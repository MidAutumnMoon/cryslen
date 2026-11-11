//! GL rendering: window, EGL surface, and the page quad. GL runs on the
//! main thread with the context current. Texture-creation failure panics.

use core::num::NonZeroU32;
use std::collections::HashMap;
use std::sync::Arc;

use egui_glow::glow;
use egui_glow::glow::HasContext;
use egui_winit::winit;
use eros::ErrorUnion;
use glutin::context::PossiblyCurrentContext;
use glutin::surface::Surface;
use glutin::surface::WindowSurface;

use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::raw_window_handle::HasWindowHandle as _;
use winit::window::Window as WinitWindow;

use crate::browse::Role;
use crate::browse::Roles;
use crate::decode;
use crate::decode::Decoded;
use crate::decode::Matrix;
use crate::decode::Pixels;
use crate::decode::Plane;
use crate::decode::Subsampling;
use crate::decode::Transfer;
use crate::decode::YuvImage;
use crate::fault::TextureError;
use crate::fault::TextureFault;
use crate::geometry;
use crate::geometry::Placement;
use crate::geometry::SpreadLayout;
use crate::geometry::Viewport;

// The quad is a fixed -1..1 strip generated from gl_VertexID; uniforms
// place it at the seat rect. Clip space is bottom-origin; winit coords
// are top-origin.
const VS: &str = concat!(
    "#version 330 core\n",
    "
    uniform vec2 u_window_size; // window size, px
    uniform vec2 u_seat_offset; // seat rect origin, screen px
    uniform vec2 u_seat_size;   // seat rect size, screen px
    out vec2 v_seat_px;
    void main() {
        vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
        vec2 pos = corner * 2.0 - 1.0;
        vec2 seat_px = (pos * 0.5 + 0.5) * u_seat_size;
        v_seat_px = seat_px;
        vec2 screen_px = u_seat_offset + seat_px;
        gl_Position = vec4(screen_px.x / u_window_size.x * 2.0 - 1.0,
                           1.0 - screen_px.y / u_window_size.y * 2.0,
                           0.0, 1.0);
    }"
);

// Per-draw constants are folded CPU-side; output is premultiplied.
const FS: &str = concat!(
    "#version 330 core\n",
    "
    in vec2 v_seat_px;
    layout(location = 0) out vec4 color;
    uniform vec2 u_page_uv_per_pixel;   // screen px -> page fraction: 1 / (scale * page size)
    uniform vec2 u_chroma_uv_per_pixel; // screen px -> chroma-page fraction, subsample-aware
    uniform mat3 u_color_matrix;  // storage samples -> gamma RGB
    uniform vec3 u_color_offset;  // u_color_matrix times the quantization anchors
    uniform float u_alpha_normalization; // depth normalize of the alpha sample; 1.0 for 8-bit storage
    uniform int u_pixel_layout;   // PIXEL_LAYOUT_RGBA or PIXEL_LAYOUT_YUV
    uniform int u_transfer_code;  // TRANSFER_SRGB or TRANSFER_PQ
    uniform int u_dither;         // 0 = 8-bit source, 1 = deep source
    uniform sampler2D u_rgba_or_luma;
    uniform sampler2D u_cb;
    uniform sampler2D u_cr;
    uniform sampler2D u_alpha;
    const int PIXEL_LAYOUT_RGBA = 0;
    const int PIXEL_LAYOUT_YUV = 1;
    const int TRANSFER_SRGB = 0;
    const int TRANSFER_PQ = 1;
    // ST 2084 constants.
    const float PQ_M1 = 2610.0 / 16384.0;
    const float PQ_M2 = 2523.0 / 4096.0 * 128.0;
    const float PQ_C1 = 3424.0 / 4096.0;
    const float PQ_C2 = 2413.0 / 4096.0 * 32.0;
    const float PQ_C3 = 2392.0 / 4096.0 * 32.0;
    // PQ is absolute: diffuse white (203 nit) maps to display white and
    // brighter highlights clip, the fixed HDR->SDR compromise.
    const float PQ_DIFFUSE_WHITE = 203.0;
    const float PQ_PEAK = 10000.0;
    const float OUTPUT_MAX = 255.0;
    vec3 to_srgb(vec3 rgb) {
        if (u_transfer_code == TRANSFER_SRGB) {
            return rgb;
        }
        vec3 pq_scaled = pow(clamp(rgb, 0.0, 1.0), vec3(1.0 / PQ_M2));
        vec3 luminance = PQ_PEAK * pow(max(pq_scaled - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * pq_scaled), vec3(1.0 / PQ_M1));
        vec3 sdr_linear = clamp(luminance / PQ_DIFFUSE_WHITE, 0.0, 1.0);
        // Linear light to the sRGB transfer curve.
        return mix(sdr_linear * 12.92, 1.055 * pow(sdr_linear, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), sdr_linear));
    }
    float bayer4(vec2 fragment_px) {
        float thresholds[16] = float[16](
            0.0, 8.0, 2.0, 10.0,
            12.0, 4.0, 14.0, 6.0,
            3.0, 11.0, 1.0, 9.0,
            15.0, 7.0, 13.0, 5.0
        );
        ivec2 cell = ivec2(fragment_px) & 3;
        return thresholds[cell.x + 4 * cell.y] / 16.0;
    }
    void main() {
        // Fragment centers land on texel centers at 1:1 zoom with
        // whole-pixel origins.
        vec2 seat_px = v_seat_px;
        // Dither the straight color at half an output quantum; an 8-bit
        // source has nothing to hide.
        float dither_offset = 0.0;
        if (u_dither != 0) {
            dither_offset = (bayer4(gl_FragCoord.xy) - 0.5) / OUTPUT_MAX;
        }
        if (u_pixel_layout == PIXEL_LAYOUT_RGBA) {
            vec4 rgba_texel = texture(u_rgba_or_luma, seat_px * u_page_uv_per_pixel);
            color = vec4((to_srgb(rgba_texel.rgb) + dither_offset) * rgba_texel.a, rgba_texel.a);
            return;
        }
        vec3 yuv_samples = vec3(
            texture(u_rgba_or_luma, seat_px * u_page_uv_per_pixel).r,
            texture(u_cb, seat_px * u_chroma_uv_per_pixel).r,
            texture(u_cr, seat_px * u_chroma_uv_per_pixel).r);
        float alpha = texture(u_alpha, seat_px * u_page_uv_per_pixel).r * u_alpha_normalization;
        color = vec4((to_srgb(u_color_matrix * yuv_samples - u_color_offset) + dither_offset) * alpha, alpha);
    }"
);

/// Seat rect and page UV agree within float rounding; page dims are
/// whole pixels.
const SEAT_UV_TOLERANCE: f32 = 1e-3;

/// Page-keyed texture cache and spread draw; a seat draws its largest
/// resident role.
pub struct PageTextures {
    gl: Arc<glow::Context>,
    pipeline: Pipeline,
    pages: HashMap<usize, [Option<PageTexture>; Role::COUNT]>,
    max_texture_edge: u32,
}

struct UniformLocations {
    window_size: <glow::Context as glow::HasContext>::UniformLocation,
    seat_offset: <glow::Context as glow::HasContext>::UniformLocation,
    seat_size: <glow::Context as glow::HasContext>::UniformLocation,
    page_uv_per_pixel:
        <glow::Context as glow::HasContext>::UniformLocation,
    chroma_uv_per_pixel:
        <glow::Context as glow::HasContext>::UniformLocation,
    color_matrix: <glow::Context as glow::HasContext>::UniformLocation,
    color_offset: <glow::Context as glow::HasContext>::UniformLocation,
    alpha_normalization:
        <glow::Context as glow::HasContext>::UniformLocation,
    pixel_layout: <glow::Context as glow::HasContext>::UniformLocation,
    transfer_code: <glow::Context as glow::HasContext>::UniformLocation,
    dither: <glow::Context as glow::HasContext>::UniformLocation,
}

/// One cached role's GPU state.
struct PageTexture {
    /// Screen px -> page fraction, before the zoom-scale divide.
    page_uv_per_pixel: (f32, f32),
    transfer: Transfer,
    needs_dither: bool,
    layout: PixelLayout,
}

enum PixelLayout {
    Rgba {
        texture: <glow::Context as glow::HasContext>::Texture,
        dims: (u32, u32),
    },
    Yuv {
        luma: <glow::Context as glow::HasContext>::Texture,
        cb: <glow::Context as glow::HasContext>::Texture,
        cr: <glow::Context as glow::HasContext>::Texture,
        alpha: <glow::Context as glow::HasContext>::Texture,
        /// Screen px -> chroma-page fraction, subsample-folded.
        chroma_uv_per_pixel: (f32, f32),
        color_conversion: ColorConversion,
        /// Storage-sample normalization for the alpha plane.
        alpha_normalization: f32,
    },
}

impl PixelLayout {
    #[expect(
        clippy::rest_pattern_accessible_field,
        reason = "the code depends only on the variant"
    )]
    fn shader_code(&self) -> i32 {
        match self {
            Self::Rgba { .. } => PIXEL_LAYOUT_RGBA,
            Self::Yuv { .. } => PIXEL_LAYOUT_YUV,
        }
    }
}

const PIXEL_LAYOUT_RGBA: i32 = 0;
const PIXEL_LAYOUT_YUV: i32 = 1;

/// Fold storage samples to gamma RGB at upload: `matrix * samples - offset`.
/// Samples are 0..1 storage units; R16 sampling already divides by 65535.
#[derive(Debug, Clone, Copy)]
struct ColorConversion {
    /// Column-major 3x3.
    matrix: [f32; 9],
    offset: [f32; 3],
}

#[expect(
    clippy::indexing_slicing,
    reason = "Role::index is bounded by Role::COUNT"
)]
fn role_slot(
    slots: &[Option<PageTexture>; Role::COUNT],
    role: Role,
) -> Option<&PageTexture> {
    slots[role.index()].as_ref()
}

#[expect(
    clippy::indexing_slicing,
    reason = "Role::index is bounded by Role::COUNT"
)]
fn role_slot_mut(
    slots: &mut [Option<PageTexture>; Role::COUNT],
    role: Role,
) -> &mut Option<PageTexture> {
    &mut slots[role.index()]
}

/// The largest resident role; slots are in ascending [`Role::ALL`] order.
fn best_texture(
    slots: &[Option<PageTexture>; Role::COUNT],
) -> Option<&PageTexture> {
    slots.iter().rev().find_map(Option::as_ref)
}

/// Refuse a page over `MAX_TEXTURE_SIZE` before upload; an oversized
/// upload samples black.
fn check_texture_dims(
    dims: (u32, u32),
    max_edge: u32,
) -> Result<(), TextureError> {
    if dims.0 == 0 || dims.1 == 0 || dims.0 > max_edge || dims.1 > max_edge
    {
        return Err(ErrorUnion::new(TextureFault { dims, max_edge }));
    }
    Ok(())
}

impl PageTextures {
    pub fn new(gl: Arc<glow::Context>) -> eros::Result<Self> {
        // SAFETY: the returned names land in Self, whose Drop deletes
        // them. The error path below leaks them into the process-fatal
        // exit.
        let pipeline = unsafe { create_pipeline(&gl, VS, FS) }?;
        #[expect(
            clippy::map_err_ignore,
            reason = "the dropped conversion error says only that the value is out of range, which the message states"
        )]
        // SAFETY: MAX_TEXTURE_SIZE is a pure query with no object names.
        let max_texture_edge = u32::try_from(unsafe {
            gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE)
        })
        .map_err(|_| {
            eros::error!("GL MAX_TEXTURE_SIZE is not a positive GLint")
        })?;
        Ok(Self {
            gl,
            pipeline,
            pages: HashMap::new(),
            max_texture_edge,
        })
    }

    /// Upload `image` as `page`'s `role`; no-op if present, never evicts.
    pub fn insert(
        &mut self,
        page: usize,
        role: Role,
        image: &Decoded,
    ) -> Result<(), TextureError> {
        if self.has(page, role) {
            return Ok(());
        }
        let dims = (image.width(), image.height());
        check_texture_dims(dims, self.max_texture_edge)?;
        // SAFETY: the size check above keeps the page inside this
        // context's texture limit, and `upload_page_texture` asserts its
        // own plane byte extents.
        let texture = unsafe { upload_page_texture(&self.gl, image) };
        // SAFETY: the context is current and the entry's names are live
        // until the cache removes it.
        unsafe {
            label_entry(&self.gl, page, role, &texture);
        }
        *role_slot_mut(self.pages.entry(page).or_default(), role) =
            Some(texture);
        Ok(())
    }

    /// Keeps the page's role slot.
    #[expect(
        clippy::rest_pattern_accessible_field,
        reason = "the update only reads the fields that decide reuse"
    )]
    pub fn update(
        &mut self,
        page: usize,
        role: Role,
        image: &Decoded,
    ) -> Result<(), TextureError> {
        let dims = (image.width(), image.height());
        check_texture_dims(dims, self.max_texture_edge)?;
        if let Some(PageTexture {
            layout:
                PixelLayout::Rgba {
                    texture,
                    dims: present_dims,
                },
            transfer,
            ..
        }) = self
            .pages
            .get(&page)
            .and_then(|slots| role_slot(slots, role))
        {
            if *present_dims == dims && *transfer == image.transfer() {
                if let Pixels::Rgba8(bytes) = image.pixels() {
                    // SAFETY: the size check above and `Decoded`'s
                    // extent check keep the byte length right, and the
                    // texture is live until this entry is removed.
                    unsafe {
                        update_rgba(&self.gl, *texture, bytes, dims);
                    }
                    return Ok(());
                }
            }
        }
        let empty = if let Some(slots) = self.pages.get_mut(&page) {
            if let Some(texture) = role_slot_mut(slots, role).take() {
                // SAFETY: the entry left the slots, so this is its only
                // deletion.
                unsafe {
                    delete_entry(&self.gl, &texture);
                }
            }
            slots.iter().all(Option::is_none)
        } else {
            false
        };
        if empty {
            self.pages.remove(&page);
        }
        self.insert(page, role, image)
    }

    pub fn roles(&self, page: usize) -> Roles {
        let Some(slots) = self.pages.get(&page) else {
            return Roles::default();
        };
        let mut roles = Roles::default();
        for (slot, role) in slots.iter().zip(Role::ALL) {
            if slot.is_some() {
                roles.set(role);
            }
        }
        roles
    }

    fn has(&self, page: usize, role: Role) -> bool {
        self.pages
            .get(&page)
            .is_some_and(|slots| role_slot(slots, role).is_some())
    }

    pub fn has_any(&self, page: usize) -> bool {
        self.pages
            .get(&page)
            .is_some_and(|slots| slots.iter().any(Option::is_some))
    }

    /// Drop every texture whose page and role `keep` rejects.
    pub fn retain(&mut self, keep: impl Fn(usize, Role) -> bool) {
        let gl = &self.gl;
        self.pages.retain(|&page, slots| {
            for (slot, role) in slots.iter_mut().zip(Role::ALL) {
                if slot.is_none() || keep(page, role) {
                    continue;
                }
                if let Some(texture) = slot.take() {
                    // SAFETY: the texture leaves the slots in this same
                    // step, so this is its only deletion.
                    unsafe {
                        delete_entry(gl, &texture);
                    }
                }
            }
            slots.iter().any(Option::is_some)
        });
    }

    pub fn clear_all(&mut self) {
        self.retain(|_, _| false);
    }

    /// Draw the shown spread, clipped to `viewport`; each seat draws its
    /// largest resident role. Does not clear the framebuffer.
    pub fn draw(
        &self,
        window_size: (u32, u32),
        viewport: Viewport,
        spread_layout: &SpreadLayout,
        seat_pages: [Option<usize>; 2],
        placement: &Placement,
    ) {
        if window_size.0 == 0 || window_size.1 == 0 {
            return;
        }
        // SAFETY: the pipeline is Self's field, live until Drop.
        unsafe {
            begin_frame(&self.gl, &self.pipeline, window_size);
        }
        // GL scissor is bottom-origin and the viewport hugs the window's
        // top-left, so only the height converts.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "viewport size is clamped non-negative whole physical px held as f32"
        )]
        let (width, height) =
            (viewport.size().0 as u32, viewport.size().1 as u32);
        // SAFETY: the context is current. This only toggles scissor state.
        unsafe {
            self.gl.enable(glow::SCISSOR_TEST);
        }
        // SAFETY: same contract as above, setting the scissor box.
        unsafe {
            self.gl.scissor(
                0,
                gl_int(window_size.1.saturating_sub(height)),
                gl_int(width),
                gl_int(height),
            );
        }
        for (rect, page) in
            spread_layout.rects.iter().zip(seat_pages.iter())
        {
            let Some(rect) = rect else { continue };
            let Some(page) = page else { continue };
            let Some(texture) =
                self.pages.get(page).and_then(best_texture)
            else {
                continue;
            };
            let page_uv_per_pixel = texture.page_uv_per_pixel;
            // A seat rect is in page pixels; a texture's UV spans the page.
            let uv_extent = (
                rect.size.0 * page_uv_per_pixel.0,
                rect.size.1 * page_uv_per_pixel.1,
            );
            let aligned = (uv_extent.0 - 1.0).abs() <= SEAT_UV_TOLERANCE
                && (uv_extent.1 - 1.0).abs() <= SEAT_UV_TOLERANCE;
            if !aligned {
                tracing::debug!(
                    page = *page,
                    rect = ?rect.size,
                    ?page_uv_per_pixel,
                    "seat layout disagrees with the texture's page dims"
                );
            }
            debug_assert!(
                aligned,
                "seat rect and texture page dims disagree for page \
                 {page}: rect {:?}, page uv {page_uv_per_pixel:?}",
                rect.size
            );
            // Whole-pixel rect origin: at 1:1 zoom a fractional origin
            // makes NEAREST draw uneven texel rows.
            let screen_origin = (
                rect.pos
                    .0
                    .mul_add(placement.scale, placement.offset.0)
                    .round(),
                rect.pos
                    .1
                    .mul_add(placement.scale, placement.offset.1)
                    .round(),
            );
            let screen_size = (
                rect.size.0 * placement.scale,
                rect.size.1 * placement.scale,
            );
            if screen_size.0 <= 0.0 || screen_size.1 <= 0.0 {
                continue;
            }
            // SAFETY: &self pins the texture, so its GL names outlive the call.
            unsafe {
                draw_page(
                    &self.gl,
                    &self.pipeline,
                    texture,
                    screen_origin,
                    screen_size,
                    placement.scale,
                );
            }
        }
        // SAFETY: the context is current. This restores unclipped raster
        // state.
        unsafe {
            self.gl.disable(glow::SCISSOR_TEST);
        }
    }
}

impl Drop for PageTextures {
    fn drop(&mut self) {
        // GL objects must be deleted while their context is alive. The
        // window that owns the context is dropped after this struct.
        self.clear_all();
        // SAFETY: the context outlives this drop. These names are
        // deleted once, here.
        unsafe {
            delete_pipeline(&self.gl, &self.pipeline);
        }
    }
}

/// # Safety
///
/// The context must be current. The caller must delete the returned name.
unsafe fn compile_shader(
    gl: &glow::Context,
    kind: u32,
    src: &str,
) -> eros::Result<<glow::Context as HasContext>::Shader> {
    let shader = gl.create_shader(kind).map_err(|error| {
        eros::error!("GL shader object creation failed: {error}")
    })?;
    gl.shader_source(shader, src);
    gl.compile_shader(shader);
    if !gl.get_shader_compile_status(shader) {
        return Err(eros::error!(
            "GL shader compile failed: {}",
            gl.get_shader_info_log(shader)
        ));
    }
    Ok(shader)
}

/// The linked program, cached uniform locations, and the empty VAO a
/// core profile requires.
struct Pipeline {
    program: <glow::Context as HasContext>::Program,
    uniform_locations: UniformLocations,
    vao: <glow::Context as glow::HasContext>::VertexArray,
}

/// Create the linked program and empty VAO. Objects from a failed step
/// leak into the process-fatal exit.
///
/// # Safety
///
/// The context must be current. The returned names are live until deleted.
unsafe fn create_pipeline(
    gl: &glow::Context,
    vs: &str,
    fs: &str,
) -> eros::Result<Pipeline> {
    let program = gl.create_program().map_err(|error| {
        eros::error!("GL program object creation failed: {error}")
    })?;
    let vao = gl.create_vertex_array().map_err(|error| {
        eros::error!("GL vertex array object creation failed: {error}")
    })?;

    let vertex_shader = compile_shader(gl, glow::VERTEX_SHADER, vs)?;
    gl.attach_shader(program, vertex_shader);
    let fragment_shader = compile_shader(gl, glow::FRAGMENT_SHADER, fs)?;
    gl.attach_shader(program, fragment_shader);
    gl.link_program(program);
    if !gl.get_program_link_status(program) {
        return Err(eros::error!(
            "GL program link failed: {}",
            gl.get_program_info_log(program)
        ));
    }
    gl.delete_shader(vertex_shader);
    gl.delete_shader(fragment_shader);

    // A core profile still requires a VAO.
    gl.bind_vertex_array(Some(vao));

    // Sampler units are fixed. Draw rebinds textures only. A silently
    // skipped sampler uniform would point every unit at 0, so a missing
    // location fails creation.
    gl.use_program(Some(program));
    let location = |name: &str| {
        #[expect(
            clippy::expect_used,
            reason = "declared and used in the in-tree shader sources, so a miss is an in-tree regression"
        )]
        gl.get_uniform_location(program, name)
            .expect("uniform location in page shader")
    };
    for (unit, name) in [
        (0_i32, "u_rgba_or_luma"),
        (1, "u_cb"),
        (2, "u_cr"),
        (3, "u_alpha"),
    ] {
        gl.uniform_1_i32(Some(&location(name)), unit);
    }
    let uniform_locations = UniformLocations {
        window_size: location("u_window_size"),
        seat_offset: location("u_seat_offset"),
        seat_size: location("u_seat_size"),
        page_uv_per_pixel: location("u_page_uv_per_pixel"),
        chroma_uv_per_pixel: location("u_chroma_uv_per_pixel"),
        color_matrix: location("u_color_matrix"),
        color_offset: location("u_color_offset"),
        alpha_normalization: location("u_alpha_normalization"),
        pixel_layout: location("u_pixel_layout"),
        transfer_code: location("u_transfer_code"),
        dither: location("u_dither"),
    };

    label(gl, glow::PROGRAM, program.0.get(), "page program");
    label(gl, glow::VERTEX_ARRAY, vao.0.get(), "page quad");

    Ok(Pipeline {
        program,
        uniform_locations,
        vao,
    })
}

/// Set the frame state the page draws depend on: the viewport, the
/// premultiplied compositing pages draw with, and the window-size
/// uniform. egui's own paint re-sets its blend state afterwards.
///
/// # Safety
///
/// The context must be current. `pipeline`'s names are live.
unsafe fn begin_frame(
    gl: &glow::Context,
    pipeline: &Pipeline,
    window_size: (u32, u32),
) {
    gl.viewport(0, 0, gl_int(window_size.0), gl_int(window_size.1));
    gl.enable(glow::BLEND);
    gl.blend_func_separate(
        glow::ONE,
        glow::ONE_MINUS_SRC_ALPHA,
        glow::ONE,
        glow::ONE_MINUS_SRC_ALPHA,
    );
    gl.use_program(Some(pipeline.program));
    gl.bind_vertex_array(Some(pipeline.vao));
    let window_px = geometry::px_f32(window_size);
    gl.uniform_2_f32(
        Some(&pipeline.uniform_locations.window_size),
        window_px.0,
        window_px.1,
    );
}

/// Draw one seat's quad. `u_pixel_layout` guards the layout-specific
/// uniforms.
///
/// # Safety
///
/// The context must be current. `pipeline`'s names and `page_texture`'s
/// texture names are live for the duration of the call.
#[expect(
    clippy::rest_pattern_accessible_field,
    reason = "the draw ignores the RGBA texture dims"
)]
unsafe fn draw_page(
    gl: &glow::Context,
    pipeline: &Pipeline,
    page_texture: &PageTexture,
    screen_origin: (f32, f32),
    screen_size: (f32, f32),
    scale: f32,
) {
    let locations = &pipeline.uniform_locations;
    gl.use_program(Some(pipeline.program));
    gl.uniform_2_f32(
        Some(&locations.seat_offset),
        screen_origin.0,
        screen_origin.1,
    );
    gl.uniform_2_f32(
        Some(&locations.seat_size),
        screen_size.0,
        screen_size.1,
    );
    gl.uniform_1_i32(
        Some(&locations.pixel_layout),
        page_texture.layout.shader_code(),
    );
    gl.uniform_1_i32(
        Some(&locations.transfer_code),
        transfer_code(page_texture.transfer),
    );
    gl.uniform_1_i32(
        Some(&locations.dither),
        i32::from(page_texture.needs_dither),
    );
    gl.uniform_2_f32(
        Some(&locations.page_uv_per_pixel),
        page_texture.page_uv_per_pixel.0 / scale,
        page_texture.page_uv_per_pixel.1 / scale,
    );
    gl.active_texture(glow::TEXTURE0);
    match &page_texture.layout {
        PixelLayout::Rgba { texture, .. } => {
            gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
        }
        PixelLayout::Yuv {
            luma,
            cb,
            cr,
            alpha,
            chroma_uv_per_pixel,
            color_conversion,
            alpha_normalization,
        } => {
            gl.uniform_2_f32(
                Some(&locations.chroma_uv_per_pixel),
                chroma_uv_per_pixel.0 / scale,
                chroma_uv_per_pixel.1 / scale,
            );
            gl.uniform_1_f32(
                Some(&locations.alpha_normalization),
                *alpha_normalization,
            );
            gl.uniform_matrix_3_f32_slice(
                Some(&locations.color_matrix),
                false,
                &color_conversion.matrix,
            );
            gl.uniform_3_f32(
                Some(&locations.color_offset),
                color_conversion.offset[0],
                color_conversion.offset[1],
                color_conversion.offset[2],
            );
            gl.bind_texture(glow::TEXTURE_2D, Some(*luma));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(*cb));
            gl.active_texture(glow::TEXTURE2);
            gl.bind_texture(glow::TEXTURE_2D, Some(*cr));
            gl.active_texture(glow::TEXTURE3);
            gl.bind_texture(glow::TEXTURE_2D, Some(*alpha));
        }
    }
    gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
}

/// # Safety
///
/// The context must be current. Each name is live and deleted at most once.
unsafe fn delete_pipeline(gl: &glow::Context, pipeline: &Pipeline) {
    gl.delete_program(pipeline.program);
    gl.delete_vertex_array(pipeline.vao);
}

/// # Safety
///
/// The context must be current. The names are live and deleted at most once.
#[expect(
    clippy::rest_pattern_accessible_field,
    reason = "deletion touches the texture names only"
)]
unsafe fn delete_entry(gl: &glow::Context, page_texture: &PageTexture) {
    match &page_texture.layout {
        PixelLayout::Rgba { texture, .. } => gl.delete_texture(*texture),
        PixelLayout::Yuv {
            luma,
            cb,
            cr,
            alpha,
            ..
        } => {
            gl.delete_texture(*luma);
            gl.delete_texture(*cb);
            gl.delete_texture(*cr);
            gl.delete_texture(*alpha);
        }
    }
}

#[expect(
    clippy::cast_possible_wrap,
    reason = "GL enums, capped texture edges, and window px fit i32; the GL ABI takes i32 enums"
)]
fn gl_int(value: u32) -> i32 {
    value as i32
}

/// Name one object when the driver exposes `KHR_debug`.
///
/// # Safety
///
/// The context must be current and `name` must be live.
unsafe fn label(
    gl: &glow::Context,
    identifier: u32,
    name: u32,
    text: &str,
) {
    if gl.supports_debug() {
        gl.object_label(identifier, name, Some(text));
    }
}

/// # Safety
///
/// The context must be current and the textures' names must be live.
#[expect(
    clippy::rest_pattern_accessible_field,
    reason = "the labels read the texture names only"
)]
unsafe fn label_entry(
    gl: &glow::Context,
    page: usize,
    role: Role,
    page_texture: &PageTexture,
) {
    if !gl.supports_debug() {
        return;
    }
    let role = role.name();
    match &page_texture.layout {
        PixelLayout::Rgba { texture, .. } => label(
            gl,
            glow::TEXTURE,
            texture.0.get(),
            &format!("page {page} {role} rgba"),
        ),
        PixelLayout::Yuv {
            luma,
            cb,
            cr,
            alpha,
            ..
        } => {
            label(
                gl,
                glow::TEXTURE,
                luma.0.get(),
                &format!("page {page} {role} luma"),
            );
            label(
                gl,
                glow::TEXTURE,
                cb.0.get(),
                &format!("page {page} {role} cb"),
            );
            label(
                gl,
                glow::TEXTURE,
                cr.0.get(),
                &format!("page {page} {role} cr"),
            );
            label(
                gl,
                glow::TEXTURE,
                alpha.0.get(),
                &format!("page {page} {role} alpha"),
            );
        }
    }
}

#[derive(Clone, Copy)]
enum MagnificationFilter {
    Nearest,
    Linear,
}

impl MagnificationFilter {
    fn gl_filter(self) -> u32 {
        match self {
            Self::Nearest => glow::NEAREST,
            Self::Linear => glow::LINEAR,
        }
    }
}

fn magnification_filter(decoded: &Decoded) -> MagnificationFilter {
    if (decoded.width(), decoded.height()) == decoded.page_dims() {
        MagnificationFilter::Nearest
    } else {
        MagnificationFilter::Linear
    }
}

/// Dither hides precision loss from a PQ or >8-bit source; 8-bit sRGB
/// has none.
fn needs_dither(transfer: Transfer, plane_depth: Option<u8>) -> bool {
    transfer == Transfer::Pq
        || plane_depth.is_some_and(|depth| depth > BYTE_DEPTH)
}

/// Create a texture with mipmapped minification and the requested
/// magnification filter.
///
/// # Safety
///
/// The context must be current. The returned name is live until deleted.
unsafe fn new_texture(
    gl: &glow::Context,
    magnification_filter: MagnificationFilter,
) -> <glow::Context as HasContext>::Texture {
    #[expect(
        clippy::expect_used,
        reason = "texture-creation failure panics by policy"
    )]
    let texture = gl.create_texture().expect("GL texture object");
    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_WRAP_S,
        gl_int(glow::CLAMP_TO_EDGE),
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_WRAP_T,
        gl_int(glow::CLAMP_TO_EDGE),
    );
    let min_filter = glow::LINEAR_MIPMAP_LINEAR;
    let mag_filter = magnification_filter.gl_filter();
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_MIN_FILTER,
        gl_int(min_filter),
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_MAG_FILTER,
        gl_int(mag_filter),
    );
    texture
}

/// # Safety
///
/// The context must be current. `data` holds exactly `dims.0 * dims.1 * 4`
/// bytes and each dim is within `MAX_TEXTURE_SIZE`.
unsafe fn upload_rgba(
    gl: &glow::Context,
    data: &[u8],
    dims: (u32, u32),
    magnification_filter: MagnificationFilter,
) -> <glow::Context as HasContext>::Texture {
    let texture = new_texture(gl, magnification_filter);
    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
    gl.tex_image_2d(
        glow::TEXTURE_2D,
        0,
        gl_int(glow::RGBA8),
        gl_int(dims.0),
        gl_int(dims.1),
        0,
        glow::RGBA,
        glow::UNSIGNED_BYTE,
        glow::PixelUnpackData::Slice(Some(data)),
    );
    gl.generate_mipmap(glow::TEXTURE_2D);
    texture
}

/// # Safety
///
/// The context must be current, `texture` is live, and `data` holds
/// exactly `dims.0 * dims.1 * 4` bytes.
unsafe fn update_rgba(
    gl: &glow::Context,
    texture: <glow::Context as HasContext>::Texture,
    data: &[u8],
    dims: (u32, u32),
) {
    debug_assert_eq!(
        data.len(),
        dims.0 as usize * dims.1 as usize * 4,
        "rgba byte length"
    );
    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
    gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);
    gl.tex_sub_image_2d(
        glow::TEXTURE_2D,
        0,
        0,
        0,
        gl_int(dims.0),
        gl_int(dims.1),
        glow::RGBA,
        glow::UNSIGNED_BYTE,
        glow::PixelUnpackData::Slice(Some(data)),
    );
    gl.generate_mipmap(glow::TEXTURE_2D);
}

/// # Safety
///
/// The context must be current. `data` holds one sample per texel of
/// `dims`, two bytes each when `wide`, and each dim is within
/// `MAX_TEXTURE_SIZE`.
unsafe fn upload_plane(
    gl: &glow::Context,
    data: &[u8],
    dims: (u32, u32),
    wide: bool,
    magnification_filter: MagnificationFilter,
) -> <glow::Context as HasContext>::Texture {
    let (internal, ty) = if wide {
        (glow::R16, glow::UNSIGNED_SHORT)
    } else {
        (glow::R8, glow::UNSIGNED_BYTE)
    };
    let texture = new_texture(gl, magnification_filter);
    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
    // Odd row widths: GL's default 4-byte unpack alignment would desync
    // every row after the first.
    gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
    gl.tex_image_2d(
        glow::TEXTURE_2D,
        0,
        gl_int(internal),
        gl_int(dims.0),
        gl_int(dims.1),
        0,
        glow::RED,
        ty,
        glow::PixelUnpackData::Slice(Some(data)),
    );
    gl.generate_mipmap(glow::TEXTURE_2D);
    texture
}

/// # Safety
///
/// The context must be current. The asserts inside enforce the plane
/// byte-length precondition of the uploads.
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are the YUV plane names the two textures hold"
)]
unsafe fn upload_page_texture(
    gl: &glow::Context,
    decoded: &Decoded,
) -> PageTexture {
    let magnification_filter = magnification_filter(decoded);
    let page_uv_per_pixel = page_uv_per_pixel(decoded.page_dims());
    let (width, height) = (decoded.width(), decoded.height());
    match &decoded.pixels() {
        Pixels::Rgba8(rgba_bytes) => {
            assert_eq!(
                rgba_bytes.len(),
                width as usize * height as usize * 4,
                "rgba byte length"
            );
            PageTexture {
                page_uv_per_pixel,
                transfer: decoded.transfer(),
                needs_dither: needs_dither(decoded.transfer(), None),
                layout: PixelLayout::Rgba {
                    texture: upload_rgba(
                        gl,
                        rgba_bytes,
                        (width, height),
                        magnification_filter,
                    ),
                    dims: (width, height),
                },
            }
        }
        Pixels::Yuv(yuv) => {
            let ((luma_w, luma_h), (chroma_w, chroma_h)) =
                decode::plane_dims((width, height), yuv.subsampling);
            let wide = matches!(yuv.luma, Plane::U16(_));
            let depth = yuv.depth.clamp(MIN_PLANE_DEPTH, MAX_PLANE_DEPTH);
            let expect = |plane: &Plane,
                          want_samples: usize,
                          what: &str| {
                let want_bytes = want_samples * if wide { 2 } else { 1 };
                assert_eq!(
                    plane.byte_len(),
                    want_bytes,
                    "{what} plane bytes"
                );
            };
            expect(&yuv.luma, luma_w as usize * luma_h as usize, "luma");
            let luma_bytes = decode::plane_bytes(&yuv.luma);
            let luma_texture = upload_plane(
                gl,
                luma_bytes,
                (luma_w, luma_h),
                wide,
                magnification_filter,
            );
            let (cb_texture, cr_texture) = if yuv.subsampling
                == Subsampling::Mono
            {
                // 1x1 chroma at the neutral code, which the folded
                // quantization anchor cancels exactly.
                let neutral = constant_plane(depth, 1_u32 << (depth - 1));
                let neutral_bytes = decode::plane_bytes(&neutral);
                (
                    upload_plane(
                        gl,
                        neutral_bytes,
                        (1, 1),
                        wide,
                        MagnificationFilter::Linear,
                    ),
                    upload_plane(
                        gl,
                        neutral_bytes,
                        (1, 1),
                        wide,
                        MagnificationFilter::Linear,
                    ),
                )
            } else {
                expect(
                    &yuv.cb,
                    chroma_w as usize * chroma_h as usize,
                    "cb",
                );
                expect(
                    &yuv.cr,
                    chroma_w as usize * chroma_h as usize,
                    "cr",
                );
                let cb_bytes = decode::plane_bytes(&yuv.cb);
                let cr_bytes = decode::plane_bytes(&yuv.cr);
                (
                    upload_plane(
                        gl,
                        cb_bytes,
                        (chroma_w, chroma_h),
                        wide,
                        MagnificationFilter::Linear,
                    ),
                    upload_plane(
                        gl,
                        cr_bytes,
                        (chroma_w, chroma_h),
                        wide,
                        MagnificationFilter::Linear,
                    ),
                )
            };
            let alpha_texture = yuv.alpha.as_ref().map_or_else(
                // 1x1 opaque: the max code, which the alpha normalization
                // maps to 1.0.
                || {
                    let opaque =
                        constant_plane(depth, (1_u32 << depth) - 1);
                    upload_plane(
                        gl,
                        decode::plane_bytes(&opaque),
                        (1, 1),
                        wide,
                        MagnificationFilter::Linear,
                    )
                },
                |alpha| {
                    expect(
                        alpha,
                        width as usize * height as usize,
                        "alpha",
                    );
                    upload_plane(
                        gl,
                        decode::plane_bytes(alpha),
                        (width, height),
                        wide,
                        MagnificationFilter::Linear,
                    )
                },
            );
            PageTexture {
                page_uv_per_pixel,
                transfer: decoded.transfer(),
                needs_dither: needs_dither(
                    decoded.transfer(),
                    Some(depth),
                ),
                layout: PixelLayout::Yuv {
                    luma: luma_texture,
                    cb: cb_texture,
                    cr: cr_texture,
                    alpha: alpha_texture,
                    chroma_uv_per_pixel: chroma_uv_per_pixel(
                        decoded.page_dims(),
                        yuv.subsampling,
                    ),
                    color_conversion: page_color_conversion(yuv, wide),
                    alpha_normalization: depth_normalization(depth, wide),
                },
            }
        }
    }
}

/// Page fractions per screen pixel before zoom. Use the page's dims; a
/// preview covers the whole page at lower resolution.
fn page_uv_per_pixel(page_dims: (u32, u32)) -> (f32, f32) {
    let page_px = geometry::px_f32(page_dims);
    (1.0 / page_px.0, 1.0 / page_px.1)
}

fn chroma_uv_per_pixel(
    page_dims: (u32, u32),
    subsampling: Subsampling,
) -> (f32, f32) {
    let (_, (chroma_w, chroma_h)) =
        decode::plane_dims(page_dims, subsampling);
    let chroma_px = geometry::px_f32((chroma_w, chroma_h));
    let ratio = subsampling.chroma_ratio();
    (ratio.0 / chroma_px.0, ratio.1 / chroma_px.1)
}

fn constant_plane(depth: u8, code: u32) -> Plane {
    if depth > BYTE_DEPTH {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "callers pass a depth clamped to 16, so the code fits u16"
        )]
        let wide = code as u16;
        Plane::U16(vec![wide].into_boxed_slice())
    } else {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "an 8-bit depth keeps codes within u8"
        )]
        let byte = code as u8;
        Plane::U8(vec![byte].into_boxed_slice())
    }
}

const TRANSFER_SRGB: i32 = 0;
const TRANSFER_PQ: i32 = 1;

fn transfer_code(transfer: Transfer) -> i32 {
    match transfer {
        Transfer::Srgb => TRANSFER_SRGB,
        Transfer::Pq => TRANSFER_PQ,
    }
}

/// Normalize R16 output for U16-stored planes: sampling divides by
/// [`R16_STORAGE_MAX`] but only `depth` bits are significant. 1.0 for 8-bit
/// storage.
#[expect(
    clippy::cast_precision_loss,
    reason = "the depth mask 2^depth - 1 is at most 65535, exact in f32"
)]
fn depth_normalization(depth: u8, wide: bool) -> f32 {
    if !wide {
        return 1.0;
    }
    R16_STORAGE_MAX
        / ((1_u32 << depth.clamp(MIN_PLANE_DEPTH, MAX_PLANE_DEPTH)) - 1)
            as f32
}

const R16_STORAGE_MAX: f32 = 65535.0;
const R8_STORAGE_MAX: f32 = 255.0;

/// 8-bit storage; wider planes use `R16` and need depth normalization.
const BYTE_DEPTH: u8 = 8;

const MIN_PLANE_DEPTH: u8 = 1;
const MAX_PLANE_DEPTH: u8 = 16;

/// YUV storage-code quantization: anchor codes, sample-to-RGB scales,
/// and capacity. Limited range scales the BT studio swing by 2^(depth-8).
struct CodeQuantization {
    luma_black: f32,
    chroma_neutral: f32,
    luma_scale: f32,
    chroma_scale: f32,
    storage_max: f32,
}

#[expect(
    clippy::cast_precision_loss,
    reason = "codes and scales stay below 65536, exact in f32"
)]
fn code_quantization(
    depth: u8,
    full_range: bool,
    wide: bool,
) -> CodeQuantization {
    let max_code = (1_u32 << depth) - 1;
    let storage_max = if wide {
        R16_STORAGE_MAX
    } else {
        R8_STORAGE_MAX
    };
    if full_range {
        CodeQuantization {
            luma_black: 0.0,
            chroma_neutral: (1_u32 << (depth - 1)) as f32,
            luma_scale: 1.0,
            chroma_scale: 1.0,
            storage_max,
        }
    } else {
        let swing = u32::from(depth.saturating_sub(8));
        CodeQuantization {
            luma_black: (16 << swing) as f32,
            chroma_neutral: (128 << swing) as f32,
            luma_scale: max_code as f32 / (219 << swing) as f32,
            chroma_scale: max_code as f32 / (224 << swing) as f32,
            storage_max,
        }
    }
}

/// Fold storage samples to gamma RGB: depth normalization, quantization,
/// and the KR/KB matrix (or GBR permutation). The offset is the matrix
/// times the anchor codes.
fn page_color_conversion(yuv: &YuvImage, wide: bool) -> ColorConversion {
    let depth = yuv.depth.clamp(MIN_PLANE_DEPTH, MAX_PLANE_DEPTH);
    let quantization = code_quantization(depth, yuv.full_range, wide);
    let normalization = depth_normalization(depth, wide);
    let (
        base_matrix,
        luma_scale,
        chroma_scale,
        luma_black,
        chroma_neutral,
    ) = if yuv.matrix == Matrix::Identity {
        // H.273 identity planes are GBR in the (Y, Cb, Cr) slots,
        // quantized as luma on all three axes: permute into RGB and
        // anchor every axis at luma black. Mono identity repeats the
        // luma plane to RGB.
        let base_matrix = if yuv.subsampling == Subsampling::Mono {
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        };
        (
            base_matrix,
            quantization.luma_scale,
            quantization.luma_scale,
            quantization.luma_black,
            quantization.luma_black,
        )
    } else {
        (
            yuv_to_rgb_matrix(yuv.matrix),
            quantization.luma_scale,
            quantization.chroma_scale,
            quantization.luma_black,
            quantization.chroma_neutral,
        )
    };
    let mut matrix = base_matrix;
    for (column, matrix_column) in
        matrix.as_chunks_mut::<3>().0.iter_mut().enumerate()
    {
        let scale = if column == 0 {
            luma_scale
        } else {
            chroma_scale
        } * normalization;
        for cell in matrix_column {
            *cell *= scale;
        }
    }
    let anchors = [
        luma_black / quantization.storage_max,
        chroma_neutral / quantization.storage_max,
        chroma_neutral / quantization.storage_max,
    ];
    let mut offset = [0.0_f32; 3];
    for (matrix_column, anchor) in
        matrix.as_chunks::<3>().0.iter().zip(anchors)
    {
        for (&cell, slot) in matrix_column.iter().zip(&mut offset) {
            *slot = cell.mul_add(anchor, *slot);
        }
    }
    ColorConversion { matrix, offset }
}

/// Full-range YUV→RGB from the exact KR/KB of each standard.
/// KG = 1 - KR - KB. Identity pages never reach here;
/// `page_color_conversion` routes them through the GBR permutation.
#[expect(
    clippy::similar_names,
    reason = "the green coefficients are the Cb and Cr terms of the BT matrix derivation"
)]
fn yuv_to_rgb_matrix(matrix: Matrix) -> [f32; 9] {
    let (k_red, k_blue) = match matrix {
        Matrix::Bt601 => BT601_KR_KB,
        Matrix::Bt709 => BT709_KR_KB,
        Matrix::Bt2020 => BT2020_KR_KB,
        Matrix::Identity => (0.0, 0.0),
    };
    let k_green = 1.0 - k_red - k_blue;
    let green_from_cb = 2.0 * k_blue * (1.0 - k_blue) / k_green;
    let green_from_cr = 2.0 * k_red * (1.0 - k_red) / k_green;
    [
        1.0,
        1.0,
        1.0, //
        0.0,
        -green_from_cb,
        2.0 * (1.0 - k_blue), //
        2.0 * (1.0 - k_red),
        -green_from_cr,
        0.0,
    ]
}

const BT601_KR_KB: (f32, f32) = (0.299, 0.114);
const BT709_KR_KB: (f32, f32) = (0.2126, 0.0722);
const BT2020_KR_KB: (f32, f32) = (0.2627, 0.0593);

/// Window + EGL context + surface. vsync is `DontWait`, so the app must call
/// `pre_present_and_swap()` to present.
pub struct Window {
    winit: WinitWindow,
    gl: Arc<glow::Context>,
    surface: Surface<WindowSurface>,
    context: PossiblyCurrentContext,
    /// Alpha config matched. The fallback may still carry alpha bits; the
    /// opaque hint and clear hide them.
    translucent: bool,
    /// Size the surface was last resized to. Creation counts as the
    /// first resize.
    resized_to: (u32, u32),
}

impl Window {
    /// Create the window and its desktop OpenGL 3.3 core context.
    ///
    /// # Safety
    ///
    /// The event loop must outlive the returned window. The platform
    /// display behind its raw handle backs every GL object created here.
    pub unsafe fn new(event_loop: &ActiveEventLoop) -> eros::Result<Self> {
        use glutin::config::Api as ConfigApi;
        use glutin::config::ConfigTemplateBuilder;
        use glutin::context::ContextApi;
        use glutin::context::ContextAttributesBuilder;
        use glutin::context::GlProfile;
        use glutin::context::NotCurrentGlContext as _;
        use glutin::context::Version;
        use glutin::display::Display;
        use glutin::display::DisplayApiPreference;
        use glutin::display::GlDisplay as _;
        use glutin::prelude::GlSurface as _;
        use glutin::surface::SurfaceAttributesBuilder;
        use glutin::surface::SwapInterval;
        use std::ffi::CString;
        use winit::dpi::LogicalSize;
        use winit::raw_window_handle::HasDisplayHandle as _;

        use winit::platform::wayland::WindowAttributesExtWayland as _;
        use winit::window::WindowAttributes;

        // glutin-winit's DisplayBuilder picks a config through a closure
        // that cannot fail, so using it would force a panic on an empty
        // config list.
        let raw_display = event_loop.display_handle()?.as_raw();
        let gl_display =
            Display::new(raw_display, DisplayApiPreference::Egl)?;
        // Prefer a config with alpha so the desktop shows through; a machine
        // without one runs opaque.
        let translucent_template = ConfigTemplateBuilder::new()
            .with_api(ConfigApi::OPENGL)
            .with_alpha_size(8)
            .build();
        let (gl_config, translucent) = if let Some(config) =
            gl_display.find_configs(translucent_template)?.next()
        {
            (config, true)
        } else {
            // The second pass drops the alpha demand. The default
            // template also demands alpha, which would repeat the
            // same empty query.
            let plain = ConfigTemplateBuilder::new()
                .with_api(ConfigApi::OPENGL)
                .with_alpha_size(0)
                .build();
            let config = gl_display.find_configs(plain)?.next().ok_or_else(
                || {
                    eros::error!(
                        "no desktop OpenGL 3.3 core config on this EGL display"
                    )
                },
            )?;
            (config, false)
        };

        let attrs = WindowAttributes::default()
            .with_name("cryslen", "cryslen")
            .with_resizable(true)
            .with_inner_size(LogicalSize {
                width: INITIAL_WINDOW_WIDTH,
                height: INITIAL_WINDOW_HEIGHT,
            })
            .with_title("cryslen")
            // Without this hint winit marks the surface opaque even though the
            // buffer has alpha.
            .with_transparent(translucent);

        let window =
            glutin_winit::finalize_window(event_loop, attrs, &gl_config)?;
        let raw_window = window.window_handle()?.as_raw();
        let ctx_attrs = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
            .with_profile(GlProfile::Core)
            .build(Some(raw_window));
        let not_current =
            gl_display.create_context(&gl_config, &ctx_attrs)?;

        let (width, height): (u32, u32) = window.inner_size().into();
        let surface_attrs =
            SurfaceAttributesBuilder::<WindowSurface>::new().build(
                raw_window,
                NonZeroU32::new(width)
                    .ok_or_else(|| eros::error!("window width is zero"))?,
                NonZeroU32::new(height).ok_or_else(|| {
                    eros::error!("window height is zero")
                })?,
            );
        let surface = gl_display
            .create_window_surface(&gl_config, &surface_attrs)?;
        let context = not_current.make_current(&surface)?;
        // Non-blocking present; `pre_present_and_swap` requests a Wayland
        // frame callback.
        surface.set_swap_interval(&context, SwapInterval::DontWait)?;

        let gl = glow::Context::from_loader_function(|symbol| {
            #[expect(
                clippy::expect_used,
                reason = "symbol names are static ASCII GL entry points, NUL-free by construction"
            )]
            let symbol = CString::new(symbol)
                .expect("GL proc name without interior NUL");
            gl_display.get_proc_address(&symbol)
        });
        #[cfg(debug_assertions)]
        let mut gl = gl;
        #[cfg(debug_assertions)]
        if gl.supports_debug() {
            // SAFETY: the context is current, and the callback only logs.
            unsafe {
                gl.debug_message_callback(
                    |source, kind, id, severity, message| {
                        if severity == glow::DEBUG_SEVERITY_NOTIFICATION {
                            return;
                        }
                        tracing::debug!(
                            source,
                            kind,
                            id,
                            severity,
                            message = message,
                            "GL debug message"
                        );
                    },
                );
            }
            // SAFETY: the context is current and supports debug output.
            unsafe {
                gl.enable(glow::DEBUG_OUTPUT);
            }
        }
        let gl = Arc::new(gl);
        Ok(Self {
            winit: window,
            gl,
            surface,
            context,
            translucent,
            resized_to: (width, height),
        })
    }

    pub fn glow(&self) -> Arc<glow::Context> {
        Arc::clone(&self.gl)
    }

    pub fn translucent(&self) -> bool {
        self.translucent
    }

    pub fn winit(&self) -> &WinitWindow {
        &self.winit
    }

    /// Resize the surface when the window size changed since the last
    /// call. A zero size (minimized) and an unchanged size are no-ops.
    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        use glutin::prelude::GlSurface as _;

        let (Some(width), Some(height)) =
            (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        else {
            return;
        };
        if (size.width, size.height) == self.resized_to {
            return;
        }
        self.surface.resize(&self.context, width, height);
        self.resized_to = (size.width, size.height);
    }

    pub fn pre_present_and_swap(&self) -> eros::Result<()> {
        use glutin::prelude::GlSurface as _;
        self.winit.pre_present_notify();
        self.surface.swap_buffers(&self.context)?;
        Ok(())
    }
}

/// Initial window size, logical px.
const INITIAL_WINDOW_WIDTH: f64 = 1000.0;
const INITIAL_WINDOW_HEIGHT: f64 = 750.0;

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;
    use crate::decode::Format;
    use crate::decode::Transfer;
    use crate::decode::YuvImage;

    // GL needs a context; these tests pin the CPU-side values it consumes.

    /// Matrix element at (`row`, `col`); storage is column-major.
    #[expect(
        clippy::indexing_slicing,
        reason = "row and col are literal 0..3 coordinates into the fixed 3x3 matrix under test"
    )]
    fn element(matrix: &[f32; 9], row: usize, col: usize) -> f32 {
        matrix[col * 3 + row]
    }

    /// A zero-filled YUV page of `depth` bits, `subsampling`, range, and
    /// matrix. U16 storage for depths past 8, matching the decoder's tie.
    fn yuv_page(
        depth: u8,
        subsampling: Subsampling,
        full_range: bool,
        matrix: Matrix,
    ) -> Decoded {
        let size = (4_u32, 4);
        let ((luma_w, luma_h), (chroma_w, chroma_h)) =
            decode::plane_dims(size, subsampling);
        let plane = |samples: usize| {
            if depth > 8 {
                Plane::U16(vec![0_u16; samples].into_boxed_slice())
            } else {
                Plane::U8(vec![0_u8; samples].into_boxed_slice())
            }
        };
        let (cb, cr) = if subsampling == Subsampling::Mono {
            (Plane::U8(Box::new([])), Plane::U8(Box::new([])))
        } else {
            (
                plane(chroma_w as usize * chroma_h as usize),
                plane(chroma_w as usize * chroma_h as usize),
            )
        };
        Decoded::yuv(
            size.0,
            size.1,
            Format::Avif,
            Transfer::Srgb,
            YuvImage {
                depth,
                subsampling,
                full_range,
                matrix,
                luma: plane(luma_w as usize * luma_h as usize),
                cb,
                cr,
                alpha: None,
            },
        )
        .expect("yuv page")
    }

    fn yuv_image(decoded: &Decoded) -> &YuvImage {
        let Pixels::Yuv(yuv) = decoded.pixels() else {
            panic!("test page must be YUV");
        };
        yuv
    }

    /// The folded conversion through one row: `matrix * samples - offset`.
    fn convert(
        conversion: &ColorConversion,
        samples: [f32; 3],
    ) -> [f32; 3] {
        let mut out = [0.0_f32; 3];
        for (row, slot) in out.iter_mut().enumerate() {
            #[expect(
                clippy::indexing_slicing,
                reason = "row and col are 0..3 coordinates into the fixed 3-vectors under test"
            )]
            let converted = (0..3)
                .map(|col| {
                    element(&conversion.matrix, row, col) * samples[col]
                })
                .sum::<f32>()
                - conversion.offset[row];
            *slot = converted;
        }
        out
    }

    /// Code values of one depth's quantization (black, white, neutral) in
    /// code units, and the largest code the storage layout holds.
    fn code_quantization_codes(
        depth: u8,
        full_range: bool,
    ) -> (f32, f32, f32, f32) {
        let scale = 2.0_f32.powi(i32::from(depth) - 8);
        let storage_max =
            2.0_f32.powi(if depth > 8 { 16 } else { 8 }) - 1.0;
        let codes = if full_range {
            (
                0.0,
                2.0_f32.powi(i32::from(depth)) - 1.0,
                2.0_f32.powi(i32::from(depth) - 1),
            )
        } else {
            (16.0 * scale, 235.0 * scale, 128.0 * scale)
        };
        (codes.0, codes.1, codes.2, storage_max)
    }

    #[test]
    fn color_conversion_anchors_black_and_white_at_every_depth() {
        for depth in [8_u8, 10, 12, 16] {
            for full_range in [true, false] {
                let decoded = yuv_page(
                    depth,
                    Subsampling::Yuv444,
                    full_range,
                    Matrix::Bt601,
                );
                let conversion = page_color_conversion(
                    yuv_image(&decoded),
                    depth > BYTE_DEPTH,
                );
                let (black, white, neutral, storage_max) =
                    code_quantization_codes(depth, full_range);
                for (code, expected) in [(black, 0.0), (white, 1.0)] {
                    // Chroma at its neutral code contributes nothing.
                    let rgb = convert(
                        &conversion,
                        [
                            code / storage_max,
                            neutral / storage_max,
                            neutral / storage_max,
                        ],
                    );
                    for (row, channel) in rgb.into_iter().enumerate() {
                        assert!(
                            (channel - expected).abs() < 1e-4,
                            "depth {depth} full {full_range} row {row}: {channel} vs {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn identity_pages_permute_gbr() {
        let color = page_color_conversion(
            yuv_image(&yuv_page(
                8,
                Subsampling::Yuv444,
                true,
                Matrix::Identity,
            )),
            false,
        );
        // Column j is plane j's contribution: Y->G, U->B, V->R.
        for (col, row) in [(0, 1), (1, 2), (2, 0)] {
            for probe in 0..3 {
                let want = f32::from(probe == row);
                assert!(
                    (element(&color.matrix, probe, col) - want).abs()
                        < 1e-6,
                    "col {col} row {probe}"
                );
            }
        }
        // Full range anchors everything at black zero.
        assert_eq!(color.offset, [0.0; 3]);

        let mono = page_color_conversion(
            yuv_image(&yuv_page(
                10,
                Subsampling::Mono,
                true,
                Matrix::Identity,
            )),
            true,
        );
        for row in 0..3 {
            assert!(
                (element(&mono.matrix, row, 1)).abs() < 1e-6,
                "mono chroma col must be inert"
            );
            assert!(
                (element(&mono.matrix, row, 2)).abs() < 1e-6,
                "mono chroma col must be inert"
            );
        }
        assert_eq!(mono.offset, [0.0; 3]);
        // The norm-folded luma column still spans the range: white (code
        // 1023 in u16 storage) reads 1 on every channel.
        let white = 1023.0 / 65535.0;
        let rgb = convert(&mono, [white, white, white]);
        for (row, channel) in rgb.into_iter().enumerate() {
            assert!(
                (channel - 1.0).abs() < 1e-4,
                "mono row {row}: {channel}"
            );
        }
    }

    #[test]
    fn dither_only_for_deep_or_pq_sources() {
        assert!(!needs_dither(Transfer::Srgb, Some(8)));
        assert!(needs_dither(Transfer::Srgb, Some(10)));
        assert!(needs_dither(Transfer::Pq, Some(8)));
        assert!(!needs_dither(Transfer::Srgb, None));
        assert!(needs_dither(Transfer::Pq, None));
    }

    #[test]
    fn magnification_filter_uses_page_resolution() {
        let page = Decoded::rgba(
            100,
            100,
            Format::Png,
            Transfer::Srgb,
            vec![0_u8; 100 * 100 * 4].into_boxed_slice(),
        )
        .expect("valid page");
        assert!(matches!(
            magnification_filter(&page),
            MagnificationFilter::Nearest
        ));
        let preview = page.with_page_dims((1000, 2000));
        assert!(matches!(
            magnification_filter(&preview),
            MagnificationFilter::Linear
        ));
    }
}
