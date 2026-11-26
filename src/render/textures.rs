//! GPU page texture store, upload/update, shared 1x1 stubs, and direct quad pass.

#![expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "scoped to GL helpers sharing one current-context invariant"
)]

use std::sync::Arc;

use egui_glow::glow;
use egui_glow::glow::Context;
use egui_glow::glow::HasContext as _;
use egui_glow::glow::PixelUnpackData;
use egui_glow::glow::Texture;
use egui_glow::glow::UniformLocation;
use indoc::indoc;

use crate::image::color;
use crate::image::color::YuvMatrix;
use crate::image::pixels;
use crate::image::pixels::Depth;
use crate::image::pixels::Image;
use crate::image::pixels::PageInfo;
use crate::image::pixels::Pixels;
use crate::image::pixels::SamplePlanes;
use crate::image::pixels::THUMB_EDGE;
use crate::image::pixels::TargetRole;
use crate::image::pixels::TextureFault;
use crate::image::pixels::ThumbImage;
use crate::image::pixels::Transfer;
use crate::image::pixels::YuvSpec;
use crate::render::cache::InsertOutcome;
use crate::render::cache::Role;
use crate::render::cache::Slot;
use crate::render::cache::SlotTable;
use crate::render::ewa;
use crate::render::ewa::Resampler;
use crate::render::program;
use crate::render::program::ALPHA_SAMPLER_UNIT;
use crate::render::program::BASE_MIP_LEVEL;
use crate::render::program::CB_SAMPLER_UNIT;
use crate::render::program::CR_SAMPLER_UNIT;
use crate::render::program::DrawSource;
use crate::render::program::DrawTarget;
use crate::render::program::EmptyVao;
use crate::render::program::FIRST_VERTEX_INDEX;
use crate::render::program::GLSL_BAYER4;
use crate::render::program::GLSL_TRANSFER_HEADER;
use crate::render::program::GLSL_VERSION;
use crate::render::program::NO_BORDER;
use crate::render::program::PIXEL_LAYOUT_MONO;
use crate::render::program::PixelLayout;
use crate::render::program::QUAD_VERTEX_COUNT;
use crate::render::program::QuadUniforms;
use crate::render::program::RGBA_OR_LUMA_SAMPLER_UNIT;
use crate::render::program::SeatGeometry;
use crate::render::program::ShaderPipeline;
use crate::render::program::StubTextures;
use crate::render::program::VERTEX_SHADER_SOURCE;
use crate::render::program::YuvTexture;
use crate::render::program::YuvUniforms;

const RGBA_BYTES_PER_PIXEL: usize = 4;
const RGBA_UNPACK_ALIGNMENT: i32 = 4;
const PLANE_UNPACK_ALIGNMENT: i32 = 1;

const PAGE_FRAGMENT_BODY: &str = indoc! {"
    in vec2 v_seat_px;
    layout(location = 0) out vec4 out_color;
    uniform vec2 u_page_uv_per_screen_px;
    uniform vec2 u_chroma_uv_per_screen_px; // screen px -> chroma texture UV
    uniform mat3 u_color_matrix;             // storage samples -> gamma RGB
    uniform vec3 u_color_offset;             // u_color_matrix times the quantization anchors
    uniform float u_alpha_normalization;     // depth normalize of the alpha sample; 1.0 for 8-bit storage
    uniform vec2 u_luma_gain_offset;
    uniform int u_pixel_layout;
    uniform int u_dither;                    // 0 = 8-bit sRGB, 1 = PQ or >8-bit source
    uniform sampler2D u_rgba_or_luma;
    uniform sampler2D u_cb;
    uniform sampler2D u_cr;
    uniform sampler2D u_alpha;
    void main() {
        // Fragment centers land on texel centers at 1:1 zoom with
        // whole-pixel origins.
        vec2 seat_px = v_seat_px;
        float dither = 0.0;
        if (u_dither != 0) {
            dither = dither_offset(gl_FragCoord.xy);
        }
        if (u_pixel_layout == PIXEL_LAYOUT_MONO) {
            float luma_sample = texture(u_rgba_or_luma, seat_px * u_page_uv_per_screen_px).r;
            float gray = clamp(
                to_srgb(vec3(luma_sample * u_luma_gain_offset.x - u_luma_gain_offset.y)).r + dither,
                0.0,
                1.0);
            out_color = vec4(gray, gray, gray, 1.0);
            return;
        }
        if (u_pixel_layout == PIXEL_LAYOUT_RGBA) {
            vec4 rgba_texel = texture(u_rgba_or_luma, seat_px * u_page_uv_per_screen_px);
            out_color = vec4((to_srgb(rgba_texel.rgb) + dither) * rgba_texel.a, rgba_texel.a);
            return;
        }
        vec3 yuv_samples = vec3(
            texture(u_rgba_or_luma, seat_px * u_page_uv_per_screen_px).r,
            texture(u_cb, seat_px * u_chroma_uv_per_screen_px).r,
            texture(u_cr, seat_px * u_chroma_uv_per_screen_px).r);
        float alpha = texture(u_alpha, seat_px * u_page_uv_per_screen_px).r * u_alpha_normalization;
        out_color = vec4((to_srgb(u_color_matrix * yuv_samples - u_color_offset) + dither) * alpha, alpha);
    }
"};

#[derive(Debug, Clone, Copy)]
pub struct DrawQuad {
    pub page: usize,
    pub origin: (f64, f64),
    pub size: (f64, f64),
    pub page_dims: (u32, u32),
}

struct SharedStubs {
    gl: Arc<Context>,
    textures: StubTextures,
}

impl SharedStubs {
    fn new(gl: Arc<Context>) -> Self {
        const OPAQUE_ALPHA_8BIT: u8 = u8::MAX;
        const STUB_DIMS: (u32, u32) = (1, 1);

        let opaque_alpha = upload_plane(
            &gl,
            &[OPAQUE_ALPHA_8BIT],
            STUB_DIMS,
            MagnificationFilter::Linear,
        );
        program::label(
            &gl,
            glow::TEXTURE,
            opaque_alpha.0.get(),
            "stub opaque alpha",
        );
        let neutral_chroma = upload_plane(
            &gl,
            &[color::CHROMA_NEUTRAL_8BIT],
            STUB_DIMS,
            MagnificationFilter::Linear,
        );
        program::label(
            &gl,
            glow::TEXTURE,
            neutral_chroma.0.get(),
            "stub neutral chroma",
        );
        Self {
            gl,
            textures: StubTextures {
                opaque_alpha,
                neutral_chroma,
            },
        }
    }
}

impl Drop for SharedStubs {
    fn drop(&mut self) {
        // SAFETY: Context is current (`Window` drop order) and `self` owns both stub textures.
        unsafe {
            self.gl.delete_texture(self.textures.opaque_alpha);
            self.gl.delete_texture(self.textures.neutral_chroma);
        }
    }
}

struct Pipeline {
    shader: ShaderPipeline,
    quad: QuadUniforms,
    page_uv_per_screen_px: UniformLocation,
    dither: UniformLocation,
    pixel_layout: UniformLocation,
    transfer_code: UniformLocation,
    luma_gain_offset: UniformLocation,
    yuv: YuvUniforms,
}

fn create_pipeline(gl: Arc<Context>) -> eros::Result<Pipeline> {
    let shader = program::link(
        gl,
        VERTEX_SHADER_SOURCE,
        &[
            GLSL_VERSION,
            GLSL_TRANSFER_HEADER,
            GLSL_BAYER4,
            PAGE_FRAGMENT_BODY,
        ],
        &[
            (RGBA_OR_LUMA_SAMPLER_UNIT, "u_rgba_or_luma"),
            (CB_SAMPLER_UNIT, "u_cb"),
            (CR_SAMPLER_UNIT, "u_cr"),
            (ALPHA_SAMPLER_UNIT, "u_alpha"),
        ],
        "page program",
    )?;
    let location = |name: &str| {
        program::uniform_location(&shader.gl, shader.program, name)
    };
    let quad = QuadUniforms::lookup(&shader.gl, shader.program);
    let yuv = YuvUniforms::lookup(
        &shader.gl,
        shader.program,
        "u_chroma_uv_per_screen_px",
    );
    Ok(Pipeline {
        quad,
        page_uv_per_screen_px: location("u_page_uv_per_screen_px"),
        dither: location("u_dither"),
        pixel_layout: location("u_pixel_layout"),
        transfer_code: location("u_transfer_code"),
        luma_gain_offset: location("u_luma_gain_offset"),
        yuv,
        shader,
    })
}

/// One cached role's GPU state.
pub struct PageTexture {
    gl: Arc<Context>,
    texture_dims: (u32, u32),
    page_info: PageInfo,
    transfer: Transfer,
    needs_dither: bool,
    resident_bytes: usize,
    layout: PixelLayout,
}

impl PageTexture {
    const fn as_draw_source(&self) -> DrawSource<'_> {
        DrawSource {
            dims: self.texture_dims,
            transfer: self.transfer,
            needs_dither: self.needs_dither,
            layout: &self.layout,
        }
    }
}

impl Slot for PageTexture {
    fn dims(&self) -> (u32, u32) {
        self.texture_dims
    }

    fn page_info(&self) -> PageInfo {
        self.page_info
    }

    fn bytes(&self) -> usize {
        self.resident_bytes
    }
}

impl Drop for PageTexture {
    fn drop(&mut self) {
        // SAFETY: `Window` drops `PageRenderer` while the context is still
        // current and alive, and `self` owns these texture handles uniquely
        // (`None` slots point to shared stubs owned by `PageRenderer`).
        unsafe {
            match &self.layout {
                PixelLayout::Rgba { texture } => {
                    self.gl.delete_texture(*texture);
                }
                PixelLayout::Mono { luma, .. } => {
                    self.gl.delete_texture(*luma);
                }
                PixelLayout::Yuv(yuv) => {
                    self.gl.delete_texture(yuv.luma);
                    for texture in
                        [yuv.cb, yuv.cr, yuv.alpha].into_iter().flatten()
                    {
                        self.gl.delete_texture(texture);
                    }
                }
            }
        }
    }
}

impl From<TargetRole> for Role {
    fn from(role: TargetRole) -> Self {
        match role {
            TargetRole::Fit => Self::Fit,
            TargetRole::Full => Self::Full,
        }
    }
}

pub struct PageRenderer {
    gl: Arc<Context>,
    vao: EmptyVao,
    pipeline: Pipeline,
    ewa: Resampler,
    stubs: SharedStubs,
    cache: SlotTable<PageTexture>,
    max_texture_edge: u32,
}

impl PageRenderer {
    pub fn new(gl: Arc<Context>) -> eros::Result<Self> {
        // SAFETY: MAX_TEXTURE_SIZE is a pure query on the current context.
        let gl_max_edge =
            unsafe { gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE) };
        let Ok(max_texture_edge) = u32::try_from(gl_max_edge) else {
            eros::bail!("GL MAX_TEXTURE_SIZE is not a positive GLint");
        };
        if max_texture_edge < THUMB_EDGE {
            return Err(eros::error!(
                "GL MAX_TEXTURE_SIZE ({max_texture_edge}) is below THUMB_EDGE ({THUMB_EDGE})",
            ));
        }
        let vao = EmptyVao::new(Arc::clone(&gl), "page quad")?;
        let pipeline = create_pipeline(Arc::clone(&gl))?;
        let ewa = Resampler::new(Arc::clone(&gl), gl_max_edge)?;
        let stubs = SharedStubs::new(Arc::clone(&gl));
        Ok(Self {
            gl,
            vao,
            pipeline,
            ewa,
            stubs,
            cache: SlotTable::default(),
            max_texture_edge,
        })
    }

    #[must_use]
    pub const fn max_texture_edge(&self) -> u32 {
        self.max_texture_edge
    }

    #[must_use]
    pub const fn cache(&self) -> &SlotTable<PageTexture> {
        &self.cache
    }

    pub const fn cache_mut(&mut self) -> &mut SlotTable<PageTexture> {
        &mut self.cache
    }

    /// Upload `image` as `page`'s `role`; no-op if present, never evicts.
    pub fn insert_image(
        &mut self,
        page: usize,
        role: TargetRole,
        image: &Image,
    ) -> Result<InsertOutcome, TextureFault> {
        use std::time::Duration;
        use std::time::Instant;

        const SLOW_UPLOAD_THRESHOLD: Duration = Duration::from_millis(30);

        let slot_role = Role::from(role);
        if self.cache.has(page, slot_role) {
            return Ok(InsertOutcome::Kept);
        }
        let texture_dims = (image.width(), image.height());
        TextureFault::check_dims(texture_dims, self.max_texture_edge)?;
        let upload_started = Instant::now();
        let page_texture =
            upload_page_texture(Arc::clone(&self.gl), image);
        let elapsed = upload_started.elapsed();
        tracing::debug!(
            page,
            role = ?slot_role,
            width = texture_dims.0,
            height = texture_dims.1,
            ?elapsed,
            "uploaded"
        );
        if elapsed >= SLOW_UPLOAD_THRESHOLD {
            tracing::info!(
                page,
                role = ?slot_role,
                width = texture_dims.0,
                height = texture_dims.1,
                ?elapsed,
                "slow upload"
            );
        }
        label_entry(&self.gl, page, slot_role.into(), &page_texture);
        Ok(self.cache.insert(page, slot_role, page_texture))
    }

    /// Upload an 8-bit sRGB thumbnail to `page`'s `Role::Thumb` slot.
    pub fn insert_thumb(&mut self, page: usize, thumb: &ThumbImage) {
        if self.cache.has(page, Role::Thumb) {
            return;
        }
        debug_assert!(
            TextureFault::check_dims(thumb.dims, self.max_texture_edge)
                .is_ok(),
            "ThumbImage ({:?}) must fit max_texture_edge ({})",
            thumb.dims,
            self.max_texture_edge
        );
        let page_texture =
            upload_thumb_texture(Arc::clone(&self.gl), thumb);
        label_entry(&self.gl, page, Role::Thumb.into(), &page_texture);
        self.cache.insert(page, Role::Thumb, page_texture);
    }

    /// Updates `page`'s live playback slot in place when dimensions and format match.
    pub fn update_playback(
        &mut self,
        page: usize,
        image: &Image,
    ) -> Result<(), TextureFault> {
        let texture_dims = (image.width(), image.height());
        TextureFault::check_dims(texture_dims, self.max_texture_edge)?;
        if let Some(PageTexture {
            texture_dims: present_dims,
            layout: PixelLayout::Rgba { texture },
            transfer,
            ..
        }) = self.cache.playback(page)
            && *present_dims == texture_dims
            && *transfer == image.transfer()
            && let Pixels::Rgba8(bytes) = image.pixels()
        {
            update_rgba(&self.gl, *texture, bytes, texture_dims);
            return Ok(());
        }
        let page_texture =
            upload_page_texture(Arc::clone(&self.gl), image);
        label_entry(&self.gl, page, "playback", &page_texture);
        self.cache.set_playback(page, page_texture);
        Ok(())
    }

    /// Does not clear the framebuffer.
    pub fn draw(
        &mut self,
        window_size: (u32, u32),
        clip_size: (u32, u32),
        quads: &[DrawQuad],
    ) {
        let Some(target) = DrawTarget::new(window_size, clip_size) else {
            return;
        };
        program::begin_frame(&self.gl, self.vao.handle(), target);
        for quad in quads {
            if quad.size.0 <= 0.0 || quad.size.1 <= 0.0 {
                continue;
            }
            let Some(page_texture) =
                self.cache.best_slot(quad.page, quad.size)
            else {
                continue;
            };
            check_seat_page_dims(
                quad.page,
                quad.page_dims,
                page_texture.page_info.dims,
            );
            let seat = SeatGeometry {
                origin: quad.origin,
                size: quad.size,
            };
            let Some(slice) = ewa::compute_visible_slice(
                seat,
                target.clip,
                page_texture.texture_dims,
            ) else {
                continue;
            };
            let drew_ewa = slice.is_minified()
                && self.ewa.draw_minified(
                    target,
                    self.stubs.textures,
                    page_texture.as_draw_source(),
                    seat,
                    &slice,
                );
            if !drew_ewa {
                draw_page(
                    &self.gl,
                    &self.pipeline,
                    self.stubs.textures,
                    target,
                    page_texture,
                    seat,
                    slice.source_per_screen,
                );
            }
        }
        // SAFETY: Context is current; restore default FBO, unclipped
        // raster state, and texture unit 0 before egui paints.
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.gl.disable(glow::SCISSOR_TEST);
            self.gl.active_texture(glow::TEXTURE0);
        }
    }
}

fn check_seat_page_dims(
    page: usize,
    layout_dims: (u32, u32),
    texture_page_dims: (u32, u32),
) {
    if layout_dims != texture_page_dims {
        tracing::debug!(
            page,
            ?layout_dims,
            ?texture_page_dims,
            "seat layout disagrees with the texture's page dims"
        );
    }
    debug_assert_eq!(
        layout_dims, texture_page_dims,
        "seat layout and texture page dims disagree for page {page}"
    );
}

fn draw_page(
    gl: &Context,
    pipeline: &Pipeline,
    stubs: StubTextures,
    target: DrawTarget,
    page_texture: &PageTexture,
    seat: SeatGeometry,
    source_per_screen_px: (f64, f64),
) {
    // SAFETY: Context is current; `pipeline` and `page_texture` own live objects
    // and `begin_frame` bound the empty VAO.
    unsafe {
        gl.use_program(Some(pipeline.shader.program));
        pipeline.quad.bind(gl, target.window, seat);
        gl.uniform_2_f32(
            Some(&pipeline.page_uv_per_screen_px),
            program::gl_f32(1.0 / seat.size.0),
            program::gl_f32(1.0 / seat.size.1),
        );
        gl.uniform_1_i32(
            Some(&pipeline.dither),
            i32::from(page_texture.needs_dither),
        );
        gl.uniform_1_i32(
            Some(&pipeline.transfer_code),
            program::transfer_code(page_texture.transfer),
        );
        match &page_texture.layout {
            PixelLayout::Mono { luma, gain, offset } => {
                gl.uniform_1_i32(
                    Some(&pipeline.pixel_layout),
                    PIXEL_LAYOUT_MONO,
                );
                program::bind_mono_luma(
                    gl,
                    pipeline.luma_gain_offset,
                    *luma,
                    (*gain, *offset),
                );
            }
            PixelLayout::Rgba { texture } => {
                program::bind_rgba_source(
                    gl,
                    pipeline.pixel_layout,
                    *texture,
                );
            }
            PixelLayout::Yuv(yuv) => {
                pipeline.yuv.bind(
                    gl,
                    pipeline.pixel_layout,
                    yuv,
                    stubs,
                    source_per_screen_px,
                );
            }
        }
        gl.draw_arrays(
            glow::TRIANGLE_STRIP,
            FIRST_VERTEX_INDEX,
            QUAD_VERTEX_COUNT,
        );
    }
}

fn label_entry(
    gl: &Context,
    page: usize,
    slot_label: &str,
    page_texture: &PageTexture,
) {
    if !gl.supports_debug() {
        return;
    }
    match &page_texture.layout {
        PixelLayout::Rgba { texture } => program::label(
            gl,
            glow::TEXTURE,
            texture.0.get(),
            &format!("page {page} {slot_label} rgba"),
        ),
        PixelLayout::Mono { luma, .. } => program::label(
            gl,
            glow::TEXTURE,
            luma.0.get(),
            &format!("page {page} {slot_label} luma"),
        ),
        PixelLayout::Yuv(yuv) => {
            program::label(
                gl,
                glow::TEXTURE,
                yuv.luma.0.get(),
                &format!("page {page} {slot_label} luma"),
            );
            for (texture, name) in
                [(yuv.cb, "cb"), (yuv.cr, "cr"), (yuv.alpha, "alpha")]
            {
                if let Some(plane_texture) = texture {
                    program::label(
                        gl,
                        glow::TEXTURE,
                        plane_texture.0.get(),
                        &format!("page {page} {slot_label} {name}"),
                    );
                }
            }
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

fn magnification_filter(
    texture_dims: (u32, u32),
    page_dims: (u32, u32),
) -> MagnificationFilter {
    if texture_dims == page_dims {
        MagnificationFilter::Nearest
    } else {
        MagnificationFilter::Linear
    }
}

trait GlSample: bytemuck::Pod {
    const INTERNAL_FORMAT: u32;
    const PIXEL_TYPE: u32;
}

impl GlSample for u8 {
    const INTERNAL_FORMAT: u32 = glow::R8;
    const PIXEL_TYPE: u32 = glow::UNSIGNED_BYTE;
}

impl GlSample for u16 {
    const INTERNAL_FORMAT: u32 = glow::R16;
    const PIXEL_TYPE: u32 = glow::UNSIGNED_SHORT;
}

/// Dither hides precision loss from a PQ or >8-bit source; 8-bit sRGB
/// has none.
const fn needs_dither(transfer: Transfer, depth: Depth) -> bool {
    matches!(transfer, Transfer::Pq) || depth.is_wide()
}

fn upload_rgba(
    gl: &Context,
    data: &[u8],
    texture_dims: (u32, u32),
    magnification: MagnificationFilter,
) -> Texture {
    assert_eq!(
        data.len(),
        texture_dims.0 as usize
            * texture_dims.1 as usize
            * RGBA_BYTES_PER_PIXEL,
        "rgba byte length"
    );
    let texture = program::create_2d_texture(
        gl,
        glow::LINEAR,
        magnification.gl_filter(),
        None,
    );
    // SAFETY: `create_2d_texture` leaves `texture` bound on the current
    // context and `data` matches `texture_dims.0 * texture_dims.1 * 4` RGBA8 bytes.
    unsafe {
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, RGBA_UNPACK_ALIGNMENT);
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            BASE_MIP_LEVEL,
            glow::RGBA8.cast_signed(),
            program::gl_size(texture_dims.0),
            program::gl_size(texture_dims.1),
            NO_BORDER,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            PixelUnpackData::Slice(Some(data)),
        );
    }
    texture
}

fn update_rgba(
    gl: &Context,
    texture: Texture,
    data: &[u8],
    texture_dims: (u32, u32),
) {
    assert_eq!(
        data.len(),
        texture_dims.0 as usize
            * texture_dims.1 as usize
            * RGBA_BYTES_PER_PIXEL,
        "rgba byte length"
    );
    // SAFETY: `texture` is owned live by a `PageTexture` entry on the current
    // context, and `data` matches `texture_dims.0 * texture_dims.1 * 4` RGBA8 bytes.
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, RGBA_UNPACK_ALIGNMENT);
        gl.tex_sub_image_2d(
            glow::TEXTURE_2D,
            BASE_MIP_LEVEL,
            0,
            0,
            program::gl_size(texture_dims.0),
            program::gl_size(texture_dims.1),
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            PixelUnpackData::Slice(Some(data)),
        );
    }
}

fn upload_plane<Sample>(
    gl: &Context,
    samples: &[Sample],
    plane_dims: (u32, u32),
    magnification: MagnificationFilter,
) -> Texture
where
    Sample: GlSample,
{
    assert_eq!(
        samples.len(),
        plane_dims.0 as usize * plane_dims.1 as usize,
        "plane sample count"
    );
    let data = bytemuck::cast_slice(samples);
    let texture = program::create_2d_texture(
        gl,
        glow::LINEAR,
        magnification.gl_filter(),
        None,
    );
    // SAFETY: `create_2d_texture` leaves `texture` bound on the current
    // context and `data` matches `plane_dims.0 * plane_dims.1 * size_of::<Sample>()` bytes.
    unsafe {
        // Odd row widths: GL's default 4-byte unpack alignment would desync
        // every row after the first.
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, PLANE_UNPACK_ALIGNMENT);
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            BASE_MIP_LEVEL,
            Sample::INTERNAL_FORMAT.cast_signed(),
            program::gl_size(plane_dims.0),
            program::gl_size(plane_dims.1),
            NO_BORDER,
            glow::RED,
            Sample::PIXEL_TYPE,
            PixelUnpackData::Slice(Some(data)),
        );
    }
    texture
}

fn upload_gray_layout<Sample>(
    gl: &Context,
    depth: Depth,
    full_range: bool,
    [luma]: &[Box<[Sample]>; 1],
    alpha: Option<&[Sample]>,
    texture_dims: (u32, u32),
    magnification: MagnificationFilter,
) -> PixelLayout
where
    Sample: GlSample,
{
    const UNUSED_CHROMA_UV_PER_LUMA_TEXEL: (f64, f64) = (1.0, 1.0);

    let luma_texture = upload_plane(gl, luma, texture_dims, magnification);
    let (gain, offset) = color::gray_gain_offset(depth, full_range);
    alpha.map_or_else(
        || PixelLayout::Mono {
            luma: luma_texture,
            gain,
            offset,
        },
        |alpha| {
            // `EWA_MONO` packs `(gray2, weight2)` with no alpha channel,
            // so gray+alpha rides the YUV path with shared neutral chroma stubs.
            PixelLayout::Yuv(YuvTexture {
                luma: luma_texture,
                cb: None,
                cr: None,
                alpha: Some(upload_plane(
                    gl,
                    alpha,
                    texture_dims,
                    magnification,
                )),
                chroma_uv_per_luma_texel: UNUSED_CHROMA_UV_PER_LUMA_TEXEL,
                color_conversion: YuvMatrix::gray(gain, offset),
                alpha_normalization: color::depth_normalization(depth),
            })
        },
    )
}

fn upload_yuv_layout<Sample>(
    gl: &Context,
    depth: Depth,
    spec: YuvSpec,
    [luma, cb, cr]: &[Box<[Sample]>; 3],
    alpha: Option<&[Sample]>,
    texture_dims: (u32, u32),
    magnification: MagnificationFilter,
) -> PixelLayout
where
    Sample: GlSample,
{
    let ((luma_width, luma_height), (chroma_width, chroma_height)) =
        pixels::plane_dims(texture_dims, spec.subsampling);
    let luma_texture =
        upload_plane(gl, luma, (luma_width, luma_height), magnification);
    let color_conversion =
        color::yuv_color_conversion(depth, spec.full_range, spec.matrix);
    let (alpha_texture, alpha_normalization) = alpha.map_or_else(
        || (None, color::depth_normalization(Depth::D8)),
        |alpha| {
            (
                Some(upload_plane(gl, alpha, texture_dims, magnification)),
                color::depth_normalization(depth),
            )
        },
    );
    let chroma_texels =
        (f64::from(chroma_width), f64::from(chroma_height));
    let chroma_ratio = spec.subsampling.chroma_ratio();
    PixelLayout::Yuv(YuvTexture {
        luma: luma_texture,
        cb: Some(upload_plane(
            gl,
            cb,
            (chroma_width, chroma_height),
            MagnificationFilter::Linear,
        )),
        cr: Some(upload_plane(
            gl,
            cr,
            (chroma_width, chroma_height),
            MagnificationFilter::Linear,
        )),
        alpha: alpha_texture,
        chroma_uv_per_luma_texel: (
            chroma_ratio.0 / chroma_texels.0,
            chroma_ratio.1 / chroma_texels.1,
        ),
        color_conversion,
        alpha_normalization,
    })
}

fn upload_page_texture(gl: Arc<Context>, image: &Image) -> PageTexture {
    let texture_dims = (image.width(), image.height());
    let page_dims = image.page_dims();
    let magnification = magnification_filter(texture_dims, page_dims);
    let layout = match image.pixels() {
        Pixels::Rgba8(rgba_bytes) => PixelLayout::Rgba {
            texture: upload_rgba(
                &gl,
                rgba_bytes,
                texture_dims,
                magnification,
            ),
        },
        Pixels::Gray { full_range, planes } => {
            let depth = planes.depth();
            match planes {
                SamplePlanes::U8 { planes, alpha } => upload_gray_layout(
                    &gl,
                    depth,
                    *full_range,
                    planes,
                    alpha.as_deref(),
                    texture_dims,
                    magnification,
                ),
                SamplePlanes::U16 { planes, alpha, .. } => {
                    upload_gray_layout(
                        &gl,
                        depth,
                        *full_range,
                        planes,
                        alpha.as_deref(),
                        texture_dims,
                        magnification,
                    )
                }
            }
        }
        Pixels::Yuv { spec, planes } => {
            let depth = planes.depth();
            match planes {
                SamplePlanes::U8 { planes, alpha } => upload_yuv_layout(
                    &gl,
                    depth,
                    *spec,
                    planes,
                    alpha.as_deref(),
                    texture_dims,
                    magnification,
                ),
                SamplePlanes::U16 { planes, alpha, .. } => {
                    upload_yuv_layout(
                        &gl,
                        depth,
                        *spec,
                        planes,
                        alpha.as_deref(),
                        texture_dims,
                        magnification,
                    )
                }
            }
        }
    };
    PageTexture {
        gl,
        texture_dims,
        page_info: image.page_info(),
        transfer: image.transfer(),
        needs_dither: needs_dither(
            image.transfer(),
            image.pixels().depth(),
        ),
        resident_bytes: image.resident_bytes(),
        layout,
    }
}

fn upload_thumb_texture(
    gl: Arc<Context>,
    thumb: &ThumbImage,
) -> PageTexture {
    let magnification = magnification_filter(thumb.dims, thumb.info.dims);
    let texture = upload_rgba(&gl, &thumb.rgba, thumb.dims, magnification);
    PageTexture {
        gl,
        texture_dims: thumb.dims,
        page_info: thumb.info,
        transfer: Transfer::Srgb,
        needs_dither: false,
        resident_bytes: thumb.rgba.len(),
        layout: PixelLayout::Rgba { texture },
    }
}
