//! GL rendering: the page quad renderer.
//!
//! GL is used from the main thread only and `Window` makes the GL context
//! current once, so the `# Safety` docs below presuppose a current context.
//!
//! Runtime texture-creation failure panics. Recovery is deliberately
//! not built. An upstream `egui_glow` unwrap can panic at runtime
//! anyway, so recovery cannot be total. The failure is also rare on
//! target hardware.
//!
//! Chroma is sampled bilinearly at centered (interstitial) positions,
//! the JPEG convention and the AVIF default. avifChromaSamplePosition is
//! not applied, so declared co-sited chroma biases by half a texel.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

use egui_glow::glow;
use egui_glow::glow::HasContext;

use crate::decode::plane_bytes;
use crate::decode::plane_dims;
use crate::decode::Decoded;
use crate::decode::Matrix;
use crate::decode::Pixels;
use crate::decode::Plane;
use crate::decode::Subsampling;
use crate::decode::Transfer;
use crate::view;
use crate::view::Placement;
use crate::view::Seat;
use crate::view::SpreadLayout;
use crate::view::Viewport;

/// Both dialect variants of one shader source, so the pair cannot drift:
/// each variant is its version preamble concatenated with the shared body.
macro_rules! shader_variants {
    ($core:ident, $es:ident, $core_preamble:literal, $es_preamble:literal, $body:literal) => {
        const $core: &str = concat!($core_preamble, $body);
        const $es: &str = concat!($es_preamble, $body);
    };
}

// Fixed -1..1 strip. The VS places it at the seat's screen rect from
// uniforms, so pan/zoom never touch vertex data. Top-origin math lives in
// the VS, because GL clip space is bottom-origin while winit coords are
// top-origin.
shader_variants!(
    VS_CORE,
    VS_ES,
    "#version 330 core\n",
    "#version 300 es\n",
    "
    in vec2 pos;
    uniform vec2 u_win;    // window size, px
    uniform vec2 u_offset; // seat rect origin, screen px
    uniform vec2 u_size;   // seat rect size, screen px
    out vec2 v_px;
    void main() {
        vec2 px = u_offset + (pos * 0.5 + 0.5) * u_size;
        v_px = px;
        gl_Position = vec4(px.x / u_win.x * 2.0 - 1.0, 1.0 - px.y / u_win.y * 2.0, 0.0, 1.0);
    }"
);

// highp is required: mediump may be fp16, where the ulp at 16k image px is
// 8.0 and wrecks the coordinate math. ES3 mandates highp support. The
// samplers declare highp too, because ES defaults them to lowp and strict
// hardware quantizes >8-bit samples to ~8 bits through the return path.
//
// One shader, both pixel layouts and both transfers: mode and transfer
// arrive as uniforms, so there are no per-variant programs. Every
// per-draw constant (the coordinate maps, the depth normalization, the
// quantization anchors) is folded CPU-side into `u_lmap`/`u_cmap`,
// `u_mat`, and `u_off`, leaving the color path one matrix multiply and
// one subtract per fragment. Output is premultiplied for compositing
// over the background.
shader_variants!(
    FS_CORE,
    FS_ES,
    "#version 330 core\n",
    "#version 300 es\nprecision highp float;\nprecision highp sampler2D;\n",
    "
    in vec2 v_px;
    out vec4 color;
    uniform vec2 u_offset;   // page rect origin, screen px
    uniform vec2 u_lmap;     // screen px -> luma texel: 1 / (scale * image size)
    uniform vec2 u_cmap;     // screen px -> chroma texel, subsample-aware
    uniform mat3 u_mat;      // storage samples -> gamma RGB
    uniform vec3 u_off;      // u_mat times the quantization anchors
    uniform float u_norm;    // depth normalize of the alpha sample; 1.0 for 8-bit storage
    uniform int u_mode;      // 0 = RGBA, 1 = YUV
    uniform int u_transfer;  // 0 = sRGB passthrough, 1 = PQ tone-mapped to SDR
    uniform sampler2D u_tex0; // RGBA or luma
    uniform sampler2D u_tex1; // Cb (1x1 midpoint for mono)
    uniform sampler2D u_tex2; // Cr
    uniform sampler2D u_tex3; // alpha (1x1 opaque when the page carries none)
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
    vec3 display(vec3 x) {
        if (u_transfer == 0) {
            return x;
        }
        vec3 p = pow(clamp(x, 0.0, 1.0), vec3(1.0 / PQ_M2));
        vec3 lum = PQ_PEAK * pow(max(p - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * p), vec3(1.0 / PQ_M1));
        vec3 sdr = clamp(lum / PQ_DIFFUSE_WHITE, 0.0, 1.0);
        // Linear light to the sRGB curve: the 12.92 slope below the
        // 0.0031308 threshold, else 1.055 * s^(1/2.4) - 0.055.
        return mix(sdr * 12.92, 1.055 * pow(sdr, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), sdr));
    }
    float bayer4(vec2 frag) {
        float tiles[16] = float[16](
            0.0, 8.0, 2.0, 10.0,
            12.0, 4.0, 14.0, 6.0,
            3.0, 11.0, 1.0, 9.0,
            15.0, 7.0, 13.0, 5.0
        );
        return tiles[int(mod(frag.x, 4.0)) + 4 * int(mod(frag.y, 4.0))] / 16.0;
    }
    void main() {
        // Fragment centers land on texel centers at 1:1 zoom with
        // whole-pixel origins (px = k + 0.5), so the linear maps already
        // address texel centers with no half-texel offset.
        vec2 rel = v_px - u_offset;
        // Ordered dither at half an output quantum, applied to the
        // straight color, hides the deep->8-bit contouring of gradients.
        float dither = (bayer4(gl_FragCoord.xy) - 0.5) / 255.0;
        if (u_mode == 0) {
            vec4 px = texture(u_tex0, rel * u_lmap);
            color = vec4((display(px.rgb) + dither) * px.a, px.a);
            return;
        }
        vec3 yuv = vec3(texture(u_tex0, rel * u_lmap).r,
                        texture(u_tex1, rel * u_cmap).r,
                        texture(u_tex2, rel * u_cmap).r);
        float a = texture(u_tex3, rel * u_lmap).r * u_norm;
        color = vec4((display(u_mat * yuv - u_off) + dither) * a, a);
    }"
);

/// Page-keyed texture cache plus the spread draw. Each entry is an RGBA
/// texture or a YUV plane triple.
pub struct PageTextures {
    gl: Arc<glow::Context>,
    pipeline: Pipeline,
    seats: [Option<usize>; 2],
    textures: HashMap<usize, Entry>,
    max_texture_edge: u32,
}

/// Uniform locations, resolved once at link time so draws never look
/// names up by string. Every name is declared and statically used by
/// this file's in-tree shaders, so a missing location is an in-tree
/// regression and fails pipeline creation instead of silently skipping
/// per-draw state.
struct Locations {
    win: <glow::Context as glow::HasContext>::UniformLocation,
    offset: <glow::Context as glow::HasContext>::UniformLocation,
    size: <glow::Context as glow::HasContext>::UniformLocation,
    lmap: <glow::Context as glow::HasContext>::UniformLocation,
    cmap: <glow::Context as glow::HasContext>::UniformLocation,
    mat: <glow::Context as glow::HasContext>::UniformLocation,
    off: <glow::Context as glow::HasContext>::UniformLocation,
    norm: <glow::Context as glow::HasContext>::UniformLocation,
    mode: <glow::Context as glow::HasContext>::UniformLocation,
    transfer: <glow::Context as glow::HasContext>::UniformLocation,
}

/// One cached page's GPU state, freed by budget eviction and on drop.
/// The variant is the pixel layout, so an entry carries exactly the
/// textures and folded maps its draw path samples: an RGBA entry has
/// no chroma map or conversion to leave stale. Every per-draw constant
/// is folded here at upload, so the frame path only divides by the
/// live zoom scale.
enum Entry {
    /// Single RGBA texture (alpha in its fourth channel).
    Rgba {
        texture: <glow::Context as glow::HasContext>::Texture,
        /// Screen px -> texel, before the zoom-scale divide.
        lmap_base: (f32, f32),
        transfer: Transfer,
        cost: usize,
    },
    /// Luma + Cb + Cr + alpha plane textures. Mono binds 1x1 midpoint
    /// chroma; a page without alpha binds a 1x1 opaque stub.
    Yuv {
        luma: <glow::Context as glow::HasContext>::Texture,
        cb: <glow::Context as glow::HasContext>::Texture,
        cr: <glow::Context as glow::HasContext>::Texture,
        alpha: <glow::Context as glow::HasContext>::Texture,
        /// Screen px -> luma texel, before the zoom-scale divide.
        lmap_base: (f32, f32),
        /// Screen px -> chroma texel, subsample-folded, before the divide.
        cmap_base: (f32, f32),
        /// Storage samples -> gamma RGB, quantization folded.
        conversion: Conversion,
        /// Depth-normalize factor for the alpha sample (1.0 for 8-bit
        /// storage).
        norm: f32,
        transfer: Transfer,
        cost: usize,
    },
}

impl Entry {
    /// Approximate GPU bytes, counted against [`TEXTURE_BUDGET_BYTES`].
    #[expect(
        clippy::rest_pattern_accessible_field,
        reason = "both variants carry the budget figure in the same slot, so the others stay unnamed"
    )]
    fn cost(&self) -> usize {
        match self {
            Self::Rgba { cost, .. } | Self::Yuv { cost, .. } => *cost,
        }
    }
}

/// One page's storage-sample-to-RGB math, folded at upload:
/// `rgb = matrix * samples - offset`, with `samples` in 0..1 storage
/// units (R16 samplers already divide by 65535).
#[derive(Debug, Clone, Copy)]
struct Conversion {
    /// Column-major 3x3.
    matrix: [f32; 9],
    offset: [f32; 3],
}

impl PageTextures {
    pub fn new(gl: Arc<glow::Context>) -> eros::Result<Self> {
        let (vs, fs) = if gl.version().is_embedded { (VS_ES, FS_ES) } else { (VS_CORE, FS_CORE) };
        // SAFETY: the returned names land in Self, whose Drop deletes
        // them. The error path below leaks them into the process-fatal
        // exit.
        let pipeline = unsafe { create_pipeline(&gl, vs, fs) }?;
        #[expect(
            clippy::map_err_ignore,
            reason = "the dropped conversion error says only that the value is out of range, which the message states"
        )]
        // SAFETY: MAX_TEXTURE_SIZE is a pure query with no object names.
        let max_texture_edge = u32::try_from(unsafe { gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE) })
            .map_err(|_| eros::error!("GL MAX_TEXTURE_SIZE is not a positive GLint"))?;
        Ok(Self {
            gl,
            pipeline,
            seats: [None, None],
            textures: HashMap::new(),
            max_texture_edge,
        })
    }

    /// Record that `seat` draws `page`, uploading its textures only on a
    /// cache miss. A bound page is never evicted, so a held seat keeps
    /// drawing. An upload the GPU refuses leaves the seat unbound
    /// (nothing draws) and returns as the page's error; the next bind
    /// retries, so the error holds exactly while the page stays wanted.
    pub fn bind(&mut self, seat: Seat, page: usize, decoded: &Decoded) -> eros::Result<()> {
        if *self.seat_slot(seat) == Some(page) {
            return Ok(());
        }
        if let Err(error) = self.ensure_cached(page, decoded) {
            self.clear(seat);
            return Err(error);
        }
        *self.seat_slot(seat) = Some(page);
        Ok(())
    }

    /// Ensure `page`'s textures are cached without binding any seat, so
    /// the frame that later shows the page is a pure rebind.
    pub fn stage(&mut self, page: usize, decoded: &Decoded) -> eros::Result<()> {
        self.ensure_cached(page, decoded)
    }

    /// Upload `page` into the cache on a miss. A page over this context's
    /// `MAX_TEXTURE_SIZE` would upload as silently black, so it is refused
    /// as the page's error instead.
    fn ensure_cached(&mut self, page: usize, decoded: &Decoded) -> eros::Result<()> {
        if self.textures.contains_key(&page) {
            return Ok(());
        }
        let (width, height) = (decoded.width, decoded.height);
        if width == 0 || height == 0 || width > self.max_texture_edge || height > self.max_texture_edge {
            return Err(eros::error!(
                "page {}: {}x{} exceeds this GPU's {}-texel texture edge limit",
                page + 1,
                width,
                height,
                self.max_texture_edge
            ));
        }
        // SAFETY: the size check above keeps the page inside this
        // context's texture limit, and `upload_entry` asserts its own
        // plane byte extents.
        let entry = unsafe { upload_entry(&self.gl, decoded) };
        self.textures.insert(page, entry);
        self.evict_over_budget();
        Ok(())
    }

    /// Stop drawing `seat`. The page's textures stay cached.
    pub fn clear(&mut self, seat: Seat) {
        *self.seat_slot(seat) = None;
    }

    /// Drop every cached page and unbind both seats. A book switch
    /// renumbers pages from zero, so nothing of the old book can stay.
    pub fn clear_all(&mut self) {
        for entry in mem::take(&mut self.textures).into_values() {
            // SAFETY: the take empties the map, so this loop holds the
            // only reference to each entry.
            unsafe { delete_entry(&self.gl, &entry); }
        }
        self.seats = [None, None];
    }

    fn seat_slot(&mut self, seat: Seat) -> &mut Option<usize> {
        let [left, right] = &mut self.seats;
        match seat {
            Seat::Left => left,
            Seat::Right => right,
        }
    }

    /// Drop cached pages until the map fits [`TEXTURE_BUDGET_BYTES`].
    fn evict_over_budget(&mut self) {
        let victims = eviction_victims(
            self.textures.iter().map(|(page, entry)| (*page, entry.cost())),
            self.seats,
            TEXTURE_BUDGET_BYTES,
        );
        self.textures.retain(|page, entry| {
            if victims.contains(page) {
                // SAFETY: the entry leaves the map in this same step, so this is its only deletion.
                unsafe { delete_entry(&self.gl, entry); }
                false
            } else {
                true
            }
        });
    }

    /// Draw the spread into `viewport`, laid out by `layout` with
    /// `placement`. Scissor-clipped to the viewport, so pages never
    /// render under chrome even when panned past it.
    ///
    /// Does not clear. The app owns the framebuffer clear each frame.
    pub fn draw(
        &self,
        size: (u32, u32),
        viewport: Viewport,
        layout: &SpreadLayout,
        placement: &Placement,
    ) {
        if size.0 == 0 || size.1 == 0 {
            return;
        }
        // SAFETY: the pipeline is Self's field, live until Drop.
        unsafe { begin_frame(&self.gl, &self.pipeline, size); }
        // GL scissor is bottom-origin and the viewport hugs the window's
        // top-left, so only the height converts.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "viewport size is whole physical px carried as f32"
        )]
        let (width, height) = (viewport.size().0 as u32, viewport.size().1 as u32);
        // SAFETY: the context is current. This only toggles scissor state.
        unsafe { self.gl.enable(glow::SCISSOR_TEST); }
        // SAFETY: same contract as above, setting the scissor box.
        unsafe {
            self.gl.scissor(
                0,
                gl_int(size.1.saturating_sub(height)),
                gl_int(width),
                gl_int(height),
            );
        }
        for (rect, page) in layout.rects.iter().zip(self.seats.iter()) {
            let Some(rect) = rect else { continue };
            let Some(page) = page else { continue };
            let Some(entry) = self.textures.get(page) else { continue };
            // Whole-pixel rect origin: at 1:1 zoom a fractional origin
            // makes NEAREST draw uneven texel rows.
            let screen_origin = (
                rect.pos.0.mul_add(placement.scale, placement.offset.0).round(),
                rect.pos.1.mul_add(placement.scale, placement.offset.1).round(),
            );
            let screen_size = (rect.size.0 * placement.scale, rect.size.1 * placement.scale);
            if screen_size.0 <= 0.0 || screen_size.1 <= 0.0 {
                continue;
            }
            // SAFETY: &self pins the entry, so its texture names outlive the call.
            unsafe {
                draw_page(
                    &self.gl,
                    &self.pipeline,
                    entry,
                    screen_origin,
                    screen_size,
                    placement.scale,
                );
            }
        }
        // SAFETY: the context is current. This restores unclipped raster
        // state.
        unsafe { self.gl.disable(glow::SCISSOR_TEST); }
    }
}

impl Drop for PageTextures {
    fn drop(&mut self) {
        // GL objects must be deleted while their context is alive. The
        // window that owns the context is dropped after this struct.
        for entry in mem::take(&mut self.textures).into_values() {
            // SAFETY: the take empties the map, so this loop holds the
            // only reference to each entry.
            unsafe { delete_entry(&self.gl, &entry); }
        }
        // SAFETY: the context outlives this drop. These names are
        // deleted once, here.
        unsafe { delete_pipeline(&self.gl, &self.pipeline); }
    }
}

/// Compile `src` as `kind`, failing with the GL info log.
///
/// # Safety
///
/// The context must be current. The caller must delete the returned name.
unsafe fn compile_shader(
    gl: &glow::Context,
    kind: u32,
    src: &str,
) -> eros::Result<<glow::Context as HasContext>::Shader> {
    let shader = gl
        .create_shader(kind)
        .map_err(|error| eros::error!("GL shader object creation failed: {error}"))?;
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

/// The linked program and its fixed companions. The names are owned by
/// [`PageTextures`] and deleted by its Drop.
struct Pipeline {
    program: <glow::Context as HasContext>::Program,
    locations: Locations,
    vao: <glow::Context as glow::HasContext>::VertexArray,
    vbo: <glow::Context as HasContext>::Buffer,
}

/// Create the linked program with its cached uniform locations, the
/// vertex buffer, and the VAO wiring. Both shaders are deleted once the
/// link succeeds. Objects left behind by a failed step leak into the
/// process-fatal error path (module policy).
///
/// # Safety
///
/// The context must be current. The returned names are live until deleted.
unsafe fn create_pipeline(gl: &glow::Context, vs: &str, fs: &str) -> eros::Result<Pipeline> {
    let program = gl
        .create_program()
        .map_err(|error| eros::error!("GL program object creation failed: {error}"))?;
    let vbo = gl
        .create_buffer()
        .map_err(|error| eros::error!("GL buffer object creation failed: {error}"))?;
    let vao = gl
        .create_vertex_array()
        .map_err(|error| eros::error!("GL vertex array object creation failed: {error}"))?;

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

    gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    gl.buffer_data_u8_slice(
        glow::ARRAY_BUFFER,
        bytemuck::cast_slice(&[-1.0_f32, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
        glow::STATIC_DRAW,
    );

    gl.bind_vertex_array(Some(vao));
    #[expect(
        clippy::expect_used,
        reason = "the attribute is declared and used in this file's in-tree shader sources, so failure means an in-tree regression"
    )]
    let loc = gl.get_attrib_location(program, "pos").expect("pos attribute in page shader");
    gl.enable_vertex_attrib_array(loc);
    gl.vertex_attrib_pointer_f32(loc, 2, glow::FLOAT, false, 8, 0);

    // Sampler units are fixed. Draw rebinds textures only. A silently
    // skipped sampler uniform would point every unit at 0, so a missing
    // location fails creation.
    gl.use_program(Some(program));
    let location = |name: &str| {
        #[expect(
            clippy::expect_used,
            reason = "the uniform is declared and statically used by this file's in-tree shader sources, so a missing location means an in-tree regression"
        )]
        gl.get_uniform_location(program, name).expect("uniform location in page shader")
    };
    for (unit, name) in [(0_i32, "u_tex0"), (1, "u_tex1"), (2, "u_tex2"), (3, "u_tex3")] {
        gl.uniform_1_i32(Some(&location(name)), unit);
    }
    let locations = Locations {
        win: location("u_win"),
        offset: location("u_offset"),
        size: location("u_size"),
        lmap: location("u_lmap"),
        cmap: location("u_cmap"),
        mat: location("u_mat"),
        off: location("u_off"),
        norm: location("u_norm"),
        mode: location("u_mode"),
        transfer: location("u_transfer"),
    };

    Ok(Pipeline { program, locations, vao, vbo })
}

/// Set the frame state the page draws depend on: the viewport, the
/// premultiplied compositing pages draw with, and the window-size
/// uniform. egui's own paint re-sets its blend state afterwards.
///
/// # Safety
///
/// The context must be current. `pipeline`'s names are live.
unsafe fn begin_frame(gl: &glow::Context, pipeline: &Pipeline, size: (u32, u32)) {
    gl.viewport(0, 0, gl_int(size.0), gl_int(size.1));
    gl.enable(glow::BLEND);
    gl.blend_func_separate(
        glow::ONE,
        glow::ONE_MINUS_SRC_ALPHA,
        glow::ONE,
        glow::ONE_MINUS_SRC_ALPHA,
    );
    gl.use_program(Some(pipeline.program));
    gl.bind_vertex_array(Some(pipeline.vao));
    let win = view::px_f32(size);
    gl.uniform_2_f32(Some(&pipeline.locations.win), win.0, win.1);
}

/// `u_mode` value of [`Entry::Rgba`]: the fragment shader samples one
/// RGBA texture.
const MODE_RGBA: i32 = 0;

/// `u_mode` value of [`Entry::Yuv`]: the fragment shader samples the
/// plane set.
const MODE_YUV: i32 = 1;

/// Draw one seat's quad. Every uniform is per-draw constant except the
/// coordinate maps' zoom-scale divide. The shader's `u_mode` branch
/// guards the mode-specific uniforms, so each variant sets only its
/// branch reads and the unused ones stay stale without effect.
///
/// # Safety
///
/// The context must be current. `pipeline`'s names and `entry`'s texture
/// names are live for the duration of the call.
#[expect(
    clippy::rest_pattern_accessible_field,
    reason = "the draw path names the state it samples; the budget figure needs no binding"
)]
unsafe fn draw_page(
    gl: &glow::Context,
    pipeline: &Pipeline,
    entry: &Entry,
    screen_origin: (f32, f32),
    screen_size: (f32, f32),
    scale: f32,
) {
    let locations = &pipeline.locations;
    gl.use_program(Some(pipeline.program));
    gl.uniform_2_f32(Some(&locations.offset), screen_origin.0, screen_origin.1);
    gl.uniform_2_f32(Some(&locations.size), screen_size.0, screen_size.1);
    gl.active_texture(glow::TEXTURE0);
    match entry {
        Entry::Rgba { texture, lmap_base, transfer, .. } => {
            gl.uniform_1_i32(Some(&locations.mode), MODE_RGBA);
            gl.uniform_1_i32(Some(&locations.transfer), transfer_id(*transfer));
            gl.uniform_2_f32(
                Some(&locations.lmap),
                lmap_base.0 / scale,
                lmap_base.1 / scale,
            );
            gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
        }
        Entry::Yuv {
            luma,
            cb,
            cr,
            alpha,
            lmap_base,
            cmap_base,
            conversion,
            norm,
            transfer,
            ..
        } => {
            gl.uniform_1_i32(Some(&locations.mode), MODE_YUV);
            gl.uniform_1_i32(Some(&locations.transfer), transfer_id(*transfer));
            gl.uniform_2_f32(
                Some(&locations.lmap),
                lmap_base.0 / scale,
                lmap_base.1 / scale,
            );
            gl.uniform_2_f32(
                Some(&locations.cmap),
                cmap_base.0 / scale,
                cmap_base.1 / scale,
            );
            gl.uniform_1_f32(Some(&locations.norm), *norm);
            gl.uniform_matrix_3_f32_slice(Some(&locations.mat), false, &conversion.matrix);
            gl.uniform_3_f32(
                Some(&locations.off),
                conversion.offset[0],
                conversion.offset[1],
                conversion.offset[2],
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

/// Delete the pipeline objects [`create_pipeline`] returned.
///
/// # Safety
///
/// The context must be current. Each name is live and deleted at most once.
unsafe fn delete_pipeline(gl: &glow::Context, pipeline: &Pipeline) {
    gl.delete_program(pipeline.program);
    gl.delete_vertex_array(pipeline.vao);
    gl.delete_buffer(pipeline.vbo);
}

/// Delete an entry's texture objects.
///
/// # Safety
///
/// The context must be current. The names are live and deleted at most once.
#[expect(
    clippy::rest_pattern_accessible_field,
    reason = "deletion touches the texture names only, so the folded draw state stays unnamed"
)]
unsafe fn delete_entry(gl: &glow::Context, entry: &Entry) {
    match entry {
        Entry::Rgba { texture, .. } => gl.delete_texture(*texture),
        Entry::Yuv { luma, cb, cr, alpha, .. } => {
            gl.delete_texture(*luma);
            gl.delete_texture(*cb);
            gl.delete_texture(*cr);
            gl.delete_texture(*alpha);
        }
    }
}

#[expect(
    clippy::cast_possible_wrap,
    reason = "GL enums, 16384-capped texture edges, and window px all sit far below i32. This is the GL ABI's u32-enum to i32-parameter seam"
)]
fn gl_int(value: u32) -> i32 {
    value as i32
}

/// Magnification policy for one texture.
#[derive(Clone, Copy)]
enum Mag {
    /// Zoomed-in texels stay sharp squares.
    Sharp,
    /// Bilinear interpolation between texels.
    Smooth,
}

impl Mag {
    fn gl(self) -> u32 {
        match self {
            Self::Sharp => glow::NEAREST,
            Self::Smooth => glow::LINEAR,
        }
    }
}

/// Texture with viewer defaults. CLAMP keeps edge texels from bleeding.
/// `filterable` false (GLES R16) means NEAREST and no mips.
///
/// # Safety
///
/// The context must be current. The returned name is live until deleted.
unsafe fn new_texture(gl: &glow::Context, filterable: bool, mag: Mag) -> <glow::Context as HasContext>::Texture {
    #[expect(
        clippy::expect_used,
        reason = "texture-creation failure panics by policy"
    )]
    let texture = gl.create_texture().expect("GL texture object");
    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, gl_int(glow::CLAMP_TO_EDGE));
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, gl_int(glow::CLAMP_TO_EDGE));
    let (min, mag) = if filterable {
        // A fit view of a 50MP page minifies ~8x, where LINEAR-only shimmers.
        (glow::LINEAR_MIPMAP_LINEAR, mag.gl())
    } else {
        (glow::NEAREST, glow::NEAREST)
    };
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, gl_int(min));
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, gl_int(mag));
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
) -> <glow::Context as HasContext>::Texture {
    let texture = new_texture(gl, true, Mag::Sharp);
    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
    gl.tex_image_2d(
        glow::TEXTURE_2D, 0, gl_int(glow::RGBA8),
        gl_int(dims.0), gl_int(dims.1), 0,
        glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(data)),
    );
    gl.generate_mipmap(glow::TEXTURE_2D);
    texture
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
    filterable: bool,
    mag: Mag,
) -> <glow::Context as HasContext>::Texture {
    let (internal, ty) = if wide { (glow::R16, glow::UNSIGNED_SHORT) } else { (glow::R8, glow::UNSIGNED_BYTE) };
    let texture = new_texture(gl, filterable, mag);
    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
    // Odd row widths: GL's default 4-byte unpack alignment would desync
    // every row after the first.
    gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
    gl.tex_image_2d(
        glow::TEXTURE_2D, 0, gl_int(internal),
        gl_int(dims.0), gl_int(dims.1), 0,
        glow::RED, ty, glow::PixelUnpackData::Slice(Some(data)),
    );
    if filterable {
        gl.generate_mipmap(glow::TEXTURE_2D);
    }
    texture
}

/// Upload one page's textures plus the folded draw state that goes with
/// them.
///
/// # Safety
///
/// The context must be current. The asserts inside enforce the plane
/// byte-length precondition of the uploads.
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are the YUV plane names the two textures hold"
)]
unsafe fn upload_entry(gl: &glow::Context, decoded: &Decoded) -> Entry {
    let (width, height) = (decoded.width, decoded.height);
    match &decoded.pixels {
        Pixels::Rgba8(rgba_bytes) => {
            assert_eq!(rgba_bytes.len(), width as usize * height as usize * 4, "rgba byte length");
            Entry::Rgba {
                texture: upload_rgba(gl, rgba_bytes, (width, height)),
                lmap_base: lmap_base((width, height)),
                transfer: decoded.transfer,
                cost: entry_cost_bytes(decoded, true),
            }
        }
        Pixels::Yuv(yuv) => {
            let ((luma_w, luma_h), (chroma_w, chroma_h)) =
                plane_dims((width, height), yuv.subsampling);
            let wide = matches!(yuv.luma, Plane::U16(_));
            // GL_R16 is not filterable in GLES. LINEAR makes the texture
            // incomplete (samples black).
            let filterable = !(gl.version().is_embedded && wide);
            let depth = yuv.depth.clamp(MIN_PLANE_DEPTH, MAX_PLANE_DEPTH);
            let expect = |plane: &Plane, want_samples: usize, what: &str| {
                let want_bytes = want_samples * if wide { 2 } else { 1 };
                assert_eq!(plane.byte_len(), want_bytes, "{what} plane bytes");
            };
            expect(&yuv.luma, luma_w as usize * luma_h as usize, "luma");
            let luma_texture =
                upload_plane(gl, plane_bytes(&yuv.luma), (luma_w, luma_h), wide, filterable, Mag::Sharp);
            let (cb_texture, cr_texture) = if yuv.subsampling == Subsampling::Mono {
                // 1x1 chroma at the neutral code, which the folded
                // quantization anchor cancels exactly.
                let neutral = stub_plane(depth, 1_u32 << (depth - 1));
                (upload_plane(gl, plane_bytes(&neutral), (1, 1), wide, filterable, Mag::Smooth),
                 upload_plane(gl, plane_bytes(&neutral), (1, 1), wide, filterable, Mag::Smooth))
            } else {
                expect(&yuv.cb, chroma_w as usize * chroma_h as usize, "cb");
                expect(&yuv.cr, chroma_w as usize * chroma_h as usize, "cr");
                (upload_plane(gl, plane_bytes(&yuv.cb), (chroma_w, chroma_h), wide, filterable, Mag::Smooth),
                 upload_plane(gl, plane_bytes(&yuv.cr), (chroma_w, chroma_h), wide, filterable, Mag::Smooth))
            };
            let alpha_texture = yuv.alpha.as_ref().map_or_else(
                // 1x1 opaque: the max code, which `u_norm` maps to 1.0.
                || upload_plane(gl, plane_bytes(&stub_plane(depth, (1_u32 << depth) - 1)), (1, 1), wide, filterable, Mag::Smooth),
                |alpha| {
                    expect(alpha, width as usize * height as usize, "alpha");
                    upload_plane(gl, plane_bytes(alpha), (width, height), wide, filterable, Mag::Smooth)
                },
            );
            Entry::Yuv {
                luma: luma_texture,
                cb: cb_texture,
                cr: cr_texture,
                alpha: alpha_texture,
                lmap_base: lmap_base((width, height)),
                cmap_base: cmap_base((width, height), yuv.subsampling),
                conversion: page_conversion(decoded),
                norm: depth_norm(depth, wide),
                transfer: decoded.transfer,
                cost: entry_cost_bytes(decoded, filterable),
            }
        }
    }
}

/// The `u_lmap` map before the zoom divide: screen px -> luma texel at
/// scale 1.
fn lmap_base(size: (u32, u32)) -> (f32, f32) {
    let img = view::px_f32(size);
    (1.0 / img.0, 1.0 / img.1)
}

/// The `u_cmap` map before the zoom divide: screen px -> chroma texel at
/// scale 1, with the subsample ratio folded in.
fn cmap_base(size: (u32, u32), subsampling: Subsampling) -> (f32, f32) {
    let (_, (chroma_w, chroma_h)) = plane_dims(size, subsampling);
    let chroma = view::px_f32((chroma_w, chroma_h));
    let ratio = subsampling.chroma_ratio();
    (ratio.0 / chroma.0, ratio.1 / chroma.1)
}

/// One 1x1 stub plane holding `code` at `depth`: mono's neutral chroma,
/// the opaque alpha's max code. Uploaded through `plane_bytes` like any
/// real plane, so the storage layout has one home.
fn stub_plane(depth: u8, code: u32) -> Plane {
    if depth > 8 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the depth clamp keeps codes within u16"
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

/// Cache budget: approximate texture bytes the cache may hold.
const TEXTURE_BUDGET_BYTES: usize = 3 * 1024 * 1024 * 1024;

/// Approximate texture bytes one page occupies: plane bytes, plus the mip
/// chain (`4/3` of the base levels) where mips are generated.
fn entry_cost_bytes(decoded: &Decoded, mipmapped: bool) -> usize {
    let base = match &decoded.pixels {
        Pixels::Rgba8(rgba) => rgba.len(),
        Pixels::Yuv(yuv) => {
            let per_sample = if matches!(yuv.luma, Plane::U16(_)) { 2 } else { 1 };
            let chroma = match yuv.subsampling {
                // Mono uploads a 1x1 midpoint texture per chroma channel.
                Subsampling::Mono => 2 * per_sample,
                Subsampling::Yuv420 | Subsampling::Yuv422 | Subsampling::Yuv444 => {
                    yuv.cb.byte_len() + yuv.cr.byte_len()
                }
            };
            yuv.luma.byte_len()
                + chroma
                // A page without alpha uploads a 1x1 opaque stub instead.
                + yuv.alpha.as_ref().map_or(per_sample, Plane::byte_len)
        }
    };
    #[expect(
        clippy::integer_division,
        reason = "the mip chain costs 4/3 of the base levels, rounded down for budgeting"
    )]
    if mipmapped { base * 4 / 3 } else { base }
}

/// Pages to drop so the cached costs fit `budget` after a fresh insert.
/// Candidates are unbound pages, farthest from the nearest bound page
/// first, ties by higher page id. Bound pages never go, so the survivors
/// can still sit above the budget until the next insert.
fn eviction_victims(
    costs: impl IntoIterator<Item = (usize, usize)>,
    bound: [Option<usize>; 2],
    budget: usize,
) -> Vec<usize> {
    let mut candidates: Vec<(usize, usize)> = costs.into_iter().collect();
    let mut remaining: usize = candidates.iter().map(|&(_, cost)| cost).sum();
    candidates.retain(|&(page, _)| !bound.contains(&Some(page)));
    let distance = |page: usize| {
        bound
            .iter()
            .flatten()
            .map(|bound_page| page.abs_diff(*bound_page))
            .min()
            .unwrap_or(usize::MAX)
    };
    candidates.sort_unstable_by_key(|&(page, _)| Reverse((distance(page), page)));
    let mut victims = Vec::new();
    for (page, cost) in candidates {
        if remaining <= budget {
            break;
        }
        remaining -= cost;
        victims.push(page);
    }
    victims
}

/// The `u_transfer` value of a page's sample encoding: 0 = sRGB
/// passthrough, 1 = PQ tone-mapped.
fn transfer_id(transfer: Transfer) -> i32 {
    match transfer {
        Transfer::Srgb => 0,
        Transfer::Pq => 1,
    }
}

/// Normalizes R16 sampler output for U16-stored planes: sampling divides by
/// [`R16_STORAGE_MAX`] but only `depth` bits are significant. 1.0 for 8-bit
/// storage.
#[expect(
    clippy::cast_precision_loss,
    reason = "the depth mask 2^depth - 1 is at most 65535, exact in f32"
)]
fn depth_norm(depth: u8, wide: bool) -> f32 {
    if !wide {
        return 1.0;
    }
    R16_STORAGE_MAX / ((1_u32 << depth.clamp(MIN_PLANE_DEPTH, MAX_PLANE_DEPTH)) - 1) as f32
}

/// Largest value an R16 (u16) texture sampler can return.
const R16_STORAGE_MAX: f32 = 65535.0;

/// Plane bit depths the R8/R16 texture paths accept.
const MIN_PLANE_DEPTH: u8 = 1;
const MAX_PLANE_DEPTH: u8 = 16;

/// The quantization of one page's storage codes at `depth`: the anchor
/// codes (luma black, chroma neutral), the sample-to-RGB scales, and the
/// storage capacity. Full range spans 0..2^depth-1 with chroma neutral at
/// half; limited range scales the BT studio swing by 2^(depth-8).
struct Quantization {
    /// Luma black, in code units.
    luma_black: f32,
    /// Chroma neutral, 2^(depth-1) under both ranges.
    chroma_neutral: f32,
    /// Slope from a depth-normalized luma sample to RGB: 1.0 at full
    /// range, `max_code` / 219·2^(depth-8) at limited range.
    luma_scale: f32,
    /// Slope from a depth-normalized chroma sample to RGB: 1.0 at full
    /// range, `max_code` / 224·2^(depth-8) at limited range.
    chroma_scale: f32,
    /// Codes the storage can hold: 255 or 65535.
    storage_max: f32,
}

#[expect(
    clippy::cast_precision_loss,
    reason = "codes and scales stay below 65536, exact in f32"
)]
fn quantization(depth: u8, full_range: bool) -> Quantization {
    let max_code = (1_u32 << depth) - 1;
    let storage_max = if depth > 8 { R16_STORAGE_MAX } else { 255.0 };
    if full_range {
        Quantization {
            luma_black: 0.0,
            chroma_neutral: (1_u32 << (depth - 1)) as f32,
            luma_scale: 1.0,
            chroma_scale: 1.0,
            storage_max,
        }
    } else {
        let swing = u32::from(depth.saturating_sub(8));
        Quantization {
            luma_black: (16 << swing) as f32,
            chroma_neutral: (128 << swing) as f32,
            luma_scale: max_code as f32 / (219 << swing) as f32,
            chroma_scale: max_code as f32 / (224 << swing) as f32,
            storage_max,
        }
    }
}

/// Fold a page's whole sample-to-RGB math at upload. The stored matrix
/// maps storage-normalized samples (0..1 as the sampler returns them)
/// through depth normalization, quantization, and the KR/KB matrix (or
/// the GBR permutation of an identity page); the stored offset is that
/// matrix times the anchor codes.
fn page_conversion(decoded: &Decoded) -> Conversion {
    let Pixels::Yuv(yuv) = &decoded.pixels else {
        return Conversion { matrix: IDENTITY3, offset: [0.0; 3] };
    };
    let depth = yuv.depth.clamp(MIN_PLANE_DEPTH, MAX_PLANE_DEPTH);
    let quantization = quantization(depth, yuv.full_range);
    let norm = depth_norm(depth, depth > 8);
    let (base, luma_scale, chroma_scale, luma_black, chroma_neutral) =
        if yuv.matrix == Matrix::Identity {
            // H.273 identity planes are GBR in the (Y, Cb, Cr) slots,
            // quantized as luma on all three axes: permute into RGB and
            // anchor every axis at luma black. Mono identity repeats the
            // luma plane to RGB.
            let base = if yuv.subsampling == Subsampling::Mono {
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
            };
            (base, quantization.luma_scale, quantization.luma_scale, quantization.luma_black, quantization.luma_black)
        } else {
            (
                full_range_yuv_matrix(yuv.matrix),
                quantization.luma_scale,
                quantization.chroma_scale,
                quantization.luma_black,
                quantization.chroma_neutral,
            )
        };
    let mut matrix = base;
    for (column, matrix_column) in matrix.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let scale = if column == 0 { luma_scale } else { chroma_scale } * norm;
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
    for (matrix_column, anchor) in matrix.as_chunks::<3>().0.iter().zip(anchors) {
        for (&cell, slot) in matrix_column.iter().zip(&mut offset) {
            *slot = cell.mul_add(anchor, *slot);
        }
    }
    Conversion { matrix, offset }
}

/// Full-range YUV→RGB from the exact KR/KB of each standard, stored
/// column-major. KG = 1 - KR - KB. Identity pages never reach here;
/// `page_conversion` routes them through the GBR permutation.
#[expect(
    clippy::similar_names,
    reason = "the green coefficients are the Cb and Cr terms of the BT matrix derivation"
)]
fn full_range_yuv_matrix(matrix: Matrix) -> [f32; 9] {
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
        1.0, 1.0, 1.0, //
        0.0, -green_from_cb, 2.0 * (1.0 - k_blue), //
        2.0 * (1.0 - k_red), -green_from_cr, 0.0,
    ]
}

/// Luma coefficients (KR, KB) of each conversion standard.
const BT601_KR_KB: (f32, f32) = (0.299, 0.114);
const BT709_KR_KB: (f32, f32) = (0.2126, 0.0722);
const BT2020_KR_KB: (f32, f32) = (0.2627, 0.0593);

const IDENTITY3: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

#[cfg(test)]
#[expect(clippy::expect_used, reason = "a panic in test code is inconsequential")]
mod tests {
    use super::*;
    use crate::decode::Format;
    use crate::decode::Transfer;
    use crate::decode::YuvImage;

    // The GL draw path itself is out of reach headlessly. These tests pin
    // the CPU-side values it consumes.

    /// Row `row`, column `col` of the math matrix (storage is column-major).
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
        let ((luma_w, luma_h), (chroma_w, chroma_h)) = plane_dims(size, subsampling);
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
            (plane(chroma_w as usize * chroma_h as usize), plane(chroma_w as usize * chroma_h as usize))
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

    /// The folded conversion through one row: `matrix * samples - offset`.
    fn convert(conversion: &Conversion, samples: [f32; 3]) -> [f32; 3] {
        let mut out = [0.0_f32; 3];
        for (row, slot) in out.iter_mut().enumerate() {
            #[expect(
                clippy::indexing_slicing,
                reason = "row and col are 0..3 coordinates into the fixed 3-vectors under test"
            )]
            let converted = (0..3)
                .map(|col| element(&conversion.matrix, row, col) * samples[col])
                .sum::<f32>()
                - conversion.offset[row];
            *slot = converted;
        }
        out
    }

    /// Code values of one depth's quantization (black, white, neutral) in
    /// code units, and the largest code the storage layout holds.
    fn quantization_codes(depth: u8, full_range: bool) -> (f32, f32, f32, f32) {
        let scale = 2.0_f32.powi(i32::from(depth) - 8);
        let storage_max = 2.0_f32.powi(if depth > 8 { 16 } else { 8 }) - 1.0;
        let codes = if full_range {
            (0.0, 2.0_f32.powi(i32::from(depth)) - 1.0, 2.0_f32.powi(i32::from(depth) - 1))
        } else {
            (16.0 * scale, 235.0 * scale, 128.0 * scale)
        };
        (codes.0, codes.1, codes.2, storage_max)
    }

    #[test]
    fn conversion_anchors_black_and_white_at_every_depth() {
        for depth in [8_u8, 10, 12, 16] {
            for full_range in [true, false] {
                let decoded = yuv_page(depth, Subsampling::Yuv444, full_range, Matrix::Bt601);
                let conversion = page_conversion(&decoded);
                let (black, white, neutral, storage_max) = quantization_codes(depth, full_range);
                for (code, expected) in [(black, 0.0), (white, 1.0)] {
                    // Chroma at its neutral code contributes nothing.
                    let rgb = convert(
                        &conversion,
                        [code / storage_max, neutral / storage_max, neutral / storage_max],
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
        let color = page_conversion(&yuv_page(8, Subsampling::Yuv444, true, Matrix::Identity));
        // Column j is plane j's contribution: Y->G, U->B, V->R.
        for (col, row) in [(0, 1), (1, 2), (2, 0)] {
            for probe in 0..3 {
                let want = f32::from(probe == row);
                assert!(
                    (element(&color.matrix, probe, col) - want).abs() < 1e-6,
                    "col {col} row {probe}"
                );
            }
        }
        // Full range anchors everything at black zero.
        assert_eq!(color.offset, [0.0; 3]);

        let mono = page_conversion(&yuv_page(10, Subsampling::Mono, true, Matrix::Identity));
        for row in 0..3 {
            assert!((element(&mono.matrix, row, 1)).abs() < 1e-6, "mono chroma col must be inert");
            assert!((element(&mono.matrix, row, 2)).abs() < 1e-6, "mono chroma col must be inert");
        }
        assert_eq!(mono.offset, [0.0; 3]);
        // The norm-folded luma column still spans the range: white (code
        // 1023 in u16 storage) reads 1 on every channel.
        let white = 1023.0 / 65535.0;
        let rgb = convert(&mono, [white, white, white]);
        for (row, channel) in rgb.into_iter().enumerate() {
            assert!((channel - 1.0).abs() < 1e-4, "mono row {row}: {channel}");
        }
    }

    #[test]
    fn mono_stub_cancels_exactly() {
        for depth in [8_u8, 10, 12] {
            for full_range in [true, false] {
                let decoded = yuv_page(depth, Subsampling::Mono, full_range, Matrix::Bt601);
                let conversion = page_conversion(&decoded);
                let (black, white, neutral, storage_max) = quantization_codes(depth, full_range);
                for (code, expected) in [(black, 0.0), (white, 1.0)] {
                    let rgb = convert(
                        &conversion,
                        [code / storage_max, neutral / storage_max, neutral / storage_max],
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
    fn shader_ids() {
        assert_ne!(MODE_RGBA, MODE_YUV, "the mode uniform must pick a branch");
        assert_eq!(MODE_RGBA, 0, "the shader samples RGBA at u_mode 0");
        assert_eq!(MODE_YUV, 1, "the shader samples the plane set at u_mode 1");
        assert_eq!(transfer_id(Transfer::Srgb), 0);
        assert_eq!(transfer_id(Transfer::Pq), 1);
    }

    #[expect(
        clippy::integer_division,
        reason = "mirrors entry_cost_bytes' truncated mip-chain cost"
    )]
    #[test]
    fn entry_cost_counts_the_mip_chain_where_present() {
        let rgba = Decoded::rgba(
            4,
            4,
            Format::Png,
            Transfer::Srgb,
            vec![0_u8; 64].into_boxed_slice(),
        )
        .expect("rgba page");
        let yuv420 = Decoded::yuv(
            4,
            4,
            Format::Avif,
            Transfer::Srgb,
            YuvImage {
                depth: 8,
                subsampling: Subsampling::Yuv420,
                full_range: true,
                matrix: Matrix::Bt709,
                luma: Plane::U8(vec![0_u8; 16].into_boxed_slice()),
                cb: Plane::U8(vec![0_u8; 4].into_boxed_slice()),
                cr: Plane::U8(vec![0_u8; 4].into_boxed_slice()),
                alpha: None,
            },
        )
        .expect("yuv page");
        let with_alpha = Decoded::yuv(
            4,
            4,
            Format::Avif,
            Transfer::Pq,
            YuvImage {
                depth: 8,
                subsampling: Subsampling::Yuv420,
                full_range: true,
                matrix: Matrix::Bt709,
                luma: Plane::U8(vec![0_u8; 16].into_boxed_slice()),
                cb: Plane::U8(vec![0_u8; 4].into_boxed_slice()),
                cr: Plane::U8(vec![0_u8; 4].into_boxed_slice()),
                alpha: Some(Plane::U8(vec![0_u8; 16].into_boxed_slice())),
            },
        )
        .expect("yuv page with alpha");
        // Mono planes here are R16, unfilterable under GLES: no mip chain.
        let mono = Decoded::yuv(
            2,
            2,
            Format::Avif,
            Transfer::Srgb,
            YuvImage {
                depth: 10,
                subsampling: Subsampling::Mono,
                full_range: true,
                matrix: Matrix::Identity,
                luma: Plane::U16(vec![0_u16; 4].into_boxed_slice()),
                cb: Plane::U8(Box::default()),
                cr: Plane::U8(Box::default()),
                alpha: None,
            },
        )
        .expect("mono page");
        for (decoded, mipmapped, want) in [
            (&rgba, true, 64 * 4 / 3),
            // the alpha-less yuv page pays a 1x1 opaque stub
            (&yuv420, true, (16 + 4 + 4 + 1) * 4 / 3),
            (&with_alpha, true, (16 + 4 + 4 + 16) * 4 / 3),
            // mono: luma + two chroma stubs + the alpha stub, all wide
            (&mono, false, 4 * 2 + 2 * 2 + 2),
        ] {
            assert_eq!(entry_cost_bytes(decoded, mipmapped), want);
        }
    }

    #[test]
    fn eviction_victims_table() {
        let far = [(3_usize, 10_usize), (19, 10), (10, 10), (20, 10)];
        let tied = [(8_usize, 10_usize), (12, 10), (10, 10)];
        let bound = [(10_usize, 10_usize), (11, 10), (12, 10)];
        for (costs, seats, budget, want) in [
            // 19 sits one page from seat 20, so 3 is the farther victim
            (&far[..], [Some(10), Some(20)], 35, vec![3_usize]),
            (&far[..], [Some(10), Some(20)], 25, vec![3_usize, 19]),
            // an equal distance from the seat breaks the tie to the higher page
            (&tied[..], [Some(10), None], 25, vec![12]),
            // bound pages never go, even at budget 0
            (&bound[..], [Some(10), Some(11)], 0, vec![12]),
            // everything fits: no victims
            (&bound[..], [Some(10), Some(11)], 30, vec![]),
        ] {
            assert_eq!(
                eviction_victims(costs.iter().copied(), seats, budget),
                want,
                "costs {costs:?} seats {seats:?} budget {budget}"
            );
        }
    }
}
