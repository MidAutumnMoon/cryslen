//! Shared GLSL snippets, shader/program compilation, uniform helpers, and frame setup.

#![expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "scoped to GL helpers sharing one current-context invariant"
)]

use std::sync::Arc;

use egui_glow::glow;
use egui_glow::glow::Context;
use egui_glow::glow::HasContext as _;
use egui_glow::glow::Program;
use egui_glow::glow::Shader;
use egui_glow::glow::Texture;
use egui_glow::glow::UniformLocation;
use egui_glow::glow::VertexArray;
use indoc::indoc;

use crate::image::color::YuvMatrix;
use crate::image::pixels::Transfer;

pub const QUAD_VERTEX_COUNT: i32 = 4;
pub const FIRST_VERTEX_INDEX: i32 = 0;
pub const BASE_MIP_LEVEL: i32 = 0;
pub const NO_BORDER: i32 = 0;

pub const RGBA_OR_LUMA_SAMPLER_UNIT: i32 = 0;
pub const CB_SAMPLER_UNIT: i32 = 1;
pub const CR_SAMPLER_UNIT: i32 = 2;
pub const ALPHA_SAMPLER_UNIT: i32 = 3;

pub const PIXEL_LAYOUT_RGBA: i32 = 0;
pub const PIXEL_LAYOUT_YUV: i32 = 1;
pub const PIXEL_LAYOUT_MONO: i32 = 2;

pub const TRANSFER_SRGB: i32 = 0;
pub const TRANSFER_PQ: i32 = 1;

pub const GLSL_VERSION: &str = "#version 330 core\n";

// The quad is a fixed -1..1 strip generated from gl_VertexID; uniforms
// place it at the seat rect. Clip space is bottom-origin; winit coords
// are top-origin.
pub const VERTEX_SHADER_SOURCE: &str = indoc! {"
    #version 330 core

    uniform vec2 u_window_size_px;
    uniform vec2 u_seat_origin_px;
    uniform vec2 u_seat_size_px;
    out vec2 v_seat_px;
    void main() {
        vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
        vec2 seat_px = corner * u_seat_size_px;
        v_seat_px = seat_px;
        vec2 screen_px = u_seat_origin_px + seat_px;
        gl_Position = vec4(screen_px.x / u_window_size_px.x * 2.0 - 1.0,
                           1.0 - screen_px.y / u_window_size_px.y * 2.0,
                           0.0, 1.0);
    }
"};

pub const FULLSCREEN_VERTEX_SHADER_SOURCE: &str = indoc! {"
    #version 330 core

    void main() {
        vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
        gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
    }
"};

pub const GLSL_TRANSFER_HEADER: &str = indoc! {"
    uniform int u_transfer_code;  // TRANSFER_SRGB or TRANSFER_PQ
    const int PIXEL_LAYOUT_RGBA = 0;
    const int PIXEL_LAYOUT_YUV = 1;
    const int PIXEL_LAYOUT_MONO = 2;
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
    const float PQ_DIFFUSE_WHITE_NITS = 203.0;
    const float PQ_PEAK_NITS = 10000.0;
    const float SRGB_LINEAR_THRESHOLD = 0.0031308;
    const float SRGB_LINEAR_SLOPE = 12.92;
    const float SRGB_GAMMA_SCALE = 1.055;
    const float SRGB_GAMMA_OFFSET = 0.055;
    const float SRGB_GAMMA_EXPONENT = 2.4;
    vec3 to_srgb(vec3 encoded_rgb) {
        if (u_transfer_code == TRANSFER_SRGB) {
            return encoded_rgb;
        }
        vec3 pq_scaled = pow(clamp(encoded_rgb, 0.0, 1.0), vec3(1.0 / PQ_M2));
        vec3 luminance_nits = PQ_PEAK_NITS * pow(max(pq_scaled - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * pq_scaled), vec3(1.0 / PQ_M1));
        vec3 sdr_linear = clamp(luminance_nits / PQ_DIFFUSE_WHITE_NITS, 0.0, 1.0);
        return mix(
            sdr_linear * SRGB_LINEAR_SLOPE,
            SRGB_GAMMA_SCALE * pow(sdr_linear, vec3(1.0 / SRGB_GAMMA_EXPONENT)) - SRGB_GAMMA_OFFSET,
            step(vec3(SRGB_LINEAR_THRESHOLD), sdr_linear));
    }
"};

pub const GLSL_BAYER4: &str = indoc! {"
    const int BAYER_SIDE = 4;
    const int BAYER_MASK = BAYER_SIDE - 1;
    const float BAYER_CELL_COUNT = float(BAYER_SIDE * BAYER_SIDE);
    const float DITHER_CENTER_BIAS = 0.5;
    const float OUTPUT_8BIT_MAX = 255.0;
    float dither_offset(vec2 fragment_px) {
        float thresholds[16] = float[16](
            0.0, 8.0, 2.0, 10.0,
            12.0, 4.0, 14.0, 6.0,
            3.0, 11.0, 1.0, 9.0,
            15.0, 7.0, 13.0, 5.0
        );
        ivec2 cell = ivec2(fragment_px) & BAYER_MASK;
        float bayer_unit = thresholds[cell.x + BAYER_SIDE * cell.y] / BAYER_CELL_COUNT;
        return (bayer_unit - DITHER_CENTER_BIAS) / OUTPUT_8BIT_MAX;
    }
"};

pub fn transfer_code(transfer: Transfer) -> i32 {
    match transfer {
        Transfer::Srgb => TRANSFER_SRGB,
        Transfer::Pq => TRANSFER_PQ,
    }
}

pub fn gl_size(value: u32) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "GL uniforms and textures take f32; narrowing happens at this boundary"
)]
pub fn gl_f32(value: f64) -> f32 {
    value as f32
}

/// Name one object when the driver exposes `KHR_debug`.
pub fn label(
    gl: &Context,
    object_namespace: u32,
    object_id: u32,
    debug_label: &str,
) {
    if gl.supports_debug() {
        // SAFETY: `gl` is current with `KHR_debug` supported and `object_id` is live.
        unsafe {
            gl.object_label(
                object_namespace,
                object_id,
                Some(debug_label),
            );
        }
    }
}

pub struct EmptyVao {
    gl: Arc<Context>,
    handle: VertexArray,
}

impl EmptyVao {
    #[must_use]
    pub const fn handle(&self) -> VertexArray {
        self.handle
    }

    pub fn new(gl: Arc<Context>, debug_name: &str) -> eros::Result<Self> {
        // SAFETY: Context is current (`Window` invariant).
        let vertex_array =
            unsafe { gl.create_vertex_array() }.map_err(|error| {
                eros::error!(
                    "GL vertex array object creation failed: {error}"
                )
            })?;
        label(&gl, glow::VERTEX_ARRAY, vertex_array.0.get(), debug_name);
        // SAFETY: Context is current and `vertex_array` was just created.
        unsafe {
            gl.bind_vertex_array(Some(vertex_array));
        }
        Ok(Self {
            gl,
            handle: vertex_array,
        })
    }
}

impl Drop for EmptyVao {
    fn drop(&mut self) {
        // SAFETY: Context is current (`Window` drop order) and `self` owns `handle`.
        unsafe {
            self.gl.delete_vertex_array(self.handle);
        }
    }
}

struct ShaderGuard<'context> {
    gl: &'context Context,
    shader: Shader,
}

impl Drop for ShaderGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.gl` is the current context and `self.shader` is owned
        // uniquely by this guard.
        unsafe {
            self.gl.delete_shader(self.shader);
        }
    }
}

fn compile_shader<'context>(
    gl: &'context Context,
    shader_stage: u32,
    sources: &[&str],
) -> eros::Result<ShaderGuard<'context>> {
    // SAFETY: Caller ensures the GL context is current on the render thread.
    let shader =
        unsafe { gl.create_shader(shader_stage) }.map_err(|error| {
            eros::error!("GL shader object creation failed: {error}")
        })?;
    let guard = ShaderGuard { gl, shader };
    let source = sources.concat();
    // SAFETY: `guard.shader` was just created on the current context.
    unsafe {
        gl.shader_source(guard.shader, &source);
        gl.compile_shader(guard.shader);
        if !gl.get_shader_compile_status(guard.shader) {
            let info_log = gl.get_shader_info_log(guard.shader);
            return Err(eros::error!(
                "GL shader compile failed: {info_log}"
            ));
        }
    }
    Ok(guard)
}

pub struct ShaderPipeline {
    pub gl: Arc<Context>,
    pub program: Program,
}

impl Drop for ShaderPipeline {
    fn drop(&mut self) {
        // SAFETY: Context is current (`Window` drop order) and `self` owns `program`.
        unsafe {
            self.gl.delete_program(self.program);
        }
    }
}

pub fn uniform_location(
    gl: &Context,
    program: Program,
    name: &str,
) -> UniformLocation {
    // SAFETY: Context is current and `program` is linked.
    let found = unsafe { gl.get_uniform_location(program, name) };
    #[expect(
        clippy::expect_used,
        reason = "declared and used in the in-tree shader sources, so a miss is an in-tree regression"
    )]
    found.expect("uniform location in shader")
}

pub fn link(
    gl: Arc<Context>,
    vertex_source: &str,
    fragment_sources: &[&str],
    samplers: &[(i32, &str)],
    debug_name: &str,
) -> eros::Result<ShaderPipeline> {
    // SAFETY: Context is current.
    let program = unsafe { gl.create_program() }.map_err(|error| {
        eros::error!("GL program object creation failed: {error}")
    })?;
    let gl_program = ShaderPipeline { gl, program };
    {
        let vertex_shader = compile_shader(
            &gl_program.gl,
            glow::VERTEX_SHADER,
            &[vertex_source],
        )?;
        let fragment_shader = compile_shader(
            &gl_program.gl,
            glow::FRAGMENT_SHADER,
            fragment_sources,
        )?;
        // SAFETY: Context is current; `gl_program.program` and both shaders are live.
        unsafe {
            gl_program
                .gl
                .attach_shader(gl_program.program, vertex_shader.shader);
            gl_program
                .gl
                .attach_shader(gl_program.program, fragment_shader.shader);
            gl_program.gl.link_program(gl_program.program);
            if !gl_program.gl.get_program_link_status(gl_program.program) {
                let info_log =
                    gl_program.gl.get_program_info_log(gl_program.program);
                return Err(eros::error!(
                    "GL program link failed ({debug_name}): {info_log}"
                ));
            }
        }
    }
    // Sampler units are fixed; draw calls rebind textures only.
    // SAFETY: Context is current and `gl_program.program` is linked.
    unsafe {
        gl_program.gl.use_program(Some(gl_program.program));
        for &(unit, name) in samplers {
            let location =
                uniform_location(&gl_program.gl, gl_program.program, name);
            gl_program.gl.uniform_1_i32(Some(&location), unit);
        }
    }
    label(
        &gl_program.gl,
        glow::PROGRAM,
        gl_program.program.0.get(),
        debug_name,
    );
    Ok(gl_program)
}

#[derive(Clone, Copy)]
pub struct DrawTarget {
    pub window: (i32, i32),
    pub clip: (i32, i32),
}

impl DrawTarget {
    pub fn new(window: (u32, u32), clip_size: (u32, u32)) -> Option<Self> {
        let window = (gl_size(window.0), gl_size(window.1));
        let clip = (
            gl_size(clip_size.0).min(window.0),
            gl_size(clip_size.1).min(window.1),
        );
        (clip.0 > 0 && clip.1 > 0).then_some(Self { window, clip })
    }
}

#[derive(Clone, Copy)]
pub struct SeatGeometry {
    pub origin: (f64, f64),
    pub size: (f64, f64),
}

pub fn restore_screen_state(gl: &Context, target: DrawTarget) {
    // SAFETY: Context is current.
    unsafe {
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.viewport(0, 0, target.window.0, target.window.1);
        gl.enable(glow::SCISSOR_TEST);
        // GL scissor is bottom-origin; viewport sits at the window's top-left.
        gl.scissor(
            0,
            target.window.1 - target.clip.1,
            target.clip.0,
            target.clip.1,
        );
        gl.enable(glow::BLEND);
        gl.blend_func_separate(
            glow::ONE,
            glow::ONE_MINUS_SRC_ALPHA,
            glow::ONE,
            glow::ONE_MINUS_SRC_ALPHA,
        );
    }
}

pub fn begin_frame(gl: &Context, vao: VertexArray, target: DrawTarget) {
    restore_screen_state(gl, target);
    // SAFETY: Context is current and `vao` is owned by `PageRenderer`.
    unsafe {
        gl.bind_vertex_array(Some(vao));
    }
}

pub struct QuadUniforms {
    window_size: UniformLocation,
    seat_origin: UniformLocation,
    seat_size: UniformLocation,
}

impl QuadUniforms {
    pub fn lookup(gl: &Context, program: Program) -> Self {
        let location = |name: &str| uniform_location(gl, program, name);
        Self {
            window_size: location("u_window_size_px"),
            seat_origin: location("u_seat_origin_px"),
            seat_size: location("u_seat_size_px"),
        }
    }

    pub fn bind(
        &self,
        gl: &Context,
        window: (i32, i32),
        seat: SeatGeometry,
    ) {
        // SAFETY: Context is current; `self` belongs to the active program.
        unsafe {
            gl.uniform_2_f32(
                Some(&self.window_size),
                gl_f32(f64::from(window.0)),
                gl_f32(f64::from(window.1)),
            );
            gl.uniform_2_f32(
                Some(&self.seat_origin),
                gl_f32(seat.origin.0),
                gl_f32(seat.origin.1),
            );
            gl.uniform_2_f32(
                Some(&self.seat_size),
                gl_f32(seat.size.0),
                gl_f32(seat.size.1),
            );
        }
    }
}

pub fn bind_mono_luma(
    gl: &Context,
    luma_gain_offset: UniformLocation,
    luma: Texture,
    (gain, offset): (f32, f32),
) {
    // SAFETY: Context is current; active program owns location and `luma` is live.
    unsafe {
        gl.uniform_2_f32(Some(&luma_gain_offset), gain, offset);
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(luma));
    }
}

pub fn bind_rgba_source(
    gl: &Context,
    pixel_layout: UniformLocation,
    texture: Texture,
) {
    // SAFETY: Context is current; active program owns location and `texture` is live.
    unsafe {
        gl.uniform_1_i32(Some(&pixel_layout), PIXEL_LAYOUT_RGBA);
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
    }
}

#[derive(Clone, Copy)]
pub struct StubTextures {
    pub opaque_alpha: Texture,
    pub neutral_chroma: Texture,
}

pub struct YuvTexture {
    pub luma: Texture,
    pub cb: Option<Texture>,
    pub cr: Option<Texture>,
    pub alpha: Option<Texture>,
    /// Luma texel -> chroma UV, subsample-folded for this texture's dims.
    pub chroma_uv_per_luma_texel: (f64, f64),
    pub color_conversion: YuvMatrix,
    /// Storage-sample normalization for the alpha plane.
    pub alpha_normalization: f32,
}

pub enum PixelLayout {
    Rgba {
        texture: Texture,
    },
    Mono {
        luma: Texture,
        gain: f32,
        offset: f32,
    },
    Yuv(YuvTexture),
}

#[derive(Clone, Copy)]
pub struct DrawSource<'source> {
    pub dims: (u32, u32),
    pub transfer: Transfer,
    pub needs_dither: bool,
    pub layout: &'source PixelLayout,
}

pub struct YuvUniforms {
    chroma_uv_per_step: UniformLocation,
    color_matrix: UniformLocation,
    color_offset: UniformLocation,
    alpha_normalization: UniformLocation,
}

impl YuvUniforms {
    pub fn lookup(
        gl: &Context,
        program: Program,
        step_uniform_name: &str,
    ) -> Self {
        let location = |name: &str| uniform_location(gl, program, name);
        Self {
            chroma_uv_per_step: location(step_uniform_name),
            color_matrix: location("u_color_matrix"),
            color_offset: location("u_color_offset"),
            alpha_normalization: location("u_alpha_normalization"),
        }
    }

    pub fn bind(
        &self,
        gl: &Context,
        pixel_layout: UniformLocation,
        yuv: &YuvTexture,
        stubs: StubTextures,
        luma_texels_per_step: (f64, f64),
    ) {
        // SAFETY: Context is current; `self` belongs to the active program and textures are live.
        unsafe {
            gl.uniform_1_i32(Some(&pixel_layout), PIXEL_LAYOUT_YUV);
            gl.uniform_2_f32(
                Some(&self.chroma_uv_per_step),
                gl_f32(
                    yuv.chroma_uv_per_luma_texel.0
                        * luma_texels_per_step.0,
                ),
                gl_f32(
                    yuv.chroma_uv_per_luma_texel.1
                        * luma_texels_per_step.1,
                ),
            );
            gl.uniform_1_f32(
                Some(&self.alpha_normalization),
                yuv.alpha_normalization,
            );
            gl.uniform_matrix_3_f32_slice(
                Some(&self.color_matrix),
                false,
                &yuv.color_conversion.matrix,
            );
            gl.uniform_3_f32(
                Some(&self.color_offset),
                yuv.color_conversion.offset[0],
                yuv.color_conversion.offset[1],
                yuv.color_conversion.offset[2],
            );
            for (unit, texture) in [
                (glow::TEXTURE0, yuv.luma),
                (glow::TEXTURE1, yuv.cb.unwrap_or(stubs.neutral_chroma)),
                (glow::TEXTURE2, yuv.cr.unwrap_or(stubs.neutral_chroma)),
                (glow::TEXTURE3, yuv.alpha.unwrap_or(stubs.opaque_alpha)),
            ] {
                gl.active_texture(unit);
                gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            }
        }
    }
}

/// Create and configure a single-level 2D texture bound to `TEXTURE_2D`.
pub fn create_2d_texture(
    gl: &Context,
    minification_filter: u32,
    magnification_filter: u32,
    debug_name: Option<&str>,
) -> Texture {
    // SAFETY: Context is current.
    unsafe {
        #[expect(
            clippy::expect_used,
            reason = "texture-creation failure panics by policy"
        )]
        let texture = gl.create_texture().expect("GL texture object");
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        for (parameter, value) in [
            (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
            (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
            (glow::TEXTURE_MIN_FILTER, minification_filter),
            (glow::TEXTURE_MAG_FILTER, magnification_filter),
        ] {
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                parameter,
                value.cast_signed(),
            );
        }
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_BASE_LEVEL,
            BASE_MIP_LEVEL,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAX_LEVEL,
            BASE_MIP_LEVEL,
        );
        if let Some(text) = debug_name {
            label(gl, glow::TEXTURE, texture.0.get(), text);
        }
        texture
    }
}
