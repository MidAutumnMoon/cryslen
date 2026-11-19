//! GL rendering: window, EGL surface, and the page quad. GL runs on the
//! main thread with the context current. Texture-creation failure panics.

#![expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "scoped to render.rs, where batched GL calls share one current-context invariant"
)]

use core::f64::consts::PI;
use core::num::NonZeroU32;
use std::collections::HashMap;
use std::sync::Arc;

use egui_glow::glow;
use egui_glow::glow::Context;
use egui_glow::glow::Framebuffer;
use egui_glow::glow::HasContext as _;
use egui_glow::glow::PixelUnpackData;
use egui_glow::glow::Program;
use egui_glow::glow::Shader;
use egui_glow::glow::Texture;
use egui_glow::glow::UniformLocation;
use egui_glow::glow::VertexArray;
use egui_winit::winit;
use enum_map::EnumMap;
use glutin::context::PossiblyCurrentContext;
use glutin::surface::Surface;
use glutin::surface::WindowSurface;
use indoc::indoc;

use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::OwnedDisplayHandle;
use winit::raw_window_handle::HasWindowHandle as _;
use winit::window::Window as WinitWindow;

use crate::browse::Role;
use crate::browse::Roles;
use crate::cache;
use crate::decode;
use crate::decode::Matrix;
use crate::decode::Subsampling;
use crate::decode::Transfer;
use crate::fault::TextureFault;
use crate::geometry;
use crate::geometry::PageRect;
use crate::geometry::Placement;
use crate::geometry::SpreadLayout;
use crate::geometry::Viewport;
use crate::protocol::ValidatedImage;
use crate::protocol::WirePixels;

// The quad is a fixed -1..1 strip generated from gl_VertexID; uniforms
// place it at the seat rect. Clip space is bottom-origin; winit coords
// are top-origin.
const VERTEX_SHADER_SOURCE: &str = indoc! {"
    #version 330 core

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
    }
"};

const FULLSCREEN_VERTEX_SHADER_SOURCE: &str = indoc! {"
    #version 330 core

    void main() {
        vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
        gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
    }
"};

const GLSL_TRANSFER_HEADER: &str = indoc! {"
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
    const float PQ_DIFFUSE_WHITE = 203.0;
    const float PQ_PEAK = 10000.0;
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
"};

const GLSL_BAYER4: &str = indoc! {"
    const float OUTPUT_MAX = 255.0;
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
"};

#[derive(Clone, Copy)]
enum EwaVariant {
    Mono,
    Color,
}

fn glsl_ewa_lut() -> String {
    let lut_len = f64::from(EWA_LUT_WIDTH);
    format!(
        indoc! {"
            const float EWA_SUPPORT = {support:.9};
            const float EWA_LUT_SCALE = {scale:.9};
            const float EWA_LUT_BIAS = {bias:.9};
            vec2 ewa_weights(sampler1D lut, float distance) {{
                float lut_coord = clamp(distance / EWA_SUPPORT, 0.0, 1.0);
                return texture(lut, lut_coord * EWA_LUT_SCALE + EWA_LUT_BIAS).rg;
            }}
        "},
        support = EWA_SUPPORT,
        scale = (lut_len - 1.0) / lut_len,
        bias = 0.5 / lut_len,
    )
}

fn fragment_shader_source() -> String {
    format!(
        "{header}{transfer}{bayer}{body}",
        header = "#version 330 core\n",
        transfer = GLSL_TRANSFER_HEADER,
        bayer = GLSL_BAYER4,
        body = indoc! {"
            in vec2 v_seat_px;
            layout(location = 0) out vec4 color;
            uniform vec2 u_page_uv_per_screen;
            uniform vec2 u_chroma_uv_per_step;  // screen px -> chroma texture UV
            uniform mat3 u_color_matrix;        // storage samples -> gamma RGB
            uniform vec3 u_color_offset;        // u_color_matrix times the quantization anchors
            uniform float u_alpha_normalization; // depth normalize of the alpha sample; 1.0 for 8-bit storage
            uniform vec2 u_luma_gain_offset;
            uniform int u_pixel_layout;
            uniform int u_dither;               // 0 = 8-bit source, 1 = deep source
            uniform sampler2D u_rgba_or_luma;
            uniform sampler2D u_cb;
            uniform sampler2D u_cr;
            uniform sampler2D u_alpha;
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
                if (u_pixel_layout == PIXEL_LAYOUT_MONO) {
                    float luma_sample = texture(u_rgba_or_luma, seat_px * u_page_uv_per_screen).r;
                    float gray = clamp(
                        to_srgb(vec3(luma_sample * u_luma_gain_offset.x - u_luma_gain_offset.y)).r + dither_offset,
                        0.0,
                        1.0);
                    color = vec4(gray, gray, gray, 1.0);
                    return;
                }
                if (u_pixel_layout == PIXEL_LAYOUT_RGBA) {
                    vec4 rgba_texel = texture(u_rgba_or_luma, seat_px * u_page_uv_per_screen);
                    color = vec4((to_srgb(rgba_texel.rgb) + dither_offset) * rgba_texel.a, rgba_texel.a);
                    return;
                }
                vec3 yuv_samples = vec3(
                    texture(u_rgba_or_luma, seat_px * u_page_uv_per_screen).r,
                    texture(u_cb, seat_px * u_chroma_uv_per_step).r,
                    texture(u_cr, seat_px * u_chroma_uv_per_step).r);
                float alpha = texture(u_alpha, seat_px * u_page_uv_per_screen).r * u_alpha_normalization;
                color = vec4((to_srgb(u_color_matrix * yuv_samples - u_color_offset) + dither_offset) * alpha, alpha);
            }
        "},
    )
}

fn ewa_horizontal_shader_source(variant: EwaVariant) -> String {
    let variant_define = match variant {
        EwaVariant::Mono => "#define EWA_MONO 1\n",
        EwaVariant::Color => "#define EWA_MONO 0\n",
    };
    format!(
        "{header}{variant_define}{transfer}{lut}{body}",
        header = "#version 330 core\n",
        transfer = GLSL_TRANSFER_HEADER,
        lut = glsl_ewa_lut(),
        body = indoc! {"
            #if EWA_MONO
            layout(location = 0) out vec4 out_accum;
            uniform vec2 u_luma_gain_offset;
            #else
            layout(location = 0) out vec4 out_primary;
            layout(location = 1) out vec4 out_secondary;
            layout(location = 2) out vec4 out_weights;
            uniform vec2 u_chroma_uv_per_step; // luma texel -> chroma texture UV
            uniform mat3 u_color_matrix;
            uniform vec3 u_color_offset;
            uniform float u_alpha_normalization;
            uniform int u_pixel_layout;
            uniform sampler2D u_cb;
            uniform sampler2D u_cr;
            uniform sampler2D u_alpha;
            #endif
            uniform int u_visible_origin_x;
            uniform int u_source_y_start;
            uniform ivec2 u_texture_size;
            uniform float u_seat_offset_x;
            uniform float u_source_per_screen_x;
            uniform sampler2D u_rgba_or_luma;
            uniform sampler1D u_ewa_lut;
            void main() {
                float filter_source_per_screen = max(1.0, u_source_per_screen_x);
                float screen_per_source = 1.0 / filter_source_per_screen;
                float support_radius = EWA_SUPPORT * filter_source_per_screen;
                float screen_x = float(u_visible_origin_x) + gl_FragCoord.x;
                float center_x = (screen_x - u_seat_offset_x) * u_source_per_screen_x;
                int source_x_start = max(0, int(floor(center_x - support_radius)));
                int source_x_end = min(u_texture_size.x - 1, int(ceil(center_x + support_radius)));
                int source_y = clamp(u_source_y_start + int(floor(gl_FragCoord.y)), 0, u_texture_size.y - 1);
                #if EWA_MONO
                vec2 accum = vec2(0.0);
                #else
                vec2 texture_size = vec2(u_texture_size);
                vec4 accum_primary = vec4(0.0);
                vec4 accum_secondary = vec4(0.0);
                #endif
                vec2 weight_sum = vec2(0.0);
                for (int source_x = source_x_start; source_x <= source_x_end; ++source_x) {
                    float distance = abs(((float(source_x) + 0.5) - center_x) * screen_per_source);
                    vec2 weights = ewa_weights(u_ewa_lut, distance);
                    #if EWA_MONO
                    float luma_sample = texelFetch(u_rgba_or_luma, ivec2(source_x, source_y), 0).r;
                    float gray = to_srgb(vec3(luma_sample * u_luma_gain_offset.x - u_luma_gain_offset.y)).r;
                    accum += gray * weights;
                    #else
                    vec4 premultiplied_texel;
                    if (u_pixel_layout == PIXEL_LAYOUT_RGBA) {
                        vec4 rgba_texel = texelFetch(u_rgba_or_luma, ivec2(source_x, source_y), 0);
                        premultiplied_texel = vec4(to_srgb(rgba_texel.rgb) * rgba_texel.a, rgba_texel.a);
                    } else {
                        vec2 luma_px = vec2(float(source_x) + 0.5, float(source_y) + 0.5);
                        vec2 chroma_uv = luma_px * u_chroma_uv_per_step;
                        vec3 yuv_samples = vec3(
                            texelFetch(u_rgba_or_luma, ivec2(source_x, source_y), 0).r,
                            texture(u_cb, chroma_uv).r,
                            texture(u_cr, chroma_uv).r);
                        float alpha = texture(u_alpha, luma_px / texture_size).r * u_alpha_normalization;
                        vec3 rgb = to_srgb(u_color_matrix * yuv_samples - u_color_offset);
                        premultiplied_texel = vec4(rgb * alpha, alpha);
                    }
                    accum_primary += premultiplied_texel * weights.x;
                    accum_secondary += premultiplied_texel * weights.y;
                    #endif
                    weight_sum += weights;
                }
                #if EWA_MONO
                out_accum = vec4(accum, weight_sum);
                #else
                out_primary = accum_primary;
                out_secondary = accum_secondary;
                out_weights = vec4(weight_sum, 0.0, 0.0);
                #endif
            }
        "},
    )
}

fn ewa_vertical_shader_source(variant: EwaVariant) -> String {
    let variant_define = match variant {
        EwaVariant::Mono => "#define EWA_MONO 1\n",
        EwaVariant::Color => "#define EWA_MONO 0\n",
    };
    format!(
        "{header}{variant_define}{bayer}{lut}{body}",
        header = "#version 330 core\n",
        bayer = GLSL_BAYER4,
        lut = glsl_ewa_lut(),
        body = indoc! {"
            const float MIN_WEIGHT_SUM = 1e-6;
            in vec2 v_seat_px;
            layout(location = 0) out vec4 color;
            uniform int u_visible_origin_x;
            uniform int u_visible_width;
            uniform int u_source_y_start;
            uniform int u_source_y_end;
            uniform float u_source_per_screen_y;
            uniform int u_dither;
            uniform sampler2D u_scratch_primary;
            #if !EWA_MONO
            uniform sampler2D u_scratch_secondary;
            uniform sampler2D u_scratch_weights;
            #endif
            uniform sampler1D u_ewa_lut;
            void main() {
                float filter_source_per_screen = max(1.0, u_source_per_screen_y);
                float screen_per_source = 1.0 / filter_source_per_screen;
                float support_radius = EWA_SUPPORT * filter_source_per_screen;
                float center_y = v_seat_px.y * u_source_per_screen_y;
                int source_y_start = max(u_source_y_start, int(floor(center_y - support_radius)));
                int source_y_end = min(u_source_y_end - 1, int(ceil(center_y + support_radius)));
                int scratch_x = clamp(int(floor(gl_FragCoord.x)) - u_visible_origin_x, 0, u_visible_width - 1);
                #if EWA_MONO
                float accum = 0.0;
                #else
                vec4 accum = vec4(0.0);
                #endif
                float weight_sum = 0.0;
                for (int source_y = source_y_start; source_y <= source_y_end; ++source_y) {
                    float distance = abs(((float(source_y) + 0.5) - center_y) * screen_per_source);
                    vec2 weights = ewa_weights(u_ewa_lut, distance);
                    ivec2 scratch_texel = ivec2(scratch_x, source_y - u_source_y_start);
                    // Subtract the second singular term: its eigenvalue is negative.
                    #if EWA_MONO
                    vec4 horizontal_sample = texelFetch(u_scratch_primary, scratch_texel, 0);
                    accum += horizontal_sample.x * weights.x - horizontal_sample.y * weights.y;
                    weight_sum += horizontal_sample.z * weights.x - horizontal_sample.w * weights.y;
                    #else
                    vec4 primary = texelFetch(u_scratch_primary, scratch_texel, 0);
                    vec4 secondary = texelFetch(u_scratch_secondary, scratch_texel, 0);
                    vec2 horizontal_weights = texelFetch(u_scratch_weights, scratch_texel, 0).rg;
                    accum += primary * weights.x - secondary * weights.y;
                    weight_sum += horizontal_weights.x * weights.x - horizontal_weights.y * weights.y;
                    #endif
                }
                #if EWA_MONO
                float gray = clamp(accum / max(weight_sum, MIN_WEIGHT_SUM), 0.0, 1.0);
                if (u_dither != 0) {
                    gray = clamp(gray + (bayer4(gl_FragCoord.xy) - 0.5) / OUTPUT_MAX, 0.0, 1.0);
                }
                color = vec4(gray, gray, gray, 1.0);
                #else
                vec4 premultiplied = accum / max(weight_sum, MIN_WEIGHT_SUM);
                float alpha = clamp(premultiplied.a, 0.0, 1.0);
                vec3 rgb = clamp(premultiplied.rgb, vec3(0.0), vec3(alpha));
                if (u_dither != 0) {
                    float dither_offset = (bayer4(gl_FragCoord.xy) - 0.5) / OUTPUT_MAX;
                    rgb = clamp(rgb + dither_offset * alpha, vec3(0.0), vec3(alpha));
                }
                color = vec4(rgb, alpha);
                #endif
            }
        "},
    )
}

/// Seat rect and page dims agree within float rounding; page dims are
/// whole pixels.
const SEAT_PAGE_DIMS_TOLERANCE: f32 = 1e-3;

const ROLE_COVER_MARGIN_PX: f32 = 1.0;
const MINIFY_SCALE_TOLERANCE: f32 = 1e-3;
const SLICE_PAD_TEXELS: f32 = 1.0;
const QUAD_VERTEX_COUNT: i32 = 4;
const EWA_LUT_SAMPLER_UNIT: i32 = 4;
const EWA_LUT_TEXTURE_UNIT: u32 = glow::TEXTURE4;

const EWA_JINC_FIRST_ZERO: f64 = 1.219_669_891_266_504_5;
const EWA_SUPPORT: f64 = 3.238_315_484_166_236;
const EWA_LUT_INTERVALS: i32 = 128;
const EWA_LUT_WIDTH: i32 = EWA_LUT_INTERVALS + 1;
const EWA_LUT_LEN: usize = EWA_LUT_WIDTH as usize;
const EWA_GRID_LEN: usize = EWA_LUT_LEN * 2 - 1;

pub struct PageTextures {
    gl: Arc<Context>,
    vao: EmptyVao,
    pipeline: Pipeline,
    ewa: EwaResampler,
    pages: HashMap<usize, EnumMap<Role, Option<PageTexture>>>,
    max_texture_edge: u32,
}

struct EmptyVao {
    gl: Arc<Context>,
    vao: VertexArray,
}

impl Drop for EmptyVao {
    fn drop(&mut self) {
        // SAFETY: Context is current (`Window` drop order) and `self` owns `vao`.
        unsafe {
            self.gl.delete_vertex_array(self.vao);
        }
    }
}

struct ColorSourceUniforms {
    chroma_uv_per_step: UniformLocation,
    color_matrix: UniformLocation,
    color_offset: UniformLocation,
    alpha_normalization: UniformLocation,
    pixel_layout: UniformLocation,
    transfer_code: UniformLocation,
}

struct HorizontalCommonUniforms {
    visible_origin_x: UniformLocation,
    source_y_start: UniformLocation,
    texture_size: UniformLocation,
    seat_offset_x: UniformLocation,
    source_per_screen_x: UniformLocation,
}

struct VerticalUniforms {
    window_size: UniformLocation,
    seat_offset: UniformLocation,
    seat_size: UniformLocation,
    visible_origin_x: UniformLocation,
    visible_width: UniformLocation,
    source_y_start: UniformLocation,
    source_y_end: UniformLocation,
    source_per_screen_y: UniformLocation,
    dither: UniformLocation,
}

struct HorizontalMonoProgram {
    objects: ProgramObjects,
    common: HorizontalCommonUniforms,
    luma_gain_offset: UniformLocation,
    transfer_code: UniformLocation,
}

struct HorizontalColorProgram {
    objects: ProgramObjects,
    common: HorizontalCommonUniforms,
    color: ColorSourceUniforms,
}

struct VerticalProgram {
    objects: ProgramObjects,
    uniforms: VerticalUniforms,
}

struct EwaResampler {
    gl: Arc<Context>,
    max_texture_edge: u32,
    lut: Texture,
    fbo: Framebuffer,
    scratch: [Option<Texture>; SCRATCH_TARGET_COUNT],
    scratch_capacity: (u32, u32),
    horizontal_mono: HorizontalMonoProgram,
    vertical_mono: VerticalProgram,
    horizontal_color: HorizontalColorProgram,
    vertical_color: VerticalProgram,
}

impl Drop for EwaResampler {
    fn drop(&mut self) {
        // SAFETY: Context is current (`Window` drop order); `self` owns `lut`, `scratch`, and `fbo`.
        unsafe {
            self.gl.delete_texture(self.lut);
            for texture in self.scratch.into_iter().flatten() {
                self.gl.delete_texture(texture);
            }
            self.gl.delete_framebuffer(self.fbo);
        }
    }
}

/// One cached role's GPU state.
struct PageTexture {
    gl: Arc<Context>,
    dims: (u32, u32),
    page_dims: (u32, u32),
    transfer: Transfer,
    needs_dither: bool,
    layout: PixelLayout,
}

impl Drop for PageTexture {
    fn drop(&mut self) {
        // SAFETY: `Window` drops `PageTextures` while the context is still
        // current and alive, and `self` owns these texture handles uniquely.
        unsafe {
            match self.layout {
                PixelLayout::Rgba { texture } => {
                    self.gl.delete_texture(texture);
                }
                PixelLayout::Mono { luma, .. } => {
                    self.gl.delete_texture(luma);
                }
                PixelLayout::Yuv {
                    luma,
                    cb,
                    cr,
                    alpha,
                    ..
                } => {
                    for texture in [luma, cb, cr, alpha] {
                        self.gl.delete_texture(texture);
                    }
                }
            }
        }
    }
}

enum PixelLayout {
    Rgba {
        texture: Texture,
    },
    Mono {
        luma: Texture,
        gain: f32,
        offset: f32,
    },
    Yuv {
        luma: Texture,
        cb: Texture,
        cr: Texture,
        alpha: Texture,
        /// Luma texel -> chroma UV, subsample-folded for this texture's dims.
        chroma_uv_per_texel: (f32, f32),
        color_conversion: ColorConversion,
        /// Storage-sample normalization for the alpha plane.
        alpha_normalization: f32,
    },
}

const PIXEL_LAYOUT_RGBA: i32 = 0;
const PIXEL_LAYOUT_YUV: i32 = 1;
const PIXEL_LAYOUT_MONO: i32 = 2;

#[derive(Clone, Copy)]
struct DrawTarget {
    window: (u32, u32),
    clip: (u32, u32),
}

#[derive(Clone, Copy)]
struct SeatGeometry {
    origin: (f32, f32),
    size: (f32, f32),
}

/// Fold storage samples to gamma RGB at upload: `matrix * samples - offset`.
/// Samples are 0..1 storage units; R16 sampling already divides by 65535.
#[derive(Debug, Clone, Copy)]
struct ColorConversion {
    /// Column-major 3x3.
    matrix: [f32; 9],
    offset: [f32; 3],
}

fn slot_roles(slots: &EnumMap<Role, Option<PageTexture>>) -> Roles {
    let mut roles = Roles::default();
    for (role, slot) in slots {
        if slot.is_some() {
            roles.set(role);
        }
    }
    roles
}

/// Keep `Fit` whenever it covers `screen_edge` so zooming back out avoids
/// re-filtering `Full`.
fn best_texture(
    slots: &EnumMap<Role, Option<PageTexture>>,
    screen_edge: f32,
) -> Option<&PageTexture> {
    if let Some(fit) = slots[Role::Fit].as_ref() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "texture edge stays far below f32's exact-integer range"
        )]
        let fit_edge = fit.dims.0.max(fit.dims.1) as f32;
        if fit_edge + ROLE_COVER_MARGIN_PX >= screen_edge {
            return Some(fit);
        }
    }
    slots[slot_roles(slots).best()?].as_ref()
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct VisibleSlice {
    visible_origin: (i32, i32),
    visible_size: (u32, u32),
    source_y_range: (u32, u32),
    source_per_screen: (f32, f32),
}

impl VisibleSlice {
    fn source_row_count(self) -> u32 {
        self.source_y_range.1 - self.source_y_range.0
    }

    fn is_minified(self) -> bool {
        self.source_per_screen.0 > 1.0 + MINIFY_SCALE_TOLERANCE
            || self.source_per_screen.1 > 1.0 + MINIFY_SCALE_TOLERANCE
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "coordinates fit i32 and exact f32, and clamping keeps u32 casts non-negative"
)]
fn compute_visible_slice(
    seat: SeatGeometry,
    clip_size: (u32, u32),
    texture_dims: (u32, u32),
) -> Option<VisibleSlice> {
    if seat.size.0 <= 0.0
        || seat.size.1 <= 0.0
        || clip_size.0 == 0
        || clip_size.1 == 0
        || texture_dims.0 == 0
        || texture_dims.1 == 0
    {
        return None;
    }
    let clip_width = gl_int(clip_size.0);
    let clip_height = gl_int(clip_size.1);
    let left = (seat.origin.0.floor() as i32).clamp(0, clip_width);
    let right =
        ((seat.origin.0 + seat.size.0).ceil() as i32).clamp(0, clip_width);
    let top = (seat.origin.1.floor() as i32).clamp(0, clip_height);
    let bottom = ((seat.origin.1 + seat.size.1).ceil() as i32)
        .clamp(0, clip_height);
    if right <= left || bottom <= top {
        return None;
    }
    let texture_px = geometry::px_f32(texture_dims);
    let source_per_screen_x = texture_px.0 / seat.size.0;
    let source_per_screen_y = texture_px.1 / seat.size.1;
    let support_radius_y = (EWA_SUPPORT as f32)
        .mul_add(source_per_screen_y.max(1.0), SLICE_PAD_TEXELS);
    let center_y_start =
        (top as f32 + 0.5 - seat.origin.1) * source_per_screen_y;
    let center_y_end =
        (bottom as f32 - 0.5 - seat.origin.1) * source_per_screen_y;
    let texture_height = gl_int(texture_dims.1);
    let source_y_start = ((center_y_start - support_radius_y).floor()
        as i32)
        .clamp(0, texture_height - 1);
    let source_y_end = ((center_y_end + support_radius_y).ceil() as i32
        + 1)
    .clamp(source_y_start + 1, texture_height);
    Some(VisibleSlice {
        visible_origin: (left, top),
        visible_size: ((right - left) as u32, (bottom - top) as u32),
        source_y_range: (source_y_start as u32, source_y_end as u32),
        source_per_screen: (source_per_screen_x, source_per_screen_y),
    })
}

/// Refuse a page over `MAX_TEXTURE_SIZE` before upload; an oversized
/// upload samples black.
fn check_texture_dims(
    dims: (u32, u32),
    max_edge: u32,
) -> Result<(), TextureFault> {
    if dims.0 == 0 || dims.1 == 0 || dims.0 > max_edge || dims.1 > max_edge
    {
        return Err(TextureFault { dims, max_edge });
    }
    Ok(())
}

impl PageTextures {
    fn new(gl: Arc<Context>) -> eros::Result<Self> {
        // SAFETY: MAX_TEXTURE_SIZE is a pure query on the current context.
        let Ok(max_texture_edge) = u32::try_from(unsafe {
            gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE)
        }) else {
            eros::bail!("GL MAX_TEXTURE_SIZE is not a positive GLint");
        };
        if max_texture_edge < cache::THUMB_EDGE {
            return Err(eros::error!(
                "GL MAX_TEXTURE_SIZE ({max_texture_edge}) is below THUMB_EDGE ({})",
                cache::THUMB_EDGE
            ));
        }
        // SAFETY: Context is current (module invariant).
        let raw_vao =
            unsafe { gl.create_vertex_array() }.map_err(|error| {
                eros::error!(
                    "GL vertex array object creation failed: {error}"
                )
            })?;
        let vao = EmptyVao {
            gl: Arc::clone(&gl),
            vao: raw_vao,
        };
        label(&gl, glow::VERTEX_ARRAY, vao.vao.0.get(), "page quad");
        // SAFETY: Context is current and `vao.vao` was just created.
        unsafe {
            gl.bind_vertex_array(Some(vao.vao));
        }
        let pipeline = create_pipeline(Arc::clone(&gl))?;
        let ewa = EwaResampler::new(Arc::clone(&gl), max_texture_edge)?;
        Ok(Self {
            gl,
            vao,
            pipeline,
            ewa,
            pages: HashMap::new(),
            max_texture_edge,
        })
    }

    pub fn max_texture_edge(&self) -> u32 {
        self.max_texture_edge
    }

    /// Upload `image` as `page`'s `role`; no-op if present, never evicts.
    pub fn insert(
        &mut self,
        page: usize,
        role: Role,
        image: &ValidatedImage,
    ) -> Result<(), TextureFault> {
        if self.has(page, role) {
            return Ok(());
        }
        let dims = (image.width(), image.height());
        check_texture_dims(dims, self.max_texture_edge)?;
        let texture = upload_page_texture(Arc::clone(&self.gl), image);
        label_entry(&self.gl, page, role, &texture);
        self.pages.entry(page).or_default()[role] = Some(texture);
        Ok(())
    }

    /// Keeps the page's role slot.
    pub fn update(
        &mut self,
        page: usize,
        role: Role,
        image: &ValidatedImage,
    ) -> Result<(), TextureFault> {
        let dims = (image.width(), image.height());
        check_texture_dims(dims, self.max_texture_edge)?;
        if let Some(PageTexture {
            dims: present_dims,
            layout: PixelLayout::Rgba { texture },
            transfer,
            ..
        }) =
            self.pages.get(&page).and_then(|slots| slots[role].as_ref())
            && *present_dims == dims
            && *transfer == image.transfer()
            && let WirePixels::Rgba8(bytes) = image.pixels()
        {
            update_rgba(&self.gl, *texture, bytes, dims);
            return Ok(());
        }
        let empty = self.pages.get_mut(&page).is_some_and(|slots| {
            slots[role] = None;
            slots.values().all(Option::is_none)
        });
        if empty {
            self.pages.remove(&page);
        }
        self.insert(page, role, image)
    }

    pub fn roles(&self, page: usize) -> Roles {
        self.pages
            .get(&page)
            .map_or_else(Roles::default, slot_roles)
    }

    fn has(&self, page: usize, role: Role) -> bool {
        self.pages
            .get(&page)
            .is_some_and(|slots| slots[role].is_some())
    }

    pub fn has_any(&self, page: usize) -> bool {
        self.pages
            .get(&page)
            .is_some_and(|slots| slots.values().any(Option::is_some))
    }

    pub fn retain(&mut self, keep: impl Fn(usize, Role) -> bool) {
        self.pages.retain(|&page, slots| {
            for (role, slot) in slots.iter_mut() {
                if slot.is_some() && !keep(page, role) {
                    *slot = None;
                }
            }
            slots.values().any(Option::is_some)
        });
    }

    pub fn clear_all(&mut self) {
        self.pages.clear();
    }

    /// Does not clear the framebuffer.
    pub fn draw(
        &mut self,
        window_size: (u32, u32),
        viewport: Viewport,
        spread_layout: &SpreadLayout,
        seat_pages: [Option<usize>; 2],
        placement: &Placement,
    ) {
        if window_size.0 == 0 || window_size.1 == 0 {
            return;
        }
        // GL scissor is bottom-origin and the viewport hugs the window's
        // top-left, so only the height converts.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "viewport size is clamped non-negative whole physical px held as f32"
        )]
        let (viewport_width, viewport_height) =
            (viewport.size().0 as u32, viewport.size().1 as u32);
        let clip_size = (
            viewport_width.min(window_size.0),
            viewport_height.min(window_size.1),
        );
        if clip_size.0 == 0 || clip_size.1 == 0 {
            return;
        }
        let target = DrawTarget {
            window: window_size,
            clip: clip_size,
        };
        begin_frame(&self.gl, self.vao.vao, target);
        for (rect, page) in
            spread_layout.rects.iter().zip(seat_pages.iter())
        {
            let Some(rect) = rect else { continue };
            let Some(page) = page else { continue };
            let screen_size = (
                rect.size.0 * placement.scale,
                rect.size.1 * placement.scale,
            );
            if screen_size.0 <= 0.0 || screen_size.1 <= 0.0 {
                continue;
            }
            let screen_edge = screen_size.0.max(screen_size.1);
            let Some(texture) = self
                .pages
                .get(page)
                .and_then(|slots| best_texture(slots, screen_edge))
            else {
                continue;
            };
            check_seat_page_dims(*page, rect, texture.page_dims);
            // Whole-pixel rect origin: at 1:1 zoom a fractional origin
            // makes NEAREST draw uneven texel rows.
            let seat = SeatGeometry {
                origin: (
                    rect.pos
                        .0
                        .mul_add(placement.scale, placement.offset.0)
                        .round(),
                    rect.pos
                        .1
                        .mul_add(placement.scale, placement.offset.1)
                        .round(),
                ),
                size: screen_size,
            };
            let Some(slice) =
                compute_visible_slice(seat, clip_size, texture.dims)
            else {
                continue;
            };
            let drew_ewa = slice.is_minified()
                && self.ewa.draw_minified(target, texture, seat, slice);
            if !drew_ewa {
                draw_page(
                    &self.gl,
                    &self.pipeline,
                    target,
                    texture,
                    seat,
                    slice.source_per_screen,
                );
            }
        }
        // SAFETY: Context is current (`Window`); restore default FBO, unclipped
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
    rect: &PageRect,
    page_dims: (u32, u32),
) {
    // A seat rect is in page pixels and spans the whole page.
    let expected = geometry::px_f32(page_dims);
    let aligned = (rect.size.0 - expected.0).abs()
        <= SEAT_PAGE_DIMS_TOLERANCE
        && (rect.size.1 - expected.1).abs() <= SEAT_PAGE_DIMS_TOLERANCE;
    if !aligned {
        tracing::debug!(
            page,
            rect = ?rect.size,
            ?page_dims,
            "seat layout disagrees with the texture's page dims"
        );
    }
    debug_assert!(
        aligned,
        "seat rect and texture page dims disagree for page \
         {page}: rect {:?}, page dims {page_dims:?}",
        rect.size
    );
}

struct ShaderGuard<'gl> {
    gl: &'gl Context,
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

fn compile_shader<'gl>(
    gl: &'gl Context,
    kind: u32,
    src: &str,
) -> eros::Result<ShaderGuard<'gl>> {
    // SAFETY: Caller ensures the GL context is current on the render thread.
    let raw = unsafe { gl.create_shader(kind) }.map_err(|error| {
        eros::error!("GL shader object creation failed: {error}")
    })?;
    let guard = ShaderGuard { gl, shader: raw };
    // SAFETY: `guard.shader` was just created on the current context.
    unsafe {
        gl.shader_source(guard.shader, src);
        gl.compile_shader(guard.shader);
        if !gl.get_shader_compile_status(guard.shader) {
            let info = gl.get_shader_info_log(guard.shader);
            return Err(eros::error!("GL shader compile failed: {info}"));
        }
    }
    Ok(guard)
}

struct ProgramObjects {
    gl: Arc<Context>,
    program: Program,
}

impl Drop for ProgramObjects {
    fn drop(&mut self) {
        // SAFETY: Context is current (`Window` drop order) and `self` owns `program`.
        unsafe {
            self.gl.delete_program(self.program);
        }
    }
}

struct Pipeline {
    objects: ProgramObjects,
    window_size: UniformLocation,
    seat_offset: UniformLocation,
    seat_size: UniformLocation,
    page_uv_per_screen: UniformLocation,
    luma_gain_offset: UniformLocation,
    dither: UniformLocation,
    color: ColorSourceUniforms,
}

fn uniform_location(
    gl: &Context,
    program: Program,
    name: &str,
) -> UniformLocation {
    // SAFETY: Context is current (module invariant) and `program` is linked.
    let found = unsafe { gl.get_uniform_location(program, name) };
    #[expect(
        clippy::expect_used,
        reason = "declared and used in the in-tree shader sources, so a miss is an in-tree regression"
    )]
    found.expect("uniform location in shader")
}

fn link_program(
    gl: Arc<Context>,
    vertex_source: &str,
    fragment_source: &str,
    samplers: &[(i32, &str)],
    debug_name: &str,
) -> eros::Result<ProgramObjects> {
    // SAFETY: Context is current (module invariant).
    let program = unsafe { gl.create_program() }.map_err(|error| {
        eros::error!("GL program object creation failed: {error}")
    })?;
    let objects = ProgramObjects { gl, program };
    {
        let vertex_shader = compile_shader(
            &objects.gl,
            glow::VERTEX_SHADER,
            vertex_source,
        )?;
        let fragment_shader = compile_shader(
            &objects.gl,
            glow::FRAGMENT_SHADER,
            fragment_source,
        )?;
        // SAFETY: Context is current; `objects.program` and both shaders are live.
        unsafe {
            objects
                .gl
                .attach_shader(objects.program, vertex_shader.shader);
            objects
                .gl
                .attach_shader(objects.program, fragment_shader.shader);
            objects.gl.link_program(objects.program);
            if !objects.gl.get_program_link_status(objects.program) {
                let info =
                    objects.gl.get_program_info_log(objects.program);
                return Err(eros::error!(
                    "GL program link failed ({debug_name}): {info}"
                ));
            }
        }
    }
    // Sampler units are fixed; draw calls rebind textures only.
    // SAFETY: Context is current and `objects.program` is linked.
    unsafe {
        objects.gl.use_program(Some(objects.program));
        for &(unit, name) in samplers {
            let location =
                uniform_location(&objects.gl, objects.program, name);
            objects.gl.uniform_1_i32(Some(&location), unit);
        }
    }
    label(
        &objects.gl,
        glow::PROGRAM,
        objects.program.0.get(),
        debug_name,
    );
    Ok(objects)
}

fn create_pipeline(gl: Arc<Context>) -> eros::Result<Pipeline> {
    let fragment_source = fragment_shader_source();
    let objects = link_program(
        gl,
        VERTEX_SHADER_SOURCE,
        &fragment_source,
        &[
            (0, "u_rgba_or_luma"),
            (1, "u_cb"),
            (2, "u_cr"),
            (3, "u_alpha"),
        ],
        "page program",
    )?;
    let location =
        |name: &str| uniform_location(&objects.gl, objects.program, name);
    let color = ColorSourceUniforms::lookup(&objects.gl, objects.program);
    Ok(Pipeline {
        window_size: location("u_window_size"),
        seat_offset: location("u_seat_offset"),
        seat_size: location("u_seat_size"),
        page_uv_per_screen: location("u_page_uv_per_screen"),
        luma_gain_offset: location("u_luma_gain_offset"),
        dither: location("u_dither"),
        color,
        objects,
    })
}

fn jinc(radius: f64) -> f64 {
    const SERIES_TERMS: usize = 35;

    let half_arg = 0.5 * PI * radius;
    let neg_square = -(half_arg * half_arg);
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    for term_index in 1..=SERIES_TERMS {
        #[expect(
            clippy::cast_precision_loss,
            reason = "term_index <= 35, exact in f64"
        )]
        let index_f64 = term_index as f64;
        term *= neg_square / (index_f64 * (index_f64 + 1.0));
        sum += term;
    }
    sum
}

fn ewa_lanczos3(radius: f64) -> f64 {
    if radius >= EWA_SUPPORT {
        return 0.0;
    }
    jinc(radius) * jinc(radius * (EWA_JINC_FIRST_ZERO / EWA_SUPPORT))
}

fn dot_product(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(lhs, rhs)| lhs * rhs).sum()
}

fn project_out(vector: &mut [f64], unit_basis: &[f64]) {
    let projection = dot_product(vector, unit_basis);
    for (slot, &basis) in vector.iter_mut().zip(unit_basis) {
        *slot = projection.mul_add(-basis, *slot);
    }
}

fn normalize_vector(vector: &mut [f64]) {
    let norm = vector
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for slot in vector.iter_mut() {
            *slot /= norm;
        }
    }
}

fn power_eigen_symmetric(
    matrix: &[f64],
    center_index: usize,
    initial_vector: &[f64],
    orthogonal_basis: Option<&[f64]>,
) -> (f64, Vec<f64>) {
    const POWER_ITERATIONS: usize = 32;

    let mut vector = initial_vector.to_vec();
    if let Some(basis) = orthogonal_basis {
        project_out(&mut vector, basis);
    }
    normalize_vector(&mut vector);

    let mut next = vec![0.0_f64; EWA_GRID_LEN];
    let rows = matrix.as_chunks::<EWA_GRID_LEN>().0;
    for _ in 0..POWER_ITERATIONS {
        for (target, row) in next.iter_mut().zip(rows) {
            *target = dot_product(row, &vector);
        }
        if let Some(basis) = orthogonal_basis {
            project_out(&mut next, basis);
        }
        normalize_vector(&mut next);
        vector.copy_from_slice(&next);
    }

    let eigenvalue: f64 = vector
        .iter()
        .zip(rows)
        .map(|(&component, row)| component * dot_product(row, &vector))
        .sum();
    if vector.get(center_index).is_some_and(|&center| center < 0.0) {
        for slot in &mut vector {
            *slot = -*slot;
        }
    }
    (eigenvalue, vector)
}

/// Rank-2 symmetric eigendecomposition of circular 2D `EWA-Lanczos3`:
/// `K(dx, dy) ≈ g0(|dx|) * g0(|dy|) - g1(|dx|) * g1(|dy|)`.
fn ewa_rank2_lut() -> [[f32; 2]; EWA_LUT_LEN] {
    #[expect(
        clippy::cast_precision_loss,
        reason = "EWA_GRID_LEN is 257, exact in f64"
    )]
    let grid_step = 2.0 * EWA_SUPPORT / (EWA_GRID_LEN - 1) as f64;
    #[expect(
        clippy::cast_precision_loss,
        reason = "grid index <= 256, exact in f64"
    )]
    let coords: Vec<f64> = (0..EWA_GRID_LEN)
        .map(|index| (index as f64).mul_add(grid_step, -EWA_SUPPORT))
        .collect();

    let mut matrix = Vec::with_capacity(EWA_GRID_LEN * EWA_GRID_LEN);
    for &coord_y in &coords {
        for &coord_x in &coords {
            matrix.push(ewa_lanczos3(coord_x.hypot(coord_y)));
        }
    }

    let center_index = EWA_LUT_LEN - 1;
    let primary_init: Vec<f64> = coords
        .iter()
        .map(|&coord| (0.5 * PI * coord / EWA_SUPPORT).cos())
        .collect();
    let (primary_eigenvalue, primary_vector) =
        power_eigen_symmetric(&matrix, center_index, &primary_init, None);

    for (row, &row_entry) in matrix
        .as_chunks_mut::<EWA_GRID_LEN>()
        .0
        .iter_mut()
        .zip(&primary_vector)
    {
        let row_scale = primary_eigenvalue * row_entry;
        for (cell, &col_entry) in row.iter_mut().zip(&primary_vector) {
            *cell = row_scale.mul_add(-col_entry, *cell);
        }
    }

    let secondary_init: Vec<f64> = coords
        .iter()
        .map(|&coord| (2.0 * coord / EWA_SUPPORT).cos())
        .collect();
    let (secondary_eigenvalue, secondary_vector) = power_eigen_symmetric(
        &matrix,
        center_index,
        &secondary_init,
        Some(&primary_vector),
    );

    let primary_scale = primary_eigenvalue.abs().sqrt();
    let secondary_scale = secondary_eigenvalue.abs().sqrt();
    let mut lut = [[0.0_f32; 2]; EWA_LUT_LEN];
    #[expect(
        clippy::cast_possible_truncation,
        reason = "LUT stores GPU texture weights in f32"
    )]
    if let (Some(primary_half), Some(secondary_half)) = (
        primary_vector.get(center_index..),
        secondary_vector.get(center_index..),
    ) {
        for ((slot, &primary_weight), &secondary_weight) in
            lut.iter_mut().zip(primary_half).zip(secondary_half)
        {
            *slot = [
                (primary_scale * primary_weight) as f32,
                (secondary_scale * secondary_weight) as f32,
            ];
        }
    }
    if let Some(last) = lut.last_mut() {
        *last = [0.0, 0.0];
    }
    lut
}

fn upload_ewa_lut(gl: &Context) -> Texture {
    let lut = ewa_rank2_lut();
    // SAFETY: Context is current (module invariant); `lut` holds `EWA_LUT_WIDTH` RG floats.
    unsafe {
        #[expect(
            clippy::expect_used,
            reason = "texture-creation failure panics by policy"
        )]
        let texture = gl.create_texture().expect("GL EWA LUT texture");
        gl.bind_texture(glow::TEXTURE_1D, Some(texture));
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);
        gl.tex_parameter_i32(
            glow::TEXTURE_1D,
            glow::TEXTURE_WRAP_S,
            gl_int(glow::CLAMP_TO_EDGE),
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_1D,
            glow::TEXTURE_MIN_FILTER,
            gl_int(glow::LINEAR),
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_1D,
            glow::TEXTURE_MAG_FILTER,
            gl_int(glow::LINEAR),
        );
        gl.tex_image_1d(
            glow::TEXTURE_1D,
            0,
            gl_int(glow::RG32F),
            EWA_LUT_WIDTH,
            0,
            glow::RG,
            glow::FLOAT,
            PixelUnpackData::Slice(Some(bytemuck::cast_slice(&lut))),
        );
        gl.bind_texture(glow::TEXTURE_1D, None);
        label(gl, glow::TEXTURE, texture.0.get(), "ewa rank2 lut");
        texture
    }
}

const SCRATCH_TARGET_COUNT: usize = 3;
const MONO_SCRATCH_TARGET_COUNT: usize = 1;

const SCRATCH_SPECS: [(u32, u32, &str); SCRATCH_TARGET_COUNT] = [
    (glow::RGBA16F, glow::RGBA, "ewa scratch primary"),
    (glow::RGBA16F, glow::RGBA, "ewa scratch secondary"),
    (glow::RG16F, glow::RG, "ewa scratch weights"),
];

enum ScratchTargets {
    Mono(Texture),
    Color([Texture; SCRATCH_TARGET_COUNT]),
}

fn alloc_scratch_texture(
    gl: &Context,
    existing: Option<Texture>,
    dims: (u32, u32),
    internal_format: u32,
    format: u32,
    debug_name: &str,
) -> Texture {
    // SAFETY: Context is current (module invariant); caller keeps `dims` within `1..=max_texture_edge`.
    unsafe {
        #[expect(
            clippy::expect_used,
            reason = "texture-creation failure panics by policy"
        )]
        let texture = existing.unwrap_or_else(|| {
            let created =
                gl.create_texture().expect("GL scratch texture object");
            gl.bind_texture(glow::TEXTURE_2D, Some(created));
            for (param, value) in [
                (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_MIN_FILTER, glow::NEAREST),
                (glow::TEXTURE_MAG_FILTER, glow::NEAREST),
            ] {
                gl.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    param,
                    gl_int(value),
                );
            }
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_BASE_LEVEL,
                0,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAX_LEVEL,
                0,
            );
            label(gl, glow::TEXTURE, created.0.get(), debug_name);
            created
        });
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            gl_int(internal_format),
            gl_int(dims.0),
            gl_int(dims.1),
            0,
            format,
            glow::FLOAT,
            PixelUnpackData::Slice(None),
        );
        texture
    }
}

fn lookup_horizontal_common(
    gl: &Context,
    program: Program,
) -> HorizontalCommonUniforms {
    let location = |name: &str| uniform_location(gl, program, name);
    HorizontalCommonUniforms {
        visible_origin_x: location("u_visible_origin_x"),
        source_y_start: location("u_source_y_start"),
        texture_size: location("u_texture_size"),
        seat_offset_x: location("u_seat_offset_x"),
        source_per_screen_x: location("u_source_per_screen_x"),
    }
}

fn create_vertical_program(
    gl: Arc<Context>,
    variant: EwaVariant,
    samplers: &[(i32, &str)],
    debug_name: &str,
) -> eros::Result<VerticalProgram> {
    let fragment_source = ewa_vertical_shader_source(variant);
    let objects = link_program(
        gl,
        VERTEX_SHADER_SOURCE,
        &fragment_source,
        samplers,
        debug_name,
    )?;
    let location =
        |name: &str| uniform_location(&objects.gl, objects.program, name);
    let uniforms = VerticalUniforms {
        window_size: location("u_window_size"),
        seat_offset: location("u_seat_offset"),
        seat_size: location("u_seat_size"),
        visible_origin_x: location("u_visible_origin_x"),
        visible_width: location("u_visible_width"),
        source_y_start: location("u_source_y_start"),
        source_y_end: location("u_source_y_end"),
        source_per_screen_y: location("u_source_per_screen_y"),
        dither: location("u_dither"),
    };
    Ok(VerticalProgram { objects, uniforms })
}

fn bind_horizontal_common(
    gl: &Context,
    common: &HorizontalCommonUniforms,
    slice: VisibleSlice,
    texture_dims: (u32, u32),
    seat_origin_x: f32,
) {
    // SAFETY: Context is current (module invariant) and the active program owns `common`.
    unsafe {
        gl.uniform_1_i32(
            Some(&common.visible_origin_x),
            slice.visible_origin.0,
        );
        gl.uniform_1_i32(
            Some(&common.source_y_start),
            gl_int(slice.source_y_range.0),
        );
        gl.uniform_2_i32(
            Some(&common.texture_size),
            gl_int(texture_dims.0),
            gl_int(texture_dims.1),
        );
        gl.uniform_1_f32(Some(&common.seat_offset_x), seat_origin_x);
        gl.uniform_1_f32(
            Some(&common.source_per_screen_x),
            slice.source_per_screen.0,
        );
    }
}

impl EwaResampler {
    fn new(gl: Arc<Context>, max_texture_edge: u32) -> eros::Result<Self> {
        let horizontal_mono_source =
            ewa_horizontal_shader_source(EwaVariant::Mono);
        let horizontal_mono_objects = link_program(
            Arc::clone(&gl),
            FULLSCREEN_VERTEX_SHADER_SOURCE,
            &horizontal_mono_source,
            &[(0, "u_rgba_or_luma"), (EWA_LUT_SAMPLER_UNIT, "u_ewa_lut")],
            "ewa horiz mono",
        )?;
        let horizontal_mono = HorizontalMonoProgram {
            common: lookup_horizontal_common(
                &horizontal_mono_objects.gl,
                horizontal_mono_objects.program,
            ),
            luma_gain_offset: uniform_location(
                &horizontal_mono_objects.gl,
                horizontal_mono_objects.program,
                "u_luma_gain_offset",
            ),
            transfer_code: uniform_location(
                &horizontal_mono_objects.gl,
                horizontal_mono_objects.program,
                "u_transfer_code",
            ),
            objects: horizontal_mono_objects,
        };

        let horizontal_color_source =
            ewa_horizontal_shader_source(EwaVariant::Color);
        let horizontal_color_objects = link_program(
            Arc::clone(&gl),
            FULLSCREEN_VERTEX_SHADER_SOURCE,
            &horizontal_color_source,
            &[
                (0, "u_rgba_or_luma"),
                (1, "u_cb"),
                (2, "u_cr"),
                (3, "u_alpha"),
                (EWA_LUT_SAMPLER_UNIT, "u_ewa_lut"),
            ],
            "ewa horiz color",
        )?;
        let horizontal_color = HorizontalColorProgram {
            common: lookup_horizontal_common(
                &horizontal_color_objects.gl,
                horizontal_color_objects.program,
            ),
            color: ColorSourceUniforms::lookup(
                &horizontal_color_objects.gl,
                horizontal_color_objects.program,
            ),
            objects: horizontal_color_objects,
        };

        let vertical_mono = create_vertical_program(
            Arc::clone(&gl),
            EwaVariant::Mono,
            &[
                (0, "u_scratch_primary"),
                (EWA_LUT_SAMPLER_UNIT, "u_ewa_lut"),
            ],
            "ewa vert mono",
        )?;
        let vertical_color = create_vertical_program(
            Arc::clone(&gl),
            EwaVariant::Color,
            &[
                (0, "u_scratch_primary"),
                (1, "u_scratch_secondary"),
                (2, "u_scratch_weights"),
                (EWA_LUT_SAMPLER_UNIT, "u_ewa_lut"),
            ],
            "ewa vert color",
        )?;

        let lut = upload_ewa_lut(&gl);
        // SAFETY: Context is current (module invariant).
        let fbo = unsafe { gl.create_framebuffer() }.map_err(|error| {
            // SAFETY: Context is current (module invariant) and `lut` is owned locally.
            unsafe {
                gl.delete_texture(lut);
            }
            eros::error!("GL framebuffer creation failed: {error}")
        })?;
        label(&gl, glow::FRAMEBUFFER, fbo.0.get(), "ewa scratch fbo");

        Ok(Self {
            gl,
            max_texture_edge,
            lut,
            fbo,
            scratch: [None; SCRATCH_TARGET_COUNT],
            scratch_capacity: (0, 0),
            horizontal_mono,
            vertical_mono,
            horizontal_color,
            vertical_color,
        })
    }

    /// Leaves `self.fbo` bound on success; restores `FRAMEBUFFER = 0` on failure.
    fn bind_scratch_fbo(
        &mut self,
        dims: (u32, u32),
        variant: EwaVariant,
    ) -> Option<ScratchTargets> {
        if dims.0 == 0
            || dims.1 == 0
            || dims.0 > self.max_texture_edge
            || dims.1 > self.max_texture_edge
        {
            return None;
        }
        let needs_grow = self.scratch[0].is_none()
            || dims.0 > self.scratch_capacity.0
            || dims.1 > self.scratch_capacity.1;
        if needs_grow {
            self.scratch_capacity = (
                self.scratch_capacity.0.max(dims.0),
                self.scratch_capacity.1.max(dims.1),
            );
        }
        let target_count = match variant {
            EwaVariant::Mono => MONO_SCRATCH_TARGET_COUNT,
            EwaVariant::Color => SCRATCH_TARGET_COUNT,
        };
        for (index, (slot, &(internal_format, format, debug_name))) in
            self.scratch.iter_mut().zip(&SCRATCH_SPECS).enumerate()
        {
            if (needs_grow && slot.is_some())
                || (index < target_count && slot.is_none())
            {
                *slot = Some(alloc_scratch_texture(
                    &self.gl,
                    *slot,
                    self.scratch_capacity,
                    internal_format,
                    format,
                    debug_name,
                ));
            }
        }
        let [Some(primary), secondary_slot, weights_slot] = self.scratch
        else {
            return None;
        };
        let all_draw_buffers = [
            glow::COLOR_ATTACHMENT0,
            glow::COLOR_ATTACHMENT1,
            glow::COLOR_ATTACHMENT2,
        ];
        let mono_draw_buffer = [glow::COLOR_ATTACHMENT0];
        let (targets, attachments, draw_buffers): (
            ScratchTargets,
            [Option<Texture>; SCRATCH_TARGET_COUNT],
            &[u32],
        ) = match variant {
            EwaVariant::Color => {
                let [Some(secondary), Some(weights)] =
                    [secondary_slot, weights_slot]
                else {
                    return None;
                };
                (
                    ScratchTargets::Color([primary, secondary, weights]),
                    [Some(primary), Some(secondary), Some(weights)],
                    &all_draw_buffers,
                )
            }
            EwaVariant::Mono => (
                ScratchTargets::Mono(primary),
                [Some(primary), None, None],
                &mono_draw_buffer,
            ),
        };

        // SAFETY: Context is current (module invariant); `self.fbo` and `attachments` are live.
        unsafe {
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
            for (&attachment, &texture) in
                all_draw_buffers.iter().zip(&attachments)
            {
                self.gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    attachment,
                    glow::TEXTURE_2D,
                    texture,
                    0,
                );
            }
            self.gl.draw_buffers(draw_buffers);
            let status =
                self.gl.check_framebuffer_status(glow::FRAMEBUFFER);
            if status != glow::FRAMEBUFFER_COMPLETE {
                tracing::debug!(
                    status,
                    needs_color = matches!(variant, EwaVariant::Color),
                    ?dims,
                    "ewa scratch framebuffer incomplete; falling back"
                );
                self.gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                return None;
            }
        }
        Some(targets)
    }

    fn draw_minified(
        &mut self,
        target: DrawTarget,
        page_texture: &PageTexture,
        seat: SeatGeometry,
        slice: VisibleSlice,
    ) -> bool {
        const ONE_TEXEL_PER_STEP: (f32, f32) = (1.0, 1.0);

        let scratch_dims =
            (slice.visible_size.0, slice.source_row_count());
        let variant =
            if matches!(page_texture.layout, PixelLayout::Mono { .. }) {
                EwaVariant::Mono
            } else {
                EwaVariant::Color
            };
        let Some(scratch) = self.bind_scratch_fbo(scratch_dims, variant)
        else {
            return false;
        };

        // Scratch rows carry signed lobes outside viewport bounds.
        // SAFETY: Context is current (module invariant); `self.fbo` is complete and bound.
        unsafe {
            self.gl.disable(glow::SCISSOR_TEST);
            self.gl.disable(glow::BLEND);
            self.gl.viewport(
                0,
                0,
                gl_int(scratch_dims.0),
                gl_int(scratch_dims.1),
            );
            self.gl.active_texture(EWA_LUT_TEXTURE_UNIT);
            self.gl.bind_texture(glow::TEXTURE_1D, Some(self.lut));

            if let PixelLayout::Mono { luma, gain, offset } =
                page_texture.layout
            {
                self.gl.use_program(Some(
                    self.horizontal_mono.objects.program,
                ));
                bind_horizontal_common(
                    &self.gl,
                    &self.horizontal_mono.common,
                    slice,
                    page_texture.dims,
                    seat.origin.0,
                );
                bind_mono_source(
                    &self.gl,
                    self.horizontal_mono.luma_gain_offset,
                    self.horizontal_mono.transfer_code,
                    page_texture.transfer,
                    luma,
                    (gain, offset),
                );
            } else {
                self.gl.use_program(Some(
                    self.horizontal_color.objects.program,
                ));
                bind_horizontal_common(
                    &self.gl,
                    &self.horizontal_color.common,
                    slice,
                    page_texture.dims,
                    seat.origin.0,
                );
                self.horizontal_color.color.bind(
                    &self.gl,
                    page_texture,
                    ONE_TEXEL_PER_STEP,
                );
            }
            self.gl.draw_arrays(
                glow::TRIANGLE_STRIP,
                0,
                QUAD_VERTEX_COUNT,
            );
        }

        restore_screen_state(&self.gl, target);
        let vertical = match scratch {
            ScratchTargets::Mono(_) => &self.vertical_mono,
            ScratchTargets::Color(_) => &self.vertical_color,
        };
        let window_px = geometry::px_f32(target.window);
        // SAFETY: Context is current (module invariant) and default FBO is restored.
        unsafe {
            self.gl.use_program(Some(vertical.objects.program));
            self.gl.uniform_2_f32(
                Some(&vertical.uniforms.window_size),
                window_px.0,
                window_px.1,
            );
            self.gl.uniform_2_f32(
                Some(&vertical.uniforms.seat_offset),
                seat.origin.0,
                seat.origin.1,
            );
            self.gl.uniform_2_f32(
                Some(&vertical.uniforms.seat_size),
                seat.size.0,
                seat.size.1,
            );
            self.gl.uniform_1_i32(
                Some(&vertical.uniforms.visible_origin_x),
                slice.visible_origin.0,
            );
            self.gl.uniform_1_i32(
                Some(&vertical.uniforms.visible_width),
                gl_int(slice.visible_size.0),
            );
            self.gl.uniform_1_i32(
                Some(&vertical.uniforms.source_y_start),
                gl_int(slice.source_y_range.0),
            );
            self.gl.uniform_1_i32(
                Some(&vertical.uniforms.source_y_end),
                gl_int(slice.source_y_range.1),
            );
            self.gl.uniform_1_f32(
                Some(&vertical.uniforms.source_per_screen_y),
                slice.source_per_screen.1,
            );
            self.gl.uniform_1_i32(
                Some(&vertical.uniforms.dither),
                i32::from(page_texture.needs_dither),
            );
            match scratch {
                ScratchTargets::Mono(primary) => {
                    self.gl.active_texture(glow::TEXTURE0);
                    self.gl.bind_texture(glow::TEXTURE_2D, Some(primary));
                }
                ScratchTargets::Color([primary, secondary, weights]) => {
                    for (unit, texture) in [
                        (glow::TEXTURE0, primary),
                        (glow::TEXTURE1, secondary),
                        (glow::TEXTURE2, weights),
                    ] {
                        self.gl.active_texture(unit);
                        self.gl
                            .bind_texture(glow::TEXTURE_2D, Some(texture));
                    }
                }
            }
            self.gl.active_texture(EWA_LUT_TEXTURE_UNIT);
            self.gl.bind_texture(glow::TEXTURE_1D, Some(self.lut));
            self.gl.draw_arrays(
                glow::TRIANGLE_STRIP,
                0,
                QUAD_VERTEX_COUNT,
            );
        }
        true
    }
}

fn restore_screen_state(gl: &Context, target: DrawTarget) {
    // SAFETY: Context is current (module invariant).
    unsafe {
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.viewport(
            0,
            0,
            gl_int(target.window.0),
            gl_int(target.window.1),
        );
        gl.enable(glow::SCISSOR_TEST);
        gl.scissor(
            0,
            gl_int(target.window.1.saturating_sub(target.clip.1)),
            gl_int(target.clip.0),
            gl_int(target.clip.1),
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

fn begin_frame(gl: &Context, vao: VertexArray, target: DrawTarget) {
    restore_screen_state(gl, target);
    // SAFETY: Context is current (module invariant) and `vao` is owned by `PageTextures`.
    unsafe {
        gl.bind_vertex_array(Some(vao));
    }
}

impl ColorSourceUniforms {
    fn lookup(gl: &Context, program: Program) -> Self {
        let location = |name: &str| uniform_location(gl, program, name);
        Self {
            chroma_uv_per_step: location("u_chroma_uv_per_step"),
            color_matrix: location("u_color_matrix"),
            color_offset: location("u_color_offset"),
            alpha_normalization: location("u_alpha_normalization"),
            pixel_layout: location("u_pixel_layout"),
            transfer_code: location("u_transfer_code"),
        }
    }

    fn bind(
        &self,
        gl: &Context,
        page_texture: &PageTexture,
        luma_texels_per_step: (f32, f32),
    ) {
        // SAFETY: Context is current (module invariant); `self` belongs to the active
        // program and `page_texture` owns live textures.
        unsafe {
            gl.uniform_1_i32(
                Some(&self.transfer_code),
                transfer_code(page_texture.transfer),
            );
            match &page_texture.layout {
                PixelLayout::Rgba { texture } => {
                    gl.uniform_1_i32(
                        Some(&self.pixel_layout),
                        PIXEL_LAYOUT_RGBA,
                    );
                    gl.active_texture(glow::TEXTURE0);
                    gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
                }
                PixelLayout::Yuv {
                    luma,
                    cb,
                    cr,
                    alpha,
                    chroma_uv_per_texel,
                    color_conversion,
                    alpha_normalization,
                } => {
                    gl.uniform_1_i32(
                        Some(&self.pixel_layout),
                        PIXEL_LAYOUT_YUV,
                    );
                    gl.uniform_2_f32(
                        Some(&self.chroma_uv_per_step),
                        chroma_uv_per_texel.0 * luma_texels_per_step.0,
                        chroma_uv_per_texel.1 * luma_texels_per_step.1,
                    );
                    gl.uniform_1_f32(
                        Some(&self.alpha_normalization),
                        *alpha_normalization,
                    );
                    gl.uniform_matrix_3_f32_slice(
                        Some(&self.color_matrix),
                        false,
                        &color_conversion.matrix,
                    );
                    gl.uniform_3_f32(
                        Some(&self.color_offset),
                        color_conversion.offset[0],
                        color_conversion.offset[1],
                        color_conversion.offset[2],
                    );
                    for (unit, texture) in [
                        (glow::TEXTURE0, *luma),
                        (glow::TEXTURE1, *cb),
                        (glow::TEXTURE2, *cr),
                        (glow::TEXTURE3, *alpha),
                    ] {
                        gl.active_texture(unit);
                        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                    }
                }
                PixelLayout::Mono { .. } => {
                    debug_assert!(
                        false,
                        "mono layout routes through the mono path"
                    );
                }
            }
        }
    }
}

fn bind_mono_source(
    gl: &Context,
    luma_gain_offset_location: UniformLocation,
    transfer_location: UniformLocation,
    transfer: Transfer,
    luma: Texture,
    (gain, offset): (f32, f32),
) {
    // SAFETY: Context is current (module invariant); active program owns locations and `luma` is live.
    unsafe {
        gl.uniform_2_f32(Some(&luma_gain_offset_location), gain, offset);
        gl.uniform_1_i32(
            Some(&transfer_location),
            transfer_code(transfer),
        );
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(luma));
    }
}

fn draw_page(
    gl: &Context,
    pipeline: &Pipeline,
    target: DrawTarget,
    page_texture: &PageTexture,
    seat: SeatGeometry,
    source_per_screen: (f32, f32),
) {
    let window_px = geometry::px_f32(target.window);
    // SAFETY: Context is current (module invariant); `pipeline` and `page_texture` own live objects
    // and `begin_frame` bound the empty VAO.
    unsafe {
        gl.use_program(Some(pipeline.objects.program));
        gl.uniform_2_f32(
            Some(&pipeline.window_size),
            window_px.0,
            window_px.1,
        );
        gl.uniform_2_f32(
            Some(&pipeline.seat_offset),
            seat.origin.0,
            seat.origin.1,
        );
        gl.uniform_2_f32(
            Some(&pipeline.seat_size),
            seat.size.0,
            seat.size.1,
        );
        gl.uniform_2_f32(
            Some(&pipeline.page_uv_per_screen),
            1.0 / seat.size.0,
            1.0 / seat.size.1,
        );
        gl.uniform_1_i32(
            Some(&pipeline.dither),
            i32::from(page_texture.needs_dither),
        );
        if let PixelLayout::Mono { luma, gain, offset } =
            page_texture.layout
        {
            gl.uniform_1_i32(
                Some(&pipeline.color.pixel_layout),
                PIXEL_LAYOUT_MONO,
            );
            bind_mono_source(
                gl,
                pipeline.luma_gain_offset,
                pipeline.color.transfer_code,
                page_texture.transfer,
                luma,
                (gain, offset),
            );
        } else {
            pipeline.color.bind(gl, page_texture, source_per_screen);
        }
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, QUAD_VERTEX_COUNT);
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
fn label(gl: &Context, identifier: u32, name: u32, text: &str) {
    if gl.supports_debug() {
        // SAFETY: `gl` is current with `KHR_debug` supported and `name` is live.
        unsafe {
            gl.object_label(identifier, name, Some(text));
        }
    }
}

fn label_entry(
    gl: &Context,
    page: usize,
    role: Role,
    page_texture: &PageTexture,
) {
    if !gl.supports_debug() {
        return;
    }
    match &page_texture.layout {
        PixelLayout::Rgba { texture } => label(
            gl,
            glow::TEXTURE,
            texture.0.get(),
            &format!("page {page} {role} rgba"),
        ),
        PixelLayout::Mono { luma, .. } => label(
            gl,
            glow::TEXTURE,
            luma.0.get(),
            &format!("page {page} {role} luma"),
        ),
        PixelLayout::Yuv {
            luma,
            cb,
            cr,
            alpha,
            ..
        } => {
            for (slot, name) in
                [(luma, "luma"), (cb, "cb"), (cr, "cr"), (alpha, "alpha")]
            {
                label(
                    gl,
                    glow::TEXTURE,
                    slot.0.get(),
                    &format!("page {page} {role} {name}"),
                );
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
    dims: (u32, u32),
    page_dims: (u32, u32),
) -> MagnificationFilter {
    if dims == page_dims {
        MagnificationFilter::Nearest
    } else {
        MagnificationFilter::Linear
    }
}

/// Dither hides precision loss from a PQ or >8-bit source; 8-bit sRGB
/// has none.
fn needs_dither(transfer: Transfer, plane_depth: Option<u8>) -> bool {
    transfer == Transfer::Pq
        || plane_depth.is_some_and(|depth| depth > decode::BYTE_DEPTH)
}

/// Create a single-level texture with linear minification and the
/// requested magnification filter.
fn new_texture(
    gl: &Context,
    magnification_filter: MagnificationFilter,
) -> Texture {
    let mag_filter = magnification_filter.gl_filter();
    // SAFETY: Caller ensures the GL context is current on the render thread.
    unsafe {
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
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            gl_int(glow::LINEAR),
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            gl_int(mag_filter),
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_BASE_LEVEL,
            0,
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAX_LEVEL, 0);
        texture
    }
}

fn upload_rgba(
    gl: &Context,
    data: &[u8],
    dims: (u32, u32),
    magnification_filter: MagnificationFilter,
) -> Texture {
    assert_eq!(
        data.len(),
        dims.0 as usize * dims.1 as usize * 4,
        "rgba byte length"
    );
    let texture = new_texture(gl, magnification_filter);
    // SAFETY: `texture` was just created on the current context and `data`
    // matches `dims.0 * dims.1 * 4` RGBA8 bytes.
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            gl_int(glow::RGBA8),
            gl_int(dims.0),
            gl_int(dims.1),
            0,
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
    dims: (u32, u32),
) {
    assert_eq!(
        data.len(),
        dims.0 as usize * dims.1 as usize * 4,
        "rgba byte length"
    );
    // SAFETY: `texture` is owned live by a `PageTexture` entry on the current
    // context, and `data` matches `dims.0 * dims.1 * 4` RGBA8 bytes.
    unsafe {
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
            PixelUnpackData::Slice(Some(data)),
        );
    }
}

fn upload_plane(
    gl: &Context,
    data: &[u8],
    dims: (u32, u32),
    wide: bool,
    magnification_filter: MagnificationFilter,
) -> Texture {
    let sample_bytes = if wide { size_of::<u16>() } else { 1 };
    assert_eq!(
        data.len(),
        dims.0 as usize * dims.1 as usize * sample_bytes,
        "plane byte length"
    );
    let (internal, ty) = if wide {
        (glow::R16, glow::UNSIGNED_SHORT)
    } else {
        (glow::R8, glow::UNSIGNED_BYTE)
    };
    let texture = new_texture(gl, magnification_filter);
    // SAFETY: `texture` was just created on the current context and `data`
    // matches `dims.0 * dims.1 * sample_bytes` bytes.
    unsafe {
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
            PixelUnpackData::Slice(Some(data)),
        );
    }
    texture
}

#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are the YUV plane names the two textures hold"
)]
fn upload_page_texture(
    gl: Arc<Context>,
    image: &ValidatedImage,
) -> PageTexture {
    let dims = (image.width(), image.height());
    let page_dims = image.page_dims();
    let magnification_filter = magnification_filter(dims, page_dims);
    match image.pixels() {
        WirePixels::Rgba8(rgba_bytes) => {
            let texture =
                upload_rgba(&gl, rgba_bytes, dims, magnification_filter);
            PageTexture {
                gl,
                dims,
                page_dims,
                transfer: image.transfer(),
                needs_dither: needs_dither(image.transfer(), None),
                layout: PixelLayout::Rgba { texture },
            }
        }
        WirePixels::Yuv {
            depth,
            subsampling,
            full_range,
            matrix,
            luma,
            cb,
            cr,
            alpha,
        } => {
            let depth = *depth;
            let ((luma_w, luma_h), (chroma_w, chroma_h)) =
                decode::plane_dims(dims, *subsampling);
            let wide = depth > decode::BYTE_DEPTH;
            let luma_texture = upload_plane(
                &gl,
                luma,
                (luma_w, luma_h),
                wide,
                magnification_filter,
            );
            let color_conversion = page_color_conversion(
                depth,
                *subsampling,
                *full_range,
                *matrix,
                wide,
            );
            let layout = if *subsampling == Subsampling::Mono
                && alpha.is_none()
            {
                PixelLayout::Mono {
                    luma: luma_texture,
                    gain: color_conversion.matrix[0],
                    offset: color_conversion.offset[0],
                }
            } else {
                let (cb_texture, cr_texture) = if *subsampling
                    == Subsampling::Mono
                {
                    // `color_conversion` zeroes columns 1 and 2.
                    let neutral_code = 1_u32 << (depth - 1);
                    (
                        upload_constant_plane(&gl, depth, neutral_code),
                        upload_constant_plane(&gl, depth, neutral_code),
                    )
                } else {
                    (
                        upload_plane(
                            &gl,
                            cb,
                            (chroma_w, chroma_h),
                            wide,
                            MagnificationFilter::Linear,
                        ),
                        upload_plane(
                            &gl,
                            cr,
                            (chroma_w, chroma_h),
                            wide,
                            MagnificationFilter::Linear,
                        ),
                    )
                };
                let alpha_texture = alpha.as_ref().map_or_else(
                    // 1x1 opaque: the max code, which the alpha normalization
                    // maps to 1.0.
                    || {
                        upload_constant_plane(
                            &gl,
                            depth,
                            (1_u32 << depth) - 1,
                        )
                    },
                    |alpha| {
                        upload_plane(
                            &gl,
                            alpha,
                            dims,
                            wide,
                            MagnificationFilter::Linear,
                        )
                    },
                );
                let chroma_px = geometry::px_f32((chroma_w, chroma_h));
                let ratio = subsampling.chroma_ratio();
                PixelLayout::Yuv {
                    luma: luma_texture,
                    cb: cb_texture,
                    cr: cr_texture,
                    alpha: alpha_texture,
                    chroma_uv_per_texel: (
                        ratio.0 / chroma_px.0,
                        ratio.1 / chroma_px.1,
                    ),
                    color_conversion,
                    alpha_normalization: depth_normalization(depth, wide),
                }
            };
            PageTexture {
                gl,
                dims,
                page_dims,
                transfer: image.transfer(),
                needs_dither: needs_dither(image.transfer(), Some(depth)),
                layout,
            }
        }
    }
}

fn upload_constant_plane(
    gl: &Context,
    depth: u8,
    sample_code: u32,
) -> Texture {
    if depth > decode::BYTE_DEPTH {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "depth stays within decode::MAX_SAMPLE_DEPTH, so sample_code fits u16"
        )]
        let wide = sample_code as u16;
        #[expect(
            clippy::host_endian_bytes,
            reason = "GL UNSIGNED_SHORT uploads expect native-endian bytes"
        )]
        upload_plane(
            gl,
            &wide.to_ne_bytes(),
            (1, 1),
            true,
            MagnificationFilter::Linear,
        )
    } else {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "depths up to decode::BYTE_DEPTH keep codes within u8"
        )]
        upload_plane(
            gl,
            &[sample_code as u8],
            (1, 1),
            false,
            MagnificationFilter::Linear,
        )
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
    debug_assert!(
        (1..=decode::MAX_SAMPLE_DEPTH).contains(&depth),
        "YUV bit depth out of range"
    );
    if !wide {
        return 1.0;
    }
    R16_STORAGE_MAX / ((1_u32 << depth) - 1) as f32
}

const R16_STORAGE_MAX: f32 = 65535.0;
const R8_STORAGE_MAX: f32 = 255.0;

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
fn page_color_conversion(
    depth: u8,
    subsampling: Subsampling,
    full_range: bool,
    color_matrix: Matrix,
    wide: bool,
) -> ColorConversion {
    debug_assert!(
        (1..=decode::MAX_SAMPLE_DEPTH).contains(&depth),
        "YUV bit depth out of range"
    );
    let quantization = code_quantization(depth, full_range, wide);
    let normalization = depth_normalization(depth, wide);
    let (
        base_matrix,
        luma_scale,
        chroma_scale,
        luma_black,
        chroma_neutral,
    ) = if subsampling == Subsampling::Mono {
        (
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            quantization.luma_scale,
            0.0,
            quantization.luma_black,
            0.0,
        )
    } else if color_matrix == Matrix::Identity {
        // H.273 identity planes are GBR in the (Y, Cb, Cr) slots,
        // quantized as luma on all three axes: permute into RGB and
        // anchor every axis at luma black.
        (
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            quantization.luma_scale,
            quantization.luma_scale,
            quantization.luma_black,
            quantization.luma_black,
        )
    } else {
        (
            yuv_to_rgb_matrix(color_matrix),
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

/// Window, EGL context, and page texture store. vsync is `DontWait`, so
/// the app must call `pre_present_and_swap()` to present.
pub struct Window {
    // Drop order: `pages` while `context` is current, `context` before
    // `surface`, `surface` before `winit`, and `_display` last.
    pages: PageTextures,
    gl: Arc<Context>,
    context: PossiblyCurrentContext,
    surface: Surface<WindowSurface>,
    winit: WinitWindow,
    _display: OwnedDisplayHandle,
    /// Alpha config matched. The fallback may still carry alpha bits; the
    /// opaque hint and clear hide them.
    translucent: bool,
    /// Size the surface was last resized to. Creation counts as the
    /// first resize.
    resized_to: (u32, u32),
}

impl Window {
    /// Create the window with a desktop OpenGL 3.3 core context.
    pub fn new(event_loop: &ActiveEventLoop) -> eros::Result<Self> {
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
        let display = event_loop.owned_display_handle();
        let raw_display = display.display_handle()?.as_raw();
        // SAFETY: `display` outlives `gl_display` locally and `_display` drops
        // last in `Window`.
        let gl_display = unsafe {
            Display::new(raw_display, DisplayApiPreference::Egl)
        }?;
        // Prefer a config with alpha so the desktop shows through; a machine
        // without one runs opaque.
        let translucent_template = ConfigTemplateBuilder::new()
            .with_api(ConfigApi::OPENGL)
            .with_alpha_size(8)
            .build();
        // SAFETY: `translucent_template` sets no raw window handle.
        let mut configs =
            unsafe { gl_display.find_configs(translucent_template) }?;
        let (gl_config, translucent) = if let Some(config) = configs.next()
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
            // SAFETY: `plain` sets no raw window handle.
            let config = unsafe { gl_display.find_configs(plain) }?
                .next()
                .ok_or_else(|| {
                    eros::error!(
                        "no desktop OpenGL 3.3 core config on this EGL display"
                    )
                })?;
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
        // SAFETY: `window` owns `raw_window` and outlives the context.
        let not_current =
            unsafe { gl_display.create_context(&gl_config, &ctx_attrs) }?;

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
        // SAFETY: `window` owns `raw_window` and outlives `surface`.
        let surface = unsafe {
            gl_display.create_window_surface(&gl_config, &surface_attrs)
        }?;
        let context = not_current.make_current(&surface)?;
        // Non-blocking present; `pre_present_and_swap` requests a Wayland
        // frame callback.
        surface.set_swap_interval(&context, SwapInterval::DontWait)?;

        // SAFETY: `context` is current and `gl_display` is alive for lookups.
        let gl = unsafe {
            Context::from_loader_function(|symbol| {
                #[expect(
                    clippy::expect_used,
                    reason = "symbol names are static ASCII GL entry points, NUL-free by construction"
                )]
                let symbol = CString::new(symbol)
                    .expect("GL proc name without interior NUL");
                gl_display.get_proc_address(&symbol)
            })
        };
        #[cfg(debug_assertions)]
        let mut gl = gl;
        #[cfg(debug_assertions)]
        if gl.supports_debug() {
            // SAFETY: the context is current, supports debug output, and the
            // callback only logs.
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
                gl.enable(glow::DEBUG_OUTPUT);
            }
        }
        let gl = Arc::new(gl);
        let pages = PageTextures::new(Arc::clone(&gl))?;
        Ok(Self {
            pages,
            gl,
            context,
            surface,
            winit: window,
            _display: display,
            translucent,
            resized_to: (width, height),
        })
    }

    pub fn pages(&self) -> &PageTextures {
        &self.pages
    }

    pub fn pages_mut(&mut self) -> &mut PageTextures {
        &mut self.pages
    }

    pub fn glow(&self) -> Arc<Context> {
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
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;

    // GL needs a context; these tests pin the CPU-side values it consumes.

    /// Matrix element at (`row`, `col`); storage is column-major.
    #[expect(
        clippy::indexing_slicing,
        reason = "row and col are literal 0..3 coordinates into the fixed 3x3 matrix under test"
    )]
    fn element(matrix: &[f32; 9], row: usize, col: usize) -> f32 {
        matrix[col * 3 + row]
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
        for subsampling in [Subsampling::Yuv444, Subsampling::Mono] {
            for depth in [8_u8, 10, 12, 16] {
                for full_range in [true, false] {
                    let conversion = page_color_conversion(
                        depth,
                        subsampling,
                        full_range,
                        Matrix::Bt601,
                        depth > decode::BYTE_DEPTH,
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
                                "{subsampling:?} depth {depth} full {full_range} row {row}: {channel} vs {expected}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn identity_pages_permute_gbr() {
        let color = page_color_conversion(
            8,
            Subsampling::Yuv444,
            true,
            Matrix::Identity,
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
            10,
            Subsampling::Mono,
            true,
            Matrix::Identity,
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
        assert!(matches!(
            magnification_filter((100, 100), (100, 100)),
            MagnificationFilter::Nearest
        ));
        assert!(matches!(
            magnification_filter((100, 100), (1000, 2000)),
            MagnificationFilter::Linear
        ));
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        clippy::indexing_slicing,
        reason = "indices stay within 0..EWA_GRID_LEN, which is exact in f64"
    )]
    fn ewa_rank2_lut_approximates_circular_lanczos3() {
        assert!(jinc(EWA_JINC_FIRST_ZERO).abs() < 1e-12);
        assert!(jinc(EWA_SUPPORT).abs() < 1e-12);

        let lut = ewa_rank2_lut();
        assert_eq!(lut[EWA_LUT_LEN - 1], [0.0, 0.0]);

        let center_index = EWA_LUT_LEN - 1;
        let grid_step = 2.0 * EWA_SUPPORT / (EWA_GRID_LEN - 1) as f64;
        let lut_weights = |index_from_center: usize| -> (f64, f64) {
            let entry = lut[index_from_center];
            (f64::from(entry[0]), f64::from(entry[1]))
        };
        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        for row in 0..EWA_GRID_LEN {
            let coord_y = (row as f64).mul_add(grid_step, -EWA_SUPPORT);
            let (row_primary, row_secondary) =
                lut_weights(row.abs_diff(center_index));
            for col in 0..EWA_GRID_LEN {
                let coord_x =
                    (col as f64).mul_add(grid_step, -EWA_SUPPORT);
                let (col_primary, col_secondary) =
                    lut_weights(col.abs_diff(center_index));
                let approx = row_secondary
                    .mul_add(-col_secondary, row_primary * col_primary);
                let exact = ewa_lanczos3(coord_x.hypot(coord_y));
                let difference = exact - approx;
                squared_error =
                    difference.mul_add(difference, squared_error);
                squared_reference =
                    exact.mul_add(exact, squared_reference);
            }
        }
        let relative_l2_error = (squared_error / squared_reference).sqrt();
        assert!(
            relative_l2_error < 0.02,
            "rank-2 rel L2 error {relative_l2_error}"
        );
    }

    #[test]
    fn visible_slice_clips_viewport_and_pads_source_rows() {
        // 2x minification panned off top edge: center rows cover [1001..2999].
        let panned = compute_visible_slice(
            SeatGeometry {
                origin: (-200.0, -500.0),
                size: (2031.0, 2880.0),
            },
            (1280, 1000),
            (4062, 5760),
        )
        .expect("panned slice");
        assert_eq!(panned.visible_origin, (0, 0));
        assert_eq!(panned.visible_size, (1280, 1000));
        assert!((900..=993).contains(&panned.source_y_range.0));
        assert!((3007..3100).contains(&panned.source_y_range.1));
        assert!(panned.is_minified());

        let exact = compute_visible_slice(
            SeatGeometry {
                origin: (0.0, 0.0),
                size: (1805.0, 2560.0),
            },
            (2560, 1440),
            (1805, 2560),
        )
        .expect("1:1 slice");
        assert!(!exact.is_minified());
        assert!(
            compute_visible_slice(
                SeatGeometry {
                    origin: (2600.0, 0.0),
                    size: (500.0, 500.0),
                },
                (2560, 1440),
                (1805, 2560),
            )
            .is_none()
        );
    }
}
