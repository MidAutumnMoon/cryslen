//! Two-pass circular rank-2 EWA-Lanczos3 GL minifier (Mono + Color).

#![expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "scoped to GL helpers sharing one current-context invariant"
)]

use core::ops::Range;
use std::sync::Arc;

use egui_glow::glow;
use egui_glow::glow::Context;
use egui_glow::glow::Framebuffer;
use egui_glow::glow::HasContext as _;
use egui_glow::glow::PixelUnpackData;
use egui_glow::glow::Texture;
use egui_glow::glow::UniformLocation;
use indoc::indoc;

use crate::image::pixels::Transfer;
use crate::render::program;
use crate::render::program::ALPHA_SAMPLER_UNIT;
use crate::render::program::BASE_MIP_LEVEL;
use crate::render::program::CB_SAMPLER_UNIT;
use crate::render::program::CR_SAMPLER_UNIT;
use crate::render::program::DrawSource;
use crate::render::program::DrawTarget;
use crate::render::program::FIRST_VERTEX_INDEX;
use crate::render::program::FULLSCREEN_VERTEX_SHADER_SOURCE;
use crate::render::program::GLSL_BAYER4;
use crate::render::program::GLSL_TRANSFER_HEADER;
use crate::render::program::GLSL_VERSION;
use crate::render::program::NO_BORDER;
use crate::render::program::PixelLayout;
use crate::render::program::QUAD_VERTEX_COUNT;
use crate::render::program::QuadUniforms;
use crate::render::program::RGBA_OR_LUMA_SAMPLER_UNIT;
use crate::render::program::SeatGeometry;
use crate::render::program::ShaderPipeline;
use crate::render::program::StubTextures;
use crate::render::program::VERTEX_SHADER_SOURCE;
use crate::render::program::YuvUniforms;

const MINIFY_SCALE_TOLERANCE: f64 = 1e-3;
const SLICE_PAD_TEXELS: f64 = 1.0;
const FLOAT_UNPACK_ALIGNMENT: i32 = 4;

const SCRATCH_PRIMARY_SAMPLER_UNIT: i32 = 0;
const SCRATCH_SECONDARY_SAMPLER_UNIT: i32 = 1;
const SCRATCH_WEIGHTS_SAMPLER_UNIT: i32 = 2;
const EWA_LUT_SAMPLER_UNIT: i32 = 4;
const EWA_LUT_TEXTURE_UNIT: u32 = glow::TEXTURE4;

const EWA_SUPPORT_TEXELS: f64 = 3.238_315_484_166_236;
const EWA_JINC_FIRST_ZERO: f64 = 1.219_669_891_266_504_5;
const EWA_RANK: usize = 2;
const EWA_LUT_INTERVALS: i32 = 128;
const EWA_LUT_TEXEL_COUNT_GL: i32 = EWA_LUT_INTERVALS + 1;
const EWA_LUT_TEXEL_COUNT: usize =
    EWA_LUT_TEXEL_COUNT_GL.cast_unsigned() as usize;
const EWA_GRID_LEN: usize = EWA_LUT_TEXEL_COUNT * 2 - 1;

const COLOR_SCRATCH_TARGET_COUNT: usize = 3;
const MONO_SCRATCH_TARGET_COUNT: usize = 1;

struct ScratchTargetSpec {
    internal_format: u32,
    format: u32,
    debug_name: &'static str,
}

const SCRATCH_TARGET_SPECS: [ScratchTargetSpec;
    COLOR_SCRATCH_TARGET_COUNT] = [
    ScratchTargetSpec {
        internal_format: glow::RGBA16F,
        format: glow::RGBA,
        debug_name: "ewa scratch primary",
    },
    ScratchTargetSpec {
        internal_format: glow::RGBA16F,
        format: glow::RGBA,
        debug_name: "ewa scratch secondary",
    },
    ScratchTargetSpec {
        internal_format: glow::RG16F,
        format: glow::RG,
        debug_name: "ewa scratch weights",
    },
];

#[derive(Clone, Copy)]
enum EwaVariant {
    Mono,
    Color,
}

impl EwaVariant {
    const fn glsl_define(self) -> &'static str {
        match self {
            Self::Mono => "#define EWA_MONO 1\n",
            Self::Color => "#define EWA_MONO 0\n",
        }
    }
}

const GLSL_EWA_LUT: &str = indoc! {"
    uniform float u_ewa_support;
    uniform vec2 u_ewa_lut_scale_bias;
    vec2 ewa_weights(sampler1D lut, float distance) {
        float lut_coord = clamp(distance / u_ewa_support, 0.0, 1.0);
        return texture(lut, lut_coord * u_ewa_lut_scale_bias.x + u_ewa_lut_scale_bias.y).rg;
    }
"};

const EWA_HORIZONTAL_BODY: &str = indoc! {"
    #if EWA_MONO
    layout(location = 0) out vec4 out_accumulated;
    uniform vec2 u_luma_gain_offset;
    #else
    layout(location = 0) out vec4 out_primary;
    layout(location = 1) out vec4 out_secondary;
    layout(location = 2) out vec4 out_weights;
    uniform vec2 u_chroma_uv_per_luma_texel;
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
    uniform ivec2 u_texture_dims_texels;
    uniform float u_seat_origin_x_px;
    uniform float u_source_per_screen_x;
    uniform sampler2D u_rgba_or_luma;
    uniform sampler1D u_ewa_lut;
    void main() {
        float filter_source_per_screen = max(1.0, u_source_per_screen_x);
        float screen_per_source = 1.0 / filter_source_per_screen;
        float support_radius = u_ewa_support * filter_source_per_screen;
        float screen_x = float(u_visible_origin_x) + gl_FragCoord.x;
        float center_x = (screen_x - u_seat_origin_x_px) * u_source_per_screen_x;
        int source_x_start = max(0, int(floor(center_x - support_radius)));
        int source_x_end = min(u_texture_dims_texels.x - 1, int(ceil(center_x + support_radius)));
        int source_y = clamp(u_source_y_start + int(floor(gl_FragCoord.y)), 0, u_texture_dims_texels.y - 1);
        #if EWA_MONO
        vec2 accumulated = vec2(0.0);
        #else
        vec2 texture_dims_texels = vec2(u_texture_dims_texels);
        vec4 accumulated_primary = vec4(0.0);
        vec4 accumulated_secondary = vec4(0.0);
        #endif
        vec2 weight_sum = vec2(0.0);
        for (int source_x = source_x_start; source_x <= source_x_end; ++source_x) {
            float distance = abs(((float(source_x) + 0.5) - center_x) * screen_per_source);
            vec2 weights = ewa_weights(u_ewa_lut, distance);
            #if EWA_MONO
            float luma_sample = texelFetch(u_rgba_or_luma, ivec2(source_x, source_y), 0).r;
            float gray = to_srgb(vec3(luma_sample * u_luma_gain_offset.x - u_luma_gain_offset.y)).r;
            accumulated += gray * weights;
            #else
            vec4 premultiplied_texel;
            if (u_pixel_layout == PIXEL_LAYOUT_RGBA) {
                vec4 rgba_texel = texelFetch(u_rgba_or_luma, ivec2(source_x, source_y), 0);
                premultiplied_texel = vec4(to_srgb(rgba_texel.rgb) * rgba_texel.a, rgba_texel.a);
            } else {
                vec2 luma_texel = vec2(float(source_x) + 0.5, float(source_y) + 0.5);
                vec2 chroma_uv = luma_texel * u_chroma_uv_per_luma_texel;
                vec3 yuv_samples = vec3(
                    texelFetch(u_rgba_or_luma, ivec2(source_x, source_y), 0).r,
                    texture(u_cb, chroma_uv).r,
                    texture(u_cr, chroma_uv).r);
                float alpha = texture(u_alpha, luma_texel / texture_dims_texels).r * u_alpha_normalization;
                vec3 rgb = to_srgb(u_color_matrix * yuv_samples - u_color_offset);
                premultiplied_texel = vec4(rgb * alpha, alpha);
            }
            accumulated_primary += premultiplied_texel * weights.x;
            accumulated_secondary += premultiplied_texel * weights.y;
            #endif
            weight_sum += weights;
        }
        #if EWA_MONO
        out_accumulated = vec4(accumulated, weight_sum);
        #else
        out_primary = accumulated_primary;
        out_secondary = accumulated_secondary;
        out_weights = vec4(weight_sum, 0.0, 0.0);
        #endif
    }
"};

const EWA_VERTICAL_BODY: &str = indoc! {"
    const float MIN_WEIGHT_SUM = 1e-6;
    in vec2 v_seat_px;
    layout(location = 0) out vec4 out_color;
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
        float support_radius = u_ewa_support * filter_source_per_screen;
        float center_y = v_seat_px.y * u_source_per_screen_y;
        int source_y_start = max(u_source_y_start, int(floor(center_y - support_radius)));
        int source_y_end = min(u_source_y_end - 1, int(ceil(center_y + support_radius)));
        int scratch_x = clamp(int(floor(gl_FragCoord.x)) - u_visible_origin_x, 0, u_visible_width - 1);
        #if EWA_MONO
        float accumulated = 0.0;
        #else
        vec4 accumulated = vec4(0.0);
        #endif
        float weight_sum = 0.0;
        for (int source_y = source_y_start; source_y <= source_y_end; ++source_y) {
            float distance = abs(((float(source_y) + 0.5) - center_y) * screen_per_source);
            vec2 weights = ewa_weights(u_ewa_lut, distance);
            ivec2 scratch_texel = ivec2(scratch_x, source_y - u_source_y_start);
            // Subtract the second singular term: its eigenvalue is negative.
            #if EWA_MONO
            vec4 horiz_accum = texelFetch(u_scratch_primary, scratch_texel, 0);
            accumulated += horiz_accum.x * weights.x - horiz_accum.y * weights.y;
            weight_sum += horiz_accum.z * weights.x - horiz_accum.w * weights.y;
            #else
            vec4 primary = texelFetch(u_scratch_primary, scratch_texel, 0);
            vec4 secondary = texelFetch(u_scratch_secondary, scratch_texel, 0);
            vec2 horiz_weights = texelFetch(u_scratch_weights, scratch_texel, 0).rg;
            accumulated += primary * weights.x - secondary * weights.y;
            weight_sum += horiz_weights.x * weights.x - horiz_weights.y * weights.y;
            #endif
        }
        #if EWA_MONO
        float gray = clamp(accumulated / max(weight_sum, MIN_WEIGHT_SUM), 0.0, 1.0);
        if (u_dither != 0) {
            gray = clamp(gray + dither_offset(gl_FragCoord.xy), 0.0, 1.0);
        }
        out_color = vec4(gray, gray, gray, 1.0);
        #else
        vec4 premultiplied = accumulated / max(weight_sum, MIN_WEIGHT_SUM);
        float alpha = clamp(premultiplied.a, 0.0, 1.0);
        vec3 rgb = clamp(premultiplied.rgb, vec3(0.0), vec3(alpha));
        if (u_dither != 0) {
            rgb = clamp(rgb + dither_offset(gl_FragCoord.xy) * alpha, vec3(0.0), vec3(alpha));
        }
        out_color = vec4(rgb, alpha);
        #endif
    }
"};

#[derive(Debug, Clone, PartialEq)]
pub struct VisibleSlice {
    pub origin_x_px: i32,
    pub width_px: i32,
    pub source_y_range: Range<i32>,
    pub source_per_screen: (f64, f64),
}

impl VisibleSlice {
    fn source_row_count(&self) -> i32 {
        self.source_y_range.end - self.source_y_range.start
    }

    pub fn is_minified(&self) -> bool {
        self.source_per_screen.0 > 1.0 + MINIFY_SCALE_TOLERANCE
            || self.source_per_screen.1 > 1.0 + MINIFY_SCALE_TOLERANCE
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "both endpoints are whole numbers clamped to 0..=limit (i32)"
)]
fn clip_span(start: f64, end: f64, limit: i32) -> Option<Range<i32>> {
    if limit <= 0 || start.is_nan() || end.is_nan() {
        return None;
    }
    let max_bound = f64::from(limit);
    let clamped_start = start.floor().clamp(0.0, max_bound);
    let clamped_end = end.ceil().clamp(0.0, max_bound);
    (clamped_end > clamped_start)
        .then_some(clamped_start as i32..clamped_end as i32)
}

pub fn compute_visible_slice(
    seat: SeatGeometry,
    clip_size: (i32, i32),
    texture_dims: (u32, u32),
) -> Option<VisibleSlice> {
    if seat.size.0 <= 0.0
        || seat.size.1 <= 0.0
        || !seat.origin.0.is_finite()
        || !seat.origin.1.is_finite()
        || !seat.size.0.is_finite()
        || !seat.size.1.is_finite()
        || texture_dims.0 == 0
        || texture_dims.1 == 0
    {
        return None;
    }
    let horizontal_span = clip_span(
        seat.origin.0,
        seat.origin.0 + seat.size.0,
        clip_size.0,
    )?;
    let vertical_span = clip_span(
        seat.origin.1,
        seat.origin.1 + seat.size.1,
        clip_size.1,
    )?;
    let source_per_screen_x = f64::from(texture_dims.0) / seat.size.0;
    let source_per_screen_y = f64::from(texture_dims.1) / seat.size.1;
    let support_radius_y = EWA_SUPPORT_TEXELS
        .mul_add(source_per_screen_y.max(1.0), SLICE_PAD_TEXELS);
    let center_y_start = (f64::from(vertical_span.start) + 0.5
        - seat.origin.1)
        * source_per_screen_y;
    let center_y_end =
        (f64::from(vertical_span.end) - 0.5 - seat.origin.1)
            * source_per_screen_y;
    let texture_height = program::gl_size(texture_dims.1);
    let source_y_range = clip_span(
        (center_y_start - support_radius_y)
            .min(f64::from(texture_height - 1)),
        center_y_end + support_radius_y + 1.0,
        texture_height,
    )?;
    Some(VisibleSlice {
        origin_x_px: horizontal_span.start,
        width_px: horizontal_span.end - horizontal_span.start,
        source_y_range,
        source_per_screen: (source_per_screen_x, source_per_screen_y),
    })
}

struct HorizontalCommon {
    program: ShaderPipeline,
    visible_origin_x: UniformLocation,
    source_y_start: UniformLocation,
    texture_dims_texels: UniformLocation,
    seat_origin_x: UniformLocation,
    source_per_screen_x: UniformLocation,
    transfer_code: UniformLocation,
}

impl HorizontalCommon {
    fn lookup(program: ShaderPipeline) -> Self {
        let location = |name: &str| {
            program::uniform_location(&program.gl, program.program, name)
        };
        Self {
            visible_origin_x: location("u_visible_origin_x"),
            source_y_start: location("u_source_y_start"),
            texture_dims_texels: location("u_texture_dims_texels"),
            seat_origin_x: location("u_seat_origin_x_px"),
            source_per_screen_x: location("u_source_per_screen_x"),
            transfer_code: location("u_transfer_code"),
            program,
        }
    }

    fn bind(
        &self,
        slice: &VisibleSlice,
        texture_dims: (u32, u32),
        seat_origin_x: f64,
        transfer: Transfer,
    ) {
        let gl = &self.program.gl;
        // SAFETY: Context is current and `self` owns the bound program.
        unsafe {
            gl.use_program(Some(self.program.program));
            gl.uniform_1_i32(
                Some(&self.visible_origin_x),
                slice.origin_x_px,
            );
            gl.uniform_1_i32(
                Some(&self.source_y_start),
                slice.source_y_range.start,
            );
            gl.uniform_2_i32(
                Some(&self.texture_dims_texels),
                program::gl_size(texture_dims.0),
                program::gl_size(texture_dims.1),
            );
            gl.uniform_1_f32(
                Some(&self.seat_origin_x),
                program::gl_f32(seat_origin_x),
            );
            gl.uniform_1_f32(
                Some(&self.source_per_screen_x),
                program::gl_f32(slice.source_per_screen.0),
            );
            gl.uniform_1_i32(
                Some(&self.transfer_code),
                program::transfer_code(transfer),
            );
        }
    }
}

struct HorizontalMonoProgram {
    common: HorizontalCommon,
    luma_gain_offset: UniformLocation,
}

struct HorizontalColorProgram {
    common: HorizontalCommon,
    pixel_layout: UniformLocation,
    yuv: YuvUniforms,
}

struct VerticalProgram {
    program: ShaderPipeline,
    quad: QuadUniforms,
    visible_origin_x: UniformLocation,
    visible_width: UniformLocation,
    source_y_start: UniformLocation,
    source_y_end: UniformLocation,
    source_per_screen_y: UniformLocation,
    dither: UniformLocation,
}

enum ScratchTargets {
    Mono(Texture),
    Color([Texture; COLOR_SCRATCH_TARGET_COUNT]),
}

pub struct Resampler {
    gl: Arc<Context>,
    max_texture_edge: i32,
    weight_lut: Texture,
    scratch_framebuffer: Framebuffer,
    scratch_textures: [Option<Texture>; COLOR_SCRATCH_TARGET_COUNT],
    scratch_capacity: (i32, i32),
    horizontal_mono: HorizontalMonoProgram,
    vertical_mono: VerticalProgram,
    horizontal_color: HorizontalColorProgram,
    vertical_color: VerticalProgram,
}

impl Drop for Resampler {
    fn drop(&mut self) {
        // SAFETY: Context is current (`Window` drop order); `self` owns `weight_lut`, `scratch_textures`, and `scratch_framebuffer`.
        unsafe {
            self.gl.delete_texture(self.weight_lut);
            for texture in self.scratch_textures.into_iter().flatten() {
                self.gl.delete_texture(texture);
            }
            self.gl.delete_framebuffer(self.scratch_framebuffer);
        }
    }
}

impl Resampler {
    pub fn new(
        gl: Arc<Context>,
        max_texture_edge: i32,
    ) -> eros::Result<Self> {
        let horizontal_mono_program = link_ewa_program(
            Arc::clone(&gl),
            FULLSCREEN_VERTEX_SHADER_SOURCE,
            &[
                GLSL_VERSION,
                EwaVariant::Mono.glsl_define(),
                GLSL_TRANSFER_HEADER,
                GLSL_EWA_LUT,
                EWA_HORIZONTAL_BODY,
            ],
            &[
                (RGBA_OR_LUMA_SAMPLER_UNIT, "u_rgba_or_luma"),
                (EWA_LUT_SAMPLER_UNIT, "u_ewa_lut"),
            ],
            "ewa horiz mono",
        )?;
        let luma_gain_offset = program::uniform_location(
            &horizontal_mono_program.gl,
            horizontal_mono_program.program,
            "u_luma_gain_offset",
        );
        let horizontal_mono = HorizontalMonoProgram {
            common: HorizontalCommon::lookup(horizontal_mono_program),
            luma_gain_offset,
        };

        let horizontal_color_program = link_ewa_program(
            Arc::clone(&gl),
            FULLSCREEN_VERTEX_SHADER_SOURCE,
            &[
                GLSL_VERSION,
                EwaVariant::Color.glsl_define(),
                GLSL_TRANSFER_HEADER,
                GLSL_EWA_LUT,
                EWA_HORIZONTAL_BODY,
            ],
            &[
                (RGBA_OR_LUMA_SAMPLER_UNIT, "u_rgba_or_luma"),
                (CB_SAMPLER_UNIT, "u_cb"),
                (CR_SAMPLER_UNIT, "u_cr"),
                (ALPHA_SAMPLER_UNIT, "u_alpha"),
                (EWA_LUT_SAMPLER_UNIT, "u_ewa_lut"),
            ],
            "ewa horiz color",
        )?;
        let pixel_layout = program::uniform_location(
            &horizontal_color_program.gl,
            horizontal_color_program.program,
            "u_pixel_layout",
        );
        let yuv = YuvUniforms::lookup(
            &horizontal_color_program.gl,
            horizontal_color_program.program,
            "u_chroma_uv_per_luma_texel",
        );
        let horizontal_color = HorizontalColorProgram {
            common: HorizontalCommon::lookup(horizontal_color_program),
            pixel_layout,
            yuv,
        };

        let vertical_mono = create_vertical_program(
            Arc::clone(&gl),
            EwaVariant::Mono,
            &[
                (SCRATCH_PRIMARY_SAMPLER_UNIT, "u_scratch_primary"),
                (EWA_LUT_SAMPLER_UNIT, "u_ewa_lut"),
            ],
            "ewa vert mono",
        )?;
        let vertical_color = create_vertical_program(
            Arc::clone(&gl),
            EwaVariant::Color,
            &[
                (SCRATCH_PRIMARY_SAMPLER_UNIT, "u_scratch_primary"),
                (SCRATCH_SECONDARY_SAMPLER_UNIT, "u_scratch_secondary"),
                (SCRATCH_WEIGHTS_SAMPLER_UNIT, "u_scratch_weights"),
                (EWA_LUT_SAMPLER_UNIT, "u_ewa_lut"),
            ],
            "ewa vert color",
        )?;

        // SAFETY: Context is current.
        let scratch_framebuffer = unsafe { gl.create_framebuffer() }
            .map_err(|error| {
                eros::error!("GL framebuffer creation failed: {error}")
            })?;
        program::label(
            &gl,
            glow::FRAMEBUFFER,
            scratch_framebuffer.0.get(),
            "ewa scratch fbo",
        );
        let weight_lut = upload_ewa_lut(&gl);

        Ok(Self {
            gl,
            max_texture_edge,
            weight_lut,
            scratch_framebuffer,
            scratch_textures: [None; COLOR_SCRATCH_TARGET_COUNT],
            scratch_capacity: (0, 0),
            horizontal_mono,
            vertical_mono,
            horizontal_color,
            vertical_color,
        })
    }

    /// Leaves `self.scratch_framebuffer` bound on success; restores `FRAMEBUFFER = 0` on failure.
    fn bind_scratch_framebuffer(
        &mut self,
        dims: (i32, i32),
        variant: EwaVariant,
    ) -> Option<ScratchTargets> {
        if dims.0 <= 0
            || dims.1 <= 0
            || dims.0 > self.max_texture_edge
            || dims.1 > self.max_texture_edge
        {
            return None;
        }
        let needs_grow = self.scratch_textures[0].is_none()
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
            EwaVariant::Color => COLOR_SCRATCH_TARGET_COUNT,
        };
        for (index, (slot, spec)) in self
            .scratch_textures
            .iter_mut()
            .zip(&SCRATCH_TARGET_SPECS)
            .enumerate()
        {
            if (needs_grow && slot.is_some())
                || (index < target_count && slot.is_none())
            {
                *slot = Some(allocate_scratch_texture(
                    &self.gl,
                    *slot,
                    self.scratch_capacity,
                    spec,
                ));
            }
        }
        let [Some(primary), secondary_slot, weights_slot] =
            self.scratch_textures
        else {
            return None;
        };
        let color_draw_buffers = [
            glow::COLOR_ATTACHMENT0,
            glow::COLOR_ATTACHMENT1,
            glow::COLOR_ATTACHMENT2,
        ];
        let mono_draw_buffer = [glow::COLOR_ATTACHMENT0];
        let (targets, attachments, draw_buffers): (
            ScratchTargets,
            [Option<Texture>; COLOR_SCRATCH_TARGET_COUNT],
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
                    &color_draw_buffers,
                )
            }
            EwaVariant::Mono => (
                ScratchTargets::Mono(primary),
                [Some(primary), None, None],
                &mono_draw_buffer,
            ),
        };

        // SAFETY: Context is current; `self.scratch_framebuffer` and `attachments` are live.
        unsafe {
            self.gl.bind_framebuffer(
                glow::FRAMEBUFFER,
                Some(self.scratch_framebuffer),
            );
            for (&attachment, &texture) in
                color_draw_buffers.iter().zip(&attachments)
            {
                self.gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    attachment,
                    glow::TEXTURE_2D,
                    texture,
                    BASE_MIP_LEVEL,
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

    pub fn draw_minified(
        &mut self,
        target: DrawTarget,
        stubs: StubTextures,
        source: DrawSource<'_>,
        seat: SeatGeometry,
        slice: &VisibleSlice,
    ) -> bool {
        use core::slice as slice_ext;

        const ONE_TEXEL_PER_STEP: (f64, f64) = (1.0, 1.0);

        let scratch_dims = (slice.width_px, slice.source_row_count());
        let variant = if matches!(source.layout, PixelLayout::Mono { .. })
        {
            EwaVariant::Mono
        } else {
            EwaVariant::Color
        };
        let Some(scratch) =
            self.bind_scratch_framebuffer(scratch_dims, variant)
        else {
            return false;
        };

        // Scratch rows carry signed lobes outside viewport bounds.
        // SAFETY: Context is current; `self.scratch_framebuffer` is complete and bound.
        unsafe {
            self.gl.disable(glow::SCISSOR_TEST);
            self.gl.disable(glow::BLEND);
            self.gl.viewport(0, 0, scratch_dims.0, scratch_dims.1);
            self.gl.active_texture(EWA_LUT_TEXTURE_UNIT);
            self.gl
                .bind_texture(glow::TEXTURE_1D, Some(self.weight_lut));

            match source.layout {
                PixelLayout::Mono { luma, gain, offset } => {
                    self.horizontal_mono.common.bind(
                        slice,
                        source.dims,
                        seat.origin.0,
                        source.transfer,
                    );
                    program::bind_mono_luma(
                        &self.gl,
                        self.horizontal_mono.luma_gain_offset,
                        *luma,
                        (*gain, *offset),
                    );
                }
                PixelLayout::Rgba { texture } => {
                    self.horizontal_color.common.bind(
                        slice,
                        source.dims,
                        seat.origin.0,
                        source.transfer,
                    );
                    program::bind_rgba_source(
                        &self.gl,
                        self.horizontal_color.pixel_layout,
                        *texture,
                    );
                }
                PixelLayout::Yuv(yuv) => {
                    self.horizontal_color.common.bind(
                        slice,
                        source.dims,
                        seat.origin.0,
                        source.transfer,
                    );
                    self.horizontal_color.yuv.bind(
                        &self.gl,
                        self.horizontal_color.pixel_layout,
                        yuv,
                        stubs,
                        ONE_TEXEL_PER_STEP,
                    );
                }
            }
            self.gl.draw_arrays(
                glow::TRIANGLE_STRIP,
                FIRST_VERTEX_INDEX,
                QUAD_VERTEX_COUNT,
            );
        }

        program::restore_screen_state(&self.gl, target);
        let (vertical, scratch_textures): (&VerticalProgram, &[Texture]) =
            match &scratch {
                ScratchTargets::Mono(primary) => {
                    (&self.vertical_mono, slice_ext::from_ref(primary))
                }
                ScratchTargets::Color(targets) => {
                    (&self.vertical_color, targets)
                }
            };
        // SAFETY: Context is current and default FBO is restored.
        unsafe {
            self.gl.use_program(Some(vertical.program.program));
            vertical.quad.bind(&self.gl, target.window, seat);
            self.gl.uniform_1_i32(
                Some(&vertical.visible_origin_x),
                slice.origin_x_px,
            );
            self.gl.uniform_1_i32(
                Some(&vertical.visible_width),
                slice.width_px,
            );
            self.gl.uniform_1_i32(
                Some(&vertical.source_y_start),
                slice.source_y_range.start,
            );
            self.gl.uniform_1_i32(
                Some(&vertical.source_y_end),
                slice.source_y_range.end,
            );
            self.gl.uniform_1_f32(
                Some(&vertical.source_per_screen_y),
                program::gl_f32(slice.source_per_screen.1),
            );
            self.gl.uniform_1_i32(
                Some(&vertical.dither),
                i32::from(source.needs_dither),
            );
            for (&unit, &texture) in
                [glow::TEXTURE0, glow::TEXTURE1, glow::TEXTURE2]
                    .iter()
                    .zip(scratch_textures)
            {
                self.gl.active_texture(unit);
                self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            }
            self.gl.active_texture(EWA_LUT_TEXTURE_UNIT);
            self.gl
                .bind_texture(glow::TEXTURE_1D, Some(self.weight_lut));
            self.gl.draw_arrays(
                glow::TRIANGLE_STRIP,
                FIRST_VERTEX_INDEX,
                QUAD_VERTEX_COUNT,
            );
        }
        true
    }
}

fn link_ewa_program(
    gl: Arc<Context>,
    vertex_source: &str,
    fragment_parts: &[&str],
    samplers: &[(i32, &str)],
    debug_name: &str,
) -> eros::Result<ShaderPipeline> {
    let gl_program = program::link(
        gl,
        vertex_source,
        fragment_parts,
        samplers,
        debug_name,
    )?;
    let texel_count = f64::from(EWA_LUT_TEXEL_COUNT_GL);
    let (support, scale, bias) = (
        program::gl_f32(EWA_SUPPORT_TEXELS),
        program::gl_f32(f64::from(EWA_LUT_INTERVALS) / texel_count),
        program::gl_f32(0.5 / texel_count),
    );
    let support_location = program::uniform_location(
        &gl_program.gl,
        gl_program.program,
        "u_ewa_support",
    );
    let scale_bias_location = program::uniform_location(
        &gl_program.gl,
        gl_program.program,
        "u_ewa_lut_scale_bias",
    );
    // SAFETY: Context is current and `gl_program.program` is active from `link`.
    unsafe {
        gl_program
            .gl
            .uniform_1_f32(Some(&support_location), support);
        gl_program.gl.uniform_2_f32(
            Some(&scale_bias_location),
            scale,
            bias,
        );
    }
    Ok(gl_program)
}

fn create_vertical_program(
    gl: Arc<Context>,
    variant: EwaVariant,
    samplers: &[(i32, &str)],
    debug_name: &str,
) -> eros::Result<VerticalProgram> {
    let gl_program = link_ewa_program(
        gl,
        VERTEX_SHADER_SOURCE,
        &[
            GLSL_VERSION,
            variant.glsl_define(),
            GLSL_BAYER4,
            GLSL_EWA_LUT,
            EWA_VERTICAL_BODY,
        ],
        samplers,
        debug_name,
    )?;
    let location = |name: &str| {
        program::uniform_location(&gl_program.gl, gl_program.program, name)
    };
    Ok(VerticalProgram {
        quad: QuadUniforms::lookup(&gl_program.gl, gl_program.program),
        visible_origin_x: location("u_visible_origin_x"),
        visible_width: location("u_visible_width"),
        source_y_start: location("u_source_y_start"),
        source_y_end: location("u_source_y_end"),
        source_per_screen_y: location("u_source_per_screen_y"),
        dither: location("u_dither"),
        program: gl_program,
    })
}

fn allocate_scratch_texture(
    gl: &Context,
    existing: Option<Texture>,
    dims: (i32, i32),
    spec: &ScratchTargetSpec,
) -> Texture {
    // SAFETY: Context is current; caller keeps `dims` within `1..=max_texture_edge`.
    unsafe {
        let texture = existing.unwrap_or_else(|| {
            program::create_2d_texture(
                gl,
                glow::NEAREST,
                glow::NEAREST,
                Some(spec.debug_name),
            )
        });
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            BASE_MIP_LEVEL,
            spec.internal_format.cast_signed(),
            dims.0,
            dims.1,
            NO_BORDER,
            spec.format,
            glow::FLOAT,
            PixelUnpackData::Slice(None),
        );
        texture
    }
}

fn jinc(radius: f64) -> f64 {
    use core::f64::consts::PI;

    const SERIES_TERMS: u32 = 35;

    let half_argument = 0.5 * PI * radius;
    let neg_square = -(half_argument * half_argument);
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    for term_index in 1..=SERIES_TERMS {
        let index_f64 = f64::from(term_index);
        term *= neg_square / (index_f64 * (index_f64 + 1.0));
        sum += term;
    }
    sum
}

fn ewa_lanczos3(radius: f64) -> f64 {
    if radius >= EWA_SUPPORT_TEXELS {
        return 0.0;
    }
    jinc(radius)
        * jinc(radius * (EWA_JINC_FIRST_ZERO / EWA_SUPPORT_TEXELS))
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
    let norm = dot_product(vector, vector).sqrt();
    if norm > 0.0 {
        for slot in vector {
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

    let mut next_vector = [0.0_f64; EWA_GRID_LEN];
    let rows = matrix.as_chunks::<EWA_GRID_LEN>().0;
    for _ in 0..POWER_ITERATIONS {
        for (target, row) in next_vector.iter_mut().zip(rows) {
            *target = dot_product(row, &vector);
        }
        if let Some(basis) = orthogonal_basis {
            project_out(&mut next_vector, basis);
        }
        normalize_vector(&mut next_vector);
        vector.copy_from_slice(&next_vector);
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
fn ewa_rank2_lut() -> [[f32; EWA_RANK]; EWA_LUT_TEXEL_COUNT] {
    use core::f64::consts::PI;

    let grid_step = EWA_SUPPORT_TEXELS / f64::from(EWA_LUT_INTERVALS);
    let coords: Vec<f64> = (-EWA_LUT_INTERVALS..=EWA_LUT_INTERVALS)
        .map(|index| f64::from(index) * grid_step)
        .collect();

    let mut matrix = Vec::with_capacity(EWA_GRID_LEN * EWA_GRID_LEN);
    for &coord_y in &coords {
        for &coord_x in &coords {
            matrix.push(ewa_lanczos3(coord_x.hypot(coord_y)));
        }
    }

    let center_index = EWA_LUT_TEXEL_COUNT - 1;
    let primary_init: Vec<f64> = coords
        .iter()
        .map(|&coord| (0.5 * PI * coord / EWA_SUPPORT_TEXELS).cos())
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
        .map(|&coord| (2.0 * coord / EWA_SUPPORT_TEXELS).cos())
        .collect();
    let (secondary_eigenvalue, secondary_vector) = power_eigen_symmetric(
        &matrix,
        center_index,
        &secondary_init,
        Some(&primary_vector),
    );

    let primary_scale = primary_eigenvalue.abs().sqrt();
    let secondary_scale = secondary_eigenvalue.abs().sqrt();
    let mut lut = [[0.0_f32; EWA_RANK]; EWA_LUT_TEXEL_COUNT];
    if let (Some(primary_half), Some(secondary_half)) = (
        primary_vector.get(center_index..),
        secondary_vector.get(center_index..),
    ) {
        for ((slot, &primary_weight), &secondary_weight) in
            lut.iter_mut().zip(primary_half).zip(secondary_half)
        {
            *slot = [
                program::gl_f32(primary_scale * primary_weight),
                program::gl_f32(secondary_scale * secondary_weight),
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
    // SAFETY: Context is current; `lut` holds `EWA_LUT_TEXEL_COUNT` RG floats.
    unsafe {
        #[expect(
            clippy::expect_used,
            reason = "texture-creation failure panics by policy"
        )]
        let texture = gl.create_texture().expect("GL EWA LUT texture");
        gl.bind_texture(glow::TEXTURE_1D, Some(texture));
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, FLOAT_UNPACK_ALIGNMENT);
        gl.tex_parameter_i32(
            glow::TEXTURE_1D,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE.cast_signed(),
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_1D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR.cast_signed(),
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_1D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR.cast_signed(),
        );
        gl.tex_image_1d(
            glow::TEXTURE_1D,
            BASE_MIP_LEVEL,
            glow::RG32F.cast_signed(),
            EWA_LUT_TEXEL_COUNT_GL,
            NO_BORDER,
            glow::RG,
            glow::FLOAT,
            PixelUnpackData::Slice(Some(bytemuck::cast_slice(&lut))),
        );
        gl.bind_texture(glow::TEXTURE_1D, None);
        program::label(
            gl,
            glow::TEXTURE,
            texture.0.get(),
            "ewa rank2 lut",
        );
        texture
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;

    #[test]
    fn ewa_rank2_lut_approximates_circular_lanczos3() {
        assert!(jinc(EWA_JINC_FIRST_ZERO).abs() < 1e-12);
        assert!(jinc(EWA_SUPPORT_TEXELS).abs() < 1e-12);

        let lut = ewa_rank2_lut();
        assert_eq!(lut.last().copied(), Some([0.0, 0.0]));

        let grid_step = EWA_SUPPORT_TEXELS / f64::from(EWA_LUT_INTERVALS);
        let lut_weights = |offset: i32| -> (f64, f64) {
            let index = usize::try_from(offset.unsigned_abs())
                .expect("offset fits usize");
            let entry = lut.get(index).expect("offset inside LUT");
            (f64::from(entry[0]), f64::from(entry[1]))
        };
        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        for row in -EWA_LUT_INTERVALS..=EWA_LUT_INTERVALS {
            let coord_y = f64::from(row) * grid_step;
            let (row_primary, row_secondary) = lut_weights(row);
            for col in -EWA_LUT_INTERVALS..=EWA_LUT_INTERVALS {
                let coord_x = f64::from(col) * grid_step;
                let (col_primary, col_secondary) = lut_weights(col);
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
        let panned = compute_visible_slice(
            SeatGeometry {
                origin: (-200.0, -500.0),
                size: (2031.0, 2880.0),
            },
            (1280, 1000),
            (4062, 5760),
        )
        .expect("panned slice");
        assert_eq!(panned.origin_x_px, 0);
        assert_eq!(panned.width_px, 1280);
        assert!((900..=993).contains(&panned.source_y_range.start));
        assert!((3007..3100).contains(&panned.source_y_range.end));
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

        let huge_pan = compute_visible_slice(
            SeatGeometry {
                origin: (-1e12, -1e12),
                size: (2e12, 2e12),
            },
            (1280, 1000),
            (4062, 5760),
        )
        .expect("huge pan still covers viewport");
        assert_eq!(huge_pan.origin_x_px, 0);
        assert_eq!(huge_pan.width_px, 1280);
        assert!(
            0 <= huge_pan.source_y_range.start
                && huge_pan.source_y_range.start
                    < huge_pan.source_y_range.end
                && huge_pan.source_y_range.end <= 5760
        );

        let tiny_scale = compute_visible_slice(
            SeatGeometry {
                origin: (0.0, 0.0),
                size: (1e-9, 1e-9),
            },
            (1280, 1000),
            (4062, 5760),
        )
        .expect("tiny scale covers one pixel");
        assert!(
            0 <= tiny_scale.source_y_range.start
                && tiny_scale.source_y_range.start
                    < tiny_scale.source_y_range.end
                && tiny_scale.source_y_range.end <= 5760
        );
    }
}
