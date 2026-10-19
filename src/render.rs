//! GL rendering: window/context (glutin-on-winit) and the page quad renderer.
//! All GL `unsafe` lives here behind safe functions (the only other unsafe in
//! cryslen is decode.rs's libavif FFI block, behind the `avif` feature).
//!
//! Two page slots, YUV plane textures with shader-side conversion, spread
//! layout draw. Chroma siting (avifChromaSamplePosition) is not applied — the
//! default co-sited/midway assumption biases chroma by at most half a texel,
//! invisible in practice. The app clears the framebuffer each frame; `draw`
//! renders only the page quads.

use core::num::NonZeroU32;
use std::sync::Arc;

use egui_winit::winit;
use glow::HasContext;
use winit::raw_window_handle::HasWindowHandle as _;

use crate::decode::{plane_bytes, plane_dims, Decoded, Matrix, Pixels, Plane, Subsampling};
use crate::view::{SpreadLayout, View};

/// Fixed -1..1 strip; the VS places it at the slot's screen rect from
/// uniforms, so pan/zoom never touch vertex data. Top-origin math lives in
/// the VS (GL clip space is bottom-origin; winit coords are top-origin).
const VS_CORE: &str = r#"
    #version 330 core
    in vec2 pos;
    uniform vec2 u_win;    // window size, px
    uniform vec2 u_offset; // slot rect origin, screen px
    uniform vec2 u_size;   // slot rect size, screen px
    out vec2 v_px;
    void main() {
        vec2 px = u_offset + (pos * 0.5 + 0.5) * u_size;
        v_px = px;
        gl_Position = vec4(px.x / u_win.x * 2.0 - 1.0, 1.0 - px.y / u_win.y * 2.0, 0.0, 1.0);
    }"#;

/// One shader, both pixel layouts: mode/subsampling/depth/range/matrix arrive
/// as uniforms, so there are no per-variant programs. Chroma is sampled
/// bilinearly at cosited scaled coordinates; for 444 that is exact.
const FS_CORE: &str = r#"
    #version 330 core
    in vec2 v_px;
    out vec4 color;
    uniform vec2 u_offset;
    uniform vec2 u_scale;  // screen px per image px
    uniform vec2 u_img;    // image size, texels
    uniform vec2 u_csize;  // chroma plane size, texels
    uniform int u_mode;    // 0 = RGBA, 1 = YUV
    uniform int u_chroma;  // 0 mono, 1 = 420, 2 = 422, 3 = 444
    uniform float u_norm;  // depth normalize (1.0 for 8-bit storage)
    uniform vec3 u_yuv_off;
    uniform mat3 u_mat;
    uniform sampler2D u_tex0; // RGBA or Y
    uniform sampler2D u_tex1; // Cb (1x1 midpoint for mono)
    uniform sampler2D u_tex2; // Cr
    void main() {
        // Fragment centers interpolate to image texel centers at 1:1 zoom
        // (ip = k + 0.5), so ip / u_img already addresses texel centers —
        // no extra half-texel offset.
        vec2 ip = (v_px - u_offset) / u_scale;
        if (u_mode == 0) { color = vec4(texture(u_tex0, ip / u_img).rgb, 1.0); return; }
        float y = texture(u_tex0, ip / u_img).r;
        vec2 ratio = u_chroma == 1 ? vec2(0.5) : vec2(u_chroma == 2 ? 0.5 : 1.0, 1.0);
        vec2 cuv = ip * ratio / u_csize;
        vec3 yuv = vec3(y, texture(u_tex1, cuv).r, texture(u_tex2, cuv).r);
        color = vec4(u_mat * (yuv * u_norm - u_yuv_off), 1.0);
    }"#;

const VS_ES: &str = r#"
    #version 300 es
    in vec2 pos;
    uniform vec2 u_win;
    uniform vec2 u_offset;
    uniform vec2 u_size;
    out vec2 v_px;
    void main() {
        vec2 px = u_offset + (pos * 0.5 + 0.5) * u_size;
        v_px = px;
        gl_Position = vec4(px.x / u_win.x * 2.0 - 1.0, 1.0 - px.y / u_win.y * 2.0, 0.0, 1.0);
    }"#;

// highp: mediump may be fp16 — ulp at 16k image px is 8.0, wrecking the
// coordinate math (challenge F5; ES3 mandates highp support).
const FS_ES: &str = r#"
    #version 300 es
    precision highp float;
    in vec2 v_px;
    out vec4 color;
    uniform vec2 u_offset;
    uniform vec2 u_scale;
    uniform vec2 u_img;
    uniform vec2 u_csize;
    uniform int u_mode;
    uniform int u_chroma;
    uniform float u_norm;
    uniform vec3 u_yuv_off;
    uniform mat3 u_mat;
    uniform sampler2D u_tex0;
    uniform sampler2D u_tex1;
    uniform sampler2D u_tex2;
    void main() {
        vec2 ip = (v_px - u_offset) / u_scale;
        if (u_mode == 0) { color = vec4(texture(u_tex0, ip / u_img).rgb, 1.0); return; }
        float y = texture(u_tex0, ip / u_img).r;
        vec2 ratio = u_chroma == 1 ? vec2(0.5) : vec2(u_chroma == 2 ? 0.5 : 1.0, 1.0);
        vec2 cuv = ip * ratio / u_csize;
        vec3 yuv = vec3(y, texture(u_tex1, cuv).r, texture(u_tex2, cuv).r);
        color = vec4(u_mat * (yuv * u_norm - u_yuv_off), 1.0);
    }"#;

/// Window + EGL context + surface. vsync is DontWait; the app must call
/// `pre_present_and_swap()` to present (Wayland frame pacing inside).
pub struct Window {
    pub winit: winit::window::Window,
    gl: Arc<glow::Context>,
    surface: glutin::surface::Surface<glutin::surface::WindowSurface>,
    context: glutin::context::PossiblyCurrentContext,
}

impl Window {
    pub fn new(event_loop: &winit::event_loop::ActiveEventLoop) -> Window {
        unsafe { Self::create(event_loop) }
    }

    /// glutin-on-winit creation (egui_glow's pure_glow pattern): prefer
    /// desktop GL, fall back to GLES; swap is non-blocking.
    unsafe fn create(event_loop: &winit::event_loop::ActiveEventLoop) -> Window {
        use glutin::context::NotCurrentGlContext as _;
        use glutin::display::{GetGlDisplay as _, GlDisplay as _};
        use glutin::prelude::GlSurface as _;

        let attrs = winit::window::WindowAttributes::default()
            .with_resizable(true)
            .with_inner_size(winit::dpi::LogicalSize { width: INITIAL_WINDOW_WIDTH, height: INITIAL_WINDOW_HEIGHT })
            .with_title("cryslen");

        let (mut window, gl_config) =
            glutin_winit::DisplayBuilder::new()
                .with_preference(glutin_winit::ApiPreference::FallbackEgl)
                .with_window_attributes(Some(attrs.clone()))
                .build(event_loop, glutin::config::ConfigTemplateBuilder::new(), |mut it| {
                    it.next().expect("no GL config")
                })
                .expect("create GL config");
        let gl_display = gl_config.display();

        let raw = window.as_ref().map(|w| w.window_handle().unwrap().as_raw());
        let ctx_attrs = glutin::context::ContextAttributesBuilder::new().build(raw);
        let fallback = glutin::context::ContextAttributesBuilder::new()
            .with_context_api(glutin::context::ContextApi::Gles(None))
            .build(raw);
        let not_current = gl_display.create_context(&gl_config, &ctx_attrs).unwrap_or_else(|_| {
            gl_config.display().create_context(&gl_config, &fallback).unwrap()
        });

        let window = window.take().unwrap_or_else(|| {
            glutin_winit::finalize_window(event_loop, attrs, &gl_config).unwrap()
        });
        let (w, h): (u32, u32) = window.inner_size().into();
        let surface_attrs =
            glutin::surface::SurfaceAttributesBuilder::<glutin::surface::WindowSurface>::new()
                .build(
                    window.window_handle().unwrap().as_raw(),
                    NonZeroU32::new(w).unwrap(),
                    NonZeroU32::new(h).unwrap(),
                );
        let surface = gl_display.create_window_surface(&gl_config, &surface_attrs).unwrap();
        let context = not_current.make_current(&surface).unwrap();
        // Non-blocking present + window.pre_present_notify() (in
        // pre_present_and_swap) gives Wayland frame-callback pacing without a
        // blocking eglSwapBuffers owning the event loop — same choice eframe
        // made.
        surface
            .set_swap_interval(&context, glutin::surface::SwapInterval::DontWait)
            .unwrap();

        let gl = Arc::new(glow::Context::from_loader_function(|s| {
            let s = std::ffi::CString::new(s).unwrap();
            gl_display.get_proc_address(&s)
        }));
        Window { winit: window, gl, surface, context }
    }

    pub fn glow(&self) -> Arc<glow::Context> {
        Arc::clone(&self.gl)
    }

    pub fn resize(&self, size: winit::dpi::PhysicalSize<u32>) {
        use glutin::prelude::GlSurface as _;
        self.surface
            .resize(&self.context, size.width.try_into().unwrap(), size.height.try_into().unwrap());
    }

    /// Frame-callback notify then non-blocking swap.
    pub fn pre_present_and_swap(&self) {
        use glutin::prelude::GlSurface as _;
        self.winit.pre_present_notify();
        self.surface.swap_buffers(&self.context).unwrap();
    }
}

/// Up to two page slots, each an RGBA texture or a YUV plane triple.
/// Upload happens on the main thread at frame time; pan/zoom are uniforms.
/// Each slot remembers which page it holds, so `set` with an unchanged page
/// is a no-op — callers never need to mirror GPU state on their side.
pub struct PageSlots {
    gl: Arc<glow::Context>,
    program: <glow::Context as glow::HasContext>::Program,
    vao: <glow::Context as glow::HasContext>::VertexArray,
    slots: [Slot; 2],
    vbo: <glow::Context as glow::HasContext>::Buffer,
    max_tex: u32,
}

/// One page slot's GPU state. Plain data; objects freed on replace/clear.
struct Slot {
    /// Page whose pixels the textures hold (`None` when empty).
    page: Option<usize>,
    texture: Option<Textures>,
    size: (u32, u32),
    uniforms: PageUniforms,
    /// Depth-normalize factor for U16-stored planes (1.0 otherwise).
    norm: f32,
}

enum Textures {
    /// Single RGBA texture.
    Rgba(<glow::Context as glow::HasContext>::Texture),
    /// Y + Cb + Cr plane textures (Cb/Cr empty for mono, bound to a 1x1).
    Yuv {
        y: <glow::Context as glow::HasContext>::Texture,
        cb: <glow::Context as glow::HasContext>::Texture,
        cr: <glow::Context as glow::HasContext>::Texture,
    },
}

/// Initial window size, logical px.
const INITIAL_WINDOW_WIDTH: f64 = 1000.0;
const INITIAL_WINDOW_HEIGHT: f64 = 750.0;

/// Neutral gray chroma for mono pages, in 8-bit storage (128 of 255;
/// the wide-plane equivalent is `1 << (depth - 1)`).
const MONO_CHROMA_MIDPOINT_8: u8 = 128;

const IDENTITY3: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// Uniform set describing an RGBA slot (mode 0); also the empty-slot default.
const RGBA_UNIFORMS: PageUniforms = PageUniforms {
    mode: 0,
    subsampling: Subsampling::Yuv444,
    depth: 8,
    full_range: true,
    matrix: Matrix::Identity,
};

fn empty_slot() -> Slot {
    Slot { page: None, texture: None, size: (0, 0), uniforms: RGBA_UNIFORMS, norm: 1.0 }
}

impl PageSlots {
    pub fn new(gl: Arc<glow::Context>) -> PageSlots {
        let (vs, fs) = if gl.version().is_embedded { (VS_ES, FS_ES) } else { (VS_CORE, FS_CORE) };
        let compile = |kind, src| unsafe {
            let shader = gl.create_shader(kind).unwrap();
            gl.shader_source(shader, src);
            gl.compile_shader(shader);
            assert!(
                gl.get_shader_compile_status(shader),
                "{}",
                gl.get_shader_info_log(shader)
            );
            shader
        };
        let (program, vbo, vao);
        unsafe {
            program = gl.create_program().unwrap();
            vbo = gl.create_buffer().unwrap();
            vao = gl.create_vertex_array().unwrap();

            gl.attach_shader(program, compile(glow::VERTEX_SHADER, vs));
            gl.attach_shader(program, compile(glow::FRAGMENT_SHADER, fs));
            gl.link_program(program);
            assert!(gl.get_program_link_status(program), "{}", gl.get_program_info_log(program));

            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&[-1.0f32, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
                glow::STATIC_DRAW,
            );

            gl.bind_vertex_array(Some(vao));
            let loc = gl.get_attrib_location(program, "pos").unwrap();
            gl.enable_vertex_attrib_array(loc);
            gl.vertex_attrib_pointer_f32(loc, 2, glow::FLOAT, false, 8, 0);

            // Sampler units are fixed; draw only rebinds textures.
            gl.use_program(Some(program));
            for (unit, name) in ["u_tex0", "u_tex1", "u_tex2"].iter().enumerate() {
                gl.uniform_1_i32(gl.get_uniform_location(program, name).as_ref(), unit as i32);
            }
        }
        let max_tex = unsafe { gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE) } as u32;
        PageSlots { gl, program, vao, slots: [empty_slot(), empty_slot()], vbo, max_tex }
    }

    /// Upload decoded pixels of `page` into visual slot `i` (0 = left,
    /// 1 = right); re-uploading the same page is a no-op. Panics loudly on
    /// MAX_TEXTURE_SIZE violations (challenge finding: never fail black).
    pub fn set(&mut self, slot: usize, page: usize, decoded: &Decoded) {
        if self.slots[slot].page == Some(page) {
            return;
        }
        let (width, height) = (decoded.width, decoded.height);
        // Fail loudly instead of uploading an oversized texture that GL
        // silently rejects (it then samples as black).
        assert!(width > 0 && height > 0, "decoded page has zero dimension");
        assert!(
            width <= self.max_tex && height <= self.max_tex,
            "image {width}x{height} exceeds GL max texture size {} (tiling not implemented)",
            self.max_tex
        );
        self.clear(slot);
        let gl = &self.gl;
        let textures = match &decoded.pixels {
            Pixels::Rgba8(rgba_bytes) => {
                assert_eq!(rgba_bytes.len(), width as usize * height as usize * 4, "rgba byte length");
                Textures::Rgba(upload_rgba(gl, rgba_bytes, (width, height)))
            }
            Pixels::Yuv(yuv) => {
                let ((luma_w, luma_h), (chroma_w, chroma_h)) =
                    plane_dims((width, height), yuv.subsampling);
                let wide = matches!(yuv.y, Plane::U16(_));
                // GL_R16 is not filterable in GLES: LINEAR makes the texture
                // incomplete (samples black); NEAREST without mips is the
                // safe fallback on an ES context.
                let filterable = !(gl.version().is_embedded && wide);
                let expect = |plane: &Plane, want_samples: usize, what: &str| {
                    // U16 planes are two bytes per sample.
                    let want_bytes = want_samples * if wide { 2 } else { 1 };
                    assert_eq!(plane_bytes(plane).len(), want_bytes, "{what} plane bytes");
                };
                expect(&yuv.y, luma_w as usize * luma_h as usize, "y");
                let luma_texture =
                    upload_plane(gl, plane_bytes(&yuv.y), (luma_w, luma_h), wide, filterable);
                let (cb_texture, cr_texture) = if yuv.subsampling == Subsampling::Mono {
                    // 1x1 chroma at the range midpoint; the matrix cancels it
                    // to neutral gray. LE pairs match plane_bytes' U16 layout.
                    let midpoint_bytes: Vec<u8> = if wide {
                        (1u32 << (yuv.depth.clamp(MIN_PLANE_DEPTH, MAX_PLANE_DEPTH) - 1))
                            .to_le_bytes()
                            .to_vec()
                    } else {
                        MONO_CHROMA_MIDPOINT_8.to_le_bytes().to_vec()
                    };
                    (upload_plane(gl, &midpoint_bytes, (1, 1), wide, filterable),
                     upload_plane(gl, &midpoint_bytes, (1, 1), wide, filterable))
                } else {
                    expect(&yuv.cb, chroma_w as usize * chroma_h as usize, "cb");
                    expect(&yuv.cr, chroma_w as usize * chroma_h as usize, "cr");
                    (upload_plane(gl, plane_bytes(&yuv.cb), (chroma_w, chroma_h), wide, filterable),
                     upload_plane(gl, plane_bytes(&yuv.cr), (chroma_w, chroma_h), wide, filterable))
                };
                self.slots[slot].norm = depth_norm(yuv.depth, wide);
                Textures::Yuv { y: luma_texture, cb: cb_texture, cr: cr_texture }
            }
        };
        self.slots[slot].page = Some(page);
        self.slots[slot].texture = Some(textures);
        self.slots[slot].size = (width, height);
        self.slots[slot].uniforms = PageUniforms::from(&decoded.pixels);
    }

    /// Clear a slot (page left the spread — undecoded-but-wanted pages keep
    /// their last frame instead, per the hold-last-frame rule).
    pub fn clear(&mut self, slot: usize) {
        if let Some(textures) = self.slots[slot].texture.take() {
            delete_textures(&self.gl, &textures);
            self.slots[slot].page = None;
            self.slots[slot].size = (0, 0);
            self.slots[slot].norm = 1.0;
        }
    }

    /// Decoded size of the page held by the slot, if any.
    pub fn size(&self, slot: usize) -> Option<(u32, u32)> {
        self.slots[slot].texture.as_ref().map(|_| self.slots[slot].size)
    }

    /// Draw the spread laid out by `layout` with `view`. Sets viewport,
    /// disables blend, pixel-snaps offsets; one quad per occupied slot.
    /// YUV pages sample their planes and convert in-shader (subsampling,
    /// depth, range, matrix passed as uniforms derived from `Pixels`).
    /// Does NOT clear — the app owns the framebuffer clear each frame.
    pub fn draw(&self, win: (f32, f32), layout: &SpreadLayout, view: &View) {
        let gl = &self.gl;
        if win.0 <= 0.0 || win.1 <= 0.0 {
            return;
        }
        unsafe {
            // Own the state this draw depends on: egui's paint leaves BLEND
            // enabled, and nothing resets the viewport after a resize.
            gl.viewport(0, 0, win.0 as i32, win.1 as i32);
            gl.disable(glow::BLEND);
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));
            let uniform_location =
                |name: &str| gl.get_uniform_location(self.program, name);
            gl.uniform_2_f32(uniform_location("u_win").as_ref(), win.0, win.1);
            // `rects` is [left_slot, right_slot] by array position.
            for (slot, rect) in layout.rects.iter().enumerate() {
                let Some(rect) = rect else { continue };
                let Some(slot_state) = self.slots.get(slot) else { continue };
                let Some(textures) = &slot_state.texture else { continue };
                // Whole-pixel rect origin; LINEAR samples shimmer at 1:1 zoom
                // on fractional offsets otherwise.
                let screen_origin = (
                    (view.offset.0 + rect.pos.0 * view.scale).round(),
                    (view.offset.1 + rect.pos.1 * view.scale).round(),
                );
                let screen_size = (rect.size.0 * view.scale, rect.size.1 * view.scale);
                if screen_size.0 <= 0.0 || screen_size.1 <= 0.0 {
                    continue;
                }
                let page_uniforms = slot_state.uniforms;
                let (_, (chroma_width, chroma_height)) =
                    plane_dims(slot_state.size, page_uniforms.subsampling);
                let (luma_anchor, chroma_anchor) = range_offsets(page_uniforms.full_range);
                gl.uniform_2_f32(uniform_location("u_offset").as_ref(), screen_origin.0, screen_origin.1);
                gl.uniform_2_f32(uniform_location("u_size").as_ref(), screen_size.0, screen_size.1);
                gl.uniform_2_f32(uniform_location("u_scale").as_ref(), view.scale, view.scale);
                gl.uniform_2_f32(
                    uniform_location("u_img").as_ref(),
                    slot_state.size.0 as f32,
                    slot_state.size.1 as f32,
                );
                gl.uniform_2_f32(
                    uniform_location("u_csize").as_ref(),
                    chroma_width as f32,
                    chroma_height as f32,
                );
                gl.uniform_1_i32(uniform_location("u_mode").as_ref(), page_uniforms.mode as i32);
                gl.uniform_1_i32(
                    uniform_location("u_chroma").as_ref(),
                    chroma_id(page_uniforms.subsampling) as i32,
                );
                gl.uniform_1_f32(uniform_location("u_norm").as_ref(), slot_state.norm);
                gl.uniform_3_f32(
                    uniform_location("u_yuv_off").as_ref(),
                    luma_anchor,
                    chroma_anchor,
                    chroma_anchor,
                );
                gl.uniform_matrix_3_f32_slice(
                    uniform_location("u_mat").as_ref(),
                    false,
                    &yuv_to_rgb_matrix(page_uniforms.matrix, page_uniforms.full_range),
                );
                gl.active_texture(glow::TEXTURE0);
                match textures {
                    Textures::Rgba(rgba_texture) => {
                        gl.bind_texture(glow::TEXTURE_2D, Some(*rgba_texture))
                    }
                    Textures::Yuv { y, cb, cr } => {
                        gl.bind_texture(glow::TEXTURE_2D, Some(*y));
                        gl.active_texture(glow::TEXTURE1);
                        gl.bind_texture(glow::TEXTURE_2D, Some(*cb));
                        gl.active_texture(glow::TEXTURE2);
                        gl.bind_texture(glow::TEXTURE_2D, Some(*cr));
                    }
                }
                gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            }
        }
    }
}

impl Drop for PageSlots {
    fn drop(&mut self) {
        // Delete while the context still lives; the app must drop Pages
        // before Window (challenge F7 teardown ordering).
        for slot in &mut self.slots {
            if let Some(textures) = slot.texture.take() {
                delete_textures(&self.gl, &textures);
            }
        }
        unsafe {
            self.gl.delete_program(self.program);
            self.gl.delete_vertex_array(self.vao);
            self.gl.delete_buffer(self.vbo);
        }
    }
}

fn delete_textures(gl: &glow::Context, textures: &Textures) {
    unsafe {
        match textures {
            Textures::Rgba(rgba_texture) => gl.delete_texture(*rgba_texture),
            Textures::Yuv { y, cb, cr } => {
                gl.delete_texture(*y);
                gl.delete_texture(*cb);
                gl.delete_texture(*cr);
            }
        }
    }
}

/// Texture with viewer defaults; CLAMP keeps edge texels from bleeding.
/// `filterable` false (GLES R16) means NEAREST and no mips.
fn new_texture(gl: &glow::Context, filterable: bool) -> <glow::Context as HasContext>::Texture {
    let texture = unsafe { gl.create_texture().unwrap() };
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as _);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as _);
        let (min, mag) = if filterable {
            // Mipmapped minification: fit view of a 50MP page minifies ~8x;
            // LINEAR-only shimmers (challenge F3).
            (glow::LINEAR_MIPMAP_LINEAR, glow::LINEAR)
        } else {
            (glow::NEAREST, glow::NEAREST)
        };
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, min as _);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, mag as _);
    }
    texture
}

fn upload_rgba(
    gl: &glow::Context,
    data: &[u8],
    dims: (u32, u32),
) -> <glow::Context as HasContext>::Texture {
    let texture = new_texture(gl, true);
    unsafe {
        gl.tex_image_2d(
            glow::TEXTURE_2D, 0, glow::RGBA8 as _,
            dims.0 as i32, dims.1 as i32, 0,
            glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(data)),
        );
        gl.generate_mipmap(glow::TEXTURE_2D);
    }
    texture
}

/// Single-channel plane: R8 for 8-bit, R16 for wider. Mipmaps unless the
/// format is unfilterable on this context.
fn upload_plane(
    gl: &glow::Context,
    data: &[u8],
    dims: (u32, u32),
    wide: bool,
    filterable: bool,
) -> <glow::Context as HasContext>::Texture {
    let (internal, ty) = if wide { (glow::R16, glow::UNSIGNED_SHORT) } else { (glow::R8, glow::UNSIGNED_BYTE) };
    let texture = new_texture(gl, filterable);
    unsafe {
        // Odd row widths: GL's default 4-byte unpack alignment would desync
        // every row after the first.
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        gl.tex_image_2d(
            glow::TEXTURE_2D, 0, internal as _,
            dims.0 as i32, dims.1 as i32, 0,
            glow::RED, ty, glow::PixelUnpackData::Slice(Some(data)),
        );
        if filterable {
            gl.generate_mipmap(glow::TEXTURE_2D);
        }
    }
    texture
}

/// How the shader should read a slot's texture set. Derived CPU-side; passed
/// as uniforms. Purely data — no per-variant shader programs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageUniforms {
    /// 0 = RGBA, 1 = YUV
    pub mode: u32,
    pub subsampling: Subsampling,
    pub depth: u8,
    pub full_range: bool,
    pub matrix: crate::decode::Matrix,
}

impl From<&Pixels> for PageUniforms {
    fn from(p: &Pixels) -> Self {
        match p {
            Pixels::Rgba8(_) => RGBA_UNIFORMS,
            Pixels::Yuv(yuv) => PageUniforms {
                mode: 1,
                subsampling: yuv.subsampling,
                depth: yuv.depth,
                full_range: yuv.full_range,
                matrix: yuv.matrix,
            },
        }
    }
}

/// Shader subsampling id: 0 mono, 1 = 420, 2 = 422, 3 = 444.
fn chroma_id(s: Subsampling) -> u32 {
    match s {
        Subsampling::Mono => 0,
        Subsampling::Yuv420 => 1,
        Subsampling::Yuv422 => 2,
        Subsampling::Yuv444 => 3,
    }
}

/// Chroma neutral point of full-range YUV (0.5 in 0..1 storage).
const FULL_RANGE_CHROMA_NEUTRAL: f32 = 0.5;
/// Limited-range (BT studio swing) anchors: Y 16..235, C 16..240 of 255.
const LIMITED_LUMA_BLACK: f32 = 16.0 / 255.0;
const LIMITED_CHROMA_NEUTRAL: f32 = 128.0 / 255.0;
/// Limited-range excursion scales: 255/219 for luma, 255/224 for chroma.
const LIMITED_LUMA_SCALE: f32 = 255.0 / 219.0;
const LIMITED_CHROMA_SCALE: f32 = 255.0 / 224.0;
/// Largest value an R16 (u16) texture sampler can return.
const R16_STORAGE_MAX: f32 = 65535.0;

/// Quantization anchors subtracted in-shader: (luma, chroma).
fn range_offsets(full_range: bool) -> (f32, f32) {
    if full_range {
        (0.0, FULL_RANGE_CHROMA_NEUTRAL)
    } else {
        (LIMITED_LUMA_BLACK, LIMITED_CHROMA_NEUTRAL)
    }
}

/// Normalizes R16 sampler output for U16-stored planes: sampling divides by
/// [`R16_STORAGE_MAX`] but only `depth` bits are significant. 1.0 for 8-bit
/// storage.
fn depth_norm(depth: u8, wide: bool) -> f32 {
    if !wide {
        return 1.0;
    }
    R16_STORAGE_MAX / ((1u32 << depth.clamp(MIN_PLANE_DEPTH, MAX_PLANE_DEPTH)) - 1) as f32
}

/// Plane bit depths the R8/R16 texture paths accept.
const MIN_PLANE_DEPTH: u8 = 1;
const MAX_PLANE_DEPTH: u8 = 16;

/// Luma coefficients (KR, KB) of each conversion standard; KG = 1 - KR - KB.
const BT601_KR_KB: (f32, f32) = (0.299, 0.114);
const BT709_KR_KB: (f32, f32) = (0.2126, 0.0722);
const BT2020_KR_KB: (f32, f32) = (0.2627, 0.0593);

/// YUV→RGB conversion for `rgb = M · (yuv * norm - offsets)`, stored
/// column-major for upload with transpose = false. Derived from the exact
/// KR/KB of each standard; limited range folds in the studio-swing scales
/// ([`LIMITED_LUMA_SCALE`], [`LIMITED_CHROMA_SCALE`]) so the anchors stay
/// the plain offsets from [`range_offsets`]. `Identity` passes planes
/// through as RGB.
fn yuv_to_rgb_matrix(matrix: Matrix, full_range: bool) -> [f32; 9] {
    let (k_red, k_blue) = match matrix {
        Matrix::Bt601 => BT601_KR_KB,
        Matrix::Bt709 => BT709_KR_KB,
        Matrix::Bt2020 => BT2020_KR_KB,
        Matrix::Identity => return IDENTITY3,
    };
    let k_green = 1.0 - k_red - k_blue;
    let (luma_scale, chroma_scale) =
        if full_range { (1.0, 1.0) } else { (LIMITED_LUMA_SCALE, LIMITED_CHROMA_SCALE) };
    let green_from_cb = chroma_scale * 2.0 * k_blue * (1.0 - k_blue) / k_green;
    let green_from_cr = chroma_scale * 2.0 * k_red * (1.0 - k_red) / k_green;
    [
        luma_scale,
        luma_scale,
        luma_scale,
        0.0,
        -green_from_cb,
        chroma_scale * 2.0 * (1.0 - k_blue),
        chroma_scale * 2.0 * (1.0 - k_red),
        -green_from_cr,
        0.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Row `row`, column `col` of the math matrix (storage is column-major).
    fn element(matrix: &[f32; 9], row: usize, col: usize) -> f32 {
        matrix[col * 3 + row]
    }

    #[test]
    fn chroma_ids() {
        assert_eq!(chroma_id(Subsampling::Mono), 0);
        assert_eq!(chroma_id(Subsampling::Yuv420), 1);
        assert_eq!(chroma_id(Subsampling::Yuv422), 2);
        assert_eq!(chroma_id(Subsampling::Yuv444), 3);
    }

    #[test]
    fn bt601_limited_matches_documented_constants() {
        let matrix = yuv_to_rgb_matrix(Matrix::Bt601, false);
        for row in 0..3 {
            assert!(
                (element(&matrix, row, 0) - LIMITED_LUMA_SCALE).abs() < 1e-4,
                "luma scale row {row}"
            );
        }
        assert!((element(&matrix, 0, 2) - 1.596027).abs() < 1e-3);
        assert!((element(&matrix, 1, 1) - -0.391762).abs() < 1e-3);
        assert!((element(&matrix, 1, 2) - -0.812968).abs() < 1e-3);
        assert!((element(&matrix, 2, 1) - 2.017232).abs() < 1e-3);
        assert_eq!(element(&matrix, 0, 1), 0.0);
        assert_eq!(element(&matrix, 2, 2), 0.0);
    }

    #[test]
    fn bt709_limited_matches_documented_constants() {
        let matrix = yuv_to_rgb_matrix(Matrix::Bt709, false);
        assert!((element(&matrix, 0, 2) - 1.792741).abs() < 1e-3);
        assert!((element(&matrix, 1, 1) - -0.212899).abs() < 1e-3);
        assert!((element(&matrix, 1, 2) - -0.532909).abs() < 1e-3);
        assert!((element(&matrix, 2, 1) - 2.112402).abs() < 1e-3);
    }

    #[test]
    fn full_range_matrices() {
        let matrix = yuv_to_rgb_matrix(Matrix::Bt601, true);
        assert_eq!(element(&matrix, 0, 0), 1.0);
        // 2(1 - KR), exact per the standard's coefficients.
        assert!((element(&matrix, 0, 2) - 2.0 * (1.0 - BT601_KR_KB.0)).abs() < 1e-6);
        assert!((element(&matrix, 2, 1) - 2.0 * (1.0 - BT601_KR_KB.1)).abs() < 1e-6);
        let matrix = yuv_to_rgb_matrix(Matrix::Bt709, true);
        assert!((element(&matrix, 0, 2) - 2.0 * (1.0 - BT709_KR_KB.0)).abs() < 1e-6);
        assert!((element(&matrix, 2, 1) - 2.0 * (1.0 - BT709_KR_KB.1)).abs() < 1e-6);
        assert_eq!(
            yuv_to_rgb_matrix(Matrix::Identity, false),
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn limited_range_anchors_hit_black_and_white() {
        let matrix = yuv_to_rgb_matrix(Matrix::Bt601, false);
        let (luma_anchor, chroma_anchor) = range_offsets(false);
        assert_eq!((luma_anchor, chroma_anchor), (LIMITED_LUMA_BLACK, LIMITED_CHROMA_NEUTRAL));
        // Studio-swing black and white map to 0.0 and 1.0.
        let black = LIMITED_LUMA_BLACK;
        let white = 235.0 / 255.0;
        for (raw_luma, expected) in [(black, 0.0), (white, 1.0)] {
            for row in 0..3 {
                let got = element(&matrix, row, 0) * (raw_luma - luma_anchor);
                assert!((got - expected).abs() < 1e-4, "row {row}: {got} vs {expected}");
            }
        }
    }

    #[test]
    fn quantization_helpers() {
        assert_eq!(range_offsets(true), (0.0, FULL_RANGE_CHROMA_NEUTRAL));
        assert_eq!(depth_norm(8, false), 1.0);
        assert_eq!(depth_norm(MAX_PLANE_DEPTH, true), 1.0);
        assert!((depth_norm(10, true) - R16_STORAGE_MAX / 1023.0).abs() < 1e-4);
        assert!((depth_norm(12, true) - R16_STORAGE_MAX / 4095.0).abs() < 1e-4);
    }

    #[test]
    fn page_uniforms_from_pixels() {
        assert_eq!(PageUniforms::from(&Pixels::Rgba8(Box::default())), RGBA_UNIFORMS);
        let yuv_pixels = Pixels::Yuv(crate::decode::YuvImage {
            depth: 10,
            subsampling: Subsampling::Yuv420,
            full_range: false,
            matrix: Matrix::Bt2020,
            y: Plane::U8(Box::default()), cb: Plane::U8(Box::default()), cr: Plane::U8(Box::default()),
        });
        let uniforms = PageUniforms::from(&yuv_pixels);
        assert_eq!(
            (uniforms.mode, uniforms.subsampling, uniforms.depth, uniforms.full_range, uniforms.matrix),
            (1, Subsampling::Yuv420, 10, false, Matrix::Bt2020)
        );
    }
}
