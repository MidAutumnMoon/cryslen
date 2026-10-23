//! GL rendering: the page quad renderer. All GL `unsafe` lives here and in
//! window.rs, behind safe functions (the only other unsafe in cryslen is
//! decode.rs's libavif FFI block, behind the `avif` feature). Runtime GL
//! calls (resource creation, swap) panic since context loss leaves nothing to
//! show.
//!
//! A page-keyed texture cache plus the two-seat spread draw, YUV plane
//! textures with shader-side conversion. The app owns what each seat
//! displays and drives the cache with per-seat bind/clear calls.
//! Chroma siting (avifChromaSamplePosition) is not applied: the
//! default co-sited/midway assumption biases chroma by at most half a texel,
//! invisible in practice.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::Arc;

use glow::HasContext;

use crate::decode::{plane_bytes, plane_dims, Decoded, Matrix, Pixels, Plane, Subsampling};
use crate::view::{Placement, SpreadLayout};

/// Fixed -1..1 strip. The VS places it at the seat's screen rect from
/// uniforms, so pan/zoom never touch vertex data. Top-origin math lives in
/// the VS, because GL clip space is bottom-origin while winit coords are
/// top-origin.
const VS_CORE: &str = r#"
    #version 330 core
    in vec2 pos;
    uniform vec2 u_win;    // window size, px
    uniform vec2 u_offset; // seat rect origin, screen px
    uniform vec2 u_size;   // seat rect size, screen px
    out vec2 v_px;
    void main() {
        vec2 px = u_offset + (pos * 0.5 + 0.5) * u_size;
        v_px = px;
        gl_Position = vec4(px.x / u_win.x * 2.0 - 1.0, 1.0 - px.y / u_win.y * 2.0, 0.0, 1.0);
    }"#;

/// One shader, both pixel layouts: mode/subsampling/depth/range/matrix arrive
/// as uniforms, so there are no per-variant programs. Chroma is sampled
/// bilinearly at cosited scaled coordinates, exact for 444.
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
        // (ip = k + 0.5), so ip / u_img already addresses texel centers
        // with no extra half-texel offset.
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

// highp is required: mediump may be fp16, where the ulp at 16k image px is
// 8.0 and wrecks the coordinate math. ES3 mandates highp support.
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

/// Page-keyed texture cache plus the spread draw. Each entry is an RGBA
/// texture or a YUV plane triple. Uploads run on the GL thread.
pub struct PageTextures {
    gl: Arc<glow::Context>,
    program: <glow::Context as glow::HasContext>::Program,
    vao: <glow::Context as glow::HasContext>::VertexArray,
    seats: [Option<usize>; 2],
    textures: HashMap<usize, Entry>,
    vbo: <glow::Context as glow::HasContext>::Buffer,
    max_tex: u32,
}

/// One cached page's GPU state, freed by budget eviction and on drop.
struct Entry {
    textures: Textures,
    size: (u32, u32),
    uniforms: PageUniforms,
    /// Depth-normalize factor for U16-stored planes (1.0 otherwise).
    norm: f32,
    /// Approximate GPU bytes, counted against [`TEXTURE_BUDGET_BYTES`].
    cost: usize,
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

/// Neutral gray chroma for mono pages, in 8-bit storage (128 of 255;
/// the wide-plane equivalent is `1 << (depth - 1)`).
const MONO_CHROMA_MIDPOINT_8: u8 = 128;

const IDENTITY3: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// Uniform set describing an RGBA page.
const RGBA_UNIFORMS: PageUniforms = PageUniforms {
    mode: Mode::Rgba,
    subsampling: Subsampling::Yuv444,
    depth: 8,
    full_range: true,
    matrix: Matrix::Identity,
};

impl PageTextures {
    pub fn new(gl: Arc<glow::Context>) -> PageTextures {
        let (vs, fs) = if gl.version().is_embedded { (VS_ES, FS_ES) } else { (VS_CORE, FS_CORE) };
        let compile = |kind, src| unsafe {
            let shader = gl.create_shader(kind).expect("GL shader object");
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
            program = gl.create_program().expect("GL program object");
            vbo = gl.create_buffer().expect("GL buffer object");
            vao = gl.create_vertex_array().expect("GL vertex array object");

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
            let loc = gl.get_attrib_location(program, "pos").expect("pos attribute in page shader");
            gl.enable_vertex_attrib_array(loc);
            gl.vertex_attrib_pointer_f32(loc, 2, glow::FLOAT, false, 8, 0);

            // Sampler units are fixed. Draw rebinds textures only.
            gl.use_program(Some(program));
            for (unit, name) in ["u_tex0", "u_tex1", "u_tex2"].iter().enumerate() {
                gl.uniform_1_i32(gl.get_uniform_location(program, name).as_ref(), unit as i32);
            }
        }
        let max_tex = unsafe { gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE) } as u32;
        PageTextures { gl, program, vao, seats: [None, None], textures: HashMap::new(), vbo, max_tex }
    }

    /// Record that `seat` draws `page`, uploading its textures only on a
    /// cache miss. Binding an unchanged page is a no-op. A bound page is
    /// never evicted, so a held seat keeps drawing.
    ///
    /// # Panics
    ///
    /// Panics if the page exceeds this context's MAX_TEXTURE_SIZE. Decode
    /// rejects edges over its own cap first, so only a GPU advertising a
    /// smaller limit can trip this.
    pub fn bind(&mut self, seat: usize, page: usize, decoded: &Decoded) {
        if self.seats[seat] == Some(page) {
            return;
        }
        let (width, height) = (decoded.width, decoded.height);
        // An oversized upload is silently rejected and then samples black.
        assert!(width > 0 && height > 0, "decoded page has zero dimension");
        assert!(
            width <= self.max_tex && height <= self.max_tex,
            "image {width}x{height} exceeds GL max texture size {} (tiling not implemented)",
            self.max_tex
        );
        self.seats[seat] = Some(page);
        if self.textures.contains_key(&page) {
            return;
        }
        let entry = upload_entry(&self.gl, decoded);
        self.textures.insert(page, entry);
        self.evict_over_budget();
    }

    /// Stop drawing `seat`. The page's textures stay cached.
    pub fn clear(&mut self, seat: usize) {
        self.seats[seat] = None;
    }

    /// Drop cached pages until the map fits [`TEXTURE_BUDGET_BYTES`].
    fn evict_over_budget(&mut self) {
        let costs: HashMap<usize, usize> =
            self.textures.iter().map(|(page, entry)| (*page, entry.cost)).collect();
        for page in eviction_victims(&costs, self.seats, TEXTURE_BUDGET_BYTES) {
            let entry = self.textures.remove(&page).expect("victim still cached");
            delete_textures(&self.gl, &entry.textures);
        }
    }

    /// Draw the spread laid out by `layout` with `placement`.
    ///
    /// Does not clear. The app owns the framebuffer clear each frame.
    pub fn draw(&self, win: (f32, f32), layout: &SpreadLayout, placement: &Placement) {
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
            for (seat, rect) in layout.rects.iter().enumerate() {
                let Some(rect) = rect else { continue };
                let Some(page) = self.seats.get(seat).copied().flatten() else { continue };
                let Some(entry) = self.textures.get(&page) else { continue };
                // Whole-pixel rect origin: LINEAR samples shimmer at 1:1 zoom
                // on fractional offsets otherwise.
                let screen_origin = (
                    (placement.offset.0 + rect.pos.0 * placement.scale).round(),
                    (placement.offset.1 + rect.pos.1 * placement.scale).round(),
                );
                let screen_size = (rect.size.0 * placement.scale, rect.size.1 * placement.scale);
                if screen_size.0 <= 0.0 || screen_size.1 <= 0.0 {
                    continue;
                }
                let page_uniforms = entry.uniforms;
                let (_, (chroma_width, chroma_height)) =
                    plane_dims(entry.size, page_uniforms.subsampling);
                let (luma_anchor, chroma_anchor) = range_offsets(page_uniforms.full_range);
                gl.uniform_2_f32(uniform_location("u_offset").as_ref(), screen_origin.0, screen_origin.1);
                gl.uniform_2_f32(uniform_location("u_size").as_ref(), screen_size.0, screen_size.1);
                gl.uniform_2_f32(uniform_location("u_scale").as_ref(), placement.scale, placement.scale);
                gl.uniform_2_f32(
                    uniform_location("u_img").as_ref(),
                    entry.size.0 as f32,
                    entry.size.1 as f32,
                );
                gl.uniform_2_f32(
                    uniform_location("u_csize").as_ref(),
                    chroma_width as f32,
                    chroma_height as f32,
                );
                gl.uniform_1_i32(
                    uniform_location("u_mode").as_ref(),
                    mode_id(page_uniforms.mode) as i32,
                );
                gl.uniform_1_i32(
                    uniform_location("u_chroma").as_ref(),
                    chroma_id(page_uniforms.subsampling) as i32,
                );
                gl.uniform_1_f32(uniform_location("u_norm").as_ref(), entry.norm);
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
                match &entry.textures {
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

impl Drop for PageTextures {
    fn drop(&mut self) {
        // GL objects must be deleted while their context is alive. The
        // window that owns the context is dropped after this struct.
        for (_, entry) in self.textures.drain() {
            delete_textures(&self.gl, &entry.textures);
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

/// Texture with viewer defaults. CLAMP keeps edge texels from bleeding.
/// `filterable` false (GLES R16) means NEAREST and no mips.
fn new_texture(gl: &glow::Context, filterable: bool) -> <glow::Context as HasContext>::Texture {
    let texture = unsafe { gl.create_texture().expect("GL texture object") };
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as _);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as _);
        let (min, mag) = if filterable {
            // Mipmapped minification: fit view of a 50MP page minifies ~8x;
            // LINEAR-only shimmer.
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

/// Upload one page's textures plus the draw state that goes with them.
fn upload_entry(gl: &glow::Context, decoded: &Decoded) -> Entry {
    let (width, height) = (decoded.width, decoded.height);
    let (textures, norm, mips) = match &decoded.pixels {
        Pixels::Rgba8(rgba_bytes) => {
            assert_eq!(rgba_bytes.len(), width as usize * height as usize * 4, "rgba byte length");
            (Textures::Rgba(upload_rgba(gl, rgba_bytes, (width, height))), 1.0, true)
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
            (Textures::Yuv { y: luma_texture, cb: cb_texture, cr: cr_texture },
             depth_norm(yuv.depth, wide),
             filterable)
        }
    };
    Entry {
        textures,
        size: (width, height),
        norm,
        uniforms: PageUniforms::from(&decoded.pixels),
        cost: entry_cost_bytes(decoded, mips),
    }
}

/// Cache budget: approximate texture bytes the cache may hold.
const TEXTURE_BUDGET_BYTES: usize = 3 * 1024 * 1024 * 1024;

/// Approximate texture bytes one page occupies: plane bytes, plus the mip
/// chain (`4/3` of the base levels) where mips are generated.
fn entry_cost_bytes(decoded: &Decoded, mips: bool) -> usize {
    let base = match &decoded.pixels {
        Pixels::Rgba8(rgba) => rgba.len(),
        Pixels::Yuv(yuv) => {
            let per_sample = if matches!(yuv.y, Plane::U16(_)) { 2 } else { 1 };
            let chroma = match yuv.subsampling {
                // Mono uploads a 1x1 midpoint texture per chroma channel.
                Subsampling::Mono => 2 * per_sample,
                _ => plane_bytes(&yuv.cb).len() + plane_bytes(&yuv.cr).len(),
            };
            plane_bytes(&yuv.y).len() + chroma
        }
    };
    if mips { base * 4 / 3 } else { base }
}

/// Pages to drop so the cached costs fit `budget` after a fresh insert.
/// Candidates are unbound pages, farthest from the nearest bound page
/// first, ties by higher page id. Bound pages never go, so the survivors
/// can still sit above the budget until the next insert.
fn eviction_victims(
    costs: &HashMap<usize, usize>,
    bound: [Option<usize>; 2],
    budget: usize,
) -> Vec<usize> {
    let mut remaining: usize = costs.values().sum();
    let mut candidates: Vec<usize> = costs
        .keys()
        .copied()
        .filter(|page| !bound.contains(&Some(*page)))
        .collect();
    let distance = |page: usize| {
        bound.iter().flatten().map(|b| page.abs_diff(*b)).min().unwrap_or(usize::MAX)
    };
    candidates.sort_unstable_by_key(|page| Reverse((distance(*page), *page)));
    let mut victims = Vec::new();
    for page in candidates {
        if remaining <= budget {
            break;
        }
        remaining -= costs[&page];
        victims.push(page);
    }
    victims
}

/// Pixel layout of the cached page: interleaved RGBA or planar YUV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Rgba,
    Yuv,
}

/// How the shader should read a seat's texture set, derived CPU-side.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PageUniforms {
    mode: Mode,
    subsampling: Subsampling,
    depth: u8,
    full_range: bool,
    matrix: Matrix,
}

impl From<&Pixels> for PageUniforms {
    fn from(p: &Pixels) -> Self {
        match p {
            Pixels::Rgba8(_) => RGBA_UNIFORMS,
            Pixels::Yuv(yuv) => PageUniforms {
                mode: Mode::Yuv,
                subsampling: yuv.subsampling,
                depth: yuv.depth,
                full_range: yuv.full_range,
                matrix: yuv.matrix,
            },
        }
    }
}

/// Shader mode id: 0 = RGBA, 1 = YUV.
fn mode_id(mode: Mode) -> u32 {
    match mode {
        Mode::Rgba => 0,
        Mode::Yuv => 1,
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

/// Luma coefficients (KR, KB) of each conversion standard. KG = 1 - KR - KB.
const BT601_KR_KB: (f32, f32) = (0.299, 0.114);
const BT709_KR_KB: (f32, f32) = (0.2126, 0.0722);
const BT2020_KR_KB: (f32, f32) = (0.2627, 0.0593);

/// YUV→RGB conversion for `rgb = M · (yuv * norm - offsets)`, stored
/// column-major for upload with transpose = false. Derived from the exact
/// KR/KB of each standard. Limited range folds in the studio-swing scales
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
    use crate::decode::Format;
    use crate::decode::YuvImage;

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
        let yuv_pixels = Pixels::Yuv(YuvImage {
            depth: 10,
            subsampling: Subsampling::Yuv420,
            full_range: false,
            matrix: Matrix::Bt2020,
            y: Plane::U8(Box::default()), cb: Plane::U8(Box::default()), cr: Plane::U8(Box::default()),
        });
        let uniforms = PageUniforms::from(&yuv_pixels);
        assert_eq!(
            (uniforms.mode, uniforms.subsampling, uniforms.depth, uniforms.full_range, uniforms.matrix),
            (Mode::Yuv, Subsampling::Yuv420, 10, false, Matrix::Bt2020)
        );
    }

    #[test]
    fn rgba_entry_cost_counts_the_mip_chain() {
        let bytes = vec![0u8; 4 * 4 * 4].into_boxed_slice();
        let decoded = Decoded::rgba(4, 4, Format::Png, bytes).expect("rgba page");
        assert_eq!(entry_cost_bytes(&decoded, true), 64 * 4 / 3);
    }

    #[test]
    fn yuv420_entry_cost_counts_the_mip_chain() {
        let yuv = YuvImage {
            depth: 8,
            subsampling: Subsampling::Yuv420,
            full_range: true,
            matrix: Matrix::Bt709,
            y: Plane::U8(vec![0u8; 4 * 4].into_boxed_slice()),
            cb: Plane::U8(vec![0u8; 2 * 2].into_boxed_slice()),
            cr: Plane::U8(vec![0u8; 2 * 2].into_boxed_slice()),
        };
        let decoded = Decoded::yuv(4, 4, Format::Avif, yuv).expect("yuv page");
        assert_eq!(entry_cost_bytes(&decoded, true), (16 + 4 + 4) * 4 / 3);
    }

    #[test]
    fn r16_gles_entry_cost_skips_mips() {
        let yuv = YuvImage {
            depth: 10,
            subsampling: Subsampling::Mono,
            full_range: true,
            matrix: Matrix::Identity,
            y: Plane::U16(vec![0u16; 4].into_boxed_slice()),
            cb: Plane::U8(Box::default()),
            cr: Plane::U8(Box::default()),
        };
        let decoded = Decoded::yuv(2, 2, Format::Avif, yuv).expect("yuv page");
        assert_eq!(entry_cost_bytes(&decoded, false), 4 * 2 + 2 * 2);
    }

    #[test]
    fn eviction_drops_unbound_pages_farthest_first() {
        let costs = HashMap::from([(3usize, 10usize), (19, 10), (10, 10), (20, 10)]);
        // 19 sits one page from seat 1's anchor, so 3 is the farther victim.
        assert_eq!(eviction_victims(&costs, [Some(10), Some(20)], 35), vec![3]);
        assert_eq!(eviction_victims(&costs, [Some(10), Some(20)], 25), vec![3, 19]);
    }

    #[test]
    fn bound_pages_are_never_evicted() {
        let costs = HashMap::from([(10usize, 10usize), (11, 10), (12, 10)]);
        assert_eq!(eviction_victims(&costs, [Some(10), Some(11)], 0), vec![12]);
        assert_eq!(eviction_victims(&costs, [Some(10), Some(11)], 30), vec![]);
    }

    #[test]
    fn eviction_ties_break_to_the_higher_page() {
        let costs = HashMap::from([(8usize, 10usize), (12, 10), (10, 10)]);
        assert_eq!(eviction_victims(&costs, [Some(10), None], 25), vec![12]);
    }
}
