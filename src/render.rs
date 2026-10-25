//! GL rendering: the page quad renderer.
//!
//! GL is used from the main thread only and `Window` makes the GL context
//! current once, so the `# Safety` docs below presuppose a current context.
//!
//! Runtime texture-creation failure panics. Recovery is deliberately
//! not built. An upstream `egui_glow` unwrap can panic at runtime
//! anyway, so recovery could not be total. The failure is also rare on
//! target hardware.
//!
//! Chroma siting (avifChromaSamplePosition) is not applied: the default
//! co-sited/midway assumption biases chroma by at most half a texel,
//! invisible in practice.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

use glow::HasContext;

use crate::decode::{plane_bytes, plane_dims, Decoded, Matrix, Pixels, Plane, Subsampling};
use crate::view;
use crate::view::{Placement, Seat, SpreadLayout};

/// Fixed -1..1 strip. The VS places it at the seat's screen rect from
/// uniforms, so pan/zoom never touch vertex data. Top-origin math lives in
/// the VS, because GL clip space is bottom-origin while winit coords are
/// top-origin.
const VS_CORE: &str = "
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
    }";

/// One shader, both pixel layouts: mode/subsampling/depth/range/matrix arrive
/// as uniforms, so there are no per-variant programs. Chroma is sampled
/// bilinearly at cosited scaled coordinates, exact for 444.
const FS_CORE: &str = "
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
    uniform sampler2D u_tex0; // RGBA or luma
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
    }";

const VS_ES: &str = "
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
    }";

// highp is required: mediump may be fp16, where the ulp at 16k image px is
// 8.0 and wrecks the coordinate math. ES3 mandates highp support.
const FS_ES: &str = "
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
    }";

/// Page-keyed texture cache plus the spread draw. Each entry is an RGBA
/// texture or a YUV plane triple.
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
    /// Luma + Cb + Cr plane textures. Mono binds 1x1 midpoint chroma.
    Yuv {
        luma: <glow::Context as glow::HasContext>::Texture,
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
    pub fn new(gl: Arc<glow::Context>) -> eros::Result<Self> {
        let (vs, fs) = if gl.version().is_embedded { (VS_ES, FS_ES) } else { (VS_CORE, FS_CORE) };
        // SAFETY: the returned names land in Self, whose Drop deletes them;
        // the error path below leaks them into the process-fatal exit.
        let (program, vao, vbo) = unsafe { create_pipeline(&gl, vs, fs) }?;
        #[expect(
            clippy::map_err_ignore,
            reason = "the dropped conversion error says only that the value is out of range, which the message states"
        )]
        // SAFETY: MAX_TEXTURE_SIZE is a pure query with no object names.
        let max_tex = u32::try_from(unsafe { gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE) })
            .map_err(|_| eros::error!("GL MAX_TEXTURE_SIZE is not a positive GLint"))?;
        Ok(Self { gl, program, vao, seats: [None, None], textures: HashMap::new(), vbo, max_tex })
    }

    /// Record that `seat` draws `page`, uploading its textures only on a
    /// cache miss. Binding an unchanged page is a no-op. A bound page is
    /// never evicted, so a held seat keeps drawing.
    ///
    /// # Panics
    ///
    /// Panics if the page exceeds this context's `MAX_TEXTURE_SIZE`. Decode
    /// rejects edges over its own cap first, so only a GPU advertising a
    /// smaller limit can trip this.
    pub fn bind(&mut self, seat: Seat, page: usize, decoded: &Decoded) {
        if *self.seat_slot(seat) == Some(page) {
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
        *self.seat_slot(seat) = Some(page);
        if self.textures.contains_key(&page) {
            return;
        }
        // SAFETY: the edge asserts above keep the page inside this
        // context's texture limit.
        let entry = unsafe { upload_entry(&self.gl, decoded) };
        self.textures.insert(page, entry);
        self.evict_over_budget();
    }

    /// Stop drawing `seat`. The page's textures stay cached.
    pub fn clear(&mut self, seat: Seat) {
        *self.seat_slot(seat) = None;
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
            self.textures.iter().map(|(page, entry)| (*page, entry.cost)),
            self.seats,
            TEXTURE_BUDGET_BYTES,
        );
        self.textures.retain(|page, entry| {
            if victims.contains(page) {
                // SAFETY: the entry leaves the map in this same step, so this is its only deletion.
                unsafe { delete_textures(&self.gl, &entry.textures); }
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
        viewport: view::Viewport,
        layout: &SpreadLayout,
        placement: &Placement,
    ) {
        if size.0 == 0 || size.1 == 0 {
            return;
        }
        // SAFETY: program and vao are Self's fields, live until Drop.
        unsafe { begin_frame(&self.gl, self.program, self.vao, size); }
        // Pages stop at the chrome edge. GL scissor is bottom-origin,
        // window coordinates are top-origin.
        let (viewport_origin, content) = (viewport.origin(), viewport.size());
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "viewport origin and size are whole physical px carried as f32"
        )]
        let (left, top, width, height) =
            (viewport_origin.0 as u32, viewport_origin.1 as u32, content.0 as u32, content.1 as u32);
        // SAFETY: the context is current; this only toggles scissor state.
        unsafe { self.gl.enable(glow::SCISSOR_TEST); }
        // SAFETY: same contract, setting the scissor box.
        unsafe {
            self.gl.scissor(
                gl_int(left),
                gl_int(size.1.saturating_sub(top + height)),
                gl_int(width),
                gl_int(height),
            );
        }
        for (rect, page) in layout.rects.iter().zip(self.seats.iter()) {
            let Some(rect) = rect else { continue };
            let Some(page) = page else { continue };
            let Some(entry) = self.textures.get(page) else { continue };
            // Whole-pixel window-px origin: LINEAR samples shimmer at 1:1
            // zoom on fractional offsets otherwise.
            let screen_origin = (
                viewport_origin.0 + rect.pos.0.mul_add(placement.scale, placement.offset.0).round(),
                viewport_origin.1 + rect.pos.1.mul_add(placement.scale, placement.offset.1).round(),
            );
            let screen_size = (rect.size.0 * placement.scale, rect.size.1 * placement.scale);
            if screen_size.0 <= 0.0 || screen_size.1 <= 0.0 {
                continue;
            }
            // SAFETY: &self pins the entry, so its texture names outlive the call.
            unsafe {
                draw_page(&self.gl, self.program, entry, screen_origin, screen_size, placement.scale);
            }
        }
        // SAFETY: the context is current; restores unclipped raster state.
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
            unsafe { delete_textures(&self.gl, &entry.textures); }
        }
        // SAFETY: the context outlives this drop; these names are deleted once, here.
        unsafe { delete_pipeline(&self.gl, self.program, self.vao, self.vbo); }
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

/// Create the linked program, the vertex buffer, and the VAO wiring.
/// Both shaders are deleted once the link succeeds. Objects left behind
/// by a failed step leak into the process-fatal error path (module policy).
///
/// # Safety
///
/// The context must be current. The returned names are live until deleted.
unsafe fn create_pipeline(
    gl: &glow::Context,
    vs: &str,
    fs: &str,
) -> eros::Result<(
    <glow::Context as HasContext>::Program,
    <glow::Context as HasContext>::VertexArray,
    <glow::Context as HasContext>::Buffer,
)> {
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

    // Sampler units are fixed. Draw rebinds textures only.
    gl.use_program(Some(program));
    for (unit, name) in [(0_i32, "u_tex0"), (1, "u_tex1"), (2, "u_tex2")] {
        gl.uniform_1_i32(gl.get_uniform_location(program, name).as_ref(), unit);
    }

    Ok((program, vao, vbo))
}

/// Own the frame state the page draws depend on. egui's paint leaves BLEND
/// enabled, and nothing resets the viewport after a resize.
///
/// # Safety
///
/// The context must be current. `program` and `vao` are live names.
unsafe fn begin_frame(
    gl: &glow::Context,
    program: <glow::Context as HasContext>::Program,
    vao: <glow::Context as HasContext>::VertexArray,
    size: (u32, u32),
) {
    gl.viewport(0, 0, gl_int(size.0), gl_int(size.1));
    gl.disable(glow::BLEND);
    gl.use_program(Some(program));
    gl.bind_vertex_array(Some(vao));
    let win = view::px_f32(size);
    gl.uniform_2_f32(gl.get_uniform_location(program, "u_win").as_ref(), win.0, win.1);
}

/// Set one seat's uniforms, bind its textures, draw the quad.
///
/// # Safety
///
/// The context must be current. `program` and `entry`'s texture names are
/// live for the duration of the call.
unsafe fn draw_page(
    gl: &glow::Context,
    program: <glow::Context as HasContext>::Program,
    entry: &Entry,
    screen_origin: (f32, f32),
    screen_size: (f32, f32),
    scale: f32,
) {
    let uniform_location = |name: &str| gl.get_uniform_location(program, name);
    let page_uniforms = entry.uniforms;
    let (_, (chroma_width, chroma_height)) =
        plane_dims(entry.size, page_uniforms.subsampling);
    let (luma_anchor, chroma_anchor) = range_offsets(page_uniforms.full_range);
    gl.uniform_2_f32(uniform_location("u_offset").as_ref(), screen_origin.0, screen_origin.1);
    gl.uniform_2_f32(uniform_location("u_size").as_ref(), screen_size.0, screen_size.1);
    gl.uniform_2_f32(uniform_location("u_scale").as_ref(), scale, scale);
    let img = view::px_f32(entry.size);
    gl.uniform_2_f32(uniform_location("u_img").as_ref(), img.0, img.1);
    let csize = view::px_f32((chroma_width, chroma_height));
    gl.uniform_2_f32(uniform_location("u_csize").as_ref(), csize.0, csize.1);
    gl.uniform_1_i32(uniform_location("u_mode").as_ref(), mode_id(page_uniforms.mode));
    gl.uniform_1_i32(uniform_location("u_chroma").as_ref(), chroma_id(page_uniforms.subsampling));
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
            gl.bind_texture(glow::TEXTURE_2D, Some(*rgba_texture));
        }
        Textures::Yuv { luma, cb, cr } => {
            gl.bind_texture(glow::TEXTURE_2D, Some(*luma));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(*cb));
            gl.active_texture(glow::TEXTURE2);
            gl.bind_texture(glow::TEXTURE_2D, Some(*cr));
        }
    }
    gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
}

/// Delete the pipeline objects [`create_pipeline`] returned.
///
/// # Safety
///
/// The context must be current. Each name is live and deleted at most once.
unsafe fn delete_pipeline(
    gl: &glow::Context,
    program: <glow::Context as HasContext>::Program,
    vao: <glow::Context as HasContext>::VertexArray,
    vbo: <glow::Context as HasContext>::Buffer,
) {
    gl.delete_program(program);
    gl.delete_vertex_array(vao);
    gl.delete_buffer(vbo);
}

/// Delete an entry's texture objects.
///
/// # Safety
///
/// The context must be current. The names are live and deleted at most once.
unsafe fn delete_textures(gl: &glow::Context, textures: &Textures) {
    match textures {
        Textures::Rgba(rgba_texture) => gl.delete_texture(*rgba_texture),
        Textures::Yuv { luma, cb, cr } => {
            gl.delete_texture(*luma);
            gl.delete_texture(*cb);
            gl.delete_texture(*cr);
        }
    }
}

#[expect(
    clippy::cast_possible_wrap,
    reason = "GL enums, 16384-capped texture edges, and window px all sit far below i32; this is the GL ABI's u32-enum to i32-parameter seam"
)]
fn gl_int(value: u32) -> i32 {
    value as i32
}

/// Texture with viewer defaults. CLAMP keeps edge texels from bleeding.
/// `filterable` false (GLES R16) means NEAREST and no mips.
///
/// # Safety
///
/// The context must be current. The returned name is live until deleted.
unsafe fn new_texture(gl: &glow::Context, filterable: bool) -> <glow::Context as HasContext>::Texture {
    #[expect(
        clippy::expect_used,
        reason = "texture-creation failure panics by policy"
    )]
    let texture = gl.create_texture().expect("GL texture object");
    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, gl_int(glow::CLAMP_TO_EDGE));
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, gl_int(glow::CLAMP_TO_EDGE));
    let (min, mag) = if filterable {
        // Mipmapped minification: fit view of a 50MP page minifies ~8x;
        // LINEAR-only shimmer.
        (glow::LINEAR_MIPMAP_LINEAR, glow::LINEAR)
    } else {
        (glow::NEAREST, glow::NEAREST)
    };
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, gl_int(min));
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, gl_int(mag));
    texture
}

/// Upload `dims` of RGBA8 and generate its mip chain.
///
/// # Safety
///
/// The context must be current. `data` holds exactly `dims.0 * dims.1 * 4`
/// bytes and each dim is within `MAX_TEXTURE_SIZE`.
unsafe fn upload_rgba(
    gl: &glow::Context,
    data: &[u8],
    dims: (u32, u32),
) -> <glow::Context as HasContext>::Texture {
    let texture = new_texture(gl, true);
    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
    gl.tex_image_2d(
        glow::TEXTURE_2D, 0, gl_int(glow::RGBA8),
        gl_int(dims.0), gl_int(dims.1), 0,
        glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(data)),
    );
    gl.generate_mipmap(glow::TEXTURE_2D);
    texture
}

/// Single-channel plane: R8 for 8-bit, R16 for wider. Mipmaps unless the
/// format is unfilterable on this context.
///
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
) -> <glow::Context as HasContext>::Texture {
    let (internal, ty) = if wide { (glow::R16, glow::UNSIGNED_SHORT) } else { (glow::R8, glow::UNSIGNED_BYTE) };
    let texture = new_texture(gl, filterable);
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

/// Upload one page's textures plus the draw state that goes with them.
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
    let (textures, norm, mips) = match &decoded.pixels {
        Pixels::Rgba8(rgba_bytes) => {
            assert_eq!(rgba_bytes.len(), width as usize * height as usize * 4, "rgba byte length");
            (Textures::Rgba(upload_rgba(gl, rgba_bytes, (width, height))), 1.0, true)
        }
        Pixels::Yuv(yuv) => {
            let ((luma_w, luma_h), (chroma_w, chroma_h)) =
                plane_dims((width, height), yuv.subsampling);
            let wide = matches!(yuv.luma, Plane::U16(_));
            // GL_R16 is not filterable in GLES: LINEAR makes the texture
            // incomplete (samples black).
            let filterable = !(gl.version().is_embedded && wide);
            let expect = |plane: &Plane, want_samples: usize, what: &str| {
                let want_bytes = want_samples * if wide { 2 } else { 1 };
                assert_eq!(plane_bytes(plane).len(), want_bytes, "{what} plane bytes");
            };
            expect(&yuv.luma, luma_w as usize * luma_h as usize, "luma");
            let luma_texture =
                upload_plane(gl, plane_bytes(&yuv.luma), (luma_w, luma_h), wide, filterable);
            let (cb_texture, cr_texture) = if yuv.subsampling == Subsampling::Mono {
                // 1x1 chroma at the range midpoint; the matrix cancels it
                // to neutral gray. Native pairs match plane_bytes' U16 layout.
                #[expect(
                    clippy::host_endian_bytes,
                    reason = "the midpoint pairs with plane_bytes' native-endian U16 layout"
                )]
                let midpoint_bytes: Vec<u8> = if wide {
                    (1_u32 << (yuv.depth.clamp(MIN_PLANE_DEPTH, MAX_PLANE_DEPTH) - 1))
                        .to_ne_bytes()
                        .to_vec()
                } else {
                    MONO_CHROMA_MIDPOINT_8.to_ne_bytes().to_vec()
                };
                (upload_plane(gl, &midpoint_bytes, (1, 1), wide, filterable),
                 upload_plane(gl, &midpoint_bytes, (1, 1), wide, filterable))
            } else {
                expect(&yuv.cb, chroma_w as usize * chroma_h as usize, "cb");
                expect(&yuv.cr, chroma_w as usize * chroma_h as usize, "cr");
                (upload_plane(gl, plane_bytes(&yuv.cb), (chroma_w, chroma_h), wide, filterable),
                 upload_plane(gl, plane_bytes(&yuv.cr), (chroma_w, chroma_h), wide, filterable))
            };
            (Textures::Yuv { luma: luma_texture, cb: cb_texture, cr: cr_texture },
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
            let per_sample = if matches!(yuv.luma, Plane::U16(_)) { 2 } else { 1 };
            let chroma = match yuv.subsampling {
                // Mono uploads a 1x1 midpoint texture per chroma channel.
                Subsampling::Mono => 2 * per_sample,
                Subsampling::Yuv420 | Subsampling::Yuv422 | Subsampling::Yuv444 => {
                    plane_bytes(&yuv.cb).len() + plane_bytes(&yuv.cr).len()
                }
            };
            plane_bytes(&yuv.luma).len() + chroma
        }
    };
    #[expect(
        clippy::integer_division,
        reason = "the mip chain costs 4/3 of the base levels, rounded down for budgeting"
    )]
    if mips { base * 4 / 3 } else { base }
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
    fn from(pixels: &Pixels) -> Self {
        match pixels {
            Pixels::Rgba8(_) => RGBA_UNIFORMS,
            Pixels::Yuv(yuv) => Self {
                mode: Mode::Yuv,
                subsampling: yuv.subsampling,
                depth: yuv.depth,
                full_range: yuv.full_range,
                matrix: yuv.matrix,
            },
        }
    }
}

fn mode_id(mode: Mode) -> i32 {
    match mode {
        Mode::Rgba => 0,
        Mode::Yuv => 1,
    }
}

fn chroma_id(subsampling: Subsampling) -> i32 {
    match subsampling {
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
#[expect(
    clippy::similar_names,
    reason = "the green coefficients are the Cb and Cr terms of the BT matrix derivation"
)]
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
#[expect(clippy::expect_used, reason = "a panic in test code is inconsequential")]
mod tests {
    use super::*;
    use crate::decode::Format;
    use crate::decode::YuvImage;

    /// Row `row`, column `col` of the math matrix (storage is column-major).
    #[expect(
        clippy::indexing_slicing,
        reason = "row and col are literal 0..3 coordinates into the fixed 3x3 matrix under test"
    )]
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
        assert!((element(&matrix, 0, 2) - 1.596_027).abs() < 1e-3);
        assert!((element(&matrix, 1, 1) - -0.391_762).abs() < 1e-3);
        assert!((element(&matrix, 1, 2) - -0.812_968).abs() < 1e-3);
        assert!((element(&matrix, 2, 1) - 2.017_232).abs() < 1e-3);
        assert_eq!(element(&matrix, 0, 1), 0.0);
        assert_eq!(element(&matrix, 2, 2), 0.0);
    }

    #[test]
    fn bt709_limited_matches_documented_constants() {
        let matrix = yuv_to_rgb_matrix(Matrix::Bt709, false);
        assert!((element(&matrix, 0, 2) - 1.792_741).abs() < 1e-3);
        assert!((element(&matrix, 1, 1) - -0.212_899).abs() < 1e-3);
        assert!((element(&matrix, 1, 2) - -0.532_909).abs() < 1e-3);
        assert!((element(&matrix, 2, 1) - 2.112_402).abs() < 1e-3);
    }

    #[test]
    fn full_range_matrices() {
        let bt601 = yuv_to_rgb_matrix(Matrix::Bt601, true);
        assert_eq!(element(&bt601, 0, 0), 1.0);
        // (0, 2) is 2(1 - KR) and (2, 1) is 2(1 - KB). Full range leaves
        // the chroma scale at 1, so BT.601 gives exactly 1.402 and 1.772.
        assert!((element(&bt601, 0, 2) - 1.402).abs() < 1e-6);
        assert!((element(&bt601, 2, 1) - 1.772).abs() < 1e-6);
        let bt709 = yuv_to_rgb_matrix(Matrix::Bt709, true);
        // The same two entries with BT.709's coefficients.
        assert!((element(&bt709, 0, 2) - 1.5748).abs() < 1e-6);
        assert!((element(&bt709, 2, 1) - 1.8556).abs() < 1e-6);
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
            luma: Plane::U8(Box::default()), cb: Plane::U8(Box::default()), cr: Plane::U8(Box::default()),
        });
        let uniforms = PageUniforms::from(&yuv_pixels);
        assert_eq!(
            (uniforms.mode, uniforms.subsampling, uniforms.depth, uniforms.full_range, uniforms.matrix),
            (Mode::Yuv, Subsampling::Yuv420, 10, false, Matrix::Bt2020)
        );
    }

    #[expect(
        clippy::integer_division,
        reason = "mirrors entry_cost_bytes' truncated mip-chain cost"
    )]
    #[test]
    fn rgba_entry_cost_counts_the_mip_chain() {
        let bytes = vec![0_u8; 4 * 4 * 4].into_boxed_slice();
        let decoded = Decoded::rgba(4, 4, Format::Png, bytes).expect("rgba page");
        assert_eq!(entry_cost_bytes(&decoded, true), 64 * 4 / 3);
    }

    #[expect(
        clippy::integer_division,
        reason = "mirrors entry_cost_bytes' truncated mip-chain cost"
    )]
    #[test]
    fn yuv420_entry_cost_counts_the_mip_chain() {
        let yuv = YuvImage {
            depth: 8,
            subsampling: Subsampling::Yuv420,
            full_range: true,
            matrix: Matrix::Bt709,
            luma: Plane::U8(vec![0_u8; 4 * 4].into_boxed_slice()),
            cb: Plane::U8(vec![0_u8; 2 * 2].into_boxed_slice()),
            cr: Plane::U8(vec![0_u8; 2 * 2].into_boxed_slice()),
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
            luma: Plane::U16(vec![0_u16; 4].into_boxed_slice()),
            cb: Plane::U8(Box::default()),
            cr: Plane::U8(Box::default()),
        };
        let decoded = Decoded::yuv(2, 2, Format::Avif, yuv).expect("yuv page");
        assert_eq!(entry_cost_bytes(&decoded, false), 4 * 2 + 2 * 2);
    }

    #[test]
    fn eviction_drops_unbound_pages_farthest_first() {
        let costs = [(3_usize, 10_usize), (19, 10), (10, 10), (20, 10)];
        // 19 sits one page from seat 1's anchor, so 3 is the farther victim.
        assert_eq!(eviction_victims(costs, [Some(10), Some(20)], 35), vec![3]);
        assert_eq!(eviction_victims(costs, [Some(10), Some(20)], 25), vec![3, 19]);
    }

    #[test]
    fn bound_pages_are_never_evicted() {
        let costs = [(10_usize, 10_usize), (11, 10), (12, 10)];
        assert_eq!(eviction_victims(costs, [Some(10), Some(11)], 0), vec![12]);
        assert_eq!(eviction_victims(costs, [Some(10), Some(11)], 30), vec![]);
    }

    #[test]
    fn eviction_ties_break_to_the_higher_page() {
        let costs = [(8_usize, 10_usize), (12, 10), (10, 10)];
        assert_eq!(eviction_victims(costs, [Some(10), None], 25), vec![12]);
    }
}
