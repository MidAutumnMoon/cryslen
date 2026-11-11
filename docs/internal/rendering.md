# Rendering: Wayland, GL, egui

winit owns the window and events, glutin binds EGL, glow calls GL, egui-winit feeds events, and egui_glow paints egui. Page textures stay outside egui's paint list. The GUI crates are Wayland-only. Role policy and retention are in `docs/internal/cache.md`.

## Frame loop

- Present without blocking; a blocking swap would own the event loop. Notify the window before presenting so Wayland's frame callback paces frames.
- Park the loop in a blocking wait; delayed work arms a deadline and the resume event redraws. Keep arming and firing separate, or a repaint clobbers the tighter deadline.
- egui's repaint callback posts a user event through a proxy created before the loop runs.
- While no seat of the on-screen spread has a texture and none failed, arm the status grace so the wait word can appear.

## GL rules

- The context floor is desktop OpenGL 3.3 core; the EGL config query requires `EGL_OPENGL_BIT`.
- Clip space is bottom-origin; flip in the vertex shader and keep the rest top-origin.
- The page quad is generated from `gl_VertexID`; no vertex buffer or attribute is bound. The empty VAO is the only binding a core profile requires.
- Each texture owns wrap and filter state; sampler objects would leak page filtering onto egui's texture units.
- The frame prologue sets the viewport and premultiplied blend for page quads; egui paints after and sets its own state.
- Snap quad origins to whole pixels; nearest sampling at 1:1 draws uneven rows at a fractional origin.
- Minify with mipmaps and trilinear filtering. Magnify a preview linearly and page-resolution pixels nearest, so zoom stays crisp. Chroma reconstructs bilinearly.
- Dither only when the source has more precision than the 8-bit output: PQ or a plane deeper than 8 bits. An 8-bit sRGB source is not perturbed.
- Set the unpack alignment to 1 for single-channel planes.
- `R16` planes are texture-filterable, so they minify and mipmap like `R8`.
- Refuse a page over the maximum texture size before upload; an oversized upload samples black.
- Delete GL objects while the context lives; the owner drops the egui renderer before the window.

## Quads and YUV

One program handles RGBA and YUV, sRGB and PQ; per-draw constants are folded on the CPU. Chroma samples bilinearly.

A seat rect is in page pixels and a texture's UV spans the page; each draw compares them and logs a mismatch at debug (`debug_assert!` in debug builds).

## Page textures

The cache maps each page to one slot per role, in `Role::ALL` order; a seat draws the largest resident slot.