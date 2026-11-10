# Rendering: Wayland, GL, egui

One hand-wired stack: winit owns the window and events, glutin binds EGL, glow calls GL, egui-winit feeds events, and egui_glow paints egui. That wiring keeps page textures outside egui's paint list and the repaint policy local. The GUI crates are built without default features and with Wayland only, keeping the dependency tree Wayland-only. Role policy and retention are in `docs/internal/cache.md`.

## Frame loop

- Present without blocking; a blocking swap would own the event loop. Notify the window before presenting so Wayland's frame callback paces frames.
- Park the loop in a blocking wait; delayed work arms a deadline and the resume event redraws. Keep arming and firing separate, or a repaint clobbers the tighter deadline.
- egui's repaint callback posts a user event through a proxy created before the loop runs.
- While no seat of the on-screen spread has a texture and none failed, arm the status grace so the wait word can appear.

## GL rules

- Clip space is bottom-origin; flip in the vertex shader and keep the rest top-origin.
- The frame prologue sets the viewport and premultiplied blend for page quads; egui paints after and sets its own state.
- Snap quad origins to whole pixels; nearest sampling at 1:1 draws uneven rows at a fractional origin.
- Minify with mipmaps and trilinear filtering; magnify a downscaled representation linearly so it does not paint its pixel grid, and the page's own pixels with nearest so zoom stays crisp. Chroma reconstructs bilinearly.
- Set the unpack alignment to 1 for single-channel planes.
- On GLES, a 16-bit single-channel texture samples black under linear filtering; bind it as nearest and skip mips for a wide plane.
- ES fragment shaders need `highp`; mediump can be fp16.
- Refuse a page over the maximum texture size before upload; an oversized upload samples black.
- Delete GL objects while the context lives; the owner drops the egui renderer before the window.

## Quads and YUV

A fixed -1..1 strip is placed by the vertex shader from uniforms, so pan, zoom, and rect changes update uniforms only. One program handles RGBA and YUV, sRGB and PQ. Per-draw constants are folded on the CPU: range anchors, depth normalization, the KR/KB matrix, the mono chroma stub, and alpha normalization. Chroma samples bilinearly at centered positions; declared co-sited metadata carries a half-texel bias, invisible in practice. Render tests pin black and white at every depth and the preview page-to-texture mapping.

A seat rect is in page pixels and a texture's UV spans the page; every draw checks that the two agree, since a stale rect clamps one edge into a band. The app derives layout and placement from the spread the frame treats as on screen, so the rects and their textures always describe the same spread.

## Uploads

An upload costs one texture per plane plus mipmaps. Decoded images queue in the app and drain under a per-frame budget; a non-empty queue re-arms. Inserting keeps a present entry, adds a new texture, and refuses an oversize page as a texture fault; retention applies the frame's role bounds and owns removal. A seat draws full, then fit, then thumb; a book switch clears all. A retained page is a texture hit.