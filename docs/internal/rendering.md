# Rendering: Wayland, GL, egui

Reference for `src/render.rs`, `src/window.rs`, and the frame loop in `src/app.rs`. The stack is winit + glutin + glow + egui_glow, wired by hand: winit owns the window and events, glutin binds EGL, glow calls GL, egui-winit feeds events, and egui_glow paints egui. Wiring it directly keeps page textures outside egui's paint list and the repaint policy local. `Cargo.toml` sets `default-features = false` and enables `wayland` on the GUI crates, keeping the dependency tree Wayland-only. Artifact roles and retention policy are in `docs/internal/cache.md`.

## Frame loop

- Present with `SwapInterval::DontWait` and `pre_present_notify`; a blocking swap owns the event loop.
- Park in `ControlFlow::Wait`; delayed work arms `WaitUntil`, and `new_events(ResumeTimeReached)` redraws. Keep arm and fire separate, or a repaint clobbers the tighter deadline.
- egui's repaint callback posts a user event through a proxy created before `run_app`.
- While the shown spread waits, arm the status grace so the wait word can appear.

## GL rules

- Clip space is bottom-origin; flip in the vertex shader and keep the rest top-origin.
- `begin_frame` sets the viewport and premultiplied blend for page quads; egui paints after and sets its own state.
- Snap quad origins to whole pixels; NEAREST at 1:1 draws uneven rows at a fractional origin.
- Minify with mipmaps and `LINEAR_MIPMAP_LINEAR`; magnify with NEAREST. Chroma reconstructs bilinearly.
- Set `UNPACK_ALIGNMENT` to 1 for single-channel planes.
- On GLES, R16 samples black under LINEAR; bind it as NEAREST and skip mips for a wide plane.
- ES fragment shaders need `precision highp`; mediump can be fp16.
- Refuse a page over `MAX_TEXTURE_SIZE` before upload; an oversized upload samples black.
- Delete GL objects while the context lives. `Gl` keeps `window` last and calls `egui_glow.destroy()` in `Drop`.

## Quads and YUV

A fixed -1..1 strip is placed by the vertex shader from uniforms, so pan, zoom, and rect changes update uniforms only. One program handles RGBA and YUV, sRGB and PQ. Per-draw constants are folded on the CPU (`page_conversion`, `upload_entry`): range anchors, depth normalization, the KR/KB matrix, the mono chroma stub, and alpha normalization. Chroma samples bilinearly at centered positions; declared co-sited metadata carries a half-texel bias, invisible in practice. Render tests pin black and white at every depth and the preview page-to-texture mapping.

## Uploads

An upload costs one texture per plane plus mipmaps. Decoded artifacts queue in the app and drain under a per-frame budget; a non-empty queue re-arms. `PageTextures::insert` keeps a present slot, adds a new texture, and refuses an oversize page as a `TextureFault`; `retain` applies `app::kept` and owns removal. A seat draws native, then view, then thumb; `clear_all` runs on a book switch. A retained page is a texture hit.