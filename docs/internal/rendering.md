# Rendering: Wayland, GL, egui

Wayland-only GUI: `winit` owns the window and events, `glutin` binds EGL, `glow` calls GL, `App` (`src/app.rs`) builds `RawInput` for `egui`, and `egui_glow::Painter` paints the UI. Page textures stay outside egui's paint list. Role budgets and eviction are in `docs/internal/cache.md`.

## Frame loop (`src/app.rs`)

- Two-phase loop: `about_to_wait` reconciles state and requests a redraw only on visible change; `RedrawRequested` draws the current state.
- Set `ControlFlow` only in `about_to_wait` from real future deadlines (never arm `WaitUntil(now)`).
- Present without blocking, notifying the window before swap so Wayland's frame callback paces frames.
- Background threads wake the parked loop via a `Wake` user event created before the loop runs.

## GL rules (`src/render.rs`)

- Floor is desktop OpenGL 3.3 core; the EGL config query requires `EGL_OPENGL_BIT`.
- Clip space is bottom-origin; the vertex shader flips Y so layout stays top-origin.
- Bind wrap and filter state on each texture; sampler objects leak page filtering onto egui's texture units.
- Page quads draw with premultiplied blend before egui, which manages its own GL state.
- Snap quad origins to whole pixels so 1:1 nearest sampling stays even.
- Set unpack alignment to 1 for single-channel planes (odd row widths desync under GL's 4-byte default).
- Refuse textures past `GL_MAX_TEXTURE_SIZE` before upload; oversized uploads sample black.
- Destroy GL objects while the context lives (drop egui and page textures before the window).

## Sampling and slot selection (`src/render.rs`)

- Minify with two-pass circular EWA-Lanczos3; magnify previews linearly and page-resolution pixels nearest; reconstruct chroma bilinearly; dither only PQ or >8-bit sources.
- A seat draws its active playback frame first, otherwise `Fit` while it covers the screen size, or the highest-resolution resident role.
