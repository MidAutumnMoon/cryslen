# Rendering: Wayland, GL, egui

Wayland-only GUI: `winit` owns the window and events, `glutin` binds EGL, `glow` calls GL, `Window` (`src/render/window.rs`) hosts `egui_glow::Painter` + `PageRenderer` (`src/render/textures.rs`), and `App` (`src/app/run.rs`) emits layout-decoupled `DrawQuad`s (`src/app/layout.rs`). Role budgets and eviction are in `docs/internal/cache.md`.

## Frame loop (`src/app/run.rs`)

- Two-phase loop: `about_to_wait` reconciles book/loader/playback state and requests a redraw only on visible change; `RedrawRequested` runs one unified `draw_frame`.
- Set `ControlFlow` only in `about_to_wait` from real future deadlines (never arm `WaitUntil(now)`).
- Present without blocking (`Window::present`), notifying the window before swap (`pre_present_notify`) so Wayland's frame callback paces frames.
- Background threads wake the parked loop via a `Wake` user event created before the loop runs.

## GL rules (`src/render/window.rs`, `src/render/program.rs`, `src/render/textures.rs`)

- Floor is desktop OpenGL 3.3 core; the EGL config query requires `EGL_OPENGL_BIT`.
- Clip space is bottom-origin; the vertex shader flips Y so layout stays top-origin.
- Bind wrap and filter state on each texture; share one immutable pair of 1x1 stub textures (`StubTextures`) across all YUV uploads rather than allocating per-page stubs.
- Page quads draw with premultiplied blend before egui, which manages its own GL state.
- Snap quad origins to whole pixels so 1:1 nearest sampling stays even; linear magnification applies only when upscaling thumbnails (`role == Role::Thumb` below native size).
- Set unpack alignment to 1 for single-channel planes and clamp textures to `glow::MAX_TEXTURE_SIZE`.
- Destroy `Egui` and `PageRenderer` before the EGL context/surface in strict `Window` field drop order.

## Sampling and slot selection (`src/render/ewa.rs`, `src/render/cache.rs`)

- Minify with two-pass circular EWA-Lanczos3 (`Resampler`, mono or color passes); magnify previews linearly and page-resolution pixels nearest; reconstruct chroma bilinearly; dither only PQ or >8-bit sources.
- A seat draws its active playback frame first, otherwise `Fit` while it covers the screen size, or the highest-resolution resident role in `SlotTable`.
