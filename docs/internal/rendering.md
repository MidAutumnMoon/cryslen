# Rendering: the Wayland/GL stack and its sharp edges

Reference for anyone touching render.rs or the event loop. The working
stack is winit + glutin + glow + egui_glow with no eframe and no X11.
Knowing why each piece exists prevents rewrites that regress latency.

## Stack layers (what each crate answers)

| crate | question it answers |
|---|---|
| winit | "give me a window and its events" (no drawing) |
| glutin (+ glutin-winit) | "bind an EGL context to that window" |
| glow | "let me call GL functions" |
| egui-winit | "feed winit events into egui" |
| egui_glow | "paint egui's output through GL" |

eframe is those five pre-wired. notan is a parallel game-flavored
packaging of the same layers. Going direct costs the boilerplate in
render.rs and buys page textures outside egui's paint list, zero
framework opinions about repaint, and no hidden X11.

## Feature flags that actually matter

`egui-winit`'s *default* features are `["clipboard", "links",
"winit/default", "x11", "wayland"]`. `winit/default` re-imports X11 and
`clipboard` pulls `arboard` → `x11rb`. To get a Wayland-only build,
every crate needs `default-features = false` plus explicit `wayland`
features (see Cargo.toml). That trim removes a large fraction of the
dependency tree and a third of the binary. If clipboard is ever
wanted, prefer `smithay-clipboard` (Wayland-native) over egui-winit's
arboard bundle.

## Frame pacing (the part that silently breaks)

- Present with `SwapInterval::DontWait` **and** call
  `window.pre_present_notify()` before swapping. A blocking
  eglSwapBuffers owns the event loop. The frame-callback pair is what
  eframe does and keeps input responsive.
- The loop parks in `ControlFlow::Wait` after every frame (zero idle
  CPU). Anything that must happen later arms `WaitUntil`. The redraw
  fires from `new_events(StartCause::ResumeTimeReached)`. Arming and
  firing must be separate events, or a repaint request renders
  immediately and clobbers the tighter deadline it just set.
- egui repaint requests need a bridge:
  `egui_ctx.set_request_repaint_callback` → send a user event. The
  proxy must come from `EventLoop::create_proxy` *before* `run_app`.
  winit 0.30 has no `create_proxy` on `ActiveEventLoop`.
- While a decode is pending, re-arm at the status-grace deadline or
  the "decoding…" flag never appears (the loop is parked).

## GL state gotchas

- GL clip space is bottom-origin, winit coords are top-origin. Do the
  flip in the vertex shader (`1 - 2*px.y/u_win.y`) and keep everything
  else top-origin.
- Set `gl.viewport` before your own draw. egui's paint sets it for
  itself, runs after you, and nothing resets it on resize.
- egui's painter leaves `BLEND` enabled. Disable it before drawing page
  quads or transparent PNGs ghost against stale buffers.
- Snap quad origins to whole pixels. LINEAR sampling shimmers at 1:1
  zoom on fractional offsets.
- Generate mipmaps and use `LINEAR_MIPMAP_LINEAR` for minification.
  A fit view minifies a photo page several-fold, where LINEAR-only
  aliases hard.
- `UNPACK_ALIGNMENT = 1` before uploading single-channel planes. Odd
  row widths desync every row after the first under GL's default
  4-byte alignment.
- `R16` is not filterable on GLES contexts: LINEAR makes the texture
  incomplete (samples black). Fall back to NEAREST without mipmaps
  when `is_embedded && wide`.
- `precision highp` in the ES fragment shader. mediump may be fp16,
  where a large image's coordinate ulp grows to whole texels and
  wrecks the math. ES3 mandates highp support.
- Assert against `MAX_TEXTURE_SIZE` before uploading. GL silently
  rejects oversized `tex_image_2d` and the texture then samples black.
  The bounds checks in shader-land cannot catch it.
- GL objects must be deleted while their context is alive. Keep the
  window/context as the last field of the owning struct (Rust drop
  order) and call `egui_glow.destroy()` explicitly on exit or it logs
  a leak warning.

## Quad strategy

A fixed -1..1 strip whose *vertex shader* places it at the slot's
screen rect from uniforms. Pan/zoom/rect changes never touch vertex
data or re-upload textures. The rasterizer clips exactly and there is
no letterbox overdraw (the app owns the clear). The alternative
(fullscreen quad + fragment bounds-check) works but writes every pixel
and complicates tiling later.

## YUV-in-shader (why AVIF renders without a CPU conversion)

One program handles RGBA and YUV via uniforms: mode (0/1), chroma id
(mono/420/422/444 → chroma UV scaling), depth normalizer, range
offsets, and a mat3. Deriving the matrix on the CPU keeps the shader
trivial:

- KR/KB per standard: BT.601 (0.299, 0.114), BT.709 (0.2126, 0.0722),
  BT.2020 (0.2627, 0.0593). KG = 1 - KR - KB.
- Limited range folds the studio-swing scales into the matrix (luma
  255/219, chroma 255/224) so the subtracted anchors stay the plain
  16/255 (luma black) and 128/255 (chroma neutral). Full range uses
  anchors 0 and 0.5 with an identity scale.
- Sampling divides an R16 texture by 65535. Only `depth` bits are
  significant, so multiply by `65535 / (2^depth - 1)`.
- Mono pages get a 1×1 midpoint chroma texture (128, or `1 << (depth-1)`
  as LE u16). The matrix cancels it to neutral gray.
- Chroma is sampled bilinearly at cosited scaled coordinates: exact for
  444, sub-texel for subsampled. Chroma siting metadata is not applied.
  The bias is invisible in practice.

Unit-test the matrix constants against the published coefficient
tables and that studio-swing black/white map to 0.0/1.0. That catches
a whole class of "slightly wrong colors" regressions without a GPU.

## Upload policy

Texture uploads are expensive (full re-upload plus mipmap generation),
so slots are page-keyed: `set(slot, page, decoded)` no-ops on the same
page id and the app never mirrors GPU state. This is also what makes a
future low-res-then-full-res or animation-frame pipeline cheap. Grow
the identity key instead of adding a mirror array.
