# Architecture Plan

Constraints:

- **Fewest concepts**: Minimize parallel states, message translations, and files a reader must hold in mind. Prefer deleting a redundant layer over adding a coordinator.
- **Simplest road**: Check whether a smaller change in policy or representation reaches the same outcome before adding threads, state machines, or IPC plumbing.

## Consolidate modules and parent state

- **Flatten `src/cache/`**:
  - Fold `src/cache/prefetch.rs` into `src/browse.rs` or `src/worker.rs`.
  - Move `src/cache/presentation.rs` to `src/presentation.rs` or fold into `src/browse.rs`.
  - Collapse `src/cache/mod.rs` to `src/cache.rs` owning only `ImageLimits`, `derive`, and `Store`.
- **Keep `src/render.rs` whole**: Keep GLSL shaders alongside the CPU `ColorConversion` math that feeds their uniforms.
- **Unify `Reading` per-page state (`src/app.rs`)**:
  - Replace the parallel `Reading::{facts, faults, playback}` maps cross-checked by `App::frame` and `presentation::build_frame` with one per-page state entry (`Gl::pages` remains on `Gl` for GL context lifetime).
  - Factor `Reading`'s transitions (`poll_book_updates`, `drain_deliveries`, `sync_motion`, `sync_playback`) off `App` and `Gl` so they can be unit-tested in `app::tests`.
