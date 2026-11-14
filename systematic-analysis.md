# Systematic Analysis & Remaining Plan

A structural review of `cryslen`, applying two governing constraints:

- **Fewest concepts**: Minimize parallel states, message translations, and files a reader must hold in mind at once. Prefer deleting a redundant layer over adding a coordinator.
- **Simplest road to the outcome**: Before adding new threads, state machines, or cross-process plumbing, check whether a smaller change in policy or representation reaches the same outcome.

Serving-model simplification (`lmtuqoxo`), progressive archive slots (`kssltzxt`), and low-copy pixel paths (`nxypznyr`) are landed. Three structural items remain.

---

## Part 1: Open Root Causes

### 1. Edge-limit traps and split parent state (`app.rs`, `cache/presentation.rs`, `render.rs`, `fault.rs`)

- **Four parallel maps for one page's state in the parent**: Page state is split across `Reading::{facts, faults, playback}` and `Gl::pages` (`PageTextures`). Every frame (`App::frame`, `presentation::build_frame`, `ui::summarize_spread`) cross-checks them separately.
- **A `Role::Full` texture-limit refusal poisons the whole page**: `decode::DECODE_EDGE_LIMIT` (16,384), `cache::MAX_EDGE` (65,536), and GPU `max_texture_edge` (`GL_MAX_TEXTURE_SIZE`) are uncoordinated. If `Role::Fit` displays and zoom triggers `full_wanted` for `Role::Full`, a `TextureFault` in `App::drain_deliveries` inserts a page-wide `PageFault` into `reading.faults`—hiding `facts` in `ui::summarize_spread`, stopping `sync_playback`, and blaming a page whose `Fit` texture is already valid.
- **Cross-process error coupling in `fault.rs`**: `TextureFault` (a parent GL error) lives inside `fault::PageError` (whose doc states "the error types stay in the child") solely so `App::drain_deliveries` can call `.widen()` into `PageFault`.

### 2. Two-stage handshake pollutes `ChildMsg` (`protocol.rs`, `book.rs`, `worker.rs`)

- `PendingBookOpen::finish` accepts `Hello` on the one-shot channel and then waits on `child_rx` for `ChildMsg::BookInfo` or `ChildMsg::OpenFailed`. Those two handshake-only variants stay in `ChildMsg` forever and must be rejected in `book::event_from_message`.

---

## Part 2: Remaining Plan

### 1. Resolution & Texture-Limit Handling (`cache/presentation.rs`, `render.rs`, `app.rs`, `fault.rs`)

- **`FIT_EDGE` without resize thrashing (`analysis.md`)**:
  - *Trap to avoid*: Keying `Fit` directly by live `viewport` physical pixels would invalidate the on-disk `view` store and re-decode the book whenever a tiling compositor resizes the window or toggles fullscreen (`f`).
  - *Balanced fix*: Derive `fit_edge` from a stable ceiling (e.g., monitor/fullscreen physical edge or coarse tier in `SetView`), or decouple `full_wanted` on active zoom (`Fit`'s `user_moved` / zoomed past fit) so unzoomed 4K+ viewports do not double-load `Fit` + `Full` on every page.
- **Guard `Role::Full` by `max_texture_edge` and isolate upload faults**:
  - Pass `max_texture_edge` from `PageTextures` into `presentation::build_frame` so `full_wanted` never requests `Role::Full` when `page_dims` exceed `max_texture_edge`.
  - In `App::drain_deliveries`, record a `PageFault` on upload failure only when `gl.pages` has no resident texture for that page; if `Fit` or `Thumb` is already resident, log and keep the working texture.
  - Remove `TextureFault` from child-side `fault::PageError`.

### 2. Keep the `BookView` Invariant; Simplify the Handshake (`book.rs`, `protocol.rs`, `worker.rs`)

- **Keep initial `BookProcess::open` synchronous**: `Book` guarantees a valid non-empty `BookView` (`PageCount >= 1`). Making the initial open asynchronous would force `Option<BookView>` across `App`, `Reading`, `presentation`, and `ui`. Because `Source::open` only reads directory entries or archive headers (`cy-unpack` decompresses asynchronously), keeping initial open synchronous preserves that invariant.
- **Fold `BookInfo` / open error into `Hello`**: Have the child open `Source` before connecting `Hello` (sending `Result<(BookInfo, IpcReceiver<ChildMsg>, IpcSender<ParentMsg>), String>` in the one-shot handshake) and delete `ChildMsg::BookInfo` and `ChildMsg::OpenFailed`, collapsing `PendingBookOpen::finish` into one wait.

### 3. Consolidate Modules & Unify `Reading` (`src/cache/`, `src/app.rs`, `src/render.rs`)

- **Flatten `src/cache/`**:
  - Fold `src/cache/prefetch.rs` (~90 non-test lines over `browse::advance` and `browse::spread_for`) into `src/browse.rs` (or `src/worker.rs`).
  - Move `src/cache/presentation.rs` out to `src/presentation.rs` (parent-side frame planner) or fold into `src/browse.rs`.
  - Turn `src/cache/mod.rs` into a flat `src/cache.rs` owning only `derive` and `Store`.
- **Keep `src/render.rs` as one file** (keeping GLSL shaders alongside the CPU `ColorConversion` math that feeds their uniforms).
- **Unify parent per-page state & extract `Reading` transitions (`src/app.rs`)**:
  - Collapse `Reading::facts`, `faults`, and `playback` into one per-page state entry (`gl.pages` remains on `Gl` for GL context lifetime).
  - Factor `Reading`'s transitions (`poll_book_updates`, `drain_deliveries`, `sync_motion`, `sync_playback`) off `ActiveEventLoop` so they can be unit-tested in `app::tests`.

---

## Summary of Trade-Offs & Decisions

| Area | Over-Engineered Route | Balanced Route |
|---|---|---|
| `FIT_EDGE` on 4K+ displays | Dynamic per-viewport `fit_edge` (thrashes disk cache on resize) | Stable display-ceiling `fit_edge` or gate `full_wanted` on active zoom |
| Initial book open & handshake | Async initial open in `App` (forces `Option<BookView>` everywhere) | Keep synchronous header open; fold `BookInfo` / error into `Hello` |
| Module organization (`src/`) | Split `app.rs` and `render.rs` into many small subfiles | Flatten `src/cache/`, unify `Reading` per-page state, keep shaders + color math in `render.rs` |
