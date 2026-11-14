# Systematic Analysis & Remaining Plan

Governing constraints for future refactors:

- **Fewest concepts**: Minimize parallel states, message translations, and files a reader must hold in mind at once. Prefer deleting a redundant layer over adding a coordinator.
- **Simplest road to the outcome**: Before adding new threads, state machines, or cross-process plumbing, check whether a smaller change in policy or representation reaches the same outcome.

---

## Open Root Causes

### 1. Split parent state (`app.rs`, `cache/presentation.rs`, `render.rs`)

- **Parallel maps for one page's state in the parent**: Page state is split across `Reading::{facts, faults, playback}` and `Gl::pages` (`PageTextures`). Every frame (`App::frame`, `presentation::build_frame`) cross-checks them separately.

### 2. Two-stage handshake pollutes `ChildMsg` (`protocol.rs`, `book.rs`, `worker.rs`)

- `PendingBookOpen::finish` accepts `Hello` on the one-shot channel and then waits on `child_rx` for `ChildMsg::BookInfo` or `ChildMsg::OpenFailed`. Those two handshake-only variants stay in `ChildMsg` forever and must be rejected in `book::event_from_message`.

---

## Remaining Plan

### 1. Keep the `BookView` Invariant; Simplify the Handshake (`book.rs`, `protocol.rs`, `worker.rs`)

- **Keep initial `BookProcess::open` synchronous**: `Book` guarantees a valid non-empty `BookView` (`PageCount >= 1`). Making the initial open asynchronous would force `Option<BookView>` across `App`, `Reading`, `presentation`, and `ui`. Because `Source::open` only reads directory entries or archive headers (`cy-unpack` decompresses asynchronously), keeping initial open synchronous preserves that invariant.
- **Fold `BookInfo` / open error into `Hello`**: Have the child open `Source` before connecting `Hello` (sending `Result<(BookInfo, IpcReceiver<ChildMsg>, IpcSender<ParentMsg>), String>` in the one-shot handshake) and delete `ChildMsg::BookInfo` and `ChildMsg::OpenFailed`, collapsing `PendingBookOpen::finish` into one wait.

### 2. Consolidate Modules & Unify `Reading` (`src/cache/`, `src/app.rs`, `src/render.rs`)

- **Flatten `src/cache/`**:
  - Fold `src/cache/prefetch.rs` (~95 non-test lines over `browse::advance` and `browse::spread_for`) into `src/browse.rs` (or `src/worker.rs`).
  - Move `src/cache/presentation.rs` out to `src/presentation.rs` (parent-side frame planner) or fold into `src/browse.rs`.
  - Turn `src/cache/mod.rs` into a flat `src/cache.rs` owning only `ImageLimits`, `derive`, and `Store`.
- **Keep `src/render.rs` as one file** (keeping GLSL shaders alongside the CPU `ColorConversion` math that feeds their uniforms).
- **Unify parent per-page state & extract `Reading` transitions (`src/app.rs`)**:
  - Collapse `Reading::{facts, faults, playback}` into one per-page state entry (`gl.pages` remains on `Gl` for GL context lifetime).
  - Factor `Reading`'s transitions (`poll_book_updates`, `drain_deliveries`, `sync_motion`, `sync_playback`) off `ActiveEventLoop` so they can be unit-tested in `app::tests`.

---

## Trade-Offs & Decisions

| Area | Over-Engineered Route | Balanced Route |
|---|---|---|
| Initial book open & handshake | Async initial open in `App` (forces `Option<BookView>` everywhere) | Keep synchronous header open; fold `BookInfo` / error into `Hello` |
| Module organization (`src/`) | Split `app.rs` and `render.rs` into many small subfiles | Flatten `src/cache/`, unify `Reading` per-page state, keep shaders + color math in `render.rs` |
