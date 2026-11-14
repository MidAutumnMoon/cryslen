# Systematic Analysis & Architecture Plan

A top-to-bottom review of `cryslen`’s architecture and code quality, tracing the open problems in `performance-plan.md` and `analysis.md` to their structural roots and applying two governing constraints:

- **Fewest concepts (`too-many-concepts`)**: Minimize the number of parallel states, message translations, and files a reader must hold in mind at once. Prefer deleting a redundant layer over adding a coordinator.
- **Simplest road to the outcome (`all-roads-lead-to-rome`)**: When a requirement or fix looks like it needs new threads, state machines, or cross-process plumbing, step back and ask whether a small change in policy or representation reaches the same observable outcome with a fraction of the mechanism.

---

## Part 1: Root-Cause Review

### 1. Why serving stalls happen: `Still` requests `Thumb`, and `Job` batches roles (`cache/presentation.rs`, `worker.rs`)

The serving issues in `performance-plan.md` (§2 and §3) look like worker scheduling bugs, but their root cause starts in `presentation::request_roles` and the shape of `worker::Job`:

- **A still frame requests `Role::Thumb` even though it can never draw it (`src/cache/presentation.rs:236-241`)**:
  - In `Motion::Still`, `choose_presented` only displays spreads where `has_settled_image()` (`Fit` or `Full`) is true, and `PageTextures::draw` always binds `best_texture` (`Full > Fit > Thumb`).
  - Yet `request_roles` in the `still` branch unconditionally requests `Role::Thumb` alongside `Role::Fit` for both the demand and on-screen spreads.
  - Because `Session::next_request` groups all requested roles for a page into a single multi-role `Job { page, roles, .. }`, every still page turn forces `Session::serve` (`src/worker.rs:405-509`) to produce, save, IPC-transfer, and GPU-upload `Thumb` in the same critical-path batch as `Fit`.
- **Consequence 1 — `Thumb` derivation and disk writes delay `Fit` delivery (`performance-plan.md` §2)**: `Session::serve` derives `Fit`, writes `Fit` to disk, derives `Thumb`, writes `Thumb` to disk, and only at the end calls `publish(Fit)` and `publish(Thumb)`.
- **Consequence 2 — A warm `Fit` + cold `Thumb` decodes the full page (`performance-plan.md` §2)**: Because the still frame asked for `[Fit, Thumb]`, if `Fit` hits the store and `Thumb` misses, the flat decode gate (`roles.contains(Thumb) && thumb.is_none()` at `src/worker.rs:441`) decodes the full native image from scratch—even though line 468 then derives `Thumb` from `view.as_ref().unwrap_or(&native)`.
- **Consequence 3 — Store hits read the encoded file first (`performance-plan.md` §3)**: `Session::serve` calls `source.bytes(page)` at entry (`src/worker.rs:414`) before checking `self.hashes` or `self.store`, paying 3–13 ms of file I/O on a 30 MB page before discovering the role is already in the store.

### 2. Why the child process has too many moving parts (`worker.rs`, `protocol.rs`, `book.rs`)

- **Leftover in-process `Worker` layer**: `Worker` (`Worker::spawn`, `WorkerCommand`, `Update`, `Worker::detach`) was built when decoding ran as a thread inside the UI process. When `cy --worker` moved each book into a child process, `worker::main` was wrapped around `Worker`.
- **Three threads, two internal channels, three event enums**: Inside a process dedicated to one book, `worker::main` blocks on `parent_rx`, translates `ParentMsg` → `WorkerCommand` over an `mpsc::channel` to `cy-decode`, which sends `Update` over a `crossbeam_channel` to `cy-writer`, which translates `Update` → `ChildMsg` (`WireImage::from_decoded`), which the parent’s `cy-book-reader` translates into `BookUpdate`.
- **Fragile message-ordering invariant**: `worker::main` only spawns `Worker` when the first `ParentMsg::SetView` arrives (`src/worker.rs:994-1055`). Any `SetRequests`, `SetMotion`, or `Playback` arriving before `SetView` is silently dropped, forcing callers (`Reading::start`, `tests/child.rs`) to obey an implicit temporal ordering.
- **Head-of-line blocking between requests and store warm-up (`performance-plan.md` §4, `analysis.md` #1)**: Prefetch never delivers images to the parent (`the_prefetch_warms_the_store_without_delivering`)—it only populates `Store`. Yet because prefetch and interactive `SetRequests` run on the same `cy-decode` loop, even a **2 ms cached store hit** has to wait up to cientos of milliseconds behind an in-flight speculative decode.

### 3. Why archives stall on page 0 (`source.rs`, `worker.rs`)

- **Whole-archive barrier (`OnceLock<PageSet>`)**: `Source::open_zip` and `Source::open_sevenzip` spawn `cy-unpack` to decompress every member into a `Vec<PageRead>` and send the finished vector over `page_set_rx` at the end. `worker::run` blocks on `source.wait_ready()` before entering its loop (`src/worker.rs:770`). Page 0 waits for the last page of the archive.
- **Caller-discipline precondition**: `Source::bytes` requires `wait_ready()` to have been called first on archives and returns a misleading `Err(BadIndex)` ("no such page") if called early (`src/source.rs:225`).

### 4. Why `Reading` and `PageTextures` have split state and edge-limit traps (`app.rs`, `cache/presentation.rs`, `render.rs`, `fault.rs`)

- **Four parallel maps for one page’s state in the parent**: A page’s state is split across `reading.facts` (`BTreeMap<usize, PageInfo>`), `reading.faults` (`BTreeMap<usize, PageFault>`), `reading.playback` (`BTreeMap<usize, PagePlayback>`), and `gl.pages` (`HashMap<usize, [Option<PageTexture>; 3]>`). Every frame (`App::frame`, `presentation::build_frame`, `ui::summarize_spread`) must cross-check `facts` against `gl.pages` and `faults`.
- **A `Role::Full` texture-limit refusal poisons the whole page**: `decode::DECODE_EDGE_LIMIT` (16,384), `cache::MAX_EDGE` (65,536), and the GPU’s `max_texture_edge` (`GL_MAX_TEXTURE_SIZE`) are not coordinated. If `Role::Fit` uploads and displays fine, and the user zooms in so `Role::Full` is requested, a `TextureFault` on `Role::Full` in `App::drain_deliveries` (`src/app.rs:606-610`) inserts a page-wide `PageFault` into `reading.faults`—hiding `PageInfo`, halting playback, and blaming a page whose `Fit` texture is already valid on screen.
- **Cross-process error coupling in `fault.rs`**: `TextureFault` (a parent GL error) is included in `fault::PageError` (whose doc states "the error types stay in the child") solely so `App::drain_deliveries` can call `.widen()` to produce a `PageFault`.

### 5. Avoidable copies and per-call thread churn (`decode/`, `cache/mod.rs`)

- **Full-image clone when `page_edge <= FIT_EDGE`**: `Decoded` stores `Box<[u8]>` / `Box<[u16]>`. When a page already fits within `FIT_EDGE`, `cache::derive` calls `page.clone()` (`src/cache/mod.rs:88, 102`), deep-copying the pixel buffer.
- **15 MB heap copy before every store write**: `cache::encode` (`src/cache/mod.rs:689`) allocates `Vec::with_capacity(32 + page.resident_bytes())` and copies all planes into it before `write_atomic` writes the slice to disk—the double-copy noted in `performance-plan.md`. For `Plane::U16`, `cache/mod.rs` loops sample-by-sample instead of using `bytemuck::cast_slice` like `protocol.rs`.
- **Double buffer allocation for >8-bit AVIF (`src/decode/avif.rs:305-330`)**: `copy_plane` copies row bytes into a `Vec<u8>` and then maps 2-byte chunks into a new `Box<[u16]>`.
- **Thread spawn/join per JXL callback (`src/decode/jxl.rs:134-164`, `analysis.md` #5)**: `Pool::run` spawns OS threads via `std::thread::scope` on every `JxlParallelRunner::run` call.

### 6. Split path rules and handshake ceremony (`app.rs`, `book.rs`, `catalog.rs`, `source.rs`)

- **Symlink mismatch between parent and child**: `app::classify` and `catalog::sibling` use `Path::is_file()` / `is_dir()` (following symlinks), while `Source::open_dir` uses `DirEntry::file_type().is_file()` (rejecting symlinks). Passing a symlinked image on the CLI passes `app::classify`, opens the parent directory in the child, and drops the image from the page list.
- **Two-stage handshake pollutes `ChildMsg`**: `PendingBookOpen::finish` accepts `Hello` on one channel and then waits on `child_rx` for `ChildMsg::BookInfo` or `ChildMsg::OpenFailed`. Those two handshake-only variants remain in `ChildMsg` forever and must be guarded against in `book::event_from_message`.

---

## Part 2: Redesign Plan (Balanced for Simplicity)

### 1. [Done] Simplify the Serving Model (`cache/presentation.rs`, `worker.rs`)

Instead of adding a complex multi-role scheduler, remove the work that shouldn't happen and make each work item a single `(page, Role)`:

- **Stop requesting `Role::Thumb` on still frames (`src/cache/presentation.rs:236-249`)**:
  - In `request_roles`, when `still` is true, request `Role::Fit` (if `!resident.has_settled_image()`) and `Role::Full` (if `on_screen` and `upscaled`), never `Role::Thumb`.
  - `Thumb` returns to its true role: requested on demand during `Motion::Skimming`, and warmed into `Store` by background prefetch.
- **Replace multi-role `Job` with single-role steps in `Session` (`src/worker.rs`)**:
  - Each step produces one `(page, Role)` following the natural hierarchy `Full → Fit → Thumb`:
    1. **Check `hashes` + `Store` before `Source::bytes`**: If `self.hashes.get(&page)` has a hash, call `store.load(hash, role)` first; only call `source.bytes(page)` on a store miss (`performance-plan.md` §3).
    2. **Publish before persist**: When serving a requested role, call `publish` as soon as the `Decoded` image is in memory, *before* `store.save` (`performance-plan.md` §2).
    3. **Keep a 1-slot `last_fit: Option<(u128, Decoded)>` memo in `Session`**: When `Thumb` misses the store, derive `Thumb` from `last_fit` (if same hash) or `store.load(hash, Role::Fit)` in ~1.2 ms before falling back to a full page decode (`performance-plan.md` §2). Because prefetch warms `Fit` then `Thumb` as separate loop ticks, the worker checks `rx` for new requests between `Fit` and `Thumb`, and `Thumb` derives immediately from `last_fit`.
- **Eliminate head-of-line blocking on cached turns (`performance-plan.md` §4)**:
  - Flatten `worker::main` + `Worker` so the child starts its session immediately on open (defaulting to page 0, `Normal`, `Forward` so early `SetRequests` are never dropped) and collapse `WorkerCommand` / `Update`.
  - **Two-step path for §4**:
    - *Step A (zero concurrency)*: With `Thumb` removed from still requests, `fast_image_resize` landed, and `IDLE_PRELOAD` tuned above normal reading turn cadence, measure whether head-of-line stalls remain.
    - *Step B (if slow JXL/AVIF decodes still stall cached turns)*: Because prefetch **never** publishes to the parent (it only writes to `Store`, which is already safe for concurrent writers via PID temp locks), run store warm-up on a background thread that only writes to `Store`, while the main child thread serves `SetRequests` (hitting `Store` in ~1.5 ms without waiting on any in-flight decode) and `Playback` directly onto `child_tx`—eliminating `cy-writer` as well.

### 2. [Done] Progressive Archive Slots Without Extra Concepts (`source.rs`, `catalog.rs`, `app.rs`, `book.rs`, `protocol.rs`, `worker.rs`, `ui.rs`)

- **Replace `OnceLock<PageSet>` + `page_set_rx` with per-page slots (`src/source.rs`)**:
  - Give `Backend::Archive` an `Arc<ArchiveSlots>` containing `Mutex<Box<[Option<PageRead>]>>` (one slot per listed page) and a `Condvar`.
  - `cy-unpack` fills each slot as its entry is read and notifies all waiters on the `Condvar` (filling any remaining empty slots with the error verdict on early abort/panic).
  - `Source::bytes(index)` waits on slot `index` until it is filled.
  - **Result**: `Source::wait_ready()` and `page_set_rx` are deleted; `Source::bytes(index)` is total for both directories and archives; and page 0 decodes as soon as entry 0 unpacks instead of waiting for the whole archive.
- **Remove cross-process unpack/fatal states and simplify the wait indicator (`protocol.rs`, `worker.rs`, `book.rs`, `ui.rs`)**:
  - Delete `ChildMsg::{Unpacked, Fatal}`, `ChildEvent::{Unpacked, Fatal}`, `BookUpdate::Unpacked`, `BookView::unpacking`, `Book::mark_unpacked`, and `ui::Wait`; display `" · loading…"` once the wait grace elapses.
- **Unify symlink rules and normalize CLI paths in `catalog.rs`**:
  - Share symlink-following `entry_is_file` across `catalog::sibling` and `Source::open_dir`, plus `normalize_path` so relative and `.` CLI arguments resolve against the working directory while preserving symlinks.
  - Keep `catalog::sibling` synchronous inside `Book::step` so `PendingBookOpen::begin` stays on the event-loop thread (preserving Linux `PR_SET_PDEATHSIG` ownership).

### 3. Low-Copy Pixels & Decoder Reuse (`decode/`, `cache/mod.rs`)

- **Use `Arc<[u8]>` / `Arc<[u16]>` for planes in `Decoded` (`src/decode/mod.rs`)**:
  - Making `Decoded` cheaply cloneable turns identity derivations (`page_edge <= FIT_EDGE` in `cache::derive`) into an O(1) refcount bump instead of a multi-megabyte allocation and memcpy.
- **Write `Store::save` directly from header + plane slices (`src/cache/mod.rs`)**:
  - Write the small header buffer followed by `decode::plane_bytes` directly to the locked temp file instead of concatenating a full-image `Vec<u8>` in `encode()`. Use `bytemuck::cast_slice` / `pod_collect_to_vec` for `Plane::U16` in `cache` just as `protocol.rs` already does.
- **Single-allocation >8-bit plane copy in `decode/avif.rs`**:
  - Copy >8-bit rows directly into `Vec<u16>` instead of `Vec<u8>` → `Box<[u16]>`.
- **Persistent worker pool for `decode/jxl.rs` (`analysis.md` #5)**:
  - Keep a small reusable thread pool in the child process for `JxlParallelRunner` instead of spawning and joining scoped OS threads per callback.

### 4. Pragmatic Resolution & Texture-Limit Handling (`cache/`, `render.rs`, `app.rs`, `fault.rs`)

- **`FIT_EDGE` without resize thrashing (`analysis.md` #2)**:
  - *Trap to avoid*: Keying `Fit` directly by live `viewport` physical pixels would invalidate the on-disk `view` store and re-decode the book every time a tiling compositor resizes the window or toggles fullscreen (`f`).
  - *Simpler road*: Either derive `fit_edge` from a stable ceiling (e.g. monitor/fullscreen physical edge or coarse tier passed in `SetView`) or, if `2560` is already visually indistinguishable from native once bilinearly filtered on the target display, decouple `full_wanted`'s zoom threshold (e.g. only request `Full` when the user actively zooms past fit, `fit.user_moved`, or when `scale > 1.0`) so unzoomed 4K viewports don't double-load `Fit` + `Full` on every page.
- **Guard `Role::Full` by `max_texture_edge` and don't poison working pages (`src/cache/presentation.rs`, `src/app.rs`, `src/fault.rs`)**:
  - Pass `max_texture_edge` into `build_frame` so `full_wanted` never requests `Role::Full` when `page_edge > max_texture_edge`.
  - In `App::drain_deliveries`, only record a `PageFault` on upload failure if `gl.pages` has no resident texture for that page; if `Fit` or `Thumb` is already resident, log and keep the working texture.
  - Remove `TextureFault` from child-side `fault::PageError`.

### 5. Keep the `BookView` Invariant; Simplify the Handshake (`book.rs`, `protocol.rs`, `app.rs`)

- **Keep initial `BookProcess::open` synchronous before `App::new` (revisiting `analysis.md` #7)**:
  - *Why not make initial open async?* Today `Book` guarantees it *always* owns a valid `BookView` (`PageCount >= 1`). Making the initial open asynchronous in `App` would force `BookView` to become `Option<BookView>`, infecting `App`, `Reading`, `presentation`, and `ui` with "no book loaded yet" states. Because `Source::open` only reads directory entries or the archive central directory/header (a few milliseconds; actual archive unpacking is already on `cy-unpack`), keeping initial open synchronous preserves the `BookView` invariant across the whole UI.
- **Fold `Result<BookInfo, String>` into `Hello` (`src/protocol.rs`, `src/book.rs`, `src/worker.rs`)**:
  - Have the child open `Source` first and send `(Result<BookInfo, String>, IpcReceiver<ChildMsg>, IpcSender<ParentMsg>)` in the one-shot `Hello` connection.
  - Delete `ChildMsg::BookInfo` and `ChildMsg::OpenFailed` from `ChildMsg`, collapsing `PendingBookOpen::finish` into a single wait.

### 6. Consolidate Modules to Reduce File-Jumping (`src/`)

Rather than splitting `app.rs` and `render.rs` into many small files, **reduce** the number of files and align boundaries with ownership:

- **Collapse `src/cache/` into flat modules**:
  - `src/cache/prefetch.rs` is 90 lines of pure spread-walk arithmetic over `browse::advance` and `browse::spread_for`; fold it into `browse.rs` (or `worker.rs`).
  - Move `src/cache/presentation.rs` to `src/presentation.rs` (parent-side frame planner) or fold into `browse.rs`.
  - Turn `src/cache/mod.rs` into a flat `src/cache.rs` owning only `derive` and `Store`.
- **Unify parent per-page state in `Reading` (`src/app.rs`)**:
  - Keep `render.rs` as one file (keeping the GLSL shader and the CPU `ColorConversion` math that feeds its uniforms side-by-side).
  - Inside `app.rs`, extract `Reading`’s state transitions (`poll_book_updates`, `drain_deliveries`, `sync_motion`, `sync_playback`) into methods that don't require a live `ActiveEventLoop` so they can be unit-tested directly in `app::tests`.

---

## Summary of Trade-Offs & Decisions

| Area | Over-Engineered / Complex Route | Balanced Route (Chosen) |
|---|---|---|
| `Thumb` blocking `Fit` on still turns (`performance-plan.md` §2) | Multi-stage priority preemption inside `Session::serve` | Stop requesting `Thumb` on `Still` frames; single-role worker steps + 1-slot `last_fit` memo |
| Cached turn blocked by preload (`performance-plan.md` §4) | 3+ threads with shared mutable session state and writer queue | Flatten child to 1–2 threads; prefetch only writes to `Store`, main child thread serves `SetRequests` directly |
| Archive first-page latency (`source.rs`) | Separate random-access ZIP reader vs streaming 7z reader | Per-page `Mutex<Box<[Option<PageRead>]>>` + `Condvar` (`ArchiveSlots`); delete `wait_ready()`, `Unpacked`, and `Fatal` |
| `FIT_EDGE` on 4K displays (`analysis.md` #2) | Dynamic per-viewport `fit_edge` (thrashes disk cache on every window resize) | Stable display-ceiling `fit_edge` or gate `full_wanted` on active zoom (`user_moved` / `scale > 1.0`) |
| Initial book open (`analysis.md` #7) | Async initial open in `App` (forces `Option<BookView>` everywhere) | Keep synchronous initial header open (preserves non-empty `BookView` invariant); fold `BookInfo` into `Hello` |
| Module organization (`src/`) | Split `app.rs` and `render.rs` into 6+ new subfiles | Flatten `src/cache/` directory; keep shaders + color math together in `render.rs` |
