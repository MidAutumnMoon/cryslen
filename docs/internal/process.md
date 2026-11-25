# Processes: one child per book

`cy` is the windowed parent; `cy --worker` serves one book over `src/protocol.rs`.

The parent owns the window, UI, GL, app state, VRAM textures, and the per-book thumb pack; the child owns the book source, archive unpacking, decode, and derivation. A child crash becomes a book fault.

## Lifecycle

- Open (`src/book.rs`, `src/worker.rs`): fork on the event-loop thread so Linux's parent-death signal stays bound to a long-lived thread. Because `accept(2)` cannot unblock if the child exits early, the child connects before opening the source and the parent accepts on a helper thread under a timeout.
- Archive (`src/source.rs`): list pages up front, unpack into per-page slots in the background, and serve each page as its slot fills. Per-page failures stay page-local; an aborted pass fills remaining empty slots with its error. Directories are ready immediately.
- Switch (`src/app.rs`): kill the old child, start the new open, and keep the previous frame on screen until the new book adopts.
- Worker (`src/worker.rs`): runs the decode pool alongside the IPC and playback loop (scheduling and caches in `docs/internal/cache.md`).

## Playback

- Parent (`src/reading.rs`, `src/animation.rs`) owns display timing and session lifecycle; child (`src/worker.rs`) owns decode and loop policy.
- One frame request is in flight per session; a monotonic generation rejects stale frames. A playback failure keeps the page's last raster.

## Wire invariants

- Page identity is the spawn-time book index; messages carry no paths.
- The parent sends the whole `WorkPlan` and is the authority for resident state; the child sends only deliveries and failures.
- Full and fit planes cross via shared memory per plane (`src/protocol.rs`); the parent validates extents before GL upload.

## Trust boundary and limits

The child runs with the user's filesystem access and shares stderr with the parent. Each transferred image copies once into shared memory. Switches and quit cancel running unpacks and decodes by killing the child.
