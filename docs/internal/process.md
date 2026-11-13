# Processes: one child per book

`cy` is the windowed parent; `cy --worker` is the child that serves one book. The parent mode runs the windowed app and owns the parent's handle to a child; the child mode runs one book's serving loop; both share one wire vocabulary.

The boundary follows ownership. The parent owns the window, UI, GL, the app state, the request set, and the texture maps. The child owns its book's source, store, decode, derivation, and archive materialization. A switch kills the old book's process, so its queue dies with it. The child decodes; the UI thread draws. A child crash becomes a book fault.

## Reading order

- Open: spawn, then handshake. The handshake must start on the event-loop thread, because the child's parent-death signal follows the thread that forked it.
- Archive: list pages immediately, materialize in one background pass, then serve. Per-page failures stay page-local; a failed pass answers the rest. A directory is ready immediately.
- Switch: kill the old child, open the new one off the event loop, and adopt its view when it lands. The book owns the lifecycle; the app keeps the last frame on screen meanwhile.
- Child: foreground loop serving requests and playback directly to the parent IPC channel, paired with a `cy-prefetch` thread that warms the store.
- Only one update type reaches the UI thread; wire messages stop at the reader thread.
- A page fault is text; a book fault is the process.
- The store, prefetch, and retention are in `docs/internal/cache.md`.

## Playback

- The parent owns display timing and the session lifecycle; the child owns decode and the loop policy.
- One frame request is in flight. `generation` rejects a stopped session's answers.
- A playback failure keeps the page's last raster.

## Invariants

- Page identity is the index in the book the child was spawned with; no message carries a path.
- The app sends the whole request set and uses its texture maps as the authority; the child acknowledges nothing.
- Pixels cross as a wire image: metadata plus one shared-memory block per plane. The shared blocks avoid serialization; each plane is copied once per direction.
- A child that never connects cannot be woken from the wait, so the handshake accepts on its own thread under the same deadline as the first message.
- Killing the old child at switch start ends switch-time decode concurrency; see the limits in `analysis.md`.

## Limits

The child has the user's full filesystem access. Every image is copied twice. Child and parent log to the same stderr target, so a line can come from either process. The initial open blocks startup; switches are asynchronous. An archive's first page waits for the whole materialization; while that pass runs the worker is parked, so a switch or quit kills the child instead of unwinding the pass.