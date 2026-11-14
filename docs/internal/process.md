# Processes: one child per book

`cy` is the windowed parent; `cy --worker` serves one book over shared wire types.

The parent owns the window, UI, GL, app state, request set, and textures; the child owns the book's source, store, decode, derivation, and archive unpacking. A child crash becomes a book fault.

## Reading order

- Open: fork on the event-loop thread so the child's parent-death signal stays bound to it, then finish the handshake off the loop.
- Archive: list pages immediately, unpack entries into per-page slots in the background, and serve each page as its slot fills. Per-page failures stay page-local; an aborted pass fills empty slots with its error. A directory is ready immediately.
- Switch: kill the old child, start a new open, and adopt its view when it lands; the app keeps the last frame on screen meanwhile.
- Child: foreground loop serving requests and playback directly to the parent IPC channel, paired with a `cy-prefetch` thread that warms the store.
- A page fault is text; a book fault is the process.
- The store, prefetch, and retention are in `docs/internal/cache.md`.

## Playback

- The parent owns display timing and the session lifecycle; the child owns decode and the loop policy.
- One frame request is in flight. `generation` rejects a stopped session's answers.
- A playback failure keeps the page's last raster.

## Invariants

- Page identity is the index in the book the child was spawned with; no message carries a path.
- The app sends the whole request set and uses its texture maps as the authority; the child acknowledges nothing.
- Pixels cross as a wire image (metadata plus one shared-memory block per plane), avoiding serialization; the parent validates extents and uploads directly to GL.
- A child that never connects cannot be woken from `accept(2)`, so the child connects before opening the source and the parent accepts on its own thread under the open timeout.

## Limits

The child has the user's full filesystem access. Every image is copied once into shared memory. Child and parent share stderr. A page request in an archive waits only until its own entry unpacks or fails; switches and quit stop running unpacks and decodes by killing the child.