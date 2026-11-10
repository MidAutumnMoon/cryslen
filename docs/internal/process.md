# Processes: one child per book

`cy` is the windowed parent; `cy --worker` is the child that serves one book. `src/parent.rs` and `src/child.rs` are the two modes, `src/book.rs` owns the parent's handle to a child, and `src/protocol.rs` is the wire vocabulary.

The boundary follows ownership. The parent owns the window, UI, GL, `App`, the request set, and the texture maps. The child owns its book's `Source`, `Store`, decode, derivation, and archive unpack. A switch kills the old book's process, so its queue dies with it. The child decodes; the UI thread draws. A child crash becomes a book fault.

## Reading order

- `BookProcess::open` / `PendingBookOpen::{begin, finish}` — spawn and handshake. `begin` must run on the event-loop thread: the child's parent-death signal follows the thread that forked it.
- `Book::step` / `App::poll_book_updates` — the asynchronous switch and adoption. `Book` owns the lifecycle; `App` keeps the frame.
- `child::main` — the child's message loop and its single writer.
- `BookUpdate` — the only update type the UI thread sees; `ChildMsg` stops at the reader thread.
- `fault.rs` — `PageFault` is text, `BookFault` is the process.
- `docs/internal/cache.md` — the store, the prefetch, and retention.

## Invariants

- Page identity is the index in the book the child was spawned with; no message carries a path.
- The app sends the whole request set and uses its texture maps as the authority; the child acknowledges nothing.
- Pixels cross as `WireImage`: metadata plus one `IpcSharedMemory` per plane. The shared blocks avoid serialization; each plane is copied once per direction.
- `accept(2)` on a child that never connected cannot be woken, so `finish` accepts on its own thread under the same deadline as the first message.
- Killing the old child at switch start ends switch-time decode concurrency; see the limits in `analysis.md`.

## Limits

The child has the user's full filesystem access. Every image is copied twice. Child and parent log to the same stderr target, so a line can come from either process. The initial open blocks startup; switches are asynchronous.