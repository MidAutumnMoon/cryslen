# Processes: one child per book

`cy` is the windowed parent; `cy --worker` is the child that serves one
book. `src/parent.rs` and `src/child.rs` are the two modes, `src/book.rs`
owns the parent's handle to a child, and `src/protocol.rs` is the wire
vocabulary.

The boundary follows ownership. The parent owns the window, UI, GL,
`App`, the want set, and the texture maps. The child owns its book's
`Source`, `Store`, decode, derivation, and archive unpack. A switch ends
the old book by ending its process, so its queue ends with it. The child
decodes; the UI thread draws. A child crash becomes a book fault.

## Reading order

- `BookProcess::open` / `PendingBookOpen::{begin, finish}` — spawn and
  handshake. `begin` must run on the event-loop thread: the child's
  parent-death signal follows the thread that forked it.
- `App::switch_book` / `App::poll_book_updates` — the asynchronous
  switch and adoption.
- `child::main` — the child's message loop and its single writer.
- `BookEvent` — the only event type the UI thread sees; `ChildMsg` stops
  at the reader thread.
- `fault.rs` — `PageFault` is text, `BookFault` is the process.
- `docs/internal/cache.md` — the store, the plan, and retention.

## Invariants

- Page identity is the index in the book the child was spawned with; no
  message carries a path.
- The app sends the whole want set and uses its texture maps as the
  authority; the child acknowledges nothing.
- Pixels cross as `WireImage`: metadata plus one `IpcSharedMemory` per
  plane. The shared blocks avoid serialization; each plane is copied once
  per direction.
- `accept(2)` on a child that never connected cannot be woken, so
  `finish` accepts on its own thread under the same deadline as the first
  message.
- Killing the old child at switch start ends switch-time decode
  concurrency; `analysis.md` item 1.

## Limits

The child has the user's full filesystem access. Every artifact is copied
twice. Child and parent log to the same stderr target, so a line can come
from either process. The initial open blocks startup; switches are
asynchronous.