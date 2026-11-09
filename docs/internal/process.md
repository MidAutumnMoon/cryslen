# Processes: one child per book

Reference for `src/parent.rs`, `src/child.rs`, `src/book.rs`, and
`src/protocol.rs`. One binary, two modes: `cy` is the parent, and
`cy --worker <ipc-server-name> <path>` is the child that serves exactly
one book. The store's format and the plan's policy live in
`docs/internal/cache.md`.

The split follows the two owners of state. The parent owns the window:
winit, egui, GL, `App`, the want set, the texture maps, and `BookView`
(the book's names, count, archive flag, unpacking flag). The child owns
the book: `Source`, `Store`, decode, mint, `worker::Session`, and the
archive unpack thread. The child never initializes winit or GL; the
parent never decodes, mints, or opens the store.

The `cy` binary only initializes logging and calls `cryslen::run(&args)`,
which dispatches to the private `parent` and `child` modules. Only the
wire types and the layers the end-to-end test names stay exported.

| | parent | child |
|---|---|---|
| owns | event loop, GL context, `App`, `BookView`, textures, upload budget | `Source`, `Store`, decode, plan, mint, unpack |
| lifetime | the process | one book; killed at switch and at exit |
| mode | `cy` | `cy --worker` |

One process per book means a switch ends the old book's work by ending
its process. There is no worker generation, no shared worker queue, and
no per-generation routing: the child's receiver belongs to that book
and drops with it. Killing the old child at switch start is also what
removes the switch-time in-flight decode that `analysis.md` item 1
described.

## Bootstrap

1. The parent creates `IpcOneShotServer::<Hello>::new()`. Its name is
   the only rendezvous the child needs.
2. `Command::new(env::current_exe())` with `--worker`, the server name,
   and the page path. stdin is null; stderr is inherited, so child logs
   land on the parent's stderr. The path travels as an `OsString` in
   argv, never as a serde payload, so a non-UTF-8 path survives.
3. The child sets `PDEATHSIG` to `SIGKILL`, exits if `getppid() == 1`
   (the parent died before the signal was armed), creates a
   `(ChildMsg receiver, ParentMsg sender)` pair with `ipc::channel`,
   connects to the server, and sends the pair as the `Hello`.
4. The parent accepts on a short-lived `cy-book-accept` thread, because
   `accept` is a blocking `accept(2)` that killing a child which never
   connected cannot wake. Accept and the first message share one
   `OPEN_TIMEOUT` (30 s) deadline. On timeout the parent first kills and
   reaps the child through the `ForkedChild` guard, because a child that
   connected but never sent leaves the accept blocked on that socket;
   it then connects a dummy sender to the server name and closes it,
   which ends a listener that had no client, joins the thread, and
   reports `BookFault::Timeout`.
5. The child opens the source and sends `BookInfo` or `OpenFailed`. The
   parent reads that first message directly from the channel, builds
   `BookView` from it (deriving the page count from `names` and the
   archive flag from the path), starts the reader thread, and the open
   returns.

A one-shot handshake has nothing to clean up: the endpoint pair is
unique to this spawn, and a failed handshake is a dead process, not a
stale connection.

## Protocol

Three messages down, six up. Both are serde enums over `ipc-channel`;
every payload is owned data.

| direction | message | meaning |
|---|---|---|
| parent → child | `SetView { index, mode, direction }` | retune: anchor page, spread mode, navigation direction |
| | `SetMotion(Motion)` | one bit: turning or still |
| | `SetWants(Vec<Want>)` | the full want set, sent only on change |
| child → parent | `BookInfo` | names, unpacking; ends the open |
| | `OpenFailed(String)` | the source did not open |
| | `Ready { page, kind, image }` | one artifact, pixels in `WireImage` |
| | `Failed { page, fault: PageFault }` | one page failed; `PageFault` is a summary |
| | `Unpacked` | the unpack pass ended |
| | `Fatal(String)` | the child's worker could not start; it exits |

Page identity is the index in the book the child was spawned with, so
no message carries a path. The parent sends wants as a whole set and
the child keeps no acknowledgement; delivery is inferred from the
texture maps exactly as it was in-process.

The child's loop turns `SetView` into
`Plan::build(view::spread_for(index, mode, len), direction, mode, len)`
and retunes the worker; `SetMotion` and `SetWants` update it. The loop
ends when the parent disconnects or kills the child. Worker events go
through a bounded writer queue, which mints `WireImage` from the
decoded planes and sends `ChildMsg`.

The archive pass has no status thread: the worker observes
`Source::unpacking()` in its own poll and emits `worker::Event::Unpacked`
once, which the child maps to `ChildMsg::Unpacked` and the app applies
by clearing the view's unpacking flag.

The parent's reader thread converts on the way in: `Ready` becomes
`worker::Ready { image: Box<Decoded> }` after `WireImage::to_decoded`,
a failed conversion becomes `Failed` with a `PageFault`, and both an
IPC error and `Fatal` become `Died(String)`. It then calls the wake
closure, so a parked loop picks the event up. The UI thread sees
`BookEvent` only, never `ChildMsg`.

## WireImage: pixels through shared memory

A decoded artifact does not fit the serialized messages: a view is
about 16.7 MiB, a 50 MP native up to 191 MiB, and an IPC pipe would
copy that through the kernel and back. `WireImage` carries one
`IpcSharedMemory` block per plane instead.

```rust
struct WireImage {
    width: u32, height: u32, format: Format, transfer: Transfer,
    page_dims: (u32, u32), pixels: WirePixels,
}
enum WirePixels {
    Rgba8(IpcSharedMemory),
    Yuv { depth: u8, subsampling: Subsampling, full_range: bool,
          matrix: Matrix, luma: IpcSharedMemory, cb: IpcSharedMemory,
          cr: IpcSharedMemory, alpha: Option<IpcSharedMemory> },
}
```

`WireImage::from_decoded` allocates a block per plane at `plane_bytes`
and copies the decoded plane into it. `to_decoded` copies each block
back into a `Decoded` built through `Decoded::rgba`/`yuv`, so the
extents are validated by the constructors, and `depth > 8` selects
`Plane::U16` (native endian). Only metadata and handles cross the
channel.

This first implementation copies on both sides. It removes the
serialization, not the memcpy: an artifact costs one copy into shared
memory and one copy back into a `Decoded` before upload. Zero-copy
upload, whether by mapping the segment in GL or by writing into a
buffer the renderer owns, is a later optimization and would change
this contract.

## Lifecycle

- **The initial open is synchronous.** `parent::run` creates the
  `EventLoop` and its proxy first, then calls
  `BookProcess::open(path, wake)`, then computes the landing index and
  builds `App::new(view, process, index, proxy)`. The open still
  blocks startup through spawn, handshake, and `BookInfo`, but the loop
  and the wake path exist from the start; the window appears with a
  book already known.
- **A switch is asynchronous.** `switch_book` derives the sibling path
  from the `BookView`, kills the current child, keeps the old view and
  its GPU textures on screen, and forks the new child with
  `PendingBookOpen::begin` on the loop's thread. A `cy-book-open`
  thread then calls `finish`, which posts `(BookView, BookProcess)` on
  a bounded channel and wakes the loop, so drawing continues and
  nothing polls. Only one open is pending at a time; a switch during a
  pending open is ignored.
- **Events wake the loop.** `PendingBookOpen::finish` hands the
  per-book reader thread a wake closure, which is the proxy's
  `send_event(UserEvent::Wake)`. The thread calls it after every
  `ChildMsg`, its final `Died` included, so a parked loop drains the
  book's events and draws the artifact without a timer poll.
- **Adoption.** On `Ok((view, process))` the app replaces its view and
  process, lands on the first spread, clears sent wants, cached
  artifacts, page errors, the upload queue and error, and the presented
  spread, clears every GPU texture, resets fit and skim, then sends
  `SetView` and a cold want set. On `Err(fault)` the old book stays.
- **Kill.** `BookProcess::kill` sends `SIGKILL`, waits for the child
  (reaping it), and joins the reader thread. `Drop` calls it, so a
  dropped book cannot leak a process. `App::exiting` kills the child
  before GL teardown.
- **PDEATHSIG.** If the parent dies abruptly the child is killed with
  it. Nothing needs flushing: the store is content-addressed and every
  temp file is disposable.

## Failure model

Two layers, split by who owns the failure.

- `PageFault` is one page as text: a `summary` string, so a typed
  `PageError` never crosses the wire. Upload errors convert from
  `PageError`; the child's reports arrive as the same text. Page faults
  are sticky per page in the app's error map.
- `BookFault` is the book: `Spawn` (exec failed), `Handshake` (accept
  failed, a protocol violation, or the stream ended before
  `BookInfo`), `Open` (the child said `OpenFailed`), `Timeout` (no
  accept and no `BookInfo` within 30 s; the child is killed), and
  `Died` (the stream ended after adoption, or the child said `Fatal`).
  A `BookFault` is shown in the status line; it does not replace the
  book on screen.
- A child crash is a `BookFault`, not a page fault. The reader thread
  turns the IPC error into `Died` and wakes the loop; the app takes and
  kills the process, notes the fault, and keeps the last known
  `BookView` and textures. Sends go nowhere (send errors are ignored),
  navigation stops updating, and a switch still works because it
  spawns a new process.
- A panicking decoder is caught inside the child and reported as a
  failed page, so it never becomes a book fault.

## The shared store

Every child of the session opens the same `$TMPDIR/cryslen-cache`
tree, keyed by the xxh3-128 of the encoded page bytes. The key needs no
owner, so a page produced by one book's process is a hit for another's,
and a book reopened after a switch finds every preview it left. Format,
headers, and caps are in `docs/internal/cache.md`.

Two processes can write one namespace at once, so a write goes through
`<hash:032x>.<pid>.tmp` and `rename`: a reader sees a complete entry or
a miss, never a half-written file. The pid in the name is the writer,
because several pids can write the same namespace; `Store::open` drops
a temp file whose pid is dead (checked with `kill(pid, 0)`, ESRCH) or,
when the pid cannot be checked, whose mtime is older than an hour.
Live writers are never touched, and `entries`/`prune` ignore `.tmp`.

The caps are shared, not per process, and pruning is by mtime, so one
child's save can evict an entry another child's plan counts as
produced. That costs one re-mint on the next want; a wrong page cannot
follow, because every key is content.

## Why no actor framework or Tokio

The surface is four and six message variants over two endpoints, with
one reader thread and one writer thread per book. A runtime would be
machinery without a job. Decode is synchronous and CPU-bound and
already parallel inside its decoder (dav1d's threads, jxl's runner), so
Tokio would either run it on a blocking pool that is `spawn_blocking`
with extra layers, or starve the reactor. The channels that exist are
`ipc-channel` across the boundary and bounded `crossbeam-channel`
inside each side, so backpressure is explicit rather than implied.

Supervision is the operating system. A child that hangs, panics, or
dies is a `BookFault` with a process to kill; an actor framework's
supervision tree would duplicate that, and its restarts would run in
one address space where the process boundary has already contained the
failure and released the memory.

## Residual limits

- **Every artifact is copied twice.** The child copies the minted
  planes into shared memory; the parent copies them back into a
  `Decoded` before upload. The transport avoids serialization, not
  memcpy: a view pays about 2 × 16.7 MiB of copy and a 50 MP native
  about 2 × 191 MiB. Removing it needs the upload to read the segment.
- **The child is not sandboxed.** It runs as the same user with the
  same filesystem access. The boundary buys crash isolation, memory
  release at switch, and a responsive UI thread, not containment of
  hostile image data.
- **The initial open still blocks startup.** Spawn, handshake, and
  `BookInfo` run synchronously in `parent::run`, before `App` exists
  and before the loop runs; only switches are asynchronous.
- **A switch during a pending open is dropped.** One open at a time;
  a sibling key press while an open is in flight changes nothing until
  the open lands.
- **`SIGKILL` is the normal stop path.** A killed child can leave a
  `.tmp` file; the dead-pid scan at the next `Store::open` removes it.
- **Child logs are unattributed.** stderr is inherited and both modes
  log under the same lib target, so a child's lines interleave with the
  parent's and neither is tagged with its book. Filter with
  `RUST_LOG=cryslen=debug`.