# Processes: one child per book

Reference for `src/parent.rs`, `src/child.rs`, `src/book.rs`, and
`src/protocol.rs`. One binary, two modes: `cy` is the parent, and
`cy --worker <ipc-server-name> <path>` is the child that serves one
book. The store and the plan are described in `docs/internal/cache.md`.

The parent owns the window: winit, egui, GL, `App`, the want set, the
texture maps, and `BookView`. The child owns the book: `Source`, `Store`,
decode, mint, `worker::Session`, and the archive unpack thread. The child
never initializes winit or GL; the parent never decodes, mints, or opens
the store.

| | parent | child |
|---|---|---|
| owns | event loop, GL context, `App`, `BookView`, textures, upload budget | `Source`, `Store`, decode, plan, mint, unpack |
| lifetime | the process | one book; killed at switch and at exit |
| mode | `cy` | `cy --worker` |

One process per book means a switch ends the old book's work by ending
its process; the handle's event receiver drops with it, so no old event
can arrive. Killing the old child at switch start is what removes the
switch-time in-flight decode that `analysis.md` item 1 described.

## Bootstrap

1. The parent creates `IpcOneShotServer::<Hello>::new()`. Its name is the
   only rendezvous the child needs.
2. `Command::new(env::current_exe())` with `--worker`, the server name,
   and the page path. stdin is null; stderr is inherited, so child logs
   land on the parent's stderr. The path travels in argv as an
   `OsString`, never as a serde payload, so a non-UTF-8 path survives.
3. The child arms `PDEATHSIG` with `SIGKILL`, exits if `getppid() == 1`
   (the parent died before the signal was armed), opens a `ChildMsg`
   channel and a `ParentMsg` channel, connects to the server, and sends
   `(ChildMsg receiver, ParentMsg sender)` as the `Hello`.
4. The parent accepts on a `cy-book-accept` thread, because `accept` is a
   blocking `accept(2)` that killing a child which never connected cannot
   wake. Accept and the first message share one `OPEN_TIMEOUT` (30 s)
   deadline. On timeout the parent kills and reaps the child through the
   `ForkedChild` guard, then wakes a listener that has no client by
   connecting a dummy sender and closing it, so the accept thread can
   join; it reports `BookFault::Timeout`.
5. The child opens the source and sends `BookInfo` or `OpenFailed`. The
   parent reads that first message from the channel, starts the reader
   thread, builds `BookView` from it, and the open returns.

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

Page identity is the index in the book the child was spawned with, so no
message carries a path. The parent sends wants as a whole set; the child
keeps no acknowledgement, and delivery is inferred from the texture maps
exactly as it was in-process.

The child's loop turns `SetView` into
`Plan::build(view::spread_for(index, mode, len), direction, mode, len)`
and retunes the worker; the first `SetView` spawns it. `SetMotion` and
`SetWants` update a running worker. The loop ends when the parent
disconnects or kills the child. Worker events go through a bounded writer
queue, which mints `WireImage` from the decoded planes and sends
`ChildMsg`.

The archive pass has no status thread: the worker observes
`Source::unpacking()` in its own poll and emits `worker::Event::Unpacked`
once, which the child maps to `ChildMsg::Unpacked`; the app clears the
view's unpacking flag.

The parent's reader thread converts on the way in: `Ready` becomes
`worker::Ready { image: Box<Decoded> }` after `WireImage::to_decoded`, a
failed conversion becomes `Failed` with a `PageFault`, and both an IPC
error and `Fatal` become `Died(String)`. It then calls the wake closure,
so a parked loop picks the event up. The UI thread sees `BookEvent` only,
never `ChildMsg`.

## WireImage: pixels through shared memory

A decoded artifact does not fit the serialized messages: a view is about
16.7 MiB, a 50 MP native up to 191 MiB, and an IPC pipe would copy that
through the kernel and back. `WireImage` instead holds the pixel metadata
and one `IpcSharedMemory` block per plane (`Rgba8`, or `Yuv` with depth,
subsampling, range, matrix, and optional alpha).

`WireImage::from_decoded` copies each plane into a new block; `to_decoded`
copies each block back into a `Decoded` built through
`Decoded::rgba`/`yuv`, so the constructors validate the extents, and
`depth > 8` selects `Plane::U16` (native endian). Only metadata and
handles cross the channel.

The transport removes the serialization, not the copy: an artifact costs
one copy into shared memory and one back into a `Decoded` before upload.
Zero-copy upload, by mapping the segment in GL or by writing into a
renderer-owned buffer, is a later optimization and would change this
contract.

## Lifecycle

- **The initial open is synchronous.** `parent::run` creates the
  `EventLoop` and its proxy, calls `BookProcess::open(path, wake)`,
  computes the landing index, and builds
  `App::new(view, process, index, proxy)`. Startup blocks through spawn,
  handshake, and `BookInfo`, but the loop and the wake path exist from
  the start; the window appears with a book already known.
- **A switch is asynchronous.** `switch_book` derives the sibling path
  from the `BookView`, kills the current child, keeps the old view and its
  GPU textures on screen, and forks the new child with
  `PendingBookOpen::begin` on the loop's thread. A `cy-book-open` thread
  then calls `finish`, which posts `(BookView, BookProcess)` on a bounded
  channel and wakes the loop, so drawing continues and no timer polls.
  Only one open is pending at a time; a switch during a pending open is
  ignored.
- **Events wake the loop.** The reader thread gets a wake closure, the
  proxy's `send_event(UserEvent::Wake)`, and calls it after every
  `ChildMsg`, its final `Died` included, so a parked loop drains the
  book's events and draws the artifact.
- **Adoption.** On `Ok((view, process))` the app replaces its view and
  process and lands on the first spread with `SetView` and a cold want
  set. It clears the sent wants, cached artifacts, page faults, upload
  queue, refused page, presented spread, every GPU texture, the fit, and
  the skim. On `Err(fault)` the old book stays.
- **Kill.** `BookProcess::kill` sends `SIGKILL`, waits for the child
  (reaping it), and joins the reader thread. `Drop` calls it, so a
  dropped book cannot leak a process. `App::exiting` kills the child
  before GL teardown.
- **PDEATHSIG.** If the parent dies abruptly the child is killed with it.
  Nothing needs flushing: the store is content-addressed and every temp
  file is disposable.

## Failure model

Two layers, split by who owns the failure.

- `PageFault` is one page as text: a `summary` string, so a typed
  `PageError` never crosses the wire. Upload errors convert from
  `PageError`; the child's reports arrive as the same text. Page faults
  are sticky per page in the app's fault map.
- `BookFault` is the book: `Spawn` (a process or thread would not start),
  `Handshake` (accept failed, a protocol violation, or the stream ended
  before `BookInfo`), `Open` (the child said `OpenFailed`, or the book has
  no pages), `Timeout` (no accept and no `BookInfo` within 30 s; the child
  is killed), and `Died` (the stream ended after `BookInfo`, or the child
  said `Fatal`). After adoption a `BookFault` is shown in the status
  line; it never replaces the book on screen.
- A child crash is a `BookFault`, not a page fault. The reader thread
  turns the IPC error into `Died` and wakes the loop; the app takes and
  kills the process, notes the fault, and keeps the last known `BookView`
  and textures. Sends go nowhere (send errors are ignored), so the view
  gains no new pixels, and a switch still works because it spawns a new
  process.
- A panicking decoder is caught inside the child and reported as a failed
  page, so it never becomes a book fault.

## The shared store

Every child of the session opens the same `$TMPDIR/cryslen-cache` tree,
keyed by the xxh3-128 of the encoded page bytes. The key needs no owner,
so a page produced by one book's process is a hit for another's, and a
book reopened after a switch finds every preview it left. Several
children write this tree at once and the caps are shared, so one child's
prune can evict an entry another child's plan counts as produced. The
format, temp-file, and pruning rules are in `docs/internal/cache.md`.

## Runtime

There is no async runtime. Cross-process traffic is `ipc-channel`, and
the worker's events pass through a bounded `crossbeam-channel` queue, so
the writer depth and the want set bound what is in flight. Decode is
synchronous, CPU-bound, and already parallel inside its decoder (dav1d's
threads, jxl's runner), so a reactor would add only a blocking pool.
Supervision is the operating system: a child that hangs, panics, or dies
is a `BookFault` with a process to kill.

## Residual limits

- **Every artifact is copied twice.** The child copies the minted planes
  into shared memory; the parent copies them back into a `Decoded` before
  upload. A view pays about 2 × 16.7 MiB, a 50 MP native about
  2 × 191 MiB. Removing it needs the upload to read the segment.
- **The child is not sandboxed.** It runs as the same user with the same
  filesystem access. The boundary buys crash isolation, memory release at
  switch, and a responsive UI thread, not containment of hostile image
  data.
- **The initial open still blocks startup.** Spawn, handshake, and
  `BookInfo` run synchronously in `parent::run`, before `App` exists and
  before the loop runs; only switches are asynchronous.
- **`SIGKILL` is the normal stop path.** A killed child can leave a
  `.tmp` file; the dead-pid scan at the next `Store::open` removes it.
- **Child logs are unattributed.** Both modes log under the same target,
  so a child's lines interleave with the parent's and neither is tagged
  with its book.