# cryslen architecture scratchpad

Working notes. Goal: flatten the program into one `main`, recover the real
data flow, then rebuild the abstraction from that flow.

Size: 17,081 lines of Rust total; **10,931 production**, 6,150 test.
~135 `struct`/`enum`/`trait` declarations (grep count, private included).

| file | prod | test |
|---|---:|---:|
| app.rs | 1842 | 844 |
| worker.rs | 1561 | 1079 |
| render.rs | 1621 | 166 |
| cache.rs | 1016 | 712 |
| decode/webp.rs | 542 | 243 |
| decode/mod.rs | 511 | 713 |
| browse.rs | 459 | 267 |
| source.rs | 418 | 622 |
| book.rs | 409 | 18 |
| ui.rs | 397 | 282 |
| decode/avif.rs | 394 | 35 |
| presentation.rs | 321 | 441 |
| geometry.rs | 287 | 349 |
| catalog.rs | 244 | 65 |
| protocol.rs | 231 | 103 |
| animation.rs | 215 | 94 |
| decode/jxl.rs | 181 | 47 |
| testkit.rs | 117 | 0 |
| decode/image.rs | 46 | 52 |
| fault.rs | 62 | 11 |
| lib.rs | 38 | 7 |
| main.rs | 19 | 0 |

---

## 1. The program flattened into one `main`

### 1.1 What the process really is

One job: **turn key presses into a changed frame without waiting for a
decode.** Everything else serves the "without waiting" part.

The essential state, written as locals in `main`:

```
book        : path, names[], kind, display labels
child       : process, to_child, from_child, death
ui          : mode, fit_mode, zoom_pin, placement
reading     : wanted_spread, shown_spread, direction, motion
page_facts  : page -> (dims, format)          # immutable
page_faults : page -> text                    # sticky
page_pixels : page -> {thumb, fit, full}      # GPU residency
in_flight   : Set<(page, role)>               # asked, not yet arrived
playback    : page -> session(role, player)
prompt      : digits | none
clock       : wait_grace, toast, playback, egui_repaint
```

Nothing else is application state.

### 1.2 One frame, in the order the code runs it

`RedrawRequested` → `App::frame` (`app.rs:954`):

1. `sync_surface_size` — window inner size → GL surface resize.
2. `poll_child(limits)` (`app.rs:901`)
   - opening: handshake landed → `adopt_book`: clear GPU textures, reset
     `Reading`, send limits + view + cold requests → `Live`.
   - live: drain child messages → deliveries queue, page faults, playback
     frames; flush playback commands; death → `Faulted`.
3. `drain_deliveries` (`app.rs:1019`; `Reading::drain_deliveries`
   `app.rs:372`)
   - pop deliveries under a 16-image / 8 MiB budget,
   - upload to `PageTextures`,
   - **only on upload success**: record `PageInfo{dims, format}`,
   - on refusal: fault the page if it is non-thumb and no settled role
     exists.
4. `present_reading` (`app.rs:1322`)
   a. `FrameInput { target, shown, mode, motion, direction, len, limits }`
   b. `choose_presented` → `shown` (target, else nearest between previous
      shown and target, else previous shown).
   c. shown changed → `Fit::commit()` (invalidate placement memo).
   d. `record_shown(mode, shown)` — writes the shown back.
   e. `on_screen = shown.unwrap_or(target)`.
   f. `seat_dims` from `pages` → `geometry::layout` → spread size.
   g. `Fit::resolve(spread size, viewport)` → placement.
   h. `build_frame` → `{demand, requests, retention, playback targets}`.
   i. retain textures, sync playback, tick playback (upload frames, next
      commands, arm a deadline).
   j. if `requests != sent_requests` → send the whole set, remember it.
   k. `summarize_spread` → status; tick timers.
5. `present_frame` (`app.rs:724`)
   a. egui `run` once to lay out the status bar → `Response` (page surface).
   b. `content_viewport` from the response rect.
   c. viewport changed → `Fit::refit()` + request redraw.
   d. no jump prompt → `apply_page_input` (drag → pan, wheel/pinch → zoom)
      **during layout**; `Fit::resolve` a second time.
   e. clear framebuffer, draw page quads, paint egui, swap.
6. `ControlFlow::Wait` — park until the next event/deadline.

### 1.3 The child, flattened

`worker::run` (`worker.rs:1396`):

```
source = Source::open(path)          # listing now; archive unpacks in a thread
store  = Store::open()               # $TMPDIR/cryslen-cache/{thumb,view}
memory = LRU(hash -> Decoded, 192 MiB)
state  = {
  prefetch ring, motion, hashes[page], stored[hash] -> roles,
  failed[page], failed_full[page], in_flight[page], requests, served,
  preload gate
}
spawn 2..=6 workers:
  job = next_job(max_prefetch):
     1. first request not yet served/failed, page not in flight
     2. else prefetch ring page missing its role
        (head + still -> Fit, else Thumb)
     3. else wait; tail preload only after 300 ms of quiet
  produce(page):
     bytes = source.bytes(page); hash = xxh3_128(bytes)
     memory[hash] -> Fit
     else store[hash, Thumb] -> publish if requested
     else store[hash, Fit]   -> Fit
     if needed: decode(bytes) -> native
        publish Full if requested
        derive Fit -> publish if requested + persist
        derive Thumb from Fit -> persist (+publish if requested)
main loop (same thread as IPC):
  apply messages (limits, view, motion, requests, playback)
  serve one playback step (decode a frame on the IPC thread)
  block on recv
```

### 1.4 Fact inventory: who owns each truth

| fact | authoritative owner | copies / mirrors |
|---|---|---|
| page names & order | child `Source` → wire `BookInfo` → parent `BookView` | child `Source.names`; parent `PageNames`; `PageNames.count` duplicates `Vec::len` |
| page index identity | the `usize` in every message | both processes must agree for the session lifetime |
| content identity | child `hashes[page] -> u128` | store filenames; memory-cache keys; `stored[hash]` ledger |
| page dims / format | parent `PageState.info`, derived from an **uploaded** pixel image | child `Decoded.page_dims/format` in every wire image |
| pixel residency (role set) | parent `PageTextures` | child `served`; child `stored[hash]`; child `memory_cache`; disk store |
| page fault | parent `PageState.fault` (sticky) + child `failed`/`failed_full` | wire `PageFault` text |
| book fault | parent `ChildProcess::Faulted` | child death only |
| spread on screen | parent `ReadingPosition.shown` | last frame's `Fit.settled` |
| where the user wants to be | parent `ReadingPosition.target` (a page index, often but not always a spread start) | child `PageState.prefetch` origin |
| reading mode | parent `ViewState.mode` | `ShownSpread.mode`; child copy from `SetView` |
| direction | parent `Reading.direction` | child copy from `SetView` |
| motion | parent `Reading.motion` (from `HeldKeys`) | child copy from `SetMotion` |
| desired roles | parent `Frame.requests` → `Reading.sent_requests` | child `requests` + `served` |
| what to prefetch next | parent `demand_spread` (skimming) | child `Prefetch::build` + `next_prefetch_job` (independent policy) |
| fit mode & placement | parent `ViewState.fit` | layout recomputed each frame |
| playback session | parent `PageState.playback` (`Player`) | child `PlaybackRunner.playing_pages` |
| wait/toast/playback deadlines | `ViewState.timers` + `WakeEvent::Repaint` | event-loop `ControlFlow` + egui repaint callback |
| book lifecycle | `BookSlot` × `ChildProcess` × `CurrentBook` | `BookView` inside `CurrentBook` |

**Core finding:** six independent mechanisms represent "which pixels do we
have": disk store, memory LRU, child `stored` ledger, child `served` set,
parent textures, parent request diff. Each is defensible alone; together
they are one fact with six homes.

### 1.5 Phase table: who touches what

| step | intent | facts | GPU pixels | child | clock | ui |
|---|---|---|---|---|---|---|
| `poll_child` | – | – | clear on adopt | drain/send | – | – |
| `drain_deliveries` | – | **write** | **write** | – | – | – |
| `present_reading` | read/record shown | read | retain/update | playback + demands | arm | status |
| `dispatch` | **write** | – | – | send | toast | – |
| `present_frame` | fit **write** | – | draw | – | arm | layout |

A single frame function is the only place all six meet. In the current
code it meets them by *calling six owners in sequence*; in the target it
should be one pure decision plus explicit effects.

---

## 2. The recovered data flow

### 2.1 Identity chain

```
path ──Source::open──▶ names[] ──index──▶ every IPC message
                         │
                         └─bytes(page)──▶ xxh3_128 ──▶ store key, memory key
```

Two identities coexist: **index** (session-scoped, cheap, what the reader
means) and **content hash** (durable, what the cache means). The parent
only knows the index. The child pays one source read to learn the hash,
then memoizes it. That is why a cold session cannot consult the disk store
without reading the file, and why `MemoryCache` is keyed by hash.

Consequence: `Store` cannot answer "do you have page N?", so the child
keeps `hashes` and `stored` to compensate.

### 2.2 Fact chain

```
decode(bytes) ──▶ Decoded{page_dims, format, ...}
     └─ WireImage{page_dims, format, ...} ──▶ parent
          └─ PageTextures::insert ──▶ PageState.info        (app.rs:402)
```

Page facts ride every pixel delivery but are *recorded* only when a GPU
upload succeeds. The planner's knowledge of page dimensions therefore
depends on the GPU. `debug_assert_eq!` guards fact immutability; the type
does not.

### 2.3 Residency chain (the feedback loop)

```
        ┌───────────────── parent ─────────────────┐
        │  plan: want(page,role)                   │
        │      minus roles the GPU already has     │
        └───────────────┬──────────────────────────┘
                        │ SetRequests (whole set, diffed)
        ┌───────────────▼──────────────────────────┐
        │  child: serve each request once per      │
        │  "stay in the set" epoch; prefetch the   │
        │  ring at its own guessed roles           │
        └───────────────┬──────────────────────────┘
                        │ Ready{page,role,image} on shared memory
        ┌───────────────▼──────────────────────────┐
        │  parent: upload; roles become residency  │
        │  next plan sees them and drops the req   │
        └──────────────────────────────────────────┘
```

Ack = "the request left the set". No negative ack. A refused thumb upload
stays requested but already served, so the child never resends it while
the page remains in the set.

### 2.4 Intent chain

```
winit key ─▶ Action ─▶ dispatch:
   turn:      position.turn(dir, mode, len, motion)  ─▶ direction, begin_turn
   first/last/jump: set_target(...)                  ─▶ after_page_change
   mode:      commit_shown(old) → switch_mode → set_target → begin_turn
   fit:       fit.select(mode)
   book:      start_open(path)
   quit/fullscreen/...
                │
                ▼
         SetView{page, mode, direction}   (sent immediately)
         SetMotion{motion}                (on held-key change)
         request_redraw
```

The turn decision uses `motion` *and* `shown` (skim guard), so navigation
depends on display state. That is why `ReadingPosition` carries `shown` as
real state, not an output.

### 2.5 Role order and spread distance: two leaked representations

- `[Role::Full, Role::Fit, Role::Thumb]` is written out four times:
  `Roles::best` (`browse.rs:249`), `best_texture` (`render.rs:253`),
  `wanted_roles` (`worker.rs:539`), `derivation` (`cache.rs:90`).
  The role order is a fact without a home.
- `Spread::contains` is re-implemented as `spread_contains`
  (`worker.rs:87`).
- `FIT_KEEP_SPREADS = 2` (`presentation.rs:240`) is compared with
  `spread.first.abs_diff(...)`: **page-index distance, not spread
  distance**. In Normal mode that is 2 spreads; in Manga/Comic it is 1.
  The constant's name states the wrong unit.

### 2.6 The viewport / status cycle

The status bar's height defines the page viewport, but the status text
needs the plan, and the plan needs the viewport. The code breaks the cycle
by planning with `view.last_viewport` (previous frame), then laying out
egui, then resolving placement a second time in `present_frame`. Effects:

- zoom shown in the status lags one frame;
- `Full` demand is computed from the pre-input scale;
- `last_viewport` + `refit` exist solely to close the loop;
- `Fit::resolve` runs twice per frame.

That is the hidden cost behind `present_reading` + `present_frame`.

---

## 3. What is real design, what is friction

### 3.1 Real design (domain truth, keep)

| unit | why it is real |
|---|---|
| two-process split | kill-on-switch stops in-flight decode instantly; `PDEATHSIG` cleanup; shared-memory pixels; a decoder panic cannot take the window |
| `Spread` pairing arithmetic (`browse.rs`) | cover alone, odd tail alone, rtl seat order |
| `ReadingPosition.target/shown` + commit-on-still | skim shows stale pages; release must commit what the eye saw |
| `Role` tiers + `Roles` | three resolutions with different costs/consumers |
| `Motion` from key auto-repeat | drives thumb-only skimming |
| whole-set demand, once per epoch | unidirectional, no per-request ack, parent authoritative |
| content-hash store + atomic rename + per-namespace caps | dedupe across books, crash-safe, multi-process |
| shared-memory planes + parent validation | zero-copy boundary plus a trust check |
| playback: parent timing, child decode, generation id | two clocks, one owner each |
| `PageFault` text across the boundary | error types stay local |
| `Fit` placement math, zoom anchor, clamp | real viewport behavior |
| archive progressive slots, per-page errors | 7z solid streams decode in storage order |
| `PageTextures` role slots, largest-role draw | GPU cache with real limits |
| upload budget | a huge page must not stall the UI |
| deadline/wake loop | must park and still animate |
| `ImageLimits` through all layers | display edge + GPU edge are one session fact |
| `DisplayTransform` at decode | layout must see upright dims |

### 3.2 Friction (accidental complexity)

| unit | evidence | cost |
|---|---|---|
| five lifecycle types: `BookSlot`, `ChildProcess`, `CurrentBook`, `BookView`, `BookOpen` | `app.rs:602-659`, `book.rs` | a 2×4 state matrix (`Initial/Loaded` × `Opening/Live/Faulted/Stopped`) with hand-maintained pairing; `send` targets `self.child` while book state is `self.book` |
| `TextureSink` | `app.rs:169`, sole non-test impl | test seam in the production call path |
| facts recorded in `drain_deliveries` | `app.rs:402-412` | ingestion + GPU upload + fact derivation in one function |
| `FrameInput`/`Frame`/`PageSummary`/`Retention` | `presentation.rs` | a plan split into 2 calls (choose, build), 3 projections of the same state (`app.rs:1352,1417`, `ui.rs:116`) |
| `Frame.demand` | `app.rs:1447` | debug log only |
| `cold_requests` | `presentation.rs:147`, `app.rs:704` | duplicates `request_roles` on an empty scene |
| parent request policy vs child prefetch policy | `presentation.rs:192-231` vs `worker.rs:600-634` | same question answered twice |
| `Reading` (10 fields, 5 methods) | `app.rs:269` | intent + residency + uploads + playback + prompt + diagnostics |
| `ViewState` (4 fields) | `app.rs:661` | mode + placement + UI timers + viewport |
| `PageState{info,fault,playback}` | `app.rs:216` | facts + pixels + async session in one record |
| `PagePlayback::Refused` + `Player::RequestState` + `next_generation` | `app.rs:211`, `animation.rs:84` | one session protocol hand-rolled in two places |
| `stored: HashMap<u128, Roles>` | `worker.rs:439` | store mirror because the store cannot answer `has` |
| `MemoryCache` | `worker.rs:315` | second cache, same key/role shape, different medium |
| `served` + `retain` | `worker.rs:449,514` | ack emulation |
| `failed`/`failed_full` | `worker.rs:442-444` | one rule split by role |
| `Prefetch` + `in_head_still` + `PreloadGate` | `worker.rs:201,430,415` | urgency policy as a ring + bool + countdown |
| `Fit.settled`/`user_moved` + 6 invalidation sites | `geometry.rs:203`; `refit/select/begin_turn/commit/resolve` | memoized placement with call-site invalidation |
| `Timers` + `Repaint` + `ControlFlow` + egui callback | `ui.rs:56`, `app.rs:676,1557` | three ways to arm one deadline |
| `WakeEvent::Wake` vs `Repaint` | `app.rs:79` | two wake kinds for one effect |
| 18 `request_redraw` sites | `app.rs` | no single "state changed" point |
| per-frame `page_name` string surgery | `book.rs:89` → `catalog::page_name` | display formatting in the frame |
| `archive`/`dir_name` in `BookView` | `book.rs:51-53` | `dir_name` display-only; `archive` a redundant bool for `BookKind` |
| protocol re-exports domain vocabulary | `protocol.rs:20-34` | wire and domain are the same names |
| `BookSlot::Initial(PageName)` + `Opening{land_on}` | `app.rs:607,629` | two "opening" representations |
| `apply_page_input` inside draw | `app.rs:777` | pointer input mutates fit while presenting |
| `jump_digits` in `Reading` | `app.rs:283` | prompt state mixed with navigation |
| `Spread::contains` duplicated | `worker.rs:87` | two definitions of the same predicate |
| role order written 4× | `browse.rs:249`, `render.rs:253`, `worker.rs:539`, `cache.rs:90` | ordering fact without a home |

### 3.3 Verdict

Of ~130 production types, **roughly 100 carry a fact or a boundary**
(spread math, role, direction, placement, decode, store, wire, GL, source).
About **25–30 are composition artifacts** concentrated in `app.rs`,
`worker.rs`'s scheduler policy, and `presentation.rs`.

By lines: the accidental layer is **~1,500–2,000 production lines (~15%)**,
but it is the layer every feature touches.

The important asymmetry: the **hard** parts (spread arithmetic, YUV
decode, cache format, GL) are well factored; the **soft** part (who owns
the frame decision, the book lifecycle, and the notion of "having" a
pixel) is tangled. That is why new features feel expensive.

---

## 4. Latent flows / edge cases obscured by the split

Tag: S = structural, O = observable.

1. S **No negative ack.** Ack is "request left the set". A refused or
   dropped delivery stays requested but already served. Production
   refusals are limited to the dims check (unreachable for Thumb/Fit,
   which derivation clamps), so this is a protocol hole, not a live bug.
2. S **Facts ride on pixels.** Dims/format are recorded only on upload
   success (`app.rs:402`). The failure modes are defensive today, but
   layout/status knowledge is coupled to the GPU.
3. O **Switch drift.** During `Opening`, the old `Reading` still accepts
   turns; `SetView`/`SetRequests` are dropped because the child is not
   `Live`. A mode switch in that window can re-pair the drawn spread;
   adopt then resets everything. Invisible in the common case.
4. S **Facts belong to the GPU.** Same cause as 2: status and layout
   depend on upload timing, not decode timing.
5. O **Two prefetchers.** The child's ring is built around the far
   target, not the next on-screen spread while skimming. In-flight jobs
   are not cancelled by ring membership. Wasted decode, not wrong frames.
6. O **Playback blocks IPC.** The child's main loop decodes frames, so a
   `Stop` waits behind the current frame decode.
7. O **Store ledger staleness.** Another worker can prune entries the
   ledger still believes present → missed prefetch, not wrong display.
8. **`MemoryCache` and `Store` evict independently.** No reconciliation;
   memory may hold a Fit the store dropped (fine) and vice versa.
9. **Archive slots never free.** A read page's `Arc<[u8]>` lives for the
   session. For 7z this is a *constraint*, not a leak: the solid reader
   is forward-only, so the slot is the only readable copy.
10. O **Unit leak.** `FIT_KEEP_SPREADS` is a page-index window
    (`presentation.rs:240,250`); paired modes get half the intended
    retention.
11. S **A worker panic shrinks the pool silently.** Only decode and
    playback are `catch_unwind`-guarded; `produce_page`/store work is
    not, and `worker_loop` does not restart a dead worker. The child
    process stays alive but serves less or nothing.

---

## 5. The redesign

### 5.1 Principles recovered from the flattening

1. **Facts and pixels are different facts.** Dims/format are immutable
   page facts; residency is a revocable cache. Never one record for both.
2. **One planner.** Presented spread, layout, placement, requests,
   retention, playback, status: one pure function over one snapshot.
3. **One lifecycle.** `Opening | Live | Failed`; `Live` owns the child,
   facts, position, playback.
4. **One authority for demand.** The parent decides *what* is needed; the
   child decides *how* and *in what order*.
5. **One clock.** A deadline set owned by the loop.
6. **One writer per phase.** Input mutates intent; ingest mutates facts
   and residency; plan is read-only; apply performs effects; present draws.
7. **Types for invariants, not for stages.** Valid spread starts, facts,
   role sets. Not `PageSummary`/`FrameInput`/`TextureSink`.

### 5.2 Target module map

```
src/
  main.rs                  args → parent | --worker

  model/                   pure; no IO, GL, clocks
    book.rs                PageCount, PageNames, ReadingMode, Spread, Direction
    reading.rs             Position { wanted: Spread, shown: Option<Shown> }, Motion
    role.rs                Role (+ DESC order), Roles, Request
    page.rs                PageFact { dims, format }, PageFault
    plan.rs                Scene -> Plan        (the only frame decision)
    status.rs              Scene -> StatusLine  (pure projection)
    placement.rs           Viewport, FitMode, Placement, Fit { mode, pin }, layout

  host/                    parent process
    app.rs                 winit + egui + loop + ingest/apply/present
    session.rs             Session { Opening, Live, Failed }, child handle, switch
    page_table.rs          PageFacts { fact, fault } + Pages query
    playback.rs            parent timing, session id, Player
    input.rs               keys → Action; held keys → Motion; prompt
    clock.rs               Deadlines { wait, toast, playback, egui }
    render.rs              Window, PageTextures (GL)
    ui.rs                  status bar, prompt

  child/                   worker process
    main.rs                handshake + loop
    source.rs              directory/zip/7z page bytes
    store.rs               content-hash disk cache
    decode/                format backends + animation
    produce.rs             bytes → hash → memory/store/decode → roles
    schedule.rs            demand queue, warm queue, in-flight, failure memory
    playback.rs            animation sessions (child side)

  wire.rs                  Hello, ParentMsg, ChildMsg, WireImage, validation
```

Net: ~10–15 types deleted, ~12 merged, ~6 added.

### 5.3 The server contract (wire)

Only what the service can act on; no reading policy.

```rust
enum ParentMsg {
    Limits(ImageLimits),
    Demand { required: Vec<Request>, warm: Vec<Request> },
    Playback(PlaybackCommand),
}
enum ChildMsg {
    Ready { page: usize, role: Role, image: WireImage },
    Failed { page: usize, role: Option<Role>, fault: PageFault },
    Playback(PlaybackEvent<WireImage>),
}
```

- `required`: serve and publish, in order.
- `warm`: load/derive and persist, never publish. Ordered; the child may
  keep the leading entries in its memory tier (its own cache policy), but
  warming never reaches the parent.
- Whole-set send on change; the child serves each `required` element once
  per epoch (epoch = entered the set). That single rule replaces
  `requests`, `served`, `in_head_still`, and the prefetch role guess.
- `Failed { role: Some(..) }` is the missing negative ack: it tells the
  parent to stop asking for that role (a texture limit, a derivation
  failure) without faulting the page. `role: None` is a page fault.
- `page`, `mode`, `direction`, `motion` disappear from the wire: they were
  the child's copy of the parent's reading model.
- Cold start: `adopt` runs the same `plan` on an empty `PageFacts` and
  sends `Demand{Fit for the target spread}` before the first frame, so
  decode starts before egui font scan (the role `cold_requests` played).

### 5.4 The frame pipeline

```
ingest()    drain child → PageFacts.fact/fault + uploads → textures;
            death → Session::Failed
chrome()    egui run: status text (from the last Plan) → page rect + response
input()     response/pointer → view pin (zoom/pan)
plan()      plan(&Scene) -> Plan                    (pure, fresh viewport)
apply()     record shown, retain textures, sync playback,
            send Demand if changed, arm deadlines
present()   draw page quads; egui paint; swap
```

The plan runs after chrome and pointer input, so placement and `Full`
demand use this frame's viewport and zoom (today they use the previous
frame's and then resolve a second time). The status text therefore shows
the last plan; only the zoom readout lags one frame, as it already does.
A fixed-height status bar would let the plan run before chrome and remove
even that lag.

```rust
struct Plan {
    presented: Option<Spread>,
    layout: SpreadLayout,           // from facts
    placement: Placement,
    demand: Vec<Request>,           // required
    warm: Vec<Request>,             // prefetch ring, roles chosen here
    retain: Retention,
    playback: [Option<(usize, Role)>; 2],
    status: Status,                 // waiting, fault, named page, zoom
}
```

The plan's inputs are a `Scene` borrowing: `Book`, `Position`, `View`
(fit mode + pin), `PageFacts`, `PageTextures::roles`, `Viewport`,
`ImageLimits`. No closure rebuilt three times.

`Pages` is the read-only query the plan uses:

```rust
fn fact(&self, page: usize) -> Option<PageFact>;
fn fault(&self, page: usize) -> Option<&PageFault>;
fn roles(&self, page: usize) -> Roles;      // only PageTextures implements this for real
```

Facts/faults come from `PageFacts`; roles from the GPU. Tests substitute a
`HashMap`; there is no mutating sink trait in the production path.

### 5.5 Placement without invalidation

```
placement = match view.pin.filter(|pin| pin.wanted == position.wanted) {
    Some(pin) => clamp(pin.placement, total, viewport),
    None => fit_as(view.fit_mode, total, viewport),
}
```

The pin is keyed to the **wanted** spread, not the presented one: a zoom
made during the hold before the target arrives must carry into it (the
current `commit` behavior). A page turn changes `wanted`, so the pin
drops. A viewport change re-fits/clamps naturally. A mode switch changes
the pairing, so it drops. Deletes `settled`, `user_moved`, `refit`,
`commit`, `begin_turn`, `select`.

### 5.6 Child: one production pipeline

```
schedule:
   required (ordered) ++ warm (ordered)
   skip in-flight page, failed role, already-answered request
produce(page, roles):
   bytes = source.bytes(page)
   hash  = hashes.entry(page).or_insert(xxh3_128(bytes))
   hit memory(hash, role) -> publish/persist
   hit store(hash, role)  -> memory.put, publish/persist
   missing -> decode once; derive roles; publish required, persist all
```

- `stored` becomes a **hint** (`warmed: HashSet<(page, role)>`) owned by
  the scheduler; a stale hint costs one wasted check, never a wrong frame.
- Failure memory is one `HashSet<(page, Role)>`; source failures cover
  Fit+Thumb, `Full` is its own entry.
- Playback remains the one session protocol.

### 5.7 What this deletes

`Reading`, `ViewState`, `PageState`, `PageSummary`, `FrameInput`, `Frame`,
`Retention` (becomes private to plan), `TextureSink`, `MotionChange`,
`BookSlot`, `CurrentBook`, `ChildProcess`, `cold_requests`, `Timers`,
`Prefetch`, `PreloadGate`, `JobKind`, `PageJob.kind`, `stored` ledger,
`FitOrigin`, `served`, `WakeEvent`, `BookView.archive/dir_name`.

Note: `BookView` itself folds into `Book`; `ChildProcess`/`BookSlot`/
`CurrentBook` fold into `Session::Live`.

Wire: `SetView`, `SetMotion`, `SetRequests` → `Demand`.

### 5.8 Target `main` skeleton

```rust
fn main() -> ExitCode {
    match classify(env::args_os()) {
        Worker { server, path } => child::main(server, path),
        Parent { path, land_on } => host::run(path, land_on),
    }
}

fn host::run(path, land_on) -> eros::Result<()> {
    let event_loop = EventLoop::<()>::with_user_event().build()?;
    let mut app = App::new(path, land_on, event_loop.create_proxy());
    event_loop.run_app(&mut app)?;
    app.take_fatal_error().map_or(Ok(()), Err)
}

impl ApplicationHandler for App {
    fn window_event(.., event) {
        match event {
            RedrawRequested => self.frame(event_loop)?,
            KeyboardInput { .. } => { self.input.key(key, &mut self.session, &mut self.view); self.redraw(); }
            Resized(_) => { self.gl.resize(); self.redraw(); }
            ..
        }
    }
}

fn App::frame(..) {
    let gl = self.gl.as_mut()?;
    self.session.ingest(gl);                        // facts, faults, uploads
    let (page_rect, response) = gl.chrome(&self.last_status);
    self.input.pointer(&response, gl.viewport(page_rect), &mut self.view);
    let plan = self.session.plan(gl, &self.view, gl.viewport(page_rect));
    self.session.apply(&plan, gl);                  // shown, retain, demand, playback
    gl.present(&plan, page_rect);                   // egui paint + pages + swap
    self.last_status = plan.status.clone();
    self.event_loop.set_control_flow(WaitUntil(self.clock.next()));
}
```

### 5.9 Migration slices (smallest reversible first)

1. **Split facts from pixels.** Move `info` recording to delivery ingest;
   split `PageState` into facts/faults; delete `PageSummary`.
2. **One planner.** Merge `choose_presented` + `build_frame` + layout +
   placement into `plan.rs`; delete `cold_requests` and `Frame.demand`;
   make the frame read-only over the reading model.
3. **One lifecycle.** `Session { Opening, Live, Failed }`; `Live` owns
   `{ book, child, facts, position, playback }`; delete `BookSlot` /
   `CurrentBook` / `ChildProcess`.
4. **One clock.** `Deadlines` + `next_deadline()`; one `Wake`; delete
   `Timers` and `Repaint` posts.
5. **Move demand policy to the parent.** Emit `warm`; delete
   `SetView`/`SetMotion`/`Prefetch`/`in_head_still`.
6. **Collapse `Fit`.** Pin identity; delete the six mutators.
7. **Names once.** Precompute `PageLabel` at adopt.
8. **Split `Reading` leftovers**: prompt → input, playback → playback,
   deliveries → ingest.

Slices 1–4 keep the protocol; 5 changes it.

---

## 6. Open questions / tradeoffs

- **Single process?** Flattening shows the boundary carries names, facts,
  pixels, demand, faults, playback. The only hard justification left is
  cancellation by kill (Rust cannot cancel a decode) and `PDEATHSIG`. If
  cancellation tokens became acceptable, a thread-per-book design would
  delete the whole wire layer and duplicate identity/residency. Keep the
  child while switch latency matters.
- **Store index.** Either `Store::has(hash, role)` (one stat) or the
  scheduler keeps a hint. Prefer a hint that cannot be wrong.
- **Facts before pixels.** Child publishes `Facts{page}` on first decode
  → status/layout stop depending on upload. Adds one message; optional.
- **Warm vs publish.** Moving `warm` to the parent adds a ~64-page ring
  to parent state, but removes a whole policy from the child and lets the
  ring origin follow the on-screen spread while skimming.
- **Fixed-height chrome.** Computing the viewport before the egui pass
  breaks the status/viewport cycle; it constrains the status bar to one
  measured line. Alternative: keep the two-pass plan, but confine it to
  one function and one explicit `Viewport` input.
- **`ReadingMode` in the child.** Gone under `Demand`; if kept for ring
  ordering, pass one `View` message, never role policy.