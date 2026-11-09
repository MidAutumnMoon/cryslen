# Cache: three representations, one plan, one store

Reference for anyone touching `preview.rs`, `plan.rs`, `worker.rs`, or
the presentation half of `app.rs`/`render.rs`.

The viewer holds three representations of a page, one production
order, and one ownership rule: the app's texture maps decide what is
missing, and the worker serves exactly that. No component assumes a
delivery stuck.

| representation | edge | bytes (3:2) | job | lifetime |
|---|---|---|---|---|
| thumb | 192 | ~96 KiB | the skim and a future gallery | tmpfs, library-wide |
| view | 2560 | ~17.5 MiB | reading at fit, revisits | tmpfs, current book |
| native | full | up to 191 MiB @ 50 MP | zoom only | delivered on want, never stored |

At fit the view image is the display representation; the native is
decoded only when the zoom scale exceeds `VIEW_EDGE / source edge`, and
it is never uploaded on a turn. The view edge must therefore cover the
fitted page's long edge in physical pixels. At 1024 a 1400 px display
asked for a native decode on every settled turn and showed the upscaled
view until it arrived; at 2560 a 2560×1440 window needs no native at
fit. Above a 2560 px fitted page (a 4K single-page view) the native is
requested again; deriving the edge from the viewport is the eventual
fix.
`decoded.width()`/`height()` are the representation's pixel
dimensions; `decoded.source()` is the page's oriented dimensions and is
what layout and the status line use, so a thumb→view promotion never
refits.

## The store (tmpfs)

`$TMPDIR/cryslen-cache/{thumb,view}/<hex xxh3-128 of the encoded page
bytes>`. Content addressing means there is no invalidation step: a
changed page is simply a different key, and identical pages across
books share an entry. The store is tmpfs (on this host `/tmp` is zram),
so it lives for the system's uptime and needs no user-facing cleanup.

The root resolves through `env::temp_dir()`, so a `TMPDIR` set per
app instance shortens the lifetime to that instance; on a normal
desktop session it is `/tmp` (zram here), cleared at boot.

Files carry a `CYPV` header: version, artifact tag, pixel layout
(RGBA8 or YUV with depth/subsampling/range/matrix), the mint edge,
source dims, image dims, `Format`, `Transfer`, then the plane payload.
`load` reconstructs through `Decoded::rgba`/`yuv` and rejects an entry
whose edge is not the current constant, so changing `THUMB_EDGE` or
`VIEW_EDGE` is a miss that re-mints, never a stale size. A corrupt file
is a miss, not a displayed page. Writes go through `<name>.tmp` +
`rename`.

Caps are per kind: 512 MiB of thumbs, 1.5 GiB of views. A save that
crosses its namespace's cap prunes that namespace, oldest mtime first;
`load` touches mtime, so the order approximates last use. The save
returns the entries it evicted, so the worker forgets them in its plan
ledger instead of claiming a pruned preview. A store that cannot be
created (read-only tmp, foreign owner, symlink) degrades to disabled;
previews still work, they just re-mint every session.

## The plan

`Plan::build(anchor, direction, mode, len)` produces one ordered list:
the anchor spread, its neighbours by increasing page distance, the
navigation direction's side first at equal distance, bounded to 512
pages ahead and 64 behind and never wrapping. `head()` is the first
three spreads. This single order answers "what should be decoded next"
for every consumer; there is no pending set, recency ledger, or
per-cache ranking.

## Production (worker)

One thread, two inputs, one store.

- **Wants** (`Want { page, kind }`, ordered by the app): the artifacts
  the app's presentation is missing *right now*. They are served first,
  in the app's order. Wants of one page are grouped so a thumb and a
  view cost one decode. The view is delivered before the thumb: a still
  frame waits on it, and the app's upload budget would otherwise defer
  it behind the thumb by a frame.
- **The plan**: the order in which pages are worth producing into the
  store ahead of time. It never delivers anything. It produces thumbs
  for the window and views for `head()` while `Still`, so the next turn
  is served from the store without decoding.
  A job beyond `head()` is preload, not demand: it starts only after
  the app has been silent for 300 ms, and the burst continues until the
  app speaks again. A reader who is turning pages leaves the worker
  idle between turns, so a want is served without waiting out a
  speculative decode; a reader who pauses still gets the deep preload.
- A want is answered from the store when the plan already produced it;
  otherwise the page is decoded once and every wanted preview is
  minted, persisted, and delivered. The thumb is minted from the view
  (one area pass over 1.5 MP instead of the source), the native is the
  page's own pixels and is never stored.
- Delivery is never assumed. The app sends its want set only when it
  changes; a want that leaves the set and comes back is served again,
  while an unchanged set is not re-fetched. The worker keeps no
  session-long "sent" ledger: `produced` records what this session put
  in the store (or loaded from it) for the plan's benefit, and an
  eviction clears the entry, so the plan rebuilds a page that returns
  to its window. A failed save still counts as produced, so a store
  that refuses writes cannot make the plan spin.
- Failures are reported once per page and skipped by both paths.

`Motion` is one bit from the app. A held navigation key (repeat events)
or two presses closer than 100 ms are a flip; everything else is a
still turn, which waits for the view instead of flashing a thumb.
250 ms of quiet returns to `Still`. It changes what is drawn (a skim
presents whatever is resident, including thumbs) and what the plan
produces (a skim never mints views).

## Presentation (app)

Requested = `spread_for(index)`. A spread is presentable when any of
its pages has any representation uploaded.

1. The requested spread, if presentable.
2. Else the presentable spread closest to the requested that lies
   between the presented and requested spreads; this advances
   monotonically under a held key and never replays a passed page.
3. Else keep the presented spread.

Seats are always the presented spread's pages, so pairing identity is
correct even when one seat is a thumb and the other a view. The seat
dims come from `source()`, never the representation.

Each frame, after retention, the app computes the wants: for the
requested spread and the shown one, the missing **view** (still), the
missing **thumb** (always, so the next skim is free), and the missing
**native** (shown pages, when the placement upscales the view). It
sends the set on change. The app's texture maps are the only authority
on what it can draw; the worker's delivery ledger is not.

`PageTextures` holds three maps and no budget. Retention is a
performance policy: thumbs stay for the session, views for the
presented and requested spreads and their neighbours, natives for the
presented spread only. `kept` also covers every want by construction,
so an artifact must leave the sent set before retention can drop it and
the worker always sees the round trip; if it is needed again, it is
simply asked for again.

Uploads drain from a queue with a per-frame budget; a non-empty queue
arms the next frame. The queue is the only reason a repaint can happen
without visible change.

## What was removed

`Screen`'s hold/commit policy, `page_watched`, `stage_forward`, the
decoded `Book` and its pending trail/recency/eviction, and the GPU
byte budget with distance eviction. Also gone: the worker's `shipped`
ledger (delivery is demand-driven), the `want_view`/`want_native`
single-slot requests, the 250 ms request-echo grace, and the separate
`serve`/`serve_view`/`serve_native` paths. The remaining knobs are the
three edges, the two store caps, the plan bounds, and the two motion
intervals.

## Diagnosing a stall

Logs go to stderr through `ino_tracing` (no timestamps; each line carries
its own duration). The tracing target is the binary's crate name, so use
`RUST_LOG=cy=debug` (not `cryslen`); a bare `RUST_LOG=debug` works too.
The stall lines are `info` and appear without any filter.

- `slow frame` (>50 ms) carries `total`, `drain`, `queue`, `presented`.
- `slow upload` (>30 ms) names the representation, its dims, and the
  GPU insert time.
- `holding for a view` (>250 ms while still) names the requested and
  presented spreads and the queue depth: the app is waiting on the
  worker, not rendering.
- `unpack pass ended`, `slow member read`: archive I/O.
- `turn` (index, gap, repeat, motion): how a navigation was classified.
- `presented` (spreads, `seats`, `took`): what each seat drew (thumb,
  view, native) and how long after the last turn it appeared.
- `requested`: the want set the app just sent.
- `targets`, `target`, `target served`: the worker's queue and the
  served page (`waited` = time since the set arrived, `thumb_hit` /
  `view_hit` = store hits, `elapsed` = serve cost).
- `job` (head or preload), `preload` bursts: what the worker was doing
  around a hold.
- `decoded`, `view minted`, `thumb minted`, `preview load/save/prune`,
  `publish`: per-artifact cost.

## Honest limits

- The mint is roughly 15 ns per source sample in release: ~125 ms for
  an 8 MP page and ~0.75 s for a 50 MP page. Head spreads are minted
  during a reading dwell, so a turn does not pay it; a cold start or a
  jump holds one mint. A debug build is ~30x slower and will look like
  a freeze.
- Minting always costs a full decode: jxl-rs exposes no scaled decode
  (`skip_preview`/`preview_size` only), libavif-sys exposes no
  `avifImageScale`, and zune-jpeg's `max_width` rejects oversized
  images rather than scaling them. A fresh 300-page book therefore
  costs about three minutes of background decode for the window.
- A page far into the window is thumb-decoded, then view-decoded when
  it reaches `head()`. The second decode happens during a reading
  dwell, not on a turn.
- A jump to unprepared content can wait out one in-flight decode plus
  its own; a still frame keeps the previous spread up meanwhile, a skim
  shows the destination's thumb.
- Store pruning is LRU by file mtime, not by distance to the reader.
  The plan rebuilds an evicted page when it returns to its window; a
  want that arrives first pays the decode.