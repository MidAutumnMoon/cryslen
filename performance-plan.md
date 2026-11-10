# Performance: large pages

Baseline for decoding and viewing large pages, and the direction the next work
should take. Numbers are one laptop (Ryzen 5 7530U, 12 threads, release
build), so treat them as ratios.

## Baseline

### Browsing session (`/tmp/cy.log`)

The session's pages are all at or below the 2560 view edge, so the view artifact
was a copy and the numbers isolate scheduling rather than decode.

| observation | value |
|---|---|
| still turn, turn → presented | 23.6, 30.5, 41.1, 44.4, 52.0, 61.4, 83.4, 110.2, 183.1 ms (median 52.0) |
| skim turn, turn → presented | 0.3–9.3 ms; repeated 8.5 ms = one 120 Hz frame |
| cached serve (load + publish) | 1.0–2.6 ms |
| store load, 2560 view | 0.95–2.5 ms |
| GPU upload, 2560 view | 1.5–2.9 ms |
| cold serve, page 0 | 58.7 ms = decode 32.6 + view derive 2.3 + view save 7.1 + thumb derive 16.2 |
| worst still stall | 183.1 ms against 178.6 ms of in-flight preload (decode 101.9 + thumb derive 76.7) |

The diagnosis is queuing: a cached demand answers in ~2 ms and presents in one
frame, and every slow still turn coincides with one speculative job holding the
worker. Decoding contributes only to plan warm-up and cold pages.

### Corpus (`/srv/Pool/Pictures/Wallpapers`, 512 files, 363 jpg / 138 png / 10 jpeg, 2.0 GB)

Per page, one decode path equivalent to the app's, plus one derivation at the
2560 view edge (and at the 192 thumb edge) from the decoded page.

| page | decode + RGBA | view derive | thumb derive |
|---|---|---|---|
| 3840×2160 jpg | 55 ms | 90 ms | 28 ms |
| 4851×3213 png | 143 ms | 182 ms | 64 ms |
| 6320×3840 jpg | 328 ms | 208 ms | 86 ms |
| 10080×5640 jpg | 415 ms | 345 ms | 201 ms |
| 13739×7716 png | 524 ms | 539 ms | 355 ms |

The hand-written area filter costs about a decode per view (1.6× at 4k, 0.8× at
10k). Thumbs in the app normally derive from the view, so they cost ~15 ms
there. A still turn on a cold 4k page is therefore ~170 ms of work and on a 10k
page ~790 ms, before IPC and upload.

Store and transport do not explain the stutter: a view save is 7 ms (two copies
of 14.7 MB before the write), a load 2 ms, shared-memory and parent-side copies
~3 ms together, and the frame floor on a 120 Hz panel is 8.5 ms.

### Decoder alternatives, measured

| input | `image` (zune-jpeg) | libjpeg-turbo, RGBA |
|---|---|---|
| 6320×3840 | 328 ms | 257 ms |
| 7680×4320 | 351 ms | 255 ms |
| 10080×5640 | 415 ms | 306 ms |

JPEG scaled decode does not help: decoding a 10080×5640 page directly to 2560
cost 245 ms against 257 ms for the full page, and the usable scale stops at 1/8.
PNG: the pure-Rust backend decodes the 13739×7716 page in 448 ms; `zune-png`
took 643 ms on the same page.

## Direction

### 1. Derivation is the first lever

Landed in xvplzwtv: 3840×2160 view 90 → 12 ms, thumb 28 → 1.2 ms; 10080×5640 view 345 → 22 ms (release, same machine). A cold 4k page prepared in 80 ms end to end.

### 2. Speculative work must not precede delivery

A demand should publish what the frame needs before any look-ahead work. Two
habits in the serving path conflict with that:

- Deriving and saving a thumb before publishing the view. A still frame waits
  on the view; the thumb is a skim artifact.
- Decoding a page to make a thumb when the page's view is already in the store.
  A stored view is the cheap source for a thumb; decoding the page is the
  expensive fallback.

Saving a preview before publishing it repeats that mistake. Send first and
persist after, or make the store write cheap enough that the order stops
mattering.

### 3. A store hit should not read the page

A store hit needs the page's hash alone: once the hash is known for the session,
re-reading and re-hashing the encoded file are unnecessary. For 30 MB pages the
read is 3–13 ms per turn, invisible in the existing serve timing because it
happens before the timer starts. The hash memo already holds the hashes; a read
is only needed for a decode.

### 4. Preload and demand should not share a queue

Turns currently wait out at most one speculative job. Shorter jobs (item 1)
reduce the stall but do not remove it. Two routes, in ascending cost:

- Stretch `IDLE_PRELOAD` past the reader's turn cadence, so the tail only warms
  after the reader has settled. One constant, but it trades skim warm-up and
  leaves a per-hit cost when a turn does land inside a job.
- Put speculative decode on its own thread. The source already tolerates
  concurrent readers and the store already tolerates concurrent writers; a
  shared produced/in-flight ledger and a publish path to the child's single
  writer are still needed. Account for doubled transient decode memory: a 106 MP
  page already peaks well above the artifact ceiling while decoding.

Take the first step now, the second after items 1–3 have been measured; item 1
halves the stall on its own.

### 5. Decoder backend, only if item 1 leaves it dominant

After item 1 the decoder is 60–85% of a cold page, so it becomes the next
lever. libjpeg-turbo is 1.25–1.36× faster on JPEG and skips the RGB→RGBA pass,
at the cost of a native dependency and reimplementing the EXIF orientation
support the current backend provides. Defer it until derivation and scheduling
are fixed, and revisit it only if cold pages remain a problem.

### 6. Viewing, if the policy is allowed to change

A still frame currently holds the previous spread until the view arrives. A
resident thumb for the requested page could be shown in the meantime; thumbs
are usually preloaded, so the page would appear immediately and sharpen. This
reverses the rule in `docs/internal/cache.md` that only a presentable spread is
drawn, so it is a product decision.

## Measured and rejected

- JPEG scaled decode: no gain, capped at 1/8 (`docs/internal/decoding.md`).
- `zune-png`: slower than the current PNG backend on the 13.7k page.
- Store format, IPC transport, GPU upload: single-digit milliseconds, so none
  needs work before the items above.

## Acceptance

Measure with the same corpus and release build:

- A still turn to a cached page stays at the frame floor (one 120 Hz frame).
- A cold 4k page completes in under 90 ms of preparation; a cold 10k page
  under 500 ms.
- No still turn pays a speculative job's full duration.
- Derivation output keeps a hard edge intact and its channels in place, pinned
  by the public derivation test.
