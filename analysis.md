# Browsing: open items, limits, and rejected mechanisms

The design this review argued for is in place: one ranked prefetch, a content-addressed preview store, request-driven delivery, and a skim with no replay. See `docs/internal/cache.md`.

Do not revive the rejected mechanisms: six-slot residency, animated skeletons, a pending/replay trail, GPU byte budgets with distance eviction, or insert-before-protect. Response, browsing, and landing are separate promises served by requests, stored thumbs, and the prefetch's head fit images.

The model that explains the throughput limits: with turn rate `r`, preparation rate `μ`, and `C` pages already cached,

```
runway = C / (r - μ)    for r > μ
```

At 20 pages/s with 1.8 pages/s of decoding, 60 cached pages buy about 3.3 s of continuous skimming; five seconds of reading prepares roughly nine more. No cache policy can make an unseen page's real colors appear in 30 ms if its decode takes 550 ms: prepare earlier, decode cheaper, or show fewer samples.

## Open

- **`FIT_EDGE` is a fixed constant (`2560`).** It covers a 2560×1440 fit; fitted pages above that trigger `full_wanted` for `Role::Full`. Do not key `Fit` by live viewport pixels (window resizes would thrash the on-disk store); use a stable display ceiling or gate `full_wanted` on active zoom (see `systematic-analysis.md`).
- **The derived cache lives in `$TMPDIR` (`env::temp_dir()`) and dies at reboot.** Entries survive process restarts, never a reboot. A bounded, versioned persistent preview/fit cache would make first preparation reusable across reboots; an explicit pre-prepare pass is the same lever for unusually expensive books.

## Limits

Measure in logs before acting on either limit:

- **In-book decode and unpack waits.** Switching kills the old child, ending switch-time decode concurrency (`docs/internal/process.md`). Inside a book, `cy-prefetch` runs off the request loop so cached turns skip prefetch stalls, but a cold request still waits for an active foreground `decode::decode` (up to ~0.75 s for a 50 MP JXL) or for its archive entry to unpack on `cy-unpack`; a still frame holds the previous spread meanwhile. Decoder-side cancellation remains the lever if that in-book wait bites.
- **Large uploads run on the UI thread.** `App::drain_deliveries` bounds batches (`16` images / `8 MiB`, one image always through), though a ~190 MiB full image can still hitch one frame. Instrumented with `slow upload` (≥ 30 ms); measure before moving uploads off-thread.

## Considered and left as is

- No hard byte ceiling on GPU maps: retention is role-bounded (`Role::Full` on screen only, `Role::Fit` within `FIT_KEEP_SPREADS` spread starts of on-screen or target, `Role::Thumb` for the book). Revisit only on OOM.
- Failures are sticky per book, with no retry/backoff. Revisit only if transient I/O errors appear.
