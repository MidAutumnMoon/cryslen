# Browsing: open items and what the process boundary closed

The design this review argued for is in place: one ranked plan, a content-addressed preview store, request-driven delivery, and a skim with no replay. See `docs/internal/cache.md`.

Do not revive the rejected mechanisms: six-slot residency, animated skeletons, a pending/replay trail, GPU byte budgets with distance eviction, or insert-before-protect. Response, browsing, and landing are different promises; today they are served by wants, stored thumbs, and the plan's head views.

The model that still explains the limits: with a turn rate `r`, a preparation rate `μ`, and `C` pages already cached,

```
runway = C / (r - μ)    for r > μ
```

At 20 pages/s with 1.8 pages/s of decoding, 60 cached pages buy about 3.3 s of continuous skimming; five seconds of reading prepares roughly nine more. No cache policy can make an unseen page's real colors appear in 30 ms if its decode takes 550 ms: prepare earlier, decode cheaper, or show fewer samples.

## Open

2. **`VIEW_EDGE` is a constant.** 2560 covers a 2560×1440 fit; a fitted page above that asks for a native at fit (`native_wanted`). Derive the edge from physical viewport pixels and key entries by it; the store header already carries the edge, so a changed edge is a miss, not a stale size.

4. **The derived cache lives in `$TMPDIR` and dies at reboot.** Entries survive process restarts, never a reboot. A bounded, versioned persistent preview/view cache would make the first preparation reusable; an explicit "prepare this book" pass is the same lever for unusually expensive books.

5. **`jxl::Pool::run` scopes and joins threads per invocation.** Measure a persistent executor and an alternative backend on the lossless corpus before treating JXL speed as fixed.

6. **Large uploads still run on the UI thread.** The drain budget (16 artifacts / 8 MiB, one artifact always through) bounds it, but a ~190 MiB native can hitch one frame. Not seen in the logs (`slow upload` is instrumented): measure before changing.

## Resolved

Item numbers stay fixed because other documents cite them. Items 1 and 7 are described in `docs/internal/process.md`.

1. **A demand job can wait out at most one book's in-flight decode.** A switch kills the old child, so its decode, archive buffers, and store handles end with it: no switch runs two decodes, and the new book waits only on its own worker. Inside one book a want can still wait for the decode already inside `decode::decode` (up to ~0.75 s for a 50 MP JXL), because the worker is still one thread; a still frame keeps the previous spread up meanwhile. Decoder-side cancellation remains the lever if that in-book wait bites.

3. **A held-key skim ends at the release.** A page-turn key the platform repeats starts a skim; `Released` or focus loss ends it. The final page's view want is sent in the release frame, so the store hit or decode starts there instead of after `SKIM_DECAY`. Rapid taps demand their own view.

7. **`Book::step` no longer opens the source on the UI thread.** It kills the old child, returns, and starts a thread that runs `PendingBookOpen::finish`; the app adopts the result when it arrives, and the old `BookView` and textures stay on screen while it runs. A slow archive listing now delays only the new book, never input or drawing. The session's initial open is still synchronous, a residual limit in `docs/internal/process.md`.

## Considered and left as is

- No hard byte ceiling on the GPU maps: retention is role-bounded (natives for the anchor only, views within `VIEW_KEEP_SPREADS` spread starts of anchor/target, thumbs for the session). Revisit only on OOM.
- Failures are sticky per session, with no retry/backoff. Revisit only if transient I/O errors appear.