# Cache: representations, prefetch, store

## Roles and layout

- Role selection (`Thumb`, `Fit`, `Full`), demand requests, and texture retention live in `src/presentation.rs`.
- `Thumb` serves skims, `Fit` serves settled reading at session `fit_edge`, and `Full` bypasses persistent caches for zoom (`src/cache.rs`, `src/presentation.rs`).
- Layout and status use page dimensions, so role promotion keeps spread pairing unchanged.
- Animated pages persist only frame 0.

## Store

`$TMPDIR/cryslen-cache/{thumb,view}/<xxh3-128 of encoded page bytes>`, implemented in `src/cache.rs`. `view` stores `Role::Fit`.

- Entries are little-endian and versioned by format version, page hash, role, and target edge; the resampling kernel is outside the key, so existing entries keep their look until pruned.
- Writes lock a unique temp file until atomic rename, allowing store open to sweep orphaned temps; loading refreshes mtime for shared per-namespace LRU caps across child processes. Failures degrade to re-deriving.

## Prefetch and worker

- Spread warm-up ordering (`Prefetch::build`), the priority worker pool, `MemoryCache` (process RAM `Fit` LRU), and `Store` coordination live in `src/worker.rs`.
- Resampling (`src/cache.rs`) averages straight-alpha RGBA because the renderer premultiplies at draw; alpha-aware resampling would avoid edge color bleed at roughly double memory and time.

## Presentation and input

- Presentable spread resolution, fallback order, and GPU upload/retention live in `src/presentation.rs` and `src/render.rs`.
- Skim vs. still motion triggers live in `src/app.rs` (platform key auto-repeat starts a skim; key release or focus loss ends it).

## Limits

A cold page without a cached `Fit` image costs a source decode (`decoding.md`) before display. Slow frames, slow uploads, and long waits log at `info`; cache and worker traces require `RUST_LOG=debug`.
