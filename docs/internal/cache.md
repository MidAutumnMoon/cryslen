# Cache: representations, prefetch, store

Reference for `src/cache/` (store, prefetch, presentation), `src/worker.rs`, and `src/render.rs`. Process ownership and transport are in `docs/internal/process.md`.

## Direction

The parent holds `BookView` and the GPU textures. The child holds one book's source, store, decode, and derivation. The app's texture maps decide what is missing; the worker serves that set and fills the store from the prefetch.

`Role::Thumb` serves skims, `Role::Fit` serves still reading at fit, and `Role::Full` is the page's own decode for zoom, bypassing the store. `FIT_EDGE` must cover the fitted page's long edge. A full image is requested only when the page's own long edge and the fitted edge both exceed `FIT_EDGE`.

`Decoded::width()` and `Decoded::height()` are image pixels; `page_dims()` is the page's. Layout and status read `page_dims`, so promotion leaves the layout unchanged.

## Store

`$TMPDIR/cryslen-cache/{thumb,view}/<xxh3-128 of the encoded page bytes>`. Content addressing gives free invalidation and shares pages across books. Every child opens the same tree, so switches keep its entries. The `view` namespace stores the fit role; the name is kept for existing entries.

The format is defined in `cache/mod.rs`: little-endian, versioned, with the target edge in the header. A changed edge is a miss that re-derives; a corrupt file is a miss. Writes are atomic (`<hash>.<pid>.tmp` then rename), and `Store::open` sweeps temps left by dead writers.

Entry identity is the format version, page hash, role, and target edge; the resampling kernel is outside it, so cached entries keep their look until pruned.

Each namespace has its own byte cap and mtime-LRU pruning (`THUMB_CAP_BYTES`, `FIT_CAP_BYTES`). `load` refreshes mtime; `save` returns evicted hashes so the worker drops them from its ledger. Caps are shared across children, so a prune can evict another child's planned page; that page re-derives on its next request. A failed store degrades to re-deriving.

## Prefetch

`Prefetch::build` gives one ordered, bounded list from the origin: neighbours by page distance, the travel direction first on ties, stopping at the book ends. `head()` is the leading `HEAD_SPREADS`. The prefetch list orders the store warm-up; the app's requests are the delivery path.

## Production

One decode thread per book. Requests come first, grouped per page so one decode serves all its roles. A request hits the store or decodes once; a missing preview is derived and persisted. Derivation is a SIMD convolution (`fast_image_resize`): a box kernel for thumbs, Hamming for fit images. RGBA averages straight alpha because the renderer premultiplies at draw; that bleeds color across transparent edges. Alpha-aware resampling removes the bleed but needs a source-sized temporary and roughly double the time. The prefetch list warms ahead; work beyond `head()` is preload and waits for `IDLE_PRELOAD` of app silence. `Motion` picks fit images for still reading and thumbs for skims. A page-turn key the platform auto-repeats starts a skim; `Released` or focus loss ends it, so the final page's fit image is requested in the release frame.

## Presentation

A spread is presentable when every seat has a page fact plus a resident texture, or the page failed. Draw the target spread when presentable; otherwise the nearest presentable start between shown and target; otherwise the shown spread. Seats use `page_dims`, so pairing survives promotion.

Each frame requests the missing roles for the on-screen spread (the shown spread, or the target while nothing is presentable) and a demand spread: the target spread while still, the next spread in the travel direction while skimming. Still frames request fit images and thumbs, plus full for an upscaled on-screen page; skims request thumbs.

Ending a skim commits the shown spread into the target, promoting the looked-at thumb in place. The target is the prefetch list's lookahead; the shown spread carries its mode, so a pairing change cannot commit it. A still turn moves the target ahead; unseen turns are discarded at exit.

Retention is derived from the frame: every request, plus role-bounded extras (thumbs for the book, fit images within `FIT_KEEP_SPREADS` of the on-screen spread or the target, full images for the on-screen spread). Because retention covers every request, a role leaves the sent set before retention drops it. Uploads drain on a per-frame budget (`UPLOAD_BUDGET_*`); a backlog re-arms the next frame.

## Limits

A missing role costs a decode; decoders return the page at its encoded size. A cold page costs its decode before showing.

`slow frame`, `slow upload`, and `holding for a view` log at `info`; others need `RUST_LOG=cryslen=debug`.