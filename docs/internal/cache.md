# Cache: representations, plan, store

Reference for `src/preview.rs`, `src/plan.rs`, `src/worker.rs`, and the presentation half of `src/app.rs` and `src/render.rs`. Process ownership and transport are in `docs/internal/process.md`.

## Direction

The parent holds `BookView` and the GPU textures. The child holds one book's source, store, decode, and derivation. The app's texture maps decide what is missing; the worker serves that set and fills the store from the plan.

`Thumb` serves skims, `View` serves still reading at fit, and `Native` is the page's own decode for zoom, bypassing the store. `VIEW_EDGE` must cover the fitted page's long edge. A native is requested only when the page's own long edge and the fitted edge both exceed `VIEW_EDGE`.

`Decoded::width()` and `Decoded::height()` are artifact pixels; `page_dims()` is the page's. Layout and status read `page_dims`, so promotion leaves the layout unchanged.

## Store

`$TMPDIR/cryslen-cache/{thumb,view}/<xxh3-128 of the encoded page bytes>`. Content addressing gives free invalidation and shares pages across books. Every child opens the same tree, so switches keep its entries.

The format is defined in `preview.rs`: little-endian, versioned, with the target edge in the header. A changed edge is a miss that re-derives; a corrupt file is a miss. Writes are atomic (`<hash>.<pid>.tmp` then rename), and `Store::open` sweeps temps left by dead writers.

Entry identity is the format version, page hash, kind, and target edge; the resampling kernel is outside it, so cached entries keep their look until pruned.

Each namespace has its own byte cap and mtime-LRU pruning (`THUMB_CAP_BYTES`, `VIEW_CAP_BYTES`). `load` refreshes mtime; `save` returns evicted hashes so the worker drops them from its ledger. Caps are shared across children, so a prune can evict another child's planned page; that page re-derives on its next want. A failed store degrades to re-deriving.

## Plan

`Plan::build` gives one ordered, bounded list from the anchor: neighbours by page distance, the travel direction first on ties, stopping at the book ends. `head()` is the leading `HEAD_SPREADS`. The plan writes the store; the app's wants are the delivery path.

## Production

One decode thread per book. Wants come first, grouped per page so one decode serves all its artifacts. A want hits the store or decodes once; a missing preview is derived and persisted. Derivation is a SIMD convolution (`fast_image_resize`): a box kernel for thumbs, Hamming for views. RGBA averages straight alpha because the renderer premultiplies at draw; that bleeds color across transparent edges. Alpha-aware resampling removes the bleed but needs a source-sized temporary and roughly double the time. The plan warms ahead; work beyond `head()` is preload and waits for `IDLE_PRELOAD` of app silence. `Motion` picks views for still reading and thumbs for skims. A page-turn key the platform auto-repeats starts a skim; `Released` or focus loss ends it, so the final page's view is wanted in the release frame.

## Presentation

A spread is presentable when every seat has a page fact plus a resident texture, or the page failed. Draw the requested spread when presentable; otherwise the nearest presentable start between shown and requested; otherwise the shown spread. Seats use `page_dims`, so pairing survives promotion.

Each frame recomputes wants for the requested and shown spreads from the texture maps: view and thumb while still, thumb while skimming, native for a shown upscaled page. Send the set on change.

Ending a skim commits `shown` into `requested`, so the thumb being looked at promotes in place. `requested` is lookahead for the plan and wants; `shown` is the last spread on screen and carries its mode, so a pairing change cannot commit it. A still turn moves `requested` ahead; a gesture's unseen turns are discarded at exit.

Retention is role-bounded: thumbs for the book, views within `VIEW_KEEP_SPREADS` of shown or requested, natives for the shown spread. `kept` covers every want, so an artifact leaves the sent set before retention drops it. Uploads drain on a per-frame budget (`UPLOAD_BUDGET_*`); a backlog re-arms the next frame.

## Limits

A missing artifact costs a decode; decoders return the page at its encoded size. A cold page costs its decode before showing.

`slow frame`, `slow upload`, and `holding for a view` log at `info`; others need `RUST_LOG=cryslen=debug`.