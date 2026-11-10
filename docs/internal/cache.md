# Cache: representations, plan, store

Reference for `src/preview.rs`, `src/plan.rs`, `src/worker.rs`, and the
presentation half of `src/app.rs` and `src/render.rs`. Process ownership
and transport are in `docs/internal/process.md`.

## Direction

The parent holds `BookView` and the GPU textures. The child holds one
book's source, store, decode, and mint. The app's texture maps decide
what is missing; the worker serves that set and fills the store from the
plan.

`Thumb` serves skims, `View` serves still reading at fit, and `Native` is
the page's own decode for zoom, bypassing the store. `VIEW_EDGE` must
cover the fitted page's long edge; a larger fitted page requests a native.
Edge constants live in `preview.rs`.

`Decoded::width()` and `Decoded::height()` are artifact pixels;
`page_dims()` is the page's. Layout and status read `page_dims`, so
promotion leaves the layout unchanged.

## Store

`$TMPDIR/cryslen-cache/{thumb,view}/<xxh3-128 of the encoded page bytes>`.
Content addressing gives free invalidation and shares pages across books.
Every child opens the same tree, so switches keep its entries.

The format is defined in `preview.rs`: little-endian, versioned, with the
mint edge in the header. A changed edge is a miss that re-mints; a corrupt
file is a miss. Writes are atomic (`<hash>.<pid>.tmp` then rename), and
`Store::open` sweeps temps left by dead writers.

Each namespace has its own byte cap and mtime-LRU pruning
(`THUMB_CAP_BYTES`, `VIEW_CAP_BYTES`). `load` refreshes mtime; `save`
returns evicted hashes so the worker drops them from its ledger. Caps are
shared across children, so a prune can evict another child's planned page;
that page re-mints on its next want. A failed store degrades to re-minting.

## Plan

`Plan::build` gives one ordered, bounded list from the anchor: neighbours
by page distance, the travel direction first on ties, stopping at the book
ends. `head()` is the leading `HEAD_SPREADS`. The plan writes the store;
the app's wants are the delivery path.

## Production

One decode thread per book. Wants come first, grouped per page so one
decode serves all its artifacts. A want hits the store or decodes once; a
missing preview is minted and persisted, and the thumb mints from the view
when available. The plan warms ahead; work beyond `head()` is preload and
waits for `IDLE_PRELOAD` of app silence. `Motion` picks views for still
reading and thumbs for skims.

## Presentation

A spread is presentable when every seat has a page fact plus a resident
texture, or the page failed. Draw the requested spread when presentable;
otherwise the nearest presentable start between presented and requested;
otherwise the presented spread. Seats use `page_dims`, so pairing survives
promotion.

Each frame recomputes wants for the requested and presented spreads from
the texture maps: view and thumb while still, thumb while skimming, native
for a shown upscaled page. Send the set on change.

Retention is role-bounded: thumbs for the book, views within
`VIEW_KEEP_SPREADS` of presented or requested, natives for the presented
spread. `kept` covers every want, so an artifact leaves the sent set
before retention drops it. Uploads drain on a per-frame budget
(`UPLOAD_BUDGET_*`); a backlog re-arms the next frame.

## Limits

Minting costs a full decode; decoders return the page at its encoded size.
A cold page costs its decode before showing.

`slow frame`, `slow upload`, and `holding for a view` log at `info`; others
need `RUST_LOG=cryslen=debug`.