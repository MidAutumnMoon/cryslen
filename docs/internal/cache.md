# Cache: representations, prefetch, store

## Direction

The parent holds the book view and the GPU textures. The child holds one book's source, store, decode, and derivation. The app's texture maps decide what is missing; the worker serves that set and fills the store from the prefetch.

Thumbs serve skims, fit images serve settled reading at fit, and full images are the page's own decode for zoom, bypassing the store. The fit edge must cover the fitted page's long edge. A full image is requested only when the page's own long edge and the fitted edge both exceed the fit edge.

A decoded image's dimensions can differ from its page's; layout and status use the page's, so promoting a thumb to a full image leaves the layout unchanged.

## Store

`$TMPDIR/cryslen-cache/{thumb,view}/<xxh3-128 of the encoded page bytes>`. Content addressing gives free invalidation and shares pages across books. Every child opens the same tree, so switches keep its entries. The `view` namespace stores the fit role; the name is kept for existing entries.

Entries are little-endian, versioned, and carry the target edge in the header. A changed edge is a miss that re-derives, as is a corrupt file. Writes are atomic (temp file, then rename); a writer holds an exclusive lock on its temp until the rename, so opening the store sweeps temps no live writer holds.

Entry identity is the format version, page hash, role, and target edge; the resampling kernel is outside it, so cached entries keep their look until pruned.

Each namespace has its own byte cap and mtime-LRU pruning. Loading refreshes an entry's mtime; saving returns the hashes it evicted so the worker drops them from its ledger. Caps are shared across children, so a prune can evict another child's planned page; that page re-derives on its next request. A failed store degrades to re-deriving.

## Prefetch

The prefetch is one ordered, bounded list from the origin: neighbours by page distance, the travel direction first on ties, stopping at the book ends. Its leading spreads are the head, which need fit images rather than thumbs. The list orders the store warm-up; the app's requests are the delivery path.

## Production

One decode thread per book. Requests come first, grouped per page so one decode serves all its roles. A request hits the store or decodes once; a missing preview is derived and persisted. Derivation is a SIMD convolution: a box kernel for thumbs, Hamming for fit images. RGBA averages straight alpha because the renderer premultiplies at draw; that bleeds color across transparent edges. Alpha-aware resampling removes the bleed but needs a source-sized temporary and roughly double the time. The prefetch warms ahead; work beyond the head is preload and waits for a quiet period of app silence. Motion picks fit images for still reading and thumbs for skims. A page-turn key the platform auto-repeats starts a skim; a key release or focus loss ends it.

## Presentation

A spread is presentable when every seat has a page fact plus a resident texture, or the page failed. Draw the target spread when presentable; otherwise the nearest presentable start between shown and target; otherwise the shown spread. Seats use page dimensions, so pairing survives promotion.

Each frame requests the missing roles for the on-screen spread (the shown spread, or the target while nothing is presentable) and a demand spread: the target spread while still, the next spread in the travel direction while skimming. Still frames request fit images and thumbs, plus full for an upscaled on-screen page; skims request thumbs.

Ending a skim commits the shown spread into the target, promoting the looked-at thumb in place. The target is the prefetch's lookahead; the shown spread carries its mode, so a pairing change cannot commit it. A still turn moves the target ahead; unseen turns are discarded at exit.

Retention is derived from the frame: every request, plus role-bounded extras (thumbs for the book, fit images within a few spreads of the on-screen spread or the target, full images for the on-screen spread). Because retention covers every request, a role leaves the sent set before retention drops it. Uploads drain under a per-frame budget; a backlog re-arms the next frame.

## Limits

A missing role costs a decode; decoders return the page at its encoded size. A cold page costs its decode before showing.

Slow frames, slow uploads, and long waits for a view log at info; everything else needs `RUST_LOG=debug`.
