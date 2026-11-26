# Cache: roles, thumb store, decode pool

## Roles and layout

- Role selection (`Thumb`, `Fit`, `Full`), `WorkPlan` construction, and VRAM budget eviction live in `src/app/reading.rs`; slot storage and eviction live in `src/render/cache.rs` (`SlotTable<Item>`), GPU upload in `src/render/textures.rs`, and derivation in `src/image/resample.rs`.
- Layout and status use `PageInfo` (`dims`, `format`, `animated`), so role promotion keeps spread pairing unchanged; mismatched cached thumbnails are evicted on `Fit`/`Full` arrival and overwritten in the thumb store.
- Only thumbnails persist to disk (frame 0 for animated pages).

## Thumb store

- One shared `redb` database of lossy WebP thumbnails at `$XDG_CACHE_HOME/cryslen/thumbs-v1.redb` (`src/book/thumbs.rs`), opened once per `App` as `ThumbStore` with per-book `ThumbHandle` instances.
- Safe across concurrent `cy` windows (`MultiWriter`), recreates on corruption or schema/table mismatch, tracks page presence in RAM so warm opens skip payload reads, flushes asynchronously via MVCC outside the handle mutex, and prunes oldest books by LRU.

## Decode pool (`src/book/loader.rs`)

- In-process worker pool (`DecodePool`) over a strict 3-tier `WorkPlan`: `current` (`Fit`/`Full` `Request`s) preempts `lookahead`, which preempts background `thumb_anchor` warmup.
- Preemption cancels lower-tier decodes cooperatively (`src/image/decode/jxl.rs`), leaving canceled pages unfaulted; workers purge mimalloc arenas and flush dirty thumbs on exit.
- A bounded native raster cache lets zoom (`Full`) skip re-decoding recent pages; `Thumb` generation after `Fit` derives from the delivered `Fit` image off the critical path.

## Presentation and input

- VRAM holds per-role byte-budgeted caches (`src/app/reading.rs`, `src/render/cache.rs`).
- Key-repeat skims (`src/app/run.rs`, `src/app/reading.rs`) pace turns to one spread ahead of screen, show warm thumbnails immediately, and pause `Fit` work while every seat has a preview so stopping sharpens on an idle CPU.
- Discrete turns wait a brief grace window for `Fit` before falling back to a warm thumbnail, then sharpen in place.
