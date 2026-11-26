# Cache: roles, thumb pack, worker pool

## Roles and layout

- Role selection (`Thumb`, `Fit`, `Full`), `WorkPlan` construction, and VRAM budget eviction live in `src/reading.rs` and `src/render.rs`; derivation lives in `src/cache.rs`.
- Layout and status use native page dimensions, so role promotion keeps spread pairing unchanged.
- Only thumbnails persist to disk (frame 0 for animated pages).

## Thumb pack (parent-owned)

- One shared `redb` store of lossy WebP thumbnails at `$XDG_CACHE_HOME/cryslen/thumbs-v1.redb` (`src/cache.rs`).
- Safe across concurrent `cy` windows (`MultiWriter`), recreates on corruption or schema mismatch, tracks page presence in RAM so warm opens avoid payload reads, and evicts oldest books by LRU.

## Worker pool (child-owned)

- Fixed worker pool (`src/worker.rs`) over a strict 3-tier `WorkPlan`: `current` preempts `lookahead`, which preempts background `thumbs`.
- Preemption cancels lower-tier decodes cooperatively (`src/decode/jxl.rs`), leaving canceled pages unfaulted.
- A bounded child native cache lets zoom (`Full`) skip re-decoding recent pages; background `thumbs` bypass it.

## Presentation and input

- VRAM holds per-role byte-budgeted caches (`src/reading.rs`, `src/render.rs`).
- Key-repeat skims (`src/app.rs`, `src/reading.rs`) pace turns to one spread ahead of screen, show warm thumbnails immediately, and pause `Fit` work while every seat has a preview so stopping sharpens on an idle CPU.
- Discrete turns wait a brief grace window for `Fit` before falling back to a warm thumbnail, then sharpen in place.
