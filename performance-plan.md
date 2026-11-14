# Performance: remaining levers

Derivation (`fast_image_resize`), request/`cy-prefetch` lane separation, hash-before-read store hits, and low-copy store/IPC paths are in place. Decoding dominates cold pages.

JPEG scaled decode and `zune-png` were measured and rejected (`docs/internal/decoding.md`).

## Measured JPEG backend comparison

Single laptop baseline (Ryzen 5 7530U, 12 threads, release build, 512-file `/srv/Pool/Pictures/Wallpapers` corpus); treat numbers as ratios.

| input | `image` (`zune-jpeg`) | `libjpeg-turbo`, RGBA |
|---|---|---|
| 6320×3840 | 328 ms | 257 ms |
| 7680×4320 | 351 ms | 255 ms |
| 10080×5640 | 415 ms | 306 ms |

## Direction (measure before changing)

Check real sessions first (`RUST_LOG=info` for `slow frame`, `slow upload`, `holding for a fit image`; `RUST_LOG=debug` for `target served`, `decoded`):

1. **Decoder backend (`libjpeg-turbo`)**: 1.25–1.36× faster on large JPEGs and skips the RGB→RGBA pass, at the cost of a native dependency and reimplementing EXIF orientation. Revisit only if cold JPEG decodes dominate real browsing delays.
2. **Cold still-turn placeholder**: A still frame holds the previous spread until `Fit` arrives (`docs/internal/cache.md`). Rendering a resident `Thumb` in the meantime would show and sharpen immediately, reversing the settled-spread rule as a product choice if cold decode waits bite.
