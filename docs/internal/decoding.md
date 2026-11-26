# Decoding: choices and traps

Measured ratios are from one laptop.

## Backends

- JPEG/PNG/GIF/BMP/TIFF (`src/image/decode/raster.rs`): `image` crate (`zune-jpeg`; `image`'s PNG backend measured faster than `zune-png` on noise-heavy input). `turbojpeg` scaled decode saves little because entropy decode dominates.
- WebP (`src/image/decode/webp.rs`): `libwebp-sys`; opaque lossy pages keep planar YUV for the display shader.
- AVIF (`src/image/decode/avif.rs`): `libavif` + `dav1d` (`libaom` measured ~3x slower); keep YUV planes for the shader to avoid a CPU pass and cut 4:2:0 upload size.
- JXL (`src/image/decode/jxl.rs`): `jxl` (`jxl-rs`) with a custom parallel runner (`jxl-oxide` measured ~4x slower; the default runner is single-threaded).
- AVIF and JXL cost 1–2 orders of magnitude more per pixel than JPEG.

## Traps

- GIF animation (`src/image/decode/gif.rs`, `src/book/playback.rs`): frame-0 decode inspects subsequent headers via `skip_frame_decoding(true)` to set `PageInfo.animated`, preventing playback worker startup on still GIFs; `image`'s `AnimationDecoder` composites frames to the logical screen, and `gif::Decoder::repeat()` preserves absent-vs-infinite loop counts.
- Size caps: reject zero edges and pixel counts past `MAX_PIXELS` before allocating (`src/image/pixels.rs`); texture edge limits are enforced during derivation (`src/image/resample.rs`) and upload (`src/render/textures.rs`).

## Out of scope

Reduced-resolution JXL/AVIF decode, hardware AV1/JXL still decode, pure-Rust AV1 decode, external thumbnail caches, and EXIF IFD1 thumbnails.

## Related docs

Scheduling and caches: `docs/internal/cache.md`. Concurrency and book lifecycle: `docs/internal/process.md`.
