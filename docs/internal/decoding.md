# Decoding: choices and traps

Measured ratios are from one laptop.

## Backends

- JPEG/PNG/GIF/BMP/TIFF (`src/decode/raster.rs`): `image` crate (`zune-jpeg`; `image`'s PNG backend measured faster than `zune-png` on noise-heavy input). `turbojpeg` scaled decode saves little because entropy decode dominates.
- WebP (`src/decode/webp.rs`): `libwebp-sys`; opaque lossy pages keep planar YUV for the display shader.
- AVIF (`src/decode/avif.rs`): `libavif` + `dav1d` (`libaom` measured ~3x slower); keep YUV planes for the shader to avoid a CPU pass and cut 4:2:0 upload size.
- JXL (`src/decode/jxl.rs`): `jxl` (`jxl-rs`) with a custom parallel runner (`jxl-oxide` measured ~4x slower; the default runner is single-threaded).
- AVIF and JXL cost 1–2 orders of magnitude more per pixel than JPEG.

## Traps

- GIF animation (`src/decode/mod.rs`): `image`'s `AnimationDecoder` already composites frames to the logical screen; read loop counts from `gif::Decoder::repeat()` because `image`'s wrapper reports a missing repeat extension as infinite.
- Size caps: reject zero edges and pixel counts past the allocation ceiling before allocating (`src/decode/mod.rs`); texture edge limits are enforced during derivation (`src/cache.rs`) and upload (`src/render.rs`).

## Out of scope

Reduced-resolution JXL/AVIF decode, hardware AV1/JXL still decode, pure-Rust AV1 decode, external thumbnail caches, and EXIF IFD1 thumbnails.

## Related docs

Scheduling and caches: `docs/internal/cache.md`. Process and IPC boundary: `docs/internal/process.md`.
