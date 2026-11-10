# Decoding: choices and traps

Reference for `src/decode/`. Treat the ratios here as one laptop's measurements; benchmark rules are in `build-and-bench.md`.

## Choices

- JPEG and PNG use `image`. JPEG runs through zune-jpeg; its PNG backend measured faster than zune-png on noise-heavy input.
- AVIF uses `libavif-sys` with the dav1d codec (`Cargo.toml`). The libaom wrapper measured about 3x slower.
- JXL uses the `jxl` crate (jxl-rs) with a runner; jxl-oxide measured about 4x slower.
- AVIF and JXL cost one to two orders more per pixel than JPEG. turbojpeg's 1/2..1/8 scaled decode saves little because entropy decode dominates.

## JXL (`jxl.rs`)

- Supply a `JxlParallelRunner`; the default path is single-threaded and several times slower. `Pool` is the scoped-thread model.
- Decode into RGBA via `JxlOutputBuffer` and `JxlPixelFormat::rgba8(extra_channels)`; the crate interleaves channels and fills opaque alpha.
- The whole file is in memory, so `NeedMoreInput` means truncation: report a corrupt page.

## AVIF (`avif.rs`)

- Create the decoder behind an RAII guard, cap `maxThreads`, set IO memory, then parse and take the image.
- Read `yuvPlanes` row by row using `yuvRowBytes`; depth > 8 means native `u16` samples.
- Keep YUV planes and convert in the display shader. `avifImageYUVToRGB` is a second full-frame pass; the shader path also cuts 4:2:0 upload size.
- Map matrix coefficients, range, depth (8/10/12/16), and subsampling (400/420/422/444).

## Guards

`decode/mod.rs` rejects a pixel count over the 512 MiB allocation ceiling and an edge over `DECODE_EDGE_LIMIT` before allocating. The renderer's `MAX_TEXTURE_SIZE` query is the final GPU check.

EXIF orientation: read `decoder.orientation()` before constructing the `DynamicImage`, then `apply_orientation`. Orientation fixtures need real files; the test encoder cannot write EXIF tags.

## Unavailable

Reduced-resolution JXL and AVIF, hardware AV1 or JXL for stills, and a pure-Rust AV1 bitstream decoder are absent. External thumbnail caches and EXIF IFD1 thumbs are out of scope; the viewer decodes its own pages.

## Where to look

`worker.rs` owns wants, plan scheduling, and the panic boundary; `source.rs` owns archive unpack. Scheduling and retention are in `cache.md`; transport is in `process.md`.