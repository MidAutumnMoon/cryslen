# Decoding: choices and traps

The ratios here are one laptop's measurements; benchmark rules are in `build-and-bench.md`.

## Choices

- JPEG and PNG use the `image` crate. JPEG runs through zune-jpeg; its PNG backend measured faster than zune-png on noise-heavy input.
- AVIF uses libavif with the dav1d codec. The libaom wrapper measured about 3x slower.
- JXL uses the jxl crate (jxl-rs) with a parallel runner; jxl-oxide measured about 4x slower.
- AVIF and JXL cost one to two orders more per pixel than JPEG. turbojpeg's scaled decode saves little because entropy decode dominates.

## JXL

- Supply a parallel runner; the default path is single-threaded and several times slower. Scoped threads are the runner model.
- Decode into RGBA with an extra alpha channel; the crate interleaves channels and fills opaque alpha.
- The whole file is in memory, so a request for more input means truncation: report a corrupt page.

## AVIF

- Create the decoder behind an RAII guard, cap the thread count, set the IO memory limit, then parse and take the image.
- Read the planes row by row using each row's stride; depth above 8 bits means native 16-bit samples.
- Keep YUV planes and convert in the display shader. A full-frame YUV-to-RGB conversion is a second pass over the image; the shader path also cuts 4:2:0 upload size.
- Map matrix coefficients, range, depth (8/10/12/16), and subsampling (400/420/422/444).

## Animation

- GIF playback uses the `image` crate's `AnimationDecoder`, which composites frames to the logical screen. Do not add a second compositor.
- Loop counts come from `gif::Decoder::repeat()`; the `image` wrapper reports a file without the extension as infinite.

## Guards

The decode entry point rejects a pixel count over the 512 MiB allocation ceiling and an edge over the decoder's limit before allocating. The renderer's maximum texture size is the final GPU check.

EXIF orientation: read the orientation before constructing the decoded image, then apply it. Orientation fixtures need real files; the test encoder cannot write EXIF tags.

## Unavailable

Reduced-resolution JXL and AVIF, hardware AV1 or JXL for stills, and a pure-Rust AV1 bitstream decoder are absent. External thumbnail caches and EXIF IFD1 thumbs are out of scope; the viewer decodes its own pages.

## Division

The child owns the message loop, requests, prefetch scheduling, and the panic boundary; the source owns archive materialization. Scheduling and retention are in `cache.md`; transport is in `process.md`.