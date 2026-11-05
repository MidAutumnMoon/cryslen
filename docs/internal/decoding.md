# Decoding: which libraries to use and why

Reference for anyone touching decode paths. Speed claims are ratios
measured on one laptop; remeasure before quoting them externally.

## Relative cost of one page turn

- JPEG and PNG are effectively free at any size a viewer meets. The
  `image` crate decodes JPEG through zune-jpeg, so the classic "image
  crate is slow" advice is stale. Its png backend also measured faster
  than zune-png on noise-heavy input.
- AVIF and JXL cost one to two orders of magnitude more per pixel than
  JPEG. A full-resolution photo page takes hundreds of milliseconds,
  not tens. That is inherent to the formats, not the libraries.
- Within that slow class, decoder choice changes the cost by whole
  factors.

## Library selection

- **AVIF: `libavif-sys` with `default-features = false, features =
  ["codec-dav1d"]`.** dav1d is the fast decoder browsers use; the
  `avif-decode` crate wraps libaom and measured about 3x slower. The
  `codec-rav1e` default only adds an encoder the viewer never calls.
- **JXL: the `jxl` crate (jxl-rs).** Pure Rust, maintained by the libjxl
  team, shipped in Chrome and Firefox. With a runner it is roughly 4x
  faster than jxl-oxide; the `libjxl-sys` binding died years ago.
- **turbojpeg scaled decode (1/2, 1/4, 1/8) does not pay for a
  viewer**: the entropy decode dominates, so even 1/8 output costs near
  half a full decode. It only pays if decode-at-display-size lands.

## jxl-rs integration essentials

Three non-obvious points:

1. **You must supply a `JxlParallelRunner`**, or the decoder takes its
   single-thread path, several times slower. A ~20-line scoped-thread
   pool (`std::thread::scope`, static chunking, capped thread count)
   recovers full speed. `jxl-image-rs-integration` hard-codes no runner,
   so it is slow too.
2. Force `JxlPixelFormat::rgba8(basic_info.extra_channels.len())` and
   read into an RGBA buffer via `JxlOutputBuffer`: the decoder
   interleaves RGBA and fills opaque alpha itself, and a file's own
   alpha decodes intact. Higher depths go through `JxlDataFormat`
   (`U16`, `F16`, `F32`).
3. **`NeedsMoreInput` with no bytes left is terminal.** The whole file
   is already in the buffer, so zero remaining bytes means none can
   arrive. Return a corrupt page rather than retrying the same call.

Guard against hostile headers: `width.checked_mul(height)` against a
pixel cap (the image crate's own 512 MiB allocation ceiling) before
allocating, and cap each edge at `DECODE_EDGE_LIMIT`. That is the
texture edge common desktop GPUs support, so an oversized page fails
before the costly decode; the authoritative per-GPU check is the
renderer's runtime query, reported as the page's texture fault.

## libavif/dav1d integration essentials

FFI sequence: `avifDecoderCreate` behind an RAII guard, a modest
`maxThreads`, `avifDecoderSetIOMemory`, `avifDecoderParse`,
`avifDecoderNextImage`, then read `(*image).yuvPlanes` row-by-row
honoring `yuvRowBytes` strides (native-endian `u16` arrays when depth >
8).

**Do not call `avifImageYUVToRGB`.** It is a second full-frame pass,
single-threaded, costing about as much as the entire dav1d bitstream
decode. Ship the Y/Cb/Cr planes to the GPU and convert in the display
shader, which also shrinks texture upload by roughly 2.5x for 4:2:0.
Map `matrixCoefficients` per `avifCalcYUVCoefficients`
(unspecified/BT470BG/BT601 all mean BT.601), keep `yuvRange`, support
8/10/12/16 depths and 400/420/422/444.

## EXIF orientation

`decoder.orientation()` must be read before decoding to a
`DynamicImage`, then applied with `apply_orientation`. Fixtures for it
must come from real files: the encoder in tests cannot write orientation
tags.

## What does not exist (do not search again)

- **Reduced-resolution JXL decode**: no API in libjxl or jxl-rs.
  `djxl -s 8` is a no-op, and responsive-encoded files do not help.
  jxl-oxide's `render_loading_frame` still renders full resolution.
- **AVIF scaled decode**: dav1d has none. Embedded `thmb` items parse
  in avif-parse but almost no encoder writes them, and progressive AVIF
  files are absent in the wild.
- **Hardware AV1 for stills**: decode blocks exist on newer GPUs but are
  video-shaped (mostly 4:2:0 8-bit, no film-grain synthesis, per-session
  setup). Older iGPUs expose no AV1 VA-API profile at all.
- **Hardware JPEG XL**: nothing ships in consumer silicon.
- **Pure-Rust AV1 bitstream decode**: does not exist. Container parsing
  (avif-parse) is pure Rust; the codec is not.

## Decode worker pattern

One worker thread serves the wanted spread; retunes arrive on a channel
and are observed between decodes, so a newer spread overtakes the
backlog and one in-flight decode always completes. Every retune queues
the spread plus its ±3 neighborhood and bounds the pending trail to ±64
pages; the wanted spread itself never drops. Decided and pending pages
live in one book: failures are sticky per page, and eviction takes
never-wanted pages first, then least recently wanted, then farthest from
the spread, never a wanted page. The book's byte budget is in the
hundreds of MiB; an oversized wanted spread stays over budget rather
than losing a shown page.

An archive unpacks once on its own thread; `Source::bytes` answers
`NotYet` until a page is filled, and the claimed unpacked sizes are
guarded against available memory at open. The worker naps 15 ms on
`NotYet`, so an unpack wait never blocks a stop or a newer retune.
Decode runs behind `catch_unwind`: a panicking decoder degrades to a
failed page instead of killing the only worker. After a short grace the
status line holds the last complete spread and says `unpacking…` or
`decoding…`; a spinner would force repaints of a parked loop.

## Thumbnail sources (known, currently unused)

The freedesktop thumbnail cache (`~/.cache/thumbnails`, MD5-of-URI
names, mtime/size freshness) reads and decodes almost instantly and
Dolphin populates it. Unused by deliberate choice ("the file explorer is
the gallery"). EXIF IFD1 embedded thumbnails are the only embedded
previews that exist in real photo collections.
