# Decoding: which libraries to use and why

Reference for anyone touching decode paths. Speed claims are orders of
magnitude and ratios measured on one laptop, not absolutes. Remeasure
on the target machine before quoting numbers externally.

## Relative cost of one page turn

- JPEG and PNG are effectively free at any size a viewer meets. Any
  `image`-crate build since 0.25 decodes JPEG through zune-jpeg, so the
  classic "image crate is slow" advice is stale. The png crate inside
  image also measured faster than zune-png on noise-heavy input, so
  swapping PNG backends buys nothing.
- AVIF and JXL cost one to two orders of magnitude more per pixel than
  JPEG. A full-resolution photo page takes hundreds of milliseconds,
  not tens, on any software decoder. This is inherent to the formats,
  not to the libraries.
- Within that slow class, decoder *choice* moves the needle by whole
  factors. See below.

## Library selection

- **AVIF: `libavif-sys` with `default-features = false, features =
  ["codec-dav1d"]`.** dav1d is the fast decoder (the same one browsers
  use). The `avif-decode` crate wraps libaom and measured about 3x
  slower. The `codec-rav1e` default only adds an encoder the viewer
  never calls.
- **JXL: the `jxl` crate (jxl-rs).** Pure Rust, maintained by the libjxl
  team, shipped as the decoder in Chrome and Firefox. With a runner it
  is roughly 4x faster than jxl-oxide, and the `libjxl-sys` binding
  died years ago. Do not "optimize" back to jxl-oxide.
- **turbojpeg scaled decode (1/2, 1/4, 1/8) does not pay for a
  viewer**: the entropy decode dominates, so even 1/8 output still
  costs near half a full decode. Only attractive if a fit-view
  decode-at-display-size feature ever lands.

## jxl-rs integration essentials

The type-state API needs three `process()` passes (headers, frame info,
pixels). Two non-obvious points:

1. **You must supply a `JxlParallelRunner`** or you get the slow
   single-thread path, several times over. The trait is "run
   `fun(0..num)`, possibly in parallel". A ~20-line scoped-thread pool
   (`std::thread::scope`, static chunking, capped thread count)
   recovers full speed. The `jxl-image-rs-integration` crate hard-codes
   no runner, so it is the slow path too.
2. Force the output format: `JxlPixelFormat { color_type: Rgba,
   color_data_format: U8 { bit_depth: 8 }, extra_channel_format:
   vec![None; basic_info.extra_channels.len()] }`, read into an RGBA
   buffer via `JxlOutputBuffer`. The decoder interleaves RGBA and fills
   opaque alpha itself. A file with its own alpha channel decodes with
   that alpha intact. Supply one `extra_channel_format` slot per extra
   channel. `None` means no separate buffer for it. Request higher bit
   depths through `JxlDataFormat` (`U16`, `F16`, `F32`).

Guard against hostile headers: `width.checked_mul(height)` against a
pixel cap (the image crate's own 512 MiB allocation ceiling) before
allocating.

Each edge is also capped at the texture limit the renderer accepts
(`MAX_TEXTURE_EDGE`). The check runs where a decoder first learns its
dimensions, so an oversized page fails before the costly decode.

## libavif/dav1d integration essentials

FFI sequence: `avifDecoderCreate` behind an RAII guard, a modest
`maxThreads` (dav1d saturates fast on stills, roughly a handful of
threads),
`avifDecoderSetIOMemory`, `avifDecoderParse`, `avifDecoderNextImage`,
then read `(*image).yuvPlanes` row-by-row honoring `yuvRowBytes`
strides (native-endian `u16` arrays when depth > 8).

**Do not call `avifImageYUVToRGB`.** It is a second full-frame pass,
single-threaded, costing about as much as the entire dav1d bitstream
decode. Ship the Y/Cb/Cr planes to the GPU and convert in the display
shader. That also shrinks texture upload by roughly 2.5x for 4:2:0.
Map `matrixCoefficients` per `avifCalcYUVCoefficients`
(unspecified/BT470BG/BT601 all mean BT.601), keep `yuvRange`, support
8/10/12/16 depths and 400/420/422/444.

## EXIF orientation (all image-crate formats)

`ImageReader::with_format(..).into_decoder()` → `decoder.orientation()`
→ `DynamicImage::from_decoder(..)` → `img.apply_orientation(orientation)`
→ `into_rgba8()`. Zero extra dependencies. The encoder in tests cannot
write orientation tags, so fixtures for it must come from real files.

## What does not exist (do not search again)

- **Reduced-resolution JXL decode**: no API in libjxl or jxl-rs.
  `djxl -s 8` is a no-op and responsive-encoded files do not help.
  jxl-oxide's partial-decode `render_loading_frame` still renders full
  resolution.
- **AVIF scaled decode**: dav1d has none. Embedded `thmb` items parse
  in avif-parse but essentially no encoder writes them, and progressive
  AVIF files are absent in the wild.
- **Hardware AV1 for stills**: decode blocks exist on newer GPUs but
  are video-shaped (mostly 4:2:0 8-bit, no film-grain synthesis,
  per-session setup). The wrong tool for a pager. Older iGPUs expose no
  AV1 VA-API profile at all.
- **Hardware JPEG XL**: nothing ships in consumer silicon.
- **Pure-Rust AV1 bitstream decode**: does not exist. Container parsing
  (avif-parse) is pure Rust, the codec is not.

## Decode worker pattern (what makes skimming feel right)

- One worker thread, mpsc retune channel carrying the wanted `Spread`.
  Every retune joins the spread to a pending trail with its ±3
  neighborhood. The neighborhood covers the full next spread in the
  double-page modes. The trail is bounded to ±64 pages of the spread's
  first page. Past that the reader has moved on, so pages are dropped
  instead of queued. The wanted spread itself never drops. The next
  decode is the undecided pending page nearest the spread, with pages
  ahead of the reader beating pages behind at equal distance. The
  channel is observed between decodes, so a newer spread overtakes the
  backlog. One in-flight decode always completes, and no cancellation
  tokens are needed.
- Decided pages and the pending trail live in one book under one lock.
  Stored views are `SeatView::Ready(Arc<Decoded>) | SeatView::Failed(String)`,
  with `Pending` synthesized for snapshots. Failures are
  sticky per page, so an error follows its page across turns.
- Byte budget in the hundreds of MiB. Eviction victims are never-wanted
  pages first, then least recently wanted, then farthest from the
  spread. The wanted spread is never evicted. Each wanted page is
  decided or pending at every instant. An oversized spread stays over
  budget rather than losing a shown page.
- The app reads one snapshot per frame, giving each seat Ready, Failed,
  or Pending. There is no unknown state.
- Wrap decode in `catch_unwind`: a panicking decoder must degrade to a
  `Failed` page, not kill the only worker.
- Per-page latency is decode-bound, so hold-last-frame plus a status
  flag after a short grace period beats any spinner. A spinner forces
  repaints of a parked loop.

## Thumbnail sources (known, currently unused)

The freedesktop thumbnail cache (`~/.cache/thumbnails`, MD5-of-URI
names, mtime/size freshness fields) reads + decodes essentially
instantly and Dolphin populates it. Unused by deliberate choice here
("the file explorer is the gallery"), though the mechanics are sound if
ever wanted. EXIF IFD1 embedded thumbnails are the only embedded-preview
ecology that exists in real photo collections.
