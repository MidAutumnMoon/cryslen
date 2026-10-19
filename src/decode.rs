//! Decoding: raw page bytes -> `Decoded` (RGBA or YUV planes).
//!
//! Dispatch is by magic bytes where possible, then extension. JPEG/PNG get
//! EXIF orientation applied. AVIF (feature `avif`) decodes to YUV planes for
//! shader-side color conversion. JXL (feature `jxl`) uses jxl-rs with the
//! parallel runner from research/bench-decode.
//!
//! `unsafe` in cryslen lives in exactly two places: render.rs (GL) and the
//! libavif FFI block below (feature `avif`). Nothing else touches it.
//!
//! All functions here are panics-clean for corrupt input: they return
//! `DecodeError`, never panic (wrap decoder panics with catch_unwind where a
//! crate documents panics; the worker also has a boundary as belt-and-braces).
//! The FFI `unsafe` for AVIF is confined to the `avif` submodule.



/// Decoded page: dimensions plus pixels in one of two layouts. Construct via
/// [`Decoded::rgba`] / [`Decoded::yuv`], which validate that pixel extents
/// match the dimensions — extent bugs surface as worker decode errors here,
/// not as display-time panics.
#[derive(Clone)]
pub struct Decoded {
    pub width: u32,
    pub height: u32,
    pub pixels: Pixels,
    pub format: Format,
}

impl Decoded {
    /// Interleaved RGBA, `4 * w * h` bytes, row-major, top-first.
    pub fn rgba(
        width: u32,
        height: u32,
        format: Format,
        bytes: Box<[u8]>,
    ) -> Result<Decoded, DecodeError> {
        let want = checked_extent(width, height, 4)?;
        if bytes.len() != want {
            return Err(DecodeError::Corrupt("rgba extent"));
        }
        Ok(Decoded { width, height, pixels: Pixels::Rgba8(bytes), format })
    }

    /// Planar YUV sized per [`plane_dims`] for these dimensions. Called only
    /// by the `avif` decoder.
    #[cfg_attr(not(feature = "avif"), allow(dead_code))]
    pub fn yuv(
        width: u32,
        height: u32,
        format: Format,
        yuv: YuvImage,
    ) -> Result<Decoded, DecodeError> {
        let (_, (cw, ch)) = plane_dims((width, height), yuv.subsampling);
        let samples = |p: &Plane| match p {
            Plane::U8(b) => b.len(),
            Plane::U16(b) => b.len(),
        };
        if samples(&yuv.y) != width as usize * height as usize
            || (yuv.subsampling != Subsampling::Mono
                && (samples(&yuv.cb) != cw as usize * ch as usize
                    || samples(&yuv.cr) != cw as usize * ch as usize))
        {
            return Err(DecodeError::Corrupt("yuv plane extent"));
        }
        Ok(Decoded { width, height, pixels: Pixels::Yuv(yuv), format })
    }

    /// Approximate resident size, for cache budget accounting.
    pub fn resident_bytes(&self) -> usize {
        fn plane(p: &Plane) -> usize {
            match p {
                Plane::U8(b) => b.len(),
                Plane::U16(b) => b.len() * 2,
            }
        }
        match &self.pixels {
            Pixels::Rgba8(b) => b.len(),
            Pixels::Yuv(y) => plane(&y.y) + plane(&y.cb) + plane(&y.cr),
        }
    }
}

/// Pixel count with a hostile-header cap (matches the image crate's 512 MiB
/// allocation ceiling) times `bytes_per_pixel`.
fn checked_extent(w: u32, h: u32, bpp: usize) -> Result<usize, DecodeError> {
    let px = (w as u64).checked_mul(h as u64).filter(|&px| px <= MAX_PIXELS as u64);
    match px {
        Some(px) if px > 0 => Ok(px as usize * bpp),
        _ => Err(DecodeError::Corrupt("bad dimensions")),
    }
}

/// Hostile-header cap shared by all decoders, in pixels.
pub const MAX_PIXELS: usize = 512 << 18;

#[derive(Clone)]
pub enum Pixels {
    /// Interleaved RGBA, `4 * width * height` bytes, row-major, top-first.
    Rgba8(Box<[u8]>),
    /// Planar YUV; the renderer converts to RGB in its shader. Constructed
    /// only by the `avif` decoder.
    #[cfg_attr(not(feature = "avif"), allow(dead_code))]
    Yuv(YuvImage),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Jpeg,
    Png,
    Gif,
    Webp,
    Bmp,
    Tiff,
    Avif,
    Jxl,
}

impl Format {
    pub fn label(self) -> &'static str {
        match self {
            Format::Jpeg => "jpeg",
            Format::Png => "png",
            Format::Gif => "gif",
            Format::Webp => "webp",
            Format::Bmp => "bmp",
            Format::Tiff => "tiff",
            Format::Avif => "avif",
            Format::Jxl => "jxl",
        }
    }
}

/// Planar YUV image. 8-bit planes are one byte per sample; 10/12-bit are
/// `u16` samples with `depth` significant bits (normalize by `(1<<depth)-1`).
/// Dimensions ride on [`Decoded`]; plane sizes derive from them.
#[derive(Clone)]
pub struct YuvImage {
    pub depth: u8,
    pub subsampling: Subsampling,
    /// Limited (video) or full (JPEG) range quantization.
    pub full_range: bool,
    pub matrix: Matrix,
    pub y: Plane,
    pub cb: Plane,
    pub cr: Plane,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// Yuv420/Yuv422 arrive only from the `avif` decoder.
#[cfg_attr(not(feature = "avif"), allow(dead_code))]
pub enum Subsampling {
    Mono,
    Yuv420,
    Yuv422,
    Yuv444,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// The BT matrices arrive only from the `avif` decoder.
#[cfg_attr(not(feature = "avif"), allow(dead_code))]
pub enum Matrix {
    Bt601,
    Bt709,
    Bt2020,
    Identity,
}

#[derive(Clone)]
// Plane data in non-`avif` builds arrives only from the image crate as RGBA.
#[cfg_attr(not(feature = "avif"), allow(dead_code))]
pub enum Plane {
    U8(Box<[u8]>),
    U16(Box<[u16]>),
}

/// Plane dims (y, chroma) for an image size and subsampling. Mono has no
/// chroma (renderers substitute a 1x1 midpoint). Pure; testable.
pub fn plane_dims(size: (u32, u32), s: Subsampling) -> ((u32, u32), (u32, u32)) {
    let (w, h) = size;
    match s {
        Subsampling::Mono => ((w, h), (1, 1)),
        Subsampling::Yuv420 => ((w, h), (w.div_ceil(2), h.div_ceil(2))),
        Subsampling::Yuv422 => ((w, h), (w.div_ceil(2), h)),
        Subsampling::Yuv444 => ((w, h), (w, h)),
    }
}

/// Byte view of a plane (u16 planes are little-endian native pairs).
pub fn plane_bytes(p: &Plane) -> &[u8] {
    match p {
        Plane::U8(b) => b,
        Plane::U16(b) => bytemuck::cast_slice(b),
    }
}

#[derive(Debug)]
pub enum DecodeError {
    NotAnImage,
    Corrupt(&'static str),
    Unsupported(&'static str),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::NotAnImage => write!(f, "not an image"),
            DecodeError::Corrupt(what) => write!(f, "corrupt: {what}"),
            DecodeError::Unsupported(what) => write!(f, "unsupported: {what}"),
        }
    }
}

/// Decode one page. `name` is the page filename (extension fallback).
/// Static (first frame of animations; animated decode comes later, if ever).
pub fn decode(bytes: &[u8], name: &str) -> Result<Decoded, DecodeError> {
    let format = match sniff(bytes).or_else(|| format_from_name(name)) {
        Some(format) => format,
        None => return Err(DecodeError::NotAnImage),
    };
    match format {
        #[cfg(feature = "avif")]
        Format::Avif => avif::decode(bytes),
        #[cfg(not(feature = "avif"))]
        Format::Avif => Err(DecodeError::Unsupported("avif (rebuild with --features avif)")),
        #[cfg(feature = "jxl")]
        Format::Jxl => jxl::decode(bytes),
        #[cfg(not(feature = "jxl"))]
        Format::Jxl => Err(DecodeError::Unsupported("jxl (rebuild with --features jxl)")),
        format => decode_image(bytes, format),
    }
}

/// Extension -> format table. Also the listing filter for page sources:
/// one table decides both what can decode and what gets listed.
pub(crate) fn format_from_name(name: &str) -> Option<Format> {
    let ext = std::path::Path::new(name)
        .extension()
        .and_then(|e| e.to_str())?
        .to_ascii_lowercase();
    Some(match ext.as_str() {
        "jpg" | "jpeg" => Format::Jpeg,
        "png" => Format::Png,
        "gif" => Format::Gif,
        "webp" => Format::Webp,
        "bmp" => Format::Bmp,
        "tif" | "tiff" => Format::Tiff,
        "avif" | "avifs" => Format::Avif,
        "jxl" => Format::Jxl,
        _ => return None,
    })
}

/// Magic-byte sniffing. `None` only means "no known signature": the caller
/// falls back to the extension.
fn sniff(b: &[u8]) -> Option<Format> {
    if b.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(Format::Jpeg);
    }
    if b.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some(Format::Png);
    }
    if b.starts_with(b"GIF8") {
        return Some(Format::Gif);
    }
    if b.len() >= 12 && b.starts_with(b"RIFF") && &b[8..12] == b"WEBP" {
        return Some(Format::Webp);
    }
    if b.starts_with(b"BM") {
        return Some(Format::Bmp);
    }
    if b.starts_with(&[0x49, 0x49, 0x2A, 0x00]) || b.starts_with(&[0x4D, 0x4D, 0x00, 0x2A]) {
        return Some(Format::Tiff);
    }
    // JXL: raw codestream or ISOBMFF container signature.
    if b.starts_with(&[0xFF, 0x0A])
        || b.starts_with(&[0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A])
    {
        return Some(Format::Jxl);
    }
    if sniff_ftyp_avif(b) {
        return Some(Format::Avif);
    }
    None
}

/// AVIF lives in an ISOBMFF `ftyp` box; check the major + compatible brands.
fn sniff_ftyp_avif(b: &[u8]) -> bool {
    if b.len() < 12 || &b[4..8] != b"ftyp" {
        return false;
    }
    let box_len = u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let end = box_len.max(12).min(b.len());
    b[8..end].as_chunks::<4>().0.iter().any(|brand| brand == b"avif" || brand == b"avis")
}

/// image-crate path: decode with a concrete decoder so EXIF orientation is
/// readable, then apply it (research/10: the 7-line verified fix).
fn decode_image(bytes: &[u8], format: Format) -> Result<Decoded, DecodeError> {
    use image::ImageDecoder as _;

    let image_format = match format {
        Format::Jpeg => image::ImageFormat::Jpeg,
        Format::Png => image::ImageFormat::Png,
        Format::Gif => image::ImageFormat::Gif,
        Format::Webp => image::ImageFormat::WebP,
        Format::Bmp => image::ImageFormat::Bmp,
        Format::Tiff => image::ImageFormat::Tiff,
        Format::Avif | Format::Jxl => return Err(DecodeError::Unsupported("internal dispatch")),
    };
    let reader = image::ImageReader::with_format(std::io::Cursor::new(bytes), image_format);
    let mut decoder = reader.into_decoder().map_err(|e| image_error(format, e))?;
    let orientation = decoder.orientation().map_err(|e| image_error(format, e))?;
    let mut img = image::DynamicImage::from_decoder(decoder).map_err(|e| image_error(format, e))?;
    img.apply_orientation(orientation);
    let rgba = img.to_rgba8();
    let (width, height) = (rgba.width(), rgba.height());
    Decoded::rgba(width, height, format, rgba.into_raw().into_boxed_slice())
}

/// The frozen error type is `'static`, so the decoder's detail goes to stderr
/// and the page gets a short label.
fn image_error(format: Format, e: image::ImageError) -> DecodeError {
    eprintln!("cy: decode {}: {e}", format.label());
    match e {
        image::ImageError::Unsupported(_) => DecodeError::Unsupported("image variant"),
        _ => DecodeError::Corrupt(format.label()),
    }
}

/// libavif FFI: decoder over the in-memory file, YUV planes copied out
/// directly (avifImageYUVToRGB would cost a full RGB conversion we do on the
/// GPU instead). Ported from research/bench-decode.
#[cfg(feature = "avif")]
mod avif {
    use libavif_sys::*;



    use super::{DecodeError, Decoded, Format, Matrix, Plane, Subsampling, YuvImage};

    /// dav1d parallelism saturates quickly on stills; capped, knob baked in.
    const MAX_THREADS: usize = 8;

    pub fn decode(bytes: &[u8]) -> Result<Decoded, DecodeError> {
        // SAFETY: the decoder is a private per-call object; every libavif call
        // receives either it or pointers read from it. The guard destroys it
        // on every exit path; `bytes` outlives the decoder (destroyed first).
        unsafe {
            let dec = DecoderGuard(avifDecoderCreate());
            if dec.0.is_null() {
                return Err(DecodeError::Unsupported("avif: decoder create"));
            }
            (*dec.0).maxThreads = threads();
            avifDecoderSetIOMemory(dec.0, bytes.as_ptr(), bytes.len());
            check(avifDecoderParse(dec.0), "avif parse")?;
            check(avifDecoderNextImage(dec.0), "avif image")?;

            let img = (*dec.0).image;
            if img.is_null() {
                return Err(DecodeError::Corrupt("avif: no image"));
            }
            let (width, height) = ((*img).width as usize, (*img).height as usize);
            let depth = (*img).depth;
            if !matches!(depth, 8 | 10 | 12 | 16) {
                return Err(DecodeError::Corrupt("avif: bad depth"));
            }
            if (width as u64) * (height as u64) > super::MAX_PIXELS as u64 {
                return Err(DecodeError::Corrupt("avif: bad dimensions"));
            }
            let subsampling = match (*img).yuvFormat {
                AVIF_PIXEL_FORMAT_YUV444 => Subsampling::Yuv444,
                AVIF_PIXEL_FORMAT_YUV422 => Subsampling::Yuv422,
                AVIF_PIXEL_FORMAT_YUV420 => Subsampling::Yuv420,
                AVIF_PIXEL_FORMAT_YUV400 => Subsampling::Mono,
                _ => return Err(DecodeError::Unsupported("avif: pixel format")),
            };
            let full_range = (*img).yuvRange == AVIF_RANGE_FULL;
            let matrix = match (*img).matrixCoefficients as u32 {
                AVIF_MATRIX_COEFFICIENTS_IDENTITY => Matrix::Identity,
                AVIF_MATRIX_COEFFICIENTS_BT709 => Matrix::Bt709,
                AVIF_MATRIX_COEFFICIENTS_BT2020_NCL => Matrix::Bt2020,
                // avifCalcYUVCoefficients treats unspecified/BG/601 as BT.601.
                AVIF_MATRIX_COEFFICIENTS_UNSPECIFIED
                | AVIF_MATRIX_COEFFICIENTS_BT470BG
                | AVIF_MATRIX_COEFFICIENTS_BT601 => Matrix::Bt601,
                _ => return Err(DecodeError::Unsupported("avif: matrix coefficients")),
            };

            let (cw, ch) = match subsampling {
                Subsampling::Mono => (0, 0),
                Subsampling::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
                Subsampling::Yuv422 => (width.div_ceil(2), height),
                Subsampling::Yuv444 => (width, height),
            };
            let (y, cb, cr) = if subsampling == Subsampling::Mono {
                (Plane::U8(Box::new([])), Plane::U8(Box::new([])), Plane::U8(Box::new([])))
            } else {
                (
                    copy_plane((*img).yuvPlanes[0], (*img).yuvRowBytes[0], width, height, depth)?,
                    copy_plane((*img).yuvPlanes[1], (*img).yuvRowBytes[1], cw, ch, depth)?,
                    copy_plane((*img).yuvPlanes[2], (*img).yuvRowBytes[2], cw, ch, depth)?,
                )
            };

            super::Decoded::yuv(
                width as u32,
                height as u32,
                Format::Avif,
                YuvImage {
                    depth: depth as u8,
                    subsampling,
                    full_range,
                    matrix,
                    y,
                    cb,
                    cr,
                },
            )
        }
    }

    fn threads() -> i32 {
        std::thread::available_parallelism().map_or(1, |n| n.get()).min(MAX_THREADS) as i32
    }

    unsafe fn check(code: avifResult, what: &'static str) -> Result<(), DecodeError> {
        if code == AVIF_RESULT_OK {
            return Ok(());
        }
        let detail = std::ffi::CStr::from_ptr(avifResultToString(code)).to_string_lossy();
        eprintln!("cy: {what}: {detail}");
        Err(DecodeError::Corrupt(what))
    }

    /// Copy one plane row by row (libavif rows are padded to `stride`).
    /// Depth > 8 planes are native-endian `u16` arrays on the C side.
    unsafe fn copy_plane(
        ptr: *const u8,
        stride: u32,
        w: usize,
        h: usize,
        depth: u32,
    ) -> Result<Plane, DecodeError> {
        if ptr.is_null() {
            return Err(DecodeError::Corrupt("avif: missing plane"));
        }
        let stride = stride as usize;
        if depth == 8 {
            if stride < w {
                return Err(DecodeError::Corrupt("avif: bad stride"));
            }
            let mut out = Vec::with_capacity(w * h);
            for row in 0..h {
                let src = std::slice::from_raw_parts(ptr.add(row * stride), w);
                out.extend_from_slice(src);
            }
            Ok(Plane::U8(out.into_boxed_slice()))
        } else {
            let bytes = w * 2;
            if stride < bytes {
                return Err(DecodeError::Corrupt("avif: bad stride"));
            }
            let mut out = Vec::with_capacity(w * h);
            for row in 0..h {
                let src = std::slice::from_raw_parts(ptr.add(row * stride) as *const u16, w);
                out.extend_from_slice(src);
            }
            Ok(Plane::U16(out.into_boxed_slice()))
        }
    }

    /// Destroys the decoder on every exit path.
    struct DecoderGuard(*mut avifDecoder);
    impl Drop for DecoderGuard {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { avifDecoderDestroy(self.0) };
            }
        }
    }
}

/// jxl-rs low-level decode with a scoped-thread parallel runner, ported from
/// research/bench-decode (`jxl_rs_decode_mt`). Output is forced to 8-bit RGB
/// (the verified path) and expanded to RGBA here.
#[cfg(feature = "jxl")]
mod jxl {
    use jxl::api::{
        states, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlParallelRunner, JxlParallelRunnerFun, JxlPixelFormat, ProcessingResult,
    };

    use super::{DecodeError, Decoded, Format};

    const MAX_THREADS: usize = 16;
    /// Decode output cap, in pixels: matches the image crate's 512 MiB
    /// allocation ceiling for hostile headers.
    const MAX_PIXELS: usize = 512 << 18;

    pub fn decode(bytes: &[u8]) -> Result<Decoded, DecodeError> {
        let mut runner = Pool(threads());
        let mut input = std::io::BufReader::with_capacity(1 << 20, std::io::Cursor::new(bytes));

        // pass 1: headers -> image size
        let dec = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut dec = match dec.process(&mut input, Some(&mut runner)) {
            Ok(ProcessingResult::Complete { result }) => result,
            Ok(ProcessingResult::NeedsMoreInput { .. }) => {
                return Err(DecodeError::Corrupt("jxl: truncated header"))
            }
            Err(e) => return Err(jxl_error(e)),
        };
        let (w, h) = dec.basic_info().size;
        let pixels = w.checked_mul(h).filter(|&px| px <= MAX_PIXELS);
        let Some(pixels) = pixels else {
            return Err(DecodeError::Corrupt("jxl: bad dimensions"));
        };

        dec.set_pixel_format(JxlPixelFormat {
            color_type: JxlColorType::Rgb,
            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
            extra_channel_format: Vec::new(),
        })
        .map_err(jxl_error)?;

        // pass 2: frame info
        let mut dec = loop {
            match dec.process(&mut input, Some(&mut runner)) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => dec = fallback,
                Err(e) => return Err(jxl_error(e)),
            }
        };

        // pass 3: pixels
        let mut buf = vec![0u8; pixels * 3];
        let mut output = JxlOutputBuffer::new(&mut buf, h, w * 3);
        loop {
            match dec.process(&mut input, std::slice::from_mut(&mut output), Some(&mut runner)) {
                Ok(ProcessingResult::Complete { .. }) => break,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => dec = fallback,
                Err(e) => return Err(jxl_error(e)),
            }
        }

        Decoded::rgba(w as u32, h as u32, Format::Jxl, expand_rgba(&buf))
    }

    fn threads() -> usize {
        std::thread::available_parallelism().map_or(1, |n| n.get()).min(MAX_THREADS)
    }

    fn jxl_error(e: jxl::error::Error) -> DecodeError {
        eprintln!("cy: decode jxl: {e}");
        DecodeError::Corrupt("jxl")
    }

    fn expand_rgba(rgb: &[u8]) -> Box<[u8]> {
        let mut rgba = vec![0u8; rgb.len() / 3 * 4].into_boxed_slice();
        let (pixels, _) = rgba.as_chunks_mut::<4>();
        let (src_pixels, _) = rgb.as_chunks::<3>();
        for (dst, src) in pixels.iter_mut().zip(src_pixels) {
            dst[..3].copy_from_slice(src);
            dst[3] = 0xFF;
        }
        rgba
    }

    /// Scoped-thread runner (from bench-decode). Per-task errors inside the
    /// threads are swallowed; a genuinely broken codestream fails the
    /// sequential `process` call right after.
    struct Pool(usize);
    impl JxlParallelRunner for Pool {
        fn run(&mut self, num: usize, fun: &JxlParallelRunnerFun<'_>) -> jxl::error::Result<()> {
            let n = self.0.min(num);
            if n < 2 {
                return (0..num).try_for_each(fun);
            }
            let chunk = num.div_ceil(n);
            std::thread::scope(|s| {
                for t in 0..n {
                    let fun: &JxlParallelRunnerFun = fun;
                    s.spawn(move || {
                        for i in t * chunk..((t + 1) * chunk).min(num) {
                            let _ = fun(i);
                        }
                    });
                }
            });
            Ok(())
        }
        fn num_threads(&self) -> usize {
            self.0
        }
    }
}

#[cfg(test)]
pub(crate) mod testkit {
    //! Scratch inputs shared by the page/decode/worker unit tests.

    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Solid-color PNG straight from the image encoder.
    pub fn tiny_png(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        let img = image::RgbaImage::from_pixel(w, h, image::Rgba(rgba));
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .expect("encode png");
        buf
    }

    /// Unique temp dir, removed on drop (tests avoid a tempfile dep).
    pub struct TempDir(PathBuf);
    static DIRS: AtomicU64 = AtomicU64::new(0);

    impl TempDir {
        pub fn new(tag: &str) -> TempDir {
            let path = std::env::temp_dir().join(format!(
                "cryslen-{}-{}-{}",
                tag,
                std::process::id(),
                DIRS.fetch_add(1, Ordering::Relaxed),
            ));
            std::fs::create_dir_all(&path).expect("create temp dir");
            TempDir(path)
        }

        pub fn path(&self) -> &Path {
            &self.0
        }

        pub fn write(&self, name: &str, bytes: &[u8]) {
            std::fs::write(self.0.join(name), bytes).expect("write temp file");
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_roundtrip() {
        let png = testkit::tiny_png(4, 3, [10, 20, 30, 255]);
        let page = decode(&png, "p.png").expect("png decodes");
        assert_eq!((page.width, page.height), (4, 3));
        assert_eq!(page.format, Format::Png);
        match page.pixels {
            Pixels::Rgba8(b) => {
                assert_eq!(b.len(), 4 * 4 * 3);
                assert_eq!(&b[..4], &[10, 20, 30, 255]);
            }
            Pixels::Yuv(_) => panic!("png must decode to rgba"),
        }
    }

    // The orientation code path (decoder.orientation + apply_orientation)
    // runs for every image-crate decode above; the encoder cannot write
    // EXIF orientation tags, so a transformed fixture is not synthesizable
    // here. Real-world verification was research/10's exif-check scratch.

    #[test]
    fn jpeg_roundtrip() {
        let img = image::RgbImage::from_pixel(5, 2, image::Rgb([200, 100, 50]));
        let mut jpg = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut jpg), image::ImageFormat::Jpeg)
            .expect("encode jpeg");
        let page = decode(&jpg, "p.jpg").expect("jpeg decodes");
        assert_eq!(page.format, Format::Jpeg);
        assert_eq!((page.width, page.height), (5, 2));
    }

    #[test]
    fn truncated_jpeg_is_corrupt_not_panic() {
        let img = image::RgbImage::from_pixel(32, 32, image::Rgb([1, 2, 3]));
        let mut jpg = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut jpg), image::ImageFormat::Jpeg)
            .unwrap();
        assert!(matches!(decode(&jpg[..20], "p.jpg"), Err(DecodeError::Corrupt(_))));
    }

    #[test]
    fn garbage_is_not_an_image() {
        assert!(matches!(decode(b"hello world", "noext"), Err(DecodeError::NotAnImage)));
        assert!(matches!(decode(b"hello world", "note.txt"), Err(DecodeError::NotAnImage)));
        assert!(matches!(decode(&[], "x"), Err(DecodeError::NotAnImage)));
    }

    #[test]
    fn sniff_magics() {
        assert_eq!(sniff(&[0xFF, 0xD8, 0xFF, 0xE0]), Some(Format::Jpeg));
        assert_eq!(sniff(b"\x89PNG\r\n\x1a\n"), Some(Format::Png));
        assert_eq!(sniff(b"GIF89a"), Some(Format::Gif));
        assert_eq!(sniff(b"RIFF\x00\x00\x00\x00WEBPVP8 "), Some(Format::Webp));
        assert_eq!(sniff(b"BM\x36\x00"), Some(Format::Bmp));
        assert_eq!(sniff(b"II*\x00"), Some(Format::Tiff));
        assert_eq!(sniff(b"\xff\x0a"), Some(Format::Jxl));
        // ISOBMFF ftyp with avif major brand.
        let mut avif = vec![0, 0, 0, 0x14];
        avif.extend_from_slice(b"ftypavif");
        assert_eq!(sniff(&avif), Some(Format::Avif));
        // Non-avif brands (e.g. HEIC's mif1) do not sniff as avif.
        let mut heic = vec![0, 0, 0, 0x14];
        heic.extend_from_slice(b"ftypmif1");
        assert_eq!(sniff(&heic), None);
        assert_eq!(sniff(b"plain text"), None);
    }

    #[test]
    fn extension_fallback_and_table() {
        assert_eq!(format_from_name("a.JPG"), Some(Format::Jpeg));
        assert_eq!(format_from_name("a.tiff"), Some(Format::Tiff));
        assert_eq!(format_from_name("a.avif"), Some(Format::Avif));
        assert_eq!(format_from_name("a.jxl"), Some(Format::Jxl));
        assert_eq!(format_from_name("cover.cbz"), None);
        assert_eq!(format_from_name("noext"), None);
    }

    #[test]
    fn avif_and_jxl_without_features() {
        // No cheap encoder exists for either format in-tree, so only the
        // dispatch decision is testable in default builds: sniffed pages
        // reach the feature gate and come back Unsupported, never panic.
        #[cfg(not(feature = "avif"))]
        assert!(matches!(
            decode(b"\x00\x00\x00\x0cftypavif", "a.avif"),
            Err(DecodeError::Unsupported(_))
        ));
        #[cfg(not(feature = "jxl"))]
        assert!(matches!(decode(&[0xFF, 0x0A, 0x00], "a.jxl"), Err(DecodeError::Unsupported(_))));
        #[cfg(feature = "avif")]
        assert!(matches!(decode(b"\x00\x00\x00\x0cftypavif", "a.avif"), Err(DecodeError::Corrupt(_))));
        #[cfg(feature = "jxl")]
        assert!(matches!(decode(&[0xFF, 0x0A, 0x00], "a.jxl"), Err(DecodeError::Corrupt(_))));
    }
}
