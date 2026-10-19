//! Decoding: raw page bytes -> `Decoded` (RGBA or YUV planes).
//!
//! Dispatch is by magic bytes where possible, then extension. JPEG/PNG get
//! EXIF orientation applied. AVIF (feature `avif`) decodes to YUV planes for
//! shader-side color conversion. JXL (feature `jxl`) uses jxl-rs with a
//! scoped-thread parallel runner.
//!
//! All functions here are panics-clean for corrupt input: they return
//! `DecodeError`, never panic. Decoder panics are caught with catch_unwind
//! where a crate documents them. The worker adds its own boundary as
//! belt-and-braces. The FFI `unsafe` for AVIF is confined to the `avif`
//! submodule.

/// Decoded page: dimensions plus pixels in one of two layouts. Construct via
/// [`Decoded::rgba`] / [`Decoded::yuv`], which validate that pixel extents
/// match the dimensions. Extent bugs surface as worker decode errors here,
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
        check_dims(width, height)?;
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
        check_dims(width, height)?;
        let (_, (cw, ch)) = plane_dims((width, height), yuv.subsampling);
        let samples = |p: &Plane| match p {
            Plane::U8(bytes) => bytes.len(),
            Plane::U16(bytes) => bytes.len(),
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
                Plane::U8(bytes) => bytes.len(),
                Plane::U16(bytes) => bytes.len() * 2,
            }
        }
        match &self.pixels {
            Pixels::Rgba8(bytes) => bytes.len(),
            Pixels::Yuv(y) => plane(&y.y) + plane(&y.cb) + plane(&y.cr),
        }
    }
}

/// Pixel count with a hostile-header cap (matches the image crate's 512 MiB
/// allocation ceiling) times `bytes_per_pixel`.
fn checked_extent(width: u32, height: u32, bytes_per_pixel: usize) -> Result<usize, DecodeError> {
    let pixel_count = (width as u64)
        .checked_mul(height as u64)
        .filter(|&count| count <= MAX_PIXELS as u64);
    match pixel_count {
        Some(count) if count > 0 => Ok(count as usize * bytes_per_pixel),
        _ => Err(DecodeError::Corrupt("bad dimensions")),
    }
}

/// Hostile-header cap shared by all decoders, in pixels.
pub const MAX_PIXELS: usize = 512 << 18;

/// Largest page edge accepted as one GL texture (the common desktop limit).
/// Bigger edges fail as sticky page errors.
const MAX_TEXTURE_EDGE: u32 = 16384;

fn check_dims(width: u32, height: u32) -> Result<(), DecodeError> {
    if width > MAX_TEXTURE_EDGE || height > MAX_TEXTURE_EDGE {
        return Err(DecodeError::Unsupported("page edge over the GPU texture limit"));
    }
    Ok(())
}

#[derive(Clone)]
pub enum Pixels {
    /// Interleaved RGBA, `4 * width * height` bytes, row-major, top-first.
    Rgba8(Box<[u8]>),
    /// Planar YUV, converted to RGB in the renderer's shader. Constructed
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

/// Planar YUV image. 8-bit planes are one byte per sample. 10/12-bit planes
/// are `u16` samples with `depth` significant bits (normalize by
/// `(1<<depth)-1`). Dimensions ride on [`Decoded`]. Plane sizes derive from
/// them.
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
/// chroma (renderers substitute a 1x1 midpoint).
pub fn plane_dims(size: (u32, u32), s: Subsampling) -> ((u32, u32), (u32, u32)) {
    let (width, height) = size;
    match s {
        Subsampling::Mono => ((width, height), (1, 1)),
        Subsampling::Yuv420 => ((width, height), (width.div_ceil(2), height.div_ceil(2))),
        Subsampling::Yuv422 => ((width, height), (width.div_ceil(2), height)),
        Subsampling::Yuv444 => ((width, height), (width, height)),
    }
}

/// Byte view of a plane (u16 planes are little-endian native pairs).
pub fn plane_bytes(plane: &Plane) -> &[u8] {
    match plane {
        Plane::U8(bytes) => bytes,
        Plane::U16(words) => bytemuck::cast_slice(words),
    }
}

#[derive(Debug)]
pub enum DecodeError {
    NotAnImage,
    Corrupt(&'static str),
    Unsupported(&'static str),
    /// `image`-crate failure, labelled with the format it was decoded as.
    Image(Format, image::ImageError),
    #[cfg(feature = "jxl")]
    Jxl(::jxl::error::Error),
    /// libavif call failure: what step failed and the avifResult detail.
    #[cfg(feature = "avif")]
    Avif(&'static str, String),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::NotAnImage => write!(f, "not an image"),
            DecodeError::Corrupt(what) => write!(f, "corrupt: {what}"),
            DecodeError::Unsupported(what) => write!(f, "unsupported: {what}"),
            DecodeError::Image(format, e) => write!(f, "{}: {e}", format.label()),
            #[cfg(feature = "jxl")]
            DecodeError::Jxl(e) => write!(f, "jxl: {e}"),
            #[cfg(feature = "avif")]
            DecodeError::Avif(what, detail) => write!(f, "{what}: {detail}"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Decode one page. `name` is the page filename (extension fallback).
/// Static: animated images show their first frame.
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
fn sniff(bytes: &[u8]) -> Option<Format> {
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(Format::Jpeg);
    }
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some(Format::Png);
    }
    if bytes.starts_with(b"GIF8") {
        return Some(Format::Gif);
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return Some(Format::Webp);
    }
    if bytes.starts_with(b"BM") {
        return Some(Format::Bmp);
    }
    if bytes.starts_with(&[0x49, 0x49, 0x2A, 0x00]) || bytes.starts_with(&[0x4D, 0x4D, 0x00, 0x2A]) {
        return Some(Format::Tiff);
    }
    // JXL: raw codestream or ISOBMFF container signature.
    if bytes.starts_with(&[0xFF, 0x0A])
        || bytes.starts_with(&[0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A])
    {
        return Some(Format::Jxl);
    }
    if sniff_ftyp_avif(bytes) {
        return Some(Format::Avif);
    }
    None
}

/// AVIF files start with an ISOBMFF `ftyp` box.
fn sniff_ftyp_avif(bytes: &[u8]) -> bool {
    if bytes.len() < 12 || &bytes[4..8] != b"ftyp" {
        return false;
    }
    let box_len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let end = box_len.max(12).min(bytes.len());
    bytes[8..end].as_chunks::<4>().0.iter().any(|brand| brand == b"avif" || brand == b"avis")
}

/// image-crate path: decode with a concrete decoder so EXIF orientation is
/// readable, then applies it.
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
    let mut decoder = reader.into_decoder().map_err(|e| DecodeError::Image(format, e))?;
    // Dimensions are known before the pixel pass. Orientation swaps at most
    // w and h, and the edge cap checks both, so this pre-check is complete.
    let (width, height) = decoder.dimensions();
    check_dims(width, height)?;
    let orientation = decoder.orientation().map_err(|e| DecodeError::Image(format, e))?;
    let mut oriented =
        image::DynamicImage::from_decoder(decoder).map_err(|e| DecodeError::Image(format, e))?;
    oriented.apply_orientation(orientation);
    let rgba = oriented.to_rgba8();
    let (width, height) = (rgba.width(), rgba.height());
    Decoded::rgba(width, height, format, rgba.into_raw().into_boxed_slice())
}

/// libavif FFI: decoder over the in-memory file, YUV planes copied out
/// directly (avifImageYUVToRGB would cost a full RGB conversion we do on the
/// GPU instead.
#[cfg(feature = "avif")]
mod avif {
    use libavif_sys::*;



    use super::{DecodeError, Decoded, Format, Matrix, Plane, Subsampling, YuvImage};

    /// dav1d parallelism saturates quickly on stills, so the thread count
    /// is capped (knob baked in).
    const MAX_THREADS: usize = 8;

    pub fn decode(bytes: &[u8]) -> Result<Decoded, DecodeError> {
        // SAFETY: the decoder is a private per-call object; every libavif call
        // receives either it or pointers read from it. The guard destroys it
        // on every exit path; `bytes` outlives the decoder (destroyed first).
        unsafe {
            let decoder_guard = DecoderGuard(avifDecoderCreate());
            if decoder_guard.0.is_null() {
                return Err(DecodeError::Unsupported("avif: decoder create"));
            }
            (*decoder_guard.0).maxThreads = threads();
            avifDecoderSetIOMemory(decoder_guard.0, bytes.as_ptr(), bytes.len());
            check(avifDecoderParse(decoder_guard.0), "avif parse")?;
            check(avifDecoderNextImage(decoder_guard.0), "avif image")?;

            let avif_image_ptr = (*decoder_guard.0).image;
            if avif_image_ptr.is_null() {
                return Err(DecodeError::Corrupt("avif: no image"));
            }
            let (width, height) =
                ((*avif_image_ptr).width as usize, (*avif_image_ptr).height as usize);
            let depth = (*avif_image_ptr).depth;
            if !matches!(depth, 8 | 10 | 12 | 16) {
                return Err(DecodeError::Corrupt("avif: bad depth"));
            }
            if (width as u64) * (height as u64) > super::MAX_PIXELS as u64 {
                return Err(DecodeError::Corrupt("avif: bad dimensions"));
            }
            super::check_dims(width as u32, height as u32)?;
            let subsampling = match (*avif_image_ptr).yuvFormat {
                AVIF_PIXEL_FORMAT_YUV444 => Subsampling::Yuv444,
                AVIF_PIXEL_FORMAT_YUV422 => Subsampling::Yuv422,
                AVIF_PIXEL_FORMAT_YUV420 => Subsampling::Yuv420,
                AVIF_PIXEL_FORMAT_YUV400 => Subsampling::Mono,
                _ => return Err(DecodeError::Unsupported("avif: pixel format")),
            };
            let full_range = (*avif_image_ptr).yuvRange == AVIF_RANGE_FULL;
            let matrix = match (*avif_image_ptr).matrixCoefficients as u32 {
                AVIF_MATRIX_COEFFICIENTS_IDENTITY => Matrix::Identity,
                AVIF_MATRIX_COEFFICIENTS_BT709 => Matrix::Bt709,
                AVIF_MATRIX_COEFFICIENTS_BT2020_NCL => Matrix::Bt2020,
                // avifCalcYUVCoefficients treats unspecified/BG/601 as BT.601.
                AVIF_MATRIX_COEFFICIENTS_UNSPECIFIED
                | AVIF_MATRIX_COEFFICIENTS_BT470BG
                | AVIF_MATRIX_COEFFICIENTS_BT601 => Matrix::Bt601,
                _ => return Err(DecodeError::Unsupported("avif: matrix coefficients")),
            };

            let (chroma_width, chroma_height) = match subsampling {
                Subsampling::Mono => (0, 0),
                Subsampling::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
                Subsampling::Yuv422 => (width.div_ceil(2), height),
                Subsampling::Yuv444 => (width, height),
            };
            let (luma_plane, blue_plane, red_plane) = if subsampling == Subsampling::Mono {
                (Plane::U8(Box::new([])), Plane::U8(Box::new([])), Plane::U8(Box::new([])))
            } else {
                (
                    copy_plane(
                        (*avif_image_ptr).yuvPlanes[0],
                        (*avif_image_ptr).yuvRowBytes[0],
                        width,
                        height,
                        depth,
                    )?,
                    copy_plane(
                        (*avif_image_ptr).yuvPlanes[1],
                        (*avif_image_ptr).yuvRowBytes[1],
                        chroma_width,
                        chroma_height,
                        depth,
                    )?,
                    copy_plane(
                        (*avif_image_ptr).yuvPlanes[2],
                        (*avif_image_ptr).yuvRowBytes[2],
                        chroma_width,
                        chroma_height,
                        depth,
                    )?,
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
                    y: luma_plane,
                    cb: blue_plane,
                    cr: red_plane,
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
        Err(DecodeError::Avif(what, detail.into_owned()))
    }

    /// Copy one plane row by row (libavif rows are padded to `stride`).
    /// Depth > 8 planes are native-endian `u16` arrays on the C side.
    unsafe fn copy_plane(
        plane_ptr: *const u8,
        row_stride: u32,
        width: usize,
        height: usize,
        depth: u32,
    ) -> Result<Plane, DecodeError> {
        if plane_ptr.is_null() {
            return Err(DecodeError::Corrupt("avif: missing plane"));
        }
        let row_stride = row_stride as usize;
        if depth == 8 {
            if row_stride < width {
                return Err(DecodeError::Corrupt("avif: bad stride"));
            }
            let mut out = Vec::with_capacity(width * height);
            for row in 0..height {
                let src = std::slice::from_raw_parts(plane_ptr.add(row * row_stride), width);
                out.extend_from_slice(src);
            }
            Ok(Plane::U8(out.into_boxed_slice()))
        } else {
            let bytes = width * 2;
            if row_stride < bytes {
                return Err(DecodeError::Corrupt("avif: bad stride"));
            }
            let mut out = Vec::with_capacity(width * height);
            for row in 0..height {
                let src = std::slice::from_raw_parts(
                    plane_ptr.add(row * row_stride) as *const u16,
                    width,
                );
                out.extend_from_slice(src);
            }
            Ok(Plane::U16(out.into_boxed_slice()))
        }
    }

    struct DecoderGuard(*mut avifDecoder);
    impl Drop for DecoderGuard {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { avifDecoderDestroy(self.0) };
            }
        }
    }
}

/// jxl-rs low-level decode with a scoped-thread parallel runner. Output is
/// requested as 8-bit RGB and expanded to RGBA here.
#[cfg(feature = "jxl")]
mod jxl {
    use jxl::api::{
        states, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlParallelRunner, JxlParallelRunnerFun, JxlPixelFormat, ProcessingResult,
    };

    use super::{DecodeError, Decoded, Format};

    const MAX_THREADS: usize = 16;

    /// Buffered read chunk for the codestream input.
    const READ_CHUNK_BYTES: usize = 1 << 20;

    pub fn decode(bytes: &[u8]) -> Result<Decoded, DecodeError> {
        let mut runner = Pool(threads());
        let mut input =
            std::io::BufReader::with_capacity(READ_CHUNK_BYTES, std::io::Cursor::new(bytes));

        // pass 1: headers -> image size
        let decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut decoder = match decoder.process(&mut input, Some(&mut runner)) {
            Ok(ProcessingResult::Complete { result }) => result,
            Ok(ProcessingResult::NeedsMoreInput { .. }) => {
                return Err(DecodeError::Corrupt("jxl: truncated header"))
            }
            Err(e) => return Err(DecodeError::Jxl(e)),
        };
        let (width, height) = decoder.basic_info().size;
        let pixels =
            width.checked_mul(height).filter(|&px| px <= super::MAX_PIXELS);
        let Some(pixel_count) = pixels else {
            return Err(DecodeError::Corrupt("jxl: bad dimensions"));
        };
        super::check_dims(width as u32, height as u32)?;

        decoder.set_pixel_format(JxlPixelFormat {
            color_type: JxlColorType::Rgb,
            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
            extra_channel_format: Vec::new(),
        })
        .map_err(DecodeError::Jxl)?;

        // pass 2: frame info
        let mut decoder = loop {
            match decoder.process(&mut input, Some(&mut runner)) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
                Err(e) => return Err(DecodeError::Jxl(e)),
            }
        };

        // pass 3: pixels
        let mut rgb_buffer = vec![0u8; pixel_count * 3];
        let mut output = JxlOutputBuffer::new(&mut rgb_buffer, height, width * 3);
        loop {
            match decoder
                .process(&mut input, std::slice::from_mut(&mut output), Some(&mut runner))
            {
                Ok(ProcessingResult::Complete { .. }) => break,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
                Err(e) => return Err(DecodeError::Jxl(e)),
            }
        }

        Decoded::rgba(width as u32, height as u32, Format::Jxl, expand_rgba(&rgb_buffer))
    }

    fn threads() -> usize {
        std::thread::available_parallelism().map_or(1, |n| n.get()).min(MAX_THREADS)
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

    /// Scoped-thread runner. Per-task errors inside the threads are
    /// swallowed. A genuinely broken codestream fails the sequential
    /// `process` call right after.
    struct Pool(usize);
    impl JxlParallelRunner for Pool {
        fn run(&mut self, num: usize, fun: &JxlParallelRunnerFun<'_>) -> jxl::error::Result<()> {
            let thread_count = self.0.min(num);
            if thread_count < 2 {
                return (0..num).try_for_each(fun);
            }
            let chunk = num.div_ceil(thread_count);
            std::thread::scope(|s| {
                for thread_index in 0..thread_count {
                    let fun: &JxlParallelRunnerFun = fun;
                    s.spawn(move || {
                        for task_index in thread_index * chunk
                            ..((thread_index + 1) * chunk).min(num)
                        {
                            let _ = fun(task_index);
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

    // decoder.orientation + apply_orientation runs in every image-crate
    // decode above. The in-tree encoder cannot write EXIF orientation tags,
    // so no transformed fixture can be synthesized here.

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
    fn truncated_jpeg_is_an_error_not_panic() {
        let img = image::RgbImage::from_pixel(32, 32, image::Rgb([1, 2, 3]));
        let mut jpg = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut jpg), image::ImageFormat::Jpeg)
            .unwrap();
        assert!(matches!(decode(&jpg[..20], "p.jpg"), Err(DecodeError::Image(Format::Jpeg, _))));
    }

    #[test]
    fn edge_cap_at_gpu_texture_limit() {
        let over = Decoded::rgba(16385, 1, Format::Png, vec![0u8; 4 * 16385].into_boxed_slice());
        assert!(matches!(over, Err(DecodeError::Unsupported(_))));
        let at_limit =
            Decoded::rgba(16384, 1, Format::Png, vec![0u8; 4 * 16384].into_boxed_slice());
        assert!(at_limit.is_ok());
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
        assert!(matches!(decode(b"\x00\x00\x00\x0cftypavif", "a.avif"), Err(DecodeError::Avif(..))));
        #[cfg(feature = "jxl")]
        assert!(matches!(decode(&[0xFF, 0x0A, 0x00], "a.jxl"), Err(DecodeError::Corrupt(_))));
    }
}
