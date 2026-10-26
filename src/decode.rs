//! Decoding: raw page bytes -> `Decoded` (RGBA or YUV planes).
//!
//! Dispatch is by magic bytes where possible, then extension. AVIF decodes
//! to YUV planes and the rest to RGBA.
//!
//! All functions here return `DecodeError` for corrupt input instead of
//! panicking. The FFI `unsafe` for AVIF is confined to the `avif`
//! submodule.

use std::error::Error;
use std::fmt;

use ::jxl::error::Error as JxlError;

/// Decoded page. Construct via [`Decoded::rgba`] / [`Decoded::yuv`], which
/// validate the pixel extents against the dimensions. A wrong extent then
/// surfaces as a construction error, not a display-time panic.
#[derive(Clone)]
pub struct Decoded {
    pub width: u32,
    pub height: u32,
    pub pixels: Pixels,
    pub format: Format,
}

impl Decoded {
    pub(crate) fn rgba(
        width: u32,
        height: u32,
        format: Format,
        bytes: Box<[u8]>,
    ) -> Result<Self, DecodeError> {
        check_dims(width, height)?;
        let want = checked_extent(width, height, 4)?;
        if bytes.len() != want {
            return Err(DecodeError::Corrupt("rgba extent"));
        }
        Ok(Self { width, height, pixels: Pixels::Rgba8(bytes), format })
    }

    pub(crate) fn yuv(
        width: u32,
        height: u32,
        format: Format,
        yuv: YuvImage,
    ) -> Result<Self, DecodeError> {
        check_dims(width, height)?;
        let (_, (cw, ch)) = plane_dims((width, height), yuv.subsampling);
        let samples = |plane: &Plane| match plane {
            Plane::U8(bytes) => bytes.len(),
            Plane::U16(bytes) => bytes.len(),
        };
        if samples(&yuv.luma) != width as usize * height as usize
            || (yuv.subsampling != Subsampling::Mono
                && (samples(&yuv.cb) != cw as usize * ch as usize
                    || samples(&yuv.cr) != cw as usize * ch as usize))
        {
            return Err(DecodeError::Corrupt("yuv plane extent"));
        }
        Ok(Self { width, height, pixels: Pixels::Yuv(yuv), format })
    }

    /// Approximate resident size, for cache budget accounting.
    pub fn resident_bytes(&self) -> usize {
        fn plane(plane: &Plane) -> usize {
            match plane {
                Plane::U8(bytes) => bytes.len(),
                Plane::U16(bytes) => bytes.len() * 2,
            }
        }
        match &self.pixels {
            Pixels::Rgba8(bytes) => bytes.len(),
            Pixels::Yuv(yuv) => plane(&yuv.luma) + plane(&yuv.cb) + plane(&yuv.cr),
        }
    }
}

/// Pixel count with a hostile-header cap. The cap matches the image crate's
/// 512 MiB allocation ceiling.
fn checked_pixels(width: u32, height: u32, what: &'static str) -> Result<u32, DecodeError> {
    width
        .checked_mul(height)
        .filter(|&count| (1..=MAX_PIXELS).contains(&count))
        .ok_or(DecodeError::Corrupt(what))
}

fn checked_extent(width: u32, height: u32, bytes_per_pixel: usize) -> Result<usize, DecodeError> {
    Ok(checked_pixels(width, height, "bad dimensions")? as usize * bytes_per_pixel)
}

const MAX_PIXELS: u32 = 512 << 18;

/// Largest page edge accepted as one GL texture. 0x4000 is the common
/// desktop limit.
const MAX_TEXTURE_EDGE: u32 = 0x4000;

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
            Self::Jpeg => "jpeg",
            Self::Png => "png",
            Self::Gif => "gif",
            Self::Webp => "webp",
            Self::Bmp => "bmp",
            Self::Tiff => "tiff",
            Self::Avif => "avif",
            Self::Jxl => "jxl",
        }
    }
}

/// 8-bit planes are one byte per sample. Deeper planes are `u16` samples
/// with `depth` significant bits. Dimensions ride on [`Decoded`]. Plane
/// sizes follow [`plane_dims`]. Mono carries luma only, so its chroma
/// planes stay empty.
#[derive(Clone)]
pub struct YuvImage {
    pub depth: u8,
    pub subsampling: Subsampling,
    /// Limited (video) or full (JPEG) range quantization.
    pub full_range: bool,
    pub matrix: Matrix,
    pub luma: Plane,
    pub cb: Plane,
    pub cr: Plane,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    Mono,
    Yuv420,
    Yuv422,
    Yuv444,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Matrix {
    Bt601,
    Bt709,
    Bt2020,
    Identity,
}

#[derive(Clone)]
pub enum Plane {
    U8(Box<[u8]>),
    U16(Box<[u16]>),
}

/// Plane dims (luma, chroma) for an image size and subsampling. Mono chroma
/// is 1x1 so renderers can bind a midpoint texture.
pub fn plane_dims(size: (u32, u32), subsampling: Subsampling) -> ((u32, u32), (u32, u32)) {
    let (width, height) = size;
    match subsampling {
        Subsampling::Mono => ((width, height), (1, 1)),
        Subsampling::Yuv420 => ((width, height), (width.div_ceil(2), height.div_ceil(2))),
        Subsampling::Yuv422 => ((width, height), (width.div_ceil(2), height)),
        Subsampling::Yuv444 => ((width, height), (width, height)),
    }
}

/// Byte view of a plane. `u16` planes become native-endian byte pairs.
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
    Image(Format, image::ImageError),
    Jxl(JxlError),
    /// libavif call failure with its step and avifResult detail.
    Avif(&'static str, String),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAnImage => write!(f, "not an image"),
            Self::Corrupt(what) => write!(f, "corrupt: {what}"),
            Self::Unsupported(what) => write!(f, "unsupported: {what}"),
            Self::Image(format, error) => write!(f, "{}: {error}", format.label()),
            Self::Jxl(error) => write!(f, "jxl: {error}"),
            Self::Avif(what, detail) => write!(f, "{what}: {detail}"),
        }
    }
}

impl Error for DecodeError {}

/// Decode one page. `name` is the page filename, used as the extension
/// fallback. Animated images show their first frame.
pub fn decode(bytes: &[u8], name: &str) -> Result<Decoded, DecodeError> {
    let Some(format) = sniff(bytes).or_else(|| format_from_name(name)) else {
        return Err(DecodeError::NotAnImage);
    };
    match format {
        Format::Avif => avif::decode(bytes),
        Format::Jxl => jxl::decode(bytes),
        Format::Jpeg
        | Format::Png
        | Format::Gif
        | Format::Webp
        | Format::Bmp
        | Format::Tiff => decode_image(bytes, format),
    }
}

/// Format for a page filename's extension. The same table decides what can
/// decode and what gets listed as a page.
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is the crate's visibility style for crate-wide items, even inside private modules"
)]
pub(crate) fn format_from_name(name: &str) -> Option<Format> {
    use std::path::Path;

    let ext = Path::new(name)
        .extension()
        .and_then(|extension| extension.to_str())?
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

/// `None` only means no known signature.
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
    if bytes.starts_with(b"RIFF") && bytes.get(8..12).is_some_and(|tag| tag == b"WEBP") {
        return Some(Format::Webp);
    }
    if bytes.starts_with(b"BM") {
        return Some(Format::Bmp);
    }
    if bytes.starts_with(&[0x49, 0x49, 0x2A, 0x00]) || bytes.starts_with(&[0x4D, 0x4D, 0x00, 0x2A]) {
        return Some(Format::Tiff);
    }
    // Raw codestream or ISOBMFF container signature.
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

fn sniff_ftyp_avif(bytes: &[u8]) -> bool {
    ftyp_brands(bytes).is_some_and(|brands| {
        brands.as_chunks::<4>().0.iter().any(|brand| brand == b"avif" || brand == b"avis")
    })
}

/// Brand region of the leading `ftyp` box. A box length under 12 still
/// yields the 4-byte major brand.
fn ftyp_brands(bytes: &[u8]) -> Option<&[u8]> {
    #[expect(
        clippy::big_endian_bytes,
        reason = "the ISOBMFF box length is big-endian by specification"
    )]
    let box_len = u32::from_be_bytes(bytes.get(..4)?.try_into().ok()?) as usize;
    let kind: [u8; 4] = bytes.get(4..8)?.try_into().ok()?;
    if kind != *b"ftyp" {
        return None;
    }
    let end = box_len.max(12).min(bytes.len());
    bytes.get(8..end)
}

/// image-crate path: decode with a concrete decoder so EXIF orientation is
/// readable, then applies it.
fn decode_image(bytes: &[u8], format: Format) -> Result<Decoded, DecodeError> {
    use std::io::Cursor;
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
    let reader = image::ImageReader::with_format(Cursor::new(bytes), image_format);
    let mut decoder = reader.into_decoder().map_err(|error| DecodeError::Image(format, error))?;
    // Orientation only swaps w and h, and the cap checks both, so this
    // pre-orientation check catches rotated pages too.
    let (width, height) = decoder.dimensions();
    check_dims(width, height)?;
    let orientation = decoder.orientation().map_err(|error| DecodeError::Image(format, error))?;
    let mut oriented = image::DynamicImage::from_decoder(decoder)
        .map_err(|error| DecodeError::Image(format, error))?;
    oriented.apply_orientation(orientation);
    let rgba = oriented.into_rgba8();
    Decoded::rgba(rgba.width(), rgba.height(), format, rgba.into_raw().into_boxed_slice())
}

/// libavif FFI: decoder over the in-memory file, YUV planes copied out
/// directly.
mod avif {
    use libavif_sys::avifDecoder;
    use libavif_sys::avifDecoderCreate;
    use libavif_sys::avifDecoderDestroy;
    use libavif_sys::avifDecoderNextImage;
    use libavif_sys::avifDecoderParse;
    use libavif_sys::avifDecoderSetIOMemory;
    use libavif_sys::avifImage;
    use libavif_sys::avifResult;
    use libavif_sys::avifResultToString;
    use libavif_sys::AVIF_MATRIX_COEFFICIENTS_BT2020_NCL;
    use libavif_sys::AVIF_MATRIX_COEFFICIENTS_BT470BG;
    use libavif_sys::AVIF_MATRIX_COEFFICIENTS_BT601;
    use libavif_sys::AVIF_MATRIX_COEFFICIENTS_BT709;
    use libavif_sys::AVIF_MATRIX_COEFFICIENTS_IDENTITY;
    use libavif_sys::AVIF_MATRIX_COEFFICIENTS_UNSPECIFIED;
    use libavif_sys::AVIF_PIXEL_FORMAT_YUV400;
    use libavif_sys::AVIF_PIXEL_FORMAT_YUV420;
    use libavif_sys::AVIF_PIXEL_FORMAT_YUV422;
    use libavif_sys::AVIF_PIXEL_FORMAT_YUV444;
    use libavif_sys::AVIF_RANGE_FULL;
    use libavif_sys::AVIF_RESULT_OK;
    use std::marker::PhantomData;

    use super::{DecodeError, Decoded, Format, Matrix, Plane, Subsampling, YuvImage};

    /// dav1d parallelism saturates quickly on stills, so the thread count
    /// is capped.
    const MAX_THREADS: usize = 8;

    pub fn decode(bytes: &[u8]) -> Result<Decoded, DecodeError> {
        // SAFETY: `DecoderGuard::new`'s contract leaves callers the `'bytes`
        // borrow, which the signature enforces.
        let decoder = unsafe { DecoderGuard::new(bytes)? };
        // SAFETY: the decoder is untouched since `new`.
        let frame = unsafe { decoder.image()? };

        let (width, height) = (frame.width, frame.height);
        let depth = match frame.depth {
            8 => 8_u8,
            10 => 10,
            12 => 12,
            16 => 16,
            _ => return Err(DecodeError::Corrupt("avif: bad depth")),
        };
        super::checked_pixels(width, height, "avif: bad dimensions")?;
        super::check_dims(width, height)?;
        let subsampling = match frame.yuvFormat {
            AVIF_PIXEL_FORMAT_YUV444 => Subsampling::Yuv444,
            AVIF_PIXEL_FORMAT_YUV422 => Subsampling::Yuv422,
            AVIF_PIXEL_FORMAT_YUV420 => Subsampling::Yuv420,
            AVIF_PIXEL_FORMAT_YUV400 => Subsampling::Mono,
            _ => return Err(DecodeError::Unsupported("avif: pixel format")),
        };
        let full_range = frame.yuvRange == AVIF_RANGE_FULL;
        let matrix = match u32::from(frame.matrixCoefficients) {
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
        // SAFETY: the guard still owns the frame, so the luma plane is live.
        // `copy_plane` rejects null planes and short strides.
        let luma_plane =
            unsafe { copy_plane(frame.yuvPlanes[0], frame.yuvRowBytes[0], width, height, u32::from(depth)) }?;
        let (blue_plane, red_plane) = if subsampling == Subsampling::Mono {
            (Plane::U8(Box::new([])), Plane::U8(Box::new([])))
        } else {
            (
                // SAFETY: same contract as the luma plane.
                unsafe {
                    copy_plane(frame.yuvPlanes[1], frame.yuvRowBytes[1], chroma_width, chroma_height, u32::from(depth))
                }?,
                // SAFETY: same contract as the luma plane.
                unsafe {
                    copy_plane(frame.yuvPlanes[2], frame.yuvRowBytes[2], chroma_width, chroma_height, u32::from(depth))
                }?,
            )
        };

        super::Decoded::yuv(
            width,
            height,
            Format::Avif,
            YuvImage {
                depth,
                subsampling,
                full_range,
                matrix,
                luma: luma_plane,
                cb: blue_plane,
                cr: red_plane,
            },
        )
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "the thread count is min(available, MAX_THREADS) before the C-int maxThreads field, so the narrowing cannot occur"
    )]
    fn threads() -> i32 {
        use std::num::NonZeroUsize;
        use std::thread;

        thread::available_parallelism().map_or(1, NonZeroUsize::get).min(MAX_THREADS) as i32
    }

    unsafe fn check(code: avifResult, what: &'static str) -> Result<(), DecodeError> {
        use std::ffi::CStr;

        if code == AVIF_RESULT_OK {
            return Ok(());
        }
        let detail = CStr::from_ptr(avifResultToString(code)).to_string_lossy();
        Err(DecodeError::Avif(what, detail.into_owned()))
    }

    /// Copies one plane row by row. libavif pads rows to `stride`.
    ///
    /// # Safety
    ///
    /// `plane_ptr` must be a live plane's pointer paired with `row_stride`
    /// as that plane's stride, holding `height` rows of samples.
    unsafe fn copy_plane(
        plane_ptr: *const u8,
        row_stride: u32,
        width: u32,
        height: u32,
        depth: u32,
    ) -> Result<Plane, DecodeError> {
        use std::slice;

        if plane_ptr.is_null() {
            return Err(DecodeError::Corrupt("avif: missing plane"));
        }
        let (width, height) = (width as usize, height as usize);
        let row_stride = row_stride as usize;
        if depth == 8 {
            if row_stride < width {
                return Err(DecodeError::Corrupt("avif: bad stride"));
            }
            let mut out = Vec::with_capacity(width * height);
            for row in 0..height {
                let src = slice::from_raw_parts(plane_ptr.add(row * row_stride), width);
                out.extend_from_slice(src);
            }
            Ok(Plane::U8(out.into_boxed_slice()))
        } else {
            let bytes = width * 2;
            if row_stride < bytes {
                return Err(DecodeError::Corrupt("avif: bad stride"));
            }
            let mut raw = Vec::with_capacity(bytes * height);
            for row in 0..height {
                let src = slice::from_raw_parts(plane_ptr.add(row * row_stride), bytes);
                raw.extend_from_slice(src);
            }
            #[expect(
                clippy::host_endian_bytes,
                reason = "libavif's u16 plane samples are native-endian C arrays"
            )]
            let samples = raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| u16::from_ne_bytes(*pair))
                .collect::<Box<[u16]>>();
            Ok(Plane::U16(samples))
        }
    }

    /// Owns the C decoder for one decode call. The `'bytes` lifetime pins
    /// the input, which the decoder reads through a raw pointer until
    /// destroyed.
    struct DecoderGuard<'bytes> {
        ptr: *mut avifDecoder,
        _bytes: PhantomData<&'bytes [u8]>,
    }

    impl<'bytes> DecoderGuard<'bytes> {
        /// Creates the decoder and decodes the first image.
        ///
        /// # Safety
        ///
        /// The marker exists because the body performs unchecked libavif
        /// calls. Input retention is carried by the `'bytes` lifetime, so
        /// callers owe nothing beyond it.
        unsafe fn new(bytes: &'bytes [u8]) -> Result<Self, DecodeError> {
            let ptr = avifDecoderCreate();
            if ptr.is_null() {
                return Err(DecodeError::Unsupported("avif: decoder create"));
            }
            let guard = Self { ptr, _bytes: PhantomData };
            (*guard.ptr).maxThreads = threads();
            avifDecoderSetIOMemory(guard.ptr, bytes.as_ptr(), bytes.len());
            check(avifDecoderParse(guard.ptr), "avif parse")?;
            check(avifDecoderNextImage(guard.ptr), "avif image")?;
            Ok(guard)
        }

        /// The decoded frame. The decoder owns it, so copy from it before
        /// the guard drops.
        ///
        /// # Safety
        ///
        /// The marker exists because the body dereferences the decoder
        /// pointer. The decoder must be untouched since `new`, and module
        /// privacy ensures no other decoder call happens between the two.
        unsafe fn image(&self) -> Result<&avifImage, DecodeError> {
            let frame = (*self.ptr).image;
            if frame.is_null() {
                return Err(DecodeError::Corrupt("avif: no image"));
            }
            Ok(&*frame)
        }
    }

    impl Drop for DecoderGuard<'_> {
        fn drop(&mut self) {
            // SAFETY: the guard solely owns the decoder from
            // `avifDecoderCreate`, and drop runs exactly once.
            unsafe { avifDecoderDestroy(self.ptr); }
        }
    }
}

mod jxl {
    use jxl::api::Event;
    use jxl::api::JxlDecoder;
    use jxl::api::JxlDecoderOptions;
    use jxl::api::JxlOutputBuffer;
    use jxl::api::JxlParallelRunner;
    use jxl::api::JxlParallelRunnerFun;
    use jxl::api::JxlPixelFormat;
    use jxl::error::Error as JxlError;
    use jxl::error::Result as JxlResult;

    use super::{DecodeError, Decoded, Format};

    const MAX_THREADS: usize = 16;

    pub fn decode(bytes: &[u8]) -> Result<Decoded, DecodeError> {
        use std::slice;

        let mut runner = Pool(threads());
        // The whole file is in memory, so `NeedMoreInput` means the file is
        // truncated.
        let mut input: &[u8] = bytes;
        let mut decoder = JxlDecoder::new(JxlDecoderOptions::default());

        loop {
            let event = decoder
                .process(&mut input, None, Some(&mut runner))
                .map_err(DecodeError::Jxl)?;
            if event == Event::BasicInfo {
                break;
            }
            if let Event::NeedMoreInput { size_hint } = event {
                tracing::debug!(size_hint, "truncated jxl header");
                return Err(DecodeError::Corrupt("jxl: truncated header"));
            }
            // A fresh decoder emits BasicInfo or NeedMoreInput before all else.
        }

        let info = decoder
            .basic_info()
            .ok_or(DecodeError::Corrupt("jxl: missing basic info"))?;
        #[expect(
            clippy::map_err_ignore,
            reason = "the conversion error only says the value does not fit, which the message already states"
        )]
        let (width, height) = (
            u32::try_from(info.size.0).map_err(|_| DecodeError::Corrupt("jxl: bad dimensions"))?,
            u32::try_from(info.size.1).map_err(|_| DecodeError::Corrupt("jxl: bad dimensions"))?,
        );
        let pixel_count = super::checked_pixels(width, height, "jxl: bad dimensions")?;
        super::check_dims(width, height)?;

        // Files without their own alpha still get opaque alpha from the
        // decoder.
        let extra_channels = info.extra_channels.len();
        decoder
            .set_pixel_format(JxlPixelFormat::rgba8(extra_channels))
            .map_err(DecodeError::Jxl)?;

        let mut rgba = vec![0_u8; pixel_count as usize * 4];
        loop {
            let mut output = JxlOutputBuffer::new(&mut rgba, height as usize, width as usize * 4);
            match decoder
                .process(&mut input, Some(slice::from_mut(&mut output)), Some(&mut runner))
                .map_err(DecodeError::Jxl)?
            {
                Event::FrameComplete { has_more_frames } => {
                    if has_more_frames {
                        tracing::debug!("jxl animation: later frames not decoded");
                    }
                    break;
                }
                Event::NeedMoreInput { size_hint } => {
                    tracing::debug!(size_hint, "truncated jxl image");
                    return Err(DecodeError::Corrupt("jxl: truncated image"));
                }
                Event::BasicInfo | Event::FrameHeader | Event::Complete => {}
            }
        }

        Decoded::rgba(width, height, Format::Jxl, rgba.into_boxed_slice())
    }

    fn threads() -> usize {
        use std::num::NonZeroUsize;
        use std::thread;

        thread::available_parallelism().map_or(1, NonZeroUsize::get).min(MAX_THREADS)
    }

    /// Scoped-thread runner. A failing task fails the whole run, which is
    /// how the crate's driver expects a broken codestream to be reported.
    /// The driver invokes this only with two or more threads, so no
    /// sequential path is needed.
    pub(super) struct Pool(usize);

    impl Pool {
        #[cfg(test)]
        pub(super) fn new(threads: usize) -> Self {
            Self(threads)
        }
    }

    impl JxlParallelRunner for Pool {
        fn run(&mut self, num: usize, fun: &JxlParallelRunnerFun<'_>) -> JxlResult<()> {
            use std::thread;

            use parking_lot::Mutex;

            let thread_count = self.0.min(num);
            let chunk = num.div_ceil(thread_count);
            let failure: Mutex<Option<JxlError>> = Mutex::new(None);
            thread::scope(|scope| {
                for thread_index in 0..thread_count {
                    let fun: &JxlParallelRunnerFun = fun;
                    let failure = &failure;
                    scope.spawn(move || {
                        for task_index in thread_index * chunk
                            ..((thread_index + 1) * chunk).min(num)
                        {
                            if let Err(task_error) = fun(task_index) {
                                failure.lock().get_or_insert(task_error);
                                break;
                            }
                        }
                    });
                }
            });
            failure.into_inner().map_or(Ok(()), Err)
        }
        fn num_threads(&self) -> usize {
            self.0
        }
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, clippy::panic, reason = "a panic in test code is inconsequential")]
mod tests {
    use super::*;
    use crate::testkit;

    /// `decode()` dispatches on magic bytes, then the filename extension; anything else is `NotAnImage`.
    #[test]
    fn format_detection() {
        // sniff: one row per magic
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
        // Non-avif brands like HEIC's mif1 do not sniff as avif.
        let mut heic = vec![0, 0, 0, 0x14];
        heic.extend_from_slice(b"ftypmif1");
        assert_eq!(sniff(&heic), None);
        assert_eq!(sniff(b"plain text"), None);

        // extension fallback: uppercase and aliases accepted, non-images rejected
        assert_eq!(format_from_name("a.JPG"), Some(Format::Jpeg));
        assert_eq!(format_from_name("a.tiff"), Some(Format::Tiff));
        assert_eq!(format_from_name("a.avif"), Some(Format::Avif));
        assert_eq!(format_from_name("a.jxl"), Some(Format::Jxl));
        assert_eq!(format_from_name("cover.cbz"), None);
        assert_eq!(format_from_name("noext"), None);

        // garbage: no magic and no usable extension is NotAnImage
        assert!(matches!(decode(b"hello world", "noext"), Err(DecodeError::NotAnImage)));
        assert!(matches!(decode(b"hello world", "note.txt"), Err(DecodeError::NotAnImage)));
        assert!(matches!(decode(&[], "x"), Err(DecodeError::NotAnImage)));
    }

    /// The in-tree encoders produce pages `decode()` reads back.
    // The encoder cannot write EXIF orientation tags, so orientation
    // handling has no testable fixture.
    #[test]
    fn encoder_roundtrips() {
        // png: format, dims, extent, and the first pixel
        let png = testkit::tiny_png(4, 3, [10, 20, 30, 255]);
        let png_page = decode(&png, "p.png").expect("png decodes");
        assert_eq!((png_page.width, png_page.height), (4, 3));
        assert_eq!(png_page.format, Format::Png);
        match png_page.pixels {
            Pixels::Rgba8(bytes) => {
                assert_eq!(bytes.len(), 4 * 4 * 3);
                assert_eq!(bytes.get(..4), Some(&[10, 20, 30, 255][..]));
            }
            Pixels::Yuv(_) => panic!("png must decode to rgba"),
        }

        // jpeg: format and dims
        let jpeg = testkit::tiny_jpeg(5, 2, [200, 100, 50]);
        let jpeg_page = decode(&jpeg, "p.jpg").expect("jpeg decodes");
        assert_eq!(jpeg_page.format, Format::Jpeg);
        assert_eq!((jpeg_page.width, jpeg_page.height), (5, 2));

        // gif: format and dims
        let gif = testkit::tiny_gif(3, 2, [200, 100, 50]);
        let gif_page = decode(&gif, "p.gif").expect("gif decodes");
        assert_eq!(gif_page.format, Format::Gif);
        assert_eq!((gif_page.width, gif_page.height), (3, 2));

        // bmp: format and dims
        let bmp = testkit::tiny_bmp(2, 3, [200, 100, 50]);
        let bmp_page = decode(&bmp, "p.bmp").expect("bmp decodes");
        assert_eq!(bmp_page.format, Format::Bmp);
        assert_eq!((bmp_page.width, bmp_page.height), (2, 3));

        // tiff: format and dims
        let tiff = testkit::tiny_tiff(3, 1, [200, 100, 50]);
        let tiff_page = decode(&tiff, "p.tiff").expect("tiff decodes");
        assert_eq!(tiff_page.format, Format::Tiff);
        assert_eq!((tiff_page.width, tiff_page.height), (3, 1));
    }

    /// Decoded constructors cap page edges at the GPU texture limit and pixel counts at the allocation ceiling.
    #[test]
    fn constructor_caps() {
        Decoded::rgba(
            MAX_TEXTURE_EDGE,
            1,
            Format::Png,
            vec![0_u8; 4 * MAX_TEXTURE_EDGE as usize].into_boxed_slice(),
        )
        .expect("page at the edge cap constructs");
        // An edge over the limit is Unsupported.
        let over = Decoded::rgba(
            MAX_TEXTURE_EDGE + 1,
            1,
            Format::Png,
            vec![0_u8; 4 * (MAX_TEXTURE_EDGE + 1) as usize].into_boxed_slice(),
        );
        assert!(matches!(over, Err(DecodeError::Unsupported(_))));
        // Dims inside the edge cap can still exceed the pixel-count cap,
        // which must reject the page before any buffer is sized for it.
        assert!(matches!(
            Decoded::rgba(MAX_TEXTURE_EDGE, MAX_TEXTURE_EDGE, Format::Png, Box::default()),
            Err(DecodeError::Corrupt("bad dimensions"))
        ));
    }

    /// `plane_dims` rounds chroma up on odd edges; `Decoded::yuv` rejects wrong plane extents and counts resident bytes with u16 samples double.
    #[test]
    fn yuv_planes() {
        // odd edges round chroma up. Mono carries a 1x1 chroma stub.
        assert_eq!(plane_dims((5, 3), Subsampling::Mono), ((5, 3), (1, 1)));
        assert_eq!(plane_dims((5, 3), Subsampling::Yuv420), ((5, 3), (3, 2)));
        assert_eq!(plane_dims((5, 3), Subsampling::Yuv422), ((5, 3), (3, 3)));
        assert_eq!(plane_dims((5, 3), Subsampling::Yuv444), ((5, 3), (5, 3)));

        // a well-sized mono page constructs; an empty luma plane does not
        let mono = |luma: Plane| YuvImage {
            depth: 10,
            subsampling: Subsampling::Mono,
            full_range: true,
            matrix: Matrix::Bt601,
            luma,
            cb: Plane::U8(Box::new([])),
            cr: Plane::U8(Box::new([])),
        };
        let real = mono(Plane::U16(vec![0; 32 * 16].into_boxed_slice()));
        let page = Decoded::yuv(32, 16, Format::Avif, real).expect("mono yuv page constructs");
        // The worker's book budget consumes this count.
        assert_eq!(page.resident_bytes(), 32 * 16 * 2);
        let empty = mono(Plane::U16(Box::new([])));
        assert!(matches!(
            Decoded::yuv(32, 16, Format::Avif, empty),
            Err(DecodeError::Corrupt("yuv plane extent"))
        ));
        // A 420 image whose chroma planes are stubs: a 4x2 image needs
        // 2x1 chroma samples.
        let undersized = YuvImage {
            depth: 8,
            subsampling: Subsampling::Yuv420,
            full_range: true,
            matrix: Matrix::Bt601,
            luma: Plane::U8(vec![0; 4 * 2].into_boxed_slice()),
            cb: Plane::U8(Box::new([])),
            cr: Plane::U8(Box::new([])),
        };
        assert!(matches!(
            Decoded::yuv(4, 2, Format::Avif, undersized),
            Err(DecodeError::Corrupt("yuv plane extent"))
        ));
    }

    /// The mono AVIF fixture decodes to a 10-bit u16 luma plane with empty chroma and the gradient intact.
    // 32x16 10-bit 4:0:0 fixture, a black-to-white grayscale gradient.
    #[test]
    fn avif_fixture_decodes() {
        let page = decode(include_bytes!("testdata/mono.avif"), "a.avif").expect("mono avif decodes");
        assert_eq!(page.format, Format::Avif);
        assert_eq!((page.width, page.height), (32, 16));
        let Pixels::Yuv(yuv) = &page.pixels else {
            panic!("avif must decode to yuv planes")
        };
        assert_eq!(yuv.subsampling, Subsampling::Mono);
        let Plane::U16(luma) = &yuv.luma else {
            panic!("10-bit luma must be u16 samples")
        };
        assert_eq!(luma.len(), 32 * 16);
        assert!(luma.first() < luma.last());
        assert!(matches!(&yuv.cb, Plane::U8(bytes) if bytes.is_empty()));
        assert!(matches!(&yuv.cr, Plane::U8(bytes) if bytes.is_empty()));
    }

    /// The jxl fixtures decode to RGBA: gray/alpha pixel content, wide row-stride order, an animated first frame.
    // wide.jxl and animated.jxl are verbatim copies of exif.jxl and
    // 5_frames_numbered_jxli.jxl from jxl 0.8.0's tests/testdata.
    #[test]
    fn jxl_fixtures_decode() {
        // gray: solid halves, gray replicated to r=g=b
        let gray = decode(include_bytes!("testdata/gray.jxl"), "a.jxl").expect("gray jxl decodes");
        assert_eq!((gray.width, gray.height), (8, 8));
        assert_eq!(gray.format, Format::Jxl);
        match gray.pixels {
            Pixels::Rgba8(bytes) => {
                assert_eq!(bytes.get(..4), Some(&[0x20, 0x20, 0x20, 0xFF][..]));
                assert_eq!(bytes.get(128..132), Some(&[0xE0, 0xE0, 0xE0, 0xFF][..]));
            }
            Pixels::Yuv(_) => panic!("jxl must decode to rgba"),
        }

        // alpha: the file's own alpha rides in the fourth byte
        let alpha = decode(include_bytes!("testdata/alpha.jxl"), "a.jxl").expect("alpha jxl decodes");
        match alpha.pixels {
            Pixels::Rgba8(bytes) => {
                assert_eq!(bytes.get(..4), Some(&[0x40, 0x80, 0xC0, 0xFF][..]));
                assert_eq!(bytes.get(128..132), Some(&[0x40, 0x80, 0xC0, 0x80][..]));
            }
            Pixels::Yuv(_) => panic!("jxl must decode to rgba"),
        }

        // wide: 256x128 pins the row/stride order. Opaque alpha on every
        // pixel proves the decoder wrote all rows of the zero-initialised
        // buffer
        let wide = decode(include_bytes!("testdata/wide.jxl"), "a.jxl").expect("wide jxl decodes");
        assert_eq!((wide.width, wide.height), (256, 128));
        let Pixels::Rgba8(wide_rgba) = &wide.pixels else {
            panic!("jxl must decode to rgba")
        };
        assert!(wide_rgba.as_chunks::<4>().0.iter().all(|px| px[3] == 0xFF));

        // animated: the numbered frames paint digits, so the first frame
        // carries non-black pixels
        let animated =
            decode(include_bytes!("testdata/animated.jxl"), "a.jxl").expect("animated jxl decodes");
        assert_eq!((animated.width, animated.height), (256, 256));
        let Pixels::Rgba8(animated_rgba) = &animated.pixels else {
            panic!("jxl must decode to rgba")
        };
        assert!(
            animated_rgba
                .as_chunks::<4>()
                .0
                .iter()
                .any(|px| px[0] != 0 || px[1] != 0 || px[2] != 0)
        );
    }

    /// Every truncation point of a jxl page ends the decode with the matching "truncated" error instead of hanging.
    // A truncation bug would hang the decode instead of failing it, hence
    // the watchdog thread. The cuts cover the header, the frame header, and
    // the pixels. A decode panic closes the channel and fails the recv, so
    // plain asserts on the main thread suffice.
    #[test]
    fn truncated_jxl_ends_the_decode() {
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let fixture = include_bytes!("testdata/gray.jxl");
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for (cut, message) in [
                (1_usize, "jxl: truncated header"),
                (2, "jxl: truncated header"),
                (8, "jxl: truncated image"),
                (35, "jxl: truncated image"),
            ] {
                let truncated = fixture.get(..cut).expect("cut inside the fixture");
                let Ok(()) = tx.send((cut, message, decode(truncated, "a.jxl"))) else { return; };
            }
        });
        for _ in 0..4 {
            let (cut, message, result) =
                rx.recv_timeout(Duration::from_secs(5)).expect("truncated jxl decode finishes");
            assert!(
                matches!(result, Err(DecodeError::Corrupt(actual)) if actual == message),
                "cut {cut} reported the wrong truncation"
            );
        }
    }

    /// The pool reports the first failing task, and every task stays reachable under the chunk partition.
    // The fixtures are single-group, so only this test exercises the
    // runner's parallel path.
    #[test]
    fn jxl_pool_reports_the_first_task_error() {
        use ::jxl::api::JxlParallelRunner as _;

        use super::jxl::Pool;

        let mut pool = Pool::new(4);
        let failing = |task: usize| {
            if task == 1 {
                Err(JxlError::InvalidSignature)
            } else {
                Ok(())
            }
        };
        assert!(matches!(pool.run(4, &failing), Err(JxlError::InvalidSignature)));

        let passing = |_: usize| Ok(());
        pool.run(4, &passing).expect("every passing task succeeds");

        // Two threads over four tasks pin the chunk partition. A range
        // that drops the last task would leave the failure unreported.
        let mut two_threads = Pool::new(2);
        let failing_last = |task: usize| {
            if task == 3 {
                Err(JxlError::InvalidSignature)
            } else {
                Ok(())
            }
        };
        assert!(matches!(
            two_threads.run(4, &failing_last),
            Err(JxlError::InvalidSignature)
        ));
    }
}
