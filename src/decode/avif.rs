//! AVIF decode through libavif. The FFI `unsafe` is confined to this
//! module.

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
use libavif_sys::AVIF_TRANSFER_CHARACTERISTICS_HLG;
use libavif_sys::AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084;
use libavif_sys::AVIF_TRANSFORM_IMIR;
use libavif_sys::AVIF_TRANSFORM_IROT;
use std::marker::PhantomData;

use eros::ErrorUnion;
use eros::IntoUnion as _;

use crate::fault::AvifFault;
use crate::fault::DecodeError;
use crate::fault::DecodeRefusal;

use super::plane_dims;
use super::transform_plane;
use super::Decoded;
use super::DisplayTransform;
use super::Format;
use super::Matrix;
use super::Plane;
use super::Subsampling;
use super::Transfer;
use super::YuvImage;

/// dav1d parallelism saturates quickly on stills, so the thread count
/// is capped.
const MAX_THREADS: usize = 8;

pub(super) fn decode(bytes: &[u8]) -> Result<Decoded, DecodeError> {
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
        _ => {
            return Err(ErrorUnion::new(DecodeRefusal::Corrupt(
                "avif: bad depth",
            )))
        }
    };
    super::checked_pixels(width, height, "avif: bad dimensions")
        .union()?;
    super::check_dims(width, height).union()?;
    let subsampling = match frame.yuvFormat {
        AVIF_PIXEL_FORMAT_YUV444 => Subsampling::Yuv444,
        AVIF_PIXEL_FORMAT_YUV422 => Subsampling::Yuv422,
        AVIF_PIXEL_FORMAT_YUV420 => Subsampling::Yuv420,
        AVIF_PIXEL_FORMAT_YUV400 => Subsampling::Mono,
        _ => {
            return Err(ErrorUnion::new(DecodeRefusal::Unsupported(
                "avif: pixel format",
            )))
        }
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
        _ => {
            return Err(ErrorUnion::new(DecodeRefusal::Unsupported(
                "avif: matrix coefficients",
            )))
        }
    };
    // HLG has no fixed SDR map; a wrong render is worse than a
    // refused page.
    let transfer = match u32::from(frame.transferCharacteristics) {
        AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084 => Transfer::Pq,
        AVIF_TRANSFER_CHARACTERISTICS_HLG => {
            return Err(ErrorUnion::new(DecodeRefusal::Unsupported(
                "avif: HLG transfer",
            )));
        }
        _ => Transfer::Srgb,
    };
    let transform = display_transform(frame);
    // A quarter turn makes 4:2:2 subsample vertically instead of
    // horizontally, which no plane geometry in the model expresses.
    if transform.swaps_dims() && subsampling == Subsampling::Yuv422 {
        return Err(ErrorUnion::new(DecodeRefusal::Unsupported(
            "avif: rotated 4:2:2",
        )));
    }

    // Mono skips the chroma copy below, so its 1x1 stub dims from
    // `plane_dims` are unused.
    let (_, (chroma_width, chroma_height)) =
        plane_dims((width, height), subsampling);
    // SAFETY: the guard still owns the frame, so the luma plane is live.
    // `copy_plane` rejects null planes and short strides.
    let luma_plane = unsafe {
        copy_plane(
            frame.yuvPlanes[0],
            frame.yuvRowBytes[0],
            width,
            height,
            u32::from(depth),
        )
    }?;
    let (blue_plane, red_plane) = if subsampling == Subsampling::Mono {
        (Plane::U8(Box::new([])), Plane::U8(Box::new([])))
    } else {
        (
            // SAFETY: same contract as the luma plane.
            unsafe {
                copy_plane(
                    frame.yuvPlanes[1],
                    frame.yuvRowBytes[1],
                    chroma_width,
                    chroma_height,
                    u32::from(depth),
                )
            }?,
            // SAFETY: same contract as the luma plane.
            unsafe {
                copy_plane(
                    frame.yuvPlanes[2],
                    frame.yuvRowBytes[2],
                    chroma_width,
                    chroma_height,
                    u32::from(depth),
                )
            }?,
        )
    };
    // SAFETY: same contract as the luma plane, at luma dimensions.
    let alpha_plane = unsafe {
        (!frame.alphaPlane.is_null())
            .then(|| {
                copy_plane(
                    frame.alphaPlane,
                    frame.alphaRowBytes,
                    width,
                    height,
                    u32::from(depth),
                )
            })
            .transpose()?
    };

    // The display transform applies to every plane alike; chroma
    // dimensions rotate coherently with the luma grid.
    let (luma_plane, _) =
        transform_plane(luma_plane, (width, height), transform);
    let (blue_plane, _) = transform_plane(
        blue_plane,
        (chroma_width, chroma_height),
        transform,
    );
    let (red_plane, _) = transform_plane(
        red_plane,
        (chroma_width, chroma_height),
        transform,
    );
    let alpha_plane = alpha_plane
        .map(|alpha| transform_plane(alpha, (width, height), transform).0);
    let (width, height) = if transform.swaps_dims() {
        (height, width)
    } else {
        (width, height)
    };

    super::Decoded::yuv(
        width,
        height,
        Format::Avif,
        transfer,
        YuvImage {
            depth,
            subsampling,
            full_range,
            matrix,
            luma: luma_plane,
            cb: blue_plane,
            cr: red_plane,
            alpha: alpha_plane,
        },
    )
    .union()
}

fn display_transform(frame: &avifImage) -> DisplayTransform {
    let flags = frame.transformFlags;
    let quarters = if flags & AVIF_TRANSFORM_IROT != 0 {
        frame.irot.angle % 4
    } else {
        0
    };
    let mirror_axis =
        (flags & AVIF_TRANSFORM_IMIR != 0).then_some(frame.imir.axis);
    DisplayTransform {
        quarters,
        mirror_axis,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the thread count is min(available, MAX_THREADS) before the C-int maxThreads field, so the narrowing cannot occur"
)]
fn threads() -> i32 {
    use std::num::NonZeroUsize;
    use std::thread;

    thread::available_parallelism()
        .map_or(1, NonZeroUsize::get)
        .min(MAX_THREADS) as i32
}

unsafe fn check(
    code: avifResult,
    what: &'static str,
) -> Result<(), AvifFault> {
    use std::ffi::CStr;

    if code == AVIF_RESULT_OK {
        return Ok(());
    }
    let detail =
        CStr::from_ptr(avifResultToString(code)).to_string_lossy();
    Err(AvifFault {
        step: what,
        detail: detail.into_owned(),
    })
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
        return Err(ErrorUnion::new(DecodeRefusal::Corrupt(
            "avif: missing plane",
        )));
    }
    let (width, height) = (width as usize, height as usize);
    let row_stride = row_stride as usize;
    if depth == 8 {
        if row_stride < width {
            return Err(ErrorUnion::new(DecodeRefusal::Corrupt(
                "avif: bad stride",
            )));
        }
        let mut out = Vec::with_capacity(width * height);
        for row in 0..height {
            let src = slice::from_raw_parts(
                plane_ptr.add(row * row_stride),
                width,
            );
            out.extend_from_slice(src);
        }
        Ok(Plane::U8(out.into_boxed_slice()))
    } else {
        let bytes = width * 2;
        if row_stride < bytes {
            return Err(ErrorUnion::new(DecodeRefusal::Corrupt(
                "avif: bad stride",
            )));
        }
        let mut raw = Vec::with_capacity(bytes * height);
        for row in 0..height {
            let src = slice::from_raw_parts(
                plane_ptr.add(row * row_stride),
                bytes,
            );
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
            return Err(ErrorUnion::new(DecodeRefusal::Unsupported(
                "avif: decoder create",
            )));
        }
        let guard = Self {
            ptr,
            _bytes: PhantomData,
        };
        (*guard.ptr).maxThreads = threads();
        // The decoder's own caps run before the pixel decode, so an
        // oversized hostile file fails at parse instead of funding a
        // multi-gigabyte dav1d picture first.
        (*guard.ptr).imageSizeLimit = super::MAX_PIXELS;
        (*guard.ptr).imageDimensionLimit = super::DECODE_EDGE_LIMIT;
        avifDecoderSetIOMemory(guard.ptr, bytes.as_ptr(), bytes.len());
        check(avifDecoderParse(guard.ptr), "avif parse").union()?;
        check(avifDecoderNextImage(guard.ptr), "avif image").union()?;
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
            return Err(ErrorUnion::new(DecodeRefusal::Corrupt(
                "avif: no image",
            )));
        }
        Ok(&*frame)
    }
}

impl Drop for DecoderGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: the guard solely owns the decoder from
        // `avifDecoderCreate`, and drop runs exactly once.
        unsafe {
            avifDecoderDestroy(self.ptr);
        }
    }
}
