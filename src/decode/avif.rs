//! AVIF decode through libavif. The FFI `unsafe` is confined to this
//! module.

use bytemuck::Pod;
use image::metadata::Orientation;
use libavif_sys::avifDecoder;
use libavif_sys::avifImage;
use libavif_sys::avifResult;
use std::marker::PhantomData;

use super::Decoded;
use super::GrayPlanes;
use super::Samples;
use super::Subsampling;
use super::WideDepth;
use super::YuvPlanes;

/// dav1d parallelism saturates quickly on stills, so the thread count
/// is capped.
const MAX_THREADS: i32 = 8;

pub(super) fn decode(
    bytes: &[u8],
    max_threads: usize,
) -> eros::Result<Decoded> {
    use libavif_sys::AVIF_MATRIX_COEFFICIENTS_BT470BG;
    use libavif_sys::AVIF_MATRIX_COEFFICIENTS_BT601;
    use libavif_sys::AVIF_MATRIX_COEFFICIENTS_BT709;
    use libavif_sys::AVIF_MATRIX_COEFFICIENTS_BT2020_NCL;
    use libavif_sys::AVIF_MATRIX_COEFFICIENTS_IDENTITY;
    use libavif_sys::AVIF_MATRIX_COEFFICIENTS_UNSPECIFIED;
    use libavif_sys::AVIF_PIXEL_FORMAT_YUV400;
    use libavif_sys::AVIF_PIXEL_FORMAT_YUV420;
    use libavif_sys::AVIF_PIXEL_FORMAT_YUV422;
    use libavif_sys::AVIF_PIXEL_FORMAT_YUV444;
    use libavif_sys::AVIF_RANGE_FULL;
    use libavif_sys::AVIF_TRANSFER_CHARACTERISTICS_HLG;
    use libavif_sys::AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084;

    use super::Format;
    use super::GrayImage;
    use super::Matrix;
    use super::Transfer;
    use super::YuvImage;

    let decoder = DecoderGuard::new(bytes, max_threads)?;
    let frame = decoder.image()?;

    let (width, height) = (frame.width, frame.height);
    let wide_depth = match frame.depth {
        8 => None,
        10 => Some(WideDepth::D10),
        12 => Some(WideDepth::D12),
        16 => Some(WideDepth::D16),
        _ => eros::bail!("avif: bad depth"),
    };
    super::admit_dims(width, height, "avif: bad dimensions")?;
    let full_range = frame.yuvRange == AVIF_RANGE_FULL;
    // HLG has no fixed SDR map; a wrong render is worse than a
    // refused page.
    let transfer = match u32::from(frame.transferCharacteristics) {
        AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084 => Transfer::Pq,
        AVIF_TRANSFER_CHARACTERISTICS_HLG => {
            eros::bail!("avif: unsupported HLG transfer");
        }
        _ => Transfer::Srgb,
    };
    let orientation = orientation(frame);
    let yuv = if frame.yuvFormat == AVIF_PIXEL_FORMAT_YUV400 {
        None
    } else {
        let subsampling = match frame.yuvFormat {
            AVIF_PIXEL_FORMAT_YUV444 => Subsampling::Yuv444,
            AVIF_PIXEL_FORMAT_YUV422 => Subsampling::Yuv422,
            AVIF_PIXEL_FORMAT_YUV420 => Subsampling::Yuv420,
            _ => eros::bail!("avif: unsupported pixel format"),
        };
        let matrix = match u32::from(frame.matrixCoefficients) {
            AVIF_MATRIX_COEFFICIENTS_IDENTITY => Matrix::Identity,
            AVIF_MATRIX_COEFFICIENTS_BT709 => Matrix::Bt709,
            AVIF_MATRIX_COEFFICIENTS_BT2020_NCL => Matrix::Bt2020,
            // avifCalcYUVCoefficients treats unspecified/BG/601 as BT.601.
            AVIF_MATRIX_COEFFICIENTS_UNSPECIFIED
            | AVIF_MATRIX_COEFFICIENTS_BT470BG
            | AVIF_MATRIX_COEFFICIENTS_BT601 => Matrix::Bt601,
            _ => eros::bail!("avif: unsupported matrix coefficients"),
        };
        // A quarter turn makes 4:2:2 subsample vertically instead of
        // horizontally, which no plane geometry in the model expresses.
        if super::orientation_swaps_dims(orientation)
            && subsampling == Subsampling::Yuv422
        {
            eros::bail!("avif: unsupported rotated 4:2:2");
        }
        Some((subsampling, matrix))
    };

    let Some((subsampling, matrix)) = yuv else {
        let samples = match wide_depth {
            // SAFETY: the guard keeps `frame`'s planes live; `copy_plane` rejects null/short strides.
            None => Samples::U8(unsafe { copy_gray_planes::<u8>(frame) }?),
            Some(depth) => Samples::U16 {
                depth,
                // SAFETY: the guard keeps `frame`'s planes live; `copy_plane` rejects null/short strides.
                planes: unsafe { copy_gray_planes::<u16>(frame) }?,
            },
        };
        return Ok(Decoded::gray(
            width,
            height,
            Format::Avif,
            transfer,
            GrayImage {
                full_range,
                samples,
            },
        )?
        .apply_orientation(orientation));
    };

    let samples = match wide_depth {
        // SAFETY: the guard keeps `frame`'s planes live; `copy_plane` rejects null/short strides.
        None => Samples::U8(unsafe {
            copy_yuv_planes::<u8>(frame, subsampling)
        }?),
        Some(depth) => Samples::U16 {
            depth,
            // SAFETY: the guard keeps `frame`'s planes live; `copy_plane` rejects null/short strides.
            planes: unsafe { copy_yuv_planes::<u16>(frame, subsampling) }?,
        },
    };

    Ok(Decoded::yuv(
        width,
        height,
        Format::Avif,
        transfer,
        YuvImage {
            subsampling,
            full_range,
            matrix,
            samples,
        },
    )?
    .apply_orientation(orientation))
}

/// Mirroring runs before rotation, matching the property order HEIF writers use.
fn orientation(frame: &avifImage) -> Orientation {
    use libavif_sys::AVIF_TRANSFORM_IMIR;
    use libavif_sys::AVIF_TRANSFORM_IROT;

    const QUARTER_TURNS_PER_ROTATION: u8 = 4;
    const IMIR_HORIZONTAL_AXIS: u8 = 1;

    let flags = frame.transformFlags;
    let ccw_quarters = if flags & AVIF_TRANSFORM_IROT != 0 {
        frame.irot.angle % QUARTER_TURNS_PER_ROTATION
    } else {
        0
    };
    let mirror_axis =
        (flags & AVIF_TRANSFORM_IMIR != 0).then_some(frame.imir.axis);
    match mirror_axis {
        None => match ccw_quarters {
            0 => Orientation::NoTransforms,
            1 => Orientation::Rotate270,
            2 => Orientation::Rotate180,
            _ => Orientation::Rotate90,
        },
        Some(IMIR_HORIZONTAL_AXIS) => match ccw_quarters {
            0 => Orientation::FlipHorizontal,
            1 => Orientation::Rotate90FlipH,
            2 => Orientation::FlipVertical,
            _ => Orientation::Rotate270FlipH,
        },
        Some(_) => match ccw_quarters {
            0 => Orientation::FlipVertical,
            1 => Orientation::Rotate270FlipH,
            2 => Orientation::FlipHorizontal,
            _ => Orientation::Rotate90FlipH,
        },
    }
}

fn clamp_threads(max_threads: usize) -> i32 {
    i32::try_from(max_threads)
        .unwrap_or(MAX_THREADS)
        .clamp(1, MAX_THREADS)
}

fn check(code: avifResult, what: &'static str) -> eros::Result<()> {
    use libavif_sys::AVIF_RESULT_OK;
    use libavif_sys::avifResultToString;
    use std::ffi::CStr;

    if code == AVIF_RESULT_OK {
        return Ok(());
    }
    // SAFETY: `avifResultToString` returns a static NUL-terminated string for
    // any `avifResult`.
    let message_ptr = unsafe { avifResultToString(code) };
    // SAFETY: `message_ptr` points to a static NUL-terminated string.
    let detail = unsafe { CStr::from_ptr(message_ptr) }.to_string_lossy();
    eros::bail!("{what}: {detail}")
}

/// # Safety
///
/// Live non-null planes in `frame` must hold `frame.height` rows at their
/// declared row strides.
unsafe fn copy_gray_planes<Sample>(
    frame: &avifImage,
) -> eros::Result<GrayPlanes<Sample>>
where
    Sample: Copy + Default + Pod,
{
    // SAFETY: caller guarantees `frame`'s luma plane is live.
    let luma = unsafe {
        copy_plane::<Sample>(
            frame.yuvPlanes[0],
            frame.yuvRowBytes[0],
            frame.width,
            frame.height,
        )
    }?;
    // SAFETY: caller guarantees `frame`'s alpha plane is live when non-null.
    let alpha = unsafe {
        (!frame.alphaPlane.is_null())
            .then(|| {
                copy_plane::<Sample>(
                    frame.alphaPlane,
                    frame.alphaRowBytes,
                    frame.width,
                    frame.height,
                )
            })
            .transpose()?
    };
    Ok(GrayPlanes { luma, alpha })
}

/// # Safety
///
/// Live non-null planes in `frame` must hold their subsampled dimensions at
/// their declared row strides.
unsafe fn copy_yuv_planes<Sample>(
    frame: &avifImage,
    subsampling: Subsampling,
) -> eros::Result<YuvPlanes<Sample>>
where
    Sample: Copy + Default + Pod,
{
    // SAFETY: caller guarantees `frame`'s luma and alpha planes are live.
    let gray = unsafe { copy_gray_planes::<Sample>(frame) }?;
    let (_, (chroma_width, chroma_height)) =
        super::plane_dims((frame.width, frame.height), subsampling);
    // SAFETY: caller guarantees `frame`'s Cb plane is live.
    let cb = unsafe {
        copy_plane::<Sample>(
            frame.yuvPlanes[1],
            frame.yuvRowBytes[1],
            chroma_width,
            chroma_height,
        )
    }?;
    // SAFETY: caller guarantees `frame`'s Cr plane is live.
    let cr = unsafe {
        copy_plane::<Sample>(
            frame.yuvPlanes[2],
            frame.yuvRowBytes[2],
            chroma_width,
            chroma_height,
        )
    }?;
    Ok(YuvPlanes {
        luma: gray.luma,
        cb,
        cr,
        alpha: gray.alpha,
    })
}

/// # Safety
///
/// When non-null, `plane_ptr` must hold `height` live rows spaced by
/// `row_stride` bytes.
unsafe fn copy_plane<Sample>(
    plane_ptr: *const u8,
    row_stride: u32,
    width: u32,
    height: u32,
) -> eros::Result<Box<[Sample]>>
where
    Sample: Copy + Default + Pod,
{
    if plane_ptr.is_null() {
        eros::bail!("avif: missing plane");
    }
    let width = width as usize;
    let height = height as usize;
    let row_stride = row_stride as usize;
    let row_bytes = width * size_of::<Sample>();
    if row_stride < row_bytes {
        eros::bail!("avif: bad stride");
    }
    if width == 0 || height == 0 {
        return Ok(Box::new([]));
    }
    let mut out = vec![Sample::default(); width * height];
    // SAFETY: `plane_ptr` holds `height` live rows at `row_stride >= row_bytes`.
    unsafe {
        copy_strided_bytes(
            plane_ptr,
            row_stride,
            row_bytes,
            height,
            bytemuck::cast_slice_mut(&mut out),
        );
    }
    Ok(out.into_boxed_slice())
}

/// # Safety
///
/// `plane_ptr` must hold `height` live rows at `row_stride`, each at
/// least `row_bytes` wide, and `target.len() == row_bytes * height`.
unsafe fn copy_strided_bytes(
    plane_ptr: *const u8,
    row_stride: usize,
    row_bytes: usize,
    height: usize,
    target: &mut [u8],
) {
    use std::slice;

    if row_stride == row_bytes {
        // SAFETY: `row_stride == row_bytes` means all rows are contiguous.
        let source = unsafe {
            slice::from_raw_parts(plane_ptr, row_bytes * height)
        };
        target.copy_from_slice(source);
        return;
    }
    for (row, target_row) in target.chunks_exact_mut(row_bytes).enumerate()
    {
        // SAFETY: `row < height`, so the row pointer stays within the allocated plane.
        let row_ptr = unsafe { plane_ptr.add(row * row_stride) };
        // SAFETY: each row holds `row_bytes` live sample bytes.
        let source = unsafe { slice::from_raw_parts(row_ptr, row_bytes) };
        target_row.copy_from_slice(source);
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
    fn new(bytes: &'bytes [u8], max_threads: usize) -> eros::Result<Self> {
        use libavif_sys::AVIF_IMAGE_CONTENT_SAMPLE_TRANSFORMS;
        use libavif_sys::AVIF_TRUE;
        use libavif_sys::avifDecoderCreate;
        use libavif_sys::avifDecoderNextImage;
        use libavif_sys::avifDecoderParse;
        use libavif_sys::avifDecoderSetIOMemory;

        const NO_DIMENSION_LIMIT: u32 = 0;

        // SAFETY: allocates an independent decoder instance.
        let ptr = unsafe { avifDecoderCreate() };
        if ptr.is_null() {
            eros::bail!("avif: decoder create failed");
        }
        let guard = Self {
            ptr,
            _bytes: PhantomData,
        };
        {
            // SAFETY: `guard.ptr` is newly allocated, non-null, and uniquely
            // owned; the mutable borrow ends before subsequent C calls.
            let decoder = unsafe { &mut *guard.ptr };
            decoder.maxThreads = clamp_threads(max_threads);
            // The decoder's pixel cap runs before the pixel decode, so an
            // oversized hostile file fails at parse instead of funding a
            // multi-gigabyte dav1d picture first. Clear libavif's per-axis
            // edge cap so tall strips under `MAX_PIXELS` still decode.
            decoder.imageSizeLimit = super::MAX_PIXELS;
            decoder.imageDimensionLimit = NO_DIMENSION_LIMIT;
            // Opt into 16-bit sample-transform items (off by default in
            // libavif 1.4); gain maps stay off as rendering skips them.
            decoder.imageContentToDecode |=
                AVIF_IMAGE_CONTENT_SAMPLE_TRANSFORMS;
            // Neither EXIF nor XMP is read from decoded AVIF frames; skipping
            // Exif also avoids failing on malformed Exif TIFF offsets.
            let enable = AVIF_TRUE.cast_signed();
            decoder.ignoreExif = enable;
            decoder.ignoreXMP = enable;
        }
        // SAFETY: `guard.ptr` is uniquely owned and `bytes` stays live until
        // `guard` drops.
        let io = unsafe {
            avifDecoderSetIOMemory(guard.ptr, bytes.as_ptr(), bytes.len())
        };
        check(io, "avif io")?;
        // SAFETY: `guard.ptr` is valid with attached input bytes.
        let parsed = unsafe { avifDecoderParse(guard.ptr) };
        check(parsed, "avif parse")?;
        // SAFETY: `guard.ptr` parsed cleanly above.
        let next = unsafe { avifDecoderNextImage(guard.ptr) };
        check(next, "avif image")?;
        Ok(guard)
    }

    fn image(&self) -> eros::Result<&avifImage> {
        // SAFETY: `self.ptr` stays live while `self` is borrowed and was
        // decoded in `new`.
        let frame = unsafe { (*self.ptr).image };
        if frame.is_null() {
            eros::bail!("avif: no image");
        }
        // SAFETY: `frame` is non-null and owned by `self.ptr` while `&self`
        // stays live.
        Ok(unsafe { &*frame })
    }
}

impl Drop for DecoderGuard<'_> {
    fn drop(&mut self) {
        use libavif_sys::avifDecoderDestroy;

        // SAFETY: the guard solely owns the decoder from
        // `avifDecoderCreate`, and drop runs exactly once.
        unsafe {
            avifDecoderDestroy(self.ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_plane_handles_contiguous_and_padded_strides() {
        let raw_u8 = [10_u8, 20, 30, 40];
        // SAFETY: `raw_u8` holds two contiguous 2-byte rows.
        let contiguous_u8 =
            unsafe { copy_plane::<u8>(raw_u8.as_ptr(), 2, 2, 2) };
        assert!(matches!(
            contiguous_u8,
            Ok(bytes) if *bytes == [10, 20, 30, 40]
        ));

        let padded_u8 = [10_u8, 20, 0xff, 0xff, 30, 40];
        // SAFETY: `padded_u8` holds 2 rows at stride 4 with the last row 2 bytes wide.
        let strided_u8 =
            unsafe { copy_plane::<u8>(padded_u8.as_ptr(), 4, 2, 2) };
        assert!(matches!(
            strided_u8,
            Ok(bytes) if *bytes == [10, 20, 30, 40]
        ));

        let padded_u16 = [100_u16, 200, 0xffff, 300, 400];
        let raw_u16: &[u8] = bytemuck::cast_slice(&padded_u16);
        // SAFETY: `raw_u16` holds 2 rows of 4 bytes spaced by stride 6 (10 bytes total).
        let strided_u16 =
            unsafe { copy_plane::<u16>(raw_u16.as_ptr(), 6, 2, 2) };
        assert!(matches!(
            strided_u16,
            Ok(words) if *words == [100, 200, 300, 400]
        ));
    }
}
