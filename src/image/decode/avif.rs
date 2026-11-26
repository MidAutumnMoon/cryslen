//! AVIF decode through libavif. The FFI `unsafe` is confined to this
//! module.

use std::marker::PhantomData;

use bytemuck::Pod;
use image::metadata::Orientation;
use libavif_sys::avifDecoder;
use libavif_sys::avifImage;
use libavif_sys::avifResult;

use crate::image::pixels;
use crate::image::pixels::Image;
use crate::image::pixels::SamplePlanes;
use crate::image::pixels::Subsampling;
use crate::image::pixels::WideDepth;

/// dav1d parallelism saturates quickly on stills, so the thread count
/// is capped.
const MAX_DAV1D_THREADS: i32 = 8;

enum AvifDepth {
    U8,
    U16(WideDepth),
}

pub(super) fn decode(
    bytes: &[u8],
    max_threads: usize,
) -> eros::Result<Image> {
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

    use crate::image::pixels::Format;
    use crate::image::pixels::Matrix;
    use crate::image::pixels::Transfer;
    use crate::image::pixels::YuvSpec;

    let decoder = DecoderGuard::new(bytes, max_threads)?;
    let frame = decoder.image()?;

    let (width, height) = (frame.width, frame.height);
    let avif_depth = match frame.depth {
        8 => AvifDepth::U8,
        10 => AvifDepth::U16(WideDepth::D10),
        12 => AvifDepth::U16(WideDepth::D12),
        16 => AvifDepth::U16(WideDepth::D16),
        _ => eros::bail!("avif: bad depth"),
    };
    pixels::admit_dims(width, height, "avif: bad dimensions")?;
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
    if frame.yuvFormat == AVIF_PIXEL_FORMAT_YUV400 {
        let planes = match avif_depth {
            AvifDepth::U8 => {
                // SAFETY: the guard keeps `frame`'s planes live; `copy_plane` rejects null/short strides.
                let (luma, alpha) =
                    unsafe { copy_gray_planes::<u8>(frame) }?;
                SamplePlanes::U8 {
                    planes: [luma],
                    alpha,
                }
            }
            AvifDepth::U16(depth) => {
                // SAFETY: the guard keeps `frame`'s planes live; `copy_plane` rejects null/short strides.
                let (luma, alpha) =
                    unsafe { copy_gray_planes::<u16>(frame) }?;
                SamplePlanes::U16 {
                    depth,
                    planes: [luma],
                    alpha,
                }
            }
        };
        return Ok(Image::gray(
            width,
            height,
            Format::Avif,
            transfer,
            full_range,
            planes,
        )?
        .apply_orientation(orientation));
    }

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
    if pixels::orientation_swaps_dims(orientation)
        && subsampling == Subsampling::Yuv422
    {
        eros::bail!("avif: unsupported rotated 4:2:2");
    }

    let planes = match avif_depth {
        AvifDepth::U8 => {
            // SAFETY: the guard keeps `frame`'s planes live; `copy_plane` rejects null/short strides.
            let (yuv, alpha) =
                unsafe { copy_yuv_planes::<u8>(frame, subsampling) }?;
            SamplePlanes::U8 { planes: yuv, alpha }
        }
        AvifDepth::U16(depth) => {
            // SAFETY: the guard keeps `frame`'s planes live; `copy_plane` rejects null/short strides.
            let (yuv, alpha) =
                unsafe { copy_yuv_planes::<u16>(frame, subsampling) }?;
            SamplePlanes::U16 {
                depth,
                planes: yuv,
                alpha,
            }
        }
    };

    Ok(Image::yuv(
        width,
        height,
        Format::Avif,
        transfer,
        YuvSpec {
            subsampling,
            full_range,
            matrix,
        },
        planes,
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
        .unwrap_or(MAX_DAV1D_THREADS)
        .clamp(1, MAX_DAV1D_THREADS)
}

fn check(code: avifResult, what: &'static str) -> eros::Result<()> {
    use std::ffi::CStr;

    use libavif_sys::AVIF_RESULT_OK;
    use libavif_sys::avifResultToString;

    if code == AVIF_RESULT_OK {
        return Ok(());
    }
    // SAFETY: `avifResultToString` returns a static NUL-terminated string for
    // any `avifResult`.
    let message_pointer = unsafe { avifResultToString(code) };
    // SAFETY: `message_pointer` points to a static NUL-terminated string.
    let detail =
        unsafe { CStr::from_ptr(message_pointer) }.to_string_lossy();
    eros::bail!("{what}: {detail}")
}

type GrayPlanes<Sample> = (Box<[Sample]>, Option<Box<[Sample]>>);
type YuvPlanes<Sample> = ([Box<[Sample]>; 3], Option<Box<[Sample]>>);

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
    Ok((luma, alpha))
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
    let (luma, alpha) = unsafe { copy_gray_planes::<Sample>(frame) }?;
    let (_, (chroma_width, chroma_height)) =
        pixels::plane_dims((frame.width, frame.height), subsampling);
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
    Ok(([luma, cb, cr], alpha))
}

/// # Safety
///
/// When non-null, `plane_pointer` must hold `height` live rows spaced by
/// `row_stride` bytes.
unsafe fn copy_plane<Sample>(
    plane_pointer: *const u8,
    row_stride: u32,
    width: u32,
    height: u32,
) -> eros::Result<Box<[Sample]>>
where
    Sample: Copy + Default + Pod,
{
    if plane_pointer.is_null() {
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
    let mut samples = vec![Sample::default(); width * height];
    // SAFETY: `plane_pointer` holds `height` live rows at `row_stride >= row_bytes`.
    unsafe {
        copy_strided_bytes(
            plane_pointer,
            row_stride,
            row_bytes,
            height,
            bytemuck::cast_slice_mut(&mut samples),
        );
    }
    Ok(samples.into_boxed_slice())
}

/// # Safety
///
/// `plane_pointer` must hold `height` live rows at `row_stride`, each at
/// least `row_bytes` wide, and `target.len() == row_bytes * height`.
unsafe fn copy_strided_bytes(
    plane_pointer: *const u8,
    row_stride: usize,
    row_bytes: usize,
    height: usize,
    target: &mut [u8],
) {
    use std::slice;

    if row_stride == row_bytes {
        // SAFETY: `row_stride == row_bytes` means all rows are contiguous.
        let source = unsafe {
            slice::from_raw_parts(plane_pointer, row_bytes * height)
        };
        target.copy_from_slice(source);
        return;
    }
    for (row, target_row) in target.chunks_exact_mut(row_bytes).enumerate()
    {
        // SAFETY: `row < height`, so the row pointer stays within the allocated plane.
        let row_pointer = unsafe { plane_pointer.add(row * row_stride) };
        // SAFETY: each row holds `row_bytes` live sample bytes.
        let source =
            unsafe { slice::from_raw_parts(row_pointer, row_bytes) };
        target_row.copy_from_slice(source);
    }
}

/// Owns the C decoder for one decode call. The `'bytes` lifetime pins
/// the input, which the decoder reads through a raw pointer until
/// destroyed.
struct DecoderGuard<'bytes> {
    decoder: *mut avifDecoder,
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
        let decoder = unsafe { avifDecoderCreate() };
        if decoder.is_null() {
            eros::bail!("avif: decoder create failed");
        }
        let guard = Self {
            decoder,
            _bytes: PhantomData,
        };
        {
            // SAFETY: `guard.decoder` is newly allocated, non-null, and uniquely
            // owned; the mutable borrow ends before subsequent C calls.
            let raw_decoder = unsafe { &mut *guard.decoder };
            raw_decoder.maxThreads = clamp_threads(max_threads);
            // The decoder's pixel cap runs before the pixel decode, so an
            // oversized hostile file fails at parse instead of funding a
            // multi-gigabyte dav1d picture first. Clear libavif's per-axis
            // edge cap so tall strips under `MAX_PIXELS` still decode.
            raw_decoder.imageSizeLimit = pixels::MAX_PIXELS;
            raw_decoder.imageDimensionLimit = NO_DIMENSION_LIMIT;
            // Opt into 16-bit sample-transform items (off by default in
            // libavif 1.4); gain maps stay off as rendering skips them.
            raw_decoder.imageContentToDecode |=
                AVIF_IMAGE_CONTENT_SAMPLE_TRANSFORMS;
            // Neither EXIF nor XMP is read from decoded AVIF frames; skipping
            // Exif also avoids failing on malformed Exif TIFF offsets.
            let enable = AVIF_TRUE.cast_signed();
            raw_decoder.ignoreExif = enable;
            raw_decoder.ignoreXMP = enable;
        }
        // SAFETY: `guard.decoder` is uniquely owned and `bytes` stays live until
        // `guard` drops.
        let io = unsafe {
            avifDecoderSetIOMemory(
                guard.decoder,
                bytes.as_ptr(),
                bytes.len(),
            )
        };
        check(io, "avif io")?;
        // SAFETY: `guard.decoder` is valid with attached input bytes.
        let parsed = unsafe { avifDecoderParse(guard.decoder) };
        check(parsed, "avif parse")?;
        // SAFETY: `guard.decoder` parsed cleanly above.
        let next = unsafe { avifDecoderNextImage(guard.decoder) };
        check(next, "avif image")?;
        Ok(guard)
    }

    fn image(&self) -> eros::Result<&avifImage> {
        // SAFETY: `self.decoder` stays live while `self` is borrowed and was
        // decoded in `new`.
        let frame = unsafe { (*self.decoder).image };
        if frame.is_null() {
            eros::bail!("avif: no image");
        }
        // SAFETY: `frame` is non-null and owned by `self.decoder` while `&self`
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
            avifDecoderDestroy(self.decoder);
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;
    use crate::image::decode::codec;
    use crate::image::pixels::Format;
    use crate::image::pixels::Pixels;

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

    // 32x16 10-bit 4:0:0 fixture, a black-to-white grayscale gradient.
    #[test]
    fn avif_fixture_decodes() {
        let page = codec::decode(
            include_bytes!("../../../tests/fixtures/mono.avif"),
            "a.avif",
        )
        .expect("mono avif decodes");
        assert_eq!(page.format(), Format::Avif);
        assert_eq!((page.width(), page.height()), (32, 16));
        let Pixels::Gray {
            planes:
                SamplePlanes::U16 {
                    planes: [luma],
                    alpha,
                    ..
                },
            ..
        } = page.pixels()
        else {
            panic!("mono avif must decode to 16-bit gray plane")
        };
        assert_eq!(luma.len(), 32 * 16);
        assert!(luma.first() < luma.last());
        assert!(alpha.is_none());
    }
}
