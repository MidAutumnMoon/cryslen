//! WebP decode through libwebp. FFI `unsafe` is confined to this module.

#![expect(
    clippy::little_endian_bytes,
    reason = "RIFF container specifies little-endian byte order"
)]

use std::marker::PhantomData;
use std::ptr;

use image::metadata::Orientation;
use libwebp_sys::VP8StatusCode;
use libwebp_sys::WEBP_CSP_MODE;
use libwebp_sys::WebPAnimDecoder;
use libwebp_sys::WebPAnimDecoderOptions;
use libwebp_sys::WebPAnimInfo;
use libwebp_sys::WebPBitstreamFeatures;
use libwebp_sys::WebPData;
use libwebp_sys::WebPDecoderConfig;

use super::Decoded;
use super::Format;
use super::Subsampling;

/// `features.format` value for lossy VP8 bitstreams (`decode.h`).
const WEBP_FORMAT_LOSSY: i32 = 1;

pub(super) fn decode(
    bytes: &[u8],
    max_threads: usize,
) -> eros::Result<Decoded> {
    let features = read_features(bytes)?;
    let (Ok(width), Ok(height)) = (
        u32::try_from(features.width),
        u32::try_from(features.height),
    ) else {
        eros::bail!("webp: bad dimensions");
    };
    super::admit_dims(width, height, "webp: bad dimensions")?;

    let orientation = exif_orientation(bytes);
    let use_threads = max_threads > 1;

    if features.has_animation != 0 {
        return decode_animated_first_frame(bytes, orientation, use_threads);
    }
    if features.format == WEBP_FORMAT_LOSSY && features.has_alpha == 0 {
        return decode_yuv(bytes, (width, height), orientation, use_threads);
    }
    decode_rgba(bytes, (width, height), orientation, use_threads)
}

fn read_features(bytes: &[u8]) -> eros::Result<WebPBitstreamFeatures> {
    use libwebp_sys::WebPGetFeatures;

    let mut features = WebPBitstreamFeatures {
        width: 0,
        height: 0,
        has_alpha: 0,
        has_animation: 0,
        format: 0,
        pad: [0; 5],
    };
    // SAFETY: `bytes` is live and `features` is initialized.
    let status = unsafe {
        WebPGetFeatures(bytes.as_ptr(), bytes.len(), &raw mut features)
    };
    check_status(status, "webp header")?;
    Ok(features)
}

fn init_external_config(
    guard: &mut DecoderConfigGuard,
    dims: (u32, u32),
    colorspace: WEBP_CSP_MODE,
    use_threads: bool,
) -> eros::Result<()> {
    let config = &mut guard.0;
    config.options.use_threads = i32::from(use_threads);
    config.output.colorspace = colorspace;
    config.output.width = i32::try_from(dims.0)?;
    config.output.height = i32::try_from(dims.1)?;
    config.output.is_external_memory = 1;
    Ok(())
}

fn decode_yuv(
    bytes: &[u8],
    dims: (u32, u32),
    orientation: Orientation,
    use_threads: bool,
) -> eros::Result<Decoded> {
    use libwebp_sys::WebPDecode;
    use libwebp_sys::WebPYUVABuffer;

    use super::Matrix;
    use super::Samples;
    use super::Transfer;
    use super::YuvImage;
    use super::YuvPlanes;

    let ((luma_width, luma_height), (chroma_width, chroma_height)) =
        super::plane_dims(dims, Subsampling::Yuv420);
    let mut luma = vec![0_u8; luma_width as usize * luma_height as usize];
    let mut cb =
        vec![0_u8; chroma_width as usize * chroma_height as usize];
    let mut cr =
        vec![0_u8; chroma_width as usize * chroma_height as usize];

    let mut guard = DecoderConfigGuard::new()?;
    init_external_config(
        &mut guard,
        dims,
        WEBP_CSP_MODE::MODE_YUV,
        use_threads,
    )?;
    let luma_stride = i32::try_from(luma_width)?;
    let chroma_stride = i32::try_from(chroma_width)?;
    guard.0.output.u.YUVA = WebPYUVABuffer {
        y: luma.as_mut_ptr(),
        u: cb.as_mut_ptr(),
        v: cr.as_mut_ptr(),
        a: ptr::null_mut(),
        y_stride: luma_stride,
        u_stride: chroma_stride,
        v_stride: chroma_stride,
        a_stride: 0,
        y_size: luma.len(),
        u_size: cb.len(),
        v_size: cr.len(),
        a_size: 0,
    };

    // SAFETY: `YUVA` points into live `luma`, `cb`, and `cr` matching `dims`
    // with `is_external_memory = 1`.
    let status = unsafe {
        WebPDecode(bytes.as_ptr(), bytes.len(), &raw mut guard.0)
    };
    drop(guard);
    check_status(status, "webp decode")?;

    Ok(Decoded::yuv(
        dims.0,
        dims.1,
        Format::Webp,
        Transfer::Srgb,
        YuvImage {
            subsampling: Subsampling::Yuv420,
            full_range: false,
            matrix: Matrix::Bt601,
            samples: Samples::U8(YuvPlanes {
                luma: luma.into_boxed_slice(),
                cb: cb.into_boxed_slice(),
                cr: cr.into_boxed_slice(),
                alpha: None,
            }),
        },
    )?
    .apply_orientation(orientation))
}

fn decode_rgba(
    bytes: &[u8],
    dims: (u32, u32),
    orientation: Orientation,
    use_threads: bool,
) -> eros::Result<Decoded> {
    use libwebp_sys::WebPDecode;
    use libwebp_sys::WebPRGBABuffer;

    let byte_len = super::rgba_byte_len(dims.0, dims.1)?;
    let mut rgba = vec![0_u8; byte_len];

    let mut guard = DecoderConfigGuard::new()?;
    init_external_config(
        &mut guard,
        dims,
        WEBP_CSP_MODE::MODE_RGBA,
        use_threads,
    )?;
    let bytes_per_pixel = u32::try_from(super::RGBA_BYTES_PER_PIXEL)?;
    let stride = i32::try_from(dims.0.saturating_mul(bytes_per_pixel))?;
    guard.0.output.u.RGBA = WebPRGBABuffer {
        rgba: rgba.as_mut_ptr(),
        stride,
        size: rgba.len(),
    };

    // SAFETY: `RGBA` points into live `rgba` sized for `dims` with
    // `is_external_memory = 1`.
    let status = unsafe {
        WebPDecode(bytes.as_ptr(), bytes.len(), &raw mut guard.0)
    };
    drop(guard);
    check_status(status, "webp decode")?;

    finish_rgba(rgba.into_boxed_slice(), dims, orientation)
}

fn finish_rgba(
    rgba: Box<[u8]>,
    dims: (u32, u32),
    orientation: Orientation,
) -> eros::Result<Decoded> {
    use super::Transfer;

    Ok(Decoded::rgba(dims.0, dims.1, Format::Webp, Transfer::Srgb, rgba)?
        .apply_orientation(orientation))
}

fn decode_animated_first_frame(
    bytes: &[u8],
    orientation: Orientation,
    use_threads: bool,
) -> eros::Result<Decoded> {
    use std::mem::MaybeUninit;
    use std::slice;

    use libwebp_sys::WebPAnimDecoderGetNext;
    use libwebp_sys::WebPAnimDecoderOptionsInit;

    let mut options = MaybeUninit::<WebPAnimDecoderOptions>::uninit();
    // SAFETY: `options` points to aligned uninitialized `WebPAnimDecoderOptions`.
    let init_ok =
        unsafe { WebPAnimDecoderOptionsInit(options.as_mut_ptr()) };
    if init_ok == 0 {
        eros::bail!("webp: anim options init failed");
    }
    // SAFETY: `WebPAnimDecoderOptionsInit` succeeded.
    let mut options = unsafe { options.assume_init() };
    options.color_mode = WEBP_CSP_MODE::MODE_RGBA;
    options.use_threads = i32::from(use_threads);

    let decoder = AnimDecoderGuard::new(bytes, &options)?;
    let info = decoder.info()?;
    let dims = (info.canvas_width, info.canvas_height);
    let byte_len = super::rgba_byte_len(dims.0, dims.1)?;

    let mut frame_ptr = ptr::null_mut();
    let mut timestamp = 0_i32;
    // SAFETY: `decoder.ptr` is live and `bytes` outlives it via `'bytes`.
    let got_frame = unsafe {
        WebPAnimDecoderGetNext(
            decoder.ptr,
            &raw mut frame_ptr,
            &raw mut timestamp,
        )
    };
    if got_frame == 0 || frame_ptr.is_null() {
        eros::bail!("webp: missing animation frame");
    }
    // SAFETY: `WebPAnimDecoderGetNext` returned `byte_len` RGBA bytes owned
    // by `decoder` while `decoder` remains live.
    let frame_slice =
        unsafe { slice::from_raw_parts(frame_ptr, byte_len) };
    let owned = frame_slice.to_vec().into_boxed_slice();
    drop(decoder);

    finish_rgba(owned, dims, orientation)
}

struct DecoderConfigGuard(WebPDecoderConfig);

impl DecoderConfigGuard {
    fn new() -> eros::Result<Self> {
        WebPDecoderConfig::new()
            .map(Self)
            .map_err(|()| eros::error!("webp: decoder config init failed"))
    }
}

impl Drop for DecoderConfigGuard {
    fn drop(&mut self) {
        use libwebp_sys::WebPFreeDecBuffer;

        // SAFETY: `self.0.output` was initialized by `WebPDecoderConfig::new`.
        unsafe {
            WebPFreeDecBuffer(&raw mut self.0.output);
        }
    }
}

struct AnimDecoderGuard<'bytes> {
    ptr: *mut WebPAnimDecoder,
    _bytes: PhantomData<&'bytes [u8]>,
}

impl<'bytes> AnimDecoderGuard<'bytes> {
    fn new(
        bytes: &'bytes [u8],
        options: &WebPAnimDecoderOptions,
    ) -> eros::Result<Self> {
        use libwebp_sys::WebPAnimDecoderNew;

        let data = WebPData {
            bytes: bytes.as_ptr(),
            size: bytes.len(),
        };
        // SAFETY: `data` (`bytes` outlives the guard via `'bytes`) and `options` are initialized.
        let ptr = unsafe { WebPAnimDecoderNew(&raw const data, options) };
        if ptr.is_null() {
            eros::bail!("webp: anim decoder init failed");
        }
        Ok(Self {
            ptr,
            _bytes: PhantomData,
        })
    }

    fn info(&self) -> eros::Result<WebPAnimInfo> {
        use libwebp_sys::WebPAnimDecoderGetInfo;

        let mut info = WebPAnimInfo::default();
        // SAFETY: `self.ptr` is live (`AnimDecoderGuard::new`) and `info` is initialized.
        let ok =
            unsafe { WebPAnimDecoderGetInfo(self.ptr, &raw mut info) };
        if ok == 0 {
            eros::bail!("webp: anim info unavailable");
        }
        Ok(info)
    }
}

impl Drop for AnimDecoderGuard<'_> {
    fn drop(&mut self) {
        use libwebp_sys::WebPAnimDecoderDelete;

        // SAFETY: `self.ptr` is uniquely owned from `WebPAnimDecoderNew`.
        unsafe {
            WebPAnimDecoderDelete(self.ptr);
        }
    }
}

fn check_status(
    status: VP8StatusCode,
    what: &'static str,
) -> eros::Result<()> {
    match status {
        VP8StatusCode::VP8_STATUS_OK => Ok(()),
        VP8StatusCode::VP8_STATUS_OUT_OF_MEMORY => {
            eros::bail!("{what}: out of memory")
        }
        VP8StatusCode::VP8_STATUS_INVALID_PARAM => {
            eros::bail!("{what}: invalid parameter")
        }
        VP8StatusCode::VP8_STATUS_BITSTREAM_ERROR => {
            eros::bail!("{what}: corrupt bitstream")
        }
        VP8StatusCode::VP8_STATUS_UNSUPPORTED_FEATURE => {
            eros::bail!("{what}: unsupported feature")
        }
        VP8StatusCode::VP8_STATUS_SUSPENDED => {
            eros::bail!("{what}: suspended")
        }
        VP8StatusCode::VP8_STATUS_USER_ABORT => {
            eros::bail!("{what}: aborted")
        }
        VP8StatusCode::VP8_STATUS_NOT_ENOUGH_DATA => {
            eros::bail!("{what}: truncated data")
        }
    }
}

fn exif_orientation(bytes: &[u8]) -> Orientation {
    find_exif_chunk(bytes)
        .map(|chunk| chunk.strip_prefix(b"Exif\0\0").unwrap_or(chunk))
        .and_then(Orientation::from_exif_chunk)
        .unwrap_or(Orientation::NoTransforms)
}

const RIFF_HEADER_BYTES: usize = 12;
const CHUNK_HEADER_BYTES: usize = 8;
const FOURCC_BYTES: usize = 4;

// WebP RIFF spec: only `VP8X` extended containers may carry an `EXIF` chunk.
fn find_exif_chunk(bytes: &[u8]) -> Option<&[u8]> {
    if bytes.get(..FOURCC_BYTES) != Some(b"RIFF")
        || bytes.get(CHUNK_HEADER_BYTES..RIFF_HEADER_BYTES)
            != Some(b"WEBP")
    {
        return None;
    }
    if bytes.get(RIFF_HEADER_BYTES..RIFF_HEADER_BYTES + FOURCC_BYTES)
        != Some(b"VP8X")
    {
        return None;
    }
    let mut offset = RIFF_HEADER_BYTES;
    while let Some(header) =
        bytes.get(offset..offset.checked_add(CHUNK_HEADER_BYTES)?)
    {
        let (fourcc, rest) = header.split_first_chunk::<FOURCC_BYTES>()?;
        let len_bytes = rest.first_chunk::<FOURCC_BYTES>()?;
        let chunk_len = u32::from_le_bytes(*len_bytes) as usize;
        let payload_start = offset.checked_add(CHUNK_HEADER_BYTES)?;
        let payload_end = payload_start.checked_add(chunk_len)?;
        if fourcc == b"EXIF" {
            return bytes.get(payload_start..payload_end);
        }
        let padded_len = chunk_len.checked_add(chunk_len & 1)?;
        offset = payload_start.checked_add(padded_len)?;
    }
    None
}

#[cfg(test)]
#[expect(
    clippy::big_endian_bytes,
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential and EXIF test fixtures write big-endian TIFF headers"
)]
mod tests {
    use super::*;
    use crate::decode;
    use crate::decode::Pixels;

    fn encode_rgba(
        width: u32,
        height: u32,
        rgba: &[u8],
        lossless: bool,
    ) -> Vec<u8> {
        use std::slice;

        use libwebp_sys::WebPEncodeLosslessRGBA;
        use libwebp_sys::WebPEncodeRGBA;
        use libwebp_sys::WebPFree;

        const LOSSY_QUALITY: f32 = 90.0;

        let stride = i32::try_from(width * 4).expect("stride fits i32");
        let width = i32::try_from(width).expect("width fits i32");
        let height = i32::try_from(height).expect("height fits i32");
        let mut out_ptr = ptr::null_mut();
        let len = if lossless {
            // SAFETY: `rgba` holds `width * height * 4` bytes for this call.
            unsafe {
                WebPEncodeLosslessRGBA(
                    rgba.as_ptr(),
                    width,
                    height,
                    stride,
                    &raw mut out_ptr,
                )
            }
        } else {
            // SAFETY: `rgba` holds `width * height * 4` bytes for this call.
            unsafe {
                WebPEncodeRGBA(
                    rgba.as_ptr(),
                    width,
                    height,
                    stride,
                    LOSSY_QUALITY,
                    &raw mut out_ptr,
                )
            }
        };
        assert!(len > 0 && !out_ptr.is_null(), "webp test encode failed");
        // SAFETY: libwebp returned a non-null buffer of `len` bytes.
        let owned =
            unsafe { slice::from_raw_parts(out_ptr, len) }.to_vec();
        // SAFETY: `out_ptr` was allocated by libwebp.
        unsafe {
            WebPFree(out_ptr.cast());
        }
        owned
    }

    fn riff_webp(chunks: &[([u8; 4], &[u8])]) -> Vec<u8> {
        let mut body = Vec::new();
        for (fourcc, payload) in chunks {
            body.extend_from_slice(fourcc);
            body.extend_from_slice(
                &u32::try_from(payload.len())
                    .expect("chunk fits u32")
                    .to_le_bytes(),
            );
            body.extend_from_slice(payload);
            if payload.len() % 2 == 1 {
                body.push(0);
            }
        }
        let mut riff = Vec::new();
        riff.extend_from_slice(b"RIFF");
        riff.extend_from_slice(
            &u32::try_from(4 + body.len())
                .expect("riff fits u32")
                .to_le_bytes(),
        );
        riff.extend_from_slice(b"WEBP");
        riff.extend_from_slice(&body);
        riff
    }

    fn vp8x_payload(flags: u8, width: u32, height: u32) -> [u8; 10] {
        let mut vp8x = [0_u8; 10];
        vp8x[0] = flags;
        let width_m1 = (width - 1).to_le_bytes();
        let height_m1 = (height - 1).to_le_bytes();
        vp8x[4..7].copy_from_slice(&width_m1[..3]);
        vp8x[7..10].copy_from_slice(&height_m1[..3]);
        vp8x
    }

    fn wrap_vp8x_with_exif(
        plain_webp: &[u8],
        width: u32,
        height: u32,
        orientation: u16,
    ) -> Vec<u8> {
        const VP8X_EXIF_FLAG: u8 = 0x08;

        let vp8x = vp8x_payload(VP8X_EXIF_FLAG, width, height);
        let chunk_start = RIFF_HEADER_BYTES;
        let fourcc: [u8; FOURCC_BYTES] = plain_webp
            .get(chunk_start..chunk_start + FOURCC_BYTES)
            .expect("chunk fourcc present")
            .try_into()
            .expect("fourcc fits");
        let len_bytes: [u8; FOURCC_BYTES] = plain_webp
            .get(
                chunk_start + FOURCC_BYTES
                    ..chunk_start + CHUNK_HEADER_BYTES,
            )
            .expect("chunk length present")
            .try_into()
            .expect("length fits");
        let payload_start = chunk_start + CHUNK_HEADER_BYTES;
        let payload_len = u32::from_le_bytes(len_bytes) as usize;
        let bitstream = plain_webp
            .get(payload_start..payload_start + payload_len)
            .expect("bitstream payload present");

        let mut exif = Vec::new();
        exif.extend_from_slice(b"Exif\0\0MM\0*\0\0\0\x08");
        exif.extend_from_slice(&1_u16.to_be_bytes());
        exif.extend_from_slice(&0x0112_u16.to_be_bytes());
        exif.extend_from_slice(&3_u16.to_be_bytes());
        exif.extend_from_slice(&1_u32.to_be_bytes());
        exif.extend_from_slice(&orientation.to_be_bytes());
        exif.extend_from_slice(&[0_u8; 2]);

        riff_webp(&[
            (*b"VP8X", &vp8x),
            (fourcc, bitstream),
            (*b"EXIF", &exif),
        ])
    }

    fn wrap_animated_webp(
        first_frame_webp: &[u8],
        second_frame_webp: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        const VP8X_ANIMATION_FLAG: u8 = 0x02;
        const ANMF_HEADER_BYTES: usize = 16;
        const FRAME_DURATION_MS: u8 = 50;

        let vp8x = vp8x_payload(VP8X_ANIMATION_FLAG, width, height);
        let anim = [0_u8; 6];
        let width_m1 = (width - 1).to_le_bytes();
        let height_m1 = (height - 1).to_le_bytes();
        let make_frame = |frame: &[u8]| {
            let mut header = [0_u8; ANMF_HEADER_BYTES];
            header[6..9].copy_from_slice(&width_m1[..3]);
            header[9..12].copy_from_slice(&height_m1[..3]);
            header[12] = FRAME_DURATION_MS;
            let mut anmf = header.to_vec();
            anmf.extend_from_slice(
                frame
                    .get(RIFF_HEADER_BYTES..)
                    .expect("frame riff header present"),
            );
            anmf
        };
        let first_anmf = make_frame(first_frame_webp);
        let second_anmf = make_frame(second_frame_webp);
        riff_webp(&[
            (*b"VP8X", &vp8x),
            (*b"ANIM", &anim),
            (*b"ANMF", &first_anmf),
            (*b"ANMF", &second_anmf),
        ])
    }

    #[test]
    fn webp_decodes_yuv_rgba_and_animation() {
        use crate::decode::Matrix;
        use crate::decode::Samples;

        let lossy =
            encode_rgba(5, 3, &[200_u8, 120, 60, 255].repeat(15), false);
        let decoded_lossy =
            decode::decode(&lossy, "page.webp").expect("lossy webp");
        assert_eq!(decoded_lossy.format(), Format::Webp);
        assert_eq!(
            (decoded_lossy.width(), decoded_lossy.height()),
            (5, 3)
        );
        let Pixels::Yuv(yuv) = decoded_lossy.pixels() else {
            panic!("opaque lossy webp must decode to yuv");
        };
        assert_eq!(yuv.subsampling, Subsampling::Yuv420);
        assert_eq!(yuv.matrix, Matrix::Bt601);
        assert!(!yuv.full_range);
        assert!(matches!(
            &yuv.samples,
            Samples::U8(planes) if planes.alpha.is_none()
        ));

        let lossless =
            encode_rgba(4, 2, &[10_u8, 20, 30, 255].repeat(8), true);
        let decoded_lossless =
            decode::decode(&lossless, "lossless.webp").expect("lossless");
        let Pixels::Rgba8(lossless_bytes) = decoded_lossless.pixels()
        else {
            panic!("lossless webp must decode to rgba");
        };
        assert_eq!(lossless_bytes.get(..4), Some(&[10, 20, 30, 255][..]));

        let alpha =
            encode_rgba(3, 2, &[40_u8, 80, 120, 128].repeat(6), false);
        let decoded_alpha =
            decode::decode(&alpha, "alpha.webp").expect("alpha webp");
        let Pixels::Rgba8(alpha_bytes) = decoded_alpha.pixels() else {
            panic!("alpha webp must decode to rgba");
        };
        assert_eq!(alpha_bytes.len(), 3 * 2 * 4);

        let second_frame =
            encode_rgba(4, 2, &[200_u8, 100, 50, 255].repeat(8), true);
        let animated = wrap_animated_webp(&lossless, &second_frame, 4, 2);
        let decoded_anim =
            decode::decode(&animated, "anim.webp").expect("anim webp");
        let Pixels::Rgba8(anim_bytes) = decoded_anim.pixels() else {
            panic!("animated webp must decode to rgba");
        };
        assert_eq!(anim_bytes.get(..4), Some(&[10, 20, 30, 255][..]));
    }

    #[test]
    fn exif_orientation_rotates_webp_output() {
        let plain =
            encode_rgba(4, 2, &[50_u8, 100, 150, 255].repeat(8), false);
        // Orientation 6 = 90 deg CW: 4x2 becomes 2x4.
        let rotated = wrap_vp8x_with_exif(&plain, 4, 2, 6);
        let decoded = decode::decode(&rotated, "rot.webp")
            .expect("rotated webp decodes");
        assert_eq!((decoded.width(), decoded.height()), (2, 4));
    }
}
