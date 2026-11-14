//! Decoding: raw page bytes to `Decoded` (RGBA or YUV planes).

#![expect(
    clippy::definition_in_module_root,
    reason = "the decode tree's shared vocabulary lives in its module root by design"
)]

use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;

use ::image::Delay;
use eros::ErrorUnion;
use eros::IntoUnion as _;
use serde::Deserialize;
use serde::Serialize;

use crate::fault::DecodeError;
use crate::fault::DecodeRefusal;

mod avif;
mod image;
mod jxl;

/// Construct via [`Decoded::rgba`] / [`Decoded::yuv`], which validate the
/// pixel extents against the dimensions.
#[derive(Clone)]
pub struct Decoded {
    /// This representation's pixel dimensions, not the page's.
    width: u32,
    height: u32,
    pixels: Pixels,
    /// Sample encoding the renderer must undo before the framebuffer.
    transfer: Transfer,
    format: Format,
    page_dims: (u32, u32),
}

/// Sample encoding. SDR pages are sRGB-class gamma, passed through to the
/// sRGB framebuffer; PQ pages are absolute-luminance HDR, tone-mapped to
/// SDR in the shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Transfer {
    Srgb,
    Pq,
}

impl Decoded {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn pixels(&self) -> &Pixels {
        &self.pixels
    }

    pub fn transfer(&self) -> Transfer {
        self.transfer
    }

    pub fn format(&self) -> Format {
        self.format
    }

    /// The page's dimensions, however far this representation is
    /// downsampled.
    pub fn page_dims(&self) -> (u32, u32) {
        self.page_dims
    }

    #[must_use]
    pub fn with_page_dims(mut self, page_dims: (u32, u32)) -> Self {
        self.page_dims = page_dims;
        self
    }

    pub fn rgba(
        width: u32,
        height: u32,
        format: Format,
        transfer: Transfer,
        bytes: Box<[u8]>,
    ) -> Result<Self, DecodeRefusal> {
        validate_rgba_extent(width, height, bytes.len())?;
        Ok(Self {
            width,
            height,
            pixels: Pixels::Rgba8(bytes),
            transfer,
            format,
            page_dims: (width, height),
        })
    }

    pub fn yuv(
        width: u32,
        height: u32,
        format: Format,
        transfer: Transfer,
        yuv: YuvImage,
    ) -> Result<Self, DecodeRefusal> {
        let wide = yuv.depth > BYTE_DEPTH;
        let matches_depth =
            |plane: &Plane| matches!(plane, Plane::U16(_)) == wide;
        if !matches_depth(&yuv.luma)
            || yuv
                .alpha
                .as_ref()
                .is_some_and(|alpha| !matches_depth(alpha))
            || (yuv.subsampling != Subsampling::Mono
                && (!matches_depth(&yuv.cb) || !matches_depth(&yuv.cr)))
        {
            return Err(DecodeRefusal::Corrupt("yuv plane depth"));
        }
        validate_yuv_extents(
            (width, height),
            yuv.depth,
            yuv.subsampling,
            [yuv.luma.byte_len(), yuv.cb.byte_len(), yuv.cr.byte_len()],
            yuv.alpha.as_ref().map(Plane::byte_len),
        )?;
        Ok(Self {
            width,
            height,
            pixels: Pixels::Yuv(yuv),
            transfer,
            format,
            page_dims: (width, height),
        })
    }

    pub fn resident_bytes(&self) -> usize {
        match &self.pixels {
            Pixels::Rgba8(bytes) => bytes.len(),
            Pixels::Yuv(yuv) => {
                yuv.luma.byte_len()
                    + yuv.cb.byte_len()
                    + yuv.cr.byte_len()
                    + yuv.alpha.as_ref().map_or(0, Plane::byte_len)
            }
        }
    }
}

/// 8-bit storage; deeper planes store one `u16` per sample.
pub const BYTE_DEPTH: u8 = 8;

const MAX_SAMPLE_DEPTH: u8 = 16;

/// Pixel count of `(width, height)` when non-empty and within
/// [`MAX_PIXELS`].
pub fn pixel_count(width: u32, height: u32) -> Option<u32> {
    width
        .checked_mul(height)
        .filter(|&count| (1..=MAX_PIXELS).contains(&count))
}

/// Validated [`pixel_count`], mapping failure to
/// [`DecodeRefusal::Corrupt`].
pub fn admit_dims(
    width: u32,
    height: u32,
    bad_dims_reason: &'static str,
) -> Result<u32, DecodeRefusal> {
    pixel_count(width, height)
        .ok_or(DecodeRefusal::Corrupt(bad_dims_reason))
}

pub fn validate_rgba_extent(
    width: u32,
    height: u32,
    bytes: usize,
) -> Result<(), DecodeRefusal> {
    let pixels = admit_dims(width, height, "bad dimensions")?;
    if bytes != pixels as usize * 4 {
        return Err(DecodeRefusal::Corrupt("rgba extent"));
    }
    Ok(())
}

pub fn validate_yuv_extents(
    dims: (u32, u32),
    depth: u8,
    subsampling: Subsampling,
    [luma_bytes, chroma_blue_bytes, chroma_red_bytes]: [usize; 3],
    alpha_bytes: Option<usize>,
) -> Result<(), DecodeRefusal> {
    let (width, height) = dims;
    let luma_pixels = admit_dims(width, height, "bad dimensions")?;
    let bytes_per_sample = match depth {
        1..=BYTE_DEPTH => 1,
        9..=MAX_SAMPLE_DEPTH => size_of::<u16>(),
        _ => return Err(DecodeRefusal::Corrupt("yuv depth")),
    };
    let expected_luma_bytes = luma_pixels as usize * bytes_per_sample;
    let (_, (chroma_width, chroma_height)) =
        plane_dims((width, height), subsampling);
    let expected_chroma_bytes =
        admit_dims(chroma_width, chroma_height, "bad dimensions")?
            as usize
            * bytes_per_sample;
    if luma_bytes != expected_luma_bytes
        || alpha_bytes.is_some_and(|len| len != expected_luma_bytes)
        || (subsampling != Subsampling::Mono
            && (chroma_blue_bytes != expected_chroma_bytes
                || chroma_red_bytes != expected_chroma_bytes))
    {
        return Err(DecodeRefusal::Corrupt("yuv plane extent"));
    }
    Ok(())
}

/// The image crate's 512 MiB allocation ceiling, as RGBA pixels.
#[expect(
    clippy::integer_division,
    reason = "the 512 MiB ceiling in 4-byte pixels divides exactly"
)]
pub const MAX_PIXELS: u32 = (512 << 20) / 4;

#[derive(Clone)]
pub enum Pixels {
    /// Interleaved RGBA, `4 * width * height` bytes, row-major, top-first.
    Rgba8(Box<[u8]>),
    /// Planar YUV, converted to RGB in the renderer's shader.
    Yuv(YuvImage),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
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
    #[must_use]
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

    /// Whether the parent may request playback.
    #[must_use]
    pub fn may_animate(self) -> bool {
        matches!(self, Self::Gif)
    }
}

/// Pass count. `Finite` counts total passes, including the first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Loops {
    Infinite,
    Finite(u32),
}

/// 8-bit planes are one byte per sample; deeper planes are `u16` with
/// `depth` significant bits. Mono carries luma only; alpha, when present,
/// is a luma-sized full-range plane.
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
    pub alpha: Option<Plane>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Subsampling {
    Mono,
    Yuv420,
    Yuv422,
    Yuv444,
}

impl Subsampling {
    /// Chroma texels per luma texel on each axis: the sampling ratio
    /// behind `plane_dims`'s texel counts. Mono carries no chroma, so
    /// its ratio is identity for the renderer's midpoint stub.
    #[must_use]
    pub fn chroma_ratio(self) -> (f32, f32) {
        match self {
            Self::Mono | Self::Yuv444 => (1.0, 1.0),
            Self::Yuv420 => (0.5, 0.5),
            Self::Yuv422 => (0.5, 1.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
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

impl Plane {
    pub fn sample_count(&self) -> usize {
        match self {
            Self::U8(bytes) => bytes.len(),
            Self::U16(words) => words.len(),
        }
    }

    pub fn byte_len(&self) -> usize {
        match self {
            Self::U8(bytes) => bytes.len(),
            Self::U16(words) => words.len() * 2,
        }
    }
}

/// Mono's chroma dims are 1x1 so renderers can bind a midpoint texture.
pub fn plane_dims(
    size: (u32, u32),
    subsampling: Subsampling,
) -> ((u32, u32), (u32, u32)) {
    let (width, height) = size;
    match subsampling {
        Subsampling::Mono => ((width, height), (1, 1)),
        Subsampling::Yuv420 => {
            ((width, height), (width.div_ceil(2), height.div_ceil(2)))
        }
        Subsampling::Yuv422 => {
            ((width, height), (width.div_ceil(2), height))
        }
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

/// The AVIF irot/imir display transform, applied at decode so the renderer
/// and the layout see upright pixels with plain dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayTransform {
    /// Quarter-turns counter-clockwise (the `irot` angle), 0..=3.
    pub quarters: u8,
    /// The `imir` axis: 0 exchanges top and bottom, 1 exchanges left and
    /// right.
    pub mirror_axis: Option<u8>,
}

impl DisplayTransform {
    pub const UPRIGHT: Self = Self {
        quarters: 0,
        mirror_axis: None,
    };

    pub fn swaps_dims(self) -> bool {
        self.quarters % 2 == 1
    }

    #[must_use]
    pub fn out_dims(self, (width, height): (u32, u32)) -> (u32, u32) {
        if self.swaps_dims() {
            (height, width)
        } else {
            (width, height)
        }
    }
}

/// Transforms one row-major plane; empty stub planes stay empty.
pub fn transform_plane(
    plane: Plane,
    dims: (u32, u32),
    transform: DisplayTransform,
) -> Plane {
    if transform == DisplayTransform::UPRIGHT {
        return plane;
    }
    match plane {
        Plane::U8(samples) if samples.is_empty() => Plane::U8(samples),
        Plane::U16(samples) if samples.is_empty() => Plane::U16(samples),
        Plane::U8(samples) => {
            Plane::U8(transform_samples(&samples, dims, transform))
        }
        Plane::U16(samples) => {
            Plane::U16(transform_samples(&samples, dims, transform))
        }
    }
}

/// Mirroring runs before rotation, matching the property order HEIF writers use.
fn transform_samples<T>(
    samples: &[T],
    (width, height): (u32, u32),
    transform: DisplayTransform,
) -> Box<[T]>
where
    T: Copy,
{
    let (out_width, out_height) = transform.out_dims((width, height));
    let (width, height) = (width as usize, height as usize);
    debug_assert_eq!(
        samples.len(),
        width * height,
        "plane sample count must match dimensions"
    );
    let mut out = Vec::with_capacity(samples.len());
    for out_y in 0..out_height as usize {
        for out_x in 0..out_width as usize {
            let (mut source_x, mut source_y) = match transform.quarters {
                1 => (width - 1 - out_y, out_x),
                2 => (width - 1 - out_x, height - 1 - out_y),
                3 => (out_y, height - 1 - out_x),
                _ => (out_x, out_y),
            };
            match transform.mirror_axis {
                Some(1) => source_x = width - 1 - source_x,
                Some(_) => source_y = height - 1 - source_y,
                None => {}
            }
            #[expect(
                clippy::indexing_slicing,
                reason = "source coordinates stay within (width, height)"
            )]
            out.push(samples[source_y * width + source_x]);
        }
    }
    out.into_boxed_slice()
}

/// Decode one page's first raster. `name` is the page filename, used as
/// the extension fallback. Animated pages stream the rest through
/// [`open_animation`].
pub fn decode(bytes: &[u8], name: &str) -> Result<Decoded, DecodeError> {
    let Some(format) = sniff(bytes).or_else(|| format_from_name(name))
    else {
        return Err(ErrorUnion::new(DecodeRefusal::NotAnImage));
    };
    match format {
        Format::Avif => avif::decode(bytes),
        Format::Jxl => jxl::decode(bytes),
        Format::Jpeg
        | Format::Png
        | Format::Gif
        | Format::Webp
        | Format::Bmp
        | Format::Tiff => image::decode(bytes, format),
    }
}

/// An animated page's frame stream. Frames arrive sequentially; there is
/// no seek.
pub struct Animation {
    bytes: Arc<[u8]>,
    format: Format,
    loops: Loops,
    frames: Frames,
}

impl Animation {
    #[must_use]
    pub fn loops(&self) -> Loops {
        self.loops
    }

    /// Restart the pass at frame 0.
    pub fn rewind(&mut self) -> Result<(), DecodeError> {
        self.frames = Frames::new(Arc::clone(&self.bytes), self.format)?;
        Ok(())
    }

    /// `None` means the pass ended.
    pub fn next_frame(&mut self) -> Option<Result<Frame, DecodeError>> {
        self.frames.next_frame()
    }
}

/// One composited animation frame at the page canvas size.
#[derive(Clone)]
pub struct Frame {
    pub image: Decoded,
    pub delay: Duration,
}

struct Frames {
    inner: ::image::Frames<'static>,
    format: Format,
}

impl Frames {
    fn new(bytes: Arc<[u8]>, format: Format) -> Result<Self, DecodeError> {
        use ::image::codecs::gif::GifDecoder;
        use ::image::AnimationDecoder as _;
        use ::image::ImageDecoder as _;

        if format != Format::Gif {
            return Err(ErrorUnion::new(DecodeRefusal::Unsupported(
                "animation format",
            )));
        }
        let decoder = GifDecoder::new(Cursor::new(bytes)).union()?;
        let (width, height) = decoder.dimensions();
        admit_dims(width, height, "bad dimensions").union()?;
        Ok(Self {
            inner: decoder.into_frames(),
            format,
        })
    }

    fn next_frame(&mut self) -> Option<Result<Frame, DecodeError>> {
        let frame = match self.inner.next() {
            None => return None,
            Some(Ok(frame)) => frame,
            Some(Err(error)) => {
                return Some(Err(ErrorUnion::new(error)));
            }
        };
        let delay = delay_duration(frame.delay());
        let buffer = frame.into_buffer();
        let (width, height) = buffer.dimensions();
        let rgba = buffer.into_raw().into_boxed_slice();
        Some(
            Decoded::rgba(
                width,
                height,
                self.format,
                Transfer::Srgb,
                rgba,
            )
            .map(|image| Frame { image, delay })
            .union(),
        )
    }
}

/// `Ok(None)` means the format has no animation path; a still GIF
/// streams one frame.
pub fn open_animation(
    bytes: Arc<[u8]>,
    name: &str,
) -> Result<Option<Animation>, DecodeError> {
    let Some(format) = sniff(&bytes).or_else(|| format_from_name(name))
    else {
        return Err(ErrorUnion::new(DecodeRefusal::NotAnImage));
    };
    if !format.may_animate() {
        return Ok(None);
    }
    let loops = gif_loops(&bytes);
    let frames = Frames::new(Arc::clone(&bytes), format)?;
    tracing::debug!(format = format.label(), ?loops, "animation opened");
    Ok(Some(Animation {
        bytes,
        format,
        loops,
        frames,
    }))
}

/// GIF's NETSCAPE repeat field. A file without the extension reports
/// `Finite(0)`; the `image` wrapper would call that a forever loop.
fn gif_loops(bytes: &[u8]) -> Loops {
    use gif::DecodeOptions;
    use gif::Repeat;

    match DecodeOptions::new().read_info(Cursor::new(bytes)) {
        Ok(decoder) => match decoder.repeat() {
            Repeat::Infinite => Loops::Infinite,
            Repeat::Finite(0) => Loops::Finite(1),
            Repeat::Finite(repeats) => {
                Loops::Finite(u32::from(repeats) + 1)
            }
        },
        Err(error) => {
            tracing::debug!(
                %error,
                "gif loop metadata unreadable; playing once"
            );
            Loops::Finite(1)
        }
    }
}

/// `image` reports delays as an exact millisecond ratio.
fn delay_duration(delay: Delay) -> Duration {
    let (numer, denom) = delay.numer_denom_ms();
    Duration::from_millis(u64::from(numer)) / denom.max(1)
}

/// Format for a page filename's extension. The same table decides what can
/// decode and what gets listed as a page.
pub fn format_from_name(name: &str) -> Option<Format> {
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

fn sniff(bytes: &[u8]) -> Option<Format> {
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(Format::Jpeg);
    }
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A])
    {
        return Some(Format::Png);
    }
    if bytes.starts_with(b"GIF8") {
        return Some(Format::Gif);
    }
    if bytes.starts_with(b"RIFF")
        && bytes.get(8..12).is_some_and(|tag| tag == b"WEBP")
    {
        return Some(Format::Webp);
    }
    if bytes.starts_with(b"BM") {
        return Some(Format::Bmp);
    }
    if bytes.starts_with(&[0x49, 0x49, 0x2A, 0x00])
        || bytes.starts_with(&[0x4D, 0x4D, 0x00, 0x2A])
    {
        return Some(Format::Tiff);
    }
    // Raw codestream or ISOBMFF container signature.
    if bytes.starts_with(&[0xFF, 0x0A])
        || bytes.starts_with(&[
            0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A,
            0x87, 0x0A,
        ])
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
        brands
            .as_chunks::<4>()
            .0
            .iter()
            .any(|brand| brand == b"avif" || brand == b"avis")
    })
}

/// Brand region of the leading `ftyp` box. A box length under 12 still
/// yields the 4-byte major brand.
fn ftyp_brands(bytes: &[u8]) -> Option<&[u8]> {
    #[expect(
        clippy::big_endian_bytes,
        reason = "the ISOBMFF box length is big-endian by specification"
    )]
    let box_len =
        u32::from_be_bytes(bytes.get(..4)?.try_into().ok()?) as usize;
    let kind: [u8; 4] = bytes.get(4..8)?.try_into().ok()?;
    if kind != *b"ftyp" {
        return None;
    }
    let end = box_len.max(12).min(bytes.len());
    bytes.get(8..end)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;

    use crate::testkit;
    use ::image::codecs::gif::Repeat;

    /// The refusal a decode returned, for tests that assert on the
    /// verdict rather than its message.
    fn refusal(
        result: Result<Decoded, DecodeError>,
    ) -> Option<DecodeRefusal> {
        match result {
            Ok(_) => None,
            Err(error) => {
                error.downcast_inner_ref::<DecodeRefusal>().cloned()
            }
        }
    }

    #[test]
    fn format_detection() {
        // sniff: one row per magic
        assert_eq!(sniff(&[0xFF, 0xD8, 0xFF, 0xE0]), Some(Format::Jpeg));
        assert_eq!(sniff(b"\x89PNG\r\n\x1a\n"), Some(Format::Png));
        assert_eq!(sniff(b"GIF89a"), Some(Format::Gif));
        assert_eq!(
            sniff(b"RIFF\x00\x00\x00\x00WEBPVP8 "),
            Some(Format::Webp)
        );
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
        assert_eq!(
            refusal(decode(b"hello world", "noext")),
            Some(DecodeRefusal::NotAnImage)
        );
        assert_eq!(
            refusal(decode(b"hello world", "note.txt")),
            Some(DecodeRefusal::NotAnImage)
        );
        assert_eq!(
            refusal(decode(&[], "x")),
            Some(DecodeRefusal::NotAnImage)
        );
    }

    #[test]
    fn constructor_caps() {
        const WIDE_EDGE: u32 = 32_768;

        Decoded::rgba(
            WIDE_EDGE,
            1,
            Format::Png,
            Transfer::Srgb,
            vec![0_u8; 4 * WIDE_EDGE as usize].into_boxed_slice(),
        )
        .expect("tall or wide strip under MAX_PIXELS constructs");
        assert!(matches!(
            Decoded::rgba(
                0,
                10,
                Format::Png,
                Transfer::Srgb,
                Box::default(),
            ),
            Err(DecodeRefusal::Corrupt("bad dimensions"))
        ));
        // Over the pixel-count cap must reject the page before any buffer
        // is sized for it.
        assert!(matches!(
            Decoded::rgba(
                WIDE_EDGE,
                WIDE_EDGE,
                Format::Png,
                Transfer::Srgb,
                Box::default()
            ),
            Err(DecodeRefusal::Corrupt("bad dimensions"))
        ));
    }

    #[test]
    fn yuv_planes() {
        // odd edges round chroma up. Mono carries a 1x1 chroma stub.
        assert_eq!(
            plane_dims((5, 3), Subsampling::Mono),
            ((5, 3), (1, 1))
        );
        assert_eq!(
            plane_dims((5, 3), Subsampling::Yuv420),
            ((5, 3), (3, 2))
        );
        assert_eq!(
            plane_dims((5, 3), Subsampling::Yuv422),
            ((5, 3), (3, 3))
        );
        assert_eq!(
            plane_dims((5, 3), Subsampling::Yuv444),
            ((5, 3), (5, 3))
        );

        // a well-sized mono page constructs; an empty luma plane does not;
        // a short alpha plane does not either
        let mono = |luma: Plane, alpha: Option<Plane>| YuvImage {
            depth: 10,
            subsampling: Subsampling::Mono,
            full_range: true,
            matrix: Matrix::Bt601,
            luma,
            cb: Plane::U8(Box::new([])),
            cr: Plane::U8(Box::new([])),
            alpha,
        };
        let real =
            mono(Plane::U16(vec![0; 32 * 16].into_boxed_slice()), None);
        let page =
            Decoded::yuv(32, 16, Format::Avif, Transfer::Srgb, real)
                .expect("mono yuv page constructs");
        // The app's upload drain charges this count against its
        // per-frame byte budget.
        assert_eq!(page.resident_bytes(), 32 * 16 * 2);
        let empty = mono(Plane::U16(Box::new([])), None);
        assert!(matches!(
            Decoded::yuv(32, 16, Format::Avif, Transfer::Srgb, empty),
            Err(DecodeRefusal::Corrupt("yuv plane extent"))
        ));
        let short_alpha = mono(
            Plane::U16(vec![0; 32 * 16].into_boxed_slice()),
            Some(Plane::U16(vec![0; 5].into_boxed_slice())),
        );
        assert!(matches!(
            Decoded::yuv(
                32,
                16,
                Format::Avif,
                Transfer::Srgb,
                short_alpha
            ),
            Err(DecodeRefusal::Corrupt("yuv plane extent"))
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
            alpha: None,
        };
        assert!(matches!(
            Decoded::yuv(4, 2, Format::Avif, Transfer::Srgb, undersized),
            Err(DecodeRefusal::Corrupt("yuv plane extent"))
        ));
    }

    #[test]
    fn display_transform_maps_samples() {
        fn check(
            samples: &[u8],
            dims: (u32, u32),
            transform: DisplayTransform,
            want_dims: (u32, u32),
            want_samples: &[u8],
        ) {
            assert_eq!(transform.out_dims(dims), want_dims);
            let plane = Plane::U8(samples.to_vec().into_boxed_slice());
            assert!(matches!(
                transform_plane(plane, dims, transform),
                Plane::U8(got) if *got == *want_samples
            ));
        }
        // A 2x1 plane [A B]: 90° CCW stands B above A.
        let quarters = |quarters: u8| DisplayTransform {
            quarters,
            mirror_axis: None,
        };
        check(b"AB", (2, 1), quarters(1), (1, 2), b"BA");
        // 180° reads the row backwards.
        check(b"AB", (2, 1), quarters(2), (2, 1), b"BA");
        // 270° CCW keeps [A B] standing: A above B.
        check(b"AB", (2, 1), quarters(3), (1, 2), b"AB");
        // 360° is identity.
        check(b"AB", (2, 1), quarters(0), (2, 1), b"AB");
        // A 2x2 mirror, both axes: axis 0 exchanges rows, axis 1 columns.
        let mirror = |axis: u8| DisplayTransform {
            quarters: 0,
            mirror_axis: Some(axis),
        };
        check(b"ABCD", (2, 2), mirror(0), (2, 2), b"CDAB");
        check(b"ABCD", (2, 2), mirror(1), (2, 2), b"BADC");
        // Mirror runs before rotate: a horizontal mirror then a 90° turn
        // transposes the grid.
        let mirrored_then_turned = DisplayTransform {
            quarters: 1,
            mirror_axis: Some(1),
        };
        check(b"ABCD", (2, 2), mirrored_then_turned, (2, 2), b"ACBD");
        assert!(matches!(
            transform_plane(Plane::U8(Box::new([])), (1, 1), quarters(1)),
            Plane::U8(got) if got.is_empty()
        ));
    }

    // 32x16 10-bit 4:0:0 fixture, a black-to-white grayscale gradient.
    #[test]
    fn avif_fixture_decodes() {
        let page =
            decode(include_bytes!("../testdata/mono.avif"), "a.avif")
                .expect("mono avif decodes");
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

    // wide.jxl and animated.jxl are verbatim copies of exif.jxl and
    // 5_frames_numbered_jxli.jxl from jxl 0.8.0's tests/testdata.
    #[test]
    fn jxl_fixtures_decode() {
        // gray: solid halves, gray replicated to r=g=b
        let gray = decode(include_bytes!("../testdata/gray.jxl"), "a.jxl")
            .expect("gray jxl decodes");
        assert_eq!((gray.width, gray.height), (8, 8));
        assert_eq!(gray.format, Format::Jxl);
        match gray.pixels {
            Pixels::Rgba8(bytes) => {
                assert_eq!(
                    bytes.get(..4),
                    Some(&[0x20, 0x20, 0x20, 0xFF][..])
                );
                assert_eq!(
                    bytes.get(128..132),
                    Some(&[0xE0, 0xE0, 0xE0, 0xFF][..])
                );
            }
            Pixels::Yuv(_) => panic!("jxl must decode to rgba"),
        }

        // alpha: the file's own alpha rides in the fourth byte
        let alpha =
            decode(include_bytes!("../testdata/alpha.jxl"), "a.jxl")
                .expect("alpha jxl decodes");
        match alpha.pixels {
            Pixels::Rgba8(bytes) => {
                assert_eq!(
                    bytes.get(..4),
                    Some(&[0x40, 0x80, 0xC0, 0xFF][..])
                );
                assert_eq!(
                    bytes.get(128..132),
                    Some(&[0x40, 0x80, 0xC0, 0x80][..])
                );
            }
            Pixels::Yuv(_) => panic!("jxl must decode to rgba"),
        }

        // wide: 256x128 pins the row/stride order. Opaque alpha on every
        // pixel proves the decoder wrote all rows of the zero-initialised
        // buffer
        let wide = decode(include_bytes!("../testdata/wide.jxl"), "a.jxl")
            .expect("wide jxl decodes");
        assert_eq!((wide.width, wide.height), (256, 128));
        let Pixels::Rgba8(wide_rgba) = &wide.pixels else {
            panic!("jxl must decode to rgba")
        };
        assert!(wide_rgba
            .as_chunks::<4>()
            .0
            .iter()
            .all(|px| px[3] == 0xFF));

        // animated: the numbered frames paint digits, so the first frame
        // carries non-black pixels
        let animated =
            decode(include_bytes!("../testdata/animated.jxl"), "a.jxl")
                .expect("animated jxl decodes");
        assert_eq!((animated.width, animated.height), (256, 256));
        let Pixels::Rgba8(animated_rgba) = &animated.pixels else {
            panic!("jxl must decode to rgba")
        };
        assert!(animated_rgba
            .as_chunks::<4>()
            .0
            .iter()
            .any(|px| px[0] != 0 || px[1] != 0 || px[2] != 0));
    }

    fn arc(bytes: Vec<u8>) -> Arc<[u8]> {
        Arc::from(bytes.into_boxed_slice())
    }

    fn first_pixel(image: &Decoded) -> [u8; 4] {
        let Pixels::Rgba8(bytes) = image.pixels() else {
            panic!("a gif frame decodes to rgba");
        };
        let mut pixel = [0; 4];
        pixel.copy_from_slice(bytes.get(..4).expect("one pixel"));
        pixel
    }

    #[test]
    fn animated_gif_streams_composited_frames() {
        let colors =
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
        let bytes = testkit::tiny_animated_gif(
            4,
            3,
            &colors,
            50,
            Repeat::Infinite,
        );
        let mut animation = open_animation(arc(bytes), "a.gif")
            .expect("animated gif decodes")
            .expect("gif has a frame stream");
        assert_eq!(animation.loops(), Loops::Infinite);
        for color in colors {
            let frame = animation
                .next_frame()
                .expect("frame present")
                .expect("frame decodes");
            assert_eq!(
                (frame.image.width(), frame.image.height()),
                (4, 3),
                "frames use the logical screen"
            );
            assert_eq!(first_pixel(&frame.image), color);
            assert_eq!(frame.delay, Duration::from_millis(50));
        }
        assert!(animation.next_frame().is_none(), "pass ended");
    }

    #[test]
    fn gif_loop_metadata_keeps_absent_distinct_from_infinite() {
        // No NETSCAPE extension: the decoder's zero default means one
        // play, not forever.
        let still = testkit::tiny_gif(2, 2, [10, 20, 30]);
        let mut once = open_animation(arc(still), "a.gif")
            .expect("still gif decodes")
            .expect("gif has a frame stream");
        assert_eq!(once.loops(), Loops::Finite(1));
        assert!(once.next_frame().is_some());
        assert!(once.next_frame().is_none());

        // The field counts repetitions after the first, so 2 means 3
        // total passes.
        let finite = testkit::tiny_animated_gif(
            2,
            2,
            &[[1, 2, 3, 255], [4, 5, 6, 255]],
            100,
            Repeat::Finite(2),
        );
        let animation = open_animation(arc(finite), "a.gif")
            .expect("finite gif decodes")
            .expect("gif has a frame stream");
        assert_eq!(animation.loops(), Loops::Finite(3));
    }

    #[test]
    fn animation_rewinds() {
        let colors =
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
        let bytes = testkit::tiny_animated_gif(
            2,
            2,
            &colors,
            20,
            Repeat::Infinite,
        );
        let mut animation = open_animation(arc(bytes), "a.gif")
            .expect("gif decodes")
            .expect("gif has a frame stream");
        for color in colors {
            let frame = animation
                .next_frame()
                .expect("frame present")
                .expect("valid");
            assert_eq!(first_pixel(&frame.image), color);
        }
        assert!(animation.next_frame().is_none(), "pass ended");
        animation.rewind().expect("rewind");
        let rewound = animation
            .next_frame()
            .expect("frame 0 again")
            .expect("valid");
        assert_eq!(first_pixel(&rewound.image), colors[0]);
    }

    // A truncation bug would hang the decode instead of failing it, hence
    // the watchdog thread. The cuts cover the header, the frame header, and
    // the pixels. A decode panic closes the channel and fails the recv, so
    // plain asserts on the main thread suffice.
    #[test]
    fn truncated_jxl_ends_the_decode() {
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let fixture = include_bytes!("../testdata/gray.jxl");
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for cut in [1_usize, 2, 8, 35] {
                let truncated =
                    fixture.get(..cut).expect("cut inside the fixture");
                let result = decode(truncated, "a.jxl");
                if tx.send((cut, result)).is_err() {
                    return;
                }
            }
        });
        for _ in 0..4 {
            let (cut, result) = rx
                .recv_timeout(Duration::from_secs(5))
                .expect("truncated jxl decode finishes");
            assert!(
                matches!(refusal(result), Some(DecodeRefusal::Corrupt(_))),
                "cut {cut} did not fail as corruption"
            );
        }
    }
}
