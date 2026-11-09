//! Decoding: raw page bytes -> `Decoded` (RGBA or YUV planes).
//!
//! Dispatch is by magic bytes, then extension; AVIF decodes to YUV
//! planes and the rest to RGBA. Corrupt input fails as `DecodeError`,
//! but the decoder libraries can still panic on hostile bytes; the
//! worker's `catch_unwind` catches that.

#![expect(
    clippy::definition_in_module_root,
    reason = "the decode tree's shared vocabulary lives in its module root by design"
)]

use eros::ErrorUnion;

use crate::fault::DecodeError;
use crate::fault::DecodeRefusal;

mod avif;
mod image;
mod jxl;

/// Construct via [`Decoded::rgba`] / [`Decoded::yuv`], which validate the
/// pixel extents against the dimensions.
#[derive(Clone)]
pub struct Decoded {
    /// Pixel dimensions of this representation. A preview is smaller
    /// than the page; the page's own dimensions live in [`Self::source`].
    width: u32,
    height: u32,
    pixels: Pixels,
    /// Sample encoding the renderer must undo before the framebuffer.
    transfer: Transfer,
    format: Format,
    /// The page's oriented dimensions, which the layout pairs and fits
    /// against. Equal to `width`/`height` for a native decode, and
    /// unchanged by preview minting.
    source: (u32, u32),
}

/// Sample encoding. SDR pages are sRGB-class gamma, passed through to the
/// sRGB framebuffer; PQ pages are absolute-luminance HDR, tone-mapped to
/// SDR in the shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    /// The page's oriented dimensions, independent of how far this
    /// representation is downsampled.
    pub fn source(&self) -> (u32, u32) {
        self.source
    }

    /// Record the page dimensions this representation was minted from.
    pub fn with_source(mut self, source: (u32, u32)) -> Self {
        self.source = source;
        self
    }

    pub fn rgba(
        width: u32,
        height: u32,
        format: Format,
        transfer: Transfer,
        bytes: Box<[u8]>,
    ) -> Result<Self, DecodeRefusal> {
        check_dims(width, height)?;
        let want = checked_extent(width, height, 4)?;
        if bytes.len() != want {
            return Err(DecodeRefusal::Corrupt("rgba extent"));
        }
        Ok(Self {
            width,
            height,
            pixels: Pixels::Rgba8(bytes),
            transfer,
            format,
            source: (width, height),
        })
    }

    pub fn yuv(
        width: u32,
        height: u32,
        format: Format,
        transfer: Transfer,
        yuv: YuvImage,
    ) -> Result<Self, DecodeRefusal> {
        check_dims(width, height)?;
        let (_, (chroma_width, chroma_height)) =
            plane_dims((width, height), yuv.subsampling);
        if yuv.luma.sample_count() != width as usize * height as usize
            || yuv.alpha.as_ref().is_some_and(|alpha| {
                alpha.sample_count() != width as usize * height as usize
            })
            || (yuv.subsampling != Subsampling::Mono
                && (yuv.cb.sample_count()
                    != chroma_width as usize * chroma_height as usize
                    || yuv.cr.sample_count()
                        != chroma_width as usize * chroma_height as usize))
        {
            return Err(DecodeRefusal::Corrupt("yuv plane extent"));
        }
        Ok(Self {
            width,
            height,
            pixels: Pixels::Yuv(yuv),
            transfer,
            format,
            source: (width, height),
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

fn checked_pixels(
    width: u32,
    height: u32,
    what: &'static str,
) -> Result<u32, DecodeRefusal> {
    width
        .checked_mul(height)
        .filter(|&count| (1..=MAX_PIXELS).contains(&count))
        .ok_or(DecodeRefusal::Corrupt(what))
}

fn checked_extent(
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
) -> Result<usize, DecodeRefusal> {
    Ok(checked_pixels(width, height, "bad dimensions")? as usize
        * bytes_per_pixel)
}

/// The image crate's 512 MiB allocation ceiling, as RGBA pixels.
#[expect(
    clippy::integer_division,
    reason = "the 512 MiB ceiling in 4-byte pixels divides exactly"
)]
const MAX_PIXELS: u32 = (512 << 20) / 4;

/// Decode-time admission cap on a page edge: pages over it fail before
/// any buffer is funded for them. 0x4000 is the texture edge common
/// desktop GPUs support; the authoritative per-GPU check is the
/// renderer's runtime query, reported as `TextureFault`.
const DECODE_EDGE_LIMIT: u32 = 0x4000;

fn check_dims(width: u32, height: u32) -> Result<(), DecodeRefusal> {
    if width > DECODE_EDGE_LIMIT || height > DECODE_EDGE_LIMIT {
        return Err(DecodeRefusal::Unsupported(
            "page edge over the GPU texture limit",
        ));
    }
    Ok(())
}

#[derive(Clone)]
pub enum Pixels {
    /// Interleaved RGBA, `4 * width * height` bytes, row-major, top-first.
    Rgba8(Box<[u8]>),
    /// Planar YUV, converted to RGB in the renderer's shader.
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

/// 8-bit planes are one byte per sample; deeper planes are `u16` samples
/// with `depth` significant bits. Mono carries luma only, so its chroma
/// planes stay empty. Alpha, when the file carries one, is a luma-sized
/// full-range plane.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    Mono,
    Yuv420,
    Yuv422,
    Yuv444,
}

impl Subsampling {
    /// Chroma texels per luma texel on each axis: the sampling ratio
    /// behind [`plane_dims`]'s texel counts. Mono carries no chroma, so
    /// its ratio is identity for the renderer's midpoint stub.
    pub fn chroma_ratio(self) -> (f32, f32) {
        match self {
            Self::Mono | Self::Yuv444 => (1.0, 1.0),
            Self::Yuv420 => (0.5, 0.5),
            Self::Yuv422 => (0.5, 1.0),
        }
    }
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

    pub fn is_upright(self) -> bool {
        self == Self::UPRIGHT
    }

    pub fn swaps_dims(self) -> bool {
        self.quarters % 2 == 1
    }
}

/// Apply `transform` to one plane, returning the plane and its new dims.
/// Empty planes (mono's chroma stubs) pass through untouched.
fn transform_plane(
    plane: Plane,
    dims: (u32, u32),
    transform: DisplayTransform,
) -> (Plane, (u32, u32)) {
    if transform.is_upright() {
        return (plane, dims);
    }
    match plane {
        Plane::U8(bytes) => {
            let (samples, dims) =
                transform_samples(bytes, dims, transform);
            (Plane::U8(samples), dims)
        }
        Plane::U16(words) => {
            let (samples, dims) =
                transform_samples(words, dims, transform);
            (Plane::U16(samples), dims)
        }
    }
}

/// Mirror-then-rotate resampling by sample index. Mirroring runs first,
/// matching the property order HEIF writers use.
fn transform_samples<T>(
    samples: Box<[T]>,
    (width, height): (u32, u32),
    transform: DisplayTransform,
) -> (Box<[T]>, (u32, u32))
where
    T: Copy,
{
    if samples.is_empty() {
        return (samples, (width, height));
    }
    let (out_width, out_height) = if transform.swaps_dims() {
        (height, width)
    } else {
        (width, height)
    };
    let (width, height) = (width as usize, height as usize);
    let source_index = |x: usize, y: usize| {
        let (mut sx, mut sy) = match transform.quarters {
            1 => (width - 1 - y, x),
            2 => (width - 1 - x, height - 1 - y),
            3 => (y, height - 1 - x),
            _ => (x, y),
        };
        match transform.mirror_axis {
            Some(1) => sx = width - 1 - sx,
            Some(_) => sy = height - 1 - sy,
            None => {}
        }
        sy * width + sx
    };
    let mut out = Vec::with_capacity(samples.len());
    for y in 0..out_height as usize {
        for x in 0..out_width as usize {
            let index = source_index(x, y);
            #[expect(
                clippy::indexing_slicing,
                reason = "out dims keep the sample count, so the mapped index stays in range"
            )]
            out.push(samples[index]);
        }
    }
    (out.into_boxed_slice(), (out_width, out_height))
}

/// Decode one page. `name` is the page filename, used as the extension
/// fallback. Animated images show their first frame.
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
        Decoded::rgba(
            DECODE_EDGE_LIMIT,
            1,
            Format::Png,
            Transfer::Srgb,
            vec![0_u8; 4 * DECODE_EDGE_LIMIT as usize].into_boxed_slice(),
        )
        .expect("page at the edge cap constructs");
        let over = Decoded::rgba(
            DECODE_EDGE_LIMIT + 1,
            1,
            Format::Png,
            Transfer::Srgb,
            vec![0_u8; 4 * (DECODE_EDGE_LIMIT + 1) as usize]
                .into_boxed_slice(),
        );
        assert!(matches!(over, Err(DecodeRefusal::Unsupported(_))));
        // Dims inside the edge cap can still exceed the pixel-count cap,
        // which must reject the page before any buffer is sized for it.
        assert!(matches!(
            Decoded::rgba(
                DECODE_EDGE_LIMIT,
                DECODE_EDGE_LIMIT,
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
        // The worker's book budget consumes this count.
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
            let plane = Plane::U8(samples.to_vec().into_boxed_slice());
            let (plane, dims) = transform_plane(plane, dims, transform);
            assert_eq!(
                (dims, plane_bytes(&plane)),
                (want_dims, want_samples)
            );
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
        // Empty planes (mono chroma stubs) pass through.
        let (empty, dims) =
            transform_plane(Plane::U8(Box::new([])), (0, 0), quarters(1));
        assert!(matches!(empty, Plane::U8(bytes) if bytes.is_empty()));
        assert_eq!(dims, (0, 0));
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
