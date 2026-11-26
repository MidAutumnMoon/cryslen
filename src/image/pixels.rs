//! Core image types, validation bounds, and EXIF orientation transforms.

use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use eros::MsgError;
use image::metadata::Orientation;
use strum::Display;
use strum::FromRepr;

pub const RGBA_BYTES_PER_PIXEL: usize = 4;
pub const THUMB_EDGE: u32 = 192;
pub const MAX_THUMB_BYTES: usize = 256 * 1024;

/// The image crate's 512 MiB allocation ceiling, as RGBA pixels.
pub const MAX_PIXELS: u32 =
    (512 * 1024 * 1024) >> RGBA_BYTES_PER_PIXEL.ilog2();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, FromRepr)]
#[repr(u8)]
#[strum(serialize_all = "lowercase")]
pub enum Format {
    Jpeg = 0,
    Png = 1,
    Gif = 2,
    Webp = 3,
    Bmp = 4,
    Tiff = 5,
    Avif = 6,
    Jxl = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    D8,
    D10,
    D12,
    D16,
}

impl Depth {
    #[must_use]
    pub const fn is_wide(self) -> bool {
        !matches!(self, Self::D8)
    }

    #[must_use]
    pub const fn bits(self) -> u8 {
        match self {
            Self::D8 => 8,
            Self::D10 => 10,
            Self::D12 => 12,
            Self::D16 => 16,
        }
    }

    #[must_use]
    pub const fn sample_bytes(self) -> usize {
        if self.is_wide() {
            size_of::<u16>()
        } else {
            size_of::<u8>()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WideDepth {
    D10,
    D12,
    D16,
}

impl WideDepth {
    #[must_use]
    pub const fn depth(self) -> Depth {
        match self {
            Self::D10 => Depth::D10,
            Self::D12 => Depth::D12,
            Self::D16 => Depth::D16,
        }
    }
}

/// Sample encoding. SDR pages are sRGB-class gamma; PQ pages are
/// absolute-luminance HDR, tone-mapped to SDR in the shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transfer {
    Srgb,
    Pq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    Yuv420,
    Yuv422,
    Yuv444,
}

impl Subsampling {
    /// Chroma texels per luma texel on each axis.
    #[must_use]
    pub fn chroma_ratio(self) -> (f64, f64) {
        match self {
            Self::Yuv444 => (1.0, 1.0),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YuvSpec {
    pub subsampling: Subsampling,
    /// Limited (video) or full (JPEG) range quantization.
    pub full_range: bool,
    pub matrix: Matrix,
}

/// Native page dimensions, decoded format, and animation flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageInfo {
    pub dims: (u32, u32),
    pub format: Format,
    pub animated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageMeta {
    /// Decoded dimensions of this representation; may be downsampled from `page_dims`.
    pub width: u32,
    pub height: u32,
    /// Original unscaled page dimensions.
    pub page_dims: (u32, u32),
    pub format: Format,
    /// Sample encoding the renderer must undo before the framebuffer.
    pub transfer: Transfer,
    /// True only when the page has more than one frame.
    pub animated: bool,
}

impl ImageMeta {
    const fn still(
        width: u32,
        height: u32,
        format: Format,
        transfer: Transfer,
    ) -> Self {
        Self {
            width,
            height,
            page_dims: (width, height),
            format,
            transfer,
            animated: false,
        }
    }

    #[must_use]
    pub const fn page_info(self) -> PageInfo {
        PageInfo {
            dims: self.page_dims,
            format: self.format,
            animated: self.animated,
        }
    }
}

/// Resolution tier produced by the decode pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetRole {
    Fit,
    Full,
}

/// Session-wide target box and GPU texture ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageLimits {
    fit_box: (u32, u32),
    max_texture_edge: u32,
}

impl ImageLimits {
    /// Clamps `max_texture_edge >= THUMB_EDGE` and each `fit_box` axis
    /// into `[THUMB_EDGE, max_texture_edge]`.
    pub fn new(fit_box: (u32, u32), max_texture_edge: u32) -> Self {
        let max_texture_edge = max_texture_edge.max(THUMB_EDGE);
        Self {
            fit_box: (
                fit_box.0.clamp(THUMB_EDGE, max_texture_edge),
                fit_box.1.clamp(THUMB_EDGE, max_texture_edge),
            ),
            max_texture_edge,
        }
    }

    #[must_use]
    pub const fn fit_box(self) -> (u32, u32) {
        self.fit_box
    }

    #[must_use]
    pub const fn max_texture_edge(self) -> u32 {
        self.max_texture_edge
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureFault {
    pub dims: (u32, u32),
    pub max_edge: u32,
}

impl TextureFault {
    pub fn check_dims(
        dims: (u32, u32),
        max_edge: u32,
    ) -> Result<(), Self> {
        if dims.0 == 0
            || dims.1 == 0
            || dims.0 > max_edge
            || dims.1 > max_edge
        {
            return Err(Self { dims, max_edge });
        }
        Ok(())
    }
}

impl Display for TextureFault {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}x{} exceeds this GPU's {}-texel texture edge limit",
            self.dims.0, self.dims.1, self.max_edge
        )
    }
}

impl Error for TextureFault {}

/// Homogeneous sample planes keeping native `u8`/`u16` alignment and a
/// bit depth consistent across all color and alpha planes by type.
#[derive(Clone, PartialEq, Eq)]
pub enum SamplePlanes<const PLANES: usize> {
    U8 {
        planes: [Box<[u8]>; PLANES],
        alpha: Option<Box<[u8]>>,
    },
    U16 {
        depth: WideDepth,
        planes: [Box<[u16]>; PLANES],
        alpha: Option<Box<[u16]>>,
    },
}

impl<const PLANES: usize> SamplePlanes<PLANES> {
    #[must_use]
    pub const fn depth(&self) -> Depth {
        match self {
            Self::U8 { .. } => Depth::D8,
            Self::U16 { depth, .. } => depth.depth(),
        }
    }

    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::U8 { planes, alpha } => {
                planes_byte_len(planes, alpha.as_deref())
            }
            Self::U16 { planes, alpha, .. } => {
                planes_byte_len(planes, alpha.as_deref())
            }
        }
    }

    fn transform_uniform(
        &self,
        dims: (u32, u32),
        orientation: Orientation,
    ) -> Self {
        match self {
            Self::U8 { planes, alpha } => Self::U8 {
                planes: planes.each_ref().map(|plane| {
                    transform_samples(plane, dims, orientation)
                }),
                alpha: alpha.as_ref().map(|plane| {
                    transform_samples(plane, dims, orientation)
                }),
            },
            Self::U16 {
                depth,
                planes,
                alpha,
            } => Self::U16 {
                depth: *depth,
                planes: planes.each_ref().map(|plane| {
                    transform_samples(plane, dims, orientation)
                }),
                alpha: alpha.as_ref().map(|plane| {
                    transform_samples(plane, dims, orientation)
                }),
            },
        }
    }
}

impl SamplePlanes<3> {
    fn transform_yuv(
        &self,
        luma_dims: (u32, u32),
        chroma_dims: (u32, u32),
        orientation: Orientation,
    ) -> Self {
        match self {
            Self::U8 {
                planes: [luma, cb, cr],
                alpha,
            } => Self::U8 {
                planes: [
                    transform_samples(luma, luma_dims, orientation),
                    transform_samples(cb, chroma_dims, orientation),
                    transform_samples(cr, chroma_dims, orientation),
                ],
                alpha: alpha.as_ref().map(|plane| {
                    transform_samples(plane, luma_dims, orientation)
                }),
            },
            Self::U16 {
                depth,
                planes: [luma, cb, cr],
                alpha,
            } => Self::U16 {
                depth: *depth,
                planes: [
                    transform_samples(luma, luma_dims, orientation),
                    transform_samples(cb, chroma_dims, orientation),
                    transform_samples(cr, chroma_dims, orientation),
                ],
                alpha: alpha.as_ref().map(|plane| {
                    transform_samples(plane, luma_dims, orientation)
                }),
            },
        }
    }
}

fn planes_byte_len<Sample, const PLANES: usize>(
    planes: &[Box<[Sample]>; PLANES],
    alpha: Option<&[Sample]>,
) -> usize {
    planes
        .iter()
        .map(|plane| size_of_val(plane.as_ref()))
        .sum::<usize>()
        + alpha.map_or(0, size_of_val)
}

#[derive(Clone, PartialEq, Eq)]
pub enum Pixels {
    /// Interleaved RGBA, `4 * width * height` bytes, row-major, top-first.
    Rgba8(Box<[u8]>),
    /// Planar grayscale, converted to RGB in the renderer's shader.
    Gray {
        full_range: bool,
        planes: SamplePlanes<1>,
    },
    /// Planar YUV, converted to RGB in the renderer's shader.
    Yuv {
        spec: YuvSpec,
        planes: SamplePlanes<3>,
    },
}

impl Pixels {
    #[must_use]
    pub const fn depth(&self) -> Depth {
        match self {
            Self::Rgba8(_) => Depth::D8,
            Self::Gray { planes, .. } => planes.depth(),
            Self::Yuv { planes, .. } => planes.depth(),
        }
    }
}

/// Decoded image whose pixel extents and sample widths passed validation.
#[derive(Clone, PartialEq, Eq)]
pub struct Image {
    meta: ImageMeta,
    pixels: Pixels,
}

impl Image {
    #[must_use]
    pub const fn page_info(&self) -> PageInfo {
        self.meta.page_info()
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.meta.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.meta.height
    }

    #[must_use]
    pub const fn pixels(&self) -> &Pixels {
        &self.pixels
    }

    #[must_use]
    pub const fn transfer(&self) -> Transfer {
        self.meta.transfer
    }

    #[must_use]
    pub const fn format(&self) -> Format {
        self.meta.format
    }

    #[must_use]
    pub const fn page_dims(&self) -> (u32, u32) {
        self.meta.page_dims
    }

    #[must_use]
    pub const fn animated(&self) -> bool {
        self.meta.animated
    }

    #[must_use]
    pub const fn with_page_dims(mut self, page_dims: (u32, u32)) -> Self {
        self.meta.page_dims = page_dims;
        self
    }

    #[must_use]
    pub const fn with_animated(mut self, animated: bool) -> Self {
        self.meta.animated = animated;
        self
    }

    pub fn rgba(
        width: u32,
        height: u32,
        format: Format,
        transfer: Transfer,
        bytes: Box<[u8]>,
    ) -> Result<Self, MsgError> {
        validate_rgba_extent(width, height, bytes.len())?;
        Ok(Self {
            meta: ImageMeta::still(width, height, format, transfer),
            pixels: Pixels::Rgba8(bytes),
        })
    }

    pub fn gray(
        width: u32,
        height: u32,
        format: Format,
        transfer: Transfer,
        full_range: bool,
        planes: SamplePlanes<1>,
    ) -> Result<Self, MsgError> {
        let (luma_bytes, alpha_bytes) = match &planes {
            SamplePlanes::U8 {
                planes: [luma],
                alpha,
            } => (
                size_of_val(luma.as_ref()),
                alpha.as_deref().map(size_of_val),
            ),
            SamplePlanes::U16 {
                planes: [luma],
                alpha,
                ..
            } => (
                size_of_val(luma.as_ref()),
                alpha.as_deref().map(size_of_val),
            ),
        };
        validate_gray_extents(
            (width, height),
            planes.depth(),
            luma_bytes,
            alpha_bytes,
        )?;
        Ok(Self {
            meta: ImageMeta::still(width, height, format, transfer),
            pixels: Pixels::Gray { full_range, planes },
        })
    }

    pub fn yuv(
        width: u32,
        height: u32,
        format: Format,
        transfer: Transfer,
        spec: YuvSpec,
        planes: SamplePlanes<3>,
    ) -> Result<Self, MsgError> {
        let (plane_bytes, alpha_bytes) = match &planes {
            SamplePlanes::U8 {
                planes: [luma, cb, cr],
                alpha,
            } => (
                [
                    size_of_val(luma.as_ref()),
                    size_of_val(cb.as_ref()),
                    size_of_val(cr.as_ref()),
                ],
                alpha.as_deref().map(size_of_val),
            ),
            SamplePlanes::U16 {
                planes: [luma, cb, cr],
                alpha,
                ..
            } => (
                [
                    size_of_val(luma.as_ref()),
                    size_of_val(cb.as_ref()),
                    size_of_val(cr.as_ref()),
                ],
                alpha.as_deref().map(size_of_val),
            ),
        };
        validate_yuv_extents(
            (width, height),
            planes.depth(),
            spec.subsampling,
            plane_bytes,
            alpha_bytes,
        )?;
        Ok(Self {
            meta: ImageMeta::still(width, height, format, transfer),
            pixels: Pixels::Yuv { spec, planes },
        })
    }

    pub fn gray_u8(
        width: u32,
        height: u32,
        format: Format,
        transfer: Transfer,
        luma: Box<[u8]>,
    ) -> Result<Self, MsgError> {
        Self::gray(
            width,
            height,
            format,
            transfer,
            true,
            SamplePlanes::U8 {
                planes: [luma],
                alpha: None,
            },
        )
    }

    pub fn gray_u16(
        width: u32,
        height: u32,
        format: Format,
        transfer: Transfer,
        depth: WideDepth,
        luma: Box<[u16]>,
    ) -> Result<Self, MsgError> {
        Self::gray(
            width,
            height,
            format,
            transfer,
            true,
            SamplePlanes::U16 {
                depth,
                planes: [luma],
                alpha: None,
            },
        )
    }

    #[must_use]
    pub fn apply_orientation(mut self, orientation: Orientation) -> Self {
        if orientation == Orientation::NoTransforms {
            return self;
        }
        let dims = (self.meta.width, self.meta.height);
        debug_assert!(
            !(orientation_swaps_dims(orientation)
                && matches!(
                    &self.pixels,
                    Pixels::Yuv { spec, .. }
                        if spec.subsampling == Subsampling::Yuv422
                )),
            "rotated 4:2:2 must be rejected before orientation"
        );
        self.pixels = match &self.pixels {
            Pixels::Rgba8(bytes) => {
                Pixels::Rgba8(transform_rgba(bytes, dims, orientation))
            }
            Pixels::Gray { full_range, planes } => Pixels::Gray {
                full_range: *full_range,
                planes: planes.transform_uniform(dims, orientation),
            },
            Pixels::Yuv { spec, planes } => {
                let (luma_dims, chroma_dims) =
                    plane_dims(dims, spec.subsampling);
                Pixels::Yuv {
                    spec: *spec,
                    planes: planes.transform_yuv(
                        luma_dims,
                        chroma_dims,
                        orientation,
                    ),
                }
            }
        };
        let target_dims = oriented_dims(orientation, dims);
        self.meta.width = target_dims.0;
        self.meta.height = target_dims.1;
        self.meta.page_dims = target_dims;
        self
    }

    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        match &self.pixels {
            Pixels::Rgba8(bytes) => bytes.len(),
            Pixels::Gray { planes, .. } => planes.resident_bytes(),
            Pixels::Yuv { planes, .. } => planes.resident_bytes(),
        }
    }
}

/// Decoded 8-bit sRGB thumbnail ready for GPU upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThumbImage {
    pub dims: (u32, u32),
    pub info: PageInfo,
    pub rgba: Vec<u8>,
}

/// Compact WebP thumbnail stored in the shared thumbnail database.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedThumb {
    pub page_dims: (u32, u32),
    pub format: Format,
    pub webp: Vec<u8>,
}

impl EncodedThumb {
    #[must_use]
    pub fn is_valid_size(&self) -> bool {
        !self.webp.is_empty() && self.webp.len() <= MAX_THUMB_BYTES
    }
}

/// Pixel count of `(width, height)` when non-empty and within [`MAX_PIXELS`].
pub fn pixel_count(width: u32, height: u32) -> Option<u32> {
    width
        .checked_mul(height)
        .filter(|&count| (1..=MAX_PIXELS).contains(&count))
}

pub fn admit_dims(
    width: u32,
    height: u32,
    bad_dims_reason: &'static str,
) -> Result<u32, MsgError> {
    pixel_count(width, height)
        .ok_or_else(|| MsgError::from_static(bad_dims_reason))
}

struct YuvLayout {
    luma_samples: usize,
    chroma_samples: usize,
}

pub fn rgba_byte_len(width: u32, height: u32) -> Result<usize, MsgError> {
    let pixels = admit_dims(width, height, "bad dimensions")?;
    (pixels as usize)
        .checked_mul(RGBA_BYTES_PER_PIXEL)
        .ok_or_else(|| MsgError::from_static("bad dimensions"))
}

fn validate_rgba_extent(
    width: u32,
    height: u32,
    bytes: usize,
) -> Result<(), MsgError> {
    if bytes != rgba_byte_len(width, height)? {
        return Err(MsgError::from_static("rgba extent"));
    }
    Ok(())
}

fn validate_gray_extents(
    dims: (u32, u32),
    depth: Depth,
    luma_bytes: usize,
    alpha_bytes: Option<usize>,
) -> Result<(), MsgError> {
    let luma_samples =
        admit_dims(dims.0, dims.1, "bad dimensions")? as usize;
    let expected_bytes = luma_samples
        .checked_mul(depth.sample_bytes())
        .ok_or_else(|| MsgError::from_static("gray plane extent"))?;
    if luma_bytes != expected_bytes
        || alpha_bytes.is_some_and(|len| len != expected_bytes)
    {
        return Err(MsgError::from_static("gray plane extent"));
    }
    Ok(())
}

fn yuv_layout(
    dims: (u32, u32),
    subsampling: Subsampling,
) -> Result<YuvLayout, MsgError> {
    let luma_samples =
        admit_dims(dims.0, dims.1, "bad dimensions")? as usize;
    let (_, (chroma_width, chroma_height)) = plane_dims(dims, subsampling);
    let chroma_samples =
        admit_dims(chroma_width, chroma_height, "bad dimensions")?
            as usize;
    Ok(YuvLayout {
        luma_samples,
        chroma_samples,
    })
}

fn validate_yuv_extents(
    dims: (u32, u32),
    depth: Depth,
    subsampling: Subsampling,
    [luma_bytes, chroma_blue_bytes, chroma_red_bytes]: [usize; 3],
    alpha_bytes: Option<usize>,
) -> Result<(), MsgError> {
    let layout = yuv_layout(dims, subsampling)?;
    let sample_bytes = depth.sample_bytes();
    let expected_luma_bytes = layout
        .luma_samples
        .checked_mul(sample_bytes)
        .ok_or_else(|| MsgError::from_static("yuv plane extent"))?;
    let expected_chroma_bytes = layout
        .chroma_samples
        .checked_mul(sample_bytes)
        .ok_or_else(|| MsgError::from_static("yuv plane extent"))?;
    if luma_bytes != expected_luma_bytes
        || alpha_bytes.is_some_and(|len| len != expected_luma_bytes)
        || chroma_blue_bytes != expected_chroma_bytes
        || chroma_red_bytes != expected_chroma_bytes
    {
        return Err(MsgError::from_static("yuv plane extent"));
    }
    Ok(())
}

pub fn plane_dims(
    size: (u32, u32),
    subsampling: Subsampling,
) -> ((u32, u32), (u32, u32)) {
    let (width, height) = size;
    match subsampling {
        Subsampling::Yuv420 => {
            ((width, height), (width.div_ceil(2), height.div_ceil(2)))
        }
        Subsampling::Yuv422 => {
            ((width, height), (width.div_ceil(2), height))
        }
        Subsampling::Yuv444 => ((width, height), (width, height)),
    }
}

pub fn orientation_swaps_dims(orientation: Orientation) -> bool {
    matches!(
        orientation,
        Orientation::Rotate90
            | Orientation::Rotate270
            | Orientation::Rotate90FlipH
            | Orientation::Rotate270FlipH
    )
}

#[must_use]
fn oriented_dims(
    orientation: Orientation,
    (width, height): (u32, u32),
) -> (u32, u32) {
    if orientation_swaps_dims(orientation) {
        (height, width)
    } else {
        (width, height)
    }
}

fn transform_rgba(
    bytes: &[u8],
    dims: (u32, u32),
    orientation: Orientation,
) -> Box<[u8]> {
    if bytes.is_empty() {
        return Box::new([]);
    }
    let (pixels, tail) = bytes.as_chunks::<RGBA_BYTES_PER_PIXEL>();
    debug_assert!(tail.is_empty(), "rgba buffer must be 4-byte aligned");
    transform_samples(pixels, dims, orientation)
        .into_vec()
        .into_flattened()
        .into_boxed_slice()
}

fn transform_samples<Sample>(
    samples: &[Sample],
    (width, height): (u32, u32),
    orientation: Orientation,
) -> Box<[Sample]>
where
    Sample: Copy,
{
    let (target_width, target_height) =
        oriented_dims(orientation, (width, height));
    let (width, height) = (width as usize, height as usize);
    debug_assert_eq!(
        samples.len(),
        width * height,
        "plane sample count must match dimensions"
    );
    let mut target_samples = Vec::with_capacity(samples.len());
    for target_y in 0..target_height as usize {
        for target_x in 0..target_width as usize {
            let (source_x, source_y) = match orientation {
                Orientation::NoTransforms => (target_x, target_y),
                Orientation::Rotate90 => (target_y, height - 1 - target_x),
                Orientation::Rotate180 => {
                    (width - 1 - target_x, height - 1 - target_y)
                }
                Orientation::Rotate270 => (width - 1 - target_y, target_x),
                Orientation::FlipHorizontal => {
                    (width - 1 - target_x, target_y)
                }
                Orientation::FlipVertical => {
                    (target_x, height - 1 - target_y)
                }
                Orientation::Rotate90FlipH => (target_y, target_x),
                Orientation::Rotate270FlipH => {
                    (width - 1 - target_y, height - 1 - target_x)
                }
            };
            #[expect(
                clippy::indexing_slicing,
                reason = "source coordinates stay within (width, height)"
            )]
            target_samples.push(samples[source_y * width + source_x]);
        }
    }
    target_samples.into_boxed_slice()
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;

    #[test]
    fn constructor_caps() {
        const WIDE_EDGE: u32 = 32_768;

        Image::rgba(
            WIDE_EDGE,
            1,
            Format::Png,
            Transfer::Srgb,
            vec![0_u8; 4 * WIDE_EDGE as usize].into_boxed_slice(),
        )
        .expect("tall or wide strip under MAX_PIXELS constructs");
        assert_eq!(
            Image::rgba(
                0,
                10,
                Format::Png,
                Transfer::Srgb,
                Box::default(),
            )
            .err(),
            Some(MsgError::from_static("bad dimensions"))
        );
        // Over the pixel-count cap must reject the page before any buffer
        // is sized for it.
        assert_eq!(
            Image::rgba(
                WIDE_EDGE,
                WIDE_EDGE,
                Format::Png,
                Transfer::Srgb,
                Box::default()
            )
            .err(),
            Some(MsgError::from_static("bad dimensions"))
        );
    }

    #[test]
    fn gray_and_yuv_planes() {
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

        let page = Image::gray(
            32,
            16,
            Format::Avif,
            Transfer::Srgb,
            true,
            SamplePlanes::U16 {
                depth: WideDepth::D10,
                planes: [vec![0; 32 * 16].into_boxed_slice()],
                alpha: None,
            },
        )
        .expect("gray page constructs");
        assert_eq!(page.resident_bytes(), 32 * 16 * 2);

        assert_eq!(
            Image::gray(
                32,
                16,
                Format::Avif,
                Transfer::Srgb,
                true,
                SamplePlanes::U16 {
                    depth: WideDepth::D10,
                    planes: [Box::new([])],
                    alpha: None,
                },
            )
            .err(),
            Some(MsgError::from_static("gray plane extent"))
        );
        assert_eq!(
            Image::gray(
                32,
                16,
                Format::Avif,
                Transfer::Srgb,
                true,
                SamplePlanes::U16 {
                    depth: WideDepth::D10,
                    planes: [vec![0; 32 * 16].into_boxed_slice()],
                    alpha: Some(vec![0; 5].into_boxed_slice()),
                },
            )
            .err(),
            Some(MsgError::from_static("gray plane extent"))
        );
        assert_eq!(
            Image::yuv(
                4,
                2,
                Format::Avif,
                Transfer::Srgb,
                YuvSpec {
                    subsampling: Subsampling::Yuv420,
                    full_range: true,
                    matrix: Matrix::Bt601,
                },
                SamplePlanes::U8 {
                    planes: [
                        vec![0; 4 * 2].into_boxed_slice(),
                        Box::new([]),
                        Box::new([]),
                    ],
                    alpha: None,
                },
            )
            .err(),
            Some(MsgError::from_static("yuv plane extent"))
        );
    }

    #[test]
    fn orientation_maps_samples() {
        fn check(
            samples: &[u8],
            dims: (u32, u32),
            orientation: Orientation,
            want_dims: (u32, u32),
            want_samples: &[u8],
        ) {
            let page = Image::gray_u8(
                dims.0,
                dims.1,
                Format::Avif,
                Transfer::Srgb,
                samples.to_vec().into_boxed_slice(),
            )
            .expect("gray page constructs")
            .apply_orientation(orientation);
            assert_eq!((page.width(), page.height()), want_dims);
            assert_eq!(page.page_dims(), want_dims);
            assert!(matches!(
                page.pixels(),
                Pixels::Gray {
                    planes: SamplePlanes::U8 { planes: [luma], .. },
                    ..
                } if &**luma == want_samples
            ));
        }
        check(b"AB", (2, 1), Orientation::NoTransforms, (2, 1), b"AB");
        check(b"AB", (2, 1), Orientation::Rotate90, (1, 2), b"AB");
        check(b"AB", (2, 1), Orientation::Rotate180, (2, 1), b"BA");
        check(b"AB", (2, 1), Orientation::Rotate270, (1, 2), b"BA");
        check(
            b"ABCD",
            (2, 2),
            Orientation::FlipHorizontal,
            (2, 2),
            b"BADC",
        );
        check(b"ABCD", (2, 2), Orientation::FlipVertical, (2, 2), b"CDAB");
        check(
            b"ABCD",
            (2, 2),
            Orientation::Rotate90FlipH,
            (2, 2),
            b"ACBD",
        );
        check(
            b"ABCD",
            (2, 2),
            Orientation::Rotate270FlipH,
            (2, 2),
            b"DBCA",
        );
    }
}
