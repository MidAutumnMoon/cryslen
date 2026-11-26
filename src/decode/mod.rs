//! Decoding: raw page bytes to `Decoded` (RGBA, Gray, or YUV planes).

#![expect(
    clippy::definition_in_module_root,
    reason = "the decode tree's shared vocabulary lives in its module root by design"
)]

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use bytes::Bytes;
use eros::MsgError;
use image::metadata::Orientation;
use serde::Deserialize;
use serde::Serialize;
use strum::Display;
use strum::FromRepr;

mod avif;
mod jxl;
mod raster;
mod webp;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct ImageMeta {
    /// Decoded dimensions of this representation; may be downsampled from `page_dims`.
    pub width: u32,
    pub height: u32,
    /// Original unscaled page dimensions.
    pub page_dims: (u32, u32),
    pub format: Format,
    /// Sample encoding the renderer must undo before the framebuffer.
    pub transfer: Transfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Depth {
    D8,
    D10,
    D12,
    D16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
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

impl Depth {
    #[must_use]
    pub const fn is_wide(self) -> bool {
        !matches!(self, Self::D8)
    }

    #[must_use]
    pub const fn wide(self) -> Option<WideDepth> {
        match self {
            Self::D8 => None,
            Self::D10 => Some(WideDepth::D10),
            Self::D12 => Some(WideDepth::D12),
            Self::D16 => Some(WideDepth::D16),
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct GraySpec {
    pub depth: Depth,
    pub full_range: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct YuvSpec {
    pub depth: Depth,
    pub subsampling: Subsampling,
    /// Limited (video) or full (JPEG) range quantization.
    pub full_range: bool,
    pub matrix: Matrix,
}

/// Construct via [`Decoded::rgba`] / [`Decoded::gray`] / [`Decoded::yuv`],
/// which validate the pixel extents against the dimensions.
#[derive(Clone, PartialEq, Eq)]
pub struct Decoded {
    meta: ImageMeta,
    pixels: Pixels,
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
    pub fn meta(&self) -> ImageMeta {
        self.meta
    }

    pub fn width(&self) -> u32 {
        self.meta.width
    }

    pub fn height(&self) -> u32 {
        self.meta.height
    }

    pub fn pixels(&self) -> &Pixels {
        &self.pixels
    }

    pub fn transfer(&self) -> Transfer {
        self.meta.transfer
    }

    pub fn format(&self) -> Format {
        self.meta.format
    }

    /// The page's dimensions, however far this representation is
    /// downsampled.
    pub fn page_dims(&self) -> (u32, u32) {
        self.meta.page_dims
    }

    #[must_use]
    pub fn with_page_dims(mut self, page_dims: (u32, u32)) -> Self {
        self.meta.page_dims = page_dims;
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
            meta: ImageMeta {
                width,
                height,
                page_dims: (width, height),
                format,
                transfer,
            },
            pixels: Pixels::Rgba8(bytes),
        })
    }

    pub fn gray(
        width: u32,
        height: u32,
        format: Format,
        transfer: Transfer,
        gray: GrayImage,
    ) -> Result<Self, MsgError> {
        let (luma_bytes, alpha_bytes) = gray.byte_lens();
        validate_gray_extents(
            (width, height),
            gray.depth(),
            luma_bytes,
            alpha_bytes,
        )?;
        Ok(Self {
            meta: ImageMeta {
                width,
                height,
                page_dims: (width, height),
                format,
                transfer,
            },
            pixels: Pixels::Gray(gray),
        })
    }

    pub fn yuv(
        width: u32,
        height: u32,
        format: Format,
        transfer: Transfer,
        yuv: YuvImage,
    ) -> Result<Self, MsgError> {
        let (plane_bytes, alpha_bytes) = yuv.byte_lens();
        validate_yuv_extents(
            (width, height),
            yuv.depth(),
            yuv.subsampling,
            plane_bytes,
            alpha_bytes,
        )?;
        Ok(Self {
            meta: ImageMeta {
                width,
                height,
                page_dims: (width, height),
                format,
                transfer,
            },
            pixels: Pixels::Yuv(yuv),
        })
    }

    #[must_use]
    fn apply_orientation(mut self, orientation: Orientation) -> Self {
        if orientation == Orientation::NoTransforms {
            return self;
        }
        let dims = (self.meta.width, self.meta.height);
        debug_assert!(
            !(orientation_swaps_dims(orientation)
                && matches!(
                    &self.pixels,
                    Pixels::Yuv(yuv)
                        if yuv.subsampling == Subsampling::Yuv422
                )),
            "rotated 4:2:2 must be rejected before orientation"
        );
        self.pixels = match &self.pixels {
            Pixels::Rgba8(bytes) => {
                Pixels::Rgba8(transform_rgba(bytes, dims, orientation))
            }
            Pixels::Gray(gray) => Pixels::Gray(GrayImage {
                full_range: gray.full_range,
                samples: match &gray.samples {
                    Samples::U8(planes) => {
                        Samples::U8(planes.transform(dims, orientation))
                    }
                    Samples::U16 { depth, planes } => Samples::U16 {
                        depth: *depth,
                        planes: planes.transform(dims, orientation),
                    },
                },
            }),
            Pixels::Yuv(yuv) => {
                let (luma_dims, chroma_dims) =
                    plane_dims(dims, yuv.subsampling);
                Pixels::Yuv(YuvImage {
                    subsampling: yuv.subsampling,
                    full_range: yuv.full_range,
                    matrix: yuv.matrix,
                    samples: match &yuv.samples {
                        Samples::U8(planes) => Samples::U8(planes.transform(
                            luma_dims,
                            chroma_dims,
                            orientation,
                        )),
                        Samples::U16 { depth, planes } => Samples::U16 {
                            depth: *depth,
                            planes: planes.transform(
                                luma_dims,
                                chroma_dims,
                                orientation,
                            ),
                        },
                    },
                })
            }
        };
        let out_dims = oriented_dims(orientation, dims);
        self.meta.width = out_dims.0;
        self.meta.height = out_dims.1;
        self.meta.page_dims = out_dims;
        self
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
            GrayImage {
                full_range: true,
                samples: Samples::U8(GrayPlanes { luma, alpha: None }),
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
            GrayImage {
                full_range: true,
                samples: Samples::U16 {
                    depth,
                    planes: GrayPlanes { luma, alpha: None },
                },
            },
        )
    }

    pub fn resident_bytes(&self) -> usize {
        match &self.pixels {
            Pixels::Rgba8(bytes) => bytes.len(),
            Pixels::Gray(gray) => gray.byte_len(),
            Pixels::Yuv(yuv) => yuv.byte_len(),
        }
    }
}

/// Pixel count of `(width, height)` when non-empty and within
/// [`MAX_PIXELS`].
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

pub const RGBA_BYTES_PER_PIXEL: usize = 4;

#[derive(Debug, Clone, Copy)]
pub struct YuvLayout {
    pub luma_samples: usize,
    pub chroma_samples: usize,
}

pub fn rgba_byte_len(width: u32, height: u32) -> Result<usize, MsgError> {
    let pixels = admit_dims(width, height, "bad dimensions")?;
    (pixels as usize)
        .checked_mul(RGBA_BYTES_PER_PIXEL)
        .ok_or_else(|| MsgError::from_static("bad dimensions"))
}

pub fn validate_rgba_extent(
    width: u32,
    height: u32,
    bytes: usize,
) -> Result<(), MsgError> {
    if bytes != rgba_byte_len(width, height)? {
        return Err(MsgError::from_static("rgba extent"));
    }
    Ok(())
}

pub fn validate_gray_extents(
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

pub fn yuv_layout(
    dims: (u32, u32),
    subsampling: Subsampling,
) -> Result<YuvLayout, MsgError> {
    let luma_samples =
        admit_dims(dims.0, dims.1, "bad dimensions")? as usize;
    let (_, (chroma_width, chroma_height)) = plane_dims(dims, subsampling);
    let chroma_samples =
        admit_dims(chroma_width, chroma_height, "bad dimensions")? as usize;
    Ok(YuvLayout {
        luma_samples,
        chroma_samples,
    })
}

pub fn validate_yuv_extents(
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

/// The image crate's 512 MiB allocation ceiling, as RGBA pixels.
pub const MAX_PIXELS: u32 =
    (512 * 1024 * 1024) >> RGBA_BYTES_PER_PIXEL.ilog2();

#[derive(Clone, PartialEq, Eq)]
pub enum Pixels {
    /// Interleaved RGBA, `4 * width * height` bytes, row-major, top-first.
    Rgba8(Box<[u8]>),
    /// Planar grayscale, converted to RGB in the renderer's shader.
    Gray(GrayImage),
    /// Planar YUV, converted to RGB in the renderer's shader.
    Yuv(YuvImage),
}
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Deserialize,
    Serialize,
    Display,
    FromRepr
)]
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

impl Format {
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

#[derive(Clone, PartialEq, Eq)]
pub struct GrayPlanes<Sample> {
    pub luma: Box<[Sample]>,
    pub alpha: Option<Box<[Sample]>>,
}

impl<Sample> GrayPlanes<Sample> {
    fn byte_lens(&self) -> (usize, Option<usize>) {
        (
            size_of_val(&*self.luma),
            self.alpha.as_deref().map(size_of_val),
        )
    }

    fn transform(&self, dims: (u32, u32), orientation: Orientation) -> Self
    where
        Sample: Copy,
    {
        Self {
            luma: transform_samples(&self.luma, dims, orientation),
            alpha: self
                .alpha
                .as_ref()
                .map(|alpha| transform_samples(alpha, dims, orientation)),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct YuvPlanes<Sample> {
    pub luma: Box<[Sample]>,
    pub cb: Box<[Sample]>,
    pub cr: Box<[Sample]>,
    pub alpha: Option<Box<[Sample]>>,
}

impl<Sample> YuvPlanes<Sample> {
    fn byte_lens(&self) -> ([usize; 3], Option<usize>) {
        (
            [
                size_of_val(&*self.luma),
                size_of_val(&*self.cb),
                size_of_val(&*self.cr),
            ],
            self.alpha.as_deref().map(size_of_val),
        )
    }

    fn transform(
        &self,
        luma_dims: (u32, u32),
        chroma_dims: (u32, u32),
        orientation: Orientation,
    ) -> Self
    where
        Sample: Copy,
    {
        Self {
            luma: transform_samples(&self.luma, luma_dims, orientation),
            cb: transform_samples(&self.cb, chroma_dims, orientation),
            cr: transform_samples(&self.cr, chroma_dims, orientation),
            alpha: self
                .alpha
                .as_ref()
                .map(|alpha| transform_samples(alpha, luma_dims, orientation)),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum Samples<Planes8, Planes16> {
    U8(Planes8),
    U16 {
        depth: WideDepth,
        planes: Planes16,
    },
}

impl<Planes8, Planes16> Samples<Planes8, Planes16> {
    #[must_use]
    pub const fn depth(&self) -> Depth {
        match self {
            Self::U8(_) => Depth::D8,
            Self::U16 { depth, .. } => depth.depth(),
        }
    }
}

pub type GraySamples = Samples<GrayPlanes<u8>, GrayPlanes<u16>>;
pub type YuvSamples = Samples<YuvPlanes<u8>, YuvPlanes<u16>>;

/// 8-bit planes are one byte per sample; deeper planes are `u16` with
/// `depth` significant bits. Alpha, when present, is a luma-sized
/// full-range plane.
#[derive(Clone, PartialEq, Eq)]
pub struct GrayImage {
    pub full_range: bool,
    pub samples: GraySamples,
}

impl GrayImage {
    #[must_use]
    pub const fn depth(&self) -> Depth {
        self.samples.depth()
    }

    #[must_use]
    pub const fn spec(&self) -> GraySpec {
        GraySpec {
            depth: self.depth(),
            full_range: self.full_range,
        }
    }

    #[must_use]
    pub fn has_alpha(&self) -> bool {
        match &self.samples {
            Samples::U8(planes) => planes.alpha.is_some(),
            Samples::U16 { planes, .. } => planes.alpha.is_some(),
        }
    }

    fn byte_lens(&self) -> (usize, Option<usize>) {
        match &self.samples {
            Samples::U8(planes) => planes.byte_lens(),
            Samples::U16 { planes, .. } => planes.byte_lens(),
        }
    }

    fn byte_len(&self) -> usize {
        let (luma, alpha) = self.byte_lens();
        luma + alpha.unwrap_or(0)
    }
}

/// 8-bit planes are one byte per sample; deeper planes are `u16` with
/// `depth` significant bits. Alpha, when present, is a luma-sized
/// full-range plane.
#[derive(Clone, PartialEq, Eq)]
pub struct YuvImage {
    pub subsampling: Subsampling,
    pub full_range: bool,
    pub matrix: Matrix,
    pub samples: YuvSamples,
}

impl YuvImage {
    #[must_use]
    pub const fn depth(&self) -> Depth {
        self.samples.depth()
    }

    #[must_use]
    pub const fn spec(&self) -> YuvSpec {
        YuvSpec {
            depth: self.depth(),
            subsampling: self.subsampling,
            full_range: self.full_range,
            matrix: self.matrix,
        }
    }

    #[must_use]
    pub fn has_alpha(&self) -> bool {
        match &self.samples {
            Samples::U8(planes) => planes.alpha.is_some(),
            Samples::U16 { planes, .. } => planes.alpha.is_some(),
        }
    }

    fn byte_lens(&self) -> ([usize; 3], Option<usize>) {
        match &self.samples {
            Samples::U8(planes) => planes.byte_lens(),
            Samples::U16 { planes, .. } => planes.byte_lens(),
        }
    }

    fn byte_len(&self) -> usize {
        let ([luma, cb, cr], alpha) = self.byte_lens();
        luma + cb + cr + alpha.unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Subsampling {
    Yuv420,
    Yuv422,
    Yuv444,
}

impl Subsampling {
    /// Chroma texels per luma texel on each axis: the sampling ratio
    /// behind `plane_dims`'s texel counts.
    #[must_use]
    pub fn chroma_ratio(self) -> (f64, f64) {
        match self {
            Self::Yuv444 => (1.0, 1.0),
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

fn orientation_swaps_dims(orientation: Orientation) -> bool {
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
    let (out_width, out_height) = oriented_dims(orientation, (width, height));
    let (width, height) = (width as usize, height as usize);
    debug_assert_eq!(
        samples.len(),
        width * height,
        "plane sample count must match dimensions"
    );
    let mut out = Vec::with_capacity(samples.len());
    for out_y in 0..out_height as usize {
        for out_x in 0..out_width as usize {
            let (source_x, source_y) = match orientation {
                Orientation::NoTransforms => (out_x, out_y),
                Orientation::Rotate90 => (out_y, height - 1 - out_x),
                Orientation::Rotate180 => {
                    (width - 1 - out_x, height - 1 - out_y)
                }
                Orientation::Rotate270 => (width - 1 - out_y, out_x),
                Orientation::FlipHorizontal => (width - 1 - out_x, out_y),
                Orientation::FlipVertical => (out_x, height - 1 - out_y),
                Orientation::Rotate90FlipH => (out_y, out_x),
                Orientation::Rotate270FlipH => {
                    (width - 1 - out_y, height - 1 - out_x)
                }
            };
            #[expect(
                clippy::indexing_slicing,
                reason = "source coordinates stay within (width, height)"
            )]
            out.push(samples[source_y * width + source_x]);
        }
    }
    out.into_boxed_slice()
}

pub const R8_STORAGE_MAX: f32 = 255.0;
pub const R16_STORAGE_MAX: f32 = 65535.0;
pub const CHROMA_NEUTRAL_8BIT: u8 = 128;

/// Fold storage samples to gamma RGB: `matrix * samples - offset`.
/// Samples are `0..1` storage units; `R16` sampling divides by `65535`.
#[derive(Debug, Clone, Copy)]
pub struct ColorConversion {
    /// Column-major 3x3.
    pub matrix: [f32; 9],
    pub offset: [f32; 3],
}

impl ColorConversion {
    #[must_use]
    pub const fn gray(gain: f32, offset: f32) -> Self {
        Self {
            matrix: [gain, gain, gain, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            offset: [offset; 3],
        }
    }

    #[must_use]
    pub fn apply(&self, samples: [f32; 3]) -> [f32; 3] {
        let mut rgb = [0.0_f32; 3];
        for (column, sample) in
            self.matrix.as_chunks::<3>().0.iter().zip(samples)
        {
            for (&cell, slot) in column.iter().zip(&mut rgb) {
                *slot = cell.mul_add(sample, *slot);
            }
        }
        for (slot, offset) in rgb.iter_mut().zip(self.offset) {
            *slot -= offset;
        }
        rgb
    }
}

#[must_use]
pub fn max_storage_code(depth: Depth) -> f32 {
    if depth.is_wide() {
        R16_STORAGE_MAX
    } else {
        R8_STORAGE_MAX
    }
}

fn max_sample_code(depth: Depth) -> u16 {
    u16::MAX >> (Depth::D16.bits() - depth.bits())
}

/// Compensate for GL `R16` normalization: sampling divides by
/// [`R16_STORAGE_MAX`] even when only `depth` bits are used.
#[must_use]
pub fn depth_normalization(depth: Depth) -> f32 {
    max_storage_code(depth) / f32::from(max_sample_code(depth))
}

struct RangeQuantization {
    luma_black: f32,
    luma_scale: f32,
    chroma_scale: f32,
    chroma_neutral: f32,
}

fn range_quantization(depth: Depth, full_range: bool) -> RangeQuantization {
    const STUDIO_LUMA_BLACK_8BIT: u16 = 16;
    const STUDIO_LUMA_SWING_8BIT: u16 = 219;
    const STUDIO_CHROMA_SWING_8BIT: u16 = 224;

    let depth_scale = 1_u16 << (depth.bits() - Depth::D8.bits());
    let chroma_neutral =
        f32::from(u16::from(CHROMA_NEUTRAL_8BIT) * depth_scale);
    let (luma_black, luma_scale, chroma_scale) = if full_range {
        (0.0, 1.0, 1.0)
    } else {
        let max_code = f32::from(max_sample_code(depth));
        (
            f32::from(STUDIO_LUMA_BLACK_8BIT * depth_scale),
            max_code / f32::from(STUDIO_LUMA_SWING_8BIT * depth_scale),
            max_code / f32::from(STUDIO_CHROMA_SWING_8BIT * depth_scale),
        )
    };
    RangeQuantization {
        luma_black,
        luma_scale,
        chroma_scale,
        chroma_neutral,
    }
}

#[must_use]
pub fn gray_gain_offset(depth: Depth, full_range: bool) -> (f32, f32) {
    let range = range_quantization(depth, full_range);
    let gain = range.luma_scale * depth_normalization(depth);
    let offset = gain * (range.luma_black / max_storage_code(depth));
    (gain, offset)
}

/// Fold storage samples to gamma RGB: depth normalization, quantization,
/// and the KR/KB matrix (or GBR permutation). The offset is the matrix
/// times the anchor codes.
#[must_use]
pub fn yuv_color_conversion(
    depth: Depth,
    full_range: bool,
    color_matrix: Matrix,
) -> ColorConversion {
    const GBR_TO_RGB_MATRIX: [f32; 9] =
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];

    let range = range_quantization(depth, full_range);
    let (mut matrix, luma_scale, chroma_scale, luma_black, chroma_neutral) =
        if color_matrix == Matrix::Identity {
            // H.273 identity planes are GBR in the (Y, Cb, Cr) slots,
            // quantized as luma on all three axes: permute into RGB and
            // anchor every axis at luma black.
            (
                GBR_TO_RGB_MATRIX,
                range.luma_scale,
                range.luma_scale,
                range.luma_black,
                range.luma_black,
            )
        } else {
            (
                yuv_to_rgb_matrix(color_matrix),
                range.luma_scale,
                range.chroma_scale,
                range.luma_black,
                range.chroma_neutral,
            )
        };
    let storage_max = max_storage_code(depth);
    let normalization = depth_normalization(depth);
    for (column, matrix_column) in
        matrix.as_chunks_mut::<3>().0.iter_mut().enumerate()
    {
        let scale = if column == 0 {
            luma_scale
        } else {
            chroma_scale
        } * normalization;
        for cell in matrix_column {
            *cell *= scale;
        }
    }
    let anchors = [
        luma_black / storage_max,
        chroma_neutral / storage_max,
        chroma_neutral / storage_max,
    ];
    let mut offset = [0.0_f32; 3];
    for (matrix_column, anchor) in
        matrix.as_chunks::<3>().0.iter().zip(anchors)
    {
        for (&cell, slot) in matrix_column.iter().zip(&mut offset) {
            *slot = cell.mul_add(anchor, *slot);
        }
    }
    ColorConversion { matrix, offset }
}

/// Full-range YUV->RGB from the exact KR/KB of each standard.
/// KG = 1 - KR - KB. Identity pages never reach here;
/// `yuv_color_conversion` routes them through the GBR permutation.
#[expect(
    clippy::similar_names,
    reason = "the green coefficients are the Cb and Cr terms of the BT matrix derivation"
)]
fn yuv_to_rgb_matrix(matrix: Matrix) -> [f32; 9] {
    const BT601_KR_KB: (f32, f32) = (0.299, 0.114);
    const BT709_KR_KB: (f32, f32) = (0.2126, 0.0722);
    const BT2020_KR_KB: (f32, f32) = (0.2627, 0.0593);

    let (k_red, k_blue) = match matrix {
        Matrix::Bt601 => BT601_KR_KB,
        Matrix::Bt709 => BT709_KR_KB,
        Matrix::Bt2020 => BT2020_KR_KB,
        Matrix::Identity => (0.0, 0.0),
    };
    let k_green = 1.0 - k_red - k_blue;
    let green_from_cb = 2.0 * k_blue * (1.0 - k_blue) / k_green;
    let green_from_cr = 2.0 * k_red * (1.0 - k_red) / k_green;
    [
        1.0,
        1.0,
        1.0, //
        0.0,
        -green_from_cb,
        2.0 * (1.0 - k_blue), //
        2.0 * (1.0 - k_red),
        -green_from_cr,
        0.0,
    ]
}

/// Map a normalized `[0, 1]` PQ channel value to sRGB `[0, 1]` using the
/// same 203-nit diffuse-white clip as `GLSL_TRANSFER_HEADER`.
#[must_use]
pub fn pq_to_srgb_channel(value: f32) -> f32 {
    const PQ_M1: f32 = 2610.0 / 16384.0;
    const PQ_M2: f32 = 2523.0 / 4096.0 * 128.0;
    const PQ_C1: f32 = 3424.0 / 4096.0;
    const PQ_C2: f32 = 2413.0 / 4096.0 * 32.0;
    const PQ_C3: f32 = 2392.0 / 4096.0 * 32.0;
    const PQ_DIFFUSE_WHITE_NITS: f32 = 203.0;
    const PQ_PEAK_NITS: f32 = 10_000.0;
    const SRGB_LINEAR_THRESHOLD: f32 = 0.003_130_8;
    const SRGB_LINEAR_SLOPE: f32 = 12.92;
    const SRGB_GAMMA_SCALE: f32 = 1.055;
    const SRGB_GAMMA_OFFSET: f32 = 0.055;
    const SRGB_GAMMA_EXPONENT: f32 = 2.4;

    let pq_scaled = value.clamp(0.0, 1.0).powf(1.0 / PQ_M2);
    let luminance_nits = PQ_PEAK_NITS
        * ((pq_scaled - PQ_C1).max(0.0) / PQ_C3.mul_add(-pq_scaled, PQ_C2))
            .powf(1.0 / PQ_M1);
    let sdr_linear = (luminance_nits / PQ_DIFFUSE_WHITE_NITS).clamp(0.0, 1.0);
    if sdr_linear <= SRGB_LINEAR_THRESHOLD {
        sdr_linear * SRGB_LINEAR_SLOPE
    } else {
        SRGB_GAMMA_SCALE
            .mul_add(sdr_linear.powf(1.0 / SRGB_GAMMA_EXPONENT), -SRGB_GAMMA_OFFSET)
    }
}

pub enum DecodeOutcome {
    Ready(Decoded),
    Canceled,
}

#[cfg(test)]
pub fn decode(bytes: &[u8], name: &str) -> eros::Result<Decoded> {
    const TEST_MAX_THREADS: usize = 2;
    let cancel = AtomicBool::new(false);
    match decode_cancellable(bytes, name, TEST_MAX_THREADS, &cancel)? {
        DecodeOutcome::Ready(decoded) => Ok(decoded),
        DecodeOutcome::Canceled => eros::bail!("decode canceled"),
    }
}

/// Decode one page's first raster, aborting cooperatively when `cancel`
/// flips during JXL decoding.
pub fn decode_cancellable(
    bytes: &[u8],
    name: &str,
    max_threads: usize,
    cancel: &AtomicBool,
) -> eros::Result<DecodeOutcome> {
    if cancel.load(Ordering::Relaxed) {
        return Ok(DecodeOutcome::Canceled);
    }
    let Some(format) = sniff(bytes).or_else(|| format_from_name(name))
    else {
        eros::bail!("not an image");
    };
    let max_threads = max_threads.max(1);
    match format {
        Format::Avif => avif::decode(bytes, max_threads).map(DecodeOutcome::Ready),
        Format::Jxl => jxl::decode(bytes, max_threads, cancel),
        Format::Webp => webp::decode(bytes, max_threads).map(DecodeOutcome::Ready),
        Format::Jpeg
        | Format::Png
        | Format::Gif
        | Format::Bmp
        | Format::Tiff => raster::decode(bytes, format).map(DecodeOutcome::Ready),
    }
}

/// An animated page's frame stream. Frames arrive sequentially; there is
/// no seek.
pub struct Animation {
    bytes: Bytes,
    loops: Loops,
    frames: image::Frames<'static>,
}

impl Animation {
    #[must_use]
    pub fn loops(&self) -> Loops {
        self.loops
    }

    /// Restart the pass at frame 0.
    pub fn rewind(&mut self) -> eros::Result<()> {
        self.frames = open_gif_frames(self.bytes.clone())?;
        Ok(())
    }

    /// `None` means the pass ended.
    pub fn next_frame(&mut self) -> Option<eros::Result<Frame>> {
        let frame = match self.frames.next() {
            None => return None,
            Some(Ok(frame)) => frame,
            Some(Err(error)) => {
                return Some(Err(error.into()));
            }
        };
        let delay = Duration::from(frame.delay());
        let buffer = frame.into_buffer();
        let (width, height) = buffer.dimensions();
        let rgba = buffer.into_raw().into_boxed_slice();
        Some(
            Decoded::rgba(
                width,
                height,
                Format::Gif,
                Transfer::Srgb,
                rgba,
            )
            .map(|image| Frame { image, delay })
            .map_err(Into::into),
        )
    }
}

/// One composited animation frame at the page canvas size.
#[derive(Clone)]
pub struct Frame {
    pub image: Decoded,
    pub delay: Duration,
}

fn open_gif_frames(
    bytes: Bytes,
) -> eros::Result<image::Frames<'static>> {
    use image::AnimationDecoder as _;
    use image::ImageDecoder as _;
    use image::Limits;
    use image::codecs::gif::GifDecoder;
    use std::io::Cursor;

    const CANVAS_BUFFERS: u64 = 2;

    let mut decoder = GifDecoder::new(Cursor::new(bytes))?;
    let (width, height) = decoder.dimensions();
    admit_dims(width, height, "bad dimensions")?;
    let screen_bytes =
        u64::from(width) * u64::from(height) * RGBA_BYTES_PER_PIXEL as u64;
    let max_frame_bytes =
        u64::from(MAX_PIXELS) * RGBA_BYTES_PER_PIXEL as u64;
    let mut limits = Limits::default();
    limits.max_alloc = Some(screen_bytes * CANVAS_BUFFERS + max_frame_bytes);
    decoder.set_limits(limits)?;
    Ok(decoder.into_frames())
}

/// `Ok(None)` means the format has no animation path; a still GIF
/// streams one frame.
pub fn open_animation(
    bytes: Bytes,
    name: &str,
) -> eros::Result<Option<Animation>> {
    let Some(format) = sniff(&bytes).or_else(|| format_from_name(name))
    else {
        eros::bail!("not an image");
    };
    if !format.may_animate() {
        return Ok(None);
    }
    let loops = gif_loops(&bytes)?;
    let frames = open_gif_frames(bytes.clone())?;
    tracing::debug!(%format, ?loops, "animation opened");
    Ok(Some(Animation {
        bytes,
        loops,
        frames,
    }))
}

/// GIF's NETSCAPE repeat field. A file without the extension reports
/// `Finite(0)`; the `image` wrapper would call that a forever loop.
fn gif_loops(bytes: &[u8]) -> eros::Result<Loops> {
    use gif::DecodeOptions;
    use gif::Repeat;
    use std::io::Cursor;

    let decoder = DecodeOptions::new().read_info(Cursor::new(bytes))?;
    Ok(match decoder.repeat() {
        Repeat::Infinite => Loops::Infinite,
        Repeat::Finite(0) => Loops::Finite(1),
        Repeat::Finite(repeats) => {
            Loops::Finite(u32::from(repeats) + 1)
        }
    })
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

pub fn encode_webp_rgba(
    dims: (u32, u32),
    rgba: &[u8],
    quality: f32,
) -> eros::Result<Vec<u8>> {
    webp::encode_rgba(dims, rgba, quality)
}

pub fn decode_webp_rgba(
    bytes: &[u8],
    max_edge: u32,
) -> eros::Result<((u32, u32), Vec<u8>)> {
    webp::decode_rgba_bounded(bytes, max_edge)
}

#[must_use]
pub fn webp_dims(bytes: &[u8]) -> Option<(u32, u32)> {
    webp::dims(bytes).ok()
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
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
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
    const BOX_FIELD_BYTES: usize = 4;
    const BOX_HEADER_BYTES: usize = BOX_FIELD_BYTES * 2;
    const MIN_FTYP_BOX_BYTES: usize = BOX_HEADER_BYTES + BOX_FIELD_BYTES;

    let (&box_len_bytes, rest) = bytes.split_first_chunk::<BOX_FIELD_BYTES>()?;
    #[expect(
        clippy::big_endian_bytes,
        reason = "the ISOBMFF box length is big-endian by specification"
    )]
    let box_len = u32::from_be_bytes(box_len_bytes) as usize;
    let kind = rest.first_chunk::<BOX_FIELD_BYTES>()?;
    if kind != b"ftyp" {
        return None;
    }
    let end = box_len.max(MIN_FTYP_BOX_BYTES).min(bytes.len());
    bytes.get(BOX_HEADER_BYTES..end)
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
    use image::codecs::gif::Repeat;

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

        // garbage: no magic and no usable extension refuses
        assert!(decode(b"hello world", "noext").is_err());
        assert!(decode(b"hello world", "note.txt").is_err());
        assert!(decode(&[], "x").is_err());
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
        assert_eq!(
            Decoded::rgba(
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
            Decoded::rgba(
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
        // odd edges round chroma up.
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

        let gray = |luma: Box<[u16]>, alpha: Option<Box<[u16]>>| GrayImage {
            full_range: true,
            samples: Samples::U16 {
                depth: WideDepth::D10,
                planes: GrayPlanes { luma, alpha },
            },
        };
        let real = gray(vec![0; 32 * 16].into_boxed_slice(), None);
        let page = Decoded::gray(32, 16, Format::Avif, Transfer::Srgb, real)
            .expect("gray page constructs");
        // The app's upload drain charges this count against its
        // per-frame byte budget.
        assert_eq!(page.resident_bytes(), 32 * 16 * 2);
        let empty = gray(Box::new([]), None);
        assert_eq!(
            Decoded::gray(32, 16, Format::Avif, Transfer::Srgb, empty)
                .err(),
            Some(MsgError::from_static("gray plane extent"))
        );
        let short_alpha = gray(
            vec![0; 32 * 16].into_boxed_slice(),
            Some(vec![0; 5].into_boxed_slice()),
        );
        assert_eq!(
            Decoded::gray(
                32,
                16,
                Format::Avif,
                Transfer::Srgb,
                short_alpha
            )
            .err(),
            Some(MsgError::from_static("gray plane extent"))
        );
        // A 420 image whose chroma planes are stubs: a 4x2 image needs
        // 2x1 chroma samples.
        let undersized = YuvImage {
            subsampling: Subsampling::Yuv420,
            full_range: true,
            matrix: Matrix::Bt601,
            samples: Samples::U8(YuvPlanes {
                luma: vec![0; 4 * 2].into_boxed_slice(),
                cb: Box::new([]),
                cr: Box::new([]),
                alpha: None,
            }),
        };
        assert_eq!(
            Decoded::yuv(4, 2, Format::Avif, Transfer::Srgb, undersized)
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
            let page = Decoded::gray_u8(
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
                Pixels::Gray(GrayImage {
                    samples: Samples::U8(planes),
                    ..
                }) if &*planes.luma == want_samples
            ));
        }
        check(b"AB", (2, 1), Orientation::NoTransforms, (2, 1), b"AB");
        // [A B]: 90° CW stands A above B.
        check(b"AB", (2, 1), Orientation::Rotate90, (1, 2), b"AB");
        // 180° reads the row backwards.
        check(b"AB", (2, 1), Orientation::Rotate180, (2, 1), b"BA");
        // 270° CW (90° CCW) stands B above A.
        check(b"AB", (2, 1), Orientation::Rotate270, (1, 2), b"BA");
        // A 2x2 grid [AB / CD]:
        check(b"ABCD", (2, 2), Orientation::FlipHorizontal, (2, 2), b"BADC");
        check(b"ABCD", (2, 2), Orientation::FlipVertical, (2, 2), b"CDAB");
        check(b"ABCD", (2, 2), Orientation::Rotate90FlipH, (2, 2), b"ACBD");
        check(b"ABCD", (2, 2), Orientation::Rotate270FlipH, (2, 2), b"DBCA");
    }

    // 32x16 10-bit 4:0:0 fixture, a black-to-white grayscale gradient.
    #[test]
    fn avif_fixture_decodes() {
        let page = decode(
            include_bytes!("../../tests/fixtures/mono.avif"),
            "a.avif",
        )
        .expect("mono avif decodes");
        assert_eq!(page.format(), Format::Avif);
        assert_eq!((page.width(), page.height()), (32, 16));
        let Pixels::Gray(gray) = &page.pixels else {
            panic!("mono avif must decode to gray plane")
        };
        let Samples::U16 { planes, .. } = &gray.samples else {
            panic!("10-bit luma must be u16 samples")
        };
        assert_eq!(planes.luma.len(), 32 * 16);
        assert!(planes.luma.first() < planes.luma.last());
        assert!(planes.alpha.is_none());
    }

    // wide.jxl and animated.jxl are verbatim copies of exif.jxl and
    // 5_frames_numbered_jxli.jxl from jxl 0.8.0's tests/testdata.
    #[test]
    fn jxl_fixtures_decode() {
        // gray: solid halves
        let gray = decode(
            include_bytes!("../../tests/fixtures/gray.jxl"),
            "a.jxl",
        )
        .expect("gray jxl decodes");
        assert_eq!((gray.width(), gray.height()), (8, 8));
        assert_eq!(gray.format(), Format::Jxl);
        match gray.pixels {
            Pixels::Gray(gray) => {
                assert!(gray.full_range);
                let Samples::U8(planes) = gray.samples else {
                    panic!("8-bit mono jxl must use U8 luma");
                };
                assert!(planes.alpha.is_none());
                assert_eq!(planes.luma.first().copied(), Some(0x20));
                assert_eq!(planes.luma.get(32).copied(), Some(0xE0));
            }
            Pixels::Rgba8(_) | Pixels::Yuv(_) => {
                panic!("opaque gray jxl must decode to gray")
            }
        }

        // alpha: the file's own alpha rides in the fourth byte
        let alpha = decode(
            include_bytes!("../../tests/fixtures/alpha.jxl"),
            "a.jxl",
        )
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
            Pixels::Gray(_) | Pixels::Yuv(_) => {
                panic!("jxl must decode to rgba")
            }
        }

        // wide: 256x128 pins the row/stride order. Opaque alpha on every
        // pixel proves the decoder wrote all rows of the zero-initialised
        // buffer
        let wide = decode(
            include_bytes!("../../tests/fixtures/wide.jxl"),
            "a.jxl",
        )
        .expect("wide jxl decodes");
        assert_eq!((wide.width(), wide.height()), (256, 128));
        let Pixels::Rgba8(wide_rgba) = &wide.pixels else {
            panic!("jxl must decode to rgba")
        };
        assert!(
            wide_rgba.as_chunks::<4>().0.iter().all(|px| px[3] == 0xFF)
        );

        // animated: the numbered frames paint digits, so the first frame
        // carries non-black pixels
        let animated = decode(
            include_bytes!("../../tests/fixtures/animated.jxl"),
            "a.jxl",
        )
        .expect("animated jxl decodes");
        assert_eq!((animated.width(), animated.height()), (256, 256));
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
        let mut animation = open_animation(Bytes::from(bytes), "a.gif")
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
        let mut once = open_animation(Bytes::from(still), "a.gif")
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
        let animation = open_animation(Bytes::from(finite), "a.gif")
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
        let mut animation = open_animation(Bytes::from(bytes), "a.gif")
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

        let fixture = include_bytes!("../../tests/fixtures/gray.jxl");
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
            assert!(result.is_err(), "cut {cut} did not fail");
        }
    }

    #[test]
    fn oversized_gif_frame_is_rejected_by_limits() {
        // 1x1 logical screen, 2-color GCT, and a 20000x20000 frame descriptor
        // (1.6 GiB RGBA, exceeding MAX_PIXELS * 4).
        let bytes = vec![
            b'G', b'I', b'F', b'8', b'9', b'a', // header
            0x01, 0x00, 0x01, 0x00, // 1x1 logical screen
            0x80, 0x00, 0x00, // GCT flag, 2 colors
            0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, // 2 RGB colors
            0x2C, // image separator
            0x00, 0x00, 0x00, 0x00, // left, top = 0, 0
            0x20, 0x4E, 0x20, 0x4E, // width, height = 20000, 20000 (LE)
            0x00, // local color table = 0
            0x02, 0x00, // LZW min code size 2, terminator
            0x3B, // trailer
        ];
        let mut animation = open_animation(Bytes::from(bytes), "big.gif")
            .expect("header opens")
            .expect("gif animation");
        let frame = animation
            .next_frame()
            .expect("oversized frame reports error");
        assert!(frame.is_err(), "oversized frame must hit memory limit");
    }
}
