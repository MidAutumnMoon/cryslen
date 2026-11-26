//! Wire types shared by the parent and each per-book child process.

#![expect(
    clippy::pub_use,
    reason = "the wire module is the crate's public vocabulary for the child integration test"
)]

use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Deref;
use std::time::Duration;

use eros::MsgError;
use ipc_channel::ipc::IpcReceiver;
use ipc_channel::ipc::IpcSender;
use ipc_channel::ipc::IpcSharedMemory;
use serde::Deserialize;
use serde::Serialize;

use crate::decode;
use crate::decode::Decoded;
use crate::decode::Samples;

pub use crate::browse::PageNames;
pub use crate::browse::Request;
pub use crate::browse::WorkRole;
pub use crate::decode::Depth;
pub use crate::decode::Format;
pub use crate::decode::GraySpec;
pub use crate::decode::ImageMeta;
pub use crate::decode::Matrix;
pub use crate::decode::Subsampling;
pub use crate::decode::Transfer;
pub use crate::decode::YuvSpec;

pub const THUMB_EDGE: u32 = 192;

/// Session-wide target box and GPU texture ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
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

    pub fn fit_box(self) -> (u32, u32) {
        self.fit_box
    }

    pub fn max_texture_edge(self) -> u32 {
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

/// `generation` changes on every restart, so a stopped session's frames
/// cannot reach the new one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct PlaybackId {
    pub page: usize,
    pub generation: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum PlaybackCommand {
    /// Open the page's stream.
    Start { id: PlaybackId },
    /// Produce the next frame. A pass boundary is invisible to the parent.
    Next { id: PlaybackId },
    /// Drop the session. The child answers nothing.
    Stop { id: PlaybackId },
}

/// One child answer about a session. Frames arrive in order.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum PlaybackEvent<Image> {
    /// One composited frame, clamped to `max_texture_edge`.
    Frame {
        id: PlaybackId,
        delay: Duration,
        image: Image,
    },
    /// Every allowed pass has been played.
    Finished { id: PlaybackId },
    /// Playback failed. The page stays valid.
    Failed { id: PlaybackId },
}

/// What a freshly opened child knows about its book.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BookInfo {
    pub names: PageNames,
    pub archive: bool,
}

/// Pixel storage for one image, one shared region per plane.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum WirePixels {
    Rgba8(IpcSharedMemory),
    Gray {
        spec: GraySpec,
        luma: IpcSharedMemory,
        alpha: Option<IpcSharedMemory>,
    },
    Yuv {
        spec: YuvSpec,
        luma: IpcSharedMemory,
        cb: IpcSharedMemory,
        cr: IpcSharedMemory,
        alpha: Option<IpcSharedMemory>,
    },
}

/// A decoded image in the shape the parent's renderer uploads.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WireImage {
    pub meta: ImageMeta,
    pub pixels: WirePixels,
}

/// Wire image whose extents, bit depth, and plane byte lengths passed validation.
#[derive(Debug, Clone)]
pub struct ValidatedImage(WireImage);

impl Deref for ValidatedImage {
    type Target = WireImage;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl WireImage {
    pub fn from_decoded(image: &Decoded) -> Self {
        use crate::decode::Pixels;

        let pixels = match image.pixels() {
            Pixels::Rgba8(bytes) => {
                WirePixels::Rgba8(IpcSharedMemory::from_bytes(bytes))
            }
            Pixels::Gray(gray) => {
                let (luma, alpha) = match &gray.samples {
                    Samples::U8(planes) => (
                        shared_samples(&planes.luma),
                        planes.alpha.as_deref().map(shared_samples),
                    ),
                    Samples::U16 { planes, .. } => (
                        shared_samples(&planes.luma),
                        planes.alpha.as_deref().map(shared_samples),
                    ),
                };
                WirePixels::Gray {
                    spec: gray.spec(),
                    luma,
                    alpha,
                }
            }
            Pixels::Yuv(yuv) => {
                let (luma, cb, cr, alpha) = match &yuv.samples {
                    Samples::U8(planes) => (
                        shared_samples(&planes.luma),
                        shared_samples(&planes.cb),
                        shared_samples(&planes.cr),
                        planes.alpha.as_deref().map(shared_samples),
                    ),
                    Samples::U16 { planes, .. } => (
                        shared_samples(&planes.luma),
                        shared_samples(&planes.cb),
                        shared_samples(&planes.cr),
                        planes.alpha.as_deref().map(shared_samples),
                    ),
                };
                WirePixels::Yuv {
                    spec: yuv.spec(),
                    luma,
                    cb,
                    cr,
                    alpha,
                }
            }
        };
        Self {
            meta: image.meta(),
            pixels,
        }
    }

    /// Checks that the pixel buffers match the declared extents, bit
    /// depth, and admission bounds.
    pub fn validate(self) -> Result<ValidatedImage, MsgError> {
        match &self.pixels {
            WirePixels::Rgba8(bytes) => decode::validate_rgba_extent(
                self.meta.width,
                self.meta.height,
                bytes.len(),
            )?,
            WirePixels::Gray { spec, luma, alpha } => {
                decode::validate_gray_extents(
                    (self.meta.width, self.meta.height),
                    spec.depth,
                    luma.len(),
                    alpha.as_ref().map(|plane| plane.len()),
                )?;
            }
            WirePixels::Yuv {
                spec,
                luma,
                cb,
                cr,
                alpha,
            } => decode::validate_yuv_extents(
                (self.meta.width, self.meta.height),
                spec.depth,
                spec.subsampling,
                [luma.len(), cb.len(), cr.len()],
                alpha.as_ref().map(|plane| plane.len()),
            )?,
        }
        Ok(ValidatedImage(self))
    }

    /// Bytes held across all shared-memory pixel planes.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        match &self.pixels {
            WirePixels::Rgba8(bytes) => bytes.len(),
            WirePixels::Gray { luma, alpha, .. } => {
                luma.len() + alpha.as_ref().map_or(0, |plane| plane.len())
            }
            WirePixels::Yuv {
                luma,
                cb,
                cr,
                alpha,
                ..
            } => {
                luma.len()
                    + cb.len()
                    + cr.len()
                    + alpha.as_ref().map_or(0, |plane| plane.len())
            }
        }
    }
}

fn shared_samples<Sample>(samples: &[Sample]) -> IpcSharedMemory
where
    Sample: bytemuck::Pod,
{
    IpcSharedMemory::from_bytes(bytemuck::cast_slice(samples))
}

const MAX_THUMB_BYTES: usize = 256 * 1024;

/// Compact WebP thumbnail attached to `Fit` delivery and persisted in the
/// parent's shared thumbnail cache.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct WireThumb {
    pub page_dims: (u32, u32),
    pub format: Format,
    pub webp: Vec<u8>,
}

/// Thumbnail whose metadata, WebP header, and edge bounds passed validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedThumb {
    wire: WireThumb,
    dims: (u32, u32),
}

impl ValidatedThumb {
    #[must_use]
    pub const fn dims(&self) -> (u32, u32) {
        self.dims
    }

    #[must_use]
    pub fn into_wire(self) -> WireThumb {
        self.wire
    }
}

impl Deref for ValidatedThumb {
    type Target = WireThumb;

    fn deref(&self) -> &Self::Target {
        &self.wire
    }
}

impl WireThumb {
    pub fn validate(self) -> Result<ValidatedThumb, MsgError> {
        decode::admit_dims(
            self.page_dims.0,
            self.page_dims.1,
            "thumb page dimensions",
        )?;
        if self.webp.is_empty() || self.webp.len() > MAX_THUMB_BYTES {
            return Err(MsgError::from_static("thumb payload"));
        }
        let Some(dims) = decode::webp_dims(&self.webp) else {
            return Err(MsgError::from_static("thumb payload"));
        };
        if TextureFault::check_dims(dims, THUMB_EDGE).is_err() {
            return Err(MsgError::from_static("thumb dimensions"));
        }
        Ok(ValidatedThumb { wire: self, dims })
    }
}

/// Three-tier demand plan sent from parent to child.
///
/// `current` runs first at max threads and cancels lower tiers;
/// `lookahead` runs next once `current` is empty; `thumbs` warms missing
/// thumbnail cache entries when both upper tiers are empty.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct WorkPlan {
    pub current: Vec<Request>,
    pub lookahead: Vec<usize>,
    pub thumbs: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ParentMsg {
    SetLimits(ImageLimits),
    SetWork(WorkPlan),
    Playback(PlaybackCommand),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ChildMsg {
    Ready {
        page: usize,
        role: WorkRole,
        image: WireImage,
        thumb: Option<WireThumb>,
    },
    Thumb {
        page: usize,
        thumb: WireThumb,
    },
    Failed {
        page: usize,
        fault: String,
    },
    Playback(PlaybackEvent<WireImage>),
}

/// Initial worker handshake sent on connect.
#[derive(Debug, Deserialize, Serialize)]
pub enum Hello {
    Opened {
        info: BookInfo,
        messages: IpcReceiver<ChildMsg>,
        commands: IpcSender<ParentMsg>,
    },
    Failed(String),
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;

    /// One solid color, so a wrong plane layout or byte order shows.
    const PIXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];

    #[test]
    fn rgba_image_round_trips_through_the_wire() {
        // Non-square, so a width/height swap cannot pass.
        let pixels = [PIXEL; 6].concat().into_boxed_slice();
        let decoded = Decoded::rgba(
            3,
            2,
            Format::Png,
            Transfer::Srgb,
            pixels.clone(),
        )
        .expect("a well-formed rgba page");

        let wire = WireImage::from_decoded(&decoded);
        assert_eq!(
            (wire.meta.width, wire.meta.height),
            (3, 2),
            "the wire keeps the pixel extents",
        );
        assert_eq!(
            wire.meta.format,
            Format::Png,
            "the wire keeps the format"
        );
        assert_eq!(
            wire.meta.page_dims,
            (3, 2),
            "the wire keeps the page dimensions",
        );
        let WirePixels::Rgba8(bytes) = &wire.pixels else {
            panic!("an rgba page stays rgba on the wire");
        };
        assert_eq!(
            bytes.to_vec(),
            pixels.to_vec(),
            "the wire keeps the pixels",
        );
        assert_eq!(wire.resident_bytes(), pixels.len());
        let mut truncated = wire.clone();
        let validated =
            wire.validate().expect("valid rgba wire image validates");
        assert_eq!(validated.resident_bytes(), pixels.len());

        truncated.meta.width = 4;
        assert_eq!(
            truncated
                .validate()
                .expect_err("truncated rgba image is rejected"),
            MsgError::from_static("rgba extent")
        );
    }

    #[test]
    fn planar_wire_validation_checks_depth_and_plane_extents() {
        use crate::decode::GrayImage;
        use crate::decode::GrayPlanes;
        use crate::decode::WideDepth;
        use crate::decode::YuvImage;
        use crate::decode::YuvPlanes;

        fn check_planar(
            decoded: &Decoded,
            expected_bytes: usize,
            expected_error: &'static str,
            corrupt_depth: impl FnOnce(&mut WirePixels),
        ) {
            let wire = WireImage::from_decoded(decoded);
            assert_eq!(wire.resident_bytes(), expected_bytes);
            let mut mismatched_depth = wire.clone();
            let mut bad_dims = wire.clone();
            wire.validate().expect("valid wire image validates");

            corrupt_depth(&mut mismatched_depth.pixels);
            assert_eq!(
                mismatched_depth
                    .validate()
                    .expect_err("mismatched plane depth is rejected"),
                MsgError::from_static(expected_error)
            );

            bad_dims.meta.width += 2;
            assert_eq!(
                bad_dims
                    .validate()
                    .expect_err("mismatched plane width is rejected"),
                MsgError::from_static(expected_error)
            );
        }

        let yuv = Decoded::yuv(
            4,
            2,
            Format::Avif,
            Transfer::Srgb,
            YuvImage {
                subsampling: Subsampling::Yuv420,
                full_range: false,
                matrix: Matrix::Bt709,
                samples: Samples::U16 {
                    depth: WideDepth::D10,
                    planes: YuvPlanes {
                        luma: vec![512_u16; 4 * 2].into_boxed_slice(),
                        cb: vec![512_u16; 2].into_boxed_slice(),
                        cr: vec![512_u16; 2].into_boxed_slice(),
                        alpha: None,
                    },
                },
            },
        )
        .expect("valid yuv page");
        check_planar(
            &yuv,
            (4 * 2 + 2 + 2) * size_of::<u16>(),
            "yuv plane extent",
            |pixels| {
                if let WirePixels::Yuv { spec, .. } = pixels {
                    spec.depth = Depth::D8;
                }
            },
        );

        let gray = Decoded::gray(
            4,
            2,
            Format::Avif,
            Transfer::Srgb,
            GrayImage {
                full_range: true,
                samples: Samples::U16 {
                    depth: WideDepth::D10,
                    planes: GrayPlanes {
                        luma: vec![512_u16; 4 * 2].into_boxed_slice(),
                        alpha: Some(
                            vec![1023_u16; 4 * 2].into_boxed_slice(),
                        ),
                    },
                },
            },
        )
        .expect("valid gray page");
        check_planar(
            &gray,
            (4 * 2 + 4 * 2) * size_of::<u16>(),
            "gray plane extent",
            |pixels| {
                if let WirePixels::Gray { spec, .. } = pixels {
                    spec.depth = Depth::D8;
                }
            },
        );
    }
}
