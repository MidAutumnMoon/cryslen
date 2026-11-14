//! Wire types shared by the parent and each per-book child process.

#![expect(
    clippy::pub_use,
    reason = "the wire module is the crate's public vocabulary for the child integration test"
)]

use ipc_channel::ipc::IpcReceiver;
use ipc_channel::ipc::IpcSender;
use ipc_channel::ipc::IpcSharedMemory;
use serde::Deserialize;
use serde::Serialize;

use crate::decode;
use crate::decode::Decoded;
use crate::decode::Pixels;
use crate::fault::DecodeRefusal;

pub use crate::animation::PlaybackCommand;
pub use crate::animation::PlaybackEvent;
pub use crate::animation::PlaybackId;
pub use crate::browse::Direction;
pub use crate::browse::Motion;
pub use crate::browse::ReadingMode;
pub use crate::browse::Request;
pub use crate::browse::Role;
pub use crate::decode::Format;
pub use crate::decode::Matrix;
pub use crate::decode::Subsampling;
pub use crate::decode::Transfer;
pub use crate::fault::PageFault;

/// What a freshly opened child knows about its book.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[non_exhaustive]
pub struct BookInfo {
    pub names: Vec<String>,
    pub archive: bool,
}

/// Pixel storage for one image, one shared region per plane.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum WirePixels {
    Rgba8(IpcSharedMemory),
    Yuv {
        depth: u8,
        subsampling: Subsampling,
        full_range: bool,
        matrix: Matrix,
        luma: IpcSharedMemory,
        cb: IpcSharedMemory,
        cr: IpcSharedMemory,
        alpha: Option<IpcSharedMemory>,
    },
}

/// A decoded image in the shape the parent's renderer uploads.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[non_exhaustive]
pub struct WireImage {
    pub width: u32,
    pub height: u32,
    pub format: Format,
    pub transfer: Transfer,
    pub page_dims: (u32, u32),
    pub pixels: WirePixels,
}

impl WireImage {
    pub fn from_decoded(image: &Decoded) -> Self {
        let pixels = match image.pixels() {
            Pixels::Rgba8(bytes) => {
                WirePixels::Rgba8(IpcSharedMemory::from_bytes(bytes))
            }
            Pixels::Yuv(yuv) => WirePixels::Yuv {
                depth: yuv.depth,
                subsampling: yuv.subsampling,
                full_range: yuv.full_range,
                matrix: yuv.matrix,
                luma: shared_plane(&yuv.luma),
                cb: shared_plane(&yuv.cb),
                cr: shared_plane(&yuv.cr),
                alpha: yuv.alpha.as_ref().map(shared_plane),
            },
        };
        Self {
            width: image.width(),
            height: image.height(),
            format: image.format(),
            transfer: image.transfer(),
            page_dims: image.page_dims(),
            pixels,
        }
    }

    /// Checks that the pixel buffers match the declared extents, bit
    /// depth, and admission bounds.
    pub fn validate(&self) -> Result<(), DecodeRefusal> {
        match &self.pixels {
            WirePixels::Rgba8(bytes) => decode::validate_rgba_extent(
                self.width,
                self.height,
                bytes.len(),
            ),
            WirePixels::Yuv {
                depth,
                subsampling,
                luma,
                cb,
                cr,
                alpha,
                ..
            } => decode::validate_yuv_extents(
                (self.width, self.height),
                *depth,
                *subsampling,
                [luma.len(), cb.len(), cr.len()],
                alpha.as_ref().map(|plane| plane.len()),
            ),
        }
    }

    /// Bytes held across all shared-memory pixel planes.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        match &self.pixels {
            WirePixels::Rgba8(bytes) => bytes.len(),
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

fn shared_plane(plane: &decode::Plane) -> IpcSharedMemory {
    IpcSharedMemory::from_bytes(decode::plane_bytes(plane))
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ParentMsg {
    SetView {
        page: usize,
        mode: ReadingMode,
        direction: Direction,
    },
    SetMotion(Motion),
    SetRequests(Vec<Request>),
    Playback(PlaybackCommand),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ChildMsg {
    BookInfo(BookInfo),
    OpenFailed(String),
    Ready {
        page: usize,
        role: Role,
        image: WireImage,
    },
    Failed {
        page: usize,
        fault: PageFault,
    },
    Playback(PlaybackEvent<WireImage>),
}

/// What the child sends on connect: the receiver for its messages and the
/// sender for the parent's.
pub type Hello = (IpcReceiver<ChildMsg>, IpcSender<ParentMsg>);

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
            (wire.width, wire.height),
            (3, 2),
            "the wire keeps the pixel extents",
        );
        assert_eq!(wire.format, Format::Png, "the wire keeps the format");
        assert_eq!(
            wire.page_dims,
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
        wire.validate().expect("valid rgba wire image validates");

        let mut truncated = wire;
        truncated.width = 4;
        assert!(matches!(
            truncated.validate(),
            Err(DecodeRefusal::Corrupt("rgba extent"))
        ));
    }

    #[test]
    fn yuv_wire_validation_checks_depth_and_plane_extents() {
        use crate::decode::Plane;
        use crate::decode::YuvImage;

        let yuv = YuvImage {
            depth: 10,
            subsampling: Subsampling::Yuv420,
            full_range: false,
            matrix: Matrix::Bt709,
            luma: Plane::U16(vec![512_u16; 4 * 2].into_boxed_slice()),
            cb: Plane::U16(vec![512_u16; 2].into_boxed_slice()),
            cr: Plane::U16(vec![512_u16; 2].into_boxed_slice()),
            alpha: None,
        };
        let decoded =
            Decoded::yuv(4, 2, Format::Avif, Transfer::Srgb, yuv)
                .expect("valid yuv page");
        let wire = WireImage::from_decoded(&decoded);
        assert_eq!(wire.resident_bytes(), (8 + 2 + 2) * 2);
        wire.validate().expect("valid yuv wire image validates");

        let mut bad_depth = wire.clone();
        if let WirePixels::Yuv { depth, .. } = &mut bad_depth.pixels {
            *depth = 0;
        }
        assert!(matches!(
            bad_depth.validate(),
            Err(DecodeRefusal::Corrupt("yuv depth"))
        ));

        let mut bad_dims = wire;
        bad_dims.width = 6;
        assert!(matches!(
            bad_dims.validate(),
            Err(DecodeRefusal::Corrupt("yuv plane extent"))
        ));
    }
}
