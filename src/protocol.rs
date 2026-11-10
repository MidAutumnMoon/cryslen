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
use crate::decode::Plane;
use crate::decode::YuvImage;
use crate::fault::DecodeRefusal;

pub use crate::decode::Format;
pub use crate::decode::Matrix;
pub use crate::decode::Subsampling;
pub use crate::decode::Transfer;
pub use crate::fault::PageFault;
pub use crate::plan::Direction;
pub use crate::preview::Artifact;
pub use crate::view::ViewMode;
pub use crate::worker::Motion;
pub use crate::worker::Want;

/// What a freshly opened child knows about its book.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[non_exhaustive]
pub struct BookInfo {
    pub names: Vec<String>,
    pub unpacking: bool,
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

    /// Rebuild the decoded image, validating extents through
    /// [`Decoded::rgba`] / [`Decoded::yuv`].
    pub fn to_decoded(&self) -> Result<Decoded, DecodeRefusal> {
        let decoded = match &self.pixels {
            WirePixels::Rgba8(bytes) => Decoded::rgba(
                self.width,
                self.height,
                self.format,
                self.transfer,
                bytes.to_vec().into_boxed_slice(),
            )?,
            WirePixels::Yuv {
                depth,
                subsampling,
                full_range,
                matrix,
                luma,
                cb,
                cr,
                alpha,
            } => Decoded::yuv(
                self.width,
                self.height,
                self.format,
                self.transfer,
                YuvImage {
                    depth: *depth,
                    subsampling: *subsampling,
                    full_range: *full_range,
                    matrix: *matrix,
                    luma: plane_from_shared(luma, *depth),
                    cb: plane_from_shared(cb, *depth),
                    cr: plane_from_shared(cr, *depth),
                    alpha: alpha
                        .as_ref()
                        .map(|plane| plane_from_shared(plane, *depth)),
                },
            )?,
        };
        Ok(decoded.with_page_dims(self.page_dims))
    }
}

fn shared_plane(plane: &Plane) -> IpcSharedMemory {
    IpcSharedMemory::from_bytes(decode::plane_bytes(plane))
}

/// Deeper planes travel as native-endian byte pairs, matching
/// [`decode::plane_bytes`].
fn plane_from_shared(shared: &IpcSharedMemory, depth: u8) -> Plane {
    if depth > 8 {
        // The cast is the fast path; the fallback keeps an unaligned
        // block from panicking.
        let bytes = shared;
        let words = bytemuck::try_cast_slice(bytes).map_or_else(
            |_| bytemuck::pod_collect_to_vec(bytes),
            <[u16]>::to_vec,
        );
        Plane::U16(words.into_boxed_slice())
    } else {
        Plane::U8(shared.to_vec().into_boxed_slice())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ParentMsg {
    SetView {
        index: usize,
        mode: ViewMode,
        direction: Direction,
    },
    SetMotion(Motion),
    SetWants(Vec<Want>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ChildMsg {
    BookInfo(BookInfo),
    OpenFailed(String),
    Ready {
        page: usize,
        kind: Artifact,
        image: WireImage,
    },
    Failed {
        page: usize,
        fault: PageFault,
    },
    Unpacked,
    Fatal(String),
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
        let pixels = [PIXEL; 4].concat().into_boxed_slice();
        let decoded = Decoded::rgba(
            2,
            2,
            Format::Png,
            Transfer::Srgb,
            pixels.clone(),
        )
        .expect("a well-formed rgba page");

        let wire = WireImage::from_decoded(&decoded);
        assert_eq!(
            (wire.width, wire.height),
            (2, 2),
            "the wire keeps the pixel extents",
        );
        assert_eq!(wire.format, Format::Png, "the wire keeps the format");
        assert_eq!(
            wire.page_dims,
            (2, 2),
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

        let back = wire.to_decoded().expect("the wire image decodes");
        assert_eq!(
            (back.width(), back.height()),
            (2, 2),
            "the pixels decode back at the same extents",
        );
        assert_eq!(
            back.page_dims(),
            (2, 2),
            "the page dimensions survive the trip",
        );
        assert_eq!(
            back.format(),
            Format::Png,
            "the format survives the trip",
        );
        match back.pixels() {
            Pixels::Rgba8(back_bytes) => {
                assert_eq!(
                    back_bytes.to_vec(),
                    pixels.to_vec(),
                    "the pixels survive the trip",
                );
            }
            Pixels::Yuv(_) => panic!("an rgba page decodes as rgba"),
        }
    }
}
