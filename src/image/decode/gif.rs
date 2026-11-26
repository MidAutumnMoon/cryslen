//! GIF still decoding (with animation detection) and streaming frames.

use std::sync::Arc;
use std::time::Duration;

use crate::image::pixels;
use crate::image::pixels::Format;
use crate::image::pixels::Image;
use crate::image::pixels::MAX_PIXELS;
use crate::image::pixels::RGBA_BYTES_PER_PIXEL;
use crate::image::pixels::Transfer;

/// Pass count. `Finite` counts total passes, including the first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loops {
    Infinite,
    Finite(u32),
}

/// One composited animation frame at the page canvas size.
#[derive(Clone)]
pub struct Frame {
    pub image: Image,
    pub delay: Duration,
}

/// An animated page's frame stream. Frames arrive sequentially; there is
/// no seek.
pub struct Animation {
    bytes: Arc<[u8]>,
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
        self.frames = open_gif_frames(Arc::clone(&self.bytes))?;
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
            Image::rgba(width, height, Format::Gif, Transfer::Srgb, rgba)
                .map(|image| Frame { image, delay })
                .map_err(Into::into),
        )
    }
}

/// Decode frame 0 of a GIF and mark `meta.animated` when a second frame exists.
pub(super) fn decode(bytes: &[u8]) -> eros::Result<Image> {
    use std::io::Cursor;

    use image::DynamicImage;
    use image::ImageDecoder as _;
    use image::ImageFormat;
    use image::ImageReader;

    let reader =
        ImageReader::with_format(Cursor::new(bytes), ImageFormat::Gif);
    let decoder = reader.into_decoder()?;
    let (width, height) = decoder.dimensions();
    pixels::admit_dims(width, height, "bad dimensions")?;
    let rgba = DynamicImage::from_decoder(decoder)?.into_rgba8();
    let image = Image::rgba(
        rgba.width(),
        rgba.height(),
        Format::Gif,
        Transfer::Srgb,
        rgba.into_raw().into_boxed_slice(),
    )?;
    Ok(image.with_animated(has_multiple_frames(bytes)))
}

fn has_multiple_frames(bytes: &[u8]) -> bool {
    use std::io::Cursor;

    let mut options = gif::DecodeOptions::new();
    options.skip_frame_decoding(true);
    let Ok(mut decoder) = options.read_info(Cursor::new(bytes)) else {
        return false;
    };
    matches!(decoder.next_frame_info(), Ok(Some(_)))
        && matches!(decoder.next_frame_info(), Ok(Some(_)))
}

fn open_gif_frames(
    bytes: Arc<[u8]>,
) -> eros::Result<image::Frames<'static>> {
    use std::io::Cursor;

    use image::AnimationDecoder as _;
    use image::ImageDecoder as _;
    use image::Limits;
    use image::codecs::gif::GifDecoder;

    const CANVAS_BUFFERS: u64 = 2;

    let mut decoder = GifDecoder::new(Cursor::new(bytes))?;
    let (width, height) = decoder.dimensions();
    pixels::admit_dims(width, height, "bad dimensions")?;
    let screen_bytes =
        u64::from(width) * u64::from(height) * RGBA_BYTES_PER_PIXEL as u64;
    let max_frame_bytes =
        u64::from(MAX_PIXELS) * RGBA_BYTES_PER_PIXEL as u64;
    let mut limits = Limits::default();
    limits.max_alloc =
        Some(screen_bytes * CANVAS_BUFFERS + max_frame_bytes);
    decoder.set_limits(limits)?;
    Ok(decoder.into_frames())
}

/// Open a GIF animation stream; a still GIF streams one frame.
pub fn open_animation(bytes: Arc<[u8]>) -> eros::Result<Animation> {
    let loops = gif_loops(&bytes)?;
    let frames = open_gif_frames(Arc::clone(&bytes))?;
    tracing::debug!(?loops, "animation opened");
    Ok(Animation {
        bytes,
        loops,
        frames,
    })
}

/// GIF's NETSCAPE repeat field. A file without the extension reports
/// `Finite(0)`; the `image` wrapper would call that a forever loop.
fn gif_loops(bytes: &[u8]) -> eros::Result<Loops> {
    use std::io::Cursor;

    use gif::DecodeOptions;
    use gif::Repeat;

    let decoder = DecodeOptions::new().read_info(Cursor::new(bytes))?;
    Ok(match decoder.repeat() {
        Repeat::Infinite => Loops::Infinite,
        Repeat::Finite(0) => Loops::Finite(1),
        Repeat::Finite(repeats) => Loops::Finite(u32::from(repeats) + 1),
    })
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
    use crate::image::pixels::Pixels;
    use crate::testkit;
    use image::codecs::gif::Repeat;

    fn first_pixel(image: &Image) -> [u8; 4] {
        let Pixels::Rgba8(bytes) = image.pixels() else {
            panic!("a gif frame decodes to rgba");
        };
        let mut pixel = [0; 4];
        pixel.copy_from_slice(bytes.get(..4).expect("one pixel"));
        pixel
    }

    #[test]
    fn still_vs_animated_gif_detection() {
        let still = testkit::tiny_gif(2, 2, [10, 20, 30]);
        let decoded_still =
            codec::decode(&still, "still.gif").expect("still gif decodes");
        assert!(!decoded_still.animated());

        let animated = testkit::tiny_animated_gif(
            2,
            2,
            &[[255, 0, 0, 255], [0, 255, 0, 255]],
            50,
            Repeat::Infinite,
        );
        let decoded_anim = codec::decode(&animated, "anim.gif")
            .expect("anim gif decodes");
        assert!(decoded_anim.animated());
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
        let mut animation = open_animation(Arc::from(bytes))
            .expect("animated gif decodes");
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
        let still = testkit::tiny_gif(2, 2, [10, 20, 30]);
        let mut once =
            open_animation(Arc::from(still)).expect("still gif decodes");
        assert_eq!(once.loops(), Loops::Finite(1));
        assert!(once.next_frame().is_some());
        assert!(once.next_frame().is_none());

        let finite = testkit::tiny_animated_gif(
            2,
            2,
            &[[1, 2, 3, 255], [4, 5, 6, 255]],
            100,
            Repeat::Finite(2),
        );
        let animation =
            open_animation(Arc::from(finite)).expect("finite gif decodes");
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
        let mut animation =
            open_animation(Arc::from(bytes)).expect("gif decodes");
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

    #[test]
    fn oversized_gif_frame_is_rejected_by_limits() {
        let bytes = vec![
            b'G', b'I', b'F', b'8', b'9', b'a', // header
            0x01, 0x00, 0x01, 0x00, // 1x1 logical screen
            0x80, 0x00, 0x00, // GCT flag, 2 colors
            0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, // 2 RGB colors
            0x2C, // image separator
            0x00, 0x00, 0x00, 0x00, // left, top = 0, 0
            0x20, 0x4E, 0x20,
            0x4E, // width, height = 20000, 20000 (LE)
            0x00, // local color table = 0
            0x02, 0x00, // LZW min code size 2, terminator
            0x3B, // trailer
        ];
        let mut animation =
            open_animation(Arc::from(bytes)).expect("header opens");
        let frame = animation
            .next_frame()
            .expect("oversized frame reports error");
        assert!(frame.is_err(), "oversized frame must hit memory limit");
    }
}
