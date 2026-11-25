use std::sync::Arc;

use bytes::Bytes;
use eros::ErrorUnion;
use image::DynamicImage;
use image::ImageFormat;
use image::Rgb;
use image::RgbImage;
use image::codecs::gif::Repeat;

use crate::protocol::ImageLimits;
use crate::source::Source;

pub const TEST_FIT_BOX: (u32, u32) = (2560, 1440);

pub const SEVENZ_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/two_pages.7z"
);

pub fn limits() -> ImageLimits {
    const MAX_TEXTURE_EDGE: u32 = 16_384;
    ImageLimits::new(TEST_FIT_BOX, MAX_TEXTURE_EDGE)
}

fn encode(img: &DynamicImage, format: ImageFormat) -> Vec<u8> {
    use std::io::Cursor;

    let mut buf = Vec::new();
    img.write_to(&mut Cursor::new(&mut buf), format)
        .expect("encode image");
    buf
}

pub fn tiny_png(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
    use image::Rgba;
    use image::RgbaImage;

    let img = RgbaImage::from_pixel(width, height, Rgba(rgba));
    encode(&DynamicImage::ImageRgba8(img), ImageFormat::Png)
}

pub fn tiny_jpeg(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let img = RgbImage::from_pixel(width, height, Rgb(rgb));
    encode(&DynamicImage::ImageRgb8(img), ImageFormat::Jpeg)
}

pub fn tiny_gif(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let img = RgbImage::from_pixel(width, height, Rgb(rgb));
    encode(&DynamicImage::ImageRgb8(img), ImageFormat::Gif)
}

/// A GIF whose frames are solid colors, all with the same delay.
pub fn tiny_animated_gif(
    width: u32,
    height: u32,
    colors: &[[u8; 4]],
    delay_ms: u32,
    repeat: Repeat,
) -> Vec<u8> {
    use image::Delay;
    use image::Frame;
    use image::Rgba;
    use image::RgbaImage;
    use image::codecs::gif::GifEncoder;

    let mut bytes = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut bytes);
        encoder.set_repeat(repeat).expect("repeat is accepted");
        for &color in colors {
            let image = RgbaImage::from_pixel(width, height, Rgba(color));
            let frame = Frame::from_parts(
                image,
                0,
                0,
                Delay::from_numer_denom_ms(delay_ms, 1),
            );
            encoder.encode_frame(frame).expect("frame encodes");
        }
    }
    bytes
}

pub fn tiny_bmp(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let img = RgbImage::from_pixel(width, height, Rgb(rgb));
    encode(&DynamicImage::ImageRgb8(img), ImageFormat::Bmp)
}

pub fn tiny_tiff(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let img = RgbImage::from_pixel(width, height, Rgb(rgb));
    encode(&DynamicImage::ImageRgb8(img), ImageFormat::Tiff)
}

pub fn readable_page(source: &Source, index: usize) -> Bytes {
    match source.bytes(index) {
        Ok(bytes) => bytes,
        Err(error) => panic!("page {index} failed: {error}"),
    }
}

pub fn unreadable_page(source: &Source, index: usize) -> Arc<ErrorUnion> {
    match source.bytes(index) {
        Err(error) => error,
        Ok(_) => panic!("page {index} succeeded"),
    }
}

/// Poll `probe` until it returns `Some`, or the test timeout passes.
///
/// # Panics
///
/// Panics with `what` on timeout.
pub fn eventually<Value>(
    what: &str,
    mut probe: impl FnMut() -> Option<Value>,
) -> Value {
    use std::thread;
    use std::time::Duration;
    use std::time::Instant;

    const EVENTUALLY_TIMEOUT: Duration = Duration::from_secs(20);

    let deadline = Instant::now() + EVENTUALLY_TIMEOUT;
    loop {
        if let Some(value) = probe() {
            return value;
        }
        if Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
            continue;
        }
        panic!("{what}");
    }
}
