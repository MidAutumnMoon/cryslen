use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use eros::ErrorUnion;
use image::codecs::gif::Repeat;

use crate::fault::DecodeRefusal;
use crate::fault::PageError;
use crate::source::Source;

fn encode(
    img: &image::DynamicImage,
    format: image::ImageFormat,
) -> Vec<u8> {
    use std::io::Cursor;

    let mut buf = Vec::new();
    img.write_to(&mut Cursor::new(&mut buf), format)
        .expect("encode image");
    buf
}

pub fn tiny_png(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
    let img =
        image::RgbaImage::from_pixel(width, height, image::Rgba(rgba));
    encode(
        &image::DynamicImage::ImageRgba8(img),
        image::ImageFormat::Png,
    )
}

pub fn tiny_jpeg(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let img = image::RgbImage::from_pixel(width, height, image::Rgb(rgb));
    encode(
        &image::DynamicImage::ImageRgb8(img),
        image::ImageFormat::Jpeg,
    )
}

pub fn tiny_gif(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let img = image::RgbImage::from_pixel(width, height, image::Rgb(rgb));
    encode(
        &image::DynamicImage::ImageRgb8(img),
        image::ImageFormat::Gif,
    )
}

/// A GIF whose frames are solid colors, all with the same delay.
pub fn tiny_animated_gif(
    width: u32,
    height: u32,
    colors: &[[u8; 4]],
    delay_ms: u32,
    repeat: Repeat,
) -> Vec<u8> {
    use image::codecs::gif::GifEncoder;

    let mut bytes = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut bytes);
        encoder.set_repeat(repeat).expect("repeat is accepted");
        for &color in colors {
            let image = image::RgbaImage::from_pixel(
                width,
                height,
                image::Rgba(color),
            );
            let frame = image::Frame::from_parts(
                image,
                0,
                0,
                image::Delay::from_numer_denom_ms(delay_ms, 1),
            );
            encoder.encode_frame(frame).expect("frame encodes");
        }
    }
    bytes
}

pub fn tiny_bmp(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let img = image::RgbImage::from_pixel(width, height, image::Rgb(rgb));
    encode(
        &image::DynamicImage::ImageRgb8(img),
        image::ImageFormat::Bmp,
    )
}

pub fn tiny_tiff(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let img = image::RgbImage::from_pixel(width, height, image::Rgb(rgb));
    encode(
        &image::DynamicImage::ImageRgb8(img),
        image::ImageFormat::Tiff,
    )
}

pub fn failed_page(message: &'static str) -> Arc<PageError> {
    Arc::new(ErrorUnion::new(DecodeRefusal::Corrupt(message)))
}

/// Materialize the source and return one page's bytes; panics if that
/// page fails.
pub fn readable_page(source: &Source, index: usize) -> Arc<[u8]> {
    source.wait_ready().expect("source materialized");
    match source.bytes(index) {
        Ok(bytes) => bytes,
        Err(error) => panic!("page {index} failed: {error}"),
    }
}

/// Materialize the source and return one page's failure; panics if that
/// page reads.
pub fn unreadable_page(source: &Source, index: usize) -> Arc<PageError> {
    source.wait_ready().expect("source materialized");
    match source.bytes(index) {
        Err(error) => error,
        Ok(_) => panic!("page {index} materialized"),
    }
}

/// Poll `probe` until it returns `Some`, or `secs` seconds pass.
///
/// # Panics
///
/// Panics with `what` on timeout.
pub fn eventually<T>(
    secs: u64,
    what: &str,
    mut probe: impl FnMut() -> Option<T>,
) -> T {
    use std::thread;
    use std::time::Duration;
    use std::time::Instant;

    let deadline = Instant::now() + Duration::from_secs(secs);
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

/// Unique temp dir, removed on drop (tests avoid a tempfile dep).
pub struct TempDir(PathBuf);
static DIRS: AtomicU64 = AtomicU64::new(0);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        use std::env;
        use std::process;

        let path = env::temp_dir().join(format!(
            "cryslen-{}-{}-{}",
            tag,
            process::id(),
            DIRS.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir_all(&path).expect("create temp dir");
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn write(&self, name: &str, bytes: &[u8]) {
        fs::write(self.0.join(name), bytes).expect("write temp file");
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // A failed removal is a leaked scratch dir, never a test failure.
        let Ok(()) = fs::remove_dir_all(&self.0) else {
            return;
        };
    }
}
