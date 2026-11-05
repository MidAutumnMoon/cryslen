use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::decode::Decoded;
use crate::decode::Format;
use crate::decode::Transfer;
use crate::fault::PageError;
use crate::worker::SeatView;

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

pub fn ready_seat(width: u32, height: u32) -> SeatView {
    SeatView::Ready(Arc::new(
        Decoded::rgba(
            width,
            height,
            Format::Png,
            Transfer::Srgb,
            vec![0_u8; width as usize * height as usize * 4]
                .into_boxed_slice(),
        )
        .expect("valid rgba page"),
    ))
}

/// A page error carrying just `message`, for seeding failure cells.
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is the crate's visibility style for crate-wide items, even inside private modules"
)]
pub(crate) fn failed_page(message: &str) -> Arc<PageError> {
    Arc::new(eros::ErrorUnion::new(eros::MsgError::from_owned(
        message.to_owned(),
    )))
}

/// A seat failed with `message`, for seeding fold and seating tests.
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is the crate's visibility style for crate-wide items, even inside private modules"
)]
pub(crate) fn failed_seat(message: &str) -> SeatView {
    SeatView::Failed(failed_page(message))
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
