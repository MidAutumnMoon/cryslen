//! Scratch inputs shared by the source/decode/worker unit tests.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::decode::Decoded;
use crate::decode::Format;
use crate::worker::SeatView;

/// Solid-color PNG straight from the image encoder.
pub fn tiny_png(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
    use std::io::Cursor;

    let img = image::RgbaImage::from_pixel(width, height, image::Rgba(rgba));
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
        .expect("encode png");
    buf
}

/// Solid-color JPEG straight from the image encoder.
pub fn tiny_jpeg(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    use std::io::Cursor;

    let img = image::RgbImage::from_pixel(width, height, image::Rgb(rgb));
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Jpeg)
        .expect("encode jpeg");
    buf
}

/// A decided `Ready` seat holding a blank `width`x`height` page.
pub fn ready_seat(width: u32, height: u32) -> SeatView {
    use std::sync::Arc;

    SeatView::Ready(Arc::new(
        Decoded::rgba(
            width,
            height,
            Format::Png,
            vec![0_u8; width as usize * height as usize * 4].into_boxed_slice(),
        )
        .expect("valid rgba page"),
    ))
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
        let Ok(()) = fs::remove_dir_all(&self.0) else { return; };
    }
}
