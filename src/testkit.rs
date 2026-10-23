//! Scratch inputs shared by the source/decode/worker unit tests.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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
        let _ = fs::remove_dir_all(&self.0);
    }
}
