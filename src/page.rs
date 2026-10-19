//! Page model: a directory or archive viewed as a natural-sorted list of pages,
//! plus the pure spread-pairing math (cover alone, then pairs).
//!
//!
//! The spread functions are pure and unit-tested; the `Source` owns persistent
//! archive handles so zip files are not reopened per page.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::decode::format_from_name;

/// Where pages come from; decided by `Source::open` from one path argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceKind {
    Directory(PathBuf),
    Zip(PathBuf),
    SevenZip(PathBuf),
}

#[derive(Debug)]
pub enum OpenError {
    NotFound(PathBuf),
    Empty,
    Unsupported(PathBuf),
    Zip(String),
    SevenZip(String),
    Io(String),
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenError::NotFound(p) => write!(f, "not found: {}", p.display()),
            OpenError::Empty => write!(f, "no image pages found"),
            OpenError::Unsupported(p) => write!(f, "unsupported source: {}", p.display()),
            OpenError::Zip(e) => write!(f, "zip: {e}"),
            OpenError::SevenZip(e) => write!(f, "7z: {e}"),
            OpenError::Io(e) => write!(f, "io: {e}"),
        }
    }
}

/// A page source with persistent handles. Cheap to keep alive for the whole
/// session; shared with the decode worker behind an `Arc`.
pub struct Source {
    pub kind: SourceKind,
    names: Vec<String>,
    zip: Option<Mutex<zip::ZipArchive<std::io::BufReader<std::fs::File>>>>,
}

impl Source {
    /// `path` is a directory, or `.zip`/`.cbz`/`.7z`/`.cb7` archive. Listing is
    /// filtered to image extensions and natural-sorted (natord).
    pub fn open(path: &Path) -> Result<Source, OpenError> {
        if std::fs::metadata(path).map_err(|_| OpenError::NotFound(path.to_path_buf()))?.is_dir() {
            return Source::open_dir(path);
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "zip" | "cbz" => Source::open_zip(path),
            "7z" | "cb7" => Source::open_sevenzip(path),
            _ => Err(OpenError::Unsupported(path.to_path_buf())),
        }
    }

    fn open_dir(dir: &Path) -> Result<Source, OpenError> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .map_err(|e| OpenError::Io(format!("{}: {e}", dir.display())))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_ok_and(|t| t.is_file()))
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| format_from_name(name).is_some())
            .collect();
        sort_pages(&mut names);
        Source::from_names(SourceKind::Directory(dir.to_path_buf()), names)
    }

    /// The archive handle opened here stays open for the session; `bytes`
    /// reads go through it (the prototype's per-page reopen cost a central
    /// directory parse each time).
    fn open_zip(path: &Path) -> Result<Source, OpenError> {
        let file = std::fs::File::open(path).map_err(|e| OpenError::Zip(e.to_string()))?;
        let zip = zip::ZipArchive::new(std::io::BufReader::new(file))
            .map_err(|e| OpenError::Zip(e.to_string()))?;
        // Non-UTF-8 entry names are skipped (zip 9 reports them as errors).
        let mut names: Vec<String> = zip
            .file_names()
            .filter_map(|name| name.ok())
            .filter(|name| format_from_name(name).is_some())
            .map(|name| name.into_owned())
            .collect();
        sort_pages(&mut names);
        let mut source =
            Source::from_names(SourceKind::Zip(path.to_path_buf()), names)?;
        source.zip = Some(Mutex::new(zip));
        Ok(source)
    }

    /// Listing needs one archive walk; `bytes` reopens per read —
    /// sevenz-rust2's `ArchiveReader` owns its file and there is no cheap
    /// handle to keep here.
    fn open_sevenzip(path: &Path) -> Result<Source, OpenError> {
        let archive = sevenz_rust2::ArchiveReader::open(path, sevenz_rust2::Password::empty())
            .map_err(|e| OpenError::SevenZip(e.to_string()))?;
        let mut names: Vec<String> = archive
            .archive()
            .files
            .iter()
            .filter(|entry| !entry.is_directory && format_from_name(&entry.name).is_some())
            .map(|entry| entry.name.clone())
            .collect();
        drop(archive);
        sort_pages(&mut names);
        Source::from_names(SourceKind::SevenZip(path.to_path_buf()), names)
    }

    fn from_names(kind: SourceKind, names: Vec<String>) -> Result<Source, OpenError> {
        if names.is_empty() {
            return Err(OpenError::Empty);
        }
        Ok(Source { kind, names, zip: None })
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn name(&self, index: usize) -> &str {
        &self.names[index]
    }

    /// Short source label for the status line, e.g. "dir", "cbz", "cb7".
    pub fn label(&self) -> &'static str {
        match self.kind {
            SourceKind::Directory(_) => "dir",
            SourceKind::Zip(_) => "cbz",
            SourceKind::SevenZip(_) => "cb7",
        }
    }

    /// Raw encoded bytes of one page. Zip reads go through the persistent
    /// archive handle; 7z reopens per read (sevenz-rust2 limitation).
    pub fn bytes(&self, index: usize) -> Result<Box<[u8]>, OpenError> {
        let name = self
            .names
            .get(index)
            .ok_or_else(|| OpenError::Io(format!("page {index} out of range")))?;
        match &self.kind {
            SourceKind::Directory(dir) => std::fs::read(dir.join(name))
                .map(Vec::into_boxed_slice)
                .map_err(|e| OpenError::Io(e.to_string())),
            SourceKind::Zip(_) => {
                let zip = self
                    .zip
                    .as_ref()
                    .ok_or_else(|| OpenError::Zip("archive handle lost".to_string()))?;
                // A poisoned lock only means a read panicked mid-flight; the
                // archive itself is still readable, so recover the data.
                let mut archive = match zip.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
                let mut entry = archive.by_name(name).map_err(|e| OpenError::Zip(e.to_string()))?;
                let mut buf = Vec::new();
                entry.read_to_end(&mut buf).map_err(|e| OpenError::Zip(e.to_string()))?;
                Ok(buf.into_boxed_slice())
            }
            SourceKind::SevenZip(path) => {
                let mut archive = sevenz_rust2::ArchiveReader::open(path, sevenz_rust2::Password::empty())
                    .map_err(|e| OpenError::SevenZip(e.to_string()))?;
                archive
                    .read_file(name)
                    .map_err(|e| OpenError::SevenZip(e.to_string()))
                    .map(Into::into)
            }
        }
    }
}

/// Pages read in filename order for humans: c2 before c10.
fn sort_pages(names: &mut [String]) {
    names.sort_by(|a, b| natord::compare(a, b));
}

/// Which pages one screen shows. Cover is alone; afterwards pairs:
/// 0-based: `0`, `1+2`, `3+4`, ...
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spread {
    pub first: usize,
    pub second: Option<usize>,
}

/// Spread containing `index` when double-page mode is on/off.
pub fn spread_for(index: usize, double: bool) -> Spread {
    if !double {
        return Spread { first: index, second: None };
    }
    if index == 0 {
        return Spread { first: 0, second: None };
    }
    let first = 1 + ((index - 1) / 2) * 2;
    Spread { first, second: Some(first + 1) }
}

/// Next/previous spread start from `current`. Clamps at the ends.
pub fn advance(current: usize, forward: bool, double: bool, len: usize) -> usize {
    let step = if double { 2 } else { 1 };
    let next = if forward { current.saturating_add(step) } else { current.saturating_sub(step) };
    next.min(len.saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::testkit::{tiny_png, TempDir};

    use std::io::Write as _;

    /// LZMA-compressed 7z with two small PNGs (built by 7zz; the crate's
    /// encoder features are trimmed, so the fixture is checked in).
    const SEVENZ_FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/two_pages.7z"
    );

    #[test]
    fn sevenz_listing_and_read() {
        let source = Source::open(std::path::Path::new(SEVENZ_FIXTURE)).expect("7z opens");
        assert_eq!(source.len(), 2);
        assert_eq!(source.label(), "cb7");
        assert_eq!(source.names, ["p1.png", "p2.png"]);
        let bytes = source.bytes(0).expect("first page reads");
        // It is the PNG we packed (magic; IHDR big-endian dims 40x30).
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let w = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
        let h = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
        assert_eq!((w, h), (40, 30));
        let decoded = crate::decode::decode(&bytes, "p1.png").expect("decodes");
        assert_eq!((decoded.width, decoded.height), (40, 30));
    }

    #[test]
    fn dir_listing_natural_order_and_filter() {
        let dir = TempDir::new("pagedir");
        dir.write("c10.png", &tiny_png(1, 1, [0, 0, 0, 255]));
        dir.write("c2.png", &tiny_png(1, 1, [0, 0, 0, 255]));
        dir.write("c1.png", &tiny_png(1, 1, [0, 0, 0, 255]));
        dir.write("cover.PNG", &tiny_png(1, 1, [0, 0, 0, 255]));
        dir.write("notes.txt", b"not a page");
        let source = Source::open(dir.path()).expect("dir opens");
        assert_eq!(source.len(), 4);
        assert_eq!(source.label(), "dir");
        // c2 sorts before c10 (natord), extension filter drops notes.txt,
        // uppercase .PNG still counts as an image.
        assert_eq!(
            source.names,
            ["c1.png", "c2.png", "c10.png", "cover.PNG"]
        );
        assert_eq!(source.name(2), "c10.png");
    }

    #[test]
    fn dir_bytes_roundtrip() {
        let dir = TempDir::new("pagebytes");
        let png = tiny_png(3, 2, [9, 8, 7, 255]);
        dir.write("p.png", &png);
        let source = Source::open(dir.path()).unwrap();
        assert_eq!(source.bytes(0).as_deref().ok(), Some(png.as_slice()));
        assert!(source.bytes(1).is_err());
    }

    #[test]
    fn zip_listing_and_read() {
        let dir = TempDir::new("pagezip");
        let zip_path = dir.path().join("book.cbz");
        {
            let file = std::fs::File::create(&zip_path).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            for (name, color) in [("b/010.png", 1u8), ("b/002.png", 2), ("b/020.png", 3)] {
                zip.start_file(name, options).unwrap();
                zip.write_all(&tiny_png(2, 2, [color, 0, 0, 255])).unwrap();
            }
            zip.start_file("b/info.txt", options).unwrap();
            zip.write_all(b"skip me").unwrap();
            zip.finish().unwrap();
        }
        let source = Source::open(&zip_path).expect("zip opens");
        assert_eq!(source.label(), "cbz");
        assert_eq!(source.names, ["b/002.png", "b/010.png", "b/020.png"]);
        let page = source.bytes(1).expect("entry readable");
        let decoded = crate::decode::decode(&page, "b/010.png").unwrap();
        assert_eq!(decoded.width, 2);
    }

    #[test]
    fn open_errors() {
        assert!(matches!(
            Source::open(Path::new("/nonexistent-cryslen-dir")),
            Err(OpenError::NotFound(_))
        ));
        let empty = TempDir::new("pageempty");
        assert!(matches!(Source::open(empty.path()), Err(OpenError::Empty)));
        let dir = TempDir::new("pageunsup");
        dir.write("book.rar", b"not really a rar");
        assert!(matches!(
            Source::open(&dir.path().join("book.rar")),
            Err(OpenError::Unsupported(_))
        ));
    }

    #[test]
    fn cover_alone_then_pairs() {
        assert_eq!(spread_for(0, true), Spread { first: 0, second: None });
        assert_eq!(spread_for(1, true), Spread { first: 1, second: Some(2) });
        assert_eq!(spread_for(2, true), Spread { first: 1, second: Some(2) });
        assert_eq!(spread_for(3, true), Spread { first: 3, second: Some(4) });
        assert_eq!(spread_for(0, false), Spread { first: 0, second: None });
        assert_eq!(spread_for(5, false), Spread { first: 5, second: None });
    }

    #[test]
    fn odd_tail() {
        // 7 pages: index 6 lives in pair (5, 6).
        assert_eq!(spread_for(6, true), Spread { first: 5, second: Some(6) });
        // pair (5, 6) with 6 pages: layout skips the out-of-range slot.
        assert_eq!(spread_for(5, true).first, 5);
    }

    #[test]
    fn advance_steps_and_clamps() {
        assert_eq!(advance(1, true, true, 10), 3);
        assert_eq!(advance(3, false, true, 10), 1);
        assert_eq!(advance(8, true, true, 9), 8);
        assert_eq!(advance(1, false, true, 10), 0);
        assert_eq!(advance(4, true, false, 10), 5);
    }
}
