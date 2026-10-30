//! Page source: a directory or archive viewed as a natural-sorted list
//! of pages. An archive unpacks once into memory on a background
//! thread while `bytes` serves whatever the pass has filled, so no
//! read blocks and no member is ever unpacked twice. A directory keeps
//! reading files, which the OS page cache serves.

use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::io::Error as IoError;
use std::io::ErrorKind;
use std::io::Read;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::Builder;

use parking_lot::Mutex;
use zip::result::ZipError;

use crate::decode::format_from_name;

#[derive(Debug)]
pub enum SourceError {
    NotFound(PathBuf),
    Empty,
    Unsupported(PathBuf),
    /// The archive's pages unpack to more bytes than the machine has
    /// available.
    TooBig { unpacked: u64, available: u64 },
    Zip(ZipError),
    SevenZip(sevenz_rust2::Error),
    Io(IoError),
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(path) => write!(f, "not found: {}", path.display()),
            Self::Empty => write!(f, "no image pages found"),
            Self::Unsupported(path) => write!(f, "unsupported source: {}", path.display()),
            Self::TooBig { unpacked, available } => {
                const MIB: u64 = 1024 * 1024;
                write!(
                    f,
                    "unpacked pages are {} MiB but only {} MiB of memory is available",
                    unpacked / MIB,
                    available / MIB
                )
            }
            Self::Zip(error) => write!(f, "zip: {error}"),
            Self::SevenZip(error) => write!(f, "7z: {error}"),
            Self::Io(error) => write!(f, "io: {error}"),
        }
    }
}

impl Error for SourceError {}

/// What the user asked the current page's name segment to be called.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageName {
    /// Directory shown before the basename, e.g. a volume inside an archive.
    /// Flat archives and anonymous paths have none.
    pub parent: Option<String>,
    /// The basename with its extension, e.g. `image001.avif`.
    pub file: String,
}

/// Which archive family a path's extension names. Both `Source::open` and
/// main's argument classification dispatch on this one table.
enum ArchiveKind {
    Zip,
    SevenZip,
}

fn archive_kind(path: &Path) -> Option<ArchiveKind> {
    let ext = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "zip" | "cbz" => Some(ArchiveKind::Zip),
        "7z" | "cb7" => Some(ArchiveKind::SevenZip),
        _ => None,
    }
}

#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is the crate's visibility style for crate-wide items, even inside private modules"
)]
pub(crate) fn is_archive(path: &Path) -> bool {
    archive_kind(path).is_some()
}

/// The answer to one page read. It never blocks: `NotYet` reports an
/// archive still unpacking, and the caller decides how to wait.
pub enum Bytes {
    Ready(Arc<[u8]>),
    NotYet,
    Failed(String),
}

/// The unpacked pages one background pass fills. Every page ends with
/// its own result — bytes or an error — and a pass that ends early
/// names why, which every unfilled page reads as. Zip members decode
/// independently, so one bad member settles only its page; a solid 7z
/// stream breaks at its first error, so its unfilled tail shares the
/// reason while the filled prefix stays readable.
struct Fill {
    pages: HashMap<usize, Result<Arc<[u8]>, String>>,
    /// `None` while the pass runs; `Some(Ok(()))` when it ended whole;
    /// `Some(Err(reason))` when it ended early.
    pass: Option<Result<(), String>>,
}

impl Fill {
    fn empty() -> Self {
        Self { pages: HashMap::new(), pass: None }
    }

    /// One page's answer under the lock.
    fn resolve(&self, index: usize) -> Bytes {
        match self.pages.get(&index) {
            Some(Ok(bytes)) => Bytes::Ready(Arc::clone(bytes)),
            Some(Err(message)) => Bytes::Failed(message.clone()),
            None => match &self.pass {
                None => Bytes::NotYet,
                Some(Ok(())) => {
                    Bytes::Failed(format!("page {}: no entry in the archive", index + 1))
                }
                Some(Err(fatal)) => Bytes::Failed(fatal.clone()),
            },
        }
    }
}

/// Where pages come from, decided by `Source::open` from one path
/// argument. Archives share one shape: a background pass unpacks the
/// pages into memory. A directory keeps reading its files.
enum Backend {
    Directory(PathBuf),
    Archive(Arc<Mutex<Fill>>),
}

pub struct Source {
    names: Vec<String>,
    backend: Backend,
}

impl Source {
    /// Listing is filtered to image extensions and natural-sorted.
    pub fn open(path: &Path) -> Result<Self, SourceError> {
        let metadata = fs::metadata(path).map_err(|error| {
            if error.kind() == ErrorKind::NotFound {
                SourceError::NotFound(path.to_path_buf())
            } else {
                SourceError::Io(error)
            }
        })?;
        if metadata.is_dir() {
            return Self::open_dir(path);
        }
        match archive_kind(path) {
            Some(ArchiveKind::Zip) => Self::open_zip(path),
            Some(ArchiveKind::SevenZip) => Self::open_sevenzip(path),
            None => Err(SourceError::Unsupported(path.to_path_buf())),
        }
    }

    fn open_dir(dir: &Path) -> Result<Self, SourceError> {
        #[expect(
            clippy::filetype_is_file,
            reason = "directory pages are regular files, so symlinks are skipped"
        )]
        let mut names: Vec<String> = fs::read_dir(dir)
            .map_err(SourceError::Io)?
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|file_type| file_type.is_file()))
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| format_from_name(name).is_some())
            .collect();
        sort_pages(&mut names);
        Self::from_names(Backend::Directory(dir.to_path_buf()), names)
    }

    /// Raw entries carry name and size without building a decoder, so
    /// listing never trips on a member whose method this build cannot
    /// decode — that member settles as its own page's error at fill
    /// time.
    fn open_zip(path: &Path) -> Result<Self, SourceError> {
        let file = File::open(path).map_err(SourceError::Io)?;
        let mut zip = zip::ZipArchive::new(BufReader::new(file)).map_err(SourceError::Zip)?;
        let mut listed: Vec<(String, u64)> = Vec::new();
        for entry_index in 0..zip.len() {
            let entry = zip.by_index_raw(entry_index).map_err(SourceError::Zip)?;
            if entry.is_dir() {
                continue;
            }
            let name = entry.name().map_err(SourceError::Zip)?.into_owned();
            if format_from_name(&name).is_some() {
                listed.push((name, entry.size()));
            }
        }
        listed.sort_by(|left, right| natord::compare(&left.0, &right.0));
        // The empty check runs before the spawn, so a pageless archive
        // starts no fill thread.
        if listed.is_empty() {
            return Err(SourceError::Empty);
        }
        let total: u64 = listed.iter().map(|(_, size)| size).sum();
        let names: Vec<String> = listed.into_iter().map(|(name, _)| name).collect();
        guard_memory(total, mem_available())?;
        let fill_state = Arc::new(Mutex::new(Fill::empty()));
        spawn_zip_fill(zip, names.clone(), Arc::clone(&fill_state));
        Self::from_names(Backend::Archive(fill_state), names)
    }

    fn open_sevenzip(path: &Path) -> Result<Self, SourceError> {
        let archive = sevenz_rust2::ArchiveReader::open(path, sevenz_rust2::Password::empty())
            .map_err(SourceError::SevenZip)?;
        let mut listed: Vec<(String, u64)> = archive
            .archive()
            .files
            .iter()
            .filter(|entry| !entry.is_directory() && format_from_name(&entry.name).is_some())
            .map(|entry| (entry.name.clone(), entry.size()))
            .collect();
        listed.sort_by(|left, right| natord::compare(&left.0, &right.0));
        // The empty check runs before the spawn, so a pageless archive
        // starts no fill thread.
        if listed.is_empty() {
            return Err(SourceError::Empty);
        }
        let total: u64 = listed.iter().map(|(_, size)| size).sum();
        let names: Vec<String> = listed.into_iter().map(|(name, _)| name).collect();
        guard_memory(total, mem_available())?;
        let by_name: HashMap<String, usize> = names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index))
            .collect();
        let fill_state = Arc::new(Mutex::new(Fill::empty()));
        spawn_sevenzip_fill(archive, by_name, Arc::clone(&fill_state));
        Self::from_names(Backend::Archive(fill_state), names)
    }

    fn from_names(backend: Backend, names: Vec<String>) -> Result<Self, SourceError> {
        if names.is_empty() {
            return Err(SourceError::Empty);
        }
        Ok(Self { names, backend })
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Entry name of one page. For archives this includes the inner
    /// path; for directories it is the bare filename.
    pub fn name(&self, index: usize) -> Option<&str> {
        self.names.get(index).map(String::as_str)
    }

    /// The status line's name segment for one page: the basename plus its
    /// immediate parent directory. Archive entries carry their inner
    /// directory; directory pages get the opened directory itself.
    #[expect(
        clippy::indexing_slicing,
        reason = "the page list is non-empty and immutable, and every page index is clamped to its length before reaching here"
    )]
    pub fn page_name(&self, index: usize) -> PageName {
        debug_assert!(index < self.names.len(), "caller clamps the page index");
        let entry = &self.names[index];
        let file = entry.rsplit('/').next().unwrap_or(entry).to_owned();
        let parent = match &self.backend {
            Backend::Directory(dir) => {
                dir.file_name().map(|name| name.to_string_lossy().into_owned())
            }
            Backend::Archive(_) => {
                entry.rsplit_once('/').map(|(path, _)| path.rsplit('/').next().unwrap_or(path).to_owned())
            }
        };
        PageName { parent, file }
    }

    /// The encoded bytes of one page, never blocking: an archive still
    /// unpacking answers `NotYet`, and the caller decides how to wait.
    pub fn bytes(&self, index: usize) -> Bytes {
        let Some(name) = self.names.get(index) else {
            return Bytes::Failed(format!("page {}: no such page", index + 1));
        };
        match &self.backend {
            Backend::Directory(dir) => match fs::read(dir.join(name)) {
                Ok(bytes) => Bytes::Ready(bytes.into()),
                Err(error) => Bytes::Failed(error.to_string()),
            },
            Backend::Archive(fill) => fill.lock().resolve(index),
        }
    }

    /// Whether the unpack pass is still running, for the status line's
    /// wait word. A directory is never filling.
    pub fn filling(&self) -> bool {
        match &self.backend {
            Backend::Directory(_) => false,
            Backend::Archive(fill) => fill.lock().pass.is_none(),
        }
    }
}

/// One member's bytes, with its read error as the message.
fn read_member(member: &mut impl Read) -> Result<Arc<[u8]>, String> {
    let mut bytes = Vec::new();
    member.read_to_end(&mut bytes).map_err(|error| error.to_string())?;
    Ok(bytes.into())
}

/// Fill a zip's pages in reading order. Zip members decode
/// independently, so each page settles on its own result — one bad
/// member fails only its page — and the pass always reaches its end.
fn spawn_zip_fill(
    mut zip: zip::ZipArchive<BufReader<File>>,
    names: Vec<String>,
    fill: Arc<Mutex<Fill>>,
) {
    // Dropping the handle detaches the thread, so quitting never waits
    // on it.
    drop(Builder::new()
        .name("cy-unpack".into())
        .spawn(move || {
            let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
                for (index, name) in names.iter().enumerate() {
                    let page = match zip.by_name(name) {
                        Ok(mut entry) => read_member(&mut entry),
                        Err(error) => Err(error.to_string()),
                    };
                    fill.lock().pages.insert(index, page);
                }
            }));
            let mut state = fill.lock();
            state.pass = Some(match outcome {
                Ok(()) => Ok(()),
                Err(panic) => Err(panic_message(panic.as_ref())),
            });
        }));
}

/// Fill a 7z's pages in storage order, the only order a solid stream
/// permits: every member is consumed — page or not — before the next
/// decodes. One error ends the pass, so the unfilled tail reads as
/// that error while the filled prefix stays readable.
fn spawn_sevenzip_fill(
    mut reader: sevenz_rust2::ArchiveReader<File>,
    pages: HashMap<String, usize>,
    fill: Arc<Mutex<Fill>>,
) {
    drop(Builder::new()
        .name("cy-unpack".into())
        .spawn(move || {
            let outcome = panic::catch_unwind(AssertUnwindSafe(|| reader.for_each_entries(
                |entry, data| {
                    let mut member = Vec::new();
                    data.read_to_end(&mut member)?;
                    if let Some(&index) = pages.get(&entry.name) {
                        fill.lock()
                            .pages
                            .insert(index, Ok(Arc::from(member.as_slice())));
                    }
                    Ok(true)
                },
            )));
            let mut state = fill.lock();
            state.pass = Some(match outcome {
                Ok(Ok(())) => Ok(()),
                Ok(Err(error)) => {
                    tracing::warn!(%error, "7z unpack ended early");
                    Err(error.to_string())
                }
                Err(panic) => Err(panic_message(panic.as_ref())),
            });
        }));
}

/// A panic payload rendered as a failure message, for the thread panic
/// boundaries: the unpack pass here and the decoder in the worker.
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is the crate's visibility style for crate-wide items, even inside private modules"
)]
pub(crate) fn panic_message(panic: &(dyn Any + Send)) -> String {
    if let Some(text) = panic.downcast_ref::<&str>() {
        return format!("panicked: {text}");
    }
    if let Some(text) = panic.downcast_ref::<String>() {
        return format!("panicked: {text}");
    }
    String::from("panicked")
}

/// Kibibytes available, from a `/proc/meminfo` text. `None` when the
/// file or line is missing.
fn parse_mem_available(info: &str) -> Option<u64> {
    let line = info.lines().find(|line| line.starts_with("MemAvailable:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(kib * 1024)
}

/// Bytes of memory the fill may hold, or `None` when the host reports
/// none, which skips the check.
fn mem_available() -> Option<u64> {
    parse_mem_available(&fs::read_to_string("/proc/meminfo").ok()?)
}

/// Refuse an archive whose pages cannot all stay in memory at once.
/// The error names both numbers; a host reporting no availability is
/// not guarded.
fn guard_memory(unpacked: u64, available: Option<u64>) -> Result<(), SourceError> {
    match available {
        Some(available) if unpacked > available => {
            Err(SourceError::TooBig { unpacked, available })
        }
        _ => Ok(()),
    }
}

/// Pages read in filename order for humans: c2 before c10.
fn sort_pages(names: &mut [String]) {
    names.sort_by(|left, right| natord::compare(left, right));
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;
    use crate::decode;
    use crate::testkit::{eventually, tiny_png, TempDir};

    use std::io::Write as _;

    /// LZMA-compressed 7z with two small PNGs, built by 7zz. The crate's
    /// encoder features are trimmed, so the fixture is checked in.
    const SEVENZ_FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/two_pages.7z"
    );

    /// `Fill` resolution: per-page results while running, the pass's
    /// outcome for the rest once it ends.
    #[test]
    fn fill_resolution() {
        let mut fill = Fill::empty();
        assert!(matches!(fill.resolve(0), Bytes::NotYet));
        fill.pages.insert(0, Ok(Arc::from(&b"a"[..])));
        fill.pages.insert(1, Err("bad page".into()));
        assert!(matches!(fill.resolve(0), Bytes::Ready(_)));
        assert!(matches!(fill.resolve(1), Bytes::Failed(message) if message == "bad page"));
        assert!(matches!(fill.resolve(2), Bytes::NotYet));

        fill.pass = Some(Ok(()));
        assert!(matches!(fill.resolve(2), Bytes::Failed(message) if message.contains("no entry")));
        fill.pass = Some(Err("stream broke".into()));
        assert!(matches!(fill.resolve(3), Bytes::Failed(message) if message == "stream broke"));
        // Settled pages keep their own results after the pass ended.
        assert!(matches!(fill.resolve(1), Bytes::Failed(message) if message == "bad page"));
    }

    /// `/proc/meminfo` parsing: the `MemAvailable` line, and nothing else.
    #[test]
    fn meminfo_parsing() {
        let info = "MemTotal:       100 kB\nMemAvailable:   4096 kB\nSwapTotal:      2 kB\n";
        assert_eq!(parse_mem_available(info), Some(4096 * 1024));
        assert_eq!(parse_mem_available("MemTotal: 1 kB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable: notanumber kB\n"), None);
    }

    /// The memory guard refuses only an archive bigger than what the
    /// host reports available; a host reporting nothing is not guarded.
    #[test]
    fn memory_guard() {
        assert!(matches!(
            guard_memory(100, Some(50)),
            Err(SourceError::TooBig { unpacked: 100, available: 50 })
        ));
        assert!(matches!(
            guard_memory(50, Some(100)),
            Ok(())
        ));
        assert!(matches!(guard_memory(u64::MAX, None), Ok(())));
    }
    /// 7z sources list pages, read them through the fill, and the
    /// background pass settles every page.
    #[test]
    fn sevenz_source() {
        let source = Source::open(Path::new(SEVENZ_FIXTURE)).expect("7z opens");
        assert_eq!(source.names, ["p1.png", "p2.png"]);
        assert_eq!(
            source.page_name(0),
            PageName { parent: None, file: "p1.png".into() }
        );
        let bytes = eventually(20, "first page never unpacked", || match source.bytes(0) {
            Bytes::Ready(bytes) => Some(bytes),
            Bytes::Failed(message) => panic!("page 0 failed: {message}"),
            Bytes::NotYet => None,
        });
        // A repeat read returns the same bytes from the fill.
        assert!(matches!(source.bytes(0), Bytes::Ready(repeat) if repeat.as_ref() == bytes.as_ref()));
        assert_eq!(bytes.get(..8), Some(&b"\x89PNG\r\n\x1a\n"[..]));
        // The decoder is the oracle for the fixture's dims.
        let page = decode::decode(&bytes, "p1.png").expect("fixture page decodes");
        assert_eq!((page.width, page.height), (40, 30));

        // the pass settles both pages
        eventually(20, "pass never ended", || (!source.filling()).then_some(()));
        assert!(matches!(source.bytes(1), Bytes::Ready(_)));
    }

    /// Directory sources list pages natural-sorted and image-filtered, and read back written bytes.
    #[test]
    fn dir_source() {
        let dir = TempDir::new("pagedir");
        dir.write("c10.png", &tiny_png(1, 1, [0, 0, 0, 255]));
        dir.write("c2.png", &tiny_png(1, 1, [7, 8, 9, 255]));
        dir.write("c1.png", &tiny_png(1, 1, [0, 0, 0, 255]));
        dir.write("cover.PNG", &tiny_png(1, 1, [0, 0, 0, 255]));
        dir.write("notes.txt", b"not a page");
        let source = Source::open(dir.path()).expect("dir opens");
        // c2 before c10 (natural order), notes.txt filtered, uppercase
        // extension kept
        assert_eq!(source.names, ["c1.png", "c2.png", "c10.png", "cover.PNG"]);
        assert_eq!(source.len(), 4);
        assert_eq!(source.name(2), Some("c10.png"));
        assert_eq!(
            source.page_name(2),
            PageName {
                parent: Some(dir.path().file_name().unwrap().to_string_lossy().into_owned()),
                file: "c10.png".into(),
            }
        );
        // a directory is never filling; bytes reads back byte-identical
        assert!(!source.filling());
        assert!(matches!(source.bytes(1), Bytes::Ready(read) if read.as_ref() == tiny_png(1, 1, [7, 8, 9, 255]).as_slice()));
    }

    /// A zip source lists its image entries natural-sorted across inner directories, names pages by the immediate parent, and the fill reads an entry back byte-identical.
    #[test]
    fn zip_listing_and_read() {
        use zip::write::SimpleFileOptions;

        let dir = TempDir::new("pagezip");
        let zip_path = dir.path().join("book.cbz");
        {
            let file = File::create(&zip_path).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            for (name, color) in
                [("b/010.png", 1_u8), ("b/002.png", 2), ("b/020.png", 3), ("b/deep/005.png", 4)]
            {
                zip.start_file(name, options).unwrap();
                zip.write_all(&tiny_png(2, 2, [color, 0, 0, 255])).unwrap();
            }
            zip.start_file("b/info.txt", options).unwrap();
            zip.write_all(b"skip me").unwrap();
            zip.finish().unwrap();
        }
        let source = Source::open(&zip_path).expect("zip opens");
        assert_eq!(source.names, ["b/002.png", "b/010.png", "b/020.png", "b/deep/005.png"]);
        assert_eq!(source.page_name(1), PageName { parent: Some("b".into()), file: "010.png".into() });
        // A nested entry shows its immediate parent, not the whole path.
        assert_eq!(
            source.page_name(3),
            PageName { parent: Some("deep".into()), file: "005.png".into() }
        );
        // bytes(1) is b/010.png, written with color 1.
        let page = eventually(20, "zip page never unpacked", || match source.bytes(1) {
            Bytes::Ready(bytes) => Some(bytes),
            Bytes::Failed(message) => panic!("page 1 failed: {message}"),
            Bytes::NotYet => None,
        });
        assert_eq!(page.as_ref(), tiny_png(2, 2, [1, 0, 0, 255]).as_slice());
        eventually(20, "pass never ended", || (!source.filling()).then_some(()));
    }

    /// A corrupt zip member fails only its own page; the pass still
    /// settles every other page.
    #[test]
    fn zip_corrupt_member_fails_only_its_page() {
        use zip::write::SimpleFileOptions;

        let dir = TempDir::new("pagezipbad");
        let zip_path = dir.path().join("book.cbz");
        let pages = [tiny_png(2, 2, [1, 0, 0, 255]), tiny_png(2, 2, [2, 0, 0, 255]), tiny_png(2, 2, [3, 0, 0, 255])];
        {
            let file = File::create(&zip_path).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            for (index, page) in pages.iter().enumerate() {
                zip.start_file(format!("{index:03}.png"), options).unwrap();
                zip.write_all(page).unwrap();
            }
            zip.finish().unwrap();
        }
        // Flip one byte inside the middle page's stored payload: its
        // CRC no longer matches, so reading that member fails.
        {
            let mut book = fs::read(&zip_path).unwrap();
            let needle = &*pages[1];
            let at = book
                .windows(needle.len())
                .position(|window| window == needle)
                .expect("stored payload lies verbatim in the file");
            *book.get_mut(at).expect("offset points into the file") ^= 0xff;
            fs::write(&zip_path, &book).unwrap();
        }

        let source = Source::open(&zip_path).expect("zip opens");
        let failure = eventually(20, "corrupt page never settled", || match source.bytes(1) {
            Bytes::Failed(message) => Some(message),
            Bytes::Ready(_) => panic!("corrupt page unpacked"),
            Bytes::NotYet => None,
        });
        assert!(failure.to_lowercase().contains("checksum"), "{failure}");
        // The neighbors settle whole.
        assert_eq!(
            eventually(20, "page 0 never unpacked", || match source.bytes(0) {
                Bytes::Ready(bytes) => Some(bytes),
                Bytes::Failed(message) => panic!("page 0 failed: {message}"),
                Bytes::NotYet => None,
            })
            .as_ref(),
            pages[0].as_slice()
        );
        assert!(matches!(source.bytes(2), Bytes::Ready(read) if read.as_ref() == pages[2].as_slice()));
    }

    /// Every unusable source opens as its matching error: not found, empty, unsupported, io.
    #[test]
    fn open_errors() {
        assert!(matches!(
            Source::open(Path::new("/nonexistent-cryslen-dir")),
            Err(SourceError::NotFound(_))
        ));
        let empty = TempDir::new("pageempty");
        assert!(matches!(Source::open(empty.path()), Err(SourceError::Empty)));
        let dir = TempDir::new("pageunsup");
        dir.write("book.rar", b"not really a rar");
        assert!(matches!(
            Source::open(&dir.path().join("book.rar")),
            Err(SourceError::Unsupported(_))
        ));
        dir.write("plain.txt", b"not a directory");
        // A path under a regular file reports io, not NotFound.
        assert!(matches!(
            Source::open(&dir.path().join("plain.txt").join("page.png")),
            Err(SourceError::Io(_))
        ));
    }
}
