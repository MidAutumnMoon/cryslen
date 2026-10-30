//! Page source: a directory or archive viewed as a natural-sorted list
//! of pages. An archive unpacks once into memory on a background
//! thread while `bytes` serves whatever the pass has filled, so archive
//! reads never block and no member is ever unpacked twice. A directory
//! keeps reading files, which the OS page cache serves.

use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::fs::File;
use std::io;
use std::io::BufReader;
use std::io::Error as IoError;
use std::io::ErrorKind;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use zip::result::ZipError;

use crate::decode::format_from_name;

#[derive(Debug)]
pub enum SourceError {
    NotFound(PathBuf),
    Empty,
    Unsupported(PathBuf),
    /// The listing names one entry twice.
    Duplicate(String),
    /// The sizes the listing claims exceed the memory the machine
    /// reports available.
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
            Self::Duplicate(name) => write!(f, "duplicate entry name: {name}"),
            Self::TooBig { unpacked, available } => write!(
                f,
                "archive too big: {unpacked} bytes claimed, {available} bytes available"
            ),
            Self::Zip(error) => write!(f, "zip: {error}"),
            Self::SevenZip(error) => write!(f, "7z: {error}"),
            Self::Io(error) => write!(f, "io: {error}"),
        }
    }
}

impl Error for SourceError {}

/// What the status line calls the current page.
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

/// The answer to one page read. `NotYet` reports an archive still
/// unpacking; the read never waits for the unpack.
pub enum PageRead {
    Ready(Arc<[u8]>),
    NotYet,
    Failed(String),
}

/// The pages one background pass fills. A page member that fails to
/// read settles as its own page's error and the pass continues. A
/// failure that ends the pass (an undecodable block, a non-page member
/// that cannot be drained, a panic, a failed thread spawn) is what its
/// unfilled pages report, while filled pages keep their results.
struct Fill {
    pages: HashMap<usize, Result<Arc<[u8]>, String>>,
    /// `None` while the pass runs, `Some(Ok(()))` when it ended whole, `Some(Err(reason))` when it ended early.
    pass: Option<Result<(), String>>,
}

impl Fill {
    fn empty() -> Self {
        Self { pages: HashMap::new(), pass: None }
    }

    fn resolve(&self, index: usize) -> PageRead {
        match self.pages.get(&index) {
            Some(Ok(bytes)) => PageRead::Ready(Arc::clone(bytes)),
            Some(Err(message)) => PageRead::Failed(message.clone()),
            None => match &self.pass {
                None => PageRead::NotYet,
                Some(Ok(())) => PageRead::Failed(String::from("no entry in the archive")),
                Some(Err(fatal)) => PageRead::Failed(fatal.clone()),
            },
        }
    }
}

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
            reason = "the entry's own type is wanted, so symlinks stay out; `is_file()` would follow them"
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

    /// Listing reads the central directory, so a damaged local header
    /// does not refuse the whole book: that member fails when the fill
    /// reaches it.
    fn open_zip(path: &Path) -> Result<Self, SourceError> {
        let file = File::open(path).map_err(SourceError::Io)?;
        let zip = zip::ZipArchive::new(BufReader::new(file)).map_err(SourceError::Zip)?;
        let mut listed: Vec<(String, u64)> = Vec::new();
        for entry_index in 0..zip.len() {
            let entry = zip.by_index_data(entry_index).map_err(SourceError::Zip)?;
            if entry.is_dir() {
                continue;
            }
            let name = entry.name().map_err(SourceError::Zip)?.into_owned();
            if format_from_name(&name).is_some() {
                listed.push((name, entry.size()));
            }
        }
        Self::open_archive(listed, |names, fill| spawn_zip_fill(zip, names, &fill))
    }

    /// No password: a 7z with encrypted headers fails to open, and the
    /// first encrypted block ends the pass.
    fn open_sevenzip(path: &Path) -> Result<Self, SourceError> {
        let archive = sevenz_rust2::ArchiveReader::open(path, sevenz_rust2::Password::empty())
            .map_err(SourceError::SevenZip)?;
        let listed: Vec<(String, u64)> = archive
            .archive()
            .files
            .iter()
            .filter(|entry| !entry.is_directory() && format_from_name(&entry.name).is_some())
            .map(|entry| (entry.name.clone(), entry.size()))
            .collect();
        Self::open_archive(listed, |names, fill| spawn_sevenzip_fill(archive, &names, &fill))
    }

    fn open_archive(
        listed: Vec<(String, u64)>,
        spawn_fill: impl FnOnce(Vec<String>, Arc<Mutex<Fill>>),
    ) -> Result<Self, SourceError> {
        // Header sizes are claims, so the sum saturates rather than
        // wraps a liar archive past the guard.
        let unpacked_bytes: u64 =
            listed.iter().map(|&(_, size)| size).fold(0, u64::saturating_add);
        let mut names: Vec<String> = listed.into_iter().map(|(name, _)| name).collect();
        sort_pages(&mut names);
        if let Some(name) = duplicate_name(&names) {
            return Err(SourceError::Duplicate(name));
        }
        if names.is_empty() {
            return Err(SourceError::Empty);
        }
        guard_memory(unpacked_bytes, mem_available_bytes())?;
        let fill = Arc::new(Mutex::new(Fill::empty()));
        spawn_fill(names.clone(), Arc::clone(&fill));
        Self::from_names(Backend::Archive(fill), names)
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
        reason = "the page list is non-empty and immutable, and callers pass an index below its length; the debug assert restates it"
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

    /// The encoded bytes of one page. An archive still unpacking
    /// answers `NotYet`; the caller decides how to wait. A failure
    /// message carries no page number; the reader names the page.
    pub fn bytes(&self, index: usize) -> PageRead {
        let Some(name) = self.names.get(index) else {
            return PageRead::Failed(String::from("no such page"));
        };
        match &self.backend {
            Backend::Directory(dir) => match fs::read(dir.join(name)) {
                Ok(bytes) => PageRead::Ready(bytes.into()),
                Err(error) => PageRead::Failed(error.to_string()),
            },
            Backend::Archive(fill) => fill.lock().resolve(index),
        }
    }

    /// Whether the unpack pass is still running, for the status line's
    /// wait word. A directory never unpacks.
    pub fn unpacking(&self) -> bool {
        match &self.backend {
            Backend::Directory(_) => false,
            Backend::Archive(fill) => fill.lock().pass.is_none(),
        }
    }
}

/// The first name a sorted listing carries twice, if any. Two pages
/// would then share one entry's bytes, or one page would find no
/// entry, so the listing refuses them.
fn duplicate_name(sorted: &[String]) -> Option<String> {
    sorted
        .iter()
        .zip(sorted.iter().skip(1))
        .find(|(left, right)| left == right)
        .map(|(name, _)| name.clone())
}

fn sort_pages(names: &mut [String]) {
    names.sort_by(|left, right| natord::compare(left, right));
}

/// One member's bytes, with its read error as the message. A failed
/// member is drained, so the next member of a solid stream decodes
/// from its true start; an independent entry discards the rest.
fn read_member(member: &mut (impl Read + ?Sized)) -> Result<Arc<[u8]>, String> {
    let mut bytes = Vec::new();
    match member.read_to_end(&mut bytes) {
        Ok(_) => Ok(bytes.into()),
        Err(error) => {
            #[expect(
                clippy::let_underscore_must_use,
                clippy::let_underscore_untyped,
                reason = "the drain's outcome is irrelevant; only its side effect of consuming the member's remaining bytes matters"
            )]
            let _ = member.read_to_end(&mut bytes);
            Err(error.to_string())
        }
    }
}

/// Start the unpack thread for one pass body. Dropping the handle
/// detaches the thread, so quitting never waits on it. A failed spawn
/// settles the pass, so no page can wait on a thread that never ran.
fn spawn_pass(
    fill: &Arc<Mutex<Fill>>,
    body: impl FnOnce(&Arc<Mutex<Fill>>) -> Result<(), String> + Send + 'static,
) {
    use std::thread::Builder;

    let thread_fill = Arc::clone(fill);
    let spawned = Builder::new()
        .name("cy-unpack".into())
        .spawn(move || settle_pass(&thread_fill, body));
    if let Err(error) = spawned {
        fill.lock().pass = Some(Err(format!("could not start the unpack thread: {error}")));
    }
}

/// Fill a zip's pages in reading order. Zip members decode
/// independently, so each page settles on its own result: one bad
/// member fails only its page, and the pass always reaches its end.
fn spawn_zip_fill(
    mut zip: zip::ZipArchive<BufReader<File>>,
    names: Vec<String>,
    fill: &Arc<Mutex<Fill>>,
) {
    spawn_pass(fill, move |fill| {
        for (index, name) in names.iter().enumerate() {
            let page = match zip.by_name(name) {
                Ok(mut entry) => read_member(&mut entry),
                Err(error) => Err(error.to_string()),
            };
            fill.lock().pages.insert(index, page);
        }
        Ok(())
    });
}

/// Fill a 7z's pages in storage order, the only order a solid stream
/// permits: every member is consumed, page or not, before the next
/// decodes. A page member that fails to read settles as its own page's
/// error and the pass continues; a block that cannot decode or a
/// non-page member that cannot be drained ends the pass.
fn spawn_sevenzip_fill(
    mut reader: sevenz_rust2::ArchiveReader<File>,
    names: &[String],
    fill: &Arc<Mutex<Fill>>,
) {
    let by_name: HashMap<String, usize> = names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    spawn_pass(fill, move |fill| {
        reader
            .for_each_entries(|entry, data| {
                match by_name.get(&entry.name) {
                    Some(&index) => {
                        fill.lock().pages.insert(index, read_member(data));
                    }
                    None => {
                        // A solid block yields its later members only
                        // after this one is consumed, so non-pages are
                        // drained rather than buffered.
                        io::copy(data, &mut io::sink())?;
                    }
                }
                Ok(true)
            })
            .map_err(|error| error.to_string())
    });
}

fn settle_pass(fill: &Arc<Mutex<Fill>>, body: impl FnOnce(&Arc<Mutex<Fill>>) -> Result<(), String>) {
    use std::panic::{self, AssertUnwindSafe};

    let outcome = panic::catch_unwind(AssertUnwindSafe(|| body(fill)));
    let mut state = fill.lock();
    state.pass = Some(match outcome {
        Ok(Ok(())) => Ok(()),
        Ok(Err(reason)) => {
            tracing::warn!(%reason, "unpack pass ended early");
            Err(reason)
        }
        Err(payload) => Err(format!("unpack {}", panic_message(payload.as_ref()))),
    });
}

/// A panic payload rendered as a failure message, for the thread panic
/// boundaries: the unpack pass here and the decoder in the worker.
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is the crate's visibility style for crate-wide items, even inside private modules"
)]
pub(crate) fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        return format!("panicked: {text}");
    }
    if let Some(text) = payload.downcast_ref::<String>() {
        return format!("panicked: {text}");
    }
    String::from("panicked")
}

/// `MemAvailable` from a `/proc/meminfo` text, converted to bytes. The
/// line counts KiB. `None` when the line is missing or unparsable.
fn parse_mem_available_bytes(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|line| line.starts_with("MemAvailable:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(kib * 1024)
}

fn mem_available_bytes() -> Option<u64> {
    parse_mem_available_bytes(&fs::read_to_string("/proc/meminfo").ok()?)
}

fn guard_memory(unpacked: u64, available: Option<u64>) -> Result<(), SourceError> {
    match available {
        Some(available) if unpacked > available => {
            Err(SourceError::TooBig { unpacked, available })
        }
        _ => Ok(()),
    }
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

    /// LZMA solid 7z with two PNGs and a non-page text member in the
    /// same block, built by 7zz.
    const MIXED_SEVENZ_FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mixed_pages.7z"
    );

    #[test]
    fn fill_resolution() {
        let mut fill = Fill::empty();
        assert!(matches!(fill.resolve(0), PageRead::NotYet));
        fill.pages.insert(0, Ok(Arc::from(&b"a"[..])));
        fill.pages.insert(1, Err("bad page".into()));
        assert!(matches!(fill.resolve(0), PageRead::Ready(_)));
        assert!(matches!(fill.resolve(1), PageRead::Failed(message) if message == "bad page"));
        assert!(matches!(fill.resolve(2), PageRead::NotYet));

        fill.pass = Some(Err("stream broke".into()));
        assert!(matches!(fill.resolve(1), PageRead::Failed(message) if message == "bad page"));
        assert!(matches!(fill.resolve(2), PageRead::Failed(message) if message == "stream broke"));

        fill.pass = Some(Ok(()));
        assert!(
            matches!(fill.resolve(2), PageRead::Failed(message) if message == "no entry in the archive")
        );
    }

    #[test]
    fn meminfo_parsing() {
        let meminfo = "MemTotal:       100 kB\nMemAvailable:   4096 kB\nSwapTotal:      2 kB\n";
        assert_eq!(parse_mem_available_bytes(meminfo), Some(4096 * 1024));
        assert_eq!(parse_mem_available_bytes("MemTotal: 1 kB\n"), None);
        assert_eq!(parse_mem_available_bytes("MemAvailable: notanumber kB\n"), None);
    }

    #[test]
    fn memory_guard() {
        assert!(matches!(
            guard_memory(100, Some(50)),
            Err(SourceError::TooBig { unpacked: 100, available: 50 })
        ));
        guard_memory(50, Some(100)).unwrap();
        guard_memory(u64::MAX, None).unwrap();
    }

    #[test]
    fn settle_pass_outcomes() {
        let whole = Arc::new(Mutex::new(Fill::empty()));
        settle_pass(&whole, |_| Ok(()));
        assert!(matches!(&whole.lock().pass, Some(Ok(()))));

        let broke = Arc::new(Mutex::new(Fill::empty()));
        settle_pass(&broke, |_| Err("stream broke".into()));
        assert!(matches!(&broke.lock().pass, Some(Err(reason)) if reason.contains("broke")));

        let boom = Arc::new(Mutex::new(Fill::empty()));
        settle_pass(&boom, |_| panic!("boom"));
        assert!(matches!(boom.lock().resolve(0), PageRead::Failed(reason) if reason.contains("boom")));
    }

    #[test]
    fn liar_sizes_refuse() {
        let listed = vec![("a.png".to_owned(), u64::MAX), ("b.png".to_owned(), u64::MAX)];
        #[expect(
            clippy::rest_pattern_accessible_field,
            reason = "binding the ignored field as `_` instead trips unneeded_field_pattern"
        )]
        match Source::open_archive(listed, |_, _| {}) {
            Err(SourceError::TooBig { unpacked: u64::MAX, .. }) => {}
            Err(other) => panic!("liar sizes settled as {other}"),
            Ok(source) => panic!("liar sizes opened as {} pages", source.len()),
        }
    }

    #[test]
    fn duplicate_names_refuse() {
        let listed = vec![
            ("b.png".to_owned(), 8),
            ("a.png".to_owned(), 8),
            ("a.png".to_owned(), 8),
        ];
        assert!(matches!(
            Source::open_archive(listed, |_, _| {}),
            Err(SourceError::Duplicate(name)) if name == "a.png"
        ));
    }

    #[test]
    fn sevenz_listing() {
        let source = Source::open(Path::new(SEVENZ_FIXTURE)).expect("7z opens");
        assert_eq!(source.names, ["p1.png", "p2.png"]);
        assert_eq!(
            source.page_name(0),
            PageName { parent: None, file: "p1.png".into() }
        );
    }

    #[test]
    fn sevenz_read_through_fill() {
        let source = Source::open(Path::new(SEVENZ_FIXTURE)).expect("7z opens");
        let bytes = eventually(20, "first page never unpacked", || match source.bytes(0) {
            PageRead::Ready(bytes) => Some(bytes),
            PageRead::Failed(message) => panic!("page 0 failed: {message}"),
            PageRead::NotYet => None,
        });
        assert!(matches!(source.bytes(0), PageRead::Ready(repeat) if repeat.as_ref() == bytes.as_ref()));
        assert_eq!(bytes.get(..8), Some(&b"\x89PNG\r\n\x1a\n"[..]));
        // the decoder is the oracle for the fixture's dims
        let page = decode::decode(&bytes, "p1.png").expect("fixture page decodes");
        assert_eq!((page.width, page.height), (40, 30));

        eventually(20, "pass never ended", || (!source.unpacking()).then_some(()));
        assert!(matches!(source.bytes(1), PageRead::Ready(_)));
    }

    /// A non-page member in the same solid block is drained, so the
    /// later page still decodes.
    #[test]
    fn sevenz_drains_non_page_members() {
        let source = Source::open(Path::new(MIXED_SEVENZ_FIXTURE)).expect("7z opens");
        assert_eq!(source.len(), 2);
        let mut pages = Vec::new();
        for index in 0..source.len() {
            let name = source.name(index).expect("page listed");
            let bytes = eventually(20, "page never unpacked", || match source.bytes(index) {
                PageRead::Ready(bytes) => Some(bytes),
                PageRead::Failed(message) => panic!("{name} failed: {message}"),
                PageRead::NotYet => None,
            });
            let page = decode::decode(&bytes, name).expect("fixture page decodes");
            assert_eq!(page.format, decode::Format::Png);
            pages.push(bytes);
        }
        let [first, second] = pages.as_slice() else {
            panic!("two pages listed");
        };
        assert_ne!(first.as_ref(), second.as_ref());
    }

    #[test]
    fn dir_listing_and_read() {
        let dir = TempDir::new("pagedir");
        dir.write("c10.png", &tiny_png(1, 1, [0, 0, 0, 255]));
        dir.write("c2.png", &tiny_png(1, 1, [7, 8, 9, 255]));
        dir.write("c1.png", &tiny_png(1, 1, [0, 0, 0, 255]));
        dir.write("cover.PNG", &tiny_png(1, 1, [0, 0, 0, 255]));
        dir.write("notes.txt", b"not a page");
        let source = Source::open(dir.path()).expect("dir opens");
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
        assert!(!source.unpacking());
        assert!(matches!(source.bytes(1), PageRead::Ready(read) if read.as_ref() == tiny_png(1, 1, [7, 8, 9, 255]).as_slice()));
    }

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
        assert_eq!(
            source.page_name(3),
            PageName { parent: Some("deep".into()), file: "005.png".into() }
        );
        // bytes(1) is b/010.png, written with color 1.
        let page = eventually(20, "zip page never unpacked", || match source.bytes(1) {
            PageRead::Ready(bytes) => Some(bytes),
            PageRead::Failed(message) => panic!("page 1 failed: {message}"),
            PageRead::NotYet => None,
        });
        assert_eq!(page.as_ref(), tiny_png(2, 2, [1, 0, 0, 255]).as_slice());
        eventually(20, "pass never ended", || (!source.unpacking()).then_some(()));
    }

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
            PageRead::Failed(message) => Some(message),
            PageRead::Ready(_) => panic!("corrupt page unpacked"),
            PageRead::NotYet => None,
        });
        assert!(failure.to_lowercase().contains("checksum"), "{failure}");
        assert_eq!(
            eventually(20, "page 0 never unpacked", || match source.bytes(0) {
                PageRead::Ready(bytes) => Some(bytes),
                PageRead::Failed(message) => panic!("page 0 failed: {message}"),
                PageRead::NotYet => None,
            })
            .as_ref(),
            pages[0].as_slice()
        );
        assert_eq!(
            eventually(20, "page 2 never unpacked", || match source.bytes(2) {
                PageRead::Ready(bytes) => Some(bytes),
                PageRead::Failed(message) => panic!("page 2 failed: {message}"),
                PageRead::NotYet => None,
            })
            .as_ref(),
            pages[2].as_slice()
        );
    }

    /// A damaged local header must not refuse the book: the central
    /// directory still lists the page, and the fill fails only that member.
    #[test]
    fn zip_corrupt_local_header_keeps_the_book_open() {
        use zip::write::SimpleFileOptions;

        let dir = TempDir::new("pagezipheader");
        let zip_path = dir.path().join("book.cbz");
        let pages = [
            tiny_png(2, 2, [1, 0, 0, 255]),
            tiny_png(2, 2, [2, 0, 0, 255]),
            tiny_png(2, 2, [3, 0, 0, 255]),
        ];
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
        // Local headers start with `PK\x03\x04`; damage the second one
        // while leaving its central-directory entry intact.
        {
            let mut book = fs::read(&zip_path).unwrap();
            let mut headers = book
                .windows(4)
                .enumerate()
                .filter(|(_, bytes)| *bytes == b"PK\x03\x04")
                .map(|(offset, _)| offset);
            headers.next().expect("first local header");
            let second = headers.next().expect("second local header");
            *book.get_mut(second).expect("offset points into the file") ^= 0xff;
            fs::write(&zip_path, &book).unwrap();
        }

        let source = Source::open(&zip_path).expect("central directory still opens the book");
        eventually(20, "damaged header never settled", || match source.bytes(1) {
            PageRead::Failed(message) => Some(message),
            PageRead::Ready(_) => panic!("damaged local header unpacked"),
            PageRead::NotYet => None,
        });
        assert_eq!(
            eventually(20, "page 0 never unpacked", || match source.bytes(0) {
                PageRead::Ready(bytes) => Some(bytes),
                PageRead::Failed(message) => panic!("page 0 failed: {message}"),
                PageRead::NotYet => None,
            })
            .as_ref(),
            pages[0].as_slice()
        );
        assert_eq!(
            eventually(20, "page 2 never unpacked", || match source.bytes(2) {
                PageRead::Ready(bytes) => Some(bytes),
                PageRead::Failed(message) => panic!("page 2 failed: {message}"),
                PageRead::NotYet => None,
            })
            .as_ref(),
            pages[2].as_slice()
        );
    }

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
