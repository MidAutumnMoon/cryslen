//! Page source: a directory or archive viewed as a natural-sorted list
//! of pages. The `Source` owns persistent archive handles so zip and 7z
//! files are not reopened per page. 7z sources also hold a
//! decompressed-page tier filled by a background pass.

use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::fmt;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::io::Error as IoError;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use zip::result::ZipError;

use crate::decode::format_from_name;

#[derive(Debug)]
pub enum SourceError {
    NotFound(PathBuf),
    Empty,
    Unsupported(PathBuf),
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

pub(crate) fn is_archive(path: &Path) -> bool {
    archive_kind(path).is_some()
}

/// Byte budget of the 7z tier, sized to hold a whole archive's members.
const TIER_BUDGET_BYTES: usize = 512 * 1024 * 1024;

/// Decompressed 7z members keyed by page index. The harvest pass and
/// `bytes` misses fill it. A fill that exceeds the byte budget evicts in
/// insertion order. Directories lean on the OS page cache and zip
/// inflate is cheap, so only 7z sources carry a tier.
struct ByteTier {
    budget: usize,
    live_bytes: usize,
    order: VecDeque<usize>,
    map: HashMap<usize, Arc<[u8]>>,
}

impl ByteTier {
    fn new(budget: usize) -> Self {
        Self { budget, live_bytes: 0, order: VecDeque::new(), map: HashMap::new() }
    }

    fn get(&self, index: usize) -> Option<&Arc<[u8]>> {
        self.map.get(&index)
    }

    /// A member larger than the whole budget is not retained. A repeated
    /// index keeps its incumbent, so the byte count stays exact.
    #[expect(
        clippy::expect_used,
        reason = "every map key has exactly one order entry because inserts and evicts touch both together, so over budget with an empty map is impossible. The ByteTier tests pin this"
    )]
    fn insert(&mut self, index: usize, data: Arc<[u8]>) {
        if data.len() > self.budget || self.map.contains_key(&index) {
            return;
        }
        self.live_bytes += data.len();
        self.order.push_back(index);
        self.map.insert(index, data);
        while self.live_bytes > self.budget {
            let evicted = self.order.pop_front().expect("over budget implies an entry");
            self.live_bytes -= self.map.remove(&evicted).expect("order and map agree").len();
        }
    }
}

/// A poisoned lock only means a holder panicked mid-flight. The guarded
/// data is still readable, so recover the guard.
fn lock_recovered<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    match lock.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// The state a 7z source's read path needs.
struct SevenZipState {
    archive: Mutex<sevenz_rust2::ArchiveReader<File>>,
    /// Decompressed 7z members, shared with the harvest thread.
    tier: Arc<Mutex<ByteTier>>,
}

/// Where pages come from, decided by `Source::open` from one path argument.
/// Each variant carries the state its read path needs.
#[expect(
    clippy::large_enum_variant,
    reason = "one Source behind an Arc, so the enum size is never multiplied"
)]
enum Backend {
    Directory(PathBuf),
    Zip(Mutex<zip::ZipArchive<BufReader<File>>>),
    SevenZip(SevenZipState),
}

pub struct Source {
    names: Vec<String>,
    backend: Backend,
}

impl Source {
    /// Listing is filtered to image extensions and natural-sorted.
    pub fn open(path: &Path) -> Result<Self, SourceError> {
        if fs::metadata(path).map_err(|_| SourceError::NotFound(path.to_path_buf()))?.is_dir() {
            return Self::open_dir(path);
        }
        match archive_kind(path) {
            Some(ArchiveKind::Zip) => Self::open_zip(path),
            Some(ArchiveKind::SevenZip) => Self::open_sevenzip(path),
            None => Err(SourceError::Unsupported(path.to_path_buf())),
        }
    }

    fn open_dir(dir: &Path) -> Result<Self, SourceError> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .map_err(SourceError::Io)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_ok_and(|file_type| file_type.is_file()))
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| format_from_name(name).is_some())
            .collect();
        sort_pages(&mut names);
        Self::from_names(Backend::Directory(dir.to_path_buf()), names)
    }

    fn open_zip(path: &Path) -> Result<Self, SourceError> {
        let file = File::open(path).map_err(SourceError::Io)?;
        let zip = zip::ZipArchive::new(BufReader::new(file)).map_err(SourceError::Zip)?;
        // Non-UTF-8 entry names are skipped (zip 9 reports them as errors).
        let mut names: Vec<String> = zip
            .file_names()
            .filter_map(|name| name.ok())
            .filter(|name| format_from_name(name).is_some())
            .map(|name| name.into_owned())
            .collect();
        sort_pages(&mut names);
        Self::from_names(Backend::Zip(Mutex::new(zip)), names)
    }

    fn open_sevenzip(path: &Path) -> Result<Self, SourceError> {
        let archive = sevenz_rust2::ArchiveReader::open(path, sevenz_rust2::Password::empty())
            .map_err(SourceError::SevenZip)?;
        let mut names: Vec<String> = archive
            .archive()
            .files
            .iter()
            .filter(|entry| !entry.is_directory && format_from_name(&entry.name).is_some())
            .map(|entry| entry.name.clone())
            .collect();
        sort_pages(&mut names);
        // The empty check runs before the spawn, so a pageless archive
        // starts no harvest thread.
        if names.is_empty() {
            return Err(SourceError::Empty);
        }
        let pages: HashMap<String, usize> = names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index))
            .collect();
        let tier = Arc::new(Mutex::new(ByteTier::new(TIER_BUDGET_BYTES)));
        let backend = Backend::SevenZip(SevenZipState {
            archive: Mutex::new(archive),
            tier: Arc::clone(&tier),
        });
        spawn_harvest(path.to_path_buf(), pages, tier);
        Self::from_names(backend, names)
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
        debug_assert!(index < self.names.len());
        let entry = &self.names[index];
        let file = entry.rsplit('/').next().unwrap_or(entry).to_string();
        let parent = match &self.backend {
            Backend::Directory(dir) => {
                dir.file_name().map(|name| name.to_string_lossy().into_owned())
            }
            Backend::Zip(_) | Backend::SevenZip(_) => entry.rsplit_once('/').map(|(path, _)| {
                path.rsplit('/').next().unwrap_or(path).to_string()
            }),
        };
        PageName { parent, file }
    }

    /// Raw encoded bytes of one page. 7z reads consult the byte tier first
    /// and fill it on a miss, so repeat reads are an `Arc` refcount bump.
    #[expect(
        clippy::indexing_slicing,
        reason = "the page list is non-empty and immutable, and every page index is clamped to its length before reaching here"
    )]
    pub fn bytes(&self, index: usize) -> Result<Arc<[u8]>, SourceError> {
        debug_assert!(index < self.names.len());
        let name = &self.names[index];
        match &self.backend {
            Backend::Directory(dir) => fs::read(dir.join(name))
                .map(Arc::from)
                .map_err(SourceError::Io),
            Backend::Zip(zip) => {
                let mut archive = lock_recovered(zip);
                let mut entry = archive.by_name(name).map_err(SourceError::Zip)?;
                let mut buf = Vec::new();
                entry
                    .read_to_end(&mut buf)
                    .map_err(|error| SourceError::Zip(ZipError::Io(error)))?;
                Ok(buf.into())
            }
            Backend::SevenZip(backend) => {
                if let Some(cached) = lock_recovered(&backend.tier).get(index).cloned() {
                    return Ok(cached);
                }
                // The reader guard is scoped so the reader and tier locks
                // are never held together.
                let data: Arc<[u8]> = {
                    let mut archive = lock_recovered(&backend.archive);
                    archive.read_file(name).map_err(SourceError::SevenZip)?.into()
                };
                lock_recovered(&backend.tier).insert(index, data.clone());
                Ok(data)
            }
        }
    }

    #[cfg(test)]
    fn tier_has(&self, index: usize) -> bool {
        match &self.backend {
            Backend::SevenZip(backend) => lock_recovered(&backend.tier).get(index).is_some(),
            Backend::Directory(_) | Backend::Zip(_) => false,
        }
    }
}

/// Decompresses every page entry into the tier in one background pass.
/// The reader is the thread's own, not the retained one, so a `bytes`
/// miss never waits behind the pass. Every entry is read to its end even
/// when it is not a page, because a solid block's later members decode
/// only after the earlier ones are consumed. A spawn error skips the
/// pass and a read error ends it. Either way, pages load through
/// `bytes` and the retained reader.
fn spawn_harvest(path: PathBuf, pages: HashMap<String, usize>, tier: Arc<Mutex<ByteTier>>) {
    use std::thread::Builder;

    // Dropping the handle detaches the thread, so quitting never waits
    // on it.
    drop(Builder::new()
        .name("cy-harvest".into())
        .spawn(move || {
            let Ok(mut reader) =
                sevenz_rust2::ArchiveReader::open(&path, sevenz_rust2::Password::empty())
            else {
                return;
            };
            let mut member = Vec::new();
            let _ = reader.for_each_entries(|entry, data| {
                member.clear();
                data.read_to_end(&mut member)?;
                if let Some(&index) = pages.get(&entry.name) {
                    lock_recovered(&tier).insert(index, Arc::from(member.as_slice()));
                }
                Ok(true)
            });
        }));
}

/// Pages read in filename order for humans: c2 before c10.
fn sort_pages(names: &mut [String]) {
    names.sort_by(|left, right| natord::compare(left, right));
}

#[cfg(test)]
#[expect(clippy::expect_used, clippy::unwrap_used, reason = "a panic in test code is inconsequential")]
mod tests {
    use super::*;
    use crate::decode;
    use crate::testkit::{tiny_png, TempDir};

    use std::io::Write as _;
    use std::time::{Duration, Instant};

    /// LZMA-compressed 7z with two small PNGs, built by 7zz. The crate's
    /// encoder features are trimmed, so the fixture is checked in.
    const SEVENZ_FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/two_pages.7z"
    );

    #[test]
    fn sevenz_listing_and_read() {
        let source = Source::open(Path::new(SEVENZ_FIXTURE)).expect("7z opens");
        assert_eq!(source.len(), 2);
        assert_eq!(
            source.page_name(0),
            PageName { parent: None, file: "p1.png".into() }
        );
        assert_eq!(source.names, ["p1.png", "p2.png"]);
        let bytes = source.bytes(0).expect("first page reads");
        assert_eq!(bytes.get(..8), Some(&b"\x89PNG\r\n\x1a\n"[..]));
        let be_u32 = |at: usize| {
            bytes
                .get(at..at + 4)
                .and_then(|word| word.try_into().ok())
                .map(u32::from_be_bytes)
        };
        let width = be_u32(16).expect("fixture ihdr width");
        let height = be_u32(20).expect("fixture ihdr height");
        assert_eq!((width, height), (40, 30));
        let decoded_page = decode::decode(&bytes, "p1.png").expect("decodes");
        assert_eq!((decoded_page.width, decoded_page.height), (width, height));
    }

    #[test]
    fn sevenz_repeat_read_served_from_tier() {
        let source = Source::open(Path::new(SEVENZ_FIXTURE)).expect("7z opens");
        let first = source.bytes(0).expect("first page reads");
        // Either bytes or the harvest thread may have filled the tier, and
        // both write the same member's bytes.
        assert!(source.tier_has(0), "tier holds the page after a read");
        let second = source.bytes(0).expect("second page reads");
        assert_eq!(first, second);
    }

    #[test]
    fn sevenz_harvest_fills_tier() {
        use std::thread;

        let source = Source::open(Path::new(SEVENZ_FIXTURE)).expect("7z opens");
        let deadline = Instant::now() + Duration::from_secs(20);
        while !(source.tier_has(0) && source.tier_has(1)) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(50));
        }
        assert!(source.tier_has(0) && source.tier_has(1), "both fixture pages harvested");
    }

    fn member(len: usize) -> Arc<[u8]> {
        Arc::from(vec![0_u8; len])
    }

    #[test]
    fn tier_evicts_in_insertion_order() {
        let mut tier = ByteTier::new(10);
        tier.insert(0, member(6));
        tier.insert(1, member(6));
        assert!(tier.get(0).is_none());
        assert!(tier.get(1).is_some());
        assert_eq!(tier.live_bytes, 6);
        tier.insert(2, member(4));
        assert!(tier.get(1).is_some());
        assert!(tier.get(2).is_some());
        assert_eq!(tier.live_bytes, 10);
    }

    #[test]
    fn tier_rejects_oversized_member() {
        let mut tier = ByteTier::new(10);
        tier.insert(0, member(11));
        assert!(tier.get(0).is_none());
        assert_eq!(tier.live_bytes, 0);
        tier.insert(1, member(10));
        assert!(tier.get(1).is_some());
        assert_eq!(tier.live_bytes, 10);
    }

    #[test]
    fn tier_keeps_incumbent_on_repeat_insert() {
        let mut tier = ByteTier::new(100);
        tier.insert(0, Arc::from(vec![1_u8]));
        tier.insert(0, Arc::from(vec![2_u8, 2]));
        assert_eq!(tier.get(0).map(|member| member.to_vec()), Some(vec![1_u8]));
        assert_eq!(tier.live_bytes, 1);
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
        assert_eq!(
            source.page_name(2),
            PageName {
                parent: Some(dir.path().file_name().unwrap().to_string_lossy().into_owned()),
                file: "c10.png".into(),
            }
        );
        assert_eq!(source.names, ["c1.png", "c2.png", "c10.png", "cover.PNG"]);
        assert_eq!(source.name(2), Some("c10.png"));
    }

    #[test]
    fn dir_bytes_roundtrip() {
        let dir = TempDir::new("pagebytes");
        let png = tiny_png(3, 2, [9, 8, 7, 255]);
        dir.write("p.png", &png);
        let source = Source::open(dir.path()).unwrap();
        assert_eq!(source.bytes(0).as_deref().ok(), Some(png.as_slice()));
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
        // A nested entry shows its immediate parent, not the whole path.
        assert_eq!(
            source.page_name(3),
            PageName { parent: Some("deep".into()), file: "005.png".into() }
        );
        let page = source.bytes(1).expect("entry readable");
        let decoded = decode::decode(&page, "b/010.png").unwrap();
        assert_eq!(decoded.width, 2);
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
    }
}
