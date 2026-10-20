//! Page model: a directory or archive viewed as a natural-sorted list of
//! pages, plus the pure spread-pairing math (cover alone, then pairs).
//! The spread functions are pure and unit-tested. The `Source` owns
//! persistent archive handles so zip and 7z files are not reopened per
//! page. 7z sources additionally hold a decompressed-page tier filled by
//! a background pass.

use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use crate::decode::format_from_name;

/// Where pages come from, decided by `Source::open` from one path argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceKind {
    Directory(PathBuf),
    Zip(PathBuf),
    SevenZip(PathBuf),
}

#[derive(Debug)]
pub enum SourceError {
    NotFound(PathBuf),
    Empty,
    Unsupported(PathBuf),
    Zip(zip::result::ZipError),
    SevenZip(sevenz_rust2::Error),
    Io(std::io::Error),
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceError::NotFound(p) => write!(f, "not found: {}", p.display()),
            SourceError::Empty => write!(f, "no image pages found"),
            SourceError::Unsupported(p) => write!(f, "unsupported source: {}", p.display()),
            SourceError::Zip(e) => write!(f, "zip: {e}"),
            SourceError::SevenZip(e) => write!(f, "7z: {e}"),
            SourceError::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for SourceError {}

/// Byte budget of the 7z tier. Whole books measure 208-383 MiB, so one
/// tier holds an entire archive.
const TIER_BYTES: usize = 512 * 1024 * 1024;

/// Decompressed 7z members keyed by page index. The harvest pass and
/// `bytes` misses fill it. A fill that passes the byte budget evicts in
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
        ByteTier { budget, live_bytes: 0, order: VecDeque::new(), map: HashMap::new() }
    }

    fn get(&self, index: usize) -> Option<&Arc<[u8]>> {
        self.map.get(&index)
    }

    /// A member larger than the whole budget is not retained. A repeated
    /// index keeps its incumbent, so the byte count stays exact.
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

/// A page source with persistent handles. Cheap to keep alive for the whole
/// session. Shared with the decode worker behind an `Arc`.
pub struct Source {
    pub kind: SourceKind,
    names: Vec<String>,
    zip: Option<Mutex<zip::ZipArchive<std::io::BufReader<std::fs::File>>>>,
    sevenzip: Option<Mutex<sevenz_rust2::ArchiveReader<std::fs::File>>>,
    /// Decompressed 7z members, shared with the harvest thread.
    tier: Option<Arc<Mutex<ByteTier>>>,
    /// Dropping detaches the harvest thread so quitting does not wait out
    /// a pass. The thread owns only its reader and the tier `Arc` and
    /// finishes alone.
    harvest: Option<JoinHandle<()>>,
}

impl Source {
    /// `path` is a directory, or `.zip`/`.cbz`/`.7z`/`.cb7` archive. Listing is
    /// filtered to image extensions and natural-sorted (natord).
    pub fn open(path: &Path) -> Result<Source, SourceError> {
        if std::fs::metadata(path).map_err(|_| SourceError::NotFound(path.to_path_buf()))?.is_dir() {
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
            _ => Err(SourceError::Unsupported(path.to_path_buf())),
        }
    }

    fn open_dir(dir: &Path) -> Result<Source, SourceError> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .map_err(SourceError::Io)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_ok_and(|t| t.is_file()))
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| format_from_name(name).is_some())
            .collect();
        sort_pages(&mut names);
        Source::from_names(SourceKind::Directory(dir.to_path_buf()), names)
    }

    /// The archive handle opened here stays open for the session. `bytes`
    /// reads go through it, avoiding a central-directory parse per page.
    fn open_zip(path: &Path) -> Result<Source, SourceError> {
        let file = std::fs::File::open(path).map_err(SourceError::Io)?;
        let zip = zip::ZipArchive::new(std::io::BufReader::new(file)).map_err(SourceError::Zip)?;
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

    /// The reader opened here stays for the session and serves `bytes`
    /// reads. A harvest thread decompresses every page into the byte
    /// tier in the background, so a solid block is decoded once per
    /// archive.
    fn open_sevenzip(path: &Path) -> Result<Source, SourceError> {
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
        let mut source = Source::from_names(SourceKind::SevenZip(path.to_path_buf()), names)?;
        let pages: HashMap<String, usize> = source
            .names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index))
            .collect();
        let tier = Arc::new(Mutex::new(ByteTier::new(TIER_BYTES)));
        source.sevenzip = Some(Mutex::new(archive));
        source.tier = Some(Arc::clone(&tier));
        source.harvest = Some(spawn_harvest(path.to_path_buf(), pages, tier));
        Ok(source)
    }

    fn from_names(kind: SourceKind, names: Vec<String>) -> Result<Source, SourceError> {
        if names.is_empty() {
            return Err(SourceError::Empty);
        }
        Ok(Source {
            kind,
            names,
            zip: None,
            sevenzip: None,
            tier: None,
            harvest: None,
        })
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

    /// Raw encoded bytes of one page. Archive reads go through the
    /// session-held handles. 7z reads consult the byte tier first and
    /// fill it on a miss, so repeat reads are an `Arc` refcount bump.
    pub fn bytes(&self, index: usize) -> Result<Arc<[u8]>, SourceError> {
        let name = self
            .names
            .get(index)
            .ok_or_else(|| SourceError::Io(std::io::Error::other(format!("page {index} out of range"))))?;
        match &self.kind {
            SourceKind::Directory(dir) => std::fs::read(dir.join(name))
                .map(Arc::from)
                .map_err(SourceError::Io),
            SourceKind::Zip(_) => {
                let mut archive =
                    lock_recovered(self.zip.as_ref().expect("zip handle for zip sources"));
                let mut entry = archive.by_name(name).map_err(SourceError::Zip)?;
                let mut buf = Vec::new();
                entry
                    .read_to_end(&mut buf)
                    .map_err(|e| SourceError::Zip(zip::result::ZipError::Io(e)))?;
                Ok(buf.into())
            }
            SourceKind::SevenZip(_) => {
                let tier = self.tier.as_ref().expect("byte tier for 7z sources");
                if let Some(cached) = lock_recovered(tier).get(index).cloned() {
                    return Ok(cached);
                }
                // The reader guard is scoped so the reader and tier locks
                // are never held together.
                let data: Arc<[u8]> = {
                    let mut archive =
                        lock_recovered(self.sevenzip.as_ref().expect("7z handle for 7z sources"));
                    archive.read_file(name).map_err(SourceError::SevenZip)?.into()
                };
                lock_recovered(tier).insert(index, data.clone());
                Ok(data)
            }
        }
    }

    /// Whether the byte tier holds `index`. Test probe for the background
    /// fill.
    #[cfg(test)]
    fn tier_has(&self, index: usize) -> bool {
        match &self.tier {
            Some(tier) => lock_recovered(tier).get(index).is_some(),
            None => false,
        }
    }
}

/// Decompresses every page entry into the tier in one background pass.
/// The reader is the thread's own, not the retained one, so a `bytes`
/// miss never waits behind the pass. Every entry is read to its end even
/// when it is not a page, because a solid block's later members decode
/// only after the earlier ones are consumed. A read error ends the pass.
/// Pages then still come from `bytes` through the retained reader.
fn spawn_harvest(
    path: PathBuf,
    pages: HashMap<String, usize>,
    tier: Arc<Mutex<ByteTier>>,
) -> JoinHandle<()> {
    std::thread::Builder::new()
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
        })
        .expect("spawn harvest thread")
}

/// Pages read in filename order for humans: c2 before c10.
fn sort_pages(names: &mut [String]) {
    names.sort_by(|a, b| natord::compare(a, b));
}

/// Which pages one screen shows. The cover is alone, afterwards pairs,
/// 0-based: `0`, `1+2`, `3+4`, ...
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spread {
    pub first: usize,
    pub second: Option<usize>,
}

/// Spread containing `index` when double-page mode is on/off, clamped to the
/// book: an odd tail shows one page (`second: None`), never a phantom page.
pub fn spread_for(index: usize, double: bool, len: usize) -> Spread {
    let len = len.max(1);
    let index = index.min(len - 1);
    if !double {
        return Spread { first: index, second: None };
    }
    if index == 0 {
        return Spread { first: 0, second: None };
    }
    let first = spread_start(spread_number(index));
    Spread { first, second: (first + 1 < len).then_some(first + 1) }
}

/// Next/previous spread start from `current`, wrapping at the ends.
/// Forward from the last spread lands on the cover, and backward from the
/// cover on the last spread. Always returns a spread *start*.
pub fn advance(current: usize, forward: bool, double: bool, len: usize) -> usize {
    let len = len.max(1);
    let current = current.min(len - 1);
    if !double {
        let next = if forward { current + 1 } else { current + len - 1 };
        return next % len;
    }
    // The double-mode cycle is the lone cover, then the pairs.
    let spreads = 1 + (len - 1).div_ceil(2);
    let number = spread_number(current);
    let next = if forward { (number + 1) % spreads } else { (number + spreads - 1) % spreads };
    spread_start(next)
}

/// 0-based spread number of a page. The cover is 0, and pages 1 and 2
/// share 1.
fn spread_number(page: usize) -> usize {
    page.div_ceil(2)
}

/// First page of a double-mode spread number. Pairs start on odd pages.
fn spread_start(number: usize) -> usize {
    if number > 0 { 2 * number - 1 } else { 0 }
}

impl Spread {
    /// Visual seating as [left, right]. Western reads first-left. Manga
    /// reads first-right, so a lone page lands in the right slot.
    pub fn seats(&self, rtl: bool) -> [Option<usize>; 2] {
        if rtl {
            [self.second, Some(self.first)]
        } else {
            [Some(self.first), self.second]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::testkit::{tiny_png, TempDir};

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
        let source = Source::open(std::path::Path::new(SEVENZ_FIXTURE)).expect("7z opens");
        assert_eq!(source.len(), 2);
        assert_eq!(source.label(), "cb7");
        assert_eq!(source.names, ["p1.png", "p2.png"]);
        let bytes = source.bytes(0).expect("first page reads");
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
        let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
        assert_eq!((width, height), (40, 30));
        let decoded_page = crate::decode::decode(&bytes, "p1.png").expect("decodes");
        assert_eq!((decoded_page.width, decoded_page.height), (width, height));
    }

    #[test]
    fn sevenz_repeat_read_served_from_tier() {
        let source = Source::open(std::path::Path::new(SEVENZ_FIXTURE)).expect("7z opens");
        let first = source.bytes(0).expect("first page reads");
        // Either bytes or the harvest thread may have filled the tier, and
        // both write the same member's bytes.
        assert!(source.tier_has(0), "tier holds the page after a read");
        let second = source.bytes(0).expect("second page reads");
        assert_eq!(first, second);
    }

    #[test]
    fn sevenz_harvest_fills_tier() {
        let source = Source::open(std::path::Path::new(SEVENZ_FIXTURE)).expect("7z opens");
        let deadline = Instant::now() + Duration::from_secs(20);
        while !(source.tier_has(0) && source.tier_has(1)) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(source.tier_has(0) && source.tier_has(1), "both fixture pages harvested");
    }

    fn member(len: usize) -> Arc<[u8]> {
        Arc::from(vec![0u8; len])
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
        tier.insert(0, Arc::from(vec![1u8]));
        tier.insert(0, Arc::from(vec![2u8, 2]));
        assert_eq!(tier.get(0).map(|d| d.to_vec()), Some(vec![1u8]));
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
        assert_eq!(source.label(), "dir");
        assert_eq!(source.names, ["c1.png", "c2.png", "c10.png", "cover.PNG"]);
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

    #[test]
    fn cover_alone_then_pairs() {
        assert_eq!(spread_for(0, true, 10), Spread { first: 0, second: None });
        assert_eq!(spread_for(1, true, 10), Spread { first: 1, second: Some(2) });
        assert_eq!(spread_for(2, true, 10), Spread { first: 1, second: Some(2) });
        assert_eq!(spread_for(3, true, 10), Spread { first: 3, second: Some(4) });
        assert_eq!(spread_for(0, false, 10), Spread { first: 0, second: None });
        assert_eq!(spread_for(5, false, 10), Spread { first: 5, second: None });
    }

    #[test]
    fn odd_tail_clamps_at_source() {
        // 7 pages: index 6 lives in pair (5, 6).
        assert_eq!(spread_for(6, true, 7), Spread { first: 5, second: Some(6) });
        // 6 pages: the would-be pair (5, 6) shows page 5 alone, with no
        // phantom page.
        assert_eq!(spread_for(5, true, 6), Spread { first: 5, second: None });
        // Out-of-range index clamps to the last page.
        assert_eq!(spread_for(99, true, 6).first, 5);
    }

    #[test]
    fn advance_steps_and_wraps() {
        assert_eq!(advance(1, true, true, 10), 3);
        assert_eq!(advance(3, false, true, 10), 1);
        // 8 is the second page of the last spread (7, 8).
        assert_eq!(advance(8, true, true, 9), 0);
        assert_eq!(advance(1, false, true, 10), 0);
        assert_eq!(advance(4, true, false, 10), 5);
        // The cover's next spread starts at 1, not 2 (status-line bug class).
        assert_eq!(advance(0, true, true, 10), 1);
    }

    #[test]
    fn advance_wraps_at_both_ends() {
        // Single mode: last page to first and back.
        assert_eq!(advance(9, true, false, 10), 0);
        assert_eq!(advance(0, false, false, 10), 9);
        // Double mode with a lone tail page.
        assert_eq!(advance(9, true, true, 10), 0);
        assert_eq!(advance(0, false, true, 10), 9);
        // Double mode ending on a full pair.
        assert_eq!(advance(7, true, true, 9), 0);
        assert_eq!(advance(0, false, true, 9), 7);
        // Odd tail: 6 pages are 0 | 1,2 | 3,4 | 5.
        assert_eq!(advance(5, true, true, 6), 0);
        assert_eq!(advance(0, false, true, 6), 5);
    }

    #[test]
    fn advance_round_trips_through_spread_starts() {
        // A mid-spread page round-trips to its spread start, not to itself.
        for len in [1, 2, 3, 6, 9, 10] {
            for &double in &[false, true] {
                for page in 0..len {
                    for &forward in &[false, true] {
                        let stepped = advance(page, forward, double, len);
                        let back = advance(stepped, !forward, double, len);
                        assert_eq!(back, spread_for(page, double, len).first);
                    }
                }
            }
        }
    }

    #[test]
    fn advance_tiny_books() {
        assert_eq!(advance(0, true, true, 1), 0);
        assert_eq!(advance(0, false, false, 1), 0);
        // 2 pages, double mode: cover and one lone page cycle into each
        // other.
        assert_eq!(advance(0, true, true, 2), 1);
        assert_eq!(advance(1, true, true, 2), 0);
        // A zero-length book never exists, but the math must tolerate it.
        assert_eq!(advance(0, true, true, 0), 0);
    }

    #[test]
    fn seats_follow_reading_direction() {
        let s = Spread { first: 1, second: Some(2) };
        assert_eq!(s.seats(false), [Some(1), Some(2)]);
        assert_eq!(s.seats(true), [Some(2), Some(1)]);
        let lone = Spread { first: 5, second: None };
        assert_eq!(lone.seats(false), [Some(5), None]);
        assert_eq!(lone.seats(true), [None, Some(5)]);
    }
}
