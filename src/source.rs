//! Page source: a directory or archive viewed as a natural-sorted list
//! of pages. An archive unpacks once into memory on a background
//! thread while `bytes` serves whatever the pass has filled, so archive
//! reads never block and no member is ever unpacked twice. A directory
//! keeps reading files, which the OS page cache serves.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::fs::File;
use std::io;
use std::io::BufReader;
use std::io::ErrorKind;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use eros::ErrorUnion;
use eros::IntoUnion as _;
use parking_lot::Mutex;

use crate::decode::format_from_name;
use crate::fault::BadIndex;
use crate::fault::OpenError;
use crate::fault::OpenRefusal;
use crate::fault::PageError;
use crate::fault::PanicFault;
use crate::view::PageCount;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageName {
    pub parent: Option<String>,
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

pub fn is_archive(path: &Path) -> bool {
    archive_kind(path).is_some()
}

#[derive(Debug)]
pub enum PageRead {
    Ready(Arc<[u8]>),
    NotYet,
    Failed(Arc<PageError>),
}

/// The pages one background pass fills. A page member that fails to
/// read settles as its own page's error and the pass continues. A
/// failure that ends the pass (an undecodable block, a non-page member
/// that cannot be drained, a panic, a failed thread spawn) is what its
/// unfilled pages report, while filled pages keep their results.
#[derive(Debug)]
struct Fill {
    pages: HashMap<usize, Result<Arc<[u8]>, Arc<PageError>>>,
    /// `None` while the pass runs, `Some(Ok(()))` when it ended whole, `Some(Err(reason))` when it ended early.
    pass: Option<Result<(), Arc<PageError>>>,
}

impl Fill {
    fn empty() -> Self {
        Self {
            pages: HashMap::new(),
            pass: None,
        }
    }

    fn resolve(&self, index: usize) -> PageRead {
        match self.pages.get(&index) {
            Some(Ok(bytes)) => PageRead::Ready(Arc::clone(bytes)),
            Some(Err(error)) => PageRead::Failed(Arc::clone(error)),
            None => match &self.pass {
                None => PageRead::NotYet,
                // Every listed member is inserted by the pass that
                // ended whole, so a miss names the caller, not the book.
                Some(Ok(())) => {
                    PageRead::Failed(Arc::new(ErrorUnion::new(BadIndex)))
                }
                Some(Err(fatal)) => PageRead::Failed(Arc::clone(fatal)),
            },
        }
    }
}

#[derive(Debug)]
enum Backend {
    Directory(PathBuf),
    Archive(Arc<Mutex<Fill>>),
}

#[derive(Debug)]
pub struct Source {
    names: Vec<String>,
    count: PageCount,
    backend: Backend,
    path: PathBuf,
}

impl Source {
    /// Listing is filtered to image extensions and natural-sorted.
    pub fn open(path: &Path) -> Result<Self, OpenError> {
        let metadata = match fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Err(ErrorUnion::new(OpenRefusal::NotFound(
                    path.to_path_buf(),
                )))
            }
            Err(error) => return Err(ErrorUnion::new(error)),
        };
        if metadata.is_dir() {
            return Self::open_dir(path);
        }
        match archive_kind(path) {
            Some(ArchiveKind::Zip) => Self::open_zip(path),
            Some(ArchiveKind::SevenZip) => Self::open_sevenzip(path),
            None => Err(ErrorUnion::new(OpenRefusal::Unsupported(
                path.to_path_buf(),
            ))),
        }
    }

    /// The next (`forward`) or previous sibling book of the same kind in
    /// this book's parent directory, in natural order: archives beside
    /// archives, directories beside directories. `None` at either end or
    /// when the parent holds no order to walk.
    pub fn sibling(&self, forward: bool) -> Option<PathBuf> {
        let parent = self.path.parent()?;
        let current = self.path.file_name()?;
        let archive = is_archive(&self.path);
        let mut siblings: Vec<(String, PathBuf)> = fs::read_dir(parent)
            .ok()?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|entry| {
                if archive {
                    entry.is_file() && is_archive(entry)
                } else {
                    entry.is_dir()
                }
            })
            .filter_map(|entry| {
                let name =
                    entry.file_name()?.to_string_lossy().into_owned();
                Some((name, entry))
            })
            .collect();
        siblings.sort_by(|left, right| compare_names(&left.0, &right.0));
        let position = siblings
            .iter()
            .position(|(name, _)| OsStr::new(name.as_str()) == current)?;
        let next = if forward {
            position.checked_add(1)?
        } else {
            position.checked_sub(1)?
        };
        siblings.get(next).map(|(_, sibling)| sibling.clone())
    }

    fn open_dir(dir: &Path) -> Result<Self, OpenError> {
        #[expect(
            clippy::filetype_is_file,
            reason = "the entry's own type is wanted, so symlinks stay out; `is_file()` would follow them"
        )]
        let mut names: Vec<String> = fs::read_dir(dir)
            .union()?
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_type()
                    .is_ok_and(|file_type| file_type.is_file())
            })
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| format_from_name(name).is_some())
            .collect();
        sort_page_names(&mut names);
        Self::from_names(
            Backend::Directory(dir.to_path_buf()),
            names,
            dir.to_path_buf(),
        )
    }

    /// Listing reads the central directory, so a damaged local header
    /// does not refuse the whole book: that member fails when the fill
    /// reaches it.
    fn open_zip(path: &Path) -> Result<Self, OpenError> {
        let zip = zip::ZipArchive::new(BufReader::new(
            File::open(path).union()?,
        ))
        .union()?;
        let mut listed: Vec<(String, u64)> = Vec::new();
        for entry_index in 0..zip.len() {
            let entry = zip.by_index_data(entry_index).union()?;
            if entry.is_dir() {
                continue;
            }
            let name = entry.name().union()?.into_owned();
            if format_from_name(&name).is_some() {
                listed.push((name, entry.size()));
            }
        }
        let (names, unpacked) = validated_listing(listed).union()?;
        guard_memory(unpacked, mem_available_bytes()).union()?;
        let fill = Arc::new(Mutex::new(Fill::empty()));
        let source = Self::from_names(
            Backend::Archive(Arc::clone(&fill)),
            names.clone(),
            path.to_path_buf(),
        )?;
        spawn_zip_fill(zip, names, &fill);
        Ok(source)
    }

    /// No password: a 7z with encrypted headers fails to open, and the
    /// first encrypted block ends the pass.
    fn open_sevenzip(path: &Path) -> Result<Self, OpenError> {
        let archive = sevenz_rust2::ArchiveReader::open(
            path,
            sevenz_rust2::Password::empty(),
        )
        .union()?;
        let listed: Vec<(String, u64)> = archive
            .archive()
            .files
            .iter()
            .filter(|entry| {
                !entry.is_directory()
                    && format_from_name(&entry.name).is_some()
            })
            .map(|entry| (entry.name.clone(), entry.size()))
            .collect();
        let (names, unpacked) = validated_listing(listed).union()?;
        guard_memory(unpacked, mem_available_bytes()).union()?;
        let fill = Arc::new(Mutex::new(Fill::empty()));
        let source = Self::from_names(
            Backend::Archive(Arc::clone(&fill)),
            names.clone(),
            path.to_path_buf(),
        )?;
        spawn_sevenzip_fill(archive, &names, &fill);
        Ok(source)
    }

    fn from_names(
        backend: Backend,
        names: Vec<String>,
        path: PathBuf,
    ) -> Result<Self, OpenError> {
        let Some(count) = PageCount::new(names.len()) else {
            return Err(ErrorUnion::new(OpenRefusal::Empty));
        };
        Ok(Self {
            names,
            count,
            backend,
            path,
        })
    }

    pub fn len(&self) -> PageCount {
        self.count
    }

    /// Entry name of one page. For archives this includes the inner
    /// path; for directories it is the bare filename.
    pub fn name(&self, index: usize) -> Option<&str> {
        self.names.get(index).map(String::as_str)
    }

    /// The status line's name segment: the basename plus its immediate parent
    /// directory.
    #[expect(
        clippy::indexing_slicing,
        reason = "the list is non-empty and callers stay below its length; the debug assert restates it"
    )]
    pub fn page_name(&self, index: usize) -> PageName {
        debug_assert!(
            index < self.names.len(),
            "caller clamps the page index"
        );
        let entry = &self.names[index];
        let file = entry.rsplit('/').next().unwrap_or(entry).to_owned();
        let parent = match &self.backend {
            Backend::Directory(dir) => dir
                .file_name()
                .map(|name| name.to_string_lossy().into_owned()),
            Backend::Archive(_) => {
                entry.rsplit_once('/').map(|(path, _)| {
                    path.rsplit('/').next().unwrap_or(path).to_owned()
                })
            }
        };
        PageName { parent, file }
    }

    /// The encoded bytes of one page. An archive still unpacking answers
    /// `NotYet` without waiting; the caller decides how to wait.
    pub fn bytes(&self, index: usize) -> PageRead {
        let Some(name) = self.names.get(index) else {
            return PageRead::Failed(Arc::new(ErrorUnion::new(BadIndex)));
        };
        match &self.backend {
            Backend::Directory(dir) => match fs::read(dir.join(name)) {
                Ok(bytes) => PageRead::Ready(bytes.into()),
                Err(error) => {
                    PageRead::Failed(Arc::new(ErrorUnion::new(error)))
                }
            },
            Backend::Archive(fill) => fill.lock().resolve(index),
        }
    }

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

fn sort_page_names(names: &mut [String]) {
    names.sort_by(|left, right| compare_names(left, right));
}

fn validated_listing(
    listed: Vec<(String, u64)>,
) -> Result<(Vec<String>, u64), OpenRefusal> {
    // Header sizes are claims, so the sum saturates rather than wraps
    // a liar archive past the guard.
    let unpacked_bytes: u64 = listed
        .iter()
        .map(|&(_, size)| size)
        .fold(0, u64::saturating_add);
    let mut names: Vec<String> =
        listed.into_iter().map(|(name, _)| name).collect();
    sort_page_names(&mut names);
    if let Some(name) = duplicate_name(&names) {
        return Err(OpenRefusal::Duplicate(name));
    }
    Ok((names, unpacked_bytes))
}

/// Natural order over UTF-8 names: ASCII digit runs compare numerically
/// (fewer leading zeros first); remaining text compares by code point.
/// Only identical names compare equal.
fn compare_names(left: &str, right: &str) -> Ordering {
    let (mut left, mut right) = (left.as_bytes(), right.as_bytes());
    loop {
        match (left.first(), right.first()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(&left_byte), Some(&right_byte)) => {
                if left_byte.is_ascii_digit()
                    && right_byte.is_ascii_digit()
                {
                    let (left_run, left_rest) = leading_digits(left);
                    let (right_run, right_rest) = leading_digits(right);
                    let order = compare_digit_runs(left_run, right_run);
                    if order != Ordering::Equal {
                        return order;
                    }
                    left = left_rest;
                    right = right_rest;
                } else {
                    let order = left_byte.cmp(&right_byte);
                    if order == Ordering::Equal {
                        left = left.split_at(1).1;
                        right = right.split_at(1).1;
                    } else {
                        return order;
                    }
                }
            }
        }
    }
}

fn leading_digits(bytes: &[u8]) -> (&[u8], &[u8]) {
    let end = bytes
        .iter()
        .position(|byte| !byte.is_ascii_digit())
        .unwrap_or(bytes.len());
    bytes.split_at(end)
}

fn strip_leading_zeros(digits: &[u8]) -> &[u8] {
    let start = digits
        .iter()
        .position(|&byte| byte != b'0')
        .unwrap_or(digits.len());
    digits.split_at(start).1
}

/// By value, then fewer leading zeros.
fn compare_digit_runs(left: &[u8], right: &[u8]) -> Ordering {
    let (left_value, right_value) =
        (strip_leading_zeros(left), strip_leading_zeros(right));
    let (left_zeros, right_zeros) = (
        left.len() - left_value.len(),
        right.len() - right_value.len(),
    );
    left_value
        .len()
        .cmp(&right_value.len())
        .then_with(|| left_value.cmp(right_value))
        .then_with(|| left_zeros.cmp(&right_zeros))
}

/// One member's bytes, or its read error. A failed member is drained,
/// so the next member of a solid stream decodes from its true start;
/// an independent entry discards the rest.
fn read_member(
    member: &mut (impl Read + ?Sized),
) -> Result<Arc<[u8]>, PageError> {
    let mut bytes = Vec::new();
    match member.read_to_end(&mut bytes) {
        Ok(_) => Ok(bytes.into()),
        Err(error) => {
            #[expect(
                clippy::let_underscore_must_use,
                clippy::let_underscore_untyped,
                reason = "only the drain's side effect matters; its result is irrelevant"
            )]
            let _ = member.read_to_end(&mut bytes);
            Err(ErrorUnion::new(error))
        }
    }
}

/// Fill a zip's pages in reading order. Zip members decode
/// independently, so each page settles on its own result.
fn spawn_zip_fill(
    mut zip: zip::ZipArchive<BufReader<File>>,
    names: Vec<String>,
    fill: &Arc<Mutex<Fill>>,
) {
    spawn_pass(fill, move |fill| {
        for (index, name) in names.iter().enumerate() {
            let page = match zip.by_name(name) {
                Ok(mut entry) => read_member(&mut entry).map_err(Arc::new),
                Err(error) => Err(Arc::new(ErrorUnion::new(error))),
            };
            fill.lock().pages.insert(index, page);
        }
        Ok(())
    });
}

/// Start the unpack thread for one pass body. Dropping the handle
/// detaches the thread, so quitting never waits on it. A failed spawn
/// settles the pass, so no page can wait on a thread that never ran.
fn spawn_pass(
    fill: &Arc<Mutex<Fill>>,
    body: impl FnOnce(&Arc<Mutex<Fill>>) -> Result<(), PageError>
        + Send
        + 'static,
) {
    use std::thread::Builder;

    let thread_fill = Arc::clone(fill);
    let spawned = Builder::new()
        .name("cy-unpack".into())
        .spawn(move || settle_pass(&thread_fill, body));
    if let Err(error) = spawned {
        tracing::warn!(?error, "could not start the unpack thread");
        fill.lock().pass = Some(Err(Arc::new(ErrorUnion::new(error))));
    }
}

/// Fill a 7z's pages in storage order, the only order a solid stream
/// permits: every member is consumed, page or not, before the next
/// decodes.
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
                        fill.lock().pages.insert(
                            index,
                            read_member(data).map_err(Arc::new),
                        );
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
            .map_err(ErrorUnion::new)
    });
}

fn settle_pass(
    fill: &Arc<Mutex<Fill>>,
    body: impl FnOnce(&Arc<Mutex<Fill>>) -> Result<(), PageError>,
) {
    use std::panic;
    use std::panic::AssertUnwindSafe;

    let outcome = panic::catch_unwind(AssertUnwindSafe(|| body(fill)));
    let mut state = fill.lock();
    state.pass = Some(match outcome {
        Ok(Ok(())) => Ok(()),
        Ok(Err(reason)) => {
            tracing::warn!(?reason, "unpack pass ended early");
            Err(Arc::new(reason))
        }
        Err(payload) => Err(Arc::new(ErrorUnion::new(
            PanicFault::from_payload("unpack", payload.as_ref()),
        ))),
    });
}

/// `MemAvailable` from a `/proc/meminfo` text, converted to bytes. The
/// line counts KiB. `None` when the line is missing or unparsable.
fn parse_mem_available_bytes(meminfo: &str) -> Option<u64> {
    let line = meminfo
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(kib * 1024)
}

fn mem_available_bytes() -> Option<u64> {
    parse_mem_available_bytes(&fs::read_to_string("/proc/meminfo").ok()?)
}

/// Kept free of the archive's unpacked pages: the decode book, texture
/// uploads, and the rest of the system also live in `MemAvailable`.
const MEM_RESERVE_BYTES: u64 = 1024 * 1024 * 1024;

fn guard_memory(
    unpacked: u64,
    available: Option<u64>,
) -> Result<(), OpenRefusal> {
    match available {
        Some(available)
            if unpacked > available.saturating_sub(MEM_RESERVE_BYTES) =>
        {
            Err(OpenRefusal::TooBig {
                unpacked,
                available,
            })
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
    use crate::fault::BadIndex;
    use crate::fault::DecodeRefusal;
    use crate::fault::PanicFault;
    use crate::testkit::eventually;
    use crate::testkit::eventually_failed;
    use crate::testkit::eventually_ready;
    use crate::testkit::failed_page;
    use crate::testkit::tiny_png;
    use crate::testkit::TempDir;

    use std::io::Error as IoError;
    use std::io::Write as _;

    fn failed_reason(what: &'static str) -> PageError {
        ErrorUnion::new(DecodeRefusal::Corrupt(what))
    }

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
    fn a_pages_own_result_answers_before_the_pass_verdict() {
        let mut fill = Fill::empty();
        assert!(matches!(fill.resolve(0), PageRead::NotYet));
        let good: Arc<[u8]> = Arc::from(&b"a"[..]);
        let bad = failed_page("bad page");
        fill.pages.insert(0, Ok(Arc::clone(&good)));
        fill.pages.insert(1, Err(Arc::clone(&bad)));
        assert!(
            matches!(fill.resolve(0), PageRead::Ready(bytes) if Arc::ptr_eq(&bytes, &good))
        );
        assert!(
            matches!(fill.resolve(1), PageRead::Failed(error) if Arc::ptr_eq(&error, &bad))
        );
        assert!(matches!(fill.resolve(2), PageRead::NotYet));

        let fatal = failed_page("stream broke");
        fill.pass = Some(Err(Arc::clone(&fatal)));
        assert!(
            matches!(fill.resolve(1), PageRead::Failed(error) if Arc::ptr_eq(&error, &bad))
        );
        assert!(
            matches!(fill.resolve(2), PageRead::Failed(error) if Arc::ptr_eq(&error, &fatal))
        );

        fill.pass = Some(Ok(()));
        // A whole pass inserts every listed member, so a miss after it
        // names the caller.
        assert!(
            matches!(fill.resolve(2), PageRead::Failed(error) if error.downcast_inner_ref::<BadIndex>().is_some())
        );
    }

    #[test]
    fn natural_order_compares_digit_runs_by_value() {
        // Mixed padding widths order by value, not byte-wise.
        let mut names = vec![
            "9.png".to_owned(),
            "010.png".to_owned(),
            "1.png".to_owned(),
        ];
        sort_page_names(&mut names);
        assert_eq!(names, ["1.png", "9.png", "010.png"]);
        let mut padded =
            vec!["page10.png".to_owned(), "page09.png".to_owned()];
        sort_page_names(&mut padded);
        assert_eq!(padded, ["page09.png", "page10.png"]);
        // Equal values keep a deterministic order: fewer zeros first.
        assert_eq!(compare_names("10.png", "010.png"), Ordering::Less);
        assert_eq!(compare_names("010.png", "10.png"), Ordering::Greater);
        assert_eq!(compare_names("10.png", "10.png"), Ordering::Equal);
        // Non-digits compare byte-wise; case and spaces stay significant.
        assert_eq!(
            compare_names("Page2.png", "page3.png"),
            Ordering::Less
        );
        assert_eq!(
            compare_names("page 2.png", "page2.png"),
            Ordering::Less
        );
        assert_eq!(compare_names("page.png", "page2.png"), Ordering::Less);
        assert_eq!(compare_names("b", "a"), Ordering::Greater);
        assert_eq!(compare_names("", ""), Ordering::Equal);
        assert_eq!(compare_names("", "a"), Ordering::Less);
    }

    #[test]
    fn natural_order_handles_non_ascii_prefixes() {
        assert_eq!(
            compare_names("漫画-2.jpg", "漫画-10.jpg"),
            Ordering::Less
        );
        assert_eq!(
            compare_names("第1話.cbz", "第10話.cbz"),
            Ordering::Less
        );
        // Code-point order, not pinyin.
        assert_eq!(compare_names("阿", "白"), Ordering::Greater);
        assert_eq!(
            compare_names("06-漫画.jpg", "漫画-06.jpg"),
            Ordering::Less
        );
    }

    #[test]
    fn natural_order_is_total() {
        let names = [
            "",
            "0",
            "00",
            "1",
            "01",
            "9",
            "10",
            "010",
            "A",
            "a",
            "a0",
            "a1",
            "a01",
            "b",
            "page 2",
            "page2",
            "z",
            "é",
            "第1話",
            "第10話",
            "漫画-2",
            "漫画-10",
            "阿",
            "白",
        ];
        for &left in &names {
            for &right in &names {
                let forward = compare_names(left, right);
                assert_eq!(
                    forward,
                    compare_names(right, left).reverse(),
                    "{left:?} vs {right:?}"
                );
                assert_eq!(
                    forward == Ordering::Equal,
                    left == right,
                    "{left:?} vs {right:?}"
                );
            }
        }
        for &left in &names {
            for &middle in &names {
                if compare_names(left, middle) == Ordering::Greater {
                    continue;
                }
                for &right in &names {
                    if compare_names(middle, right) == Ordering::Greater {
                        continue;
                    }
                    assert_ne!(
                        compare_names(left, right),
                        Ordering::Greater,
                        "{left:?} <= {middle:?} <= {right:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn meminfo_parsing() {
        let meminfo = "MemTotal:       100 kB\nMemAvailable:   4096 kB\nSwapTotal:      2 kB\n";
        assert_eq!(parse_mem_available_bytes(meminfo), Some(4096 * 1024));
        assert_eq!(parse_mem_available_bytes("MemTotal: 1 kB\n"), None);
        assert_eq!(
            parse_mem_available_bytes("MemAvailable: notanumber kB\n"),
            None
        );
    }

    #[test]
    fn memory_guard() {
        assert!(matches!(
            guard_memory(100, Some(50)),
            Err(OpenRefusal::TooBig {
                unpacked: 100,
                available: 50
            })
        ));
        guard_memory(50, Some(MEM_RESERVE_BYTES + 100)).unwrap();
        guard_memory(u64::MAX, None).unwrap();
        // The guard keeps a reserve free: an archive that would leave
        // less than it is refused even though it still fits.
        let available = 4 * 1024 * 1024 * 1024;
        guard_memory(available - MEM_RESERVE_BYTES, Some(available))
            .unwrap();
        #[expect(
            clippy::rest_pattern_accessible_field,
            reason = "naming the ignored field as `_` would trip unneeded_field_pattern"
        )]
        let Err(OpenRefusal::TooBig { unpacked, .. }) = guard_memory(
            available - MEM_RESERVE_BYTES + 1,
            Some(available),
        ) else {
            panic!("an archive eating the reserve must refuse");
        };
        assert_eq!(unpacked, available - MEM_RESERVE_BYTES + 1);
    }

    #[test]
    fn settle_pass_outcomes() {
        let whole = Arc::new(Mutex::new(Fill::empty()));
        settle_pass(&whole, |_| Ok(()));
        assert!(matches!(&whole.lock().pass, Some(Ok(()))));

        let broke = Arc::new(Mutex::new(Fill::empty()));
        settle_pass(&broke, |_| Err(failed_reason("stream broke")));
        let settled = broke.lock();
        let Some(Err(reason)) = &settled.pass else {
            panic!("the broken pass must settle as an error");
        };
        assert_eq!(
            reason.downcast_inner_ref::<DecodeRefusal>(),
            Some(&DecodeRefusal::Corrupt("stream broke"))
        );
        drop(settled);

        let boom = Arc::new(Mutex::new(Fill::empty()));
        settle_pass(&boom, |_| panic!("boom"));
        let PageRead::Failed(error) = boom.lock().resolve(0) else {
            panic!("a panic settles as its unfilled pages' failure");
        };
        let fault = error
            .downcast_inner_ref::<PanicFault>()
            .expect("a panic settles as a PanicFault");
        assert!(fault.message.contains("boom"));
    }

    #[test]
    fn liar_sizes_refuse() {
        let listed = vec![
            ("a.png".to_owned(), u64::MAX),
            ("b.png".to_owned(), u64::MAX),
        ];
        let (names, unpacked) =
            validated_listing(listed).expect("liar names still list");
        assert_eq!(names, ["a.png", "b.png"]);
        assert_eq!(unpacked, u64::MAX);
        #[expect(
            clippy::rest_pattern_accessible_field,
            reason = "naming the ignored field as `_` would trip unneeded_field_pattern"
        )]
        let Err(OpenRefusal::TooBig { unpacked, .. }) =
            guard_memory(unpacked, Some(4 * 1024 * 1024 * 1024))
        else {
            panic!("liar sizes must refuse against the guard");
        };
        assert_eq!(unpacked, u64::MAX);
    }

    #[test]
    fn duplicate_names_refuse() {
        let listed = vec![
            ("b.png".to_owned(), 8),
            ("a.png".to_owned(), 8),
            ("a.png".to_owned(), 8),
        ];
        assert_eq!(
            validated_listing(listed).expect_err("duplicates must refuse"),
            OpenRefusal::Duplicate(String::from("a.png"))
        );
    }

    #[test]
    fn sevenz_listing() {
        let source =
            Source::open(Path::new(SEVENZ_FIXTURE)).expect("7z opens");
        assert_eq!(source.names, ["p1.png", "p2.png"]);
        assert_eq!(
            source.page_name(0),
            PageName {
                parent: None,
                file: "p1.png".into()
            }
        );
    }

    #[test]
    fn sevenz_read_through_fill() {
        let source =
            Source::open(Path::new(SEVENZ_FIXTURE)).expect("7z opens");
        let bytes = eventually_ready(&source, 0);
        assert!(
            matches!(source.bytes(0), PageRead::Ready(repeat) if repeat.as_ref() == bytes.as_ref())
        );
        assert_eq!(bytes.get(..8), Some(&b"\x89PNG\r\n\x1a\n"[..]));
        // the decoder is the oracle for the fixture's dims
        let page = decode::decode(&bytes, "p1.png")
            .expect("fixture page decodes");
        assert_eq!((page.width(), page.height()), (40, 30));

        eventually(20, "pass never ended", || {
            (!source.unpacking()).then_some(())
        });
        assert!(matches!(source.bytes(1), PageRead::Ready(_)));
    }

    #[test]
    fn sevenz_drains_non_page_members() {
        let source = Source::open(Path::new(MIXED_SEVENZ_FIXTURE))
            .expect("7z opens");
        assert_eq!(source.len().get(), 2);
        let mut pages = Vec::new();
        for index in 0..source.len().get() {
            let name = source.name(index).expect("page listed");
            let bytes = eventually_ready(&source, index);
            let page = decode::decode(&bytes, name)
                .expect("fixture page decodes");
            assert_eq!(page.format(), decode::Format::Png);
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
        assert_eq!(
            source.names,
            ["c1.png", "c2.png", "c10.png", "cover.PNG"]
        );
        assert_eq!(source.len().get(), 4);
        assert_eq!(source.name(2), Some("c10.png"));
        assert_eq!(
            source.page_name(2),
            PageName {
                parent: Some(
                    dir.path()
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned()
                ),
                file: "c10.png".into(),
            }
        );
        assert!(!source.unpacking());
        assert!(
            matches!(source.bytes(1), PageRead::Ready(read) if read.as_ref() == tiny_png(1, 1, [7, 8, 9, 255]).as_slice())
        );
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
            for (name, color) in [
                ("b/010.png", 1_u8),
                ("b/002.png", 2),
                ("b/020.png", 3),
                ("b/deep/005.png", 4),
            ] {
                zip.start_file(name, options).unwrap();
                zip.write_all(&tiny_png(2, 2, [color, 0, 0, 255]))
                    .unwrap();
            }
            zip.start_file("b/info.txt", options).unwrap();
            zip.write_all(b"skip me").unwrap();
            zip.finish().unwrap();
        }
        let source = Source::open(&zip_path).expect("zip opens");
        assert_eq!(
            source.names,
            ["b/002.png", "b/010.png", "b/020.png", "b/deep/005.png"]
        );
        assert_eq!(
            source.page_name(1),
            PageName {
                parent: Some("b".into()),
                file: "010.png".into()
            }
        );
        assert_eq!(
            source.page_name(3),
            PageName {
                parent: Some("deep".into()),
                file: "005.png".into()
            }
        );
        // bytes(1) is b/010.png, written with color 1.
        let page = eventually_ready(&source, 1);
        assert_eq!(
            page.as_ref(),
            tiny_png(2, 2, [1, 0, 0, 255]).as_slice()
        );
        eventually(20, "pass never ended", || {
            (!source.unpacking()).then_some(())
        });
    }

    #[test]
    fn zip_corruption_stays_page_local() {
        use zip::write::SimpleFileOptions;

        let dir = TempDir::new("pagezipbad");
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
                zip.start_file(format!("{index:03}.png"), options)
                    .unwrap();
                zip.write_all(page).unwrap();
            }
            zip.finish().unwrap();
        }
        // Two corruption modes in one book: a flipped byte inside page
        // 1's stored payload (its CRC stops matching) and a damaged
        // page 2 local header (its central-directory entry stays
        // whole). Each broken member must fail alone.
        {
            let mut book = fs::read(&zip_path).unwrap();
            let needle = &*pages[1];
            let payload = book
                .windows(needle.len())
                .position(|window| window == needle)
                .expect("stored payload lies verbatim in the file");
            *book
                .get_mut(payload)
                .expect("offset points into the file") ^= 0xff;
            let mut headers = book
                .windows(4)
                .enumerate()
                .filter(|(_, bytes)| *bytes == b"PK\x03\x04")
                .map(|(offset, _)| offset);
            headers.next().expect("first local header");
            headers.next().expect("second local header");
            let third = headers.next().expect("third local header");
            *book.get_mut(third).expect("offset points into the file") ^=
                0xff;
            fs::write(&zip_path, &book).unwrap();
        }

        let source = Source::open(&zip_path)
            .expect("the central directory still opens the book");
        let failure = eventually_failed(&source, 1);
        let read_error = failure
            .downcast_inner_ref::<IoError>()
            .expect("a corrupt member settles as an io error");
        // zip reports a payload CRC mismatch as invalid data.
        assert_eq!(read_error.kind(), ErrorKind::InvalidData);
        eventually_failed(&source, 2);
        assert_eq!(
            eventually_ready(&source, 0).as_ref(),
            pages[0].as_slice()
        );
    }

    #[test]
    fn open_errors() {
        let missing = Source::open(Path::new("/nonexistent-cryslen-dir"))
            .expect_err("a missing path must refuse");
        assert!(matches!(
            missing.downcast_inner_ref::<OpenRefusal>(),
            Some(OpenRefusal::NotFound(_))
        ));
        let empty_dir = TempDir::new("pageempty");
        let empty = Source::open(empty_dir.path())
            .expect_err("an empty directory must refuse");
        assert!(matches!(
            empty.downcast_inner_ref::<OpenRefusal>(),
            Some(OpenRefusal::Empty)
        ));
        let dir = TempDir::new("pageunsup");
        dir.write("book.rar", b"not really a rar");
        let unsupported = Source::open(&dir.path().join("book.rar"))
            .expect_err("an unsupported extension must refuse");
        assert!(matches!(
            unsupported.downcast_inner_ref::<OpenRefusal>(),
            Some(OpenRefusal::Unsupported(_))
        ));
        dir.write("plain.txt", b"not a directory");
        // A path under a regular file reports io, not NotFound.
        let under_file =
            Source::open(&dir.path().join("plain.txt").join("page.png"))
                .expect_err("a path under a regular file must fail as io");
        assert!(under_file.downcast_inner_ref::<IoError>().is_some());
    }

    #[test]
    fn sibling_books_step_between_their_own_kind() {
        let parent = TempDir::new("siblings");
        let volume = |name: &str| {
            let dir = parent.path().join(name);
            fs::create_dir_all(&dir).expect("volume dir creates");
            fs::write(dir.join("p.png"), tiny_png(1, 1, [0, 0, 0, 255]))
                .expect("volume page writes");
            dir
        };
        // Natural order puts "v1" < "v2" < "v10" by digit value.
        let first = volume("v1");
        let middle = volume("v2");
        let last = volume("v10");
        fs::write(parent.path().join("notes.txt"), b"not a book")
            .expect("plain file writes");

        let book = Source::open(&middle).expect("volume opens");
        assert_eq!(
            book.sibling(true),
            Some(last),
            "forward follows natural order"
        );
        assert_eq!(
            book.sibling(false),
            Some(first.clone()),
            "backward follows natural order"
        );
        let cover = Source::open(&first).expect("volume opens");
        assert_eq!(
            cover.sibling(false),
            None,
            "the shelf's start has no earlier book"
        );

        // Archives step between archives only.
        let shelf = TempDir::new("siblings-archives");
        let write_cbz = |name: &str| {
            use zip::write::SimpleFileOptions;

            let path = shelf.path().join(name);
            let file = File::create(&path).expect("zip creates");
            let mut zip = zip::ZipWriter::new(file);
            zip.start_file("p.png", SimpleFileOptions::default())
                .expect("entry starts");
            zip.write_all(&tiny_png(1, 1, [0, 0, 0, 255]))
                .expect("entry writes");
            zip.finish().expect("zip finishes");
            path
        };
        let before = write_cbz("a.cbz");
        let after = write_cbz("b.cbz");
        fs::write(shelf.path().join("notes.txt"), b"not a book")
            .expect("plain file writes");
        let archive = Source::open(&before).expect("zip opens");
        // The volumes directory is a sibling entry of the other kind and
        // must not answer.
        assert_eq!(
            archive.sibling(true),
            Some(after),
            "archives step between archives"
        );
        assert_eq!(
            archive.sibling(false),
            None,
            "the first archive has no earlier book"
        );
    }
}
