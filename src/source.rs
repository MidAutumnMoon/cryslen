//! Page source: a directory or archive as a natural-sorted page list.
//! An archive materializes once on a background thread; `wait_ready`
//! gates `bytes`, which is total after it returns.

use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io;
use std::io::BufReader;
use std::io::ErrorKind;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use eros::ErrorUnion;
use eros::IntoUnion as _;

use crate::browse::PageCount;
use crate::catalog;
use crate::catalog::ArchiveKind;
use crate::decode::format_from_name;
use crate::fault::BadIndex;
use crate::fault::OpenError;
use crate::fault::OpenRefusal;
use crate::fault::PageError;
use crate::fault::PanicFault;

type PageRead = Result<Arc<[u8]>, Arc<PageError>>;
type PageSet = Vec<PageRead>;

#[derive(Debug)]
enum Backend {
    Directory(PathBuf),
    Archive { page_set_rx: Receiver<PageSet> },
}

#[derive(Debug)]
pub struct Source {
    names: Vec<String>,
    count: PageCount,
    backend: Backend,
    /// Filled once by `wait_ready`; `bytes` reads it for archives.
    pages: OnceLock<PageSet>,
}

impl Source {
    pub fn open(path: &Path) -> Result<Self, OpenError> {
        let metadata = match fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Err(ErrorUnion::new(OpenRefusal::NotFound(
                    path.to_path_buf(),
                )));
            }
            Err(error) => return Err(ErrorUnion::new(error)),
        };
        if metadata.is_dir() {
            return Self::open_dir(path);
        }
        match catalog::archive_kind(path) {
            Some(ArchiveKind::Zip) => Self::open_zip(path),
            Some(ArchiveKind::SevenZip) => Self::open_sevenzip(path),
            None => Err(ErrorUnion::new(OpenRefusal::Unsupported(
                path.to_path_buf(),
            ))),
        }
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
        Self::from_names(Backend::Directory(dir.to_path_buf()), names)
    }

    /// Listing reads the central directory, so a damaged local header fails
    /// only that member when materialization reaches it.
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
        let (page_set_tx, page_set_rx) = crossbeam_channel::bounded(1);
        let source = Self::from_names(
            Backend::Archive { page_set_rx },
            names.clone(),
        )?;
        spawn_zip_materialize(zip, names, &page_set_tx);
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
        let (page_set_tx, page_set_rx) = crossbeam_channel::bounded(1);
        let source = Self::from_names(
            Backend::Archive { page_set_rx },
            names.clone(),
        )?;
        spawn_sevenzip_materialize(archive, &names, &page_set_tx);
        Ok(source)
    }

    fn from_names(
        backend: Backend,
        names: Vec<String>,
    ) -> Result<Self, OpenError> {
        let Some(count) = PageCount::new(names.len()) else {
            return Err(ErrorUnion::new(OpenRefusal::Empty));
        };
        Ok(Self {
            names,
            count,
            backend,
            pages: OnceLock::new(),
        })
    }

    pub fn len(&self) -> PageCount {
        self.count
    }

    /// Every page's entry name, in reading order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Entry name of one page. For archives this includes the inner
    /// path; for directories it is the bare filename.
    pub fn name(&self, index: usize) -> Option<&str> {
        self.names.get(index).map(String::as_str)
    }

    /// Blocks until an archive's pages are materialized; a directory is
    /// always ready. Afterwards [`Self::bytes`] is total.
    pub fn wait_ready(&self) -> Result<(), Arc<PageError>> {
        if self.pages.get().is_some() {
            return Ok(());
        }
        let Backend::Archive { page_set_rx } = &self.backend else {
            return Ok(());
        };
        let page_set = page_set_rx.recv().map_err(|error| {
            Arc::new(ErrorUnion::new(PanicFault {
                origin: "unpack",
                message: format!(
                    "the unpack thread stopped without a page set: {error}"
                ),
            }))
        })?;
        if self.pages.set(page_set).is_err() {
            tracing::debug!("the page set was already materialized");
        }
        Ok(())
    }

    /// The encoded bytes of one page. For an archive the caller must
    /// have called [`Self::wait_ready`] first.
    pub fn bytes(&self, index: usize) -> PageRead {
        let Some(name) = self.names.get(index) else {
            return Err(Arc::new(ErrorUnion::new(BadIndex)));
        };
        match &self.backend {
            Backend::Directory(dir) => match fs::read(dir.join(name)) {
                Ok(bytes) => Ok(bytes.into()),
                Err(error) => Err(Arc::new(ErrorUnion::new(error))),
            },
            Backend::Archive {
                page_set_rx: _page_set_rx,
            } => self
                .pages
                .get()
                .and_then(|page_set| page_set.get(index))
                .cloned()
                .unwrap_or_else(|| {
                    Err(Arc::new(ErrorUnion::new(BadIndex)))
                }),
        }
    }

    pub fn is_archive(&self) -> bool {
        matches!(
            &self.backend,
            Backend::Archive {
                page_set_rx: _page_set_rx
            }
        )
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
    names.sort_by(|left, right| catalog::compare_names(left, right));
}

fn validated_listing(
    listed: Vec<(String, u64)>,
) -> Result<(Vec<String>, u64), OpenRefusal> {
    // Header sizes are claims; saturating addition keeps the sum within
    // the guard even for a liar archive.
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

/// One member's bytes, or its read error. A failed member is drained,
/// so the next member of a solid stream decodes from its true start;
/// an independent entry discards the rest.
fn read_member(
    member: &mut (impl Read + ?Sized),
) -> Result<Arc<[u8]>, PageError> {
    let started = Instant::now();
    let mut bytes = Vec::new();
    match member.read_to_end(&mut bytes) {
        Ok(_) => {
            let elapsed = started.elapsed();
            if elapsed >= Duration::from_millis(200) {
                tracing::debug!(
                    bytes = bytes.len(),
                    elapsed = ?elapsed,
                    "slow member read"
                );
            }
            Ok(bytes.into())
        }
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

/// Materialize a zip's pages in reading order.
fn spawn_zip_materialize(
    mut zip: zip::ZipArchive<BufReader<File>>,
    names: Vec<String>,
    tx: &Sender<PageSet>,
) {
    let page_count = names.len();
    spawn_materialize(page_count, tx, move |slots| {
        for (index, name) in names.iter().enumerate() {
            let page = match zip.by_name(name) {
                Ok(mut entry) => read_member(&mut entry).map_err(Arc::new),
                Err(error) => Err(Arc::new(ErrorUnion::new(error))),
            };
            if let Some(slot) = slots.get_mut(index) {
                *slot = Some(page);
            }
        }
        Ok(())
    });
}

/// The thread detaches when its handle drops, so quitting proceeds
/// immediately; a failed spawn answers every page with the spawn error.
fn spawn_materialize(
    page_count: usize,
    tx: &Sender<PageSet>,
    body: impl FnOnce(&mut [Option<PageRead>]) -> Result<(), PageError>
        + Send
        + 'static,
) {
    use std::panic;
    use std::panic::AssertUnwindSafe;
    use std::thread::Builder;

    let thread_tx = tx.clone();
    let spawned =
        Builder::new().name("cy-unpack".into()).spawn(move || {
            let started = Instant::now();
            let mut slots: Vec<Option<PageRead>> = vec![None; page_count];
            let outcome =
                panic::catch_unwind(AssertUnwindSafe(|| body(&mut slots)));
            let verdict = match outcome {
                Ok(Ok(())) => Ok(()),
                Ok(Err(reason)) => {
                    tracing::warn!(?reason, "unpack pass ended early");
                    Err(Arc::new(reason))
                }
                Err(payload) => Err(Arc::new(ErrorUnion::new(
                    PanicFault::from_payload("unpack", payload.as_ref()),
                ))),
            };
            tracing::debug!(
                elapsed = ?started.elapsed(),
                ok = verdict.is_ok(),
                "unpack pass ended"
            );
            if thread_tx.send(finish_pages(slots, verdict)).is_err() {
                tracing::debug!("the page set has no reader");
            }
        });
    if let Err(error) = spawned {
        tracing::warn!(?error, "could not start the unpack thread");
        let error: Arc<PageError> = Arc::new(ErrorUnion::new(error));
        let page_set = vec![Err(Arc::clone(&error)); page_count];
        if tx.send(page_set).is_err() {
            tracing::debug!("the page set has no reader");
        }
    }
}

/// An unfilled page takes the pass verdict, or `BadIndex` if the pass
/// ended whole.
fn finish_pages(
    slots: Vec<Option<PageRead>>,
    verdict: Result<(), Arc<PageError>>,
) -> PageSet {
    let fallback = verdict
        .err()
        .unwrap_or_else(|| Arc::new(ErrorUnion::new(BadIndex)));
    slots
        .into_iter()
        .map(|slot| slot.unwrap_or_else(|| Err(Arc::clone(&fallback))))
        .collect()
}

/// Materialize a 7z's pages in storage order, the only order a solid
/// stream permits.
fn spawn_sevenzip_materialize(
    mut reader: sevenz_rust2::ArchiveReader<File>,
    names: &[String],
    tx: &Sender<PageSet>,
) {
    let by_name: HashMap<String, usize> = names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    let page_count = names.len();
    spawn_materialize(page_count, tx, move |slots| {
        reader
            .for_each_entries(|entry, data| {
                match by_name.get(&entry.name) {
                    Some(&index) => {
                        if let Some(slot) = slots.get_mut(index) {
                            *slot =
                                Some(read_member(data).map_err(Arc::new));
                        }
                    }
                    None => {
                        // A solid block yields its later members only
                        // after this one is consumed, so non-pages are drained.
                        io::copy(data, &mut io::sink())?;
                    }
                }
                Ok(true)
            })
            .map_err(ErrorUnion::new)
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

/// Kept free of the archive's unpacked pages: the worker's in-flight
/// decode, the app's page textures, the preview store, and the rest of
/// the system also live in `MemAvailable`.
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
    use crate::catalog::compare_names;
    use crate::catalog::page_name;
    use crate::catalog::sibling;
    use crate::catalog::PageName;
    use crate::decode;
    use crate::fault::BadIndex;
    use crate::fault::PanicFault;
    use crate::testkit::failed_page;
    use crate::testkit::readable_page;
    use crate::testkit::tiny_png;
    use crate::testkit::unreadable_page;
    use crate::testkit::TempDir;

    use std::cmp::Ordering;
    use std::io::Error as IoError;
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
    fn a_failed_pass_answers_only_unfilled_pages() {
        let good: Arc<[u8]> = Arc::from(&b"a"[..]);
        let bad = failed_page("bad page");
        let fatal = failed_page("stream broke");
        let pages = finish_pages(
            vec![
                Some(Ok(Arc::clone(&good))),
                Some(Err(Arc::clone(&bad))),
                None,
            ],
            Err(Arc::clone(&fatal)),
        );
        let [first, second, third] = pages.as_slice() else {
            panic!("three page slots");
        };
        assert!(matches!(first, Ok(bytes) if Arc::ptr_eq(bytes, &good)));
        assert!(matches!(second, Err(error) if Arc::ptr_eq(error, &bad)));
        assert!(matches!(third, Err(error) if Arc::ptr_eq(error, &fatal)));

        // A whole pass that missed a page reports `BadIndex`.
        let whole =
            finish_pages(vec![Some(Ok(Arc::clone(&good))), None], Ok(()));
        let [_, missing] = whole.as_slice() else {
            panic!("two page slots");
        };
        assert!(
            matches!(missing, Err(error) if error.downcast_inner_ref::<BadIndex>().is_some())
        );
    }

    #[test]
    fn natural_order_compares_digit_runs_by_value() {
        // Mixed padding widths order by numeric value.
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
        // Code-point order for non-ASCII text.
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
    fn a_panicking_materializer_answers_unfilled_pages() {
        let (tx, rx) = crossbeam_channel::bounded(1);
        spawn_materialize(2, &tx, |slots| {
            if let Some(slot) = slots.get_mut(0) {
                *slot = Some(Ok(Arc::from(&b"a"[..])));
            }
            panic!("boom");
        });
        let pages = rx.recv().expect("the materializer sends a page set");
        let [first, second] = pages.as_slice() else {
            panic!("two page slots");
        };
        assert!(first.is_ok());
        let Err(error) = second else {
            panic!("the unfilled page takes the panic verdict");
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
            page_name(&source.names, true, None, 0),
            PageName {
                parent_dir: None,
                file: "p1.png".into()
            }
        );
    }

    #[test]
    fn sevenz_materializes_once_and_drains_non_page_members() {
        let source = Source::open(Path::new(MIXED_SEVENZ_FIXTURE))
            .expect("7z opens");
        source.wait_ready().expect("7z materializes");
        assert_eq!(source.len().get(), 2);
        let mut pages = Vec::new();
        for index in 0..source.len().get() {
            let name = source.name(index).expect("page listed");
            let bytes = source.bytes(index).expect("page materialized");
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
        let dir_name = dir
            .path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        assert_eq!(
            page_name(&source.names, false, Some(&dir_name), 2),
            PageName {
                parent_dir: Some(dir_name),
                file: "c10.png".into(),
            }
        );
        assert!(!source.is_archive());
        assert!(
            matches!(source.bytes(1), Ok(read) if read.as_ref() == tiny_png(1, 1, [7, 8, 9, 255]).as_slice())
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
            page_name(&source.names, true, None, 1),
            PageName {
                parent_dir: Some("b".into()),
                file: "010.png".into()
            }
        );
        assert_eq!(
            page_name(&source.names, true, None, 3),
            PageName {
                parent_dir: Some("deep".into()),
                file: "005.png".into()
            }
        );
        // bytes(1) is b/010.png, written with color 1.
        let page = readable_page(&source, 1);
        assert_eq!(
            page.as_ref(),
            tiny_png(2, 2, [1, 0, 0, 255]).as_slice()
        );
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
        let failure = unreadable_page(&source, 1);
        let read_error = failure
            .downcast_inner_ref::<IoError>()
            .expect("a corrupt member settles as an io error");
        // zip reports a payload CRC mismatch as invalid data.
        assert_eq!(read_error.kind(), ErrorKind::InvalidData);
        unreadable_page(&source, 2);
        assert_eq!(
            readable_page(&source, 0).as_ref(),
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
        // A path under a regular file reports io.
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

        assert_eq!(
            sibling(&middle, false, true),
            Some(last),
            "forward follows natural order"
        );
        assert_eq!(
            sibling(&middle, false, false),
            Some(first.clone()),
            "backward follows natural order"
        );
        assert_eq!(
            sibling(&first, false, false),
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
        // The volumes directory is a sibling entry of the other kind and
        // must not answer.
        assert_eq!(
            sibling(&before, true, true),
            Some(after),
            "archives step between archives"
        );
        assert_eq!(
            sibling(&before, true, false),
            None,
            "the first archive has no earlier book"
        );
    }
}
