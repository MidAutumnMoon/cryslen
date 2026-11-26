//! Page source: a directory or archive as a natural-sorted page list.
//! Archives unpack progressively into per-page slots on a background
//! thread; [`Source::bytes`] waits on the requested page's slot.

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::fs;
use std::fs::File;
use std::io;
use std::io::BufReader;
use std::io::ErrorKind;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use eros::ErrorUnion;
use parking_lot::Mutex;
use sevenz_rust2::Archive;
use zip::ZipArchive;

use crate::book::catalog;
use crate::book::catalog::PageNames;
use crate::book::sevenz;
use crate::book::sevenz::SevenZipPlan;
use crate::image::decode::codec;

const ARCHIVE_READ_BUFFER_BYTES: usize = 64 * 1024;
const MAX_PREALLOCATED_BYTES: usize = 16 * 1024 * 1024;
const READ_TO_END_PROBE_BYTES: usize = 32;

/// Preserves `BufReader`'s buffer across adjacent `SeekFrom::Start` seeks
/// from `zip` and `sevenz-rust2`.
struct SeekBufReader<Reader> {
    inner: BufReader<Reader>,
    position: u64,
}

impl<Reader: Read> SeekBufReader<Reader> {
    fn new(inner: Reader, capacity: usize) -> Self {
        Self {
            inner: BufReader::with_capacity(capacity, inner),
            position: 0,
        }
    }
}

impl<Reader: Read> Read for SeekBufReader<Reader> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let bytes_read = self.inner.read(buf)?;
        self.position = self
            .position
            .saturating_add(u64::try_from(bytes_read).unwrap_or(u64::MAX));
        Ok(bytes_read)
    }
}

fn relative_offset(from: u64, to: u64) -> Option<i64> {
    if to >= from {
        i64::try_from(to - from).ok()
    } else {
        i64::try_from(from - to).ok().and_then(i64::checked_neg)
    }
}

impl<Reader: Read + Seek> Seek for SeekBufReader<Reader> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let relative = match pos {
            SeekFrom::Start(target) => {
                relative_offset(self.position, target)
                    .map(|delta| (delta, target))
            }
            SeekFrom::Current(offset) => self
                .position
                .checked_add_signed(offset)
                .map(|target| (offset, target)),
            SeekFrom::End(_) => None,
        };
        if let Some((delta, target)) = relative {
            self.inner.seek_relative(delta)?;
            self.position = target;
            return Ok(target);
        }
        self.position = self.inner.seek(pos)?;
        Ok(self.position)
    }

    fn stream_position(&mut self) -> io::Result<u64> {
        Ok(self.position)
    }
}

struct CancellableRead<'cancel, Reader: ?Sized> {
    inner: &'cancel mut Reader,
    cancel: &'cancel AtomicBool,
}

impl<Reader: Read + ?Sized> Read for CancellableRead<'_, Reader> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(io::Error::new(
                ErrorKind::Interrupted,
                "canceled",
            ));
        }
        self.inner.read(buf)
    }
}

pub fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic payload")
}

pub fn catch_panic<Output>(
    operation: &str,
    body: impl FnOnce() -> eros::Result<Output>,
) -> eros::Result<Output> {
    use std::panic;
    use std::panic::AssertUnwindSafe;

    match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(result) => result,
        Err(payload) => eros::bail!(
            "{operation} panicked: {}",
            panic_message(payload.as_ref())
        ),
    }
}

pub type PageRead = Result<Arc<[u8]>, Arc<ErrorUnion>>;
pub type ReadyHook = Arc<dyn Fn() + Send + Sync>;

struct ArchiveSlots {
    slots: Box<[OnceLock<PageRead>]>,
    on_ready: Mutex<Option<ReadyHook>>,
    cancel: AtomicBool,
    resident_bytes: AtomicU64,
}

impl Debug for ArchiveSlots {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArchiveSlots")
            .field("count", &self.slots.len())
            .finish_non_exhaustive()
    }
}

impl ArchiveSlots {
    fn new(count: usize) -> Self {
        use std::iter;

        let slots = iter::repeat_with(OnceLock::new).take(count).collect();
        Self {
            slots,
            on_ready: Mutex::new(None),
            cancel: AtomicBool::new(false),
            resident_bytes: AtomicU64::new(0),
        }
    }

    fn is_canceled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    fn is_ready(&self, index: usize) -> bool {
        self.slots
            .get(index)
            .is_none_or(|slot| slot.get().is_some())
    }

    fn set_on_ready(&self, hook: Option<ReadyHook>) {
        *self.on_ready.lock() = hook;
    }

    fn notify_ready(&self) {
        let hook = self.on_ready.lock().clone();
        if let Some(hook) = hook {
            hook();
        }
    }

    fn get(&self, index: usize) -> PageRead {
        self.slots.get(index).map_or_else(
            || Err(Arc::new(eros::error!("no such page"))),
            |slot| slot.wait().clone(),
        )
    }

    fn fill(&self, index: usize, page: PageRead) {
        let added = page.as_ref().map_or(0, |bytes| bytes.len() as u64);
        if let Some(slot) = self.slots.get(index)
            && slot.set(page).is_ok()
        {
            if added > 0 {
                self.resident_bytes.fetch_add(added, Ordering::Relaxed);
            }
            self.notify_ready();
        }
    }

    fn finish(&self, verdict: Result<(), Arc<ErrorUnion>>) {
        let fallback = verdict.err().unwrap_or_else(|| {
            Arc::new(eros::error!("missing archive entry"))
        });
        let mut filled_any = false;
        for slot in &self.slots {
            if slot.get().is_none()
                && slot.set(Err(Arc::clone(&fallback))).is_ok()
            {
                filled_any = true;
            }
        }
        if filled_any {
            self.notify_ready();
        }
    }
}

#[cfg(test)]
pub struct TestArchiveSlots(Arc<ArchiveSlots>);

#[cfg(test)]
impl TestArchiveSlots {
    pub fn fill(&self, index: usize, bytes: Arc<[u8]>) {
        self.0.fill(index, Ok(bytes));
    }
}

enum UnpackTask {
    Zip {
        zip: ZipArchive<SeekBufReader<File>>,
        indices: Vec<usize>,
    },
    SevenZip {
        archive: Box<Archive>,
        reader: SeekBufReader<File>,
        plan: SevenZipPlan,
    },
}

/// Listed book ready to start unpacking once adopted.
pub struct ListedBook {
    source: Source,
    unpack: Option<UnpackTask>,
}

impl ListedBook {
    #[must_use]
    pub fn names(&self) -> &PageNames {
        self.source.names()
    }

    #[must_use]
    pub fn is_archive(&self) -> bool {
        self.source.is_archive()
    }

    #[must_use]
    pub fn start(self) -> Source {
        if let (Some(unpack), Backend::Archive(slots)) =
            (self.unpack, &self.source.backend)
        {
            match unpack {
                UnpackTask::Zip { zip, indices } => {
                    spawn_zip_materialize(zip, indices, slots);
                }
                UnpackTask::SevenZip {
                    archive,
                    reader,
                    plan,
                } => {
                    spawn_sevenzip_materialize(
                        *archive,
                        reader,
                        plan,
                        &self.source.names,
                        slots,
                    );
                }
            }
        }
        self.source
    }
}

#[derive(Debug)]
enum Backend {
    Directory(PathBuf),
    Archive(Arc<ArchiveSlots>),
}

#[derive(Debug)]
pub struct Source {
    names: PageNames,
    backend: Backend,
}

impl Source {
    #[cfg(test)]
    pub fn open(path: &Path) -> eros::Result<Self> {
        Ok(Self::open_listed(path, 0)?.start())
    }

    pub fn open_listed(
        path: &Path,
        reclaimable_bytes: u64,
    ) -> eros::Result<ListedBook> {
        use crate::book::catalog::ArchiveKind;

        let metadata = match fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                eros::bail!("not found: {}", path.display());
            }
            Err(error) => return Err(error.into()),
        };
        if metadata.is_dir() {
            let source = Self::open_dir(path)?;
            return Ok(ListedBook {
                source,
                unpack: None,
            });
        }
        match catalog::archive_kind(path) {
            Some(ArchiveKind::Zip) => {
                Self::open_zip_listed(path, reclaimable_bytes)
            }
            Some(ArchiveKind::SevenZip) => {
                Self::open_sevenzip_listed(path, reclaimable_bytes)
            }
            None => eros::bail!("unsupported source: {}", path.display()),
        }
    }

    fn open_dir(dir: &Path) -> eros::Result<Self> {
        let mut names: Vec<String> = fs::read_dir(dir)?
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                codec::format_from_name(&name)?;
                catalog::entry_is_file(&entry).then_some(name)
            })
            .collect();
        names.sort_by(|left, right| catalog::compare_names(left, right));
        Self::from_names(Backend::Directory(dir.to_path_buf()), names)
    }

    /// Listing reads the central directory, so a damaged local header fails
    /// only that member when materialization reaches it.
    fn open_zip_listed(
        path: &Path,
        reclaimable_bytes: u64,
    ) -> eros::Result<ListedBook> {
        let reader = SeekBufReader::new(
            File::open(path)?,
            ARCHIVE_READ_BUFFER_BYTES,
        );
        let zip = ZipArchive::new(reader)?;
        let mut entries = Vec::new();
        for entry_index in 0..zip.len() {
            let entry = zip.by_index_data(entry_index)?;
            if entry.is_dir() {
                continue;
            }
            let name = entry.name()?.into_owned();
            if codec::format_from_name(&name).is_some() {
                entries.push(ListedEntry {
                    name,
                    unpacked_bytes: entry.size(),
                    archive_index: entry_index,
                });
            }
        }
        let listing = validated_listing(entries)?;
        let source = Self::open_archive_slots(
            listing.names,
            listing.unpacked_bytes,
            reclaimable_bytes,
        )?;
        Ok(ListedBook {
            source,
            unpack: Some(UnpackTask::Zip {
                zip,
                indices: listing.archive_indices,
            }),
        })
    }

    /// No password: a 7z with encrypted headers fails to open, and the
    /// first encrypted block ends the pass.
    fn open_sevenzip_listed(
        path: &Path,
        reclaimable_bytes: u64,
    ) -> eros::Result<ListedBook> {
        use sevenz_rust2::Password;

        let mut reader = SeekBufReader::new(
            File::open(path)?,
            ARCHIVE_READ_BUFFER_BYTES,
        );
        let password = Password::empty();
        let available = sevenz::mem_available_bytes()
            .map(|bytes| bytes.saturating_add(reclaimable_bytes));
        let header_limit_kib =
            sevenz::sevenzip_header_limit_kib(available);
        let archive = Archive::read_with_memory_limit(
            &mut reader,
            &password,
            header_limit_kib,
        )?;
        let entries: Vec<ListedEntry> = archive
            .files
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                !entry.is_directory()
                    && codec::format_from_name(&entry.name).is_some()
            })
            .map(|(file_index, entry)| ListedEntry {
                name: entry.name.clone(),
                unpacked_bytes: entry.size(),
                archive_index: file_index,
            })
            .collect();
        let listing = validated_listing(entries)?;
        let plan = SevenZipPlan::build(
            &archive.stream_map.file_block_index,
            archive.blocks.len(),
            &listing.archive_indices,
        );
        let needed_bytes = listing
            .unpacked_bytes
            .saturating_add(plan.decoder_peak_bytes(&archive));
        let source = Self::open_archive_slots(
            listing.names,
            needed_bytes,
            reclaimable_bytes,
        )?;
        Ok(ListedBook {
            source,
            unpack: Some(UnpackTask::SevenZip {
                archive: Box::new(archive),
                reader,
                plan,
            }),
        })
    }

    fn open_archive_slots(
        names: Vec<String>,
        needed_bytes: u64,
        reclaimable_bytes: u64,
    ) -> eros::Result<Self> {
        sevenz::guard_memory(
            needed_bytes,
            sevenz::mem_available_bytes(),
            reclaimable_bytes,
        )?;
        let slots = Arc::new(ArchiveSlots::new(names.len()));
        Self::from_names(Backend::Archive(slots), names)
    }

    #[cfg(test)]
    pub fn from_test_slots(
        names: Vec<String>,
    ) -> eros::Result<(Self, TestArchiveSlots)> {
        let slots = Arc::new(ArchiveSlots::new(names.len()));
        let source =
            Self::from_names(Backend::Archive(Arc::clone(&slots)), names)?;
        Ok((source, TestArchiveSlots(slots)))
    }

    fn from_names(
        backend: Backend,
        names: Vec<String>,
    ) -> eros::Result<Self> {
        let Some(names) = PageNames::new(names) else {
            eros::bail!("no image pages found");
        };
        Ok(Self { names, backend })
    }

    /// Every page's entry name, in reading order.
    #[must_use]
    pub fn names(&self) -> &PageNames {
        &self.names
    }

    /// Entry name of one page. For archives this includes the inner
    /// path; for directories it is the bare filename.
    #[must_use]
    pub fn name(&self, index: usize) -> Option<&str> {
        self.names.get(index).map(String::as_str)
    }

    /// The encoded bytes of one page. For an archive, waits until that
    /// page's slot is filled.
    pub fn bytes(&self, index: usize) -> PageRead {
        let Some(name) = self.names.get(index) else {
            return Err(Arc::new(eros::error!("no such page")));
        };
        match &self.backend {
            Backend::Directory(dir) => match fs::read(dir.join(name)) {
                Ok(bytes) => Ok(Arc::from(bytes)),
                Err(error) => Err(Arc::new(error.into())),
            },
            Backend::Archive(slots) => slots.get(index),
        }
    }

    #[must_use]
    pub fn is_archive(&self) -> bool {
        matches!(&self.backend, Backend::Archive(_))
    }

    #[must_use]
    pub fn is_ready(&self, index: usize) -> bool {
        match &self.backend {
            Backend::Directory(_) => true,
            Backend::Archive(slots) => slots.is_ready(index),
        }
    }

    pub fn set_on_ready(&self, hook: Option<ReadyHook>) {
        if let Backend::Archive(slots) = &self.backend {
            slots.set_on_ready(hook);
        }
    }

    pub fn cancel(&self) {
        if let Backend::Archive(slots) = &self.backend {
            slots.cancel();
        }
    }

    #[must_use]
    pub fn archive_resident_bytes(&self) -> u64 {
        match &self.backend {
            Backend::Directory(_) => 0,
            Backend::Archive(slots) => {
                slots.resident_bytes.load(Ordering::Relaxed)
            }
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

struct ListedEntry {
    name: String,
    unpacked_bytes: u64,
    archive_index: usize,
}

#[derive(Debug)]
struct ArchiveListing {
    names: Vec<String>,
    archive_indices: Vec<usize>,
    unpacked_bytes: u64,
}

fn validated_listing(
    mut entries: Vec<ListedEntry>,
) -> eros::Result<ArchiveListing> {
    // Header sizes are claims; saturating addition keeps the sum within
    // the guard even for a liar archive.
    let unpacked_bytes: u64 = entries
        .iter()
        .map(|entry| entry.unpacked_bytes)
        .fold(0, u64::saturating_add);
    entries.sort_by(|left, right| {
        catalog::compare_names(&left.name, &right.name)
    });
    let (names, archive_indices): (Vec<_>, Vec<_>) = entries
        .into_iter()
        .map(|entry| (entry.name, entry.archive_index))
        .unzip();
    if let Some(name) = duplicate_name(&names) {
        eros::bail!("duplicate entry name: {name}");
    }
    Ok(ArchiveListing {
        names,
        archive_indices,
        unpacked_bytes,
    })
}

fn read_member(
    member: &mut (impl Read + ?Sized),
    expected_bytes: u64,
    cancel: &AtomicBool,
) -> eros::Result<Arc<[u8]>> {
    use std::time::Duration;
    use std::time::Instant;

    const SLOW_MEMBER_READ_THRESHOLD: Duration =
        Duration::from_millis(200);

    let started = Instant::now();
    let capacity = usize::try_from(expected_bytes)
        .unwrap_or(MAX_PREALLOCATED_BYTES)
        .min(MAX_PREALLOCATED_BYTES)
        .saturating_add(READ_TO_END_PROBE_BYTES);
    let mut bytes = Vec::with_capacity(capacity);
    let mut reader = CancellableRead {
        inner: member,
        cancel,
    };
    reader.read_to_end(&mut bytes)?;
    if bytes.capacity()
        > bytes.len().saturating_add(READ_TO_END_PROBE_BYTES)
    {
        bytes.shrink_to_fit();
    }
    let elapsed = started.elapsed();
    if elapsed >= SLOW_MEMBER_READ_THRESHOLD {
        tracing::debug!(
            bytes = bytes.len(),
            elapsed = ?elapsed,
            "slow member read"
        );
    }
    Ok(Arc::from(bytes))
}

fn spawn_zip_materialize(
    mut zip: ZipArchive<SeekBufReader<File>>,
    entry_indices: Vec<usize>,
    slots: &Arc<ArchiveSlots>,
) {
    spawn_materialize(slots, move |slots| {
        for (page_index, entry_index) in
            entry_indices.into_iter().enumerate()
        {
            if slots.is_canceled() {
                break;
            }
            let page = match zip.by_index(entry_index) {
                Ok(mut entry) => {
                    let expected_bytes = entry.size();
                    read_member(&mut entry, expected_bytes, &slots.cancel)
                        .map_err(Arc::new)
                }
                Err(error) => Err(Arc::new(error.into())),
            };
            slots.fill(page_index, page);
        }
        Ok(())
    });
}

/// The thread detaches when its handle drops, so quitting proceeds
/// immediately; a failed spawn answers every page with the spawn error.
fn spawn_materialize(
    slots: &Arc<ArchiveSlots>,
    body: impl FnOnce(&ArchiveSlots) -> eros::Result<()> + Send + 'static,
) {
    use std::thread::Builder;
    use std::time::Instant;

    use rustfs_mimalloc::MiMalloc;

    let thread_slots = Arc::clone(slots);
    let spawned =
        Builder::new().name("cy-unpack".into()).spawn(move || {
            let started = Instant::now();
            let verdict =
                match catch_panic("unpack", || body(&thread_slots)) {
                    Ok(()) => Ok(()),
                    Err(reason) => {
                        tracing::warn!(%reason, "unpack pass ended early");
                        Err(Arc::new(reason))
                    }
                };
            tracing::debug!(
                elapsed = ?started.elapsed(),
                ok = verdict.is_ok(),
                "unpack pass ended"
            );
            thread_slots.finish(verdict);
            MiMalloc::collect(true);
            MiMalloc::arenas_purge();
        });
    if let Err(error) = spawned {
        tracing::warn!(?error, "could not start the unpack thread");
        let error: Arc<ErrorUnion> = Arc::new(error.into());
        slots.finish(Err(error));
    }
}

fn spawn_sevenzip_materialize(
    archive: Archive,
    mut reader: SeekBufReader<File>,
    plan: SevenZipPlan,
    names: &[String],
    slots: &Arc<ArchiveSlots>,
) {
    use std::collections::HashMap;

    use sevenz_rust2::BlockDecoder;
    use sevenz_rust2::Password;

    const BLOCK_DECODER_THREADS: u32 = 1;

    let page_by_name: HashMap<String, usize> = names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    spawn_materialize(slots, move |slots| {
        for page_index in plan.empty_pages {
            slots.fill(page_index, Ok(Arc::from([])));
        }
        let password = Password::empty();
        for planned in plan.blocks {
            if slots.is_canceled() {
                break;
            }
            let mut remaining_pages = planned.page_count.get();
            let decoder = BlockDecoder::new(
                BLOCK_DECODER_THREADS,
                planned.index,
                &archive,
                &password,
                &mut reader,
            );
            decoder.for_each_entries(&mut |entry, data| {
                if slots.is_canceled() {
                    return Ok(false);
                }
                let page_index = (!entry.is_directory())
                    .then(|| page_by_name.get(&entry.name).copied())
                    .flatten();
                if let Some(index) = page_index {
                    slots.fill(
                        index,
                        read_member(data, entry.size(), &slots.cancel)
                            .map_err(Arc::new),
                    );
                    remaining_pages = remaining_pages.saturating_sub(1);
                }
                Ok(remaining_pages > 0 && !slots.is_canceled())
            })?;
        }
        Ok(())
    });
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
    use crate::testkit;

    use tempfile::TempDir;
    use zip::ZipWriter;

    const MIXED_SEVENZ_FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mixed_pages.7z"
    );

    fn failed_page(message: &'static str) -> Arc<ErrorUnion> {
        use eros::MsgError;

        Arc::new(ErrorUnion::new(MsgError::from_static(message)))
    }

    #[test]
    fn a_failed_pass_answers_only_unfilled_pages() {
        let good: Arc<[u8]> = Arc::from(b"a".as_slice());
        let bad = failed_page("bad page");
        let fatal = failed_page("stream broke");
        let slots = ArchiveSlots::new(3);
        slots.fill(0, Ok(Arc::clone(&good)));
        slots.fill(1, Err(Arc::clone(&bad)));
        slots.finish(Err(Arc::clone(&fatal)));
        assert!(
            matches!(slots.get(0), Ok(bytes) if bytes.as_ptr() == good.as_ptr() && bytes.len() == good.len())
        );
        assert!(
            matches!(slots.get(1), Err(error) if Arc::ptr_eq(&error, &bad))
        );
        assert!(
            matches!(slots.get(2), Err(error) if Arc::ptr_eq(&error, &fatal))
        );

        let whole = ArchiveSlots::new(2);
        whole.fill(0, Ok(good));
        whole.finish(Ok(()));
        let missed = whole.get(1).expect_err("missed slot must fail");
        assert!(missed.to_string().contains("missing archive entry"));
    }

    #[test]
    fn a_panicking_materializer_answers_unfilled_pages() {
        let slots = Arc::new(ArchiveSlots::new(2));
        spawn_materialize(&slots, |slots| {
            slots.fill(0, Ok(Arc::from(b"a".as_slice())));
            panic!("boom");
        });
        assert_eq!(slots.get(0).unwrap().as_ref(), b"a");
        let Err(error) = slots.get(1) else {
            panic!("the unfilled page takes the panic verdict");
        };
        assert!(error.to_string().contains("boom"));
    }

    #[test]
    fn archive_slots_serve_filled_pages_before_pass_ends() {
        use std::sync::mpsc;

        let slots = Arc::new(ArchiveSlots::new(2));
        let (release_sender, release_receiver) = mpsc::channel();
        spawn_materialize(&slots, move |slots| {
            slots.fill(0, Ok(Arc::from(b"first".as_slice())));
            release_receiver.recv().expect("gate releases slot 1");
            slots.fill(1, Ok(Arc::from(b"second".as_slice())));
            Ok(())
        });

        assert_eq!(slots.get(0).expect("slot 0 ready").as_ref(), b"first");
        release_sender.send(()).expect("release slot 1");
        assert_eq!(
            slots.get(1).expect("slot 1 ready").as_ref(),
            b"second"
        );
    }

    #[test]
    fn duplicate_names_refuse() {
        let entries = vec![
            ListedEntry {
                name: "b.png".to_owned(),
                unpacked_bytes: 8,
                archive_index: 0,
            },
            ListedEntry {
                name: "a.png".to_owned(),
                unpacked_bytes: 8,
                archive_index: 1,
            },
            ListedEntry {
                name: "a.png".to_owned(),
                unpacked_bytes: 8,
                archive_index: 2,
            },
        ];
        let error = validated_listing(entries)
            .expect_err("duplicates must refuse");
        assert!(error.to_string().contains("a.png"));
    }

    #[test]
    fn seek_buf_reader_preserves_buffer_for_in_window_seeks() {
        use std::io::Cursor;

        const BUFFER_BYTES: usize = 64;

        struct CountingReader {
            inner: Cursor<Vec<u8>>,
            reads: usize,
            seeks: usize,
        }

        impl Read for CountingReader {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                self.reads += 1;
                self.inner.read(buf)
            }
        }

        impl Seek for CountingReader {
            fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
                self.seeks += 1;
                self.inner.seek(pos)
            }
        }

        let data: Vec<u8> = (0..=u8::MAX).collect();
        let mut reader = SeekBufReader::new(
            CountingReader {
                inner: Cursor::new(data.clone()),
                reads: 0,
                seeks: 0,
            },
            BUFFER_BYTES,
        );

        let mut head = [0_u8; 8];
        reader.read_exact(&mut head).expect("read initial slice");
        assert_eq!(Some(head.as_slice()), data.get(0..8));
        assert_eq!(reader.stream_position().expect("current position"), 8);

        let mut slice = [0_u8; 4];
        for (seek_to, expected_position) in [
            (SeekFrom::Start(32), 32),
            (SeekFrom::Start(16), 16),
            (SeekFrom::Current(-8), 12),
        ] {
            assert_eq!(
                reader.seek(seek_to).expect("in-window seek"),
                expected_position
            );
            reader.read_exact(&mut slice).expect("read in-window slice");
            let offset =
                usize::try_from(expected_position).expect("offset fits");
            assert_eq!(
                Some(slice.as_slice()),
                data.get(offset..offset + slice.len())
            );
        }
        let underlying = reader.inner.get_ref();
        assert_eq!((underlying.reads, underlying.seeks), (1, 0));

        for (seek_to, expected_position) in [
            (SeekFrom::Start(200), 200),
            (SeekFrom::End(-4), 252),
            (SeekFrom::Current(-4), 252),
        ] {
            assert_eq!(
                reader.seek(seek_to).expect("out-of-window seek"),
                expected_position
            );
            reader.read_exact(&mut slice).expect("read after seek");
            let offset =
                usize::try_from(expected_position).expect("offset fits");
            assert_eq!(
                Some(slice.as_slice()),
                data.get(offset..offset + slice.len())
            );
        }
    }

    #[test]
    fn sevenz_materializes_once_and_drains_non_page_members() {
        use crate::image::pixels::Format;

        let source = Source::open(Path::new(MIXED_SEVENZ_FIXTURE))
            .expect("7z opens");
        let page_count = source.names().count().get();
        assert_eq!(page_count, 2);
        let mut pages = Vec::new();
        for index in 0..page_count {
            let name = source.name(index).expect("page listed");
            let bytes = source.bytes(index).expect("page materialized");
            let page =
                codec::decode(&bytes, name).expect("fixture page decodes");
            assert_eq!(page.format(), Format::Png);
            pages.push(bytes);
        }
        let [first, second] = pages.as_slice() else {
            panic!("two pages listed");
        };
        assert_ne!(first.as_ref(), second.as_ref());
    }

    #[test]
    fn dir_listing_and_read() {
        let dir = TempDir::new().expect("create temp dir");
        fs::write(
            dir.path().join("c10.png"),
            testkit::tiny_png(1, 1, [0, 0, 0, 255]),
        )
        .expect("write c10.png");
        fs::write(
            dir.path().join("c2.png"),
            testkit::tiny_png(1, 1, [7, 8, 9, 255]),
        )
        .expect("write c2.png");
        fs::write(
            dir.path().join("c1.png"),
            testkit::tiny_png(1, 1, [0, 0, 0, 255]),
        )
        .expect("write c1.png");
        fs::write(
            dir.path().join("cover.PNG"),
            testkit::tiny_png(1, 1, [0, 0, 0, 255]),
        )
        .expect("write cover.PNG");
        fs::write(dir.path().join("notes.txt"), b"not a page")
            .expect("write notes.txt");
        let source = Source::open(dir.path()).expect("dir opens");
        assert_eq!(
            source.names,
            ["c1.png", "c2.png", "c10.png", "cover.PNG"]
        );
        assert_eq!(source.names().count().get(), 4);
        assert_eq!(source.name(2), Some("c10.png"));
        assert!(!source.is_archive());
        let expected = testkit::tiny_png(1, 1, [7, 8, 9, 255]);
        assert!(
            matches!(source.bytes(1), Ok(read) if read.as_ref() == expected.as_slice())
        );
    }

    #[cfg(unix)]
    #[test]
    fn dir_listing_follows_symlinks_and_skips_broken_entries() {
        use std::os::unix::fs as unix_fs;

        let target_dir = TempDir::new().expect("create temp dir");
        let target_png = testkit::tiny_png(2, 2, [9, 8, 7, 255]);
        fs::write(target_dir.path().join("real.png"), &target_png)
            .expect("write real.png");
        let nested_dir = target_dir.path().join("subdir");
        fs::create_dir_all(&nested_dir).expect("nested dir creates");

        let book = TempDir::new().expect("create temp dir");
        unix_fs::symlink(
            target_dir.path().join("real.png"),
            book.path().join("page.png"),
        )
        .expect("file symlink creates");
        unix_fs::symlink(
            target_dir.path().join("missing.png"),
            book.path().join("broken.png"),
        )
        .expect("broken symlink creates");
        unix_fs::symlink(&nested_dir, book.path().join("subdir.png"))
            .expect("dir symlink creates");

        let source = Source::open(book.path()).expect("dir opens");
        assert_eq!(source.names, ["page.png"]);
        assert_eq!(
            source.bytes(0).expect("symlinked page reads").as_ref(),
            target_png.as_slice()
        );
    }

    #[test]
    fn zip_listing_and_read() {
        use std::io::Write as _;

        use zip::CompressionMethod;
        use zip::write::SimpleFileOptions;

        let dir = TempDir::new().expect("create temp dir");
        let zip_path = dir.path().join("book.cbz");
        {
            let file = File::create(&zip_path).unwrap();
            let mut zip = ZipWriter::new(file);
            let options = SimpleFileOptions::default()
                .compression_method(CompressionMethod::Stored);
            for (name, color) in [
                ("b/010.png", 1_u8),
                ("b/002.png", 2),
                ("b/020.png", 3),
                ("b/deep/005.png", 4),
            ] {
                zip.start_file(name, options).unwrap();
                zip.write_all(&testkit::tiny_png(
                    2,
                    2,
                    [color, 0, 0, 255],
                ))
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
        let page = testkit::readable_page(&source, 1);
        assert_eq!(
            page.as_ref(),
            testkit::tiny_png(2, 2, [1, 0, 0, 255]).as_slice()
        );
    }

    #[test]
    fn zip_corruption_stays_page_local() {
        use std::io::Error as IoError;
        use std::io::Write as _;

        use zip::CompressionMethod;
        use zip::write::SimpleFileOptions;

        let dir = TempDir::new().expect("create temp dir");
        let zip_path = dir.path().join("book.cbz");
        let pages = [
            testkit::tiny_png(2, 2, [1, 0, 0, 255]),
            testkit::tiny_png(2, 2, [2, 0, 0, 255]),
            testkit::tiny_png(2, 2, [3, 0, 0, 255]),
        ];
        {
            let file = File::create(&zip_path).unwrap();
            let mut zip = ZipWriter::new(file);
            let options = SimpleFileOptions::default()
                .compression_method(CompressionMethod::Stored);
            for (index, page) in pages.iter().enumerate() {
                zip.start_file(format!("{index:03}.png"), options)
                    .unwrap();
                zip.write_all(page).unwrap();
            }
            zip.finish().unwrap();
        }
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
        let failure = testkit::unreadable_page(&source, 1);
        let read_error = failure
            .downcast_inner_ref::<IoError>()
            .expect("a corrupt member settles as an io error");
        assert_eq!(read_error.kind(), ErrorKind::InvalidData);
        testkit::unreadable_page(&source, 2);
        assert_eq!(
            testkit::readable_page(&source, 0).as_ref(),
            pages[0].as_slice()
        );
    }

    #[test]
    fn cancel_stops_unpack_early() {
        use std::io::Write as _;

        use zip::CompressionMethod;
        use zip::write::SimpleFileOptions;

        let dir = TempDir::new().expect("create temp dir");
        let zip_path = dir.path().join("book.cbz");
        {
            let file = File::create(&zip_path).unwrap();
            let mut zip = ZipWriter::new(file);
            let options = SimpleFileOptions::default()
                .compression_method(CompressionMethod::Stored);
            for index in 0..8 {
                zip.start_file(format!("{index:03}.png"), options)
                    .unwrap();
                zip.write_all(&testkit::tiny_png(
                    2,
                    2,
                    [index, 0, 0, 255],
                ))
                .unwrap();
            }
            zip.finish().unwrap();
        }
        let listed = Source::open_listed(&zip_path, 0).expect("listed");
        listed.source.cancel();
        let source = listed.start();
        let _error = testkit::unreadable_page(&source, 7);
    }

    #[test]
    fn open_errors() {
        use std::io::Error as IoError;

        let missing = Source::open(Path::new("/nonexistent-cryslen-dir"))
            .expect_err("a missing path must refuse");
        assert!(missing.to_string().contains("not found"));
        let empty_dir = TempDir::new().expect("create temp dir");
        let empty = Source::open(empty_dir.path())
            .expect_err("an empty directory must refuse");
        assert!(empty.to_string().contains("no image pages found"));
        let dir = TempDir::new().expect("create temp dir");
        fs::write(dir.path().join("book.rar"), b"not really a rar")
            .expect("write book.rar");
        let unsupported = Source::open(&dir.path().join("book.rar"))
            .expect_err("an unsupported extension must refuse");
        assert!(unsupported.to_string().contains("unsupported source"));
        fs::write(dir.path().join("plain.txt"), b"not a directory")
            .expect("write plain.txt");
        let under_file =
            Source::open(&dir.path().join("plain.txt").join("page.png"))
                .expect_err("a path under a regular file must fail as io");
        assert!(under_file.downcast_inner_ref::<IoError>().is_some());
    }
}
