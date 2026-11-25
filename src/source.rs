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
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;

use bytes::Bytes;
use eros::ErrorUnion;
use parking_lot::Mutex;
use sevenz_rust2::Archive;
use zip::ZipArchive;

use crate::browse::PageNames;
use crate::catalog;
use crate::decode;

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

type PageRead = Result<Bytes, Arc<ErrorUnion>>;
type ReadyHook = Arc<dyn Fn() + Send + Sync>;

struct ArchiveSlots {
    slots: Box<[OnceLock<PageRead>]>,
    on_ready: Mutex<Option<ReadyHook>>,
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
        }
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
        if let Some(slot) = self.slots.get(index)
            && slot.set(page).is_ok()
        {
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
    pub fn fill(&self, index: usize, bytes: Bytes) {
        self.0.fill(index, Ok(bytes));
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
    pub fn open(path: &Path) -> eros::Result<Self> {
        use std::io::ErrorKind;

        use crate::catalog::ArchiveKind;

        let metadata = match fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                eros::bail!("not found: {}", path.display());
            }
            Err(error) => return Err(error.into()),
        };
        if metadata.is_dir() {
            return Self::open_dir(path);
        }
        match catalog::archive_kind(path) {
            Some(ArchiveKind::Zip) => Self::open_zip(path),
            Some(ArchiveKind::SevenZip) => Self::open_sevenzip(path),
            None => eros::bail!("unsupported source: {}", path.display()),
        }
    }

    fn open_dir(dir: &Path) -> eros::Result<Self> {
        let mut names: Vec<String> = fs::read_dir(dir)?
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                decode::format_from_name(&name)?;
                catalog::entry_is_file(&entry).then_some(name)
            })
            .collect();
        sort_page_names(&mut names);
        Self::from_names(Backend::Directory(dir.to_path_buf()), names)
    }

    /// Listing reads the central directory, so a damaged local header fails
    /// only that member when materialization reaches it.
    fn open_zip(path: &Path) -> eros::Result<Self> {
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
            if decode::format_from_name(&name).is_some() {
                entries.push(ListedEntry {
                    name,
                    unpacked_bytes: entry.size(),
                    archive_index: entry_index,
                });
            }
        }
        let listing = validated_listing(entries)?;
        let (source, slots) = Self::open_archive_slots(
            listing.names,
            listing.unpacked_bytes,
        )?;
        spawn_zip_materialize(zip, listing.archive_indices, &slots);
        Ok(source)
    }

    /// No password: a 7z with encrypted headers fails to open, and the
    /// first encrypted block ends the pass.
    fn open_sevenzip(path: &Path) -> eros::Result<Self> {
        use sevenz_rust2::Password;

        let mut reader = SeekBufReader::new(
            File::open(path)?,
            ARCHIVE_READ_BUFFER_BYTES,
        );
        let password = Password::empty();
        let header_limit_kib =
            sevenzip_header_limit_kib(mem_available_bytes());
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
                    && decode::format_from_name(&entry.name).is_some()
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
        let (source, slots) =
            Self::open_archive_slots(listing.names, needed_bytes)?;
        spawn_sevenzip_materialize(
            archive,
            reader,
            plan,
            &source.names,
            &slots,
        );
        Ok(source)
    }

    fn open_archive_slots(
        names: Vec<String>,
        needed_bytes: u64,
    ) -> eros::Result<(Self, Arc<ArchiveSlots>)> {
        guard_memory(needed_bytes, mem_available_bytes())?;
        let slots = Arc::new(ArchiveSlots::new(names.len()));
        let source =
            Self::from_names(Backend::Archive(Arc::clone(&slots)), names)?;
        Ok((source, slots))
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

    #[cfg(test)]
    pub fn len(&self) -> NonZeroUsize {
        self.names.count()
    }

    /// Every page's entry name, in reading order.
    pub fn names(&self) -> &PageNames {
        &self.names
    }

    /// Entry name of one page. For archives this includes the inner
    /// path; for directories it is the bare filename.
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
                Ok(bytes) => Ok(Bytes::from(bytes)),
                Err(error) => Err(Arc::new(error.into())),
            },
            Backend::Archive(slots) => slots.get(index),
        }
    }

    pub fn is_archive(&self) -> bool {
        matches!(&self.backend, Backend::Archive(_))
    }

    pub fn is_ready(&self, index: usize) -> bool {
        match &self.backend {
            Backend::Directory(_) => true,
            Backend::Archive(slots) => slots.is_ready(index),
        }
    }

    pub fn on_ready(
        &self,
        hook: impl Fn() + Send + Sync + 'static,
    ) -> ReadyGuard<'_> {
        let slots = match &self.backend {
            Backend::Directory(_) => None,
            Backend::Archive(slots) => {
                slots.set_on_ready(Some(Arc::new(hook)));
                Some(slots.as_ref())
            }
        };
        ReadyGuard { slots }
    }
}

#[must_use]
#[derive(Debug)]
pub struct ReadyGuard<'source> {
    slots: Option<&'source ArchiveSlots>,
}

impl Drop for ReadyGuard<'_> {
    fn drop(&mut self) {
        if let Some(slots) = self.slots {
            slots.set_on_ready(None);
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
    names.sort_by(|left, right| catalog::compare_names(left, right));
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
) -> eros::Result<Bytes> {
    use std::time::Duration;
    use std::time::Instant;

    let started = Instant::now();
    let capacity = usize::try_from(expected_bytes)
        .unwrap_or(MAX_PREALLOCATED_BYTES)
        .min(MAX_PREALLOCATED_BYTES)
        .saturating_add(READ_TO_END_PROBE_BYTES);
    let mut bytes = Vec::with_capacity(capacity);
    member.read_to_end(&mut bytes)?;
    if bytes.capacity()
        > bytes.len().saturating_add(READ_TO_END_PROBE_BYTES)
    {
        bytes.shrink_to_fit();
    }
    let elapsed = started.elapsed();
    if elapsed >= Duration::from_millis(200) {
        tracing::debug!(
            bytes = bytes.len(),
            elapsed = ?elapsed,
            "slow member read"
        );
    }
    Ok(Bytes::from(bytes))
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
            let page = match zip.by_index(entry_index) {
                Ok(mut entry) => {
                    let expected_bytes = entry.size();
                    read_member(&mut entry, expected_bytes)
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
            let verdict = match catch_panic("unpack", || {
                body(&thread_slots)
            }) {
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

struct PlannedBlock {
    index: usize,
    page_count: NonZeroUsize,
}

struct SevenZipPlan {
    blocks: Vec<PlannedBlock>,
    empty_pages: Vec<usize>,
}

impl SevenZipPlan {
    fn build(
        file_block_index: &[Option<usize>],
        block_count: usize,
        page_file_indices: &[usize],
    ) -> Self {
        let mut block_page_counts = vec![0_usize; block_count];
        let mut block_order = Vec::new();
        let mut empty_pages = Vec::new();
        for (page_index, &file_index) in
            page_file_indices.iter().enumerate()
        {
            let block = file_block_index
                .get(file_index)
                .copied()
                .flatten()
                .filter(|&block_index| block_index < block_count);
            if let Some(block_index) = block
                && let Some(page_count) =
                    block_page_counts.get_mut(block_index)
            {
                if *page_count == 0 {
                    block_order.push(block_index);
                }
                *page_count += 1;
            } else {
                empty_pages.push(page_index);
            }
        }
        let blocks = block_order
            .into_iter()
            .filter_map(|index| {
                block_page_counts
                    .get(index)
                    .copied()
                    .and_then(NonZeroUsize::new)
                    .map(|page_count| PlannedBlock { index, page_count })
            })
            .collect();
        Self {
            blocks,
            empty_pages,
        }
    }

    fn decoder_peak_bytes(&self, archive: &Archive) -> u64 {
        let mut peak_bytes = 0_u64;
        for planned in &self.blocks {
            let Some(block) = archive.blocks.get(planned.index) else {
                continue;
            };
            let block_unpacked_bytes = block.get_unpack_size();
            let mut block_peak_bytes = 0_u64;
            for (coder_index, coder) in block.coders.iter().enumerate() {
                let Some(dictionary_bytes) = coder_dictionary_bytes(
                    coder.encoder_method_id(),
                    coder.properties(),
                ) else {
                    continue;
                };
                let coder_unpacked_bytes =
                    block.get_unpack_size_at_index(coder_index);
                let unpacked_bytes = if coder_unpacked_bytes > 0 {
                    coder_unpacked_bytes
                } else {
                    block_unpacked_bytes
                };
                block_peak_bytes = block_peak_bytes.saturating_add(
                    lzma_growth_peak_bytes(
                        dictionary_bytes,
                        unpacked_bytes,
                    ),
                );
            }
            peak_bytes = peak_bytes.max(block_peak_bytes);
        }
        peak_bytes
    }
}

#[expect(
    clippy::little_endian_bytes,
    reason = "LZMA properties store the dictionary size little-endian by format specification"
)]
fn coder_dictionary_bytes(
    method_id: &[u8],
    properties: &[u8],
) -> Option<u64> {
    use sevenz_rust2::EncoderMethod;

    const LZMA2_MAX_DICTIONARY_CODE: u8 = 40;
    const LZMA2_MANTISSA_BASE: u64 = 2;
    const LZMA2_BASE_EXPONENT: u32 = 11;
    const LZMA_DICTIONARY_OFFSET: usize = 1;
    const LZMA_DICTIONARY_BYTES: usize = size_of::<u32>();

    match method_id {
        EncoderMethod::ID_LZMA2 => match properties.first().copied()? {
            LZMA2_MAX_DICTIONARY_CODE => Some(u64::from(u32::MAX)),
            code @ 0..LZMA2_MAX_DICTIONARY_CODE => {
                let mantissa = LZMA2_MANTISSA_BASE | u64::from(code & 1);
                let exponent = u32::from(code >> 1) + LZMA2_BASE_EXPONENT;
                Some(mantissa << exponent)
            }
            _ => None,
        },
        EncoderMethod::ID_LZMA => {
            let raw = properties
                .get(LZMA_DICTIONARY_OFFSET..)
                ?.first_chunk::<LZMA_DICTIONARY_BYTES>()?;
            Some(u64::from(u32::from_le_bytes(*raw)))
        }
        _ => None,
    }
}

/// Models `lzma-rust2`'s `LzDecoder::grow_slow` doubling reallocation
/// where old and new dictionary buffers coexist.
fn lzma_growth_peak_bytes(
    dictionary_bytes: u64,
    unpacked_bytes: u64,
) -> u64 {
    const MIN_GROWTH_BYTES: u64 = 64 * 1024;

    let target_bytes = unpacked_bytes.min(dictionary_bytes);
    if target_bytes == 0 {
        return 0;
    }
    let mut previous_bytes = 0_u64;
    let mut capacity_bytes = 0_u64;
    while capacity_bytes < target_bytes {
        previous_bytes = capacity_bytes;
        capacity_bytes = capacity_bytes
            .saturating_mul(2)
            .max(MIN_GROWTH_BYTES)
            .min(dictionary_bytes);
    }
    previous_bytes.saturating_add(capacity_bytes)
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
            slots.fill(page_index, Ok(Bytes::new()));
        }
        let password = Password::empty();
        for planned in plan.blocks {
            let mut remaining_pages = planned.page_count.get();
            let decoder = BlockDecoder::new(
                BLOCK_DECODER_THREADS,
                planned.index,
                &archive,
                &password,
                &mut reader,
            );
            decoder.for_each_entries(&mut |entry, data| {
                let page_index = (!entry.is_directory())
                    .then(|| page_by_name.get(&entry.name).copied())
                    .flatten();
                if let Some(index) = page_index {
                    slots.fill(
                        index,
                        read_member(data, entry.size()).map_err(Arc::new),
                    );
                    remaining_pages = remaining_pages.saturating_sub(1);
                }
                Ok(remaining_pages > 0)
            })?;
        }
        Ok(())
    });
}

/// `MemAvailable` from a `/proc/meminfo` text, converted to bytes. The
/// line counts KiB. `None` when the line is missing or unparsable.
fn parse_mem_available_bytes(meminfo: &str) -> Option<u64> {
    let line = meminfo
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

fn mem_available_bytes() -> Option<u64> {
    parse_mem_available_bytes(&fs::read_to_string("/proc/meminfo").ok()?)
}

/// Kept free in `MemAvailable` for in-flight decodes, GPU textures, the
/// preview store, and the rest of the system.
const MEM_RESERVE_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_SEVENZIP_HEADER_KIB: u64 = 128 * 1024;

fn sevenzip_header_limit_kib(available_bytes: Option<u64>) -> usize {
    const KIB_SHIFT: u32 = 10;

    let budget_kib = available_bytes.map_or(
        MAX_SEVENZIP_HEADER_KIB,
        |available_bytes| {
            (available_bytes.saturating_sub(MEM_RESERVE_BYTES)
                >> KIB_SHIFT)
                .min(MAX_SEVENZIP_HEADER_KIB)
        },
    );
    usize::try_from(budget_kib).unwrap_or(usize::MAX)
}

fn guard_memory(
    needed_bytes: u64,
    available_bytes: Option<u64>,
) -> eros::Result<()> {
    match available_bytes {
        Some(available_bytes)
            if needed_bytes
                > available_bytes.saturating_sub(MEM_RESERVE_BYTES) =>
        {
            eros::bail!(
                "archive too big: {needed_bytes} bytes needed, {available_bytes} bytes available"
            );
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

    use crate::catalog;
    use crate::catalog::PageName;
    use crate::testkit;

    use std::cmp::Ordering;
    use std::io::Write as _;

    use tempfile::TempDir;
    use zip::ZipWriter;

    /// LZMA solid 7z with two PNGs and a non-page text member in the
    /// same block, built by 7zz.
    const MIXED_SEVENZ_FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mixed_pages.7z"
    );

    fn failed_page(message: &'static str) -> Arc<ErrorUnion> {
        use eros::MsgError;

        Arc::new(ErrorUnion::new(MsgError::from_static(message)))
    }

    #[test]
    fn panic_message_reads_payload_kinds() {
        assert_eq!(panic_message(&String::from("boom")), "boom");
        assert_eq!(panic_message(&"static boom"), "static boom");
        assert_eq!(panic_message(&7_u8), "non-string panic payload");
    }

    #[test]
    fn a_failed_pass_answers_only_unfilled_pages() {
        let good = Bytes::from_static(b"a");
        let bad = failed_page("bad page");
        let fatal = failed_page("stream broke");
        let slots = ArchiveSlots::new(3);
        slots.fill(0, Ok(good.clone()));
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

        // A whole pass that missed a page rejects unfilled slots.
        let whole = ArchiveSlots::new(2);
        whole.fill(0, Ok(good));
        whole.finish(Ok(()));
        let missed = whole.get(1).expect_err("missed slot must fail");
        assert!(missed.to_string().contains("missing archive entry"));
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
        assert_eq!(
            catalog::compare_names("10.png", "010.png"),
            Ordering::Less
        );
        assert_eq!(
            catalog::compare_names("010.png", "10.png"),
            Ordering::Greater
        );
        assert_eq!(
            catalog::compare_names("10.png", "10.png"),
            Ordering::Equal
        );
        // Non-digits compare byte-wise; case and spaces stay significant.
        assert_eq!(
            catalog::compare_names("Page2.png", "page3.png"),
            Ordering::Less
        );
        assert_eq!(
            catalog::compare_names("page 2.png", "page2.png"),
            Ordering::Less
        );
        assert_eq!(
            catalog::compare_names("page.png", "page2.png"),
            Ordering::Less
        );
        assert_eq!(catalog::compare_names("b", "a"), Ordering::Greater);
        assert_eq!(catalog::compare_names("", ""), Ordering::Equal);
        assert_eq!(catalog::compare_names("", "a"), Ordering::Less);
    }

    #[test]
    fn natural_order_handles_non_ascii_prefixes() {
        assert_eq!(
            catalog::compare_names("漫画-2.jpg", "漫画-10.jpg"),
            Ordering::Less
        );
        assert_eq!(
            catalog::compare_names("第1話.cbz", "第10話.cbz"),
            Ordering::Less
        );
        // Code-point order for non-ASCII text.
        assert_eq!(catalog::compare_names("阿", "白"), Ordering::Greater);
        assert_eq!(
            catalog::compare_names("06-漫画.jpg", "漫画-06.jpg"),
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
                let forward = catalog::compare_names(left, right);
                assert_eq!(
                    forward,
                    catalog::compare_names(right, left).reverse(),
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
                if catalog::compare_names(left, middle)
                    == Ordering::Greater
                {
                    continue;
                }
                for &right in &names {
                    if catalog::compare_names(middle, right)
                        == Ordering::Greater
                    {
                        continue;
                    }
                    assert_ne!(
                        catalog::compare_names(left, right),
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
        assert_eq!(
            parse_mem_available_bytes(&format!(
                "MemAvailable: {} kB\n",
                u64::MAX
            )),
            None
        );
    }

    #[test]
    fn memory_guard() {
        assert!(guard_memory(100, Some(50)).is_err());
        guard_memory(50, Some(MEM_RESERVE_BYTES + 100)).unwrap();
        guard_memory(u64::MAX, None).unwrap();
        // The guard keeps a reserve free: an archive that would leave
        // less than it is refused even though it still fits.
        let available = 4 * 1024 * 1024 * 1024;
        guard_memory(available - MEM_RESERVE_BYTES, Some(available))
            .unwrap();
        assert!(
            guard_memory(
                available - MEM_RESERVE_BYTES + 1,
                Some(available),
            )
            .is_err()
        );

        let max_header_kib =
            usize::try_from(MAX_SEVENZIP_HEADER_KIB).unwrap();
        assert_eq!(sevenzip_header_limit_kib(None), max_header_kib);
        assert_eq!(
            sevenzip_header_limit_kib(Some(available)),
            max_header_kib
        );
        assert_eq!(
            sevenzip_header_limit_kib(Some(
                MEM_RESERVE_BYTES + 4 * 1024 * 1024
            )),
            4 * 1024
        );
        assert_eq!(sevenzip_header_limit_kib(Some(MEM_RESERVE_BYTES)), 0);
    }

    #[test]
    fn a_panicking_materializer_answers_unfilled_pages() {
        let slots = Arc::new(ArchiveSlots::new(2));
        spawn_materialize(&slots, |slots| {
            slots.fill(0, Ok(Bytes::from_static(b"a")));
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
        let (release_tx, release_rx) = mpsc::channel();
        spawn_materialize(&slots, move |slots| {
            slots.fill(0, Ok(Bytes::from_static(b"first")));
            release_rx.recv().expect("gate releases slot 1");
            slots.fill(1, Ok(Bytes::from_static(b"second")));
            Ok(())
        });

        assert_eq!(slots.get(0).expect("slot 0 ready").as_ref(), b"first");
        release_tx.send(()).expect("release slot 1");
        assert_eq!(
            slots.get(1).expect("slot 1 ready").as_ref(),
            b"second"
        );
    }

    #[test]
    fn liar_sizes_refuse() {
        let entries = vec![
            ListedEntry {
                name: "a.png".to_owned(),
                unpacked_bytes: u64::MAX,
                archive_index: 0,
            },
            ListedEntry {
                name: "b.png".to_owned(),
                unpacked_bytes: u64::MAX,
                archive_index: 1,
            },
        ];
        let listing =
            validated_listing(entries).expect("liar names still list");
        assert_eq!(listing.names, ["a.png", "b.png"]);
        assert_eq!(listing.unpacked_bytes, u64::MAX);
        assert!(
            guard_memory(
                listing.unpacked_bytes,
                Some(4 * 1024 * 1024 * 1024)
            )
            .is_err()
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
    fn sevenzip_plan_orders_blocks_and_skips_unused() {
        let file_block_index =
            [Some(2), Some(0), None, Some(1), Some(2), None];
        let page_file_indices = [3, 2, 0, 4];
        let plan =
            SevenZipPlan::build(&file_block_index, 3, &page_file_indices);
        let blocks: Vec<(usize, usize)> = plan
            .blocks
            .iter()
            .map(|planned| (planned.index, planned.page_count.get()))
            .collect();
        assert_eq!(blocks, [(1, 1), (2, 2)]);
        assert_eq!(plan.empty_pages, [1]);
    }

    #[test]
    fn lzma_growth_peak_bytes_estimates_realloc_spike() {
        use sevenz_rust2::EncoderMethod;

        const KIB: u64 = 1024;
        const MIB: u64 = 1024 * 1024;

        assert_eq!(
            coder_dictionary_bytes(EncoderMethod::ID_LZMA2, &[0]),
            Some(4 * KIB)
        );
        assert_eq!(
            coder_dictionary_bytes(EncoderMethod::ID_LZMA2, &[37]),
            Some(1536 * MIB)
        );
        assert_eq!(
            coder_dictionary_bytes(EncoderMethod::ID_LZMA2, &[40]),
            Some(u64::from(u32::MAX))
        );
        assert_eq!(
            coder_dictionary_bytes(EncoderMethod::ID_LZMA2, &[41]),
            None
        );
        assert_eq!(
            coder_dictionary_bytes(
                EncoderMethod::ID_LZMA,
                &[0x5d, 0, 0, 0x01, 0]
            ),
            Some(64 * KIB)
        );

        assert_eq!(
            lzma_growth_peak_bytes(1536 * MIB, 1_119_464_490),
            2560 * MIB
        );
        assert_eq!(
            lzma_growth_peak_bytes(1024 * MIB, MIB),
            512 * KIB + MIB
        );
        assert_eq!(lzma_growth_peak_bytes(16 * MIB, 147 * MIB), 24 * MIB);
    }

    #[test]
    fn sevenz_listing() {
        let source = Source::open(Path::new(testkit::SEVENZ_FIXTURE))
            .expect("7z opens");
        assert_eq!(source.names, ["p1.png", "p2.png"]);
        assert_eq!(
            catalog::page_name(&source.names, true, None, 0),
            PageName {
                parent_dir: None,
                file: "p1.png".into()
            }
        );
    }

    #[test]
    fn sevenz_materializes_once_and_drains_non_page_members() {
        use crate::decode;
        use crate::decode::Format;

        let source = Source::open(Path::new(MIXED_SEVENZ_FIXTURE))
            .expect("7z opens");
        assert_eq!(source.len().get(), 2);
        let mut pages = Vec::new();
        for index in 0..source.len().get() {
            let name = source.name(index).expect("page listed");
            let bytes = source.bytes(index).expect("page materialized");
            let page = decode::decode(&bytes, name)
                .expect("fixture page decodes");
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
        assert_eq!(source.len().get(), 4);
        assert_eq!(source.name(2), Some("c10.png"));
        let dir_name = dir
            .path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        assert_eq!(
            catalog::page_name(&source.names, false, Some(&dir_name), 2),
            PageName {
                parent_dir: Some(dir_name),
                file: "c10.png".into(),
            }
        );
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
        assert_eq!(
            catalog::page_name(&source.names, true, None, 1),
            PageName {
                parent_dir: Some("b".into()),
                file: "010.png".into()
            }
        );
        assert_eq!(
            catalog::page_name(&source.names, true, None, 3),
            PageName {
                parent_dir: Some("deep".into()),
                file: "005.png".into()
            }
        );
        // bytes(1) is b/010.png, written with color 1.
        let page = testkit::readable_page(&source, 1);
        assert_eq!(
            page.as_ref(),
            testkit::tiny_png(2, 2, [1, 0, 0, 255]).as_slice()
        );
    }

    #[test]
    fn zip_corruption_stays_page_local() {
        use std::io::Error as IoError;
        use std::io::ErrorKind;

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
        let failure = testkit::unreadable_page(&source, 1);
        let read_error = failure
            .downcast_inner_ref::<IoError>()
            .expect("a corrupt member settles as an io error");
        // zip reports a payload CRC mismatch as invalid data.
        assert_eq!(read_error.kind(), ErrorKind::InvalidData);
        testkit::unreadable_page(&source, 2);
        assert_eq!(
            testkit::readable_page(&source, 0).as_ref(),
            pages[0].as_slice()
        );
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
        // A path under a regular file reports io.
        let under_file =
            Source::open(&dir.path().join("plain.txt").join("page.png"))
                .expect_err("a path under a regular file must fail as io");
        assert!(under_file.downcast_inner_ref::<IoError>().is_some());
    }

    #[test]
    fn sibling_books_step_between_their_own_kind() {
        use crate::browse::Direction;

        let parent = TempDir::new().expect("create temp dir");
        let volume = |name: &str| {
            let dir = parent.path().join(name);
            fs::create_dir_all(&dir).expect("volume dir creates");
            fs::write(
                dir.join("p.png"),
                testkit::tiny_png(1, 1, [0, 0, 0, 255]),
            )
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
            catalog::sibling(&middle, false, Direction::Forward),
            Some(last),
            "forward follows natural order"
        );
        assert_eq!(
            catalog::sibling(&middle, false, Direction::Backward),
            Some(first.clone()),
            "backward follows natural order"
        );
        assert_eq!(
            catalog::sibling(&first, false, Direction::Backward),
            None,
            "the shelf's start has no earlier book"
        );

        // Archives step between archives only.
        let shelf = TempDir::new().expect("create temp dir");
        let write_cbz = |name: &str| {
            use zip::write::SimpleFileOptions;

            let path = shelf.path().join(name);
            let file = File::create(&path).expect("zip creates");
            let mut zip = ZipWriter::new(file);
            zip.start_file("p.png", SimpleFileOptions::default())
                .expect("entry starts");
            zip.write_all(&testkit::tiny_png(1, 1, [0, 0, 0, 255]))
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
            catalog::sibling(&before, true, Direction::Forward),
            Some(after),
            "archives step between archives"
        );
        assert_eq!(
            catalog::sibling(&before, true, Direction::Backward),
            None,
            "the first archive has no earlier book"
        );
    }
}
