//! Shared thumbnail store (`$XDG_CACHE_HOME/cryslen/thumbs-v1.redb`).
//! Opened once per app as `ThumbStore`; each open book creates a
//! lightweight `ThumbHandle` handle shared between UI reads and worker writes.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::ErrorKind;
use std::ops::RangeInclusive;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use parking_lot::Mutex;
use redb::Builder;
use redb::ConcurrencyMode;
use redb::Database;
use redb::ReadOnlyTable;
use redb::ReadTransaction;
use redb::ReadableDatabase as _;
use redb::ReadableTable as _;
use redb::Table;
use redb::TableDefinition;
use redb::TableError;
use redb::backends::FileBackend;

use crate::book::catalog;
use crate::image::decode::webp;
use crate::image::pixels;
use crate::image::pixels::EncodedThumb;
use crate::image::pixels::Format;
use crate::image::pixels::PageInfo;
use crate::image::pixels::THUMB_EDGE;
use crate::image::pixels::ThumbImage;

const THUMB_PACK_CAP_BYTES: u64 = 512 * 1024 * 1024;
const THUMB_PACK_PRUNE_NUMERATOR: u64 = 4;
const THUMB_PACK_PRUNE_DENOMINATOR: u64 = 5;
const FLUSH_ENTRY_THRESHOLD: usize = 32;

const THUMB_DATABASE_FILENAME: &str = "thumbs-v1.redb";
const THUMB_DATABASE_CACHE_BYTES: usize = 8 * 1024 * 1024;
const THUMB_DATABASE_FILE_MODE: u32 = 0o600;

type StoredThumb<'bytes> = (u32, u32, u8, &'bytes [u8]);

const BOOKS_TABLE: TableDefinition<'_, [u8; 16], (u64, u64)> =
    TableDefinition::new("books");
const THUMBS_TABLE: TableDefinition<'_, ([u8; 16], u32), StoredThumb<'_>> =
    TableDefinition::new("thumbs");

/// Shared `redb` handle kept open for the lifetime of the app.
#[derive(Clone)]
pub struct ThumbStore {
    database: Arc<Database>,
}

impl ThumbStore {
    pub fn open() -> Option<Self> {
        let root = default_cache_dir()?;
        Self::open_dir(&root)
    }

    #[cfg(test)]
    pub fn open_at(root: &Path) -> Option<Self> {
        Self::open_dir(root)
    }

    fn open_dir(root: &Path) -> Option<Self> {
        if make_dir(root).is_none() {
            tracing::warn!(root = %root.display(), "thumb pack disabled");
            return None;
        }
        let database_path = root.join(THUMB_DATABASE_FILENAME);
        let database = open_or_recreate(&database_path)?;
        Some(Self {
            database: Arc::new(database),
        })
    }

    pub fn open_book(
        &self,
        path: &Path,
        archive: bool,
        names: &[String],
    ) -> Option<ThumbHandle> {
        let key = book_key(path, archive, names)?;
        self.open_book_key(key, names.len())
    }

    #[cfg(test)]
    pub fn open_book_with_key(
        &self,
        book_key: [u8; 16],
        page_count: usize,
    ) -> Option<ThumbHandle> {
        self.open_book_key(book_key, page_count)
    }

    fn open_book_key(
        &self,
        book_key: [u8; 16],
        page_count: usize,
    ) -> Option<ThumbHandle> {
        let (present, existing_last_used) =
            scan_book(&self.database, book_key, page_count)
                .inspect_err(|error| {
                    tracing::warn!(%error, "thumb pack scan failed");
                })
                .ok()?;
        let warm = present.iter().filter(|&&hit| hit).count();
        tracing::debug!(warm, total = page_count, "thumb pack opened");
        let now_secs = now_unix_secs();
        if let Some(last_used) = existing_last_used
            && last_used != now_secs
            && let Err(error) =
                touch_book_last_used(&self.database, book_key, now_secs)
        {
            tracing::debug!(%error, "thumb pack touch failed");
        }
        Some(ThumbHandle {
            database: Arc::clone(&self.database),
            book_key,
            state: Mutex::new(ThumbState {
                present,
                pending: BTreeMap::new(),
                pruned_once: false,
            }),
            flush_lock: Mutex::new(()),
        })
    }
}

struct ThumbState {
    present: Vec<bool>,
    pending: BTreeMap<usize, EncodedThumb>,
    pruned_once: bool,
}

/// Per-book view into [`ThumbStore`]. Lock order: `flush_lock` before `state`.
pub struct ThumbHandle {
    database: Arc<Database>,
    book_key: [u8; 16],
    state: Mutex<ThumbState>,
    flush_lock: Mutex<()>,
}

impl ThumbHandle {
    #[must_use]
    pub fn has(&self, page: usize) -> bool {
        let state = self.state.lock();
        state.pending.contains_key(&page)
            || state.present.get(page).copied().unwrap_or(false)
    }

    #[must_use]
    pub fn load(&self, page: usize) -> Option<ThumbImage> {
        let pending = {
            let state = self.state.lock();
            if let Some(pending) = state.pending.get(&page) {
                Some(pending.clone())
            } else if state.present.get(page).copied().unwrap_or(false) {
                None
            } else {
                return None;
            }
        };
        if let Some(pending) = pending {
            return decode_thumb_image(
                pending.page_dims,
                pending.format,
                &pending.webp,
            );
        }
        let page_u32 = u32::try_from(page).ok()?;
        let read_transaction = self.database.begin_read().ok()?;
        let table = read_transaction.open_table(THUMBS_TABLE).ok()?;
        let entry = table.get((self.book_key, page_u32)).ok()??;
        let (page_width, page_height, format_tag, webp) = entry.value();
        let format = Format::from_repr(format_tag)?;
        let page_dims = (page_width, page_height);
        pixels::admit_dims(
            page_dims.0,
            page_dims.1,
            "thumb page dimensions",
        )
        .ok()?;
        decode_thumb_image(page_dims, format, webp)
    }

    pub fn record(&self, page: usize, thumb: EncodedThumb) {
        self.record_inner(page, thumb, false);
    }

    pub fn record_overwrite(&self, page: usize, thumb: EncodedThumb) {
        self.record_inner(page, thumb, true);
    }

    fn record_inner(
        &self,
        page: usize,
        thumb: EncodedThumb,
        overwrite: bool,
    ) {
        let should_flush = {
            let mut state = self.state.lock();
            if page >= state.present.len() {
                return;
            }
            let already = state.pending.contains_key(&page)
                || state.present.get(page).copied().unwrap_or(false);
            if !overwrite && already {
                return;
            }
            state.pending.insert(page, thumb);
            state.pending.len() >= FLUSH_ENTRY_THRESHOLD
        };
        if should_flush {
            self.flush();
        }
    }

    pub fn flush(&self) {
        if self.state.lock().pending.is_empty() {
            return;
        }
        if let Err(error) =
            self.flush_pending(now_unix_secs(), THUMB_PACK_CAP_BYTES)
        {
            tracing::warn!(%error, "thumb pack flush failed");
        }
    }

    fn flush_pending(
        &self,
        now_secs: u64,
        cap_bytes: u64,
    ) -> Result<(), redb::Error> {
        let _flush_guard = self.flush_lock.lock();
        let (items, pruned_once): (Vec<(usize, EncodedThumb)>, bool) = {
            let state = self.state.lock();
            if state.pending.is_empty() {
                return Ok(());
            }
            (
                state
                    .pending
                    .iter()
                    .map(|(&page, thumb)| (page, thumb.clone()))
                    .collect(),
                state.pruned_once,
            )
        };

        let write_transaction = self.database.begin_write()?;
        {
            let mut books_table =
                write_transaction.open_table(BOOKS_TABLE)?;
            let mut thumbs_table =
                write_transaction.open_table(THUMBS_TABLE)?;

            let mut book_bytes =
                books_table.get(self.book_key)?.map_or(0, |existing| {
                    let (_last_used_secs, existing_bytes) =
                        existing.value();
                    existing_bytes
                });

            for (page, thumb) in &items {
                let Ok(page_u32) = u32::try_from(*page) else {
                    continue;
                };
                let new_bytes = thumb.webp.len() as u64;
                let previous_bytes = thumbs_table
                    .insert(
                        (self.book_key, page_u32),
                        (
                            thumb.page_dims.0,
                            thumb.page_dims.1,
                            thumb.format as u8,
                            thumb.webp.as_slice(),
                        ),
                    )?
                    .map_or(0, |previous| {
                        let (_, _, _, previous_webp) = previous.value();
                        previous_webp.len() as u64
                    });
                book_bytes = book_bytes
                    .saturating_add(new_bytes)
                    .saturating_sub(previous_bytes);
            }

            books_table.insert(self.book_key, (now_secs, book_bytes))?;

            if !pruned_once {
                prune_old_books(
                    &mut books_table,
                    &mut thumbs_table,
                    self.book_key,
                    cap_bytes,
                )?;
            }
        }
        write_transaction.commit()?;

        let mut state = self.state.lock();
        state.pruned_once = true;
        for (page, thumb) in items {
            if let Some(slot) = state.present.get_mut(page) {
                *slot = true;
            }
            if let Entry::Occupied(entry) = state.pending.entry(page)
                && *entry.get() == thumb
            {
                entry.remove();
            }
        }
        Ok(())
    }
}

impl Drop for ThumbHandle {
    fn drop(&mut self) {
        self.flush();
    }
}

fn decode_thumb_image(
    page_dims: (u32, u32),
    format: Format,
    webp_bytes: &[u8],
) -> Option<ThumbImage> {
    let (dims, rgba) = webp::decode_rgba_bounded(webp_bytes, THUMB_EDGE)
        .inspect_err(|error| {
            tracing::debug!(%error, "thumb webp decode failed");
        })
        .ok()?;
    Some(ThumbImage {
        dims,
        info: PageInfo {
            dims: page_dims,
            format,
            animated: false,
        },
        rgba,
    })
}

fn open_or_recreate(path: &Path) -> Option<Database> {
    let file = open_regular_file(path)?;
    let error = match open_and_verify(file) {
        Ok(db) => return Some(db),
        Err(error) => error,
    };
    tracing::warn!(
        path = %path.display(),
        %error,
        "recreating corrupt or outdated thumb pack"
    );
    if let Err(remove_error) = fs::remove_file(path)
        && remove_error.kind() != ErrorKind::NotFound
    {
        tracing::warn!(
            path = %path.display(),
            error = %remove_error,
            "cannot remove thumb pack"
        );
        return None;
    }
    let fresh_file = open_regular_file(path)?;
    open_and_verify(fresh_file)
        .inspect_err(|retry_error| {
            tracing::warn!(
                path = %path.display(),
                error = %retry_error,
                "thumb pack disabled after recreate"
            );
        })
        .ok()
}

fn open_regular_file(path: &Path) -> Option<File> {
    use std::os::unix::fs::OpenOptionsExt as _;

    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                tracing::warn!(
                    path = %path.display(),
                    "thumb pack path is not a regular file"
                );
                return None;
            }
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "thumb pack disabled");
            return None;
        }
    }

    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(THUMB_DATABASE_FILE_MODE)
        .open(path)
        .inspect_err(|error| {
            tracing::warn!(path = %path.display(), %error, "thumb pack disabled");
        })
        .ok()
}

fn open_and_verify(file: File) -> Result<Database, redb::Error> {
    let database = Builder::new()
        .set_cache_size(THUMB_DATABASE_CACHE_BYTES)
        .set_concurrency_mode(ConcurrencyMode::MultiWriter)
        .create_with_backend(FileBackend::new(file))?;
    {
        let read_transaction = database.begin_read()?;
        let _books = open_if_exists(&read_transaction, BOOKS_TABLE)?;
        let _thumbs = open_if_exists(&read_transaction, THUMBS_TABLE)?;
    }
    Ok(database)
}

fn scan_book(
    database: &Database,
    book_key: [u8; 16],
    page_count: usize,
) -> Result<(Vec<bool>, Option<u64>), redb::Error> {
    let mut present = vec![false; page_count];
    let read_transaction = database.begin_read()?;
    let books = open_if_exists(&read_transaction, BOOKS_TABLE)?;
    let thumbs = open_if_exists(&read_transaction, THUMBS_TABLE)?;
    let last_used = if let (Some(books), Some(thumbs)) = (books, thumbs)
        && let Some(row) = books.get(book_key)?
    {
        for entry in thumbs.range(book_thumb_range(book_key))? {
            let (key, _) = entry?;
            let (_, page_index) = key.value();
            if let Some(slot) = present.get_mut(page_index as usize) {
                *slot = true;
            }
        }
        let (last_used_secs, _) = row.value();
        Some(last_used_secs)
    } else {
        None
    };
    Ok((present, last_used))
}

fn open_if_exists<Key, Value>(
    read_transaction: &ReadTransaction,
    definition: TableDefinition<'_, Key, Value>,
) -> Result<Option<ReadOnlyTable<Key, Value>>, redb::Error>
where
    Key: redb::Key + 'static,
    Value: redb::Value + 'static,
{
    match read_transaction.open_table(definition) {
        Ok(table) => Ok(Some(table)),
        Err(TableError::TableDoesNotExist(_)) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

const fn book_thumb_range(
    book_key: [u8; 16],
) -> RangeInclusive<([u8; 16], u32)> {
    (book_key, 0)..=(book_key, u32::MAX)
}

fn touch_book_last_used(
    database: &Database,
    book_key: [u8; 16],
    now_secs: u64,
) -> Result<(), redb::Error> {
    let write_transaction = database.begin_write()?;
    let mut updated = false;
    {
        let mut books_table = write_transaction.open_table(BOOKS_TABLE)?;
        let existing =
            books_table.get(book_key)?.map(|guard| guard.value());
        if let Some((last_used, total_bytes)) = existing
            && last_used != now_secs
        {
            books_table.insert(book_key, (now_secs, total_bytes))?;
            updated = true;
        }
    }
    if updated {
        write_transaction.commit()?;
    }
    Ok(())
}

fn prune_old_books(
    books_table: &mut Table<'_, [u8; 16], (u64, u64)>,
    thumbs_table: &mut Table<'_, ([u8; 16], u32), StoredThumb<'_>>,
    keep_key: [u8; 16],
    cap_bytes: u64,
) -> Result<(), redb::Error> {
    let mut entries = Vec::new();
    let mut total_bytes = 0_u64;
    for entry in books_table.iter()? {
        let (key_guard, value_guard) = entry?;
        let key = key_guard.value();
        let (last_used_secs, book_bytes) = value_guard.value();
        total_bytes = total_bytes.saturating_add(book_bytes);
        entries.push((last_used_secs, key, book_bytes));
    }
    if total_bytes <= cap_bytes {
        return Ok(());
    }
    entries.sort_unstable();
    let target_bytes = cap_bytes
        .checked_div(THUMB_PACK_PRUNE_DENOMINATOR)
        .map_or(cap_bytes, |fifth| {
            fifth.saturating_mul(THUMB_PACK_PRUNE_NUMERATOR)
        });
    let mut removed_count = 0_u64;
    for (_, victim_key, victim_bytes) in entries {
        if total_bytes <= target_bytes {
            break;
        }
        if victim_key == keep_key {
            continue;
        }
        thumbs_table
            .retain_in(book_thumb_range(victim_key), |_, _| false)?;
        books_table.remove(victim_key)?;
        total_bytes = total_bytes.saturating_sub(victim_bytes);
        removed_count += 1;
    }
    tracing::debug!(
        removed = removed_count,
        total_bytes,
        cap_bytes,
        "thumb pack prune"
    );
    Ok(())
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn default_cache_dir() -> Option<PathBuf> {
    use std::env;

    if let Some(xdg_cache_home) = env::var_os("XDG_CACHE_HOME")
        && !xdg_cache_home.is_empty()
    {
        let path = PathBuf::from(xdg_cache_home);
        if path.is_absolute() {
            return Some(path.join("cryslen"));
        }
    }
    let home = env::var_os("HOME")?;
    if home.is_empty() {
        return None;
    }
    Some(PathBuf::from(home).join(".cache").join("cryslen"))
}

#[expect(
    clippy::little_endian_bytes,
    reason = "book_key hashes metadata fields in fixed little-endian order"
)]
fn book_key(
    path: &Path,
    archive: bool,
    names: &[String],
) -> Option<[u8; 16]> {
    use xxhash_rust::xxh3::Xxh3;

    const KEY_FIELD_SEPARATOR: &[u8] = b"\0";

    let normalized = catalog::normalize_path(path);
    let metadata = fs::metadata(&normalized).ok()?;
    let modified = metadata
        .modified()
        .ok()?
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();

    let mut hasher = Xxh3::new();
    hasher.update(if archive {
        b"archive".as_slice()
    } else {
        b"dir".as_slice()
    });
    hasher.update(KEY_FIELD_SEPARATOR);
    hasher.update(normalized.as_os_str().as_encoded_bytes());
    hasher.update(KEY_FIELD_SEPARATOR);
    if archive {
        hasher.update(&metadata.len().to_le_bytes());
    }
    hasher.update(&modified.as_secs().to_le_bytes());
    hasher.update(&modified.subsec_nanos().to_le_bytes());
    hasher.update(&(names.len() as u64).to_le_bytes());
    if !archive {
        for name in names {
            hasher.update(name.as_bytes());
            hasher.update(KEY_FIELD_SEPARATOR);
        }
    }
    Some(hasher.digest128().to_le_bytes())
}

fn make_dir(path: &Path) -> Option<()> {
    use std::fs::DirBuilder;
    use std::os::unix::fs::DirBuilderExt as _;

    const CACHE_DIR_MODE: u32 = 0o700;

    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            let usable =
                metadata.is_dir() && !metadata.file_type().is_symlink();
            return usable.then_some(());
        }
        Err(error) => {
            if error.kind() != ErrorKind::NotFound {
                return None;
            }
        }
    }
    DirBuilder::new()
        .recursive(true)
        .mode(CACHE_DIR_MODE)
        .create(path)
        .ok()?;
    let metadata = fs::symlink_metadata(path).ok()?;
    (metadata.is_dir() && !metadata.file_type().is_symlink()).then_some(())
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::image::pixels::Image;
    use crate::image::pixels::RGBA_BYTES_PER_PIXEL;
    use crate::image::pixels::Transfer;
    use crate::image::resample;

    fn rgba_page(width: u32, height: u32) -> Image {
        let bytes = (0..(width as usize)
            * (height as usize)
            * RGBA_BYTES_PER_PIXEL)
            .map(|index| u8::try_from(index % 251).expect("channel"))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Image::rgba(width, height, Format::Png, Transfer::Srgb, bytes)
            .expect("rgba page")
            .with_page_dims((width, height))
    }

    #[test]
    fn thumb_pack_roundtrips_recovers_corrupt_db_shares_multiwriter_and_prunes()
     {
        let dir = TempDir::new().expect("temp dir");
        let root = dir.path().join("cache");
        let store = ThumbStore::open_at(&root).expect("store");
        let thumb_a = resample::derive_thumb(&rgba_page(16, 12), 1)
            .expect("thumb a");
        let thumb_b = resample::derive_thumb(&rgba_page(24, 10), 1)
            .expect("thumb b");
        let expected_a = decode_thumb_image(
            thumb_a.page_dims,
            thumb_a.format,
            &thumb_a.webp,
        )
        .expect("decoded a");
        let expected_b = decode_thumb_image(
            thumb_b.page_dims,
            thumb_b.format,
            &thumb_b.webp,
        )
        .expect("decoded b");

        let oldest_key = [0x33; 16];
        let oldest_pack = store
            .open_book_with_key(oldest_key, 4)
            .expect("oldest pack");
        oldest_pack.record(0, thumb_a);
        oldest_pack.record(2, thumb_b.clone());
        oldest_pack.record(0, thumb_b.clone());
        assert!(oldest_pack.has(0));
        assert_eq!(oldest_pack.load(0), Some(expected_a.clone()));
        oldest_pack
            .flush_pending(2, THUMB_PACK_CAP_BYTES)
            .expect("flush oldest");
        assert_eq!(oldest_pack.load(0), Some(expected_a));
        assert!(oldest_pack.load(1).is_none());

        let database_path = root.join(THUMB_DATABASE_FILENAME);
        assert!(ThumbStore::open_at(&database_path).is_none());

        let peer_store = ThumbStore::open_at(&root).expect("peer store");
        let middle_key = [0x22; 16];
        let middle_pack = peer_store
            .open_book_with_key(middle_key, 4)
            .expect("middle pack");
        middle_pack.record(1, thumb_b.clone());
        middle_pack
            .flush_pending(1, THUMB_PACK_CAP_BYTES)
            .expect("flush middle");
        drop(middle_pack);
        drop(oldest_pack);

        let reopened = peer_store
            .open_book_with_key(middle_key, 4)
            .expect("reopened pack");
        assert!(reopened.has(1));
        assert!(!reopened.has(0));
        assert_eq!(reopened.load(1), Some(expected_b));
        drop(reopened);

        let thumb_b_bytes = thumb_b.webp.len() as u64;
        let newest_key = [0x11; 16];
        let newest_pack = peer_store
            .open_book_with_key(newest_key, 4)
            .expect("newest pack");
        newest_pack.record(0, thumb_b.clone());
        newest_pack
            .flush_pending(now_unix_secs(), 3 * thumb_b_bytes)
            .expect("flush newest");
        assert!(newest_pack.has(0));
        drop(newest_pack);
        drop(peer_store);
        drop(store);

        let check_store = ThumbStore::open_at(&root).expect("check store");
        assert!(
            !check_store
                .open_book_with_key(oldest_key, 4)
                .expect("check oldest")
                .has(0),
            "oldest book should be pruned"
        );
        assert!(
            check_store
                .open_book_with_key(middle_key, 4)
                .expect("check middle")
                .has(1),
            "touched middle book should survive"
        );
        drop(check_store);

        fs::write(&database_path, b"corrupt-db-header")
            .expect("corrupt db");
        let recovered_store =
            ThumbStore::open_at(&root).expect("recovered store");
        let recovered = recovered_store
            .open_book_with_key(oldest_key, 4)
            .expect("recovered pack");
        assert!(!recovered.has(0));
        recovered.record(0, thumb_b.clone());
        recovered.flush();
        assert!(recovered.load(0).is_some());
        drop(recovered);
        drop(recovered_store);

        fs::remove_file(&database_path).expect("remove db");
        {
            const LEGACY_BOOKS_TABLE: TableDefinition<
                '_,
                [u8; 16],
                &[u8],
            > = TableDefinition::new("books");
            const LEGACY_THUMBS_TABLE: TableDefinition<
                '_,
                ([u8; 16], u32),
                &[u8],
            > = TableDefinition::new("thumbs");

            let legacy =
                Database::create(&database_path).expect("legacy db");
            let write_transaction =
                legacy.begin_write().expect("legacy write");
            {
                write_transaction
                    .open_table(LEGACY_BOOKS_TABLE)
                    .expect("legacy books");
                write_transaction
                    .open_table(LEGACY_THUMBS_TABLE)
                    .expect("legacy thumbs");
            }
            write_transaction.commit().expect("legacy commit");
        }
        let migrated_store =
            ThumbStore::open_at(&root).expect("migrated store");
        let migrated = migrated_store
            .open_book_with_key(oldest_key, 4)
            .expect("migrated pack");
        migrated.record(0, thumb_b);
        migrated.flush();
        assert!(migrated.load(0).is_some());
    }
}
