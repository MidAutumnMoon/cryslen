//! Downscaled page previews and the on-disk cache: one directory per role
//! under `env::temp_dir()`, keyed by [`content_hash`]. The little-endian
//! file format is described in `docs/internal/cache.md`.

#![expect(
    clippy::little_endian_bytes,
    reason = "the cache file format is defined little-endian"
)]

use std::borrow::Cow;
use std::env;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fs;
use std::fs::File;
use std::fs::FileTimes;
use std::fs::TryLockError;
use std::io;
use std::io::IoSlice;
use std::io::Read as _;
use std::io::Write as _;
use std::mem::DropGuard;
use std::os::unix::fs::DirBuilderExt as _;
use std::os::unix::fs::MetadataExt as _;
use std::os::unix::fs::OpenOptionsExt as _;
use std::path::Path;
use std::path::PathBuf;
use std::process;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;
use std::time::SystemTime;

use fast_image_resize::FilterType;
use fast_image_resize::PixelTrait;
use fast_image_resize::ResizeAlg;
use fast_image_resize::ResizeOptions;
use fast_image_resize::Resizer;
use fast_image_resize::images::TypedImage;
use fast_image_resize::images::TypedImageRef;
use fast_image_resize::pixels::Pixel;
use fast_image_resize::pixels::PixelComponent;
use serde::Deserialize;
use serde::Serialize;
use xxhash_rust::xxh3::xxh3_128;

use crate::browse::Role;
use crate::decode;
use crate::decode::Decoded;
use crate::decode::Format;
use crate::decode::Matrix;
use crate::decode::Pixels;
use crate::decode::Plane;
use crate::decode::Subsampling;
use crate::decode::Transfer;
use crate::decode::YuvImage;
use crate::fault::TextureFault;

pub const THUMB_EDGE: u32 = 192;
pub const THUMB_CAP_BYTES: u64 = 512 * 1024 * 1024;
pub const FIT_CAP_BYTES: u64 = 3 * 512 * 1024 * 1024;

/// Session-wide target edges clamped into the display and GPU ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct ImageLimits {
    fit_edge: u32,
    max_texture_edge: u32,
}

impl ImageLimits {
    /// Clamps `max_texture_edge >= THUMB_EDGE` and `fit_edge` into
    /// `[THUMB_EDGE, max_texture_edge]`.
    pub fn new(fit_edge: u32, max_texture_edge: u32) -> Self {
        let max_texture_edge = max_texture_edge.max(THUMB_EDGE);
        Self {
            fit_edge: fit_edge.clamp(THUMB_EDGE, max_texture_edge),
            max_texture_edge,
        }
    }

    pub fn fit_edge(self) -> u32 {
        self.fit_edge
    }

    pub fn max_texture_edge(self) -> u32 {
        self.max_texture_edge
    }
}

fn derivation(role: Role, limits: ImageLimits) -> (u32, FilterType) {
    match role {
        Role::Thumb => (THUMB_EDGE, FilterType::Box),
        Role::Fit => (limits.fit_edge(), FilterType::Lanczos3),
        Role::Full => (limits.max_texture_edge(), FilterType::Lanczos3),
    }
}

/// Content address of a page: xxh3-128 over its encoded bytes.
pub fn content_hash(bytes: &[u8]) -> u128 {
    xxh3_128(bytes)
}

/// Downsample to the role's long edge; borrows `page` unchanged when it
/// fits or resampling fails within `max_texture_edge`, and returns
/// `TextureFault` past `max_texture_edge`.
pub fn derive(
    page: &Decoded,
    role: Role,
    limits: ImageLimits,
) -> Result<Cow<'_, Decoded>, TextureFault> {
    let (edge, filter) = derivation(role, limits);
    let error = match derive_to_edge(page, edge, filter) {
        Ok(derived) => return Ok(derived),
        Err(error) => error,
    };
    let dims = (page.width(), page.height());
    let max_edge = limits.max_texture_edge();
    if dims.0.max(dims.1) <= max_edge {
        tracing::warn!(%error, ?dims, "resample failed; keeping the page");
        return Ok(Cow::Borrowed(page));
    }
    tracing::warn!(
        %error,
        ?dims,
        max_edge,
        "resample failed past texture limit"
    );
    Err(TextureFault { dims, max_edge })
}

fn derive_to_edge(
    page: &Decoded,
    edge: u32,
    filter: FilterType,
) -> eros::Result<Cow<'_, Decoded>> {
    let page_dims = page.page_dims();
    let dims = (page.width(), page.height());
    let target = fit_dims(dims, edge);
    if target == dims {
        return Ok(Cow::Borrowed(page));
    }
    resample_page(page, target, filter)
        .map(|derived| Cow::Owned(derived.with_page_dims(page_dims)))
}

#[expect(
    clippy::integer_division,
    reason = "edge scaling is integer pixel arithmetic"
)]
fn fit_dims(dims: (u32, u32), edge: u32) -> (u32, u32) {
    let (width, height) = (u64::from(dims.0), u64::from(dims.1));
    let long_edge = width.max(height);
    if long_edge <= u64::from(edge) {
        return dims;
    }
    let scaled = |axis: u64| {
        let rounded = (axis * u64::from(edge) + long_edge / 2) / long_edge;
        u32::try_from(rounded.max(1)).unwrap_or(u32::MAX)
    };
    (scaled(width), scaled(height))
}

fn resample_page(
    page: &Decoded,
    target: (u32, u32),
    filter: FilterType,
) -> eros::Result<Decoded> {
    let dims = (page.width(), page.height());
    let mut resizer = Resizer::new();
    match page.pixels() {
        Pixels::Rgba8(bytes) => {
            let pixels = resize_typed::<_, u8, 4>(
                &mut resizer,
                bytes.as_chunks::<4>().0,
                dims,
                target,
                filter,
            )?;
            Ok(Decoded::rgba(
                target.0,
                target.1,
                page.format(),
                page.transfer(),
                pixels.into_flattened().into_boxed_slice(),
            )?)
        }
        Pixels::Yuv(yuv) => {
            let ((luma_w, luma_h), (chroma_w, chroma_h)) =
                decode::plane_dims(dims, yuv.subsampling);
            let (_, (target_w, target_h)) =
                decode::plane_dims(target, yuv.subsampling);
            let planes = YuvImage {
                depth: yuv.depth,
                subsampling: yuv.subsampling,
                full_range: yuv.full_range,
                matrix: yuv.matrix,
                luma: resize_plane(
                    &mut resizer,
                    &yuv.luma,
                    (luma_w, luma_h),
                    target,
                    filter,
                )?,
                cb: resize_plane(
                    &mut resizer,
                    &yuv.cb,
                    (chroma_w, chroma_h),
                    (target_w, target_h),
                    filter,
                )?,
                cr: resize_plane(
                    &mut resizer,
                    &yuv.cr,
                    (chroma_w, chroma_h),
                    (target_w, target_h),
                    filter,
                )?,
                alpha: match yuv.alpha.as_ref() {
                    Some(alpha) => Some(resize_plane(
                        &mut resizer,
                        alpha,
                        (luma_w, luma_h),
                        target,
                        filter,
                    )?),
                    None => None,
                },
            };
            Ok(Decoded::yuv(
                target.0,
                target.1,
                page.format(),
                page.transfer(),
                planes,
            )?)
        }
    }
}

fn resize_plane(
    resizer: &mut Resizer,
    plane: &Plane,
    source_dims: (u32, u32),
    target_dims: (u32, u32),
    filter: FilterType,
) -> eros::Result<Plane> {
    // Equal extents also cover Mono's empty chroma planes, which carry a
    // 1x1 extent without samples.
    if source_dims == target_dims {
        return Ok(plane.clone());
    }
    Ok(match plane {
        Plane::U8(bytes) => Plane::U8(
            resize_typed::<_, u8, 1>(
                resizer,
                bytes,
                source_dims,
                target_dims,
                filter,
            )?
            .into_boxed_slice(),
        ),
        Plane::U16(words) => Plane::U16(
            resize_typed::<_, u16, 1>(
                resizer,
                words,
                source_dims,
                target_dims,
                filter,
            )?
            .into_boxed_slice(),
        ),
    })
}

fn resize_typed<S, C, const N: usize>(
    resizer: &mut Resizer,
    source: &[S],
    source_dims: (u32, u32),
    target_dims: (u32, u32),
    filter: FilterType,
) -> eros::Result<Vec<S>>
where
    S: Default + PartialEq + bytemuck::Pod,
    C: PixelComponent,
    Pixel<S, C, N>: PixelTrait,
{
    let Ok(count) = usize::try_from(
        u64::from(target_dims.0) * u64::from(target_dims.1),
    ) else {
        eros::bail!("buffer count overflows usize");
    };
    let mut out = vec![S::default(); count];
    let source = TypedImageRef::<Pixel<S, C, N>>::from_buffer(
        source_dims.0,
        source_dims.1,
        bytemuck::cast_slice(source),
    )?;
    let mut target = TypedImage::<Pixel<S, C, N>>::from_buffer(
        target_dims.0,
        target_dims.1,
        bytemuck::cast_slice_mut(&mut out),
    )?;
    // Straight alpha: the renderer premultiplies at draw. Alpha-aware
    // resampling avoids the color bleed at transparent edges but costs a
    // source-sized temporary and about double the time.
    let options = ResizeOptions::new()
        .resize_alg(ResizeAlg::Convolution(filter))
        .use_alpha(false);
    resizer.resize_typed(&source, &mut target, &options)?;
    Ok(out)
}

/// The shared preview cache. A store that could not claim a real
/// directory stays disabled, and every operation on it is a no-op.
pub struct Store {
    state: Option<State>,
}

struct State {
    thumb: Namespace,
    view: Namespace,
}

/// What a save did: whether the entry reached the store, and the
/// entries its prune removed (oldest first), so a caller that tracks the
/// store's contents can keep its ledger current.
#[derive(Debug, Default)]
pub struct SaveResult {
    pub persisted: bool,
    pub evicted: Vec<(Role, u128)>,
}

/// One role namespace: a directory of entries with its own byte cap.
struct Namespace {
    dir: PathBuf,
    cap: u64,
    /// Bytes written since the last scan or prune.
    total: AtomicU64,
}

impl Store {
    /// Open the shared store under `env::temp_dir()/cryslen-cache`.
    pub fn open() -> Self {
        Self::at(&env::temp_dir().join("cryslen-cache"))
    }

    /// Open the store rooted at `root`, creating it if absent. A `root`
    /// that cannot be created, or that resolves through a symlink, leaves
    /// the store disabled.
    pub fn at(root: &Path) -> Self {
        Self::with_caps(root, THUMB_CAP_BYTES, FIT_CAP_BYTES)
    }

    fn with_caps(root: &Path, thumb_cap: u64, view_cap: u64) -> Self {
        let state = prepare(root).map(|(thumb, view)| State {
            thumb: Namespace::new(thumb, thumb_cap),
            view: Namespace::new(view, view_cap),
        });
        Self { state }
    }

    /// Load `hash`, or `None` when absent, damaged, or cached for another
    /// target edge; a hit refreshes the entry's mtime for LRU.
    pub fn load(
        &self,
        hash: u128,
        role: Role,
        fit_edge: u32,
    ) -> Option<Decoded> {
        let started = Instant::now();
        let loaded = self.role_target(role, fit_edge).and_then(|target| {
            let file = File::open(target.namespace.path(hash)).ok()?;
            #[expect(
                clippy::let_underscore_must_use,
                clippy::let_underscore_untyped,
                reason = "the recency touch is advisory; its failure changes nothing"
            )]
            let _ = file.set_times(
                FileTimes::new().set_modified(SystemTime::now()),
            );
            read_stored(file, target.edge)
        });
        tracing::debug!(
            ?role,
            hit = loaded.is_some(),
            elapsed = ?started.elapsed(),
            "preview load"
        );
        loaded
    }

    /// Store `page` under `hash`, replacing the same key. Crossing the
    /// namespace's cap prunes it; the result names the removed entries.
    pub fn save(
        &self,
        hash: u128,
        role: Role,
        page: &Decoded,
        fit_edge: u32,
    ) -> SaveResult {
        let Some(target) = self.role_target(role, fit_edge) else {
            return SaveResult::default();
        };
        let started = Instant::now();
        let Ok(size) = write_encoded_atomic(
            target.namespace,
            hash,
            page,
            target.edge,
        ) else {
            return SaveResult::default();
        };
        let prev =
            target.namespace.total.fetch_add(size, Ordering::Relaxed);
        let over = prev.saturating_add(size) > target.namespace.cap;
        tracing::debug!(
            ?role,
            bytes = size,
            elapsed = ?started.elapsed(),
            "preview save"
        );
        let evicted = if over {
            target
                .namespace
                .prune()
                .into_iter()
                .map(|evicted| (role, evicted))
                .collect()
        } else {
            Vec::new()
        };
        SaveResult {
            persisted: true,
            evicted,
        }
    }

    fn role_target(
        &self,
        role: Role,
        fit_edge: u32,
    ) -> Option<RoleTarget<'_>> {
        let state = self.state.as_ref()?;
        match role {
            Role::Thumb => Some(RoleTarget {
                namespace: &state.thumb,
                edge: THUMB_EDGE,
            }),
            Role::Fit => Some(RoleTarget {
                namespace: &state.view,
                edge: fit_edge,
            }),
            Role::Full => None,
        }
    }
}

struct RoleTarget<'store> {
    namespace: &'store Namespace,
    edge: u32,
}

static NEXT_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

impl Namespace {
    fn new(dir: PathBuf, cap: u64) -> Self {
        clean_temps(&dir);
        let total = entries(&dir)
            .iter()
            .map(|entry| entry.size)
            .fold(0_u64, u64::saturating_add);
        Self {
            dir,
            cap,
            total: AtomicU64::new(total),
        }
    }

    fn path(&self, hash: u128) -> PathBuf {
        self.dir.join(format!("{hash:032x}"))
    }

    /// The per-write temp name: concurrent processes or worker threads
    /// save into one namespace, so pid plus sequence keeps each
    /// in-progress file distinct.
    fn temp_path(&self, hash: u128) -> PathBuf {
        let sequence = NEXT_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        self.dir
            .join(format!("{hash:032x}.{}-{sequence}.tmp", process::id()))
    }

    fn prune(&self) -> Vec<u128> {
        let entries = entries(&self.dir);
        let mut total = entries
            .iter()
            .map(|entry| entry.size)
            .fold(0_u64, u64::saturating_add);
        let mut removed = 0_u64;
        let mut evicted = Vec::new();
        for entry in entries {
            if total <= self.cap {
                break;
            }
            if fs::remove_file(self.dir.join(&entry.name)).is_ok() {
                total = total.saturating_sub(entry.size);
                removed += 1;
                if let Some(hash) = entry
                    .name
                    .to_str()
                    .and_then(|name| u128::from_str_radix(name, 16).ok())
                {
                    evicted.push(hash);
                }
            }
        }
        tracing::debug!(
            dir = %self.dir.display(),
            removed,
            total,
            cap = self.cap,
            "preview prune"
        );
        self.total.store(total, Ordering::Relaxed);
        evicted
    }
}

struct StoreEntry {
    modified: SystemTime,
    name: OsString,
    size: u64,
}

/// The cache files of `dir`, least recently used first. A `.tmp` file
/// belongs to a writer that has not renamed it yet; skip it.
fn entries(dir: &Path) -> Vec<StoreEntry> {
    let Ok(read) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut entries = Vec::new();
    for entry in read.flatten() {
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        let name = entry.file_name();
        if !is_hash_name(&name) {
            continue;
        }
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        entries.push(StoreEntry {
            modified,
            name,
            size: metadata.len(),
        });
    }
    entries.sort_by(|left, right| {
        left.modified
            .cmp(&right.modified)
            .then_with(|| left.name.cmp(&right.name))
    });
    entries
}

fn is_hash_name(name: &OsStr) -> bool {
    name.to_str().is_some_and(|name| {
        name.len() == 32
            && name.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

/// Drop temp files no writer holds: the kernel releases a dead writer's
/// lock, so a lock that can be taken proves the temp is unclaimed.
fn clean_temps(dir: &Path) {
    let Ok(read) = fs::read_dir(dir) else {
        return;
    };
    let mut removed = 0_usize;
    for entry in read.flatten() {
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        let name = entry.file_name();
        if !is_temp_name(&name) {
            continue;
        }
        if remove_if_unlocked(&dir.join(&name)) {
            removed += 1;
        }
    }
    if removed > 0 {
        tracing::debug!(
            dir = %dir.display(),
            removed,
            "cleaned stale preview temps"
        );
    }
}

fn remove_if_unlocked(path: &Path) -> bool {
    let Ok(file) = File::open(path) else {
        return false;
    };
    match file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => return false,
        Err(TryLockError::Error(error)) => {
            tracing::debug!(
                %error,
                path = %path.display(),
                "preview temp lock failed"
            );
            return false;
        }
    }
    // Hold the lock across unlink so a racing writer waits for removal.
    if let Err(error) = fs::remove_file(path) {
        tracing::debug!(
            %error,
            path = %path.display(),
            "stale preview temp"
        );
        return false;
    }
    drop(file);
    true
}

/// Whether `name` is a writer's temp: `<hash:032x>.<pid>[-<sequence>].tmp`.
fn is_temp_name(name: &OsStr) -> bool {
    let Some(name) = name.to_str() else {
        return false;
    };
    let Some(stem) = name.strip_suffix(".tmp") else {
        return false;
    };
    let Some((hash, writer)) = stem.split_once('.') else {
        return false;
    };
    let valid_writer = match writer.split_once('-') {
        Some((pid, sequence)) => is_digits(pid) && is_digits(sequence),
        None => is_digits(writer),
    };
    is_hash_name(OsStr::new(hash)) && valid_writer
}

fn is_digits(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit())
}

/// Refuse symlinks so the store stays inside `root`.
fn prepare(root: &Path) -> Option<(PathBuf, PathBuf)> {
    make_dir(root)?;
    let thumb = root.join(<&str>::from(Role::Thumb));
    let view = root.join(<&str>::from(Role::Fit));
    make_dir(&thumb)?;
    make_dir(&view)?;
    Some((thumb, view))
}

fn make_dir(path: &Path) -> Option<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            let usable =
                metadata.is_dir() && !metadata.file_type().is_symlink();
            return usable.then_some(());
        }
        Err(error) => {
            if error.kind() != io::ErrorKind::NotFound {
                return None;
            }
        }
    }
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(path)
        .ok()?;
    let metadata = fs::symlink_metadata(path).ok()?;
    (metadata.is_dir() && !metadata.file_type().is_symlink()).then_some(())
}

type TempGuard = DropGuard<PathBuf, fn(PathBuf)>;

fn remove_temp(path: PathBuf) {
    if let Err(error) = fs::remove_file(path) {
        tracing::debug!(%error, "leftover preview temp file");
    }
}

fn file_identity(metadata: &fs::Metadata) -> (u64, u64) {
    (metadata.dev(), metadata.ino())
}

/// Retries if `clean_temps` sweeps the entry before `file.lock()` lands;
/// keep the returned `File` locked until rename so cleaners see a live writer.
fn open_locked_temp(
    namespace: &Namespace,
    hash: u128,
) -> io::Result<(File, TempGuard)> {
    const MAX_ATTEMPTS: usize = 8;

    for _ in 0..MAX_ATTEMPTS {
        let temp = namespace.temp_path(hash);
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temp)?;
        let guard: TempGuard = DropGuard::new(temp, remove_temp);
        file.lock()?;
        let opened = file_identity(&file.metadata()?);
        let on_disk = match fs::symlink_metadata(&*guard) {
            Ok(metadata) => Some(file_identity(&metadata)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(error),
        };
        if on_disk == Some(opened) {
            return Ok((file, guard));
        }
        // Disarm before retrying so a replaced path's inode stays intact.
        drop(DropGuard::dismiss(guard));
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "temp file was swept before it could be locked",
    ))
}

fn write_atomic(
    namespace: &Namespace,
    hash: u128,
    mut slices: &mut [IoSlice<'_>],
) -> io::Result<u64> {
    let (mut file, guard) = open_locked_temp(namespace, hash)?;
    let total: usize = slices.iter().map(|slice| slice.len()).sum();
    while !slices.is_empty() {
        match file.write_vectored(slices) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "short vectored write",
                ));
            }
            Ok(written) => IoSlice::advance_slices(&mut slices, written),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    fs::rename(&*guard, namespace.path(hash))?;
    drop(DropGuard::dismiss(guard));
    Ok(u64::try_from(total).unwrap_or(u64::MAX))
}

const MAGIC: [u8; 4] = *b"CYPV";
const VERSION: u16 = 4;
const PREFIX_BYTES: usize =
    MAGIC.len() + size_of::<u16>() + size_of::<u8>();
const MAX_META_BYTES: usize = u8::MAX as usize;

#[derive(Deserialize, Serialize)]
struct CacheMeta {
    target_edge: u32,
    page_dims: (u32, u32),
    width: u32,
    height: u32,
    format: Format,
    transfer: Transfer,
    pixels: PixelMeta,
}

#[derive(Deserialize, Serialize)]
enum PixelMeta {
    Rgba8,
    Yuv {
        depth: u8,
        subsampling: Subsampling,
        full_range: bool,
        matrix: Matrix,
        has_alpha: bool,
    },
}

impl CacheMeta {
    fn from_page(page: &Decoded, target_edge: u32) -> Self {
        let pixels = match page.pixels() {
            Pixels::Rgba8(_) => PixelMeta::Rgba8,
            Pixels::Yuv(yuv) => PixelMeta::Yuv {
                depth: yuv.depth,
                subsampling: yuv.subsampling,
                full_range: yuv.full_range,
                matrix: yuv.matrix,
                has_alpha: yuv.alpha.is_some(),
            },
        };
        Self {
            target_edge,
            page_dims: page.page_dims(),
            width: page.width(),
            height: page.height(),
            format: page.format(),
            transfer: page.transfer(),
            pixels,
        }
    }
}

fn encode_header(meta: &CacheMeta) -> io::Result<Vec<u8>> {
    let mut meta_buf = [0_u8; MAX_META_BYTES];
    let encoded = postcard::to_slice(meta, &mut meta_buf)
        .map_err(io::Error::other)?;
    let Ok(meta_len) = u8::try_from(encoded.len()) else {
        return Err(io::Error::other(format!(
            "cache metadata exceeds {MAX_META_BYTES} bytes"
        )));
    };
    let mut header = Vec::with_capacity(PREFIX_BYTES + encoded.len());
    header.extend_from_slice(&MAGIC);
    header.extend_from_slice(&VERSION.to_le_bytes());
    header.push(meta_len);
    header.extend_from_slice(encoded);
    Ok(header)
}

fn write_encoded_atomic(
    namespace: &Namespace,
    hash: u128,
    page: &Decoded,
    target_edge: u32,
) -> io::Result<u64> {
    let header = encode_header(&CacheMeta::from_page(page, target_edge))?;
    match page.pixels() {
        Pixels::Rgba8(bytes) => {
            let mut slices = [IoSlice::new(&header), IoSlice::new(bytes)];
            write_atomic(namespace, hash, &mut slices)
        }
        Pixels::Yuv(yuv) => {
            // Header + 3 YUV planes + optional alpha plane.
            const YUV_MAX_SLICES: usize = 1 + 3 + 1;
            let luma = little_endian_plane_bytes(&yuv.luma);
            let cb = little_endian_plane_bytes(&yuv.cb);
            let cr = little_endian_plane_bytes(&yuv.cr);
            let alpha = yuv.alpha.as_ref().map(little_endian_plane_bytes);
            let mut slices = Vec::with_capacity(YUV_MAX_SLICES);
            slices.push(IoSlice::new(&header));
            slices.push(IoSlice::new(&luma));
            slices.push(IoSlice::new(&cb));
            slices.push(IoSlice::new(&cr));
            if let Some(alpha) = &alpha {
                slices.push(IoSlice::new(alpha));
            }
            write_atomic(namespace, hash, &mut slices)
        }
    }
}

fn little_endian_plane_bytes(plane: &Plane) -> Cow<'_, [u8]> {
    match plane {
        Plane::U8(bytes) => Cow::Borrowed(bytes),
        Plane::U16(words) => {
            if cfg!(target_endian = "little") {
                Cow::Borrowed(bytemuck::cast_slice(words))
            } else {
                let mut out =
                    Vec::with_capacity(words.len() * size_of::<u16>());
                for word in words {
                    out.extend_from_slice(&word.to_le_bytes());
                }
                Cow::Owned(out)
            }
        }
    }
}

fn read_stored(mut file: File, target_edge: u32) -> Option<Decoded> {
    let file_len = file.metadata().ok()?.len();
    if file_len < u64::try_from(PREFIX_BYTES).ok()? {
        return None;
    }
    let mut prefix = [0_u8; PREFIX_BYTES];
    file.read_exact(&mut prefix).ok()?;
    let (magic, rest) = prefix.split_first_chunk::<{ MAGIC.len() }>()?;
    let (version, rest) =
        rest.split_first_chunk::<{ size_of::<u16>() }>()?;
    let &[meta_len] = rest else {
        return None;
    };
    if *magic != MAGIC || *version != VERSION.to_le_bytes() {
        return None;
    }
    let meta_len = usize::from(meta_len);
    if meta_len == 0 {
        return None;
    }
    let header_len = u64::try_from(PREFIX_BYTES + meta_len).ok()?;
    let payload_len = file_len.checked_sub(header_len)?;
    let mut meta_buf = [0_u8; MAX_META_BYTES];
    let meta_bytes = meta_buf.get_mut(..meta_len)?;
    file.read_exact(meta_bytes).ok()?;
    let meta: CacheMeta = postcard::from_bytes(meta_bytes).ok()?;
    if meta.target_edge != target_edge {
        return None;
    }
    decode::pixel_count(meta.page_dims.0, meta.page_dims.1)?;
    let page = match meta.pixels {
        PixelMeta::Rgba8 => read_rgba(&mut file, payload_len, &meta)?,
        PixelMeta::Yuv {
            depth,
            subsampling,
            full_range,
            matrix,
            has_alpha,
        } => {
            let layout = decode::yuv_layout(
                (meta.width, meta.height),
                depth,
                subsampling,
            )
            .ok()?;
            let expected_bytes = layout.total_bytes(has_alpha)?;
            if u64::try_from(expected_bytes).ok()? != payload_len {
                return None;
            }
            let luma =
                read_plane(&mut file, layout.luma_samples, layout.is_u16)?;
            let cb = read_plane(
                &mut file,
                layout.chroma_samples,
                layout.is_u16,
            )?;
            let cr = read_plane(
                &mut file,
                layout.chroma_samples,
                layout.is_u16,
            )?;
            let alpha = if has_alpha {
                Some(read_plane(
                    &mut file,
                    layout.luma_samples,
                    layout.is_u16,
                )?)
            } else {
                None
            };
            Decoded::yuv(
                meta.width,
                meta.height,
                meta.format,
                meta.transfer,
                YuvImage {
                    depth,
                    subsampling,
                    full_range,
                    matrix,
                    luma,
                    cb,
                    cr,
                    alpha,
                },
            )
            .ok()?
        }
    };
    Some(page.with_page_dims(meta.page_dims))
}

fn read_rgba(
    file: &mut File,
    payload_len: u64,
    meta: &CacheMeta,
) -> Option<Decoded> {
    let expected_bytes =
        decode::rgba_byte_len(meta.width, meta.height).ok()?;
    if u64::try_from(expected_bytes).ok()? != payload_len {
        return None;
    }
    let mut bytes = vec![0_u8; expected_bytes];
    file.read_exact(&mut bytes).ok()?;
    Decoded::rgba(
        meta.width,
        meta.height,
        meta.format,
        meta.transfer,
        bytes.into_boxed_slice(),
    )
    .ok()
}

fn read_plane(
    file: &mut File,
    samples: usize,
    is_u16: bool,
) -> Option<Plane> {
    if !is_u16 {
        let mut bytes = vec![0_u8; samples];
        file.read_exact(&mut bytes).ok()?;
        return Some(Plane::U8(bytes.into_boxed_slice()));
    }
    let mut words = vec![0_u16; samples];
    file.read_exact(bytemuck::cast_slice_mut(&mut words)).ok()?;
    if cfg!(not(target_endian = "little")) {
        for word in &mut words {
            *word = u16::from_le(*word);
        }
    }
    Some(Plane::U16(words.into_boxed_slice()))
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use std::thread;
    use std::time::Duration;

    use tempfile::TempDir;

    use super::*;

    const TEST_FIT_EDGE: u32 = 2560;

    fn test_limits() -> ImageLimits {
        const MAX_TEXTURE_EDGE: u32 = 16_384;
        ImageLimits::new(TEST_FIT_EDGE, MAX_TEXTURE_EDGE)
    }

    fn count(width: u32, height: u32) -> usize {
        (width as usize) * (height as usize)
    }

    fn rgba_page(width: u32, height: u32) -> Decoded {
        let bytes = (0..count(width, height) * 4)
            .map(|index| u8::try_from(index % 251).expect("channel"))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Decoded::rgba(width, height, Format::Png, Transfer::Srgb, bytes)
            .expect("rgba page")
            .with_page_dims((width, height))
    }

    fn yuv_page(
        width: u32,
        height: u32,
        subsampling: Subsampling,
    ) -> Decoded {
        yuv_page_with(width, height, subsampling, 10, false)
    }

    fn yuv_page_with(
        width: u32,
        height: u32,
        subsampling: Subsampling,
        depth: u8,
        with_alpha: bool,
    ) -> Decoded {
        let ((_, _), (chroma_w, chroma_h)) =
            decode::plane_dims((width, height), subsampling);
        let plane = |samples: usize| {
            if depth > 8 {
                Plane::U16(
                    (0..samples)
                        .map(|index| {
                            u16::try_from(index % 1024).expect("sample")
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                )
            } else {
                Plane::U8(
                    (0..samples)
                        .map(|index| {
                            u8::try_from(index % 251).expect("sample")
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                )
            }
        };
        let chroma = if subsampling == Subsampling::Mono {
            0
        } else {
            count(chroma_w, chroma_h)
        };
        let luma = count(width, height);
        let yuv = YuvImage {
            depth,
            subsampling,
            full_range: false,
            matrix: Matrix::Bt709,
            luma: plane(luma),
            cb: plane(chroma),
            cr: plane(chroma),
            alpha: with_alpha.then(|| plane(luma)),
        };
        Decoded::yuv(width, height, Format::Avif, Transfer::Pq, yuv)
            .expect("yuv page")
            .with_page_dims((width, height))
    }

    fn rgba_bytes(page: &Decoded) -> Vec<u8> {
        let Pixels::Rgba8(bytes) = page.pixels() else {
            panic!("expected rgba pixels");
        };
        bytes.to_vec()
    }

    /// A page whose planes are each one constant, `values` in luma, Cb,
    /// Cr, alpha order.
    fn flat_yuv_page(
        width: u32,
        height: u32,
        subsampling: Subsampling,
        values: [u16; 4],
    ) -> Decoded {
        let ((_, _), (chroma_w, chroma_h)) =
            decode::plane_dims((width, height), subsampling);
        let plane = |samples: usize, value: u16| {
            Plane::U16(vec![value; samples].into_boxed_slice())
        };
        let chroma = if subsampling == Subsampling::Mono {
            0
        } else {
            count(chroma_w, chroma_h)
        };
        let yuv = YuvImage {
            depth: 10,
            subsampling,
            full_range: false,
            matrix: Matrix::Bt709,
            luma: plane(count(width, height), values[0]),
            cb: plane(chroma, values[1]),
            cr: plane(chroma, values[2]),
            alpha: Some(plane(count(width, height), values[3])),
        };
        Decoded::yuv(width, height, Format::Avif, Transfer::Pq, yuv)
            .expect("yuv page")
            .with_page_dims((width, height))
    }

    fn yuv_planes(page: &Decoded) -> [Vec<u16>; 4] {
        let Pixels::Yuv(yuv) = page.pixels() else {
            panic!("expected yuv pixels");
        };
        let samples = |plane: &Plane| match plane {
            Plane::U8(bytes) => {
                bytes.iter().map(|&byte| u16::from(byte)).collect()
            }
            Plane::U16(words) => words.to_vec(),
        };
        [
            samples(&yuv.luma),
            samples(&yuv.cb),
            samples(&yuv.cr),
            yuv.alpha.as_ref().map_or_else(Vec::new, samples),
        ]
    }

    fn set_modified(path: &Path, seconds: u64) {
        let modified =
            SystemTime::UNIX_EPOCH + Duration::from_secs(seconds);
        fs::set_times(path, FileTimes::new().set_modified(modified))
            .expect("set mtime");
    }

    #[test]
    fn concurrent_saves_do_not_collide() {
        use std::sync::atomic::AtomicUsize;

        const WRITER_COUNT: usize = 2;
        const SAVES_PER_WRITER: usize = 20;

        let dir = TempDir::new().expect("create temp dir");
        let root = dir.path();
        let page = rgba_page(8, 8);
        let hash = content_hash(b"same-entry");
        let store = Store::at(root);
        let active_writers = AtomicUsize::new(WRITER_COUNT);

        thread::scope(|scope| {
            let store = &store;
            let page = &page;
            let active_writers = &active_writers;
            for _ in 0..WRITER_COUNT {
                scope.spawn(move || {
                    for _ in 0..SAVES_PER_WRITER {
                        assert!(
                            store
                                .save(hash, Role::Fit, page, TEST_FIT_EDGE)
                                .persisted
                        );
                    }
                    active_writers.fetch_sub(1, Ordering::Relaxed);
                });
            }
            scope.spawn(move || {
                while active_writers.load(Ordering::Relaxed) > 0 {
                    drop(Store::at(root));
                }
            });
        });

        let loaded = Store::at(root)
            .load(hash, Role::Fit, TEST_FIT_EDGE)
            .expect("kept");
        assert_eq!((loaded.width(), loaded.height()), (8, 8));
    }

    #[test]
    fn open_cleans_unlocked_writer_temps() {
        let dir = TempDir::new().expect("create temp dir");
        let root = dir.path().join("cache");
        let page = rgba_page(2, 2);
        let hash = content_hash(b"page-kept");
        let store = Store::at(&root);
        assert!(
            store
                .save(hash, Role::Thumb, &page, TEST_FIT_EDGE)
                .persisted
        );

        // A dead writer leaves its temp unlocked.
        let stale_thumb = store
            .role_target(Role::Thumb, TEST_FIT_EDGE)
            .expect("thumb")
            .namespace
            .temp_path(hash);
        fs::write(&stale_thumb, b"partial").expect("thumb temp");
        let stale_view = store
            .role_target(Role::Fit, TEST_FIT_EDGE)
            .expect("view")
            .namespace
            .temp_path(hash);
        fs::write(&stale_view, b"partial").expect("view temp");

        // A live writer's temp stays while it holds the lock.
        let live_temp = store
            .role_target(Role::Thumb, TEST_FIT_EDGE)
            .expect("thumb")
            .namespace
            .temp_path(hash);
        drop(store);
        let locked_temp = File::options()
            .write(true)
            .create_new(true)
            .open(&live_temp)
            .expect("live temp");
        locked_temp.lock().expect("lock live temp");

        let malformed = [
            "not-a-hash.123.tmp",
            "0123456789abcdef0123456789abcdef.nope.tmp",
            "0123456789abcdef0123456789abcdef.12.tmp.bak",
        ];
        for name in malformed {
            fs::write(root.join("thumb").join(name), b"partial")
                .expect("malformed temp");
        }

        let reopened = Store::at(&root);
        assert!(!stale_thumb.exists());
        assert!(!stale_view.exists());
        assert!(live_temp.exists());
        for name in malformed {
            assert!(root.join("thumb").join(name).exists(), "{name}");
        }
        assert!(reopened.load(hash, Role::Thumb, TEST_FIT_EDGE).is_some());

        // Closing the handle releases the lock, so the next open claims
        // the temp.
        drop(locked_temp);
        Store::at(&root);
        assert!(!live_temp.exists());
    }

    #[test]
    fn rgba_roundtrip_preserves_the_page() {
        let dir = TempDir::new().expect("create temp dir");
        let page = rgba_page(3, 2).with_page_dims((9, 6));
        let store = Store::at(dir.path());
        assert!(store.state.is_some());
        let hash = content_hash(b"page-a");
        assert!(
            store
                .save(hash, Role::Thumb, &page, TEST_FIT_EDGE)
                .persisted
        );
        assert!(
            !store.save(hash, Role::Full, &page, TEST_FIT_EDGE).persisted
        );
        assert!(store.load(hash, Role::Full, TEST_FIT_EDGE).is_none());

        let loaded = store
            .load(hash, Role::Thumb, TEST_FIT_EDGE)
            .expect("cached thumb");
        assert_eq!((loaded.width(), loaded.height()), (3, 2));
        assert_eq!(loaded.page_dims(), (9, 6));
        assert_eq!(loaded.format(), Format::Png);
        assert_eq!(loaded.transfer(), Transfer::Srgb);
        assert_eq!(rgba_bytes(&loaded), rgba_bytes(&page));
    }

    #[test]
    fn yuv_plane_roundtrip_preserves_the_page() {
        let dir = TempDir::new().expect("create temp dir");
        let page =
            yuv_page(5, 3, Subsampling::Yuv420).with_page_dims((15, 9));
        let store = Store::at(dir.path());
        let hash = content_hash(b"page-b");
        assert!(
            store.save(hash, Role::Fit, &page, TEST_FIT_EDGE).persisted
        );
        assert!(
            store.load(hash, Role::Fit, TEST_FIT_EDGE + 1).is_none(),
            "a fit entry saved for one target edge misses for another"
        );

        let loaded = store
            .load(hash, Role::Fit, TEST_FIT_EDGE)
            .expect("cached view");
        assert_eq!((loaded.width(), loaded.height()), (5, 3));
        assert_eq!(loaded.page_dims(), (15, 9));
        assert_eq!(loaded.format(), Format::Avif);
        assert_eq!(loaded.transfer(), Transfer::Pq);
        let Pixels::Yuv(yuv) = loaded.pixels() else {
            panic!("expected yuv pixels");
        };
        assert_eq!(yuv.depth, 10);
        assert_eq!(yuv.subsampling, Subsampling::Yuv420);
        assert!(!yuv.full_range);
        assert_eq!(yuv.matrix, Matrix::Bt709);
        assert_eq!(yuv_planes(&loaded), yuv_planes(&page));
    }

    #[test]
    fn yuv_u8_planes_and_alpha_roundtrip() {
        let dir = TempDir::new().expect("create temp dir");
        let page = yuv_page_with(6, 4, Subsampling::Yuv444, 8, true);
        let store = Store::at(dir.path());
        let hash = content_hash(b"page-e");
        assert!(
            store
                .save(hash, Role::Thumb, &page, TEST_FIT_EDGE)
                .persisted
        );

        let loaded = store
            .load(hash, Role::Thumb, TEST_FIT_EDGE)
            .expect("cached thumb");
        let Pixels::Yuv(yuv) = loaded.pixels() else {
            panic!("expected yuv pixels");
        };
        assert_eq!(yuv.depth, 8);
        assert!(yuv.alpha.is_some());
        assert_eq!(yuv_planes(&loaded), yuv_planes(&page));
    }

    #[test]
    fn corrupt_entries_are_rejected() {
        const VERSION_START: usize = MAGIC.len();
        const VERSION_END: usize = VERSION_START + size_of::<u16>();

        let dir = TempDir::new().expect("create temp dir");
        let page = rgba_page(2, 2);
        let store = Store::at(dir.path());
        let hash = content_hash(b"page-c");
        assert!(
            store
                .save(hash, Role::Thumb, &page, TEST_FIT_EDGE)
                .persisted
        );
        let path = dir.path().join("thumb").join(format!("{hash:032x}"));
        let good = fs::read(&path).expect("read cache file");

        let mut bad_magic = good.clone();
        if let Some(byte) = bad_magic.first_mut() {
            *byte = b'X';
        }
        fs::write(&path, &bad_magic).expect("write magic");
        assert!(store.load(hash, Role::Thumb, TEST_FIT_EDGE).is_none());

        let mut bad_version = good.clone();
        if let Some(field) =
            bad_version.get_mut(VERSION_START..VERSION_END)
        {
            field.copy_from_slice(&(VERSION + 1).to_le_bytes());
        }
        fs::write(&path, &bad_version).expect("write version");
        assert!(store.load(hash, Role::Thumb, TEST_FIT_EDGE).is_none());

        for meta_len in [0_u8, u8::MAX] {
            let mut bad_meta_len = good.clone();
            if let Some(byte) = bad_meta_len.get_mut(VERSION_END) {
                *byte = meta_len;
            }
            fs::write(&path, &bad_meta_len).expect("write meta length");
            assert!(
                store.load(hash, Role::Thumb, TEST_FIT_EDGE).is_none()
            );
        }

        // Different target edge on disk misses cleanly even with matching pixels.
        let mut wrong_edge =
            encode_header(&CacheMeta::from_page(&page, THUMB_EDGE + 1))
                .expect("encode wrong-edge header");
        wrong_edge.extend_from_slice(&rgba_bytes(&page));
        fs::write(&path, &wrong_edge).expect("write wrong-edge entry");
        assert!(store.load(hash, Role::Thumb, TEST_FIT_EDGE).is_none());

        // Admissible dims whose byte size exceeds the file reject against
        // file length before allocating.
        let mut forged_dims = CacheMeta::from_page(&page, THUMB_EDGE);
        forged_dims.width = 4_096;
        forged_dims.height = 4_096;
        let oversized =
            encode_header(&forged_dims).expect("encode oversized header");
        fs::write(&path, &oversized).expect("write oversized entry");
        assert!(store.load(hash, Role::Thumb, TEST_FIT_EDGE).is_none());

        let mut truncated = good.clone();
        truncated.truncate(good.len() - 1);
        fs::write(&path, &truncated).expect("write truncated");
        assert!(store.load(hash, Role::Thumb, TEST_FIT_EDGE).is_none());

        let mut trailing = good.clone();
        trailing.push(0);
        fs::write(&path, &trailing).expect("write trailing");
        assert!(store.load(hash, Role::Thumb, TEST_FIT_EDGE).is_none());

        fs::write(&path, &good).expect("write intact");
        assert!(store.load(hash, Role::Thumb, TEST_FIT_EDGE).is_some());
    }

    #[test]
    fn disabled_store_is_a_no_op() {
        let dir = TempDir::new().expect("create temp dir");
        fs::write(dir.path().join("not-a-directory"), b"x")
            .expect("write marker file");
        let store = Store::at(&dir.path().join("not-a-directory"));
        assert!(store.state.is_none());
        let page = rgba_page(2, 2);
        let hash = content_hash(b"page-d");
        assert!(
            !store
                .save(hash, Role::Thumb, &page, TEST_FIT_EDGE)
                .persisted
        );
        assert!(store.load(hash, Role::Thumb, TEST_FIT_EDGE).is_none());

        let nested = Store::at(&dir.path().join("not-a-directory/file"));
        assert!(nested.state.is_none());
        assert!(
            !nested.save(hash, Role::Fit, &page, TEST_FIT_EDGE).persisted
        );
    }

    #[test]
    fn derive_scales_to_the_edge() {
        let page = rgba_page(500, 200).with_page_dims((1500, 600));
        let thumb =
            derive_to_edge(&page, 200, FilterType::Box).expect("thumb");
        assert_eq!((thumb.width(), thumb.height()), (200, 80));
        assert_eq!(thumb.page_dims(), (1500, 600));
        assert_eq!(thumb.format(), Format::Png);
        assert_eq!(thumb.transfer(), Transfer::Srgb);
        assert_eq!(
            u64::from(thumb.width()) * 200,
            u64::from(thumb.height()) * 500,
            "the aspect ratio is preserved"
        );

        let wide = rgba_page(1800, 900);
        let wide_thumb = derive_to_edge(&wide, 200, FilterType::Box)
            .expect("wide thumb");
        assert_eq!((wide_thumb.width(), wide_thumb.height()), (200, 100));

        let exact = rgba_page(200, 100);
        let kept = derive_to_edge(&exact, 200, FilterType::Box)
            .expect("exact kept");
        assert_eq!((kept.width(), kept.height()), (200, 100));
        assert_eq!(rgba_bytes(&kept), rgba_bytes(&exact));

        let small = rgba_page(40, 30);
        let grown = derive_to_edge(&small, 1024, FilterType::Box)
            .expect("small kept");
        assert_eq!((grown.width(), grown.height()), (40, 30));
    }

    #[test]
    #[expect(
        clippy::integer_division,
        reason = "the divisions locate a row's midpoint and quarter points; truncating to a pixel is intended"
    )]
    fn derive_preserves_a_hard_edge() {
        let (page_width, page_height) = (4000_u32, 100_u32);
        let page_row_width = count(page_width, 1);
        let mut bytes = vec![0_u8; count(page_width, page_height) * 4];
        for (index, pixel) in
            bytes.as_chunks_mut::<4>().0.iter_mut().enumerate()
        {
            let column = index % page_row_width;
            let value = if column * 2 >= page_row_width {
                [0, 0, 255, 255]
            } else {
                [255, 0, 0, 255]
            };
            pixel.copy_from_slice(&value);
        }
        let page = Decoded::rgba(
            page_width,
            page_height,
            Format::Png,
            Transfer::Srgb,
            bytes.into_boxed_slice(),
        )
        .expect("rgba page");
        let limits = test_limits();
        for (kind, edge, row_count) in [
            (Role::Thumb, THUMB_EDGE, 5_u32),
            (Role::Fit, TEST_FIT_EDGE, 64),
        ] {
            let derived = derive(&page, kind, limits).expect("derives");
            let row_width = count(derived.width(), 1);
            assert_eq!(
                (derived.width(), derived.height()),
                (edge, row_count)
            );
            let pixels = rgba_bytes(&derived);
            let rows: Vec<&[[u8; 4]]> =
                pixels.as_chunks::<4>().0.chunks(row_width).collect();
            let first = rows.first().expect("a row");
            assert_eq!(first.first(), Some(&[255, 0, 0, 255]), "{kind:?}");
            assert_eq!(first.last(), Some(&[0, 0, 255, 255]), "{kind:?}");
            assert_eq!(
                first.get(row_width / 4),
                Some(&[255, 0, 0, 255]),
                "{kind:?}"
            );
            assert_eq!(
                first.get(row_width * 3 / 4),
                Some(&[0, 0, 255, 255]),
                "{kind:?}"
            );
            for (y, row) in rows.iter().enumerate().skip(1) {
                assert_eq!(*row, *first, "{kind:?} row {y}");
            }
            let crossing = first
                .iter()
                .position(|pixel| {
                    let [_, _, blue, _] = *pixel;
                    blue > 127
                })
                .expect("edge crossing");
            assert!(
                crossing.abs_diff(row_width / 2) <= 1,
                "{kind:?}: crossing at {crossing}, edge at {}",
                row_width / 2
            );
        }
    }

    #[test]
    fn derive_keeps_straight_alpha() {
        // Alpha-aware resampling would keep red at 255.
        let bytes = vec![255, 0, 0, 255, 0, 0, 0, 0].into_boxed_slice();
        let page = Decoded::rgba(2, 1, Format::Png, Transfer::Srgb, bytes)
            .expect("rgba page");
        let derived =
            derive_to_edge(&page, 1, FilterType::Box).expect("derives");
        let pixel = rgba_bytes(&derived);
        let [red, green, blue, alpha] = pixel.as_slice() else {
            panic!("one pixel");
        };
        assert!(red.abs_diff(128) <= 1, "{pixel:?}");
        assert_eq!(*green, 0);
        assert_eq!(*blue, 0);
        assert!(alpha.abs_diff(128) <= 1, "{pixel:?}");
    }

    #[test]
    fn derive_keeps_yuv_plane_values() {
        let page = flat_yuv_page(
            4000,
            100,
            Subsampling::Yuv420,
            [400, 600, 300, 1023],
        );
        let limits = test_limits();
        for (kind, edge, row_count) in [
            (Role::Thumb, THUMB_EDGE, 5_u32),
            (Role::Fit, TEST_FIT_EDGE, 64),
        ] {
            let derived = derive(&page, kind, limits).expect("derives");
            assert_eq!(
                (derived.width(), derived.height()),
                (edge, row_count),
                "{kind:?}"
            );
            assert_eq!(derived.page_dims(), (4000, 100), "{kind:?}");
            let Pixels::Yuv(yuv) = derived.pixels() else {
                panic!("expected yuv pixels");
            };
            assert_eq!(yuv.depth, 10, "{kind:?}");
            assert_eq!(yuv.matrix, Matrix::Bt709, "{kind:?}");
            let [luma, cb, cr, alpha] = yuv_planes(&derived);
            for (name, plane, value) in [
                ("luma", luma, 400_u16),
                ("cb", cb, 600),
                ("cr", cr, 300),
                ("alpha", alpha, 1023),
            ] {
                assert!(
                    plane.iter().all(|sample| sample.abs_diff(value) <= 1),
                    "{kind:?} {name}"
                );
            }
        }
    }

    #[test]
    fn derive_keeps_yuv_plane_extents_valid() {
        let page = yuv_page(500, 200, Subsampling::Yuv420);
        let thumb = derive_to_edge(&page, 200, FilterType::Hamming)
            .expect("thumb");
        assert_eq!((thumb.width(), thumb.height()), (200, 80));
        assert_eq!(thumb.page_dims(), (500, 200));
        let Pixels::Yuv(yuv) = thumb.pixels() else {
            panic!("expected yuv pixels");
        };
        assert_eq!(yuv.luma.sample_count(), count(200, 80));
        assert_eq!(yuv.cb.sample_count(), count(100, 40));
        assert_eq!(yuv.cr.sample_count(), count(100, 40));
        assert_eq!(yuv.depth, 10);
        assert_eq!(yuv.matrix, Matrix::Bt709);
    }

    #[test]
    fn derive_keeps_odd_yuv422_chroma_consistent() {
        let page = yuv_page(9, 10, Subsampling::Yuv422);
        let thumb =
            derive_to_edge(&page, 7, FilterType::Box).expect("thumb");
        assert_eq!((thumb.width(), thumb.height()), (6, 7));
        let Pixels::Yuv(yuv) = thumb.pixels() else {
            panic!("expected yuv pixels");
        };
        assert_eq!(yuv.cb.sample_count(), count(3, 7));
        assert_eq!(yuv.cr.sample_count(), count(3, 7));
    }

    #[test]
    fn prune_drops_the_oldest_first() {
        let dir = TempDir::new().expect("create temp dir");
        let root = dir.path().join("cache");
        let probe_root = dir.path().join("probe");
        let page = rgba_page(4, 4);
        let probe_hash = content_hash(b"probe");
        assert!(
            Store::with_caps(&probe_root, THUMB_CAP_BYTES, FIT_CAP_BYTES)
                .save(probe_hash, Role::Thumb, &page, TEST_FIT_EDGE)
                .persisted
        );
        let size = fs::metadata(
            probe_root.join("thumb").join(format!("{probe_hash:032x}")),
        )
        .expect("probe entry metadata")
        .len();
        let store = Store::with_caps(&root, size + 1, FIT_CAP_BYTES);

        let first = content_hash(b"first");
        let second = content_hash(b"second");
        let third = content_hash(b"third");
        let mut evicted = Vec::new();
        for (age, hash) in [(10_u64, first), (11, second), (12, third)] {
            let saved =
                store.save(hash, Role::Thumb, &page, TEST_FIT_EDGE);
            assert!(saved.persisted);
            evicted.extend(saved.evicted);
            let path = root.join("thumb").join(format!("{hash:032x}"));
            set_modified(&path, age);
        }

        // The cap fits one entry, so each save after the first evicted
        // the oldest, and the caller was told which ones.
        assert_eq!(
            evicted,
            vec![(Role::Thumb, first), (Role::Thumb, second)]
        );
        assert!(store.load(first, Role::Thumb, TEST_FIT_EDGE).is_none());
        assert!(store.load(second, Role::Thumb, TEST_FIT_EDGE).is_none());
        assert!(store.load(third, Role::Thumb, TEST_FIT_EDGE).is_some());

        // The fit namespace has its own budget, so a fit save cannot
        // evict thumbnails and reports no evictions.
        let saved = store.save(first, Role::Fit, &page, TEST_FIT_EDGE);
        assert!(saved.persisted && saved.evicted.is_empty());
        assert!(store.load(third, Role::Thumb, TEST_FIT_EDGE).is_some());
    }

    #[test]
    fn derive_clamps_full_to_max_texture_edge() {
        let small = rgba_page(400, 200);
        let limits = ImageLimits::new(300, 500);
        let fit = derive(&small, Role::Fit, limits).expect("fit");
        assert_eq!((fit.width(), fit.height()), (300, 150));

        let full_small = derive(&small, Role::Full, limits).expect("full");
        assert_eq!((full_small.width(), full_small.height()), (400, 200));

        let wide = rgba_page(1000, 200).with_page_dims((1000, 200));
        let full_clamped =
            derive(&wide, Role::Full, limits).expect("clamped full");
        assert_eq!(
            (full_clamped.width(), full_clamped.height()),
            (500, 100)
        );
        assert_eq!(full_clamped.page_dims(), (1000, 200));
    }
}
