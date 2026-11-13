//! Downscaled page previews and the on-disk cache: one directory per role
//! under `env::temp_dir()`, keyed by [`content_hash`]. The little-endian
//! file format is described in `docs/internal/cache.md`.

#![expect(
    clippy::definition_in_module_root,
    reason = "the cache's store and derivation live in its module root by design"
)]
#![expect(
    clippy::little_endian_bytes,
    reason = "the cache file format is defined little-endian"
)]

use std::env;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::fs::File;
use std::fs::FileTimes;
use std::fs::TryLockError;
use std::io;
use std::io::Read as _;
use std::io::Write as _;
use std::mem::DropGuard;
use std::os::unix::fs::DirBuilderExt as _;
use std::os::unix::fs::OpenOptionsExt as _;
use std::path::Path;
use std::path::PathBuf;
use std::process;
use std::time::Instant;
use std::time::SystemTime;

use fast_image_resize::images::Image;
use fast_image_resize::images::ImageRef;
use fast_image_resize::FilterType;
use fast_image_resize::ImageBufferError;
use fast_image_resize::PixelType;
use fast_image_resize::ResizeAlg;
use fast_image_resize::ResizeError;
use fast_image_resize::ResizeOptions;
use fast_image_resize::Resizer;
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
use crate::fault::DecodeRefusal;

pub mod prefetch;
pub mod presentation;

pub const THUMB_EDGE: u32 = 192;
/// The settled reading image; it must cover the fitted page's long edge in
/// physical pixels.
pub const FIT_EDGE: u32 = 2560;
pub const THUMB_CAP_BYTES: u64 = 512 * 1024 * 1024;
pub const FIT_CAP_BYTES: u64 = 3 * 512 * 1024 * 1024;

/// The target edge, stored in the entry header.
fn edge(role: Role) -> Option<u32> {
    derivation(role).map(|(edge, _)| edge)
}

fn derivation(role: Role) -> Option<(u32, FilterType)> {
    match role {
        Role::Thumb => Some((THUMB_EDGE, FilterType::Box)),
        Role::Fit => Some((FIT_EDGE, FilterType::Hamming)),
        Role::Full => None,
    }
}

/// Content address of a page: xxh3-128 over its encoded bytes.
pub fn content_hash(bytes: &[u8]) -> u128 {
    xxh3_128(bytes)
}

/// Downsample to the role's long edge; returns `page` unchanged when it
/// already fits or the resample fails.
pub fn derive(page: &Decoded, role: Role) -> Decoded {
    let Some((edge, filter)) = derivation(role) else {
        return page.clone();
    };
    derive_to_edge(page, edge, filter)
}

fn derive_to_edge(
    page: &Decoded,
    edge: u32,
    filter: FilterType,
) -> Decoded {
    let page_dims = page.page_dims();
    let dims = (page.width(), page.height());
    let target = fit_dims(dims, edge);
    if target == dims {
        return page.clone();
    }
    match resample_page(page, target, filter) {
        Ok(derived) => derived.with_page_dims(page_dims),
        Err(error) => {
            tracing::warn!(
                %error,
                target = ?target,
                "resample failed; keeping the page"
            );
            page.clone()
        }
    }
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

#[derive(Debug)]
enum ResampleError {
    Buffer(ImageBufferError),
    Resize(ResizeError),
    Extent(DecodeRefusal),
}

impl fmt::Display for ResampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buffer(error) => write!(f, "{error}"),
            Self::Resize(error) => write!(f, "{error}"),
            Self::Extent(error) => write!(f, "{error}"),
        }
    }
}

fn resample_page(
    page: &Decoded,
    target: (u32, u32),
    filter: FilterType,
) -> Result<Decoded, ResampleError> {
    let dims = (page.width(), page.height());
    let mut resizer = Resizer::new();
    match page.pixels() {
        Pixels::Rgba8(bytes) => {
            let bytes = resize_buffer(
                &mut resizer,
                bytes,
                dims,
                target,
                PixelType::U8x4,
                filter,
            )?;
            Decoded::rgba(
                target.0,
                target.1,
                page.format(),
                page.transfer(),
                bytes.into_boxed_slice(),
            )
            .map_err(ResampleError::Extent)
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
            Decoded::yuv(
                target.0,
                target.1,
                page.format(),
                page.transfer(),
                planes,
            )
            .map_err(ResampleError::Extent)
        }
    }
}

fn resize_plane(
    resizer: &mut Resizer,
    plane: &Plane,
    source_dims: (u32, u32),
    target_dims: (u32, u32),
    filter: FilterType,
) -> Result<Plane, ResampleError> {
    // Equal extents also cover Mono's empty chroma planes, which carry a
    // 1x1 extent without samples.
    if source_dims == target_dims {
        return Ok(plane.clone());
    }
    Ok(match plane {
        Plane::U8(bytes) => {
            let bytes = resize_buffer(
                resizer,
                bytes,
                source_dims,
                target_dims,
                PixelType::U8,
                filter,
            )?;
            Plane::U8(bytes.into_boxed_slice())
        }
        Plane::U16(words) => {
            let source = bytemuck::cast_slice(words);
            let bytes = resize_buffer(
                resizer,
                source,
                source_dims,
                target_dims,
                PixelType::U16,
                filter,
            )?;
            let pixels = bytemuck::pod_collect_to_vec(&bytes);
            Plane::U16(pixels.into_boxed_slice())
        }
    })
}

fn resize_buffer(
    resizer: &mut Resizer,
    source: &[u8],
    source_dims: (u32, u32),
    target_dims: (u32, u32),
    pixel: PixelType,
    filter: FilterType,
) -> Result<Vec<u8>, ResampleError> {
    let source =
        ImageRef::new(source_dims.0, source_dims.1, source, pixel)
            .map_err(ResampleError::Buffer)?;
    let mut target = Image::new(target_dims.0, target_dims.1, pixel);
    // Straight alpha: the renderer premultiplies at draw. Alpha-aware
    // resampling avoids the color bleed at transparent edges but costs a
    // source-sized temporary and about double the time.
    let options = ResizeOptions::new()
        .resize_alg(ResizeAlg::Convolution(filter))
        .use_alpha(false);
    resizer
        .resize(&source, &mut target, &options)
        .map_err(ResampleError::Resize)?;
    Ok(target.into_vec())
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
    total: u64,
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

    /// Load `hash`, or `None` when absent, damaged, or not cached; a hit
    /// refreshes the entry's mtime for LRU.
    pub fn load(&self, hash: u128, role: Role) -> Option<Decoded> {
        let started = Instant::now();
        let loaded = self.namespace(role).and_then(|namespace| {
            let file = File::open(namespace.path(hash)).ok()?;
            #[expect(
                clippy::let_underscore_must_use,
                clippy::let_underscore_untyped,
                reason = "the recency touch is advisory; its failure changes nothing"
            )]
            let _ = file.set_times(
                FileTimes::new().set_modified(SystemTime::now()),
            );
            read_stored(file, role)
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
        &mut self,
        hash: u128,
        role: Role,
        page: &Decoded,
    ) -> SaveResult {
        let Some(namespace) = self.namespace_mut(role) else {
            return SaveResult::default();
        };
        let path = namespace.path(hash);
        let temp = namespace.temp_path(hash);
        let started = Instant::now();
        let encoded = encode(page, role);
        let size = u64::try_from(encoded.len()).unwrap_or(u64::MAX);
        if write_atomic(&temp, &path, &encoded).is_err() {
            return SaveResult::default();
        }
        namespace.total = namespace.total.saturating_add(size);
        let over = namespace.total > namespace.cap;
        tracing::debug!(
            ?role,
            bytes = size,
            elapsed = ?started.elapsed(),
            "preview save"
        );
        let evicted = if over {
            namespace
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

    fn namespace(&self, role: Role) -> Option<&Namespace> {
        let state = self.state.as_ref()?;
        match role {
            Role::Thumb => Some(&state.thumb),
            Role::Fit => Some(&state.view),
            Role::Full => None,
        }
    }

    fn namespace_mut(&mut self, role: Role) -> Option<&mut Namespace> {
        let state = self.state.as_mut()?;
        match role {
            Role::Thumb => Some(&mut state.thumb),
            Role::Fit => Some(&mut state.view),
            Role::Full => None,
        }
    }
}

impl Namespace {
    fn new(dir: PathBuf, cap: u64) -> Self {
        clean_temps(&dir);
        let total = entries(&dir)
            .iter()
            .map(|entry| entry.size)
            .fold(0_u64, u64::saturating_add);
        Self { dir, cap, total }
    }

    fn path(&self, hash: u128) -> PathBuf {
        self.dir.join(format!("{hash:032x}"))
    }

    /// The per-writer temp name: several processes save into one
    /// namespace, so the pid keeps their in-progress files apart.
    fn temp_path(&self, hash: u128) -> PathBuf {
        self.dir.join(format!("{hash:032x}.{}.tmp", process::id()))
    }

    fn prune(&mut self) -> Vec<u128> {
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
        self.total = total;
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
        let path = dir.join(&name);
        let Ok(file) = File::open(&path) else {
            continue;
        };
        let stale = match file.try_lock() {
            Ok(()) => true,
            Err(TryLockError::WouldBlock) => false,
            Err(TryLockError::Error(error)) => {
                tracing::debug!(
                    %error,
                    path = %path.display(),
                    "preview temp lock failed"
                );
                false
            }
        };
        if !stale {
            continue;
        }
        if let Err(error) = fs::remove_file(&path) {
            tracing::debug!(
                %error,
                path = %path.display(),
                "stale preview temp"
            );
            continue;
        }
        removed += 1;
    }
    if removed > 0 {
        tracing::debug!(
            dir = %dir.display(),
            removed,
            "cleaned stale preview temps"
        );
    }
}

/// Whether `name` is a writer's temp: `<hash:032x>.<pid>.tmp`.
fn is_temp_name(name: &OsStr) -> bool {
    let Some(name) = name.to_str() else {
        return false;
    };
    let Some((hash, rest)) = name.split_once('.') else {
        return false;
    };
    let Some((pid, suffix)) = rest.split_once('.') else {
        return false;
    };
    suffix == "tmp"
        && is_hash_name(OsStr::new(hash))
        && !pid.is_empty()
        && pid.bytes().all(|byte| byte.is_ascii_digit())
}

/// Create the store's directories, refusing a root that is not a plain
/// directory. A symlinked root could move the cache outside the intended
/// tree, so it disables the store instead.
fn prepare(root: &Path) -> Option<(PathBuf, PathBuf)> {
    make_dir(root)?;
    let thumb = root.join("thumb");
    let view = root.join("view");
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

fn write_atomic(
    temp: &Path,
    final_path: &Path,
    bytes: &[u8],
) -> io::Result<()> {
    let guard = DropGuard::new(temp, |path| {
        if let Err(error) = fs::remove_file(path) {
            tracing::debug!(%error, "leftover preview temp file");
        }
    });
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(*guard)?;
    // Hold until the rename: clean_temps treats a free lock as a dead
    // writer.
    file.lock()?;
    file.write_all(bytes)?;
    fs::rename(*guard, final_path)?;
    DropGuard::dismiss(guard);
    Ok(())
}

const MAGIC: [u8; 4] = *b"CYPV";
const VERSION: u16 = 2;

const PIXEL_RGBA: u8 = 0;
const PIXEL_YUV: u8 = 1;

const PLANE_U8: u8 = 0;
const PLANE_U16: u8 = 1;

/// Alpha reuses the plane tag byte, so absence needs its own value.
const ALPHA_ABSENT: u8 = 0;
const ALPHA_U8: u8 = 1;
const ALPHA_U16: u8 = 2;

/// Loose per-axis bound on a dimension a cache header may claim. Real
/// pages stop at the decoder's 0x4000 edge cap, so this is slack;
/// `MAX_SAMPLES` bounds the product.
const MAX_EDGE: u32 = 0x0001_0000;

/// Mirror of the decoder's 512 MiB RGBA allocation ceiling, in pixels.
const MAX_SAMPLES: usize = 128 << 20;

fn encode(page: &Decoded, role: Role) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + page.resident_bytes());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.push(role_tag(role));
    out.push(match page.pixels() {
        Pixels::Rgba8(_) => PIXEL_RGBA,
        Pixels::Yuv(_) => PIXEL_YUV,
    });
    push_u32(&mut out, edge(role).unwrap_or(0));
    for value in [
        page.page_dims().0,
        page.page_dims().1,
        page.width(),
        page.height(),
    ] {
        push_u32(&mut out, value);
    }
    out.push(format_tag(page.format()));
    out.push(transfer_tag(page.transfer()));
    match page.pixels() {
        Pixels::Rgba8(bytes) => {
            push_len(&mut out, bytes.len());
            out.extend_from_slice(bytes);
        }
        Pixels::Yuv(yuv) => {
            out.push(yuv.depth);
            out.push(subsampling_tag(yuv.subsampling));
            out.push(u8::from(yuv.full_range));
            out.push(matrix_tag(yuv.matrix));
            encode_plane(&mut out, &yuv.luma);
            encode_plane(&mut out, &yuv.cb);
            encode_plane(&mut out, &yuv.cr);
            match yuv.alpha.as_ref() {
                None => out.push(ALPHA_ABSENT),
                Some(alpha) => {
                    encode_plane_with(
                        &mut out, alpha, ALPHA_U8, ALPHA_U16,
                    );
                }
            }
        }
    }
    out
}

fn encode_plane(out: &mut Vec<u8>, plane: &Plane) {
    encode_plane_with(out, plane, PLANE_U8, PLANE_U16);
}

fn encode_plane_with(
    out: &mut Vec<u8>,
    plane: &Plane,
    u8_tag: u8,
    u16_tag: u8,
) {
    match plane {
        Plane::U8(bytes) => {
            out.push(u8_tag);
            push_len(out, bytes.len());
            out.extend_from_slice(bytes);
        }
        Plane::U16(words) => {
            out.push(u16_tag);
            push_len(out, words.len() * 2);
            for word in words.as_ref() {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
    }
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_len(out: &mut Vec<u8>, value: usize) {
    push_u32(out, u32::try_from(value).unwrap_or(u32::MAX));
}

fn role_tag(role: Role) -> u8 {
    match role {
        Role::Thumb => 0,
        Role::Fit => 1,
        Role::Full => 2,
    }
}

fn format_tag(format: Format) -> u8 {
    match format {
        Format::Jpeg => 0,
        Format::Png => 1,
        Format::Gif => 2,
        Format::Webp => 3,
        Format::Bmp => 4,
        Format::Tiff => 5,
        Format::Avif => 6,
        Format::Jxl => 7,
    }
}

fn format_from_tag(tag: u8) -> Option<Format> {
    Some(match tag {
        0 => Format::Jpeg,
        1 => Format::Png,
        2 => Format::Gif,
        3 => Format::Webp,
        4 => Format::Bmp,
        5 => Format::Tiff,
        6 => Format::Avif,
        7 => Format::Jxl,
        _ => return None,
    })
}

fn transfer_tag(transfer: Transfer) -> u8 {
    match transfer {
        Transfer::Srgb => 0,
        Transfer::Pq => 1,
    }
}

fn transfer_from_tag(tag: u8) -> Option<Transfer> {
    Some(match tag {
        0 => Transfer::Srgb,
        1 => Transfer::Pq,
        _ => return None,
    })
}

fn subsampling_tag(subsampling: Subsampling) -> u8 {
    match subsampling {
        Subsampling::Mono => 0,
        Subsampling::Yuv420 => 1,
        Subsampling::Yuv422 => 2,
        Subsampling::Yuv444 => 3,
    }
}

fn subsampling_from_tag(tag: u8) -> Option<Subsampling> {
    Some(match tag {
        0 => Subsampling::Mono,
        1 => Subsampling::Yuv420,
        2 => Subsampling::Yuv422,
        3 => Subsampling::Yuv444,
        _ => return None,
    })
}

fn matrix_tag(matrix: Matrix) -> u8 {
    match matrix {
        Matrix::Bt601 => 0,
        Matrix::Bt709 => 1,
        Matrix::Bt2020 => 2,
        Matrix::Identity => 3,
    }
}

fn matrix_from_tag(tag: u8) -> Option<Matrix> {
    Some(match tag {
        0 => Matrix::Bt601,
        1 => Matrix::Bt709,
        2 => Matrix::Bt2020,
        3 => Matrix::Identity,
        _ => return None,
    })
}

/// Header and payload reader over an open cache file. Every read is
/// charged against the file's length, so a damaged header cannot make the
/// loader allocate memory the file does not hold.
struct Reader {
    file: File,
    remaining: u64,
}

impl Reader {
    fn new(file: File) -> Option<Self> {
        let remaining = file.metadata().ok()?.len();
        Some(Self { file, remaining })
    }

    fn at_end(&self) -> bool {
        self.remaining == 0
    }

    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.bytes(N)?.try_into().ok()
    }

    fn bytes(&mut self, len: usize) -> Option<Vec<u8>> {
        let needed = u64::try_from(len).ok()?;
        if needed > self.remaining {
            return None;
        }
        self.remaining -= needed;
        let mut buffer = vec![0_u8; len];
        self.file.read_exact(&mut buffer).ok()?;
        Some(buffer)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(u8::from_le_bytes(self.array()?))
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.array()?))
    }
}

fn read_stored(file: File, role: Role) -> Option<Decoded> {
    let mut reader = Reader::new(file)?;
    if reader.array::<4>()? != MAGIC
        || reader.u16()? != VERSION
        || reader.u8()? != role_tag(role)
    {
        return None;
    }
    let pixel = reader.u8()?;
    if reader.u32()? != edge(role).unwrap_or(0) {
        return None;
    }
    let page_dims = (reader.u32()?, reader.u32()?);
    let image = (reader.u32()?, reader.u32()?);
    let format = format_from_tag(reader.u8()?)?;
    let transfer = transfer_from_tag(reader.u8()?)?;
    sample_count(page_dims)?;
    let page = match pixel {
        PIXEL_RGBA => read_rgba(&mut reader, image, format, transfer)?,
        PIXEL_YUV => read_yuv(&mut reader, image, format, transfer)?,
        _ => return None,
    };
    if !reader.at_end() {
        return None;
    }
    Some(page.with_page_dims(page_dims))
}

fn read_rgba(
    reader: &mut Reader,
    image: (u32, u32),
    format: Format,
    transfer: Transfer,
) -> Option<Decoded> {
    let want = sample_count(image)?.checked_mul(4)?;
    if usize::try_from(reader.u32()?).ok()? != want {
        return None;
    }
    let bytes = reader.bytes(want)?.into_boxed_slice();
    Decoded::rgba(image.0, image.1, format, transfer, bytes).ok()
}

fn read_yuv(
    reader: &mut Reader,
    image: (u32, u32),
    format: Format,
    transfer: Transfer,
) -> Option<Decoded> {
    let depth = reader.u8()?;
    let subsampling = subsampling_from_tag(reader.u8()?)?;
    let full_range = match reader.u8()? {
        0 => false,
        1 => true,
        _ => return None,
    };
    let matrix = matrix_from_tag(reader.u8()?)?;
    let ((luma_w, luma_h), (chroma_w, chroma_h)) =
        decode::plane_dims(image, subsampling);
    let luma = read_plane(reader, sample_count((luma_w, luma_h))?)?;
    let chroma = if subsampling == Subsampling::Mono {
        0
    } else {
        sample_count((chroma_w, chroma_h))?
    };
    let cb = read_plane(reader, chroma)?;
    let cr = read_plane(reader, chroma)?;
    let alpha = match reader.u8()? {
        ALPHA_ABSENT => None,
        tag => {
            let tag = alpha_plane_tag(tag)?;
            let wanted = sample_count((luma_w, luma_h))?;
            Some(read_plane_tag(reader, tag, wanted)?)
        }
    };
    let yuv = YuvImage {
        depth,
        subsampling,
        full_range,
        matrix,
        luma,
        cb,
        cr,
        alpha,
    };
    Decoded::yuv(image.0, image.1, format, transfer, yuv).ok()
}

fn alpha_plane_tag(tag: u8) -> Option<u8> {
    Some(match tag {
        ALPHA_U8 => PLANE_U8,
        ALPHA_U16 => PLANE_U16,
        _ => return None,
    })
}

fn read_plane(reader: &mut Reader, samples: usize) -> Option<Plane> {
    let tag = reader.u8()?;
    read_plane_tag(reader, tag, samples)
}

fn read_plane_tag(
    reader: &mut Reader,
    tag: u8,
    samples: usize,
) -> Option<Plane> {
    let bytes_per_sample = match tag {
        PLANE_U8 => 1,
        PLANE_U16 => 2,
        _ => return None,
    };
    let want = samples.checked_mul(bytes_per_sample)?;
    if usize::try_from(reader.u32()?).ok()? != want {
        return None;
    }
    let bytes = reader.bytes(want)?;
    Some(if bytes_per_sample == 1 {
        Plane::U8(bytes.into_boxed_slice())
    } else {
        Plane::U16(words(&bytes))
    })
}

fn words(bytes: &[u8]) -> Box<[u16]> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

/// Pixel count of a `dims` plane, refusing implausible extents and more
/// samples than the decoder could ever allocate.
fn sample_count(dims: (u32, u32)) -> Option<usize> {
    let (width, height) = dims;
    if width == 0 || height == 0 || width > MAX_EDGE || height > MAX_EDGE {
        return None;
    }
    let count = u64::from(width) * u64::from(height);
    let count = usize::try_from(count).ok()?;
    (count <= MAX_SAMPLES).then_some(count)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::browse::Roles;
    use crate::testkit::TempDir;

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

    fn temp_name(hash: u128, pid: u32) -> String {
        format!("{hash:032x}.{pid}.tmp")
    }

    #[test]
    fn temp_names_carry_the_writer_pid() {
        let dir = TempDir::new("preview-temp-pid");
        let store = Store::at(dir.path());
        let thumb = store.namespace(Role::Thumb).expect("thumb");
        let hash = content_hash(b"temp-pid");
        let expected = dir
            .path()
            .join("thumb")
            .join(temp_name(hash, process::id()));
        let path = thumb.temp_path(hash);
        assert_eq!(path, expected);
        assert!(
            path.file_name().is_some_and(is_temp_name),
            "the cleaner must recognize its own writer temp"
        );
    }

    #[test]
    fn open_cleans_unlocked_writer_temps() {
        let dir = TempDir::new("preview-stale-temps");
        let root = dir.path().join("cache");
        let page = rgba_page(2, 2);
        let hash = content_hash(b"page-kept");
        let mut store = Store::at(&root);
        assert!(store.save(hash, Role::Thumb, &page).persisted);
        drop(store);

        // A dead writer leaves its temp unlocked.
        let stale_pid = 4242;
        let stale_thumb =
            root.join("thumb").join(temp_name(hash, stale_pid));
        fs::write(&stale_thumb, b"partial").expect("thumb temp");
        let stale_view =
            root.join("view").join(temp_name(hash, stale_pid));
        fs::write(&stale_view, b"partial").expect("view temp");

        // A live writer's temp stays while it holds the lock.
        let live_pid = 4243;
        let live_temp = root.join("thumb").join(temp_name(hash, live_pid));
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
        assert!(reopened.load(hash, Role::Thumb).is_some());

        // Closing the handle releases the lock, so the next open claims
        // the temp.
        drop(locked_temp);
        Store::at(&root);
        assert!(!live_temp.exists());
    }

    #[test]
    fn rgba_roundtrip_preserves_the_page() {
        let dir = TempDir::new("preview-rgba");
        let page = rgba_page(3, 2).with_page_dims((9, 6));
        let mut store = Store::at(dir.path());
        assert!(store.state.is_some());
        let hash = content_hash(b"page-a");
        assert!(store.save(hash, Role::Thumb, &page).persisted);
        assert!(!store.save(hash, Role::Full, &page).persisted);
        assert!(store.load(hash, Role::Full).is_none());

        let loaded = store.load(hash, Role::Thumb).expect("cached thumb");
        assert_eq!((loaded.width(), loaded.height()), (3, 2));
        assert_eq!(loaded.page_dims(), (9, 6));
        assert_eq!(loaded.format(), Format::Png);
        assert_eq!(loaded.transfer(), Transfer::Srgb);
        assert_eq!(rgba_bytes(&loaded), rgba_bytes(&page));
    }

    #[test]
    fn yuv_plane_roundtrip_preserves_the_page() {
        let dir = TempDir::new("preview-yuv");
        let page =
            yuv_page(5, 3, Subsampling::Yuv420).with_page_dims((15, 9));
        let mut store = Store::at(dir.path());
        let hash = content_hash(b"page-b");
        assert!(store.save(hash, Role::Fit, &page).persisted);

        let loaded = store.load(hash, Role::Fit).expect("cached view");
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
        let dir = TempDir::new("preview-yuv8");
        let page = yuv_page_with(6, 4, Subsampling::Yuv444, 8, true);
        let mut store = Store::at(dir.path());
        let hash = content_hash(b"page-e");
        assert!(store.save(hash, Role::Thumb, &page).persisted);

        let loaded = store.load(hash, Role::Thumb).expect("cached thumb");
        let Pixels::Yuv(yuv) = loaded.pixels() else {
            panic!("expected yuv pixels");
        };
        assert_eq!(yuv.depth, 8);
        assert!(yuv.alpha.is_some());
        assert_eq!(yuv_planes(&loaded), yuv_planes(&page));
    }

    #[test]
    fn corrupt_entries_are_rejected() {
        let dir = TempDir::new("preview-corrupt");
        let page = rgba_page(2, 2);
        let mut store = Store::at(dir.path());
        let hash = content_hash(b"page-c");
        assert!(store.save(hash, Role::Thumb, &page).persisted);
        let path = dir.path().join("thumb").join(format!("{hash:032x}"));
        let good = fs::read(&path).expect("read cache file");

        let mut bad_magic = good.clone();
        if let Some(byte) = bad_magic.first_mut() {
            *byte = b'X';
        }
        fs::write(&path, &bad_magic).expect("write magic");
        assert!(store.load(hash, Role::Thumb).is_none());

        let mut bad_version = good.clone();
        if let Some(field) = bad_version.get_mut(4..6) {
            field.copy_from_slice(&(VERSION + 1).to_le_bytes());
        }
        fs::write(&path, &bad_version).expect("write version");
        assert!(store.load(hash, Role::Thumb).is_none());

        // The target edge sits between the pixel tag and the page dims;
        // a different edge is a different role.
        let mut bad_edge = good.clone();
        if let Some(field) = bad_edge.get_mut(8..12) {
            field.copy_from_slice(&(THUMB_EDGE + 1).to_le_bytes());
        }
        fs::write(&path, &bad_edge).expect("write edge");
        assert!(store.load(hash, Role::Thumb).is_none());

        // The RGBA payload length sits right after the 30 header bytes.
        let mut bad_length = good.clone();
        if let Some(field) = bad_length.get_mut(30..34) {
            field.copy_from_slice(&1_u32.to_le_bytes());
        }
        fs::write(&path, &bad_length).expect("write length");
        assert!(store.load(hash, Role::Thumb).is_none());

        let mut truncated = good.clone();
        truncated.truncate(good.len() - 1);
        fs::write(&path, &truncated).expect("write truncated");
        assert!(store.load(hash, Role::Thumb).is_none());

        let mut trailing = good.clone();
        trailing.push(0);
        fs::write(&path, &trailing).expect("write trailing");
        assert!(store.load(hash, Role::Thumb).is_none());

        fs::write(&path, &good).expect("write intact");
        assert!(store.load(hash, Role::Thumb).is_some());
    }

    #[test]
    fn disabled_store_is_a_no_op() {
        let dir = TempDir::new("preview-disabled");
        dir.write("not-a-directory", b"x");
        let mut store = Store::at(&dir.path().join("not-a-directory"));
        assert!(store.state.is_none());
        let page = rgba_page(2, 2);
        let hash = content_hash(b"page-d");
        assert!(!store.save(hash, Role::Thumb, &page).persisted);
        assert!(store.load(hash, Role::Thumb).is_none());

        let mut nested =
            Store::at(&dir.path().join("not-a-directory/file"));
        assert!(nested.state.is_none());
        assert!(!nested.save(hash, Role::Fit, &page).persisted);
    }

    #[test]
    fn derive_scales_to_the_edge() {
        let page = rgba_page(500, 200).with_page_dims((1500, 600));
        let thumb = derive_to_edge(&page, 200, FilterType::Box);
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
        let wide_thumb = derive_to_edge(&wide, 200, FilterType::Box);
        assert_eq!((wide_thumb.width(), wide_thumb.height()), (200, 100));

        let exact = rgba_page(200, 100);
        let kept = derive_to_edge(&exact, 200, FilterType::Box);
        assert_eq!((kept.width(), kept.height()), (200, 100));
        assert_eq!(rgba_bytes(&kept), rgba_bytes(&exact));

        let small = rgba_page(40, 30);
        let grown = derive_to_edge(&small, 1024, FilterType::Box);
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
        for (kind, edge, row_count) in
            [(Role::Thumb, THUMB_EDGE, 5_u32), (Role::Fit, FIT_EDGE, 64)]
        {
            let derived = derive(&page, kind);
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
        let derived = derive_to_edge(&page, 1, FilterType::Box);
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
        for (kind, edge, row_count) in
            [(Role::Thumb, THUMB_EDGE, 5_u32), (Role::Fit, FIT_EDGE, 64)]
        {
            let derived = derive(&page, kind);
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
        let thumb = derive_to_edge(&page, 200, FilterType::Hamming);
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
        let thumb = derive_to_edge(&page, 7, FilterType::Box);
        assert_eq!((thumb.width(), thumb.height()), (6, 7));
        let Pixels::Yuv(yuv) = thumb.pixels() else {
            panic!("expected yuv pixels");
        };
        assert_eq!(yuv.cb.sample_count(), count(3, 7));
        assert_eq!(yuv.cr.sample_count(), count(3, 7));
    }

    #[test]
    fn prune_drops_the_oldest_first() {
        let dir = TempDir::new("preview-prune");
        let root = dir.path().join("cache");
        let page = rgba_page(4, 4);
        let size =
            u64::try_from(encode(&page, Role::Thumb).len()).expect("size");
        let mut store = Store::with_caps(&root, size + 1, FIT_CAP_BYTES);

        let first = content_hash(b"first");
        let second = content_hash(b"second");
        let third = content_hash(b"third");
        let mut evicted = Vec::new();
        for (age, hash) in [(10_u64, first), (11, second), (12, third)] {
            let saved = store.save(hash, Role::Thumb, &page);
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
        assert!(store.load(first, Role::Thumb).is_none());
        assert!(store.load(second, Role::Thumb).is_none());
        assert!(store.load(third, Role::Thumb).is_some());

        // The fit namespace has its own budget, so a fit save cannot
        // evict thumbnails and reports no evictions.
        let saved = store.save(first, Role::Fit, &page);
        assert!(saved.persisted && saved.evicted.is_empty());
        assert!(store.load(third, Role::Thumb).is_some());
    }

    #[test]
    fn roles_prefer_the_largest_role() {
        let mut roles = Roles::default();
        roles.set(Role::Thumb);
        assert_eq!(roles.best(), Some(Role::Thumb));
        assert!(roles.contains(Role::Thumb));
        assert!(roles.any());
        assert!(!roles.has_settled_image());

        roles.set(Role::Fit);
        assert_eq!(roles.best(), Some(Role::Fit));
        assert!(roles.has_settled_image());

        roles.set(Role::Full);
        assert_eq!(roles.best(), Some(Role::Full));

        roles.clear(Role::Full);
        assert_eq!(roles.best(), Some(Role::Fit));
        assert!(!roles.contains(Role::Full));

        roles.clear(Role::Fit);
        assert_eq!(roles.best(), Some(Role::Thumb));
        assert!(!roles.has_settled_image());
        assert!(roles.any());

        roles.clear(Role::Thumb);
        assert_eq!(roles.best(), None);
        assert!(!roles.any());
    }
}
