//! Downscaled page derivation and the shared parent thumbnail store
//! (`$XDG_CACHE_HOME/cryslen/thumbs-v1.redb`).

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::ErrorKind;
use std::mem;
use std::ops::RangeInclusive;
use std::path::Path;
use std::path::PathBuf;
use std::thread;
use std::time::SystemTime;

use bytemuck::Pod;
use fast_image_resize::FilterType;
use fast_image_resize::PixelTrait;
use fast_image_resize::Resizer;
use fast_image_resize::pixels::Pixel;
use fast_image_resize::pixels::PixelComponent;
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

use crate::browse::WorkRole;
use crate::catalog;
use crate::decode;
use crate::decode::Decoded;
use crate::decode::Format;
use crate::decode::GrayImage;
use crate::decode::GrayPlanes;
use crate::decode::Pixels;
use crate::decode::Samples;
use crate::decode::Subsampling;
use crate::decode::Transfer;
use crate::decode::YuvImage;
use crate::decode::YuvPlanes;
use crate::protocol::ImageLimits;
use crate::protocol::THUMB_EDGE;
use crate::protocol::TextureFault;
use crate::protocol::ValidatedThumb;
use crate::protocol::WireThumb;

const THUMB_PACK_CAP_BYTES: u64 = 512 * 1024 * 1024;
const THUMB_PACK_PRUNE_NUMERATOR: u64 = 4;
const THUMB_PACK_PRUNE_DENOMINATOR: u64 = 5;
const FLUSH_ENTRY_THRESHOLD: usize = 32;
const THUMB_WEBP_QUALITY: f32 = 78.0;

fn derivation(
    role: WorkRole,
    limits: ImageLimits,
) -> ((u32, u32), FilterType) {
    match role {
        WorkRole::Fit => (limits.fit_box(), FilterType::Lanczos3),
        WorkRole::Full => {
            let max_edge = limits.max_texture_edge();
            ((max_edge, max_edge), FilterType::Lanczos3)
        }
    }
}

/// Downsample to the role's bounding box; borrows `page` unchanged when
/// it fits or resampling fails within `max_texture_edge`, and returns
/// `TextureFault` past `max_texture_edge`.
pub fn derive(
    page: &Decoded,
    role: WorkRole,
    limits: ImageLimits,
    max_threads: usize,
) -> Result<Cow<'_, Decoded>, TextureFault> {
    let (bounds, filter) = derivation(role, limits);
    let error = match derive_to_box(page, bounds, filter, max_threads) {
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

/// Derive a compact WebP thumbnail (`<= THUMB_EDGE` per axis) from a
/// decoded or `Fit`-sized page.
pub fn derive_thumb(
    page: &Decoded,
    max_threads: usize,
) -> eros::Result<ValidatedThumb> {
    let thumb = derive_to_box(
        page,
        (THUMB_EDGE, THUMB_EDGE),
        FilterType::Box,
        max_threads,
    )?;
    let rgba = thumb_rgba(&thumb);
    let webp = decode::encode_webp_rgba(
        (thumb.width(), thumb.height()),
        &rgba,
        THUMB_WEBP_QUALITY,
    )?;
    let wire = WireThumb {
        page_dims: thumb.page_dims(),
        format: thumb.format(),
        webp,
    };
    Ok(wire.validate()?)
}

fn thumb_rgba(page: &Decoded) -> Cow<'_, [u8]> {
    const RGB_CHANNELS: usize = 3;

    let dims = (page.width(), page.height());
    let transfer = page.transfer();
    match page.pixels() {
        Pixels::Rgba8(raw) => {
            if transfer == Transfer::Srgb {
                return Cow::Borrowed(raw);
            }
            let mut rgba = raw.to_vec();
            for pixel in
                rgba.as_chunks_mut::<{ decode::RGBA_BYTES_PER_PIXEL }>().0
            {
                for channel in pixel.iter_mut().take(RGB_CHANNELS) {
                    *channel = quantize_unit(decode::pq_to_srgb_channel(
                        f32::from(*channel) / decode::R8_STORAGE_MAX,
                    ));
                }
            }
            Cow::Owned(rgba)
        }
        Pixels::Gray(gray) => Cow::Owned(encode_gray_rgba8(gray, transfer)),
        Pixels::Yuv(yuv) => Cow::Owned(encode_yuv_rgba8(yuv, dims, transfer)),
    }
}

fn encode_gray_rgba8(gray: &GrayImage, transfer: Transfer) -> Vec<u8> {
    match &gray.samples {
        Samples::U8(planes) => gray_planes_to_rgba8(gray, transfer, planes),
        Samples::U16 { planes, .. } => {
            gray_planes_to_rgba8(gray, transfer, planes)
        }
    }
}

fn gray_planes_to_rgba8<Sample>(
    gray: &GrayImage,
    transfer: Transfer,
    planes: &GrayPlanes<Sample>,
) -> Vec<u8>
where
    Sample: Copy,
    f32: From<Sample>,
{
    let depth = gray.depth();
    let (gain, offset) = decode::gray_gain_offset(depth, gray.full_range);
    let storage_max = decode::max_storage_code(depth);
    let alpha_normalization = decode::depth_normalization(depth);
    let map_luma = |sample: f32| {
        let luma = (sample / storage_max).mul_add(gain, -offset);
        let srgb = apply_transfer(luma.clamp(0.0, 1.0), transfer);
        quantize_unit(srgb)
    };
    let map_alpha = |sample: f32| {
        quantize_unit(
            (sample / storage_max * alpha_normalization).clamp(0.0, 1.0),
        )
    };

    let mut bytes =
        Vec::with_capacity(planes.luma.len() * decode::RGBA_BYTES_PER_PIXEL);
    for (index, luma_sample) in planes.luma.iter().enumerate() {
        let gray_byte = map_luma(f32::from(*luma_sample));
        let alpha_byte = planes.alpha.as_deref().map_or(u8::MAX, |plane| {
            plane
                .get(index)
                .copied()
                .map_or(u8::MAX, |sample| map_alpha(f32::from(sample)))
        });
        bytes.extend_from_slice(&[gray_byte, gray_byte, gray_byte, alpha_byte]);
    }
    bytes
}

fn encode_yuv_rgba8(
    yuv: &YuvImage,
    dims: (u32, u32),
    transfer: Transfer,
) -> Vec<u8> {
    match &yuv.samples {
        Samples::U8(planes) => yuv_planes_to_rgba8(yuv, dims, transfer, planes),
        Samples::U16 { planes, .. } => {
            yuv_planes_to_rgba8(yuv, dims, transfer, planes)
        }
    }
}

fn yuv_planes_to_rgba8<Sample>(
    yuv: &YuvImage,
    (width, height): (u32, u32),
    transfer: Transfer,
    planes: &YuvPlanes<Sample>,
) -> Vec<u8>
where
    Sample: Copy,
    f32: From<Sample>,
{
    let depth = yuv.depth();
    let conversion =
        decode::yuv_color_conversion(depth, yuv.full_range, yuv.matrix);
    let storage_max = decode::max_storage_code(depth);
    let alpha_normalization = decode::depth_normalization(depth);
    let (_, chroma_dims) = decode::plane_dims((width, height), yuv.subsampling);
    let (image_width, image_height) = (width as usize, height as usize);
    let (chroma_width, chroma_height) =
        (chroma_dims.0 as usize, chroma_dims.1 as usize);
    let (shift_x, shift_y) = match yuv.subsampling {
        Subsampling::Yuv444 => (0_u32, 0_u32),
        Subsampling::Yuv422 => (1, 0),
        Subsampling::Yuv420 => (1, 1),
    };
    let mut rgba = Vec::with_capacity(
        image_width * image_height * decode::RGBA_BYTES_PER_PIXEL,
    );
    for row in 0..image_height {
        let chroma_row =
            (row >> shift_y).min(chroma_height.saturating_sub(1));
        for column in 0..image_width {
            let chroma_column =
                (column >> shift_x).min(chroma_width.saturating_sub(1));
            let luma_index = row * image_width + column;
            let chroma_index = chroma_row * chroma_width + chroma_column;
            let luma = planes
                .luma
                .get(luma_index)
                .copied()
                .map_or(0.0, f32::from)
                / storage_max;
            let blue_chroma = planes
                .cb
                .get(chroma_index)
                .copied()
                .map_or(0.0, f32::from)
                / storage_max;
            let red_chroma = planes
                .cr
                .get(chroma_index)
                .copied()
                .map_or(0.0, f32::from)
                / storage_max;
            let rgb = conversion.apply([luma, blue_chroma, red_chroma]);
            let alpha_byte =
                planes.alpha.as_deref().map_or(u8::MAX, |plane| {
                    let alpha =
                        plane.get(luma_index).copied().map_or(1.0, f32::from)
                            / storage_max
                            * alpha_normalization;
                    quantize_unit(alpha.clamp(0.0, 1.0))
                });
            for channel in rgb {
                rgba.push(quantize_unit(apply_transfer(
                    channel.clamp(0.0, 1.0),
                    transfer,
                )));
            }
            rgba.push(alpha_byte);
        }
    }
    rgba
}

fn apply_transfer(channel: f32, transfer: Transfer) -> f32 {
    match transfer {
        Transfer::Srgb => channel,
        Transfer::Pq => decode::pq_to_srgb_channel(channel),
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "channel is clamped to 0.0..=1.0 before scaling to 0..=255"
)]
fn quantize_unit(channel: f32) -> u8 {
    (channel.clamp(0.0, 1.0) * decode::R8_STORAGE_MAX).round() as u8
}

fn derive_to_box(
    page: &Decoded,
    bounds: (u32, u32),
    filter: FilterType,
    max_threads: usize,
) -> eros::Result<Cow<'_, Decoded>> {
    let page_dims = page.page_dims();
    let dims = (page.width(), page.height());
    let target = fit_dims(dims, bounds);
    if target == dims {
        return Ok(Cow::Borrowed(page));
    }
    resample_page(page, target, filter, max_threads)
        .map(|derived| Cow::Owned(derived.with_page_dims(page_dims)))
}

#[expect(
    clippy::integer_division,
    reason = "box scaling is integer pixel arithmetic"
)]
pub fn fit_dims(dims: (u32, u32), bounds: (u32, u32)) -> (u32, u32) {
    let (width, height) = (u64::from(dims.0), u64::from(dims.1));
    let (max_width, max_height) =
        (u64::from(bounds.0), u64::from(bounds.1));
    if width <= max_width && height <= max_height {
        return dims;
    }
    if width * max_height >= height * max_width {
        let scaled_height = (height * max_width + width / 2) / width;
        let target_height =
            u32::try_from(scaled_height.clamp(1, max_height))
                .unwrap_or(bounds.1);
        (bounds.0, target_height)
    } else {
        let scaled_width = (width * max_height + height / 2) / height;
        let target_width = u32::try_from(scaled_width.clamp(1, max_width))
            .unwrap_or(bounds.0);
        (target_width, bounds.1)
    }
}

fn resample_page(
    page: &Decoded,
    target: (u32, u32),
    filter: FilterType,
    max_threads: usize,
) -> eros::Result<Decoded> {
    let dims = (page.width(), page.height());
    match page.pixels() {
        Pixels::Rgba8(bytes) => {
            let mut resizer = Resizer::new();
            let pixels = resize_typed::<_, u8, { decode::RGBA_BYTES_PER_PIXEL }>(
                &mut resizer,
                bytes.as_chunks::<{ decode::RGBA_BYTES_PER_PIXEL }>().0,
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
        Pixels::Gray(gray) => {
            let samples = match &gray.samples {
                Samples::U8(planes) => Samples::U8(resize_gray::<u8>(
                    planes,
                    dims,
                    target,
                    filter,
                    max_threads,
                )?),
                Samples::U16 { depth, planes } => Samples::U16 {
                    depth: *depth,
                    planes: resize_gray::<u16>(
                        planes,
                        dims,
                        target,
                        filter,
                        max_threads,
                    )?,
                },
            };
            Ok(Decoded::gray(
                target.0,
                target.1,
                page.format(),
                page.transfer(),
                GrayImage {
                    full_range: gray.full_range,
                    samples,
                },
            )?)
        }
        Pixels::Yuv(yuv) => {
            let samples = match &yuv.samples {
                Samples::U8(planes) => Samples::U8(resize_yuv::<u8>(
                    planes,
                    dims,
                    target,
                    yuv.subsampling,
                    filter,
                    max_threads,
                )?),
                Samples::U16 { depth, planes } => Samples::U16 {
                    depth: *depth,
                    planes: resize_yuv::<u16>(
                        planes,
                        dims,
                        target,
                        yuv.subsampling,
                        filter,
                        max_threads,
                    )?,
                },
            };
            Ok(Decoded::yuv(
                target.0,
                target.1,
                page.format(),
                page.transfer(),
                YuvImage {
                    subsampling: yuv.subsampling,
                    full_range: yuv.full_range,
                    matrix: yuv.matrix,
                    samples,
                },
            )?)
        }
    }
}

fn resize_gray<Sample>(
    planes: &GrayPlanes<Sample>,
    dims: (u32, u32),
    target: (u32, u32),
    filter: FilterType,
    max_threads: usize,
) -> eros::Result<GrayPlanes<Sample>>
where
    Sample: Copy + Default + PartialEq + Pod + PixelComponent + Send + Sync,
    Pixel<Sample, Sample, 1>: PixelTrait,
{
    let mut specs = vec![(&*planes.luma, dims, target)];
    if let Some(alpha) = planes.alpha.as_deref() {
        specs.push((alpha, dims, target));
    }
    let mut resized = resize_planes(&specs, filter, max_threads)?.into_iter();
    Ok(GrayPlanes {
        luma: resized
            .next()
            .ok_or_else(|| eros::error!("missing luma plane"))?,
        alpha: resized.next(),
    })
}

fn resize_yuv<Sample>(
    planes: &YuvPlanes<Sample>,
    dims: (u32, u32),
    target: (u32, u32),
    subsampling: Subsampling,
    filter: FilterType,
    max_threads: usize,
) -> eros::Result<YuvPlanes<Sample>>
where
    Sample: Copy + Default + PartialEq + Pod + PixelComponent + Send + Sync,
    Pixel<Sample, Sample, 1>: PixelTrait,
{
    let (luma_dims, chroma_dims) = decode::plane_dims(dims, subsampling);
    let (_, target_chroma_dims) = decode::plane_dims(target, subsampling);
    let mut specs = vec![
        (&*planes.luma, luma_dims, target),
        (&*planes.cb, chroma_dims, target_chroma_dims),
        (&*planes.cr, chroma_dims, target_chroma_dims),
    ];
    if let Some(alpha) = planes.alpha.as_deref() {
        specs.push((alpha, luma_dims, target));
    }
    let mut resized = resize_planes(&specs, filter, max_threads)?.into_iter();
    Ok(YuvPlanes {
        luma: resized
            .next()
            .ok_or_else(|| eros::error!("missing luma plane"))?,
        cb: resized
            .next()
            .ok_or_else(|| eros::error!("missing cb plane"))?,
        cr: resized
            .next()
            .ok_or_else(|| eros::error!("missing cr plane"))?,
        alpha: resized.next(),
    })
}

type PlaneResize<'source, Sample> =
    (&'source [Sample], (u32, u32), (u32, u32));

fn resize_planes<Sample>(
    specs: &[PlaneResize<'_, Sample>],
    filter: FilterType,
    max_threads: usize,
) -> eros::Result<Vec<Box<[Sample]>>>
where
    Sample: Copy + Default + PartialEq + Pod + PixelComponent + Send + Sync,
    Pixel<Sample, Sample, 1>: PixelTrait,
{
    let Some((&first, rest)) = specs.split_first() else {
        return Ok(Vec::new());
    };
    if max_threads <= 1 || rest.is_empty() {
        let mut resizer = Resizer::new();
        return specs
            .iter()
            .map(|&(source, source_dims, target_dims)| {
                resize_plane(
                    &mut resizer,
                    source,
                    source_dims,
                    target_dims,
                    filter,
                )
            })
            .collect();
    }
    thread::scope(|scope| {
        let handles: Vec<_> = rest
            .iter()
            .map(|&(source, source_dims, target_dims)| {
                scope.spawn(move || {
                    let mut resizer = Resizer::new();
                    resize_plane(
                        &mut resizer,
                        source,
                        source_dims,
                        target_dims,
                        filter,
                    )
                })
            })
            .collect();
        let mut resizer = Resizer::new();
        let mut planes = Vec::with_capacity(specs.len());
        planes.push(resize_plane(
            &mut resizer,
            first.0,
            first.1,
            first.2,
            filter,
        )?);
        for handle in handles {
            planes.push(
                handle
                    .join()
                    .map_err(|_panic| eros::error!("plane resample panicked"))??,
            );
        }
        Ok(planes)
    })
}

fn resize_plane<Sample>(
    resizer: &mut Resizer,
    source: &[Sample],
    source_dims: (u32, u32),
    target_dims: (u32, u32),
    filter: FilterType,
) -> eros::Result<Box<[Sample]>>
where
    Sample: Copy + Default + PartialEq + Pod + PixelComponent,
    Pixel<Sample, Sample, 1>: PixelTrait,
{
    if source_dims == target_dims {
        return Ok(source.into());
    }
    Ok(resize_typed::<Sample, Sample, 1>(
        resizer,
        source,
        source_dims,
        target_dims,
        filter,
    )?
    .into_boxed_slice())
}

fn resize_typed<Element, Component, const CHANNELS: usize>(
    resizer: &mut Resizer,
    source: &[Element],
    source_dims: (u32, u32),
    target_dims: (u32, u32),
    filter: FilterType,
) -> eros::Result<Vec<Element>>
where
    Element: Default + PartialEq + Pod,
    Component: PixelComponent,
    Pixel<Element, Component, CHANNELS>: PixelTrait,
{
    use fast_image_resize::ResizeAlg;
    use fast_image_resize::ResizeOptions;
    use fast_image_resize::images::TypedImage;
    use fast_image_resize::images::TypedImageRef;

    let Ok(element_count) = usize::try_from(
        u64::from(target_dims.0) * u64::from(target_dims.1),
    ) else {
        eros::bail!("buffer count overflows usize");
    };
    let mut buffer = vec![Element::default(); element_count];
    let source = TypedImageRef::<Pixel<Element, Component, CHANNELS>>::from_buffer(
        source_dims.0,
        source_dims.1,
        bytemuck::cast_slice(source),
    )?;
    let mut target = TypedImage::<Pixel<Element, Component, CHANNELS>>::from_buffer(
        target_dims.0,
        target_dims.1,
        bytemuck::cast_slice_mut(&mut buffer),
    )?;
    // Straight alpha: the renderer premultiplies at draw. Alpha-aware
    // resampling avoids the color bleed at transparent edges but costs a
    // source-sized temporary and about double the time.
    let options = ResizeOptions::new()
        .resize_alg(ResizeAlg::Convolution(filter))
        .use_alpha(false);
    resizer.resize_typed(&source, &mut target, &options)?;
    Ok(buffer)
}

const THUMB_DB_FILENAME: &str = "thumbs-v1.redb";
const THUMB_DB_CACHE_BYTES: usize = 8 * 1024 * 1024;
const THUMB_DB_FILE_MODE: u32 = 0o600;

type StoredThumb<'bytes> = (u32, u32, u8, &'bytes [u8]);

const BOOKS_TABLE: TableDefinition<'_, [u8; 16], (u64, u64)> =
    TableDefinition::new("books");
const THUMBS_TABLE: TableDefinition<'_, ([u8; 16], u32), StoredThumb<'_>> =
    TableDefinition::new("thumbs");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThumbImage {
    dims: (u32, u32),
    page_dims: (u32, u32),
    format: Format,
    rgba: Vec<u8>,
}

impl ThumbImage {
    #[must_use]
    pub const fn dims(&self) -> (u32, u32) {
        self.dims
    }

    #[must_use]
    pub const fn page_dims(&self) -> (u32, u32) {
        self.page_dims
    }

    #[must_use]
    pub const fn format(&self) -> Format {
        self.format
    }

    #[must_use]
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }
}

/// Parent-owned shared thumbnail cache under
/// `$XDG_CACHE_HOME/cryslen/thumbs-v1.redb`.
pub struct ThumbPack {
    db: Database,
    book_key: [u8; 16],
    present: Vec<bool>,
    pending: BTreeMap<usize, ValidatedThumb>,
    pruned_once: bool,
}

impl ThumbPack {
    pub fn open(path: &Path, archive: bool, names: &[String]) -> Option<Self> {
        let root = default_cache_dir()?;
        let key = book_key(path, archive, names)?;
        Self::open_at(&root, key, names.len())
    }

    pub fn open_at(
        root: &Path,
        book_key: [u8; 16],
        page_count: usize,
    ) -> Option<Self> {
        if make_dir(root).is_none() {
            tracing::warn!(root = %root.display(), "thumb pack disabled");
            return None;
        }
        let db_path = root.join(THUMB_DB_FILENAME);
        let (db, present, existing_last_used) =
            open_or_recreate(&db_path, book_key, page_count)?;
        let warm = present.iter().filter(|&&hit| hit).count();
        tracing::debug!(warm, total = page_count, "thumb pack opened");
        let now_secs = now_unix_secs();
        if let Some(last_used) = existing_last_used
            && last_used != now_secs
            && let Err(error) = touch_book_last_used(&db, book_key, now_secs)
        {
            tracing::debug!(%error, "thumb pack touch failed");
        }
        Some(Self {
            db,
            book_key,
            present,
            pending: BTreeMap::new(),
            pruned_once: false,
        })
    }

    pub fn has(&self, page: usize) -> bool {
        self.pending.contains_key(&page)
            || self.present.get(page).copied().unwrap_or(false)
    }

    pub fn load(&self, page: usize) -> Option<ThumbImage> {
        if let Some(pending) = self.pending.get(&page) {
            return decode_thumb_image(
                pending.page_dims,
                pending.format,
                &pending.webp,
            );
        }
        if !self.present.get(page).copied().unwrap_or(false) {
            return None;
        }
        let page_u32 = u32::try_from(page).ok()?;
        let read_txn = self.db.begin_read().ok()?;
        let table = read_txn.open_table(THUMBS_TABLE).ok()?;
        let entry = table.get((self.book_key, page_u32)).ok()??;
        let (page_width, page_height, format_tag, webp) = entry.value();
        let format = Format::from_repr(format_tag)?;
        let page_dims = (page_width, page_height);
        decode::admit_dims(page_dims.0, page_dims.1, "thumb page dimensions")
            .ok()?;
        decode_thumb_image(page_dims, format, webp)
    }

    pub fn record(&mut self, page: usize, thumb: ValidatedThumb) {
        if self.has(page) || page >= self.present.len() {
            return;
        }
        self.pending.insert(page, thumb);
        if self.pending.len() >= FLUSH_ENTRY_THRESHOLD {
            self.flush();
        }
    }

    pub fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        if let Err(error) =
            self.flush_pending(now_unix_secs(), THUMB_PACK_CAP_BYTES)
        {
            tracing::warn!(%error, "thumb pack flush failed");
        }
    }

    fn flush_pending(
        &mut self,
        now_secs: u64,
        cap_bytes: u64,
    ) -> Result<(), redb::Error> {
        let items = mem::take(&mut self.pending);
        let write_txn = self.db.begin_write()?;
        let mut flushed_pages = Vec::with_capacity(items.len());
        {
            let mut books_table = write_txn.open_table(BOOKS_TABLE)?;
            let mut thumbs_table = write_txn.open_table(THUMBS_TABLE)?;

            let mut book_bytes = books_table
                .get(self.book_key)?
                .map_or(0, |existing| existing.value().1);

            for (page, thumb) in items {
                let Ok(page_u32) = u32::try_from(page) else {
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
                    .map_or(0, |previous| previous.value().3.len() as u64);
                book_bytes = book_bytes
                    .saturating_add(new_bytes)
                    .saturating_sub(previous_bytes);
                flushed_pages.push(page);
            }

            books_table.insert(self.book_key, (now_secs, book_bytes))?;

            if !self.pruned_once {
                prune_old_books(
                    &mut books_table,
                    &mut thumbs_table,
                    self.book_key,
                    cap_bytes,
                )?;
            }
        }
        write_txn.commit()?;
        self.pruned_once = true;

        for page in flushed_pages {
            if let Some(slot) = self.present.get_mut(page) {
                *slot = true;
            }
        }
        Ok(())
    }
}

impl Drop for ThumbPack {
    fn drop(&mut self) {
        self.flush();
    }
}

fn decode_thumb_image(
    page_dims: (u32, u32),
    format: Format,
    webp: &[u8],
) -> Option<ThumbImage> {
    let (dims, rgba) = decode::decode_webp_rgba(webp, THUMB_EDGE)
        .inspect_err(|error| {
            tracing::debug!(%error, "thumb webp decode failed");
        })
        .ok()?;
    Some(ThumbImage {
        dims,
        page_dims,
        format,
        rgba,
    })
}

fn open_or_recreate(
    path: &Path,
    book_key: [u8; 16],
    page_count: usize,
) -> Option<(Database, Vec<bool>, Option<u64>)> {
    let file = open_regular_file(path)?;
    let error = match open_and_scan(file, book_key, page_count) {
        Ok(opened) => return Some(opened),
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
    open_and_scan(fresh_file, book_key, page_count)
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
        .mode(THUMB_DB_FILE_MODE)
        .open(path)
        .inspect_err(|error| {
            tracing::warn!(path = %path.display(), %error, "thumb pack disabled");
        })
        .ok()
}

fn open_and_scan(
    file: File,
    book_key: [u8; 16],
    page_count: usize,
) -> Result<(Database, Vec<bool>, Option<u64>), redb::Error> {
    let db = Builder::new()
        .set_cache_size(THUMB_DB_CACHE_BYTES)
        .set_concurrency_mode(ConcurrencyMode::MultiWriter)
        .create_with_backend(FileBackend::new(file))?;
    let mut present = vec![false; page_count];
    let last_used = {
        let read_txn = db.begin_read()?;
        let books = open_if_exists(&read_txn, BOOKS_TABLE)?;
        let thumbs = open_if_exists(&read_txn, THUMBS_TABLE)?;
        if let (Some(books), Some(thumbs)) = (books, thumbs)
            && let Some(row) = books.get(book_key)?
        {
            for entry in thumbs.range(book_thumb_range(book_key))? {
                let (key, _) = entry?;
                if let Some(slot) = present.get_mut(key.value().1 as usize) {
                    *slot = true;
                }
            }
            Some(row.value().0)
        } else {
            None
        }
    };
    Ok((db, present, last_used))
}

fn open_if_exists<Key, Value>(
    read_txn: &ReadTransaction,
    definition: TableDefinition<'_, Key, Value>,
) -> Result<Option<ReadOnlyTable<Key, Value>>, redb::Error>
where
    Key: redb::Key + 'static,
    Value: redb::Value + 'static,
{
    match read_txn.open_table(definition) {
        Ok(table) => Ok(Some(table)),
        Err(TableError::TableDoesNotExist(_)) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

const fn book_thumb_range(book_key: [u8; 16]) -> RangeInclusive<([u8; 16], u32)> {
    (book_key, 0)..=(book_key, u32::MAX)
}

fn touch_book_last_used(
    db: &Database,
    book_key: [u8; 16],
    now_secs: u64,
) -> Result<(), redb::Error> {
    let write_txn = db.begin_write()?;
    let mut updated = false;
    {
        let mut books_table = write_txn.open_table(BOOKS_TABLE)?;
        let existing = books_table.get(book_key)?.map(|guard| guard.value());
        if let Some((last_used, total_bytes)) = existing
            && last_used != now_secs
        {
            books_table.insert(book_key, (now_secs, total_bytes))?;
            updated = true;
        }
    }
    if updated {
        write_txn.commit()?;
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
        .map_or(cap_bytes, |step| {
            step.saturating_mul(THUMB_PACK_PRUNE_NUMERATOR)
        });
    let mut removed_count = 0_u64;
    for (_, victim_key, victim_bytes) in entries {
        if total_bytes <= target_bytes {
            break;
        }
        if victim_key == keep_key {
            continue;
        }
        thumbs_table.retain_in(book_thumb_range(victim_key), |_, _| false)?;
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
fn book_key(path: &Path, archive: bool, names: &[String]) -> Option<[u8; 16]> {
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
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::decode::Depth;
    use crate::decode::Matrix;
    use crate::decode::WideDepth;
    use crate::testkit;
    use crate::testkit::TEST_FIT_BOX;

    fn pixel_count(width: u32, height: u32) -> usize {
        (width as usize) * (height as usize)
    }

    fn rgba_page(width: u32, height: u32) -> Decoded {
        let bytes =
            (0..pixel_count(width, height) * decode::RGBA_BYTES_PER_PIXEL)
                .map(|index| u8::try_from(index % 251).expect("channel"))
                .collect::<Vec<_>>()
                .into_boxed_slice();
        Decoded::rgba(width, height, Format::Png, Transfer::Srgb, bytes)
            .expect("rgba page")
            .with_page_dims((width, height))
    }

    fn rgba_bytes(page: &Decoded) -> Vec<u8> {
        let Pixels::Rgba8(bytes) = page.pixels() else {
            panic!("expected rgba pixels");
        };
        bytes.to_vec()
    }

    fn flat_yuv_page(
        width: u32,
        height: u32,
        subsampling: Subsampling,
        values: [u16; 4],
    ) -> Decoded {
        let ((_, _), (chroma_width, chroma_height)) =
            decode::plane_dims((width, height), subsampling);
        let plane = |samples: usize, value: u16| {
            vec![value; samples].into_boxed_slice()
        };
        let chroma_samples = pixel_count(chroma_width, chroma_height);
        let yuv = YuvImage {
            subsampling,
            full_range: false,
            matrix: Matrix::Bt709,
            samples: Samples::U16 {
                depth: WideDepth::D10,
                planes: YuvPlanes {
                    luma: plane(pixel_count(width, height), values[0]),
                    cb: plane(chroma_samples, values[1]),
                    cr: plane(chroma_samples, values[2]),
                    alpha: Some(plane(pixel_count(width, height), values[3])),
                },
            },
        };
        Decoded::yuv(width, height, Format::Avif, Transfer::Pq, yuv)
            .expect("yuv page")
            .with_page_dims((width, height))
    }

    fn yuv_planes(page: &Decoded) -> [Vec<u16>; 4] {
        let Pixels::Yuv(yuv) = page.pixels() else {
            panic!("expected yuv pixels");
        };
        match &yuv.samples {
            Samples::U8(planes) => {
                let widen = |bytes: &[u8]| {
                    bytes.iter().map(|&byte| u16::from(byte)).collect()
                };
                [
                    widen(&planes.luma),
                    widen(&planes.cb),
                    widen(&planes.cr),
                    planes.alpha.as_deref().map_or_else(Vec::new, widen),
                ]
            }
            Samples::U16 { planes, .. } => [
                planes.luma.to_vec(),
                planes.cb.to_vec(),
                planes.cr.to_vec(),
                planes.alpha.as_deref().map_or_else(Vec::new, <[u16]>::to_vec),
            ],
        }
    }

    #[test]
    fn derive_scales_to_the_box() {
        let page = rgba_page(500, 200).with_page_dims((1500, 600));
        let thumb = derive_to_box(&page, (200, 200), FilterType::Box, 1)
            .expect("thumb");
        assert_eq!((thumb.width(), thumb.height()), (200, 80));
        assert_eq!(thumb.page_dims(), (1500, 600));
        assert_eq!(thumb.format(), Format::Png);
        assert_eq!(thumb.transfer(), Transfer::Srgb);

        let exact = rgba_page(200, 100);
        let kept = derive_to_box(&exact, (200, 200), FilterType::Box, 1)
            .expect("exact kept");
        assert_eq!((kept.width(), kept.height()), (200, 100));
        assert_eq!(rgba_bytes(&kept), rgba_bytes(&exact));

        let wide = rgba_page(1000, 200).with_page_dims((1000, 200));
        let clamped = derive(
            &wide,
            WorkRole::Full,
            ImageLimits::new((300, 300), 500),
            2,
        )
        .expect("clamped full");
        assert_eq!((clamped.width(), clamped.height()), (500, 100));
        assert_eq!(clamped.page_dims(), (1000, 200));

        assert_eq!(fit_dims((3628, 5148), TEST_FIT_BOX), (1015, 1440));
        assert_eq!(fit_dims((4000, 100), TEST_FIT_BOX), (2560, 64));
    }

    #[test]
    fn derive_keeps_yuv_plane_values() {
        let page = flat_yuv_page(
            4000,
            100,
            Subsampling::Yuv420,
            [400, 600, 300, 1023],
        );
        let limits = testkit::limits();
        let derived =
            derive(&page, WorkRole::Fit, limits, 4).expect("derives");
        assert_eq!((derived.width(), derived.height()), (TEST_FIT_BOX.0, 64));
        assert_eq!(derived.page_dims(), (4000, 100));
        let Pixels::Yuv(yuv) = derived.pixels() else {
            panic!("expected yuv pixels");
        };
        assert_eq!(yuv.depth(), Depth::D10);
        assert_eq!(yuv.matrix, Matrix::Bt709);
        let [luma, cb, cr, alpha] = yuv_planes(&derived);
        for (name, plane, value) in [
            ("luma", luma, 400_u16),
            ("cb", cb, 600),
            ("cr", cr, 300),
            ("alpha", alpha, 1023),
        ] {
            assert!(
                plane.iter().all(|sample| sample.abs_diff(value) <= 1),
                "{name}"
            );
        }
    }

    #[test]
    fn derive_thumb_converts_planar_pages_to_8bit_srgb() {
        let assert_rgba_close = |bytes: &[u8], expected: [u8; 4], tol: u8| {
            for pixel in bytes.as_chunks::<4>().0 {
                for (&channel, target) in pixel.iter().zip(expected) {
                    assert!(
                        channel.abs_diff(target) <= tol,
                        "expected ~{expected:?}, got {pixel:?}"
                    );
                }
            }
        };

        let pq_black =
            flat_yuv_page(4, 2, Subsampling::Yuv420, [64, 512, 512, 1023]);
        assert_rgba_close(&thumb_rgba(&pq_black), [0, 0, 0, 255], 1);

        let pq_white =
            flat_yuv_page(4, 2, Subsampling::Yuv420, [940, 512, 512, 512]);
        assert_rgba_close(&thumb_rgba(&pq_white), [255, 255, 255, 128], 1);

        let gray10 = Decoded::gray(
            400,
            200,
            Format::Avif,
            Transfer::Srgb,
            GrayImage {
                full_range: true,
                samples: Samples::U16 {
                    depth: WideDepth::D10,
                    planes: GrayPlanes {
                        luma: vec![1023_u16; pixel_count(400, 200)]
                            .into_boxed_slice(),
                        alpha: Some(
                            vec![512_u16; pixel_count(400, 200)]
                                .into_boxed_slice(),
                        ),
                    },
                },
            },
        )
        .expect("gray10");
        let wire_thumb = derive_thumb(&gray10, 1).expect("gray10 webp thumb");
        let decoded = decode_thumb_image(
            wire_thumb.page_dims,
            wire_thumb.format,
            &wire_thumb.webp,
        )
        .expect("decoded gray10 webp");
        assert_eq!(decoded.dims(), (THUMB_EDGE, 96));
        assert_rgba_close(decoded.rgba(), [255, 255, 255, 128], 4);
    }

    #[test]
    fn thumb_pack_roundtrips_recovers_corrupt_db_shares_multiwriter_and_prunes() {
        let dir = TempDir::new().expect("temp dir");
        let root = dir.path().join("cache");
        let thumb_a = derive_thumb(&rgba_page(16, 12), 1).expect("thumb a");
        let thumb_b = derive_thumb(&rgba_page(24, 10), 1).expect("thumb b");
        let expected_a =
            decode_thumb_image(thumb_a.page_dims, thumb_a.format, &thumb_a.webp)
                .expect("decoded a");
        let expected_b =
            decode_thumb_image(thumb_b.page_dims, thumb_b.format, &thumb_b.webp)
                .expect("decoded b");

        // Order keys opposite to LRU eviction so key sort cannot mask timestamp ordering.
        let oldest_key = [0x33; 16];
        let mut oldest_pack =
            ThumbPack::open_at(&root, oldest_key, 4).expect("oldest pack");
        oldest_pack.record(0, thumb_a);
        oldest_pack.record(2, thumb_b.clone());
        // Duplicate record is ignored; pending load hits immediately.
        oldest_pack.record(0, thumb_b.clone());
        assert!(oldest_pack.has(0));
        assert_eq!(oldest_pack.load(0), Some(expected_a.clone()));
        oldest_pack
            .flush_pending(2, THUMB_PACK_CAP_BYTES)
            .expect("flush oldest");
        assert_eq!(oldest_pack.load(0), Some(expected_a));
        assert!(oldest_pack.load(1).is_none());

        // Non-directory root is rejected cleanly.
        let db_path = root.join(THUMB_DB_FILENAME);
        assert!(ThumbPack::open_at(&db_path, [1; 16], 2).is_none());

        // Concurrent MultiWriter handle on the same database works.
        let middle_key = [0x22; 16];
        let mut middle_pack =
            ThumbPack::open_at(&root, middle_key, 4).expect("middle pack");
        middle_pack.record(1, thumb_b.clone());
        middle_pack
            .flush_pending(1, THUMB_PACK_CAP_BYTES)
            .expect("flush middle");
        drop(middle_pack);
        drop(oldest_pack);

        // Reopening persists entries, isolates keys, and touches `last_used` past `oldest_key`.
        let reopened =
            ThumbPack::open_at(&root, middle_key, 4).expect("reopened pack");
        assert!(reopened.has(1));
        assert!(!reopened.has(0));
        assert_eq!(reopened.load(1), Some(expected_b));
        drop(reopened);

        // Book-level LRU pruning: cap fits ~2.5 entries, so flushing a third
        // book evicts `oldest_key` (last_used = 2) while keeping `middle_key`
        // (bumped by reopen) and `newest_key`.
        let thumb_b_bytes = thumb_b.webp.len() as u64;
        let newest_key = [0x11; 16];
        let mut newest_pack =
            ThumbPack::open_at(&root, newest_key, 4).expect("newest pack");
        newest_pack.record(0, thumb_b.clone());
        newest_pack
            .flush_pending(now_unix_secs(), 3 * thumb_b_bytes)
            .expect("flush newest");
        assert!(newest_pack.has(0));
        drop(newest_pack);

        assert!(
            !ThumbPack::open_at(&root, oldest_key, 4)
                .expect("check oldest")
                .has(0),
            "oldest book should be pruned"
        );
        assert!(
            ThumbPack::open_at(&root, middle_key, 4)
                .expect("check middle")
                .has(1),
            "touched middle book should survive"
        );

        // Corrupt database file recovers cleanly by recreating thumbs-v1.redb.
        fs::write(&db_path, b"corrupt-db-header").expect("corrupt db");
        let mut recovered =
            ThumbPack::open_at(&root, oldest_key, 4).expect("recovered pack");
        assert!(!recovered.has(0));
        recovered.record(0, thumb_b.clone());
        recovered.flush();
        assert!(recovered.load(0).is_some());
        drop(recovered);

        // Table-type mismatch recreates thumbs-v1.redb.
        fs::remove_file(&db_path).expect("remove db");
        {
            const LEGACY_BOOKS_TABLE: TableDefinition<'_, [u8; 16], &[u8]> =
                TableDefinition::new("books");
            const LEGACY_THUMBS_TABLE: TableDefinition<'_, ([u8; 16], u32), &[u8]> =
                TableDefinition::new("thumbs");

            let legacy = Database::create(&db_path).expect("legacy db");
            let write_txn = legacy.begin_write().expect("legacy write");
            {
                write_txn
                    .open_table(LEGACY_BOOKS_TABLE)
                    .expect("legacy books");
                write_txn
                    .open_table(LEGACY_THUMBS_TABLE)
                    .expect("legacy thumbs");
            }
            write_txn.commit().expect("legacy commit");
        }
        let mut migrated =
            ThumbPack::open_at(&root, oldest_key, 4).expect("migrated pack");
        migrated.record(0, thumb_b);
        migrated.flush();
        assert!(migrated.load(0).is_some());
    }
}
