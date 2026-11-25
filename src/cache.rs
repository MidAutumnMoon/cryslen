//! Downscaled page derivation and the per-book parent thumbnail pack
//! (`$XDG_CACHE_HOME/cryslen/thumbs/<book_key>.cyth`).

#![expect(
    clippy::little_endian_bytes,
    reason = "the thumb-pack binary format is defined little-endian"
)]

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fs;
use std::fs::File;
use std::io;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::ErrorKind;
use std::io::Read as _;
use std::io::Seek as _;
use std::io::SeekFrom;
use std::io::Write as _;
use std::mem;
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
use crate::protocol::ThumbFormat;
use crate::protocol::ValidatedThumb;
use crate::protocol::WireThumb;

const THUMB_PACK_CAP_BYTES: u64 = 512 * 1024 * 1024;
const THUMB_PACK_PRUNE_NUMERATOR: u64 = 4;
const THUMB_PACK_PRUNE_DENOMINATOR: u64 = 5;
const FLUSH_ENTRY_THRESHOLD: usize = 32;
const FLUSH_BYTE_THRESHOLD: usize = 2 * 1024 * 1024;

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

/// Derive a compact 8-bit sRGB thumbnail (`<= THUMB_EDGE` per axis) from a
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
    encode_thumb(&thumb)
}

fn encode_thumb(page: &Decoded) -> eros::Result<ValidatedThumb> {
    const RGB_CHANNELS: usize = 3;

    let width = page.width();
    let height = page.height();
    let transfer = page.transfer();
    let (pixels, bytes) = match page.pixels() {
        Pixels::Rgba8(raw) => {
            let mut rgba = raw.to_vec();
            if transfer == Transfer::Pq {
                for pixel in
                    rgba.as_chunks_mut::<{ decode::RGBA_BYTES_PER_PIXEL }>().0
                {
                    for channel in pixel.iter_mut().take(RGB_CHANNELS) {
                        *channel = quantize_unit(decode::pq_to_srgb_channel(
                            f32::from(*channel) / decode::R8_STORAGE_MAX,
                        ));
                    }
                }
            }
            (ThumbFormat::Rgba8, rgba)
        }
        Pixels::Gray(gray) => encode_gray_thumb(
            gray,
            (width, height),
            transfer,
        ),
        Pixels::Yuv(yuv) => encode_yuv_thumb(
            yuv,
            (width, height),
            transfer,
        ),
    };
    let wire = WireThumb {
        width,
        height,
        page_dims: page.page_dims(),
        format: page.format(),
        pixels,
        bytes,
    };
    Ok(wire.validate()?)
}

fn encode_gray_thumb(
    gray: &GrayImage,
    (width, height): (u32, u32),
    transfer: Transfer,
) -> (ThumbFormat, Vec<u8>) {
    let pixel_count = width as usize * height as usize;

    match &gray.samples {
        Samples::U8(planes) => {
            gray_planes_to_thumb(gray, pixel_count, transfer, planes)
        }
        Samples::U16 { planes, .. } => {
            gray_planes_to_thumb(gray, pixel_count, transfer, planes)
        }
    }
}

fn gray_planes_to_thumb<Sample>(
    gray: &GrayImage,
    pixel_count: usize,
    transfer: Transfer,
    planes: &GrayPlanes<Sample>,
) -> (ThumbFormat, Vec<u8>)
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

    planes.alpha.as_deref().map_or_else(
        || {
            let bytes = planes
                .luma
                .iter()
                .map(|sample| map_luma(f32::from(*sample)))
                .collect();
            (ThumbFormat::Gray8, bytes)
        },
        |alpha_plane| {
            let mut bytes = Vec::with_capacity(
                pixel_count * decode::RGBA_BYTES_PER_PIXEL,
            );
            for (luma_sample, alpha_sample) in
                planes.luma.iter().zip(alpha_plane.iter())
            {
                let gray_byte = map_luma(f32::from(*luma_sample));
                let alpha_byte = map_alpha(f32::from(*alpha_sample));
                bytes.extend_from_slice(&[
                    gray_byte, gray_byte, gray_byte, alpha_byte,
                ]);
            }
            (ThumbFormat::Rgba8, bytes)
        },
    )
}

fn encode_yuv_thumb(
    yuv: &YuvImage,
    dims: (u32, u32),
    transfer: Transfer,
) -> (ThumbFormat, Vec<u8>) {
    match &yuv.samples {
        Samples::U8(planes) => (
            ThumbFormat::Rgba8,
            yuv_planes_to_rgba8(yuv, dims, transfer, planes),
        ),
        Samples::U16 { planes, .. } => (
            ThumbFormat::Rgba8,
            yuv_planes_to_rgba8(yuv, dims, transfer, planes),
        ),
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

const PACK_MAGIC: [u8; 4] = *b"CYTH";
const PACK_VERSION: u16 = 1;
const PACK_HEADER_BYTES: usize = 16;
const RECORD_HEADER_BYTES: usize = 20;
const PACK_EXTENSION: &str = ".cyth";

#[derive(Clone, Copy)]
struct RecordLocation {
    payload_offset: u64,
    width: u32,
    height: u32,
    page_dims: (u32, u32),
    format: Format,
    pixels: ThumbFormat,
}

struct PackFile {
    file: File,
    valid_end: u64,
}

/// Parent-owned append-only per-book thumbnail pack under
/// `$XDG_CACHE_HOME/cryslen/thumbs/<book_key>.cyth`.
pub struct ThumbPack {
    path: PathBuf,
    cap_bytes: u64,
    file: Option<PackFile>,
    index: Vec<Option<RecordLocation>>,
    pending: BTreeMap<usize, ValidatedThumb>,
    pending_bytes: usize,
    pruned_once: bool,
}

impl ThumbPack {
    pub fn open(path: &Path, archive: bool, names: &[String]) -> Option<Self> {
        let root = default_thumb_dir()?;
        let key = book_key(path, archive, names)?;
        Self::open_at(&root, key, names.len())
    }

    pub fn open_at(root: &Path, key: u128, page_count: usize) -> Option<Self> {
        Self::open_with_cap(root, key, page_count, THUMB_PACK_CAP_BYTES)
    }

    fn open_with_cap(
        root: &Path,
        key: u128,
        page_count: usize,
        cap_bytes: u64,
    ) -> Option<Self> {
        if make_dir(root).is_none() {
            tracing::warn!(root = %root.display(), "thumb pack disabled");
            return None;
        }
        let path = root.join(format!("{key:032x}{PACK_EXTENSION}"));
        let mut index = vec![None; page_count];
        let file = open_existing_pack(&path, &mut index);
        Some(Self {
            path,
            cap_bytes,
            file,
            index,
            pending: BTreeMap::new(),
            pending_bytes: 0,
            pruned_once: false,
        })
    }

    pub fn has(&self, page: usize) -> bool {
        self.pending.contains_key(&page)
            || self.index.get(page).is_some_and(Option::is_some)
    }

    pub fn load(&self, page: usize) -> Option<ValidatedThumb> {
        use std::os::unix::fs::FileExt as _;

        if let Some(pending) = self.pending.get(&page) {
            return Some(pending.clone());
        }
        let location = self.index.get(page).copied().flatten()?;
        let pack = self.file.as_ref()?;
        let payload_bytes =
            location.pixels.byte_len(location.width, location.height)?;
        let mut bytes = vec![0_u8; payload_bytes];
        pack.file
            .read_exact_at(&mut bytes, location.payload_offset)
            .ok()?;
        WireThumb {
            width: location.width,
            height: location.height,
            page_dims: location.page_dims,
            format: location.format,
            pixels: location.pixels,
            bytes,
        }
        .validate()
        .ok()
    }

    pub fn record(&mut self, page: usize, thumb: ValidatedThumb) {
        if self.has(page) || page >= self.index.len() {
            return;
        }
        self.pending_bytes =
            self.pending_bytes.saturating_add(thumb.bytes.len());
        self.pending.insert(page, thumb);
        if self.pending.len() >= FLUSH_ENTRY_THRESHOLD
            || self.pending_bytes >= FLUSH_BYTE_THRESHOLD
        {
            self.flush();
        }
    }

    pub fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        if let Err(error) = self.flush_pending() {
            tracing::debug!(%error, path = %self.path.display(), "thumb pack flush failed");
            return;
        }
        if !self.pruned_once {
            self.pruned_once = true;
            if let (Some(dir), Some(file_name)) =
                (self.path.parent(), self.path.file_name())
            {
                prune_packs(dir, self.cap_bytes, file_name);
            }
        }
    }

    fn flush_pending(&mut self) -> io::Result<()> {
        use std::fs::OpenOptions;
        use std::os::unix::fs::OpenOptionsExt as _;

        const PACK_FILE_MODE: u32 = 0o600;

        if let Ok(metadata) = fs::symlink_metadata(&self.path)
            && metadata.file_type().is_symlink()
        {
            return Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "thumb pack path is a symlink",
            ));
        }

        let (target_file, mut offset) = match &self.file {
            None => {
                let mut file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .mode(PACK_FILE_MODE)
                    .open(&self.path)?;
                file.write_all(&encode_pack_header(self.index.len()))?;
                (file, PACK_HEADER_BYTES as u64)
            }
            Some(pack) => {
                let mut file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&self.path)?;
                file.set_len(pack.valid_end)?;
                file.seek(SeekFrom::Start(pack.valid_end))?;
                (file, pack.valid_end)
            }
        };

        let items = mem::take(&mut self.pending);
        self.pending_bytes = 0;

        let mut new_locations = Vec::with_capacity(items.len());
        let mut writer = BufWriter::new(target_file);
        for (page, thumb) in items {
            let Some(record_header) = encode_record_header(page, &thumb)
            else {
                continue;
            };
            writer.write_all(&record_header)?;
            writer.write_all(&thumb.bytes)?;
            let payload_offset = offset + RECORD_HEADER_BYTES as u64;
            let payload_bytes = thumb.bytes.len() as u64;
            new_locations.push((
                page,
                RecordLocation {
                    payload_offset,
                    width: thumb.width,
                    height: thumb.height,
                    page_dims: thumb.page_dims,
                    format: thumb.format,
                    pixels: thumb.pixels,
                },
            ));
            offset += RECORD_HEADER_BYTES as u64 + payload_bytes;
        }
        writer.flush()?;
        let file = writer.into_inner().map_err(io::IntoInnerError::into_error)?;
        for (page, location) in new_locations {
            if let Some(slot) = self.index.get_mut(page) {
                *slot = Some(location);
            }
        }
        self.file = Some(PackFile {
            file,
            valid_end: offset,
        });
        Ok(())
    }
}

impl Drop for ThumbPack {
    fn drop(&mut self) {
        self.flush();
    }
}

fn open_existing_pack(
    path: &Path,
    index: &mut [Option<RecordLocation>],
) -> Option<PackFile> {
    let metadata = fs::symlink_metadata(path).ok()?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return None;
    }
    let file = File::open(path).ok()?;
    if let Err(error) = file.set_modified(SystemTime::now()) {
        tracing::debug!(%error, path = %path.display(), "thumb pack mtime touch failed");
    }
    let file_bytes = file.metadata().map_or(0, |meta| meta.len());
    if file_bytes < PACK_HEADER_BYTES as u64 {
        return None;
    }

    let page_count = index.len();
    let mut reader = BufReader::new(&file);
    let mut header = [0_u8; PACK_HEADER_BYTES];
    reader.read_exact(&mut header).ok()?;
    if decode_pack_header(&header) != Some(page_count) {
        return None;
    }

    let mut offset = PACK_HEADER_BYTES as u64;
    let mut record = [0_u8; RECORD_HEADER_BYTES];
    loop {
        if reader.read_exact(&mut record).is_err() {
            break;
        }
        let Some((page, location, payload_bytes)) =
            decode_record_header(&record, page_count, offset)
        else {
            break;
        };
        let Some(next_offset) =
            offset.checked_add(RECORD_HEADER_BYTES as u64 + payload_bytes)
        else {
            break;
        };
        if next_offset > file_bytes {
            break;
        }
        let Ok(payload_skip) = i64::try_from(payload_bytes) else {
            break;
        };
        if reader.seek_relative(payload_skip).is_err() {
            break;
        }
        if let Some(slot) = index.get_mut(page) {
            *slot = Some(location);
        }
        offset = next_offset;
    }
    drop(reader);
    Some(PackFile {
        file,
        valid_end: offset,
    })
}

fn encode_pack_header(page_count: usize) -> [u8; PACK_HEADER_BYTES] {
    let mut header = [0_u8; PACK_HEADER_BYTES];
    header[..4].copy_from_slice(&PACK_MAGIC);
    header[4..6].copy_from_slice(&PACK_VERSION.to_le_bytes());
    header[8..16].copy_from_slice(&(page_count as u64).to_le_bytes());
    header
}

fn decode_pack_header(header: &[u8; PACK_HEADER_BYTES]) -> Option<usize> {
    if header[..4] != PACK_MAGIC {
        return None;
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    if version != PACK_VERSION {
        return None;
    }
    let page_count = u64::from_le_bytes([
        header[8], header[9], header[10], header[11], header[12], header[13],
        header[14], header[15],
    ]);
    usize::try_from(page_count).ok()
}

fn encode_record_header(
    page: usize,
    thumb: &ValidatedThumb,
) -> Option<[u8; RECORD_HEADER_BYTES]> {
    let page = u32::try_from(page).ok()?;
    let width = u16::try_from(thumb.width).ok()?;
    let height = u16::try_from(thumb.height).ok()?;
    let mut record = [0_u8; RECORD_HEADER_BYTES];
    record[0..4].copy_from_slice(&page.to_le_bytes());
    record[4..6].copy_from_slice(&width.to_le_bytes());
    record[6..8].copy_from_slice(&height.to_le_bytes());
    record[8..12].copy_from_slice(&thumb.page_dims.0.to_le_bytes());
    record[12..16].copy_from_slice(&thumb.page_dims.1.to_le_bytes());
    record[16] = format_to_code(thumb.format);
    record[17] = pixel_to_code(thumb.pixels);
    Some(record)
}

fn decode_record_header(
    record: &[u8; RECORD_HEADER_BYTES],
    page_count: usize,
    record_offset: u64,
) -> Option<(usize, RecordLocation, u64)> {
    let page = u32::from_le_bytes([record[0], record[1], record[2], record[3]])
        as usize;
    if page >= page_count {
        return None;
    }
    let width = u32::from(u16::from_le_bytes([record[4], record[5]]));
    let height = u32::from(u16::from_le_bytes([record[6], record[7]]));
    if width == 0 || height == 0 || width > THUMB_EDGE || height > THUMB_EDGE {
        return None;
    }
    let page_width =
        u32::from_le_bytes([record[8], record[9], record[10], record[11]]);
    let page_height =
        u32::from_le_bytes([record[12], record[13], record[14], record[15]]);
    decode::pixel_count(page_width, page_height)?;
    let format = code_to_format(record[16])?;
    let pixels = code_to_pixel(record[17])?;
    let payload_bytes = pixels.byte_len(width, height)? as u64;
    let payload_offset =
        record_offset.checked_add(RECORD_HEADER_BYTES as u64)?;
    Some((
        page,
        RecordLocation {
            payload_offset,
            width,
            height,
            page_dims: (page_width, page_height),
            format,
            pixels,
        },
        payload_bytes,
    ))
}

const fn format_to_code(format: Format) -> u8 {
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

const fn code_to_format(code: u8) -> Option<Format> {
    match code {
        0 => Some(Format::Jpeg),
        1 => Some(Format::Png),
        2 => Some(Format::Gif),
        3 => Some(Format::Webp),
        4 => Some(Format::Bmp),
        5 => Some(Format::Tiff),
        6 => Some(Format::Avif),
        7 => Some(Format::Jxl),
        _ => None,
    }
}

const fn pixel_to_code(pixels: ThumbFormat) -> u8 {
    match pixels {
        ThumbFormat::Gray8 => 0,
        ThumbFormat::Rgba8 => 1,
    }
}

const fn code_to_pixel(code: u8) -> Option<ThumbFormat> {
    match code {
        0 => Some(ThumbFormat::Gray8),
        1 => Some(ThumbFormat::Rgba8),
        _ => None,
    }
}

fn default_thumb_dir() -> Option<PathBuf> {
    use std::env;

    if let Some(xdg_cache_home) = env::var_os("XDG_CACHE_HOME")
        && !xdg_cache_home.is_empty()
    {
        let path = PathBuf::from(xdg_cache_home);
        if path.is_absolute() {
            return Some(path.join("cryslen").join("thumbs"));
        }
    }
    let home = env::var_os("HOME")?;
    if home.is_empty() {
        return None;
    }
    Some(PathBuf::from(home).join(".cache").join("cryslen").join("thumbs"))
}

fn book_key(path: &Path, archive: bool, names: &[String]) -> Option<u128> {
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
    Some(hasher.digest128())
}

fn prune_packs(dir: &Path, cap_bytes: u64, keep_name: &OsStr) {
    struct PackEntry {
        modified: SystemTime,
        name: OsString,
        bytes: u64,
    }

    let Ok(dir_entries) = fs::read_dir(dir) else {
        return;
    };
    let mut entries = Vec::new();
    let mut total_bytes = 0_u64;
    for entry in dir_entries.flatten() {
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        let name = entry.file_name();
        if !name
            .to_str()
            .is_some_and(|text| text.ends_with(PACK_EXTENSION))
        {
            continue;
        }
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        let bytes = metadata.len();
        total_bytes = total_bytes.saturating_add(bytes);
        entries.push(PackEntry {
            modified,
            name,
            bytes,
        });
    }
    if total_bytes <= cap_bytes {
        return;
    }
    entries.sort_by(|left, right| {
        left.modified
            .cmp(&right.modified)
            .then_with(|| left.name.cmp(&right.name))
    });
    let target_bytes = cap_bytes
        .checked_div(THUMB_PACK_PRUNE_DENOMINATOR)
        .map_or(cap_bytes, |step| {
            step.saturating_mul(THUMB_PACK_PRUNE_NUMERATOR)
        });
    let mut removed_count = 0_u64;
    for entry in entries {
        if total_bytes <= target_bytes {
            break;
        }
        if entry.name == keep_name {
            continue;
        }
        if fs::remove_file(dir.join(&entry.name)).is_ok() {
            total_bytes = total_bytes.saturating_sub(entry.bytes);
            removed_count += 1;
        }
    }
    tracing::debug!(
        dir = %dir.display(),
        removed = removed_count,
        total_bytes,
        cap_bytes,
        "thumb pack prune"
    );
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

    fn set_modified(path: &Path, seconds: u64) {
        use std::fs::FileTimes;
        use std::time::Duration;

        let modified =
            SystemTime::UNIX_EPOCH + Duration::from_secs(seconds);
        fs::set_times(path, FileTimes::new().set_modified(modified))
            .expect("set mtime");
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
        let assert_rgba_close = |bytes: &[u8], expected: [u8; 4]| {
            for pixel in bytes.as_chunks::<4>().0 {
                for (&channel, target) in pixel.iter().zip(expected) {
                    assert!(
                        channel.abs_diff(target) <= 1,
                        "expected ~{expected:?}, got {pixel:?}"
                    );
                }
            }
        };

        let pq_black = flat_yuv_page(
            400,
            200,
            Subsampling::Yuv420,
            [64, 512, 512, 1023],
        );
        let black_thumb = derive_thumb(&pq_black, 2).expect("pq black");
        assert_eq!(black_thumb.pixels, ThumbFormat::Rgba8);
        assert_rgba_close(&black_thumb.bytes, [0, 0, 0, 255]);

        let pq_white = flat_yuv_page(
            400,
            200,
            Subsampling::Yuv420,
            [940, 512, 512, 512],
        );
        let white_thumb = derive_thumb(&pq_white, 2).expect("pq white");
        assert_eq!(white_thumb.pixels, ThumbFormat::Rgba8);
        assert_rgba_close(&white_thumb.bytes, [255, 255, 255, 128]);

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
        let gray10_thumb = derive_thumb(&gray10, 2).expect("gray10 thumb");
        assert_eq!(gray10_thumb.pixels, ThumbFormat::Rgba8);
        assert_rgba_close(&gray10_thumb.bytes, [255, 255, 255, 128]);

        let gray8 = Decoded::gray(
            400,
            200,
            Format::Png,
            Transfer::Srgb,
            GrayImage {
                full_range: true,
                samples: Samples::U8(GrayPlanes {
                    luma: vec![120_u8; pixel_count(400, 200)].into_boxed_slice(),
                    alpha: None,
                }),
            },
        )
        .expect("gray8");
        let gray8_thumb = derive_thumb(&gray8, 1).expect("gray8 thumb");
        assert_eq!(gray8_thumb.pixels, ThumbFormat::Gray8);
        assert!(gray8_thumb.bytes.iter().all(|&byte| byte == 120));
    }

    #[test]
    fn thumb_pack_roundtrips_recovers_torn_tail_and_prunes() {
        let dir = TempDir::new().expect("temp dir");
        let root = dir.path().join("thumbs");
        let gray8 = Decoded::gray(
            400,
            200,
            Format::Png,
            Transfer::Srgb,
            GrayImage {
                full_range: true,
                samples: Samples::U8(GrayPlanes {
                    luma: vec![120_u8; pixel_count(400, 200)].into_boxed_slice(),
                    alpha: None,
                }),
            },
        )
        .expect("gray8");
        let gray_thumb = derive_thumb(&gray8, 1).expect("gray thumb");
        let rgba_thumb =
            derive_thumb(&rgba_page(400, 200), 1).expect("rgba thumb");

        let oldest_key = 0x1111_u128;
        let mut oldest_pack =
            ThumbPack::open_at(&root, oldest_key, 4).expect("oldest pack");
        oldest_pack.record(0, gray_thumb.clone());
        oldest_pack.record(2, rgba_thumb.clone());
        // Duplicate record is ignored; pending load hits immediately.
        oldest_pack.record(0, rgba_thumb.clone());
        assert!(oldest_pack.has(0));
        assert_eq!(oldest_pack.load(0), Some(gray_thumb.clone()));
        oldest_pack.flush();
        assert_eq!(oldest_pack.load(0), Some(gray_thumb.clone()));
        assert_eq!(oldest_pack.load(2), Some(rgba_thumb.clone()));
        assert!(oldest_pack.load(1).is_none());
        drop(oldest_pack);

        // Append torn bytes at EOF: reopen recovers intact entries and
        // truncates the torn tail on the next flush.
        let oldest_pack_path =
            root.join(format!("{oldest_key:032x}{PACK_EXTENSION}"));
        assert!(ThumbPack::open_at(&oldest_pack_path, 1, 2).is_none());
        let mut pack_bytes = fs::read(&oldest_pack_path).expect("read pack");
        pack_bytes.extend_from_slice(&[0xff, 0xaa, 0x55, 0x00, 0x01]);
        fs::write(&oldest_pack_path, &pack_bytes).expect("write torn tail");

        let mut reopened =
            ThumbPack::open_at(&root, oldest_key, 4).expect("reopened pack");
        assert_eq!(reopened.load(0), Some(gray_thumb));
        assert_eq!(reopened.load(2), Some(rgba_thumb.clone()));
        reopened.record(3, rgba_thumb.clone());
        reopened.flush();
        assert_eq!(reopened.load(3), Some(rgba_thumb.clone()));
        drop(reopened);

        // Book-level LRU pruning: cap fits ~1.5 packs, so saving a third
        // book evicts the oldest book while preserving the active pack.
        let oldest_pack_bytes =
            fs::metadata(&oldest_pack_path).expect("pack len").len();
        let middle_key = 0x2222_u128;
        let newest_key = 0x3333_u128;
        let mut middle_pack = ThumbPack::open_with_cap(
            &root,
            middle_key,
            4,
            THUMB_PACK_CAP_BYTES,
        )
        .expect("middle pack");
        middle_pack.record(0, rgba_thumb.clone());
        middle_pack.flush();
        drop(middle_pack);

        set_modified(&oldest_pack_path, 10);
        set_modified(&root.join(format!("{middle_key:032x}{PACK_EXTENSION}")), 20);

        let mut newest_pack = ThumbPack::open_with_cap(
            &root,
            newest_key,
            4,
            oldest_pack_bytes + 10,
        )
        .expect("newest pack");
        newest_pack.record(0, rgba_thumb);
        newest_pack.flush();
        assert!(
            !oldest_pack_path.exists(),
            "oldest book pack should be pruned"
        );
        assert!(
            root.join(format!("{newest_key:032x}{PACK_EXTENSION}"))
                .exists()
        );
    }
}
