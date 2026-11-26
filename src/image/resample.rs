//! Downscaled page derivation and thumbnail RGBA/WebP generation.

use std::borrow::Cow;
use std::thread;

use bytemuck::Pod;
use fast_image_resize::FilterType;
use fast_image_resize::PixelTrait;
use fast_image_resize::Resizer;
use fast_image_resize::pixels::Pixel;
use fast_image_resize::pixels::PixelComponent;

use crate::image::color;
use crate::image::color::YuvMatrix;
use crate::image::decode::webp;
use crate::image::pixels;
use crate::image::pixels::Depth;
use crate::image::pixels::EncodedThumb;
use crate::image::pixels::Image;
use crate::image::pixels::ImageLimits;
use crate::image::pixels::Pixels;
use crate::image::pixels::RGBA_BYTES_PER_PIXEL;
use crate::image::pixels::SamplePlanes;
use crate::image::pixels::Subsampling;
use crate::image::pixels::THUMB_EDGE;
use crate::image::pixels::TargetRole;
use crate::image::pixels::TextureFault;
use crate::image::pixels::Transfer;

const THUMB_WEBP_QUALITY: f32 = 78.0;

fn derivation(
    role: TargetRole,
    limits: ImageLimits,
) -> ((u32, u32), FilterType) {
    match role {
        TargetRole::Fit => (limits.fit_box(), FilterType::Lanczos3),
        TargetRole::Full => {
            let max_edge = limits.max_texture_edge();
            ((max_edge, max_edge), FilterType::Lanczos3)
        }
    }
}

/// Downsample to the role's bounding box; borrows `page` unchanged when
/// it fits or resampling fails within `max_texture_edge`, and returns
/// `TextureFault` past `max_texture_edge`.
pub fn derive(
    page: &Image,
    role: TargetRole,
    limits: ImageLimits,
    max_threads: usize,
) -> Result<Cow<'_, Image>, TextureFault> {
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
    page: &Image,
    max_threads: usize,
) -> eros::Result<EncodedThumb> {
    let thumb = derive_to_box(
        page,
        (THUMB_EDGE, THUMB_EDGE),
        FilterType::Box,
        max_threads,
    )?;
    let rgba = thumb_rgba(&thumb);
    let webp = webp::encode_rgba(
        (thumb.width(), thumb.height()),
        &rgba,
        THUMB_WEBP_QUALITY,
    )?;
    let encoded = EncodedThumb {
        page_dims: thumb.page_dims(),
        format: thumb.format(),
        webp,
    };
    if !encoded.is_valid_size() {
        eros::bail!("thumb payload exceeds cap");
    }
    Ok(encoded)
}

fn thumb_rgba(page: &Image) -> Cow<'_, [u8]> {
    const RGB_CHANNELS: usize = 3;

    let dims = (page.width(), page.height());
    let transfer = page.transfer();
    match page.pixels() {
        Pixels::Rgba8(raw) => {
            if transfer == Transfer::Srgb {
                return Cow::Borrowed(raw);
            }
            let mut rgba = raw.to_vec();
            for pixel in rgba.as_chunks_mut::<RGBA_BYTES_PER_PIXEL>().0 {
                for channel in pixel.iter_mut().take(RGB_CHANNELS) {
                    *channel = quantize_unit(color::pq_to_srgb_channel(
                        f32::from(*channel) / color::R8_STORAGE_MAX,
                    ));
                }
            }
            Cow::Owned(rgba)
        }
        Pixels::Gray { full_range, planes } => {
            let depth = planes.depth();
            let (gain, offset) =
                color::gray_gain_offset(depth, *full_range);
            let conversion = YuvMatrix::gray(gain, offset);
            let rgba = match planes {
                SamplePlanes::U8 {
                    planes: [luma],
                    alpha,
                } => planar_to_rgba8(
                    dims,
                    depth,
                    transfer,
                    conversion,
                    Subsampling::Yuv444,
                    luma,
                    None,
                    alpha.as_deref(),
                ),
                SamplePlanes::U16 {
                    planes: [luma],
                    alpha,
                    ..
                } => planar_to_rgba8(
                    dims,
                    depth,
                    transfer,
                    conversion,
                    Subsampling::Yuv444,
                    luma,
                    None,
                    alpha.as_deref(),
                ),
            };
            Cow::Owned(rgba)
        }
        Pixels::Yuv { spec, planes } => {
            let depth = planes.depth();
            let conversion = color::yuv_color_conversion(
                depth,
                spec.full_range,
                spec.matrix,
            );
            let rgba = match planes {
                SamplePlanes::U8 {
                    planes: [luma, cb, cr],
                    alpha,
                } => planar_to_rgba8(
                    dims,
                    depth,
                    transfer,
                    conversion,
                    spec.subsampling,
                    luma,
                    Some((cb, cr)),
                    alpha.as_deref(),
                ),
                SamplePlanes::U16 {
                    planes: [luma, cb, cr],
                    alpha,
                    ..
                } => planar_to_rgba8(
                    dims,
                    depth,
                    transfer,
                    conversion,
                    spec.subsampling,
                    luma,
                    Some((cb, cr)),
                    alpha.as_deref(),
                ),
            };
            Cow::Owned(rgba)
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "self-contained planar-to-RGBA helper"
)]
fn planar_to_rgba8<Sample>(
    dims: (u32, u32),
    depth: Depth,
    transfer: Transfer,
    conversion: YuvMatrix,
    subsampling: Subsampling,
    luma: &[Sample],
    chroma: Option<(&[Sample], &[Sample])>,
    alpha: Option<&[Sample]>,
) -> Vec<u8>
where
    Sample: Copy + Into<f32>,
{
    let (width, height) = dims;
    let storage_max = color::max_storage_code(depth);
    let alpha_normalization = color::depth_normalization(depth);
    let (_, chroma_dims) =
        pixels::plane_dims((width, height), subsampling);
    let (image_width, image_height) = (width as usize, height as usize);
    let (chroma_width, chroma_height) =
        (chroma_dims.0 as usize, chroma_dims.1 as usize);
    let (shift_x, shift_y) = match subsampling {
        Subsampling::Yuv444 => (0_u32, 0_u32),
        Subsampling::Yuv422 => (1, 0),
        Subsampling::Yuv420 => (1, 1),
    };
    let mut rgba = Vec::with_capacity(
        image_width * image_height * RGBA_BYTES_PER_PIXEL,
    );
    for row in 0..image_height {
        let chroma_row =
            (row >> shift_y).min(chroma_height.saturating_sub(1));
        for column in 0..image_width {
            let chroma_column =
                (column >> shift_x).min(chroma_width.saturating_sub(1));
            let luma_index = row * image_width + column;
            let chroma_index = chroma_row * chroma_width + chroma_column;
            let luma_sample =
                luma.get(luma_index).copied().map_or(0.0, Into::into)
                    / storage_max;
            let (blue_chroma, red_chroma) =
                chroma.map_or((0.0, 0.0), |(cb, cr)| {
                    (
                        cb.get(chroma_index)
                            .copied()
                            .map_or(0.0, Into::into)
                            / storage_max,
                        cr.get(chroma_index)
                            .copied()
                            .map_or(0.0, Into::into)
                            / storage_max,
                    )
                });
            let rgb =
                conversion.apply([luma_sample, blue_chroma, red_chroma]);
            let alpha_byte = alpha.map_or(u8::MAX, |plane| {
                let sample =
                    plane.get(luma_index).copied().map_or(1.0, |raw| {
                        raw.into() / storage_max * alpha_normalization
                    });
                quantize_unit(sample.clamp(0.0, 1.0))
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
        Transfer::Pq => color::pq_to_srgb_channel(channel),
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "channel is clamped to 0.0..=1.0 before scaling to 0..=255"
)]
fn quantize_unit(channel: f32) -> u8 {
    (channel.clamp(0.0, 1.0) * color::R8_STORAGE_MAX).round() as u8
}

fn derive_to_box(
    page: &Image,
    bounds: (u32, u32),
    filter: FilterType,
    max_threads: usize,
) -> eros::Result<Cow<'_, Image>> {
    let page_dims = page.page_dims();
    let animated = page.animated();
    let dims = (page.width(), page.height());
    let target = fit_dims(dims, bounds);
    if target == dims {
        return Ok(Cow::Borrowed(page));
    }
    resample_page(page, target, filter, max_threads).map(|derived| {
        Cow::Owned(
            derived.with_page_dims(page_dims).with_animated(animated),
        )
    })
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
    page: &Image,
    target: (u32, u32),
    filter: FilterType,
    max_threads: usize,
) -> eros::Result<Image> {
    let dims = (page.width(), page.height());
    match page.pixels() {
        Pixels::Rgba8(bytes) => {
            let mut resizer = Resizer::new();
            let output = resize_typed::<_, u8, RGBA_BYTES_PER_PIXEL>(
                &mut resizer,
                bytes.as_chunks::<RGBA_BYTES_PER_PIXEL>().0,
                dims,
                target,
                filter,
            )?;
            Ok(Image::rgba(
                target.0,
                target.1,
                page.format(),
                page.transfer(),
                output.into_flattened().into_boxed_slice(),
            )?)
        }
        Pixels::Gray { full_range, planes } => {
            let target_planes = match planes {
                SamplePlanes::U8 {
                    planes: [luma],
                    alpha,
                } => {
                    let (target_planes, target_alpha) =
                        resize_homogeneous(
                            [(luma.as_ref(), dims, target)],
                            alpha
                                .as_deref()
                                .map(|plane| (plane, dims, target)),
                            filter,
                            max_threads,
                        )?;
                    SamplePlanes::U8 {
                        planes: target_planes,
                        alpha: target_alpha,
                    }
                }
                SamplePlanes::U16 {
                    depth,
                    planes: [luma],
                    alpha,
                } => {
                    let (target_planes, target_alpha) =
                        resize_homogeneous(
                            [(luma.as_ref(), dims, target)],
                            alpha
                                .as_deref()
                                .map(|plane| (plane, dims, target)),
                            filter,
                            max_threads,
                        )?;
                    SamplePlanes::U16 {
                        depth: *depth,
                        planes: target_planes,
                        alpha: target_alpha,
                    }
                }
            };
            Ok(Image::gray(
                target.0,
                target.1,
                page.format(),
                page.transfer(),
                *full_range,
                target_planes,
            )?)
        }
        Pixels::Yuv { spec, planes } => {
            let (luma_dims, chroma_dims) =
                pixels::plane_dims(dims, spec.subsampling);
            let (_, target_chroma_dims) =
                pixels::plane_dims(target, spec.subsampling);
            let target_planes = match planes {
                SamplePlanes::U8 {
                    planes: [luma, cb, cr],
                    alpha,
                } => {
                    let (target_planes, target_alpha) =
                        resize_homogeneous(
                            [
                                (luma.as_ref(), luma_dims, target),
                                (
                                    cb.as_ref(),
                                    chroma_dims,
                                    target_chroma_dims,
                                ),
                                (
                                    cr.as_ref(),
                                    chroma_dims,
                                    target_chroma_dims,
                                ),
                            ],
                            alpha
                                .as_deref()
                                .map(|plane| (plane, luma_dims, target)),
                            filter,
                            max_threads,
                        )?;
                    SamplePlanes::U8 {
                        planes: target_planes,
                        alpha: target_alpha,
                    }
                }
                SamplePlanes::U16 {
                    depth,
                    planes: [luma, cb, cr],
                    alpha,
                } => {
                    let (target_planes, target_alpha) =
                        resize_homogeneous(
                            [
                                (luma.as_ref(), luma_dims, target),
                                (
                                    cb.as_ref(),
                                    chroma_dims,
                                    target_chroma_dims,
                                ),
                                (
                                    cr.as_ref(),
                                    chroma_dims,
                                    target_chroma_dims,
                                ),
                            ],
                            alpha
                                .as_deref()
                                .map(|plane| (plane, luma_dims, target)),
                            filter,
                            max_threads,
                        )?;
                    SamplePlanes::U16 {
                        depth: *depth,
                        planes: target_planes,
                        alpha: target_alpha,
                    }
                }
            };
            Ok(Image::yuv(
                target.0,
                target.1,
                page.format(),
                page.transfer(),
                *spec,
                target_planes,
            )?)
        }
    }
}

type PlaneResize<'source, Sample> =
    (&'source [Sample], (u32, u32), (u32, u32));

type ResizedPlanes<Sample, const PLANES: usize> =
    ([Box<[Sample]>; PLANES], Option<Box<[Sample]>>);

fn resize_homogeneous<Sample, const PLANES: usize>(
    specs: [PlaneResize<'_, Sample>; PLANES],
    alpha: Option<PlaneResize<'_, Sample>>,
    filter: FilterType,
    max_threads: usize,
) -> eros::Result<ResizedPlanes<Sample, PLANES>>
where
    Sample:
        Copy + Default + PartialEq + Pod + PixelComponent + Send + Sync,
    Pixel<Sample, Sample, 1>: PixelTrait,
{
    let mut jobs: Vec<PlaneResize<'_, Sample>> = specs.into();
    if let Some(alpha_spec) = alpha {
        jobs.push(alpha_spec);
    }
    let mut resized = resize_plane_jobs(&jobs, filter, max_threads)?;
    let target_alpha = alpha.and_then(|_| resized.pop());
    let target_planes: [Box<[Sample]>; PLANES] =
        resized.try_into().map_err(|_vec: Vec<Box<[Sample]>>| {
            eros::error!("unexpected plane count")
        })?;
    Ok((target_planes, target_alpha))
}

fn resize_plane_jobs<Sample>(
    jobs: &[PlaneResize<'_, Sample>],
    filter: FilterType,
    max_threads: usize,
) -> eros::Result<Vec<Box<[Sample]>>>
where
    Sample:
        Copy + Default + PartialEq + Pod + PixelComponent + Send + Sync,
    Pixel<Sample, Sample, 1>: PixelTrait,
{
    let Some((&first, rest)) = jobs.split_first() else {
        return Ok(Vec::new());
    };
    if max_threads <= 1 || rest.is_empty() {
        let mut resizer = Resizer::new();
        return jobs
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
        let mut planes = Vec::with_capacity(jobs.len());
        planes.push(resize_plane(
            &mut resizer,
            first.0,
            first.1,
            first.2,
            filter,
        )?);
        for handle in handles {
            planes.push(handle.join().map_err(|_panic| {
                eros::error!("plane resample panicked")
            })??);
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
    let source =
        TypedImageRef::<Pixel<Element, Component, CHANNELS>>::from_buffer(
            source_dims.0,
            source_dims.1,
            bytemuck::cast_slice(source),
        )?;
    let mut target =
        TypedImage::<Pixel<Element, Component, CHANNELS>>::from_buffer(
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

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;
    use crate::image::decode::webp as webp_decode;
    use crate::image::pixels::Format;
    use crate::image::pixels::Matrix;
    use crate::image::pixels::WideDepth;
    use crate::image::pixels::YuvSpec;
    use crate::testkit;
    use crate::testkit::TEST_FIT_BOX;

    fn pixel_count(width: u32, height: u32) -> usize {
        (width as usize) * (height as usize)
    }

    fn rgba_page(width: u32, height: u32) -> Image {
        let bytes = (0..pixel_count(width, height) * RGBA_BYTES_PER_PIXEL)
            .map(|index| u8::try_from(index % 251).expect("channel"))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Image::rgba(width, height, Format::Png, Transfer::Srgb, bytes)
            .expect("rgba page")
            .with_page_dims((width, height))
    }

    fn rgba_bytes(page: &Image) -> Vec<u8> {
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
    ) -> Image {
        let ((_, _), (chroma_width, chroma_height)) =
            pixels::plane_dims((width, height), subsampling);
        let plane = |samples: usize, value: u16| {
            vec![value; samples].into_boxed_slice()
        };
        let chroma_samples = pixel_count(chroma_width, chroma_height);
        Image::yuv(
            width,
            height,
            Format::Avif,
            Transfer::Pq,
            YuvSpec {
                subsampling,
                full_range: false,
                matrix: Matrix::Bt709,
            },
            SamplePlanes::U16 {
                depth: WideDepth::D10,
                planes: [
                    plane(pixel_count(width, height), values[0]),
                    plane(chroma_samples, values[1]),
                    plane(chroma_samples, values[2]),
                ],
                alpha: Some(plane(pixel_count(width, height), values[3])),
            },
        )
        .expect("yuv page")
        .with_page_dims((width, height))
    }

    fn yuv_planes(page: &Image) -> [Vec<u16>; 4] {
        let Pixels::Yuv { planes, .. } = page.pixels() else {
            panic!("expected yuv pixels");
        };
        match planes {
            SamplePlanes::U8 {
                planes: [luma, cb, cr],
                alpha,
            } => [
                luma.iter().copied().map(u16::from).collect(),
                cb.iter().copied().map(u16::from).collect(),
                cr.iter().copied().map(u16::from).collect(),
                alpha.as_ref().map_or_else(Vec::new, |plane| {
                    plane.iter().copied().map(u16::from).collect()
                }),
            ],
            SamplePlanes::U16 {
                planes: [luma, cb, cr],
                alpha,
                ..
            } => [
                luma.to_vec(),
                cb.to_vec(),
                cr.to_vec(),
                alpha
                    .as_ref()
                    .map_or_else(Vec::new, |plane| plane.to_vec()),
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
            TargetRole::Full,
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
            derive(&page, TargetRole::Fit, limits, 4).expect("derives");
        assert_eq!(
            (derived.width(), derived.height()),
            (TEST_FIT_BOX.0, 64)
        );
        assert_eq!(derived.page_dims(), (4000, 100));
        let Pixels::Yuv { planes, .. } = derived.pixels() else {
            panic!("expected yuv pixels");
        };
        let Pixels::Yuv { spec, .. } = derived.pixels() else {
            panic!("expected yuv pixels");
        };
        assert_eq!(planes.depth(), Depth::D10);
        assert_eq!(spec.matrix, Matrix::Bt709);
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
        let assert_rgba_close =
            |bytes: &[u8], expected: [u8; 4], tol: u8| {
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

        let gray10 = Image::gray(
            400,
            200,
            Format::Avif,
            Transfer::Srgb,
            true,
            SamplePlanes::U16 {
                depth: WideDepth::D10,
                planes: [vec![1023_u16; pixel_count(400, 200)]
                    .into_boxed_slice()],
                alpha: Some(
                    vec![512_u16; pixel_count(400, 200)]
                        .into_boxed_slice(),
                ),
            },
        )
        .expect("gray10");
        let encoded = derive_thumb(&gray10, 1).expect("gray10 webp thumb");
        let (dims, rgba) =
            webp_decode::decode_rgba_bounded(&encoded.webp, THUMB_EDGE)
                .expect("decoded gray10 webp");
        assert_eq!(dims, (THUMB_EDGE, 96));
        assert_rgba_close(&rgba, [255, 255, 255, 128], 4);
    }
}
