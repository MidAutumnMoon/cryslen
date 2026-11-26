//! Color conversion math: YUV/Gray quantization and PQ-to-sRGB transfer.

use crate::image::pixels::Depth;
use crate::image::pixels::Matrix;

pub const R8_STORAGE_MAX: f32 = 255.0;
pub const R16_STORAGE_MAX: f32 = 65535.0;
pub const CHROMA_NEUTRAL_8BIT: u8 = 128;

/// Fold storage samples to gamma RGB: `matrix * samples - offset`.
/// Samples are `0..1` storage units; `R16` sampling divides by `65535`.
#[derive(Debug, Clone, Copy)]
pub struct YuvMatrix {
    /// Column-major 3x3.
    pub matrix: [f32; 9],
    pub offset: [f32; 3],
}

impl YuvMatrix {
    #[must_use]
    pub const fn gray(gain: f32, offset: f32) -> Self {
        Self {
            matrix: [gain, gain, gain, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            offset: [offset; 3],
        }
    }

    #[must_use]
    pub fn apply(&self, samples: [f32; 3]) -> [f32; 3] {
        let mut rgb = [0.0_f32; 3];
        for (column, sample) in
            self.matrix.as_chunks::<3>().0.iter().zip(samples)
        {
            for (&cell, slot) in column.iter().zip(&mut rgb) {
                *slot = cell.mul_add(sample, *slot);
            }
        }
        for (slot, offset) in rgb.iter_mut().zip(self.offset) {
            *slot -= offset;
        }
        rgb
    }
}

#[must_use]
pub fn max_storage_code(depth: Depth) -> f32 {
    if depth.is_wide() {
        R16_STORAGE_MAX
    } else {
        R8_STORAGE_MAX
    }
}

fn max_sample_code(depth: Depth) -> u16 {
    u16::MAX >> (Depth::D16.bits() - depth.bits())
}

/// Compensate for GL `R16` normalization: sampling divides by
/// [`R16_STORAGE_MAX`] even when only `depth` bits are used.
#[must_use]
pub fn depth_normalization(depth: Depth) -> f32 {
    max_storage_code(depth) / f32::from(max_sample_code(depth))
}

struct RangeQuantization {
    luma_black: f32,
    luma_scale: f32,
    chroma_scale: f32,
    chroma_neutral: f32,
}

fn range_quantization(
    depth: Depth,
    full_range: bool,
) -> RangeQuantization {
    const STUDIO_LUMA_BLACK_8BIT: u16 = 16;
    const STUDIO_LUMA_SWING_8BIT: u16 = 219;
    const STUDIO_CHROMA_SWING_8BIT: u16 = 224;

    let depth_scale = 1_u16 << (depth.bits() - Depth::D8.bits());
    let chroma_neutral =
        f32::from(u16::from(CHROMA_NEUTRAL_8BIT) * depth_scale);
    let (luma_black, luma_scale, chroma_scale) = if full_range {
        (0.0, 1.0, 1.0)
    } else {
        let max_code = f32::from(max_sample_code(depth));
        (
            f32::from(STUDIO_LUMA_BLACK_8BIT * depth_scale),
            max_code / f32::from(STUDIO_LUMA_SWING_8BIT * depth_scale),
            max_code / f32::from(STUDIO_CHROMA_SWING_8BIT * depth_scale),
        )
    };
    RangeQuantization {
        luma_black,
        luma_scale,
        chroma_scale,
        chroma_neutral,
    }
}

#[must_use]
pub fn gray_gain_offset(depth: Depth, full_range: bool) -> (f32, f32) {
    let range = range_quantization(depth, full_range);
    let gain = range.luma_scale * depth_normalization(depth);
    let offset = gain * (range.luma_black / max_storage_code(depth));
    (gain, offset)
}

/// Fold storage samples to gamma RGB: depth normalization, quantization,
/// and the KR/KB matrix (or GBR permutation). The offset is the matrix
/// times the anchor codes.
#[must_use]
pub fn yuv_color_conversion(
    depth: Depth,
    full_range: bool,
    color_matrix: Matrix,
) -> YuvMatrix {
    const GBR_TO_RGB_MATRIX: [f32; 9] =
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];

    let range = range_quantization(depth, full_range);
    let (mut matrix, luma_scale, chroma_scale, luma_black, chroma_neutral) =
        if color_matrix == Matrix::Identity {
            // H.273 identity planes are GBR in the (Y, Cb, Cr) slots,
            // quantized as luma on all three axes: permute into RGB and
            // anchor every axis at luma black.
            (
                GBR_TO_RGB_MATRIX,
                range.luma_scale,
                range.luma_scale,
                range.luma_black,
                range.luma_black,
            )
        } else {
            (
                yuv_to_rgb_matrix(color_matrix),
                range.luma_scale,
                range.chroma_scale,
                range.luma_black,
                range.chroma_neutral,
            )
        };
    let storage_max = max_storage_code(depth);
    let normalization = depth_normalization(depth);
    for (column, matrix_column) in
        matrix.as_chunks_mut::<3>().0.iter_mut().enumerate()
    {
        let scale = if column == 0 {
            luma_scale
        } else {
            chroma_scale
        } * normalization;
        for cell in matrix_column {
            *cell *= scale;
        }
    }
    let anchors = [
        luma_black / storage_max,
        chroma_neutral / storage_max,
        chroma_neutral / storage_max,
    ];
    let mut offset = [0.0_f32; 3];
    for (matrix_column, anchor) in
        matrix.as_chunks::<3>().0.iter().zip(anchors)
    {
        for (&cell, slot) in matrix_column.iter().zip(&mut offset) {
            *slot = cell.mul_add(anchor, *slot);
        }
    }
    YuvMatrix { matrix, offset }
}

/// Full-range YUV->RGB from the exact KR/KB of each standard.
/// KG = 1 - KR - KB. Identity pages never reach here;
/// `yuv_color_conversion` routes them through the GBR permutation.
#[expect(
    clippy::similar_names,
    reason = "the green coefficients are the Cb and Cr terms of the BT matrix derivation"
)]
fn yuv_to_rgb_matrix(matrix: Matrix) -> [f32; 9] {
    const BT601_KR_KB: (f32, f32) = (0.299, 0.114);
    const BT709_KR_KB: (f32, f32) = (0.2126, 0.0722);
    const BT2020_KR_KB: (f32, f32) = (0.2627, 0.0593);

    let (k_red, k_blue) = match matrix {
        Matrix::Bt601 => BT601_KR_KB,
        Matrix::Bt709 => BT709_KR_KB,
        Matrix::Bt2020 => BT2020_KR_KB,
        Matrix::Identity => (0.0, 0.0),
    };
    let k_green = 1.0 - k_red - k_blue;
    let green_from_cb = 2.0 * k_blue * (1.0 - k_blue) / k_green;
    let green_from_cr = 2.0 * k_red * (1.0 - k_red) / k_green;
    [
        1.0,
        1.0,
        1.0, //
        0.0,
        -green_from_cb,
        2.0 * (1.0 - k_blue), //
        2.0 * (1.0 - k_red),
        -green_from_cr,
        0.0,
    ]
}

/// Map a normalized `[0, 1]` PQ channel value to sRGB `[0, 1]` using the
/// same 203-nit diffuse-white clip as `GLSL_TRANSFER_HEADER`.
#[must_use]
pub fn pq_to_srgb_channel(value: f32) -> f32 {
    const PQ_M1: f32 = 2610.0 / 16384.0;
    const PQ_M2: f32 = 2523.0 / 4096.0 * 128.0;
    const PQ_C1: f32 = 3424.0 / 4096.0;
    const PQ_C2: f32 = 2413.0 / 4096.0 * 32.0;
    const PQ_C3: f32 = 2392.0 / 4096.0 * 32.0;
    const PQ_DIFFUSE_WHITE_NITS: f32 = 203.0;
    const PQ_PEAK_NITS: f32 = 10_000.0;
    const SRGB_LINEAR_THRESHOLD: f32 = 0.003_130_8;
    const SRGB_LINEAR_SLOPE: f32 = 12.92;
    const SRGB_GAMMA_SCALE: f32 = 1.055;
    const SRGB_GAMMA_OFFSET: f32 = 0.055;
    const SRGB_GAMMA_EXPONENT: f32 = 2.4;

    let pq_scaled = value.clamp(0.0, 1.0).powf(1.0 / PQ_M2);
    let luminance_nits = PQ_PEAK_NITS
        * ((pq_scaled - PQ_C1).max(0.0)
            / PQ_C3.mul_add(-pq_scaled, PQ_C2))
        .powf(1.0 / PQ_M1);
    let sdr_linear =
        (luminance_nits / PQ_DIFFUSE_WHITE_NITS).clamp(0.0, 1.0);
    if sdr_linear <= SRGB_LINEAR_THRESHOLD {
        sdr_linear * SRGB_LINEAR_SLOPE
    } else {
        SRGB_GAMMA_SCALE.mul_add(
            sdr_linear.powf(1.0 / SRGB_GAMMA_EXPONENT),
            -SRGB_GAMMA_OFFSET,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Matrix element at (`row`, `col`); storage is column-major.
    #[expect(
        clippy::indexing_slicing,
        reason = "row and col are literal 0..3 coordinates into the fixed 3x3 matrix under test"
    )]
    fn element(matrix: &[f32; 9], row: usize, col: usize) -> f32 {
        matrix[col * 3 + row]
    }

    /// Code values of one depth's quantization (black, white, neutral) in
    /// code units, and the largest code the storage layout holds.
    fn code_quantization_codes(
        depth: Depth,
        full_range: bool,
    ) -> (f32, f32, f32, f32) {
        let bits = depth.bits();
        let scale = 2.0_f32.powi(i32::from(bits) - 8);
        let storage_max =
            2.0_f32.powi(if depth.is_wide() { 16 } else { 8 }) - 1.0;
        let codes = if full_range {
            (
                0.0,
                2.0_f32.powi(i32::from(bits)) - 1.0,
                2.0_f32.powi(i32::from(bits) - 1),
            )
        } else {
            (16.0 * scale, 235.0 * scale, 128.0 * scale)
        };
        (codes.0, codes.1, codes.2, storage_max)
    }

    #[test]
    fn color_conversion_anchors_black_and_white_at_every_depth() {
        for depth in [Depth::D8, Depth::D10, Depth::D12, Depth::D16] {
            for full_range in [true, false] {
                for conversion in [
                    yuv_color_conversion(depth, full_range, Matrix::Bt601),
                    {
                        let (gain, offset) =
                            gray_gain_offset(depth, full_range);
                        YuvMatrix::gray(gain, offset)
                    },
                ] {
                    let (black, white, neutral, storage_max) =
                        code_quantization_codes(depth, full_range);
                    for (code, expected) in [(black, 0.0), (white, 1.0)] {
                        // Chroma at its neutral code contributes nothing.
                        let rgb = conversion.apply([
                            code / storage_max,
                            neutral / storage_max,
                            neutral / storage_max,
                        ]);
                        for (row, channel) in rgb.into_iter().enumerate() {
                            assert!(
                                (channel - expected).abs() < 1e-4,
                                "depth {depth:?} full {full_range} row {row}: {channel} vs {expected}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn identity_pages_permute_gbr() {
        let color =
            yuv_color_conversion(Depth::D8, true, Matrix::Identity);
        // Column j is plane j's contribution: Y->G, U->B, V->R.
        for (col, row) in [(0, 1), (1, 2), (2, 0)] {
            for probe in 0..3 {
                let want = f32::from(probe == row);
                assert!(
                    (element(&color.matrix, probe, col) - want).abs()
                        < 1e-6,
                    "col {col} row {probe}"
                );
            }
        }
        // Full range anchors everything at black zero.
        assert_eq!(color.offset, [0.0; 3]);
    }
}
