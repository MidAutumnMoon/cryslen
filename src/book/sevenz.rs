//! 7z block scheduling, LZMA dictionary peak estimation, and `/proc/meminfo` memory guard.

use std::fs;
use std::num::NonZeroUsize;

use sevenz_rust2::Archive;

/// Kept free in `MemAvailable` for in-flight decodes, GPU textures, the
/// preview store, and the rest of the system.
const MEM_RESERVE_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_SEVENZIP_HEADER_KIB: u64 = 128 * 1024;

pub(super) struct PlannedBlock {
    pub index: usize,
    pub page_count: NonZeroUsize,
}

pub(super) struct SevenZipPlan {
    pub blocks: Vec<PlannedBlock>,
    pub empty_pages: Vec<usize>,
}

impl SevenZipPlan {
    pub fn build(
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

    pub fn decoder_peak_bytes(&self, archive: &Archive) -> u64 {
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
                .get(LZMA_DICTIONARY_OFFSET..)?
                .first_chunk::<LZMA_DICTIONARY_BYTES>()?;
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

/// `MemAvailable` from a `/proc/meminfo` text, converted to bytes. The
/// line counts KiB. `None` when the line is missing or unparsable.
fn parse_mem_available_bytes(meminfo: &str) -> Option<u64> {
    const BYTES_PER_KIB: u64 = 1024;

    let line = meminfo
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kib.checked_mul(BYTES_PER_KIB)
}

pub(super) fn mem_available_bytes() -> Option<u64> {
    parse_mem_available_bytes(&fs::read_to_string("/proc/meminfo").ok()?)
}

pub(super) fn sevenzip_header_limit_kib(
    available_bytes: Option<u64>,
) -> usize {
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

pub(super) fn guard_memory(
    needed_bytes: u64,
    available_bytes: Option<u64>,
    reclaimable_bytes: u64,
) -> eros::Result<()> {
    match available_bytes {
        Some(available_bytes)
            if needed_bytes
                > available_bytes
                    .saturating_add(reclaimable_bytes)
                    .saturating_sub(MEM_RESERVE_BYTES) =>
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
    clippy::unwrap_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;

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
        assert!(guard_memory(100, Some(50), 0).is_err());
        guard_memory(50, Some(MEM_RESERVE_BYTES + 100), 0).unwrap();
        guard_memory(u64::MAX, None, 0).unwrap();
        let available = 4 * 1024 * 1024 * 1024;
        guard_memory(available - MEM_RESERVE_BYTES, Some(available), 0)
            .unwrap();
        assert!(
            guard_memory(
                available - MEM_RESERVE_BYTES + 1,
                Some(available),
                0,
            )
            .is_err()
        );
        // Reclaimable bytes from a currently open book count toward budget.
        guard_memory(
            available - MEM_RESERVE_BYTES + 1024,
            Some(available),
            1024,
        )
        .unwrap();

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
}
