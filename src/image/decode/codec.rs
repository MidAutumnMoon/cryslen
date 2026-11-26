//! Format sniffing, extension matching, and decoder dispatch.

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use crate::image::pixels::Format;
use crate::image::pixels::Image;

use super::avif;
use super::gif;
use super::jxl;
use super::raster;
use super::webp;

pub enum DecodeOutcome {
    Ready(Image),
    Canceled,
}

#[cfg(test)]
pub fn decode(bytes: &[u8], name: &str) -> eros::Result<Image> {
    const TEST_MAX_THREADS: usize = 2;
    let cancel = AtomicBool::new(false);
    match decode_cancellable(bytes, name, TEST_MAX_THREADS, &cancel)? {
        DecodeOutcome::Ready(decoded) => Ok(decoded),
        DecodeOutcome::Canceled => eros::bail!("decode canceled"),
    }
}

/// Decode one page's first raster, aborting cooperatively when `cancel`
/// flips during JXL decoding.
pub fn decode_cancellable(
    bytes: &[u8],
    name: &str,
    max_threads: usize,
    cancel: &AtomicBool,
) -> eros::Result<DecodeOutcome> {
    if cancel.load(Ordering::Relaxed) {
        return Ok(DecodeOutcome::Canceled);
    }
    let Some(format) = sniff(bytes).or_else(|| format_from_name(name))
    else {
        eros::bail!("not an image");
    };
    let max_threads = max_threads.max(1);
    match format {
        Format::Avif => {
            avif::decode(bytes, max_threads).map(DecodeOutcome::Ready)
        }
        Format::Jxl => jxl::decode(bytes, max_threads, cancel),
        Format::Webp => {
            webp::decode(bytes, max_threads).map(DecodeOutcome::Ready)
        }
        Format::Gif => gif::decode(bytes).map(DecodeOutcome::Ready),
        Format::Jpeg | Format::Png | Format::Bmp | Format::Tiff => {
            raster::decode(bytes, format).map(DecodeOutcome::Ready)
        }
    }
}

/// Format for a page filename's extension. The same table decides what can
/// decode and what gets listed as a page.
pub fn format_from_name(name: &str) -> Option<Format> {
    use std::path::Path;

    let extension = Path::new(name)
        .extension()
        .and_then(|extension| extension.to_str())?
        .to_ascii_lowercase();
    Some(match extension.as_str() {
        "jpg" | "jpeg" => Format::Jpeg,
        "png" => Format::Png,
        "gif" => Format::Gif,
        "webp" => Format::Webp,
        "bmp" => Format::Bmp,
        "tif" | "tiff" => Format::Tiff,
        "avif" | "avifs" => Format::Avif,
        "jxl" => Format::Jxl,
        _ => return None,
    })
}

fn sniff(bytes: &[u8]) -> Option<Format> {
    const RIFF_TYPE_OFFSET: usize = 8;
    const RIFF_HEADER_BYTES: usize = 12;

    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(Format::Jpeg);
    }
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A])
    {
        return Some(Format::Png);
    }
    if bytes.starts_with(b"GIF8") {
        return Some(Format::Gif);
    }
    if bytes.starts_with(b"RIFF")
        && bytes.get(RIFF_TYPE_OFFSET..RIFF_HEADER_BYTES) == Some(b"WEBP")
    {
        return Some(Format::Webp);
    }
    if bytes.starts_with(b"BM") {
        return Some(Format::Bmp);
    }
    if bytes.starts_with(&[0x49, 0x49, 0x2A, 0x00])
        || bytes.starts_with(&[0x4D, 0x4D, 0x00, 0x2A])
    {
        return Some(Format::Tiff);
    }
    // Raw codestream or ISOBMFF container signature.
    if bytes.starts_with(&[0xFF, 0x0A])
        || bytes.starts_with(&[
            0x00, 0x00, 0x00, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A,
            0x87, 0x0A,
        ])
    {
        return Some(Format::Jxl);
    }
    if sniff_ftyp_avif(bytes) {
        return Some(Format::Avif);
    }
    None
}

fn sniff_ftyp_avif(bytes: &[u8]) -> bool {
    ftyp_brands(bytes).is_some_and(|brands| {
        brands
            .as_chunks::<4>()
            .0
            .iter()
            .any(|brand| brand == b"avif" || brand == b"avis")
    })
}

/// Brand region of the leading `ftyp` box. A box length under 12 still
/// yields the 4-byte major brand.
fn ftyp_brands(bytes: &[u8]) -> Option<&[u8]> {
    const BOX_FIELD_BYTES: usize = 4;
    const BOX_HEADER_BYTES: usize = BOX_FIELD_BYTES * 2;
    const MIN_FTYP_BOX_BYTES: usize = BOX_HEADER_BYTES + BOX_FIELD_BYTES;

    let (&box_len_bytes, rest) =
        bytes.split_first_chunk::<BOX_FIELD_BYTES>()?;
    #[expect(
        clippy::big_endian_bytes,
        reason = "the ISOBMFF box length is big-endian by specification"
    )]
    let box_len = u32::from_be_bytes(box_len_bytes) as usize;
    let kind = rest.first_chunk::<BOX_FIELD_BYTES>()?;
    if kind != b"ftyp" {
        return None;
    }
    let end = box_len.max(MIN_FTYP_BOX_BYTES).min(bytes.len());
    bytes.get(BOX_HEADER_BYTES..end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_detection() {
        assert_eq!(sniff(&[0xFF, 0xD8, 0xFF, 0xE0]), Some(Format::Jpeg));
        assert_eq!(sniff(b"\x89PNG\r\n\x1a\n"), Some(Format::Png));
        assert_eq!(sniff(b"GIF89a"), Some(Format::Gif));
        assert_eq!(
            sniff(b"RIFF\x00\x00\x00\x00WEBPVP8 "),
            Some(Format::Webp)
        );
        assert_eq!(sniff(b"BM\x36\x00"), Some(Format::Bmp));
        assert_eq!(sniff(b"II*\x00"), Some(Format::Tiff));
        assert_eq!(sniff(b"\xff\x0a"), Some(Format::Jxl));
        let mut avif = vec![0, 0, 0, 0x14];
        avif.extend_from_slice(b"ftypavif");
        assert_eq!(sniff(&avif), Some(Format::Avif));
        let mut heic = vec![0, 0, 0, 0x14];
        heic.extend_from_slice(b"ftypmif1");
        assert_eq!(sniff(&heic), None);
        assert_eq!(sniff(b"plain text"), None);

        assert_eq!(format_from_name("a.JPG"), Some(Format::Jpeg));
        assert_eq!(format_from_name("a.tiff"), Some(Format::Tiff));
        assert_eq!(format_from_name("a.avif"), Some(Format::Avif));
        assert_eq!(format_from_name("a.jxl"), Some(Format::Jxl));
        assert_eq!(format_from_name("cover.cbz"), None);
        assert_eq!(format_from_name("noext"), None);

        assert!(decode(b"hello world", "noext").is_err());
        assert!(decode(b"hello world", "note.txt").is_err());
        assert!(decode(&[], "x").is_err());
    }
}
