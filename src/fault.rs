//! The fault taxonomy: what can go wrong, as types.
//!
//! Refusals ([`OpenRefusal`], [`DecodeRefusal`]) are classification
//! facts the caller may match on. Faults ([`AvifFault`],
//! [`PanicFault`], [`BadIndex`], [`TextureFault`]) carry structured
//! detail; the last two of those name bugs, not bad input. The error
//! unions ([`PageError`], [`OpenError`], [`DecodeError`],
//! [`TextureError`], [`SpawnError`]) are the transport: which of
//! these types a layer can fail with, stated in its signature.
//!
//! [`PageError`] is the one shared set: it is stored behind `Arc` in
//! the worker's book and the screen's error cells, and an error
//! cannot change set after the `Arc` wraps it. The page-side unions
//! ([`DecodeError`], [`TextureError`]) widen into it where a layer
//! learns the page is the unit of failure: the worker's decode
//! verdict, the screen's texture uploads. An open refusal owns no
//! page, so [`OpenError`] never widens into [`PageError`]: the app
//! pins it beside the page cells and the status line speaks for it
//! distinctly. [`SpawnError`] widens into [`OpenError`] at the book
//! switch.

use std::any::Any;
use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;

use eros::ErrorUnion;
use image::ImageError;
use jxl::error::Error as JxlError;
use zip::result::ZipError;

/// Why a source refused to open.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenRefusal {
    NotFound(PathBuf),
    Unsupported(PathBuf),
    Empty,
    /// The listing names one entry twice.
    Duplicate(String),
    /// The sizes the listing claims exceed the memory the machine
    /// reports available.
    TooBig {
        unpacked: u64,
        available: u64,
    },
}

impl fmt::Display for OpenRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(path) => write!(f, "not found: {}", path.display()),
            Self::Unsupported(path) => write!(f, "unsupported source: {}", path.display()),
            Self::Empty => write!(f, "no image pages found"),
            Self::Duplicate(name) => write!(f, "duplicate entry name: {name}"),
            Self::TooBig { unpacked, available } => write!(
                f,
                "archive too big: {unpacked} bytes claimed, {available} bytes available"
            ),
        }
    }
}

impl Error for OpenRefusal {}

/// Why bytes refused to decode: a verdict about the content, not a
/// failure in the decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeRefusal {
    NotAnImage,
    Corrupt(&'static str),
    Unsupported(&'static str),
}

impl fmt::Display for DecodeRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAnImage => write!(f, "not an image"),
            Self::Corrupt(what) => write!(f, "corrupt: {what}"),
            Self::Unsupported(what) => write!(f, "unsupported: {what}"),
        }
    }
}

impl Error for DecodeRefusal {}

/// A libavif call failure with its step and avifResult detail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvifFault {
    pub step: &'static str,
    pub detail: String,
}

impl fmt::Display for AvifFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.step, self.detail)
    }
}

impl Error for AvifFault {}

/// A panic payload that crossed a thread boundary: the decoder or the
/// unpack pass died on this page. A bug, not bad input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanicFault {
    /// Which boundary caught it, e.g. `"decoder"`.
    pub origin: &'static str,
    pub message: String,
}

impl fmt::Display for PanicFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} panicked: {}", self.origin, self.message)
    }
}

impl Error for PanicFault {}

impl PanicFault {
    /// Render a panic payload, for the thread panic boundaries: the
    /// unpack pass in the source and the decoder in the worker.
    pub fn from_payload(
        origin: &'static str,
        payload: &(dyn Any + Send),
    ) -> Self {
        let message = payload
            .downcast_ref::<&str>()
            .map(|text| (*text).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| String::from("non-string panic payload"));
        Self { origin, message }
    }
}

/// The caller asked for a page index the source does not list. A bug,
/// not bad input: a live source lists at least one page, and every
/// index below its length has a name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BadIndex;

impl fmt::Display for BadIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no such page")
    }
}

impl Error for BadIndex {}

/// A page this GPU cannot texture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureFault {
    pub dims: (u32, u32),
    pub max_edge: u32,
}

impl fmt::Display for TextureFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}x{} exceeds this GPU's {}-texel texture edge limit",
            self.dims.0, self.dims.1, self.max_edge
        )
    }
}

impl Error for TextureFault {}

/// What one page can fail with, source side and decode side together.
/// Shared by every `Arc`-pinned error cell in the worker and screen.
pub type PageError = ErrorUnion<(
    DecodeRefusal,
    AvifFault,
    ImageError,
    JxlError,
    ZipError,
    sevenz_rust2::Error,
    io::Error,
    PanicFault,
    BadIndex,
    TextureFault,
)>;

/// What opening a source can fail with. Never widens into
/// [`PageError`]: an open refusal owns no page.
pub type OpenError =
    ErrorUnion<(OpenRefusal, ZipError, sevenz_rust2::Error, io::Error)>;

/// What decoding one page can fail with; widens into [`PageError`]
/// where the page becomes the unit of failure.
pub type DecodeError =
    ErrorUnion<(DecodeRefusal, AvifFault, ImageError, JxlError)>;

/// What caching one page's textures can fail with: the size refusal.
/// Widens into [`PageError`] where the page becomes the unit of
/// failure.
pub type TextureError = ErrorUnion<(TextureFault,)>;

/// What spawning a worker or building the app can fail with; widens
/// into [`OpenError`] at the book switch.
pub type SpawnError = ErrorUnion<(io::Error,)>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusal_display_strings() {
        assert_eq!(OpenRefusal::Empty.to_string(), "no image pages found");
        assert_eq!(
            OpenRefusal::NotFound(PathBuf::from("/a")).to_string(),
            "not found: /a"
        );
        assert_eq!(
            OpenRefusal::Unsupported(PathBuf::from("/a.zip")).to_string(),
            "unsupported source: /a.zip"
        );
        assert_eq!(
            OpenRefusal::Duplicate(String::from("a.png")).to_string(),
            "duplicate entry name: a.png"
        );
        assert_eq!(
            OpenRefusal::TooBig {
                unpacked: 8,
                available: 4
            }
            .to_string(),
            "archive too big: 8 bytes claimed, 4 bytes available"
        );
        assert_eq!(DecodeRefusal::NotAnImage.to_string(), "not an image");
        assert_eq!(
            DecodeRefusal::Corrupt("bad block").to_string(),
            "corrupt: bad block"
        );
        assert_eq!(
            DecodeRefusal::Unsupported("16-bit").to_string(),
            "unsupported: 16-bit"
        );
    }

    #[test]
    fn fault_display_strings() {
        assert_eq!(
            AvifFault {
                step: "avifDecoderNextImage",
                detail: String::from("AVIF_RESULT_DECODE_COLOR_FAILED")
            }
            .to_string(),
            "avifDecoderNextImage: AVIF_RESULT_DECODE_COLOR_FAILED"
        );
        assert_eq!(
            TextureFault {
                dims: (6000, 4000),
                max_edge: 0x4000
            }
            .to_string(),
            "6000x4000 exceeds this GPU's 16384-texel texture edge limit"
        );
        assert_eq!(BadIndex.to_string(), "no such page");
    }

    #[test]
    fn panic_fault_carries_its_origin() {
        assert_eq!(
            PanicFault::from_payload("decoder", &String::from("boom"))
                .to_string(),
            "decoder panicked: boom"
        );
        assert_eq!(
            PanicFault::from_payload("unpack", &"static boom").to_string(),
            "unpack panicked: static boom"
        );
        assert_eq!(
            PanicFault::from_payload("unpack", &7_u8).to_string(),
            "unpack panicked: non-string panic payload"
        );
    }
}
