//! The fault taxonomy: what can go wrong, as types.
//!
//! Refusals ([`OpenRefusal`], [`DecodeRefusal`]) are classification
//! facts the caller may match on. Faults ([`AvifFault`],
//! [`PanicError`], [`TextureFault`]) carry structured detail. The
//! error unions ([`PageError`], [`OpenError`], [`DecodeError`]) are
//! the transport: which of these types a layer can fail with, stated
//! in its signature.
//!
//! [`PageError`] is the one shared set: it is stored behind `Arc` in
//! the worker's book and the screen's error cells, and an error
//! cannot change set after the `Arc` wraps it. Narrower unions
//! ([`DecodeError`], [`OpenError`], `ErrorUnion<(TextureFault,)>`)
//! widen into it exactly once, at the layer that learns the page is
//! the unit of failure.

use std::any::Any;
use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use ::jxl::error::Error as JxlError;
use eros::ErrorUnion;
use eros::MsgError;
use image::ImageError;
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
            Self::Empty => write!(f, "no image pages found"),
            Self::Unsupported(path) => write!(f, "unsupported source: {}", path.display()),
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
pub struct PanicError {
    /// Which boundary caught it, e.g. `"decoder"`.
    pub origin: &'static str,
    pub message: String,
}

impl fmt::Display for PanicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} panicked: {}", self.origin, self.message)
    }
}

impl Error for PanicError {}

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
pub type PageSet = (
    DecodeRefusal,
    AvifFault,
    ImageError,
    JxlError,
    ZipError,
    sevenz_rust2::Error,
    io::Error,
    PanicError,
    TextureFault,
    MsgError,
);
pub type PageError = ErrorUnion<PageSet>;

/// What opening a source can fail with.
pub type OpenSet = (OpenRefusal, ZipError, sevenz_rust2::Error, io::Error);
pub type OpenError = ErrorUnion<OpenSet>;

/// What decoding one page can fail with; widens into [`PageError`]
/// where the page becomes the unit of failure.
pub type DecodeSet = (DecodeRefusal, AvifFault, ImageError, JxlError);
pub type DecodeError = ErrorUnion<DecodeSet>;

/// What spawning a worker or building the app can fail with.
pub type SpawnError = ErrorUnion<(io::Error,)>;

/// What the status line blames: a page's own failure (decode, source,
/// or a texture the GPU refused), or a book switch that failed.
pub enum StatusFault {
    Page(Arc<PageError>),
    Noted(Arc<OpenError>),
}

impl fmt::Display for StatusFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Page(error) => error.fmt(f),
            Self::Noted(error) => error.fmt(f),
        }
    }
}

/// A panic payload rendered as a fault, for the thread panic
/// boundaries: the unpack pass and the decoder in the worker.
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is the crate's visibility style for crate-wide items, even inside private modules"
)]
pub(crate) fn from_payload(
    origin: &'static str,
    payload: &(dyn Any + Send),
) -> PanicError {
    let message = payload
        .downcast_ref::<&str>()
        .map(|text| (*text).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| String::from("non-string panic payload"));
    PanicError { origin, message }
}

/// The caller asked for a page index the source does not list: an
/// internal invariant, not a content problem.
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is the crate's visibility style for crate-wide items, even inside private modules"
)]
pub(crate) fn bad_index() -> PageError {
    ErrorUnion::new(MsgError::from_static("no such page"))
}

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
    }

    #[test]
    fn panic_fault_carries_its_origin() {
        assert_eq!(
            from_payload("decoder", &String::from("boom")).to_string(),
            "decoder panicked: boom"
        );
        assert_eq!(
            from_payload("unpack", &"static boom").to_string(),
            "unpack panicked: static boom"
        );
        assert_eq!(
            from_payload("unpack", &7_u8).to_string(),
            "unpack panicked: non-string panic payload"
        );
        assert_eq!(bad_index().to_string(), "no such page");
    }
}
