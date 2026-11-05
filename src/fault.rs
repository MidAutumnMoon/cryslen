//! The fault taxonomy: what can go wrong, as types.
//!
//! Refusals ([`OpenRefusal`], [`DecodeRefusal`]) are verdicts about the
//! input that callers may match on. Faults carry structured detail;
//! [`PanicFault`] and [`BadIndex`] name bugs, not bad input.
//!
//! [`PageError`] is the one shared set, stored behind `Arc` in the
//! source's fill, the worker's book, and the screen's upload error, so
//! an error cannot change set once wrapped. [`DecodeError`] and
//! [`TextureError`] widen into it where the page is the unit of
//! failure. [`OpenError`] never does, because an open refusal owns no
//! page; [`SpawnError`] widens into [`OpenError`] at the book switch.

use std::any::Any;
use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;

use eros::ErrorUnion;
use image::ImageError;
use jxl::error::Error as JxlError;
use zip::result::ZipError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenRefusal {
    NotFound(PathBuf),
    Unsupported(PathBuf),
    Empty,
    Duplicate(String),
    TooBig { unpacked: u64, available: u64 },
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanicFault {
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

/// A live source lists at least one page, so every index below its
/// length has a name; a miss means a caller bug.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BadIndex;

impl fmt::Display for BadIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no such page")
    }
}

impl Error for BadIndex {}

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

pub type OpenError =
    ErrorUnion<(OpenRefusal, ZipError, sevenz_rust2::Error, io::Error)>;

pub type DecodeError =
    ErrorUnion<(DecodeRefusal, AvifFault, ImageError, JxlError)>;

pub type TextureError = ErrorUnion<(TextureFault,)>;

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
