//! The fault taxonomy: matchable refusals about input, a page fault as
//! text, and a book fault with detail. [`PageError`] collects the
//! page-level errors; an open error owns no page.

use std::any::Any;
use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use eros::ErrorUnion;
use image::ImageError;
use jxl::error::Error as JxlError;
use serde::Deserialize;
use serde::Serialize;
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

/// A page's failure as text; the error types stay in the child.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PageFault {
    summary: String,
}

impl PageFault {
    pub fn new(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
        }
    }
}

impl fmt::Display for PageFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary)
    }
}

impl From<PageError> for PageFault {
    fn from(error: PageError) -> Self {
        Self::new(error.to_string())
    }
}

impl From<Arc<PageError>> for PageFault {
    fn from(error: Arc<PageError>) -> Self {
        Self::new(error.to_string())
    }
}

/// Why a book could not be opened or kept open.
#[derive(Debug)]
pub enum BookFault {
    Spawn(String),
    Handshake(String),
    Open(String),
    Timeout,
    Died(String),
}

impl fmt::Display for BookFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn(detail) => {
                write!(f, "cannot start the book process: {detail}")
            }
            Self::Handshake(detail) => {
                write!(f, "book process handshake failed: {detail}")
            }
            Self::Open(detail) => {
                write!(f, "cannot open the book: {detail}")
            }
            Self::Timeout => {
                write!(f, "the book process did not answer in time")
            }
            Self::Died(detail) => {
                write!(f, "the book process died: {detail}")
            }
        }
    }
}

impl Error for BookFault {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_fault_reads_payload_kinds() {
        let string =
            PanicFault::from_payload("decoder", &String::from("boom"));
        assert_eq!(string.origin, "decoder");
        assert_eq!(string.message, "boom");

        let text = PanicFault::from_payload("unpack", &"static boom");
        assert_eq!(text.origin, "unpack");
        assert_eq!(text.message, "static boom");

        let other = PanicFault::from_payload("unpack", &7_u8);
        assert_eq!(other.origin, "unpack");
        assert_eq!(other.message, "non-string panic payload");
    }
}
