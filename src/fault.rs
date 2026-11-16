//! Shared fault types: texture limit violations, IPC page fault summaries,
//! and panic payload formatting.

use std::any::Any;
use std::error::Error;
use std::fmt;

use serde::Deserialize;
use serde::Serialize;

pub fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic payload")
}

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

impl From<TextureFault> for PageFault {
    fn from(error: TextureFault) -> Self {
        Self::new(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_message_reads_payload_kinds() {
        assert_eq!(panic_message(&String::from("boom")), "boom");
        assert_eq!(panic_message(&"static boom"), "static boom");
        assert_eq!(panic_message(&7_u8), "non-string panic payload");
    }
}
