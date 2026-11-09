//! cryslen: a wayland image viewer with manga mode.
//!
//! The parent process owns the window, the UI, and the GPU; each open
//! book lives in a child process that owns its source, decode, and
//! preview store. The two sides exchange [`protocol`] messages.

use std::ffi::OsStr;
use std::ffi::OsString;
use std::process::ExitCode;

pub mod protocol;

mod app;
mod book;
mod catalog;
mod child;
mod decode;
mod fault;
mod keys;
mod parent;
mod plan;
mod preview;
mod render;
mod source;
mod ui;
mod view;
mod window;
mod worker;

/// Run the mode the command line names: `cy --worker <ipc-server-name>
/// <path>` serves one book, and anything else is the windowed parent.
#[must_use]
pub fn run(args: &[OsString]) -> ExitCode {
    let worker = args
        .get(1)
        .is_some_and(|arg| arg.as_os_str() == OsStr::new("--worker"));
    if worker {
        child::main(args)
    } else {
        parent::main(args)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod testkit;
