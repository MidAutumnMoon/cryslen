//! cryslen: a wayland image viewer with manga mode.
//! The parent owns the window, the UI, and the GPU; each open book lives
//! in a child process and talks to the parent over [`protocol`].

use std::ffi::OsStr;
use std::ffi::OsString;
use std::process::ExitCode;

pub mod protocol;

mod animation;
mod app;
mod book;
mod browse;
mod cache;
mod catalog;
mod decode;
mod fault;
mod geometry;
mod render;
mod source;
mod ui;
mod worker;

/// Run the mode the command line names: `cy --worker <ipc-server-name>
/// <path>` serves one book, and anything else is the windowed parent.
pub fn run(args: &[OsString]) -> ExitCode {
    let worker_mode = args
        .get(1)
        .is_some_and(|arg| arg.as_os_str() == OsStr::new("--worker"));
    if worker_mode {
        worker::main(args)
    } else {
        app::main(args)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod testkit;
