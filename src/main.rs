//! Usage: cy <dir | archive | image-file>.

mod app;
mod book;
mod image;
mod render;
#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod testkit;

use std::process::ExitCode;

use rustfs_mimalloc::MiMalloc;

#[global_allocator]
static ALLOC: MiMalloc = MiMalloc;

fn main() -> ExitCode {
    use std::env;

    // Dropping the guard stops the non-blocking log writer, so it must
    // live for the whole process.
    let _log_guard = ino_tracing::init_tracing_subscriber();

    // env::args panics on non-UTF-8 argv, which file managers pass.
    let args: Vec<_> = env::args_os().collect();
    app::run::main(&args)
}
