//! Usage: cy <dir | archive | image-file>.

use std::env;
use std::process::ExitCode;

use rustfs_mimalloc::MiMalloc;

#[global_allocator]
static ALLOC: MiMalloc = MiMalloc;

fn main() -> ExitCode {
    // Dropping the guard stops the non-blocking log writer, so it must
    // live for the whole process.
    let _log_guard = ino_tracing::init_tracing_subscriber();

    let args: Vec<_> = env::args_os().collect();
    cryslen::run(&args)
}
