//! Usage: cy <dir | file.zip | file.7z | image-file>.

mod app;
mod decode;
mod fault;
mod keys;
mod render;
mod screen;
mod source;
#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
pub(crate) mod testkit;
mod ui;
mod view;
mod window;
mod worker;

use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use app::App;
use app::UserEvent;
use source::Source;

use rustfs_mimalloc::MiMalloc;

#[global_allocator]
static ALLOC: MiMalloc = MiMalloc;

#[derive(Debug, PartialEq)]
enum Arg {
    Open(PathBuf),
    Image { dir: PathBuf, name: String },
}

#[expect(
    clippy::use_debug,
    reason = "the fatal exit prints the error's Debug on purpose: the full report with location, source chain, and backtrace status"
)]
fn main() -> ExitCode {
    use std::env;

    // Dropping the guard stops the non-blocking log writer, so it must
    // live for the whole process.
    let _log_guard = ino_tracing::init_tracing_subscriber();

    // env::args panics on non-UTF-8 argv, which file managers pass.
    let Some(arg_os) = env::args_os().nth(1) else {
        eprintln!("usage: cy <dir | archive | image-file>");
        return ExitCode::from(2);
    };
    let open = match classify(Path::new(&arg_os)) {
        Ok(open) => open,
        Err(error) => {
            eprintln!("cy: {error}");
            return ExitCode::FAILURE;
        }
    };
    let (open_path, land_on) = match &open {
        Arg::Open(path) => (path.as_path(), None),
        Arg::Image { dir, name } => (dir.as_path(), Some(name.as_str())),
    };

    match run(open_path, land_on) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // The full report: root fault, creation location, source
            // chain, backtrace status. `RUST_BACKTRACE` arms the trace.
            eprintln!("cy: {error:?}");
            ExitCode::FAILURE
        }
    }
}

fn classify(path: &Path) -> Result<Arg, fault::OpenRefusal> {
    let Some(name) = path.file_name() else {
        return Ok(Arg::Open(path.to_path_buf()));
    };
    if !path.is_file() || source::is_archive(path) {
        return Ok(Arg::Open(path.to_path_buf()));
    }
    let name = name.to_string_lossy().into_owned();
    if decode::format_from_name(&name).is_none() {
        return Err(fault::OpenRefusal::Unsupported(path.to_path_buf()));
    }
    // Path::parent() yields Some("") for bare relative names ("pic.jpg").
    let dir = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(Arg::Image {
        dir: dir.to_path_buf(),
        name,
    })
}

fn landing<'entries>(
    mut entries: impl Iterator<Item = &'entries str>,
    want: &str,
) -> usize {
    entries
        .position(|entry| entry.eq_ignore_ascii_case(want))
        .unwrap_or(0)
}

fn run(open_path: &Path, land_on: Option<&str>) -> eros::Result<()> {
    use egui_winit::winit::event_loop::EventLoop;

    let source = Arc::new(Source::open(open_path)?);

    let index = land_on.map_or(0, |name| {
        // Every page index below `len` has a name, so the indices survive
        // the filter.
        landing(
            (0..source.len().get()).filter_map(|page| source.name(page)),
            name,
        )
    });

    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let mut app = App::new(source, index, event_loop.create_proxy())?;
    event_loop.run_app(&mut app)?;
    if let Some(error) = app.take_fatal_error() {
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::TempDir;

    #[test]
    fn classify_argument_kinds() {
        // classify never reads the content, only the extension.
        let dir = TempDir::new("main-arg");
        dir.write("page.png", b"x");
        dir.write("book.cbz", b"not really a zip");
        dir.write("notes.txt", b"not a page");

        let rows = [
            (
                dir.path().join("page.png"),
                Ok(Arg::Image {
                    dir: dir.path().to_path_buf(),
                    name: String::from("page.png"),
                }),
            ),
            (
                dir.path().join("book.cbz"),
                Ok(Arg::Open(dir.path().join("book.cbz"))),
            ),
            (
                dir.path().to_path_buf(),
                Ok(Arg::Open(dir.path().to_path_buf())),
            ),
            (
                dir.path().join("notes.txt"),
                Err(fault::OpenRefusal::Unsupported(
                    dir.path().join("notes.txt"),
                )),
            ),
        ];
        for (path, want) in rows {
            assert_eq!(classify(&path), want, "{}", path.display());
        }
    }

    #[test]
    fn landing_resolves_the_image_argument() {
        let entries = ["cover.png", "Page2.png", "p3.png"];
        let names = || entries.iter().copied();
        assert_eq!(landing(names(), "Page2.png"), 1);
        assert_eq!(landing(names(), "page2.PNG"), 1);
        assert_eq!(landing(names(), "missing.png"), 0);
    }
}
