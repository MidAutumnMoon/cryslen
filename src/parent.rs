//! Parent process: parse the command line, classify the argument, and
//! run the windowed application.

use std::ffi::OsString;
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use crate::app;
use crate::app::App;
use crate::app::WakeEvent;
use crate::book::BookProcess;
use crate::book::Wake;
use crate::catalog;
use crate::decode;
use crate::fault;

#[derive(Debug, PartialEq)]
enum Argument {
    Book(PathBuf),
    Image { dir: PathBuf, name: String },
}

#[expect(
    clippy::use_debug,
    reason = "the fatal exit prints the full error report"
)]
pub fn main(args: &[OsString]) -> ExitCode {
    let Some(argument) = args.get(1) else {
        eprintln!("usage: cy <dir | archive | image-file>");
        return ExitCode::from(2);
    };
    let open = match classify(Path::new(argument)) {
        Ok(open) => open,
        Err(error) => {
            eprintln!("cy: {error}");
            return ExitCode::FAILURE;
        }
    };
    let (open_path, land_on) = match &open {
        Argument::Book(path) => (path.as_path(), None),
        Argument::Image { dir, name } => {
            (dir.as_path(), Some(name.as_str()))
        }
    };

    match run(open_path, land_on) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // `RUST_BACKTRACE` arms the backtrace in the Debug report.
            eprintln!("cy: {error:?}");
            ExitCode::FAILURE
        }
    }
}

fn classify(path: &Path) -> Result<Argument, fault::OpenRefusal> {
    let Some(name) = path.file_name() else {
        return Ok(Argument::Book(path.to_path_buf()));
    };
    if !path.is_file() || catalog::is_archive(path) {
        return Ok(Argument::Book(path.to_path_buf()));
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
    Ok(Argument::Image {
        dir: dir.to_path_buf(),
        name,
    })
}

fn run(open_path: &Path, land_on: Option<&str>) -> eros::Result<()> {
    use egui_winit::winit::event_loop::EventLoop;

    // The loop exists before the first open, so the book's reader thread
    // can wake it like any later book's does.
    let event_loop = EventLoop::<WakeEvent>::with_user_event().build()?;
    let wake: Wake = {
        let proxy = event_loop.create_proxy();
        Arc::new(move || app::post(&proxy, WakeEvent::Wake))
    };
    // The initial open is the one blocking open; every switch after it
    // runs off the event loop.
    let (view, process) = BookProcess::open(open_path, wake)?;
    let index = land_on.map_or(0, |name| {
        // Every page index below `len` has a name, so the indices survive
        // the filter.
        catalog::landing(
            (0..view.len().get()).filter_map(|page| view.name(page)),
            name,
        )
    });

    let mut app =
        App::new(view, process, index, event_loop.create_proxy());
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
        let dir = TempDir::new("parent-arg");
        dir.write("page.png", b"x");
        dir.write("book.cbz", b"not really a zip");
        dir.write("notes.txt", b"not a page");

        let rows = [
            (
                dir.path().join("page.png"),
                Ok(Argument::Image {
                    dir: dir.path().to_path_buf(),
                    name: String::from("page.png"),
                }),
            ),
            (
                dir.path().join("book.cbz"),
                Ok(Argument::Book(dir.path().join("book.cbz"))),
            ),
            (
                dir.path().to_path_buf(),
                Ok(Argument::Book(dir.path().to_path_buf())),
            ),
            (
                dir.path().join("notes.txt"),
                Err(fault::OpenRefusal::Unsupported(
                    dir.path().join("notes.txt"),
                )),
            ),
        ];
        for (path, expected) in rows {
            assert_eq!(classify(&path), expected, "{}", path.display());
        }
    }
}
