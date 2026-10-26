//! Usage: cy <dir | file.zip | file.7z | image-file>.

mod app;
mod decode;
mod keys;
mod render;
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

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use app::{App, UserEvent};
use source::Source;
use worker::Worker;

use rustfs_mimalloc::MiMalloc;

#[global_allocator]
static ALLOC: MiMalloc = MiMalloc;

/// An image file opens its parent directory at that image. A directory or
/// archive opens as itself.
enum Arg {
    Open(PathBuf),
    Image { dir: PathBuf, name: String },
}

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
            eprintln!("cy: {error}");
            ExitCode::FAILURE
        }
    }
}

/// A file argument must be an image by extension, or it is a startup
/// error.
fn classify(path: &Path) -> eros::Result<Arg> {
    // A path with no file-name component cannot be a file argument,
    // so it opens as a source.
    let Some(name) = path.file_name() else {
        return Ok(Arg::Open(path.to_path_buf()));
    };
    if !path.is_file() || source::is_archive(path) {
        return Ok(Arg::Open(path.to_path_buf()));
    }
    let name = name.to_string_lossy().into_owned();
    if decode::format_from_name(&name).is_none() {
        return Err(eros::error!("unsupported file type: {}", path.display()));
    }
    // Path::parent() yields Some("") for bare relative names ("pic.jpg").
    let dir = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(Arg::Image { dir: dir.to_path_buf(), name })
}

/// Page index to open on: the entry matching `want` case-insensitively,
/// or the cover on a miss.
fn landing<'entries>(
    mut entries: impl Iterator<Item = &'entries str>,
    want: &str,
) -> usize {
    entries.position(|entry| entry.eq_ignore_ascii_case(want)).unwrap_or(0)
}

fn run(open_path: &Path, land_on: Option<&str>) -> eros::Result<()> {
    use winit::event_loop::EventLoop;

    let source = Arc::new(Source::open(open_path)?);

    let index = land_on.map_or(0, |name| {
        // Every page index below `len` has a name, so the indices survive
        // the filter.
        landing((0..source.len()).filter_map(|page| source.name(page)), name)
    });

    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();

    let worker = Worker::spawn(Arc::clone(&source), move || {
        app::post(&proxy, UserEvent::PagesUpdated);
    })?;

    let mut app = App::new(source, worker, index, event_loop.create_proxy());
    event_loop.run_app(&mut app)?;
    if let Some(error) = app.take_fatal_error() {
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::panic, reason = "a panic in test code is inconsequential")]
mod tests {
    use super::*;
    use crate::testkit::TempDir;

    /// A path argument classifies to an image in its parent, a source, or an error.
    #[test]
    fn classify_argument_kinds() {
        // classify never reads the content, only the extension.
        let dir = TempDir::new("main-arg");
        dir.write("page.png", b"x");
        dir.write("book.cbz", b"not really a zip");
        dir.write("notes.txt", b"not a page");

        let rows = [
            // image file: the parent directory opened at that image.
            (dir.path().join("page.png"), Ok(Some("page.png"))),
            // archive file: opens as itself.
            (dir.path().join("book.cbz"), Ok(None)),
            // the directory itself: opens as itself.
            (dir.path().to_path_buf(), Ok(None)),
            // any other file: a startup error.
            (dir.path().join("notes.txt"), Err("unsupported")),
        ];
        for (path, want) in rows {
            match (classify(&path), want) {
                (Ok(Arg::Image { dir: parent, name }), Ok(Some(want_name))) => {
                    assert_eq!(parent, dir.path(), "{}: parent dir", path.display());
                    assert_eq!(name, want_name, "{}: image name", path.display());
                }
                (Ok(Arg::Open(open)), Ok(None)) => {
                    assert_eq!(open, path, "{}: source path", path.display());
                }
                (Err(error), Err(want_text)) => {
                    let message = error.to_string();
                    assert!(message.contains(want_text), "{}: {message}", path.display());
                }
                (_, want) => panic!("{}: wrong kind, wanted {want:?}", path.display()),
            }
        }
    }

    /// `landing` resolves the image argument: an exact or case-mismatched name finds its entry, a miss opens the cover.
    #[test]
    fn landing_resolves_the_image_argument() {
        let entries = ["cover.png", "Page2.png", "p3.png"];
        let names = || entries.iter().copied();
        assert_eq!(landing(names(), "Page2.png"), 1);
        assert_eq!(landing(names(), "page2.PNG"), 1);
        assert_eq!(landing(names(), "missing.png"), 0);
    }
}
