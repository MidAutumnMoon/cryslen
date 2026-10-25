//! Usage: cy <dir | file.zip | file.7z | image-file>
//! A bare image file opens its parent directory at that page.

mod app;
mod decode;
mod keys;
mod render;
mod source;
#[cfg(test)]
#[expect(clippy::expect_used, reason = "a panic in test code is inconsequential")]
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

/// Parsed path argument. An image file opens its parent directory at that
/// image. A directory or archive opens as itself.
enum Arg {
    Open(PathBuf),
    Image { dir: PathBuf, name: String },
}

fn main() -> ExitCode {
    use std::env;

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

/// Directories and archives open as sources. A file argument must be an
/// image by extension, or it is a startup error.
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

fn run(open_path: &Path, land_on: Option<&str>) -> eros::Result<()> {
    use winit::event_loop::EventLoop;

    let source = Arc::new(Source::open(open_path)?);

    let index = land_on
        .and_then(|name| {
            (0..source.len()).find(|&page_index| {
                source.name(page_index).is_some_and(|entry| entry.eq_ignore_ascii_case(name))
            })
        })
        .unwrap_or(0);

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
#[expect(clippy::expect_used, clippy::panic, reason = "a panic in test code is inconsequential")]
mod tests {
    use super::*;
    use crate::testkit::{tiny_png, TempDir};

    #[test]
    fn image_file_argument_opens_parent_dir() {
        let dir = TempDir::new("main-arg-img");
        dir.write("page.png", &tiny_png(1, 1, [0, 0, 0, 255]));
        match classify(&dir.path().join("page.png")).expect("image file accepted") {
            Arg::Image { dir: parent, name } => {
                assert_eq!(parent, dir.path());
                assert_eq!(name, "page.png");
            }
            Arg::Open(_) => panic!("image file classified as a source path"),
        }
    }

    #[test]
    fn directory_and_archive_arguments_open_as_sources() {
        let dir = TempDir::new("main-arg-src");
        dir.write("book.cbz", b"not really a zip");
        assert!(matches!(classify(&dir.path().join("book.cbz")), Ok(Arg::Open(_))));
        assert!(matches!(classify(dir.path()), Ok(Arg::Open(_))));
    }

    #[test]
    fn other_file_arguments_error() {
        let dir = TempDir::new("main-arg-bad");
        dir.write("notes.txt", b"not a page");
        let Err(error) = classify(&dir.path().join("notes.txt")) else {
            panic!("txt file accepted as an image or source");
        };
        assert!(error.to_string().contains("unsupported"), "{error}");
    }
}
