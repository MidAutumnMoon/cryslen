//! cryslen (`cy`) — minimal wayland image/comic pager.
//! Usage: cy <dir | file.zip | file.7z | image-file>
//! A bare image file opens its parent directory at that page.

mod app;
mod decode;
mod page;
mod render;
mod ui;
mod view;
mod worker;

use std::path::Path;
use std::sync::Arc;

use app::{App, UserEvent};
use page::Source;
use worker::Worker;

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: cy <dir | archive | image-file>");
        std::process::exit(2);
    });
    let path = Path::new(&arg);

    let start = if path.is_file() && !is_archive(path) {
        // Path::parent() yields Some("") for bare relative names ("pic.jpg").
        let dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
        Some((dir.to_path_buf(), name))
    } else {
        None
    };
    let open_path = start.as_ref().map_or(path, |(dir, _)| dir.as_path());

    let source = match Source::open(open_path) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("cy: {e}");
            std::process::exit(1);
        }
    };

    // Page to land on: the named file if given, else the first page.
    let index = start
        .as_ref()
        .and_then(|(_, name)| name.as_ref())
        .and_then(|name| (0..source.len()).find(|&i| source.name(i).eq_ignore_ascii_case(name)))
        .unwrap_or(0);

    let event_loop =
        winit::event_loop::EventLoop::<UserEvent>::with_user_event().build().unwrap();
    let proxy = event_loop.create_proxy();

    let worker = Worker::spawn(Arc::clone(&source), move || {
        let _ = proxy.send_event(UserEvent::PagesUpdated);
    });
    worker.focus(index);

    let mut app = App::new(source, worker, index, event_loop.create_proxy());
    event_loop.run_app(&mut app).expect("event loop");
}

fn is_archive(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else { return false };
    let ext = ext.to_ascii_lowercase();
    matches!(ext.as_str(), "zip" | "cbz" | "7z" | "cb7")
}
