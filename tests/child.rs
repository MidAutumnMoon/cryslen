//! End-to-end check of the child IPC protocol: the real `cy --worker`
//! process opens a book, decodes a page, and hands the pixels back
//! through the shared wire types.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
#![expect(
    clippy::tests_outside_test_module,
    reason = "an integration test file is already test-only"
)]

use std::env;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::path::PathBuf;
use std::process;
use std::process::Child;
use std::process::Command;
use std::process::ExitStatus;
use std::process::Stdio;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::mpsc::RecvTimeoutError;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use cryslen::protocol::Artifact;
use cryslen::protocol::ChildMsg;
use cryslen::protocol::Direction;
use cryslen::protocol::Format;
use cryslen::protocol::Hello;
use cryslen::protocol::ParentMsg;
use cryslen::protocol::ViewMode;
use cryslen::protocol::Want;
use cryslen::protocol::WirePixels;
use ipc_channel::ipc::IpcOneShotServer;

/// The page is one solid color, so the decoded pixels prove a real
/// decode.
const PIXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];

/// The page's edge. Small enough that the thumb derivation is the page
/// itself.
const PAGE_EDGE: u32 = 2;

/// RGBA bytes in the page: four channels per pixel.
const PAGE_BYTES: usize = (PAGE_EDGE * PAGE_EDGE * 4) as usize;

/// Generous bound on each protocol step: a healthy child answers in
/// milliseconds, but a cold process start can be slow.
const STEP_TIMEOUT: Duration = Duration::from_secs(30);

/// How often the handshake wait rechecks the child process.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

static ROOTS: AtomicU64 = AtomicU64::new(0);

/// A scratch root unique within this test process, removed on drop.
struct TempRoot(PathBuf);

impl TempRoot {
    fn new() -> Self {
        let path = env::temp_dir().join(format!(
            "cryslen-child-{}-{}",
            process::id(),
            ROOTS.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir_all(&path).expect("create the test's scratch root");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        // Removal is best-effort; a leaked scratch dir is acceptable.
        let Ok(()) = fs::remove_dir_all(&self.0) else {
            return;
        };
    }
}

/// Kills and reaps the child however the test ends, including panics.
struct ChildGuard(Option<Child>);

impl ChildGuard {
    fn new(child: Child) -> Self {
        Self(Some(child))
    }

    /// The exit status once the child is gone, without blocking.
    fn exited(&mut self) -> Option<ExitStatus> {
        let child = self.0.as_mut()?;
        child.try_wait().expect("poll the child process")
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let Some(mut child) = self.0.take() else {
            return;
        };
        if let Err(error) = child.kill() {
            eprintln!("kill the child: {error}");
        }
        if let Err(error) = child.wait() {
            eprintln!("reap the child: {error}");
        }
    }
}

/// A solid `PAGE_EDGE`-square RGBA page the test can recognize after the
/// round trip.
fn write_page(dir: &Path) {
    let page = image::RgbaImage::from_pixel(
        PAGE_EDGE,
        PAGE_EDGE,
        image::Rgba(PIXEL),
    );
    let mut encoded = Cursor::new(Vec::<u8>::new());
    image::DynamicImage::ImageRgba8(page)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .expect("encode the test page");
    fs::write(dir.join("page.png"), encoded.into_inner())
        .expect("write the test page");
}

#[test]
fn child_serves_a_decoded_thumb() {
    let binary = env!("CARGO_BIN_EXE_cy");

    // `root` outlives the child guard below, so the child is dead before
    // its scratch dirs go away.
    let root = TempRoot::new();
    let book = root.path().join("book");
    let store = root.path().join("store");
    fs::create_dir_all(&book).expect("create the book directory");
    fs::create_dir_all(&store).expect("create the child's tmpdir");
    write_page(&book);

    let (server, name) = IpcOneShotServer::<Hello>::new()
        .expect("start a one-shot ipc server");

    // The child's own temp store is private to this test, so no other
    // run can serve the page from the shared cache.
    let child = Command::new(binary)
        .arg("--worker")
        .arg(&name)
        .arg(&book)
        .env("TMPDIR", &store)
        .env("RUST_LOG", "off")
        .stdin(Stdio::null())
        .spawn()
        .unwrap_or_else(|error| {
            panic!("spawn {binary} --worker: {error}")
        });
    let mut guard = ChildGuard::new(child);

    // `accept` blocks until a client connects, so it runs beside a wait
    // that notices a child that died before the handshake.
    let (accepted_tx, accepted_rx) = mpsc::channel();
    let accept_thread = thread::spawn(move || {
        let _sent = accepted_tx.send(server.accept());
    });

    let started = Instant::now();
    let hello = loop {
        match accepted_rx.recv_timeout(POLL_INTERVAL) {
            Ok(Ok((_, hello))) => break hello,
            Ok(Err(error)) => {
                panic!("the child failed the handshake: {error}");
            }
            Err(RecvTimeoutError::Disconnected) => {
                panic!("the accept thread ended without a handshake");
            }
            Err(RecvTimeoutError::Timeout) => {
                if let Some(status) = guard.exited() {
                    panic!(
                        "{binary} exited before the handshake: {status}"
                    );
                }
                assert!(
                    started.elapsed() <= STEP_TIMEOUT,
                    "{binary} did not connect within {STEP_TIMEOUT:?}",
                );
            }
        }
    };
    accept_thread
        .join()
        .expect("the accept thread finished before the handshake");

    let (messages, commands) = hello;

    let info = match messages.try_recv_timeout(STEP_TIMEOUT) {
        Ok(ChildMsg::BookInfo(info)) => info,
        Ok(message) => panic!("expected BookInfo first, got {message:?}"),
        Err(error) => panic!("no BookInfo from the child: {error}"),
    };
    assert_eq!(
        info.names.len(),
        1,
        "the child lists the one page that was written",
    );
    assert_eq!(info.names, ["page.png"], "the child lists the page");
    assert!(!info.unpacking, "a directory never unpacks");

    commands
        .send(ParentMsg::SetView {
            index: 0,
            mode: ViewMode::Normal,
            direction: Direction::Forward,
        })
        .expect("send SetView");
    commands
        .send(ParentMsg::SetWants(vec![Want::new(0, Artifact::Thumb)]))
        .expect("send SetWants");

    let image = match messages.try_recv_timeout(STEP_TIMEOUT) {
        Ok(ChildMsg::Ready {
            page: 0,
            kind: Artifact::Thumb,
            image,
        }) => image,
        Ok(message) => {
            panic!("expected a page 0 thumb, got {message:?}")
        }
        Err(error) => panic!("no Ready from the child: {error}"),
    };

    assert_eq!(
        (image.width, image.height),
        (PAGE_EDGE, PAGE_EDGE),
        "the thumb is the page's own size",
    );
    assert_eq!(
        image.page_dims,
        (PAGE_EDGE, PAGE_EDGE),
        "the thumb reports the page's dimensions",
    );
    assert_eq!(
        image.format,
        Format::Png,
        "the child preserved the page's format",
    );
    let WirePixels::Rgba8(bytes) = &image.pixels else {
        panic!("a png page arrived as yuv");
    };
    assert_eq!(
        bytes.len(),
        PAGE_BYTES,
        "the thumb holds one rgba per pixel",
    );
    let expected = [PIXEL; 4].concat();
    assert_eq!(
        bytes.to_vec(),
        expected,
        "the thumb holds the page's color",
    );
}
