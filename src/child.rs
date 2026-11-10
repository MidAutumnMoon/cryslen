//! Child process: owns one book's source, worker, and preview store.
//! The parent starts it as `cy --worker`; the book is its whole state, and
//! UI requests arrive as [`ParentMsg`].

use std::ffi::OsStr;
use std::ffi::OsString;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::thread;

use crossbeam_channel::Receiver;
use ipc_channel::ipc;
use ipc_channel::ipc::IpcSender;
use nix::sys::prctl;
use nix::sys::signal::Signal;
use nix::unistd::getppid;

use crate::fault::PageFault;
use crate::prefetch::Prefetch;
use crate::protocol::BookInfo;
use crate::protocol::ChildMsg;
use crate::protocol::Hello;
use crate::protocol::ParentMsg;
use crate::protocol::WireImage;
use crate::source::Source;
use crate::geometry;
use crate::worker::Update;
use crate::worker::Worker;

/// Messages a producer may queue for the writer. Small because the
/// parent consumes at frame pace; this bounds updates awaiting
/// conversion, while the request set bounds images in flight.
const WRITER_QUEUE_DEPTH: usize = 2;

fn into_message(update: Update) -> ChildMsg {
    match update {
        Update::Ready { page, role, image } => ChildMsg::Ready {
            page,
            role,
            image: WireImage::from_decoded(&image),
        },
        Update::Failed { page, error } => ChildMsg::Failed {
            page,
            fault: PageFault::from(error),
        },
        Update::Unpacked => ChildMsg::Unpacked,
        Update::Fatal(summary) => ChildMsg::Fatal(summary),
    }
}

/// Whether this message ends the writer's run.
fn last(update: &Update) -> bool {
    matches!(update, Update::Fatal(_))
}

pub fn main(args: &[OsString]) -> ExitCode {
    let Some((server, path)) = parse(args) else {
        tracing::error!(
            "--worker needs an ipc server name and a book path"
        );
        return ExitCode::FAILURE;
    };
    // The kernel kills this process with SIGKILL once the parent dies, so a
    // parent crash ends the decode.
    if let Err(error) = prctl::set_pdeathsig(Signal::SIGKILL) {
        tracing::error!(?error, "could not arm the parent-death signal");
        return ExitCode::FAILURE;
    }
    // The parent may have died before the signal was armed; exit.
    if getppid().as_raw() == 1 {
        tracing::warn!("the parent already exited");
        return ExitCode::FAILURE;
    }

    let Some(server_name) = server.to_str() else {
        tracing::error!("the ipc server name is not utf-8");
        return ExitCode::FAILURE;
    };
    let (child_tx, child_rx) = match ipc::channel::<ChildMsg>() {
        Ok(channels) => channels,
        Err(error) => {
            tracing::error!(?error, "could not create the child channel");
            return ExitCode::FAILURE;
        }
    };
    let (parent_tx, parent_rx) = match ipc::channel::<ParentMsg>() {
        Ok(channels) => channels,
        Err(error) => {
            tracing::error!(?error, "could not create the parent channel");
            return ExitCode::FAILURE;
        }
    };
    let hello = match IpcSender::<Hello>::connect(server_name.to_owned()) {
        Ok(hello) => hello,
        Err(error) => {
            tracing::error!(?error, "could not connect to the parent");
            return ExitCode::FAILURE;
        }
    };
    if hello.send((child_rx, parent_tx)).is_err() {
        tracing::error!("could not hand the channels to the parent");
        return ExitCode::FAILURE;
    }

    let opened = match Source::open(path) {
        Ok(source) => source,
        Err(error) => {
            tracing::warn!(?error, "book open refused");
            if child_tx
                .send(ChildMsg::OpenFailed(error.to_string()))
                .is_err()
            {
                tracing::warn!("the parent left before the open verdict");
            }
            return ExitCode::FAILURE;
        }
    };
    let info = BookInfo {
        names: opened.names().to_vec(),
        unpacking: opened.unpacking(),
    };
    if child_tx.send(ChildMsg::BookInfo(info)).is_err() {
        tracing::warn!("the parent left before the book info");
        return ExitCode::FAILURE;
    }

    // One thread owns the child's only `IpcSender`, so messages reach the
    // parent in the order they were queued.
    let (to_parent_tx, to_parent_rx) =
        crossbeam_channel::bounded::<Update>(WRITER_QUEUE_DEPTH);
    let writer = match thread::Builder::new()
        .name("cy-writer".into())
        .spawn(move || write(&child_tx, &to_parent_rx))
    {
        Ok(writer) => writer,
        Err(error) => {
            tracing::error!(?error, "could not start the writer thread");
            return ExitCode::FAILURE;
        }
    };

    let len = opened.len();
    let source = Arc::new(opened);
    let mut worker: Option<Worker> = None;
    let mut worker_failed = false;
    while let Ok(message) = parent_rx.recv() {
        match message {
            ParentMsg::SetView {
                page,
                mode,
                direction,
            } => {
                let prefetch = Prefetch::build(
                    geometry::spread_for(page, mode, len),
                    direction,
                    mode,
                    len,
                );
                if let Some(running) = worker.as_ref() {
                    running.set_prefetch(prefetch);
                } else {
                    let queue = to_parent_tx.clone();
                    let notify = move |event| {
                        // A disconnected queue only means the writer
                        // already left, so the update has no reader.
                        let Ok(()) = queue.send(event) else {
                            return;
                        };
                    };
                    match Worker::spawn(Arc::clone(&source), prefetch, notify)
                    {
                        Ok(spawned) => worker = Some(spawned),
                        Err(error) => {
                            tracing::error!(
                                ?error,
                                "could not start the decode worker"
                            );
                            if to_parent_tx
                                .send(Update::Fatal(error.to_string()))
                                .is_err()
                            {
                                tracing::debug!(
                                    "the writer already exited"
                                );
                            }
                            worker_failed = true;
                            break;
                        }
                    }
                }
            }
            ParentMsg::SetMotion(motion) => {
                if let Some(running) = worker.as_ref() {
                    running.set_motion(motion);
                }
            }
            ParentMsg::SetRequests(requests) => {
                // The parent always sends its first view before any
                // request, so a request without a worker has no prefetch to
                // serve.
                if let Some(running) = worker.as_ref() {
                    running.set_requests(requests);
                } else {
                    tracing::debug!(
                        "requests arrived before the first view"
                    );
                }
            }
        }
    }
    if let Some(mut running) = worker {
        // The parent is switching or quitting; waiting out an in-flight
        // decode would only delay the exit.
        running.detach();
    }
    if worker_failed {
        // The writer delivers the verdict before the process goes away.
        if writer.join().is_err() {
            tracing::warn!("the writer thread panicked");
        }
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn parse(args: &[OsString]) -> Option<(&OsStr, &Path)> {
    let [_program, flag, server, path] = args else {
        return None;
    };
    if flag.as_os_str() != OsStr::new("--worker") {
        return None;
    }
    Some((server.as_os_str(), Path::new(path)))
}

/// Serialize updates onto the parent channel until a fatal verdict
/// ends the run.
fn write(child_tx: &IpcSender<ChildMsg>, queue: &Receiver<Update>) {
    while let Ok(update) = queue.recv() {
        let last = last(&update);
        let Ok(()) = child_tx.send(into_message(update)) else {
            return;
        };
        if last {
            return;
        }
    }
}
