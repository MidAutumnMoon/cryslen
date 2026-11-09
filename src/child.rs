//! Child process: owns one book's source, worker, and preview store.
//!
//! Started by the parent as `cy --worker <ipc-server-name> <path>`. The
//! child never touches winit or GL; everything it learns about the UI
//! arrives as [`ParentMsg`].

use std::ffi::OsStr;
use std::ffi::OsString;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::thread;

use crossbeam_channel::bounded;
use crossbeam_channel::Receiver;
use ipc_channel::ipc;
use ipc_channel::ipc::IpcSender;
use nix::sys::prctl;
use nix::sys::signal::Signal;
use nix::unistd::getppid;

use crate::fault::PageFault;
use crate::plan::Plan;
use crate::protocol::BookInfo;
use crate::protocol::ChildMsg;
use crate::protocol::Hello;
use crate::protocol::ParentMsg;
use crate::protocol::WireImage;
use crate::source::Source;
use crate::view;
use crate::worker::Event;
use crate::worker::Worker;

/// Messages a producer may queue for the writer. Small because the
/// parent consumes at frame pace; this bounds events awaiting
/// conversion, while the want set bounds images in flight.
const OUT_QUEUE: usize = 2;

/// One message on its way to the parent, from the worker or the main
/// thread.
enum Out {
    Event(Event),
    Fatal(String),
}

impl Out {
    fn into_message(self) -> ChildMsg {
        match self {
            Self::Event(Event::Ready(ready)) => ChildMsg::Ready {
                page: ready.page,
                kind: ready.kind,
                image: WireImage::from_decoded(&ready.image),
            },
            Self::Event(Event::Failed { page, error }) => {
                ChildMsg::Failed {
                    page,
                    fault: PageFault::from(error),
                }
            }
            Self::Event(Event::Unpacked) => ChildMsg::Unpacked,
            Self::Fatal(summary) => ChildMsg::Fatal(summary),
        }
    }

    /// Whether nothing follows this message, so the writer can finish.
    fn last(&self) -> bool {
        matches!(self, Self::Fatal(_))
    }
}

pub fn main(args: &[OsString]) -> ExitCode {
    let Some((server, path)) = parse(args) else {
        tracing::error!(
            "--worker needs an ipc server name and a book path"
        );
        return ExitCode::FAILURE;
    };
    // The kernel kills this process with SIGKILL once the parent dies, so
    // a parent crash never leaves a decode running.
    if let Err(error) = prctl::set_pdeathsig(Signal::SIGKILL) {
        tracing::error!(?error, "could not arm the parent-death signal");
        return ExitCode::FAILURE;
    }
    // The parent may have died before the signal was armed, which leaves
    // nothing to open the book for.
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
    let (out_tx, out_rx) = bounded::<Out>(OUT_QUEUE);
    let writer = match thread::Builder::new()
        .name("cy-writer".into())
        .spawn(move || write(&child_tx, &out_rx))
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
    let mut fatal = false;
    while let Ok(message) = parent_rx.recv() {
        match message {
            ParentMsg::SetView {
                index,
                mode,
                direction,
            } => {
                let plan = Plan::build(
                    view::spread_for(index, mode, len),
                    direction,
                    mode,
                    len,
                );
                if let Some(running) = worker.as_ref() {
                    running.retune(plan);
                } else {
                    let events = out_tx.clone();
                    let notify = move |event| {
                        // A disconnected queue only means the writer
                        // already left, so the event has no reader.
                        let Ok(()) = events.send(Out::Event(event)) else {
                            return;
                        };
                    };
                    match Worker::spawn(Arc::clone(&source), plan, notify)
                    {
                        Ok(spawned) => worker = Some(spawned),
                        Err(error) => {
                            tracing::error!(
                                ?error,
                                "could not start the decode worker"
                            );
                            if out_tx
                                .send(Out::Fatal(error.to_string()))
                                .is_err()
                            {
                                tracing::debug!(
                                    "the writer already exited"
                                );
                            }
                            fatal = true;
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
            ParentMsg::SetWants(wants) => {
                // The parent always sends its first view before any want,
                // so a want without a worker has no plan to serve.
                if let Some(running) = worker.as_ref() {
                    running.retarget(wants);
                } else {
                    tracing::debug!("wants arrived before the first view");
                }
            }
        }
    }
    if let Some(mut running) = worker {
        // The parent is switching or quitting; waiting out an in-flight
        // decode would only delay the exit.
        running.detach();
    }
    if fatal {
        // The writer delivers the verdict before the process goes away.
        if writer.join().is_err() {
            tracing::warn!("the writer thread panicked");
        }
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// The two arguments the parent passes after `--worker`.
fn parse(args: &[OsString]) -> Option<(&OsStr, &Path)> {
    let [_program, flag, server, path] = args else {
        return None;
    };
    if flag.as_os_str() != OsStr::new("--worker") {
        return None;
    }
    Some((server.as_os_str(), Path::new(path)))
}

/// Serialize `Out` onto the parent channel until a fatal verdict ends the
/// run.
fn write(child_tx: &IpcSender<ChildMsg>, out: &Receiver<Out>) {
    while let Ok(item) = out.recv() {
        let last = item.last();
        let Ok(()) = child_tx.send(item.into_message()) else {
            return;
        };
        if last {
            return;
        }
    }
}
