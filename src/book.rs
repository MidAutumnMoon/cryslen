//! The parent's view of a book and the handle to its child process.

use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::process::Child;
use std::process::Command;
use std::process::Stdio;
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::Sender;
use ipc_channel::ipc::IpcOneShotServer;
use ipc_channel::ipc::IpcReceiver;
use ipc_channel::ipc::IpcSender;
use ipc_channel::TryRecvError;

use crate::catalog;
use crate::catalog::PageName;
use crate::fault::BookFault;
use crate::fault::PageFault;
use crate::protocol::BookInfo;
use crate::protocol::ChildMsg;
use crate::protocol::Hello;
use crate::protocol::ParentMsg;
use crate::view::PageCount;
use crate::worker;

/// A child that has not answered by now is wedged; the parent reports a
/// fault.
const OPEN_TIMEOUT: Duration = Duration::from_secs(30);

/// Tells the event loop that a [`BookEvent`] is waiting to be drained.
pub type Wake = Box<dyn Fn() + Send + Sync + 'static>;

/// The parent's complete book state; source, store, and worker live in
/// the child.
pub struct BookView {
    pub path: PathBuf,
    pub archive: bool,
    pub dir_name: Option<String>,
    pub names: Vec<String>,
    pub count: PageCount,
    pub unpacking: bool,
}

impl BookView {
    pub fn from_info(
        path: PathBuf,
        info: BookInfo,
    ) -> Result<Self, BookFault> {
        let Some(count) = PageCount::new(info.names.len()) else {
            return Err(BookFault::Open(String::from(
                "the book has no pages",
            )));
        };
        let archive = catalog::is_archive(&path);
        let dir_name = if archive {
            None
        } else {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        };
        Ok(Self {
            path,
            archive,
            dir_name,
            names: info.names,
            count,
            unpacking: info.unpacking,
        })
    }

    pub fn name(&self, index: usize) -> Option<&str> {
        self.names.get(index).map(String::as_str)
    }

    pub fn len(&self) -> PageCount {
        self.count
    }

    pub fn page_name(&self, index: usize) -> PageName {
        // The status line may name a seat resolved against an older
        // length; every index must still land on a listed page.
        let index = index.min(self.count.get().saturating_sub(1));
        catalog::page_name(
            &self.names,
            self.archive,
            self.dir_name.as_deref(),
            index,
        )
    }

    pub fn sibling(&self, forward: bool) -> Option<PathBuf> {
        catalog::sibling(&self.path, forward)
    }
}

pub enum BookEvent {
    Ready(worker::Ready),
    Failed { page: usize, fault: PageFault },
    Unpacked,
    Died(String),
}

pub struct BookProcess {
    child: Option<Child>,
    to_child: IpcSender<ParentMsg>,
    events: Receiver<BookEvent>,
    reader: Option<JoinHandle<()>>,
}

impl BookProcess {
    /// A failure reaps the child.
    pub fn open(
        path: &Path,
        wake: Wake,
    ) -> Result<(BookView, Self), BookFault> {
        PendingBookOpen::begin(path, wake)?.finish()
    }

    pub fn send(&self, message: ParentMsg) {
        // Send failures are ignored; the closed event stream reports the death.
        if let Err(error) = self.to_child.send(message) {
            tracing::debug!(?error, "the book process is gone");
        }
    }

    pub fn drain(&self) -> Vec<BookEvent> {
        self.events.try_iter().collect()
    }

    /// Kill and reap the child, then join its reader. Idempotent, so both
    /// an explicit switch and `Drop` can rely on it.
    pub fn kill(&mut self) {
        if let Some(mut child) = self.child.take() {
            kill_and_reap(&mut child);
        }
        if let Some(reader) = self.reader.take() {
            if reader.join().is_err() {
                tracing::warn!("the book reader panicked");
            }
        }
    }
}

impl Drop for BookProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

/// A forked child whose open has not finished; dropping it kills the
/// child, so an abandoned handshake cannot leak a process.
pub struct PendingBookOpen {
    path: PathBuf,
    server_name: String,
    server: IpcOneShotServer<Hello>,
    child: ForkedChild,
    wake: Wake,
}

impl PendingBookOpen {
    /// Create the server and fork the child on the calling thread.
    /// Linux delivers the child's `PR_SET_PDEATHSIG` when the thread that
    /// forked it exits, so this cannot run on a short-lived handshake thread.
    pub fn begin(path: &Path, wake: Wake) -> Result<Self, BookFault> {
        let (server, server_name) = IpcOneShotServer::<Hello>::new()
            .map_err(|error| BookFault::Spawn(error.to_string()))?;
        let exe = env::current_exe()
            .map_err(|error| BookFault::Spawn(error.to_string()))?;
        let child = Command::new(exe)
            .arg("--worker")
            .arg(&server_name)
            .arg(path)
            .stdin(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| BookFault::Spawn(error.to_string()))?;
        Ok(Self {
            path: path.to_path_buf(),
            server_name,
            server,
            child: ForkedChild(Some(child)),
            wake,
        })
    }

    /// A blocking `accept(2)` on a child that never connected cannot be
    /// woken, so it runs on its own thread under the first message's deadline.
    pub fn finish(self) -> Result<(BookView, BookProcess), BookFault> {
        let Self {
            path,
            server_name,
            server,
            mut child,
            wake,
        } = self;
        let deadline = Instant::now() + OPEN_TIMEOUT;
        let (accept_tx, accept_rx) = crossbeam_channel::bounded(1);
        let accept_thread = thread::Builder::new()
            .name(String::from("cy-book-accept"))
            .spawn(move || {
                let _sent = accept_tx.send(server.accept());
            })
            .map_err(|error| BookFault::Spawn(error.to_string()))?;
        let accepted = match accept_rx.recv_timeout(remaining(deadline)) {
            Ok(accepted) => accepted,
            Err(RecvTimeoutError::Timeout) => {
                // A child that connected but never sent leaves the accept
                // thread blocked on the read, and only its death closes
                // that socket; a child that never connected needs the
                // dummy connect below instead.
                child.kill();
                wake_accept(&server_name, accept_thread);
                return Err(BookFault::Timeout);
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(BookFault::Handshake(String::from(
                    "the book accept thread stopped",
                )))
            }
        };
        let (_, (child_rx, to_child)) = match accepted {
            Ok(accepted) => accepted,
            Err(error) => {
                return Err(BookFault::Handshake(error.to_string()))
            }
        };
        let info = match child_rx.try_recv_timeout(remaining(deadline)) {
            Ok(ChildMsg::BookInfo(info)) => info,
            Ok(ChildMsg::OpenFailed(detail)) => {
                return Err(BookFault::Open(detail))
            }
            Ok(message) => {
                return Err(BookFault::Handshake(format!(
                    "the child sent {message:?} before its book info"
                )))
            }
            Err(TryRecvError::Empty) => return Err(BookFault::Timeout),
            Err(error) => {
                return Err(BookFault::Handshake(error.to_string()))
            }
        };
        let (events, reader) = spawn_reader(child_rx, wake)?;
        // Any error below drops the process, whose `Drop` kills the child.
        let process = BookProcess {
            child: child.release(),
            to_child,
            events,
            reader: Some(reader),
        };
        let view = BookView::from_info(path, info)?;
        Ok((view, process))
    }
}

fn remaining(deadline: Instant) -> Duration {
    deadline.saturating_duration_since(Instant::now())
}

/// End the blocked `accept(2)`: one connect wakes the listener, and
/// closing the sender ends the read that follows.
fn wake_accept(server_name: &str, accept_thread: JoinHandle<()>) {
    let Ok(waker) = IpcSender::<Hello>::connect(server_name.to_owned())
    else {
        // Nothing can end a still-blocked accept, so the thread is not
        // joined.
        tracing::warn!("could not wake the book accept thread");
        return;
    };
    drop(waker);
    if accept_thread.join().is_err() {
        tracing::warn!("the book accept thread panicked");
    }
}

/// The child a [`PendingBookOpen`] owns until [`BookProcess`] takes it.
struct ForkedChild(Option<Child>);

impl ForkedChild {
    fn release(mut self) -> Option<Child> {
        self.0.take()
    }

    fn kill(&mut self) {
        if let Some(mut child) = self.0.take() {
            kill_and_reap(&mut child);
        }
    }
}

impl Drop for ForkedChild {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            kill_and_reap(&mut child);
        }
    }
}

fn kill_and_reap(child: &mut Child) {
    if let Err(error) = child.kill() {
        tracing::debug!(?error, "cannot signal the book process");
    }
    if let Err(error) = child.wait() {
        tracing::warn!(?error, "cannot reap the book process");
    }
}

/// Forward each message as a [`BookEvent`] and wake the event loop, so a
/// frame drains it.
fn spawn_reader(
    child_rx: IpcReceiver<ChildMsg>,
    wake: Wake,
) -> Result<(Receiver<BookEvent>, JoinHandle<()>), BookFault> {
    let (event_sender, events) = crossbeam_channel::unbounded();
    let handle = thread::Builder::new()
        .name(String::from("cy-book-reader"))
        .spawn(move || read_events(&child_rx, &event_sender, &*wake))
        .map_err(|error| BookFault::Spawn(error.to_string()))?;
    Ok((events, handle))
}

fn read_events(
    child_rx: &IpcReceiver<ChildMsg>,
    event_sender: &Sender<BookEvent>,
    wake: &dyn Fn(),
) {
    loop {
        let event = child_rx.recv().map_or_else(
            |_| BookEvent::Died(String::from("the book process exited")),
            event_from_message,
        );
        let last = matches!(event, BookEvent::Died(_));
        if event_sender.send(event).is_err() {
            return;
        }
        wake();
        if last {
            return;
        }
    }
}

fn event_from_message(message: ChildMsg) -> BookEvent {
    match message {
        // The handshake is read before this thread starts; a repeat is a
        // protocol violation.
        ChildMsg::BookInfo(_) | ChildMsg::OpenFailed(_) => {
            BookEvent::Died(String::from(
                "the child repeated its handshake",
            ))
        }
        ChildMsg::Ready { page, kind, image } => {
            match image.to_decoded() {
                Ok(decoded) => BookEvent::Ready(worker::Ready {
                    page,
                    kind,
                    image: Box::new(decoded),
                }),
                Err(refusal) => BookEvent::Failed {
                    page,
                    fault: PageFault::new(refusal.to_string()),
                },
            }
        }
        ChildMsg::Failed { page, fault } => {
            BookEvent::Failed { page, fault }
        }
        ChildMsg::Unpacked => BookEvent::Unpacked,
        ChildMsg::Fatal(detail) => BookEvent::Died(detail),
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;

    #[test]
    fn wake_accept_unblocks_a_waiting_listener() {
        let (server, name) =
            IpcOneShotServer::<Hello>::new().expect("server");
        let accept_thread = thread::spawn(move || {
            drop(server.accept());
        });
        wake_accept(&name, accept_thread);
    }
}
