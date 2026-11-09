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
/// fault instead of waiting on it forever.
const OPEN_TIMEOUT: Duration = Duration::from_secs(30);

/// Tells the event loop that a [`BookEvent`] is waiting to be drained.
/// The app supplies the closure; this module only calls it, so the book
/// layer never learns what an event loop is.
pub type Wake = Box<dyn Fn() + Send + Sync + 'static>;

/// What a book looks like from the parent: everything the UI needs, and
/// nothing that lives in the child.
pub struct BookView {
    pub path: PathBuf,
    pub archive: bool,
    pub dir_name: Option<String>,
    pub names: Vec<String>,
    pub count: PageCount,
    pub unpacking: bool,
}

impl BookView {
    /// The wire sends only what the child discovered; the count and the
    /// archive flag follow from the path the parent handed it.
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
        // Archives name pages by their inner path; a directory book's
        // pages belong to the source directory itself.
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

/// One child's message as the app consumes it. The handshake is read
/// before the reader starts, so every event here follows it.
pub enum BookEvent {
    Ready(worker::Ready),
    Failed { page: usize, fault: PageFault },
    Unpacked,
    Died(String),
}

pub struct BookProcess {
    child: Option<Child>,
    tx: IpcSender<ParentMsg>,
    events: Receiver<BookEvent>,
    reader: Option<JoinHandle<()>>,
}

impl BookProcess {
    /// Spawn the child for `path`, hand it the channels, and wait for its
    /// [`BookInfo`]. A failure leaves no process behind.
    pub fn open(
        path: &Path,
        wake: Wake,
    ) -> Result<(BookView, Self), BookFault> {
        PendingBookOpen::begin(path, wake)?.finish()
    }

    pub fn send(&self, message: ParentMsg) {
        // A dead child is reported through the event stream, not here.
        if let Err(error) = self.tx.send(message) {
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
            stop(&mut child);
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
    /// Create the server and fork the child on the calling thread, then
    /// return the state its handshake needs.
    ///
    /// The fork belongs here, not in [`Self::finish`]: Linux delivers the
    /// child's `PR_SET_PDEATHSIG` when the thread that forked it exits,
    /// so a fork made on a short-lived handshake thread would kill the
    /// child the moment that thread returned.
    pub fn begin(path: &Path, wake: Wake) -> Result<Self, BookFault> {
        let (server, server_name) = IpcOneShotServer::<Hello>::new()
            .map_err(|error| BookFault::Spawn(error.to_string()))?;
        let exe = env::current_exe()
            .map_err(|error| BookFault::Spawn(error.to_string()))?;
        // The path travels as an argument, never as a message: only the
        // channels cross the handshake.
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

    /// Run the handshake: accept the child's channels, read its book
    /// info, and start the reader. This is the slow half of an open and
    /// may run off the thread that forked the child.
    ///
    /// `accept` is a blocking `accept(2)`, and a child that never
    /// connected leaves no signal to wake it, so it runs on its own
    /// thread. The accept and the first message share one deadline.
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
        let accept = thread::Builder::new()
            .name(String::from("cy-book-accept"))
            .spawn(move || {
                let _sent = accept_tx.send(server.accept());
            })
            .map_err(|error| BookFault::Spawn(error.to_string()))?;
        let accepted = match accept_rx.recv_timeout(remaining(deadline)) {
            Ok(accepted) => accepted,
            Err(RecvTimeoutError::Timeout) => {
                // A child that connected but never sent leaves the
                // accept thread blocked on the read, and only its death
                // closes that socket; the connect then wakes a listener
                // that has no client at all.
                child.kill();
                wake_accept(&server_name, accept);
                return Err(BookFault::Timeout);
            }
            // The thread cannot end without sending, but a panic would.
            Err(RecvTimeoutError::Disconnected) => {
                return Err(BookFault::Handshake(String::from(
                    "the book accept thread stopped",
                )))
            }
        };
        let (_, (child_rx, tx)) = match accepted {
            Ok(hello) => hello,
            Err(error) => {
                return Err(BookFault::Handshake(error.to_string()))
            }
        };
        // The child's info or its refusal is the only possible first
        // message; anything else is a broken handshake.
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
            tx,
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

/// End the blocked `accept(2)`: connecting once wakes the listener, and
/// closing the sender ends the read that follows, so the thread exits.
fn wake_accept(server_name: &str, accept: JoinHandle<()>) {
    let connected = IpcSender::<Hello>::connect(server_name.to_owned());
    let Ok(dummy) = connected else {
        // The listener is already gone and the thread dies with the
        // process, so there is nothing to join.
        tracing::warn!("could not wake the book accept thread");
        return;
    };
    // Closing the sender ends the read that followed the connect.
    drop(dummy);
    if accept.join().is_err() {
        tracing::warn!("the book accept thread panicked");
    }
}

/// The child a [`PendingBookOpen`] owns until [`BookProcess`] takes it.
struct ForkedChild(Option<Child>);

impl ForkedChild {
    /// Hand the process to its permanent owner; the guard stops killing
    /// it.
    fn release(mut self) -> Option<Child> {
        self.0.take()
    }

    /// Kill and reap the child before the guard drops, so a blocked
    /// handshake can end.
    fn kill(&mut self) {
        if let Some(mut child) = self.0.take() {
            stop(&mut child);
        }
    }
}

impl Drop for ForkedChild {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            stop(&mut child);
        }
    }
}

/// Kill the child and reap it on an open's error path.
fn stop(child: &mut Child) {
    if let Err(error) = child.kill() {
        tracing::debug!(?error, "cannot signal the book process");
    }
    if let Err(error) = child.wait() {
        tracing::warn!(?error, "cannot reap the book process");
    }
}

/// The reader forwards each message as a [`BookEvent`] and wakes the
/// event loop, so a frame drains it; the receiver dies with the process,
/// so an event can never outlive its book.
fn spawn_reader(
    child_rx: IpcReceiver<ChildMsg>,
    wake: Wake,
) -> Result<(Receiver<BookEvent>, JoinHandle<()>), BookFault> {
    let (events, receiver) = crossbeam_channel::unbounded();
    let handle = thread::Builder::new()
        .name(String::from("cy-book-reader"))
        .spawn(move || read_events(&child_rx, &events, &*wake))
        .map_err(|error| BookFault::Spawn(error.to_string()))?;
    Ok((receiver, handle))
}

fn read_events(
    child_rx: &IpcReceiver<ChildMsg>,
    events: &crossbeam_channel::Sender<BookEvent>,
    wake: &dyn Fn(),
) {
    loop {
        // The app learns about a dead child from the closed stream.
        let event = child_rx.recv().map_or_else(
            |_| BookEvent::Died(String::from("the book process exited")),
            event_of,
        );
        let last = matches!(event, BookEvent::Died(_));
        // A failed send means the receiver is gone with the book, and
        // nobody is left to wake.
        if events.send(event).is_err() {
            return;
        }
        wake();
        if last {
            return;
        }
    }
}

/// Convert one wire message. A pixel payload is rebuilt into a
/// [`worker::Ready`], and a refusal becomes this page's fault.
fn event_of(message: ChildMsg) -> BookEvent {
    match message {
        // The handshake is read before this thread starts; a repeat is a
        // protocol violation, not an event.
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
        let accept = thread::spawn(move || {
            drop(server.accept());
        });
        wake_accept(&name, accept);
    }
}
