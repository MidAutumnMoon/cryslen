//! The parent's view of a book and the handle to its child process.

use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::process::Child;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::Receiver;
use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::Sender;
use crossbeam_channel::TryRecvError;
use ipc_channel::ipc::IpcOneShotServer;
use ipc_channel::ipc::IpcReceiver;
use ipc_channel::ipc::IpcSender;

use crate::animation::PlaybackEvent;
use crate::browse::PageCount;
use crate::browse::PageNames;
use crate::browse::Role;
use crate::catalog;
use crate::catalog::PageName;
use crate::fault::BookFault;
use crate::fault::PageFault;
use crate::protocol::BookInfo;
use crate::protocol::ChildMsg;
use crate::protocol::Hello;
use crate::protocol::ParentMsg;
use crate::protocol::WireImage;

/// A child that has not answered by now is wedged; the parent reports a
/// fault.
const OPEN_TIMEOUT: Duration = Duration::from_secs(30);

/// Wakes the event loop for a completed open or queued child message.
pub type Wake = Arc<dyn Fn() + Send + Sync + 'static>;

pub type OpenedBook = Result<(BookView, BookProcess), BookFault>;

/// The parent's complete book state; source, store, and worker live in
/// the child.
pub struct BookView {
    path: PathBuf,
    archive: bool,
    dir_name: Option<String>,
    names: PageNames,
}

impl BookView {
    pub fn from_info(path: PathBuf, info: BookInfo) -> Self {
        let archive = info.archive;
        let dir_name = if archive {
            None
        } else {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        };
        Self {
            path,
            archive,
            dir_name,
            names: info.names,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn name(&self, index: usize) -> Option<&str> {
        self.names.get(index).map(String::as_str)
    }

    pub fn landing(&self, name: &str) -> usize {
        catalog::landing(self.names.iter().map(String::as_str), name)
    }

    pub fn len(&self) -> PageCount {
        self.names.count()
    }

    pub fn page_name(&self, index: usize) -> PageName {
        // The status line may name a seat resolved against an older
        // length; every index must still land on a listed page.
        let index = index.min(self.names.count().get().saturating_sub(1));
        catalog::page_name(
            &self.names,
            self.archive,
            self.dir_name.as_deref(),
            index,
        )
    }

    pub fn sibling(&self, forward: bool) -> Option<PathBuf> {
        catalog::sibling(&self.path, self.archive, forward)
    }
}

/// One image delivered by the child: the page, its role, and the pixels
/// the parent uploads.
pub struct Delivery {
    pub page: usize,
    pub role: Role,
    pub image: Box<WireImage>,
}

pub enum BookUpdate {
    Ready(Delivery),
    Playback(PlaybackEvent<Box<WireImage>>),
    Failed { page: usize, fault: PageFault },
}

enum ProcessEvent {
    Update(BookUpdate),
    Died(String),
}

pub struct BookProcess {
    child: ChildGuard,
    to_child: IpcSender<ParentMsg>,
    events: Receiver<ProcessEvent>,
    reader: Option<JoinHandle<()>>,
}

impl BookProcess {
    pub fn send(&self, message: ParentMsg) {
        // Send failures are ignored; the closed event stream reports the death.
        if let Err(error) = self.to_child.send(message) {
            tracing::debug!(?error, "the book process is gone");
        }
    }

    fn drain(&self) -> (Vec<BookUpdate>, Option<String>) {
        let mut updates = Vec::new();
        let mut died = None;
        for event in self.events.try_iter() {
            match event {
                ProcessEvent::Update(update) => updates.push(update),
                ProcessEvent::Died(detail) => died = Some(detail),
            }
        }
        (updates, died)
    }

    /// Kill and reap the child, then join its reader. Idempotent, so both
    /// an explicit switch and `Drop` can rely on it.
    pub fn kill(&mut self) {
        self.child.kill();
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

pub struct BookOpen {
    results: Receiver<OpenedBook>,
}

impl BookOpen {
    /// Call from a long-lived thread: Linux sends the child's
    /// `PR_SET_PDEATHSIG` when the thread that forked it exits.
    pub fn start(path: &Path, wake: Wake) -> Result<Self, BookFault> {
        let (server, server_name) = IpcOneShotServer::<Hello>::new()
            .map_err(|error| BookFault::Spawn(error.to_string()))?;
        let binary = env::current_exe()
            .map_err(|error| BookFault::Spawn(error.to_string()))?;
        let child = Command::new(binary)
            .arg("--worker")
            .arg(&server_name)
            .arg(path)
            .stdin(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| BookFault::Spawn(error.to_string()))?;
        let child = ChildGuard(Some(child));
        let path = path.to_path_buf();
        let (result_sender, results) = crossbeam_channel::bounded(1);
        thread::Builder::new()
            .name(String::from("cy-book-open"))
            .spawn(move || {
                let opened = finish_open(
                    path,
                    &server_name,
                    server,
                    child,
                    Arc::clone(&wake),
                );
                if result_sender.send(opened).is_ok() {
                    wake();
                }
            })
            .map_err(|error| BookFault::Spawn(error.to_string()))?;
        Ok(Self { results })
    }

    pub fn try_take(&self) -> Option<OpenedBook> {
        match self.results.try_recv() {
            Ok(opened) => Some(opened),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err(BookFault::Died(
                String::from("the book open ended without a book"),
            ))),
        }
    }
}

/// Keeps [`BookView`] alive across switches and faults.
pub struct Book {
    view: BookView,
    phase: Phase,
    wake: Wake,
}

enum Phase {
    Live(BookProcess),
    Opening(BookOpen),
    Faulted(BookFault),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BookChange {
    None,
    Adopted,
    Faulted,
}

impl Book {
    pub fn new(view: BookView, process: BookProcess, wake: Wake) -> Self {
        Self {
            view,
            phase: Phase::Live(process),
            wake,
        }
    }

    pub fn view(&self) -> &BookView {
        &self.view
    }

    pub fn fault(&self) -> Option<&BookFault> {
        if let Phase::Faulted(fault) = &self.phase {
            Some(fault)
        } else {
            None
        }
    }

    pub fn send(&self, message: ParentMsg) {
        if let Phase::Live(process) = &self.phase {
            process.send(message);
        }
    }

    /// Opens a sibling book (`true` advances forward); no-op while opening.
    pub fn switch(&mut self, forward: bool) -> BookChange {
        if matches!(self.phase, Phase::Opening(_)) {
            return BookChange::None;
        }
        let Some(path) = self.view.sibling(forward) else {
            return BookChange::None;
        };
        if let Phase::Live(process) = &mut self.phase {
            process.kill();
        }
        tracing::debug!(path = %path.display(), "book switch");
        match BookOpen::start(&path, Arc::clone(&self.wake)) {
            Ok(open) => {
                self.phase = Phase::Opening(open);
                BookChange::None
            }
            Err(fault) => {
                tracing::warn!(?fault, "book open failed");
                self.phase = Phase::Faulted(fault);
                BookChange::Faulted
            }
        }
    }

    pub fn poll(&mut self) -> (Vec<BookUpdate>, BookChange) {
        match &mut self.phase {
            Phase::Live(process) => {
                let (updates, died) = process.drain();
                if let Some(detail) = died {
                    tracing::warn!(%detail, "the book process failed");
                    self.phase = Phase::Faulted(BookFault::Died(detail));
                    return (updates, BookChange::Faulted);
                }
                (updates, BookChange::None)
            }
            Phase::Opening(open) => {
                let Some(opened) = open.try_take() else {
                    return (Vec::new(), BookChange::None);
                };
                match opened {
                    Ok((view, process)) => {
                        self.view = view;
                        self.phase = Phase::Live(process);
                        (Vec::new(), BookChange::Adopted)
                    }
                    Err(fault) => {
                        tracing::warn!(?fault, "book open failed");
                        self.phase = Phase::Faulted(fault);
                        (Vec::new(), BookChange::Faulted)
                    }
                }
            }
            Phase::Faulted(_) => (Vec::new(), BookChange::None),
        }
    }
}

/// Runs `accept(2)` on `cy-book-accept` under `OPEN_TIMEOUT` because a
/// never-connecting child leaves `accept(2)` blocked forever.
fn finish_open(
    path: PathBuf,
    server_name: &str,
    server: IpcOneShotServer<Hello>,
    mut child: ChildGuard,
    wake: Wake,
) -> OpenedBook {
    let (accept_tx, accept_rx) = crossbeam_channel::bounded(1);
    let accept_thread = thread::Builder::new()
        .name(String::from("cy-book-accept"))
        .spawn(move || {
            let _sent = accept_tx.send(server.accept());
        })
        .map_err(|error| BookFault::Spawn(error.to_string()))?;
    let accepted = match accept_rx.recv_timeout(OPEN_TIMEOUT) {
        Ok(accepted) => accepted,
        Err(RecvTimeoutError::Timeout) => {
            // A child that connected but never sent leaves the accept
            // thread blocked on the read, and only its death closes that
            // socket; a child that never connected needs the dummy connect
            // below instead.
            child.kill();
            wake_accept(server_name, accept_thread);
            return Err(BookFault::Timeout);
        }
        Err(RecvTimeoutError::Disconnected) => {
            return Err(BookFault::Handshake(String::from(
                "the book accept thread stopped",
            )))
        }
    };
    let (_, hello) = match accepted {
        Ok(accepted) => accepted,
        Err(error) => {
            return Err(BookFault::Handshake(error.to_string()));
        }
    };
    let (info, messages, commands) = match hello {
        Hello::Opened {
            info,
            messages,
            commands,
        } => (info, messages, commands),
        Hello::Failed(detail) => return Err(BookFault::Open(detail)),
    };
    let view = BookView::from_info(path, info);
    let (events, reader) = spawn_reader(messages, wake)?;
    let process = BookProcess {
        child: child.release(),
        to_child: commands,
        events,
        reader: Some(reader),
    };
    Ok((view, process))
}

/// Connecting then dropping the sender unblocks `accept(2)` and ends the read.
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

struct ChildGuard(Option<Child>);

impl ChildGuard {
    fn release(mut self) -> Self {
        Self(self.0.take())
    }

    fn kill(&mut self) {
        if let Some(mut child) = self.0.take() {
            kill_and_reap(&mut child);
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        self.kill();
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

fn spawn_reader(
    child_rx: IpcReceiver<ChildMsg>,
    wake: Wake,
) -> Result<(Receiver<ProcessEvent>, JoinHandle<()>), BookFault> {
    let (event_sender, events) = crossbeam_channel::unbounded();
    let handle = thread::Builder::new()
        .name(String::from("cy-book-reader"))
        .spawn(move || read_events(&child_rx, &event_sender, &*wake))
        .map_err(|error| BookFault::Spawn(error.to_string()))?;
    Ok((events, handle))
}

fn read_events(
    child_rx: &IpcReceiver<ChildMsg>,
    event_sender: &Sender<ProcessEvent>,
    wake: &dyn Fn(),
) {
    loop {
        let event = child_rx.recv().map_or_else(
            |_| {
                ProcessEvent::Died(String::from("the book process exited"))
            },
            |message| ProcessEvent::Update(event_from_message(message)),
        );
        let last = matches!(event, ProcessEvent::Died(_));
        if event_sender.send(event).is_err() {
            return;
        }
        wake();
        if last {
            return;
        }
    }
}

fn event_from_message(message: ChildMsg) -> BookUpdate {
    match message {
        ChildMsg::Ready { page, role, image } => match image.validate() {
            Ok(()) => BookUpdate::Ready(Delivery {
                page,
                role,
                image: Box::new(image),
            }),
            Err(refusal) => BookUpdate::Failed {
                page,
                fault: PageFault::new(refusal.to_string()),
            },
        },
        ChildMsg::Playback(event) => {
            BookUpdate::Playback(validated_event(event))
        }
        ChildMsg::Failed { page, fault } => {
            BookUpdate::Failed { page, fault }
        }
    }
}

/// A malformed frame answers `Failed`.
fn validated_event(
    event: PlaybackEvent<WireImage>,
) -> PlaybackEvent<Box<WireImage>> {
    match event {
        PlaybackEvent::Frame { id, delay, image } => {
            match image.validate() {
                Ok(()) => PlaybackEvent::Frame {
                    id,
                    delay,
                    image: Box::new(image),
                },
                Err(refusal) => {
                    tracing::warn!(
                        ?id,
                        %refusal,
                        "playback frame refused"
                    );
                    PlaybackEvent::Failed { id }
                }
            }
        }
        PlaybackEvent::Finished { id } => PlaybackEvent::Finished { id },
        PlaybackEvent::Failed { id } => PlaybackEvent::Failed { id },
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
