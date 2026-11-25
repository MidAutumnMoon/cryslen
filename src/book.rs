use std::fmt::Display;
use std::path::Path;
use std::path::PathBuf;
use std::process::Child;
use std::sync::Arc;
use std::thread::Builder;
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use eros::ErrorUnion;
use ipc_channel::ipc::IpcOneShotServer;
use ipc_channel::ipc::IpcReceiver;
use ipc_channel::ipc::IpcSender;

use crate::browse::Role;
use crate::protocol::BookInfo;
use crate::protocol::ChildMsg;
use crate::protocol::Hello;
use crate::protocol::ParentMsg;
use crate::protocol::PlaybackEvent;
use crate::protocol::ValidatedImage;
use crate::protocol::WireImage;

/// A child that has not answered by now is wedged; the parent reports a
/// fault.
const OPEN_TIMEOUT: Duration = Duration::from_secs(30);

pub enum PageEvent {
    Ready {
        page: usize,
        role: Role,
        image: Box<ValidatedImage>,
    },
    Playback(PlaybackEvent<Box<ValidatedImage>>),
    Failed {
        page: usize,
        fault: String,
    },
}

pub enum BookEvent {
    Page(PageEvent),
    Died(String),
}

pub struct BookProcess {
    child: ChildGuard,
    to_child: IpcSender<ParentMsg>,
    events: Receiver<BookEvent>,
    reader: Option<JoinHandle<()>>,
}

impl BookProcess {
    pub fn send(&self, message: ParentMsg) {
        // Send failures are ignored; the closed event stream reports the death.
        if let Err(error) = self.to_child.send(message) {
            tracing::debug!(?error, "the book process is gone");
        }
    }

    pub fn drain(&self) -> impl Iterator<Item = BookEvent> + '_ {
        self.events.try_iter()
    }

    pub fn kill(&mut self) {
        self.child.kill();
        if let Some(reader) = self.reader.take()
            && reader.join().is_err()
        {
            tracing::warn!("the book reader panicked");
        }
    }
}

impl Drop for BookProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

pub struct BookOpen {
    results: Receiver<eros::Result<(PathBuf, BookInfo, BookProcess)>>,
}

fn spawn_error(error: impl Display) -> ErrorUnion {
    eros::error!("cannot start the book process: {error}")
}

impl BookOpen {
    /// Call from a long-lived thread: Linux sends the child's
    /// `PR_SET_PDEATHSIG` when the thread that forked it exits.
    pub fn start(
        path: &Path,
        wake: impl Fn() + Send + Sync + 'static,
    ) -> eros::Result<Self> {
        use std::env;
        use std::process::Command;
        use std::process::Stdio;

        let wake: Arc<dyn Fn() + Send + Sync + 'static> = Arc::new(wake);
        let (server, server_name) =
            IpcOneShotServer::<Hello>::new().map_err(spawn_error)?;
        let binary = env::current_exe().map_err(spawn_error)?;
        let child = Command::new(binary)
            .arg("--worker")
            .arg(&server_name)
            .arg(path)
            .stdin(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(spawn_error)?;
        let child = ChildGuard(Some(child));
        let path = path.to_path_buf();
        let (result_sender, results) = crossbeam_channel::bounded(1);
        Builder::new()
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
            .map_err(spawn_error)?;
        Ok(Self { results })
    }

    pub fn try_take(
        &self,
    ) -> Option<eros::Result<(PathBuf, BookInfo, BookProcess)>> {
        use crossbeam_channel::TryRecvError;

        match self.results.try_recv() {
            Ok(opened) => Some(opened),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err(eros::error!(
                "the book process died: the book open ended without a book"
            ))),
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
    wake: Arc<dyn Fn() + Send + Sync + 'static>,
) -> eros::Result<(PathBuf, BookInfo, BookProcess)> {
    use crossbeam_channel::RecvTimeoutError;

    let (accept_tx, accept_rx) = crossbeam_channel::bounded(1);
    let accept_thread = Builder::new()
        .name(String::from("cy-book-accept"))
        .spawn(move || {
            let _sent = accept_tx.send(server.accept());
        })
        .map_err(spawn_error)?;
    let accepted = match accept_rx.recv_timeout(OPEN_TIMEOUT) {
        Ok(accepted) => accepted,
        Err(RecvTimeoutError::Timeout) => {
            // A child that connected but never sent leaves the accept
            // thread blocked on the read, and only its death closes that
            // socket; a child that never connected needs the dummy connect
            // below instead.
            child.kill();
            wake_accept(server_name, accept_thread);
            eros::bail!("the book process did not answer in time");
        }
        Err(RecvTimeoutError::Disconnected) => {
            eros::bail!(
                "book process handshake failed: the book accept thread stopped"
            );
        }
    };
    let (_, hello) = match accepted {
        Ok(accepted) => accepted,
        Err(error) => {
            eros::bail!("book process handshake failed: {error}");
        }
    };
    let (info, messages, commands) = match hello {
        Hello::Opened {
            info,
            messages,
            commands,
        } => (info, messages, commands),
        Hello::Failed(detail) => {
            eros::bail!("cannot open the book: {detail}");
        }
    };
    let (events, reader) = spawn_reader(messages, wake)?;
    let process = BookProcess {
        child,
        to_child: commands,
        events,
        reader: Some(reader),
    };
    Ok((path, info, process))
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
    match child.try_wait() {
        Ok(Some(status)) => {
            tracing::warn!(%status, "the book process already exited");
            return;
        }
        Ok(None) => {}
        Err(error) => {
            tracing::debug!(?error, "cannot poll the book process");
        }
    }
    if let Err(error) = child.kill() {
        tracing::debug!(?error, "cannot signal the book process");
    }
    if let Err(error) = child.wait() {
        tracing::warn!(?error, "cannot reap the book process");
    }
}

fn spawn_reader(
    child_rx: IpcReceiver<ChildMsg>,
    wake: Arc<dyn Fn() + Send + Sync + 'static>,
) -> eros::Result<(Receiver<BookEvent>, JoinHandle<()>)> {
    let (event_sender, events) = crossbeam_channel::unbounded();
    let handle = Builder::new()
        .name(String::from("cy-book-reader"))
        .spawn(move || read_events(&child_rx, &event_sender, &*wake))
        .map_err(spawn_error)?;
    Ok((events, handle))
}

fn read_events(
    child_rx: &IpcReceiver<ChildMsg>,
    event_sender: &Sender<BookEvent>,
    wake: &dyn Fn(),
) {
    use ipc_channel::IpcError;

    loop {
        let event = match child_rx.recv() {
            Ok(message) => BookEvent::Page(event_from_message(message)),
            Err(IpcError::Disconnected) => {
                BookEvent::Died(String::from("the book process exited"))
            }
            Err(
                error @ (IpcError::Io(_) | IpcError::SerializationError(_)),
            ) => {
                tracing::warn!(%error, "book ipc receive failed");
                BookEvent::Died(format!(
                    "the book process IPC failed: {error}"
                ))
            }
        };
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

fn event_from_message(message: ChildMsg) -> PageEvent {
    match message {
        ChildMsg::Ready { page, role, image } => match image.validate() {
            Ok(image) => PageEvent::Ready {
                page,
                role,
                image: Box::new(image),
            },
            Err(refusal) => PageEvent::Failed {
                page,
                fault: refusal.to_string(),
            },
        },
        ChildMsg::Playback(event) => {
            PageEvent::Playback(validated_event(event))
        }
        ChildMsg::Failed { page, fault } => {
            PageEvent::Failed { page, fault }
        }
    }
}

fn validated_event(
    event: PlaybackEvent<WireImage>,
) -> PlaybackEvent<Box<ValidatedImage>> {
    match event {
        PlaybackEvent::Frame { id, delay, image } => {
            match image.validate() {
                Ok(image) => PlaybackEvent::Frame {
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
        use std::thread;

        let (server, name) =
            IpcOneShotServer::<Hello>::new().expect("server");
        let accept_thread = thread::spawn(move || {
            drop(server.accept());
        });
        wake_accept(&name, accept_thread);
    }
}
