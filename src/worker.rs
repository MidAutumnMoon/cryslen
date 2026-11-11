//! Decode worker: the per-book child process. It owns the book's source,
//! decode loop, and preview store; the parent starts it as `cy --worker`
//! and talks to it over [`protocol`]. Nothing decoded survives here.

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::process::ExitCode;
use std::sync::mpsc;
use std::sync::mpsc::RecvTimeoutError;
use std::sync::Arc;
use std::thread;
use std::thread::Builder;
use std::thread::JoinHandle;
use std::time::Duration;
use std::time::Instant;

use crossbeam_channel::Receiver;
use eros::ErrorUnion;
use ipc_channel::ipc;
use ipc_channel::ipc::IpcSender;
use rustix::process::getppid;
use rustix::process::set_parent_process_death_signal;
use rustix::process::Pid;
use rustix::process::Signal;

use crate::animation::DecodedEvent;
use crate::browse::spread_for;
use crate::browse::Motion;
use crate::browse::Request;
use crate::browse::Role;
use crate::browse::Roles;
use crate::cache;
use crate::cache::prefetch::Prefetch;
use crate::cache::Store;
use crate::decode;
use crate::decode::Animation;
use crate::decode::Decoded;
use crate::decode::Frame;
use crate::decode::Loops;
use crate::fault::BadIndex;
use crate::fault::DecodeError;
use crate::fault::PageError;
use crate::fault::PageFault;
use crate::fault::PanicFault;
use crate::fault::SpawnError;
use crate::protocol::BookInfo;
use crate::protocol::ChildMsg;
use crate::protocol::Hello;
use crate::protocol::ParentMsg;
use crate::protocol::PlaybackCommand;
use crate::protocol::PlaybackEvent;
use crate::protocol::PlaybackId;
use crate::protocol::WireImage;
use crate::source::PageRead;
use crate::source::Source;

/// Nap between checks while a page unpacks; the loop drains the channel
/// after each, so a stop or request waits at most this long.
const FILL_POLL: Duration = Duration::from_millis(15);

/// Silence required before preload beyond the head starts, so a turning
/// reader leaves the worker idle between turns.
const IDLE_PRELOAD: Duration = Duration::from_millis(300);

/// One message from the decode thread.
pub enum Update {
    Ready {
        page: usize,
        role: Role,
        image: Box<Decoded>,
    },
    Failed {
        page: usize,
        error: Arc<PageError>,
    },
    Playback(DecodedEvent),
    /// Every page now reads without waiting on an unpack pass. Sent
    /// once, on the first loop iteration that observes it.
    Unpacked,
    Fatal(String),
}

enum WorkerCommand {
    SetPrefetch(Prefetch),
    Motion(Motion),
    /// Replace the set of roles the app is missing. The app sends
    /// it on change; a request leaving the set is how it says it got the
    /// role.
    Requests(Vec<Request>),
    Playback(PlaybackCommand),
    Stop,
}

/// A playback step from the parent, in arrival order.
#[derive(Clone, Copy, Debug)]
enum PlaybackStep {
    Start { id: PlaybackId, role: Role },
    Next { id: PlaybackId },
}

impl PlaybackStep {
    #[expect(
        clippy::rest_pattern_accessible_field,
        reason = "the id is all the caller needs from either step"
    )]
    fn id(self) -> PlaybackId {
        match self {
            Self::Start { id, .. } | Self::Next { id } => id,
        }
    }
}

/// The child's playback cursor for one playing page; the session owns
/// the loop policy.
struct PlaybackSession {
    id: PlaybackId,
    role: Role,
    animation: Animation,
    passes: u32,
}

impl PlaybackSession {
    /// `None` once every allowed pass has been played.
    fn advance(&mut self) -> Result<Option<Frame>, DecodeError> {
        let mut rewound = false;
        loop {
            if let Some(frame) = self.animation.next_frame().transpose()? {
                return Ok(Some(frame));
            }
            if rewound {
                return Ok(None);
            }
            self.passes = self.passes.saturating_add(1);
            if !self.should_repeat() {
                return Ok(None);
            }
            self.animation.rewind()?;
            rewound = true;
        }
    }

    fn should_repeat(&self) -> bool {
        match self.animation.loops() {
            Loops::Infinite => true,
            Loops::Finite(total) => self.passes < total,
        }
    }
}

/// A page and the roles to produce. A `requested` job is delivered; a
/// `preload` job is speculative.
#[derive(Clone, Copy, Debug)]
struct Job {
    page: usize,
    roles: Roles,
    requested: bool,
    preload: bool,
}

struct Session {
    prefetch: Prefetch,
    motion: Motion,
    store: Store,
    /// Content hash of each page's bytes, computed once per session.
    hashes: HashMap<usize, u128>,
    /// Roles this session put in the store or loaded, by content hash.
    /// Evictions clear the entry, and a failed save still counts so a
    /// refusing store cannot make the worker spin.
    stored: HashMap<u128, Roles>,
    /// A page's first failure is its verdict for the session.
    failed: HashMap<usize, Arc<PageError>>,
    /// What the app is missing, in its priority order.
    requests: Vec<Request>,
    requests_at: Instant,
    /// Requests already handed over since they last entered the set.
    served: HashSet<Request>,
    playing_pages: HashMap<usize, PlaybackSession>,
    playback_steps: VecDeque<PlaybackStep>,
    preload_allowed: bool,
}

impl Session {
    fn new(prefetch: Prefetch, store: Store) -> Self {
        Self {
            prefetch,
            motion: Motion::Still,
            store,
            hashes: HashMap::new(),
            stored: HashMap::new(),
            failed: HashMap::new(),
            requests: Vec::new(),
            requests_at: Instant::now(),
            served: HashSet::new(),
            playing_pages: HashMap::new(),
            playback_steps: VecDeque::new(),
            preload_allowed: false,
        }
    }

    /// Apply one message. `false` means the loop must stop.
    fn apply(&mut self, message: WorkerCommand) -> bool {
        // Playback is not a page turn: it must not reset the quiet window
        // that lets the prefetch tail run.
        if !matches!(
            message,
            WorkerCommand::Stop | WorkerCommand::Playback(_)
        ) {
            self.preload_allowed = false;
        }
        match message {
            WorkerCommand::SetPrefetch(prefetch) => {
                tracing::debug!(
                    origin = ?prefetch.spreads().first(),
                    motion = ?self.motion,
                    "prefetch set"
                );
                self.prefetch = prefetch;
            }
            WorkerCommand::Motion(motion) => {
                tracing::debug!(?motion, "motion");
                self.motion = motion;
            }
            WorkerCommand::Requests(requests) => {
                tracing::debug!(count = requests.len(), "requests");
                self.served.retain(|request| requests.contains(request));
                self.requests = requests;
                self.requests_at = Instant::now();
            }
            WorkerCommand::Playback(command) => {
                self.apply_playback(command);
            }
            WorkerCommand::Stop => return false,
        }
        true
    }

    fn apply_playback(&mut self, command: PlaybackCommand) {
        match command {
            PlaybackCommand::Start { id, role } => {
                // A restart replaces everything still queued for the page.
                self.playback_steps
                    .retain(|step| step.id().page != id.page);
                self.playback_steps
                    .push_back(PlaybackStep::Start { id, role });
            }
            PlaybackCommand::Next { id } => {
                self.playback_steps.push_back(PlaybackStep::Next { id });
            }
            PlaybackCommand::Stop { id } => {
                if self
                    .playing_pages
                    .get(&id.page)
                    .is_some_and(|session| session.id == id)
                {
                    self.playing_pages.remove(&id.page);
                }
                self.playback_steps.retain(|step| step.id() != id);
            }
        }
    }

    fn next_prefetch_job(&self) -> Option<Job> {
        let head = self.prefetch.head();
        for spread in self.prefetch.spreads() {
            let in_head = head.contains(spread);
            for page in
                [Some(spread.first), spread.second].into_iter().flatten()
            {
                if self.failed.contains_key(&page) {
                    continue;
                }
                let stored = self
                    .hashes
                    .get(&page)
                    .and_then(|hash| self.stored.get(hash))
                    .copied()
                    .unwrap_or_default();
                let mut roles = Roles::default();
                if !stored.contains(Role::Thumb) {
                    roles.set(Role::Thumb);
                }
                if in_head
                    && self.motion == Motion::Still
                    && !stored.contains(Role::Fit)
                {
                    roles.set(Role::Fit);
                }
                if roles.any() {
                    return Some(Job {
                        page,
                        roles,
                        requested: false,
                        preload: !in_head,
                    });
                }
            }
        }
        None
    }

    /// The first requested page with an unserved role, with the page's whole
    /// group so one decode serves it.
    fn next_request(&self) -> Option<Job> {
        let mut found: Option<Job> = None;
        for request in &self.requests {
            if self.served.contains(request)
                || self.failed.contains_key(&request.page)
            {
                continue;
            }
            match found.as_mut() {
                Some(job) if job.page == request.page => {
                    job.roles.set(request.role);
                }
                Some(_) => break,
                None => {
                    let mut roles = Roles::default();
                    roles.set(request.role);
                    found = Some(Job {
                        page: request.page,
                        roles,
                        requested: true,
                        preload: false,
                    });
                }
            }
        }
        found
    }

    fn request_wait(&self) -> Duration {
        self.requests_at.elapsed()
    }

    fn hash(&mut self, page: usize, bytes: &[u8]) -> u128 {
        *self
            .hashes
            .entry(page)
            .or_insert_with(|| cache::content_hash(bytes))
    }

    /// Hand one role to the app and mark it served for the current set.
    fn publish(
        &mut self,
        page: usize,
        role: Role,
        image: Decoded,
        notify: &dyn Fn(Update),
    ) {
        self.served.insert(Request { page, role });
        tracing::debug!(
            page,
            ?role,
            width = image.width(),
            height = image.height(),
            "publish"
        );
        notify(Update::Ready {
            page,
            role,
            image: Box::new(image),
        });
    }

    fn mark_stored(&mut self, hash: u128, role: Role) {
        self.stored.entry(hash).or_default().set(role);
    }

    /// Save one role; the evictions it reports leave the ledger.
    fn persist(
        &mut self,
        page: usize,
        hash: u128,
        role: Role,
        image: Decoded,
    ) -> Decoded {
        let saved = self.store.save(hash, role, &image);
        if !saved.persisted {
            tracing::debug!(page, ?role, "preview save missed the store");
        }
        for (evicted_role, evicted_hash) in saved.evicted {
            if let Some(roles) = self.stored.get_mut(&evicted_hash) {
                roles.clear(evicted_role);
            }
        }
        self.mark_stored(hash, role);
        image
    }

    /// Record a page's verdict and tell the app once.
    fn record_failure(
        &mut self,
        page: usize,
        error: Arc<PageError>,
        notify: &dyn Fn(Update),
    ) {
        if self.failed.insert(page, Arc::clone(&error)).is_some() {
            return;
        }
        notify(Update::Failed { page, error });
    }

    /// Load what the store answers, decode once for the rest, and save every
    /// preview; a requested job also publishes.
    fn serve(
        &mut self,
        source: &Source,
        job: Job,
        notify: &dyn Fn(Update),
    ) {
        let page = job.page;
        let roles = job.roles;
        let requested = job.requested;
        let bytes = match source.bytes(page) {
            PageRead::NotYet => {
                thread::sleep(FILL_POLL);
                return;
            }
            PageRead::Failed(error) => {
                self.record_failure(page, error, notify);
                return;
            }
            PageRead::Ready(bytes) => bytes,
        };
        let serve_started = Instant::now();
        let hash = self.hash(page, &bytes);
        let mut thumb = roles
            .contains(Role::Thumb)
            .then(|| self.store.load(hash, Role::Thumb))
            .flatten();
        let mut view = roles
            .contains(Role::Fit)
            .then(|| self.store.load(hash, Role::Fit))
            .flatten();
        let thumb_hit = thumb.is_some();
        let view_hit = view.is_some();
        if thumb.is_some() {
            self.mark_stored(hash, Role::Thumb);
        }
        if view.is_some() {
            self.mark_stored(hash, Role::Fit);
        }
        if roles.contains(Role::Full)
            || (roles.contains(Role::Fit) && view.is_none())
            || (roles.contains(Role::Thumb) && thumb.is_none())
        {
            let decode_started = Instant::now();
            let native = match decode_page(source, page, &bytes) {
                Ok(native) => native,
                Err(error) => {
                    self.record_failure(page, error, notify);
                    return;
                }
            };
            tracing::debug!(
                page,
                elapsed = ?decode_started.elapsed(),
                "decoded"
            );
            if roles.contains(Role::Fit) && view.is_none() {
                let started = Instant::now();
                let derived = cache::derive(&native, Role::Fit);
                tracing::debug!(
                    page,
                    elapsed = ?started.elapsed(),
                    "fit derived"
                );
                view = Some(self.persist(page, hash, Role::Fit, derived));
            }
            if roles.contains(Role::Thumb) && thumb.is_none() {
                let started = Instant::now();
                let from = view.as_ref().unwrap_or(&native);
                let derived = cache::derive(from, Role::Thumb);
                tracing::debug!(
                    page,
                    elapsed = ?started.elapsed(),
                    "thumb derived"
                );
                thumb =
                    Some(self.persist(page, hash, Role::Thumb, derived));
            }
            // Natives bypass the store; a requested job delivers them.
            if roles.contains(Role::Full) && requested {
                self.publish(page, Role::Full, native, notify);
            }
        }
        if requested {
            tracing::debug!(
                page,
                thumb = roles.contains(Role::Thumb),
                view = roles.contains(Role::Fit),
                native = roles.contains(Role::Full),
                thumb_hit,
                view_hit,
                waited = ?self.requests_at.elapsed(),
                elapsed = ?serve_started.elapsed(),
                "target served"
            );
            // The fit image goes first: a still frame waits on it, and the
            // app's upload budget would otherwise defer it behind the
            // thumb by a frame.
            if roles.contains(Role::Fit) {
                if let Some(view) = view {
                    self.publish(page, Role::Fit, view, notify);
                }
            }
            if roles.contains(Role::Thumb) {
                if let Some(thumb) = thumb {
                    self.publish(page, Role::Thumb, thumb, notify);
                }
            }
        }
    }

    fn serve_playback(
        &mut self,
        source: &Source,
        step: PlaybackStep,
        notify: &dyn Fn(Update),
    ) {
        match step {
            PlaybackStep::Start { id, role } => {
                self.start_playback(source, id, role, notify);
            }
            PlaybackStep::Next { id } => self.next_playback(id, notify),
        }
    }

    /// A successful start emits nothing; the first `Next` produces the
    /// first frame. A start that cannot open the stream answers `Failed`.
    fn start_playback(
        &mut self,
        source: &Source,
        id: PlaybackId,
        role: Role,
        notify: &dyn Fn(Update),
    ) {
        let page = id.page;
        let bytes = match source.bytes(page) {
            PageRead::NotYet => {
                thread::sleep(FILL_POLL);
                self.playback_steps
                    .push_front(PlaybackStep::Start { id, role });
                return;
            }
            PageRead::Failed(_) => {
                tracing::debug!(?id, "playback source failed");
                self.fail_playback(id, notify);
                return;
            }
            PageRead::Ready(bytes) => bytes,
        };
        let Some(name) = source.name(page) else {
            self.fail_playback(id, notify);
            return;
        };

        self.playing_pages.remove(&page);
        let opened = panic::catch_unwind(AssertUnwindSafe(|| {
            decode::open_animation(Arc::clone(&bytes), name)
        }));
        match opened {
            Ok(Ok(Some(animation))) => {
                let loops = animation.loops();
                self.playing_pages.insert(
                    page,
                    PlaybackSession {
                        id,
                        role,
                        animation,
                        passes: 0,
                    },
                );
                tracing::debug!(?id, ?role, ?loops, "playback start");
            }
            Ok(Ok(None)) => {
                tracing::debug!(?id, ?role, "no playback path");
                self.fail_playback(id, notify);
            }
            Ok(Err(error)) => {
                tracing::warn!(
                    ?id,
                    ?role,
                    %error,
                    "playback open failed"
                );
                self.fail_playback(id, notify);
            }
            Err(payload) => {
                let fault = PanicFault::from_payload(
                    "animation",
                    payload.as_ref(),
                );
                tracing::warn!(
                    ?id,
                    ?role,
                    fault = %fault,
                    "playback open panicked"
                );
                self.fail_playback(id, notify);
            }
        }
    }

    fn next_playback(&mut self, id: PlaybackId, notify: &dyn Fn(Update)) {
        let page = id.page;
        let Some(session) = self.playing_pages.get_mut(&page) else {
            tracing::debug!(?id, "playback frame without a session");
            return;
        };
        if session.id != id {
            tracing::debug!(?id, "stale playback frame");
            return;
        }
        let role = session.role;
        let produced =
            panic::catch_unwind(AssertUnwindSafe(|| session.advance()));
        match produced {
            Ok(Ok(Some(frame))) => {
                let image = cache::derive(&frame.image, role);
                tracing::debug!(
                    ?id,
                    ?role,
                    delay = ?frame.delay,
                    width = image.width(),
                    height = image.height(),
                    "playback frame"
                );
                notify(Update::Playback(DecodedEvent::Frame {
                    id,
                    delay: frame.delay,
                    image: Box::new(image),
                }));
            }
            Ok(Ok(None)) => {
                tracing::debug!(?id, ?role, "playback finished");
                self.playing_pages.remove(&page);
                notify(Update::Playback(DecodedEvent::Finished { id }));
            }
            Ok(Err(error)) => {
                tracing::warn!(
                    ?id,
                    ?role,
                    %error,
                    "playback frame failed"
                );
                self.fail_playback(id, notify);
            }
            Err(payload) => {
                let fault = PanicFault::from_payload(
                    "animation",
                    payload.as_ref(),
                );
                tracing::warn!(
                    ?id,
                    ?role,
                    fault = %fault,
                    "playback frame panicked"
                );
                self.fail_playback(id, notify);
            }
        }
    }

    fn fail_playback(&mut self, id: PlaybackId, notify: &dyn Fn(Update)) {
        if self
            .playing_pages
            .get(&id.page)
            .is_some_and(|session| session.id == id)
        {
            self.playing_pages.remove(&id.page);
        }
        self.playback_steps.retain(|step| step.id() != id);
        notify(Update::Playback(DecodedEvent::Failed { id }));
    }
}

/// Handle to the decode thread. Every control is a send; the last
/// prefetch wins, and a newer prefetch overtakes the old one's tail.
pub struct Worker {
    tx: mpsc::Sender<WorkerCommand>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    /// Spawn the worker for `source` with `initial` as its prefetch. `notify`
    /// fires from the worker thread with each role or failure.
    pub fn spawn(
        source: Arc<Source>,
        initial: Prefetch,
        notify: impl Fn(Update) + Send + Sync + 'static,
    ) -> Result<Self, SpawnError> {
        Self::launch(source, initial, Store::open(), notify)
    }

    /// Spawn against an explicit store root for tests.
    #[cfg(test)]
    fn spawn_at(
        source: Arc<Source>,
        initial: Prefetch,
        root: &Path,
        notify: impl Fn(Update) + Send + Sync + 'static,
    ) -> Result<Self, SpawnError> {
        Self::launch(source, initial, Store::at(root), notify)
    }

    fn launch(
        source: Arc<Source>,
        initial: Prefetch,
        store: Store,
        notify: impl Fn(Update) + Send + Sync + 'static,
    ) -> Result<Self, SpawnError> {
        let (tx, rx) = mpsc::channel::<WorkerCommand>();
        let handle =
            Builder::new().name("cy-decode".into()).spawn(move || {
                run(&source, &rx, Session::new(initial, store), &notify);
            })?;
        Ok(Self {
            tx,
            handle: Some(handle),
        })
    }

    /// A failed send means the thread already exited, so the message has
    /// no reader left.
    fn tell(&self, message: WorkerCommand) {
        let Ok(()) = self.tx.send(message) else {
            return;
        };
    }

    /// Replace the prefetch; the in-flight decode still finishes.
    pub fn set_prefetch(&self, prefetch: Prefetch) {
        self.tell(WorkerCommand::SetPrefetch(prefetch));
    }

    pub fn set_motion(&self, motion: Motion) {
        self.tell(WorkerCommand::Motion(motion));
    }

    /// Serve these requests in order, ahead of the prefetch; the app sends only
    /// changes, and a request that leaves and returns is served again.
    pub fn set_requests(&self, requests: Vec<Request>) {
        self.tell(WorkerCommand::Requests(requests));
    }

    pub fn playback(&self, command: PlaybackCommand) {
        self.tell(WorkerCommand::Playback(command));
    }

    /// Stop the worker without joining it, so the caller returns while an
    /// in-flight decode finishes. The thread owns its state.
    pub fn detach(&mut self) {
        self.tell(WorkerCommand::Stop);
        self.handle = None;
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Stop wakes a parked loop so the join returns.
        self.tell(WorkerCommand::Stop);
        if let Some(handle) = self.handle.take() {
            // A panicked thread leaves nothing to act on at teardown, so
            // debug builds surface the breach.
            let joined = handle.join();
            debug_assert!(joined.is_ok(), "worker thread panicked");
        }
    }
}

/// Worker loop: drain the channel before every pick, so Stop and a newer
/// prefetch answer between decodes, and one in-flight decode completes.
fn run(
    source: &Source,
    rx: &mpsc::Receiver<WorkerCommand>,
    mut session: Session,
    notify: &dyn Fn(Update),
) {
    // An archive pass that settled before this thread started is still
    // announced; a directory has no pass to announce.
    let mut unpack_pending = source.unpacks();
    loop {
        if unpack_pending && !source.unpacking() {
            unpack_pending = false;
            notify(Update::Unpacked);
        }
        while let Ok(message) = rx.try_recv() {
            if !session.apply(message) {
                return;
            }
        }
        if let Some(job) = session.next_request() {
            tracing::debug!(
                page = job.page,
                roles = ?job.roles,
                waited = ?session.request_wait(),
                "request"
            );
            session.serve(source, job, notify);
            continue;
        }
        if let Some(step) = session.playback_steps.pop_front() {
            session.serve_playback(source, step, notify);
            continue;
        }
        if let Some(job) = session.next_prefetch_job() {
            tracing::debug!(
                page = job.page,
                thumb = job.roles.contains(Role::Thumb),
                fit = job.roles.contains(Role::Fit),
                motion = ?session.motion,
                preload = job.preload,
                "job"
            );
            if !job.preload || session.preload_allowed {
                session.serve(source, job, notify);
                continue;
            }
            // Preload yields to the app. Let the app speak first: a turning
            // reader gets its own pages ahead of speculative work.
            match rx.recv_timeout(IDLE_PRELOAD) {
                Ok(message) => {
                    if !session.apply(message) {
                        return;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    session.preload_allowed = true;
                    session.serve(source, job, notify);
                }
                Err(RecvTimeoutError::Disconnected) => return,
            }
        } else {
            let Ok(message) = rx.recv() else {
                return;
            };
            if !session.apply(message) {
                return;
            }
        }
    }
}

/// Decode one page behind the panic boundary, so a panicking decoder
/// cannot kill the only worker thread.
fn decode_page(
    source: &Source,
    page: usize,
    bytes: &[u8],
) -> Result<Decoded, Arc<PageError>> {
    let Some(name) = source.name(page) else {
        // `Source::bytes` answers every page under its length, so a name
        // miss is a caller bug.
        return Err(Arc::new(ErrorUnion::new(BadIndex)));
    };
    match panic::catch_unwind(AssertUnwindSafe(|| {
        decode::decode(bytes, name)
    })) {
        Ok(Ok(decoded)) => Ok(decoded),
        Ok(Err(error)) => Err(Arc::new(error.widen())),
        Err(payload) => Err(Arc::new(ErrorUnion::new(
            PanicFault::from_payload("decoder", payload.as_ref()),
        ))),
    }
}

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
        Update::Playback(event) => ChildMsg::Playback(wire_event(event)),
        Update::Unpacked => ChildMsg::Unpacked,
        Update::Fatal(summary) => ChildMsg::Fatal(summary),
    }
}

fn wire_event(event: DecodedEvent) -> PlaybackEvent<WireImage> {
    event.map_image(|image| WireImage::from_decoded(&image))
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
    if let Err(error) = set_parent_process_death_signal(Some(Signal::KILL))
    {
        tracing::error!(?error, "could not arm the parent-death signal");
        return ExitCode::FAILURE;
    }
    // The parent may have died before the signal was armed; exit.
    if getppid() == Some(Pid::INIT) {
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
                    spread_for(page, mode, len),
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
                    match Worker::spawn(
                        Arc::clone(&source),
                        prefetch,
                        notify,
                    ) {
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
            ParentMsg::Playback(command) => {
                if let Some(running) = worker.as_ref() {
                    running.playback(command);
                } else {
                    tracing::debug!(
                        ?command,
                        "playback arrived before the first view"
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

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;
    use crate::browse::Direction;
    use crate::browse::PageCount;
    use crate::browse::ReadingMode;
    use crate::browse::Spread;
    use crate::cache::Store;
    use crate::fault::DecodeRefusal;
    use crate::testkit::eventually;
    use crate::testkit::tiny_animated_gif;
    use crate::testkit::tiny_png;
    use crate::testkit::TempDir;
    use image::codecs::gif::Repeat;

    use parking_lot::Mutex;

    /// Page edge for the tests that check derived and full dimensions:
    /// larger than the thumb edge, smaller than the fit edge, so the
    /// thumb is scaled and the fit image is the page itself.
    const BIG: u32 = 1400;

    fn prefetch_for(origin: usize, count: usize) -> Prefetch {
        Prefetch::build(
            Spread::lone(origin),
            Direction::Forward,
            ReadingMode::Normal,
            PageCount::new(count).expect("test book is non-empty"),
        )
    }

    fn start_playback(
        worker: &Worker,
        page: usize,
        role: Role,
        generation: u32,
    ) {
        worker.playback(PlaybackCommand::Start {
            id: PlaybackId { page, generation },
            role,
        });
    }

    fn next_frame(worker: &Worker, page: usize, generation: u32) {
        worker.playback(PlaybackCommand::Next {
            id: PlaybackId { page, generation },
        });
    }

    /// A directory of distinct solid pages, `size` pixels square.
    fn page_dir(
        tag: &str,
        count: usize,
        size: u32,
    ) -> (TempDir, Arc<Source>) {
        let dir = TempDir::new(tag);
        for page in 0..count {
            let shade = u8::try_from(page % 256).expect("shade fits");
            let tint = u8::try_from(page >> 8).expect("tint fits");
            dir.write(
                &format!("p{page:04}.png"),
                &tiny_png(size, size, [shade, tint, 90, 255]),
            );
        }
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        (dir, source)
    }

    /// The worker's content hash for one page of a directory source.
    fn page_hash(source: &Source, page: usize) -> u128 {
        match source.bytes(page) {
            PageRead::Ready(bytes) => cache::content_hash(&bytes),
            PageRead::NotYet | PageRead::Failed(_) => {
                panic!("page {page} did not read")
            }
        }
    }

    fn long_edge(image: &Decoded) -> u32 {
        image.width().max(image.height())
    }

    fn first_pixel(image: &Decoded) -> [u8; 4] {
        let decode::Pixels::Rgba8(bytes) = image.pixels() else {
            panic!("a gif frame decodes to rgba");
        };
        let mut pixel = [0; 4];
        pixel.copy_from_slice(bytes.get(..4).expect("one pixel"));
        pixel
    }

    fn is_thumb(role: Role) -> bool {
        role == Role::Thumb
    }

    fn is_fit(role: Role) -> bool {
        role == Role::Fit
    }

    fn request_fit(page: usize) -> Request {
        Request {
            page,
            role: Role::Fit,
        }
    }

    fn request_thumb(page: usize) -> Request {
        Request {
            page,
            role: Role::Thumb,
        }
    }

    fn request_full(page: usize) -> Request {
        Request {
            page,
            role: Role::Full,
        }
    }

    /// The events the worker delivered, kept so several assertions can
    /// read the same stream.
    struct Updates {
        rx: mpsc::Receiver<Update>,
        seen: Mutex<Vec<Update>>,
    }

    #[expect(
        clippy::rest_pattern_accessible_field,
        reason = "the ignored fields are unused; naming them would trip unneeded_field_pattern"
    )]
    impl Updates {
        fn new() -> (Self, impl Fn(Update) + Send + Sync + 'static) {
            let (tx, rx) = mpsc::channel();
            let events = Self {
                rx,
                seen: Mutex::new(Vec::new()),
            };
            (events, move |event| {
                // A dead receiver means the test is over; the worker
                // carries on.
                let Ok(()) = tx.send(event) else {
                    return;
                };
            })
        }

        fn drain(&self) {
            while let Ok(event) = self.rx.try_recv() {
                self.seen.lock().push(event);
            }
        }

        fn ready(
            &self,
            what: &str,
            want: impl Fn(usize, Role) -> bool,
        ) -> Decoded {
            eventually(20, what, || {
                self.drain();
                let seen = self.seen.lock();
                seen.iter().find_map(|event| match event {
                    Update::Ready { page, role, image }
                        if want(*page, *role) =>
                    {
                        Some((**image).clone())
                    }
                    Update::Ready { .. }
                    | Update::Failed { .. }
                    | Update::Playback(_)
                    | Update::Unpacked
                    | Update::Fatal(_) => None,
                })
            })
        }

        fn failure(&self, page: usize) -> Arc<PageError> {
            eventually(20, "page never failed", || {
                self.drain();
                let seen = self.seen.lock();
                seen.iter().find_map(|event| match event {
                    Update::Failed {
                        page: failed,
                        error,
                    } if *failed == page => Some(Arc::clone(error)),
                    Update::Ready { .. }
                    | Update::Failed { .. }
                    | Update::Playback(_)
                    | Update::Unpacked
                    | Update::Fatal(_) => None,
                })
            })
        }

        fn delivered(
            &self,
            page: usize,
            want: impl Fn(Role) -> bool,
        ) -> usize {
            self.drain();
            let seen = self.seen.lock();
            seen.iter()
                .filter(|event| {
                    matches!(event, Update::Ready { page: delivered, role, .. }
                        if *delivered == page && want(*role))
                })
                .count()
        }

        fn failures(&self, page: usize) -> usize {
            self.drain();
            let seen = self.seen.lock();
            seen.iter()
                .filter_map(|event| match event {
                    Update::Failed {
                        page: failed,
                        error,
                    } if *failed == page => Some(error),
                    Update::Ready { .. }
                    | Update::Failed { .. }
                    | Update::Playback(_)
                    | Update::Unpacked
                    | Update::Fatal(_) => None,
                })
                .count()
        }

        /// The `offset`-th frame delivered for `page`, counting from
        /// zero in arrival order.
        fn frame(
            &self,
            page: usize,
            offset: usize,
        ) -> (Duration, Decoded) {
            eventually(20, "no animation frame", || {
                self.drain();
                let seen = self.seen.lock();
                seen.iter()
                    .filter_map(|event| match event {
                        Update::Playback(DecodedEvent::Frame {
                            id,
                            delay,
                            image,
                        }) if id.page == page => {
                            Some((*delay, (**image).clone()))
                        }
                        Update::Ready { .. }
                        | Update::Failed { .. }
                        | Update::Playback(_)
                        | Update::Unpacked
                        | Update::Fatal(_) => None,
                    })
                    .nth(offset)
            })
        }

        fn finished(&self, page: usize) {
            eventually(20, "no animation finish", || {
                self.drain();
                let seen = self.seen.lock();
                seen.iter()
                    .any(|event| {
                        matches!(
                            event,
                            Update::Playback(DecodedEvent::Finished {
                                id,
                            }) if id.page == page
                        )
                    })
                    .then_some(())
            });
        }
    }

    #[test]
    fn a_target_is_served_and_persisted() {
        let (_dir, source) = page_dir("worker-target", 1, BIG);
        let root = TempDir::new("worker-target-store");
        let (events, notify) = Updates::new();
        let worker = Worker::spawn_at(
            Arc::clone(&source),
            prefetch_for(0, 1),
            root.path(),
            notify,
        )
        .expect("worker spawns");

        worker.set_requests(vec![request_fit(0), request_thumb(0)]);
        let thumb = events
            .ready("the target never delivered a thumb", |page, role| {
                page == 0 && is_thumb(role)
            });
        let view = events.ready(
            "the target never delivered a fit image",
            |page, role| page == 0 && is_fit(role),
        );
        assert_eq!(long_edge(&thumb), cache::THUMB_EDGE);
        assert_eq!(
            long_edge(&view),
            cache::FIT_EDGE.min(BIG),
            "the fit image's long edge matches the target edge"
        );

        let store = Store::at(root.path());
        let hash = page_hash(&source, 0);
        assert!(
            store.load(hash, Role::Thumb).is_some(),
            "the thumb was not persisted"
        );
        assert!(
            store.load(hash, Role::Fit).is_some(),
            "the fit image was not persisted"
        );
    }

    #[test]
    fn the_tail_waits_for_a_quiet_app() {
        let (_dir, source) = page_dir("worker-gate", 6, 8);
        let root = TempDir::new("worker-gate-store");
        let _worker = Worker::spawn_at(
            Arc::clone(&source),
            prefetch_for(0, 6),
            root.path(),
            |_| {},
        )
        .expect("worker spawns");
        let store = Store::at(root.path());
        // The head is pages 0, 1 and 2; wait until it is fully produced.
        for page in [0, 1, 2] {
            eventually(20, "the head never reached the store", || {
                store
                    .load(page_hash(&source, page), Role::Thumb)
                    .is_some()
                    .then_some(())
            });
        }
        let head_done = Instant::now();
        // Page 3 is beyond the head: it must not start until the app has
        // been silent for the preload window.
        eventually(20, "the tail never preloaded", || {
            store
                .load(page_hash(&source, 3), Role::Thumb)
                .is_some()
                .then_some(())
        });
        let waited = head_done.elapsed();
        assert!(
            waited
                >= IDLE_PRELOAD.saturating_sub(Duration::from_millis(50)),
            "the tail started after only {waited:?}"
        );
    }

    #[test]
    fn a_failed_page_notifies_once_and_the_prefetch_advances() {
        let dir = TempDir::new("worker-fail");
        dir.write("p000.png", b"not a png at all");
        for page in 1..5_u8 {
            dir.write(
                &format!("p{page:03}.png"),
                &tiny_png(BIG, BIG, [page, 20, 30, 255]),
            );
        }
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let root = TempDir::new("worker-fail-store");
        let (events, notify) = Updates::new();
        let _worker = Worker::spawn_at(
            Arc::clone(&source),
            prefetch_for(0, 5),
            root.path(),
            notify,
        )
        .expect("worker spawns");

        let error = events.failure(0);
        assert!(
            error.downcast_inner_ref::<image::ImageError>().is_some()
                || error.downcast_inner_ref::<DecodeRefusal>().is_some(),
            "page 0 failed as {error}, expected a decode fault"
        );
        // The failure leaves the rest of the prefetch running: pages behind it
        // still reach the store.
        let store = Store::at(root.path());
        eventually(20, "the failure stalled the prefetch", || {
            store
                .load(page_hash(&source, 4), Role::Thumb)
                .is_some()
                .then_some(())
        });
        assert_eq!(
            events.failures(0),
            1,
            "the failure was announced more than once"
        );
    }

    #[test]
    fn the_prefetch_warms_the_store_without_delivering() {
        let (_dir, source) = page_dir("worker-prefetch", 1, BIG);
        let root = TempDir::new("worker-prefetch-store");
        let (events, notify) = Updates::new();
        let _worker = Worker::spawn_at(
            Arc::clone(&source),
            prefetch_for(0, 1),
            root.path(),
            notify,
        )
        .expect("worker spawns");

        let store = Store::at(root.path());
        eventually(
            20,
            "the prefetch never produced the head fit image",
            || {
                store
                    .load(page_hash(&source, 0), Role::Fit)
                    .is_some()
                    .then_some(())
            },
        );
        events.drain();
        assert_eq!(
            events.delivered(0, |_| true),
            0,
            "the prefetch delivered a role nobody requested"
        );
    }

    #[test]
    fn a_fit_leaving_the_request_set_is_served_again() {
        let (_dir, source) = page_dir("worker-recover", 1, BIG);
        let root = TempDir::new("worker-recover-store");
        let (events, notify) = Updates::new();
        let worker = Worker::spawn_at(
            Arc::clone(&source),
            prefetch_for(0, 1),
            root.path(),
            notify,
        )
        .expect("worker spawns");

        worker.set_requests(vec![request_fit(0)]);
        events.ready(
            "the target never delivered a fit image",
            |page, role| page == 0 && is_fit(role),
        );
        // An unchanged set is the app still waiting for its own upload.
        worker.set_requests(vec![request_fit(0)]);
        thread::sleep(Duration::from_millis(50));
        assert_eq!(
            events.delivered(0, is_fit),
            1,
            "an unchanged request set fetched the fit image twice"
        );
        // The app uploaded it, so it left the set; asking again later
        // must fetch it again.
        worker.set_requests(Vec::new());
        worker.set_requests(vec![request_fit(0)]);
        eventually(20, "the re-added fit image was never served", || {
            (events.delivered(0, is_fit) >= 2).then_some(())
        });
    }

    #[test]
    fn a_full_target_carries_the_page_pixels() {
        let (_dir, source) = page_dir("worker-full", 1, BIG);
        let root = TempDir::new("worker-full-store");
        let (events, notify) = Updates::new();
        let worker = Worker::spawn_at(
            Arc::clone(&source),
            prefetch_for(0, 1),
            root.path(),
            notify,
        )
        .expect("worker spawns");

        worker.set_requests(vec![request_full(0)]);
        let image = events
            .ready("the full target was never honoured", |page, role| {
                page == 0 && role == Role::Full
            });
        assert_eq!(
            (image.width(), image.height()),
            (BIG, BIG),
            "full must carry the page's own pixels"
        );
    }

    #[test]
    fn animation_frames_stream_in_order() {
        let colors =
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
        let dir = TempDir::new("worker-animation");
        dir.write(
            "p000.gif",
            &tiny_animated_gif(8, 8, &colors, 50, Repeat::Infinite),
        );
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let root = TempDir::new("worker-animation-store");
        let (events, notify) = Updates::new();
        let worker = Worker::spawn_at(
            Arc::clone(&source),
            prefetch_for(0, 1),
            root.path(),
            notify,
        )
        .expect("worker spawns");

        start_playback(&worker, 0, Role::Fit, 9);
        for (offset, color) in colors.into_iter().enumerate() {
            next_frame(&worker, 0, 9);
            let (delay, image) = events.frame(0, offset);
            assert_eq!(delay, Duration::from_millis(50));
            assert_eq!((image.width(), image.height()), (8, 8));
            assert_eq!(first_pixel(&image), color);
        }
        // The fourth frame crosses the pass boundary and repeats the
        // first, without ending the playback.
        next_frame(&worker, 0, 9);
        let (_, image) = events.frame(0, 3);
        assert_eq!(first_pixel(&image), colors[0]);
    }

    #[test]
    fn a_finite_animation_finishes_after_its_last_pass() {
        let colors =
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
        let dir = TempDir::new("worker-animation-finite");
        dir.write(
            "p000.gif",
            &tiny_animated_gif(8, 8, &colors, 50, Repeat::Finite(0)),
        );
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let root = TempDir::new("worker-animation-finite-store");
        let (events, notify) = Updates::new();
        let worker = Worker::spawn_at(
            Arc::clone(&source),
            prefetch_for(0, 1),
            root.path(),
            notify,
        )
        .expect("worker spawns");

        start_playback(&worker, 0, Role::Fit, 3);
        for (offset, color) in colors.into_iter().enumerate() {
            next_frame(&worker, 0, 3);
            let (_, image) = events.frame(0, offset);
            assert_eq!(first_pixel(&image), color);
        }
        next_frame(&worker, 0, 3);
        events.finished(0);
    }

    #[test]
    fn a_warm_store_serves_without_decoding() {
        let (_dir, source) = page_dir("worker-warm", 1, BIG);
        let root = TempDir::new("worker-warm-store");
        let store_root = root.path();

        // The first session derives and persists both roles.
        let (first, notify) = Updates::new();
        let worker = Worker::spawn_at(
            Arc::clone(&source),
            prefetch_for(0, 1),
            store_root,
            notify,
        )
        .expect("worker spawns");
        worker.set_requests(vec![request_thumb(0), request_fit(0)]);
        let thumb = first.ready(
            "the first session never delivered a thumb",
            |page, role| page == 0 && is_thumb(role),
        );
        let view = first.ready(
            "the first session never delivered a fit image",
            |page, role| page == 0 && is_fit(role),
        );
        assert_eq!(long_edge(&thumb), cache::THUMB_EDGE);
        assert_eq!(
            long_edge(&view),
            cache::FIT_EDGE.min(BIG),
            "the fit image's long edge matches the target edge"
        );
        drop(worker);

        let hash = page_hash(&source, 0);
        let mut store = Store::at(store_root);
        assert!(
            store.load(hash, Role::Thumb).is_some(),
            "the thumb was not persisted"
        );
        assert!(
            store.load(hash, Role::Fit).is_some(),
            "the fit image was not persisted"
        );

        // Replace both persisted roles with a size no decode of this
        // page could produce, so the second session's events say where
        // the pixels came from.
        let sentinel = decode::decode(
            &tiny_png(7, 7, [1, 2, 3, 255]),
            "sentinel.png",
        )
        .expect("sentinel decodes");
        assert!(store.save(hash, Role::Thumb, &sentinel).persisted);
        assert!(store.save(hash, Role::Fit, &sentinel).persisted);
        drop(store);

        // A second worker on the same root loads the cached roles.
        let (second, notify_second) = Updates::new();
        let warm = Worker::spawn_at(
            Arc::clone(&source),
            prefetch_for(0, 1),
            store_root,
            notify_second,
        )
        .expect("worker spawns");
        warm.set_requests(vec![request_thumb(0), request_fit(0)]);
        let warm_thumb = second.ready(
            "the second session never delivered a thumb",
            |page, role| page == 0 && is_thumb(role),
        );
        let warm_view = second.ready(
            "the second session never delivered a fit image",
            |page, role| page == 0 && is_fit(role),
        );
        assert_eq!(
            (warm_thumb.width(), warm_thumb.height()),
            (7, 7),
            "the thumb missed the warm store"
        );
        assert_eq!(
            (warm_view.width(), warm_view.height()),
            (7, 7),
            "the fit image missed the warm store"
        );

        // Loading leaves the roles it read unchanged.
        let kept = Store::at(store_root)
            .load(hash, Role::Fit)
            .expect("the sentinel fit stays");
        assert_eq!((kept.width(), kept.height()), (7, 7));
    }
}
