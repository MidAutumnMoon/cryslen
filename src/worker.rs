//! Per-book decode worker (`cy --worker`): serves requests and
//! playback in the foreground while `cy-prefetch` warms the store.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use eros::ErrorUnion;
use ipc_channel::ipc;
use ipc_channel::ipc::IpcReceiver;
use ipc_channel::ipc::IpcSender;
use ipc_channel::TryRecvError as IpcTryRecvError;
use parking_lot::Condvar;
use parking_lot::Mutex;
use rustix::process;
use rustix::process::Pid;
use rustix::process::Signal;

use crate::browse;
use crate::browse::Motion;
use crate::browse::PageCount;
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
use crate::protocol::BookInfo;
use crate::protocol::ChildMsg;
use crate::protocol::Hello;
use crate::protocol::ParentMsg;
use crate::protocol::PlaybackCommand;
use crate::protocol::PlaybackEvent;
use crate::protocol::PlaybackId;
use crate::protocol::WireImage;
use crate::source::Source;

/// Quiet window before prefetch preloads pages beyond the head.
const IDLE_PRELOAD: Duration = Duration::from_millis(300);

enum ChildEvent<'image> {
    Ready {
        page: usize,
        role: Role,
        image: &'image Decoded,
    },
    Failed {
        page: usize,
        error: &'image Arc<PageError>,
    },
    Playback(PlaybackEvent<&'image Decoded>),
}

enum CommandPoll {
    Message(ParentMsg),
    Empty,
    Disconnected,
}

trait CommandReceiver {
    fn try_recv(&self) -> CommandPoll;
    fn recv(&self) -> Option<ParentMsg>;
}

impl CommandReceiver for IpcReceiver<ParentMsg> {
    fn try_recv(&self) -> CommandPoll {
        match Self::try_recv(self) {
            Ok(message) => CommandPoll::Message(message),
            Err(IpcTryRecvError::Empty) => CommandPoll::Empty,
            Err(IpcTryRecvError::IpcError(_)) => CommandPoll::Disconnected,
        }
    }

    fn recv(&self) -> Option<ParentMsg> {
        Self::recv(self).ok()
    }
}

trait EventSink {
    fn emit(&self, event: ChildEvent<'_>) -> bool;
}

impl EventSink for IpcSender<ChildMsg> {
    fn emit(&self, event: ChildEvent<'_>) -> bool {
        let message = match event {
            ChildEvent::Ready { page, role, image } => ChildMsg::Ready {
                page,
                role,
                image: WireImage::from_decoded(image),
            },
            ChildEvent::Failed { page, error } => ChildMsg::Failed {
                page,
                fault: PageFault::from(Arc::clone(error)),
            },
            ChildEvent::Playback(event) => ChildMsg::Playback(
                event.map_image(WireImage::from_decoded),
            ),
        };
        self.send(message).is_ok()
    }
}

#[derive(Clone, Copy, Debug)]
enum PlaybackStep {
    Start { id: PlaybackId, role: Role },
    Next { id: PlaybackId },
}

impl PlaybackStep {
    fn id(self) -> PlaybackId {
        match self {
            Self::Start { id, .. } | Self::Next { id } => id,
        }
    }
}

struct PlaybackSession {
    id: PlaybackId,
    role: Role,
    animation: Animation,
    passes: u32,
}

impl PlaybackSession {
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

#[derive(Clone, Copy, Debug)]
struct PrefetchStep {
    page: usize,
    role: Role,
    preload: bool,
}

#[derive(Clone, Copy)]
enum ServeKind {
    Request { requested_at: Instant },
    Prefetch { preload: bool },
}

impl ServeKind {
    const fn lane(self) -> &'static str {
        match self {
            Self::Request { .. } => "request",
            Self::Prefetch { .. } => "prefetch",
        }
    }

    const fn is_request(self) -> bool {
        matches!(self, Self::Request { .. })
    }

    fn report(
        self,
        page: usize,
        role: Role,
        hit: bool,
        elapsed: Duration,
    ) {
        match self {
            Self::Request { requested_at } => tracing::debug!(
                page,
                ?role,
                hit,
                waited = ?requested_at.elapsed(),
                ?elapsed,
                "target served"
            ),
            Self::Prefetch { preload } => tracing::debug!(
                page,
                ?role,
                preload,
                hit,
                ?elapsed,
                "prefetch served"
            ),
        }
    }
}

struct BookState {
    page_count: PageCount,
    prefetch: Option<Prefetch>,
    motion: Motion,
    /// Content hash of each page's bytes, computed once per session.
    hashes: HashMap<usize, u128>,
    /// Roles known to reside in the store, by content hash. Evictions
    /// clear entries, and failed saves stay marked so a refusing store
    /// cannot make prefetch spin.
    stored: HashMap<u128, Roles>,
    /// A page's first failure is its verdict for the session.
    failed: HashMap<usize, Arc<PageError>>,
    /// The request foreground is currently decoding, so prefetch skips it
    /// even if a new `SetRequests` drops it mid-decode.
    active_request: Option<Request>,
    requests: Vec<Request>,
    /// Requests already handed over since they last entered the set.
    served: HashSet<Request>,
    preload_allowed: bool,
    activity_sequence: u64,
    stopped: bool,
}

impl BookState {
    fn new(page_count: PageCount) -> Self {
        Self {
            page_count,
            prefetch: None,
            motion: Motion::Still,
            hashes: HashMap::new(),
            stored: HashMap::new(),
            failed: HashMap::new(),
            active_request: None,
            requests: Vec::new(),
            served: HashSet::new(),
            preload_allowed: false,
            activity_sequence: 0,
            stopped: false,
        }
    }

    fn apply(
        &mut self,
        message: ParentMsg,
        playback: &mut PlaybackRunner,
    ) -> bool {
        // Ignore playback so animation frames keep the preload quiet window.
        if !matches!(message, ParentMsg::Playback(_)) {
            self.preload_allowed = false;
            self.activity_sequence =
                self.activity_sequence.wrapping_add(1);
        }
        match message {
            ParentMsg::SetView {
                page,
                mode,
                direction,
            } => {
                let prefetch = Prefetch::build(
                    browse::spread_for(page, mode, self.page_count),
                    direction,
                    mode,
                    self.page_count,
                );
                tracing::debug!(
                    origin = ?prefetch.spreads().first(),
                    motion = ?self.motion,
                    "prefetch set"
                );
                self.prefetch = Some(prefetch);
                true
            }
            ParentMsg::SetMotion(motion) => {
                tracing::debug!(?motion, "motion");
                self.motion = motion;
                true
            }
            ParentMsg::SetRequests(requests) => {
                tracing::debug!(count = requests.len(), "requests");
                self.served.retain(|request| requests.contains(request));
                self.requests = requests;
                true
            }
            ParentMsg::Playback(command) => {
                playback.apply(command);
                false
            }
        }
    }

    fn take_request(&mut self) -> Option<Request> {
        let request = *self.requests.iter().find(|request| {
            !self.served.contains(request)
                && !self.failed.contains_key(&request.page)
        })?;
        self.active_request = Some(request);
        Some(request)
    }

    fn wants_unserved(&self, page: usize, role: Role) -> bool {
        let request = Request::new(page, role);
        self.requests.contains(&request) && !self.served.contains(&request)
    }

    fn is_claimed(&self, page: usize, role: Role) -> bool {
        let request = Request::new(page, role);
        self.active_request == Some(request)
            || (self.requests.contains(&request)
                && !self.served.contains(&request))
    }

    fn next_prefetch_job(&self) -> Option<(PrefetchStep, Motion)> {
        let prefetch = self.prefetch.as_ref()?;
        let head = prefetch.head();
        for spread in prefetch.spreads() {
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
                if in_head
                    && self.motion == Motion::Still
                    && !stored.contains(Role::Fit)
                    && !self.is_claimed(page, Role::Fit)
                {
                    return Some((
                        PrefetchStep {
                            page,
                            role: Role::Fit,
                            preload: false,
                        },
                        self.motion,
                    ));
                }
                // Let a pending fit decode finish before warming thumb so
                // thumb can derive from the saved fit without re-decoding.
                let fit_pending = !stored.contains(Role::Fit)
                    && self.is_claimed(page, Role::Fit);
                if !stored.contains(Role::Thumb)
                    && !fit_pending
                    && !self.is_claimed(page, Role::Thumb)
                {
                    return Some((
                        PrefetchStep {
                            page,
                            role: Role::Thumb,
                            preload: !in_head,
                        },
                        self.motion,
                    ));
                }
            }
        }
        None
    }

    fn mark_stored(&mut self, hash: u128, role: Role) {
        self.stored.entry(hash).or_default().set(role);
    }
}

struct RequestGuard<'shared, Sink> {
    shared: &'shared Shared<Sink>,
}

impl<Sink> Drop for RequestGuard<'_, Sink> {
    fn drop(&mut self) {
        self.shared.finish_request();
    }
}

struct Shared<Sink> {
    store: Store,
    state: Mutex<BookState>,
    wake: Condvar,
    sink: Mutex<Sink>,
}

impl<Sink> Shared<Sink> {
    fn finish_request(&self) {
        let mut state = self.state.lock();
        state.active_request = None;
        drop(state);
        self.wake.notify_one();
    }
}

impl<Sink> Shared<Sink>
where
    Sink: EventSink,
{
    fn new(page_count: PageCount, store: Store, sink: Sink) -> Self {
        Self {
            store,
            state: Mutex::new(BookState::new(page_count)),
            wake: Condvar::new(),
            sink: Mutex::new(sink),
        }
    }

    fn stop(&self) {
        let mut state = self.state.lock();
        state.stopped = true;
        drop(state);
        self.wake.notify_all();
    }

    fn is_stopped(&self) -> bool {
        self.state.lock().stopped
    }

    fn apply_message(
        &self,
        message: ParentMsg,
        playback: &mut PlaybackRunner,
        requested_at: &mut Instant,
    ) {
        if matches!(message, ParentMsg::SetRequests(_)) {
            *requested_at = Instant::now();
        }
        let wake_prefetch = self.state.lock().apply(message, playback);
        if wake_prefetch {
            self.wake.notify_one();
        }
    }

    fn apply_messages(
        &self,
        rx: &impl CommandReceiver,
        playback: &mut PlaybackRunner,
        requested_at: &mut Instant,
    ) -> bool {
        loop {
            match rx.try_recv() {
                CommandPoll::Message(message) => {
                    self.apply_message(message, playback, requested_at);
                }
                CommandPoll::Empty => return true,
                CommandPoll::Disconnected => return false,
            }
        }
    }

    fn next_request(&self) -> Option<(Request, RequestGuard<'_, Sink>)> {
        let request = self.state.lock().take_request()?;
        Some((request, RequestGuard { shared: self }))
    }

    #[expect(
        clippy::significant_drop_tightening,
        reason = "condvar wait loop holds the mutex guard across iterations"
    )]
    fn next_prefetch(&self) -> Option<PrefetchStep> {
        let mut preload_wait: Option<(Instant, Instant, u64)> = None;
        let (step, motion) = {
            let mut state = self.state.lock();
            loop {
                if state.stopped {
                    return None;
                }
                let Some((step, motion)) = state.next_prefetch_job()
                else {
                    preload_wait = None;
                    self.wake.wait(&mut state);
                    continue;
                };
                if !step.preload || state.preload_allowed {
                    break (step, motion);
                }
                let sequence = state.activity_sequence;
                let (started, deadline) = match preload_wait {
                    Some((started, deadline, wait_sequence))
                        if wait_sequence == sequence =>
                    {
                        (started, deadline)
                    }
                    _ => {
                        let started = Instant::now();
                        let deadline = started + IDLE_PRELOAD;
                        preload_wait = Some((started, deadline, sequence));
                        (started, deadline)
                    }
                };
                let timed_out =
                    self.wake.wait_until(&mut state, deadline).timed_out();
                if timed_out
                    && !state.stopped
                    && state.activity_sequence == sequence
                {
                    state.preload_allowed = true;
                    tracing::debug!(
                        waited = ?started.elapsed(),
                        "preload started"
                    );
                }
            }
        };
        tracing::debug!(
            page = step.page,
            role = ?step.role,
            ?motion,
            preload = step.preload,
            "prefetch job"
        );
        Some(step)
    }

    fn hash_of(&self, page: usize) -> Option<u128> {
        self.state.lock().hashes.get(&page).copied()
    }

    fn insert_hash(&self, page: usize, bytes: &[u8]) -> u128 {
        let hash = cache::content_hash(bytes);
        *self.state.lock().hashes.entry(page).or_insert(hash)
    }

    fn load_cached(&self, hash: u128, role: Role) -> Option<Decoded> {
        let image = self.store.load(hash, role)?;
        self.state.lock().mark_stored(hash, role);
        Some(image)
    }

    fn persist(
        &self,
        page: usize,
        hash: u128,
        role: Role,
        image: &Decoded,
    ) {
        let saved = self.store.save(hash, role, image);
        if !saved.persisted {
            tracing::debug!(page, ?role, "preview not persisted");
        }
        let mut state = self.state.lock();
        for (evicted_role, evicted_hash) in saved.evicted {
            if let Some(roles) = state.stored.get_mut(&evicted_hash) {
                roles.clear(evicted_role);
            }
        }
        state.mark_stored(hash, role);
    }

    fn wants_unserved(&self, page: usize, role: Role) -> bool {
        self.state.lock().wants_unserved(page, role)
    }

    fn emit(&self, event: ChildEvent<'_>) {
        let sent = self.sink.lock().emit(event);
        if !sent {
            let mut state = self.state.lock();
            state.stopped = true;
            drop(state);
            self.wake.notify_all();
        }
    }

    fn publish(&self, page: usize, role: Role, image: &Decoded) {
        self.state.lock().served.insert(Request::new(page, role));
        tracing::debug!(
            page,
            ?role,
            width = image.width(),
            height = image.height(),
            "publish"
        );
        self.emit(ChildEvent::Ready { page, role, image });
    }

    fn record_failure(&self, page: usize, error: &Arc<PageError>) {
        let is_new = match self.state.lock().failed.entry(page) {
            Entry::Vacant(entry) => {
                entry.insert(Arc::clone(error));
                true
            }
            Entry::Occupied(_) => false,
        };
        if is_new {
            self.emit(ChildEvent::Failed { page, error });
        }
    }

    fn read_source(
        &self,
        source: &Source,
        page: usize,
    ) -> Option<Arc<[u8]>> {
        match source.bytes(page) {
            Ok(bytes) => Some(bytes),
            Err(error) => {
                self.record_failure(page, &error);
                None
            }
        }
    }

    fn decode(
        &self,
        source: &Source,
        page: usize,
        bytes: &[u8],
        lane: &str,
    ) -> Option<Decoded> {
        let started = Instant::now();
        match decode_page(source, page, bytes) {
            Ok(native) => {
                tracing::debug!(
                    page,
                    lane,
                    elapsed = ?started.elapsed(),
                    "decoded"
                );
                Some(native)
            }
            Err(error) => {
                self.record_failure(page, &error);
                None
            }
        }
    }
}

fn derive_thumb_from_fit<Sink>(
    shared: &Shared<Sink>,
    last_fit: &mut Option<(u128, Decoded)>,
    page: usize,
    hash: u128,
    lane: &str,
) -> Option<Decoded>
where
    Sink: EventSink,
{
    if let Some((fit_hash, fit)) = last_fit {
        if *fit_hash == hash {
            return Some(derive_timed(page, Role::Thumb, fit, lane));
        }
    }
    let fit = shared.load_cached(hash, Role::Fit)?;
    let thumb = derive_timed(page, Role::Thumb, &fit, lane);
    *last_fit = Some((hash, fit));
    Some(thumb)
}

fn serve_cached<Sink>(
    shared: &Shared<Sink>,
    last_fit: &mut Option<(u128, Decoded)>,
    page: usize,
    role: Role,
    hash: u128,
    kind: ServeKind,
    serve_started: Instant,
) -> bool
where
    Sink: EventSink,
{
    if let Some(image) = shared.load_cached(hash, role) {
        if kind.is_request() {
            shared.publish(page, role, &image);
        }
        kind.report(page, role, true, serve_started.elapsed());
        if role == Role::Fit {
            *last_fit = Some((hash, image));
        }
        return true;
    }
    if role == Role::Thumb {
        if let Some(thumb) = derive_thumb_from_fit(
            shared,
            last_fit,
            page,
            hash,
            kind.lane(),
        ) {
            if kind.is_request() {
                shared.publish(page, Role::Thumb, &thumb);
            }
            kind.report(page, Role::Thumb, false, serve_started.elapsed());
            shared.persist(page, hash, Role::Thumb, &thumb);
            return true;
        }
    }
    false
}

fn produce_preview<Sink>(
    source: &Source,
    shared: &Shared<Sink>,
    last_fit: &mut Option<(u128, Decoded)>,
    page: usize,
    role: Role,
    kind: ServeKind,
) where
    Sink: EventSink,
{
    let serve_started = Instant::now();
    let known_hash = shared.hash_of(page);
    if let Some(hash) = known_hash {
        if serve_cached(
            shared,
            last_fit,
            page,
            role,
            hash,
            kind,
            serve_started,
        ) {
            return;
        }
    }

    let Some(bytes) = shared.read_source(source, page) else {
        return;
    };
    let hash =
        known_hash.unwrap_or_else(|| shared.insert_hash(page, &bytes));
    if known_hash.is_none()
        && serve_cached(
            shared,
            last_fit,
            page,
            role,
            hash,
            kind,
            serve_started,
        )
    {
        return;
    }

    let Some(native) = shared.decode(source, page, &bytes, kind.lane())
    else {
        return;
    };
    let derived = derive_timed(page, role, &native, kind.lane());
    if kind.is_request() {
        shared.publish(page, role, &derived);
        kind.report(page, role, false, serve_started.elapsed());
        if role == Role::Fit && shared.wants_unserved(page, Role::Full) {
            shared.publish(page, Role::Full, &native);
            kind.report(page, Role::Full, false, serve_started.elapsed());
        }
    } else {
        kind.report(page, role, false, serve_started.elapsed());
    }
    shared.persist(page, hash, role, &derived);
    if role == Role::Fit {
        *last_fit = Some((hash, derived));
    }
}

fn serve_full<Sink>(
    source: &Source,
    shared: &Shared<Sink>,
    page: usize,
    requested_at: Instant,
) where
    Sink: EventSink,
{
    let serve_started = Instant::now();
    let kind = ServeKind::Request { requested_at };
    let Some(bytes) = shared.read_source(source, page) else {
        return;
    };
    let Some(native) = shared.decode(source, page, &bytes, kind.lane())
    else {
        return;
    };
    shared.publish(page, Role::Full, &native);
    kind.report(page, Role::Full, false, serve_started.elapsed());
}

fn serve_request<Sink>(
    source: &Source,
    shared: &Shared<Sink>,
    last_fit: &mut Option<(u128, Decoded)>,
    request: Request,
    requested_at: Instant,
) where
    Sink: EventSink,
{
    if request.role == Role::Full {
        serve_full(source, shared, request.page, requested_at);
        return;
    }
    produce_preview(
        source,
        shared,
        last_fit,
        request.page,
        request.role,
        ServeKind::Request { requested_at },
    );
}

struct PlaybackRunner {
    playing_pages: HashMap<usize, PlaybackSession>,
    steps: VecDeque<PlaybackStep>,
}

impl PlaybackRunner {
    fn new() -> Self {
        Self {
            playing_pages: HashMap::new(),
            steps: VecDeque::new(),
        }
    }

    fn apply(&mut self, command: PlaybackCommand) {
        match command {
            PlaybackCommand::Start { id, role } => {
                self.steps.retain(|step| step.id().page != id.page);
                self.steps.push_back(PlaybackStep::Start { id, role });
            }
            PlaybackCommand::Next { id } => {
                self.steps.push_back(PlaybackStep::Next { id });
            }
            PlaybackCommand::Stop { id } => {
                if self
                    .playing_pages
                    .get(&id.page)
                    .is_some_and(|session| session.id == id)
                {
                    self.playing_pages.remove(&id.page);
                }
                self.steps.retain(|step| step.id() != id);
            }
        }
    }

    fn pop_front(&mut self) -> Option<PlaybackStep> {
        self.steps.pop_front()
    }

    fn serve<Sink>(
        &mut self,
        source: &Source,
        step: PlaybackStep,
        shared: &Shared<Sink>,
    ) where
        Sink: EventSink,
    {
        match step {
            PlaybackStep::Start { id, role } => {
                self.start_playback(source, id, role, shared);
            }
            PlaybackStep::Next { id } => self.next_playback(id, shared),
        }
    }

    fn start_playback<Sink>(
        &mut self,
        source: &Source,
        id: PlaybackId,
        role: Role,
        shared: &Shared<Sink>,
    ) where
        Sink: EventSink,
    {
        let page = id.page;
        let bytes = match source.bytes(page) {
            Ok(bytes) => bytes,
            Err(error) => {
                tracing::debug!(?id, %error, "playback source failed");
                self.fail_playback(id, shared);
                return;
            }
        };
        let Some(name) = source.name(page) else {
            self.fail_playback(id, shared);
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
                self.fail_playback(id, shared);
            }
            Ok(Err(error)) => {
                tracing::warn!(
                    ?id,
                    ?role,
                    %error,
                    "playback open failed"
                );
                self.fail_playback(id, shared);
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
                self.fail_playback(id, shared);
            }
        }
    }

    fn next_playback<Sink>(
        &mut self,
        id: PlaybackId,
        shared: &Shared<Sink>,
    ) where
        Sink: EventSink,
    {
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
                shared.emit(ChildEvent::Playback(PlaybackEvent::Frame {
                    id,
                    delay: frame.delay,
                    image: &image,
                }));
            }
            Ok(Ok(None)) => {
                tracing::debug!(?id, ?role, "playback finished");
                self.playing_pages.remove(&page);
                shared.emit(ChildEvent::Playback(
                    PlaybackEvent::Finished { id },
                ));
            }
            Ok(Err(error)) => {
                tracing::warn!(
                    ?id,
                    ?role,
                    %error,
                    "playback frame failed"
                );
                self.fail_playback(id, shared);
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
                self.fail_playback(id, shared);
            }
        }
    }

    fn fail_playback<Sink>(
        &mut self,
        id: PlaybackId,
        shared: &Shared<Sink>,
    ) where
        Sink: EventSink,
    {
        if self
            .playing_pages
            .get(&id.page)
            .is_some_and(|session| session.id == id)
        {
            self.playing_pages.remove(&id.page);
        }
        self.steps.retain(|step| step.id() != id);
        shared.emit(ChildEvent::Playback(PlaybackEvent::Failed { id }));
    }
}

fn derive_timed(
    page: usize,
    role: Role,
    source: &Decoded,
    lane: &str,
) -> Decoded {
    let started = Instant::now();
    let derived = cache::derive(source, role);
    tracing::debug!(
        page,
        ?role,
        lane,
        elapsed = ?started.elapsed(),
        "derived"
    );
    derived
}

fn prefetch_loop<Sink>(source: &Source, shared: &Shared<Sink>)
where
    Sink: EventSink,
{
    let mut last_fit = None;
    while let Some(step) = shared.next_prefetch() {
        produce_preview(
            source,
            shared,
            &mut last_fit,
            step.page,
            step.role,
            ServeKind::Prefetch {
                preload: step.preload,
            },
        );
    }
}

fn run<Sink>(
    source: &Source,
    rx: &impl CommandReceiver,
    store: Store,
    sink: Sink,
) where
    Sink: EventSink + Send,
{
    let shared = Shared::new(source.len(), store, sink);
    let mut requested_at = Instant::now();
    let mut last_fit = None;
    let mut playback = PlaybackRunner::new();

    thread::scope(|scope| {
        let spawned = thread::Builder::new()
            .name("cy-prefetch".into())
            .spawn_scoped(scope, || prefetch_loop(source, &shared));
        if let Err(error) = spawned {
            tracing::warn!(?error, "prefetch thread disabled");
        }

        loop {
            if !shared.apply_messages(rx, &mut playback, &mut requested_at)
                || shared.is_stopped()
            {
                break;
            }
            if let Some((request, _guard)) = shared.next_request() {
                tracing::debug!(
                    page = request.page,
                    role = ?request.role,
                    waited = ?requested_at.elapsed(),
                    "request"
                );
                serve_request(
                    source,
                    &shared,
                    &mut last_fit,
                    request,
                    requested_at,
                );
                continue;
            }
            if let Some(step) = playback.pop_front() {
                playback.serve(source, step, &shared);
                continue;
            }
            let Some(message) = rx.recv() else {
                break;
            };
            shared.apply_message(
                message,
                &mut playback,
                &mut requested_at,
            );
        }
        shared.stop();
    });
}

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

pub fn main(args: &[OsString]) -> ExitCode {
    let Some((server, path)) = parse(args) else {
        tracing::error!(
            "--worker needs an ipc server name and a book path"
        );
        return ExitCode::FAILURE;
    };
    // The kernel kills this process with SIGKILL once the parent dies, so a
    // parent crash ends the decode.
    if let Err(error) =
        process::set_parent_process_death_signal(Some(Signal::KILL))
    {
        tracing::error!(?error, "could not arm the parent-death signal");
        return ExitCode::FAILURE;
    }
    // The parent may have died before the signal was armed; exit.
    if process::getppid() == Some(Pid::INIT) {
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
        archive: opened.is_archive(),
    };
    if child_tx.send(ChildMsg::BookInfo(info)).is_err() {
        tracing::warn!("the parent left before the book info");
        return ExitCode::FAILURE;
    }

    run(&opened, &parent_rx, Store::open(), child_tx);
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

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use std::fs;
    use std::sync::mpsc;

    use image::codecs::gif::Repeat;
    use image::ImageError;

    use super::*;
    use crate::animation::DecodedEvent;
    use crate::browse::Direction;
    use crate::browse::ReadingMode;
    use crate::cache::Store;
    use crate::decode::Pixels;
    use crate::fault::DecodeRefusal;
    use crate::testkit;
    use crate::testkit::TempDir;

    /// Page edge for the tests that check derived and full dimensions:
    /// larger than the thumb edge, smaller than the fit edge, so the
    /// thumb is scaled and the fit image is the page itself.
    const BIG: u32 = 1400;

    enum Update {
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
    }

    impl CommandReceiver for mpsc::Receiver<ParentMsg> {
        fn try_recv(&self) -> CommandPoll {
            match Self::try_recv(self) {
                Ok(message) => CommandPoll::Message(message),
                Err(mpsc::TryRecvError::Empty) => CommandPoll::Empty,
                Err(mpsc::TryRecvError::Disconnected) => {
                    CommandPoll::Disconnected
                }
            }
        }

        fn recv(&self) -> Option<ParentMsg> {
            Self::recv(self).ok()
        }
    }

    struct TestSink(Option<mpsc::Sender<Update>>);

    impl TestSink {
        const fn discard() -> Self {
            Self(None)
        }
    }

    impl EventSink for TestSink {
        fn emit(&self, event: ChildEvent<'_>) -> bool {
            let Some(tx) = &self.0 else {
                return true;
            };
            let update = match event {
                ChildEvent::Ready { page, role, image } => Update::Ready {
                    page,
                    role,
                    image: Box::new(image.clone()),
                },
                ChildEvent::Failed { page, error } => Update::Failed {
                    page,
                    error: Arc::clone(error),
                },
                ChildEvent::Playback(playback) => Update::Playback(
                    playback.map_image(|image| Box::new(image.clone())),
                ),
            };
            // A dropped receiver means the test already ended; keep the
            // worker alive until commands close.
            let Ok(()) = tx.send(update) else {
                return true;
            };
            true
        }
    }

    struct TestWorker {
        tx: Option<mpsc::Sender<ParentMsg>>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl TestWorker {
        fn spawn_at(
            source: Arc<Source>,
            initial_view: Option<usize>,
            store_root: &Path,
            sink: TestSink,
        ) -> Self {
            let (tx, rx) = mpsc::channel();
            let store = Store::at(store_root);
            let handle = thread::Builder::new()
                .name("test-worker".into())
                .spawn(move || {
                    run(&source, &rx, store, sink);
                })
                .expect("test worker thread spawns");
            let worker = Self {
                tx: Some(tx),
                handle: Some(handle),
            };
            if let Some(page) = initial_view {
                worker.set_view(page);
            }
            worker
        }

        fn send(&self, message: ParentMsg) {
            let Some(tx) = &self.tx else {
                return;
            };
            let Ok(()) = tx.send(message) else {
                return;
            };
        }

        fn set_view(&self, page: usize) {
            self.send(ParentMsg::SetView {
                page,
                mode: ReadingMode::Normal,
                direction: Direction::Forward,
            });
        }

        fn set_requests(&self, requests: Vec<Request>) {
            self.send(ParentMsg::SetRequests(requests));
        }

        fn playback(&self, command: PlaybackCommand) {
            self.send(ParentMsg::Playback(command));
        }
    }

    impl Drop for TestWorker {
        fn drop(&mut self) {
            drop(self.tx.take());
            if let Some(handle) = self.handle.take() {
                if handle.join().is_err() {
                    tracing::warn!("test worker panicked");
                }
            }
        }
    }

    fn start_playback(
        worker: &TestWorker,
        page: usize,
        role: Role,
        generation: u32,
    ) {
        worker.playback(PlaybackCommand::Start {
            id: PlaybackId { page, generation },
            role,
        });
    }

    fn next_frame(worker: &TestWorker, page: usize, generation: u32) {
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
                &testkit::tiny_png(size, size, [shade, tint, 90, 255]),
            );
        }
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        (dir, source)
    }

    /// The worker's content hash for one page of a directory source.
    fn page_hash(source: &Source, page: usize) -> u128 {
        match source.bytes(page) {
            Ok(bytes) => cache::content_hash(&bytes),
            Err(error) => panic!("page {page} did not read: {error}"),
        }
    }

    fn long_edge(image: &Decoded) -> u32 {
        image.width().max(image.height())
    }

    fn first_pixel(image: &Decoded) -> [u8; 4] {
        let Pixels::Rgba8(bytes) = image.pixels() else {
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

    impl Updates {
        fn new() -> (Self, TestSink) {
            let (tx, rx) = mpsc::channel();
            let events = Self {
                rx,
                seen: Mutex::new(Vec::new()),
            };
            (events, TestSink(Some(tx)))
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
            testkit::eventually(20, what, || {
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
                    | Update::Playback(_) => None,
                })
            })
        }

        fn failure(&self, page: usize) -> Arc<PageError> {
            testkit::eventually(20, "page never failed", || {
                self.drain();
                let seen = self.seen.lock();
                seen.iter().find_map(|event| match event {
                    Update::Failed {
                        page: failed,
                        error,
                    } if *failed == page => Some(Arc::clone(error)),
                    Update::Ready { .. }
                    | Update::Failed { .. }
                    | Update::Playback(_) => None,
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
                    | Update::Playback(_) => None,
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
            testkit::eventually(20, "no animation frame", || {
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
                        | Update::Playback(_) => None,
                    })
                    .nth(offset)
            })
        }

        fn finished(&self, page: usize) {
            testkit::eventually(20, "no animation finish", || {
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
    fn an_archive_serves_pages() {
        const SEVENZ_FIXTURE: &str = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/two_pages.7z"
        );
        let source = Arc::new(
            Source::open(Path::new(SEVENZ_FIXTURE)).expect("7z opens"),
        );
        let root = TempDir::new("worker-archive-store");
        let (events, sink) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            sink,
        );

        worker.set_requests(vec![request_thumb(0)]);
        let thumb = events
            .ready("the archive page never served", |page, role| {
                page == 0 && is_thumb(role)
            });
        assert_eq!((thumb.width(), thumb.height()), (40, 30));
    }

    #[test]
    fn a_target_is_served_and_persisted() {
        let (_dir, source) = page_dir("worker-target", 1, BIG);
        let root = TempDir::new("worker-target-store");
        let (events, sink) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            sink,
        );

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
        let _worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            TestSink::discard(),
        );
        let store = Store::at(root.path());
        // The head is pages 0, 1 and 2; wait until it is fully produced.
        for page in [0, 1, 2] {
            testkit::eventually(
                20,
                "the head never reached the store",
                || {
                    store
                        .load(page_hash(&source, page), Role::Thumb)
                        .is_some()
                        .then_some(())
                },
            );
        }
        let head_done = Instant::now();
        // Page 3 is beyond the head: it must not start until the app has
        // been silent for the preload window.
        testkit::eventually(20, "the tail never preloaded", || {
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
                &testkit::tiny_png(BIG, BIG, [page, 20, 30, 255]),
            );
        }
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let root = TempDir::new("worker-fail-store");
        let (events, sink) = Updates::new();
        let _worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            sink,
        );

        let error = events.failure(0);
        assert!(
            error.downcast_inner_ref::<ImageError>().is_some()
                || error.downcast_inner_ref::<DecodeRefusal>().is_some(),
            "page 0 failed as {error}, expected a decode fault"
        );
        // The failure leaves the rest of the prefetch running: pages behind it
        // still reach the store.
        let store = Store::at(root.path());
        testkit::eventually(
            20,
            "the failure stalled the prefetch",
            || {
                store
                    .load(page_hash(&source, 4), Role::Thumb)
                    .is_some()
                    .then_some(())
            },
        );
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
        let (events, sink) = Updates::new();
        let _worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            sink,
        );

        let store = Store::at(root.path());
        testkit::eventually(
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
        let (events, sink) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            sink,
        );

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
        testkit::eventually(
            20,
            "the re-added fit image was never served",
            || (events.delivered(0, is_fit) >= 2).then_some(()),
        );
    }

    #[test]
    fn a_full_target_carries_the_page_pixels() {
        let (_dir, source) = page_dir("worker-full", 1, BIG);
        let root = TempDir::new("worker-full-store");
        let (events, sink) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            sink,
        );

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
            &testkit::tiny_animated_gif(
                8,
                8,
                &colors,
                50,
                Repeat::Infinite,
            ),
        );
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let root = TempDir::new("worker-animation-store");
        let (events, sink) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            sink,
        );

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
            &testkit::tiny_animated_gif(
                8,
                8,
                &colors,
                50,
                Repeat::Finite(0),
            ),
        );
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let root = TempDir::new("worker-animation-finite-store");
        let (events, sink) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            sink,
        );

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
        let (first, sink) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            store_root,
            sink,
        );
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
        let store = Store::at(store_root);
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
            &testkit::tiny_png(7, 7, [1, 2, 3, 255]),
            "sentinel.png",
        )
        .expect("sentinel decodes");
        assert!(store.save(hash, Role::Thumb, &sentinel).persisted);
        assert!(store.save(hash, Role::Fit, &sentinel).persisted);
        drop(store);

        // A second worker on the same root loads the cached roles.
        let (second, sink_second) = Updates::new();
        let warm = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            store_root,
            sink_second,
        );
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

    #[test]
    fn store_hit_with_known_hash_skips_source_read() {
        const WAIT_TIMEOUT_SECS: u64 = 20;

        let (dir, source) = page_dir("worker-known-hash", 2, BIG);
        let root = TempDir::new("worker-known-hash-store");
        let requested_hash = page_hash(&source, 0);
        let prefetched_hash = page_hash(&source, 1);
        let (events, sink) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            sink,
        );

        worker.set_requests(vec![request_fit(0)]);
        events
            .ready("initial fit", |page, role| page == 0 && is_fit(role));
        let store = Store::at(root.path());
        testkit::eventually(
            WAIT_TIMEOUT_SECS,
            "warmed fit and thumb never persisted",
            || {
                (store.load(requested_hash, Role::Fit).is_some()
                    && store.load(requested_hash, Role::Thumb).is_some()
                    && store.load(prefetched_hash, Role::Fit).is_some())
                .then_some(())
            },
        );

        for name in ["p0000.png", "p0001.png"] {
            fs::remove_file(dir.path().join(name))
                .expect("source file removes");
        }
        worker.set_requests(Vec::new());
        worker.set_requests(vec![
            request_fit(0),
            request_thumb(0),
            request_fit(1),
        ]);
        testkit::eventually(
            WAIT_TIMEOUT_SECS,
            "known-hash store hits never delivered from shared store",
            || {
                (events.delivered(0, is_fit) >= 2
                    && events.delivered(0, is_thumb) >= 1
                    && events.delivered(1, is_fit) >= 1)
                    .then_some(())
            },
        );
        assert_eq!(events.failures(0), 0);
        assert_eq!(events.failures(1), 0);
    }

    #[test]
    fn thumb_miss_derives_from_cached_fit_without_decoding() {
        const SENTINEL_HEIGHT: u32 = 300;
        const SENTINEL_WIDTH: u32 = SENTINEL_HEIGHT * 2;

        let (_dir, source) = page_dir("worker-derive-fit", 1, BIG);
        let root = TempDir::new("worker-derive-fit-store");
        let hash = page_hash(&source, 0);
        let landscape_fit = decode::decode(
            &testkit::tiny_png(
                SENTINEL_WIDTH,
                SENTINEL_HEIGHT,
                [1, 2, 3, 255],
            ),
            "sentinel.png",
        )
        .expect("sentinel decodes");
        let store = Store::at(root.path());
        assert!(store.save(hash, Role::Fit, &landscape_fit).persisted);
        drop(store);

        let (events, sink) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            None,
            root.path(),
            sink,
        );
        worker.set_requests(vec![request_thumb(0)]);
        let thumb = events
            .ready("derived thumb never arrived", |page, role| {
                page == 0 && is_thumb(role)
            });
        assert_eq!(thumb.width(), cache::THUMB_EDGE);
        assert_eq!(thumb.height() * 2, thumb.width());
    }

    #[test]
    fn fit_and_full_requested_together_serve_both() {
        const PAGE_WIDTH: u32 = cache::FIT_EDGE + 40;
        const PAGE_HEIGHT: u32 = 16;

        let dir = TempDir::new("worker-fit-full");
        dir.write(
            "p0000.png",
            &testkit::tiny_png(PAGE_WIDTH, PAGE_HEIGHT, [10, 20, 30, 255]),
        );
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let root = TempDir::new("worker-fit-full-store");
        let (events, sink) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            None,
            root.path(),
            sink,
        );

        worker.set_requests(vec![
            request_fit(0),
            request_full(0),
            request_thumb(0),
        ]);
        events.ready("thumb never arrived", |page, role| {
            page == 0 && is_thumb(role)
        });
        let fit = events.ready("fit never arrived", |page, role| {
            page == 0 && is_fit(role)
        });
        let full = events.ready("full never arrived", |page, role| {
            page == 0 && role == Role::Full
        });
        assert_eq!(
            (fit.width(), fit.height()),
            (cache::FIT_EDGE, PAGE_HEIGHT)
        );
        assert_eq!(
            (full.width(), full.height()),
            (PAGE_WIDTH, PAGE_HEIGHT)
        );
        assert_eq!(events.delivered(0, |role| role == Role::Full), 1);
    }
}
