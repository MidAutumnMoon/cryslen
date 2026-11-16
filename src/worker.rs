//! Per-book decode worker (`cy --worker`): priority worker pool and IPC loop.

use std::borrow::Cow;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fmt;
use std::num::NonZeroUsize;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::OnceLock;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use ipc_channel::TryRecvError as IpcTryRecvError;
use ipc_channel::ipc;
use ipc_channel::ipc::IpcReceiver;
use ipc_channel::ipc::IpcSender;
use parking_lot::Condvar;
use parking_lot::Mutex;
use rustix::process;
use rustix::process::Pid;
use rustix::process::Signal;

use crate::browse;
use crate::browse::Direction;
use crate::browse::Motion;
use crate::browse::PageCount;
use crate::browse::ReadingMode;
use crate::browse::Request;
use crate::browse::Role;
use crate::browse::Roles;
use crate::browse::Spread;
use crate::cache;
use crate::cache::ImageLimits;
use crate::cache::Store;
use crate::decode;
use crate::decode::Animation;
use crate::decode::Decoded;
use crate::decode::Frame;
use crate::decode::Loops;
use crate::fault;
use crate::fault::PageFault;
use crate::protocol::BookInfo;
use crate::protocol::ChildMsg;
use crate::protocol::Hello;
use crate::protocol::ParentMsg;
use crate::protocol::PlaybackCommand;
use crate::protocol::PlaybackEvent;
use crate::protocol::PlaybackId;
use crate::protocol::WireImage;
use crate::source::Source;

const IDLE_PRELOAD: Duration = Duration::from_millis(300);

const PREFETCH_AHEAD_PAGES: usize = 64;
const PREFETCH_BEHIND_PAGES: usize = 16;
const PREFETCH_HEAD_SPREADS: usize = 8;
const PREFETCH_AHEAD_RATIO: usize = 2;

const MEMORY_CACHE_CAP_BYTES: usize = 192 * 1024 * 1024;

const MIN_WORKERS: usize = 2;
const MAX_WORKERS: usize = 6;
const RESERVED_FOREGROUND_WORKERS: usize = 2;
const MIN_PREFETCH_WORKERS: usize = 1;
const PREFETCH_THREADS_PER_PAGE: usize = 1;

fn worker_count() -> usize {
    thread::available_parallelism()
        .map_or(MIN_WORKERS, NonZeroUsize::get)
        .clamp(MIN_WORKERS, MAX_WORKERS)
}

const fn max_prefetch_workers(workers: usize) -> usize {
    let available = workers.saturating_sub(RESERVED_FOREGROUND_WORKERS);
    if available > MIN_PREFETCH_WORKERS {
        available
    } else {
        MIN_PREFETCH_WORKERS
    }
}

fn spread_contains(spread: Spread, page: usize) -> bool {
    spread.first == page || spread.second == Some(page)
}

enum ChildEvent<'image> {
    Ready {
        page: usize,
        role: Role,
        image: &'image Decoded,
    },
    Failed {
        page: usize,
        error: &'image dyn fmt::Display,
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
                fault: PageFault::new(error.to_string()),
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
    const fn id(self) -> PlaybackId {
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
    fn advance(&mut self) -> eros::Result<Option<Frame>> {
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

#[derive(Debug, Clone)]
struct Prefetch {
    spreads: Vec<Spread>,
    head: usize,
}

impl Prefetch {
    fn build(
        origin: Spread,
        direction: Direction,
        mode: ReadingMode,
        len: PageCount,
    ) -> Self {
        let (ahead, behind) = match direction {
            Direction::Forward => {
                (PREFETCH_AHEAD_PAGES, PREFETCH_BEHIND_PAGES)
            }
            Direction::Backward => {
                (PREFETCH_BEHIND_PAGES, PREFETCH_AHEAD_PAGES)
            }
        };
        let increasing = walk_spreads(
            origin.first,
            Direction::Forward,
            mode,
            len,
            ahead,
        );
        let decreasing = walk_spreads(
            origin.first,
            Direction::Backward,
            mode,
            len,
            behind,
        );
        let (travel, opposite) = match direction {
            Direction::Forward => (&increasing, &decreasing),
            Direction::Backward => (&decreasing, &increasing),
        };
        let mut spreads =
            Vec::with_capacity(1 + increasing.len() + decreasing.len());
        spreads.push(origin);
        let mut travel_iter = travel.iter().copied();
        let mut opposite_iter = opposite.iter().copied();
        loop {
            let mut pushed = false;
            for spread in travel_iter.by_ref().take(PREFETCH_AHEAD_RATIO) {
                spreads.push(spread);
                pushed = true;
            }
            if let Some(spread) = opposite_iter.next() {
                spreads.push(spread);
                pushed = true;
            }
            if !pushed {
                break;
            }
        }
        let head = spreads.len().min(PREFETCH_HEAD_SPREADS);
        Self { spreads, head }
    }

    fn spreads(&self) -> &[Spread] {
        &self.spreads
    }

    /// Leading spreads warmed at `Role::Fit` while still.
    fn head(&self) -> &[Spread] {
        self.spreads.get(..self.head).unwrap_or_default()
    }

    fn in_head(&self, page: usize) -> bool {
        self.head()
            .iter()
            .any(|spread| spread_contains(*spread, page))
    }

    fn contains(&self, page: usize) -> bool {
        self.spreads()
            .iter()
            .any(|spread| spread_contains(*spread, page))
    }
}

fn walk_spreads(
    start: usize,
    direction: Direction,
    mode: ReadingMode,
    len: PageCount,
    limit: usize,
) -> Vec<Spread> {
    let mut spreads = Vec::new();
    let mut current = start;
    loop {
        let next = browse::advance(current, direction, mode, len);
        let forward = direction == Direction::Forward;
        let wrapped = if forward {
            next <= current
        } else {
            next >= current
        };
        if wrapped {
            break;
        }
        let distance = if forward { next - start } else { start - next };
        if distance > limit {
            break;
        }
        spreads.push(browse::spread_for(next, mode, len));
        current = next;
    }
    spreads
}

struct MemoryCache {
    cap_bytes: usize,
    entries: HashMap<u128, Arc<Decoded>>,
    order: VecDeque<u128>,
}

impl MemoryCache {
    fn new(cap_bytes: usize) -> Self {
        Self {
            cap_bytes,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn total_bytes(&self) -> usize {
        self.entries
            .values()
            .map(|image| image.resident_bytes())
            .sum()
    }

    fn get(&mut self, hash: u128) -> Option<Arc<Decoded>> {
        let image = self.entries.get(&hash).map(Arc::clone)?;
        self.touch(hash);
        Some(image)
    }

    fn insert(&mut self, hash: u128, image: Arc<Decoded>) {
        if image.resident_bytes() > self.cap_bytes {
            return;
        }
        self.entries.insert(hash, image);
        self.touch(hash);
        while self.total_bytes() > self.cap_bytes {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
    }

    fn touch(&mut self, hash: u128) {
        self.order.retain(|key| *key != hash);
        self.order.push_back(hash);
    }
}

#[derive(Clone, Copy, Debug)]
enum JobKind {
    Request { requested_at: Instant },
    Prefetch { preload: bool },
}

impl JobKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Request { .. } => "request",
            Self::Prefetch { .. } => "prefetch",
        }
    }

    const fn is_prefetch(self) -> bool {
        matches!(self, Self::Prefetch { .. })
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

#[derive(Clone, Copy, Debug)]
struct PageJob {
    page: usize,
    kind: JobKind,
}

#[derive(Clone, Copy)]
enum PreloadGate {
    Idle,
    Countdown(Instant),
    Open,
}

#[derive(Clone, Copy)]
struct PageState {
    wanted: Roles,
    in_head_still: bool,
    in_prefetch: bool,
    stored: Roles,
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
    memory_cache: MemoryCache,
    /// Pages whose first failure settled them for the session.
    failed: HashSet<usize>,
    /// Pages whose `Role::Full` derivation failed and must not retry.
    failed_full: HashSet<usize>,
    in_flight: HashMap<usize, JobKind>,
    requests: Vec<Request>,
    requests_at: Instant,
    /// Requests already handed over since they last entered the set.
    served: HashSet<Request>,
    preload: PreloadGate,
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
            memory_cache: MemoryCache::new(MEMORY_CACHE_CAP_BYTES),
            failed: HashSet::new(),
            failed_full: HashSet::new(),
            in_flight: HashMap::new(),
            requests: Vec::new(),
            requests_at: Instant::now(),
            served: HashSet::new(),
            preload: PreloadGate::Idle,
            stopped: false,
        }
    }

    fn apply(
        &mut self,
        message: ParentMsg,
        playback: &mut PlaybackRunner,
    ) -> bool {
        // Ignore playback so animation frames keep the preload quiet
        // window.
        if !matches!(message, ParentMsg::Playback(_)) {
            self.preload = PreloadGate::Idle;
        }
        match message {
            // Handled on `Shared` before locking `state`.
            ParentMsg::SetLimits(_) => false,
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
                self.requests_at = Instant::now();
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

    fn role_failed(&self, page: usize, role: Role) -> bool {
        self.failed.contains(&page)
            || (role == Role::Full && self.failed_full.contains(&page))
    }

    fn wants_unserved(&self, page: usize, role: Role) -> bool {
        let request = Request::new(page, role);
        self.requests.contains(&request)
            && !self.served.contains(&request)
            && !self.role_failed(page, role)
    }

    fn wanted_roles(&self, page: usize) -> Roles {
        let mut roles = Roles::default();
        for role in [Role::Full, Role::Fit, Role::Thumb] {
            if self.wants_unserved(page, role) {
                roles.set(role);
            }
        }
        roles
    }

    fn claim_publish(&mut self, page: usize, role: Role) -> bool {
        if !self.wants_unserved(page, role) {
            return false;
        }
        self.served.insert(Request::new(page, role));
        true
    }

    fn stored_for_hash(&self, hash: u128) -> Roles {
        self.stored.get(&hash).copied().unwrap_or_default()
    }

    fn page_state(&self, page: usize, hash: u128) -> PageState {
        let in_head_still = self.motion == Motion::Still
            && self
                .prefetch
                .as_ref()
                .is_some_and(|prefetch| prefetch.in_head(page));
        let in_prefetch = self
            .prefetch
            .as_ref()
            .is_some_and(|prefetch| prefetch.contains(page));
        PageState {
            wanted: self.wanted_roles(page),
            in_head_still,
            in_prefetch,
            stored: self.stored_for_hash(hash),
        }
    }

    fn in_flight_prefetch(&self) -> usize {
        self.in_flight
            .values()
            .filter(|kind| kind.is_prefetch())
            .count()
    }

    fn take_request_page(&mut self) -> Option<PageJob> {
        let request = *self.requests.iter().find(|request| {
            !self.served.contains(request)
                && !self.role_failed(request.page, request.role)
                && !self.in_flight.contains_key(&request.page)
        })?;
        let kind = JobKind::Request {
            requested_at: self.requests_at,
        };
        self.in_flight.insert(request.page, kind);
        Some(PageJob {
            page: request.page,
            kind,
        })
    }

    fn next_prefetch_job(&self) -> Option<(usize, Role, bool, Motion)> {
        let prefetch = self.prefetch.as_ref()?;
        let head = prefetch.head();
        for spread in prefetch.spreads() {
            let in_head = head.contains(spread);
            for page in
                [Some(spread.first), spread.second].into_iter().flatten()
            {
                if self.failed.contains(&page)
                    || self.in_flight.contains_key(&page)
                {
                    continue;
                }
                let stored = self
                    .hashes
                    .get(&page)
                    .map_or_default(|&hash| self.stored_for_hash(hash));
                if in_head
                    && self.motion == Motion::Still
                    && !stored.contains(Role::Fit)
                {
                    return Some((page, Role::Fit, false, self.motion));
                }
                if !stored.contains(Role::Thumb) {
                    return Some((
                        page,
                        Role::Thumb,
                        !in_head,
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

struct InFlightGuard<'shared, Sink> {
    shared: &'shared Shared<Sink>,
    page: usize,
}

impl<Sink> Drop for InFlightGuard<'_, Sink> {
    fn drop(&mut self) {
        self.shared.finish_page_job(self.page);
    }
}

struct Shared<Sink> {
    store: Store,
    limits: OnceLock<ImageLimits>,
    state: Mutex<BookState>,
    wake: Condvar,
    sink: Mutex<Sink>,
}

impl<Sink> Shared<Sink> {
    fn finish_page_job(&self, page: usize) {
        self.state.lock().in_flight.remove(&page);
        self.wake.notify_all();
    }
}

impl<Sink> Shared<Sink>
where
    Sink: EventSink,
{
    fn new(page_count: PageCount, store: Store, sink: Sink) -> Self {
        Self {
            store,
            limits: OnceLock::new(),
            state: Mutex::new(BookState::new(page_count)),
            wake: Condvar::new(),
            sink: Mutex::new(sink),
        }
    }

    fn stop(&self) {
        self.state.lock().stopped = true;
        self.wake.notify_all();
    }

    fn is_stopped(&self) -> bool {
        self.state.lock().stopped
    }

    fn apply_message(
        &self,
        message: ParentMsg,
        playback: &mut PlaybackRunner,
    ) {
        if let ParentMsg::SetLimits(limits) = message {
            match self.limits.set(limits) {
                Ok(()) => tracing::debug!(
                    fit_edge = limits.fit_edge(),
                    max_texture_edge = limits.max_texture_edge(),
                    "limits set"
                ),
                Err(existing) => tracing::debug!(
                    fit_edge = existing.fit_edge(),
                    max_texture_edge = existing.max_texture_edge(),
                    "ignoring duplicate limits"
                ),
            }
            // Acquire and release `state` before notifying so idle workers
            // cannot observe `None` and park after `notify_all`.
            drop(self.state.lock());
            self.wake.notify_all();
            return;
        }
        let wake_workers = self.state.lock().apply(message, playback);
        if wake_workers {
            self.wake.notify_all();
        }
    }

    fn apply_messages(
        &self,
        rx: &impl CommandReceiver,
        playback: &mut PlaybackRunner,
    ) -> bool {
        loop {
            match rx.try_recv() {
                CommandPoll::Message(message) => {
                    self.apply_message(message, playback);
                }
                CommandPoll::Empty => return true,
                CommandPoll::Disconnected => return false,
            }
        }
    }

    fn limits(&self) -> Option<ImageLimits> {
        self.limits.get().copied()
    }

    #[expect(
        clippy::significant_drop_tightening,
        reason = "condvar wait loop holds the mutex guard across iterations"
    )]
    fn next_job(&self, max_prefetch: usize) -> Option<PageJob> {
        let mut state = self.state.lock();
        loop {
            if state.stopped {
                return None;
            }
            if self.limits.get().is_none() {
                self.wake.wait(&mut state);
                continue;
            }
            if let Some(job) = state.take_request_page() {
                return Some(job);
            }
            if state.in_flight_prefetch() >= max_prefetch {
                self.wake.wait(&mut state);
                continue;
            }
            let Some((page, role, preload, motion)) =
                state.next_prefetch_job()
            else {
                self.wake.wait(&mut state);
                continue;
            };
            if !preload || matches!(state.preload, PreloadGate::Open) {
                let kind = JobKind::Prefetch { preload };
                state.in_flight.insert(page, kind);
                drop(state);
                tracing::debug!(
                    page,
                    ?role,
                    ?motion,
                    preload,
                    "prefetch job"
                );
                return Some(PageJob { page, kind });
            }
            // Start the quiet countdown only once active work has drained.
            if !state.in_flight.is_empty() {
                self.wake.wait(&mut state);
                continue;
            }
            match state.preload {
                PreloadGate::Idle => {
                    let deadline = Instant::now() + IDLE_PRELOAD;
                    state.preload = PreloadGate::Countdown(deadline);
                    self.wake.wait_until(&mut state, deadline);
                }
                PreloadGate::Countdown(deadline) => {
                    if Instant::now() >= deadline {
                        state.preload = PreloadGate::Open;
                        tracing::debug!(?IDLE_PRELOAD, "preload started");
                        self.wake.notify_all();
                    } else {
                        self.wake.wait_until(&mut state, deadline);
                    }
                }
                PreloadGate::Open => {}
            }
        }
    }

    fn page_state(&self, page: usize, hash: u128) -> PageState {
        self.state.lock().page_state(page, hash)
    }

    fn load_memory_fit(&self, hash: u128) -> Option<Arc<Decoded>> {
        self.state.lock().memory_cache.get(hash)
    }

    fn save_memory_fit(&self, hash: u128, image: Arc<Decoded>) {
        self.state.lock().memory_cache.insert(hash, image);
    }

    fn resolve_hash(
        &self,
        source: &Source,
        page: usize,
    ) -> Option<(u128, Option<Arc<[u8]>>)> {
        let known = self.state.lock().hashes.get(&page).copied();
        if let Some(known) = known {
            return Some((known, None));
        }
        let bytes = self.read_source(source, page)?;
        let hash = cache::content_hash(&bytes);
        let stored = *self.state.lock().hashes.entry(page).or_insert(hash);
        Some((stored, Some(bytes)))
    }

    fn load_stored(&self, hash: u128, role: Role) -> Option<Decoded> {
        let limits = self.limits()?;
        let image = self.store.load(hash, role, limits.fit_edge())?;
        self.state.lock().mark_stored(hash, role);
        Some(image)
    }

    fn persist_if_missing(
        &self,
        page: usize,
        hash: u128,
        role: Role,
        image: &Decoded,
    ) {
        if self.state.lock().stored_for_hash(hash).contains(role) {
            return;
        }
        let Some(limits) = self.limits() else {
            return;
        };
        let saved = self.store.save(hash, role, image, limits.fit_edge());
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

    fn derive_for<'source>(
        &self,
        page: usize,
        role: Role,
        source: &'source Decoded,
        lane: &str,
    ) -> Option<Cow<'source, Decoded>> {
        let limits = self.limits()?;
        let started = Instant::now();
        match cache::derive(source, role, limits) {
            Ok(derived) => {
                tracing::debug!(
                    page,
                    ?role,
                    lane,
                    elapsed = ?started.elapsed(),
                    "derived"
                );
                Some(derived)
            }
            Err(fault) => {
                match role {
                    Role::Full => {
                        self.state.lock().failed_full.insert(page);
                    }
                    Role::Thumb | Role::Fit => {
                        self.record_failure(page, &fault);
                    }
                }
                None
            }
        }
    }

    fn emit(&self, event: ChildEvent<'_>) {
        let sent = self.sink.lock().emit(event);
        if !sent {
            self.state.lock().stopped = true;
            self.wake.notify_all();
        }
    }

    fn claim_and_publish(
        &self,
        page: usize,
        role: Role,
        image: &Decoded,
    ) -> bool {
        if !self.state.lock().claim_publish(page, role) {
            return false;
        }
        tracing::debug!(
            page,
            ?role,
            width = image.width(),
            height = image.height(),
            "publish"
        );
        self.emit(ChildEvent::Ready { page, role, image });
        true
    }

    fn record_failure(&self, page: usize, error: &dyn fmt::Display) {
        if self.state.lock().failed.insert(page) {
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
        max_threads: usize,
        lane: &str,
    ) -> Option<Decoded> {
        let started = Instant::now();
        match decode_page(source, page, bytes, max_threads) {
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum FitOrigin {
    Cache,
    Decode,
}

fn finish_fit<Sink>(
    shared: &Shared<Sink>,
    job: PageJob,
    hash: u128,
    fit: &Arc<Decoded>,
    origin: FitOrigin,
    thumb_done: bool,
    started: Instant,
) where
    Sink: EventSink,
{
    let page = job.page;
    let state = shared.page_state(page, hash);
    if state.wanted.contains(Role::Fit) || state.in_head_still {
        shared.save_memory_fit(hash, Arc::clone(fit));
        if state.wanted.contains(Role::Fit) {
            shared.claim_and_publish(page, Role::Fit, fit);
        }
        job.kind.report(
            page,
            Role::Fit,
            origin == FitOrigin::Cache,
            started.elapsed(),
        );
    }

    let need_thumb = state.wanted.contains(Role::Thumb)
        || (!thumb_done
            && (state.in_prefetch || state.in_head_still)
            && !state.stored.contains(Role::Thumb));
    if need_thumb
        && let Some(derived) =
            shared.derive_for(page, Role::Thumb, fit, job.kind.label())
    {
        let thumb = derived.into_owned();
        if state.wanted.contains(Role::Thumb) {
            shared.claim_and_publish(page, Role::Thumb, &thumb);
        }
        job.kind.report(page, Role::Thumb, false, started.elapsed());
        shared.persist_if_missing(page, hash, Role::Thumb, &thumb);
    }

    // Persist freshly derived `Fit` during tail preload so advancing into
    // `head` loads from the store without re-decoding.
    if origin == FitOrigin::Decode {
        shared.persist_if_missing(page, hash, Role::Fit, fit);
    }
}

fn produce_page<Sink>(
    source: &Source,
    shared: &Shared<Sink>,
    job: PageJob,
    max_threads: usize,
) where
    Sink: EventSink,
{
    let started = Instant::now();
    let page = job.page;
    let Some((hash, mut source_bytes)) = shared.resolve_hash(source, page)
    else {
        return;
    };

    let initial = shared.page_state(page, hash);
    let mut thumb_done = false;
    let mut cached_fit = shared.load_memory_fit(hash);

    // Check stored thumbnails before deriving from a cached `Fit` while
    // `MemoryCache` is cold so external store entries remain authoritative.
    let want_or_missing_thumb = initial.wanted.contains(Role::Thumb)
        || ((initial.in_prefetch || initial.in_head_still)
            && !initial.stored.contains(Role::Thumb));
    if cached_fit.is_none()
        && want_or_missing_thumb
        && let Some(loaded_thumb) = shared.load_stored(hash, Role::Thumb)
    {
        if initial.wanted.contains(Role::Thumb) {
            shared.claim_and_publish(page, Role::Thumb, &loaded_thumb);
        }
        job.kind.report(page, Role::Thumb, true, started.elapsed());
        thumb_done = true;
    }

    let want_fit =
        initial.wanted.contains(Role::Fit) || initial.in_head_still;
    if cached_fit.is_none()
        && (want_fit || (want_or_missing_thumb && !thumb_done))
        && let Some(loaded_fit) = shared.load_stored(hash, Role::Fit)
    {
        cached_fit = Some(Arc::new(loaded_fit));
    }

    if let Some(fit) = cached_fit.as_ref() {
        finish_fit(
            shared,
            job,
            hash,
            fit,
            FitOrigin::Cache,
            thumb_done,
            started,
        );
    }

    let before_decode = shared.page_state(page, hash);
    let need_fit_decode = cached_fit.is_none()
        && (before_decode.wanted.contains(Role::Fit)
            || before_decode.wanted.contains(Role::Thumb)
            || before_decode.in_head_still
            || (before_decode.in_prefetch && !thumb_done));
    if !before_decode.wanted.contains(Role::Full) && !need_fit_decode {
        return;
    }

    let Some(bytes) = source_bytes
        .take()
        .or_else(|| shared.read_source(source, page))
    else {
        return;
    };
    let Some(native) =
        shared.decode(source, page, &bytes, max_threads, job.kind.label())
    else {
        return;
    };

    let after_decode = shared.page_state(page, hash);
    if after_decode.wanted.contains(Role::Full)
        && let Some(full) =
            shared.derive_for(page, Role::Full, &native, job.kind.label())
    {
        shared.claim_and_publish(page, Role::Full, &full);
        job.kind.report(page, Role::Full, false, started.elapsed());
    }

    let need_preview = after_decode.wanted.contains(Role::Fit)
        || after_decode.wanted.contains(Role::Thumb)
        || after_decode.in_head_still
        || after_decode.in_prefetch;
    if cached_fit.is_none() && need_preview {
        let Some(derived) =
            shared.derive_for(page, Role::Fit, &native, job.kind.label())
        else {
            return;
        };
        let owned = match derived {
            Cow::Owned(derived_fit) => derived_fit,
            Cow::Borrowed(_) => native,
        };
        let decoded_fit = Arc::new(owned);
        finish_fit(
            shared,
            job,
            hash,
            &decoded_fit,
            FitOrigin::Decode,
            thumb_done,
            started,
        );
    }
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
                let message = fault::panic_message(payload.as_ref());
                tracing::warn!(
                    ?id,
                    ?role,
                    "playback open panicked: {message}"
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
        let Some(limits) = shared.limits() else {
            tracing::debug!(?id, "playback frame before limits arrived");
            self.fail_playback(id, shared);
            return;
        };
        let role = session.role;
        let produced =
            panic::catch_unwind(AssertUnwindSafe(|| session.advance()));
        match produced {
            Ok(Ok(Some(frame))) => {
                let image = match cache::derive(&frame.image, role, limits)
                {
                    Ok(image) => image,
                    Err(fault) => {
                        tracing::warn!(
                            ?id,
                            ?role,
                            %fault,
                            "playback frame exceeds texture limit"
                        );
                        self.fail_playback(id, shared);
                        return;
                    }
                };
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
                let message = fault::panic_message(payload.as_ref());
                tracing::warn!(
                    ?id,
                    ?role,
                    "playback frame panicked: {message}"
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

fn worker_loop<Sink>(
    source: &Source,
    shared: &Shared<Sink>,
    workers: usize,
) where
    Sink: EventSink,
{
    let max_prefetch = max_prefetch_workers(workers);
    while let Some(job) = shared.next_job(max_prefetch) {
        let _guard = InFlightGuard {
            shared,
            page: job.page,
        };
        if let JobKind::Request { requested_at } = job.kind {
            tracing::debug!(
                page = job.page,
                waited = ?requested_at.elapsed(),
                "request"
            );
        }
        let max_threads = if job.kind.is_prefetch() {
            PREFETCH_THREADS_PER_PAGE
        } else {
            workers
        };
        produce_page(source, shared, job, max_threads);
    }
}

fn run<Sink>(
    source: &Source,
    rx: &impl CommandReceiver,
    store: Store,
    sink: Sink,
) where
    Sink: EventSink + Send + Sync,
{
    let workers = worker_count();
    let shared = Shared::new(source.len(), store, sink);
    let mut playback = PlaybackRunner::new();

    thread::scope(|scope| {
        for index in 0..workers {
            let spawned = thread::Builder::new()
                .name(format!("cy-worker-{index}"))
                .spawn_scoped(scope, || {
                    worker_loop(source, &shared, workers);
                });
            if let Err(error) = spawned {
                tracing::warn!(
                    index,
                    ?error,
                    "worker thread spawn failed"
                );
            }
        }

        loop {
            if !shared.apply_messages(rx, &mut playback)
                || shared.is_stopped()
            {
                break;
            }
            if let Some(step) = playback.pop_front() {
                playback.serve(source, step, &shared);
                continue;
            }
            let Some(message) = rx.recv() else {
                break;
            };
            shared.apply_message(message, &mut playback);
        }
        shared.stop();
    });
}

fn decode_page(
    source: &Source,
    page: usize,
    bytes: &[u8],
    max_threads: usize,
) -> eros::Result<Decoded> {
    let Some(name) = source.name(page) else {
        // `Source::bytes` answers every page under its length, so a name
        // miss is a caller bug.
        eros::bail!("no such page");
    };
    match panic::catch_unwind(AssertUnwindSafe(|| {
        decode::decode_with_threads(bytes, name, max_threads)
    })) {
        Ok(result) => result,
        Err(payload) => eros::bail!(
            "decoder panicked: {}",
            fault::panic_message(payload.as_ref())
        ),
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
    // Connect before opening the source so a crash or kill during open
    // closes the accepted socket instead of holding `accept(2)` until
    // timeout.
    let hello = match IpcSender::<Hello>::connect(server_name.to_owned()) {
        Ok(hello) => hello,
        Err(error) => {
            tracing::error!(?error, "could not connect to the parent");
            return ExitCode::FAILURE;
        }
    };

    let opened = match Source::open(path) {
        Ok(source) => source,
        Err(error) => {
            tracing::warn!(?error, "book open refused");
            if hello.send(Hello::Failed(error.to_string())).is_err() {
                tracing::warn!("the parent left before the open verdict");
            }
            return ExitCode::FAILURE;
        }
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
    let info = BookInfo {
        names: opened.names().clone(),
        archive: opened.is_archive(),
    };
    if hello
        .send(Hello::Opened {
            info,
            messages: child_rx,
            commands: parent_tx,
        })
        .is_err()
    {
        tracing::warn!("the parent left before the handshake");
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
    use tempfile::TempDir;

    use super::*;
    use crate::cache::Store;
    use crate::decode::Pixels;
    use crate::testkit;

    /// Page edge for the tests that check derived and full dimensions:
    /// larger than the thumb edge, smaller than the fit edge, so the
    /// thumb is scaled and the fit image is the page itself.
    const BIG: u32 = 1400;
    const TEST_FIT_EDGE: u32 = 2560;

    fn test_limits() -> ImageLimits {
        const MAX_TEXTURE_EDGE: u32 = 16_384;
        ImageLimits::new(TEST_FIT_EDGE, MAX_TEXTURE_EDGE)
    }

    enum Update {
        Ready {
            page: usize,
            role: Role,
            image: Box<Decoded>,
        },
        Failed {
            page: usize,
        },
        Playback(PlaybackEvent<Box<Decoded>>),
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
                ChildEvent::Failed { page, .. } => Update::Failed { page },
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
            worker.send(ParentMsg::SetLimits(test_limits()));
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
            if let Some(handle) = self.handle.take()
                && handle.join().is_err()
            {
                tracing::warn!("test worker panicked");
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
    fn page_dir(count: usize, size: u32) -> (TempDir, Arc<Source>) {
        let dir = TempDir::new().expect("create temp dir");
        for page in 0..count {
            let shade = u8::try_from(page % 256).expect("shade fits");
            let tint = u8::try_from(page >> 8).expect("tint fits");
            fs::write(
                dir.path().join(format!("p{page:04}.png")),
                testkit::tiny_png(size, size, [shade, tint, 90, 255]),
            )
            .expect("write page");
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

        fn failure(&self, page: usize) {
            testkit::eventually(20, "page never failed", || {
                self.drain();
                let seen = self.seen.lock();
                seen.iter().find_map(|event| match event {
                    Update::Failed { page: failed } if *failed == page => {
                        Some(())
                    }
                    Update::Ready { .. }
                    | Update::Failed { .. }
                    | Update::Playback(_) => None,
                })
            });
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
                .filter(|event| {
                    matches!(event, Update::Failed { page: failed } if *failed == page)
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
                        Update::Playback(PlaybackEvent::Frame {
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
                            Update::Playback(PlaybackEvent::Finished {
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
        let root = TempDir::new().expect("create temp dir");
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
        let (_dir, source) = page_dir(1, BIG);
        let root = TempDir::new().expect("create temp dir");
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
            TEST_FIT_EDGE.min(BIG),
            "the fit image's long edge matches the target edge"
        );
        // `publish` runs before `persist`; join the worker so both files
        // finish persisting before inspecting the store.
        drop(worker);

        let store = Store::at(root.path());
        let hash = page_hash(&source, 0);
        assert!(
            store.load(hash, Role::Thumb, TEST_FIT_EDGE).is_some(),
            "the thumb was not persisted"
        );
        assert!(
            store.load(hash, Role::Fit, TEST_FIT_EDGE).is_some(),
            "the fit image was not persisted"
        );
    }

    #[test]
    fn the_tail_waits_for_a_quiet_app() {
        let count = PREFETCH_HEAD_SPREADS + 4;
        let (_dir, source) = page_dir(count, 8);
        let root = TempDir::new().expect("create temp dir");
        let _worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            TestSink::discard(),
        );
        let store = Store::at(root.path());
        // Wait until all head spreads are fully produced.
        for page in 0..PREFETCH_HEAD_SPREADS {
            testkit::eventually(
                20,
                "the head never reached the store",
                || {
                    store
                        .load(
                            page_hash(&source, page),
                            Role::Thumb,
                            TEST_FIT_EDGE,
                        )
                        .is_some()
                        .then_some(())
                },
            );
        }
        let head_done = Instant::now();
        // The first page beyond the head must not start until the quiet
        // window has elapsed after the head finishes.
        testkit::eventually(20, "the tail never preloaded", || {
            store
                .load(
                    page_hash(&source, PREFETCH_HEAD_SPREADS),
                    Role::Thumb,
                    TEST_FIT_EDGE,
                )
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
        let dir = TempDir::new().expect("create temp dir");
        fs::write(dir.path().join("p000.png"), b"not a png at all")
            .expect("write corrupt page");
        for page in 1..5_u8 {
            fs::write(
                dir.path().join(format!("p{page:03}.png")),
                testkit::tiny_png(BIG, BIG, [page, 20, 30, 255]),
            )
            .expect("write page");
        }
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let root = TempDir::new().expect("create temp dir");
        let (events, sink) = Updates::new();
        let _worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            sink,
        );

        events.failure(0);
        // The failure leaves the rest of the prefetch running: pages behind it
        // still reach the store.
        let store = Store::at(root.path());
        testkit::eventually(
            20,
            "the failure stalled the prefetch",
            || {
                store
                    .load(
                        page_hash(&source, 4),
                        Role::Thumb,
                        TEST_FIT_EDGE,
                    )
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
        let (_dir, source) = page_dir(1, BIG);
        let root = TempDir::new().expect("create temp dir");
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
                    .load(page_hash(&source, 0), Role::Fit, TEST_FIT_EDGE)
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
        let (_dir, source) = page_dir(1, BIG);
        let root = TempDir::new().expect("create temp dir");
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
        let (_dir, source) = page_dir(1, BIG);
        let root = TempDir::new().expect("create temp dir");
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
        let dir = TempDir::new().expect("create temp dir");
        fs::write(
            dir.path().join("p000.gif"),
            testkit::tiny_animated_gif(
                8,
                8,
                &colors,
                50,
                Repeat::Infinite,
            ),
        )
        .expect("write gif");
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let root = TempDir::new().expect("create temp dir");
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
        let dir = TempDir::new().expect("create temp dir");
        fs::write(
            dir.path().join("p000.gif"),
            testkit::tiny_animated_gif(
                8,
                8,
                &colors,
                50,
                Repeat::Finite(0),
            ),
        )
        .expect("write gif");
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let root = TempDir::new().expect("create temp dir");
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
        let (_dir, source) = page_dir(1, BIG);
        let root = TempDir::new().expect("create temp dir");
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
            TEST_FIT_EDGE.min(BIG),
            "the fit image's long edge matches the target edge"
        );
        drop(worker);

        let hash = page_hash(&source, 0);
        let store = Store::at(store_root);
        assert!(
            store.load(hash, Role::Thumb, TEST_FIT_EDGE).is_some(),
            "the thumb was not persisted"
        );
        assert!(
            store.load(hash, Role::Fit, TEST_FIT_EDGE).is_some(),
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
        assert!(
            store
                .save(hash, Role::Thumb, &sentinel, TEST_FIT_EDGE)
                .persisted
        );
        assert!(
            store
                .save(hash, Role::Fit, &sentinel, TEST_FIT_EDGE)
                .persisted
        );
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
            .load(hash, Role::Fit, TEST_FIT_EDGE)
            .expect("the sentinel fit stays");
        assert_eq!((kept.width(), kept.height()), (7, 7));
    }

    #[test]
    fn store_hit_with_known_hash_skips_source_read() {
        const WAIT_TIMEOUT_SECS: u64 = 20;

        let (dir, source) = page_dir(2, BIG);
        let root = TempDir::new().expect("create temp dir");
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
                (store
                    .load(requested_hash, Role::Fit, TEST_FIT_EDGE)
                    .is_some()
                    && store
                        .load(requested_hash, Role::Thumb, TEST_FIT_EDGE)
                        .is_some()
                    && store
                        .load(prefetched_hash, Role::Fit, TEST_FIT_EDGE)
                        .is_some())
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

        let (_dir, source) = page_dir(1, BIG);
        let root = TempDir::new().expect("create temp dir");
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
        assert!(
            store
                .save(hash, Role::Fit, &landscape_fit, TEST_FIT_EDGE)
                .persisted
        );
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
        const PAGE_WIDTH: u32 = TEST_FIT_EDGE + 40;
        const PAGE_HEIGHT: u32 = 16;

        let dir = TempDir::new().expect("create temp dir");
        fs::write(
            dir.path().join("p0000.png"),
            testkit::tiny_png(PAGE_WIDTH, PAGE_HEIGHT, [10, 20, 30, 255]),
        )
        .expect("write page");
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let root = TempDir::new().expect("create temp dir");
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
            (TEST_FIT_EDGE, PAGE_HEIGHT)
        );
        assert_eq!(
            (full.width(), full.height()),
            (PAGE_WIDTH, PAGE_HEIGHT)
        );
        assert_eq!(events.delivered(0, |role| role == Role::Full), 1);
    }

    fn page_count(count: usize) -> PageCount {
        PageCount::new(count).expect("test book is non-empty")
    }

    fn prefetch_starts(prefetch: &Prefetch) -> Vec<usize> {
        prefetch
            .spreads()
            .iter()
            .map(|spread| spread.first)
            .collect()
    }

    #[test]
    fn prefetch_walks_out_from_the_origin() {
        let book = page_count(10);
        let origin = browse::spread_for(4, ReadingMode::Normal, book);
        let forward = Prefetch::build(
            origin,
            Direction::Forward,
            ReadingMode::Normal,
            book,
        );
        assert_eq!(
            prefetch_starts(&forward),
            vec![4, 5, 6, 3, 7, 8, 2, 9, 1, 0]
        );
        assert!(
            forward
                .spreads()
                .iter()
                .all(|spread| spread.second.is_none())
        );

        let backward = Prefetch::build(
            origin,
            Direction::Backward,
            ReadingMode::Normal,
            book,
        );
        assert_eq!(
            prefetch_starts(&backward),
            vec![4, 3, 2, 5, 1, 0, 6, 7, 8, 9]
        );
    }

    #[test]
    fn prefetch_ends_do_not_wrap() {
        let long_book = page_count(1200);
        let opening =
            browse::spread_for(0, ReadingMode::Normal, long_book);
        let before = Prefetch::build(
            opening,
            Direction::Backward,
            ReadingMode::Normal,
            long_book,
        );
        assert_eq!(before.spreads().first(), Some(&opening));
        assert_eq!(before.spreads().len(), 1 + PREFETCH_BEHIND_PAGES);
        assert!(
            before
                .spreads()
                .iter()
                .all(|spread| spread.first <= PREFETCH_BEHIND_PAGES)
        );

        let closing =
            browse::spread_for(1199, ReadingMode::Normal, long_book);
        let after = Prefetch::build(
            closing,
            Direction::Forward,
            ReadingMode::Normal,
            long_book,
        );
        assert_eq!(after.spreads().first(), Some(&closing));
        assert_eq!(after.spreads().len(), 1 + PREFETCH_BEHIND_PAGES);
        assert!(after
            .spreads()
            .iter()
            .all(|spread| spread.first >= 1199 - PREFETCH_BEHIND_PAGES));

        let single = page_count(1);
        let only = browse::spread_for(0, ReadingMode::Manga, single);
        for direction in [Direction::Forward, Direction::Backward] {
            let alone = Prefetch::build(
                only,
                direction,
                ReadingMode::Manga,
                single,
            );
            assert_eq!(alone.spreads(), &[only]);
            assert_eq!(alone.head(), &[only]);
        }
    }

    #[test]
    fn prefetch_bounds_are_respected_and_unique() {
        use std::collections::BTreeSet;

        let book = page_count(1200);
        let origin = browse::spread_for(600, ReadingMode::Normal, book);
        let prefetch = Prefetch::build(
            origin,
            Direction::Forward,
            ReadingMode::Normal,
            book,
        );
        assert_eq!(
            prefetch.spreads().len(),
            1 + PREFETCH_AHEAD_PAGES + PREFETCH_BEHIND_PAGES
        );
        assert_eq!(prefetch.spreads().first(), Some(&origin));
        assert_eq!(prefetch.head().len(), PREFETCH_HEAD_SPREADS);
        assert!(prefetch.spreads().iter().all(|spread| {
            (600 - PREFETCH_BEHIND_PAGES..=600 + PREFETCH_AHEAD_PAGES)
                .contains(&spread.first)
        }));
        let unique: BTreeSet<usize> = prefetch
            .spreads()
            .iter()
            .map(|spread| spread.first)
            .collect();
        assert_eq!(unique.len(), prefetch.spreads().len());
    }
}
