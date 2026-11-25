//! Per-book decode worker (`cy --worker`): priority worker pool and IPC loop.

use std::borrow::Cow;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::ffi::OsString;
use std::fmt::Display;
use std::mem::DropGuard;
use std::num::NonZeroUsize;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::OnceLock;
use std::thread;
use std::thread::Builder;
use std::time::Duration;
use std::time::Instant;

use bytes::Bytes;
use ipc_channel::ipc;
use ipc_channel::ipc::IpcReceiver;
use ipc_channel::ipc::IpcSender;
use parking_lot::Condvar;
use parking_lot::Mutex;

use crate::browse;
use crate::browse::Direction;
use crate::browse::Motion;
use crate::browse::ReadingMode;
use crate::browse::Request;
use crate::browse::Role;
use crate::browse::Roles;
use crate::browse::Spread;
use crate::cache;
use crate::cache::Store;
use crate::decode;
use crate::decode::Animation;
use crate::decode::Decoded;
use crate::decode::Frame;
use crate::protocol::ChildMsg;
use crate::protocol::ImageLimits;
use crate::protocol::ParentMsg;
use crate::protocol::PlaybackCommand;
use crate::protocol::PlaybackEvent;
use crate::protocol::PlaybackId;
use crate::protocol::WireImage;
use crate::source;
use crate::source::Source;

const IDLE_PRELOAD: Duration = Duration::from_millis(1500);

const PREFETCH_AHEAD_PAGES: usize = 64;
const PREFETCH_BEHIND_PAGES: usize = 16;
const PREFETCH_HEAD_SPREADS: usize = 8;
const PREFETCH_AHEAD_RATIO: usize = 2;

const MEMORY_CACHE_CAP_BYTES: usize = 192 * 1024 * 1024;

const DEFAULT_CPU_THREADS: usize = 2;
const MIN_WORKERS: usize = 2;
const MAX_WORKERS: usize = 6;
const MAX_PREFETCH_WORKERS: usize = 1;
const PREFETCH_THREAD_DIVISOR: usize = 4;

fn available_parallelism() -> usize {
    thread::available_parallelism()
        .map_or(DEFAULT_CPU_THREADS, NonZeroUsize::get)
}

#[derive(Clone, Copy, Debug)]
enum PlaybackStep {
    Start { id: PlaybackId },
    Next { id: PlaybackId },
}

impl PlaybackStep {
    const fn id(self) -> PlaybackId {
        match self {
            Self::Start { id } | Self::Next { id } => id,
        }
    }
}

struct PlaybackSession {
    id: PlaybackId,
    animation: Animation,
    passes: u32,
}

impl PlaybackSession {
    fn advance(&mut self) -> eros::Result<Option<Frame>> {
        if let Some(frame) = self.animation.next_frame().transpose()? {
            return Ok(Some(frame));
        }
        self.passes = self.passes.saturating_add(1);
        if !self.should_repeat() {
            return Ok(None);
        }
        self.animation.rewind()?;
        self.animation.next_frame().transpose()
    }

    fn should_repeat(&self) -> bool {
        use crate::decode::Loops;

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
        len: NonZeroUsize,
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
        self.head().iter().any(|spread| spread.contains(page))
    }

    fn contains(&self, page: usize) -> bool {
        self.spreads().iter().any(|spread| spread.contains(page))
    }
}

fn walk_spreads(
    start: usize,
    direction: Direction,
    mode: ReadingMode,
    len: NonZeroUsize,
    limit: usize,
) -> Vec<Spread> {
    let mut spreads = Vec::new();
    let mut current = start;
    while let Some(next) = browse::step(current, direction, mode, len) {
        if next.abs_diff(start) > limit {
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

fn includes_preview(roles: Roles) -> bool {
    roles.contains(Role::Fit) || roles.contains(Role::Thumb)
}

struct PrefetchCandidate {
    page: usize,
    role: Role,
    preload: bool,
}

struct BookState {
    page_count: NonZeroUsize,
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
    fn new(page_count: NonZeroUsize) -> Self {
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

    fn role_failed(&self, page: usize, role: Role) -> bool {
        self.failed.contains(&page)
            || (role == Role::Full && self.failed_full.contains(&page))
    }

    fn is_unsettled(&self, request: Request) -> bool {
        !self.served.contains(&request)
            && !self.role_failed(request.page, request.role)
    }

    fn wants_unserved(&self, page: usize, role: Role) -> bool {
        let request = Request { page, role };
        self.requests.contains(&request) && self.is_unsettled(request)
    }

    fn wanted_roles(&self, page: usize) -> Roles {
        use strum::VariantArray as _;

        let mut roles = Roles::default();
        for &role in Role::VARIANTS {
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
        self.served.insert(Request { page, role });
        true
    }

    fn stored_for_hash(&self, hash: u128) -> Roles {
        self.stored.get(&hash).copied().unwrap_or_default()
    }

    fn needed_roles(&self, page: usize, hash: u128) -> Roles {
        let mut needed = self.wanted_roles(page);
        let Some(prefetch) = self.prefetch.as_ref() else {
            return needed;
        };
        if prefetch.in_head(page) && self.motion == Motion::Still {
            needed.set(Role::Fit);
        }
        if prefetch.contains(page)
            && !self.stored_for_hash(hash).contains(Role::Thumb)
        {
            needed.set(Role::Thumb);
        }
        needed
    }

    fn in_flight_prefetch(&self) -> usize {
        self.in_flight
            .values()
            .filter(|kind| kind.is_prefetch())
            .count()
    }

    fn take_request_page(&mut self, source: &Source) -> Option<PageJob> {
        let request = *self.requests.iter().find(|&&request| {
            self.is_unsettled(request)
                && !self.in_flight.contains_key(&request.page)
                && source.is_ready(request.page)
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

    fn has_pending_request(&self) -> bool {
        self.in_flight.values().any(|kind| !kind.is_prefetch())
            || self
                .requests
                .iter()
                .any(|&request| self.is_unsettled(request))
    }

    fn next_prefetch_job(&self, source: &Source) -> Option<PrefetchCandidate> {
        let prefetch = self.prefetch.as_ref()?;
        for (index, spread) in prefetch.spreads().iter().enumerate() {
            let in_head = index < prefetch.head;
            for page in
                [Some(spread.first), spread.second].into_iter().flatten()
            {
                if self.failed.contains(&page)
                    || self.in_flight.contains_key(&page)
                    || !source.is_ready(page)
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
                    return Some(PrefetchCandidate {
                        page,
                        role: Role::Fit,
                        preload: false,
                    });
                }
                if !stored.contains(Role::Thumb) {
                    return Some(PrefetchCandidate {
                        page,
                        role: Role::Thumb,
                        preload: !in_head,
                    });
                }
            }
        }
        None
    }

    fn mark_stored(&mut self, hash: u128, role: Role) {
        self.stored.entry(hash).or_default().set(role);
    }
}

struct Shared {
    store: Store,
    limits: OnceLock<ImageLimits>,
    state: Mutex<BookState>,
    wake: Condvar,
    to_parent: Mutex<IpcSender<ChildMsg>>,
}

impl Shared {
    fn new(
        page_count: NonZeroUsize,
        store: Store,
        to_parent: IpcSender<ChildMsg>,
    ) -> Self {
        Self {
            store,
            limits: OnceLock::new(),
            state: Mutex::new(BookState::new(page_count)),
            wake: Condvar::new(),
            to_parent: Mutex::new(to_parent),
        }
    }

    fn finish_page_job(&self, page: usize) {
        self.state.lock().in_flight.remove(&page);
        self.wake.notify_all();
    }

    fn stop(&self) {
        self.state.lock().stopped = true;
        self.wake.notify_all();
    }

    fn wake_all(&self) {
        // Acquire and release `state` before notifying so idle workers
        // cannot observe stale state and park after `notify_all`.
        drop(self.state.lock());
        self.wake.notify_all();
    }

    fn is_stopped(&self) -> bool {
        self.state.lock().stopped
    }

    fn update_and_reset_preload(&self, edit: impl FnOnce(&mut BookState)) {
        let mut state = self.state.lock();
        state.preload = PreloadGate::Idle;
        edit(&mut state);
        drop(state);
        self.wake.notify_all();
    }

    fn apply_message(
        &self,
        message: ParentMsg,
        playback: &mut PlaybackRunner,
    ) {
        match message {
            ParentMsg::SetLimits(limits) => {
                match self.limits.set(limits) {
                    Ok(()) => tracing::debug!(
                        fit_box = ?limits.fit_box(),
                        max_texture_edge = limits.max_texture_edge(),
                        "limits set"
                    ),
                    Err(existing) => tracing::debug!(
                        fit_box = ?existing.fit_box(),
                        max_texture_edge = existing.max_texture_edge(),
                        "ignoring duplicate limits"
                    ),
                }
                self.wake_all();
            }
            // Ignore playback so animation frames keep the preload quiet
            // window and avoid locking `state`.
            ParentMsg::Playback(command) => playback.apply(command),
            ParentMsg::SetView {
                page,
                mode,
                direction,
            } => {
                self.update_and_reset_preload(|state| {
                    let prefetch = Prefetch::build(
                        browse::spread_for(page, mode, state.page_count),
                        direction,
                        mode,
                        state.page_count,
                    );
                    tracing::debug!(
                        origin = ?prefetch.spreads().first(),
                        motion = ?state.motion,
                        "prefetch set"
                    );
                    state.prefetch = Some(prefetch);
                });
            }
            ParentMsg::SetMotion(motion) => {
                self.update_and_reset_preload(|state| {
                    tracing::debug!(?motion, "motion");
                    state.motion = motion;
                });
            }
            ParentMsg::SetRequests(requests) => {
                self.update_and_reset_preload(|state| {
                    tracing::debug!(count = requests.len(), "requests");
                    state.requests_at = Instant::now();
                    state
                        .served
                        .retain(|request| requests.contains(request));
                    state.requests = requests;
                });
            }
        }
    }

    fn apply_messages(
        &self,
        commands: &IpcReceiver<ParentMsg>,
        playback: &mut PlaybackRunner,
    ) -> bool {
        use ipc_channel::IpcError;
        use ipc_channel::TryRecvError as IpcTryRecvError;

        loop {
            match commands.try_recv() {
                Ok(message) => self.apply_message(message, playback),
                Err(IpcTryRecvError::Empty) => return true,
                Err(IpcTryRecvError::IpcError(IpcError::Disconnected)) => {
                    return false;
                }
                Err(IpcTryRecvError::IpcError(error)) => {
                    tracing::warn!(%error, "worker ipc receive failed");
                    return false;
                }
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
    fn next_job(&self, source: &Source) -> Option<PageJob> {
        let mut state = self.state.lock();
        loop {
            if state.stopped {
                return None;
            }
            if self.limits.get().is_none() {
                self.wake.wait(&mut state);
                continue;
            }
            if let Some(job) = state.take_request_page(source) {
                return Some(job);
            }
            if state.has_pending_request() {
                self.wake.wait(&mut state);
                continue;
            }
            if state.in_flight_prefetch() >= MAX_PREFETCH_WORKERS {
                self.wake.wait(&mut state);
                continue;
            }
            let Some(PrefetchCandidate {
                page,
                role,
                preload,
            }) = state.next_prefetch_job(source)
            else {
                self.wake.wait(&mut state);
                continue;
            };
            if !preload || matches!(state.preload, PreloadGate::Open) {
                let motion = state.motion;
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

    fn needed_roles(&self, page: usize, hash: u128) -> Roles {
        self.state.lock().needed_roles(page, hash)
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
    ) -> Option<(u128, Option<Bytes>)> {
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
        let image = self.store.load(hash, role, limits.fit_box())?;
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
        let evicted = self.store.save(hash, role, image, limits.fit_box());
        if evicted.is_none() {
            tracing::debug!(page, ?role, "preview not persisted");
        }
        let mut state = self.state.lock();
        if let Some(evicted) = evicted {
            for evicted_hash in evicted {
                if let Some(roles) = state.stored.get_mut(&evicted_hash) {
                    roles.clear(role);
                }
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

    fn emit(&self, message: ChildMsg) {
        let sent = self.to_parent.lock().send(message);
        if let Err(error) = sent {
            tracing::debug!(?error, "parent ipc send failed");
            self.stop();
        }
    }

    fn deliver(
        &self,
        job: PageJob,
        role: Role,
        image: &Decoded,
        hit: bool,
        started: Instant,
    ) {
        let published = if self.state.lock().claim_publish(job.page, role) {
            tracing::debug!(
                page = job.page,
                ?role,
                width = image.width(),
                height = image.height(),
                "publish"
            );
            self.emit(ChildMsg::Ready {
                page: job.page,
                role,
                image: WireImage::from_decoded(image),
            });
            true
        } else {
            false
        };
        if published || job.kind.is_prefetch() {
            job.kind.report(job.page, role, hit, started.elapsed());
        }
    }

    fn record_failure(&self, page: usize, error: &dyn Display) {
        if self.state.lock().failed.insert(page) {
            self.emit(ChildMsg::Failed {
                page,
                fault: error.to_string(),
            });
        }
    }

    fn read_source(&self, source: &Source, page: usize) -> Option<Bytes> {
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

fn finish_fit(
    shared: &Shared,
    job: PageJob,
    hash: u128,
    fit: &Arc<Decoded>,
    origin: FitOrigin,
    started: Instant,
) {
    let page = job.page;
    let needed = shared.needed_roles(page, hash);
    if needed.contains(Role::Fit) {
        shared.save_memory_fit(hash, Arc::clone(fit));
        shared.deliver(
            job,
            Role::Fit,
            fit,
            origin == FitOrigin::Cache,
            started,
        );
    }

    if needed.contains(Role::Thumb)
        && let Some(derived) =
            shared.derive_for(page, Role::Thumb, fit, job.kind.label())
    {
        shared.deliver(job, Role::Thumb, &derived, false, started);
        shared.persist_if_missing(page, hash, Role::Thumb, &derived);
    }

    // Persist freshly derived `Fit` during tail preload so advancing into
    // `head` loads from the store without re-decoding.
    if origin == FitOrigin::Decode {
        shared.persist_if_missing(page, hash, Role::Fit, fit);
    }
}

fn produce_page(
    source: &Source,
    shared: &Shared,
    job: PageJob,
    max_threads: usize,
) {
    let started = Instant::now();
    let page = job.page;
    let Some((hash, mut source_bytes)) = shared.resolve_hash(source, page)
    else {
        return;
    };

    let mut cached_fit = shared.load_memory_fit(hash);
    if cached_fit.is_none() {
        // Check stored thumbnails before deriving from a cached `Fit` while
        // `MemoryCache` is cold so external store entries remain authoritative.
        let mut needed = shared.needed_roles(page, hash);
        if needed.contains(Role::Thumb)
            && let Some(loaded_thumb) =
                shared.load_stored(hash, Role::Thumb)
        {
            shared.deliver(job, Role::Thumb, &loaded_thumb, true, started);
            needed = shared.needed_roles(page, hash);
        }
        if includes_preview(needed)
            && let Some(loaded_fit) = shared.load_stored(hash, Role::Fit)
        {
            cached_fit = Some(Arc::new(loaded_fit));
        }
    }

    if let Some(fit) = cached_fit.as_ref() {
        finish_fit(
            shared,
            job,
            hash,
            fit,
            FitOrigin::Cache,
            started,
        );
    }

    let before_decode = shared.needed_roles(page, hash);
    let need_preview_decode =
        cached_fit.is_none() && includes_preview(before_decode);
    if !before_decode.contains(Role::Full) && !need_preview_decode {
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

    let after_decode = shared.needed_roles(page, hash);
    if after_decode.contains(Role::Full)
        && let Some(full) =
            shared.derive_for(page, Role::Full, &native, job.kind.label())
    {
        shared.deliver(job, Role::Full, &full, false, started);
    }

    if cached_fit.is_none() && includes_preview(after_decode) {
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
            PlaybackCommand::Start { id } => {
                self.steps.retain(|step| step.id().page != id.page);
                self.steps.push_back(PlaybackStep::Start { id });
            }
            PlaybackCommand::Next { id } => {
                self.steps.push_back(PlaybackStep::Next { id });
            }
            PlaybackCommand::Stop { id } => self.drop_session(id),
        }
    }

    fn pop_front(&mut self) -> Option<PlaybackStep> {
        self.steps.pop_front()
    }

    fn serve(
        &mut self,
        source: &Source,
        step: PlaybackStep,
        shared: &Shared,
    ) {
        match step {
            PlaybackStep::Start { id } => {
                self.start_playback(source, id, shared);
            }
            PlaybackStep::Next { id } => self.next_playback(id, shared),
        }
    }

    fn start_playback(
        &mut self,
        source: &Source,
        id: PlaybackId,
        shared: &Shared,
    ) {
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
        match source::catch_panic("playback open", || {
            decode::open_animation(bytes.clone(), name)
        }) {
            Ok(Some(animation)) => {
                let loops = animation.loops();
                self.playing_pages.insert(
                    page,
                    PlaybackSession {
                        id,
                        animation,
                        passes: 0,
                    },
                );
                tracing::debug!(?id, ?loops, "playback start");
            }
            Ok(None) => {
                tracing::debug!(?id, "no playback path");
                self.fail_playback(id, shared);
            }
            Err(error) => {
                tracing::warn!(?id, %error, "playback open failed");
                self.fail_playback(id, shared);
            }
        }
    }

    fn next_playback(&mut self, id: PlaybackId, shared: &Shared) {
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
        match source::catch_panic("playback frame", || session.advance()) {
            Ok(Some(frame)) => {
                let image =
                    match cache::derive(&frame.image, Role::Full, limits) {
                        Ok(image) => image,
                        Err(fault) => {
                            tracing::warn!(
                                ?id,
                                %fault,
                                "playback frame exceeds texture limit"
                            );
                            self.fail_playback(id, shared);
                            return;
                        }
                    };
                tracing::debug!(
                    ?id,
                    delay = ?frame.delay,
                    width = image.width(),
                    height = image.height(),
                    "playback frame"
                );
                shared.emit(ChildMsg::Playback(PlaybackEvent::Frame {
                    id,
                    delay: frame.delay,
                    image: WireImage::from_decoded(&image),
                }));
            }
            Ok(None) => {
                tracing::debug!(?id, "playback finished");
                self.playing_pages.remove(&page);
                shared.emit(ChildMsg::Playback(PlaybackEvent::Finished {
                    id,
                }));
            }
            Err(error) => {
                tracing::warn!(?id, %error, "playback frame failed");
                self.fail_playback(id, shared);
            }
        }
    }

    fn drop_session(&mut self, id: PlaybackId) {
        if self
            .playing_pages
            .get(&id.page)
            .is_some_and(|session| session.id == id)
        {
            self.playing_pages.remove(&id.page);
        }
        self.steps.retain(|step| step.id() != id);
    }

    fn fail_playback(&mut self, id: PlaybackId, shared: &Shared) {
        self.drop_session(id);
        shared.emit(ChildMsg::Playback(PlaybackEvent::Failed { id }));
    }
}

fn worker_loop(source: &Source, shared: &Shared, cpu_threads: usize) {
    let prefetch_threads =
        cpu_threads.div_ceil(PREFETCH_THREAD_DIVISOR).max(1);
    while let Some(job) = shared.next_job(source) {
        let _guard =
            DropGuard::new(job.page, |page| shared.finish_page_job(page));
        let max_threads = if job.kind.is_prefetch() {
            prefetch_threads
        } else {
            cpu_threads
        };
        if let JobKind::Request { requested_at } = job.kind {
            tracing::debug!(
                page = job.page,
                threads = max_threads,
                waited = ?requested_at.elapsed(),
                "request"
            );
        }
        produce_page(source, shared, job, max_threads);
    }
}

fn run(
    source: &Source,
    commands: &IpcReceiver<ParentMsg>,
    store: Store,
    to_parent: IpcSender<ChildMsg>,
) {
    use ipc_channel::IpcError;

    let cpu_threads = available_parallelism();
    let workers = cpu_threads.clamp(MIN_WORKERS, MAX_WORKERS);
    let shared = Arc::new(Shared::new(source.len(), store, to_parent));
    let wake_shared = Arc::clone(&shared);
    let _ready_guard = source.on_ready(move || wake_shared.wake_all());
    let mut playback = PlaybackRunner::new();

    thread::scope(|scope| {
        let shared = shared.as_ref();
        for index in 0..workers {
            let spawned = Builder::new()
                .name(format!("cy-worker-{index}"))
                .spawn_scoped(scope, || {
                    worker_loop(source, shared, cpu_threads);
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
            if !shared.apply_messages(commands, &mut playback)
                || shared.is_stopped()
            {
                break;
            }
            if let Some(step) = playback.pop_front() {
                playback.serve(source, step, shared);
                continue;
            }
            let message = match commands.recv() {
                Ok(message) => message,
                Err(IpcError::Disconnected) => break,
                Err(error) => {
                    tracing::warn!(%error, "worker ipc receive failed");
                    break;
                }
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
    source::catch_panic("decoder", || {
        decode::decode_with_threads(bytes, name, max_threads)
    })
}

pub fn main(args: &[OsString]) -> ExitCode {
    use rustix::process;
    use rustix::process::Pid;
    use rustix::process::Signal;

    use crate::protocol::BookInfo;
    use crate::protocol::Hello;

    let [server, path] = args else {
        tracing::error!(
            "--worker needs an ipc server name and a book path"
        );
        return ExitCode::FAILURE;
    };
    let path = Path::new(path);
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

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use std::fs;
    use std::thread::JoinHandle;

    use tempfile::TempDir;

    use super::*;
    use crate::cache::Store;
    use crate::protocol::THUMB_EDGE;
    use crate::testkit;
    use crate::testkit::TEST_FIT_BOX;

    /// Page edge for the tests that check derived and full dimensions:
    /// larger than the thumb edge, smaller than the fit box, so the
    /// thumb is scaled and the fit image is the page itself.
    const BIG: u32 = 1400;

    struct TestWorker {
        to_worker: Option<IpcSender<ParentMsg>>,
        worker: Option<JoinHandle<()>>,
        drain: Option<JoinHandle<()>>,
    }

    impl TestWorker {
        fn spawn_at(
            source: Arc<Source>,
            initial_view: Option<usize>,
            store_root: &Path,
            to_parent: Option<IpcSender<ChildMsg>>,
        ) -> Self {
            let (to_worker, commands) =
                ipc::channel::<ParentMsg>().expect("parent test channel");
            let (to_parent, drain) = to_parent.map_or_else(
                || {
                    let (event_sender, event_receiver) =
                        ipc::channel::<ChildMsg>()
                            .expect("child test channel");
                    let drain = Builder::new()
                        .name("test-worker-drain".into())
                        .spawn(
                            move || {
                                while event_receiver.recv().is_ok() {}
                            },
                        )
                        .expect("drain thread spawns");
                    (event_sender, Some(drain))
                },
                |sender| (sender, None),
            );
            let store = Store::at(store_root);
            let worker = Builder::new()
                .name("test-worker".into())
                .spawn(move || {
                    run(&source, &commands, store, to_parent);
                })
                .expect("test worker thread spawns");
            let handle = Self {
                to_worker: Some(to_worker),
                worker: Some(worker),
                drain,
            };
            handle.send(ParentMsg::SetLimits(testkit::limits()));
            if let Some(page) = initial_view {
                handle.set_view(page);
            }
            handle
        }

        fn send(&self, message: ParentMsg) {
            let Some(to_worker) = &self.to_worker else {
                return;
            };
            let Ok(()) = to_worker.send(message) else {
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
            drop(self.to_worker.take());
            if let Some(worker) = self.worker.take()
                && worker.join().is_err()
            {
                tracing::warn!("test worker panicked");
            }
            if let Some(drain) = self.drain.take()
                && drain.join().is_err()
            {
                tracing::warn!("test drain thread panicked");
            }
        }
    }

    fn start_playback(worker: &TestWorker, page: usize, generation: u32) {
        worker.playback(PlaybackCommand::Start {
            id: PlaybackId { page, generation },
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

    fn long_edge(image: &WireImage) -> u32 {
        image.meta.width.max(image.meta.height)
    }

    fn first_pixel(image: &WireImage) -> [u8; 4] {
        use crate::protocol::WirePixels;

        let WirePixels::Rgba8(bytes) = &image.pixels else {
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
        messages: IpcReceiver<ChildMsg>,
        seen: Mutex<Vec<ChildMsg>>,
    }

    impl Updates {
        fn new() -> (Self, IpcSender<ChildMsg>) {
            let (sender, messages) =
                ipc::channel().expect("child test channel");
            let events = Self {
                messages,
                seen: Mutex::new(Vec::new()),
            };
            (events, sender)
        }

        fn drain(&self) {
            while let Ok(event) = self.messages.try_recv() {
                self.seen.lock().push(event);
            }
        }

        fn ready(
            &self,
            what: &str,
            want: impl Fn(usize, Role) -> bool,
        ) -> WireImage {
            testkit::eventually(what, || {
                self.drain();
                let seen = self.seen.lock();
                seen.iter().find_map(|event| match event {
                    ChildMsg::Ready { page, role, image }
                        if want(*page, *role) =>
                    {
                        Some(image.clone())
                    }
                    ChildMsg::Ready { .. }
                    | ChildMsg::Failed { .. }
                    | ChildMsg::Playback(_) => None,
                })
            })
        }

        fn failure(&self, page: usize) {
            testkit::eventually("page never failed", || {
                self.drain();
                let seen = self.seen.lock();
                seen.iter().find_map(|event| match event {
                    ChildMsg::Failed { page: failed, .. }
                        if *failed == page =>
                    {
                        Some(())
                    }
                    ChildMsg::Ready { .. }
                    | ChildMsg::Failed { .. }
                    | ChildMsg::Playback(_) => None,
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
                    matches!(event, ChildMsg::Ready { page: delivered, role, .. }
                        if *delivered == page && want(*role))
                })
                .count()
        }

        fn failures(&self, page: usize) -> usize {
            self.drain();
            let seen = self.seen.lock();
            seen.iter()
                .filter(|event| {
                    matches!(event, ChildMsg::Failed { page: failed, .. } if *failed == page)
                })
                .count()
        }

        /// The `offset`-th frame delivered for `page`, counting from
        /// zero in arrival order.
        fn frame(
            &self,
            page: usize,
            offset: usize,
        ) -> (Duration, WireImage) {
            testkit::eventually("no animation frame", || {
                self.drain();
                let seen = self.seen.lock();
                seen.iter()
                    .filter_map(|event| match event {
                        ChildMsg::Playback(PlaybackEvent::Frame {
                            id,
                            delay,
                            image,
                        }) if id.page == page => {
                            Some((*delay, image.clone()))
                        }
                        ChildMsg::Ready { .. }
                        | ChildMsg::Failed { .. }
                        | ChildMsg::Playback(_) => None,
                    })
                    .nth(offset)
            })
        }

        fn finished(&self, page: usize) {
            testkit::eventually("no animation finish", || {
                self.drain();
                let seen = self.seen.lock();
                seen.iter()
                    .any(|event| {
                        matches!(
                            event,
                            ChildMsg::Playback(PlaybackEvent::Finished {
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
        let source = Arc::new(
            Source::open(Path::new(testkit::SEVENZ_FIXTURE))
                .expect("7z opens"),
        );
        let root = TempDir::new().expect("create temp dir");
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            Some(to_parent),
        );

        worker.set_requests(vec![request_thumb(0)]);
        let thumb = events
            .ready("the archive page never served", |page, role| {
                page == 0 && is_thumb(role)
            });
        assert_eq!((thumb.meta.width, thumb.meta.height), (40, 30));
    }

    #[test]
    fn a_target_is_served_and_persisted() {
        let (_dir, source) = page_dir(1, BIG);
        let root = TempDir::new().expect("create temp dir");
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            Some(to_parent),
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
        assert_eq!(long_edge(&thumb), THUMB_EDGE);
        assert_eq!(
            (view.meta.width, view.meta.height),
            (BIG, BIG),
            "the fit image stays unscaled inside the target box"
        );
        // `publish` runs before `persist`; join the worker so both files
        // finish persisting before inspecting the store.
        drop(worker);

        let store = Store::at(root.path());
        let hash = page_hash(&source, 0);
        assert!(
            store.load(hash, Role::Thumb, TEST_FIT_BOX).is_some(),
            "the thumb was not persisted"
        );
        assert!(
            store.load(hash, Role::Fit, TEST_FIT_BOX).is_some(),
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
            None,
        );
        let store = Store::at(root.path());
        // Wait until all head spreads are fully produced.
        for page in 0..PREFETCH_HEAD_SPREADS {
            testkit::eventually("the head never reached the store", || {
                store
                    .load(
                        page_hash(&source, page),
                        Role::Thumb,
                        TEST_FIT_BOX,
                    )
                    .is_some()
                    .then_some(())
            });
        }
        let head_done = Instant::now();
        // The first page beyond the head must not start until the quiet
        // window has elapsed after the head finishes.
        testkit::eventually("the tail never preloaded", || {
            store
                .load(
                    page_hash(&source, PREFETCH_HEAD_SPREADS),
                    Role::Thumb,
                    TEST_FIT_BOX,
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
        let (events, to_parent) = Updates::new();
        let _worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            Some(to_parent),
        );

        events.failure(0);
        // The failure leaves the rest of the prefetch running: pages behind it
        // still reach the store.
        let store = Store::at(root.path());
        testkit::eventually("the failure stalled the prefetch", || {
            store
                .load(page_hash(&source, 4), Role::Thumb, TEST_FIT_BOX)
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
        let (_dir, source) = page_dir(1, BIG);
        let root = TempDir::new().expect("create temp dir");
        let (events, to_parent) = Updates::new();
        let _worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            Some(to_parent),
        );

        let store = Store::at(root.path());
        testkit::eventually(
            "the prefetch never produced the head fit image",
            || {
                store
                    .load(page_hash(&source, 0), Role::Fit, TEST_FIT_BOX)
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
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            Some(to_parent),
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
        testkit::eventually("the re-added fit image was never served", || {
            (events.delivered(0, is_fit) >= 2).then_some(())
        });
    }

    #[test]
    fn a_full_target_carries_the_page_pixels() {
        let (_dir, source) = page_dir(1, BIG);
        let root = TempDir::new().expect("create temp dir");
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            Some(to_parent),
        );

        worker.set_requests(vec![request_full(0)]);
        let image = events
            .ready("the full target was never honoured", |page, role| {
                page == 0 && role == Role::Full
            });
        assert_eq!(
            (image.meta.width, image.meta.height),
            (BIG, BIG),
            "full must carry the page's own pixels"
        );
    }

    #[test]
    fn animation_frames_stream_in_order() {
        use image::codecs::gif::Repeat;

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
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            Some(to_parent),
        );

        start_playback(&worker, 0, 9);
        for (offset, color) in colors.into_iter().enumerate() {
            next_frame(&worker, 0, 9);
            let (delay, image) = events.frame(0, offset);
            assert_eq!(delay, Duration::from_millis(50));
            assert_eq!((image.meta.width, image.meta.height), (8, 8));
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
        use image::codecs::gif::Repeat;

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
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            Some(to_parent),
        );

        start_playback(&worker, 0, 3);
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
        let (first, first_to_parent) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            store_root,
            Some(first_to_parent),
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
        assert_eq!(long_edge(&thumb), THUMB_EDGE);
        assert_eq!(
            (view.meta.width, view.meta.height),
            (BIG, BIG),
            "the fit image stays unscaled inside the target box"
        );
        drop(worker);

        let hash = page_hash(&source, 0);
        let store = Store::at(store_root);
        assert!(
            store.load(hash, Role::Thumb, TEST_FIT_BOX).is_some(),
            "the thumb was not persisted"
        );
        assert!(
            store.load(hash, Role::Fit, TEST_FIT_BOX).is_some(),
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
                .save(hash, Role::Thumb, &sentinel, TEST_FIT_BOX)
                .is_some()
        );
        assert!(
            store
                .save(hash, Role::Fit, &sentinel, TEST_FIT_BOX)
                .is_some()
        );
        drop(store);

        // A second worker on the same root loads the cached roles.
        let (second, second_to_parent) = Updates::new();
        let warm = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            store_root,
            Some(second_to_parent),
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
            (warm_thumb.meta.width, warm_thumb.meta.height),
            (7, 7),
            "the thumb missed the warm store"
        );
        assert_eq!(
            (warm_view.meta.width, warm_view.meta.height),
            (7, 7),
            "the fit image missed the warm store"
        );

        // Loading leaves the roles it read unchanged.
        let kept = Store::at(store_root)
            .load(hash, Role::Fit, TEST_FIT_BOX)
            .expect("the sentinel fit stays");
        assert_eq!((kept.width(), kept.height()), (7, 7));
    }

    #[test]
    fn store_hit_with_known_hash_skips_source_read() {
        let (dir, source) = page_dir(2, BIG);
        let root = TempDir::new().expect("create temp dir");
        let requested_hash = page_hash(&source, 0);
        let prefetched_hash = page_hash(&source, 1);
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            Some(0),
            root.path(),
            Some(to_parent),
        );

        worker.set_requests(vec![request_fit(0)]);
        events
            .ready("initial fit", |page, role| page == 0 && is_fit(role));
        let store = Store::at(root.path());
        testkit::eventually("warmed fit and thumb never persisted", || {
            (store
                .load(requested_hash, Role::Fit, TEST_FIT_BOX)
                .is_some()
                && store
                    .load(requested_hash, Role::Thumb, TEST_FIT_BOX)
                    .is_some()
                && store
                    .load(prefetched_hash, Role::Fit, TEST_FIT_BOX)
                    .is_some())
            .then_some(())
        });

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
    fn tail_prefetch_with_stored_thumb_skips_source_decode() {
        let tail_page = PREFETCH_HEAD_SPREADS;
        let (dir, _) = page_dir(tail_page, 8);
        let corrupt = b"not an image";
        fs::write(dir.path().join(format!("p{tail_page:04}.png")), corrupt)
            .expect("write corrupt tail page");
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let root = TempDir::new().expect("create temp dir");
        let tail_hash = cache::content_hash(corrupt);
        let sentinel = decode::decode(
            &testkit::tiny_png(4, 4, [1, 2, 3, 255]),
            "sentinel.png",
        )
        .expect("sentinel decodes");
        let store = Store::at(root.path());
        assert!(
            store
                .save(tail_hash, Role::Thumb, &sentinel, TEST_FIT_BOX)
                .is_some()
        );
        drop(store);

        let shared =
            Shared::new(source.len(), Store::at(root.path()), Updates::new().1);
        shared
            .limits
            .set(testkit::limits())
            .expect("limits unset");
        shared.update_and_reset_preload(|state| {
            state.prefetch = Some(Prefetch::build(
                browse::spread_for(0, ReadingMode::Normal, state.page_count),
                Direction::Forward,
                ReadingMode::Normal,
                state.page_count,
            ));
        });

        produce_page(
            &source,
            &shared,
            PageJob {
                page: tail_page,
                kind: JobKind::Prefetch { preload: true },
            },
            1,
        );
        assert!(!shared.state.lock().failed.contains(&tail_page));
        assert!(shared
            .state
            .lock()
            .stored_for_hash(tail_hash)
            .contains(Role::Thumb));
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
                .save(hash, Role::Fit, &landscape_fit, TEST_FIT_BOX)
                .is_some()
        );
        drop(store);

        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            None,
            root.path(),
            Some(to_parent),
        );
        worker.set_requests(vec![request_thumb(0)]);
        let thumb = events
            .ready("derived thumb never arrived", |page, role| {
                page == 0 && is_thumb(role)
            });
        assert_eq!(thumb.meta.width, THUMB_EDGE);
        assert_eq!(thumb.meta.height * 2, thumb.meta.width);
    }

    #[test]
    fn fit_and_full_requested_together_serve_both() {
        const PAGE_WIDTH: u32 = TEST_FIT_BOX.0 + 40;
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
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            None,
            root.path(),
            Some(to_parent),
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
            (fit.meta.width, fit.meta.height),
            (TEST_FIT_BOX.0, PAGE_HEIGHT)
        );
        assert_eq!(
            (full.meta.width, full.meta.height),
            (PAGE_WIDTH, PAGE_HEIGHT)
        );
        assert_eq!(events.delivered(0, |role| role == Role::Full), 1);
    }

    fn page_count(count: usize) -> NonZeroUsize {
        NonZeroUsize::new(count).expect("test book is non-empty")
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

    #[test]
    fn unready_archive_slots_do_not_starve_and_wake_when_filled() {
        const WORKER_PARK_SETTLE: Duration = Duration::from_millis(20);

        let names = (0..=MAX_WORKERS)
            .map(|index| format!("p{index:03}.png"))
            .collect();
        let (source, slots) =
            Source::from_test_slots(names).expect("test archive");
        let source = Arc::new(source);
        slots.fill(
            0,
            Bytes::from(testkit::tiny_png(16, 16, [10, 20, 30, 255])),
        );

        let root = TempDir::new().expect("create temp dir");
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn_at(
            Arc::clone(&source),
            None,
            root.path(),
            Some(to_parent),
        );

        worker.set_requests(
            (1..=MAX_WORKERS).chain([0]).map(request_fit).collect(),
        );
        events.ready(
            "ready page 0 starved behind unready requests",
            |page, role| page == 0 && is_fit(role),
        );

        thread::sleep(WORKER_PARK_SETTLE);
        slots.fill(
            1,
            Bytes::from(testkit::tiny_png(12, 12, [40, 50, 60, 255])),
        );
        events.ready(
            "worker never woke when requested slot filled",
            |page, role| page == 1 && is_fit(role),
        );
        drop(worker);
    }
}
