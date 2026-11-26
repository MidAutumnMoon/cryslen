//! Per-book decode worker (`cy --worker`): 3-tier 2-worker decode pool and IPC loop.

use std::borrow::Cow;
use std::collections::BTreeMap;
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
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread;
use std::thread::Builder;
use std::time::Instant;

use bytes::Bytes;
use ipc_channel::ipc;
use ipc_channel::ipc::IpcReceiver;
use ipc_channel::ipc::IpcSender;
use parking_lot::Condvar;
use parking_lot::Mutex;

use crate::browse::Request;
use crate::browse::WorkRole;
use crate::cache;
use crate::decode;
use crate::decode::Animation;
use crate::decode::DecodeOutcome;
use crate::decode::Decoded;
use crate::decode::Frame;
use crate::protocol::ChildMsg;
use crate::protocol::ImageLimits;
use crate::protocol::ParentMsg;
use crate::protocol::PlaybackCommand;
use crate::protocol::PlaybackEvent;
use crate::protocol::PlaybackId;
use crate::protocol::WorkPlan;
use crate::protocol::WireImage;
use crate::protocol::WireThumb;
use crate::source;
use crate::source::Source;

const DEFAULT_CPU_THREADS: usize = 2;

const NATIVE_CACHE_CAP_BYTES: usize = 320 * 1024 * 1024;
const NATIVE_CACHE_MAX_ENTRIES: usize = 8;

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

#[derive(Default)]
struct NativeCache {
    total_bytes: usize,
    entries: HashMap<usize, Arc<Decoded>>,
    order: VecDeque<usize>,
}

impl NativeCache {
    fn get(&mut self, page: usize) -> Option<Arc<Decoded>> {
        let image = self.entries.get(&page).map(Arc::clone)?;
        self.touch(page);
        Some(image)
    }

    fn insert(&mut self, page: usize, image: Arc<Decoded>) {
        let bytes = image.resident_bytes();
        if bytes > NATIVE_CACHE_CAP_BYTES {
            return;
        }
        if let Some(previous) = self.entries.insert(page, image) {
            self.total_bytes =
                self.total_bytes.saturating_sub(previous.resident_bytes());
        }
        self.total_bytes = self.total_bytes.saturating_add(bytes);
        self.touch(page);
        while self.total_bytes > NATIVE_CACHE_CAP_BYTES
            || self.entries.len() > NATIVE_CACHE_MAX_ENTRIES
        {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(removed) = self.entries.remove(&oldest) {
                self.total_bytes =
                    self.total_bytes.saturating_sub(removed.resident_bytes());
            }
        }
    }

    fn touch(&mut self, page: usize) {
        self.order.retain(|cached_page| *cached_page != page);
        self.order.push_back(page);
    }
}

#[derive(Clone, Copy)]
struct ThreadBudget {
    single: usize,
    pair: usize,
    thumb: usize,
}

impl ThreadBudget {
    fn for_cpus(cpu_threads: usize) -> (usize, Self) {
        const SINGLE_WORKER_MAX_CPUS: usize = 2;
        const MAX_DECODE_WORKERS: usize = 2;
        const THUMB_THREAD_DIVISOR: usize = 4;

        let workers = if cpu_threads <= SINGLE_WORKER_MAX_CPUS {
            1
        } else {
            MAX_DECODE_WORKERS
        };
        let pair = cpu_threads.checked_div(workers).unwrap_or(1).max(1);
        let thumb = cpu_threads
            .checked_div(THUMB_THREAD_DIVISOR)
            .unwrap_or(1)
            .max(1);
        (
            workers,
            Self {
                single: cpu_threads,
                pair,
                thumb,
            },
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JobTier {
    Current,
    Lookahead,
    Thumb,
}

struct InFlight {
    tier: JobTier,
    cancel: Arc<AtomicBool>,
}

impl InFlight {
    fn is_active(&self) -> bool {
        !self.cancel.load(Ordering::Relaxed)
    }

    fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

struct PageJob {
    page: usize,
    tier: JobTier,
    threads: usize,
    requested_at: Instant,
    cancel: Arc<AtomicBool>,
}

impl PageJob {
    fn is_canceled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

fn plan_contains(plan: &WorkPlan, request: Request) -> bool {
    plan.current.contains(&request)
        || (request.role == WorkRole::Fit
            && plan.lookahead.contains(&request.page))
}

struct BookState {
    plan: WorkPlan,
    work_updated_at: Instant,
    /// Requests already handed over since they last entered the plan.
    served: HashSet<Request>,
    /// Background thumbnail pages already handed over since entering `plan.thumbs`.
    served_thumbs: HashSet<usize>,
    /// Pages whose first failure settled them for the session.
    failed: HashSet<usize>,
    /// Pages whose `WorkRole::Full` derivation failed and must not retry.
    failed_full: HashSet<usize>,
    in_flight: BTreeMap<usize, InFlight>,
    native_cache: NativeCache,
    stopped: bool,
}

impl BookState {
    fn new() -> Self {
        Self {
            plan: WorkPlan::default(),
            work_updated_at: Instant::now(),
            served: HashSet::new(),
            served_thumbs: HashSet::new(),
            failed: HashSet::new(),
            failed_full: HashSet::new(),
            in_flight: BTreeMap::new(),
            native_cache: NativeCache::default(),
            stopped: false,
        }
    }

    fn role_failed(&self, page: usize, role: WorkRole) -> bool {
        self.failed.contains(&page)
            || (role == WorkRole::Full && self.failed_full.contains(&page))
    }

    fn is_unsettled(&self, request: Request) -> bool {
        !self.served.contains(&request)
            && !self.role_failed(request.page, request.role)
    }

    fn is_unsettled_fit(&self, page: usize) -> bool {
        self.is_unsettled(Request {
            page,
            role: WorkRole::Fit,
        })
    }

    fn plan_contains_current_page(&self, page: usize) -> bool {
        self.plan.current.iter().any(|request| request.page == page)
    }

    fn active_plan_tier(&self) -> Option<JobTier> {
        if self.plan.current.iter().any(|&request| self.is_unsettled(request)) {
            Some(JobTier::Current)
        } else if self
            .plan
            .lookahead
            .iter()
            .any(|&page| self.is_unsettled_fit(page))
        {
            Some(JobTier::Lookahead)
        } else if self
            .plan
            .thumbs
            .iter()
            .any(|&page| self.wants_unserved_thumb(page))
        {
            Some(JobTier::Thumb)
        } else {
            None
        }
    }

    fn has_active_tier(&self, tier: JobTier) -> bool {
        self.in_flight
            .values()
            .any(|job| job.tier == tier && job.is_active())
    }

    fn has_any_active(&self) -> bool {
        self.in_flight.values().any(InFlight::is_active)
    }

    fn wants_unserved(&self, page: usize, role: WorkRole) -> bool {
        let request = Request { page, role };
        plan_contains(&self.plan, request) && self.is_unsettled(request)
    }

    fn claim_publish(&mut self, page: usize, role: WorkRole) -> bool {
        if !self.wants_unserved(page, role) {
            return false;
        }
        self.served.insert(Request { page, role });
        true
    }

    fn wants_unserved_thumb(&self, page: usize) -> bool {
        self.plan.thumbs.contains(&page)
            && !self.served_thumbs.contains(&page)
            && !self.failed.contains(&page)
    }

    fn claim_publish_thumb(&mut self, page: usize) -> bool {
        if !self.wants_unserved_thumb(page) {
            return false;
        }
        self.served_thumbs.insert(page);
        true
    }

    fn apply_work(&mut self, plan: WorkPlan) {
        self.work_updated_at = Instant::now();
        self.served.retain(|&request| plan_contains(&plan, request));
        self.served_thumbs.retain(|page| plan.thumbs.contains(page));
        self.plan = plan;
        let active_tier = self.active_plan_tier();
        for (&page, job) in &self.in_flight {
            let keep = match job.tier {
                JobTier::Current | JobTier::Lookahead => match active_tier {
                    Some(JobTier::Current) => {
                        self.plan_contains_current_page(page)
                    }
                    Some(JobTier::Lookahead) => {
                        self.plan.lookahead.contains(&page)
                    }
                    Some(JobTier::Thumb) | None => false,
                },
                JobTier::Thumb => {
                    active_tier == Some(JobTier::Thumb)
                        && self.plan.thumbs.contains(&page)
                }
            };
            if !keep {
                job.cancel();
            }
        }
    }

    fn stop(&mut self) {
        self.stopped = true;
        for job in self.in_flight.values() {
            job.cancel();
        }
    }

    fn launch(&mut self, page: usize, tier: JobTier, threads: usize) -> PageJob {
        let cancel = Arc::new(AtomicBool::new(false));
        self.in_flight.insert(
            page,
            InFlight {
                tier,
                cancel: Arc::clone(&cancel),
            },
        );
        PageJob {
            page,
            tier,
            threads,
            requested_at: self.work_updated_at,
            cancel,
        }
    }

    fn take_job(
        &mut self,
        source: &Source,
        budget: ThreadBudget,
    ) -> Option<PageJob> {
        let mut current_pages = Vec::new();
        for &request in &self.plan.current {
            if self.is_unsettled(request)
                && !current_pages.contains(&request.page)
            {
                current_pages.push(request.page);
            }
        }
        if let Some(&page) = current_pages.iter().find(|&&page| {
            !self.in_flight.contains_key(&page) && source.is_ready(page)
        }) {
            let threads =
                if current_pages.len() == 1 && !self.has_any_active() {
                    budget.single
                } else {
                    budget.pair
                };
            return Some(self.launch(page, JobTier::Current, threads));
        }
        if !current_pages.is_empty() || self.has_active_tier(JobTier::Current)
        {
            return None;
        }

        if let Some(&page) = self.plan.lookahead.iter().find(|&&page| {
            self.is_unsettled_fit(page)
                && !self.in_flight.contains_key(&page)
                && source.is_ready(page)
        }) {
            return Some(self.launch(page, JobTier::Lookahead, budget.pair));
        }
        let lookahead_active =
            self.plan.lookahead.iter().any(|&page| self.is_unsettled_fit(page))
                || self.has_active_tier(JobTier::Lookahead);
        if lookahead_active {
            return None;
        }

        if self.has_active_tier(JobTier::Thumb) {
            return None;
        }
        let &page = self.plan.thumbs.iter().find(|&&page| {
            self.wants_unserved_thumb(page)
                && !self.in_flight.contains_key(&page)
                && source.is_ready(page)
        })?;
        Some(self.launch(page, JobTier::Thumb, budget.thumb))
    }
}

struct Shared {
    limits: OnceLock<ImageLimits>,
    budget: ThreadBudget,
    state: Mutex<BookState>,
    wake: Condvar,
    to_parent: Mutex<IpcSender<ChildMsg>>,
}

impl Shared {
    fn new(to_parent: IpcSender<ChildMsg>, budget: ThreadBudget) -> Self {
        Self {
            limits: OnceLock::new(),
            budget,
            state: Mutex::new(BookState::new()),
            wake: Condvar::new(),
            to_parent: Mutex::new(to_parent),
        }
    }

    fn finish_page_job(&self, page: usize) {
        self.state.lock().in_flight.remove(&page);
        self.wake.notify_all();
    }

    fn stop(&self) {
        self.state.lock().stop();
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
            ParentMsg::Playback(command) => playback.apply(command),
            ParentMsg::SetWork(plan) => {
                self.apply_work(plan);
            }
        }
    }

    fn apply_work(&self, plan: WorkPlan) {
        tracing::debug!(
            current = plan.current.len(),
            lookahead = plan.lookahead.len(),
            thumbs = plan.thumbs.len(),
            "work plan"
        );
        self.state.lock().apply_work(plan);
        self.wake.notify_all();
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
            if let Some(job) = state.take_job(source, self.budget) {
                return Some(job);
            }
            self.wake.wait(&mut state);
        }
    }

    fn wanted_roles(&self, page: usize) -> (bool, bool) {
        let state = self.state.lock();
        (
            state.wants_unserved(page, WorkRole::Full),
            state.wants_unserved(page, WorkRole::Fit),
        )
    }

    fn load_native(&self, page: usize) -> Option<Arc<Decoded>> {
        self.state.lock().native_cache.get(page)
    }

    fn save_native(&self, page: usize, image: Arc<Decoded>) {
        self.state.lock().native_cache.insert(page, image);
    }

    fn derive_for<'source>(
        &self,
        page: usize,
        role: WorkRole,
        source: &'source Decoded,
        threads: usize,
    ) -> Option<Cow<'source, Decoded>> {
        let limits = self.limits()?;
        let started = Instant::now();
        match cache::derive(source, role, limits, threads) {
            Ok(derived) => {
                tracing::debug!(
                    page,
                    ?role,
                    threads,
                    elapsed = ?started.elapsed(),
                    "derived"
                );
                Some(derived)
            }
            Err(fault) => {
                match role {
                    WorkRole::Full => {
                        self.state.lock().failed_full.insert(page);
                    }
                    WorkRole::Fit => {
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
        job: &PageJob,
        role: WorkRole,
        image: &Decoded,
        thumb: Option<WireThumb>,
        native_cache_hit: bool,
        started: Instant,
    ) {
        if !self.state.lock().claim_publish(job.page, role) {
            return;
        }
        tracing::debug!(
            page = job.page,
            ?role,
            width = image.width(),
            height = image.height(),
            has_thumb = thumb.is_some(),
            "publish"
        );
        self.emit(ChildMsg::Ready {
            page: job.page,
            role,
            image: WireImage::from_decoded(image),
            thumb,
        });
        tracing::debug!(
            page = job.page,
            ?role,
            tier = ?job.tier,
            native_cache_hit,
            waited = ?job.requested_at.elapsed(),
            elapsed = ?started.elapsed(),
            "served"
        );
    }

    fn deliver_thumb(
        &self,
        job: &PageJob,
        thumb: WireThumb,
        started: Instant,
    ) {
        if !self.state.lock().claim_publish_thumb(job.page) {
            return;
        }
        tracing::debug!(
            page = job.page,
            bytes = thumb.webp.len(),
            threads = job.threads,
            waited = ?job.requested_at.elapsed(),
            elapsed = ?started.elapsed(),
            "thumb"
        );
        self.emit(ChildMsg::Thumb {
            page: job.page,
            thumb,
        });
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
}

fn decode_page_source(
    source: &Source,
    shared: &Shared,
    job: &PageJob,
) -> Option<Decoded> {
    let page = job.page;
    let bytes = shared.read_source(source, page)?;
    let Some(name) = source.name(page) else {
        shared.record_failure(page, &"no such page");
        return None;
    };
    let decode_started = Instant::now();
    let decoded = match source::catch_panic("decoder", || {
        decode::decode_cancellable(&bytes, name, job.threads, &job.cancel)
    }) {
        Ok(DecodeOutcome::Ready(decoded)) => decoded,
        Ok(DecodeOutcome::Canceled) => {
            tracing::debug!(
                page,
                tier = ?job.tier,
                threads = job.threads,
                elapsed = ?decode_started.elapsed(),
                "decode canceled"
            );
            return None;
        }
        Err(error) => {
            shared.record_failure(page, &error);
            return None;
        }
    };
    tracing::debug!(
        page,
        tier = ?job.tier,
        threads = job.threads,
        elapsed = ?decode_started.elapsed(),
        "decoded"
    );
    Some(decoded)
}

fn load_or_decode(
    source: &Source,
    shared: &Shared,
    job: &PageJob,
    cache_native: bool,
) -> Option<(Arc<Decoded>, bool)> {
    if let Some(cached) = shared.load_native(job.page) {
        return Some((cached, true));
    }
    let decoded = decode_page_source(source, shared, job)?;
    let native = Arc::new(decoded);
    if cache_native {
        shared.save_native(job.page, Arc::clone(&native));
    }
    Some((native, false))
}

fn produce_page(source: &Source, shared: &Shared, job: &PageJob) {
    const ATTACHED_THUMB_THREADS: usize = 1;

    let started = Instant::now();
    let page = job.page;
    let is_thumb = job.tier == JobTier::Thumb;
    let Some((native, native_cache_hit)) =
        load_or_decode(source, shared, job, !is_thumb)
    else {
        return;
    };
    if job.is_canceled() {
        return;
    }
    if is_thumb {
        match cache::derive_thumb(&native, job.threads) {
            Ok(thumb) if !job.is_canceled() => {
                shared.deliver_thumb(job, thumb.into_wire(), started);
            }
            Ok(_) => {}
            Err(error) => {
                tracing::debug!(page, %error, "thumb derive failed");
            }
        }
        return;
    }

    let (want_full, want_fit) = shared.wanted_roles(page);
    if want_full
        && let Some(full) =
            shared.derive_for(page, WorkRole::Full, &native, job.threads)
    {
        shared.deliver(
            job,
            WorkRole::Full,
            &full,
            None,
            native_cache_hit,
            started,
        );
    }

    if job.is_canceled() {
        return;
    }

    if want_fit
        && let Some(fit) =
            shared.derive_for(page, WorkRole::Fit, &native, job.threads)
    {
        let thumb = match cache::derive_thumb(&fit, ATTACHED_THUMB_THREADS) {
            Ok(thumb) => Some(thumb.into_wire()),
            Err(error) => {
                tracing::debug!(page, %error, "thumb derive failed");
                None
            }
        };
        shared.deliver(
            job,
            WorkRole::Fit,
            &fit,
            thumb,
            native_cache_hit,
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
                let image = match cache::derive(
                    &frame.image,
                    WorkRole::Full,
                    limits,
                    1,
                ) {
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

fn worker_loop(source: &Source, shared: &Shared) {
    while let Some(job) = shared.next_job(source) {
        let _guard =
            DropGuard::new(job.page, |page| shared.finish_page_job(page));
        tracing::debug!(
            page = job.page,
            tier = ?job.tier,
            threads = job.threads,
            waited = ?job.requested_at.elapsed(),
            "job"
        );
        produce_page(source, shared, &job);
    }
}

fn run(
    source: &Source,
    commands: &IpcReceiver<ParentMsg>,
    to_parent: IpcSender<ChildMsg>,
) {
    use ipc_channel::IpcError;

    let (workers, budget) = ThreadBudget::for_cpus(available_parallelism());
    let shared = Arc::new(Shared::new(to_parent, budget));
    let wake_shared = Arc::clone(&shared);
    let _ready_guard = source.on_ready(move || wake_shared.wake_all());
    let mut playback = PlaybackRunner::new();

    thread::scope(|scope| {
        let shared = shared.as_ref();
        for index in 0..workers {
            let spawned = Builder::new()
                .name(format!("cy-worker-{index}"))
                .spawn_scoped(scope, || {
                    worker_loop(source, shared);
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

    run(&opened, &parent_rx, child_tx);
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
    use std::time::Duration;

    use tempfile::TempDir;

    use super::*;
    use crate::protocol::THUMB_EDGE;
    use crate::testkit;
    use crate::testkit::TEST_FIT_BOX;

    /// Page edge larger than `THUMB_EDGE` and smaller than `TEST_FIT_BOX`,
    /// so `Fit` stays unscaled while the attached thumb scales down.
    const UNSCALED_FIT_EDGE: u32 = 1400;

    struct TestWorker {
        to_worker: Option<IpcSender<ParentMsg>>,
        worker: Option<JoinHandle<()>>,
        drain: Option<JoinHandle<()>>,
    }

    impl TestWorker {
        fn spawn(
            source: Arc<Source>,
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
            let worker = Builder::new()
                .name("test-worker".into())
                .spawn(move || {
                    run(&source, &commands, to_parent);
                })
                .expect("test worker thread spawns");
            let handle = Self {
                to_worker: Some(to_worker),
                worker: Some(worker),
                drain,
            };
            handle.send(ParentMsg::SetLimits(testkit::limits()));
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

        fn set_work(&self, plan: WorkPlan) {
            self.send(ParentMsg::SetWork(plan));
        }

        fn set_current(&self, current: Vec<Request>) {
            self.set_work(WorkPlan {
                current,
                ..WorkPlan::default()
            });
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

    fn first_pixel(image: &WireImage) -> [u8; 4] {
        use crate::protocol::WirePixels;

        let WirePixels::Rgba8(bytes) = &image.pixels else {
            panic!("a gif frame decodes to rgba");
        };
        let mut pixel = [0; 4];
        pixel.copy_from_slice(bytes.get(..4).expect("one pixel"));
        pixel
    }

    fn request_fit(page: usize) -> Request {
        Request {
            page,
            role: WorkRole::Fit,
        }
    }

    fn request_full(page: usize) -> Request {
        Request {
            page,
            role: WorkRole::Full,
        }
    }

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
            want: impl Fn(usize, WorkRole) -> bool,
        ) -> (WireImage, Option<WireThumb>) {
            testkit::eventually(what, || {
                self.drain();
                let seen = self.seen.lock();
                seen.iter().find_map(|event| match event {
                    ChildMsg::Ready {
                        page,
                        role,
                        image,
                        thumb,
                    } if want(*page, *role) => {
                        Some((image.clone(), thumb.clone()))
                    }
                    ChildMsg::Ready { .. }
                    | ChildMsg::Thumb { .. }
                    | ChildMsg::Failed { .. }
                    | ChildMsg::Playback(_) => None,
                })
            })
        }

        fn thumb(&self, what: &str, want_page: usize) -> WireThumb {
            testkit::eventually(what, || {
                self.drain();
                let seen = self.seen.lock();
                seen.iter().find_map(|event| match event {
                    ChildMsg::Thumb { page, thumb } if *page == want_page => {
                        Some(thumb.clone())
                    }
                    ChildMsg::Ready { .. }
                    | ChildMsg::Thumb { .. }
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
                    | ChildMsg::Thumb { .. }
                    | ChildMsg::Failed { .. }
                    | ChildMsg::Playback(_) => None,
                })
            });
        }

        fn delivered(
            &self,
            page: usize,
            want: impl Fn(WorkRole) -> bool,
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

        fn ready_count(&self, page: usize) -> usize {
            self.delivered(page, |_| true)
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
                        | ChildMsg::Thumb { .. }
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
    fn an_archive_serves_fit_with_attached_thumb() {
        let source = Arc::new(
            Source::open(Path::new(testkit::SEVENZ_FIXTURE))
                .expect("7z opens"),
        );
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn(Arc::clone(&source), Some(to_parent));

        worker.set_current(vec![request_fit(0)]);
        let (image, thumb) =
            events.ready("the archive page never served", |page, role| {
                page == 0 && role == WorkRole::Fit
            });
        assert_eq!((image.meta.width, image.meta.height), (40, 30));
        let thumb = thumb
            .expect("fit delivery includes thumb")
            .validate()
            .expect("valid webp thumb");
        assert_eq!(thumb.dims(), (40, 30));
    }

    #[test]
    fn fit_delivery_includes_scaled_thumb_and_native_cache_serves_full() {
        const PAGE_WIDTH: u32 = TEST_FIT_BOX.0 + 40;
        const PAGE_HEIGHT: u32 = 100;

        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("p0000.png");
        fs::write(
            &path,
            testkit::tiny_png(PAGE_WIDTH, PAGE_HEIGHT, [10, 20, 30, 255]),
        )
        .expect("write page");
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn(Arc::clone(&source), Some(to_parent));

        worker.set_current(vec![request_fit(0)]);
        let (fit, thumb) =
            events.ready("fit never arrived", |page, role| {
                page == 0 && role == WorkRole::Fit
            });
        assert_eq!(
            (fit.meta.width, fit.meta.height),
            (TEST_FIT_BOX.0, fit.meta.height)
        );
        let thumb = thumb
            .expect("thumb attached")
            .validate()
            .expect("valid webp thumb");
        assert_eq!(thumb.dims().0.max(thumb.dims().1), THUMB_EDGE);

        // Delete the backing file: `WorkRole::Full` must still succeed from
        // `native_cache` without touching disk!
        fs::remove_file(&path).expect("remove source file");
        worker.set_current(vec![request_full(0)]);
        let (full, full_thumb) =
            events.ready("full never arrived from native cache", |page, role| {
                page == 0 && role == WorkRole::Full
            });
        assert_eq!((full.meta.width, full.meta.height), (PAGE_WIDTH, PAGE_HEIGHT));
        assert!(full_thumb.is_none());
        assert_eq!(events.failures(0), 0);
    }

    #[test]
    fn a_failed_page_notifies_once_and_plan_advances() {
        let dir = TempDir::new().expect("create temp dir");
        fs::write(dir.path().join("p000.png"), b"not a png at all")
            .expect("write corrupt page");
        for page in 1..5_u8 {
            fs::write(
                dir.path().join(format!("p{page:03}.png")),
                testkit::tiny_png(
                    UNSCALED_FIT_EDGE,
                    UNSCALED_FIT_EDGE,
                    [page, 20, 30, 255],
                ),
            )
            .expect("write page");
        }
        let source =
            Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn(Arc::clone(&source), Some(to_parent));

        worker.set_current((0..5).map(request_fit).collect());
        events.failure(0);
        events.ready("the failure stalled remaining items", |page, role| {
            page == 4 && role == WorkRole::Fit
        });
        assert_eq!(
            events.failures(0),
            1,
            "the failure was announced more than once"
        );
    }

    #[test]
    fn a_fit_leaving_plan_is_served_again_when_readded() {
        const REDUNDANT_DELIVERY_WAIT: Duration = Duration::from_millis(50);

        let (_dir, source) = page_dir(1, UNSCALED_FIT_EDGE);
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn(Arc::clone(&source), Some(to_parent));

        worker.set_current(vec![request_fit(0)]);
        events.ready("fit never delivered", |page, role| {
            page == 0 && role == WorkRole::Fit
        });
        // An unchanged work plan must not re-deliver.
        worker.set_current(vec![request_fit(0)]);
        thread::sleep(REDUNDANT_DELIVERY_WAIT);
        assert_eq!(
            events.delivered(0, |role| role == WorkRole::Fit),
            1,
            "unchanged work fetched fit twice"
        );
        // Leaving then re-entering work (e.g. VRAM eviction) re-delivers.
        worker.set_work(WorkPlan::default());
        worker.set_current(vec![request_fit(0)]);
        testkit::eventually("re-added fit never served", || {
            (events.delivered(0, |role| role == WorkRole::Fit) >= 2)
                .then_some(())
        });
    }

    #[test]
    fn unready_archive_slots_do_not_starve_and_wake_when_filled() {
        const WORKER_PARK_SETTLE: Duration = Duration::from_millis(20);

        let names = ["p000.png", "p001.png", "p002.png"]
            .into_iter()
            .map(String::from)
            .collect();
        let (source, slots) =
            Source::from_test_slots(names).expect("test archive");
        let source = Arc::new(source);
        slots.fill(
            0,
            Bytes::from(testkit::tiny_png(16, 16, [10, 20, 30, 255])),
        );

        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn(Arc::clone(&source), Some(to_parent));

        worker.set_current(vec![
            request_fit(1),
            request_fit(2),
            request_fit(0),
        ]);
        events.ready(
            "ready page 0 starved behind unready requests",
            |page, role| page == 0 && role == WorkRole::Fit,
        );

        thread::sleep(WORKER_PARK_SETTLE);
        slots.fill(
            1,
            Bytes::from(testkit::tiny_png(12, 12, [40, 50, 60, 255])),
        );
        events.ready(
            "worker never woke when requested slot filled",
            |page, role| page == 1 && role == WorkRole::Fit,
        );
    }

    #[test]
    fn background_thumb_tier_streams_thumbs_without_fit() {
        const PAGE_EDGE: u32 = 400;

        let (_dir, source) = page_dir(3, PAGE_EDGE);
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn(Arc::clone(&source), Some(to_parent));

        worker.set_work(WorkPlan {
            current: Vec::new(),
            lookahead: Vec::new(),
            thumbs: vec![0, 1, 2],
        });
        for page in 0..3 {
            events
                .thumb("background thumb never arrived", page)
                .validate()
                .expect("valid webp thumb");
            assert_eq!(
                events.ready_count(page),
                0,
                "background thumb tier must not send Fit/Full images"
            );
        }
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
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn(Arc::clone(&source), Some(to_parent));

        start_playback(&worker, 0, 9);
        for (offset, color) in colors.into_iter().enumerate() {
            next_frame(&worker, 0, 9);
            let (delay, image) = events.frame(0, offset);
            assert_eq!(delay, Duration::from_millis(50));
            assert_eq!((image.meta.width, image.meta.height), (8, 8));
            assert_eq!(first_pixel(&image), color);
        }
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
        let (events, to_parent) = Updates::new();
        let worker = TestWorker::spawn(Arc::clone(&source), Some(to_parent));

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
    fn fixed_pool_tiers_enforce_strict_priority_and_threads() {
        let names = (0..6).map(|index| format!("{index:03}.png")).collect();
        let (source, slots) =
            Source::from_test_slots(names).expect("test archive");
        for index in 0..6 {
            slots.fill(index, Bytes::new());
        }
        let to_parent = ipc::channel::<ChildMsg>().expect("test channel").0;
        let budget = ThreadBudget {
            single: 12,
            pair: 6,
            thumb: 3,
        };
        let shared = Shared::new(to_parent, budget);

        // Tier 1: single current page gets all CPU threads (12T).
        shared.apply_work(WorkPlan {
            current: vec![request_fit(0)],
            lookahead: vec![1, 2],
            thumbs: vec![3, 4],
        });
        let single_current_job = shared
            .state
            .lock()
            .take_job(&source, budget)
            .expect("single current job");
        assert_eq!(
            (
                single_current_job.page,
                single_current_job.tier,
                single_current_job.threads
            ),
            (0, JobTier::Current, 12)
        );
        assert!(
            shared.state.lock().take_job(&source, budget).is_none(),
            "lower tiers stay blocked while Tier 1 is in flight"
        );
        shared.finish_page_job(0);
        shared.state.lock().served.insert(request_fit(0));

        // Tier 2: lookahead runs at pair_threads (6T) while thumbs wait.
        let first_lookahead = shared
            .state
            .lock()
            .take_job(&source, budget)
            .expect("lookahead job 1");
        let second_lookahead = shared
            .state
            .lock()
            .take_job(&source, budget)
            .expect("lookahead job 2");
        assert_eq!(
            (
                first_lookahead.page,
                first_lookahead.tier,
                first_lookahead.threads
            ),
            (1, JobTier::Lookahead, 6)
        );
        assert_eq!(
            (
                second_lookahead.page,
                second_lookahead.tier,
                second_lookahead.threads
            ),
            (2, JobTier::Lookahead, 6)
        );
        assert!(
            shared.state.lock().take_job(&source, budget).is_none(),
            "thumbs stay blocked while lookahead is active"
        );

        // Preemption: new Tier 1 work cancels in-flight Tier 2 jobs.
        shared.apply_work(WorkPlan {
            current: vec![request_fit(5), request_fit(4)],
            lookahead: vec![1, 2],
            thumbs: vec![3],
        });
        assert!(first_lookahead.is_canceled());
        assert!(second_lookahead.is_canceled());
        let first_current = shared
            .state
            .lock()
            .take_job(&source, budget)
            .expect("current pair job 1");
        let second_current = shared
            .state
            .lock()
            .take_job(&source, budget)
            .expect("current pair job 2");
        assert_eq!(
            (first_current.page, first_current.tier, first_current.threads),
            (5, JobTier::Current, 6)
        );
        assert_eq!(
            (
                second_current.page,
                second_current.tier,
                second_current.threads
            ),
            (4, JobTier::Current, 6)
        );
        shared.finish_page_job(1);
        shared.finish_page_job(2);
        shared.finish_page_job(5);
        shared.finish_page_job(4);

        // Tier 3: at most one background thumb runs at thumb_threads (3T).
        shared.apply_work(WorkPlan {
            current: Vec::new(),
            lookahead: Vec::new(),
            thumbs: vec![3, 4],
        });
        let thumb_job = shared
            .state
            .lock()
            .take_job(&source, budget)
            .expect("background thumb job");
        assert_eq!(
            (thumb_job.page, thumb_job.tier, thumb_job.threads),
            (3, JobTier::Thumb, 3)
        );
        assert!(
            shared.state.lock().take_job(&source, budget).is_none(),
            "second worker stays idle while one background thumb is running"
        );

        // Jumping onto page 3 preempts and cancels the running Tier 3 thumb job.
        shared.apply_work(WorkPlan {
            current: vec![request_fit(3)],
            lookahead: Vec::new(),
            thumbs: vec![3, 4],
        });
        assert!(thumb_job.is_canceled());
    }
}
