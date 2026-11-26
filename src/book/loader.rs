//! In-process 3-tier decode pool (`Current > Lookahead > Thumb`) and native cache.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::fmt::Display;
use std::mem::DropGuard;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::thread;
use std::thread::Builder;
use std::thread::JoinHandle;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use parking_lot::Condvar;
use parking_lot::Mutex;

use crate::book::catalog::Direction;
use crate::book::source;
use crate::book::source::Source;
use crate::book::thumbs::ThumbHandle;
use crate::image::decode::codec;
use crate::image::decode::codec::DecodeOutcome;
use crate::image::pixels::Image;
use crate::image::pixels::ImageLimits;
use crate::image::pixels::TargetRole;
use crate::image::resample;

const DEFAULT_CPU_THREADS: usize = 2;

const NATIVE_CACHE_CAP_BYTES: usize = 320 * 1024 * 1024;
const NATIVE_CACHE_MAX_ENTRIES: usize = 8;

const THUMB_WARM_AHEAD_PAGES: usize = 256;
const THUMB_WARM_BEHIND_PAGES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Request {
    pub page: usize,
    pub role: TargetRole,
}

/// Three-tier demand plan passed from `Reading` to [`DecodePool`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkPlan {
    pub current: Vec<Request>,
    pub lookahead: Vec<usize>,
    pub thumb_anchor: Option<(usize, Direction)>,
}

pub enum PageEvent {
    Ready {
        page: usize,
        role: TargetRole,
        image: Arc<Image>,
    },
    Failed {
        page: usize,
        fault: String,
    },
}

pub struct DecodePool {
    shared: Arc<Shared>,
    events: Receiver<PageEvent>,
    workers: Vec<JoinHandle<()>>,
}

impl DecodePool {
    pub fn start(
        source: Arc<Source>,
        thumbs: Option<Arc<ThumbHandle>>,
        limits: ImageLimits,
        initial_plan: WorkPlan,
        wake: Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        let (worker_count, budget) =
            ThreadBudget::for_cpus(available_parallelism());
        let (event_sender, events) = crossbeam_channel::unbounded();
        let on_ready_source = Arc::clone(&source);
        let shared = Arc::new(Shared {
            source,
            thumbs,
            limits,
            budget,
            state: Mutex::new(BookState::new(initial_plan)),
            wake: Condvar::new(),
            events: event_sender,
            ui_wake: wake,
            remaining_workers: AtomicUsize::new(worker_count),
        });
        let weak = Arc::downgrade(&shared);
        on_ready_source.set_on_ready(Some(Arc::new(move || {
            if let Some(live) = weak.upgrade() {
                live.wake_all();
            }
        })));

        let mut workers = Vec::with_capacity(worker_count);
        for worker_index in 0..worker_count {
            let thread_shared = Arc::clone(&shared);
            match Builder::new()
                .name(format!("cy-worker-{worker_index}"))
                .spawn(move || worker_loop(&thread_shared))
            {
                Ok(handle) => workers.push(handle),
                Err(error) => {
                    tracing::warn!(
                        worker_index,
                        ?error,
                        "worker thread spawn failed"
                    );
                    shared.on_worker_exit();
                }
            }
        }
        Self {
            shared,
            events,
            workers,
        }
    }

    pub fn set_work(&self, plan: WorkPlan) {
        self.shared.apply_work(plan);
    }

    pub fn drain(&self) -> impl Iterator<Item = PageEvent> + '_ {
        self.events.try_iter()
    }

    pub fn stop(&self) {
        self.shared.source.cancel();
        self.shared.source.set_on_ready(None);
        self.shared.stop();
    }

    pub fn stop_and_join(mut self) {
        self.stop();
        for handle in self.workers.drain(..) {
            if handle.join().is_err() {
                tracing::warn!("worker thread panicked");
            }
        }
    }
}

impl Drop for DecodePool {
    fn drop(&mut self) {
        self.stop();
    }
}

fn available_parallelism() -> usize {
    thread::available_parallelism()
        .map_or(DEFAULT_CPU_THREADS, NonZeroUsize::get)
}

#[derive(Default)]
struct NativeCache {
    total_bytes: usize,
    entries: HashMap<usize, Arc<Image>>,
    order: VecDeque<usize>,
}

impl NativeCache {
    fn get(&mut self, page: usize) -> Option<Arc<Image>> {
        let image = self.entries.get(&page).map(Arc::clone)?;
        self.touch(page);
        Some(image)
    }

    fn insert(&mut self, page: usize, image: Arc<Image>) {
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
                self.total_bytes = self
                    .total_bytes
                    .saturating_sub(removed.resident_bytes());
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
        || (request.role == TargetRole::Fit
            && plan.lookahead.contains(&request.page))
}

struct BookState {
    plan: WorkPlan,
    work_updated_at: Instant,
    /// Requests already handed over since they last entered the plan.
    served: HashSet<Request>,
    /// Pages whose first failure settled them for the session.
    failed: HashSet<usize>,
    /// Pages whose `TargetRole::Full` derivation failed and must not retry.
    failed_full: HashSet<usize>,
    in_flight: BTreeMap<usize, InFlight>,
    native_cache: NativeCache,
    stopped: bool,
}

impl BookState {
    fn new(initial_plan: WorkPlan) -> Self {
        Self {
            plan: initial_plan,
            work_updated_at: Instant::now(),
            served: HashSet::new(),
            failed: HashSet::new(),
            failed_full: HashSet::new(),
            in_flight: BTreeMap::new(),
            native_cache: NativeCache::default(),
            stopped: false,
        }
    }

    fn role_failed(&self, page: usize, role: TargetRole) -> bool {
        self.failed.contains(&page)
            || (role == TargetRole::Full
                && self.failed_full.contains(&page))
    }

    fn is_unsettled(&self, request: Request) -> bool {
        !self.served.contains(&request)
            && !self.role_failed(request.page, request.role)
    }

    fn is_unsettled_fit(&self, page: usize) -> bool {
        self.is_unsettled(Request {
            page,
            role: TargetRole::Fit,
        })
    }

    fn plan_contains_current_page(&self, page: usize) -> bool {
        self.plan.current.iter().any(|request| request.page == page)
    }

    fn active_plan_tier(
        &self,
        thumbs: Option<&ThumbHandle>,
    ) -> Option<JobTier> {
        if self
            .plan
            .current
            .iter()
            .any(|&request| self.is_unsettled(request))
        {
            Some(JobTier::Current)
        } else if self
            .plan
            .lookahead
            .iter()
            .any(|&page| self.is_unsettled_fit(page))
        {
            Some(JobTier::Lookahead)
        } else if self.plan.thumb_anchor.is_some() && thumbs.is_some() {
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

    fn wants_unserved(&self, page: usize, role: TargetRole) -> bool {
        let request = Request { page, role };
        plan_contains(&self.plan, request) && self.is_unsettled(request)
    }

    fn claim_publish(&mut self, page: usize, role: TargetRole) -> bool {
        if !self.wants_unserved(page, role) {
            return false;
        }
        self.served.insert(Request { page, role });
        true
    }

    fn apply_work(
        &mut self,
        plan: WorkPlan,
        thumbs: Option<&ThumbHandle>,
    ) {
        self.work_updated_at = Instant::now();
        self.served.retain(|&request| plan_contains(&plan, request));
        self.plan = plan;
        let active_tier = self.active_plan_tier(thumbs);
        for (&page, job) in &self.in_flight {
            let keep = match active_tier {
                Some(JobTier::Current) => {
                    self.plan_contains_current_page(page)
                }
                Some(JobTier::Lookahead) => {
                    self.plan.lookahead.contains(&page)
                }
                Some(JobTier::Thumb) => job.tier == JobTier::Thumb,
                None => false,
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

    fn launch(
        &mut self,
        page: usize,
        tier: JobTier,
        threads: usize,
    ) -> PageJob {
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

    fn next_thumb_page(
        &self,
        source: &Source,
        thumbs: &ThumbHandle,
        anchor: usize,
        direction: Direction,
    ) -> Option<usize> {
        let page_count = source.names().count().get();
        let anchor = anchor.min(page_count - 1);
        let is_eligible = |&page: &usize| {
            !self.failed.contains(&page)
                && !self.in_flight.contains_key(&page)
                && !thumbs.has(page)
                && source.is_ready(page)
        };
        match direction {
            Direction::Forward => {
                let mut forward_ahead = anchor.saturating_add(1)
                    ..=anchor
                        .saturating_add(THUMB_WARM_AHEAD_PAGES)
                        .min(page_count - 1);
                let mut backward_behind = (anchor
                    .saturating_sub(THUMB_WARM_BEHIND_PAGES)
                    ..anchor)
                    .rev();
                forward_ahead
                    .find(is_eligible)
                    .or_else(|| backward_behind.find(is_eligible))
            }
            Direction::Backward => {
                let mut backward_ahead = (anchor
                    .saturating_sub(THUMB_WARM_AHEAD_PAGES)
                    ..anchor)
                    .rev();
                let mut forward_behind = anchor.saturating_add(1)
                    ..=anchor
                        .saturating_add(THUMB_WARM_BEHIND_PAGES)
                        .min(page_count - 1);
                backward_ahead
                    .find(is_eligible)
                    .or_else(|| forward_behind.find(is_eligible))
            }
        }
    }

    fn take_job(
        &mut self,
        source: &Source,
        thumbs: Option<&ThumbHandle>,
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
            // Only allocate all CPU threads when no other decode worker is busy.
            let threads =
                if current_pages.len() == 1 && self.in_flight.is_empty() {
                    budget.single
                } else {
                    budget.pair
                };
            return Some(self.launch(page, JobTier::Current, threads));
        }
        if !current_pages.is_empty()
            || self.has_active_tier(JobTier::Current)
        {
            return None;
        }

        if let Some(&page) = self.plan.lookahead.iter().find(|&&page| {
            self.is_unsettled_fit(page)
                && !self.in_flight.contains_key(&page)
                && source.is_ready(page)
        }) {
            return Some(self.launch(
                page,
                JobTier::Lookahead,
                budget.pair,
            ));
        }
        let lookahead_active = self
            .plan
            .lookahead
            .iter()
            .any(|&page| self.is_unsettled_fit(page))
            || self.has_active_tier(JobTier::Lookahead);
        if lookahead_active {
            return None;
        }

        if self.has_active_tier(JobTier::Thumb) {
            return None;
        }
        let (anchor, direction) = self.plan.thumb_anchor?;
        let thumbs = thumbs?;
        let page =
            self.next_thumb_page(source, thumbs, anchor, direction)?;
        Some(self.launch(page, JobTier::Thumb, budget.thumb))
    }
}

struct Shared {
    source: Arc<Source>,
    thumbs: Option<Arc<ThumbHandle>>,
    limits: ImageLimits,
    budget: ThreadBudget,
    state: Mutex<BookState>,
    wake: Condvar,
    events: Sender<PageEvent>,
    ui_wake: Arc<dyn Fn() + Send + Sync>,
    remaining_workers: AtomicUsize,
}

impl Shared {
    fn finish_page_job(&self, page: usize) {
        self.state.lock().in_flight.remove(&page);
        self.wake.notify_all();
    }

    fn on_worker_exit(&self) {
        use rustfs_mimalloc::MiMalloc;

        if self.remaining_workers.fetch_sub(1, Ordering::AcqRel) == 1 {
            if let Some(thumbs) = &self.thumbs {
                thumbs.flush();
            }
            MiMalloc::collect(true);
            MiMalloc::arenas_purge();
        }
    }

    fn stop(&self) {
        self.state.lock().stop();
        self.wake.notify_all();
    }

    fn wake_all(&self) {
        drop(self.state.lock());
        self.wake.notify_all();
    }

    fn apply_work(&self, plan: WorkPlan) {
        tracing::debug!(
            current = plan.current.len(),
            lookahead = plan.lookahead.len(),
            has_thumb_anchor = plan.thumb_anchor.is_some(),
            "work plan"
        );
        self.state.lock().apply_work(plan, self.thumbs.as_deref());
        self.wake.notify_all();
    }

    #[expect(
        clippy::significant_drop_tightening,
        reason = "condvar wait loop holds the mutex guard across iterations"
    )]
    fn next_job(&self) -> Option<PageJob> {
        let mut state = self.state.lock();
        loop {
            if state.stopped {
                return None;
            }
            if let Some(job) = state.take_job(
                &self.source,
                self.thumbs.as_deref(),
                self.budget,
            ) {
                return Some(job);
            }
            self.wake.wait(&mut state);
        }
    }

    fn derive_for<'source>(
        &self,
        page: usize,
        role: TargetRole,
        source: &'source Image,
        threads: usize,
    ) -> Option<Cow<'source, Image>> {
        let started = Instant::now();
        match resample::derive(source, role, self.limits, threads) {
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
                    TargetRole::Full => {
                        self.state.lock().failed_full.insert(page);
                    }
                    TargetRole::Fit => {
                        self.record_failure(page, &fault);
                    }
                }
                None
            }
        }
    }

    fn emit(&self, event: PageEvent) {
        if self.events.send(event).is_err() {
            self.stop();
            return;
        }
        (self.ui_wake)();
    }

    fn deliver(
        &self,
        job: &PageJob,
        role: TargetRole,
        image: Arc<Image>,
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
            tier = ?job.tier,
            native_cache_hit,
            waited = ?job.requested_at.elapsed(),
            elapsed = ?started.elapsed(),
            "served"
        );
        self.emit(PageEvent::Ready {
            page: job.page,
            role,
            image,
        });
    }

    fn record_failure(&self, page: usize, error: &dyn Display) {
        if self.state.lock().failed.insert(page) {
            self.emit(PageEvent::Failed {
                page,
                fault: error.to_string(),
            });
        }
    }

    fn read_source(&self, page: usize) -> Option<Arc<[u8]>> {
        match self.source.bytes(page) {
            Ok(bytes) => Some(bytes),
            Err(error) => {
                self.record_failure(page, &error);
                None
            }
        }
    }
}

fn decode_page_source(shared: &Shared, job: &PageJob) -> Option<Image> {
    let page = job.page;
    let bytes = shared.read_source(page)?;
    let Some(name) = shared.source.name(page) else {
        shared.record_failure(page, &"no such page");
        return None;
    };
    let decode_started = Instant::now();
    let decoded = match source::catch_panic("decoder", || {
        codec::decode_cancellable(&bytes, name, job.threads, &job.cancel)
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
    shared: &Shared,
    job: &PageJob,
    cache_native: bool,
) -> Option<(Arc<Image>, bool)> {
    let cached = shared.state.lock().native_cache.get(job.page);
    if let Some(cached) = cached {
        return Some((cached, true));
    }
    let decoded = decode_page_source(shared, job)?;
    let native = Arc::new(decoded);
    if cache_native {
        shared
            .state
            .lock()
            .native_cache
            .insert(job.page, Arc::clone(&native));
    }
    Some((native, false))
}

fn into_shared_image(
    derived: Cow<'_, Image>,
    backing: &Arc<Image>,
) -> Arc<Image> {
    match derived {
        Cow::Borrowed(_) => Arc::clone(backing),
        Cow::Owned(owned) => Arc::new(owned),
    }
}

fn record_missing_thumb(
    shared: &Shared,
    job: &PageJob,
    image: &Image,
    threads: usize,
) {
    let Some(thumbs) = &shared.thumbs else {
        return;
    };
    if thumbs.has(job.page) || job.is_canceled() {
        return;
    }
    match resample::derive_thumb(image, threads) {
        Ok(thumb) if !job.is_canceled() => thumbs.record(job.page, thumb),
        Ok(_) => {}
        Err(error) => {
            tracing::debug!(page = job.page, %error, "thumb derive failed");
        }
    }
}

fn produce_page(shared: &Shared, job: &PageJob) {
    const ATTACHED_THUMB_THREADS: usize = 1;

    let started = Instant::now();
    let page = job.page;
    let is_thumb = job.tier == JobTier::Thumb;
    let Some((native, native_cache_hit)) =
        load_or_decode(shared, job, !is_thumb)
    else {
        return;
    };
    if job.is_canceled() {
        return;
    }

    let (want_full, want_fit) = {
        let state = shared.state.lock();
        (
            state.wants_unserved(page, TargetRole::Full),
            state.wants_unserved(page, TargetRole::Fit),
        )
    };
    if !want_full && !want_fit {
        if is_thumb {
            record_missing_thumb(shared, job, &native, job.threads);
        }
        return;
    }

    if is_thumb {
        // Promoted background thumb job: preserve decoded raster for zoom.
        shared
            .state
            .lock()
            .native_cache
            .insert(page, Arc::clone(&native));
    }

    if want_full
        && let Some(full) =
            shared.derive_for(page, TargetRole::Full, &native, job.threads)
    {
        shared.deliver(
            job,
            TargetRole::Full,
            into_shared_image(full, &native),
            native_cache_hit,
            started,
        );
    }

    if job.is_canceled() {
        return;
    }

    if want_fit
        && let Some(fit) =
            shared.derive_for(page, TargetRole::Fit, &native, job.threads)
    {
        let fit_image = into_shared_image(fit, &native);
        shared.deliver(
            job,
            TargetRole::Fit,
            Arc::clone(&fit_image),
            native_cache_hit,
            started,
        );
        record_missing_thumb(
            shared,
            job,
            &fit_image,
            ATTACHED_THUMB_THREADS,
        );
    }
}

fn worker_loop(shared: &Shared) {
    let _exit_guard = DropGuard::new((), |()| shared.on_worker_exit());
    while let Some(job) = shared.next_job() {
        let _job_guard =
            DropGuard::new(job.page, |page| shared.finish_page_job(page));
        tracing::debug!(
            page = job.page,
            tier = ?job.tier,
            threads = job.threads,
            waited = ?job.requested_at.elapsed(),
            "job"
        );
        produce_page(shared, &job);
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::time::Duration;

    use tempfile::TempDir;

    use super::*;
    use crate::book::thumbs::ThumbStore;
    use crate::image::pixels::THUMB_EDGE;
    use crate::testkit;
    use crate::testkit::TEST_FIT_BOX;

    const UNSCALED_FIT_EDGE: u32 = 1400;

    fn noop_wake() -> Arc<dyn Fn() + Send + Sync> {
        Arc::new(|| {})
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

    fn request_fit(page: usize) -> Request {
        Request {
            page,
            role: TargetRole::Fit,
        }
    }

    fn request_full(page: usize) -> Request {
        Request {
            page,
            role: TargetRole::Full,
        }
    }

    struct LoaderHarness {
        loader: DecodePool,
        seen: Mutex<Vec<PageEvent>>,
    }

    impl LoaderHarness {
        fn start(
            source: Arc<Source>,
            thumbs: Option<Arc<ThumbHandle>>,
            plan: WorkPlan,
        ) -> Self {
            Self {
                loader: DecodePool::start(
                    source,
                    thumbs,
                    testkit::limits(),
                    plan,
                    noop_wake(),
                ),
                seen: Mutex::new(Vec::new()),
            }
        }

        fn drain(&self) {
            let mut seen = self.seen.lock();
            seen.extend(self.loader.drain());
        }

        fn ready(
            &self,
            what: &str,
            want: impl Fn(usize, TargetRole) -> bool,
        ) -> Arc<Image> {
            testkit::eventually(what, || {
                self.drain();
                self.seen.lock().iter().find_map(|event| match event {
                    PageEvent::Ready { page, role, image }
                        if want(*page, *role) =>
                    {
                        Some(Arc::clone(image))
                    }
                    PageEvent::Ready { .. } | PageEvent::Failed { .. } => {
                        None
                    }
                })
            })
        }

        fn failure(&self, page: usize) {
            testkit::eventually("page never failed", || {
                self.drain();
                self.seen.lock().iter().find_map(|event| match event {
                    PageEvent::Failed { page: failed, .. }
                        if *failed == page =>
                    {
                        Some(())
                    }
                    PageEvent::Ready { .. } | PageEvent::Failed { .. } => {
                        None
                    }
                })
            });
        }

        fn delivered(
            &self,
            page: usize,
            want: impl Fn(TargetRole) -> bool,
        ) -> usize {
            self.drain();
            self.seen
                .lock()
                .iter()
                .filter(|event| {
                    matches!(
                        event,
                        PageEvent::Ready {
                            page: delivered,
                            role,
                            ..
                        } if *delivered == page && want(*role)
                    )
                })
                .count()
        }

        fn failures(&self, page: usize) -> usize {
            self.drain();
            self.seen
                .lock()
                .iter()
                .filter(|event| {
                    matches!(
                        event,
                        PageEvent::Failed { page: failed, .. } if *failed == page
                    )
                })
                .count()
        }
    }

    #[test]
    fn an_archive_serves_fit_and_records_thumb() {
        let cache_dir = TempDir::new().expect("cache dir");
        let store = ThumbStore::open_at(cache_dir.path()).expect("store");
        let source = Arc::new(
            Source::open(Path::new(testkit::SEVENZ_FIXTURE))
                .expect("7z opens"),
        );
        let thumbs = Arc::new(
            store
                .open_book(
                    Path::new(testkit::SEVENZ_FIXTURE),
                    source.is_archive(),
                    source.names(),
                )
                .expect("book thumbs"),
        );
        let harness = LoaderHarness::start(
            Arc::clone(&source),
            Some(Arc::clone(&thumbs)),
            WorkPlan {
                current: vec![request_fit(0)],
                ..WorkPlan::default()
            },
        );

        let image = harness
            .ready("the archive page never served", |page, role| {
                page == 0 && role == TargetRole::Fit
            });
        assert_eq!((image.width(), image.height()), (40, 30));
        let thumb =
            testkit::eventually("thumb never recorded", || thumbs.load(0));
        assert_eq!(thumb.dims, (40, 30));
        harness.loader.stop_and_join();
    }

    #[test]
    fn fit_delivery_records_scaled_thumb_and_native_cache_serves_full() {
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
        let cache_dir = TempDir::new().expect("cache dir");
        let store = ThumbStore::open_at(cache_dir.path()).expect("store");
        let thumbs = Arc::new(
            store
                .open_book(dir.path(), source.is_archive(), source.names())
                .expect("book thumbs"),
        );
        let harness = LoaderHarness::start(
            Arc::clone(&source),
            Some(Arc::clone(&thumbs)),
            WorkPlan {
                current: vec![request_fit(0)],
                ..WorkPlan::default()
            },
        );

        let fit = harness.ready("fit never arrived", |page, role| {
            page == 0 && role == TargetRole::Fit
        });
        assert_eq!(
            (fit.width(), fit.height()),
            (TEST_FIT_BOX.0, fit.height())
        );
        let thumb =
            testkit::eventually("thumb never recorded", || thumbs.load(0));
        assert_eq!(thumb.dims.0.max(thumb.dims.1), THUMB_EDGE);

        // Delete the backing file: `TargetRole::Full` must still succeed from
        // `native_cache` without touching disk!
        fs::remove_file(&path).expect("remove source file");
        harness.loader.set_work(WorkPlan {
            current: vec![request_full(0)],
            ..WorkPlan::default()
        });
        let full = harness.ready(
            "full never arrived from native cache",
            |page, role| page == 0 && role == TargetRole::Full,
        );
        assert_eq!(
            (full.width(), full.height()),
            (PAGE_WIDTH, PAGE_HEIGHT)
        );
        assert_eq!(harness.failures(0), 0);
        harness.loader.stop_and_join();
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
        let harness = LoaderHarness::start(
            Arc::clone(&source),
            None,
            WorkPlan {
                current: (0..5).map(request_fit).collect(),
                ..WorkPlan::default()
            },
        );
        harness.failure(0);
        harness
            .ready("the failure stalled remaining items", |page, role| {
                page == 4 && role == TargetRole::Fit
            });
        assert_eq!(
            harness.failures(0),
            1,
            "the failure was announced more than once"
        );
        harness.loader.stop_and_join();
    }

    #[test]
    fn a_fit_leaving_plan_is_served_again_when_readded() {
        const REDUNDANT_DELIVERY_WAIT: Duration =
            Duration::from_millis(50);

        let (_dir, source) = page_dir(1, UNSCALED_FIT_EDGE);
        let harness = LoaderHarness::start(
            Arc::clone(&source),
            None,
            WorkPlan {
                current: vec![request_fit(0)],
                ..WorkPlan::default()
            },
        );

        harness.ready("fit never delivered", |page, role| {
            page == 0 && role == TargetRole::Fit
        });
        harness.loader.set_work(WorkPlan {
            current: vec![request_fit(0)],
            ..WorkPlan::default()
        });
        thread::sleep(REDUNDANT_DELIVERY_WAIT);
        assert_eq!(
            harness.delivered(0, |role| role == TargetRole::Fit),
            1,
            "unchanged work fetched fit twice"
        );
        harness.loader.set_work(WorkPlan::default());
        harness.loader.set_work(WorkPlan {
            current: vec![request_fit(0)],
            ..WorkPlan::default()
        });
        testkit::eventually("re-added fit never served", || {
            (harness.delivered(0, |role| role == TargetRole::Fit) >= 2)
                .then_some(())
        });
        harness.loader.stop_and_join();
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
            Arc::from(testkit::tiny_png(16, 16, [10, 20, 30, 255])),
        );

        let harness = LoaderHarness::start(
            Arc::clone(&source),
            None,
            WorkPlan {
                current: vec![
                    request_fit(1),
                    request_fit(2),
                    request_fit(0),
                ],
                ..WorkPlan::default()
            },
        );
        harness.ready(
            "ready page 0 starved behind unready requests",
            |page, role| page == 0 && role == TargetRole::Fit,
        );

        thread::sleep(WORKER_PARK_SETTLE);
        slots.fill(
            1,
            Arc::from(testkit::tiny_png(12, 12, [40, 50, 60, 255])),
        );
        harness.ready(
            "worker never woke when requested slot filled",
            |page, role| page == 1 && role == TargetRole::Fit,
        );
        harness.loader.stop_and_join();
    }

    #[test]
    fn background_thumb_tier_records_thumbs_without_fit() {
        const PAGE_EDGE: u32 = 400;

        let (dir, source) = page_dir(3, PAGE_EDGE);
        let cache_dir = TempDir::new().expect("cache dir");
        let store = ThumbStore::open_at(cache_dir.path()).expect("store");
        let thumbs = Arc::new(
            store
                .open_book(dir.path(), source.is_archive(), source.names())
                .expect("book thumbs"),
        );
        let harness = LoaderHarness::start(
            Arc::clone(&source),
            Some(Arc::clone(&thumbs)),
            WorkPlan {
                current: Vec::new(),
                lookahead: Vec::new(),
                thumb_anchor: Some((0, Direction::Forward)),
            },
        );
        for page in 1..3 {
            testkit::eventually("background thumb never recorded", || {
                thumbs.load(page)
            });
            assert_eq!(
                harness.delivered(page, |_| true),
                0,
                "background thumb tier must not deliver Fit/Full images"
            );
        }
        harness.loader.stop_and_join();
    }

    #[test]
    fn fixed_pool_tiers_enforce_strict_priority_and_threads() {
        let names =
            (0..6).map(|index| format!("{index:03}.png")).collect();
        let (source, slots) =
            Source::from_test_slots(names).expect("test archive");
        for index in 0..6 {
            slots.fill(index, Arc::from([]));
        }
        let source = Arc::new(source);
        let cache_dir = TempDir::new().expect("cache dir");
        let store = ThumbStore::open_at(cache_dir.path()).expect("store");
        let thumbs = Arc::new(
            store.open_book_with_key([7; 16], 6).expect("book thumbs"),
        );
        let (event_sender, _) = crossbeam_channel::unbounded();
        let budget = ThreadBudget {
            single: 12,
            pair: 6,
            thumb: 3,
        };
        let shared = Shared {
            source: Arc::clone(&source),
            thumbs: Some(Arc::clone(&thumbs)),
            limits: testkit::limits(),
            budget,
            state: Mutex::new(BookState::new(WorkPlan {
                current: vec![request_fit(0)],
                lookahead: vec![1, 2],
                thumb_anchor: Some((2, Direction::Forward)),
            })),
            wake: Condvar::new(),
            events: event_sender,
            ui_wake: noop_wake(),
            remaining_workers: AtomicUsize::new(2),
        };

        let single_current_job = shared
            .state
            .lock()
            .take_job(&source, Some(&thumbs), budget)
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
            shared
                .state
                .lock()
                .take_job(&source, Some(&thumbs), budget)
                .is_none(),
            "lower tiers stay blocked while Tier 1 is in flight"
        );
        shared.finish_page_job(0);
        shared.state.lock().served.insert(request_fit(0));

        let first_lookahead = shared
            .state
            .lock()
            .take_job(&source, Some(&thumbs), budget)
            .expect("lookahead job 1");
        let second_lookahead = shared
            .state
            .lock()
            .take_job(&source, Some(&thumbs), budget)
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
            shared
                .state
                .lock()
                .take_job(&source, Some(&thumbs), budget)
                .is_none(),
            "thumbs stay blocked while lookahead is active"
        );

        shared.apply_work(WorkPlan {
            current: vec![request_fit(5), request_fit(4)],
            lookahead: vec![1, 2],
            thumb_anchor: Some((2, Direction::Forward)),
        });
        assert!(first_lookahead.is_canceled());
        assert!(second_lookahead.is_canceled());
        let first_current = shared
            .state
            .lock()
            .take_job(&source, Some(&thumbs), budget)
            .expect("current pair job 1");
        let second_current = shared
            .state
            .lock()
            .take_job(&source, Some(&thumbs), budget)
            .expect("current pair job 2");
        assert_eq!(
            (
                first_current.page,
                first_current.tier,
                first_current.threads
            ),
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

        shared.apply_work(WorkPlan {
            current: Vec::new(),
            lookahead: Vec::new(),
            thumb_anchor: Some((2, Direction::Forward)),
        });
        let thumb_job = shared
            .state
            .lock()
            .take_job(&source, Some(&thumbs), budget)
            .expect("background thumb job");
        assert_eq!(
            (thumb_job.page, thumb_job.tier, thumb_job.threads),
            (3, JobTier::Thumb, 3)
        );
        assert!(
            shared
                .state
                .lock()
                .take_job(&source, Some(&thumbs), budget)
                .is_none(),
            "second worker stays idle while one background thumb is running"
        );

        // Jumping onto page 3 promotes the in-flight thumb job without canceling it.
        shared.apply_work(WorkPlan {
            current: vec![request_fit(3)],
            lookahead: Vec::new(),
            thumb_anchor: None,
        });
        assert!(!thumb_job.is_canceled());
    }
}
