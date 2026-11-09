//! Decode worker: the app's demand service over a store of previews.
//!
//! Two inputs decide what the thread does: the app's [`Want`]s, the
//! artifacts its presentation is missing right now, and the plan, the
//! order in which pages are worth producing into the store ahead of
//! time. Wants are served first, from the store when the plan already
//! warmed them and otherwise by decoding and minting on the spot; the
//! plan alone never delivers anything, it only fills the store.
//!
//! Nothing decoded survives here. The preview store is the only cache: a
//! page is read, hashed, minted into the artifacts it needs, persisted,
//! and the pixels drop. A later want, or a later session over the same
//! root, loads the artifact back instead of decoding again.

use std::collections::HashMap;
use std::collections::HashSet;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::sync::mpsc;
use std::sync::mpsc::RecvTimeoutError;
use std::sync::Arc;
use std::thread;
use std::thread::Builder;
use std::thread::JoinHandle;
use std::time::Duration;
use std::time::Instant;

#[cfg(test)]
use std::path::PathBuf;

use eros::ErrorUnion;

use crate::decode;
use crate::decode::Decoded;
use crate::fault::BadIndex;
use crate::fault::PageError;
use crate::fault::PanicFault;
use crate::fault::SpawnError;
use crate::plan::Plan;
use crate::preview;
use crate::preview::Artifact;
use crate::preview::Store;
use crate::source::PageRead;
use crate::source::Source;

/// How long the loop naps when a page's bytes are still unpacking. The
/// channel is drained after every nap, so a stop or a newer plan answers
/// within one nap.
const FILL_POLL: Duration = Duration::from_millis(15);

/// How long the app must stay silent before the worker spends time on
/// pages beyond the head. Those pages are preload, not demand: a reader
/// who is turning pages leaves the worker idle between turns, so the
/// next target is served without waiting out a speculative decode.
const IDLE_PRELOAD: Duration = Duration::from_millis(300);

/// What the reader is doing, which decides whether views are worth
/// minting at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Motion {
    /// The reader is looking at the spread; its head is worth a view.
    Still,
    /// The reader is turning pages faster than a view could arrive, so
    /// thumbs are all that can be shown.
    Skimming,
}

/// One artifact for the app. `image` is what the renderer uploads: the
/// stored crop for `Thumb` and `View`, the page's own pixels for `Native`.
pub struct Ready {
    pub page: usize,
    pub kind: Artifact,
    pub image: Arc<Decoded>,
}

pub enum Event {
    Ready(Ready),
    Failed { page: usize, error: Arc<PageError> },
}

/// One artifact the app is missing. The app's texture maps are the only
/// authority on what it has; the worker never assumes a delivery stuck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Want {
    pub page: usize,
    pub kind: Artifact,
}

enum ToWorker {
    Retune(Plan),
    Motion(Motion),
    /// Replace the set of artifacts the app is missing. The app sends
    /// it on change; a want leaving the set is how it says it got the
    /// artifact.
    Targets(Vec<Want>),
    Stop,
}

/// Which of a page's artifacts a step will produce or deliver.
#[derive(Clone, Copy, Debug, Default)]
struct Artifacts {
    thumb: bool,
    view: bool,
    native: bool,
}

impl Artifacts {
    fn set(&mut self, kind: Artifact) {
        match kind {
            Artifact::Thumb => self.thumb = true,
            Artifact::View => self.view = true,
            Artifact::Native => self.native = true,
        }
    }

    fn clear(&mut self, kind: Artifact) {
        match kind {
            Artifact::Thumb => self.thumb = false,
            Artifact::View => self.view = false,
            Artifact::Native => self.native = false,
        }
    }

    fn any(self) -> bool {
        self.thumb || self.view || self.native
    }
}

/// One unit of work: a page and the artifacts to produce. `demand` means
/// the app is waiting for them and they are delivered; otherwise they
/// only fill the store. `preload` means the work is speculative and must
/// not make a turning reader wait.
#[derive(Clone, Copy, Debug)]
struct Job {
    page: usize,
    needs: Artifacts,
    demand: bool,
    preload: bool,
}

/// Everything the worker thread knows: what the app wants, and what the
/// store already answers.
struct Session {
    plan: Plan,
    motion: Motion,
    store: Store,
    /// Content hash of each page's bytes, computed once per session.
    hashes: HashMap<usize, u128>,
    /// Artifacts this session put in the store, or loaded from it, keyed
    /// by the store's own content hash. The plan uses this to skip
    /// pages; a save's evictions clear the entry directly, so the plan
    /// rebuilds a page that returns to its window. A failed save still
    /// counts, so a store that refuses writes cannot make the plan spin
    /// on the same page.
    produced: HashMap<u128, Artifacts>,
    /// A page's first failure is its verdict for the session.
    failed: HashMap<usize, Arc<PageError>>,
    /// What the app is missing, in its priority order.
    targets: Vec<Want>,
    /// When the current target set arrived, for the wait diagnostics.
    targets_at: Instant,
    /// Wants already handed over since they last entered the set.
    delivered: HashSet<Want>,
    /// Whether the idle preload burst is running. Any word from the app
    /// stops it, so a turning reader never competes with the tail.
    preloading: bool,
}

impl Session {
    fn new(plan: Plan, store: Store) -> Self {
        Self {
            plan,
            motion: Motion::Still,
            store,
            hashes: HashMap::new(),
            produced: HashMap::new(),
            failed: HashMap::new(),
            targets: Vec::new(),
            targets_at: Instant::now(),
            delivered: HashSet::new(),
            preloading: false,
        }
    }

    /// Apply one message. `false` means the loop must stop.
    fn apply(&mut self, message: ToWorker) -> bool {
        if !matches!(message, ToWorker::Stop) {
            self.preloading = false;
        }
        match message {
            ToWorker::Retune(plan) => {
                tracing::debug!(
                    anchor = ?plan.spreads().first(),
                    motion = ?self.motion,
                    "plan retuned"
                );
                self.plan = plan;
            }
            ToWorker::Motion(motion) => {
                tracing::debug!(?motion, "motion");
                self.motion = motion;
            }
            ToWorker::Targets(targets) => {
                // A want that left the set was delivered and uploaded;
                // if it comes back, it must be served again.
                tracing::debug!(count = targets.len(), "targets");
                self.delivered.retain(|want| targets.contains(want));
                self.targets = targets;
                self.targets_at = Instant::now();
            }
            ToWorker::Stop => return false,
        }
        true
    }

    /// The first page in plan order that still needs an artifact, and
    /// which artifacts it needs. A failed page is never a candidate
    /// again.
    fn next_job(&self) -> Option<Job> {
        let head = self.plan.head();
        for spread in self.plan.spreads() {
            let in_head = head.contains(spread);
            for page in
                [Some(spread.first), spread.second].into_iter().flatten()
            {
                if self.failed.contains_key(&page) {
                    continue;
                }
                let produced = self
                    .hashes
                    .get(&page)
                    .and_then(|hash| self.produced.get(hash))
                    .copied()
                    .unwrap_or_default();
                let needs = Artifacts {
                    thumb: !produced.thumb,
                    view: in_head
                        && self.motion == Motion::Still
                        && !produced.view,
                    native: false,
                };
                if needs.any() {
                    return Some(Job {
                        page,
                        needs,
                        demand: false,
                        preload: !in_head,
                    });
                }
            }
        }
        None
    }

    /// The first target page with an undelivered kind, and those kinds.
    /// The app lists its wants in priority order, so a page's group is
    /// taken whole and served with a single decode.
    fn next_target(&self) -> Option<Job> {
        let mut found: Option<Job> = None;
        for want in &self.targets {
            if self.delivered.contains(want)
                || self.failed.contains_key(&want.page)
            {
                continue;
            }
            match found.as_mut() {
                Some(job) if job.page == want.page => {
                    job.needs.set(want.kind);
                }
                Some(_) => break,
                None => {
                    let mut needs = Artifacts::default();
                    needs.set(want.kind);
                    found = Some(Job {
                        page: want.page,
                        needs,
                        demand: true,
                        preload: false,
                    });
                }
            }
        }
        found
    }

    /// How long the app has been missing the current target set.
    fn target_wait(&self) -> Duration {
        self.targets_at.elapsed()
    }

    /// Content hash of the page's bytes, computed once per session.
    fn hash(&mut self, page: usize, bytes: &[u8]) -> u128 {
        *self
            .hashes
            .entry(page)
            .or_insert_with(|| preview::content_hash(bytes))
    }

    /// Hand one artifact to the app and remember it for the current target
    /// set, so an unchanged set does not fetch it twice.
    fn publish(
        &mut self,
        page: usize,
        kind: Artifact,
        image: Decoded,
        notify: &dyn Fn(Event),
    ) {
        self.delivered.insert(Want { page, kind });
        tracing::debug!(
            page,
            ?kind,
            width = image.width(),
            height = image.height(),
            "publish"
        );
        notify(Event::Ready(Ready {
            page,
            kind,
            image: Arc::new(image),
        }));
    }

    /// Remember that the store holds `hash`'s `kind`.
    fn mark_produced(&mut self, hash: u128, kind: Artifact) {
        self.produced.entry(hash).or_default().set(kind);
    }

    /// Save one artifact and remember the store has it. A save only prunes
    /// when a cap is crossed, and it reports the entries it evicted, so
    /// the ledger stays exact instead of claiming a pruned preview.
    fn keep(
        &mut self,
        page: usize,
        hash: u128,
        kind: Artifact,
        image: Decoded,
    ) -> Decoded {
        let saved = self.store.save(hash, kind, &image);
        if !saved.landed {
            tracing::debug!(page, ?kind, "preview save missed the store");
        }
        for (evicted_kind, evicted_hash) in saved.evicted {
            if let Some(artifacts) = self.produced.get_mut(&evicted_hash) {
                artifacts.clear(evicted_kind);
            }
        }
        self.mark_produced(hash, kind);
        image
    }

    /// Record a page's verdict and tell the app once.
    fn record_failure(
        &mut self,
        page: usize,
        error: Arc<PageError>,
        notify: &dyn Fn(Event),
    ) {
        if self.failed.insert(page, Arc::clone(&error)).is_some() {
            return;
        }
        notify(Event::Failed { page, error });
    }

    /// Produce a job's artifacts: load what the store already answers,
    /// decode once for the rest, and save every preview made. A demand
    /// job hands the artifacts to the app; a plan job only fills the
    /// store.
    fn serve(&mut self, source: &Source, job: Job, notify: &dyn Fn(Event)) {
        let page = job.page;
        let needs = job.needs;
        let demand = job.demand;
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
        let mut thumb = needs
            .thumb
            .then(|| self.store.load(hash, Artifact::Thumb))
            .flatten();
        let mut view = needs
            .view
            .then(|| self.store.load(hash, Artifact::View))
            .flatten();
        let thumb_hit = thumb.is_some();
        let view_hit = view.is_some();
        if thumb.is_some() {
            self.mark_produced(hash, Artifact::Thumb);
        }
        if view.is_some() {
            self.mark_produced(hash, Artifact::View);
        }
        if needs.native
            || (needs.view && view.is_none())
            || (needs.thumb && thumb.is_none())
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
            if needs.view && view.is_none() {
                let started = Instant::now();
                let minted = preview::mint(&native, preview::VIEW_EDGE);
                tracing::debug!(
                    page,
                    elapsed = ?started.elapsed(),
                    "view minted"
                );
                view = Some(self.keep(page, hash, Artifact::View, minted));
            }
            if needs.thumb && thumb.is_none() {
                let started = Instant::now();
                // The view is the cheaper source: one area pass over the
                // preview instead of a second over the full page.
                let from = view.as_ref().unwrap_or(&native);
                let minted = preview::mint(from, preview::THUMB_EDGE);
                tracing::debug!(
                    page,
                    elapsed = ?started.elapsed(),
                    "thumb minted"
                );
                thumb = Some(self.keep(page, hash, Artifact::Thumb, minted));
            }
            // Natives are the page's own pixels; the store never holds
            // them, and only a target can ask for them.
            if needs.native && demand {
                self.publish(page, Artifact::Native, native, notify);
            }
            // `native` drops here: the store is the only cache.
        }
        if demand {
            tracing::debug!(
                page,
                thumb = needs.thumb,
                view = needs.view,
                native = needs.native,
                thumb_hit,
                view_hit,
                waited = ?self.targets_at.elapsed(),
                elapsed = ?serve_started.elapsed(),
                "target served"
            );
            // The view goes first: a still frame waits on it, and the
            // app's upload budget would otherwise defer it behind the
            // thumb by a frame. A skim wants only thumbs, so it is
            // unaffected.
            if needs.view {
                if let Some(view) = view {
                    self.publish(page, Artifact::View, view, notify);
                }
            }
            if needs.thumb {
                if let Some(thumb) = thumb {
                    self.publish(page, Artifact::Thumb, thumb, notify);
                }
            }
        }
    }
}

/// Handle to the decode thread. Every control is a cheap send; the
/// thread applies the queued messages in order before its next pick, so
/// the last retune wins and a newer plan overtakes the old one's tail.
pub struct Worker {
    tx: mpsc::Sender<ToWorker>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    /// Spawn the worker for `source`, opened on `initial`. `notify` fires
    /// from the worker thread with each artifact or failure; the worker
    /// knows nothing about winit.
    pub fn spawn(
        source: Arc<Source>,
        initial: Plan,
        notify: impl Fn(Event) + Send + Sync + 'static,
    ) -> Result<Self, SpawnError> {
        Self::launch(source, initial, Store::open(), notify)
    }

    /// Spawn against an explicit store root. Tests own the root so a
    /// warm-store case can reuse one; production takes [`Store::open`].
    #[cfg(test)]
    fn spawn_at(
        source: Arc<Source>,
        initial: Plan,
        root: PathBuf,
        notify: impl Fn(Event) + Send + Sync + 'static,
    ) -> Result<Self, SpawnError> {
        Self::launch(source, initial, Store::at(root), notify)
    }

    fn launch(
        source: Arc<Source>,
        initial: Plan,
        store: Store,
        notify: impl Fn(Event) + Send + Sync + 'static,
    ) -> Result<Self, SpawnError> {
        let (tx, rx) = mpsc::channel::<ToWorker>();
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
    fn tell(&self, message: ToWorker) {
        let Ok(()) = self.tx.send(message) else {
            return;
        };
    }

    /// Replace the plan. The in-flight decode still finishes and its
    /// result still reaches the store and the app.
    pub fn retune(&self, plan: Plan) {
        self.tell(ToWorker::Retune(plan));
    }

    pub fn set_motion(&self, motion: Motion) {
        self.tell(ToWorker::Motion(motion));
    }

    /// Replace the set of artifacts the app is missing. The app calls
    /// this only when the set changes; the worker serves the wants in
    /// the given order, ahead of the plan. A want the app drops and adds
    /// again is served again.
    pub fn retarget(&self, targets: Vec<Want>) {
        self.tell(ToWorker::Targets(targets));
    }

    /// Stop the worker without joining it, so the caller never waits out
    /// an in-flight decode. The thread owns its state.
    pub fn detach(&mut self) {
        self.tell(ToWorker::Stop);
        self.handle = None;
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Stop wakes a parked loop so the join returns.
        self.tell(ToWorker::Stop);
        if let Some(handle) = self.handle.take() {
            // Join Err means the thread died of a panic. Nothing can act
            // on the payload at teardown, so debug builds surface the
            // breach and release reaps regardless.
            let joined = handle.join();
            debug_assert!(joined.is_ok(), "worker thread panicked");
        }
    }
}

/// Worker loop: the channel is drained before every pick, so Stop and a
/// newer plan answer between decodes. One decode in flight always runs to
/// completion.
fn run(
    source: &Source,
    rx: &mpsc::Receiver<ToWorker>,
    mut session: Session,
    notify: &dyn Fn(Event),
) {
    loop {
        while let Ok(message) = rx.try_recv() {
            if !session.apply(message) {
                return;
            }
        }
        if let Some(job) = session.next_target() {
            tracing::debug!(
                page = job.page,
                needs = ?job.needs,
                waited = ?session.target_wait(),
                "target"
            );
            session.serve(source, job, notify);
            continue;
        }
        if let Some(job) = session.next_job() {
            tracing::debug!(
                page = job.page,
                thumb = job.needs.thumb,
                view = job.needs.view,
                motion = ?session.motion,
                preload = job.preload,
                "job"
            );
            if !job.preload || session.preloading {
                session.serve(source, job, notify);
                continue;
            }
            // Preload must not make the app wait. Let the app speak
            // first: a reader who is turning pages must never wait for
            // a page they have not asked for.
            match rx.recv_timeout(IDLE_PRELOAD) {
                Ok(message) => {
                    if !session.apply(message) {
                        return;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    session.preloading = true;
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
        // miss names a caller bug, not bad input.
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

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;
    use crate::decode::decode as decode_bytes;
    use crate::fault::DecodeRefusal;
    use crate::plan::Direction;
    use crate::preview::Store;
    use crate::testkit::eventually;
    use crate::testkit::tiny_png;
    use crate::testkit::TempDir;
    use crate::view::PageCount;
    use crate::view::Spread;
    use crate::view::ViewMode;

    use parking_lot::Mutex;

    /// Page edge for the tests that check minted and native dimensions:
    /// larger than the thumb edge, smaller than the view edge, so the
    /// thumb is scaled and the view is the page itself.
    const BIG: u32 = 1400;

    fn plan(anchor: usize, count: usize) -> Plan {
        Plan::build(
            Spread::lone(anchor),
            Direction::Forward,
            ViewMode::Normal,
            PageCount::new(count).expect("test book is non-empty"),
        )
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
            PageRead::Ready(bytes) => preview::content_hash(&bytes),
            PageRead::NotYet | PageRead::Failed(_) => {
                panic!("page {page} did not read")
            }
        }
    }

    fn long_edge(image: &Decoded) -> u32 {
        image.width().max(image.height())
    }

    fn is_thumb(ready: &Ready) -> bool {
        matches!(ready.kind, Artifact::Thumb)
    }

    fn is_view(ready: &Ready) -> bool {
        matches!(ready.kind, Artifact::View)
    }

    fn want_view(page: usize) -> Want {
        Want {
            page,
            kind: Artifact::View,
        }
    }

    fn want_thumb(page: usize) -> Want {
        Want {
            page,
            kind: Artifact::Thumb,
        }
    }

    fn want_native(page: usize) -> Want {
        Want {
            page,
            kind: Artifact::Native,
        }
    }

    /// The events the worker delivered, kept so several assertions can
    /// read the same stream.
    struct Events {
        rx: mpsc::Receiver<Event>,
        seen: Mutex<Vec<Event>>,
    }

    #[expect(
        clippy::rest_pattern_accessible_field,
        reason = "the ignored fields are unused; naming them would trip unneeded_field_pattern"
    )]
    impl Events {
        fn new() -> (Self, impl Fn(Event) + Send + Sync + 'static) {
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

        /// Wait for the first ready event matching `want`.
        fn ready(
            &self,
            what: &str,
            want: impl Fn(&Ready) -> bool,
        ) -> Arc<Decoded> {
            eventually(20, what, || {
                self.drain();
                let seen = self.seen.lock();
                seen.iter().find_map(|event| match event {
                    Event::Ready(ready) if want(ready) => {
                        Some(Arc::clone(&ready.image))
                    }
                    Event::Ready(_) | Event::Failed { .. } => None,
                })
            })
        }

        /// Wait for `page`'s failure.
        fn failure(&self, page: usize) -> Arc<PageError> {
            eventually(20, "page never failed", || {
                self.drain();
                let seen = self.seen.lock();
                seen.iter().find_map(|event| match event {
                    Event::Failed {
                        page: failed,
                        error,
                    } if *failed == page => Some(Arc::clone(error)),
                    Event::Ready(_) | Event::Failed { .. } => None,
                })
            })
        }

        /// How many ready events for `page` match `want`.
        fn delivered(
            &self,
            page: usize,
            want: impl Fn(&Ready) -> bool,
        ) -> usize {
            self.drain();
            let seen = self.seen.lock();
            seen.iter()
                .filter(|event| {
                    matches!(event, Event::Ready(ready)
                        if ready.page == page && want(ready))
                })
                .count()
        }

        /// How many failures `page` reported.
        fn failures(&self, page: usize) -> usize {
            self.drain();
            let seen = self.seen.lock();
            seen.iter()
                .filter_map(|event| match event {
                    Event::Failed { page: failed, error }
                        if *failed == page =>
                    {
                        Some(error)
                    }
                    Event::Ready(_) | Event::Failed { .. } => None,
                })
                .count()
        }
    }

    #[test]
    fn a_target_is_served_and_persisted() {
        let (_dir, source) = page_dir("worker-target", 1, BIG);
        let root = TempDir::new("worker-target-store");
        let (events, notify) = Events::new();
        let worker = Worker::spawn_at(
            Arc::clone(&source),
            plan(0, 1),
            root.path().to_path_buf(),
            notify,
        )
        .expect("worker spawns");

        worker.retarget(vec![want_view(0), want_thumb(0)]);
        let thumb = events
            .ready("the target never delivered a thumb", |ready| {
                ready.page == 0 && is_thumb(ready)
            });
        let view = events
            .ready("the target never delivered a view", |ready| {
                ready.page == 0 && is_view(ready)
            });
        assert_eq!(long_edge(&thumb), preview::THUMB_EDGE);
        assert_eq!(
            long_edge(&view),
            preview::VIEW_EDGE.min(BIG),
            "the view is minted at the view edge, never upscaled"
        );

        // One decode fed both artifacts, and both reached the store.
        let store = Store::at(root.path().to_path_buf());
        let hash = page_hash(&source, 0);
        assert!(
            store.load(hash, Artifact::Thumb).is_some(),
            "the thumb was not persisted"
        );
        assert!(
            store.load(hash, Artifact::View).is_some(),
            "the view was not persisted"
        );
    }

    #[test]
    fn the_tail_waits_for_a_quiet_app() {
        let (_dir, source) = page_dir("worker-gate", 6, 8);
        let root = TempDir::new("worker-gate-store");
        let _worker = Worker::spawn_at(
            Arc::clone(&source),
            plan(0, 6),
            root.path().to_path_buf(),
            |_| {},
        )
        .expect("worker spawns");
        let store = Store::at(root.path().to_path_buf());
        // The head is pages 0, 1 and 2; wait until it is fully produced.
        for page in [0, 1, 2] {
            eventually(20, "the head never reached the store", || {
                store
                    .load(page_hash(&source, page), Artifact::Thumb)
                    .is_some()
                    .then_some(())
            });
        }
        let head_done = Instant::now();
        // Page 3 is beyond the head: it must not start until the app has
        // been silent for the preload window.
        eventually(20, "the tail never preloaded", || {
            store
                .load(page_hash(&source, 3), Artifact::Thumb)
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
    fn a_failed_page_notifies_once_and_the_plan_advances() {
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
        let (events, notify) = Events::new();
        let _worker = Worker::spawn_at(
            Arc::clone(&source),
            plan(0, 5),
            root.path().to_path_buf(),
            notify,
        )
        .expect("worker spawns");

        let error = events.failure(0);
        assert!(
            error.downcast_inner_ref::<image::ImageError>().is_some()
                || error.downcast_inner_ref::<DecodeRefusal>().is_some(),
            "page 0 must fail as a decode fault, not {error}"
        );
        // The failure does not stall the plan: pages behind it still
        // reach the store.
        let store = Store::at(root.path().to_path_buf());
        eventually(20, "the failure stalled the plan", || {
            store
                .load(page_hash(&source, 4), Artifact::Thumb)
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
    fn the_plan_warms_the_store_without_delivering() {
        let (_dir, source) = page_dir("worker-plan", 1, BIG);
        let root = TempDir::new("worker-plan-store");
        let (events, notify) = Events::new();
        let _worker = Worker::spawn_at(
            Arc::clone(&source),
            plan(0, 1),
            root.path().to_path_buf(),
            notify,
        )
        .expect("worker spawns");

        let store = Store::at(root.path().to_path_buf());
        eventually(20, "the plan never produced the head view", || {
            store
                .load(page_hash(&source, 0), Artifact::View)
                .is_some()
                .then_some(())
        });
        events.drain();
        assert_eq!(
            events.delivered(0, |_| true),
            0,
            "the plan delivered an artifact nobody wanted"
        );
    }

    #[test]
    fn a_view_leaving_the_target_set_is_served_again() {
        let (_dir, source) = page_dir("worker-recover", 1, BIG);
        let root = TempDir::new("worker-recover-store");
        let (events, notify) = Events::new();
        let worker = Worker::spawn_at(
            Arc::clone(&source),
            plan(0, 1),
            root.path().to_path_buf(),
            notify,
        )
        .expect("worker spawns");

        worker.retarget(vec![want_view(0)]);
        events.ready("the target never delivered a view", |ready| {
            ready.page == 0 && is_view(ready)
        });
        // An unchanged set is the app still waiting for its own upload,
        // not a new demand.
        worker.retarget(vec![want_view(0)]);
        thread::sleep(Duration::from_millis(50));
        assert_eq!(
            events.delivered(0, is_view),
            1,
            "an unchanged target set fetched the view twice"
        );
        // The app uploaded it, so it left the set; asking again later
        // must fetch it again.
        worker.retarget(Vec::new());
        worker.retarget(vec![want_view(0)]);
        eventually(20, "the re-added view was never served", || {
            (events.delivered(0, is_view) >= 2).then_some(())
        });
    }

    #[test]
    fn a_native_target_carries_the_page_pixels() {
        let (_dir, source) = page_dir("worker-native", 1, BIG);
        let root = TempDir::new("worker-native-store");
        let (events, notify) = Events::new();
        let worker = Worker::spawn_at(
            Arc::clone(&source),
            plan(0, 1),
            root.path().to_path_buf(),
            notify,
        )
        .expect("worker spawns");

        worker.retarget(vec![want_native(0)]);
        let image = events.ready(
            "the native target was never honoured",
            |ready| {
                ready.page == 0 && matches!(ready.kind, Artifact::Native)
            },
        );
        assert_eq!(
            (image.width(), image.height()),
            (BIG, BIG),
            "native must carry the page's own pixels"
        );
    }

    #[test]
    fn a_warm_store_serves_without_decoding() {
        let (_dir, source) = page_dir("worker-warm", 1, BIG);
        let root = TempDir::new("worker-warm-store");
        let layout = root.path().to_path_buf();

        // The first session mints and persists both artifacts.
        let (first, notify) = Events::new();
        let worker = Worker::spawn_at(
            Arc::clone(&source),
            plan(0, 1),
            layout.clone(),
            notify,
        )
        .expect("worker spawns");
        worker.retarget(vec![want_thumb(0), want_view(0)]);
        let thumb = first
            .ready("the first session never delivered a thumb", |ready| {
                ready.page == 0 && is_thumb(ready)
            });
        let view = first
            .ready("the first session never delivered a view", |ready| {
                ready.page == 0 && is_view(ready)
            });
        assert_eq!(long_edge(&thumb), preview::THUMB_EDGE);
        assert_eq!(
            long_edge(&view),
            preview::VIEW_EDGE.min(BIG),
            "the view is minted at the view edge, never upscaled"
        );
        drop(worker);

        let hash = page_hash(&source, 0);
        let mut store = Store::at(layout.clone());
        assert!(
            store.load(hash, Artifact::Thumb).is_some(),
            "the thumb was not persisted"
        );
        assert!(
            store.load(hash, Artifact::View).is_some(),
            "the view was not persisted"
        );

        // Replace both persisted artifacts with a size no decode of this
        // page could produce, so the second session's events say where
        // the pixels came from.
        let sentinel =
            decode_bytes(&tiny_png(7, 7, [1, 2, 3, 255]), "sentinel.png")
                .expect("sentinel decodes");
        assert!(store.save(hash, Artifact::Thumb, &sentinel).landed);
        assert!(store.save(hash, Artifact::View, &sentinel).landed);
        drop(store);

        // A second worker on the same root loads instead of decoding.
        let (second, notify_second) = Events::new();
        let warm = Worker::spawn_at(
            Arc::clone(&source),
            plan(0, 1),
            layout.clone(),
            notify_second,
        )
        .expect("worker spawns");
        warm.retarget(vec![want_thumb(0), want_view(0)]);
        let warm_thumb = second
            .ready("the second session never delivered a thumb", |ready| {
                ready.page == 0 && is_thumb(ready)
            });
        let warm_view = second
            .ready("the second session never delivered a view", |ready| {
                ready.page == 0 && is_view(ready)
            });
        assert_eq!(
            (warm_thumb.width(), warm_thumb.height()),
            (7, 7),
            "the thumb was decoded again instead of loaded"
        );
        assert_eq!(
            (warm_view.width(), warm_view.height()),
            (7, 7),
            "the view was decoded again instead of loaded"
        );

        // Loading saves nothing over the artifacts it read.
        let kept = Store::at(layout)
            .load(hash, Artifact::View)
            .expect("the sentinel view stays");
        assert_eq!((kept.width(), kept.height()), (7, 7));
    }
}
