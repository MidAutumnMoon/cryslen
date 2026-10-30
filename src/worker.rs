//! Decode worker: one thread serving the spread the app is showing. One
//! book holds every decided page and the pending trail. The wanted
//! spread never leaves the book, and every other page leaves by recency
//! then distance. Nothing is persisted to disk.

use std::cmp::Reverse;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::io;
use std::mem;
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use parking_lot::Mutex;

use crate::decode::Decoded;
use crate::source::{panic_message, PageRead, Source};
use crate::view::{Spread, ViewMode};

/// Approximate decoded bytes the book may hold. The wanted spread
/// stays resident even over this budget.
const BOOK_BUDGET_BYTES: usize = 768 * 1024 * 1024;

enum ToWorker {
    Retune(Spread),
    Stop,
}

/// The worker's answer for one wanted spread, in spread order. An
/// undecided page reads as [`SeatView::Pending`], so there is no unknown
/// state.
#[derive(Clone)]
pub struct SpreadView {
    pub first: SeatView,
    pub second: Option<SeatView>,
}

/// One page's view: decided (`Ready`, `Failed`) or still [`SeatView::Pending`].
/// The book stores decided views only. A failed page is sticky, so its
/// error follows the page across turns.
#[derive(Clone)]
pub enum SeatView {
    Ready(Arc<Decoded>),
    Failed(String),
    Pending,
}

impl SeatView {
    fn resident_bytes(&self) -> usize {
        match self {
            Self::Ready(decoded) => decoded.resident_bytes(),
            Self::Failed(message) => message.len(),
            Self::Pending => 0,
        }
    }
}

impl SpreadView {
    /// Visual seating as [left, right] under `mode`.
    pub fn seats(&self, mode: ViewMode) -> [Option<&SeatView>; 2] {
        use crate::view;

        view::seating(&self.first, self.second.as_ref(), mode)
    }
}

/// Decided pages and the pending trail under one lock. The book owns the
/// newest wanted spread; every transition preserves the invariant that
/// each of its pages is decided or pending, never neither. A wanted page
/// joins the trail in `retune` and leaves it only by being decided in
/// `store`, whose eviction cannot take a wanted page.
struct Book {
    wanted: Spread,
    decided: HashMap<usize, SeatView>,
    pending: BTreeSet<usize>,
    last_wanted: HashMap<usize, u64>,
    recency_counter: u64,
    budget: usize,
    live_bytes: usize,
}

/// Eviction rank of a page never recorded as wanted. The recency
/// counter starts above it, so such pages sort before every wanted one.
const NEVER_WANTED_RANK: u64 = 0;

impl Book {
    fn new(initial: Spread, page_count: usize) -> Self {
        let mut book = Self {
            wanted: initial,
            decided: HashMap::new(),
            pending: BTreeSet::new(),
            last_wanted: HashMap::new(),
            recency_counter: 0,
            budget: BOOK_BUDGET_BYTES,
            live_bytes: 0,
        };
        book.retune(initial, page_count);
        book
    }

    fn retune(&mut self, wanted: Spread, page_count: usize) {
        self.wanted = wanted;
        self.recency_counter += 1;
        for page in [Some(wanted.first), wanted.second].into_iter().flatten() {
            self.last_wanted.insert(page, self.recency_counter);
            self.pending.insert(page);
        }
        self.pending.extend(neighbors(wanted.first, page_count));
        retain_trail(&mut self.pending, wanted.first, &wanted);
        self.assert_wanted_covered();
    }

    fn take_next(&mut self) -> Option<usize> {
        self.pending.retain(|page| !self.decided.contains_key(page));
        self.pending
            .iter()
            .copied()
            .min_by_key(|&page| next_pick_key(page, self.wanted.first))
    }

    fn store(&mut self, page: usize, seat_view: SeatView) {
        let resident_bytes = seat_view.resident_bytes();
        if let Some(old) = self.decided.insert(page, seat_view) {
            self.live_bytes -= old.resident_bytes();
        }
        self.live_bytes += resident_bytes;
        self.pending.remove(&page);
        if self.live_bytes > self.budget {
            self.evict();
        }
        self.assert_wanted_covered();
        self.assert_live_bytes();
    }

    fn snapshot(&self, wanted: &Spread) -> SpreadView {
        SpreadView {
            first: self.seat_view(wanted.first),
            second: wanted.second.map(|page| self.seat_view(page)),
        }
    }

    fn seat_view(&self, page: usize) -> SeatView {
        self.decided.get(&page).cloned().unwrap_or(SeatView::Pending)
    }

    /// Eviction victims are never-wanted pages first, then least
    /// recently wanted, then farthest from the spread. A wanted page is
    /// never a victim.
    fn evict(&mut self) {
        if self.live_bytes <= self.budget {
            return;
        }
        let wanted = self.wanted;
        let mut victims: Vec<usize> = self
            .decided
            .keys()
            .copied()
            .filter(|&page| !wanted.contains(page))
            .collect();
        victims.sort_unstable_by_key(|&page| {
            (
                self.last_wanted.get(&page).copied().unwrap_or(NEVER_WANTED_RANK),
                Reverse(page.abs_diff(wanted.first)),
                page,
            )
        });
        for page in victims {
            if let Some(old) = self.decided.remove(&page) {
                self.live_bytes -= old.resident_bytes();
            }
            if self.live_bytes <= self.budget {
                return;
            }
        }
        // The loop can end over budget when only wanted pages remain.
    }

    fn assert_wanted_covered(&self) {
        let covered =
            |page: usize| self.decided.contains_key(&page) || self.pending.contains(&page);
        debug_assert!(covered(self.wanted.first), "first wanted page is decided or pending");
        if let Some(second) = self.wanted.second {
            debug_assert!(covered(second), "second wanted page is decided or pending");
        }
    }

    fn assert_live_bytes(&self) {
        debug_assert_eq!(
            self.live_bytes,
            self.decided.values().map(SeatView::resident_bytes).sum(),
            "live_bytes matches the summed resident bytes"
        );
    }
}

/// Orders by distance to `anchor`, with `false` (ahead) sorting before
/// `true` (behind) at equal distance.
fn next_pick_key(page: usize, anchor: usize) -> (usize, bool) {
    (page.abs_diff(anchor), page < anchor)
}

/// How many pages on each side of the wanted spread's first page to
/// prefetch. A double-mode turn steps two pages, so the far page of
/// the next spread sits at distance 3. Single-page mode over-prefetches by
/// the same margin, which the book reclaims first as never-wanted.
const PREFETCH_RADIUS: usize = 3;

fn neighbors(anchor: usize, page_count: usize) -> Vec<usize> {
    use std::iter;

    iter::once(anchor)
        .chain(
            (1..=PREFETCH_RADIUS)
                .flat_map(|distance| [anchor.checked_sub(distance), anchor.checked_add(distance)])
                .flatten(),
        )
        .filter(|&page| page < page_count)
        .collect()
}

/// How many pages on each side of the wanted spread may stay pending. A
/// held key sends retunes faster than pages decode, so pages past this
/// radius are dropped rather than queued for a reader that has already
/// moved on. The wanted spread itself never drops.
const TRAIL_RADIUS: usize = 64;

fn retain_trail(pending: &mut BTreeSet<usize>, anchor: usize, wanted: &Spread) {
    pending.retain(|&page| page.abs_diff(anchor) <= TRAIL_RADIUS || wanted.contains(page));
}

/// Handle to the worker. `retune` is cheap and never blocks.
pub struct Worker {
    tx: mpsc::Sender<ToWorker>,
    book: Arc<Mutex<Book>>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    /// Spawn the worker for `source`, opened on `initial`. `notify`
    /// fires after every decided page and should send the winit user
    /// event. The worker itself knows nothing about winit.
    pub fn spawn(
        source: Arc<Source>,
        initial: Spread,
        notify: impl Fn() + Send + Sync + 'static,
    ) -> io::Result<Self> {
        use std::thread::Builder;

        let (tx, rx) = mpsc::channel::<ToWorker>();
        let book = Arc::new(Mutex::new(Book::new(initial, source.len())));
        let thread_book = Arc::clone(&book);
        let handle = Builder::new()
            .name("cy-decode".into())
            .spawn(move || run(&source, &rx, &thread_book, &notify))?;
        Ok(Self { tx, book, handle: Some(handle) })
    }

    /// A failed send means the thread already exited, so the message has
    /// no reader left.
    fn tell(&self, message: ToWorker) {
        let Ok(()) = self.tx.send(message) else { return; };
    }

    pub fn retune(&self, wanted: Spread) {
        self.tell(ToWorker::Retune(wanted));
    }

    /// The current view for both seats of `wanted`, read under one lock.
    pub fn snapshot(&self, wanted: &Spread) -> SpreadView {
        self.book.lock().snapshot(wanted)
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

/// How long the worker naps when the wanted page's bytes are still
/// unpacking. The loop head observes the channel after every nap, so
/// Stop and newer retunes answer within one nap.
const FILL_POLL: Duration = Duration::from_millis(15);

/// Worker loop: the channel is observed between decodes, so a newer
/// spread overtakes the backlog. One decode in flight always runs to
/// completion. A page the source has not unpacked yet makes the loop
/// nap instead of block, so no wait can outlive a Stop.
fn run(
    source: &Source,
    rx: &mpsc::Receiver<ToWorker>,
    book: &Mutex<Book>,
    notify: &dyn Fn(),
) {
    let page_count = source.len();
    // Retunes accumulate while a page decides; all are replayed, in
    // arrival order, so last-wanted recency stays truthful.
    let mut retunes: Vec<Spread> = Vec::new();
    loop {
        while let Ok(message) = rx.try_recv() {
            let ToWorker::Retune(spread) = message else { return };
            retunes.push(spread);
        }
        if !retunes.is_empty() {
            let mut book = book.lock();
            for spread in mem::take(&mut retunes) {
                book.retune(spread, page_count);
            }
        }
        // The scrutinee's MutexGuard drops before the else block runs,
        // so the lock is not held across the blocking recv below.
        let Some(page_index) = book.lock().take_next() else {
            let arrived = match rx.recv() {
                Ok(ToWorker::Retune(spread)) => spread,
                Ok(ToWorker::Stop) | Err(_) => return,
            };
            retunes.push(arrived);
            continue;
        };
        let seat_view = match decide_page(source, page_index) {
            Decision::Ready(page) => SeatView::Ready(page),
            Decision::Failed(message) => SeatView::Failed(message),
            Decision::NotYet => {
                thread::sleep(FILL_POLL);
                continue;
            }
        };
        book.lock().store(page_index, seat_view);
        notify();
    }
}

enum Decision {
    Ready(Arc<Decoded>),
    Failed(String),
    NotYet,
}

/// A terminal failure labelled with the page's 1-based display number.
/// This is the only place the prefix is added.
fn page_failure(page_index: usize, message: impl fmt::Display) -> Decision {
    Decision::Failed(format!("page {}: {message}", page_index + 1))
}

/// Resolve one page to a decision: decoded, a terminal failure, or a
/// nap because the unpack has not reached it. The decoder runs behind
/// the panic boundary, so a panicking decoder must not kill the only
/// worker thread.
fn decide_page(source: &Source, page_index: usize) -> Decision {
    use crate::decode;
    use std::panic::{self, AssertUnwindSafe};

    let Some(name) = source.name(page_index) else {
        return page_failure(page_index, "no such page");
    };
    match source.bytes(page_index) {
        PageRead::Ready(bytes) => {
            let result = panic::catch_unwind(AssertUnwindSafe(|| decode::decode(&bytes, name)));
            match result {
                Ok(Ok(page)) => Decision::Ready(Arc::new(page)),
                Ok(Err(error)) => page_failure(page_index, error),
                Err(payload) => page_failure(
                    page_index,
                    format!("decoder {}", panic_message(payload.as_ref())),
                ),
            }
        }
        PageRead::Failed(message) => page_failure(page_index, message),
        PageRead::NotYet => Decision::NotYet,
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, clippy::panic, reason = "a panic in test code is inconsequential")]
mod tests {
    use super::*;
    use crate::testkit::{eventually, ready_seat, tiny_png, TempDir};

    use parking_lot::Condvar;
    use std::time::{Duration, Instant};

    #[test]
    fn neighbors_are_nearest_first() {
        assert_eq!(neighbors(5, 10), vec![5, 4, 6, 3, 7, 2, 8]);
        assert_eq!(neighbors(0, 4), vec![0, 1, 2, 3]);
        assert_eq!(neighbors(9, 10), vec![9, 8, 7, 6]);
    }

    // Pins PREFETCH_RADIUS against view's double-mode turn math.
    #[test]
    fn prefetch_window_covers_double_mode_turns() {
        use crate::view;

        for page_count in [2_usize, 9, 13] {
            let spreads = 1 + (page_count - 1).div_ceil(2);
            let mut focus = 0;
            for turn in 0..spreads {
                let window = neighbors(focus, page_count);
                let spread = view::spread_for(focus, ViewMode::Manga, page_count);
                for page in [Some(spread.first), spread.second].into_iter().flatten() {
                    assert!(
                        window.contains(&page),
                        "page count {page_count}: spread page {page} outside focus {focus}"
                    );
                }
                // The wrap turn jumps across the book, which prefetch
                // cannot cover.
                if turn + 1 < spreads {
                    let next = view::advance(focus, true, ViewMode::Manga, page_count);
                    let next_spread =
                        view::spread_for(next, ViewMode::Manga, page_count);
                    for page in [Some(next_spread.first), next_spread.second]
                        .into_iter()
                        .flatten()
                    {
                        assert!(
                            window.contains(&page),
                            "page count {page_count}: next spread page {page} outside focus {focus}"
                        );
                    }
                    focus = next;
                }
            }
        }
    }

    #[test]
    fn next_pick_is_nearest_then_ahead() {
        // A page count of 0 admits no neighbours, so the trail starts as
        // the wanted pair alone.
        let mut book = Book::new(Spread { first: 10, second: Some(11) }, 0);
        book.store(10, ready_seat(2, 1));
        book.pending.extend(8..=12);
        assert_eq!(book.take_next(), Some(11));
        assert!(!book.pending.contains(&10));
        let mut order = Vec::new();
        while let Some(page) = book.take_next() {
            order.push(page);
            book.pending.remove(&page);
        }
        assert_eq!(order, vec![11, 9, 12, 8]);
    }

    // store()'s debug asserts guard wanted coverage and live_bytes. The
    // kept/dropped checks below also protect release builds.
    #[test]
    fn eviction_rank() {
        let wanted = Spread { first: 5, second: Some(6) };
        for (budget, retunes, stores, kept, dropped) in [
            (8, vec![wanted], vec![1_usize, 3], vec![3_usize], vec![1_usize]),
            (8, vec![Spread::lone(1), wanted], vec![1, 3], vec![1], vec![3]),
            (8, vec![Spread::lone(1), Spread::lone(2), wanted], vec![1, 2], vec![2], vec![1]),
        ] {
            let mut book = Book::new(Spread::lone(0), 10);
            book.budget = budget;
            for spread in &retunes {
                book.retune(*spread, 10);
            }
            for &page in &stores {
                book.store(page, ready_seat(2, 1));
            }
            for &page in &kept {
                assert!(
                    book.decided.contains_key(&page),
                    "kept {page} evicted: retunes {retunes:?} stores {stores:?}"
                );
            }
            for &page in &dropped {
                assert!(
                    !book.decided.contains_key(&page),
                    "dropped {page} survived: retunes {retunes:?} stores {stores:?}"
                );
            }
            assert!(book.live_bytes <= book.budget, "live bytes over budget: {retunes:?} {stores:?}");
        }
    }

    #[test]
    fn saturating_store_never_drops_the_wanted_spread() {
        let mut book = Book::new(Spread::lone(1), 20);
        book.budget = 16;
        let wanted = Spread { first: 7, second: Some(8) };
        book.retune(wanted, 20);
        book.store(1, ready_seat(2, 1));
        book.store(7, ready_seat(2, 1));
        book.store(8, ready_seat(2, 1));
        assert!(book.decided.contains_key(&7));
        assert!(book.decided.contains_key(&8));
        assert!(!book.decided.contains_key(&1));
        let snapshot = book.snapshot(&wanted);
        assert!(matches!(snapshot.first, SeatView::Ready(_)));
        assert!(matches!(snapshot.second, Some(SeatView::Ready(_))));
    }

    // 10x10 rgba is 400 bytes against a budget of 10: the store itself
    // saturates the book.
    #[test]
    fn wanted_spread_survives_its_own_oversize() {
        let wanted = Spread { first: 3, second: Some(4) };
        let mut book = Book::new(wanted, 10);
        book.budget = 10;
        book.store(3, ready_seat(10, 10));
        assert!(book.decided.contains_key(&3));
    }

    fn tiny_source(tag: &str, page_count: u8) -> (TempDir, Arc<Source>) {
        let dir = TempDir::new(tag);
        for page in 0..page_count {
            dir.write(&format!("p{page:03}.png"), &tiny_png(1, 1, [page, 0, 0, 255]));
        }
        let source = Arc::new(Source::open(dir.path()).expect("temp source opens"));
        (dir, source)
    }

    /// Parks the worker after each store until the test releases that
    /// store's number. Retunes sent while parked are guaranteed queued
    /// behind it, which pins the decode order for assertions.
    #[derive(Clone)]
    struct Gate(Arc<(Mutex<GateTickets>, Condvar)>);

    struct GateTickets {
        next: u64,
        released_through: u64,
    }

    impl Gate {
        fn new() -> Self {
            Self(Arc::new((Mutex::new(GateTickets { next: 0, released_through: 0 }), Condvar::new())))
        }

        fn hold(&self) {
            let (state, cv) = &*self.0;
            let mut state = state.lock();
            state.next += 1;
            let ticket = state.next;
            // Times out so a failed assertion cannot hang Drop's join.
            let deadline = Instant::now() + Duration::from_secs(5);
            cv.wait_while_for(
                &mut state,
                |state| state.released_through < ticket,
                deadline.saturating_duration_since(Instant::now()),
            );
            drop(state);
        }

        fn release(&self, through: u64) {
            let (state, cv) = &*self.0;
            state.lock().released_through = through;
            cv.notify_all();
        }
    }

    fn wait_ready(worker: &Worker, pages: &[usize]) {
        for &page in pages {
            eventually(20, &format!("page {page} not decoded in time"), || {
                match worker.snapshot(&Spread::lone(page)).first {
                    SeatView::Ready(seat) => Some(seat),
                    SeatView::Failed(message) => panic!("page {page} failed: {message}"),
                    SeatView::Pending => None,
                }
            });
        }
    }

    #[test]
    fn worker_end_to_end() {
        let dir = TempDir::new("worker-e2e");
        dir.write("bad.png", b"not a png at all");
        dir.write("ok.png", &tiny_png(3, 2, [0, 255, 0, 255]));
        let source = Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let updates = Arc::new(Mutex::new(0_usize));
        let counted = Arc::clone(&updates);
        let ok = Spread::lone(1);
        let worker = Worker::spawn(source, ok, move || {
            *counted.lock() += 1;
        })
        .expect("worker spawns");

        wait_ready(&worker, &[1]);
        let SeatView::Ready(page) = worker.snapshot(&ok).first else {
            panic!("ok page not Ready");
        };
        assert_eq!((page.width, page.height), (3, 2));
        assert!(*updates.lock() > 0);

        let bad = Spread::lone(0);
        worker.retune(bad);
        let failed = eventually(20, "page 0 not decided in time", || {
            match worker.snapshot(&bad).first {
                SeatView::Failed(message) => Some(message),
                SeatView::Ready(_) => panic!("bad.png decoded as Ready?"),
                SeatView::Pending => None,
            }
        });
        assert!(failed.starts_with("page 1: "), "{failed}");
        assert!(!failed.contains("page 1: page 1"), "{failed}");
        // The entry is sticky across turns: turn to the ok page, then back.
        worker.retune(ok);
        wait_ready(&worker, &[1]);
        worker.retune(bad);
        let revisit = worker.snapshot(&bad).first;
        assert!(
            matches!(&revisit, SeatView::Failed(message) if *message == failed),
            "page 0 lost its sticky failure across a turn"
        );
        // Drop joins the worker here, so a broken stop path hangs the
        // suite rather than failing an assert.
    }

    /// The worker naps through an unpack instead of failing or
    /// hanging: seeded on the page the zip fill reaches last, the seat
    /// settles once that page lands, and a retune afterwards settles
    /// the first page from the fill.
    #[test]
    fn worker_waits_out_an_unpack() {
        use std::fs::File;
        use std::io::Write as _;
        use zip::write::SimpleFileOptions;

        let dir = TempDir::new("worker-unpack");
        let zip_path = dir.path().join("book.cbz");
        {
            let file = File::create(&zip_path).expect("zip creates");
            let mut zip = zip::ZipWriter::new(file);
            let options = SimpleFileOptions::default();
            // 32 MiB of zeros inflates slowly enough in a debug build that
            // the far page is still NotYet when the worker first asks
            // for it. Zeros are not a PNG, so the seat settles as a
            // decode failure; the point is that it settles at all.
            for page in 0..32_u32 {
                zip.start_file(format!("{page:03}.png"), options).expect("entry starts");
                zip.write_all(&vec![0_u8; 1024 * 1024]).expect("entry writes");
            }
            zip.finish().expect("zip finishes");
        }
        let source = Arc::new(Source::open(&zip_path).expect("zip opens"));
        let far = source.len() - 1;
        let worker =
            Worker::spawn(Arc::clone(&source), Spread::lone(far), || {}).expect("worker spawns");

        let settled = |page: usize| {
            eventually(20, "page never settled", || {
                match worker.snapshot(&Spread::lone(page)).first {
                    SeatView::Pending => None,
                    SeatView::Failed(message) => Some(message),
                    SeatView::Ready(_) => panic!("zeros decoded as an image?"),
                }
            })
        };
        let far_page = settled(far);
        assert!(far_page.starts_with(&format!("page {}: ", far + 1)), "{far_page}");
        assert!(!far_page.contains("no entry in the archive"), "{far_page}");
        assert!(!far_page.contains("could not start the unpack thread"), "{far_page}");
        worker.retune(Spread::lone(0));
        let first_page = settled(0);
        assert!(first_page.starts_with("page 1: "), "{first_page}");
    }

    #[test]
    fn retune_decoding() {
        let (_dir, source) = tiny_source("worker-retune", 50);
        let worker = Worker::spawn(source, Spread::lone(0), || {}).expect("worker spawns");
        // 50 retunes queue faster than pages decode; every page still decodes
        for page in 0..50 {
            worker.retune(Spread::lone(page));
        }
        wait_ready(&worker, &(0..50).collect::<Vec<_>>());
        worker.retune(Spread { first: 3, second: Some(4) });
        wait_ready(&worker, &[3, 4]);
        // The last-wanted records are what keep a wanted spread in the
        // book, so the wiring is pinned on them.
        let was_wanted = |page: usize| {
            worker.book.lock().last_wanted.contains_key(&page)
        };
        assert!(was_wanted(3), "page 3 never recorded as wanted");
        assert!(was_wanted(4), "page 4 never recorded as wanted");
    }

    #[test]
    fn trail_radius_bounds_decoding() {
        let (_dir, source) = tiny_source("worker-bound", 140);
        let gate = Gate::new();
        let parked = gate.clone();
        let worker = Worker::spawn(source, Spread::lone(0), move || parked.hold())
            .expect("worker spawns");
        wait_ready(&worker, &[0]);
        // All 139 retunes queue while the worker is parked on page 0.
        for page in 1..140 {
            worker.retune(Spread::lone(page));
        }
        gate.release(u64::MAX);
        wait_ready(&worker, &(75..140).collect::<Vec<_>>());
        for page in 1..75 {
            assert!(
                matches!(worker.snapshot(&Spread::lone(page)).first, SeatView::Pending),
                "page {page} decoded past the trail radius"
            );
        }
    }

    #[test]
    fn newer_retune_overtakes_far_pending() {
        let (_dir, source) = tiny_source("worker-prio", 70);
        let gate = Gate::new();
        let parked = gate.clone();
        let worker = Worker::spawn(source, Spread::lone(0), move || parked.hold())
            .expect("worker spawns");
        wait_ready(&worker, &[0]);
        worker.retune(Spread::lone(60));
        gate.release(1);
        wait_ready(&worker, &[60]);
        // The worker is parked on page 60's store, so the old window
        // pages 1 and 2 cannot have been decoded before it.
        for page in [1, 2] {
            assert!(
                matches!(worker.snapshot(&Spread::lone(page)).first, SeatView::Pending),
                "page {page} decoded before the wanted page"
            );
        }
        gate.release(u64::MAX);
    }
}
