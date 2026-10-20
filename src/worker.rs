//! Decode worker: one thread serving the spread the app is showing. One
//! book holds every decided page and the pending trail, the wanted
//! spread never leaves it, and every other page leaves by recency then
//! distance. No disk persistence at all.

use std::cmp::Reverse;
use std::collections::{BTreeSet, HashMap};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

use crate::decode::Decoded;
use crate::source::Source;
use crate::view::Spread;

/// Byte budget of the book: approximate decoded bytes it may hold. The
/// wanted spread stays resident even over this budget, and frames the
/// display holds outlive the book entirely.
const BOOK_BUDGET_BYTES: usize = 768 * 1024 * 1024;

enum ToWorker {
    Retune(Spread),
    Stop,
}

/// One page's settled outcome. A failed page is sticky, so waiting,
/// error display, and hold-last-frame are all pure reads of the book,
/// and the failure follows the page across turns.
enum Outcome {
    Ready(Arc<Decoded>),
    Failed(String),
}

impl Outcome {
    fn resident_bytes(&self) -> usize {
        match self {
            Outcome::Ready(decoded) => decoded.resident_bytes(),
            Outcome::Failed(message) => message.len(),
        }
    }
}

/// The worker's answer for one wanted spread, in spread order. There is
/// no unknown state: a page without an outcome reads as [`SeatView::Pending`].
#[derive(Clone)]
pub struct SpreadView {
    pub first: SeatView,
    pub second: Option<SeatView>,
}

#[derive(Clone)]
pub enum SeatView {
    Ready(Arc<Decoded>),
    Failed(String),
    Pending,
}

impl SpreadView {
    /// Visual seating as [left, right], mirroring [`crate::view::Spread::seats`].
    /// Manga reads first-right, so a lone page lands in the right seat.
    pub fn seats(&self, rtl: bool) -> [Option<&SeatView>; 2] {
        if rtl {
            [self.second.as_ref(), Some(&self.first)]
        } else {
            [Some(&self.first), self.second.as_ref()]
        }
    }
}

/// Decided pages and the pending trail under one lock, so every
/// transition preserves the whole-book invariant: each page of the
/// wanted spread is decided or pending, never neither. A wanted page
/// joins the trail in `retune` and leaves it only by being decided in
/// `store`, whose eviction cannot take a wanted page. The invariant has
/// no exit.
struct Book {
    outcomes: HashMap<usize, Outcome>,
    pending: BTreeSet<usize>,
    /// When each page was last wanted, written only by `retune`. A page
    /// never recorded here was never wanted and leaves the book first.
    last_wanted: HashMap<usize, u64>,
    recency_counter: u64,
    budget: usize,
    live_bytes: usize,
}

/// Eviction rank of a page never recorded as wanted. The recency
/// counter starts above it, so such pages sort before every wanted one.
const NEVER_WANTED_RANK: u64 = 0;

impl Book {
    fn new() -> Book {
        Book {
            outcomes: HashMap::new(),
            pending: BTreeSet::new(),
            last_wanted: HashMap::new(),
            recency_counter: 0,
            budget: BOOK_BUDGET_BYTES,
            live_bytes: 0,
        }
    }

    /// Point the worker at a spread. Both pages are recorded as last
    /// wanted, joined to the trail with their neighbours, and protected
    /// from trail pruning.
    fn retune(&mut self, wanted: Spread, page_count: usize) {
        self.recency_counter += 1;
        self.last_wanted.insert(wanted.first, self.recency_counter);
        if let Some(second) = wanted.second {
            self.last_wanted.insert(second, self.recency_counter);
        }
        self.pending.insert(wanted.first);
        if let Some(second) = wanted.second {
            self.pending.insert(second);
        }
        self.pending.extend(neighbors(wanted.first, page_count));
        retain_trail(&mut self.pending, wanted.first, &wanted);
        self.assert_wanted_covered(&wanted);
    }

    /// The pending page to decode next: the undecided page nearest the
    /// spread's first, ahead of the reader before behind it. Pages that
    /// already hold an outcome leave the pending set, so a page is never
    /// decoded twice.
    fn take_next(&mut self, wanted: &Spread) -> Option<usize> {
        self.pending.retain(|page| !self.outcomes.contains_key(page));
        self.pending
            .iter()
            .copied()
            .min_by_key(|&page| next_pick_key(page, wanted.first))
    }

    /// Decide a pending page, then evict until under budget. Wanted
    /// pages never go, so an oversized spread stays over budget rather
    /// than dropping a page the display is showing.
    fn store(&mut self, page: usize, outcome: Outcome, wanted: &Spread) {
        let bytes = outcome.resident_bytes();
        if let Some(old) = self.outcomes.insert(page, outcome) {
            self.live_bytes -= old.resident_bytes();
        }
        self.live_bytes += bytes;
        self.pending.remove(&page);
        if self.live_bytes > self.budget {
            self.evict(wanted);
        }
        self.assert_wanted_covered(wanted);
    }

    /// Both seats of the wanted spread in one lock acquisition. Cloning
    /// a `Ready` page is an `Arc` bump.
    fn snapshot(&self, wanted: &Spread) -> SpreadView {
        SpreadView {
            first: self.seat_view(wanted.first),
            second: wanted.second.map(|page| self.seat_view(page)),
        }
    }

    fn seat_view(&self, page: usize) -> SeatView {
        match self.outcomes.get(&page) {
            Some(Outcome::Ready(decoded)) => SeatView::Ready(Arc::clone(decoded)),
            Some(Outcome::Failed(message)) => SeatView::Failed(message.clone()),
            None => SeatView::Pending,
        }
    }

    /// Eviction victims are never-wanted pages first, then least
    /// recently wanted, then farthest from the spread. A wanted page has
    /// no path into the victim list.
    fn evict(&mut self, wanted: &Spread) {
        if self.live_bytes <= self.budget {
            return;
        }
        let mut victims: Vec<usize> = self
            .outcomes
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
            if let Some(old) = self.outcomes.remove(&page) {
                self.live_bytes -= old.resident_bytes();
            }
            if self.live_bytes <= self.budget {
                return;
            }
        }
        // Only wanted pages remain, and they stay even over budget.
    }

    fn assert_wanted_covered(&self, wanted: &Spread) {
        let covered =
            |page: usize| self.outcomes.contains_key(&page) || self.pending.contains(&page);
        debug_assert!(covered(wanted.first));
        if let Some(second) = wanted.second {
            debug_assert!(covered(second));
        }
    }
}

/// Handle to the worker. `retune` is cheap and never blocks.
pub struct Worker {
    tx: mpsc::Sender<ToWorker>,
    book: Arc<Mutex<Book>>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    /// Spawn the worker for `source`. `notify` fires after every decided
    /// page and should send the winit user event. The worker itself
    /// knows nothing about winit.
    pub fn spawn(source: Arc<Source>, notify: impl Fn() + Send + Sync + 'static) -> Worker {
        let (tx, rx) = mpsc::channel::<ToWorker>();
        let book = Arc::new(Mutex::new(Book::new()));
        let handle = std::thread::Builder::new()
            .name("cy-decode".into())
            .spawn({
                let book = Arc::clone(&book);
                move || run(&source, rx, &book, &notify)
            })
            .expect("spawn decode worker");
        Worker { tx, book, handle: Some(handle) }
    }

    /// Tell the worker which spread the app is now showing. Every
    /// retune joins the spread to the decode trail.
    pub fn retune(&self, wanted: Spread) {
        let _ = self.tx.send(ToWorker::Retune(wanted));
    }

    /// The current outcome for both seats of `wanted`, read under one
    /// lock.
    pub fn snapshot(&self, wanted: &Spread) -> SpreadView {
        self.book.lock().expect("decode book lock").snapshot(wanted)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Stop wakes a parked loop so the join returns.
        let _ = self.tx.send(ToWorker::Stop);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Worker loop: every retune records the wanted pages and joins the
/// pending trail, and pages past `TRAIL_RADIUS` of the wanted spread
/// are dropped. One decode at a time, the undecided pending page
/// nearest the spread first, ahead of the reader before behind. The
/// channel is observed between decodes, so a newer spread overtakes the
/// backlog. One decode in flight always runs to completion.
fn run(
    source: &Source,
    rx: mpsc::Receiver<ToWorker>,
    book: &Mutex<Book>,
    notify: &dyn Fn(),
) {
    let page_count = source.len();
    let (mut wanted, mut retunes): (Spread, Vec<Spread>) = match rx.recv() {
        Ok(ToWorker::Retune(wanted)) => (wanted, vec![wanted]),
        Ok(ToWorker::Stop) | Err(_) => return,
    };
    loop {
        while let Ok(message) = rx.try_recv() {
            let ToWorker::Retune(spread) = message else { return };
            retunes.push(spread);
        }
        if let Some(&newest) = retunes.last() {
            wanted = newest;
            {
                let mut book = book.lock().expect("decode book lock");
                for &spread in &retunes {
                    book.retune(spread, page_count);
                }
            }
            retunes.clear();
        }
        let Some(page_index) = book.lock().expect("decode book lock").take_next(&wanted) else {
            let arrived = match rx.recv() {
                Ok(ToWorker::Retune(spread)) => spread,
                Ok(ToWorker::Stop) | Err(_) => return,
            };
            retunes.push(arrived);
            continue;
        };
        let outcome = decode_page(source, page_index);
        {
            let mut book = book.lock().expect("decode book lock");
            let outcome = match outcome {
                Ok(page) => Outcome::Ready(page),
                Err(message) => Outcome::Failed(message),
            };
            book.store(page_index, outcome, &wanted);
        }
        notify();
    }
}

/// Decode priority key: distance to the spread's first page, and at
/// equal distance a page ahead of the reader before one behind it.
fn next_pick_key(page: usize, anchor: usize) -> (usize, bool) {
    (page.abs_diff(anchor), page < anchor)
}

/// One page through the panic boundary: a panicking decoder must not kill
/// the only worker thread. The error string is the status-line message.
fn decode_page(source: &Source, page_index: usize) -> Result<Arc<Decoded>, String> {
    let name = source.name(page_index).to_owned();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> eros::Result<Decoded> {
            let bytes = source.bytes(page_index)?;
            Ok(crate::decode::decode(&bytes, &name)?)
        },
    ));
    match result {
        Ok(Ok(page)) => Ok(Arc::new(page)),
        Ok(Err(e)) => Err(format!("page {}: {e}", page_index + 1)),
        Err(panic) => Err(format!("page {}: {}", page_index + 1, panic_message(panic))),
    }
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = panic.downcast_ref::<&str>() {
        format!("decoder panicked: {text}")
    } else if let Some(text) = panic.downcast_ref::<String>() {
        format!("decoder panicked: {text}")
    } else {
        "decoder panicked".to_string()
    }
}

/// How many pages on each side of the wanted spread to prefetch. The
/// app anchors each spread on its first page, and a double-mode turn
/// steps two pages, so the far page of the next spread sits at distance
/// 3. Single-page mode over-prefetches by the same margin, which the
/// book reclaims first as never-wanted.
const PREFETCH_RADIUS: i64 = 3;

/// Decode order around the wanted spread: its first page, then
/// neighbours out to [`PREFETCH_RADIUS`], nearest first.
fn neighbors(current: usize, page_count: usize) -> Vec<usize> {
    let mut pages: Vec<usize> = (0..=PREFETCH_RADIUS)
        .flat_map(|distance| {
            [current as i64 - distance, current as i64 + distance]
        })
        .filter(|&page| (0..page_count as i64).contains(&page))
        .map(|page| page as usize)
        .collect();
    pages.sort_unstable_by_key(|&page| (page.abs_diff(current), page));
    pages.dedup();
    pages
}

/// How many pages on each side of the wanted spread may stay pending. A
/// held key sends retunes faster than pages decode, so pages past this
/// radius are dropped rather than queued for a reader that has already
/// moved on. The wanted spread itself never drops.
const TRAIL_RADIUS: i64 = 64;

fn retain_trail(pending: &mut BTreeSet<usize>, anchor: usize, wanted: &Spread) {
    pending.retain(|&page| {
        (page as i64 - anchor as i64).abs() <= TRAIL_RADIUS || wanted.contains(page)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{tiny_png, TempDir};
    use crate::view::Spread;

    use std::sync::Condvar;
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
        for page_count in [2usize, 9, 13] {
            let spreads = 1 + (page_count - 1).div_ceil(2);
            let mut focus = 0;
            for turn in 0..spreads {
                let window = neighbors(focus, page_count);
                let spread = crate::view::spread_for(focus, true, page_count);
                for page in [Some(spread.first), spread.second].into_iter().flatten() {
                    assert!(
                        window.contains(&page),
                        "page count {page_count}: spread page {page} outside focus {focus}"
                    );
                }
                // The wrap turn jumps across the book, which prefetch
                // cannot cover.
                if turn + 1 < spreads {
                    let next = crate::view::advance(focus, true, true, page_count);
                    let next_spread = crate::view::spread_for(next, true, page_count);
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
    fn view_seating_follows_reading_direction() {
        let lone = SpreadView { first: SeatView::Pending, second: None };
        assert!(matches!(lone.seats(true), [None, Some(SeatView::Pending)]));
        assert!(matches!(lone.seats(false), [Some(SeatView::Pending), None]));
        let pair =
            SpreadView { first: SeatView::Pending, second: Some(SeatView::Failed("x".into())) };
        assert!(matches!(pair.seats(true), [Some(SeatView::Failed(_)), Some(SeatView::Pending)]));
        assert!(matches!(pair.seats(false), [Some(SeatView::Pending), Some(SeatView::Failed(_))]));
    }

    #[test]
    fn trail_retention_drops_pages_past_the_radius() {
        let mut pending: BTreeSet<usize> = (0..=200).collect();
        let wanted = Spread { first: 100, second: Some(101) };
        retain_trail(&mut pending, 100, &wanted);
        let kept = (100 - TRAIL_RADIUS as usize)..=(100 + TRAIL_RADIUS as usize);
        assert_eq!(pending, kept.collect());
    }

    #[test]
    fn next_pick_is_nearest_then_ahead() {
        let mut book = Book::new();
        let wanted = Spread { first: 10, second: Some(11) };
        book.pending.extend([10, 11]);
        book.store(10, ready_page(10), &wanted);
        book.pending.extend(8..=12);
        assert_eq!(book.take_next(&wanted), Some(11));
        assert!(!book.pending.contains(&10));
        let mut order = Vec::new();
        while let Some(page) = book.take_next(&wanted) {
            order.push(page);
            book.pending.remove(&page);
        }
        assert_eq!(order, vec![11, 9, 12, 8]);
    }

    fn ready_page(label: u8) -> Outcome {
        Outcome::Ready(Arc::new(
            crate::decode::Decoded::rgba(
                2,
                1,
                crate::decode::Format::Png,
                vec![label; 8].into_boxed_slice(),
            )
            .unwrap(),
        ))
    }

    #[test]
    fn evicts_unwanted_pages_far_first() {
        let mut book = Book::new();
        book.budget = 8;
        let wanted = Spread { first: 5, second: Some(6) };
        book.retune(wanted, 10);
        book.store(1, ready_page(1), &wanted);
        book.store(3, ready_page(3), &wanted);
        assert!(book.outcomes.contains_key(&3));
        assert!(!book.outcomes.contains_key(&1));
        assert!(book.live_bytes <= book.budget);
    }

    #[test]
    fn once_wanted_page_outlives_never_wanted_page() {
        let mut book = Book::new();
        book.budget = 8;
        let wanted = Spread { first: 5, second: Some(6) };
        book.retune(Spread { first: 1, second: None }, 10);
        book.retune(wanted, 10);
        book.store(1, ready_page(1), &wanted);
        // Page 3 is nearer the spread but was never wanted, so it goes.
        book.store(3, ready_page(3), &wanted);
        assert!(book.outcomes.contains_key(&1));
        assert!(!book.outcomes.contains_key(&3));
    }

    #[test]
    fn once_wanted_pages_evict_least_recently_wanted_first() {
        let mut book = Book::new();
        book.budget = 8;
        let wanted = Spread { first: 5, second: Some(6) };
        book.retune(Spread { first: 1, second: None }, 10);
        book.retune(Spread { first: 2, second: None }, 10);
        book.retune(wanted, 10);
        book.store(1, ready_page(1), &wanted);
        book.store(2, ready_page(2), &wanted);
        assert!(book.outcomes.contains_key(&2));
        assert!(!book.outcomes.contains_key(&1));
    }

    // The endgame pin: a full book must evict the oldest unwanted page,
    // never a page of the spread the user just turned to.
    #[test]
    fn saturating_store_never_drops_the_wanted_spread() {
        let mut book = Book::new();
        book.budget = 16;
        let wanted = Spread { first: 7, second: Some(8) };
        book.retune(Spread { first: 1, second: None }, 20);
        book.retune(wanted, 20);
        book.store(1, ready_page(1), &wanted);
        book.store(7, ready_page(7), &wanted);
        book.store(8, ready_page(8), &wanted);
        assert!(book.outcomes.contains_key(&7));
        assert!(book.outcomes.contains_key(&8));
        assert!(!book.outcomes.contains_key(&1));
        let snapshot = book.snapshot(&wanted);
        assert!(matches!(snapshot.first, SeatView::Ready(_)));
        assert!(matches!(snapshot.second, Some(SeatView::Ready(_))));
    }

    #[test]
    fn wanted_spread_survives_its_own_oversize() {
        let mut book = Book::new();
        book.budget = 10;
        let wanted = Spread { first: 3, second: Some(4) };
        book.retune(wanted, 10);
        let oversize_page = Outcome::Ready(Arc::new(
            crate::decode::Decoded::rgba(
                10,
                10,
                crate::decode::Format::Png,
                vec![0; 400].into_boxed_slice(),
            )
            .unwrap(),
        ));
        book.store(3, oversize_page, &wanted);
        assert!(book.outcomes.contains_key(&3));
    }

    #[test]
    fn retune_decodes_and_notifies() {
        let dir = TempDir::new("worker");
        for i in 0..3 {
            dir.write(&format!("p{i}.png"), &tiny_png(3, 2, [i as u8, 0, 0, 255]));
        }
        let source = Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let updates = Arc::new(Mutex::new(0usize));
        let counted = Arc::clone(&updates);
        let worker = Worker::spawn(source, move || {
            *counted.lock().unwrap() += 1;
        });
        worker.retune(Spread { first: 1, second: None });

        let deadline = Instant::now() + Duration::from_secs(20);
        while !matches!(
            worker.snapshot(&Spread { first: 1, second: None }).first,
            SeatView::Ready(_)
        ) && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        let SeatView::Ready(page) =
            worker.snapshot(&Spread { first: 1, second: None }).first
        else {
            panic!("not Ready");
        };
        assert_eq!((page.width, page.height), (3, 2));
        assert!(*updates.lock().unwrap() > 0);
        // Drop joins the worker here, so a broken stop path hangs the
        // suite rather than failing an assert.
    }

    #[test]
    fn decode_failure_is_a_sticky_book_entry() {
        let dir = TempDir::new("worker-err");
        dir.write("bad.png", b"not a png at all");
        dir.write("ok.png", &tiny_png(2, 2, [0, 255, 0, 255]));
        let source = Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let worker = Worker::spawn(source, || {});
        let wanted = Spread { first: 0, second: None };
        worker.retune(wanted);
        let deadline = Instant::now() + Duration::from_secs(20);
        while !matches!(worker.snapshot(&wanted).first, SeatView::Failed(_))
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        match worker.snapshot(&wanted).first {
            SeatView::Failed(msg) => assert!(msg.contains("page 1"), "{msg}"),
            SeatView::Ready(_) => panic!("bad.png decoded as Ready?"),
            SeatView::Pending => panic!("page 0 not decided in time"),
        }
    }

    fn tiny_source(tag: &str, page_count: usize) -> (TempDir, Arc<Source>) {
        let dir = TempDir::new(tag);
        for page in 0..page_count {
            dir.write(&format!("p{page:03}.png"), &tiny_png(1, 1, [page as u8, 0, 0, 255]));
        }
        let source = Arc::new(Source::open(dir.path()).expect("temp source opens"));
        (dir, source)
    }

    /// Parks the worker after each store until the test releases that
    /// store's number. Retunes sent while parked are guaranteed queued
    /// behind it, which pins the decode order for assertions.
    #[derive(Clone)]
    struct Gate(Arc<(Mutex<(u64, u64)>, Condvar)>);

    impl Gate {
        fn new() -> Gate {
            Gate(Arc::new((Mutex::new((0, 0)), Condvar::new())))
        }

        fn hold(&self) {
            let (state, cv) = &*self.0;
            let mut g = state.lock().unwrap();
            g.0 += 1;
            let ticket = g.0;
            // Times out so a failed assertion cannot hang Drop's join.
            let deadline = Instant::now() + Duration::from_secs(5);
            while g.1 < ticket && Instant::now() < deadline {
                let (guard, _) = cv.wait_timeout(g, Duration::from_millis(50)).unwrap();
                g = guard;
            }
        }

        fn release(&self, through: u64) {
            let (state, cv) = &*self.0;
            state.lock().unwrap().1 = through;
            cv.notify_all();
        }
    }

    fn wait_ready(worker: &Worker, pages: &[usize]) {
        let deadline = Instant::now() + Duration::from_secs(20);
        for &page in pages {
            loop {
                match worker.snapshot(&Spread { first: page, second: None }).first {
                    SeatView::Ready(_) => break,
                    SeatView::Failed(msg) => panic!("page {page} failed: {msg}"),
                    SeatView::Pending if Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    SeatView::Pending => panic!("page {page} not decoded in time"),
                }
            }
        }
    }

    #[test]
    fn rapid_retunes_decode_every_trail_page() {
        let (_dir, source) = tiny_source("worker-skim", 50);
        let worker = Worker::spawn(source, || {});
        for page in 0..50 {
            worker.retune(Spread { first: page, second: None });
        }
        wait_ready(&worker, &(0..50).collect::<Vec<_>>());
    }

    #[test]
    fn retuned_spread_decodes_both_pages() {
        let (_dir, source) = tiny_source("worker-double", 6);
        let worker = Worker::spawn(source, || {});
        worker.retune(Spread { first: 3, second: Some(4) });
        wait_ready(&worker, &[3, 4]);
        // Both pages being recorded as last wanted is what keeps a
        // wanted spread in the book, so the wiring is pinned on those
        // records.
        let book = worker.book.lock().expect("decode book lock");
        assert!(book.last_wanted.contains_key(&3));
        assert!(book.last_wanted.contains_key(&4));
    }

    #[test]
    fn trail_pages_past_the_radius_never_decode() {
        let (_dir, source) = tiny_source("worker-bound", 140);
        let gate = Gate::new();
        let parked = gate.clone();
        let worker = Worker::spawn(source, move || parked.hold());
        worker.retune(Spread { first: 0, second: None });
        wait_ready(&worker, &[0]);
        // All 139 retunes queue while the worker is parked on page 0.
        for page in 1..140 {
            worker.retune(Spread { first: page, second: None });
        }
        gate.release(u64::MAX);
        wait_ready(&worker, &(75..140).collect::<Vec<_>>());
        for page in 1..75 {
            assert!(
                matches!(
                    worker.snapshot(&Spread { first: page, second: None }).first,
                    SeatView::Pending
                ),
                "page {page} decoded past the trail radius"
            );
        }
    }

    #[test]
    fn newer_retune_overtakes_far_pending_pages() {
        let (_dir, source) = tiny_source("worker-prio", 70);
        let gate = Gate::new();
        let parked = gate.clone();
        let worker = Worker::spawn(source, move || parked.hold());
        worker.retune(Spread { first: 0, second: None });
        wait_ready(&worker, &[0]);
        worker.retune(Spread { first: 60, second: None });
        gate.release(1);
        wait_ready(&worker, &[60]);
        // The worker is parked on page 60's store, so the old window
        // pages 1 and 2 cannot have been decoded before it.
        for page in [1, 2] {
            assert!(
                matches!(
                    worker.snapshot(&Spread { first: page, second: None }).first,
                    SeatView::Pending
                ),
                "page {page} decoded before the wanted page"
            );
        }
        gate.release(u64::MAX);
    }
}
