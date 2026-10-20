//! Decode worker: one thread, a pending trail around the focus decoded
//! nearest first, byte-budget cache where viewed pages outlive
//! speculative ones. No disk persistence at all.

use std::cmp::Reverse;
use std::collections::{BTreeSet, HashMap};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

use crate::decode::Decoded;
use crate::source::Source;

/// Cache budget: approximate decoded bytes the cache may hold.
const PAGE_CACHE_BUDGET_BYTES: usize = 768 * 1024 * 1024;

enum ToWorker {
    Focus(usize),
    Stop,
}

/// Cache entry: the decode outcome, not just the success case. A failed
/// page is sticky, so waiting, error display, and hold-last-frame are all
/// pure reads of this map, and the failure follows the page across turns.
#[derive(Clone)]
pub enum PageState {
    Ready(Arc<Decoded>),
    Failed(String),
}

impl PageState {
    fn resident_bytes(&self) -> usize {
        match self {
            PageState::Ready(d) => d.resident_bytes(),
            PageState::Failed(e) => e.len(),
        }
    }
}

/// Cache shared between worker (writer) and app (reader). The byte budget is
/// the approximate sum of decoded buffer sizes. Eviction is two-tier:
/// never-viewed pages go first, farthest from the focus first, then viewed
/// pages least-recently-viewed first. The focused page never goes.
struct PageCache {
    map: HashMap<usize, PageState>,
    budget: usize,
    live_bytes: usize,
    viewed: HashMap<usize, u64>,
    stamp: u64,
}

impl PageCache {
    fn new() -> PageCache {
        PageCache {
            map: HashMap::new(),
            budget: PAGE_CACHE_BUDGET_BYTES,
            live_bytes: 0,
            viewed: HashMap::new(),
            stamp: 0,
        }
    }

    fn mark_viewed(&mut self, index: usize) {
        self.stamp += 1;
        self.viewed.insert(index, self.stamp);
    }

    /// Store a page outcome, then evict until under budget. `focused`
    /// never goes, even when it alone exceeds the budget.
    fn insert(&mut self, index: usize, page: PageState, focused: usize) {
        let bytes = page.resident_bytes();
        if let Some(old) = self.map.insert(index, page) {
            self.live_bytes -= old.resident_bytes();
        }
        self.live_bytes += bytes;
        if self.live_bytes > self.budget {
            self.evict(focused);
        }
    }

    fn evict(&mut self, focused: usize) {
        if self.live_bytes <= self.budget {
            return;
        }
        let mut speculative: Vec<usize> = self
            .map
            .keys()
            .copied()
            .filter(|&key| key != focused && !self.viewed.contains_key(&key))
            .collect();
        speculative.sort_unstable_by_key(|&key| Reverse(key.abs_diff(focused)));
        for key in speculative {
            self.remove(key);
            if self.live_bytes <= self.budget {
                return;
            }
        }
        let mut viewed: Vec<usize> =
            self.map.keys().copied().filter(|&key| key != focused).collect();
        viewed.sort_unstable_by_key(|&key| self.viewed[&key]);
        for key in viewed {
            self.remove(key);
            if self.live_bytes <= self.budget {
                return;
            }
        }
        // The focused page stays even over budget, so an oversized page
        // cannot evict itself.
    }

    fn remove(&mut self, key: usize) {
        if let Some(old) = self.map.remove(&key) {
            self.live_bytes -= old.resident_bytes();
        }
    }
}

/// Handle to the worker. `focus` is cheap and never blocks.
pub struct Worker {
    tx: mpsc::Sender<ToWorker>,
    cache: Arc<Mutex<PageCache>>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    /// Spawn the worker for `source`. `notify` fires after every cache change
    /// and should send the winit user event. The worker itself knows nothing
    /// about winit.
    pub fn spawn(source: Arc<Source>, notify: impl Fn() + Send + Sync + 'static) -> Worker {
        let (tx, rx) = mpsc::channel::<ToWorker>();
        let cache = Arc::new(Mutex::new(PageCache::new()));
        let handle = std::thread::Builder::new()
            .name("cy-decode".into())
            .spawn({
                let cache = Arc::clone(&cache);
                move || run(&source, rx, &cache, &notify)
            })
            .expect("spawn decode worker");
        Worker { tx, cache, handle: Some(handle) }
    }

    /// Tell the worker which page the user is on. Every focus counts as a
    /// viewed page and joins the decode trail.
    pub fn focus(&self, index: usize) {
        let _ = self.tx.send(ToWorker::Focus(index));
    }

    /// The decode outcome for one page, if it has been decided. Cloning a
    /// `Ready` page is an `Arc` bump.
    pub fn page(&self, index: usize) -> Option<PageState> {
        self.cache.lock().expect("decode cache lock").map.get(&index).cloned()
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

/// Worker loop: every focus joins the pending trail with its neighbors, and
/// pages past `TRAIL_RADIUS` of the focus are dropped. One decode at a
/// time, the undecided pending page nearest the focus first, ahead of the
/// reader before behind. The channel is observed between decodes, so a
/// newer focus overtakes the backlog. One decode in flight always runs
/// to completion.
fn run(
    source: &Source,
    rx: mpsc::Receiver<ToWorker>,
    cache: &Mutex<PageCache>,
    notify: &dyn Fn(),
) {
    let len = source.len();
    let mut focus = match rx.recv() {
        Ok(ToWorker::Focus(index)) => index,
        Ok(ToWorker::Stop) | Err(_) => return,
    };
    let mut arrivals: Vec<usize> = vec![focus];
    let mut pending: BTreeSet<usize> = BTreeSet::new();
    loop {
        while let Ok(message) = rx.try_recv() {
            let ToWorker::Focus(index) = message else { return };
            arrivals.push(index);
        }
        if let Some(&newest) = arrivals.last() {
            focus = newest;
            {
                let mut pages = cache.lock().expect("decode cache lock");
                for &arrived in &arrivals {
                    pages.mark_viewed(arrived);
                }
            }
            pending.extend(arrivals.iter().copied());
            pending.extend(neighbors(focus, len));
            retain_trail(&mut pending, focus);
            arrivals.clear();
        }
        let pick = {
            let pages = cache.lock().expect("decode cache lock");
            take_next(&mut pending, &pages, focus)
        };
        let Some(index) = pick else {
            let arrived = match rx.recv() {
                Ok(ToWorker::Focus(index)) => index,
                Ok(ToWorker::Stop) | Err(_) => return,
            };
            arrivals.push(arrived);
            continue;
        };
        let outcome = decode_page(source, index);
        {
            let mut pages = cache.lock().expect("decode cache lock");
            pages.insert(
                index,
                match outcome {
                    Ok(page) => PageState::Ready(page),
                    Err(message) => PageState::Failed(message),
                },
                focus,
            );
        }
        pending.remove(&index);
        notify();
    }
}

/// Decode priority key: distance to the focus first, and at equal distance
/// a page ahead of the reader before one behind it.
fn next_pick_key(page: usize, focus: usize) -> (usize, bool) {
    (page.abs_diff(focus), page < focus)
}

/// The pending page to decode next, or `None` when the trail is fully
/// decided. Pages that already hold an outcome leave the pending set,
/// so a Ready or Failed page is never decoded twice.
fn take_next(pending: &mut BTreeSet<usize>, cache: &PageCache, focus: usize) -> Option<usize> {
    pending.retain(|page| !cache.map.contains_key(page));
    pending.iter().copied().min_by_key(|&page| next_pick_key(page, focus))
}

/// One page through the panic boundary: a panicking decoder must not kill
/// the only worker thread. The error string is the status-line message.
fn decode_page(source: &Source, index: usize) -> Result<Arc<Decoded>, String> {
    let name = source.name(index).to_owned();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> eros::Result<Decoded> {
            let bytes = source.bytes(index)?;
            Ok(crate::decode::decode(&bytes, &name)?)
        },
    ));
    match result {
        Ok(Ok(page)) => Ok(Arc::new(page)),
        Ok(Err(e)) => Err(format!("page {}: {e}", index + 1)),
        Err(panic) => Err(format!("page {}: {}", index + 1, panic_message(panic))),
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

/// How many pages on each side of the focused page to prefetch. The app
/// focuses a spread's first page, and a double-mode turn steps two pages,
/// so the far page of the next spread sits at distance 3. Single-page mode
/// over-prefetches by the same margin, which the two-tier cache reclaims
/// first as speculation.
const PREFETCH_RADIUS: i64 = 3;

/// Decode order around the focused page: the page itself, then neighbours
/// out to [`PREFETCH_RADIUS`], nearest first.
fn neighbors(current: usize, len: usize) -> Vec<usize> {
    let mut pages: Vec<usize> = (0..=PREFETCH_RADIUS)
        .flat_map(|distance| {
            [current as i64 - distance, current as i64 + distance]
        })
        .filter(|&page| (0..len as i64).contains(&page))
        .map(|page| page as usize)
        .collect();
    pages.sort_unstable_by_key(|&page| (page.abs_diff(current), page));
    pages.dedup();
    pages
}

/// How many pages on each side of the focus may stay pending. A held key
/// sends focuses faster than pages decode, so pages past this radius are
/// dropped rather than queued for a reader that has already moved on.
const TRAIL_RADIUS: i64 = 64;

fn retain_trail(pending: &mut BTreeSet<usize>, focus: usize) {
    pending.retain(|&page| (page as i64 - focus as i64).abs() <= TRAIL_RADIUS);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{tiny_png, TempDir};

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
        for len in [2usize, 9, 13] {
            let spreads = 1 + (len - 1).div_ceil(2);
            let mut focus = 0;
            for turn in 0..spreads {
                let window = neighbors(focus, len);
                let spread = crate::view::spread_for(focus, true, len);
                for page in [Some(spread.first), spread.second].into_iter().flatten() {
                    assert!(
                        window.contains(&page),
                        "len {len}: spread page {page} outside focus {focus}"
                    );
                }
                // The wrap turn jumps across the book, which prefetch
                // cannot cover.
                if turn + 1 < spreads {
                    let next = crate::view::advance(focus, true, true, len);
                    let next_spread = crate::view::spread_for(next, true, len);
                    for page in [Some(next_spread.first), next_spread.second]
                        .into_iter()
                        .flatten()
                    {
                        assert!(
                            window.contains(&page),
                            "len {len}: next spread page {page} outside focus {focus}"
                        );
                    }
                    focus = next;
                }
            }
        }
    }

    #[test]
    fn trail_retention_drops_pages_past_the_radius() {
        let mut pending: BTreeSet<usize> = (0..=200).collect();
        retain_trail(&mut pending, 100);
        let kept = (100 - TRAIL_RADIUS as usize)..=(100 + TRAIL_RADIUS as usize);
        assert_eq!(pending, kept.collect());
    }

    #[test]
    fn next_pick_is_nearest_then_ahead() {
        let mut cache = PageCache::new();
        cache.insert(10, ready_page(10), 10);
        let mut pending: BTreeSet<usize> = (8..=12).collect();
        assert_eq!(take_next(&mut pending, &cache, 10), Some(11));
        assert!(!pending.contains(&10));
        let mut order = Vec::new();
        while let Some(page) = take_next(&mut pending, &cache, 10) {
            order.push(page);
            pending.remove(&page);
        }
        assert_eq!(order, vec![11, 9, 12, 8]);
    }

    fn ready_page(label: u8) -> PageState {
        PageState::Ready(Arc::new(
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
    fn cache_evicts_speculative_pages_far_first() {
        let mut cache = PageCache::new();
        cache.budget = 8;
        cache.insert(1, ready_page(1), 5);
        cache.insert(3, ready_page(3), 5);
        assert!(cache.map.contains_key(&3));
        assert!(!cache.map.contains_key(&1));
        assert!(cache.live_bytes <= cache.budget);
    }

    #[test]
    fn viewed_page_outlives_speculative_page() {
        let mut cache = PageCache::new();
        cache.budget = 8;
        cache.mark_viewed(1);
        cache.insert(1, ready_page(1), 5);
        // Page 3 is nearer the focus but never viewed, so it goes first.
        cache.insert(3, ready_page(3), 5);
        assert!(cache.map.contains_key(&1));
        assert!(!cache.map.contains_key(&3));
    }

    #[test]
    fn viewed_pages_evict_least_recent_first() {
        let mut cache = PageCache::new();
        cache.budget = 8;
        cache.mark_viewed(1);
        cache.mark_viewed(2);
        cache.insert(1, ready_page(1), 5);
        cache.insert(2, ready_page(2), 5);
        assert!(cache.map.contains_key(&2));
        assert!(!cache.map.contains_key(&1));
    }

    #[test]
    fn focused_page_survives_its_own_oversize() {
        let mut cache = PageCache::new();
        cache.budget = 10;
        let big = PageState::Ready(Arc::new(
            crate::decode::Decoded::rgba(
                10,
                10,
                crate::decode::Format::Png,
                vec![0; 400].into_boxed_slice(),
            )
            .unwrap(),
        ));
        cache.insert(3, big, 3);
        assert!(cache.map.contains_key(&3));
    }

    #[test]
    fn focus_decodes_and_caches() {
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
        worker.focus(1);

        let deadline = Instant::now() + Duration::from_secs(20);
        while !matches!(worker.page(1), Some(PageState::Ready(_))) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        let PageState::Ready(page) = worker.page(1).expect("focused page decoded in time") else {
            panic!("not Ready");
        };
        assert_eq!((page.width, page.height), (3, 2));
        assert!(*updates.lock().unwrap() > 0);
        // Drop joins the worker here, so a broken stop path hangs the
        // suite rather than failing an assert.
    }

    #[test]
    fn decode_failure_is_a_sticky_cache_entry() {
        let dir = TempDir::new("worker-err");
        dir.write("bad.png", b"not a png at all");
        dir.write("ok.png", &tiny_png(2, 2, [0, 255, 0, 255]));
        let source = Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let worker = Worker::spawn(source, || {});
        worker.focus(0);
        let deadline = Instant::now() + Duration::from_secs(20);
        while !matches!(worker.page(0), Some(PageState::Failed(_))) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        match worker.page(0) {
            Some(PageState::Failed(msg)) => assert!(msg.contains("page 1"), "{msg}"),
            Some(PageState::Ready(_)) => panic!("bad.png decoded as Ready?"),
            None => panic!("page 0 not decided in time"),
        }
    }

    fn tiny_source(tag: &str, len: usize) -> (TempDir, Arc<Source>) {
        let dir = TempDir::new(tag);
        for page in 0..len {
            dir.write(&format!("p{page:03}.png"), &tiny_png(1, 1, [page as u8, 0, 0, 255]));
        }
        let source = Arc::new(Source::open(dir.path()).expect("temp source opens"));
        (dir, source)
    }

    /// Parks the worker after each insert until the test releases that
    /// insert's number. Focuses sent while parked are guaranteed queued
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
            while !matches!(worker.page(page), Some(PageState::Ready(_))) {
                assert!(Instant::now() < deadline, "page {page} not decoded in time");
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    #[test]
    fn rapid_focuses_decode_every_trail_page() {
        let (_dir, source) = tiny_source("worker-skim", 50);
        let worker = Worker::spawn(source, || {});
        for page in 0..50 {
            worker.focus(page);
        }
        wait_ready(&worker, &(0..50).collect::<Vec<_>>());
    }

    #[test]
    fn trail_pages_past_the_radius_never_decode() {
        let (_dir, source) = tiny_source("worker-bound", 140);
        let gate = Gate::new();
        let parked = gate.clone();
        let worker = Worker::spawn(source, move || parked.hold());
        worker.focus(0);
        wait_ready(&worker, &[0]);
        // All 140 focuses queue while the worker is parked on page 0.
        for page in 1..140 {
            worker.focus(page);
        }
        gate.release(u64::MAX);
        wait_ready(&worker, &(75..140).collect::<Vec<_>>());
        for page in 1..75 {
            assert!(
                worker.page(page).is_none(),
                "page {page} decoded past the trail radius"
            );
        }
    }

    #[test]
    fn newer_focus_overtakes_far_pending_pages() {
        let (_dir, source) = tiny_source("worker-prio", 70);
        let gate = Gate::new();
        let parked = gate.clone();
        let worker = Worker::spawn(source, move || parked.hold());
        worker.focus(0);
        wait_ready(&worker, &[0]);
        worker.focus(60);
        gate.release(1);
        wait_ready(&worker, &[60]);
        // The worker is parked on page 60's insert, so the old window
        // pages 1 and 2 cannot have been decoded before it.
        for page in [1, 2] {
            assert!(
                worker.page(page).is_none(),
                "page {page} decoded before the focused page"
            );
        }
        gate.release(u64::MAX);
    }
}
