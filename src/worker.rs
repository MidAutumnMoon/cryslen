//! Decode worker: one thread, latest-wins focus, ±2 prefetch window,
//! in-memory cache with a byte budget. No disk persistence at all.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

use crate::decode::Decoded;
use crate::page::Source;

/// Cache budget: approximate decoded bytes the cache may hold.
const BUDGET_BYTES: usize = 512 * 1024 * 1024;

/// Wakes the loop so it can observe `stop` and exit (sent by `Drop`).
const SENTINEL: usize = usize::MAX;
/// Cache entry: the decode outcome, not just the success case. A failed
/// page is sticky, so waiting, error display, and hold-last-frame are all
/// pure reads of this map, and the failure follows the page across turns.
pub enum Page {
    Ready(Arc<Decoded>),
    Failed(String),
}

impl Clone for Page {
    fn clone(&self) -> Self {
        match self {
            Page::Ready(d) => Page::Ready(Arc::clone(d)),
            Page::Failed(e) => Page::Failed(e.clone()),
        }
    }
}

impl Page {
    fn resident_bytes(&self) -> usize {
        match self {
            Page::Ready(d) => d.resident_bytes(),
            Page::Failed(e) => e.len(),
        }
    }
}

/// Cache shared between worker (writer) and app (reader). The byte budget is
/// the approximate sum of decoded buffer sizes. Eviction is simplest-first:
/// drop arbitrary non-current entries. There is no LRU machinery.
struct PageCache {
    map: HashMap<usize, Page>,
    budget: usize,
    live_bytes: usize,
}

impl PageCache {
    fn new() -> PageCache {
        PageCache { map: HashMap::new(), budget: BUDGET_BYTES, live_bytes: 0 }
    }

    /// Store a page outcome, then evict until under budget. `focused` never
    /// goes. Entries outside `keep` go first, then the neighbours.
    fn insert(&mut self, index: usize, page: Page, focused: usize, keep: &[usize]) {
        let bytes = page.resident_bytes();
        if let Some(old) = self.map.insert(index, page) {
            self.live_bytes -= old.resident_bytes();
        }
        self.live_bytes += bytes;
        if self.live_bytes > self.budget {
            self.evict(focused, keep);
        }
    }

    fn evict(&mut self, focused: usize, keep: &[usize]) {
        if self.live_bytes <= self.budget {
            return;
        }
        let outside: Vec<usize> = self
            .map
            .keys()
            .copied()
            .filter(|k| *k != focused && !keep.contains(k))
            .collect();
        for key in outside {
            self.remove(key);
            if self.live_bytes <= self.budget {
                return;
            }
        }
        // Everything cached is now wanted. Drop the neighbours, never the
        // focused page, so a budget-exceeding page cannot evict itself.
        let neighbours: Vec<usize> =
            keep.iter().copied().filter(|&page| page != focused).collect();
        for key in neighbours {
            self.remove(key);
            if self.live_bytes <= self.budget {
                return;
            }
        }
    }

    fn remove(&mut self, key: usize) {
        if let Some(old) = self.map.remove(&key) {
            self.live_bytes -= old.resident_bytes();
        }
    }
}

/// Handle to the worker. `focus` is cheap and never blocks.
pub struct Worker {
    tx: mpsc::Sender<usize>,
    cache: Arc<Mutex<PageCache>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    /// Spawn the worker for `source`. `notify` fires after every cache change
    /// and should send the winit user event. The worker itself knows nothing
    /// about winit.
    pub fn spawn(source: Arc<Source>, notify: impl Fn() + Send + Sync + 'static) -> Worker {
        let (tx, rx) = mpsc::channel::<usize>();
        let cache = Arc::new(Mutex::new(PageCache::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let handle = std::thread::Builder::new()
            .name("cy-decode".into())
            .spawn({
                let cache = Arc::clone(&cache);
                let stop = Arc::clone(&stop);
                move || run(&source, rx, &stop, &cache, &notify)
            })
            .expect("spawn decode worker");
        Worker { tx, cache, stop, handle: Some(handle) }
    }

    /// Tell the worker which page the user is on. Sending is latest-wins:
    /// the worker drains pending focuses and serves only the newest.
    pub fn focus(&self, index: usize) {
        let _ = self.tx.send(index);
    }

    /// The decode outcome for one page, if it has been decided. Cloning a
    /// `Ready` page is an `Arc` bump.
    pub fn page(&self, index: usize) -> Option<Page> {
        self.cache.lock().expect("decode cache lock").map.get(&index).cloned()
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // A final dummy focus wakes the loop so it observes `stop`.
        let _ = self.tx.send(usize::MAX);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Worker loop: latest-wins focus, then decode the ±2 window around it,
/// nearest first. A focus update arriving mid-batch aborts the rest of the
/// batch so no FIFO backlog forms. One decode in flight always runs to
/// completion.
fn run(
    source: &Source,
    rx: mpsc::Receiver<usize>,
    stop: &AtomicBool,
    cache: &Mutex<PageCache>,
    notify: &dyn Fn(),
) {
    let mut current = match rx.recv() {
        Ok(index) => index,
        Err(_) => return,
    };
    while current != SENTINEL && !stop.load(Ordering::Relaxed) {
        let decode_order = window(current, source.len());
        let mut newer: Option<usize> = None;
        for &index in &decode_order {
            while let Ok(arrived) = rx.try_recv() {
                if stop.load(Ordering::Relaxed) || arrived == SENTINEL {
                    return;
                }
                newer = Some(arrived);
            }
            if newer.is_some() {
                break;
            }
            let done = cache.lock().expect("decode cache lock").map.contains_key(&index);
            if done {
                continue;
            }
            let outcome = decode_page(source, index);
            {
                let mut pages = cache.lock().expect("decode cache lock");
                pages.insert(
                    index,
                    match outcome {
                        Ok(page) => Page::Ready(page),
                        Err(message) => Page::Failed(message),
                    },
                    current,
                    &decode_order,
                );
            }
            // The cache carries the outcome, decoder detail included, for the
            // status line.
            notify();
        }
        current = match newer.or_else(|| rx.recv().ok()) {
            Some(index) => index,
            None => return,
        };
    }
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

/// How many pages on each side of the focused page to prefetch. ±2 covers
/// both modes' windows: single-page needs ±1, double-page spreads shift by
/// two pages per turn.
const PREFETCH_RADIUS: i64 = 2;

/// Decode order around the focused page: the page itself, then neighbours
/// out to [`PREFETCH_RADIUS`], nearest first (the current page included).
pub fn window(current: usize, len: usize) -> Vec<usize> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::testkit::{tiny_png, TempDir};

    use std::time::{Duration, Instant};

    #[test]
    fn window_is_nearest_first() {
        assert_eq!(window(5, 10), vec![5, 4, 6, 3, 7]);
        assert_eq!(window(0, 4), vec![0, 1, 2]);
        assert_eq!(window(9, 10), vec![9, 8, 7]);
    }

    #[test]
    fn cache_evicts_far_pages_first() {
        let mut cache = PageCache::new();
        cache.budget = 8;
        let page = |label: u8| {
            Page::Ready(Arc::new(
                crate::decode::Decoded::rgba(
                    2,
                    1,
                    crate::decode::Format::Png,
                    vec![label; 8].into_boxed_slice(),
                )
                .unwrap(),
            ))
        };
        // Live bytes 8. Inserting another page (total 16 > 8) must evict the
        // page outside the keep set, never the focused page.
        cache.insert(1, page(1), 5, &[5, 4]);
        cache.insert(5, page(5), 5, &[5, 4]);
        assert!(cache.map.contains_key(&5));
        assert!(!cache.map.contains_key(&1));
        assert!(cache.live_bytes <= cache.budget);
    }

    #[test]
    fn focused_page_survives_its_own_oversize() {
        let mut cache = PageCache::new();
        cache.budget = 10;
        let big = Page::Ready(Arc::new(
            crate::decode::Decoded::rgba(
                10,
                10,
                crate::decode::Format::Png,
                vec![0; 400].into_boxed_slice(),
            )
            .unwrap(),
        ));
        cache.insert(3, big, 3, &[3]);
        // The oversized focused page stays even though it alone busts budget.
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
        while !matches!(worker.page(1), Some(Page::Ready(_))) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        let Page::Ready(page) = worker.page(1).expect("focused page decoded in time") else {
            panic!("not Ready");
        };
        assert_eq!((page.width, page.height), (3, 2));
        assert!(*updates.lock().unwrap() > 0);
        // Drop joins the thread. A hung sentinel/stop path would hang the
        // test suite here (no GL or event loop involved).
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
        while !matches!(worker.page(0), Some(Page::Failed(_))) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        match worker.page(0) {
            Some(Page::Failed(msg)) => assert!(msg.contains("page 1"), "{msg}"),
            Some(Page::Ready(_)) => panic!("bad.png decoded as Ready?"),
            None => panic!("page 0 not decided in time"),
        }
    }
}
