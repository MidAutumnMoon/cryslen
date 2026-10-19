//! Decode worker: one thread, latest-wins focus, ±2 prefetch window,
//! in-memory cache with a byte budget. No disk persistence at all.
//!
//! The loop follows the pattern verified in research/08 (oculante-style
//! cancellation): block on focus updates, drain pending ones keeping the
//! last, decode the window nearest-first, aborting the rest whenever a newer
//! focus arrives.

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

/// Cache entry: the decode outcome, not just the success case. A failed page
/// is sticky — waiting, error display, and hold-last-frame are all pure reads
/// of the same map, and the failure follows the page across turns.
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

/// Cache shared between worker (writer) and app (reader). Byte budget is the
/// approximate sum of decoded buffer sizes; eviction is simplest-first (drop
/// arbitrary non-current entries) — no LRU machinery.
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
    /// goes; entries outside `keep` go first, then the neighbours.
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
        // Everything cached is now wanted; drop the neighbours, never the
        // focused page (so a budget-exceeding page cannot evict itself).
        let neighbours: Vec<usize> =
            keep.iter().copied().filter(|&k| k != focused).collect();
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
    /// (it should send the winit user event; the worker itself knows nothing
    /// about winit).
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
        self.cache.lock().unwrap().map.get(&index).cloned()
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
/// batch (no FIFO backlog); one decode in flight always runs to completion.
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
        let want = window(current, source.len());
        let mut newer: Option<usize> = None;
        for &index in &want {
            while let Ok(arrived) = rx.try_recv() {
                if stop.load(Ordering::Relaxed) || arrived == SENTINEL {
                    return;
                }
                newer = Some(arrived);
            }
            if newer.is_some() {
                break;
            }
            let done = cache.lock().unwrap().map.contains_key(&index);
            if done {
                continue;
            }
            let outcome = decode_page(source, index);
            {
                let mut pages = cache.lock().unwrap();
                pages.insert(
                    index,
                    match outcome {
                        Ok(page) => Page::Ready(page),
                        Err(e) => Page::Failed(e),
                    },
                    current,
                    &want,
                );
            }
            // Detail went to stderr already; the cache now carries the
            // outcome for the status line.
            notify();
        }
        current = match newer.or_else(|| rx.recv().ok()) {
            Some(index) => index,
            None => return,
        };
    }
}

/// One page through the panic boundary (research/08: a panicking decoder must
/// not kill the only worker thread).
fn decode_page(source: &Source, index: usize) -> Result<Arc<Decoded>, String> {
    let name = source.name(index).to_owned();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let bytes = source.bytes(index).map_err(|e| e.to_string())?;
        crate::decode::decode(&bytes, &name).map_err(|e| e.to_string())
    }));
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

/// Decode order around the focused page: the page itself, then neighbours
/// out to ±2 (covers both modes' windows: single-page needs ±1,
/// double-page spreads shift by two). Includes the current page — `keep[0]`
/// is not a convention here, `run` passes `current` explicitly.
pub fn window(current: usize, len: usize) -> Vec<usize> {
    let mut want: Vec<usize> = (0..=2)
        .flat_map(|d| [current as i64 - d as i64, current as i64 + d as i64])
        .filter(|&i| (0..len as i64).contains(&i))
        .map(|i| i as usize)
        .collect();
    want.sort_unstable_by_key(|&i| (i.abs_diff(current), i));
    want.dedup();
    want
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
        // Live bytes 8; inserting another page (total 16 > 8) evicts the
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
        // Drop joins the thread; if the sentinel/stop path hung, the test
        // suite would hang here (no GL or event loop involved).
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
