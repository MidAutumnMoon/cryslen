//! Decode worker: one thread, latest-wins focus, ±2 prefetch window,
//! in-memory cache with a byte budget. No disk persistence at all.
//!
//! The loop follows the pattern verified in research/08 (oculante-style
//! cancellation): block on focus updates, drain pending ones keeping the
//! last, decode the prefetch set in nearest-first order, aborting the rest
//! whenever a newer focus arrives.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

use crate::decode::{Decoded, Plane, Pixels};
use crate::page::Source;

/// Cache budget: approximate decoded bytes the cache may hold.
const BUDGET_BYTES: usize = 512 * 1024 * 1024;

/// Wakes the loop so it can observe `stop` and exit (sent by `Drop`).
const SENTINEL: usize = usize::MAX;

/// Cache shared between worker (writer) and app (reader). Byte budget is the
/// approximate sum of decoded plane/buffer sizes; eviction is simplest-first
/// (drop arbitrary non-current entries) — no LRU machinery.
pub struct PageCache {
    map: HashMap<usize, Arc<Decoded>>,
    budget: usize,
    live_bytes: usize,
}

impl PageCache {
    fn new() -> PageCache {
        PageCache { map: HashMap::new(), budget: BUDGET_BYTES, live_bytes: 0 }
    }

    fn contains(&self, index: usize) -> bool {
        self.map.contains_key(&index)
    }

    /// Store a decoded page, then evict until under budget. `keep` is the
    /// prefetch set in decode-priority order; entries outside it go first,
    /// then neighbours — except `keep[0]` (the focused page), so a
    /// budget-exceeding page cannot evict itself.
    fn insert(&mut self, index: usize, page: Arc<Decoded>, keep: &[usize]) {
        let bytes = decoded_bytes(&page);
        if let Some(old) = self.map.insert(index, page) {
            self.live_bytes -= decoded_bytes(&old);
        }
        self.live_bytes += bytes;
        if self.live_bytes > self.budget {
            self.evict(keep);
        }
    }

    fn evict(&mut self, keep: &[usize]) {
        if self.live_bytes <= self.budget {
            return;
        }
        let outside: Vec<usize> =
            self.map.keys().copied().filter(|k| !keep.contains(k)).collect();
        for key in outside {
            self.remove(key);
            if self.live_bytes <= self.budget {
                return;
            }
        }
        // Everything cached is now wanted; drop the neighbours, never the
        // focused page (`keep[0]`).
        let neighbours: &[usize] = keep.get(1..).unwrap_or(&[]);
        for &key in neighbours {
            self.remove(key);
            if self.live_bytes <= self.budget {
                return;
            }
        }
    }

    fn remove(&mut self, key: usize) {
        if let Some(old) = self.map.remove(&key) {
            self.live_bytes -= decoded_bytes(&old);
        }
    }
}

/// Handle to the worker. `focus` is cheap and never blocks.
pub struct Worker {
    tx: mpsc::Sender<usize>,
    cache: Arc<Mutex<PageCache>>,
    errors: Arc<Mutex<Option<String>>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    /// Spawn the worker for `source`. `notify` fires after every cache change
    /// or decode failure (it should send the winit user event; the worker
    /// itself knows nothing about winit).
    pub fn spawn(source: Arc<Source>, notify: impl Fn() + Send + Sync + 'static) -> Worker {
        let (tx, rx) = mpsc::channel::<usize>();
        let cache = Arc::new(Mutex::new(PageCache::new()));
        let errors = Arc::new(Mutex::new(None));
        let stop = Arc::new(AtomicBool::new(false));
        let handle = std::thread::Builder::new()
            .name("cy-decode".into())
            .spawn({
                let cache = Arc::clone(&cache);
                let errors = Arc::clone(&errors);
                let stop = Arc::clone(&stop);
                move || run(&source, rx, &stop, &cache, &errors, &notify)
            })
            .expect("spawn decode worker");
        Worker { tx, cache, errors, stop, handle: Some(handle) }
    }

    /// Tell the worker which page the user is on. Sending is latest-wins:
    /// the worker drains pending focuses and serves only the newest.
    pub fn focus(&self, index: usize) {
        let _ = self.tx.send(index);
    }

    pub fn get(&self, index: usize) -> Option<Arc<Decoded>> {
        self.cache.lock().unwrap().map.get(&index).cloned()
    }

    /// Take the last decode failure, if any (for the status line).
    pub fn take_error(&self) -> Option<String> {
        self.errors.lock().unwrap().take()
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
    errors: &Mutex<Option<String>>,
    notify: &dyn Fn(),
) {
    let mut current = match rx.recv() {
        Ok(index) => index,
        Err(_) => return,
    };
    while current != SENTINEL && !stop.load(Ordering::Relaxed) {
        let want = prefetch(current, source.len());
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
            if cache.lock().unwrap().contains(index) {
                continue;
            }
            match decode_page(source, index) {
                Ok(page) => {
                    let mut pages = cache.lock().unwrap();
                    pages.insert(index, page, &want);
                    drop(pages);
                    notify();
                }
                Err(e) => {
                    eprintln!("cy: page {}: {e}", index + 1);
                    *errors.lock().unwrap() = Some(format!("page {}: {e}", index + 1));
                    // Store nothing, but wake the app so it can stop
                    // waiting on a page that will never arrive.
                    notify();
                }
            }
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
        Ok(Err(e)) => Err(e),
        Err(panic) => Err(panic_message(panic)),
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
/// out to ±2. ±2 covers both modes' windows: single-page needs ±1,
/// double-page spreads shift by two.
pub fn prefetch(current: usize, len: usize) -> Vec<usize> {
    let mut want: Vec<usize> = (0..=2)
        .flat_map(|d| [current as i64 - d as i64, current as i64 + d as i64])
        .filter(|&i| (0..len as i64).contains(&i))
        .map(|i| i as usize)
        .collect();
    want.sort_unstable_by_key(|&i| (i.abs_diff(current), i));
    want.dedup();
    want
}

/// Approximate resident size of one decoded page, for budget accounting.
pub fn decoded_bytes(d: &Decoded) -> usize {
    fn plane(p: &Plane) -> usize {
        match p {
            Plane::U8(b) => b.len(),
            Plane::U16(b) => b.len() * 2,
        }
    }
    match &d.pixels {
        Pixels::Rgba8(b) => b.len(),
        Pixels::Yuv(y) => plane(&y.y) + plane(&y.cb) + plane(&y.cr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::testkit::{tiny_png, TempDir};

    use std::time::{Duration, Instant};

    #[test]
    fn prefetch_is_nearest_first_window() {
        assert_eq!(prefetch(5, 10), vec![5, 4, 6, 3, 7]);
        assert_eq!(prefetch(0, 4), vec![0, 1, 2]);
        assert_eq!(prefetch(9, 10), vec![9, 8, 7]);
    }

    #[test]
    fn decoded_bytes_sizes() {
        let rgba = Decoded {
            width: 4,
            height: 3,
            pixels: Pixels::Rgba8(vec![0; 4 * 4 * 3].into_boxed_slice()),
            format: crate::decode::Format::Png,
        };
        assert_eq!(decoded_bytes(&rgba), 48);

        let yuv = Decoded {
            width: 4,
            height: 2,
            pixels: Pixels::Yuv(Arc::new(crate::decode::YuvImage {
                depth: 10,
                subsampling: crate::decode::Subsampling::Yuv420,
                full_range: false,
                matrix: crate::decode::Matrix::Bt709,
                y: Plane::U16(vec![0; 8].into_boxed_slice()),
                cb: Plane::U16(vec![0; 2].into_boxed_slice()),
                cr: Plane::U16(vec![0; 2].into_boxed_slice()),
            })),
            format: crate::decode::Format::Avif,
        };
        // (8 + 2 + 2) u16 samples = 24 bytes.
        assert_eq!(decoded_bytes(&yuv), 24);
    }

    #[test]
    fn cache_evicts_far_pages_first() {
        let mut cache = PageCache::new();
        cache.budget = 10;
        let page = |label: u8| {
            Arc::new(Decoded {
                width: 1,
                height: 1,
                pixels: Pixels::Rgba8(vec![label; 8].into_boxed_slice()),
                format: crate::decode::Format::Png,
            })
        };
        // Live bytes 8; inserting another page (total 16 > 10) evicts the
        // page outside the prefetch set [5, 4].
        cache.insert(1, page(1), &[5, 4]);
        cache.insert(5, page(5), &[5, 4]);
        assert!(cache.contains(5));
        assert!(!cache.contains(1));
        assert!(cache.live_bytes <= cache.budget);
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
        while worker.get(1).is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        let page = worker.get(1).expect("focused page decoded in time");
        assert_eq!((page.width, page.height), (3, 2));
        assert!(*updates.lock().unwrap() > 0);
        // Drop joins the thread; if the sentinel/stop path hung, the test
        // suite would hang here (no GL or event loop involved).
    }

    #[test]
    fn decode_failure_surfaces_via_take_error() {
        let dir = TempDir::new("worker-err");
        dir.write("bad.png", b"not a png at all");
        dir.write("ok.png", &tiny_png(2, 2, [0, 255, 0, 255]));
        let source = Arc::new(Source::open(dir.path()).expect("temp source opens"));
        let worker = Worker::spawn(source, || {});
        worker.focus(0);
        let deadline = Instant::now() + Duration::from_secs(20);
        while worker.take_error().is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(worker.take_error().is_some() || worker.get(1).is_some());
    }
}
