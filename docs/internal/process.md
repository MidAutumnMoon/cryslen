# Concurrency: in-process book lifecycle

`cy` is a pure single-process binary (`src/main.rs` -> `src/app/run.rs`); decoded rasters live directly in `Arc<Image>` without IPC serialization or shared-memory copies.

## Lifecycle

- Open (`src/app/run.rs`, `src/book/source.rs`): background `cy-open` thread runs `Source::open_listed(path, reclaimable_bytes)`, scanning entries and checking `/proc/meminfo` against 7z LZMA memory estimates (`src/book/sevenz.rs`) before opening `ThumbHandle` (`src/book/thumbs.rs`).
- Archive (`src/book/source.rs`): directories read on demand; ZIP and 7z share uniform `ArchiveSlots` populated asynchronously on `cy-unpack` only once `ListedBook::start()` runs, with cooperative read cancellation on switch/drop and mimalloc purging after unpack.
- Switch (`src/app/run.rs`): keep the previous `LoadedBook` on screen while `cy-open` runs (crediting its `archive_resident_bytes` as `reclaimable_bytes`), then drop the old book (`DecodePool::stop` + `Source::cancel`) and clear VRAM before calling `listed.start()`.
- Decode (`src/book/loader.rs`): `DecodePool` shares `Arc<Source>` across worker threads and wakes the event loop on completion.

## Playback (`src/book/playback.rs`, `src/app/run.rs`)

- Animated GIF pages shown at a settled spread spawn an in-process `GifPlayer` (`cy-gif`) streaming decoded frames over a 1-frame `crossbeam_channel` bounded channel.
- Dropping a `GifPlayer` disconnects its channel so the producer exits; `Reading` decides settled targets while `LoadedBook` owns `GifPlayer` instances and frame timing.
