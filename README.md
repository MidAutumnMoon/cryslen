# cryslen

`cy`: a minimal Wayland image and comic pager.

- Pages from directories, `.zip`/`.cbz`, and `.7z`/`.cb7` archives
- Manga mode: right-to-left reading (default on) and double-page spreads
- Off-thread decoding with cancellation and prefetch; GPU-side pan/zoom
- AVIF (dav1d, YUV converted in the display shader) and JPEG XL (jxl-rs)

## Build

    nix develop -c cargo build --release

## Keys

`←/→` or `Space` page · `d` double-page · `m` RTL · `f` fullscreen · `r` refit ·
wheel zoom · drag pan · `q` quit. No config file; knobs live in the source.
