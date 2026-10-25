# cryslen

<div align="center">
  <img src="assets/icon-raw.avif" alt="cryslen" width="350">

### `cy`: a minimal Wayland image viewer with manga mode.
</div>

- Pages from directories, `.zip`/`.cbz`, and `.7z`/`.cb7` archives
- Opens JPEG, PNG, GIF, WebP, BMP, TIFF, AVIF, and JPEG XL files.
  Animated files show their first frame.
- View modes: `Normal` single page, `Manga` and `Comic` double page
  (right to left and left to right)
- Double-page modes show the cover alone, then page pairs
- Background decoding and prefetch keep page turns fast
- Smooth pan and zoom
- Hold an arrow key to flip through the book. Paging wraps at the ends.

## Usage

    cy <dir | archive | image-file>

Pass a single image and cy opens its folder at that page.

## Build

    nix develop -c cargo build --release

## Keys

`←/→` page by screen side · `Space` page by reading order · `n`/`m`/`c` Normal, Manga,
Comic mode · `f` fullscreen · `r` refit · wheel zoom · drag pan · `q` quit. No config
file. Knobs live in the source.
