# cryslen

<div align="center">
  <img src="assets/icon-raw.avif" alt="cryslen" width="350">

### `cy`: a minimal Wayland image and comic pager.
</div>

- Pages from directories, `.zip`/`.cbz`, and `.7z`/`.cb7` archives
- Opens JPEG, PNG, GIF, WebP, BMP, TIFF, AVIF, and JPEG XL files.
  Animated files show their first frame.
- View modes: `normal` single page, `manga` and `comic` double page
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

`←/→` or `Space` page · `n`/`m`/`c` normal, manga, comic mode · `f` fullscreen ·
`r` refit · wheel zoom · drag pan · `q` quit. No config file. Knobs live in the
source.
