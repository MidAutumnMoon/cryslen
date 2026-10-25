# cryslen

<div align="center">
  <img src="assets/icon-raw.avif" alt="cryslen icon" width="350">

### `cy`: a minimal Wayland image viewer with manga mode.
</div>

I built it from scratch because I want:

- Minimal, config-less, stateless viewer. This thing draws raw OpenGL. Knobs and shortcuts are all baked in. They fit my habit but maybe not yours.
- Support enough file formats. Besides PNG, JPEG, I use AVIF and JXL extensively. Animated formats are not yet supported (soon).
- Manga mode: right to left, double page. Of course it also has left to right mode like a normal image viewer.
- Read manga directly from archives: 7z, zip, cbz.
- Heavily utilize background decoding and prefetch to make it "feel snappy". As a result, hold arrow keys can flip pages extremely smooth. The bottleneck is always the decoder and CPU power.

## Usage

    cy <dir/ | archive.{7z,cbz,zip} | image-file.{png,jpg,avif,jxl,...}>

Proper CLI handling later.

## Build

    nix develop -c cargo build --release

flake.nix package doesn't work for now. Later!

## Screenshots (2026-10)

|![cryslen full window](assets/screenshot-2026-10/full.avif)|
|-|
|Full window with transparent background. Blur effect is by niri.|

|![cryslen left side](assets/screenshot-2026-10/left-side.avif)|
|-|
|From left to right: <br/><br/> - `current page`/`total pages`. <br/> - `immediate parent`/`filename`. <br/> - image dimension <br/> - zoom level|

|![cryslen right side](assets/screenshot-2026-10/right-side.avif)|
|-|
|Mode indicators|

## Shortcuts

- `←/→` arrow keys switch pages
- `space` page by reading order
- `n`/`m`/`c` normal, manga, comic mode (first letter of each mode)
- `f` toggle fullscreen
- `mouse wheel` zoom
- `r` reset zoom level
- `drag` move/pan
- `q` or `esc` quit

## License and disclosure

GPL-3.0.

LLM-assisted. Not quite vibe coding but not quite handmade either.
