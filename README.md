# cryslen

<div align="center">
  <img src="assets/icon-raw.avif" alt="cryslen icon" width="350">

### `cy`: a minimal Wayland image viewer with manga mode.
</div>

I built it from scratch because I want:

- Minimal, config-less, stateless viewer. This thing draws raw OpenGL. Knobs and shortcuts are all baked in. They fit my habit but maybe not yours.
- Support enough file formats. I use JXL and AVIF heavily, so support of these two is very good.
- Manga mode: right to left, double page. Of course it also has left to right mode like a normal image viewer.
- Read manga directly from archives: 7z, cb7, zip, cbz.
- Heavily utilize background decoding and prefetch to make it "feel snappy". As a result, hold arrow keys can flip pages extremely smooth. The bottleneck is always the decoder and CPU power.

## Usage

    cy <dir/ | archive.{7z,cb7,cbz,zip} | image-file.{png,jpg,gif,avif,jxl,...}>

Proper CLI handling later.

## Build/Run

    nix develop -c cargo build --release

    nix run github:MidAutumnMoon/cryslen

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

|![cryslen page jump popup](assets/screenshot-2026-11/page-jump-popup.avif)|
|-|
|The `:` page jump popup. Background is dimmed.|

## Shortcuts

- `←/→` switch pages
- `space`/`shift+space` forward/backward by reading order
- `:` brings up a popup for typing page number to jump to (like vim)
- `home`/`end` first/last spread
- `n`/`m`/`c` normal, manga, comic mode (first letter of each mode)
- `f` toggle fullscreen
- `mouse wheel` zoom
- `r` reset zoom level; `w`/`h` fit width/height; `1` actual size
- `drag` move/pan
- `b`/`shift+b` next/previous sibling archive/folder
- `q` or `esc` quit

## License and disclosure

GPL-3.0.

LLM-assisted. Not quite vibe coding but not quite handmade either.
