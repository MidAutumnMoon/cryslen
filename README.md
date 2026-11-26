# cryslen

<div align="center">
  <img src="assets/icon-raw.avif" alt="cryslen icon" width="350">

### `cy`: a minimal Wayland image viewer with manga mode.
</div>

I built it from scratch because I want:

- Config-less, stateless <sub>*except a throwaway cache*</sub>, feature-less viewer. Knobs and shortcuts are all baked in. They fit my habit but maybe not yours.
- Support enough file formats. I use JXL and AVIF heavily, so support for these two is very good.
- Manga mode: right to left, double page. Of course it also has left to right mode like a normal image viewer.
- Read manga directly from archives: 7z, cb7, zip, cbz.
- The UX *feels snappy*. Hold arrow keys can skim through pages extremely fast. Although the bottleneck is always the image dimension and your CPU.

What's not working right now:

- HDR. ← <sub>I don't care about HDR for now. Don't submit PR for HDR.</sub>
- RAR archive format.
- Very large pictures.
- Animated AVIF.
- Edge cases.

What *will not* be added (don't submit feature request or PR for that):

- X11 support.
- Runtime config or key binding remap.
- Image editing.
- Watch videos.

## Usage

    cy <dir/ | archive.{7z,cb7,cbz,zip} | image-file.{png,jpg,gif,avif,jxl,...}>

Proper CLI handling later.

## Build/Run

```sh
nix develop -c cargo build --release

nix run github:MidAutumnMoon/cryslen
```

I didn't set `lto` in cargo manifest. If you want smaller binary and better performance, build it with:

```sh
CARGO_PROFILE_RELEASE_LTO=thin
```

## Screenshots

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

**Extremely snappy skimming:**

https://github.com/user-attachments/assets/8c140333-8681-464c-b4d2-82594f4115f7

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
