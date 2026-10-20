# Build, tooling, and benchmark hazards

Reference for building cryslen and for measuring anything on this
machine.

## Nix devShell (needed for every build)

The vendored build chain for libavif-sys's dav1d: `cmake`, `meson`,
`ninja`, `perl`, `clang`, `pkg-config`, and `nasm` (dav1d's x86
assembly requires it). The flake's
`packages.default` mirrors the devShell as `nativeBuildInputs` for
`buildRustPackage` with `cargoLock.lockFile = ./Cargo.lock`, so keep
the lockfile committed.

aom's cmake greps `nasm -hf` for `-Ox` and rejects nasm 3.x, but aom
is not in this tree (`codec-dav1d` only). If `codec-aom` ever joins,
prefer yasm or a patched libaom-sys over a wrapper script.

## Cargo feature hygiene for a Wayland-only build

Every GUI-layer dep needs `default-features = false` plus explicit
`wayland` (rendering.md explains why egui-winit's defaults leak X11).
Watch new transitive defaults after upgrades. A plain version bump can
silently re-import x11rb.

## sevenz-rust2

`default-features = false, features = ["bzip2"]` keeps LZMA decode and
drops the AES stack (aes/cbc/sha2/getrandom) and the LZMA encoder,
which the read-only path never calls. The `ArchiveReader` handle is
retained for the session. A background pass decompresses every page
once into a byte-budgeted tier, so solid archives pay their block
decode once.
Non-UTF-8 entry names surface as errors from `file_names()` and are
skipped.

## Benchmarking on this machine

- **Never write large output inside a timed region on /tmp.** It is
  zram, and writing a large raw frame costs more compression time than
  the decode being measured. Pipe to `/dev/null` (keeping the encode
  format) or time in-memory buffers only.
- Warm up once, run enough iterations to smooth jitter, report the
  mean. Do not time while another cargo build runs. Compile load on
  all cores visibly moves decoder numbers.
- Corpus generation that produced stable comparisons: `magick -size
  6000x4000 plasma:fractal src.png` (noise-heavy, decoder-hostile),
  `avifenc -s 8 -j <n> src.png test.avif`, `cjxl -e 3 src.png
  test.jxl`. A smooth-PNG variant exposes the deflate-bound case.
- Multi-threaded decoders (dav1d, jxl-rs runner) report wall time.
  Single-threaded ones (zune) report per-core. Label which when
  comparing.
- `crates.io` API greps need a User-Agent header, and the HTML search
  page is JS-rendered and useless for scraping. The registry source
  cache at `~/.local/share/cargo/registry/src/` is the fastest way to
  check what a dependency actually does.

## jj (the repo's VCS)

Colocated jj+git. Bookmarks must move forward to push: work lands as
commits, then `jj bookmark set main -r @-` (or create on the tip)
before `jj git push --all`. Pushed commits become immutable. `jj new`
starts the next working-copy commit. Anonymous-head commits sit
invisible to `--all` until a bookmark covers them. `jj restore --from
<rev> <path>` is the recovery path for bad edits.

## Gates

All inside `nix develop`: `cargo clippy --all-targets -- -D warnings`
plus `cargo test`. dav1d compiles from source on the first run, then
cargo caches it.
