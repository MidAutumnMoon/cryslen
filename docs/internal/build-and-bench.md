# Build, tooling, and benchmark hazards

Reference for building cryslen and for measuring anything on this
machine.

## Nix devShell (needed for `--features avif`)

The vendored build chain for libavif-sys's dav1d: `cmake`, `meson`,
`ninja`, `perl`, `clang`, `pkg-config`, and a **nasm shim** (below).
`--features jxl` is pure Rust and needs none of this. The flake's
`packages.default` mirrors the devShell as `nativeBuildInputs` for
`buildRustPackage` with `cargoLock.lockFile = ./Cargo.lock`, so keep
the lockfile committed.

### The nasm shim (do not remove casually)

aom's cmake configure greps `nasm -hf` output for the literal "-Ox".
nasm 3.x still supports the flag but dropped it from the help text, so
configure fails with "Unsupported nasm: multipass optimization not
supported". The shim appends the line for `-hf` and execs the real
nasm otherwise.

**Cache trap:** when the toolchain changes (e.g. adding the shim),
CMake caches `CMAKE_ASM_NASM_COMPILER` by absolute store path in
`target*/build/libaom-sys-*/out/build/CMakeCache.txt`. `cargo clean -p
libaom-sys` does not clear it. Delete the build dir manually before
rebuilding, or the stale path keeps failing while the shim looks
innocent.

## Feature-gated dead code

The YUV types are constructed only by the `avif` decoder, so default
builds warn "never constructed". The pattern used:
`#[cfg_attr(not(feature = "avif"), allow(dead_code))]` on the specific
variants/enums. It is self-documenting, and the warning returns if the
constructor ever disappears for real.

## Cargo feature hygiene for a Wayland-only build

Every GUI-layer dep needs `default-features = false` plus explicit
`wayland` (rendering.md explains why egui-winit's defaults leak X11).
Watch new transitive defaults after upgrades. A plain version bump can
silently re-import x11rb.

## sevenz-rust2

`default-features = false, features = ["bzip2"]` keeps LZMA decode and
drops the AES stack (aes/cbc/sha2/getrandom) and the LZMA encoder,
which the read-only path never calls. Listing needs one archive walk.
`ArchiveReader` owns its file handle so per-read reopens are inherent
to the API (fine for non-solid archives, measurable for solid ones).
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

`cargo clippy --all-targets` (default features) and, inside `nix
develop`, `cargo clippy --all-targets --all-features -- -D warnings`
plus `cargo test --all-features`. The avif feature compiles dav1d from
source, so keep those runs for integration moments, not every edit.
