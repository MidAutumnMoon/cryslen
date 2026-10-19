# Internal notes

Working knowledge for future work on cryslen: which libraries to use
and why, API sharp edges, and build traps. Speed claims are ratios and
orders of magnitude measured on one laptop, not portable absolutes.

- [decoding.md](decoding.md): decoder selection with reasons, jxl-rs
  and libavif/dav1d integration essentials, the decode-worker pattern,
  and what does not exist (so nobody searches for it again).
- [rendering.md](rendering.md): the winit/glutin/glow/egui_glow
  layering and feature flags, frame pacing and repaint arming, GL state
  gotchas, quad strategy, YUV-in-shader math, upload policy.
- [build-and-bench.md](build-and-bench.md): nix devShell and the nasm
  shim (with its CMake cache trap), feature-gate hygiene, benchmark
  methodology (the zram /tmp hazard above all), jj push mechanics,
  gate commands.
