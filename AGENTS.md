# AGENTS.md

## Project Overview

cryslen is an image viewer, authored by MidAutumnMoon and licensed under GPL-3.0-or-later.

- **Repository**: <https://github.com/MidAutumnMoon/cryslen>

## Make changes

- Make systematic changes. Don't be scoped to the spot you are looking at.
- Make breaking changes. Trace, find, redesign, from top to bottom.
- Challenge existing designs.

## Look Things Up

- When unsure about a library, tool, or API, use web search or Context7 before guessing.
- Prefer Context7 for fresh library docs. Or read source code from local registry.
- Don't hallucinate option names, function signatures, or CLI flags. Look it up.
- Use `$CARGO_HOME` instead of `~/.cargo`.

## No `| tail` / `| head`

Do not pipe any command output through `head` or `tail`, tools will properly handle large output natively.

## Fitting New Code

- Recover the domain truth before choosing a shape: valid states, natural owner, boundaries, and relevant failure or resource constraints.
- Give each fact one authoritative home. Prefer representations and APIs that enforce invariants over comments or caller discipline.
- Read the surrounding code first. Extend an abstraction only when the new behavior shares its policy and ownership; otherwise reshape proportionally instead of adding special cases, parallel paths, or copied logic.
- Keep dependencies and data flow explicit. Do not recover required context from ambient state or reconstruct information discarded earlier.
- Design for real pressures, not hypothetical reuse. Preserve existing behavior unless a change is requested; consolidate experiments before landing.

## Coding Conventions

### nix devShell

Run cargo commands in `nix develop` devShell.

### Error Handling

- Use [`eros`](https://docs.rs/eros/latest/eros/).
- Prefer richer return types instead of convention.
- Avoid `unwrap()` and `panic` in production code (Clippy will warn).

### String

- Use [`indoc`](https://docs.rs/indoc/) for multi-line strings.
- Ignore `max_width` for long single-line strings. No need to workaround the limit by concatenating or splitting strings.

### Logging

- Use the `tracing` crate for all logging.
- Initialize the subscriber via `ino_tracing` at the start of `main()`.

## Verify Changes

After making changes (scope to the relevant app/crate):

1. **Lint**: `cargo clippy --all-features --all-targets --package <crate>` Don't act immediately on clippy suggestions. Don't suppress warnings.
2. **Test**: `cargo test --all-features --package <crate>`
3. Collect, dedup and report all triggered clippy warning to the user and require user review. Because new warnings are symptom of deeper architecural problems, don't suppress or workaround them without user's review. But ensure tests pass before considering the change complete.

Do not pipe command output through `head` or `tail`, especially `cargo test`.

### Visual smoke test

Tools, usable separately:

- App: run the release binary under `nix develop`, with `RUST_LOG=cryslen=debug` and a scratch `$TMPDIR` for a cold store. Its window `app_id` is `cryslen`; the log records navigation and frame events.
- niri: `niri msg windows` lists windows, `niri msg action focus-window --id <id>` focuses one, `niri msg --json focused-window` confirms.
- dotool: `printf 'key <name>\n' | nix run nixpkgs#dotool --` sends real input; it needs `/dev/uinput` and acts on the focused window. Key bindings: `src/keys.rs`.
- Screenshot: `niri msg action screenshot-window --id <id> --write-to-disk true` writes a PNG under `~/Pictures/Screenshots`.

A screenshot shows the settled frame, not an interim one. Delete the screenshots you create and stop the app with `pkill -x cy`.

## VCS

Use `jj` vcs exclusively. For example, `jj status`, `jj new`.

External pager is configured. Use `jj diff --git` and `jj show --git` for traditional git style diff output.

Don't commit on behalf of the user. Don't write commit message

