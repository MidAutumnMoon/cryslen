# cryslen

cryslen is an image viewer, authored by MidAutumnMoon and licensed under GPL-3.0-or-later.

- Repository: <https://github.com/MidAutumnMoon/cryslen>

## Make changes

- Make systematic changes. Don't be scoped to the spot you are looking at.
- Make breaking changes. Trace, find, redesign, from top to bottom.
- Challenge existing designs.

## Repo conventions

- Nix devShell: Run cargo commands in `nix develop` devShell. Otherwise build will break.
- Error handling Use [`eros`](https://docs.rs/eros/latest/eros/). Avoid `unwrap()` and `panic` outside tests.
- Use [`indoc`](https://docs.rs/indoc/) for multi-line strings.
- Use `tracing` for logging.

## Verify Changes

After making changes (scope to the relevant app/crate):

1. **Lint**: `cargo clippy --all-features --all-targets --package <crate>`. MUST NOT suppress warning without permission. Skip and report warnings that might signal architectural problems. Fix other warnings properly instead of taking clippy suggestions as-is.
2. **Test**: `cargo test --all-features --package <crate>`

Do not pipe command output through `head` or `tail`, especially `cargo test`.

### Visual smoke test

Tools, usable separately:

- App: run the release binary under `nix develop`, with `RUST_LOG=debug` and a scratch `$TMPDIR` for a cold store. Its window `app_id` is `cryslen`; the log records navigation and frame events.
- niri: `niri msg windows` lists windows, `niri msg action focus-window --id <id>` focuses one, `niri msg --json focused-window` confirms.
- dotool: `printf 'key <name>\n' | nix run nixpkgs#dotool --` sends real input; it needs `/dev/uinput` and acts on the focused window. Key bindings: `src/app.rs`.
- Screenshot: `niri msg action screenshot-window --id <id> --write-to-disk true` writes a PNG under `~/Pictures/Screenshots`.

A screenshot shows the settled frame, not an interim one. Delete the screenshots you create and stop the app with `pkill -x cy`.

## VCS

Use `jj` vcs exclusively. For example, `jj status`, `jj new`.

External pager is configured. Use `jj diff --git` and `jj show --git` for traditional git style diff output.

Don't commit on behalf of the user. Don't write commit message

