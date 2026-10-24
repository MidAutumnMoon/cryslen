# AGENTS.md

## Project Overview

cryslen is an image viewer, authored by MidAutumnMoon and licensed under GPL-3.0-or-later.

- **Repository**: <https://github.com/MidAutumnMoon/cryslen>

## Communication

- Be concise. Say the thing, stop. Say natural, straight and plain English. No jargon in any form.
- Don't repeat what user already said or what's already in context.
- Don't meaninglessly repeat what the user already knows or owns.
- Don't pad with disclaimers, summaries, or "hope that helps" type closings.
- If something is wrong, say what's wrong and how to fix it. Don't hedge.
- Write user-facing messages in concise and non-repetitive manner. Don't repeat what user already knows.

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

1. **Lint**: `cargo clippy --all-features --all-targets --package <crate>`
2. **Test**: `cargo test --all-features --package <crate>`
3. Collect, dedup and report all triggered clippy warning to the user and require them to review. Because new warnings are symptom of deeper architecural problems, don't silence or workaround them without user's review. Ensure tests pass before considering the change complete.

Do not pipe command output through `head` or `tail`, especially `cargo test`.

## VCS

Use `jj` vcs exclusively. For example, `jj status`, `jj new`.

`difftastic` is configured as diff tool. Use `jj diff --git` and `jj show --git` for traditional git style diff output.

