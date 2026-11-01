Your original intuition is good: **spend idle time so that future turns are already paid for.** I think the later proposals partly lost that goal while trying to make the implementation tidy.

- Six slots simplify residency, but throw away the large lookahead you originally wanted.
- Animated skeletons guarantee movement, but do not deliver “colors fly by.”
- Thumbnails are the right direction, but generating them through full decodes does not solve first-pass throughput.
- “Landing jumps the queue” sounds responsive, but cannot interrupt the expensive job already running.

The missing piece is not another cache mechanism. It is a clear separation between **navigation, presentation, and image quality**, backed by an honest model of what work costs.

## 1. Decide what “fast” promises

There are three different promises here:

| Promise | What the user experiences | What it requires |
|---|---|---|
| Immediate response | Every turn visibly affects the interface | Rendering that never waits for decoding |
| Fast visual browsing | Recognizable page content changes while skimming | Ready previews, or enough preview-generation throughput |
| Fast landing | Stopping quickly produces a readable page | Prioritized work and control over background interference |

A skeleton satisfies the first promise, not the second. A large cache can satisfy all three temporarily. A prepared preview cache can satisfy the second across an entire book.

That distinction matters because otherwise a proposal can “solve skimming” while only animating the absence of images.

Taking your reported JXL throughput as given: if navigation consumes $r$ pages per second, decoding supplies $\mu$, and you start with $C$ usable pages ahead, the approximate runway is:

$$
T_{\text{runway}} = \frac{C}{r-\mu}, \qquad r>\mu
$$

At 20 pages/s, with decoding at 1.8 pages/s, 60 cached pages buy about 3.3 seconds of continuous skimming.

That is useful! But it is a burst buffer, not a throughput fix.

Conversely, five seconds spent reading one page gives that decoder time to prepare roughly nine more. Your idle-time idea can work extremely well for normal reading followed by occasional bursts.

**I would preserve that advantage, rather than replacing it with a tiny fixed ring.**

## 2. Cache a large *perceptual* working set

The biggest change I would make is to stop treating a page as one indivisible full-resolution object.

Use a small number of representations:

| Representation | Purpose | Desired coverage |
|---|---|---|
| Preview | Recognize composition and colors during motion | Broad |
| View-sized image | Read comfortably at the current fitted size | Nearby pages |
| Native-resolution image | Zoom and inspect detail | Current page and selectively retained neighbors |

“View-sized” should be based on physical viewport pixels, with some headroom—not a universal fixed edge length.

This creates a useful middle ground between:

- Hundreds of native-resolution bitmaps.
- Six native-resolution bitmaps.
- Tiny thumbnails followed by another long wait whenever the user stops.

**Landing should request enough detail to read, not automatically require every original pixel.** If a suitable view-sized representation is ready, show it immediately. Fetch native resolution when the zoom or expected interaction warrants it.

With your current decoders, producing a smaller representation may still require a full decode. Be explicit about that:

> Smaller representations reduce retention, upload, and presentation cost. They do not necessarily reduce initial decode cost.

That is still a substantial win. A 50-megapixel RGBA image needs about 191 MiB before mipmaps. A preview with a 512-pixel maximum edge needs at most 1 MiB in RGBA. Those are radically different cache depths.

### Keep byte accounting

I strongly disagree with deleting byte budgets in favor of six slots.

With the RGBA page sizes your current decoder permits, six full-resolution textures with mipmaps can approach 4 GiB—before decoder scratch space, transfer buffers, or archive contents.

Slot count is a bound on **object count**, not a useful bound on memory pressure.

You can delete competing eviction policies. You cannot delete resource accounting.

I would use:

- One coordinated residency policy.
- Separate accounting for the resources that actually differ, including system memory and GPU allocations.
- Explicit allowance for in-flight decode output and scratch space.
- Lower-resolution fallback when a requested native representation does not fit.

Likewise, “never cache decoded pixels in RAM” is too absolute. A RAM spill tier can be valuable when uploading is much cheaper than decoding again. What is wasteful is retaining byte-identical CPU and GPU copies **without an intentional use for the CPU copy**.

Roles such as current, next, and previous are good **priorities and protections**. They need not dictate the entire cache’s structure.

## 3. Separate where the user is going from what you can show

I would keep three facts distinct:

1. **Requested position:** where input says the user wants to go.
2. **Presented position:** which page content is actually on screen.
3. **Presented quality:** preview, view-sized, or native.

Your existing `index` versus `Screen::shown` already contains part of this distinction. The problem is that `plan_screen` generally resolves the difference by holding the old spread.

For ordinary reading, briefly holding the previous readable spread can be reasonable. During sustained navigation, it becomes the frozen screen you described.

I would make two presentation policies explicit.

### Ordinary turns: exact destination, best available quality

On a turn:

- Update the requested position immediately.
- If the destination has any suitable representation, show it.
- Promote its quality as better representations arrive.
- If nothing is available, acknowledge the turn visually without pretending the old page is the destination.

A small directional transition, page-edge movement, or target-page indicator is enough. A large skeleton animation is not mandatory.

For spreads, keep the **page identities** correct even if their quality differs. A preview beside a sharp page is preferable to accidentally presenting an old page as the new partner.

### Sustained skimming: a sampled stream, not a replay queue

You said you do not need to see every page in detail. I would go further:

**You probably do not need to present every page at all.**

Suppose the user requests pages 40 through 80 while only a few images become ready. Showing 42, 49, 61, and 78 can still feel like browsing forward. Replaying 41 through 64 after the user has stopped at 80 feels broken.

The rules should be:

- Navigation never waits for an intermediate page.
- There is no backlog of intermediate pages that presentation must replay.
- A completed intermediate decode may contribute one useful sample to the current skim.
- Within a forward skim, those samples must not move backward.
- Once the user stops, old samples must not interrupt landing.

This also resolves an apparent conflict:

> “Latest want wins” should discard obsolete queued work, not necessarily discard every useful result from an already-running job.

A completed page can still be valuable as a cache entry or a skim sample. It should simply not possess the right to take over the screen.

For the first implementation, I would prefer cached representations of the exact target whenever available, and permit opportunistic samples only while a skim gesture is active.

### Infer skimming from input, not decoder failure

“Enter skim when another turn arrives while the screen lags” is a useful fallback signal, but not a good definition of user intent.

It makes behavior depend on whether the cache happens to be warm. A slow isolated turn and a held navigation key are different interactions.

Your input path currently handles pressed keys but does not use releases for navigation state. You already receive `key.repeat`; use that, and track release:

- A normal press is an ordinary turn.
- Repeated navigation indicates sustained browsing.
- Release ends that held-key gesture and prioritizes its destination.
- Focus loss clears the held state.

For rapid tapping, a short quiet-period timeout is perfectly reasonable. **Eliminating timers is not intrinsically cleaner.** Small amounts of hysteresis often make an interaction more stable.

## 4. Use one ranked plan for both work and residency

This is where I would simplify the implementation.

Instead of maintaining:

- A historical pending trail.
- A separate recency history.
- A worker cache policy.
- A renderer cache policy.
- A special forward-only staging policy.

Maintain one authoritative plan describing **which representations are currently worth having, in priority order**.

A practical starting order might be:

1. Missing readable representations for the requested spread.
2. The most likely next spread.
3. A small reversal cushion.
4. A broad preview window biased toward the navigation direction.
5. Further quality upgrades or optional preview preparation.

The exact order between nearby previews and quality upgrades should be informed by measurements. It does not require a sophisticated optimizer initially.

Use that same ranking to answer:

- What job should start next?
- What gets admitted into the cache?
- What survives pressure?
- What should be uploaded ahead of presentation?

This gives you a much stronger invariant:

> A page is decoded speculatively because the current plan expects to retain a useful representation of it—not merely because it once entered a pending set.

### Admission before execution

Before starting speculative work:

- Confirm that it is still wanted by the latest plan.
- Reserve capacity for its output.
- Choose the representation you intend to retain.
- Avoid admitting work that will immediately become the preferred eviction victim.

That addresses the store–evict–retune–decode loop at its source.

Also distinguish **nonresident** from **pending**. A representation that was deliberately not admitted should not automatically look like a missing job that must be scheduled again.

### Keep a small backward cushion

I would not delete backward support as a principle.

There are two separate things:

- Retaining a previous page that you already paid to decode.
- Spending new decode time speculatively on pages behind the reader.

The first is usually cheap and valuable. The second can have a lower priority during forward reading.

When direction changes, the bias should change with it. “Reading is forward” is a default prediction, not a permanent property of the interaction.

## 5. The thumbnail proposal is closest—but its production model matters

Your thumbnail tier is the closest proposal to what I would build. Its weak point is not the idea of thumbnails. It is **when and how you pay for them**.

These are the meaningful options:

| Option | What it buys | Remaining limitation |
|---|---|---|
| Deep session cache of originals | Instant turns through a prepared region | Expensive memory; finite runway |
| Broad session cache of smaller representations | Much deeper runway and cheaper uploads | Existing stack still pays full decode cost initially |
| Persistent preview or view-sized cache | Fast revisits and prepared-book skimming | First preparation costs time; needs invalidation and disk limits |
| Faster or genuinely reduced-resolution decoding | Better cold performance | Depends on backend, codec, and actual bitstreams |

If disk caching is acceptable, it deserves serious consideration.

Your current proposal frees all thumbnails on a book switch. That means the application repeatedly pays the slowest part of preparation. Persisting a derived preview set turns the expensive first pass into reusable work.

It can be modest:

- A versioned derived-image cache.
- Book/member identity and a suitable source fingerprint.
- A bounded disk budget.
- Ordinary small image files or a simple indexed store.

It does not need a database, a bespoke packed pixel format, or a large GPU atlas in the first version.

An optional “prepare fast browsing” operation is also a legitimate product choice for unusually expensive books. It acknowledges that complete, first-ever, content-rich skimming is a preparation problem.

**If every unseen page requires a full 550 ms decode, no cache policy can manufacture its real colors in 30 ms.** You must prepare earlier, make decoding cheaper, or show fewer real samples.

That is the unavoidable trade-off—not evidence that you have failed to discover the right UX trick.

## 6. “Idle” work is not harmless work

This is the most important challenge to the proposed sweep:

> “Worst added turn latency: one decode” is not a reassuring guarantee when a decode takes half a second—or several seconds for an outlier.

Duty cycling reduces the probability that input collides with background work. It does not reduce the worst-case residual time of a noninterruptible decode.

Likewise, a priority queue cannot make landing preempt the job already inside `decode::decode`.

You have several choices:

- Accept that residual latency and keep speculation especially valuable.
- Add cooperative cancellation where a decoder genuinely supports it.
- Allow interactive work to start alongside background work, with a shared CPU and memory budget.
- Use process isolation if strict interruption of an uncooperative decoder is essential.

I would start conservatively:

- One bounded executor shared across book generations.
- No accumulation of detached workers across switches.
- Obsolete queued jobs removed immediately.
- Cancellation checked at real boundaries.
- A clear distinction between cancellation and failure.

This bounds resource accumulation, but does not magically interrupt an existing decode. If a new book must start decoding immediately despite an old uninterruptible job, you need either spare execution capacity or actual cancellation.

Be careful adding page-level concurrency: your AVIF path already uses up to eight threads, and JXL up to sixteen. Several such jobs can oversubscribe the machine and hurt tail latency.

### Revisit the decoder implementation before accepting its speed as fixed

One concrete detail in your code deserves profiling: `jxl::Pool::run` creates and joins scoped OS threads on every runner invocation.

That may or may not dominate your large-page workload, but it is worth measuring against a persistent executor. Also benchmark an alternative native JXL backend on the same lossless corpus before concluding that all remaining improvements must come from UX.

I would not assume reduced-resolution JXL is cheap for your particular lossless files merely because an API exposes it. That needs an actual benchmark.

## 7. Changes I would make in these files

There are several important implementation consequences beyond changing cache sizes.

### `render.rs`: reserve and protect before trimming

`PageTextures::ensure_cached` inserts and evicts before `bind` records the new seat binding.

That means a newly uploaded destination can be evicted before it becomes protected, after which the seat can refer to a page absent from `textures`.

Fix that independently of any larger redesign: admission and temporary protection must precede eviction.

Longer term, let `PageTextures` manage GL resources, not choose the application’s residency policy.

### `screen.rs`: preserve geometry across quality changes

`SeatShow.dims` currently comes from `Decoded`, and `view::layout` uses those dimensions directly.

Once you introduce previews, separate:

- Oriented source dimensions.
- Representation pixel dimensions.

Otherwise replacing a thumbnail with a native image can change the fit, zoom percentage, and layout.

A quality promotion should sharpen the same page in the same place. It should not look like another page turn.

### `app.rs`: uploads are part of the latency problem

Currently `display` and `stage_forward` can upload large images and generate mipmaps on the UI thread inside `frame`.

Even if decoding is perfectly scheduled, that work can still cause input and animation hitches.

Measure it, then constrain background upload work. Small previews are especially valuable here: they can become visible without first committing hundreds of megabytes to GL.

Also, `page_watched` only watches the destination and next spread. A deep GPU-ready cache needs a display-independent path for handling all admitted prefetch results.

Coalesce “results available” notifications and drain them with a bounded work budget. **A completion may require resource handling without requiring a visible repaint.**

### `source.rs`: stop treating all archives alike

For ZIP, members are independently accessible. A distant destination need not wait for a whole forward unpack pass.

For solid 7z, sequential extraction may genuinely be required. That deserves a different policy, potentially including disk spooling for large archives.

Add cancellation checks between members and during bounded reads where practical. Generation tags prevent stale display results; they do not stop old extraction work consuming CPU and memory.

Also note that `switch_book` performs sibling discovery and `Source::open` synchronously on the UI thread. Slow storage or expensive listing can make switching feel stuck before decoding even begins.

### Failures: retry at the layer that failed

Do not replace “sticky forever” with “retry every frame.”

Use structured distinctions:

- Transient read failure: bounded retry/backoff.
- Corrupt or unsupported content: retain the result until explicit retry or source change.
- GPU resource pressure: try a smaller representation or retry after pressure changes.
- Cancellation: not an image failure.

For archives, retrying only the worker is insufficient if `Fill` still contains the same sticky source error.

## 8. How I would implement this without another giant rewrite

I would deliver it in three passes.

**First: fix scheduling and presentation using the existing representations.**

- Latest-plan-wins instead of replaying every retune.
- Admission before work.
- No pending historical trail.
- Correct upload protection.
- Small direction-aware retention.
- Explicit held-key/release handling.
- No replay of intermediate pages after landing.

**Second: add preview and view-sized representations.**

- Mint previews from useful decodes.
- Keep broad previews and a narrower readable-quality window.
- Preserve source geometry across promotions.
- Make landing readable first, then sharper.
- Add a persistent derived cache only if its benefits justify it.

**Third: expand background preparation only after measuring its interference.**

Measure at least:

- Input to first visible response.
- Input to actual destination preview.
- Release to readable landing.
- Upload-related frame stalls.
- Useful versus discarded decode work.
- Peak memory including in-flight work.

For tests, extract the planner into deterministic state transitions. Feed it input events and manually completed fake jobs. A trace such as “hold forward, reverse, release, then switch books during a decode” will reveal more than a callback parked behind a condvar.

**My recommendation:** revive your original large-prefetch idea, but make it a **large perceptual cache rather than a large collection of native bitmaps**. Combine that with one ranked plan, input-driven presentation, and an explicit cold-cache fallback.

That preserves what was good about your original vision without pretending either six slots or animated placeholders can substitute for ready image content.
