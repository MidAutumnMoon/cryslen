//! App: winit event handling, state transitions, frame composition.

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use egui_winit::winit;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::StartCause;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::ControlFlow;
use winit::event_loop::EventLoopProxy;
use winit::keyboard::ModifiersState;
use winit::window::WindowId;

use crate::decode;
use crate::decode::Format;
use crate::fault::OpenError;
use crate::fault::PageError;
use crate::fault::SpawnError;
use crate::keys::Action;
use crate::plan::Direction;
use crate::plan::Plan;
use crate::preview::Artifact;
use crate::preview::VIEW_EDGE;
use crate::render::PageTextures;
use crate::source::Source;
use crate::ui;
use crate::ui::PageFacts;
use crate::ui::StatusLine;
use crate::view;
use crate::view::FitMode;
use crate::view::PageCount;
use crate::view::Spread;
use crate::view::ViewMode;
use crate::view::Viewport;
use crate::window::Window;
use crate::worker;
use crate::worker::Motion;
use crate::worker::Want;
use crate::worker::Worker;

pub enum UserEvent {
    /// Arms a wake; `new_events` requests the redraw at the deadline.
    Repaint(Duration),
    /// One worker decision: a decoded artifact to upload, or a page
    /// failure to note. Every decision can change the frame.
    Worker(worker::Event),
}

/// A failed send means the loop is gone, leaving the event with no consumer.
fn post(proxy: &EventLoopProxy<UserEvent>, event: UserEvent) {
    let Ok(()) = proxy.send_event(event) else {
        return;
    };
}

fn spawn_worker(
    source: &Arc<Source>,
    plan: Plan,
    proxy: &EventLoopProxy<UserEvent>,
) -> Result<Worker, SpawnError> {
    let proxy = proxy.clone();
    Worker::spawn(Arc::clone(source), plan, move |event| {
        post(&proxy, UserEvent::Worker(event));
    })
}

struct Gl {
    pages: PageTextures,
    egui: egui_glow::EguiGlow,
    // Must drop last: owns the GL context used by the fields above.
    window: Window,
}

impl Drop for Gl {
    fn drop(&mut self) {
        // egui_glow reports leaked resources if destroy never runs.
        self.egui.destroy();
    }
}

/// What the layout and the status line know about one decoded page.
/// The dims are the content's, independent of which representation
/// (thumb, view, native) the cache currently holds.
#[derive(Clone, Copy)]
struct Fact {
    source: (u32, u32),
    format: Format,
}

/// Two presses closer than this cannot be read between, so they are a
/// flip rather than a read. A held key is a flip by its repeat events;
/// this covers very fast tapping and scripted input.
const SKIM_GAP: Duration = Duration::from_millis(100);

/// How long after the last navigation the skim decays to a still read,
/// which promotes the presented spread from thumbs to views.
const SKIM_DECAY: Duration = Duration::from_millis(250);

/// Artifacts one frame may upload, and the bytes they may hold. A big
/// page upload stalls the frame, so the rest waits for a fresh paint;
/// one artifact always goes through.
const UPLOAD_BUDGET_COUNT: usize = 16;
const UPLOAD_BUDGET_BYTES: usize = 8 * 1024 * 1024;

pub struct App {
    source: Arc<Source>,
    worker: Worker,
    gl: Option<Gl>,
    fatal_error: Option<eros::ErrorUnion>,
    index: usize,
    mode: ViewMode,
    fit: view::Fit,
    /// The spread the last frame presented, `None` until a page has
    /// pixels.
    presented: Option<Spread>,
    /// The content facts of every page whose pixels reached the cache.
    facts: BTreeMap<usize, Fact>,
    /// Page failures, sticky per page so a blip cannot erase them.
    errors: BTreeMap<usize, Arc<PageError>>,
    /// Decoded artifacts waiting for an upload, in arrival order.
    queue: VecDeque<worker::Ready>,
    /// The last texture the GPU refused, held until a book switch.
    upload_error: Option<Arc<PageError>>,
    /// Whether the worker produces skim thumbs or full views.
    motion: Motion,
    /// When the last navigation was dispatched, for the skim window.
    last_nav: Option<Instant>,
    /// When the requested spread stopped being presentable while still:
    /// the "waiting for a view" hold, logged once per hold.
    hold_since: Option<Instant>,
    hold_logged: bool,
    /// The target set last sent to the worker: what the app was
    /// missing when it last spoke, in the same order.
    sent: Vec<Want>,
    /// Reading direction of the last turn: plans and skim tie-breaks
    /// follow it.
    direction: Direction,
    /// A failure the app noted (a book switch that failed). No frame
    /// retries it, so it holds until a successful switch clears it.
    noted_error: Option<Arc<OpenError>>,
    timers: ui::Timers,
    /// Cursor position in viewport px, same as window px.
    cursor: (f32, f32),
    dragging: bool,
    /// Held modifier state; winit 0.30 key events carry none, only
    /// `ModifiersChanged` does.
    modifiers: ModifiersState,
    /// The viewport the last frame painted into, read by input between frames.
    last_viewport: Viewport,
    /// Wakes the parked loop from egui and timed waits, which run outside
    /// the event loop's handlers.
    proxy: EventLoopProxy<UserEvent>,
}

impl App {
    pub fn new(
        source: Arc<Source>,
        index: usize,
        proxy: EventLoopProxy<UserEvent>,
    ) -> Result<Self, SpawnError> {
        let mode = ViewMode::Normal;
        let direction = Direction::Forward;
        let spread = view::spread_for(index, mode, source.len());
        let plan = Plan::build(spread, direction, mode, source.len());
        let worker = spawn_worker(&source, plan, &proxy)?;
        // The first frame is a frame away; declare what a cold start
        // lacks now, so the worker serves the visible page before it
        // starts the plan.
        let cold = cold_wants(spread, mode);
        worker.retarget(cold.clone());
        Ok(Self {
            source,
            worker,
            gl: None,
            fatal_error: None,
            index,
            mode,
            fit: view::Fit::default(),
            presented: None,
            facts: BTreeMap::new(),
            errors: BTreeMap::new(),
            queue: VecDeque::new(),
            upload_error: None,
            motion: Motion::Still,
            last_nav: None,
            hold_since: None,
            hold_logged: false,
            sent: cold,
            direction,
            noted_error: None,
            timers: ui::Timers::default(),
            cursor: (0.0, 0.0),
            dragging: false,
            modifiers: ModifiersState::empty(),
            last_viewport: Viewport::default(),
            proxy,
        })
    }

    /// Store `error` and exit so `main` can report it after `run_app`
    /// returns.
    fn fail(
        &mut self,
        error: eros::ErrorUnion,
        event_loop: &ActiveEventLoop,
    ) {
        self.fatal_error = Some(error);
        event_loop.exit();
    }

    pub fn take_fatal_error(&mut self) -> Option<eros::ErrorUnion> {
        self.fatal_error.take()
    }

    fn spread(&self) -> Spread {
        view::spread_for(self.index, self.mode, self.source.len())
    }

    /// Whether the texture cache holds any representation of `page`.
    fn live(&self, page: usize) -> bool {
        self.gl.as_ref().is_some_and(|gl| gl.pages.has_any(page))
    }

    /// Whether the texture cache holds `page`'s `kind`.
    fn has(&self, page: usize, kind: Artifact) -> bool {
        self.gl.as_ref().is_some_and(|gl| gl.pages.has(page, kind))
    }

    /// The largest representation a seat can draw, for the presentation
    /// log.
    fn kind_of(&self, page: Option<usize>) -> &'static str {
        let Some(page) = page else {
            return "empty";
        };
        if self.has(page, Artifact::Native) {
            "native"
        } else if self.has(page, Artifact::View) {
            "view"
        } else if self.has(page, Artifact::Thumb) {
            "thumb"
        } else {
            "none"
        }
    }

    /// Whether `page` clears the bar a still frame presents: a view
    /// image or the native page. A 192 px thumb is a motion artifact;
    /// showing it while reading would be a blur the reader did not ask
    /// for, so a still frame holds the previous spread until the view
    /// arrives instead.
    fn readable(&self, page: usize) -> bool {
        self.has(page, Artifact::View) || self.has(page, Artifact::Native)
    }

    fn is_fullscreen(&self) -> bool {
        self.gl
            .as_ref()
            .is_some_and(|gl| gl.window.winit().fullscreen().is_some())
    }

    fn frame(&mut self) -> eros::Result<()> {
        let frame_started = Instant::now();
        let Some(size) = self.sync_surface_size() else {
            return Ok(());
        };

        self.drain_ready();
        let drained = frame_started.elapsed();
        let now = Instant::now();
        self.tick_motion(now);

        let requested = self.spread();
        let skimming = self.motion == Motion::Skimming;
        let chosen = choose_presented(
            requested,
            self.presented,
            self.mode,
            self.source.len(),
            &self.facts,
            self.direction == Direction::Forward,
            |page| {
                if skimming {
                    self.live(page)
                } else {
                    self.readable(page)
                }
            },
        );
        if chosen != self.presented {
            let seats = chosen.map(|spread| {
                spread.seats(self.mode).map(|seat| self.kind_of(seat))
            });
            let took =
                self.last_nav.map(|last| now.saturating_duration_since(last));
            tracing::debug!(
                from = ?self.presented,
                to = ?chosen,
                motion = ?self.motion,
                ?seats,
                ?took,
                "presented"
            );
            self.presented = chosen;
            self.fit.commit();
        }
        // A still frame that is not showing its request is waiting for a
        // view; a long one is the "frozen page" report, so log it once.
        if self.motion == Motion::Still && chosen != Some(requested) {
            let since = *self.hold_since.get_or_insert(now);
            let waited = now.saturating_duration_since(since);
            if !self.hold_logged && waited >= Duration::from_millis(250) {
                self.hold_logged = true;
                tracing::info!(
                    requested = ?requested,
                    shown = ?chosen,
                    waited = ?waited,
                    queue = self.queue.len(),
                    "holding for a view"
                );
            }
        } else {
            self.hold_since = None;
            self.hold_logged = false;
        }
        let shown = chosen.unwrap_or(requested);

        let layout = view::layout(self.seat_dims());
        let spread_size = layout.total();
        let placed = self.fit.resolve(spread_size, self.last_viewport);

        if let Some(gl) = self.gl.as_mut() {
            let (mode, len) = (self.mode, self.source.len());
            // Retention is a performance policy: a dropped artifact is
            // just a want on the next frame. Thumbs make a skim cheap,
            // so they stay for the session; views serve the presented
            // spread, the requested one, and their neighbours; natives
            // serve only the pages on screen.
            gl.pages
                .retain(|page, kind| kept(page, kind, shown, requested, mode, len));
        }
        // The app's texture maps are the authority on what it can draw;
        // the worker's delivery ledger is not. Recompute the delta every
        // frame and send it only when it changes. `kept` covers every
        // want, so an artifact must leave the sent set before retention
        // can drop it, and the worker sees the whole round trip.
        //
        // A zero-sized viewport leaves the placement inert at 1:1,
        // which says nothing about what the screen can resolve.
        let (view_width, view_height) = self.last_viewport.size();
        let measurable = view_width > 0.0 && view_height > 0.0;
        let wants = wants(
            requested,
            shown,
            self.mode,
            self.motion == Motion::Still,
            // A native is the page's own pixels; the shown spread
            // is the only one up, so `wants` only asks for those.
            |page| {
                measurable
                    && self.facts.get(&page).is_some_and(|fact| {
                        native_wanted(placed.scale, fact.source)
                    })
            },
            |page, kind| self.has(page, kind),
            |page| self.errors.contains_key(&page),
        );
        let (mode, len) = (self.mode, self.source.len());
        debug_assert!(
            wants
                .iter()
                .all(|want| kept(want.page, want.kind, shown, requested, mode, len)),
            "retention must cover every want; a want that never leaves \
             the sent set can never be re-served"
        );
        if wants != self.sent {
            tracing::debug!(?wants, "requested");
            self.worker.retarget(wants.clone());
            self.sent = wants;
        }

        let folded = ui::fold_status(shown, |page| {
            let fact = self.facts.get(&page);
            // `waiting` means the spread has no pixels yet, so a page
            // counts as decoded only once its texture is cached.
            ui::PageState {
                format: fact.and_then(|fact| {
                    self.live(page).then_some(fact.format)
                }),
                source: fact.map(|fact| fact.source),
                error: self.errors.get(&page).cloned(),
            }
        });
        if let Some(delay) = self.timers.tick_wait(now, folded.waiting) {
            post(&self.proxy, UserEvent::Repaint(delay));
        }
        if let Some(delay) = self.timers.tick_toast(now) {
            post(&self.proxy, UserEvent::Repaint(delay));
        }

        let page = folded.page.map(|(page, decoded, dims)| PageFacts {
            claimed: self
                .source
                .name(page)
                .and_then(decode::format_from_name),
            decoded,
            dims,
        });
        let status = StatusLine {
            page_count: self.source.len().get(),
            spread: shown,
            mode: self.mode,
            toast: self.timers.toasting(),
            name: self.source.page_name(folded.speaker),
            page,
            zoom: placed.scale,
            waiting: self
                .timers
                .wait_word(now, || self.source.unpacking()),
            error: ui::fault_to_show(
                folded.error,
                self.noted_error.as_ref(),
                self.upload_error.as_ref(),
            ),
        };

        // `EguiGlow::run` only records shapes, so egui lays out before
        // pages are placed; `paint` later draws the bar above them.
        let viewport = {
            let show_bar = !self.is_fullscreen();
            let Some(gl) = self.gl.as_mut() else {
                return Ok(());
            };
            let mut content = egui::Rect::NOTHING;
            gl.egui.run(gl.window.winit(), |ui| {
                content = ui::show(ui, &status, show_bar);
            });
            content_viewport(
                size,
                content,
                gl.egui.egui_ctx.pixels_per_point(),
            )
        };
        // The bar's height changes without a winit resize; the status
        // line used the old scale, so redraw once.
        if viewport != self.last_viewport {
            self.fit.refit();
            self.last_viewport = viewport;
            self.request_redraw();
        }
        let drawn = self.fit.resolve(spread_size, viewport);

        let Some(gl) = self.gl.as_mut() else {
            return Ok(());
        };

        gl.egui.painter.clear(
            [size.width, size.height],
            clear_color(gl.window.translucent()),
        );
        gl.pages.draw(
            (size.width, size.height),
            viewport,
            &layout,
            shown.seats(self.mode),
            &drawn,
        );
        gl.egui.paint(gl.window.winit());
        gl.window.pre_present_and_swap()?;
        let total = frame_started.elapsed();
        if total >= Duration::from_millis(50) {
            tracing::info!(
                total = ?total,
                drain = ?drained,
                queue = self.queue.len(),
                presented = ?self.presented,
                "slow frame"
            );
        }
        Ok(())
    }

    /// Upload the queued artifacts under this frame's budget. A queue
    /// with more left re-arms the loop immediately, the only
    /// self-sustaining wake.
    fn drain_ready(&mut self) {
        let Some(gl) = self.gl.as_mut() else {
            return;
        };
        let started = Instant::now();
        let mut uploaded = 0_usize;
        let mut bytes = 0_usize;
        while let Some(cost) =
            self.queue.front().map(|ready| ready.image.resident_bytes())
        {
            let over_budget = uploaded >= UPLOAD_BUDGET_COUNT
                || bytes.saturating_add(cost) > UPLOAD_BUDGET_BYTES;
            if uploaded > 0 && over_budget {
                break;
            }
            let Some(ready) = self.queue.pop_front() else {
                break;
            };
            self.facts.insert(
                ready.page,
                Fact {
                    source: ready.image.source(),
                    format: ready.image.format(),
                },
            );
            let upload_started = Instant::now();
            let refused =
                gl.pages.insert(ready.page, ready.kind, &ready.image);
            let elapsed = upload_started.elapsed();
            tracing::debug!(
                page = ready.page,
                kind = ?ready.kind,
                width = ready.image.width(),
                height = ready.image.height(),
                elapsed = ?elapsed,
                "uploaded"
            );
            if elapsed >= Duration::from_millis(30) {
                tracing::info!(
                    page = ready.page,
                    kind = ?ready.kind,
                    width = ready.image.width(),
                    height = ready.image.height(),
                    elapsed = ?elapsed,
                    "slow upload"
                );
            }
            if let Err(error) = refused {
                self.upload_error = Some(Arc::new(error.widen()));
            }
            uploaded += 1;
            bytes = bytes.saturating_add(cost);
        }
        if uploaded > 0 {
            tracing::debug!(
                count = uploaded,
                bytes,
                elapsed = ?started.elapsed(),
                "drained"
            );
        }
        if !self.queue.is_empty() {
            post(&self.proxy, UserEvent::Repaint(Duration::ZERO));
        }
    }

    /// The source dims of the page in each visual seat, from the content
    /// truth, so a thumb-to-view promotion never refits.
    fn seat_dims(&self) -> [Option<(u32, u32)>; 2] {
        let shown = self.presented.unwrap_or_else(|| self.spread());
        shown.seats(self.mode).map(|seat| {
            let page = seat?;
            self.facts.get(&page).map(|fact| fact.source)
        })
    }

    /// One flip starts a skim; every nav opens a fresh decay window.
    fn note_navigation(&mut self, now: Instant, repeat: bool) {
        let gap = self.last_nav.map(|last| now.saturating_duration_since(last));
        let flip = repeat || gap.is_some_and(|gap| gap < SKIM_GAP);
        if flip && self.motion == Motion::Still {
            tracing::debug!(?gap, repeat, "skim begins");
            self.motion = Motion::Skimming;
            self.worker.set_motion(Motion::Skimming);
        }
        tracing::debug!(
            index = self.index,
            ?gap,
            repeat,
            motion = ?self.motion,
            "turn"
        );
        self.last_nav = Some(now);
        post(&self.proxy, UserEvent::Repaint(SKIM_DECAY));
    }

    /// End a skim once the reader stops, and repaint so the presented
    /// page promotes from its thumb.
    fn tick_motion(&mut self, now: Instant) {
        if self.motion != Motion::Skimming {
            return;
        }
        let Some(last) = self.last_nav else {
            self.end_skim();
            return;
        };
        let quiet = now.saturating_duration_since(last);
        if quiet < SKIM_DECAY {
            // A later navigation moved the decay past the wake that
            // fired, and the arming logic keeps the earliest wake, so
            // this frame must arm the new deadline itself.
            post(
                &self.proxy,
                UserEvent::Repaint(SKIM_DECAY.saturating_sub(quiet)),
            );
            return;
        }
        self.end_skim();
    }

    fn end_skim(&mut self) {
        tracing::debug!("skim ends");
        self.motion = Motion::Still;
        self.worker.set_motion(Motion::Still);
        self.request_redraw();
    }

    fn sync_surface_size(&mut self) -> Option<PhysicalSize<u32>> {
        let gl = self.gl.as_mut()?;
        let size = gl.window.winit().inner_size();
        if size.width == 0 || size.height == 0 {
            return None;
        }
        gl.window.resize(size);
        Some(size)
    }

    fn request_redraw(&self) {
        if let Some(gl) = self.gl.as_ref() {
            gl.window.winit().request_redraw();
        }
    }

    fn geometry(&self) -> ((f32, f32), Viewport) {
        (view::layout(self.seat_dims()).total(), self.last_viewport)
    }

    fn dispatch(
        &mut self,
        action: Action,
        repeat: bool,
        event_loop: &ActiveEventLoop,
    ) {
        match action {
            Action::Quit => event_loop.exit(),
            Action::Escape => {
                if self.is_fullscreen() {
                    if let Some(gl) = self.gl.as_ref() {
                        gl.window.winit().set_fullscreen(None);
                    }
                } else {
                    event_loop.exit();
                }
            }
            Action::SelectMode(mode) => {
                let switch = view::switch_mode(
                    self.mode,
                    mode,
                    self.index,
                    self.source.len(),
                );
                self.mode = mode;
                self.index = switch.index;
                self.direction = Direction::Forward;
                self.timers.toast(Instant::now());
                if switch.pairing_changed {
                    // The fit begins the new pairing; the switch is not
                    // a navigation, so the skim does not open.
                    self.fit.begin_turn();
                }
                // The plan carries the seating order and the spread, so
                // both kinds of switch retune; only a pairing change
                // restarts the fit.
                self.worker.retune(Plan::build(
                    self.spread(),
                    Direction::Forward,
                    self.mode,
                    self.source.len(),
                ));
                self.request_redraw();
            }
            Action::ToggleFullscreen => {
                use winit::window::Fullscreen;

                if let Some(gl) = self.gl.as_ref() {
                    let request = (!self.is_fullscreen())
                        .then_some(Fullscreen::Borderless(None));
                    gl.window.winit().set_fullscreen(request);
                }
            }
            Action::ResetView => {
                self.fit.select(FitMode::Page);
                self.request_redraw();
            }
            Action::SelectFit(mode) => {
                self.fit.select(mode);
                self.request_redraw();
            }
            Action::First | Action::Last => {
                let first = matches!(action, Action::First);
                self.index = if first {
                    view::first_spread()
                } else {
                    view::last_spread(self.mode, self.source.len())
                };
                self.after_page_change(travel(first), repeat);
            }
            Action::NextBook | Action::PrevBook => {
                self.switch_book(matches!(action, Action::NextBook));
            }
            Action::Right
            | Action::Left
            | Action::Forward
            | Action::Back => {
                let forward = action.is_forward(self.mode);
                self.index = view::advance(
                    self.index,
                    forward,
                    self.mode,
                    self.source.len(),
                );
                self.after_page_change(travel(forward), repeat);
            }
        }
    }

    fn switch_book(&mut self, forward: bool) {
        let Some(path) = self.source.sibling(forward) else {
            return;
        };
        let source = match Source::open(&path) {
            Ok(source) => Arc::new(source),
            Err(error) => {
                self.noted_error = Some(Arc::new(error));
                self.request_redraw();
                return;
            }
        };
        let spread = view::spread_for(
            view::first_spread(),
            self.mode,
            source.len(),
        );
        let plan = Plan::build(
            spread,
            Direction::Forward,
            self.mode,
            source.len(),
        );
        let worker = match spawn_worker(&source, plan, &self.proxy) {
            Ok(worker) => worker,
            Err(error) => {
                self.noted_error = Some(Arc::new(error.widen()));
                self.request_redraw();
                return;
            }
        };
        let cold = cold_wants(
            view::spread_for(view::first_spread(), self.mode, source.len()),
            self.mode,
        );
        worker.retarget(cold.clone());
        // The old worker may sit mid-decode on the old book; detaching
        // keeps the switch instant.
        self.worker.detach();
        self.worker = worker;
        self.source = source;
        self.index = view::first_spread();
        self.sent = cold;
        self.noted_error = None;
        // A book switch renumbers pages from zero, so nothing of the old
        // book can stay.
        self.facts.clear();
        self.errors.clear();
        self.presented = None;
        self.queue.clear();
        // The new worker starts with no targets; tell it what the first
        // spread of the new book lacks right away.
        self.upload_error = None;
        if let Some(gl) = self.gl.as_mut() {
            gl.pages.clear_all();
        }
        self.after_page_change(Direction::Forward, false);
    }

    /// Every spread mutation retunes the worker, restarts the fit, and
    /// opens the skim window here.
    fn after_page_change(&mut self, direction: Direction, repeat: bool) {
        self.direction = direction;
        let plan = Plan::build(
            self.spread(),
            direction,
            self.mode,
            self.source.len(),
        );
        self.worker.retune(plan);
        self.fit.begin_turn();
        self.note_navigation(Instant::now(), repeat);
        self.request_redraw();
    }
}

fn travel(forward: bool) -> Direction {
    if forward {
        Direction::Forward
    } else {
        Direction::Backward
    }
}

/// Whether the view artifact is upscaled: the placement asks for more
/// than [`VIEW_EDGE`] screen px across the page's long edge, so only the
/// native decode has pixels to spare.
#[expect(
    clippy::cast_precision_loss,
    reason = "page edges and the fixed view edge stay far below f32's exact-integer range, so the threshold comparison holds"
)]
fn native_wanted(scale: f32, source: (u32, u32)) -> bool {
    let edge = source.0.max(source.1);
    edge > 0 && scale > VIEW_EDGE as f32 / edge as f32
}

/// Every artifact the frame's presentation is missing, in service
/// order: the requested spread first (a pair needs both seats before it
/// can present), then the shown one. A still frame waits for views and
/// thumbs; a skim runs on thumbs alone. A native is wanted only for a
/// page on screen: the requested spread is not up yet.
fn wants(
    requested: Spread,
    shown: Spread,
    mode: ViewMode,
    still: bool,
    upscaled: impl Fn(usize) -> bool,
    has: impl Fn(usize, Artifact) -> bool,
    errored: impl Fn(usize) -> bool,
) -> Vec<Want> {
    let readable = |page: usize| {
        has(page, Artifact::View) || has(page, Artifact::Native)
    };
    let mut wants = Vec::new();
    for spread in [requested, shown] {
        for page in spread.seats(mode).into_iter().flatten() {
            if errored(page) {
                continue;
            }
            if still {
                if !readable(page) {
                    push(&mut wants, page, Artifact::View);
                }
                if !has(page, Artifact::Thumb) {
                    push(&mut wants, page, Artifact::Thumb);
                }
                if shown.contains(page)
                    && upscaled(page)
                    && !has(page, Artifact::Native)
                {
                    push(&mut wants, page, Artifact::Native);
                }
                continue;
            }
            if !has(page, Artifact::Thumb) {
                push(&mut wants, page, Artifact::Thumb);
            }
        }
    }
    wants
}

/// What a spread with no textures at all is missing. Used when a
/// worker starts, before the first frame can measure the placement.
fn cold_wants(spread: Spread, mode: ViewMode) -> Vec<Want> {
    wants(
        spread,
        spread,
        mode,
        true,
        |_| false,
        |_, _| false,
        |_| false,
    )
}

fn push(wants: &mut Vec<Want>, page: usize, kind: Artifact) {
    let want = Want { page, kind };
    if !wants.contains(&want) {
        wants.push(want);
    }
}

/// Whether the frame keeps `page`'s `kind` resident. Must cover every
/// [`wants`] entry: a want may only leave the sent set while the app
/// still holds the artifact, or the worker would never learn it is
/// missing again.
fn kept(
    page: usize,
    kind: Artifact,
    shown: Spread,
    requested: Spread,
    mode: ViewMode,
    len: PageCount,
) -> bool {
    match kind {
        Artifact::Thumb => true,
        Artifact::View => view_kept(page, shown, requested, mode, len),
        Artifact::Native => shown.contains(page),
    }
}

/// Whether a view of `page` sits close enough to the presented or the
/// requested spread to be worth keeping for a likely turn. Two spread
/// starts covers a spread and its neighbours in every mode, including
/// the head's third spread at the start of a book (where no backward
/// neighbour exists). A view outside this window is dropped and asked
/// for again if a turn needs it.
fn view_kept(
    page: usize,
    shown: Spread,
    requested: Spread,
    mode: ViewMode,
    len: PageCount,
) -> bool {
    let start = view::spread_for(page, mode, len).first;
    start.abs_diff(shown.first) <= 2 || start.abs_diff(requested.first) <= 2
}

/// The spread a frame presents.
///
/// The requested spread wins once any of its pages clears the caller's
/// bar (a view or native while still, any artifact while skimming).
/// Otherwise
/// the frame steps through the presentable spreads whose start lies
/// between the shown and the requested one, nearest to the request
/// first, so a cold skim advances monotonically and never replays a page
/// the reader already passed. With nothing eligible, the shown spread
/// stays up rather than blanking. A cold start has no interval to
/// respect. `forward` only breaks the distance tie a cold start can
/// produce.
fn choose_presented(
    requested: Spread,
    shown: Option<Spread>,
    mode: ViewMode,
    len: PageCount,
    facts: &BTreeMap<usize, Fact>,
    forward: bool,
    live: impl Fn(usize) -> bool,
) -> Option<Spread> {
    let presentable = |spread: Spread| {
        spread
            .seats(mode)
            .into_iter()
            .flatten()
            .any(&live)
    };
    if presentable(requested) {
        return Some(requested);
    }
    let (low, high) = shown.map_or((0, usize::MAX), |shown| {
        (
            shown.first.min(requested.first),
            shown.first.max(requested.first),
        )
    });
    let mut best: Option<Spread> = None;
    for &page in facts.keys() {
        let spread = view::spread_for(page, mode, len);
        if spread.first < low
            || spread.first > high
            || !presentable(spread)
        {
            continue;
        }
        let distance = spread.first.abs_diff(requested.first);
        let closer = best.is_none_or(|current| {
            let current_distance = current.first.abs_diff(requested.first);
            distance < current_distance
                || (distance == current_distance
                    && if forward {
                        spread.first > current.first
                    } else {
                        spread.first < current.first
                    })
        });
        if closer {
            best = Some(spread);
        }
    }
    best.or(shown)
}

#[expect(
    clippy::rest_pattern_accessible_field,
    reason = "the ignored fields are unused; naming them would trip unneeded_field_pattern"
)]
impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // SAFETY: `exiting` sets `self.gl = None`, dropping the window and
        // its GL objects while the event loop's display is still alive.
        let window = match unsafe { Window::new(event_loop) } {
            Ok(window) => window,
            Err(error) => {
                self.fail(error, event_loop);
                return;
            }
        };
        let gl = window.glow();
        let pages = match PageTextures::new(Arc::clone(&gl)) {
            Ok(pages) => pages,
            Err(error) => {
                self.fail(error, event_loop);
                return;
            }
        };
        // `EguiGlow::new` panics if its painter cannot initialize; accepted upstream risk.
        let egui = egui_glow::EguiGlow::new(
            event_loop,
            Arc::clone(&gl),
            None,
            None,
            true,
        );
        // System fonts cover filename characters the built-in fonts cannot render.
        egui.egui_ctx.add_font_provider(Arc::new(
            egui_system_font_provider::SystemFontProvider::new(),
        ));
        {
            use parking_lot::Mutex;

            let proxy = Mutex::new(self.proxy.clone());
            egui.egui_ctx.set_request_repaint_callback(move |info| {
                post(&proxy.lock(), UserEvent::Repaint(info.delay));
            });
        }
        if window.translucent() {
            egui.egui_ctx.all_styles_mut(|style| {
                style.visuals.panel_fill = glass_fill();
            });
        }
        window.winit().request_redraw();
        self.gl = Some(Gl {
            pages,
            egui,
            window,
        });
    }

    fn user_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        event: UserEvent,
    ) {
        match event {
            UserEvent::Repaint(delay) => {
                let Some(deadline) = Instant::now().checked_add(delay)
                else {
                    return;
                };
                if let ControlFlow::WaitUntil(armed) =
                    event_loop.control_flow()
                {
                    if armed <= deadline {
                        return;
                    }
                }
                event_loop
                    .set_control_flow(ControlFlow::WaitUntil(deadline));
            }
            UserEvent::Worker(event) => {
                match event {
                    worker::Event::Ready(ready) => {
                        self.queue.push_back(ready);
                    }
                    worker::Event::Failed { page, error } => {
                        self.errors.insert(page, error);
                    }
                }
                self.request_redraw();
            }
        }
    }

    fn new_events(
        &mut self,
        _event_loop: &ActiveEventLoop,
        cause: StartCause,
    ) {
        if let StartCause::ResumeTimeReached { .. } = cause {
            self.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Quit must not wait out an in-flight decode; Drop would join it.
        self.worker.detach();
        self.gl = None;
    }

    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "winit's unhandled WindowEvent variants need no work"
    )]
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        use winit::event::ElementState;
        use winit::event::MouseButton;
        use winit::event::MouseScrollDelta;
        use winit::event::WindowEvent;
        use winit::keyboard::PhysicalKey;

        if matches!(
            event,
            WindowEvent::CloseRequested | WindowEvent::Destroyed
        ) {
            event_loop.exit();
        }
        let Some(gl) = self.gl.as_mut() else { return };

        let egui_wants_pointer =
            if matches!(event, WindowEvent::RedrawRequested) {
                // RedrawRequested carries nothing for egui, and egui-winit
                // answers it with `repaint: true`; honoring that answer would
                // re-request the redraw and self-sustain the loop at
                // compositor refresh while idle.
                gl.egui.egui_ctx.egui_wants_pointer_input()
            } else {
                let response =
                    gl.egui.on_window_event(gl.window.winit(), &event);
                let wants = gl.egui.egui_ctx.egui_wants_pointer_input();
                if response.repaint {
                    gl.window.winit().request_redraw();
                }
                wants
            };

        match &event {
            WindowEvent::Resized(_) => {
                // The refit happens in `frame()` when the viewport notices the new size.
                self.sync_surface_size();
                self.request_redraw();
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
            }
            WindowEvent::KeyboardInput { event: key, .. } => {
                if key.state == ElementState::Pressed
                    && !gl.egui.egui_ctx.egui_wants_keyboard_input()
                {
                    if let PhysicalKey::Code(code) = key.physical_key {
                        if let Some(action) =
                            Action::from_key(code, self.modifiers)
                        {
                            if !key.repeat || action.repeats() {
                                self.dispatch(action, key.repeat, event_loop);
                            }
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. }
                if !egui_wants_pointer =>
            {
                let wheel_delta_y = match delta {
                    MouseScrollDelta::LineDelta(_, lines) => *lines,
                    MouseScrollDelta::PixelDelta(pixels) => wheel_lines(
                        pixels.y,
                        gl.window.winit().scale_factor(),
                    ),
                };
                let (total, viewport) = self.geometry();
                // Scroll up (positive dy) zooms in.
                let zoom_factor =
                    (wheel_delta_y * WHEEL_ZOOM_SENSITIVITY).exp();
                self.fit.zoom(zoom_factor, self.cursor, total, viewport);
                self.request_redraw();
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging =
                    *state == ElementState::Pressed && !egui_wants_pointer;
            }
            WindowEvent::CursorMoved { position, .. } => {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "cursor positions are physical px, where f32 keeps subpixel precision"
                )]
                let cursor_pos = (position.x as f32, position.y as f32);
                if self.dragging && !egui_wants_pointer {
                    let (total, viewport) = self.geometry();
                    let pan_delta = (
                        cursor_pos.0 - self.cursor.0,
                        cursor_pos.1 - self.cursor.1,
                    );
                    self.fit.pan(pan_delta, total, viewport);
                    self.request_redraw();
                }
                self.cursor = cursor_pos;
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.frame() {
                    self.fail(error, event_loop);
                    return;
                }
                // Park until the next input or wake, so idle costs nothing.
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            _ => {}
        }
    }
}

/// Window background color, straight channels.
const BACKGROUND: [f32; 3] = [0.08, 0.08, 0.09];

const GLASS_ALPHA: f32 = 0.7;

/// Framebuffer clear color, premultiplied because Wayland composites
/// ARGB buffers that way.
fn clear_color(translucent: bool) -> [f32; 4] {
    let [red, green, blue] = BACKGROUND;
    if translucent {
        [
            red * GLASS_ALPHA,
            green * GLASS_ALPHA,
            blue * GLASS_ALPHA,
            GLASS_ALPHA,
        ]
    } else {
        [red, green, blue, 1.0]
    }
}

/// The page viewport from the chrome egui just laid out: the full-width
/// window area above the status bar, in physical px.
fn content_viewport(
    window: PhysicalSize<u32>,
    content: egui::Rect,
    pixels_per_point: f32,
) -> Viewport {
    let win = view::px_f32((window.width, window.height));
    // The bar's top edge rounds to a whole physical px, matching the
    // painter's clip-rect rounding.
    let bar_top = (content.max.y * pixels_per_point).round();
    Viewport::above_bottom_bar(win, bar_top)
}

/// The glass in egui's u8 color: a conversion of [`clear_color`], never a
/// second premultiplication.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the clear color's channels lie in 0..=1, so the rounded scaled value is a valid u8"
)]
fn glass_fill() -> egui::Color32 {
    let [red, green, blue, alpha] = clear_color(true);
    let channel = |value: f32| value.mul_add(255.0, 0.5) as u8;
    egui::Color32::from_rgba_premultiplied(
        channel(red),
        channel(green),
        channel(blue),
        channel(alpha),
    )
}

/// One wheel notch zooms ~×1.16, three notches ~×1.6.
const WHEEL_ZOOM_SENSITIVITY: f32 = 0.15;

/// Assumed wheel resolution for pixel-mode wheels, in logical px of
/// scroll per line.
const PIXELS_PER_WHEEL_LINE: f32 = 40.0;

/// Pixel-mode wheel motion as zoom lines, divided by scale factor so
/// zoom speed does not vary with output scaling.
#[expect(
    clippy::cast_possible_truncation,
    reason = "wheel lines feed an exponential zoom sensitivity, so one f64-to-f32 rounding is immaterial"
)]
fn wheel_lines(dy: f64, scale_factor: f64) -> f32 {
    (dy / (f64::from(PIXELS_PER_WHEEL_LINE) * scale_factor)) as f32
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;

    #[test]
    fn clear_color_premultiplies_the_glass() {
        assert_eq!(clear_color(false), [0.08, 0.08, 0.09, 1.0]);
        let [red, green, blue, alpha] = clear_color(true);
        assert!((alpha - 0.7).abs() < 1e-6, "{alpha}");
        for (channel, want) in
            [(red, 0.056), (green, 0.056), (blue, 0.063)]
        {
            assert!((channel - want).abs() < 1e-6, "{channel} vs {want}");
        }
    }

    #[test]
    fn content_viewport_subtracts_the_bar_in_physical_px() {
        let content = egui::Rect::from_min_max(
            egui::Pos2::ZERO,
            egui::pos2(1280.0, 772.0),
        );
        let integer =
            content_viewport(PhysicalSize::new(1280, 800), content, 1.0);
        assert_eq!(integer.size(), (1280.0, 772.0));
        // Fractional output scale: 772 pt at 1.25 pixels per point leaves
        // 965 physical px of content in a 1000 px window.
        let fractional =
            content_viewport(PhysicalSize::new(1600, 1000), content, 1.25);
        assert_eq!(fractional.size(), (1600.0, 965.0));
    }

    #[test]
    fn pixel_wheel_zoom_speed_does_not_vary_with_output_scale_factor() {
        assert_eq!(wheel_lines(40.0, 1.0), 1.0);
        assert_eq!(wheel_lines(80.0, 2.0), 1.0);
        assert_eq!(wheel_lines(40.0, 2.0), 0.5);
    }

    #[test]
    fn retention_covers_every_want() {
        let len = PageCount::new(120).expect("the test book is not empty");
        let missing = |_: usize, _: Artifact| false;
        let nowhere = |_: usize| false;
        for mode in [ViewMode::Normal, ViewMode::Manga, ViewMode::Comic] {
            for index in (0..len.get()).step_by(7) {
                let requested = view::spread_for(index, mode, len);
                for anchor in [
                    0,
                    index.saturating_sub(4),
                    index,
                    (index + 4).min(len.get() - 1),
                ] {
                    let shown = view::spread_for(anchor, mode, len);
                    for still in [true, false] {
                        for want in wants(
                            requested,
                            shown,
                            mode,
                            still,
                            |_| true,
                            missing,
                            nowhere,
                        ) {
                            assert!(
                                kept(
                                    want.page, want.kind, shown,
                                    requested, mode, len
                                ),
                                "{want:?} is wanted but {shown:?} / \
                                 {requested:?} would drop it in {mode:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn presentation_advances_through_presentable_spreads() {
        let len = PageCount::new(12).expect("the test book is not empty");
        let mode = ViewMode::Normal;
        let fact = Fact {
            source: (2, 2),
            format: Format::Png,
        };
        let facts: BTreeMap<usize, Fact> =
            [0, 2, 4, 8].into_iter().map(|page| (page, fact)).collect();
        let live = |page: usize| matches!(page, 0 | 2 | 4);
        let lone = Spread::lone;

        // The request wins as soon as it has a pixel.
        assert_eq!(
            choose_presented(
                lone(4),
                Some(lone(0)),
                mode,
                len,
                &facts,
                true,
                live
            ),
            Some(lone(4))
        );
        // Otherwise the nearest presentable spread between the shown
        // and the requested one; page 8 has a fact but no pixels.
        assert_eq!(
            choose_presented(
                lone(9),
                Some(lone(0)),
                mode,
                len,
                &facts,
                true,
                live
            ),
            Some(lone(4))
        );
        // Nothing between the two: the shown spread keeps its pixels.
        assert_eq!(
            choose_presented(
                lone(9),
                Some(lone(2)),
                mode,
                len,
                &facts,
                true,
                |page| page == 0
            ),
            Some(lone(2))
        );
        // A cold start picks the nearest presentable spread.
        assert_eq!(
            choose_presented(lone(5), None, mode, len, &facts, true, live),
            Some(lone(4))
        );
        // A cold start with nothing decoded presents nothing.
        assert_eq!(
            choose_presented(
                lone(5),
                None,
                mode,
                len,
                &facts,
                true,
                |_| false
            ),
            None
        );
    }
}
