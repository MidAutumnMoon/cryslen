//! App: winit event handling, state transitions, frame composition.

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::thread;
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

use crate::book::BookEvent;
use crate::book::BookProcess;
use crate::book::BookView;
use crate::book::PendingBookOpen;
use crate::book::Wake;
use crate::decode;
use crate::decode::Format;
use crate::fault::BookFault;
use crate::fault::PageFault;
use crate::keys::Action;
use crate::plan::Direction;
use crate::preview::Artifact;
use crate::preview::VIEW_EDGE;
use crate::protocol::ParentMsg;
use crate::render::PageTextures;
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

pub enum UserEvent {
    /// Arms a wake; `new_events` requests the redraw at the deadline.
    Repaint(Duration),
    /// A book's reader thread forwarded an event, or a switch landed;
    /// the loop must draw a frame to drain it.
    Wake,
}

/// A failed send means the loop is gone, leaving the event with no consumer.
pub fn post(proxy: &EventLoopProxy<UserEvent>, event: UserEvent) {
    let Ok(()) = proxy.send_event(event) else {
        return;
    };
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

/// A drained page's own dims and format, not its representation's.
#[derive(Clone, Copy)]
struct CachedPage {
    dims: (u32, u32),
    format: Format,
}

/// Two presses closer than this are a flip, not a read; a held key flips
/// by its repeat events.
const SKIM_GAP: Duration = Duration::from_millis(100);

/// How long after the last navigation a skim becomes a still read.
const SKIM_DECAY: Duration = Duration::from_millis(250);

/// Per-frame upload budget; one artifact always goes through, so a big
/// page stalls at most one frame.
const UPLOAD_BUDGET_COUNT: usize = 16;
const UPLOAD_BUDGET_BYTES: usize = 8 * 1024 * 1024;

/// Spread starts around the shown and requested spreads whose views
/// stay resident.
const VIEW_KEEP_SPREADS: usize = 2;

pub struct App {
    /// What the UI shows; it outlives a dead child until a switch lands.
    view: BookView,
    /// The live child, or `None` once it died or was killed.
    process: Option<BookProcess>,
    /// The one switch in flight; a switch while it runs is ignored.
    pending: Option<
        crossbeam_channel::Receiver<
            Result<(BookView, BookProcess), BookFault>,
        >,
    >,
    gl: Option<Gl>,
    fatal_error: Option<eros::ErrorUnion>,
    index: usize,
    mode: ViewMode,
    fit: view::Fit,
    /// Set only when every seat resolves; frames draw only this.
    presented: Option<Spread>,
    /// Page facts, written at the first artifact that leaves the queue.
    cached: BTreeMap<usize, CachedPage>,
    /// Page failures, sticky per page so a blip cannot erase them.
    errors: BTreeMap<usize, PageFault>,
    /// Decoded artifacts waiting for an upload, in arrival order.
    queue: VecDeque<worker::Ready>,
    /// The page whose texture the GPU last refused, held until a book
    /// switch; the failure itself lives in `errors`.
    upload_error: Option<usize>,
    /// Whether the worker produces skim thumbs or full views.
    motion: Motion,
    /// When the last navigation was dispatched, for the skim window.
    last_nav: Option<Instant>,
    /// When the still-frame "waiting for a view" hold began, and whether
    /// it was logged.
    hold_since: Option<Instant>,
    hold_logged: bool,
    /// The want set last sent to the worker.
    sent: Vec<Want>,
    /// Reading direction of the last turn: the child's plan and skim
    /// tie-breaks follow it.
    direction: Direction,
    /// A failed book switch, held until one succeeds.
    noted_error: Option<BookFault>,
    timers: ui::Timers,
    /// Cursor position in viewport px, same as window px.
    cursor: (f32, f32),
    dragging: bool,
    /// Held modifier state; winit 0.30 key events carry none, only
    /// `ModifiersChanged` does.
    modifiers: ModifiersState,
    /// The viewport the last frame painted into, read by input between frames.
    last_viewport: Viewport,
    /// Wakes the parked loop from egui, timed waits, and book readers,
    /// which all run outside the event loop's handlers.
    proxy: EventLoopProxy<UserEvent>,
}

impl App {
    pub fn new(
        view: BookView,
        process: BookProcess,
        index: usize,
        proxy: EventLoopProxy<UserEvent>,
    ) -> Self {
        let mode = ViewMode::Normal;
        let direction = Direction::Forward;
        // The first frame is a frame away; declare what a cold start
        // lacks now, so the child serves the visible page before it
        // starts the plan.
        let cold =
            cold_wants(view::spread_for(index, mode, view.len()), mode);
        process.send(ParentMsg::SetView {
            index,
            mode,
            direction,
        });
        process.send(ParentMsg::SetWants(cold.clone()));
        Self {
            view,
            process: Some(process),
            pending: None,
            gl: None,
            fatal_error: None,
            index,
            mode,
            fit: view::Fit::default(),
            presented: None,
            cached: BTreeMap::new(),
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
        }
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

    fn send(&self, message: ParentMsg) {
        if let Some(process) = self.process.as_ref() {
            process.send(message);
        }
    }

    fn kill_process(&mut self) {
        if let Some(mut process) = self.process.take() {
            process.kill();
        }
    }

    fn drain_events(&self) -> Vec<BookEvent> {
        self.process
            .as_ref()
            .map_or_else(Vec::new, BookProcess::drain)
    }

    fn spread(&self) -> Spread {
        view::spread_for(self.index, self.mode, self.view.len())
    }

    fn has_any(&self, page: usize) -> bool {
        self.gl.as_ref().is_some_and(|gl| gl.pages.has_any(page))
    }

    fn has(&self, page: usize, kind: Artifact) -> bool {
        self.gl.as_ref().is_some_and(|gl| gl.pages.has(page, kind))
    }

    /// The largest representation a seat can draw, for the presentation
    /// log.
    fn artifact_name(&self, page: Option<usize>) -> &'static str {
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

    /// A still frame presents only a view or native; a thumb is a motion
    /// artifact, so the previous spread stays up until the view arrives.
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

        self.poll_book_updates();
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
            self.view.len(),
            &self.cached,
            self.direction == Direction::Forward,
            |page| {
                self.errors.contains_key(&page)
                    || (self.cached.contains_key(&page)
                        && if skimming {
                            self.has_any(page)
                        } else {
                            self.readable(page)
                        })
            },
        );
        if chosen != self.presented {
            let seats = chosen.map(|spread| {
                spread
                    .seats(self.mode)
                    .map(|seat| self.artifact_name(seat))
            });
            let took = self
                .last_nav
                .map(|last| now.saturating_duration_since(last));
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
        // A long still-frame hold is the frozen-page report; log it once.
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

        let (mode, len) = (self.mode, self.view.len());
        if let Some(gl) = self.gl.as_mut() {
            // Retention is a performance policy: a dropped artifact is
            // just a want on the next frame.
            gl.pages.retain(|page, kind| {
                kept(page, kind, shown, requested, mode, len)
            });
        }
        // The texture maps are the authority, not the worker's ledger:
        // recompute the wants each frame and send only changes. A
        // zero-sized viewport says nothing about what the screen resolves.
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
                    && self.cached.get(&page).is_some_and(|entry| {
                        native_wanted(placed.scale, entry.dims)
                    })
            },
            |page, kind| self.has(page, kind),
            |page| self.errors.contains_key(&page),
        );
        debug_assert!(
            wants.iter().all(|want| kept(
                want.page, want.kind, shown, requested, mode, len
            )),
            "retention must cover every want; a want that never leaves \
             the sent set can never be re-served"
        );
        if wants != self.sent {
            tracing::debug!(?wants, "requested");
            self.send(ParentMsg::SetWants(wants.clone()));
            self.sent = wants;
        }

        let errors = &self.errors;
        let cached = &self.cached;
        let pages = self.gl.as_ref().map(|gl| &gl.pages);
        let folded = ui::fold_status(shown, |page| {
            let entry = cached.get(&page);
            // `waiting` means the spread has no pixels yet, so a page
            // counts as decoded only once its texture is cached.
            ui::PageState {
                format: entry.and_then(|entry| {
                    pages
                        .is_some_and(|pages| pages.has_any(page))
                        .then_some(entry.format)
                }),
                dims: entry.map(|entry| entry.dims),
                error: errors.get(&page),
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
                .view
                .name(page)
                .and_then(decode::format_from_name),
            decoded,
            dims,
        });
        let upload = self
            .upload_error
            .and_then(|failed| self.errors.get(&failed));
        let status = StatusLine {
            page_count: self.view.len().get(),
            spread: shown,
            mode: self.mode,
            toast: self.timers.toasting(),
            name: self.view.page_name(folded.speaker),
            page,
            zoom: placed.scale,
            waiting: self.timers.wait_word(now, || self.view.unpacking),
            error: ui::fault_to_show(
                folded.error,
                self.noted_error.as_ref(),
                upload,
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
        if let Some(presented) = self.presented {
            gl.pages.draw(
                (size.width, size.height),
                viewport,
                &layout,
                presented.seats(self.mode),
                &drawn,
            );
        }
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

    /// Upload the queued artifacts under this frame's budget. A queue with
    /// more left re-arms the loop immediately, so a backlog keeps
    /// draining.
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
            let dims = ready.image.page_dims();
            let format = ready.image.format();
            match self.cached.entry(ready.page) {
                Entry::Vacant(entry) => {
                    entry.insert(CachedPage { dims, format });
                }
                Entry::Occupied(entry) => debug_assert_eq!(
                    (entry.get().dims, entry.get().format),
                    (dims, format),
                    "page facts are fixed at the first upload"
                ),
            }
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
                self.errors
                    .insert(ready.page, PageFault::from(error.widen()));
                self.upload_error = Some(ready.page);
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

    fn seat_dims(&self) -> [Option<(u32, u32)>; 2] {
        let shown = self.presented.unwrap_or_else(|| self.spread());
        shown.seats(self.mode).map(|seat| {
            let page = seat?;
            self.cached.get(&page).map(|entry| entry.dims)
        })
    }

    /// One flip starts a skim; every nav opens a fresh decay window.
    fn note_navigation(&mut self, now: Instant, repeat: bool) {
        let gap = self
            .last_nav
            .map(|last| now.saturating_duration_since(last));
        let flip = repeat || gap.is_some_and(|gap| gap < SKIM_GAP);
        if flip && self.motion == Motion::Still {
            tracing::debug!(?gap, repeat, "skim begins");
            self.motion = Motion::Skimming;
            self.send(ParentMsg::SetMotion(Motion::Skimming));
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
        self.send(ParentMsg::SetMotion(Motion::Still));
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
                    self.view.len(),
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
                // The view carries the seating order and the spread, so
                // both kinds of switch retune; only a pairing change
                // restarts the fit.
                self.set_view();
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
                    view::last_spread(self.mode, self.view.len())
                };
                self.after_page_change(travel_direction(first), repeat);
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
                    self.view.len(),
                );
                self.after_page_change(travel_direction(forward), repeat);
            }
        }
    }

    /// Start the next sibling book without blocking the frame. The old
    /// child dies first, so its archive memory and its queued events
    /// cannot outlive the switch; the old view and its textures stay on
    /// screen until the new book lands.
    fn switch_book(&mut self, forward: bool) {
        if self.pending.is_some() {
            return;
        }
        let Some(path) = self.view.sibling(forward) else {
            return;
        };
        tracing::debug!(path = %path.display(), forward, "book switch");
        self.kill_process();
        // The fork happens here, on the loop's thread: the child's
        // parent-death signal follows the thread that forked it.
        let wake: Wake = {
            let proxy = self.proxy.clone();
            Box::new(move || post(&proxy, UserEvent::Wake))
        };
        let opened = match PendingBookOpen::begin(&path, wake) {
            Ok(opened) => opened,
            Err(error) => {
                self.noted_error = Some(error);
                self.request_redraw();
                return;
            }
        };
        let (sender, result) = crossbeam_channel::bounded(1);
        let proxy = self.proxy.clone();
        let spawned = thread::Builder::new()
            .name(String::from("cy-book-open"))
            .spawn(move || {
                // A receiver that is gone means the app is gone too.
                if sender.send(opened.finish()).is_ok() {
                    post(&proxy, UserEvent::Wake);
                }
            });
        match spawned {
            Ok(_detached) => {
                self.pending = Some(result);
            }
            Err(error) => {
                // A failed spawn drops the pending open, which kills the
                // child that was just forked for it.
                self.noted_error =
                    Some(BookFault::Spawn(error.to_string()));
                self.request_redraw();
            }
        }
    }

    /// Put a freshly opened book on screen. Page numbering restarts, so
    /// no artifact, fault, or want of the old book may survive.
    fn adopt_book(&mut self, view: BookView, process: BookProcess) {
        let index = view::first_spread();
        let cold = cold_wants(
            view::spread_for(index, self.mode, view.len()),
            self.mode,
        );
        process.send(ParentMsg::SetView {
            index,
            mode: self.mode,
            direction: Direction::Forward,
        });
        process.send(ParentMsg::SetWants(cold.clone()));
        tracing::debug!(
            path = %view.path.display(),
            count = view.len().get(),
            "book adopted"
        );
        self.view = view;
        // Assigning drops the old process, whose `Drop` kills it.
        self.process = Some(process);
        self.index = index;
        self.direction = Direction::Forward;
        self.sent = cold;
        self.presented = None;
        self.cached.clear();
        self.errors.clear();
        self.queue.clear();
        self.upload_error = None;
        self.motion = Motion::Still;
        self.last_nav = None;
        self.hold_since = None;
        self.hold_logged = false;
        self.fit.refit();
        self.noted_error = None;
        if let Some(gl) = self.gl.as_mut() {
            gl.pages.clear_all();
        }
    }

    /// Drain the child's events and adopt a finished switch. A killed
    /// child's receiver went with it, so nothing here can be a stale
    /// event of a book that already left the screen.
    fn poll_book_updates(&mut self) {
        for event in self.drain_events() {
            match event {
                BookEvent::Ready(ready) => self.queue.push_back(ready),
                BookEvent::Failed { page, fault } => {
                    self.errors.insert(page, fault);
                }
                BookEvent::Unpacked => self.view.unpacking = false,
                BookEvent::Died(detail) => {
                    tracing::warn!(%detail, "the book process failed");
                    // The process is gone; taking it reaps it and joins
                    // the reader that just reported the death.
                    self.kill_process();
                    self.noted_error
                        .get_or_insert(BookFault::Died(detail));
                }
            }
        }
        let Some(pending) = self.pending.take() else {
            return;
        };
        let opened = match pending.try_recv() {
            Ok(opened) => opened,
            Err(crossbeam_channel::TryRecvError::Empty) => {
                self.pending = Some(pending);
                return;
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                self.noted_error.get_or_insert_with(|| {
                    BookFault::Died(String::from(
                        "the book switch ended without a book",
                    ))
                });
                return;
            }
        };
        match opened {
            Ok((view, process)) => self.adopt_book(view, process),
            Err(fault) => {
                tracing::warn!(?fault, "book switch failed");
                self.noted_error = Some(fault);
            }
        }
    }

    /// Retune the child: the view carries the spread, the seating order,
    /// and the prefetch direction.
    fn set_view(&self) {
        self.send(ParentMsg::SetView {
            index: self.index,
            mode: self.mode,
            direction: self.direction,
        });
    }

    /// Every spread mutation retunes the child, restarts the fit, and
    /// opens the skim window here.
    fn after_page_change(&mut self, direction: Direction, repeat: bool) {
        self.direction = direction;
        self.set_view();
        self.fit.begin_turn();
        self.note_navigation(Instant::now(), repeat);
        self.request_redraw();
    }
}

fn travel_direction(forward: bool) -> Direction {
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
fn native_wanted(scale: f32, page_dims: (u32, u32)) -> bool {
    let edge = page_dims.0.max(page_dims.1);
    edge > 0 && scale > VIEW_EDGE as f32 / edge as f32
}

/// Every artifact the frame's presentation is missing, in service order:
/// requested spread, then shown. A still frame wants views and thumbs; a
/// skim wants thumbs; a native only for a page on screen.
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
                    push_want(&mut wants, page, Artifact::View);
                }
                if !has(page, Artifact::Thumb) {
                    push_want(&mut wants, page, Artifact::Thumb);
                }
                if shown.contains(page)
                    && upscaled(page)
                    && !has(page, Artifact::Native)
                {
                    push_want(&mut wants, page, Artifact::Native);
                }
                continue;
            }
            if !has(page, Artifact::Thumb) {
                push_want(&mut wants, page, Artifact::Thumb);
            }
        }
    }
    wants
}

/// What a spread with no textures is missing, used before the first
/// frame can measure the placement.
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

fn push_want(wants: &mut Vec<Want>, page: usize, kind: Artifact) {
    let want = Want { page, kind };
    if !wants.contains(&want) {
        wants.push(want);
    }
}

/// Whether the frame keeps `page`'s `kind` resident. Must cover every
/// [`wants`] entry, or the worker never learns the artifact is missing
/// again.
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

fn view_kept(
    page: usize,
    shown: Spread,
    requested: Spread,
    mode: ViewMode,
    len: PageCount,
) -> bool {
    let start = view::spread_for(page, mode, len).first;
    start.abs_diff(shown.first) <= VIEW_KEEP_SPREADS
        || start.abs_diff(requested.first) <= VIEW_KEEP_SPREADS
}

/// The requested spread once every page clears `can_show`; otherwise the
/// nearest presentable start between the shown and requested spreads, so
/// a skim never replays a passed page; otherwise the shown spread.
fn choose_presented(
    requested: Spread,
    shown: Option<Spread>,
    mode: ViewMode,
    len: PageCount,
    cached: &BTreeMap<usize, CachedPage>,
    forward: bool,
    can_show: impl Fn(usize) -> bool,
) -> Option<Spread> {
    let presentable = |spread: Spread| {
        spread.seats(mode).into_iter().flatten().all(&can_show)
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
    for &page in cached.keys() {
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
            UserEvent::Wake => self.request_redraw(),
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
        // Quit must not wait out an in-flight decode; the child dies, and
        // a pending open's book dies with its own result if it lands.
        self.kill_process();
        self.pending = None;
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
                                self.dispatch(
                                    action, key.repeat, event_loop,
                                );
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
        let entry = CachedPage {
            dims: (2, 2),
            format: Format::Png,
        };
        let cached: BTreeMap<usize, CachedPage> =
            [0, 2, 4, 8].into_iter().map(|page| (page, entry)).collect();
        let can_show = |page: usize| matches!(page, 0 | 2 | 4);
        let lone = Spread::lone;

        // The request wins as soon as it has a pixel.
        assert_eq!(
            choose_presented(
                lone(4),
                Some(lone(0)),
                mode,
                len,
                &cached,
                true,
                can_show
            ),
            Some(lone(4))
        );
        // Otherwise the nearest presentable spread between the shown
        // and the requested one; page 8 is cached but has no pixels.
        assert_eq!(
            choose_presented(
                lone(9),
                Some(lone(0)),
                mode,
                len,
                &cached,
                true,
                can_show
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
                &cached,
                true,
                |page| page == 0
            ),
            Some(lone(2))
        );
        // A cold start picks the nearest presentable spread.
        assert_eq!(
            choose_presented(
                lone(5),
                None,
                mode,
                len,
                &cached,
                true,
                can_show
            ),
            Some(lone(4))
        );
        // A cold start with nothing decoded presents nothing.
        assert_eq!(
            choose_presented(
                lone(5),
                None,
                mode,
                len,
                &cached,
                true,
                |_| false
            ),
            None
        );
    }

    #[test]
    fn paired_spread_waits_for_every_seat() {
        let len = PageCount::new(12).expect("the test book is not empty");
        let mode = ViewMode::Manga;
        let entry = CachedPage {
            dims: (2, 2),
            format: Format::Png,
        };
        let cached: BTreeMap<usize, CachedPage> =
            [0, 2].into_iter().map(|page| (page, entry)).collect();
        let pair = Spread {
            first: 1,
            second: Some(2),
        };
        let shown = Spread::lone(0);

        assert_eq!(
            choose_presented(
                pair,
                Some(shown),
                mode,
                len,
                &cached,
                true,
                |page| page == 2
            ),
            Some(shown)
        );
        assert_eq!(
            choose_presented(
                pair,
                Some(shown),
                mode,
                len,
                &cached,
                true,
                |page| matches!(page, 1 | 2)
            ),
            Some(pair)
        );
    }
}
