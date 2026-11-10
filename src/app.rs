//! App: winit event handling, state transitions, frame composition.

use std::collections::btree_map::Entry;
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
use winit::keyboard::KeyCode;
use winit::keyboard::ModifiersState;
use winit::window::WindowId;

use crate::book::Book;
use crate::book::BookChange;
use crate::book::BookControl;
use crate::book::BookEvent;
use crate::book::BookProcess;
use crate::book::BookView;
use crate::book::Wake;
use crate::decode;
use crate::fault::PageFault;
use crate::keys::Action;
use crate::plan::Direction;
use crate::presentation;
use crate::presentation::FrameInput;
use crate::preview::Artifact;
use crate::preview::Artifacts;
use crate::protocol::ParentMsg;
use crate::render::PageTextures;
use crate::ui;
use crate::ui::PageFacts;
use crate::ui::StatusLine;
use crate::view;
use crate::view::FitMode;
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

/// One artifact always goes through, so a big page stalls at most one frame.
const UPLOAD_BUDGET_COUNT: usize = 16;
const UPLOAD_BUDGET_BYTES: usize = 8 * 1024 * 1024;

/// Page-turn keys currently down, and whether the platform has repeated
/// them.
#[derive(Default)]
struct HeldKeys {
    page_turn_keys: BTreeMap<KeyCode, bool>,
}

impl HeldKeys {
    fn press(&mut self, code: KeyCode, repeated: bool) {
        *self.page_turn_keys.entry(code).or_default() |= repeated;
    }

    fn release(&mut self, code: KeyCode) {
        self.page_turn_keys.remove(&code);
    }

    fn clear(&mut self) {
        self.page_turn_keys.clear();
    }

    fn skimming(&self) -> bool {
        self.page_turn_keys.values().any(|&repeated| repeated)
    }
}

/// Everything that resets when a book is adopted.
struct Session {
    navigation: presentation::Navigation,
    cached: BTreeMap<usize, presentation::CachedPage>,
    /// Page faults, sticky per page so a blip cannot erase them.
    faults: BTreeMap<usize, PageFault>,
    queue: VecDeque<worker::Ready>,
    upload_fault_page: Option<usize>,
    motion: Motion,
    held_keys: HeldKeys,
    last_nav: Option<Instant>,
    hold_since: Option<Instant>,
    hold_logged: bool,
    sent_wants: Vec<Want>,
    /// Reading direction of the last turn: the child's plan and skim
    /// tie-breaks follow it.
    direction: Direction,
}

impl Session {
    /// The first frame is a frame away, so compute the cold wants up
    /// front: the child serves the visible page before it starts the plan.
    fn new(
        view: &BookView,
        mode: ViewMode,
        index: usize,
    ) -> (Self, Vec<Want>) {
        let cold = presentation::cold_wants(
            view::spread_for(index, mode, view.len()),
            mode,
        );
        let session = Self {
            navigation: presentation::Navigation::new(index),
            cached: BTreeMap::new(),
            faults: BTreeMap::new(),
            queue: VecDeque::new(),
            upload_fault_page: None,
            motion: Motion::Still,
            held_keys: HeldKeys::default(),
            last_nav: None,
            hold_since: None,
            hold_logged: false,
            sent_wants: cold.clone(),
            direction: Direction::Forward,
        };
        (session, cold)
    }

    /// Start a book: build its session, then send the view before the
    /// cold wants so the child has a plan to serve.
    fn start(book: &Book, mode: ViewMode, index: usize) -> Self {
        let (session, cold) = Self::new(book.view(), mode, index);
        book.send(ParentMsg::SetView {
            index,
            mode,
            direction: Direction::Forward,
        });
        book.send(ParentMsg::SetWants(cold));
        session
    }
}

pub struct App {
    /// The book and its lifecycle; it outlives a dead child until a switch
    /// lands.
    book: Book,
    gl: Option<Gl>,
    fatal_error: Option<eros::ErrorUnion>,
    mode: ViewMode,
    /// The fit mode survives a book switch; its placement does not.
    fit: view::Fit,
    session: Session,
    timers: ui::Timers,
    /// Cursor position in viewport px, same as window px.
    cursor: (f32, f32),
    dragging: bool,
    /// Held modifier state; winit 0.30 key events carry none, only
    /// `ModifiersChanged` does.
    modifiers: ModifiersState,
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
        let wake: Wake = Arc::new({
            let proxy = proxy.clone();
            move || post(&proxy, UserEvent::Wake)
        });
        let book = Book::new(view, process, wake);
        let session = Session::start(&book, mode, index);
        Self {
            book,
            gl: None,
            fatal_error: None,
            mode,
            fit: view::Fit::default(),
            session,
            timers: ui::Timers::default(),
            cursor: (0.0, 0.0),
            dragging: false,
            modifiers: ModifiersState::empty(),
            last_viewport: Viewport::default(),
            proxy,
        }
    }

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
        self.book.send(message);
    }

    fn artifacts(&self, page: usize) -> Artifacts {
        self.gl
            .as_ref()
            .map_or_else(Artifacts::default, |gl| gl.pages.artifacts(page))
    }

    fn artifact_name(&self, page: Option<usize>) -> &'static str {
        let Some(page) = page else {
            return "empty";
        };
        self.artifacts(page).best().map_or("none", Artifact::name)
    }

    fn target_spread(&self) -> Spread {
        view::spread_for(
            self.session.navigation.target(),
            self.mode,
            self.book.view().len(),
        )
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

        let len = self.book.view().len();
        let target = view::spread_for(
            self.session.navigation.target(),
            self.mode,
            len,
        );

        let (view_width, view_height) = self.last_viewport.size();
        let measurable = view_width > 0.0 && view_height > 0.0;
        let plan = presentation::plan_frame(
            &FrameInput {
                target,
                presented: self.session.navigation.presented_with_mode(),
                mode: self.mode,
                motion: self.session.motion,
                direction: self.session.direction,
                len,
                facts: &self.session.cached,
                faults: &self.session.faults,
                measurable,
            },
            |page| self.artifacts(page),
            |anchor| {
                let total =
                    view::layout(self.seat_dims_for(anchor)).total();
                self.fit.scale_for(total, self.last_viewport)
            },
        );
        let presented = plan.presented();
        let changed = presented != self.session.navigation.presented();
        if changed {
            let seats = presented.map(|spread| {
                spread
                    .seats(self.mode)
                    .map(|seat| self.artifact_name(seat))
            });
            let took = self
                .session
                .last_nav
                .map(|last| now.saturating_duration_since(last));
            tracing::debug!(
                from = ?self.session.navigation.presented(),
                to = ?presented,
                motion = ?self.session.motion,
                ?seats,
                ?took,
                "presented"
            );
            self.fit.commit();
        }
        // Record every frame so a mode switch retags an unchanged spread.
        self.session
            .navigation
            .record_presented(self.mode, presented);
        let anchor = plan.anchor();
        // The layout must describe the anchor the plan presents; a stale
        // layout clamps one edge.
        let layout = view::layout(self.seat_dims_for(anchor));
        let spread_size = layout.total();
        let placed = self.fit.resolve(spread_size, self.last_viewport);
        if changed {
            tracing::debug!(
                ?anchor,
                ?presented,
                total = ?spread_size,
                dims = ?self.seat_dims_for(anchor),
                scale = placed.scale,
                "placement"
            );
        }
        if self.session.motion == Motion::Still
            && presented != Some(target)
        {
            let since = *self.session.hold_since.get_or_insert(now);
            let waited = now.saturating_duration_since(since);
            if !self.session.hold_logged
                && waited >= Duration::from_millis(250)
            {
                self.session.hold_logged = true;
                tracing::info!(
                    target = ?target,
                    presented = ?presented,
                    waited = ?waited,
                    queue = self.session.queue.len(),
                    "holding for a view"
                );
            }
        } else {
            self.session.hold_since = None;
            self.session.hold_logged = false;
        }
        if let Some(gl) = self.gl.as_mut() {
            // Retention is a performance policy: a dropped artifact is
            // just a want on the next frame.
            gl.pages.retain(|page, kind| plan.retains(page, kind));
        }
        // The texture maps are the authority: recompute the wants each frame
        // and send only changes. A zero-sized viewport says nothing about what
        // the screen resolves.
        let demand = plan.demand();
        let wants = plan.into_wants();
        if wants != self.session.sent_wants {
            tracing::debug!(?wants, ?demand, "sent wants");
            self.send(ParentMsg::SetWants(wants.clone()));
            self.session.sent_wants = wants;
        }

        let faults = &self.session.faults;
        let cached = &self.session.cached;
        let pages = self.gl.as_ref().map(|gl| &gl.pages);
        let folded = ui::fold_status(anchor, |page| {
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
                fault: faults.get(&page),
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
                .book
                .view()
                .name(page)
                .and_then(decode::format_from_name),
            decoded,
            dims,
        });
        let upload_fault = self
            .session
            .upload_fault_page
            .and_then(|failed| self.session.faults.get(&failed));
        let status = StatusLine {
            page_count: self.book.view().len().get(),
            spread: anchor,
            mode: self.mode,
            toast: self.timers.toasting(),
            name: self.book.view().page_name(folded.speaker),
            page,
            zoom: placed.scale,
            waiting: self
                .timers
                .wait_word(now, || self.book.view().unpacking),
            fault: ui::fault_to_show(
                folded.fault,
                self.book.fault(),
                upload_fault,
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
        if let Some(spread) = self.session.navigation.presented() {
            gl.pages.draw(
                (size.width, size.height),
                viewport,
                &layout,
                spread.seats(self.mode),
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
                queue = self.session.queue.len(),
                presented = ?self.session.navigation.presented(),
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
        while let Some(cost) = self
            .session
            .queue
            .front()
            .map(|ready| ready.image.resident_bytes())
        {
            let over_budget = uploaded >= UPLOAD_BUDGET_COUNT
                || bytes.saturating_add(cost) > UPLOAD_BUDGET_BYTES;
            if uploaded > 0 && over_budget {
                break;
            }
            let Some(ready) = self.session.queue.pop_front() else {
                break;
            };
            let dims = ready.image.page_dims();
            let format = ready.image.format();
            match self.session.cached.entry(ready.page) {
                Entry::Vacant(entry) => {
                    entry
                        .insert(presentation::CachedPage { dims, format });
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
                self.session
                    .faults
                    .insert(ready.page, PageFault::from(error.widen()));
                self.session.upload_fault_page = Some(ready.page);
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
        if !self.session.queue.is_empty() {
            post(&self.proxy, UserEvent::Repaint(Duration::ZERO));
        }
    }

    fn anchor(&self) -> Spread {
        self.session
            .navigation
            .presented()
            .unwrap_or_else(|| self.target_spread())
    }

    fn seat_dims_for(&self, spread: Spread) -> [Option<(u32, u32)>; 2] {
        spread.seats(self.mode).map(|seat| {
            let page = seat?;
            self.session.cached.get(&page).map(|entry| entry.dims)
        })
    }

    fn sync_motion(&mut self) {
        let motion = if self.session.held_keys.skimming() {
            Motion::Skimming
        } else {
            Motion::Still
        };
        if motion == self.session.motion {
            return;
        }
        if motion == Motion::Still {
            // A turn may not have reached the screen; commit the presented
            // spread so its thumb promotes in place.
            let from = self.session.navigation.target();
            if self.session.navigation.commit_presented(self.mode) {
                tracing::debug!(
                    from,
                    to = self.session.navigation.target(),
                    "commit presented"
                );
                self.set_view();
            }
        }
        tracing::debug!(
            ?motion,
            target = self.session.navigation.target(),
            presented = ?self.session.navigation.presented(),
            "motion"
        );
        self.session.motion = motion;
        self.send(ParentMsg::SetMotion(motion));
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
        (
            view::layout(self.seat_dims_for(self.anchor())).total(),
            self.last_viewport,
        )
    }

    fn dispatch(&mut self, action: Action, event_loop: &ActiveEventLoop) {
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
                if self.session.motion == Motion::Skimming {
                    // The new pairing is a different coordinate system;
                    // commit the presented spread before switching.
                    let from = self.session.navigation.target();
                    if self.session.navigation.commit_presented(self.mode)
                    {
                        tracing::debug!(
                            from,
                            to = self.session.navigation.target(),
                            "mode switch commits presented"
                        );
                    }
                }
                let switch = view::switch_mode(
                    self.mode,
                    mode,
                    self.session.navigation.target(),
                    self.book.view().len(),
                );
                self.mode = mode;
                self.session.navigation.set_target(switch.index);
                self.session.direction = Direction::Forward;
                self.timers.toast(Instant::now());
                if switch.pairing_changed {
                    // The fit begins the new pairing; a mode switch skips
                    // the skim.
                    self.fit.begin_turn();
                }
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
                let index = if first {
                    view::first_spread()
                } else {
                    view::last_spread(self.mode, self.book.view().len())
                };
                self.session.navigation.set_target(index);
                self.after_page_change(travel_direction(first));
            }
            Action::NextBook | Action::PrevBook => {
                let change = self.book.step(BookControl::Switch(
                    matches!(action, Action::NextBook),
                ));
                self.apply_book_change(change);
            }
            Action::Right
            | Action::Left
            | Action::Forward
            | Action::Back => {
                let forward = action.is_forward(self.mode);
                let next = view::advance(
                    self.session.navigation.target(),
                    forward,
                    self.mode,
                    self.book.view().len(),
                );
                self.session.navigation.set_target(next);
                self.after_page_change(travel_direction(forward));
            }
        }
    }

    /// A switch runs off the event loop; the old book's view and textures
    /// stay on screen until the new one lands.
    fn apply_book_change(&mut self, change: BookChange) {
        match change {
            BookChange::None => {}
            BookChange::Adopted => {
                let index = view::first_spread();
                self.session =
                    Session::start(&self.book, self.mode, index);
                self.fit.refit();
                if let Some(gl) = self.gl.as_mut() {
                    gl.pages.clear_all();
                }
                tracing::debug!(
                    path = %self.book.view().path.display(),
                    count = self.book.view().len().get(),
                    "book adopted"
                );
            }
            BookChange::Faulted => {
                self.request_redraw();
            }
        }
    }

    /// Drain the child's events and adopt a finished switch. A killed
    /// child's receiver went with it, so no stale event of a book that
    /// already left the screen can arrive here.
    fn poll_book_updates(&mut self) {
        for event in self.book.drain() {
            match event {
                BookEvent::Ready(ready) => {
                    self.session.queue.push_back(ready);
                }
                BookEvent::Failed { page, fault } => {
                    self.session.faults.insert(page, fault);
                }
                BookEvent::Unpacked => self.book.mark_unpacked(),
                BookEvent::Died(detail) => {
                    let change = self.book.step(BookControl::Died(detail));
                    self.apply_book_change(change);
                }
            }
        }
        if let Some(opened) = self.book.take_opened() {
            let change = self.book.step(BookControl::Opened(opened));
            self.apply_book_change(change);
        }
    }

    fn set_view(&self) {
        self.send(ParentMsg::SetView {
            index: self.session.navigation.target(),
            mode: self.mode,
            direction: self.session.direction,
        });
    }

    fn after_page_change(&mut self, direction: Direction) {
        self.session.direction = direction;
        self.set_view();
        self.fit.begin_turn();
        self.session.last_nav = Some(Instant::now());
        tracing::debug!(
            index = self.session.navigation.target(),
            ?direction,
            motion = ?self.session.motion,
            "turn"
        );
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
        // `EguiGlow::new` panics if its painter cannot initialize; accepted
        // upstream risk.
        let egui = egui_glow::EguiGlow::new(
            event_loop,
            Arc::clone(&gl),
            None,
            None,
            true,
        );
        // System fonts cover filename characters the built-in fonts cannot
        // render.
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
        // Quit kills the child, so an in-flight decode ends with it; a
        // pending open's book dies with its own result if it lands.
        self.book.shutdown();
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
                // egui-winit answers RedrawRequested with `repaint: true`;
                // honoring that answer would re-request the redraw and
                // self-sustain the loop at compositor refresh while idle.
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
                self.sync_surface_size();
                self.request_redraw();
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
            }
            WindowEvent::KeyboardInput { event: key, .. } => {
                let PhysicalKey::Code(code) = key.physical_key else {
                    return;
                };
                match key.state {
                    ElementState::Pressed
                        if !gl
                            .egui
                            .egui_ctx
                            .egui_wants_keyboard_input() =>
                    {
                        let Some(action) =
                            Action::from_key(code, self.modifiers)
                        else {
                            return;
                        };
                        if key.repeat && !action.repeats() {
                            return;
                        }
                        if action.repeats() {
                            self.session.held_keys.press(code, key.repeat);
                            self.sync_motion();
                        }
                        self.dispatch(action, event_loop);
                    }
                    ElementState::Released => {
                        self.session.held_keys.release(code);
                        self.sync_motion();
                    }
                    ElementState::Pressed => {}
                }
            }
            WindowEvent::Focused(false) => {
                self.session.held_keys.clear();
                self.sync_motion();
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

/// The glass in egui's premultiplied u8 color, converted from [`clear_color`].
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
/// zoom speed stays constant across output scaling.
#[expect(
    clippy::cast_possible_truncation,
    reason = "wheel lines feed an exponential zoom sensitivity, so one f64-to-f32 rounding is immaterial"
)]
fn wheel_lines(dy: f64, scale_factor: f64) -> f32 {
    (dy / (f64::from(PIXELS_PER_WHEEL_LINE) * scale_factor)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tap_stays_still_and_a_repeat_skims_until_release() {
        let mut held_keys = HeldKeys::default();
        held_keys.press(KeyCode::ArrowRight, false);
        assert!(!held_keys.skimming());
        held_keys.press(KeyCode::ArrowRight, true);
        assert!(held_keys.skimming());
        held_keys.release(KeyCode::ArrowRight);
        assert!(!held_keys.skimming());
    }

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
}
