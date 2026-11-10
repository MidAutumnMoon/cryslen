//! App: the windowed parent process. Parses the command line, runs the
//! event loop, owns the reading state and input, and composes frames.

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::ffi::OsString;
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
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

use crate::animation::DecodedEvent;
use crate::animation::Due;
use crate::animation::PlaybackCommand;
use crate::animation::PlaybackId;
use crate::animation::Player;
use crate::book::Book;
use crate::book::BookChange;
use crate::book::BookControl;
use crate::book::BookProcess;
use crate::book::BookUpdate;
use crate::book::BookView;
use crate::book::Delivery;
use crate::book::Wake;
use crate::browse;
use crate::browse::Direction;
use crate::browse::Motion;
use crate::browse::PageInfo;
use crate::browse::ReadingMode;
use crate::browse::ReadingPosition;
use crate::browse::Request;
use crate::browse::Role;
use crate::browse::Roles;
use crate::browse::Spread;
use crate::cache::presentation;
use crate::cache::presentation::FrameInput;
use crate::catalog;
use crate::decode;
use crate::fault;
use crate::fault::PageFault;
use crate::geometry;
use crate::geometry::FitMode;
use crate::geometry::Viewport;
use crate::protocol::ParentMsg;
use crate::render::PageTextures;
use crate::render::Window;
use crate::ui;
use crate::ui::PageStatus;
use crate::ui::StatusLine;

pub enum WakeEvent {
    /// Arms a wake; `new_events` requests the redraw at the deadline.
    Repaint(Duration),
    /// A book's reader thread forwarded an event, or a switch landed;
    /// the loop must draw a frame to drain it.
    Wake,
}

fn min_deadline(deadline: Option<Duration>, delay: Duration) -> Duration {
    deadline.map_or(delay, |current| current.min(delay))
}

pub fn post(proxy: &EventLoopProxy<WakeEvent>, event: WakeEvent) {
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

/// One image always goes through, so a big page stalls at most one frame.
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

/// `Refused` is sticky for the book, so a decoder that already stopped
/// is not retried on every return.
enum PagePlayback {
    Playing(Player),
    Refused,
}

/// Everything that resets when a book is adopted.
struct Reading {
    position: ReadingPosition,
    facts: BTreeMap<usize, PageInfo>,
    /// Page faults, sticky per page so a blip cannot erase them.
    faults: BTreeMap<usize, PageFault>,
    deliveries: VecDeque<Delivery>,
    upload_fault_page: Option<usize>,
    motion: Motion,
    held_keys: HeldKeys,
    last_turn: Option<Instant>,
    hold_since: Option<Instant>,
    hold_logged: bool,
    sent_requests: Vec<Request>,
    /// Reading direction of the last turn: the child's prefetch and skim
    /// tie-breaks follow it.
    direction: Direction,
    playback: BTreeMap<usize, PagePlayback>,
    next_generation: u32,
}

impl Reading {
    /// The first frame is a frame away, so compute the cold requests up
    /// front: the child serves the visible page before it starts the prefetch.
    fn new(
        view: &BookView,
        mode: ReadingMode,
        index: usize,
    ) -> (Self, Vec<Request>) {
        let cold = presentation::cold_requests(
            browse::spread_for(index, mode, view.len()),
            mode,
        );
        let reading = Self {
            position: ReadingPosition::new(index),
            facts: BTreeMap::new(),
            faults: BTreeMap::new(),
            deliveries: VecDeque::new(),
            upload_fault_page: None,
            motion: Motion::Still,
            held_keys: HeldKeys::default(),
            last_turn: None,
            hold_since: None,
            hold_logged: false,
            sent_requests: cold.clone(),
            direction: Direction::Forward,
            playback: BTreeMap::new(),
            next_generation: 0,
        };
        (reading, cold)
    }

    /// Start a book: build its reading state, then send the view before the
    /// cold requests so the child has a prefetch to serve.
    fn start(book: &Book, mode: ReadingMode, index: usize) -> Self {
        let (reading, cold) = Self::new(book.view(), mode, index);
        book.send(ParentMsg::SetView {
            page: index,
            mode,
            direction: Direction::Forward,
        });
        book.send(ParentMsg::SetRequests(cold));
        reading
    }
}

pub struct App {
    /// The book and its lifecycle; it outlives a dead child until a switch
    /// lands.
    book: Book,
    gl: Option<Gl>,
    fatal_error: Option<eros::ErrorUnion>,
    mode: ReadingMode,
    /// The fit mode survives a book switch; its placement does not.
    fit: geometry::Fit,
    reading: Reading,
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
    proxy: EventLoopProxy<WakeEvent>,
}

impl App {
    pub fn new(
        view: BookView,
        process: BookProcess,
        index: usize,
        proxy: EventLoopProxy<WakeEvent>,
    ) -> Self {
        let mode = ReadingMode::Normal;
        let wake: Wake = Arc::new({
            let proxy = proxy.clone();
            move || post(&proxy, WakeEvent::Wake)
        });
        let book = Book::new(view, process, wake);
        let reading = Reading::start(&book, mode, index);
        Self {
            book,
            gl: None,
            fatal_error: None,
            mode,
            fit: geometry::Fit::default(),
            reading,
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

    fn roles(&self, page: usize) -> Roles {
        self.gl
            .as_ref()
            .map_or_else(Roles::default, |gl| gl.pages.roles(page))
    }

    fn role_name(&self, page: Option<usize>) -> &'static str {
        let Some(page) = page else {
            return "empty";
        };
        self.roles(page).best().map_or("none", Role::name)
    }

    fn target_spread(&self) -> Spread {
        browse::spread_for(
            self.reading.position.target(),
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
        self.drain_deliveries();
        let drained = frame_started.elapsed();
        let now = Instant::now();

        let len = self.book.view().len();
        let target = browse::spread_for(
            self.reading.position.target(),
            self.mode,
            len,
        );

        let (view_width, view_height) = self.last_viewport.size();
        let measurable = view_width > 0.0 && view_height > 0.0;
        let frame = presentation::build_frame(
            &FrameInput {
                target,
                shown: self.reading.position.shown_with_mode(),
                mode: self.mode,
                motion: self.reading.motion,
                direction: self.reading.direction,
                len,
                facts: &self.reading.facts,
                faults: &self.reading.faults,
                measurable,
            },
            |page| self.roles(page),
            |on_screen| {
                let total =
                    geometry::layout(self.seat_dims_for(on_screen)).total();
                self.fit.scale_for(total, self.last_viewport)
            },
        );
        let shown = frame.shown();
        let changed = shown != self.reading.position.shown();
        if changed {
            let seats = shown.map(|spread| {
                spread
                    .seats(self.mode)
                    .map(|seat| self.role_name(seat))
            });
            let took = self
                .reading
                .last_turn
                .map(|last| now.saturating_duration_since(last));
            tracing::debug!(
                from = ?self.reading.position.shown(),
                to = ?shown,
                motion = ?self.reading.motion,
                ?seats,
                ?took,
                "shown"
            );
            self.fit.commit();
        }
        // Record every frame so a mode switch retags an unchanged spread.
        self.reading
            .position
            .record_shown(self.mode, shown);
        let on_screen = frame.on_screen();
        // The layout must describe the spread the frame shows; a stale
        // layout clamps one edge.
        let layout = geometry::layout(self.seat_dims_for(on_screen));
        let spread_size = layout.total();
        let placed = self.fit.resolve(spread_size, self.last_viewport);
        if changed {
            tracing::debug!(
                ?on_screen,
                ?shown,
                total = ?spread_size,
                dims = ?self.seat_dims_for(on_screen),
                scale = placed.scale,
                "placement"
            );
        }
        if self.reading.motion == Motion::Still
            && shown != Some(target)
        {
            let since = *self.reading.hold_since.get_or_insert(now);
            let waited = now.saturating_duration_since(since);
            if !self.reading.hold_logged
                && waited >= Duration::from_millis(250)
            {
                self.reading.hold_logged = true;
                tracing::info!(
                    target = ?target,
                    shown = ?shown,
                    waited = ?waited,
                    deliveries = self.reading.deliveries.len(),
                    "holding for a fit image"
                );
            }
        } else {
            self.reading.hold_since = None;
            self.reading.hold_logged = false;
        }
        if let Some(gl) = self.gl.as_mut() {
            // Retention is a performance policy: a dropped role is just
            // a request on the next frame.
            gl.pages.retain(|page, role| frame.retains(page, role));
        }
        self.sync_playback(frame.playback_targets());
        if let Some(delay) = self.tick_playback(now) {
            post(&self.proxy, WakeEvent::Repaint(delay));
        }
        // The texture maps are the authority: recompute the requests each
        // frame and send only changes. A zero-sized viewport says nothing
        // about what the screen resolves.
        let demand = frame.demand();
        let requests = frame.into_requests();
        if requests != self.reading.sent_requests {
            tracing::debug!(?requests, ?demand, "sent requests");
            self.send(ParentMsg::SetRequests(requests.clone()));
            self.reading.sent_requests = requests;
        }

        let faults = &self.reading.faults;
        let facts = &self.reading.facts;
        let pages = self.gl.as_ref().map(|gl| &gl.pages);
        let spread_status = ui::summarize_spread(on_screen, |page| {
            let info = facts
                .get(&page)
                .filter(|_| {
                    pages.is_some_and(|pages| pages.has_any(page))
                })
                .copied();
            (info, faults.get(&page))
        });
        if let Some(delay) = self.timers.tick_wait(now, spread_status.waiting) {
            post(&self.proxy, WakeEvent::Repaint(delay));
        }
        if let Some(delay) = self.timers.tick_toast(now) {
            post(&self.proxy, WakeEvent::Repaint(delay));
        }

        let page = spread_status.page.map(|(page, info)| PageStatus {
            info,
            claimed: self
                .book
                .view()
                .name(page)
                .and_then(decode::format_from_name),
        });
        let upload_fault = self
            .reading
            .upload_fault_page
            .and_then(|failed| self.reading.faults.get(&failed));
        let status = StatusLine {
            page_count: self.book.view().len().get(),
            spread: on_screen,
            mode: self.mode,
            toast: self.timers.toasting(),
            name: self.book.view().page_name(spread_status.named_page),
            page,
            zoom: placed.scale,
            waiting: self
                .timers
                .wait_word(now, || self.book.view().unpacking),
            fault: ui::fault_to_show(
                spread_status.fault,
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
        if let Some(spread) = self.reading.position.shown() {
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
                deliveries = self.reading.deliveries.len(),
                shown = ?self.reading.position.shown(),
                "slow frame"
            );
        }
        Ok(())
    }

    /// Upload pending deliveries under this frame's budget. A non-empty
    /// queue re-arms the loop immediately, so a backlog keeps draining.
    fn drain_deliveries(&mut self) {
        let Some(gl) = self.gl.as_mut() else {
            return;
        };
        let started = Instant::now();
        let mut uploaded = 0_usize;
        let mut bytes = 0_usize;
        while let Some(cost) = self
            .reading
            .deliveries
            .front()
            .map(|delivery| delivery.image.resident_bytes())
        {
            let over_budget = uploaded >= UPLOAD_BUDGET_COUNT
                || bytes.saturating_add(cost) > UPLOAD_BUDGET_BYTES;
            if uploaded > 0 && over_budget {
                break;
            }
            let Some(delivery) = self.reading.deliveries.pop_front() else {
                break;
            };
            let dims = delivery.image.page_dims();
            let format = delivery.image.format();
            match self.reading.facts.entry(delivery.page) {
                Entry::Vacant(entry) => {
                    entry
                        .insert(PageInfo { dims, format });
                }
                Entry::Occupied(entry) => debug_assert_eq!(
                    (entry.get().dims, entry.get().format),
                    (dims, format),
                    "page facts are fixed at the first upload"
                ),
            }
            let upload_started = Instant::now();
            let refused =
                gl.pages.insert(delivery.page, delivery.role, &delivery.image);
            let elapsed = upload_started.elapsed();
            tracing::debug!(
                page = delivery.page,
                role = ?delivery.role,
                width = delivery.image.width(),
                height = delivery.image.height(),
                elapsed = ?elapsed,
                "uploaded"
            );
            if elapsed >= Duration::from_millis(30) {
                tracing::info!(
                    page = delivery.page,
                    role = ?delivery.role,
                    width = delivery.image.width(),
                    height = delivery.image.height(),
                    elapsed = ?elapsed,
                    "slow upload"
                );
            }
            if let Err(error) = refused {
                self.reading
                    .faults
                    .insert(delivery.page, PageFault::from(error.widen()));
                self.reading.upload_fault_page = Some(delivery.page);
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
        if !self.reading.deliveries.is_empty() {
            post(&self.proxy, WakeEvent::Repaint(Duration::ZERO));
        }
    }

    /// A player follows the role its seat draws; a role change restarts it.
    fn sync_playback(&mut self, targets: [Option<(usize, Role)>; 2]) {
        let mut stop = Vec::new();
        for (&page, state) in &self.reading.playback {
            let keep = match state {
                PagePlayback::Refused => true,
                PagePlayback::Playing(player) => {
                    targets.iter().flatten().any(|&(want_page, want_role)| {
                        want_page == page && want_role == player.role()
                    })
                }
            };
            if !keep {
                stop.push(page);
            }
        }
        for page in stop {
            self.stop_playback(page);
        }
        for &(page, role) in targets.iter().flatten() {
            if !self.reading.playback.contains_key(&page) {
                self.start_playback(page, role);
            }
        }
    }

    fn start_playback(&mut self, page: usize, role: Role) {
        let id = PlaybackId {
            page,
            generation: self.reading.next_generation,
        };
        self.reading.next_generation = id.generation.wrapping_add(1);
        self.send(ParentMsg::Playback(PlaybackCommand::Start { id, role }));
        self.reading.playback.insert(
            page,
            PagePlayback::Playing(Player::new(id, role)),
        );
        tracing::debug!(?id, ?role, "playback player start");
    }

    fn stop_playback(&mut self, page: usize) {
        let Some(PagePlayback::Playing(player)) = self.reading.playback.get(&page)
        else {
            return;
        };
        let id = player.id();
        let role = player.role();
        self.reading.playback.remove(&page);
        self.send(ParentMsg::Playback(PlaybackCommand::Stop { id }));
        tracing::debug!(?id, ?role, "playback player stop");
    }

    fn refuse_playback(&mut self, page: usize) {
        self.stop_playback(page);
        self.reading.playback.insert(page, PagePlayback::Refused);
    }

    fn is_current_playback(&self, id: PlaybackId) -> bool {
        matches!(
            self.reading.playback.get(&id.page),
            Some(PagePlayback::Playing(player)) if player.id() == id
        )
    }

    fn player_mut(&mut self, id: PlaybackId) -> Option<&mut Player> {
        match self.reading.playback.get_mut(&id.page) {
            Some(PagePlayback::Playing(player)) if player.id() == id => {
                Some(player)
            }
            _ => {
                tracing::debug!(?id, "stale playback update");
                None
            }
        }
    }

    /// Returns the earliest wake the players need; `None` when no
    /// repaint is needed.
    fn tick_playback(&mut self, now: Instant) -> Option<Duration> {
        let mut deadline: Option<Duration> = None;
        let mut failed = Vec::new();
        if let Some(gl) = self.gl.as_mut() {
            for (&page, state) in &mut self.reading.playback {
                let PagePlayback::Playing(player) = state else {
                    continue;
                };
                match player.tick(now) {
                    Due::Show { image, delay } => {
                        let role = player.role();
                        if let Err(error) =
                            gl.pages.update(page, role, &image)
                        {
                            tracing::warn!(
                                page,
                                ?role,
                                %error,
                                "playback frame upload failed"
                            );
                            failed.push(page);
                            continue;
                        }
                        tracing::debug!(
                            page,
                            ?role,
                            ?delay,
                            "playback frame shown"
                        );
                        deadline = Some(min_deadline(deadline, delay));
                    }
                    Due::Wait(delay) => {
                        deadline = Some(min_deadline(deadline, delay));
                    }
                    Due::Idle => {}
                }
                if player.take_next() {
                    self.book.send(ParentMsg::Playback(
                        PlaybackCommand::Next { id: player.id() },
                    ));
                }
            }
        }
        for page in failed {
            self.refuse_playback(page);
        }
        deadline
    }

    fn last_shown_spread(&self) -> Spread {
        self.reading
            .position
            .shown()
            .unwrap_or_else(|| self.target_spread())
    }

    fn seat_dims_for(&self, spread: Spread) -> [Option<(u32, u32)>; 2] {
        spread.seats(self.mode).map(|seat| {
            let page = seat?;
            self.reading.facts.get(&page).map(|entry| entry.dims)
        })
    }

    fn sync_motion(&mut self) {
        let motion = if self.reading.held_keys.skimming() {
            Motion::Skimming
        } else {
            Motion::Still
        };
        if motion == self.reading.motion {
            return;
        }
        if motion == Motion::Still {
            // A turn may not have reached the screen; commit the shown
            // spread so its thumb promotes in place.
            let from = self.reading.position.target();
            if self.reading.position.commit_shown(self.mode) {
                tracing::debug!(
                    from,
                    to = self.reading.position.target(),
                    "commit shown"
                );
                self.set_view();
            }
        }
        tracing::debug!(
            ?motion,
            target = self.reading.position.target(),
            shown = ?self.reading.position.shown(),
            "motion"
        );
        self.reading.motion = motion;
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
            geometry::layout(self.seat_dims_for(self.last_shown_spread())).total(),
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
                if self.reading.motion == Motion::Skimming {
                    // The new pairing is a different coordinate system;
                    // commit the shown spread before switching.
                    let from = self.reading.position.target();
                    if self.reading.position.commit_shown(self.mode)
                    {
                        tracing::debug!(
                            from,
                            to = self.reading.position.target(),
                            "mode switch commits shown"
                        );
                    }
                }
                let switch = browse::switch_mode(
                    self.mode,
                    mode,
                    self.reading.position.target(),
                    self.book.view().len(),
                );
                self.mode = mode;
                self.reading.position.set_target(switch.index);
                self.reading.direction = Direction::Forward;
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
                    browse::first_spread()
                } else {
                    browse::last_spread(self.mode, self.book.view().len())
                };
                self.reading.position.set_target(index);
                let direction = if first {
                    Direction::Forward
                } else {
                    Direction::Backward
                };
                self.after_page_change(direction);
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
                let Some(direction) = action.direction(self.mode) else {
                    return;
                };
                let next = browse::advance(
                    self.reading.position.target(),
                    direction,
                    self.mode,
                    self.book.view().len(),
                );
                self.reading.position.set_target(next);
                self.after_page_change(direction);
            }
        }
    }

    /// A switch runs off the event loop; the old book's view and textures
    /// stay on screen until the new one lands.
    fn apply_book_change(&mut self, change: BookChange) {
        match change {
            BookChange::None => {}
            BookChange::Adopted => {
                let index = browse::first_spread();
                self.reading =
                    Reading::start(&self.book, self.mode, index);
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
                BookUpdate::Ready(delivery) => {
                    self.reading.deliveries.push_back(delivery);
                }
                BookUpdate::Failed { page, fault } => {
                    self.reading.faults.insert(page, fault);
                }
                BookUpdate::Playback(event) => match event {
                    DecodedEvent::Frame { id, delay, image } => {
                        if let Some(player) = self.player_mut(id) {
                            if !player.on_frame(delay, image) {
                                tracing::debug!(
                                    ?id,
                                    "unrequested playback frame"
                                );
                            }
                        }
                    }
                    DecodedEvent::Finished { id } => {
                        if let Some(player) = self.player_mut(id) {
                            if player.on_finished() {
                                tracing::debug!(?id, "playback finished");
                            }
                        }
                    }
                    DecodedEvent::Failed { id } => {
                        if self.is_current_playback(id) {
                            tracing::warn!(?id, "playback failed");
                            self.refuse_playback(id.page);
                        }
                    }
                },
                BookUpdate::Unpacked => self.book.mark_unpacked(),
                BookUpdate::Died(detail) => {
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
            page: self.reading.position.target(),
            mode: self.mode,
            direction: self.reading.direction,
        });
    }

    fn after_page_change(&mut self, direction: Direction) {
        self.reading.direction = direction;
        self.set_view();
        self.fit.begin_turn();
        self.reading.last_turn = Some(Instant::now());
        tracing::debug!(
            index = self.reading.position.target(),
            ?direction,
            motion = ?self.reading.motion,
            "turn"
        );
        self.request_redraw();
    }
}

#[expect(
    clippy::rest_pattern_accessible_field,
    reason = "the ignored fields are unused; naming them would trip unneeded_field_pattern"
)]
impl ApplicationHandler<WakeEvent> for App {
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
                post(&proxy.lock(), WakeEvent::Repaint(info.delay));
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
        event: WakeEvent,
    ) {
        match event {
            WakeEvent::Repaint(delay) => {
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
            WakeEvent::Wake => self.request_redraw(),
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
                            self.reading.held_keys.press(code, key.repeat);
                            self.sync_motion();
                        }
                        self.dispatch(action, event_loop);
                    }
                    ElementState::Released => {
                        self.reading.held_keys.release(code);
                        self.sync_motion();
                    }
                    ElementState::Pressed => {}
                }
            }
            WindowEvent::Focused(false) => {
                self.reading.held_keys.clear();
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
    let win = geometry::px_f32((window.width, window.height));
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Quit,
    /// Leave fullscreen when fullscreen, quit otherwise.
    Escape,
    SelectMode(ReadingMode),
    ToggleFullscreen,
    ResetView,
    SelectFit(FitMode),
    /// The book's first spread.
    First,
    /// The book's last spread.
    Last,
    /// The next sibling book in the parent directory.
    NextBook,
    /// The previous sibling book in the parent directory.
    PrevBook,
    /// Page one spread toward the screen right.
    Right,
    /// Page one spread toward the screen left.
    Left,
    /// Page one spread forward in reading order.
    Forward,
    /// Page one spread backward in reading order.
    Back,
}

impl Action {
    /// Bindings answer only to bare keys, `Shift+Space` and `Shift+B`
    /// excepted; every other modifier combo (anything the desktop owns)
    /// stays with the desktop.
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "winit's unmapped KeyCode variants map to no action"
    )]
    pub fn from_key(
        code: KeyCode,
        modifiers: ModifiersState,
    ) -> Option<Self> {
        let bare_shift = modifiers.shift_key()
            && !modifiers.control_key()
            && !modifiers.alt_key()
            && !modifiers.super_key();
        if bare_shift {
            return match code {
                KeyCode::Space => Some(Self::Back),
                KeyCode::KeyB => Some(Self::PrevBook),
                _ => None,
            };
        }
        if !modifiers.is_empty() {
            return None;
        }
        #[expect(
            clippy::wildcard_enum_match_arm,
            reason = "winit's unmapped KeyCode variants map to no action"
        )]
        match code {
            KeyCode::KeyQ => Some(Self::Quit),
            KeyCode::Escape => Some(Self::Escape),
            KeyCode::KeyN => Some(Self::SelectMode(ReadingMode::Normal)),
            KeyCode::KeyM => Some(Self::SelectMode(ReadingMode::Manga)),
            KeyCode::KeyC => Some(Self::SelectMode(ReadingMode::Comic)),
            KeyCode::KeyF => Some(Self::ToggleFullscreen),
            KeyCode::KeyR => Some(Self::ResetView),
            KeyCode::KeyW => Some(Self::SelectFit(FitMode::Width)),
            KeyCode::KeyH => Some(Self::SelectFit(FitMode::Height)),
            KeyCode::Digit1 => Some(Self::SelectFit(FitMode::Actual)),
            KeyCode::KeyB => Some(Self::NextBook),
            KeyCode::Home => Some(Self::First),
            KeyCode::End => Some(Self::Last),
            KeyCode::ArrowRight => Some(Self::Right),
            KeyCode::ArrowLeft => Some(Self::Left),
            KeyCode::Space | KeyCode::PageDown => Some(Self::Forward),
            KeyCode::PageUp => Some(Self::Back),
            _ => None,
        }
    }

    /// Whether a held key keeps firing this action.
    pub fn repeats(self) -> bool {
        matches!(
            self,
            Self::Right | Self::Left | Self::Forward | Self::Back
        )
    }

    /// The reading direction this action advances in `mode`, if it turns
    /// pages. Manga flips the arrows (Left advances). The reading-order
    /// keys never flip.
    pub fn direction(self, mode: ReadingMode) -> Option<Direction> {
        match self {
            Self::Forward => Some(Direction::Forward),
            Self::Right => Some(if mode.rtl() {
                Direction::Backward
            } else {
                Direction::Forward
            }),
            Self::Left => Some(if mode.rtl() {
                Direction::Forward
            } else {
                Direction::Backward
            }),
            Self::Back => Some(Direction::Backward),
            Self::Quit
            | Self::Escape
            | Self::SelectMode(_)
            | Self::ToggleFullscreen
            | Self::ResetView
            | Self::SelectFit(_)
            | Self::First
            | Self::Last
            | Self::NextBook
            | Self::PrevBook => None,
        }
    }
}

#[cfg(test)]
mod input_tests {
    use super::*;

    #[test]
    fn key_map() {
        let bindings: &[(KeyCode, Option<Action>)] = &[
            (KeyCode::Escape, Some(Action::Escape)),
            (KeyCode::KeyQ, Some(Action::Quit)),
            (KeyCode::KeyN, Some(Action::SelectMode(ReadingMode::Normal))),
            (KeyCode::KeyM, Some(Action::SelectMode(ReadingMode::Manga))),
            (KeyCode::KeyC, Some(Action::SelectMode(ReadingMode::Comic))),
            (KeyCode::KeyF, Some(Action::ToggleFullscreen)),
            (KeyCode::KeyR, Some(Action::ResetView)),
            (KeyCode::KeyW, Some(Action::SelectFit(FitMode::Width))),
            (KeyCode::KeyH, Some(Action::SelectFit(FitMode::Height))),
            (KeyCode::Digit1, Some(Action::SelectFit(FitMode::Actual))),
            (KeyCode::KeyB, Some(Action::NextBook)),
            (KeyCode::ShiftLeft, None), // the Shift+Space pairing lives on Space
            (KeyCode::Home, Some(Action::First)),
            (KeyCode::End, Some(Action::Last)),
            (KeyCode::ArrowRight, Some(Action::Right)),
            (KeyCode::ArrowLeft, Some(Action::Left)),
            (KeyCode::Space, Some(Action::Forward)),
            (KeyCode::PageDown, Some(Action::Forward)),
            (KeyCode::PageUp, Some(Action::Back)),
            (KeyCode::KeyA, None),
            (KeyCode::KeyD, None),
            (KeyCode::ArrowUp, None),
        ];
        for (code, action) in bindings {
            assert_eq!(
                Action::from_key(*code, ModifiersState::empty()),
                *action,
                "{code:?}"
            );
        }
    }

    #[test]
    fn modifier_interactions() {
        let bindings: &[(KeyCode, ModifiersState, Option<Action>)] = &[
            (KeyCode::Space, ModifiersState::SHIFT, Some(Action::Back)),
            (KeyCode::KeyB, ModifiersState::SHIFT, Some(Action::PrevBook)),
            (KeyCode::KeyQ, ModifiersState::CONTROL, None),
            (KeyCode::KeyQ, ModifiersState::SUPER, None),
            (KeyCode::KeyC, ModifiersState::CONTROL, None),
            (KeyCode::Space, ModifiersState::CONTROL, None),
            (KeyCode::ArrowRight, ModifiersState::ALT, None),
            (
                KeyCode::Space,
                ModifiersState::SHIFT | ModifiersState::CONTROL,
                None,
            ),
            (KeyCode::PageDown, ModifiersState::SHIFT, None),
            (KeyCode::PageUp, ModifiersState::SHIFT, None),
        ];
        for (code, modifiers, action) in bindings {
            assert_eq!(
                Action::from_key(*code, *modifiers),
                *action,
                "{code:?} {modifiers:?}"
            );
        }
    }

    #[test]
    fn action_semantics() {
        assert!(Action::Right.repeats());
        assert!(Action::Left.repeats());
        assert!(Action::Forward.repeats());
        assert!(Action::Back.repeats());
        let one_shot = [
            Action::Quit,
            Action::Escape,
            Action::SelectMode(ReadingMode::Normal),
            Action::SelectMode(ReadingMode::Manga),
            Action::SelectMode(ReadingMode::Comic),
            Action::ToggleFullscreen,
            Action::ResetView,
            Action::SelectFit(FitMode::Page),
            Action::SelectFit(FitMode::Width),
            Action::SelectFit(FitMode::Height),
            Action::SelectFit(FitMode::Actual),
            Action::First,
            Action::Last,
            Action::NextBook,
            Action::PrevBook,
        ];
        for action in one_shot {
            assert!(!action.repeats(), "{action:?} is one-shot");
        }
        let forwards: &[(Action, ReadingMode, Direction)] = &[
            (Action::Right, ReadingMode::Normal, Direction::Forward),
            (Action::Left, ReadingMode::Normal, Direction::Backward),
            (Action::Forward, ReadingMode::Normal, Direction::Forward),
            (Action::Back, ReadingMode::Normal, Direction::Backward),
            (Action::Right, ReadingMode::Manga, Direction::Backward),
            (Action::Left, ReadingMode::Manga, Direction::Forward),
            (Action::Forward, ReadingMode::Manga, Direction::Forward),
            (Action::Back, ReadingMode::Manga, Direction::Backward),
            (Action::Right, ReadingMode::Comic, Direction::Forward),
            (Action::Left, ReadingMode::Comic, Direction::Backward),
            (Action::Forward, ReadingMode::Comic, Direction::Forward),
            (Action::Back, ReadingMode::Comic, Direction::Backward),
        ];
        for &(action, mode, want) in forwards {
            assert_eq!(
                action.direction(mode),
                Some(want),
                "{action:?} in {mode:?}"
            );
        }
    }
}

#[derive(Debug, PartialEq)]
enum Argument {
    Book(PathBuf),
    Image { dir: PathBuf, name: String },
}

#[expect(
    clippy::use_debug,
    reason = "the fatal exit prints the full error report"
)]
pub fn main(args: &[OsString]) -> ExitCode {
    let Some(argument) = args.get(1) else {
        eprintln!("usage: cy <dir | archive | image-file>");
        return ExitCode::from(2);
    };
    let open = match classify(Path::new(argument)) {
        Ok(open) => open,
        Err(error) => {
            eprintln!("cy: {error}");
            return ExitCode::FAILURE;
        }
    };
    let (open_path, land_on) = match &open {
        Argument::Book(path) => (path.as_path(), None),
        Argument::Image { dir, name } => {
            (dir.as_path(), Some(name.as_str()))
        }
    };

    match run(open_path, land_on) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // `RUST_BACKTRACE` arms the backtrace in the Debug report.
            eprintln!("cy: {error:?}");
            ExitCode::FAILURE
        }
    }
}

fn classify(path: &Path) -> Result<Argument, fault::OpenRefusal> {
    let Some(name) = path.file_name() else {
        return Ok(Argument::Book(path.to_path_buf()));
    };
    if !path.is_file() || catalog::is_archive(path) {
        return Ok(Argument::Book(path.to_path_buf()));
    }
    let name = name.to_string_lossy().into_owned();
    if decode::format_from_name(&name).is_none() {
        return Err(fault::OpenRefusal::Unsupported(path.to_path_buf()));
    }
    // Path::parent() yields Some("") for bare relative names ("pic.jpg").
    let dir = path
        .parent()
        .filter(|parent| !parent.is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(Argument::Image {
        dir: dir.to_path_buf(),
        name,
    })
}

fn run(open_path: &Path, land_on: Option<&str>) -> eros::Result<()> {
    use egui_winit::winit::event_loop::EventLoop;

    // The loop exists before the first open, so the book's reader thread
    // can wake it like any later book's does.
    let event_loop = EventLoop::<WakeEvent>::with_user_event().build()?;
    let wake: Wake = {
        let proxy = event_loop.create_proxy();
        Arc::new(move || post(&proxy, WakeEvent::Wake))
    };
    // The initial open is the one blocking open; every switch after it
    // runs off the event loop.
    let (view, process) = BookProcess::open(open_path, wake)?;
    let index = land_on.map_or(0, |name| {
        // Every page index below `len` has a name, so the indices survive
        // the filter.
        catalog::landing(
            (0..view.len().get()).filter_map(|page| view.name(page)),
            name,
        )
    });

    let mut app =
        App::new(view, process, index, event_loop.create_proxy());
    event_loop.run_app(&mut app)?;
    if let Some(error) = app.take_fatal_error() {
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod argv_tests {
    use super::*;
    use crate::testkit::TempDir;

    #[test]
    fn classify_argument_kinds() {
        // classify never reads the content, only the extension.
        let dir = TempDir::new("parent-arg");
        dir.write("page.png", b"x");
        dir.write("book.cbz", b"not really a zip");
        dir.write("notes.txt", b"not a page");

        let rows = [
            (
                dir.path().join("page.png"),
                Ok(Argument::Image {
                    dir: dir.path().to_path_buf(),
                    name: String::from("page.png"),
                }),
            ),
            (
                dir.path().join("book.cbz"),
                Ok(Argument::Book(dir.path().join("book.cbz"))),
            ),
            (
                dir.path().to_path_buf(),
                Ok(Argument::Book(dir.path().to_path_buf())),
            ),
            (
                dir.path().join("notes.txt"),
                Err(fault::OpenRefusal::Unsupported(
                    dir.path().join("notes.txt"),
                )),
            ),
        ];
        for (path, expected) in rows {
            assert_eq!(classify(&path), expected, "{}", path.display());
        }
    }
}
