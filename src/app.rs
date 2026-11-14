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

use egui::Color32;
use egui::Context;
use egui::Rect;
use egui::Response;
use egui_glow::EguiGlow;
use egui_winit::winit;
use eros::ErrorUnion;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::KeyEvent;
use winit::event::StartCause;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::ControlFlow;
use winit::event_loop::EventLoopProxy;
use winit::keyboard::KeyCode;
use winit::keyboard::ModifiersState;
use winit::window::WindowId;

use crate::animation::Due;
use crate::animation::PlaybackCommand;
use crate::animation::PlaybackEvent;
use crate::animation::PlaybackId;
use crate::animation::Player;
use crate::book::Book;
use crate::book::BookChange;
use crate::book::BookOpen;
use crate::book::BookUpdate;
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
use crate::cache::ImageLimits;
use crate::catalog;
use crate::catalog::PageName;
use crate::decode;
use crate::fault::BookFault;
use crate::fault::OpenRefusal;
use crate::fault::PageFault;
use crate::geometry;
use crate::geometry::Fit;
use crate::geometry::FitMode;
use crate::geometry::SpreadLayout;
use crate::geometry::Viewport;
use crate::protocol::ParentMsg;
use crate::protocol::WireImage;
use crate::render::PageTextures;
use crate::render::Window;
use crate::ui;
use crate::ui::PageStatus;
use crate::ui::ReadingStatus;
use crate::ui::StatusFault;
use crate::ui::StatusLine;
use crate::ui::Timers;

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
    limits: ImageLimits,
    egui: EguiGlow,
    // Must drop last: owns the GL context used by the fields above.
    window: Window,
}

fn display_limits(
    event_loop: &ActiveEventLoop,
    window: &Window,
    max_texture_edge: u32,
) -> ImageLimits {
    let monitor_edge = event_loop
        .available_monitors()
        .map(|monitor| {
            let size = monitor.size();
            size.width.max(size.height)
        })
        .max()
        .filter(|&edge| edge > 0);
    let window_size = window.winit().inner_size();
    let window_edge = window_size.width.max(window_size.height);
    let limits = ImageLimits::new(
        monitor_edge.unwrap_or(window_edge),
        max_texture_edge,
    );
    tracing::debug!(
        ?monitor_edge,
        window_edge,
        fit_edge = limits.fit_edge(),
        max_texture_edge = limits.max_texture_edge(),
        "display limits"
    );
    limits
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
    Playing(Player<Box<WireImage>>),
    Refused,
}

/// Everything that resets when a book is adopted.
struct Reading {
    position: ReadingPosition,
    facts: BTreeMap<usize, PageInfo>,
    /// Page faults, sticky per page so a blip cannot erase them.
    faults: BTreeMap<usize, PageFault>,
    deliveries: VecDeque<Delivery>,
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
    /// Digits typed in `:` mode; `None` while the prompt is closed.
    jump_digits: Option<String>,
}

impl Reading {
    fn new(index: usize) -> Self {
        Self {
            position: ReadingPosition::new(index),
            facts: BTreeMap::new(),
            faults: BTreeMap::new(),
            deliveries: VecDeque::new(),
            motion: Motion::Still,
            held_keys: HeldKeys::default(),
            last_turn: None,
            hold_since: None,
            hold_logged: false,
            sent_requests: Vec::new(),
            direction: Direction::Forward,
            playback: BTreeMap::new(),
            next_generation: 0,
            jump_digits: None,
        }
    }
}

fn page_number(digits: &str, page_count: usize) -> Option<usize> {
    digits
        .parse::<usize>()
        .ok()
        .filter(|page| (1..=page_count).contains(page))
}

struct LiveBook {
    book: Book,
    reading: Reading,
}

enum Session {
    Opening {
        open: BookOpen,
        name: PageName,
        land_on: Option<String>,
    },
    Faulted {
        name: PageName,
        fault: BookFault,
    },
    Reading(Box<LiveBook>),
    Closed,
}

struct SpreadDraw<'digits> {
    layout: SpreadLayout,
    size: (f32, f32),
    shown: Option<Spread>,
    jump_digits: Option<&'digits str>,
}

struct PreparedReading {
    reading: ReadingStatus,
    name: PageName,
    fault_page: Option<usize>,
    layout: SpreadLayout,
    spread_size: (f32, f32),
}

fn tick_timers(
    timers: &mut Timers,
    proxy: &EventLoopProxy<WakeEvent>,
    now: Instant,
    waiting: bool,
) {
    if let Some(delay) = timers.tick_wait(now, waiting) {
        post(proxy, WakeEvent::Repaint(delay));
    }
    if let Some(delay) = timers.tick_toast(now) {
        post(proxy, WakeEvent::Repaint(delay));
    }
}

fn adopt_reading(
    fit: &mut Fit,
    gl: Option<&mut Gl>,
    limits: ImageLimits,
    mode: ReadingMode,
    book: &Book,
    index: usize,
) -> Reading {
    fit.refit();
    if let Some(gl) = gl {
        gl.pages.clear_all();
    }
    let mut reading = Reading::new(index);
    let requests = presentation::cold_requests(
        browse::spread_for(index, mode, book.view().len()),
        mode,
    );
    book.send(ParentMsg::SetLimits(limits));
    book.send(ParentMsg::SetView {
        page: index,
        mode,
        direction: reading.direction,
    });
    book.send(ParentMsg::SetRequests(requests.clone()));
    reading.sent_requests = requests;
    tracing::debug!(
        path = %book.view().path().display(),
        count = book.view().len().get(),
        "book adopted"
    );
    reading
}

fn present_frame(
    gl: &mut Gl,
    fit: &mut Fit,
    last_viewport: &mut Viewport,
    size: PhysicalSize<u32>,
    status: &StatusLine<'_>,
    spread: Option<SpreadDraw<'_>>,
) -> eros::Result<()> {
    let show_status_bar = gl.window.winit().fullscreen().is_none();
    let jump = spread.as_ref().and_then(|draw| {
        let digits = draw.jump_digits?;
        let page_count = status.reading.as_ref()?.page_count;
        Some((
            digits,
            page_count,
            page_number(digits, page_count).is_some(),
        ))
    });

    // `EguiGlow::run` only records shapes, so egui lays out before pages
    // are placed; `paint` later draws the bar above them.
    let mut page_surface = None;
    gl.egui.run(gl.window.winit(), |ui| {
        let page_response = ui::show(ui, status, show_status_bar);
        if let Some((digits, page_count, valid)) = jump {
            ui::page_jump(
                ui.ctx(),
                page_response.rect,
                digits,
                page_count,
                valid,
            );
        }
        page_surface = Some(page_response);
    });
    let Some(page_surface) = page_surface else {
        return Ok(());
    };

    let viewport = content_viewport(
        size,
        page_surface.rect,
        page_surface.ctx.pixels_per_point(),
    );
    // The bar can change height without a resize; the status scale needs
    // a repaint.
    if viewport != *last_viewport {
        fit.refit();
        *last_viewport = viewport;
        gl.window.winit().request_redraw();
    }

    let drawn = spread.as_ref().map(|draw| {
        if jump.is_none()
            && apply_page_input(fit, &page_surface, draw.size, viewport)
        {
            gl.window.winit().request_redraw();
        }
        fit.resolve(draw.size, viewport)
    });

    gl.egui.painter.clear(
        [size.width, size.height],
        clear_color(gl.window.translucent()),
    );
    if let (Some(draw), Some(placement)) = (spread, drawn) {
        if let Some(shown) = draw.shown {
            gl.pages.draw(
                (size.width, size.height),
                viewport,
                &draw.layout,
                shown.seats(status.mode),
                &placement,
            );
        }
    }
    gl.egui.paint(gl.window.winit());
    gl.window.pre_present_and_swap()?;
    Ok(())
}

pub struct App {
    session: Session,
    gl: Option<Gl>,
    fatal_error: Option<ErrorUnion>,
    mode: ReadingMode,
    /// The fit mode survives a book switch; its placement does not.
    fit: Fit,
    timers: Timers,
    /// Held modifier state; winit 0.30 key events carry none, only
    /// `ModifiersChanged` does.
    modifiers: ModifiersState,
    last_viewport: Viewport,
    wake: Wake,
    /// Wakes the parked loop from egui, timed waits, and book readers,
    /// which all run outside the event loop's handlers.
    proxy: EventLoopProxy<WakeEvent>,
}

impl App {
    pub fn new(
        open_path: &Path,
        land_on: Option<&str>,
        proxy: EventLoopProxy<WakeEvent>,
    ) -> Self {
        let wake: Wake = {
            let proxy = proxy.clone();
            Arc::new(move || post(&proxy, WakeEvent::Wake))
        };
        let name = catalog::opening_name(open_path, land_on);
        let session = match BookOpen::start(open_path, Arc::clone(&wake)) {
            Ok(open) => Session::Opening {
                open,
                name,
                land_on: land_on.map(str::to_owned),
            },
            Err(fault) => {
                tracing::warn!(?fault, "book open failed");
                Session::Faulted { name, fault }
            }
        };
        Self {
            session,
            gl: None,
            fatal_error: None,
            mode: ReadingMode::Normal,
            fit: Fit::default(),
            timers: Timers::default(),
            modifiers: ModifiersState::empty(),
            last_viewport: Viewport::default(),
            wake,
            proxy,
        }
    }

    fn fail(&mut self, error: ErrorUnion, event_loop: &ActiveEventLoop) {
        self.fatal_error = Some(error);
        event_loop.exit();
    }

    pub fn take_fatal_error(&mut self) -> Option<ErrorUnion> {
        self.fatal_error.take()
    }

    fn live(&self) -> Option<&LiveBook> {
        let Session::Reading(live) = &self.session else {
            return None;
        };
        Some(live)
    }

    fn live_mut(&mut self) -> Option<&mut LiveBook> {
        let Session::Reading(live) = &mut self.session else {
            return None;
        };
        Some(live)
    }

    fn send(&self, message: ParentMsg) {
        if let Some(live) = self.live() {
            live.book.send(message);
        }
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

    fn is_fullscreen(&self) -> bool {
        self.gl
            .as_ref()
            .is_some_and(|gl| gl.window.winit().fullscreen().is_some())
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

    fn poll_initial_open(&mut self, limits: ImageLimits) {
        use std::mem;

        let Session::Opening { open, .. } = &self.session else {
            return;
        };
        let Some(opened) = open.try_take() else {
            return;
        };
        let Session::Opening { name, land_on, .. } =
            mem::replace(&mut self.session, Session::Closed)
        else {
            return;
        };
        match opened {
            Ok((view, process)) => {
                let index = land_on
                    .as_deref()
                    .map_or_else(browse::first_spread, |entry| {
                        view.landing(entry)
                    });
                let book =
                    Book::new(view, process, Arc::clone(&self.wake));
                let reading = adopt_reading(
                    &mut self.fit,
                    self.gl.as_mut(),
                    limits,
                    self.mode,
                    &book,
                    index,
                );
                self.session =
                    Session::Reading(Box::new(LiveBook { book, reading }));
            }
            Err(fault) => {
                tracing::warn!(?fault, "book open failed");
                self.session = Session::Faulted { name, fault };
            }
        }
    }

    fn frame(&mut self) -> eros::Result<()> {
        let frame_started = Instant::now();
        let Some(size) = self.sync_surface_size() else {
            return Ok(());
        };
        let Some(limits) = self.gl.as_ref().map(|gl| gl.limits) else {
            return Ok(());
        };
        self.poll_initial_open(limits);
        self.poll_book_updates(limits);
        self.drain_deliveries();
        let drained = frame_started.elapsed();
        let now = Instant::now();
        let prepared = self.prepare_reading(limits, now);

        let Self {
            session,
            gl,
            mode,
            fit,
            timers,
            last_viewport,
            proxy,
            ..
        } = self;
        let Some(gl) = gl.as_mut() else {
            return Ok(());
        };

        match session {
            Session::Closed => Ok(()),
            Session::Opening { name, .. } => {
                tick_timers(timers, proxy, now, true);
                let status = StatusLine {
                    reading: None,
                    mode: *mode,
                    toast: timers.toasting(),
                    name: name.clone(),
                    waiting: timers.waiting(now),
                    fault: None,
                };
                present_frame(gl, fit, last_viewport, size, &status, None)
            }
            Session::Faulted { name, fault } => {
                tick_timers(timers, proxy, now, false);
                let status = StatusLine {
                    reading: None,
                    mode: *mode,
                    toast: timers.toasting(),
                    name: name.clone(),
                    waiting: false,
                    fault: Some(StatusFault::Book(fault)),
                };
                present_frame(gl, fit, last_viewport, size, &status, None)
            }
            Session::Reading(live) => {
                let Some(prepared) = prepared else {
                    return Ok(());
                };
                let page_fault = prepared
                    .fault_page
                    .and_then(|page| live.reading.faults.get(&page));
                let status = StatusLine {
                    reading: Some(prepared.reading),
                    mode: *mode,
                    toast: timers.toasting(),
                    name: prepared.name,
                    waiting: timers.waiting(now),
                    fault: ui::fault_to_show(
                        page_fault,
                        live.book.fault(),
                    ),
                };
                let spread = SpreadDraw {
                    layout: prepared.layout,
                    size: prepared.spread_size,
                    shown: live.reading.position.shown(),
                    jump_digits: live.reading.jump_digits.as_deref(),
                };
                present_frame(
                    gl,
                    fit,
                    last_viewport,
                    size,
                    &status,
                    Some(spread),
                )?;
                let total = frame_started.elapsed();
                if total >= Duration::from_millis(50) {
                    tracing::info!(
                        total = ?total,
                        drain = ?drained,
                        deliveries = live.reading.deliveries.len(),
                        shown = ?live.reading.position.shown(),
                        "slow frame"
                    );
                }
                Ok(())
            }
        }
    }

    fn prepare_reading(
        &mut self,
        limits: ImageLimits,
        now: Instant,
    ) -> Option<PreparedReading> {
        let measurable = {
            let (view_width, view_height) = self.last_viewport.size();
            view_width > 0.0 && view_height > 0.0
        };
        let frame = {
            let live = self.live()?;
            let len = live.book.view().len();
            let target = browse::spread_for(
                live.reading.position.target(),
                self.mode,
                len,
            );
            presentation::build_frame(
                &FrameInput {
                    target,
                    shown: live.reading.position.shown_with_mode(),
                    mode: self.mode,
                    motion: live.reading.motion,
                    direction: live.reading.direction,
                    len,
                    limits,
                    facts: &live.reading.facts,
                    faults: &live.reading.faults,
                    measurable,
                },
                |page| self.roles(page),
                |on_screen| {
                    let total =
                        geometry::layout(self.seat_dims_for(on_screen))
                            .total();
                    self.fit.scale_for(total, self.last_viewport)
                },
            )
        };

        let shown = frame.shown();
        let previous_shown = self.live()?.reading.position.shown();
        let changed = shown != previous_shown;
        if changed {
            let seats = shown.map(|spread| {
                spread.seats(self.mode).map(|seat| self.role_name(seat))
            });
            let live = self.live()?;
            let took = live
                .reading
                .last_turn
                .map(|last| now.saturating_duration_since(last));
            tracing::debug!(
                from = ?previous_shown,
                to = ?shown,
                motion = ?live.reading.motion,
                ?seats,
                ?took,
                "shown"
            );
            self.fit.commit();
        }

        let mode = self.mode;
        if let Some(live) = self.live_mut() {
            // Record every frame so a mode switch retags an unchanged spread.
            live.reading.position.record_shown(mode, shown);
        }
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

        {
            let live = self.live_mut()?;
            let target = browse::spread_for(
                live.reading.position.target(),
                mode,
                live.book.view().len(),
            );
            if live.reading.motion == Motion::Still
                && shown != Some(target)
            {
                let since = *live.reading.hold_since.get_or_insert(now);
                let waited = now.saturating_duration_since(since);
                if !live.reading.hold_logged
                    && waited >= Duration::from_millis(250)
                {
                    live.reading.hold_logged = true;
                    tracing::info!(
                        target = ?target,
                        shown = ?shown,
                        waited = ?waited,
                        deliveries = live.reading.deliveries.len(),
                        "holding for a fit image"
                    );
                }
            } else {
                live.reading.hold_since = None;
                live.reading.hold_logged = false;
            }
        }

        if let Some(gl) = self.gl.as_mut() {
            // Retention is a performance policy: a dropped role is just a
            // request on the next frame.
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
        if let Some(live) = self.live_mut() {
            if requests != live.reading.sent_requests {
                tracing::debug!(?requests, ?demand, "sent requests");
                live.book.send(ParentMsg::SetRequests(requests.clone()));
                live.reading.sent_requests = requests;
            }
        }

        let live = self.live()?;
        let pages = self.gl.as_ref().map(|gl| &gl.pages);
        let spread_status = ui::summarize_spread(on_screen, |page| {
            let info = live
                .reading
                .facts
                .get(&page)
                .filter(|_| pages.is_some_and(|pages| pages.has_any(page)))
                .copied();
            (info, live.reading.faults.get(&page))
        });

        let page_count = live.book.view().len().get();
        let page = spread_status.page.map(|(page, info)| PageStatus {
            info,
            claimed: live
                .book
                .view()
                .name(page)
                .and_then(decode::format_from_name),
        });
        let name = live.book.view().page_name(spread_status.named_page);
        let fault_page =
            spread_status.fault.map(|_| spread_status.named_page);
        let waiting = spread_status.waiting;
        tick_timers(&mut self.timers, &self.proxy, now, waiting);
        Some(PreparedReading {
            reading: ReadingStatus {
                page_count,
                spread: on_screen,
                page,
                zoom: placed.scale,
            },
            name,
            fault_page,
            layout,
            spread_size,
        })
    }

    /// Upload pending deliveries under this frame's budget. A non-empty
    /// queue re-arms the loop immediately, so a backlog keeps draining.
    fn drain_deliveries(&mut self) {
        let Self {
            session, gl, proxy, ..
        } = self;
        let Some(gl) = gl.as_mut() else {
            return;
        };
        let Session::Reading(live) = session else {
            return;
        };
        let started = Instant::now();
        let mut uploaded = 0_usize;
        let mut bytes = 0_usize;
        while let Some(cost) = live
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
            let Some(delivery) = live.reading.deliveries.pop_front()
            else {
                break;
            };
            let dims = delivery.image.page_dims;
            let format = delivery.image.format;
            match live.reading.facts.entry(delivery.page) {
                Entry::Vacant(entry) => {
                    entry.insert(PageInfo { dims, format });
                }
                Entry::Occupied(entry) => debug_assert_eq!(
                    (entry.get().dims, entry.get().format),
                    (dims, format),
                    "page facts are fixed at the first upload"
                ),
            }
            let upload_started = Instant::now();
            match gl.pages.insert(
                delivery.page,
                delivery.role,
                &delivery.image,
            ) {
                Ok(()) => {
                    let elapsed = upload_started.elapsed();
                    tracing::debug!(
                        page = delivery.page,
                        role = ?delivery.role,
                        width = delivery.image.width,
                        height = delivery.image.height,
                        elapsed = ?elapsed,
                        "uploaded"
                    );
                    if elapsed >= Duration::from_millis(30) {
                        tracing::info!(
                            page = delivery.page,
                            role = ?delivery.role,
                            width = delivery.image.width,
                            height = delivery.image.height,
                            elapsed = ?elapsed,
                            "slow upload"
                        );
                    }
                    uploaded += 1;
                    bytes = bytes.saturating_add(cost);
                }
                Err(error) => {
                    tracing::warn!(
                        page = delivery.page,
                        role = ?delivery.role,
                        %error,
                        "texture upload refused"
                    );
                    if delivery.role != Role::Thumb
                        && !gl
                            .pages
                            .roles(delivery.page)
                            .has_settled_image()
                    {
                        live.reading
                            .faults
                            .insert(delivery.page, PageFault::from(error));
                    }
                }
            }
        }
        if uploaded > 0 {
            tracing::debug!(
                count = uploaded,
                bytes,
                elapsed = ?started.elapsed(),
                "drained"
            );
        }
        if !live.reading.deliveries.is_empty() {
            post(proxy, WakeEvent::Repaint(Duration::ZERO));
        }
    }

    /// A player follows the role its seat draws; a role change restarts it.
    fn sync_playback(&mut self, targets: [Option<(usize, Role)>; 2]) {
        let stop: Vec<usize> = {
            let Some(live) = self.live() else {
                return;
            };
            live.reading
                .playback
                .iter()
                .filter_map(|(&page, state)| {
                    let keep = match state {
                        PagePlayback::Refused => true,
                        PagePlayback::Playing(player) => targets
                            .iter()
                            .flatten()
                            .any(|&(want_page, want_role)| {
                                want_page == page
                                    && want_role == player.role()
                            }),
                    };
                    (!keep).then_some(page)
                })
                .collect()
        };
        for page in stop {
            self.stop_playback(page);
        }
        for &(page, role) in targets.iter().flatten() {
            let contains = self.live().is_some_and(|live| {
                live.reading.playback.contains_key(&page)
            });
            if !contains {
                self.start_playback(page, role);
            }
        }
    }

    fn start_playback(&mut self, page: usize, role: Role) {
        let Some(live) = self.live_mut() else {
            return;
        };
        let id = PlaybackId {
            page,
            generation: live.reading.next_generation,
        };
        live.reading.next_generation = id.generation.wrapping_add(1);
        live.book.send(ParentMsg::Playback(PlaybackCommand::Start {
            id,
            role,
        }));
        live.reading
            .playback
            .insert(page, PagePlayback::Playing(Player::new(id, role)));
        tracing::debug!(?id, ?role, "playback player start");
    }

    fn stop_playback(&mut self, page: usize) {
        let Some(live) = self.live_mut() else {
            return;
        };
        let Some(PagePlayback::Playing(player)) =
            live.reading.playback.get(&page)
        else {
            return;
        };
        let id = player.id();
        let role = player.role();
        live.reading.playback.remove(&page);
        live.book
            .send(ParentMsg::Playback(PlaybackCommand::Stop { id }));
        tracing::debug!(?id, ?role, "playback player stop");
    }

    fn refuse_playback(&mut self, page: usize) {
        self.stop_playback(page);
        if let Some(live) = self.live_mut() {
            live.reading.playback.insert(page, PagePlayback::Refused);
        }
    }

    fn is_current_playback(&self, id: PlaybackId) -> bool {
        self.live().is_some_and(|live| {
            matches!(
                live.reading.playback.get(&id.page),
                Some(PagePlayback::Playing(player)) if player.id() == id
            )
        })
    }

    fn player_mut(
        &mut self,
        id: PlaybackId,
    ) -> Option<&mut Player<Box<WireImage>>> {
        let live = self.live_mut()?;
        match live.reading.playback.get_mut(&id.page) {
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
        {
            let Self { session, gl, .. } = self;
            let (Some(gl), Session::Reading(live)) =
                (gl.as_mut(), session)
            else {
                return None;
            };
            for (&page, state) in &mut live.reading.playback {
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
                    live.book.send(ParentMsg::Playback(
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

    fn seat_dims_for(&self, spread: Spread) -> [Option<(u32, u32)>; 2] {
        spread.seats(self.mode).map(|seat| {
            let page = seat?;
            self.live()?
                .reading
                .facts
                .get(&page)
                .map(|entry| entry.dims)
        })
    }

    fn update_held_keys(&mut self, update: impl FnOnce(&mut HeldKeys)) {
        let Some(live) = self.live_mut() else {
            return;
        };
        update(&mut live.reading.held_keys);
        self.sync_motion();
    }

    fn sync_motion(&mut self) {
        let mode = self.mode;
        let Some(live) = self.live_mut() else {
            return;
        };
        let motion = if live.reading.held_keys.skimming() {
            Motion::Skimming
        } else {
            Motion::Still
        };
        if motion == live.reading.motion {
            return;
        }
        let committed = if motion == Motion::Still {
            // A turn may not have reached the screen; commit the shown
            // spread so its thumb promotes in place.
            let from = live.reading.position.target();
            if live.reading.position.commit_shown(mode) {
                tracing::debug!(
                    from,
                    to = live.reading.position.target(),
                    "commit shown"
                );
                true
            } else {
                false
            }
        } else {
            false
        };
        tracing::debug!(
            ?motion,
            target = live.reading.position.target(),
            shown = ?live.reading.position.shown(),
            "motion"
        );
        live.reading.motion = motion;
        if committed {
            self.set_view();
        }
        self.send(ParentMsg::SetMotion(motion));
        self.request_redraw();
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
            Action::SelectMode(mode) => {
                let previous = self.mode;
                self.mode = mode;
                self.timers.toast(Instant::now());
                if let Some(live) = self.live_mut() {
                    if live.reading.motion == Motion::Skimming {
                        // The new pairing is a different coordinate system;
                        // commit the shown spread before switching.
                        let from = live.reading.position.target();
                        if live.reading.position.commit_shown(previous) {
                            tracing::debug!(
                                from,
                                to = live.reading.position.target(),
                                "mode switch commits shown"
                            );
                        }
                    }
                    let switch = browse::switch_mode(
                        previous,
                        mode,
                        live.reading.position.target(),
                        live.book.view().len(),
                    );
                    live.reading.position.set_target(switch.index);
                    live.reading.direction = Direction::Forward;
                    if switch.pairing_changed {
                        // The fit begins the new pairing; a mode switch
                        // skips the skim.
                        self.fit.begin_turn();
                    }
                    self.set_view();
                }
                self.request_redraw();
            }
            Action::First | Action::Last => {
                let first = matches!(action, Action::First);
                let mode = self.mode;
                let Some(live) = self.live_mut() else {
                    return;
                };
                let index = if first {
                    browse::first_spread()
                } else {
                    browse::last_spread(mode, live.book.view().len())
                };
                live.reading.position.set_target(index);
                let direction = if first {
                    Direction::Forward
                } else {
                    Direction::Backward
                };
                self.after_page_change(direction);
            }
            Action::NextBook | Action::PrevBook => {
                let limits = self.gl.as_ref().map(|gl| gl.limits);
                let Some(live) = self.live_mut() else {
                    return;
                };
                let change =
                    live.book.switch(matches!(action, Action::NextBook));
                if let Some(limits) = limits {
                    self.apply_book_change(change, limits);
                }
            }
            Action::Right
            | Action::Left
            | Action::Forward
            | Action::Back => {
                let Some(direction) = action.direction(self.mode) else {
                    return;
                };
                let mode = self.mode;
                let Some(live) = self.live_mut() else {
                    return;
                };
                let next = browse::advance(
                    live.reading.position.target(),
                    direction,
                    mode,
                    live.book.view().len(),
                );
                live.reading.position.set_target(next);
                self.after_page_change(direction);
            }
        }
    }

    /// A switch runs off the event loop; the old book's view and textures
    /// stay on screen until the new one lands.
    fn apply_book_change(
        &mut self,
        change: BookChange,
        limits: ImageLimits,
    ) {
        match change {
            BookChange::None => {}
            BookChange::Adopted => {
                let Self {
                    session,
                    gl,
                    fit,
                    mode,
                    ..
                } = self;
                let Session::Reading(live) = session else {
                    return;
                };
                live.reading = adopt_reading(
                    fit,
                    gl.as_mut(),
                    limits,
                    *mode,
                    &live.book,
                    browse::first_spread(),
                );
            }
            BookChange::Faulted => {
                self.request_redraw();
            }
        }
    }

    /// Dropping a killed child's receiver ensures no event arrives from a
    /// replaced book.
    fn poll_book_updates(&mut self, limits: ImageLimits) {
        let (updates, change) = {
            let Some(live) = self.live_mut() else {
                return;
            };
            live.book.poll()
        };
        for event in updates {
            match event {
                BookUpdate::Ready(delivery) => {
                    if let Some(live) = self.live_mut() {
                        live.reading.deliveries.push_back(delivery);
                    }
                }
                BookUpdate::Failed { page, fault } => {
                    if let Some(live) = self.live_mut() {
                        live.reading.faults.insert(page, fault);
                    }
                }
                BookUpdate::Playback(event) => match event {
                    PlaybackEvent::Frame { id, delay, image } => {
                        if let Some(player) = self.player_mut(id) {
                            if !player.on_frame(delay, image) {
                                tracing::debug!(
                                    ?id,
                                    "unrequested playback frame"
                                );
                            }
                        }
                    }
                    PlaybackEvent::Finished { id } => {
                        if let Some(player) = self.player_mut(id) {
                            if player.on_finished() {
                                tracing::debug!(?id, "playback finished");
                            }
                        }
                    }
                    PlaybackEvent::Failed { id } => {
                        if self.is_current_playback(id) {
                            tracing::warn!(?id, "playback failed");
                            self.refuse_playback(id.page);
                        }
                    }
                },
            }
        }
        self.apply_book_change(change, limits);
    }

    fn on_jump_key(&mut self, key: &KeyEvent) {
        use winit::keyboard::PhysicalKey;

        if self.modifiers.control_key()
            || self.modifiers.alt_key()
            || self.modifiers.super_key()
        {
            return;
        }

        // Digits and Backspace repeat at the compositor's rate.
        let digit = key.text.as_deref().and_then(|text| {
            let [digit] = text.as_bytes() else {
                return None;
            };
            digit.is_ascii_digit().then(|| char::from(*digit))
        });
        if let Some(digit) = digit {
            if let Some(live) = self.live_mut() {
                if let Some(digits) = &mut live.reading.jump_digits {
                    digits.push(digit);
                }
            }
            self.request_redraw();
            return;
        }

        let PhysicalKey::Code(code) = key.physical_key else {
            return;
        };
        if code == KeyCode::Backspace {
            if let Some(live) = self.live_mut() {
                if let Some(digits) = &mut live.reading.jump_digits {
                    digits.pop();
                }
            }
            self.request_redraw();
            return;
        }
        if matches!(code, KeyCode::Enter | KeyCode::NumpadEnter) {
            if key.repeat {
                return;
            }
            let Some(live) = self.live_mut() else {
                return;
            };
            let page_count = live.book.view().len().get();
            let Some(digits) = live.reading.jump_digits.as_deref() else {
                return;
            };
            if digits.is_empty() {
                live.reading.jump_digits = None;
                self.request_redraw();
                return;
            }
            if let Some(page) = page_number(digits, page_count) {
                live.reading.jump_digits = None;
                self.jump_to_page(page);
            }
            self.request_redraw();
            return;
        }
        if code == KeyCode::Escape {
            if key.repeat {
                return;
            }
            if let Some(live) = self.live_mut() {
                live.reading.jump_digits = None;
            }
            self.request_redraw();
        }
    }

    fn jump_to_page(&mut self, page: usize) {
        let Some(index) = page.checked_sub(1) else {
            return;
        };
        let mode = self.mode;
        let Some(live) = self.live_mut() else {
            return;
        };
        let len = live.book.view().len();
        let target_start = browse::spread_for(index, mode, len).first;
        let current_start =
            browse::spread_for(live.reading.position.target(), mode, len)
                .first;
        if target_start == current_start {
            return;
        }
        live.reading.position.set_target(target_start);
        self.after_page_change(if target_start > current_start {
            Direction::Forward
        } else {
            Direction::Backward
        });
    }

    fn set_view(&self) {
        let Some(live) = self.live() else {
            return;
        };
        live.book.send(ParentMsg::SetView {
            page: live.reading.position.target(),
            mode: self.mode,
            direction: live.reading.direction,
        });
    }

    fn after_page_change(&mut self, direction: Direction) {
        if let Some(live) = self.live_mut() {
            live.reading.direction = direction;
        }
        self.set_view();
        self.fit.begin_turn();
        if let Some(live) = self.live_mut() {
            live.reading.last_turn = Some(Instant::now());
            tracing::debug!(
                index = live.reading.position.target(),
                ?direction,
                motion = ?live.reading.motion,
                "turn"
            );
        }
        self.request_redraw();
    }
}

impl ApplicationHandler<WakeEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        use egui_system_font_provider::SystemFontProvider;

        // SAFETY: `exiting` drops `self.gl` while the event loop's
        // display is still alive.
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
        let limits =
            display_limits(event_loop, &window, pages.max_texture_edge());
        // Adopt an already-ready book before egui and system font scanning
        // so page 0 decodes concurrently.
        self.poll_initial_open(limits);
        // `EguiGlow::new` panics if its painter cannot initialize; accepted
        // upstream risk.
        let egui =
            EguiGlow::new(event_loop, Arc::clone(&gl), None, None, true);
        // System fonts cover filename characters the built-in fonts cannot
        // render.
        egui.egui_ctx
            .add_font_provider(Arc::new(SystemFontProvider::new()));
        {
            use parking_lot::Mutex;

            let proxy = Mutex::new(self.proxy.clone());
            egui.egui_ctx.set_request_repaint_callback(move |info| {
                post(&proxy.lock(), WakeEvent::Repaint(info.delay));
            });
        }
        let panel_fill = window.translucent().then(glass_fill);
        apply_dark_theme(&egui.egui_ctx, panel_fill);
        window.winit().request_redraw();
        self.gl = Some(Gl {
            pages,
            limits,
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
        // Drops any live child immediately; an in-flight open drops its
        // child if it lands.
        self.session = Session::Closed;
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
        use winit::event::WindowEvent;
        use winit::keyboard::PhysicalKey;

        if matches!(
            event,
            WindowEvent::CloseRequested | WindowEvent::Destroyed
        ) {
            event_loop.exit();
        }
        let Some(gl) = self.gl.as_mut() else {
            return;
        };

        // egui-winit reports repaint on RedrawRequested; honoring it would
        // self-sustain the loop at compositor refresh.
        if !matches!(event, WindowEvent::RedrawRequested) {
            let response =
                gl.egui.on_window_event(gl.window.winit(), &event);
            if response.repaint {
                gl.window.winit().request_redraw();
            }
        }
        let egui_wants_keyboard =
            gl.egui.egui_ctx.egui_wants_keyboard_input();

        match &event {
            WindowEvent::Resized(_) => {
                self.sync_surface_size();
                self.request_redraw();
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
            }
            WindowEvent::KeyboardInput { event: key, .. } => {
                match key.state {
                    ElementState::Pressed => {
                        if self.live().is_some_and(|live| {
                            live.reading.jump_digits.is_some()
                        }) {
                            self.on_jump_key(key);
                            return;
                        }
                        if egui_wants_keyboard {
                            return;
                        }
                        if !key.repeat
                            && !self.modifiers.control_key()
                            && !self.modifiers.alt_key()
                            && !self.modifiers.super_key()
                            && key.text.as_deref() == Some(":")
                        {
                            if let Some(live) = self.live_mut() {
                                live.reading.jump_digits =
                                    Some(String::new());
                                self.request_redraw();
                            }
                            return;
                        }
                        let PhysicalKey::Code(code) = key.physical_key
                        else {
                            return;
                        };
                        let Some(action) =
                            Action::from_key(code, self.modifiers)
                        else {
                            return;
                        };
                        if key.repeat && !action.repeats() {
                            return;
                        }
                        if action.repeats() {
                            self.update_held_keys(|keys| {
                                keys.press(code, key.repeat);
                            });
                        }
                        self.dispatch(action, event_loop);
                    }
                    ElementState::Released => {
                        let PhysicalKey::Code(code) = key.physical_key
                        else {
                            return;
                        };
                        self.update_held_keys(|keys| {
                            keys.release(code);
                        });
                    }
                }
            }
            WindowEvent::Focused(false) => {
                self.update_held_keys(HeldKeys::clear);
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
    content: Rect,
    pixels_per_point: f32,
) -> Viewport {
    let win = geometry::px_f32((window.width, window.height));
    // The bar's top edge rounds to a whole physical px, matching the
    // painter's clip-rect rounding.
    let bar_top = (content.max.y * pixels_per_point).round();
    Viewport::above_bottom_bar(win, bar_top)
}

fn raw_zoom_factor(page_surface: &Response, points_per_line: f32) -> f32 {
    use egui::Event;
    use egui::MouseWheelUnit;

    let viewport_height = page_surface.ctx.viewport_rect().height();
    page_surface.ctx.input(|i| {
        let mut wheel_lines = 0.0;
        let mut pinch_factor = 1.0;
        // egui smooths wheel deltas over frames; page zoom follows each
        // raw event.
        for event in &i.raw.events {
            if let Event::MouseWheel { unit, delta, .. } = event {
                let scroll_points = match unit {
                    MouseWheelUnit::Point => delta.y,
                    MouseWheelUnit::Line => delta.y * points_per_line,
                    MouseWheelUnit::Page => delta.y * viewport_height,
                };
                wheel_lines += scroll_points / points_per_line;
            }
            if let Event::Zoom(zoom_factor) = event {
                pinch_factor *= zoom_factor;
            }
        }
        pinch_factor * (wheel_lines * WHEEL_ZOOM_PER_LINE).exp()
    })
}

fn apply_page_input(
    fit: &mut Fit,
    page_surface: &Response,
    spread_size: (f32, f32),
    viewport: Viewport,
) -> bool {
    use egui::PointerButton;

    let pixels_per_point = page_surface.ctx.pixels_per_point();
    if page_surface.dragged_by(PointerButton::Primary) {
        let delta = page_surface.drag_delta() * pixels_per_point;
        if delta.length_sq() > 0.0 {
            fit.pan((delta.x, delta.y), spread_size, viewport);
        }
    }
    if !page_surface.hovered() {
        return false;
    }
    let points_per_line = page_surface
        .ctx
        .options(|options| options.input_options.line_scroll_speed);
    if points_per_line <= 0.0 {
        return false;
    }
    let zoom_factor = raw_zoom_factor(page_surface, points_per_line);
    if (zoom_factor - 1.0).abs() <= f32::EPSILON {
        return false;
    }
    let Some(hover_pos) = page_surface.hover_pos() else {
        return false;
    };
    let anchor = (hover_pos - page_surface.rect.min) * pixels_per_point;
    fit.zoom(zoom_factor, (anchor.x, anchor.y), spread_size, viewport);
    true
}

/// The glass in egui's premultiplied u8 color, converted from [`clear_color`].
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the clear color's channels lie in 0..=1, so the rounded scaled value is a valid u8"
)]
fn glass_fill() -> Color32 {
    let [red, green, blue, alpha] = clear_color(true);
    let channel = |value: f32| value.mul_add(255.0, 0.5) as u8;
    Color32::from_rgba_premultiplied(
        channel(red),
        channel(green),
        channel(blue),
        channel(alpha),
    )
}

/// A light desktop theme would otherwise pick dark status text over the
/// dark window.
fn apply_dark_theme(egui_ctx: &Context, panel_fill: Option<Color32>) {
    use egui::Theme;

    egui_ctx.set_theme(Theme::Dark);
    if let Some(fill) = panel_fill {
        egui_ctx.global_style_mut(|style| {
            style.visuals.panel_fill = fill;
        });
    }
}

const WHEEL_ZOOM_PER_LINE: f32 = 0.2;

#[cfg(test)]
mod tests {
    use super::*;
    use egui::Pos2;
    use egui::RawInput;
    use egui::Theme;

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
    fn egui_stays_dark_when_the_desktop_is_light() {
        let ctx = Context::default();
        apply_dark_theme(&ctx, Some(glass_fill()));
        ctx.run_ui(
            RawInput {
                system_theme: Some(Theme::Light),
                ..Default::default()
            },
            |_| {},
        )
        .drop_without_applying_deltas();
        assert_eq!(ctx.theme(), Theme::Dark);
        assert_eq!(ctx.global_style().visuals.panel_fill, glass_fill());
    }

    #[test]
    fn content_viewport_subtracts_the_bar_in_physical_px() {
        let content =
            Rect::from_min_max(Pos2::ZERO, egui::pos2(1280.0, 772.0));
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

fn classify(path: &Path) -> Result<Argument, OpenRefusal> {
    let normalized = catalog::normalize_path(path);
    let Some(name) = normalized.file_name() else {
        return Ok(Argument::Book(normalized));
    };
    if !normalized.is_file() || catalog::is_archive(&normalized) {
        return Ok(Argument::Book(normalized));
    }
    let name = name.to_string_lossy().into_owned();
    if decode::format_from_name(&name).is_none() {
        return Err(OpenRefusal::Unsupported(normalized));
    }
    let dir = normalized
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(Argument::Image {
        dir: dir.to_path_buf(),
        name,
    })
}

fn run(open_path: &Path, land_on: Option<&str>) -> eros::Result<()> {
    use egui_winit::winit::event_loop::EventLoop;

    // Build the loop and fork the child here so `PR_SET_PDEATHSIG` tracks
    // the event-loop thread.
    let event_loop = EventLoop::<WakeEvent>::with_user_event().build()?;
    let mut app = App::new(open_path, land_on, event_loop.create_proxy());
    event_loop.run_app(&mut app)?;
    if let Some(error) = app.take_fatal_error() {
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod argv_tests {
    use super::*;
    use crate::testkit::TempDir;

    #[test]
    fn classify_argument_kinds() {
        use std::env;
        use std::fs;
        use std::os::unix::fs::symlink;

        // classify never reads the content, only the extension.
        let dir = TempDir::new("parent-arg");
        dir.write("page.png", b"x");
        dir.write("book.cbz", b"not really a zip");
        dir.write("notes.txt", b"not a page");
        fs::create_dir_all(dir.path().join("sub")).expect("mkdir sub");
        symlink(dir.path().join("page.png"), dir.path().join("link.png"))
            .expect("symlink page");
        symlink(dir.path().join("sub"), dir.path().join("link_dir"))
            .expect("symlink dir");
        symlink(
            dir.path().join("missing.png"),
            dir.path().join("broken.png"),
        )
        .expect("broken symlink");

        let rows = [
            (
                dir.path().join("page.png"),
                Ok(Argument::Image {
                    dir: dir.path().to_path_buf(),
                    name: String::from("page.png"),
                }),
            ),
            (
                dir.path().join("link.png"),
                Ok(Argument::Image {
                    dir: dir.path().to_path_buf(),
                    name: String::from("link.png"),
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
                dir.path().join("link_dir/."),
                Ok(Argument::Book(dir.path().join("link_dir"))),
            ),
            (
                dir.path().join("broken.png"),
                Ok(Argument::Book(dir.path().join("broken.png"))),
            ),
            (
                dir.path().join("notes.txt"),
                Err(OpenRefusal::Unsupported(
                    dir.path().join("notes.txt"),
                )),
            ),
            (
                PathBuf::from("."),
                Ok(Argument::Book(
                    env::current_dir().expect("current dir"),
                )),
            ),
        ];
        for (path, expected) in rows {
            assert_eq!(classify(&path), expected, "{}", path.display());
        }
    }
}
