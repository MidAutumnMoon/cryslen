//! App: the windowed parent process. Parses the command line, runs the
//! event loop, owns the reading state and input, and composes frames.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::num::NonZeroUsize;
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

use crate::book::BookEvent;
use crate::book::BookOpen;
use crate::book::BookProcess;
use crate::browse::Direction;
use crate::browse::Motion;
use crate::browse::ReadingMode;
use crate::browse::Spread;
use crate::catalog;
use crate::catalog::PageName;
use crate::decode;
use crate::geometry;
use crate::geometry::Fit;
use crate::geometry::FitMode;
use crate::geometry::SpreadLayout;
use crate::geometry::Viewport;
use crate::protocol::BookInfo;
use crate::protocol::ImageLimits;
use crate::protocol::ParentMsg;
use crate::reading::Reading;
use crate::render::Window;
use crate::ui;
use crate::ui::StatusLine;
use crate::ui::Timers;

enum WakeEvent {
    /// Arms a wake; `new_events` requests the redraw at the deadline.
    Repaint(Duration),
    /// A book's reader thread forwarded an event, or a switch landed;
    /// the loop must draw a frame to drain it.
    Wake,
}

fn post(proxy: &EventLoopProxy<WakeEvent>, event: WakeEvent) {
    let Ok(()) = proxy.send_event(event) else {
        return;
    };
}

struct Gl {
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

fn page_number(digits: &str, page_count: usize) -> Option<NonZeroUsize> {
    digits
        .parse::<NonZeroUsize>()
        .ok()
        .filter(|page| page.get() <= page_count)
}

enum BookSlot {
    Initial(PageName),
    Loaded(Box<Reading>),
}

impl BookSlot {
    fn reading(&self) -> Option<&Reading> {
        match self {
            Self::Initial(_) => None,
            Self::Loaded(reading) => Some(reading),
        }
    }

    fn reading_mut(&mut self) -> Option<&mut Reading> {
        match self {
            Self::Initial(_) => None,
            Self::Loaded(reading) => Some(reading),
        }
    }
}

enum ChildProcess {
    Opening {
        open: BookOpen,
        land_on: Option<String>,
    },
    Live(BookProcess),
    Faulted(String),
    Stopped,
}

impl ChildProcess {
    fn is_opening(&self) -> bool {
        matches!(self, Self::Opening { .. })
    }

    fn is_live(&self) -> bool {
        matches!(self, Self::Live(_))
    }

    fn fault(&self) -> Option<&str> {
        match self {
            Self::Faulted(fault) => Some(fault),
            Self::Opening { .. } | Self::Live(_) | Self::Stopped => None,
        }
    }

    fn send(&self, message: ParentMsg) {
        if let Self::Live(process) = self {
            process.send(message);
        }
    }
}

#[derive(Default)]
struct ViewState {
    mode: ReadingMode,
    fit: Fit,
    timers: Timers,
    last_viewport: Viewport,
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

fn present_frame(
    gl: &mut Gl,
    view: &mut ViewState,
    size: PhysicalSize<u32>,
    status: &StatusLine<'_>,
    spread: Option<(&SpreadLayout, Option<Spread>)>,
    jump_digits: Option<&str>,
) -> eros::Result<()> {
    let show_status_bar = gl.window.winit().fullscreen().is_none();
    let jump = jump_digits.and_then(|digits| {
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
    if viewport != view.last_viewport {
        view.fit.refit();
        view.last_viewport = viewport;
        gl.window.winit().request_redraw();
    }

    let placement = spread.as_ref().map(|(layout, _)| {
        let spread_size = layout.total();
        if jump.is_none()
            && apply_page_input(
                &mut view.fit,
                &page_surface,
                spread_size,
                viewport,
            )
        {
            gl.window.winit().request_redraw();
        }
        view.fit.resolve(spread_size, viewport)
    });

    gl.egui.painter.clear(
        [size.width, size.height],
        clear_color(gl.window.translucent()),
    );
    if let (Some((layout, Some(shown))), Some(placement)) =
        (spread, placement)
    {
        gl.window.pages_mut().draw(
            (size.width, size.height),
            viewport,
            layout,
            shown.seats(status.mode),
            &placement,
        );
    }
    gl.egui.paint(gl.window.winit());
    gl.window.pre_present_and_swap()?;
    Ok(())
}

struct App {
    book: BookSlot,
    child: ChildProcess,
    gl: Option<Gl>,
    fatal_error: Option<ErrorUnion>,
    view: ViewState,
    held_keys: HeldKeys,
    /// Digits typed in `:` mode; `None` while the prompt is closed.
    jump_digits: Option<String>,
    /// Held modifier state; winit 0.30 key events carry none, only
    /// `ModifiersChanged` does.
    modifiers: ModifiersState,
    /// Wakes the parked loop from egui, timed waits, and book readers,
    /// which all run outside the event loop's handlers.
    proxy: EventLoopProxy<WakeEvent>,
}

impl App {
    fn new(
        open_path: &Path,
        land_on: Option<&str>,
        proxy: EventLoopProxy<WakeEvent>,
    ) -> Self {
        let mut app = Self {
            book: BookSlot::Initial(catalog::opening_name(
                open_path, land_on,
            )),
            child: ChildProcess::Stopped,
            gl: None,
            fatal_error: None,
            view: ViewState::default(),
            held_keys: HeldKeys::default(),
            jump_digits: None,
            modifiers: ModifiersState::empty(),
            proxy,
        };
        app.start_open(open_path, land_on.map(str::to_owned));
        app
    }

    fn fail(&mut self, error: ErrorUnion, event_loop: &ActiveEventLoop) {
        self.fatal_error = Some(error);
        event_loop.exit();
    }

    fn take_fatal_error(&mut self) -> Option<ErrorUnion> {
        self.fatal_error.take()
    }

    fn start_open(&mut self, path: &Path, land_on: Option<String>) {
        if self.child.is_opening() {
            return;
        }
        // Drop any running child before spawning so two workers never overlap.
        self.child = ChildProcess::Stopped;
        let proxy = self.proxy.clone();
        self.child = match BookOpen::start(path, move || {
            post(&proxy, WakeEvent::Wake);
        }) {
            Ok(open) => ChildProcess::Opening { open, land_on },
            Err(fault) => {
                tracing::warn!(%fault, "book open failed");
                self.request_redraw();
                ChildProcess::Faulted(fault.to_string())
            }
        };
    }

    fn adopt_book(
        &mut self,
        path: PathBuf,
        info: BookInfo,
        land_on: Option<&str>,
        process: BookProcess,
        limits: ImageLimits,
    ) {
        self.held_keys.clear();
        self.jump_digits = None;
        self.view.fit.refit();
        if let Some(gl) = self.gl.as_mut() {
            gl.window.pages_mut().clear_all();
        }
        let mode = self.view.mode;
        let (reading, requests) = Reading::open(path, info, land_on, mode);
        process.send(ParentMsg::SetLimits(limits));
        process.send(ParentMsg::SetView {
            page: reading.target(),
            mode,
            direction: reading.direction(),
        });
        process.send(ParentMsg::SetRequests(requests));
        tracing::debug!(
            path = %reading.path().display(),
            count = reading.page_count().get(),
            "book adopted"
        );
        self.book = BookSlot::Loaded(Box::new(reading));
        self.child = ChildProcess::Live(process);
    }

    fn send(&self, message: ParentMsg) {
        self.child.send(message);
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

    fn poll_child(&mut self, limits: ImageLimits) {
        match &mut self.child {
            ChildProcess::Opening { open, land_on } => {
                let Some(opened) = open.try_take() else {
                    return;
                };
                let land_on = land_on.take();
                match opened {
                    Ok((path, info, process)) => {
                        self.adopt_book(
                            path,
                            info,
                            land_on.as_deref(),
                            process,
                            limits,
                        );
                    }
                    Err(fault) => {
                        tracing::warn!(%fault, "book open failed");
                        self.child =
                            ChildProcess::Faulted(fault.to_string());
                    }
                }
                self.request_redraw();
            }
            ChildProcess::Live(process) => {
                let Some(reading) = self.book.reading_mut() else {
                    return;
                };
                let mut died = None;
                for event in process.drain() {
                    match event {
                        BookEvent::Died(detail) => died = Some(detail),
                        other @ (BookEvent::Ready { .. }
                        | BookEvent::Playback(_)
                        | BookEvent::Failed { .. }) => {
                            if let Some(command) =
                                reading.apply_event(other)
                            {
                                process.send(ParentMsg::Playback(command));
                            }
                        }
                    }
                }
                if let Some(detail) = died {
                    tracing::warn!(%detail, "the book process failed");
                    self.child = ChildProcess::Faulted(format!(
                        "the book process died: {detail}"
                    ));
                    self.request_redraw();
                }
            }
            ChildProcess::Faulted(_) | ChildProcess::Stopped => {}
        }
    }

    fn frame(&mut self) -> eros::Result<()> {
        const SLOW_FRAME_THRESHOLD: Duration = Duration::from_millis(50);

        let frame_started = Instant::now();
        let Some(size) = self.sync_surface_size() else {
            return Ok(());
        };
        let Some(limits) = self.gl.as_ref().map(|gl| gl.limits) else {
            return Ok(());
        };
        self.poll_child(limits);
        self.drain_uploads();
        let drain_elapsed = frame_started.elapsed();
        let now = Instant::now();

        let Self {
            book,
            child,
            gl,
            view,
            jump_digits,
            proxy,
            ..
        } = self;
        let Some(gl) = gl.as_mut() else {
            return Ok(());
        };
        match book {
            BookSlot::Initial(name) => {
                tick_timers(
                    &mut view.timers,
                    proxy,
                    now,
                    child.is_opening(),
                );
                let status = StatusLine {
                    reading: None,
                    mode: view.mode,
                    toast: view.timers.toasting(),
                    name: name.clone(),
                    waiting: view.timers.waiting(now),
                    fault: child.fault(),
                };
                present_frame(gl, view, size, &status, None, None)
            }
            BookSlot::Loaded(reading) => {
                present_reading(
                    reading,
                    child,
                    gl,
                    view,
                    jump_digits.as_deref(),
                    size,
                    proxy,
                )?;
                let total = frame_started.elapsed();
                if total >= SLOW_FRAME_THRESHOLD {
                    tracing::info!(
                        total = ?total,
                        drain = ?drain_elapsed,
                        pending_uploads = reading.pending_upload_count(),
                        shown = ?reading.shown(),
                        "slow frame"
                    );
                }
                Ok(())
            }
        }
    }

    fn drain_uploads(&mut self) {
        const SLOW_UPLOAD_THRESHOLD: Duration = Duration::from_millis(30);

        let Self {
            book, gl, proxy, ..
        } = self;
        let (Some(gl), Some(reading)) = (gl.as_mut(), book.reading_mut())
        else {
            return;
        };
        let started = Instant::now();
        let mut uploaded_count = 0_usize;
        let mut uploaded_bytes = 0_usize;
        let textures = gl.window.pages_mut();
        while let Some(upload) =
            reading.pop_upload(uploaded_count, uploaded_bytes)
        {
            let image_bytes = upload.image.resident_bytes();
            let upload_started = Instant::now();
            match textures.insert(upload.page, upload.role, &upload.image)
            {
                Ok(()) => {
                    let elapsed = upload_started.elapsed();
                    reading.record_uploaded(upload.page, &upload.image);
                    tracing::debug!(
                        page = upload.page,
                        role = ?upload.role,
                        width = upload.image.meta.width,
                        height = upload.image.meta.height,
                        elapsed = ?elapsed,
                        "uploaded"
                    );
                    if elapsed >= SLOW_UPLOAD_THRESHOLD {
                        tracing::info!(
                            page = upload.page,
                            role = ?upload.role,
                            width = upload.image.meta.width,
                            height = upload.image.meta.height,
                            elapsed = ?elapsed,
                            "slow upload"
                        );
                    }
                    uploaded_count += 1;
                    uploaded_bytes =
                        uploaded_bytes.saturating_add(image_bytes);
                }
                Err(fault) => {
                    tracing::warn!(
                        page = upload.page,
                        role = ?upload.role,
                        %fault,
                        "texture upload refused"
                    );
                    let has_settled_image =
                        textures.roles(upload.page).has_settled_image();
                    reading.record_upload_refused(
                        upload.page,
                        upload.role,
                        has_settled_image,
                        fault,
                    );
                }
            }
        }
        if uploaded_count > 0 {
            tracing::debug!(
                count = uploaded_count,
                bytes = uploaded_bytes,
                elapsed = ?started.elapsed(),
                "drained"
            );
        }
        if reading.has_pending_uploads() {
            post(proxy, WakeEvent::Repaint(Duration::ZERO));
        }
    }

    fn update_held_keys(&mut self, update: impl FnOnce(&mut HeldKeys)) {
        let mode = self.view.mode;
        update(&mut self.held_keys);
        let Some(reading) = self.book.reading_mut() else {
            return;
        };
        let motion = if self.held_keys.skimming() {
            Motion::Skimming
        } else {
            Motion::Still
        };
        let before = reading.target();
        if !reading.set_motion(motion, mode) {
            return;
        }
        if reading.target() != before {
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
            Action::SelectFit(mode) => {
                self.view.fit.select(mode);
                self.request_redraw();
            }
            Action::SelectMode(mode) => {
                let previous = self.view.mode;
                self.view.mode = mode;
                self.view.timers.toast(Instant::now());
                if let Some(reading) = self.book.reading_mut() {
                    if reading.switch_mode(previous, mode) {
                        // The fit begins the new pairing; a mode switch
                        // skips the skim.
                        self.view.fit.begin_turn();
                    }
                    self.set_view();
                }
                self.request_redraw();
            }
            Action::First | Action::Last => {
                let mode = self.view.mode;
                let Some(reading) = self.book.reading_mut() else {
                    return;
                };
                if matches!(action, Action::First) {
                    reading.jump_to_first(Instant::now());
                } else {
                    reading.jump_to_last(mode, Instant::now());
                }
                self.after_page_change();
            }
            Action::NextBook | Action::PrevBook => {
                // Switch runs off the event loop; the old book's view and
                // textures stay on screen until the new one lands.
                if self.child.is_opening() {
                    return;
                }
                let Some(path) = self.book.reading().and_then(|reading| {
                    reading.sibling(matches!(action, Action::NextBook))
                }) else {
                    return;
                };
                tracing::debug!(path = %path.display(), "book switch");
                self.start_open(&path, None);
            }
            Action::Right
            | Action::Left
            | Action::Forward
            | Action::Back => {
                let Some(direction) = action.direction(self.view.mode)
                else {
                    return;
                };
                let mode = self.view.mode;
                let Some(reading) = self.book.reading_mut() else {
                    return;
                };
                if reading.turn(direction, mode, Instant::now()) {
                    self.after_page_change();
                }
            }
        }
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
            if let Some(digits) = &mut self.jump_digits {
                digits.push(digit);
            }
            self.request_redraw();
            return;
        }

        let PhysicalKey::Code(code) = key.physical_key else {
            return;
        };
        if code == KeyCode::Backspace {
            if let Some(digits) = &mut self.jump_digits {
                digits.pop();
            }
            self.request_redraw();
            return;
        }
        if matches!(code, KeyCode::Enter | KeyCode::NumpadEnter) {
            if key.repeat {
                return;
            }
            let Some(page_count) = self
                .book
                .reading()
                .map(|reading| reading.page_count().get())
            else {
                return;
            };
            let Some(digits) = self.jump_digits.as_deref() else {
                return;
            };
            if digits.is_empty() {
                self.jump_digits = None;
                self.request_redraw();
                return;
            }
            if let Some(page_number) = page_number(digits, page_count) {
                self.jump_digits = None;
                self.jump_to_page(page_number);
            }
            self.request_redraw();
            return;
        }
        if code == KeyCode::Escape {
            if key.repeat {
                return;
            }
            self.jump_digits = None;
            self.request_redraw();
        }
    }

    fn jump_to_page(&mut self, page_number: NonZeroUsize) {
        let mode = self.view.mode;
        let Some(reading) = self.book.reading_mut() else {
            return;
        };
        if reading.jump_to_page(page_number, mode, Instant::now()) {
            self.after_page_change();
        }
    }

    fn set_view(&self) {
        let Some(reading) = self.book.reading() else {
            return;
        };
        self.send(ParentMsg::SetView {
            page: reading.target(),
            mode: self.view.mode,
            direction: reading.direction(),
        });
    }

    fn after_page_change(&mut self) {
        self.set_view();
        self.view.fit.begin_turn();
        self.request_redraw();
    }
}

fn present_reading(
    reading: &mut Reading,
    child: &ChildProcess,
    gl: &mut Gl,
    view: &mut ViewState,
    jump_digits: Option<&str>,
    size: PhysicalSize<u32>,
    proxy: &EventLoopProxy<WakeEvent>,
) -> eros::Result<()> {
    let now = Instant::now();
    let (shown, changed) = reading.resolve_shown(view.mode, now, |page| {
        gl.window.pages().roles(page)
    });
    if changed {
        view.fit.commit();
    }
    let on_screen = reading.on_screen(view.mode);
    let dims = reading.seat_dims(view.mode, on_screen);
    let layout = geometry::layout(dims);
    let spread_size = layout.total();
    let placed = view.fit.resolve(spread_size, view.last_viewport);
    if changed {
        tracing::debug!(
            ?on_screen,
            ?shown,
            total = ?spread_size,
            ?dims,
            scale = placed.scale,
            "placement"
        );
    }

    let (view_width, view_height) = view.last_viewport.size();
    let on_screen_scale =
        (view_width > 0 && view_height > 0).then_some(placed.scale);
    if let Some(requests) = reading.update_requests(
        view.mode,
        on_screen,
        on_screen_scale,
        gl.limits,
        |page| gl.window.pages().roles(page),
    ) {
        child.send(ParentMsg::SetRequests(requests));
    }
    reading.note_hold(view.mode, now);

    gl.window.pages_mut().retain(|page, role| {
        reading.retains_texture(view.mode, on_screen, page, role)
    });
    for command in reading.sync_playback(view.mode, on_screen, |page| {
        gl.window.pages().roles(page)
    }) {
        child.send(ParentMsg::Playback(command));
    }
    let (deadline, commands) =
        reading.tick_playback(now, |page, role, image| {
            gl.window.pages_mut().update(page, role, image)
        });
    for command in commands {
        child.send(ParentMsg::Playback(command));
    }
    if let Some(delay) = deadline {
        post(proxy, WakeEvent::Repaint(delay));
    }

    let mut status = reading.status_line(
        on_screen,
        view.mode,
        placed.scale,
        child.fault(),
        |page| gl.window.pages().has_any(page),
    );
    tick_timers(
        &mut view.timers,
        proxy,
        now,
        status.waiting && child.is_live(),
    );
    status.waiting = view.timers.waiting(now);
    status.toast = view.timers.toasting();
    present_frame(
        gl,
        view,
        size,
        &status,
        Some((&layout, reading.shown())),
        jump_digits,
    )
}

impl ApplicationHandler<WakeEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        use egui_system_font_provider::SystemFontProvider;

        let window = match Window::new(event_loop) {
            Ok(window) => window,
            Err(error) => {
                self.fail(error, event_loop);
                return;
            }
        };
        let limits = display_limits(
            event_loop,
            &window,
            window.pages().max_texture_edge(),
        );
        // Adopt an already-ready book before egui and system font scanning
        // so page 0 decodes concurrently.
        self.poll_child(limits);
        // `EguiGlow::new` panics if its painter cannot initialize; accepted
        // upstream risk.
        let egui =
            EguiGlow::new(event_loop, window.glow(), None, None, true);
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
        let panel_fill = window.translucent().then_some(GLASS_FILL);
        apply_dark_theme(&egui.egui_ctx, panel_fill);
        window.winit().request_redraw();
        self.gl = Some(Gl {
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
                    && armed <= deadline
                {
                    return;
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
        self.child = ChildProcess::Stopped;
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
                        if self.jump_digits.is_some() {
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
                            if self.book.reading().is_some() {
                                self.jump_digits = Some(String::new());
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

const OPAQUE_FILL: Color32 = Color32::from_rgb(20, 20, 23);
const GLASS_ALPHA: u8 = 179;
const GLASS_FILL: Color32 = Color32::from_rgba_unmultiplied_const(
    OPAQUE_FILL.r(),
    OPAQUE_FILL.g(),
    OPAQUE_FILL.b(),
    GLASS_ALPHA,
);

/// Framebuffer clear color, premultiplied because Wayland composites
/// ARGB buffers that way.
fn clear_color(translucent: bool) -> [f32; 4] {
    let color = if translucent {
        GLASS_FILL
    } else {
        OPAQUE_FILL
    };
    color.to_normalized_gamma_f32()
}

/// The page viewport from the chrome egui just laid out: the full-width
/// window area above the status bar, in physical px.
fn content_viewport(
    window: PhysicalSize<u32>,
    content: Rect,
    pixels_per_point: f32,
) -> Viewport {
    let bar_top = f64::from(content.max.y) * f64::from(pixels_per_point);
    Viewport::above_bottom_bar((window.width, window.height), bar_top)
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
    spread_size: (f64, f64),
    viewport: Viewport,
) -> bool {
    use egui::PointerButton;

    let pixels_per_point = page_surface.ctx.pixels_per_point();
    if page_surface.dragged_by(PointerButton::Primary) {
        let delta = page_surface.drag_delta() * pixels_per_point;
        if delta.length_sq() > 0.0 {
            fit.pan(
                (f64::from(delta.x), f64::from(delta.y)),
                spread_size,
                viewport,
            );
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
    fit.zoom(
        f64::from(zoom_factor),
        (f64::from(anchor.x), f64::from(anchor.y)),
        spread_size,
        viewport,
    );
    true
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
        assert_eq!(
            clear_color(false),
            [20.0 / 255.0, 20.0 / 255.0, 23.0 / 255.0, 1.0]
        );
        assert_eq!(
            clear_color(true),
            [14.0 / 255.0, 14.0 / 255.0, 16.0 / 255.0, 179.0 / 255.0]
        );
    }

    #[test]
    fn egui_stays_dark_when_the_desktop_is_light() {
        let ctx = Context::default();
        apply_dark_theme(&ctx, Some(GLASS_FILL));
        ctx.run_ui(
            RawInput {
                system_theme: Some(Theme::Light),
                ..Default::default()
            },
            |_| {},
        )
        .drop_without_applying_deltas();
        assert_eq!(ctx.theme(), Theme::Dark);
        assert_eq!(ctx.global_style().visuals.panel_fill, GLASS_FILL);
    }

    #[test]
    fn content_viewport_subtracts_the_bar_in_physical_px() {
        let content =
            Rect::from_min_max(Pos2::ZERO, egui::pos2(1280.0, 772.0));
        let integer =
            content_viewport(PhysicalSize::new(1280, 800), content, 1.0);
        assert_eq!(integer.size(), (1280, 772));
        // Fractional output scale: 772 pt at 1.25 pixels per point leaves
        // 965 physical px of content in a 1000 px window.
        let fractional =
            content_viewport(PhysicalSize::new(1600, 1000), content, 1.25);
        assert_eq!(fractional.size(), (1600, 965));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Quit,
    /// Leave fullscreen when fullscreen, quit otherwise.
    Escape,
    SelectMode(ReadingMode),
    ToggleFullscreen,
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
            KeyCode::KeyR => Some(Self::SelectFit(FitMode::Page)),
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
    use strum::VariantArray as _;

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
            (KeyCode::KeyR, Some(Action::SelectFit(FitMode::Page))),
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
            Action::ToggleFullscreen,
            Action::First,
            Action::Last,
            Action::NextBook,
            Action::PrevBook,
        ]
        .into_iter()
        .chain(
            ReadingMode::VARIANTS
                .iter()
                .copied()
                .map(Action::SelectMode),
        )
        .chain(FitMode::VARIANTS.iter().copied().map(Action::SelectFit));
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

#[expect(
    clippy::use_debug,
    reason = "the fatal exit prints the full error report"
)]
pub fn main(args: &[OsString]) -> ExitCode {
    let Some(argument) = args.get(1) else {
        eprintln!("usage: cy <dir | archive | image-file>");
        return ExitCode::from(2);
    };
    let (open_path, land_on) = match classify(Path::new(argument)) {
        Ok(open) => open,
        Err(error) => {
            eprintln!("cy: {error}");
            return ExitCode::FAILURE;
        }
    };

    match run(&open_path, land_on.as_deref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // `RUST_BACKTRACE` arms the backtrace in the Debug report.
            eprintln!("cy: {error:?}");
            ExitCode::FAILURE
        }
    }
}

fn classify(path: &Path) -> eros::Result<(PathBuf, Option<String>)> {
    let normalized = catalog::normalize_path(path);
    let Some(name) = normalized.file_name() else {
        return Ok((normalized, None));
    };
    if !normalized.is_file() || catalog::is_archive(&normalized) {
        return Ok((normalized, None));
    }
    let name = name.to_string_lossy().into_owned();
    if decode::format_from_name(&name).is_none() {
        eros::bail!("unsupported source: {}", normalized.display());
    }
    let dir = normalized
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok((dir.to_path_buf(), Some(name)))
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
    use tempfile::TempDir;

    #[test]
    fn classify_argument_kinds() {
        use std::env;
        use std::fs;
        use std::os::unix::fs as unix_fs;

        // classify never reads the content, only the extension.
        let dir = TempDir::new().expect("create temp dir");
        fs::write(dir.path().join("page.png"), b"x")
            .expect("write page.png");
        fs::write(dir.path().join("book.cbz"), b"not really a zip")
            .expect("write book.cbz");
        fs::write(dir.path().join("notes.txt"), b"not a page")
            .expect("write notes.txt");
        fs::create_dir_all(dir.path().join("sub")).expect("mkdir sub");
        unix_fs::symlink(
            dir.path().join("page.png"),
            dir.path().join("link.png"),
        )
        .expect("symlink page");
        unix_fs::symlink(
            dir.path().join("sub"),
            dir.path().join("link_dir"),
        )
        .expect("symlink dir");
        unix_fs::symlink(
            dir.path().join("missing.png"),
            dir.path().join("broken.png"),
        )
        .expect("broken symlink");

        let rows = [
            (
                dir.path().join("page.png"),
                (dir.path().to_path_buf(), Some(String::from("page.png"))),
            ),
            (
                dir.path().join("link.png"),
                (dir.path().to_path_buf(), Some(String::from("link.png"))),
            ),
            (
                dir.path().join("book.cbz"),
                (dir.path().join("book.cbz"), None),
            ),
            (dir.path().to_path_buf(), (dir.path().to_path_buf(), None)),
            (
                dir.path().join("link_dir/."),
                (dir.path().join("link_dir"), None),
            ),
            (
                dir.path().join("broken.png"),
                (dir.path().join("broken.png"), None),
            ),
            (
                PathBuf::from("."),
                (env::current_dir().expect("current dir"), None),
            ),
        ];
        for (path, expected) in rows {
            assert_eq!(
                classify(&path).ok(),
                Some(expected),
                "{}",
                path.display()
            );
        }
        let unsupported = classify(&dir.path().join("notes.txt"))
            .expect_err("non-image file must refuse");
        assert!(unsupported.to_string().contains("unsupported source"));
    }
}
