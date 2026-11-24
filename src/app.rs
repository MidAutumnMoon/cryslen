//! App: the windowed parent process. Parses the command line, runs the
//! event loop, owns the reading state and input, and composes frames.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use egui::Color32;
use egui::Context;
use egui::Rect;
use egui::TexturesDelta;
use egui::Ui;
use egui::epaint::ClippedShape;
use egui_glow::Painter;
use egui_glow::glow::Context as GlowContext;
use eros::ErrorUnion;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::ElementState;
use winit::event::KeyEvent;
use winit::event::MouseScrollDelta;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::EventLoopProxy;
use winit::keyboard::KeyCode;
use winit::keyboard::ModifiersState;
use winit::keyboard::PhysicalKey;
use winit::window::WindowId;

use crate::book::BookOpen;
use crate::book::BookProcess;
use crate::browse::Direction;
use crate::browse::ReadingMode;
use crate::browse::Spread;
use crate::catalog;
use crate::catalog::PageName;
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
use crate::ui::StatusLine;
use crate::ui::Timers;

/// A book's reader thread forwarded an event, or an open finished; the
/// loop reconciles state in `about_to_wait`.
struct Wake;

struct Egui {
    ctx: Context,
    painter: Painter,
}

struct EguiFrame {
    shapes: Vec<ClippedShape>,
    textures_delta: TexturesDelta,
    pixels_per_point: f32,
    repaint_requested: bool,
}

impl Egui {
    fn new(
        gl: Arc<GlowContext>,
        panel_fill: Option<Color32>,
    ) -> eros::Result<Self> {
        use egui_system_font_provider::SystemFontProvider;

        let painter = Painter::new(gl, "", None, true)?;
        let ctx = Context::default();
        // System fonts cover filename characters the built-in fonts cannot
        // render.
        ctx.add_font_provider(Arc::new(SystemFontProvider::new()));
        apply_dark_theme(&ctx, panel_fill);
        Ok(Self { ctx, painter })
    }

    fn layout(
        &self,
        size: PhysicalSize<u32>,
        scale_factor: f64,
        run_ui: impl FnMut(&mut Ui),
    ) -> EguiFrame {
        use egui::Pos2;
        use egui::RawInput;
        use egui::ViewportId;
        use egui::ViewportIdMap;
        use egui::ViewportInfo;

        #[expect(
            clippy::cast_possible_truncation,
            reason = "egui pixels_per_point is f32"
        )]
        let pixels_per_point = scale_factor as f32;
        let logical = size.to_logical::<f32>(scale_factor);
        let screen_rect = (logical.width > 0.0 && logical.height > 0.0)
            .then(|| {
                Rect::from_min_size(
                    Pos2::ZERO,
                    egui::vec2(logical.width, logical.height),
                )
            });
        let mut viewports = ViewportIdMap::default();
        viewports.insert(
            ViewportId::ROOT,
            ViewportInfo {
                native_pixels_per_point: Some(pixels_per_point),
                ..Default::default()
            },
        );
        let input = RawInput {
            viewport_id: ViewportId::ROOT,
            viewports,
            screen_rect,
            max_texture_side: Some(self.painter.max_texture_side()),
            ..Default::default()
        };
        let output = self.ctx.run_ui(input, run_ui);
        let repaint_requested = output
            .viewport_output
            .get(&ViewportId::ROOT)
            .is_some_and(|viewport| viewport.repaint_delay.is_zero());
        EguiFrame {
            shapes: output.shapes,
            textures_delta: output.textures_delta,
            pixels_per_point: output.pixels_per_point,
            repaint_requested,
        }
    }

    fn clear(&self, size: PhysicalSize<u32>, color: [f32; 4]) {
        self.painter.clear([size.width, size.height], color);
    }

    fn paint(&mut self, size: PhysicalSize<u32>, mut frame: EguiFrame) {
        let clipped =
            self.ctx.tessellate(frame.shapes, frame.pixels_per_point);
        self.painter.paint_and_update_textures(
            [size.width, size.height],
            frame.pixels_per_point,
            &clipped,
            &mut frame.textures_delta,
        );
    }
}

impl Drop for Egui {
    fn drop(&mut self) {
        // egui_glow reports leaked resources if destroy never runs.
        self.painter.destroy();
    }
}

struct Gl {
    limits: ImageLimits,
    egui: Egui,
    // Must drop last: owns the GL context used by the fields above.
    window: Window,
}

fn display_limits(
    event_loop: &ActiveEventLoop,
    window: &Window,
    max_texture_edge: u32,
) -> ImageLimits {
    let monitor_box = window
        .winit()
        .current_monitor()
        .or_else(|| event_loop.primary_monitor())
        .or_else(|| {
            event_loop.available_monitors().max_by_key(|monitor| {
                let size = monitor.size();
                u64::from(size.width) * u64::from(size.height)
            })
        })
        .map(|monitor| {
            let size = monitor.size();
            (size.width, size.height)
        })
        .filter(|&(width, height)| width > 0 && height > 0);
    let window_size = window.winit().inner_size();
    let window_box = (window_size.width, window_size.height);
    let limits = ImageLimits::new(
        monitor_box.unwrap_or(window_box),
        max_texture_edge,
    );
    tracing::debug!(
        ?monitor_box,
        ?window_box,
        fit_box = ?limits.fit_box(),
        max_texture_edge = limits.max_texture_edge(),
        "display limits"
    );
    limits
}

#[derive(Default)]
enum InputMode {
    #[default]
    Browse,
    Jump(String),
}

impl InputMode {
    fn jump_digits(&self) -> Option<&str> {
        match self {
            Self::Browse => None,
            Self::Jump(digits) => Some(digits.as_str()),
        }
    }
}

#[derive(Default)]
struct PointerState {
    position: Option<(f64, f64)>,
    dragging: bool,
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

fn present_frame(
    gl: &mut Gl,
    view: &mut ViewState,
    size: PhysicalSize<u32>,
    status: &StatusLine<'_>,
    spread: Option<(&SpreadLayout, Option<Spread>)>,
    jump_digits: Option<&str>,
) -> eros::Result<bool> {
    use crate::ui;

    let show_status_bar = gl.window.winit().fullscreen().is_none();
    let jump = jump_digits.and_then(|digits| {
        let page_count = status.reading.as_ref()?.page_count;
        Some((
            digits,
            page_count,
            page_number(digits, page_count).is_some(),
        ))
    });

    let scale_factor = gl.window.winit().scale_factor();
    let mut page_rect = Rect::NOTHING;
    let frame = gl.egui.layout(size, scale_factor, |ui| {
        page_rect = ui::show(ui, status, show_status_bar);
        if let Some((digits, page_count, valid)) = jump {
            ui::page_jump(ui.ctx(), page_rect, digits, page_count, valid);
        }
    });

    let viewport =
        content_viewport(size, page_rect, frame.pixels_per_point);
    // The bar can change height without a resize; refit immediately and
    // request one follow-up frame so the status scale matches.
    let viewport_changed = viewport != view.last_viewport;
    if viewport_changed {
        view.fit.refit();
        view.last_viewport = viewport;
    }

    let placement = spread
        .as_ref()
        .map(|(layout, _)| view.fit.resolve(layout.total(), viewport));

    gl.egui.clear(size, clear_color(gl.window.translucent()));
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
    let followup_redraw = viewport_changed || frame.repaint_requested;
    gl.egui.paint(size, frame);
    gl.window.pre_present_and_swap()?;
    Ok(followup_redraw)
}

struct App {
    book: BookSlot,
    child: ChildProcess,
    gl: Option<Gl>,
    fatal_error: Option<ErrorUnion>,
    view: ViewState,
    skimming_keys: BTreeSet<KeyCode>,
    input_mode: InputMode,
    pointer: PointerState,
    needs_redraw: bool,
    /// Held modifier state; winit 0.30 key events carry none, only
    /// `ModifiersChanged` does.
    modifiers: ModifiersState,
    /// Wakes the parked loop from background opens and book readers.
    proxy: EventLoopProxy<Wake>,
}

impl App {
    fn new(
        open_path: &Path,
        land_on: Option<&str>,
        proxy: EventLoopProxy<Wake>,
    ) -> Self {
        let mut app = Self {
            book: BookSlot::Initial(catalog::opening_name(
                open_path, land_on,
            )),
            child: ChildProcess::Stopped,
            gl: None,
            fatal_error: None,
            view: ViewState::default(),
            skimming_keys: BTreeSet::new(),
            input_mode: InputMode::Browse,
            pointer: PointerState::default(),
            needs_redraw: false,
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
            let Ok(()) = proxy.send_event(Wake) else {
                return;
            };
        }) {
            Ok(open) => ChildProcess::Opening { open, land_on },
            Err(fault) => {
                tracing::warn!(%fault, "book open failed");
                self.needs_redraw = true;
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
        self.skimming_keys.clear();
        self.input_mode = InputMode::Browse;
        self.set_dragging(false);
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
        self.needs_redraw = true;
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

    fn set_dragging(&mut self, dragging: bool) {
        use winit::window::CursorIcon;

        if self.pointer.dragging == dragging {
            return;
        }
        self.pointer.dragging = dragging;
        if let Some(gl) = self.gl.as_ref() {
            let icon = if dragging {
                CursorIcon::Move
            } else {
                CursorIcon::Default
            };
            gl.window.winit().set_cursor(icon);
        }
    }

    fn cursor_in_viewport(&self) -> Option<(f64, f64)> {
        let cursor = self.pointer.position?;
        let (width, height) = self.view.last_viewport.size();
        (width > 0
            && height > 0
            && cursor.0 >= 0.0
            && cursor.1 >= 0.0
            && cursor.0 < f64::from(width)
            && cursor.1 < f64::from(height))
        .then_some(cursor)
    }

    fn browsing_cursor(&self) -> Option<(f64, f64)> {
        matches!(self.input_mode, InputMode::Browse)
            .then(|| self.cursor_in_viewport())
            .flatten()
    }

    fn current_spread_size(&self) -> Option<(f64, f64)> {
        let reading = self.book.reading()?;
        let mode = self.view.mode;
        let on_screen = reading.on_screen(mode);
        let dims = reading.seat_dims(mode, on_screen);
        let total = geometry::layout(dims).total();
        (total.0 > 0.0 && total.1 > 0.0).then_some(total)
    }

    fn update_fit(
        &mut self,
        update: impl FnOnce(&mut Fit, (f64, f64), Viewport),
    ) {
        let viewport = self.view.last_viewport;
        let (width, height) = viewport.size();
        if width == 0 || height == 0 {
            return;
        }
        let Some(spread_size) = self.current_spread_size() else {
            return;
        };
        self.view.fit.resolve(spread_size, viewport);
        update(&mut self.view.fit, spread_size, viewport);
        self.needs_redraw = true;
    }

    fn zoom_at(&mut self, factor: f64, cursor: (f64, f64)) {
        if (factor - 1.0).abs() <= f64::EPSILON {
            return;
        }
        self.update_fit(|fit, spread_size, viewport| {
            fit.zoom(factor, cursor, spread_size, viewport);
        });
    }

    fn poll_child(&mut self, limits: ImageLimits) {
        use crate::book::BookEvent;

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
                        self.needs_redraw = true;
                    }
                }
            }
            ChildProcess::Live(process) => {
                let Some(reading) = self.book.reading_mut() else {
                    return;
                };
                let mut died = None;
                let mut failed = false;
                for event in process.drain() {
                    match event {
                        BookEvent::Died(detail) => died = Some(detail),
                        other @ (BookEvent::Ready { .. }
                        | BookEvent::Playback(_)
                        | BookEvent::Failed { .. }) => {
                            if matches!(other, BookEvent::Failed { .. }) {
                                failed = true;
                            }
                            if let Some(command) =
                                reading.apply_event(other)
                            {
                                process.send(ParentMsg::Playback(command));
                            }
                        }
                    }
                }
                if failed {
                    self.needs_redraw = true;
                }
                if let Some(detail) = died {
                    tracing::warn!(%detail, "the book process failed");
                    self.child = ChildProcess::Faulted(format!(
                        "the book process died: {detail}"
                    ));
                    self.needs_redraw = true;
                }
            }
            ChildProcess::Faulted(_) | ChildProcess::Stopped => {}
        }
    }

    fn reconcile(&mut self, event_loop: &ActiveEventLoop) {
        use winit::event_loop::ControlFlow;

        let Some(limits) = self.gl.as_ref().map(|gl| gl.limits) else {
            return;
        };
        self.poll_child(limits);
        let now = Instant::now();

        let Self {
            book,
            child,
            gl,
            view,
            needs_redraw,
            ..
        } = self;
        let Some(gl) = gl.as_mut() else {
            return;
        };

        let (waiting, playback_deadline) = match book {
            BookSlot::Initial(_) => (child.is_opening(), None),
            BookSlot::Loaded(reading) => {
                let touched = drain_uploads(reading, gl);
                let (shown, changed) =
                    reading.resolve_shown(view.mode, now, |page| {
                        gl.window.pages().roles(page)
                    });
                if changed {
                    view.fit.commit();
                    *needs_redraw = true;
                }
                let on_screen = reading.on_screen(view.mode);
                if touched.iter().any(|&page| on_screen.contains(page)) {
                    *needs_redraw = true;
                }
                let dims = reading.seat_dims(view.mode, on_screen);
                let layout = geometry::layout(dims);
                let spread_size = layout.total();
                let placement =
                    view.fit.resolve(spread_size, view.last_viewport);
                if changed {
                    tracing::debug!(
                        ?on_screen,
                        ?shown,
                        total = ?spread_size,
                        ?dims,
                        scale = placement.scale,
                        "placement"
                    );
                }

                let (viewport_width, viewport_height) =
                    view.last_viewport.size();
                let on_screen_scale = (viewport_width > 0
                    && viewport_height > 0)
                    .then_some(placement.scale);
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
                    reading
                        .retains_texture(view.mode, on_screen, page, role)
                });
                for command in
                    reading.sync_playback(view.mode, on_screen, |page| {
                        gl.window.pages().roles(page)
                    })
                {
                    child.send(ParentMsg::Playback(command));
                }
                let playback =
                    reading.tick_playback(now, |page, role, image| {
                        gl.window.pages_mut().update(page, role, image)
                    });
                for command in playback.commands {
                    child.send(ParentMsg::Playback(command));
                }
                if playback.updated {
                    *needs_redraw = true;
                }

                let waiting = reading.is_waiting(on_screen, |page| {
                    gl.window.pages().has_any(page)
                });
                (waiting && child.is_live(), playback.next_deadline)
            }
        };
        let timers = view.timers.tick(now, waiting);
        if timers.changed {
            *needs_redraw = true;
        }
        let deadline = [playback_deadline, timers.next_deadline]
            .into_iter()
            .flatten()
            .min();

        if *needs_redraw {
            gl.window.winit().request_redraw();
        }
        event_loop.set_control_flow(
            deadline.map_or(ControlFlow::Wait, ControlFlow::WaitUntil),
        );
    }

    fn draw_frame(&mut self) -> eros::Result<()> {
        use std::time::Duration;

        const SLOW_FRAME_THRESHOLD: Duration = Duration::from_millis(50);

        self.needs_redraw = false;
        let frame_started = Instant::now();
        let Some(size) = self.sync_surface_size() else {
            return Ok(());
        };
        let Self {
            book,
            child,
            gl,
            view,
            input_mode,
            needs_redraw,
            ..
        } = self;
        let Some(gl) = gl.as_mut() else {
            return Ok(());
        };
        let followup_redraw = match book {
            BookSlot::Initial(name) => {
                let status = StatusLine {
                    reading: None,
                    mode: view.mode,
                    toast: view.timers.is_toasting(),
                    name: name.clone(),
                    waiting: view.timers.is_waiting()
                        && child.is_opening(),
                    fault: child.fault(),
                };
                present_frame(gl, view, size, &status, None, None)?
            }
            BookSlot::Loaded(reading) => {
                let on_screen = reading.on_screen(view.mode);
                let dims = reading.seat_dims(view.mode, on_screen);
                let layout = geometry::layout(dims);
                let placement =
                    view.fit.resolve(layout.total(), view.last_viewport);
                let mut status = reading.status_line(
                    on_screen,
                    view.mode,
                    placement.scale,
                    child.fault(),
                    |page| gl.window.pages().has_any(page),
                );
                status.waiting =
                    status.waiting && view.timers.is_waiting();
                status.toast = view.timers.is_toasting();
                present_frame(
                    gl,
                    view,
                    size,
                    &status,
                    Some((&layout, reading.shown())),
                    input_mode.jump_digits(),
                )?
            }
        };
        if followup_redraw {
            *needs_redraw = true;
        }
        let total = frame_started.elapsed();
        if total >= SLOW_FRAME_THRESHOLD {
            tracing::info!(
                total = ?total,
                shown = ?book.reading().and_then(Reading::shown),
                "slow frame"
            );
        }
        Ok(())
    }

    fn sync_motion(&mut self) {
        use crate::browse::Motion;

        let mode = self.view.mode;
        let Some(reading) = self.book.reading_mut() else {
            return;
        };
        let motion = if self.skimming_keys.is_empty() {
            Motion::Still
        } else {
            Motion::Skimming
        };
        let before = reading.target();
        if !reading.set_motion(motion, mode) {
            return;
        }
        if reading.target() != before {
            self.set_view();
        }
        self.child.send(ParentMsg::SetMotion(motion));
    }

    fn clear_skimming_keys(&mut self) {
        self.skimming_keys.clear();
        self.sync_motion();
    }

    fn on_key(
        &mut self,
        key: &KeyEvent,
        is_synthetic: bool,
        event_loop: &ActiveEventLoop,
    ) {
        if is_synthetic && key.state == ElementState::Pressed {
            return;
        }
        match key.state {
            ElementState::Pressed => {
                if let InputMode::Jump(digits) = &mut self.input_mode {
                    let Some(page_count) = self
                        .book
                        .reading()
                        .map(|reading| reading.page_count().get())
                    else {
                        return;
                    };
                    match step_jump(
                        digits,
                        key,
                        self.modifiers,
                        page_count,
                    ) {
                        JumpOutcome::Ignored => {}
                        JumpOutcome::Edited => self.needs_redraw = true,
                        JumpOutcome::Canceled => {
                            self.input_mode = InputMode::Browse;
                            self.needs_redraw = true;
                        }
                        JumpOutcome::Committed(target_page) => {
                            self.input_mode = InputMode::Browse;
                            let mode = self.view.mode;
                            if let Some(reading) = self.book.reading_mut()
                                && reading.jump_to_page(
                                    target_page,
                                    mode,
                                    Instant::now(),
                                )
                            {
                                self.after_page_change();
                            }
                            self.needs_redraw = true;
                        }
                    }
                    return;
                }
                if !key.repeat
                    && !self.modifiers.control_key()
                    && !self.modifiers.alt_key()
                    && !self.modifiers.super_key()
                    && key.text.as_deref() == Some(":")
                {
                    if self.book.reading().is_some() {
                        self.clear_skimming_keys();
                        self.set_dragging(false);
                        self.input_mode = InputMode::Jump(String::new());
                        self.needs_redraw = true;
                    }
                    return;
                }
                let PhysicalKey::Code(code) = key.physical_key else {
                    return;
                };
                let Some(action) = Action::from_key(code, self.modifiers)
                else {
                    return;
                };
                if key.repeat && !action.repeats() {
                    return;
                }
                if action.repeats() && key.repeat {
                    self.skimming_keys.insert(code);
                    self.sync_motion();
                }
                self.dispatch(action, event_loop);
            }
            ElementState::Released => {
                let PhysicalKey::Code(code) = key.physical_key else {
                    return;
                };
                if self.skimming_keys.remove(&code) {
                    self.sync_motion();
                }
            }
        }
    }

    fn dispatch(&mut self, action: Action, event_loop: &ActiveEventLoop) {
        match action {
            Action::Quit => event_loop.exit(),
            Action::Escape => {
                if self.is_fullscreen() {
                    if let Some(gl) = self.gl.as_ref() {
                        gl.window.winit().set_fullscreen(None);
                    }
                    self.needs_redraw = true;
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
                self.needs_redraw = true;
            }
            Action::SelectFit(mode) => {
                self.view.fit.select(mode);
                self.needs_redraw = true;
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
                self.needs_redraw = true;
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

    fn set_view(&self) {
        let Some(reading) = self.book.reading() else {
            return;
        };
        self.child.send(ParentMsg::SetView {
            page: reading.target(),
            mode: self.view.mode,
            direction: reading.direction(),
        });
    }

    fn after_page_change(&mut self) {
        self.set_view();
        self.view.fit.begin_turn();
    }
}

enum JumpOutcome {
    Ignored,
    Edited,
    Canceled,
    Committed(NonZeroUsize),
}

fn step_jump(
    digits: &mut String,
    key: &KeyEvent,
    modifiers: ModifiersState,
    page_count: usize,
) -> JumpOutcome {
    if modifiers.control_key()
        || modifiers.alt_key()
        || modifiers.super_key()
    {
        return JumpOutcome::Ignored;
    }

    // Digits and Backspace repeat at the compositor's rate.
    let digit = key.text.as_deref().and_then(|text| {
        let [byte] = text.as_bytes() else {
            return None;
        };
        byte.is_ascii_digit().then(|| char::from(*byte))
    });
    if let Some(digit) = digit {
        digits.push(digit);
        return JumpOutcome::Edited;
    }

    let PhysicalKey::Code(code) = key.physical_key else {
        return JumpOutcome::Ignored;
    };
    if code == KeyCode::Backspace {
        digits.pop();
        return JumpOutcome::Edited;
    }
    if key.repeat {
        return JumpOutcome::Ignored;
    }
    if matches!(code, KeyCode::Enter | KeyCode::NumpadEnter) {
        if digits.is_empty() {
            JumpOutcome::Canceled
        } else if let Some(target) = page_number(digits, page_count) {
            JumpOutcome::Committed(target)
        } else {
            JumpOutcome::Ignored
        }
    } else if code == KeyCode::Escape {
        JumpOutcome::Canceled
    } else {
        JumpOutcome::Ignored
    }
}

fn wheel_zoom_factor(delta: MouseScrollDelta, scale_factor: f64) -> f64 {
    const POINTS_PER_LINE: f64 = 40.0;
    const WHEEL_ZOOM_PER_LINE: f64 = 0.2;

    let wheel_lines = match delta {
        MouseScrollDelta::LineDelta(_, y) => f64::from(y),
        MouseScrollDelta::PixelDelta(pixels) => {
            pixels.y / scale_factor / POINTS_PER_LINE
        }
    };
    (wheel_lines * WHEEL_ZOOM_PER_LINE).exp()
}

fn drain_uploads(reading: &mut Reading, gl: &mut Gl) -> Vec<usize> {
    use std::time::Duration;

    const SLOW_UPLOAD_THRESHOLD: Duration = Duration::from_millis(30);

    let pending = reading.take_pending_uploads();
    if pending.is_empty() {
        return Vec::new();
    }
    let started = Instant::now();
    let mut uploaded_count = 0_usize;
    let mut uploaded_bytes = 0_usize;
    let mut touched = Vec::with_capacity(pending.len());
    let textures = gl.window.pages_mut();
    for upload in pending {
        let image_bytes = upload.image.resident_bytes();
        let upload_started = Instant::now();
        touched.push(upload.page);
        match textures.insert(upload.page, upload.role, &upload.image) {
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
    touched
}

impl ApplicationHandler<Wake> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
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
        let panel_fill = window.translucent().then_some(GLASS_FILL);
        let egui = match Egui::new(window.glow(), panel_fill) {
            Ok(egui) => egui,
            Err(error) => {
                self.fail(error, event_loop);
                return;
            }
        };
        self.needs_redraw = true;
        self.gl = Some(Gl {
            limits,
            egui,
            window,
        });
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: Wake) {
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.reconcile(event_loop);
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
        use winit::event::MouseButton;

        match event {
            WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                self.sync_surface_size();
                self.needs_redraw = true;
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                self.needs_redraw = true;
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
            }
            WindowEvent::KeyboardInput {
                event: key,
                is_synthetic,
                ..
            } => {
                self.on_key(&key, is_synthetic, event_loop);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current = (position.x, position.y);
                if let Some(previous) =
                    self.pointer.position.replace(current)
                    && self.pointer.dragging
                    && matches!(self.input_mode, InputMode::Browse)
                {
                    let delta =
                        (current.0 - previous.0, current.1 - previous.1);
                    if delta != (0.0, 0.0) {
                        self.update_fit(|fit, spread_size, viewport| {
                            fit.pan(delta, spread_size, viewport);
                        });
                    }
                }
            }
            WindowEvent::CursorLeft { .. } => {
                self.pointer.position = None;
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => match state {
                ElementState::Pressed => {
                    if self.book.reading().is_some()
                        && self.browsing_cursor().is_some()
                    {
                        self.set_dragging(true);
                    }
                }
                ElementState::Released => self.set_dragging(false),
            },
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(cursor) = self.browsing_cursor()
                    && let Some(gl) = self.gl.as_ref()
                {
                    let factor = wheel_zoom_factor(
                        delta,
                        gl.window.winit().scale_factor(),
                    );
                    self.zoom_at(factor, cursor);
                }
            }
            WindowEvent::PinchGesture { delta, .. } => {
                if let Some(cursor) = self.browsing_cursor() {
                    self.zoom_at(delta.exp(), cursor);
                }
            }
            WindowEvent::Focused(false) => {
                self.clear_skimming_keys();
                self.set_dragging(false);
                self.modifiers = ModifiersState::empty();
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.draw_frame() {
                    self.fail(error, event_loop);
                }
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
    let color = if translucent { GLASS_FILL } else { OPAQUE_FILL };
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

#[cfg(test)]
mod tests {
    use super::*;

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
        use egui::RawInput;
        use egui::Theme;

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
        use egui::Pos2;

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
        use strum::VariantArray as _;

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
    use crate::decode;

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
    use winit::event_loop::EventLoop;

    // Build the loop and fork the child here so `PR_SET_PDEATHSIG` tracks
    // the event-loop thread.
    let event_loop = EventLoop::<Wake>::with_user_event().build()?;
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

    #[test]
    fn classify_argument_kinds() {
        use std::env;
        use std::fs;
        use std::os::unix::fs as unix_fs;
        use tempfile::TempDir;

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
