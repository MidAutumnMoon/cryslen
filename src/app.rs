//! App: winit event handling, state transitions, frame composition.

use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::StartCause;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::ControlFlow;
use winit::event_loop::EventLoopProxy;
use winit::keyboard::ModifiersState;
use winit::window::WindowId;

use crate::decode::{Decoded, Format};
use crate::keys::Action;
use crate::render::PageTextures;
use crate::source::Source;
use crate::view;
use crate::view::{Placement, Seat, Spread, ViewMode, Viewport};
use crate::window::Window;
use crate::worker::{SeatView, SpreadView, Worker};

/// User events that wake the parked loop.
pub enum UserEvent {
    /// Arms a wake; `new_events` requests the redraw at the deadline.
    Repaint(Duration),
    PagesUpdated,
}

/// A failed send means the loop is gone, leaving the event with no consumer.
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is the crate's visibility style for crate-wide items, even inside private modules"
)]
pub(crate) fn post(proxy: &EventLoopProxy<UserEvent>, event: UserEvent) {
    let Ok(()) = proxy.send_event(event) else { return; };
}

struct State {
    index: usize,
    mode: ViewMode,
    placement: Placement,
}

const DEFAULT_STATE: State = State {
    index: 0,
    mode: ViewMode::Normal,
    placement: Placement { scale: 1.0, offset: (0.0, 0.0) },
};

struct FoldedStatus {
    waiting: bool,
    /// First failure among the spread's pages, in reading order.
    error: Option<String>,
    /// The first ready page before any failure.
    page: Option<(usize, Format, (u32, u32))>,
}

/// Fold the spread's pages into the status line's error and page facts
/// in reading order: `Spread.first` (the page read first), then its
/// partner.
fn fold_status(spread: Spread, snapshot: &SpreadView) -> FoldedStatus {
    let mut folded = FoldedStatus {
        waiting: matches!(snapshot.first, SeatView::Pending)
            || matches!(snapshot.second.as_ref(), Some(SeatView::Pending)),
        error: None,
        page: None,
    };
    let pages = [
        (Some(spread.first), Some(&snapshot.first)),
        (spread.second, snapshot.second.as_ref()),
    ];
    for (page, view) in pages.into_iter().filter_map(|(page, view)| page.zip(view)) {
        match view {
            SeatView::Failed(message) => {
                folded.error = Some(message.clone());
                break;
            }
            SeatView::Ready(decoded) if folded.page.is_none() => {
                folded.page = Some((page, decoded.format, (decoded.width, decoded.height)));
            }
            SeatView::Ready(_) | SeatView::Pending => {}
        }
    }
    folded
}

enum SeatStep<'view> {
    Show { page: usize, decoded: &'view Arc<Decoded> },
    Hold,
    Clear,
}

struct ScreenPlan<'view> {
    steps: [SeatStep<'view>; 2],
    shown_after: Option<Spread>,
    committed: bool,
    /// The ready page staged while its partner is undecided.
    stage: Option<(usize, &'view Arc<Decoded>)>,
}

fn seat_step(fact: Option<(usize, &SeatView)>, bound: Option<usize>) -> SeatStep<'_> {
    match fact {
        Some((page, SeatView::Ready(decoded))) => SeatStep::Show { page, decoded },
        // A pending page holds while the seat is empty or shows that page.
        Some((page, SeatView::Pending)) if bound.is_none_or(|bound_page| bound_page == page) => {
            SeatStep::Hold
        }
        None | Some((_, SeatView::Failed(_) | SeatView::Pending)) => SeatStep::Clear,
    }
}

/// The screen policy: a spread change is held while any wanted seat is
/// undecided; once all are decided, all seats change in one frame.
/// Before anything has been shown, ready seats appear as they arrive.
/// `bound[i]` is the page seat i currently shows.
fn plan_screen(
    shown: Option<Spread>,
    wanted: Spread,
    mode: ViewMode,
    seat_views: [Option<&SeatView>; 2],
    bound: [Option<usize>; 2],
) -> ScreenPlan<'_> {
    let [wanted_left, wanted_right] = wanted.seats(mode);
    let [view_left, view_right] = seat_views;
    let [bound_left, bound_right] = bound;
    let seat_facts = [wanted_left.zip(view_left), wanted_right.zip(view_right)];
    let [fact_left, fact_right] = seat_facts;
    let undecided =
        seat_facts.into_iter().any(|fact| matches!(fact, Some((_, SeatView::Pending))));
    if undecided && shown.is_some_and(|shown| shown != wanted) {
        let stage = seat_facts.into_iter().find_map(|fact| match fact {
            Some((page, SeatView::Ready(decoded))) => Some((page, decoded)),
            _ => None,
        });
        return ScreenPlan {
            steps: [SeatStep::Hold, SeatStep::Hold],
            shown_after: shown,
            committed: false,
            stage,
        };
    }
    let steps = [seat_step(fact_left, bound_left), seat_step(fact_right, bound_right)];
    #[expect(
        clippy::rest_pattern_accessible_field,
        reason = "the matched fields are unused here, so naming them would trip unneeded_field_pattern"
    )]
    let any_shown = steps.iter().any(|step| matches!(step, SeatStep::Show { .. }));
    // An all-failed first spread shows nothing, so `shown` stays None.
    let shown_after = (shown.is_some() || any_shown).then_some(wanted);
    ScreenPlan {
        steps,
        shown_after,
        committed: shown_after != shown,
        stage: None,
    }
}

#[derive(Default)]
struct Fit {
    /// The spread total the last refit decision settled; any other total
    /// demands a refit.
    settled: Option<(f32, f32)>,
    /// The user zoomed or panned since the last turn, viewport change, or reset.
    user_moved: bool,
}

impl Fit {
    fn should_refit(&self, total: (f32, f32)) -> bool {
        self.settled != Some(total)
    }

    fn settle(&mut self, total: (f32, f32)) {
        self.settled = Some(total);
    }

    fn note_user_move(&mut self) {
        self.user_moved = true;
    }

    /// A turn or re-pairing mode switch: pre-turn user moves must not
    /// gate the next commit's fit.
    fn begin_turn(&mut self) {
        self.user_moved = false;
    }

    fn committed(&mut self) {
        self.settled = None;
    }

    fn viewport_changed(&mut self) {
        self.refit();
    }

    fn reset_view(&mut self) {
        self.refit();
    }

    fn refit(&mut self) {
        self.settled = None;
        self.user_moved = false;
    }
}

/// The window owns the surface and context, so it is declared last to
/// outlive pages and egui.
struct Gl {
    pages: PageTextures,
    egui: egui_glow::EguiGlow,
    window: Window,
}

impl Drop for Gl {
    fn drop(&mut self) {
        // egui's GL objects must be destroyed while the window's context is
        // alive; egui_glow reports leaked resources if destroy never runs.
        self.egui.destroy();
    }
}

#[derive(Clone, Copy)]
struct SeatShow {
    page: usize,
    dims: (u32, u32),
}

pub struct App {
    source: Arc<Source>,
    worker: Worker,
    state: State,
    gl: Option<Gl>,
    fatal_error: Option<eros::ErrorUnion>,
    shown_seats: [Option<SeatShow>; 2],
    /// The last spread the screen plan committed; None before the first
    /// commit (cold start).
    shown: Option<Spread>,
    /// The spread the worker was last tuned to; `shown` lags until the
    /// plan commits it.
    retuned_spread: Spread,
    /// Cursor position in viewport px, same as window px.
    cursor: (f32, f32),
    dragging: bool,
    /// Last `ModifiersChanged` state; winit 0.30 key events carry none.
    modifiers: ModifiersState,
    waiting_since: Option<Instant>,
    mode_toast_since: Option<Instant>,
    /// The viewport the last frame painted into, read by input between frames.
    last_viewport: Viewport,
    fit: Fit,
    /// Wakes the parked loop from egui and timed waits; winit 0.30 cannot
    /// create a proxy from `ActiveEventLoop`.
    proxy: EventLoopProxy<UserEvent>,
}

impl App {
    pub fn new(
        source: Arc<Source>,
        worker: Worker,
        index: usize,
        proxy: EventLoopProxy<UserEvent>,
    ) -> Self {
        let state = State { index, ..DEFAULT_STATE };
        let spread = view::spread_for(state.index, state.mode, source.len());
        Self {
            source,
            worker,
            state,
            gl: None,
            fatal_error: None,
            shown_seats: [None, None],
            shown: None,
            retuned_spread: spread,
            cursor: (0.0, 0.0),
            dragging: false,
            modifiers: ModifiersState::empty(),
            waiting_since: None,
            mode_toast_since: None,
            last_viewport: Viewport::default(),
            fit: Fit::default(),
            proxy,
        }
    }

    /// Store `error` and exit so `main` can report it after `run_app`
    /// returns.
    fn fail(&mut self, error: eros::ErrorUnion, event_loop: &ActiveEventLoop) {
        self.fatal_error = Some(error);
        event_loop.exit();
    }

    pub fn take_fatal_error(&mut self) -> Option<eros::ErrorUnion> {
        self.fatal_error.take()
    }

    fn spread(&self) -> Spread {
        view::spread_for(self.state.index, self.state.mode, self.source.len())
    }

    fn seat_dims(&self) -> [Option<(u32, u32)>; 2] {
        self.shown_seats.each_ref().map(|seat| seat.map(|show| show.dims))
    }

    /// Fit `total` into `viewport` when due. A user zoom or pan wins over
    /// the shape-triggered refit.
    fn refit_into(&mut self, total: (f32, f32), viewport: Viewport) {
        if self.fit.should_refit(total) {
            if !self.fit.user_moved {
                self.state.placement = view::fit(total, viewport);
            }
            self.fit.settle(total);
        }
    }

    fn display(&mut self, spread: Spread, snapshot: &SpreadView) {
        let bound = self.shown_seats.each_ref().map(|seat| seat.map(|show| show.page));
        let plan = plan_screen(
            self.shown,
            spread,
            self.state.mode,
            snapshot.seats(self.state.mode),
            bound,
        );
        for ((seat, shown), step) in Seat::ALL
            .into_iter()
            .zip(self.shown_seats.each_mut())
            .zip(plan.steps)
        {
            match step {
                SeatStep::Show { page, decoded } => {
                    if let Some(gl) = self.gl.as_mut() {
                        gl.pages.bind(seat, page, decoded);
                    }
                    *shown = Some(SeatShow { page, dims: (decoded.width, decoded.height) });
                }
                SeatStep::Hold => {}
                SeatStep::Clear => {
                    *shown = None;
                    if let Some(gl) = self.gl.as_mut() {
                        gl.pages.clear(seat);
                    }
                }
            }
        }
        if let Some((page, decoded)) = plan.stage {
            if let Some(gl) = self.gl.as_mut() {
                gl.pages.stage(page, decoded);
            }
        }
        self.shown = plan.shown_after;
        if plan.committed {
            self.fit.committed();
        }
    }

    fn frame(&mut self) -> eros::Result<()> {
        use crate::decode;
        use crate::ui;
        use crate::ui::PageFacts;
        use crate::ui::StatusLine;

        let spread = self.spread();
        if spread != self.retuned_spread {
            self.retuned_spread = spread;
            self.worker.retune(spread);
        }

        let Some(size) = self.surface_size() else {
            return Ok(());
        };

        let snapshot = self.worker.snapshot(&spread);
        self.display(spread, &snapshot);

        // Refit before the status line reads the scale, so the bar does
        // not show a superseded zoom.
        let layout = view::layout(self.seat_dims());
        let total = layout.total();
        self.refit_into(total, self.last_viewport);

        let folded = fold_status(spread, &snapshot);
        self.waiting_since = folded
            .waiting
            .then(|| self.waiting_since.unwrap_or_else(Instant::now));
        // The parked loop otherwise never shows the "decoding…" flag.
        if let Some(since) = self.waiting_since {
            let elapsed = since.elapsed();
            if elapsed < WAIT_GRACE {
                post(&self.proxy, UserEvent::Repaint(WAIT_GRACE.saturating_sub(elapsed)));
            }
        }

        // Arm a wake at expiry so the toast disappears without other
        // traffic.
        self.mode_toast_since =
            self.mode_toast_since.filter(|&since| since.elapsed() < MODE_TOAST);
        if let Some(since) = self.mode_toast_since {
            let remaining = MODE_TOAST.saturating_sub(since.elapsed());
            post(&self.proxy, UserEvent::Repaint(remaining));
        }

        // The bar names the page whose facts it shows: the fold's page, else the lead.
        let speaking_page = folded.page.map_or(spread.first, |(page, _, _)| page);
        let page = folded.page.map(|(page, decoded, dims)| PageFacts {
            claimed: self.source.name(page).and_then(decode::format_from_name),
            decoded,
            dims,
        });
        let status = StatusLine {
            total: self.source.len(),
            spread,
            mode: self.state.mode,
            toast: self.mode_toast_since.is_some(),
            name: self.source.page_name(speaking_page),
            page,
            zoom: self.state.placement.scale,
            waiting: self.waiting_since.is_some_and(|since| since.elapsed() >= WAIT_GRACE),
            error: folded.error,
        };

        // `EguiGlow::run` only records shapes, so egui lays out before
        // pages are placed; `paint` later draws the bar above them.
        let viewport = {
            let Some(gl) = self.gl.as_mut() else { return Ok(()) };
            let mut content = egui::Rect::NOTHING;
            gl.egui.run(gl.window.winit(), |ui| content = ui::show(ui, &status));
            content_viewport(size, content, gl.egui.egui_ctx.pixels_per_point())
        };
        // The bar's height changes without a winit resize, so diff every
        // frame; the status line used the old scale, so redraw once.
        if viewport != self.last_viewport {
            self.fit.viewport_changed();
            self.last_viewport = viewport;
            self.request_redraw();
        }
        self.refit_into(total, viewport);

        let Some(gl) = self.gl.as_mut() else { return Ok(()) };

        gl.egui.painter.clear([size.width, size.height], clear_color(gl.window.translucent()));
        gl.pages.draw((size.width, size.height), viewport, &layout, &self.state.placement);
        gl.egui.paint(gl.window.winit());
        gl.window.pre_present_and_swap()?;
        // Reveal after presenting, so the window never maps empty.
        gl.window.winit().set_visible(true);
        Ok(())
    }

    /// Physical surface size, resized to match; None without a window or
    /// with a zero dimension.
    fn surface_size(&mut self) -> Option<PhysicalSize<u32>> {
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

    fn dispatch(&mut self, act: Action, event_loop: &ActiveEventLoop) {
        match act {
            Action::Quit => event_loop.exit(),
            Action::SelectMode(mode) => {
                let switch = view::switch_mode(
                    self.state.mode,
                    mode,
                    self.state.index,
                    self.source.len(),
                );
                self.state.mode = mode;
                self.state.index = switch.index;
                self.mode_toast_since = Some(Instant::now());
                if switch.pairing_changed {
                    self.after_page_change();
                } else {
                    self.request_redraw(); // same pairing: no page turn
                }
            }
            Action::ToggleFullscreen => {
                use winit::window::Fullscreen;

                if let Some(gl) = self.gl.as_ref() {
                    let entering = gl.window.winit().fullscreen().is_none();
                    let fullscreen_request = entering.then_some(Fullscreen::Borderless(None));
                    gl.window.winit().set_fullscreen(fullscreen_request);
                }
            }
            Action::ResetView => {
                self.fit.reset_view();
                self.request_redraw();
            }
            Action::Right | Action::Left | Action::Forward | Action::Back => {
                let forward = act.is_forward(self.state.mode);
                self.state.index = view::advance(
                    self.state.index,
                    forward,
                    self.state.mode,
                    self.source.len(),
                );
                self.after_page_change();
            }
        }
    }

    fn after_page_change(&mut self) {
        self.fit.begin_turn();
        self.request_redraw();
    }
}

#[expect(
    clippy::rest_pattern_accessible_field,
    reason = "the matched fields are unused here, so naming them would trip unneeded_field_pattern"
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
        let egui = egui_glow::EguiGlow::new(event_loop, Arc::clone(&gl), None, None, true);
        // System fonts cover filename characters the built-in fonts cannot render.
        egui.egui_ctx
            .add_font_provider(Arc::new(egui_system_font_provider::SystemFontProvider::new()));
        {
            use parking_lot::Mutex;

            let proxy = Mutex::new(self.proxy.clone());
            egui.egui_ctx.set_request_repaint_callback(move |info| {
                post(&proxy.lock(), UserEvent::Repaint(info.delay));
            });
        }
        // Opaque mode keeps egui's default panel fill.
        if window.translucent() {
            egui.egui_ctx.all_styles_mut(|style| style.visuals.panel_fill = glass_fill());
        }
        window.winit().request_redraw();
        self.gl = Some(Gl { pages, egui, window });
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Repaint(delay) => {
                let Some(deadline) = Instant::now().checked_add(delay) else { return };
                // Never delay an earlier armed wake: keep the sooner
                // deadline.
                if let ControlFlow::WaitUntil(armed) = event_loop.control_flow() {
                    if armed <= deadline {
                        return;
                    }
                }
                event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
            }
            UserEvent::PagesUpdated => self.request_redraw(),
        }
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, cause: StartCause) {
        if let StartCause::ResumeTimeReached { .. } = cause {
            self.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.gl = None;
    }

    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "winit's unhandled WindowEvent variants need no work"
    )]
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
        use winit::event::MouseButton;
        use winit::keyboard::PhysicalKey;

        if matches!(event, WindowEvent::CloseRequested | WindowEvent::Destroyed) {
            event_loop.exit();
        }
        let Some(gl) = self.gl.as_mut() else { return };

        let egui_wants_pointer = {
            let response = gl.egui.on_window_event(gl.window.winit(), &event);
            let wants = gl.egui.egui_ctx.egui_wants_pointer_input();
            if response.repaint {
                gl.window.winit().request_redraw();
            }
            wants
        };

        match &event {
            WindowEvent::Resized(_) => {
                // The refit happens in `frame()` when the viewport notices the new size.
                self.surface_size();
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
                        if let Some(act) = Action::from_key(code, self.modifiers) {
                            if !key.repeat || act.repeats() {
                                self.dispatch(act, event_loop);
                            }
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_wants_pointer => {
                let wheel_delta_y = match delta {
                    MouseScrollDelta::LineDelta(_, lines) => *lines,
                    MouseScrollDelta::PixelDelta(pixels) => {
                        wheel_lines(pixels.y, gl.window.winit().scale_factor())
                    }
                };
                let (total, viewport) = self.geometry();
                // Scroll up (positive dy) zooms in.
                let zoom_factor = (wheel_delta_y * WHEEL_ZOOM_SENSITIVITY).exp();
                let zoomed = view::zoom_at(self.state.placement, self.cursor, zoom_factor);
                self.state.placement = view::clamp(zoomed, total, viewport);
                self.fit.note_user_move();
                self.request_redraw();
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.dragging = *state == ElementState::Pressed && !egui_wants_pointer;
            }
            WindowEvent::CursorMoved { position, .. } => {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "cursor positions are physical px; f32 keeps subpixel precision at window sizes"
                )]
                let cursor_pos = (position.x as f32, position.y as f32);
                if self.dragging && !egui_wants_pointer {
                    let (total, viewport) = self.geometry();
                    let pan_delta =
                        (cursor_pos.0 - self.cursor.0, cursor_pos.1 - self.cursor.1);
                    let panned = view::pan(self.state.placement, pan_delta);
                    self.state.placement = view::clamp(panned, total, viewport);
                    self.fit.note_user_move();
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

/// How long a page may stay undecided before the status line says so.
const WAIT_GRACE: Duration = Duration::from_millis(150);

/// How long a mode switch's toast stays on screen.
const MODE_TOAST: Duration = Duration::from_millis(1500);

/// Window background color, straight channels.
const BACKGROUND: [f32; 3] = [0.08, 0.08, 0.09];

/// Background opacity when the window is translucent.
const GLASS_ALPHA: f32 = 0.8;

/// Framebuffer clear color, premultiplied because Wayland composites
/// ARGB buffers that way.
fn clear_color(translucent: bool) -> [f32; 4] {
    let [red, green, blue] = BACKGROUND;
    if translucent {
        [red * GLASS_ALPHA, green * GLASS_ALPHA, blue * GLASS_ALPHA, GLASS_ALPHA]
    } else {
        [red, green, blue, 1.0]
    }
}

/// The page viewport from the chrome egui just laid out: the full-width
/// window area above the status bar, in physical px. Uses egui's
/// `pixels_per_point`, not the window's scale factor.
fn content_viewport(window: PhysicalSize<u32>, content: egui::Rect, pixels_per_point: f32) -> Viewport {
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
    egui::Color32::from_rgba_premultiplied(channel(red), channel(green), channel(blue), channel(alpha))
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
mod tests {
    use super::*;
    use crate::testkit::ready_seat;

    #[test]
    fn fit_lifecycle() {
        let mut fit = Fit::default();
        assert!(fit.should_refit((100.0, 200.0)), "default demands a refit");
        fit.settle((100.0, 200.0));
        assert!(!fit.should_refit((100.0, 200.0)), "the settled total needs no refit");
        assert!(fit.should_refit((100.0, 201.0)), "a new total demands a refit");
        // A late-decoded page changes the shape after the user moved.
        fit.note_user_move();
        fit.settle((120.0, 200.0));
        assert!(fit.user_moved, "a user move survives a later settle");
        let discards = [Fit::viewport_changed, Fit::reset_view];
        let names = ["viewport change", "reset view"];
        for (discard, name) in discards.into_iter().zip(names) {
            fit.settle((100.0, 200.0));
            fit.note_user_move();
            discard(&mut fit);
            assert!(fit.should_refit((100.0, 200.0)), "{name} discards the settle");
            assert!(!fit.user_moved, "{name} discards the user move");
        }
        fit.settle((100.0, 200.0));
        fit.note_user_move();
        fit.begin_turn();
        assert!(!fit.should_refit((100.0, 200.0)), "begin_turn keeps the held frame frozen");
        assert!(!fit.user_moved, "begin_turn discards pre-turn moves");
        fit.note_user_move();
        fit.committed();
        assert!(fit.should_refit((100.0, 200.0)), "committed demands a refit");
        assert!(fit.user_moved, "committed keeps hold-time moves");
    }

    #[test]
    fn fold_status_table() {
        struct Expect {
            waiting: bool,
            error: Option<&'static str>,
            page: Option<(usize, Format, (u32, u32))>,
        }
        let pair = Spread { first: 0, second: Some(1) };
        let spread_view = |first: SeatView, second: Option<SeatView>| SpreadView { first, second };
        let rows = [
            // a decided pair never waits; the trailing failure still reports
            (pair, spread_view(ready_seat(2, 2), Some(SeatView::Failed("x".into()))),
                Expect { waiting: false, error: Some("x"), page: Some((0, Format::Png, (2, 2))) }),
            // a pending lead waits and yields its place to the partner
            (pair, spread_view(SeatView::Pending, Some(ready_seat(2, 2))),
                Expect { waiting: true, error: None, page: Some((1, Format::Png, (2, 2))) }),
            // a lone page
            (Spread::lone(0), spread_view(ready_seat(2, 2), None),
                Expect { waiting: false, error: None, page: Some((0, Format::Png, (2, 2))) }),
            // the first failure in reading order wins
            (pair, spread_view(SeatView::Failed("first".into()), Some(SeatView::Failed("second".into()))),
                Expect { waiting: false, error: Some("first"), page: None }),
            // both ready: the lead speaks
            (pair, spread_view(ready_seat(800, 600), Some(ready_seat(640, 480))),
                Expect { waiting: false, error: None, page: Some((0, Format::Png, (800, 600))) }),
            // a ready lead speaks before the trailing failure stops the fold
            (pair, spread_view(ready_seat(800, 600), Some(SeatView::Failed("second".into()))),
                Expect { waiting: false, error: Some("second"), page: Some((0, Format::Png, (800, 600))) }),
            // a failed lead stops the fold before the partner can speak
            (pair, spread_view(SeatView::Failed("first".into()), Some(ready_seat(640, 480))),
                Expect { waiting: false, error: Some("first"), page: None }),
            // a pending page behind a failure still waits
            (pair, spread_view(SeatView::Pending, Some(SeatView::Failed("x".into()))),
                Expect { waiting: true, error: Some("x"), page: None }),
            // a lone undecided page waits and speaks nothing
            (Spread::lone(0), spread_view(SeatView::Pending, None),
                Expect { waiting: true, error: None, page: None }),
        ];
        for (i, (spread, snapshot, want)) in rows.iter().enumerate() {
            let folded = fold_status(*spread, snapshot);
            assert_eq!(folded.waiting, want.waiting, "row {i}: waiting");
            assert_eq!(folded.error.as_deref(), want.error, "row {i}: error");
            assert_eq!(folded.page, want.page, "row {i}: page");
        }
    }

    #[test]
    fn plan_screen_table() {
        #[derive(Debug, PartialEq)]
        enum StepKind {
            Show,
            Hold,
            Clear,
        }
        type StepFacts = (StepKind, Option<(usize, (u32, u32))>);
        fn show(page: usize, dims: (u32, u32)) -> StepFacts {
            (StepKind::Show, Some((page, dims)))
        }
        const HOLD: StepFacts = (StepKind::Hold, None);
        const CLEAR: StepFacts = (StepKind::Clear, None);
        fn step_facts(step: &SeatStep<'_>) -> StepFacts {
            match step {
                SeatStep::Show { page, decoded } => {
                    (StepKind::Show, Some((*page, (decoded.width, decoded.height))))
                }
                SeatStep::Hold => (StepKind::Hold, None),
                SeatStep::Clear => (StepKind::Clear, None),
            }
        }
        struct Row {
            name: &'static str,
            shown: Option<Spread>,
            wanted: Spread,
            mode: ViewMode,
            views: [Option<SeatView>; 2],
            /// The page each seat currently shows, [left, right].
            bound: [Option<usize>; 2],
            steps: [StepFacts; 2],
            shown_after: Option<Spread>,
            committed: bool,
            stage: Option<(usize, (u32, u32))>,
        }
        let old = Spread { first: 3, second: Some(4) };
        let pair = Spread { first: 5, second: Some(6) };
        let failed = || Some(SeatView::Failed("x".into()));
        let pending = || Some(SeatView::Pending);
        // Manga seats a pair as [second, first]: left = page 6, right = 5.
        // Comic/Normal seat [first, second]: left = first.
        // `old` and `pair` were committed under Manga: bound [second, first].
        let rows = [
            Row {
                name: "a settled pair commits both seats in one frame (rtl seating)",
                shown: Some(old), wanted: pair, mode: ViewMode::Manga,
                views: [Some(ready_seat(600, 700)), Some(ready_seat(500, 700))],
                bound: [Some(4), Some(3)],
                steps: [show(6, (600, 700)), show(5, (500, 700))],
                shown_after: Some(pair), committed: true, stage: None,
            },
            Row {
                name: "an undecided left page holds every seat and stages the partner",
                shown: Some(old), wanted: pair, mode: ViewMode::Manga,
                views: [pending(), Some(ready_seat(500, 700))],
                bound: [Some(4), Some(3)],
                steps: [HOLD, HOLD],
                shown_after: Some(old), committed: false, stage: Some((5, (500, 700))),
            },
            Row {
                name: "an undecided right page holds every seat and stages the partner",
                shown: Some(old), wanted: pair, mode: ViewMode::Manga,
                views: [Some(ready_seat(600, 700)), pending()],
                bound: [Some(4), Some(3)],
                steps: [HOLD, HOLD],
                shown_after: Some(old), committed: false, stage: Some((6, (600, 700))),
            },
            Row {
                name: "a failed partner commits: the seat clears",
                shown: Some(old), wanted: pair, mode: ViewMode::Manga,
                views: [failed(), Some(ready_seat(500, 700))],
                bound: [Some(4), Some(3)],
                steps: [CLEAR, show(5, (500, 700))],
                shown_after: Some(pair), committed: true, stage: None,
            },
            Row {
                name: "both failed commits to background mid-session",
                shown: Some(old), wanted: pair, mode: ViewMode::Manga,
                views: [failed(), failed()],
                bound: [Some(4), Some(3)],
                steps: [CLEAR, CLEAR],
                shown_after: Some(pair), committed: true, stage: None,
            },
            Row {
                name: "a pending page behind a failure still holds",
                shown: Some(old), wanted: pair, mode: ViewMode::Manga,
                views: [failed(), pending()],
                bound: [Some(4), Some(3)],
                steps: [HOLD, HOLD],
                shown_after: Some(old), committed: false, stage: None,
            },
            Row {
                name: "the shown spread re-read as pending holds invisibly",
                shown: Some(pair), wanted: pair, mode: ViewMode::Manga,
                views: [pending(), Some(ready_seat(500, 700))],
                bound: [Some(6), Some(5)],
                steps: [HOLD, show(5, (500, 700))],
                shown_after: Some(pair), committed: false, stage: None,
            },
            Row {
                name: "a seat left on the wrong page by a mid-hold flip clears",
                shown: Some(pair), wanted: pair, mode: ViewMode::Manga,
                views: [pending(), Some(ready_seat(500, 700))],
                // bound is pair's Comic arrangement: left = 5, right = 6.
                bound: [Some(5), Some(6)],
                steps: [CLEAR, show(5, (500, 700))],
                shown_after: Some(pair), committed: false, stage: None,
            },
            Row {
                name: "a lone Manga page parks right and clears left",
                shown: Some(old), wanted: Spread::lone(7), mode: ViewMode::Manga,
                views: [None, Some(ready_seat(700, 700))],
                bound: [Some(4), Some(3)],
                steps: [CLEAR, show(7, (700, 700))],
                shown_after: Some(Spread::lone(7)), committed: true, stage: None,
            },
            Row {
                name: "ltr seating binds the first page left",
                shown: Some(old), wanted: pair, mode: ViewMode::Comic,
                views: [Some(ready_seat(500, 700)), Some(ready_seat(600, 700))],
                bound: [Some(4), Some(3)],
                steps: [show(5, (500, 700)), show(6, (600, 700))],
                shown_after: Some(pair), committed: true, stage: None,
            },
            Row {
                name: "a Normal turn holds the unwanted seat until commit",
                shown: Some(Spread::lone(2)), wanted: Spread::lone(3),
                mode: ViewMode::Normal,
                views: [pending(), None],
                bound: [Some(2), None],
                steps: [HOLD, HOLD],
                shown_after: Some(Spread::lone(2)), committed: false, stage: None,
            },
            Row {
                name: "a Normal commit clears the unwanted seat in the swap",
                shown: Some(Spread::lone(2)), wanted: Spread::lone(3),
                mode: ViewMode::Normal,
                views: [Some(ready_seat(300, 400)), None],
                bound: [Some(2), None],
                steps: [show(3, (300, 400)), CLEAR],
                shown_after: Some(Spread::lone(3)), committed: true, stage: None,
            },
            Row {
                name: "cold start with nothing decided keeps nothing",
                shown: None, wanted: pair, mode: ViewMode::Manga,
                views: [pending(), pending()],
                bound: [None, None],
                steps: [HOLD, HOLD],
                shown_after: None, committed: false, stage: None,
            },
            Row {
                name: "cold start shows a decided seat at once",
                shown: None, wanted: pair, mode: ViewMode::Manga,
                views: [Some(ready_seat(600, 700)), pending()],
                bound: [None, None],
                steps: [show(6, (600, 700)), HOLD],
                shown_after: Some(pair), committed: true, stage: None,
            },
            Row {
                name: "cold start survives an all-failed first spread",
                shown: None, wanted: pair, mode: ViewMode::Manga,
                views: [failed(), failed()],
                bound: [None, None],
                steps: [CLEAR, CLEAR],
                shown_after: None, committed: false, stage: None,
            },
            Row {
                name: "cold start mixes a cleared failure with a shown page",
                shown: None, wanted: pair, mode: ViewMode::Manga,
                views: [failed(), Some(ready_seat(500, 700))],
                bound: [None, None],
                steps: [CLEAR, show(5, (500, 700))],
                shown_after: Some(pair), committed: true, stage: None,
            },
            Row {
                name: "steady state re-shows the same seats",
                shown: Some(pair), wanted: pair, mode: ViewMode::Manga,
                views: [Some(ready_seat(600, 700)), Some(ready_seat(500, 700))],
                bound: [Some(6), Some(5)],
                steps: [show(6, (600, 700)), show(5, (500, 700))],
                shown_after: Some(pair), committed: false, stage: None,
            },
            Row {
                name: "a direction flip on a decided spread re-mirrors at once",
                shown: Some(pair), wanted: pair, mode: ViewMode::Comic,
                views: [Some(ready_seat(500, 700)), Some(ready_seat(600, 700))],
                bound: [Some(6), Some(5)],
                steps: [show(5, (500, 700)), show(6, (600, 700))],
                shown_after: Some(pair), committed: false, stage: None,
            },
            Row {
                name: "a direction flip mid-hold keeps the old arrangement",
                shown: Some(old), wanted: pair, mode: ViewMode::Comic,
                views: [Some(ready_seat(500, 700)), pending()],
                bound: [Some(4), Some(3)],
                steps: [HOLD, HOLD],
                shown_after: Some(old), committed: false, stage: Some((5, (500, 700))),
            },
        ];
        for Row { name, shown, wanted, mode, views, bound, steps, shown_after, committed, stage } in rows {
            let views = views.each_ref().map(Option::as_ref);
            let plan = plan_screen(shown, wanted, mode, views, bound);
            assert_eq!(plan.steps.each_ref().map(step_facts), steps, "{name}: steps");
            assert_eq!(plan.shown_after, shown_after, "{name}: shown");
            assert_eq!(plan.committed, committed, "{name}: committed");
            assert_eq!(
                plan.stage.map(|(page, decoded)| (page, (decoded.width, decoded.height))),
                stage,
                "{name}: stage"
            );
        }
    }

    #[expect(
        clippy::float_cmp_const,
        reason = "the opaque variant returns the same constants untouched, so equality is exact"
    )]
    #[test]
    fn background_glass() {
        let [red, green, blue, alpha] = clear_color(true);
        assert!((alpha - GLASS_ALPHA).abs() < 1e-6);
        for (channel, straight) in [red, green, blue].into_iter().zip(BACKGROUND) {
            assert!((straight.mul_add(GLASS_ALPHA, -channel)).abs() < 1e-6);
        }
        assert_eq!(clear_color(false), [BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 1.0]);
        // The bar must repeat the translucent background. Exact bytes, so
        // no formula or constructor change can drift them apart silently.
        assert_eq!(glass_fill(), egui::Color32::from_rgba_premultiplied(16, 16, 18, 204));
    }

    #[test]
    fn content_viewport_subtracts_the_bar_in_physical_px() {
        let content = egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1280.0, 772.0));
        let integer = content_viewport(PhysicalSize::new(1280, 800), content, 1.0);
        assert_eq!(integer.size(), (1280.0, 772.0));
        // Fractional output scale: 772 pt at 1.25 pixels per point leaves
        // 965 physical px of content in a 1000 px window.
        let fractional = content_viewport(PhysicalSize::new(1600, 1000), content, 1.25);
        assert_eq!(fractional.size(), (1600.0, 965.0));
    }

    #[test]
    fn pixel_wheel_zoom_speed_does_not_vary_with_output_scale_factor() {
        assert_eq!(wheel_lines(40.0, 1.0), 1.0);
        assert_eq!(wheel_lines(80.0, 2.0), 1.0);
        assert_eq!(wheel_lines(40.0, 2.0), 0.5);
    }
}
