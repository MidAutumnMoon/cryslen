//! App: winit event handling, state transitions, frame composition.
//!
//! State is a single plain struct. Key commands map to an `Action` first
//! (pure, testable), then mutate state.

use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::StartCause;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::ControlFlow;
use winit::event_loop::EventLoopProxy;
use winit::window::WindowId;

use crate::decode::{Decoded, Format};
use crate::keys::Action;
use crate::render::PageTextures;
use crate::source::Source;
use crate::view;
use crate::view::{Placement, Seat, Spread, SpreadLayout, ViewMode, Viewport};
use crate::window::Window;
use crate::worker::{SeatView, SpreadView, Worker};

/// User events that wake the parked loop.
pub enum UserEvent {
    /// Arms a wake only. The redraw fires from `new_events` when the
    /// deadline is reached.
    Repaint(Duration),
    /// The decode book changed.
    PagesUpdated,
}

/// A failed send means the loop is gone, so the event has no consumer
/// left and quitting already won.
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is the crate's visibility style for crate-wide items, even inside private modules"
)]
pub(crate) fn post(proxy: &EventLoopProxy<UserEvent>, event: UserEvent) {
    let Ok(()) = proxy.send_event(event) else { return; };
}

struct State {
    /// Index of the spread's first page (a spread start).
    index: usize,
    mode: ViewMode,
    placement: Placement,
}

/// Baked-in defaults (no config file, ever).
const DEFAULT_STATE: State = State {
    index: 0,
    mode: ViewMode::Normal,
    placement: Placement { scale: 1.0, offset: (0.0, 0.0) },
};

/// Status facts folded over both seats of a spread.
struct FoldedStatus {
    /// A wanted seat is still undecided.
    waiting: bool,
    /// First failure among the seats, in reading order.
    error: Option<String>,
    /// Format and dims of the first Ready seat in reading order.
    page: Option<(Format, (u32, u32))>,
}

/// Fold the seats' views into status facts, visiting the seats in reading
/// order so the spread's lead page speaks. `seat_views[i]` is the worker's
/// view for `spread.seats(mode)[i]`. A failed page stops the fold, because
/// its facts do not exist and the partner's would describe the wrong page.
fn fold_status(spread: Spread, mode: ViewMode, seat_views: &[Option<SeatView>; 2]) -> FoldedStatus {
    use std::iter;

    let waiting = iter::zip(spread.seats(mode), seat_views)
        .any(|(wanted, view)| wanted.is_some() && matches!(view, Some(SeatView::Pending)));
    let mut folded = FoldedStatus { waiting, error: None, page: None };
    // Manga reads the right seat first.
    let [left, right] = seat_views;
    let reading_order = if mode.rtl() { [right, left] } else { [left, right] };
    for seat_view in reading_order {
        match seat_view {
            Some(SeatView::Failed(message)) => {
                folded.error = Some(message.clone());
                break;
            }
            Some(SeatView::Ready(decoded)) if folded.page.is_none() => {
                folded.page = Some((decoded.format, (decoded.width, decoded.height)));
            }
            _ => {}
        }
    }
    folded
}

#[derive(Default)]
struct Fit {
    /// Spread total the refit decision was settled for. None demands a refit.
    settled: Option<(f32, f32)>,
    /// The user zoomed or panned since the last fit.
    user_moved: bool,
}

impl Fit {
    /// Whether the placement must be fitted to `total` this frame.
    fn should_refit(&self, total: (f32, f32)) -> bool {
        self.settled != Some(total)
    }

    /// Settle the refit decision for `total`, fitted or not.
    fn settle(&mut self, total: (f32, f32)) {
        self.settled = Some(total);
    }

    fn note_user_move(&mut self) {
        self.user_moved = true;
    }

    /// The new spread fits afresh once its dims settle. User placement
    /// does not carry across pages.
    fn page_turned(&mut self) {
        self.refit();
    }

    /// Fit afresh when the viewport changed. User placement does not
    /// survive.
    fn viewport_changed(&mut self) {
        self.refit();
    }

    /// Fit afresh, discarding user placement.
    fn reset_view(&mut self) {
        self.refit();
    }

    fn refit(&mut self) {
        self.settled = None;
        self.user_moved = false;
    }
}

/// Everything that exists only between `resumed` and exit (winit
/// requirement). The window owns the GL context, so it drops last,
/// after pages and egui.
struct Gl {
    pages: PageTextures,
    egui: egui_glow::EguiGlow,
    window: Window,
}

impl Drop for Gl {
    fn drop(&mut self) {
        // The drop body runs before the field drops, so egui's GL objects
        // die here while the context is alive. egui_glow logs a leak
        // warning if destroy never runs.
        self.egui.destroy();
    }
}

pub struct App {
    source: Arc<Source>,
    worker: Worker,
    state: State,
    gl: Option<Gl>,
    fatal_error: Option<eros::ErrorUnion>,
    /// What each seat holds, the last frame included. Worker eviction
    /// can drop the book's entry, never the pixels held here.
    held: [Option<Arc<Decoded>>; 2],
    /// The spread last sent to the worker. `frame()` asserts the
    /// on-screen spread equals this, so every spread change must
    /// retune.
    retuned_spread: Spread,
    /// Cursor position in viewport px, equal to window px because the
    /// viewport hugs the window's top-left.
    cursor: (f32, f32),
    dragging: bool,
    /// Set while any wanted seat is undecided. There is no spinner
    /// animation because it would force repaints of the parked loop.
    waiting_since: Option<Instant>,
    /// Set while the mode-switch toast is showing in the status bar.
    mode_toast_since: Option<Instant>,
    /// Layout of the spread currently on screen. Pan and zoom read this
    /// memo between frames.
    last_layout: SpreadLayout,
    /// The viewport the last frame painted into. Pan and zoom read this
    /// memo between frames. `frame()` refreshes it after every egui pass.
    last_viewport: Viewport,
    fit: Fit,
    /// Proxy for egui's repaint bridge. winit 0.30 cannot create one from
    /// `ActiveEventLoop`.
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
        worker.retune(spread);
        Self {
            source,
            worker,
            state,
            gl: None,
            fatal_error: None,
            held: [None, None],
            retuned_spread: spread,
            cursor: (0.0, 0.0),
            dragging: false,
            waiting_since: None,
            mode_toast_since: None,
            last_layout: SpreadLayout::default(),
            last_viewport: Viewport::default(),
            fit: Fit::default(),
            proxy,
        }
    }

    /// Record `error` and leave the event loop. The stored error outlives
    /// the loop so `main` can report it.
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

    /// Fit `total` into `viewport` when a refit is due. A user zoom or
    /// pan wins over the shape-triggered refit.
    fn refit_into(&mut self, total: (f32, f32), viewport: Viewport) {
        if self.fit.should_refit(total) {
            if !self.fit.user_moved {
                self.state.placement = view::fit(total, viewport);
            }
            self.fit.settle(total);
        }
    }

    fn display(&mut self, spread: Spread, snapshot: &SpreadView) -> [Option<SeatView>; 2] {
        #[expect(
            clippy::redundant_closure_for_method_calls,
            reason = "map hands the item owned, and the fn path needs it by reference"
        )]
        let seat_views: [Option<SeatView>; 2] =
            snapshot.seats(self.state.mode).map(|seat| seat.cloned());
        let wanted_pages = spread.seats(self.state.mode);
        for ((seat, wanted), (view, held)) in Seat::ALL
            .into_iter()
            .zip(wanted_pages)
            .zip(seat_views.each_ref().into_iter().zip(self.held.each_mut()))
        {
            #[expect(
                clippy::match_same_arms,
                reason = "the hold arms document different seat states, and merging loses the case map"
            )]
            match (wanted, view) {
                (None, _) => {
                    *held = None;
                    if let Some(gl) = self.gl.as_mut() {
                        gl.pages.clear(seat);
                    }
                }
                (Some(page), Some(SeatView::Ready(decoded))) => {
                    *held = Some(Arc::clone(decoded));
                    if let Some(gl) = self.gl.as_mut() {
                        gl.pages.bind(seat, page, decoded);
                    }
                }
                // The seat holds its last frame.
                (Some(_), Some(SeatView::Pending | SeatView::Failed(_))) => {}
                // A wanted seat always has a view.
                (Some(_), None) => {}
            }
        }
        let held_dims = self
            .held
            .each_ref()
            .map(|seat_held| seat_held.as_ref().map(|decoded| (decoded.width, decoded.height)));
        self.last_layout = view::layout(held_dims);
        seat_views
    }

    fn frame(&mut self) -> eros::Result<()> {
        use crate::decode;
        use crate::ui;
        use crate::ui::PageFacts;
        use crate::ui::StatusLine;

        let Some(size) = self.surface_size() else {
            return Ok(());
        };

        let spread = self.spread();
        debug_assert_eq!(spread, self.retuned_spread, "every spread change must retune the worker");
        let snapshot = self.worker.snapshot(&spread);
        let seat_views = self.display(spread, &snapshot);

        // Refit with the memo before the status line reads the scale, so
        // the bar does not show a superseded zoom.
        let total = self.last_layout.total();
        self.refit_into(total, self.last_viewport);

        let folded = fold_status(spread, self.state.mode, &seat_views);
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

        // Wake once at expiry so the toast disappears without other
        // traffic.
        self.mode_toast_since =
            self.mode_toast_since.filter(|&since| since.elapsed() < MODE_TOAST);
        if let Some(since) = self.mode_toast_since {
            let remaining = MODE_TOAST.saturating_sub(since.elapsed());
            post(&self.proxy, UserEvent::Repaint(remaining));
        }

        let claimed = self.source.name(spread.first).and_then(decode::format_from_name);
        let page = folded.page.map(|(decoded, dims)| PageFacts { decoded, claimed, dims });
        let status = StatusLine {
            total: self.source.len(),
            spread,
            mode: self.state.mode,
            toast: self.mode_toast_since.is_some(),
            name: self.source.page_name(spread.first),
            page,
            zoom: self.state.placement.scale,
            waiting: self.waiting_since.is_some_and(|since| since.elapsed() >= WAIT_GRACE),
            error: folded.error,
        };

        // egui lays out first: `EguiGlow::run` only records shapes, so
        // the chrome rect is settled before pages are placed, and paint
        // still draws the bar after the pages.
        let viewport = {
            let Some(gl) = self.gl.as_mut() else { return Ok(()) };
            let mut content = egui::Rect::NOTHING;
            gl.egui.run(gl.window.winit(), |ui| content = ui::show(ui, &status));
            content_viewport(size, content, gl.egui.egui_ctx.pixels_per_point())
        };
        // The bar's height moves with fonts and egui zoom, with no winit
        // resize event, so the viewport is diffed every frame. The status
        // line above was built at the old scale, so redraw once to
        // correct it.
        if viewport != self.last_viewport {
            self.fit.viewport_changed();
            self.last_viewport = viewport;
            self.request_redraw();
        }
        self.refit_into(total, viewport);

        let Some(gl) = self.gl.as_mut() else { return Ok(()) };

        gl.egui.painter.clear([size.width, size.height], clear_color(gl.window.translucent()));
        gl.pages.draw((size.width, size.height), viewport, &self.last_layout, &self.state.placement);
        gl.egui.paint(gl.window.winit());
        gl.window.pre_present_and_swap()?;
        // Reveal only after the first frame is presented, so the window
        // never maps empty.
        gl.window.winit().set_visible(true);
        Ok(())
    }

    /// Physical surface size, resizing the GL surface to match.
    /// None when there is no window yet or it is minimized.
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

    /// The (spread total, viewport) pair that pan and zoom work in, from
    /// the last frame's memos.
    fn geometry(&self) -> ((f32, f32), Viewport) {
        (self.last_layout.total(), self.last_viewport)
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
                    self.request_redraw(); // same pages, seats swap
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
        self.fit.page_turned();
        self.request_redraw();
        let spread = self.spread();
        self.retuned_spread = spread;
        self.worker.retune(spread);
    }
}

#[expect(
    clippy::rest_pattern_accessible_field,
    reason = "the matched fields are unused here, so naming them would trip unneeded_field_pattern"
)]
impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // SAFETY: the app owns this window and is dropped before the event
        // loop is, so the loop's display outlives every GL object in it.
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
        // `EguiGlow::new` can panic. Accepted upstream risk.
        let egui = egui_glow::EguiGlow::new(event_loop, Arc::clone(&gl), None, None, true);
        // System fonts join the family fallback chains lazily, for filename
        // characters the built-in fonts cannot render.
        egui.egui_ctx
            .add_font_provider(Arc::new(egui_system_font_provider::SystemFontProvider::new()));
        {
            use egui::mutex::Mutex;

            let proxy = Mutex::new(self.proxy.clone());
            egui.egui_ctx.set_request_repaint_callback(move |info| {
                post(&proxy.lock(), UserEvent::Repaint(info.delay));
            });
        }
        // Opaque mode deliberately keeps egui's default panel fill. The
        // fill applies once here; frame()'s clear follows the mode every
        // frame.
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
                // A repaint request must never delay an already-armed
                // earlier wake, so keep the sooner deadline.
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

        let egui_wants = {
            let response = gl.egui.on_window_event(gl.window.winit(), &event);
            let wants = gl.egui.egui_ctx.egui_wants_pointer_input();
            if response.repaint {
                gl.window.winit().request_redraw();
            }
            wants
        };

        match &event {
            WindowEvent::Resized(_) => {
                // The refit lands in frame(), where the viewport memo
                // notices the new size.
                self.surface_size();
                self.request_redraw();
            }
            WindowEvent::KeyboardInput { event: key, .. } => {
                if key.state == ElementState::Pressed {
                    if let PhysicalKey::Code(code) = key.physical_key {
                        if let Some(act) = Action::from_key(code) {
                            if !key.repeat || act.repeats() {
                                self.dispatch(act, event_loop);
                            }
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_wants => {
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
                self.dragging = *state == ElementState::Pressed && !egui_wants;
            }
            WindowEvent::CursorMoved { position, .. } => {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "cursor positions are physical px, exact in f32"
                )]
                let cursor_pos = (position.x as f32, position.y as f32);
                if self.dragging && !egui_wants {
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
                // Park until the next input, so idle costs nothing.
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

/// The page viewport from the chrome egui just laid out: the window
/// above the status bar's top edge, in physical px, full width. The
/// scale is egui's own (`pixels_per_point` = egui zoom x output scale),
/// never the window's scale factor.
fn content_viewport(window: PhysicalSize<u32>, content: egui::Rect, pixels_per_point: f32) -> Viewport {
    let win = view::px_f32((window.width, window.height));
    // The bar's top edge rounds to a whole physical px, matching the
    // rounding of egui's own clip rects.
    let bar_top = (content.max.y * pixels_per_point).round();
    Viewport::above_bottom_bar(win, bar_top)
}

/// The glass as egui's u8 color: only a conversion of [`clear_color`],
/// never a second premultiplication.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the clear color's channels are clamped 0..=1 values scaled into u8"
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

    // frame() and dispatch() are GL/winit-bound, so this mod pins the pure
    // helpers they consume.

    /// A fit refits until settled, user moves survive settles, and every discard resets both.
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
        let discards = [Fit::page_turned, Fit::viewport_changed, Fit::reset_view];
        let names = ["page turn", "viewport change", "reset view"];
        for (discard, name) in discards.into_iter().zip(names) {
            fit.settle((100.0, 200.0));
            fit.note_user_move();
            discard(&mut fit);
            assert!(fit.should_refit((100.0, 200.0)), "{name} discards the settle");
            assert!(!fit.user_moved, "{name} discards the user move");
        }
    }

    /// The fold, row by row: who waits, which error wins, which page speaks.
    #[test]
    fn fold_status_table() {
        struct Expect {
            waiting: bool,
            error: Option<&'static str>,
            page: Option<(Format, (u32, u32))>,
        }
        let pair = Spread { first: 0, second: Some(1) };
        let single = Spread::lone(0);
        let rows = [
            // a decided pair never waits; the failure still reports
            (pair, ViewMode::Comic, [Some(ready_seat(2, 2)), Some(SeatView::Failed("x".into()))],
                Expect { waiting: false, error: Some("x"), page: Some((Format::Png, (2, 2))) }),
            // a wanted pending seat waits, whichever seat it sits in
            (pair, ViewMode::Comic, [Some(SeatView::Pending), Some(ready_seat(2, 2))],
                Expect { waiting: true, error: None, page: Some((Format::Png, (2, 2))) }),
            // an unwanted seat is never waited on
            (single, ViewMode::Normal, [Some(ready_seat(2, 2)), None],
                Expect { waiting: false, error: None, page: Some((Format::Png, (2, 2))) }),
            // the first failure in reading order wins; comic reads left first
            (pair, ViewMode::Comic,
                [Some(SeatView::Failed("seat0".into())), Some(SeatView::Failed("seat1".into()))],
                Expect { waiting: false, error: Some("seat0"), page: None }),
            // manga reads the right seat first, so its error wins
            (pair, ViewMode::Manga,
                [Some(SeatView::Failed("seat0".into())), Some(SeatView::Failed("seat1".into()))],
                Expect { waiting: false, error: Some("seat1"), page: None }),
            // a ready lead speaks before the trailing failure stops the fold
            (pair, ViewMode::Comic,
                [Some(ready_seat(2, 2)), Some(SeatView::Failed("seat1".into()))],
                Expect { waiting: false, error: Some("seat1"), page: Some((Format::Png, (2, 2))) }),
            // both ready: the lead page speaks; manga's lead is the right seat
            (pair, ViewMode::Manga, [Some(ready_seat(800, 600)), Some(ready_seat(640, 480))],
                Expect { waiting: false, error: None, page: Some((Format::Png, (640, 480))) }),
            // comic's lead is the left seat
            (pair, ViewMode::Comic, [Some(ready_seat(800, 600)), Some(ready_seat(640, 480))],
                Expect { waiting: false, error: None, page: Some((Format::Png, (800, 600))) }),
            // a still-deciding lead yields its place to the other page
            (pair, ViewMode::Manga, [Some(ready_seat(800, 600)), Some(SeatView::Pending)],
                Expect { waiting: true, error: None, page: Some((Format::Png, (800, 600))) }),
            // a failed lead stops the fold before the partner can speak
            (pair, ViewMode::Manga,
                [Some(ready_seat(800, 600)), Some(SeatView::Failed("seat1".into()))],
                Expect { waiting: false, error: Some("seat1"), page: None }),
            // a pending seat behind a failed one still waits
            (pair, ViewMode::Manga, [Some(SeatView::Pending), Some(SeatView::Failed("x".into()))],
                Expect { waiting: true, error: Some("x"), page: None }),
            // an empty seat never speaks
            (pair, ViewMode::Manga, [None, Some(ready_seat(800, 600))],
                Expect { waiting: false, error: None, page: Some((Format::Png, (800, 600))) }),
        ];
        for (i, (spread, mode, seat_views, want)) in rows.iter().enumerate() {
            let folded = fold_status(*spread, *mode, seat_views);
            assert_eq!(folded.waiting, want.waiting, "row {i}: waiting");
            assert_eq!(folded.error.as_deref(), want.error, "row {i}: error");
            assert_eq!(folded.page, want.page, "row {i}: page");
        }
    }

    /// The background glass is one premultiplied color, pinned as exact bytes.
    #[expect(
        clippy::float_cmp_const,
        reason = "the opaque variant returns the same constants untouched, so equality is exact"
    )]
    #[test]
    fn background_glass() {
        let [red, green, blue, alpha] = clear_color(true);
        assert!((alpha - GLASS_ALPHA).abs() < 1e-6);
        // Each channel stores its straight value times alpha.
        for (channel, straight) in [red, green, blue].into_iter().zip(BACKGROUND) {
            assert!((straight.mul_add(GLASS_ALPHA, -channel)).abs() < 1e-6);
        }
        assert_eq!(clear_color(false), [BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 1.0]);
        // The bar must repeat the background glass. Exact bytes, so no
        // formula or constructor change can drift them apart silently.
        assert_eq!(glass_fill(), egui::Color32::from_rgba_premultiplied(16, 16, 18, 204));
    }

    /// The content viewport subtracts the bar in physical px, scaling the egui content rect by egui's own pixels-per-point.
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

    /// Pixel-mode wheel motion divides by the output scale factor, so the same physical scroll yields the same zoom lines at any scale.
    #[test]
    fn pixel_wheel_zoom_speed_does_not_vary_with_output_scale_factor() {
        assert_eq!(wheel_lines(40.0, 1.0), 1.0);
        assert_eq!(wheel_lines(80.0, 2.0), 1.0);
        assert_eq!(wheel_lines(40.0, 2.0), 0.5);
    }
}
