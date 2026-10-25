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
use crate::view::{Placement, Seat, Spread, SpreadLayout, ViewMode};
use crate::window::Window;
use crate::worker::{SeatView, SpreadView, Worker};

/// User events that wake the parked loop.
pub enum UserEvent {
    /// Wake the loop after this delay. Arming only: `new_events`
    /// fires the redraw when the deadline is reached.
    Repaint(Duration),
    /// The decode book changed. Redraw to pick up newly decided pages.
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

/// Everything mode-ish about the viewing session.
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
/// view for `spread.seats(mode)[i]`. A seat still deciding yields its place
/// to the other one. A failed page stops the fold: its facts do not exist,
/// and the partner's would describe the wrong page.
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

/// The auto-fit protocol: which spread total the refit decision is
/// settled for, and whether the user has placed the spread since.
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

    /// User zoom or pan.
    fn note_user_move(&mut self) {
        self.user_moved = true;
    }

    /// Page turn: the new spread fits afresh once its dims settle. User
    /// placement does not carry across pages.
    fn page_turned(&mut self) {
        self.refit();
    }

    /// Resize: fit afresh, discarding user placement.
    fn resized(&mut self) {
        self.refit();
    }

    /// Reset view: fit afresh, discarding user placement.
    fn reset_view(&mut self) {
        self.refit();
    }

    fn refit(&mut self) {
        self.settled = None;
        self.user_moved = false;
    }
}

/// The GL bundle: everything that exists only between `resumed` and exit
/// (winit requirement). Field order is drop order: pages and egui are
/// dropped while the window (which owns the GL context) is still alive.
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
    cursor: (f32, f32),
    dragging: bool,
    /// Set while any wanted seat is undecided. The last frame stays on
    /// screen (hold-last-frame rule) and the status line flags waiting
    /// after [`WAIT_GRACE`]. No spinner animation: it would force repaints.
    waiting_since: Option<Instant>,
    /// Set while the mode-switch toast is showing in the status bar.
    mode_toast_since: Option<Instant>,
    /// Layout of the spread currently on screen. Pan and zoom read this
    /// memo between frames.
    last_layout: SpreadLayout,
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

    /// Spread for the current index/mode, clamped to the book.
    fn spread(&self) -> Spread {
        view::spread_for(self.state.index, self.state.mode, self.source.len())
    }

    /// Show each seat's Ready page, hold undecided and failed seats on
    /// their last frame, and refresh the layout memo. Returns the
    /// seat-ordered views for the status fold.
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

    /// One frame: decide, upload, draw, present.
    fn frame(&mut self) -> eros::Result<()> {
        use crate::decode;
        use crate::ui;
        use crate::ui::PageFacts;
        use crate::ui::StatusLine;

        let Some(size) = self.surface_size() else {
            return Ok(());
        };
        let win = view::px_f32((size.width, size.height));

        let spread = self.spread();
        debug_assert_eq!(spread, self.retuned_spread, "every spread change must retune the worker");
        let snapshot = self.worker.snapshot(&spread);
        let seat_views = self.display(spread, &snapshot);

        // A user zoom or pan wins over the shape-triggered refit.
        let total = self.last_layout.total();
        if self.fit.should_refit(total) {
            if !self.fit.user_moved {
                self.state.placement = view::fit(total, win);
            }
            self.fit.settle(total);
        }

        let folded = fold_status(spread, self.state.mode, &seat_views);
        self.waiting_since = folded
            .waiting
            .then(|| self.waiting_since.unwrap_or_else(Instant::now));
        // Arm a wake at grace expiry. The parked loop otherwise never
        // shows the "decoding…" flag.
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

        let Some(gl) = self.gl.as_mut() else { return Ok(()) };
        gl.egui.painter.clear([size.width, size.height], clear_color(gl.window.translucent()));
        gl.pages.draw((size.width, size.height), &self.last_layout, &self.state.placement);
        gl.egui.run(gl.window.winit(), |ui| ui::show(ui, &status));
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

    /// (spread total, window size) in physical px, the coordinate frame that
    /// pan and zoom work in.
    fn geometry(&self) -> ((f32, f32), (f32, f32)) {
        let window = self.gl.as_ref().map(|gl| {
            let window_size = gl.window.winit().inner_size();
            view::px_f32((window_size.width, window_size.height))
        });
        (self.last_layout.total(), window.unwrap_or((0.0, 0.0)))
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
        // Opaque mode deliberately keeps egui's default panel fill over
        // the background. Glass mode applies the fill once here; a
        // future runtime glass toggle must re-apply it, while frame()'s
        // clear follows the mode on its own.
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

        // Give egui the event first. If it wants the pointer, we don't
        // pan or zoom.
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
                self.surface_size();
                self.fit.resized();
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
                let (total, window) = self.geometry();
                // Scroll up (positive dy) zooms in.
                let zoom_factor = (wheel_delta_y * WHEEL_ZOOM_SENSITIVITY).exp();
                let zoomed = view::zoom_at(self.state.placement, self.cursor, zoom_factor);
                self.state.placement = view::clamp(zoomed, total, window);
                self.fit.note_user_move();
                self.request_redraw();
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.dragging = *state == ElementState::Pressed && !egui_wants;
            }
            WindowEvent::CursorMoved { position, .. } => {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "cursor positions are window-scale px, exact in f32"
                )]
                let cursor_pos = (position.x as f32, position.y as f32);
                if self.dragging && !egui_wants {
                    let (total, window) = self.geometry();
                    let pan_delta =
                        (cursor_pos.0 - self.cursor.0, cursor_pos.1 - self.cursor.1);
                    let panned = view::pan(self.state.placement, pan_delta);
                    self.state.placement = view::clamp(panned, total, window);
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

/// Window background color, straight (un-multiplied) channels.
const BACKGROUND: [f32; 3] = [0.08, 0.08, 0.09];

/// Background opacity when the window is translucent.
const GLASS_ALPHA: f32 = 0.85;

/// Wayland composites ARGB buffers as premultiplied alpha, so the
/// translucent variant scales the color channels by alpha. Stored
/// straight, the composite would come out brighter than intended.
fn clear_color(translucent: bool) -> [f32; 4] {
    let [red, green, blue] = BACKGROUND;
    if translucent {
        [red * GLASS_ALPHA, green * GLASS_ALPHA, blue * GLASS_ALPHA, GLASS_ALPHA]
    } else {
        [red, green, blue, 1.0]
    }
}

/// [`clear_color`]'s translucent channels as egui's premultiplied u8
/// color. The premultiplied glass lives in one place, so the bar
/// repeating the background holds structurally.
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

/// Wheel zoom sensitivity: one notch is ~×1.16, three notches ~×1.6.
const WHEEL_ZOOM_SENSITIVITY: f32 = 0.15;

/// Assumed wheel resolution for pixel-mode wheels: how many physical
/// px of scroll equal one "line".
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
#[expect(clippy::expect_used, reason = "a panic in test code is inconsequential")]
mod tests {
    use super::*;

    fn ready(width: u32, height: u32) -> SeatView {
        SeatView::Ready(Arc::new(
            Decoded::rgba(
                width,
                height,
                Format::Png,
                vec![0_u8; width as usize * height as usize * 4].into_boxed_slice(),
            )
            .expect("valid rgba page"),
        ))
    }

    #[test]
    fn clear_color_is_premultiplied() {
        let [red, green, blue, alpha] = clear_color(true);
        assert!((alpha - GLASS_ALPHA).abs() < 1e-6);
        // Each channel stores its straight value times alpha.
        for (channel, straight) in [red, green, blue].into_iter().zip(BACKGROUND) {
            assert!((straight.mul_add(GLASS_ALPHA, -channel)).abs() < 1e-6);
        }
        // The premultiplied invariant: no channel exceeds alpha.
        assert!(red <= alpha && green <= alpha && blue <= alpha);
    }

    #[expect(
        clippy::float_cmp_const,
        reason = "the opaque variant returns the same constants untouched, so equality is exact"
    )]
    #[test]
    fn opaque_clear_color_is_the_plain_background() {
        assert_eq!(clear_color(false), [BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 1.0]);
    }

    #[test]
    fn glass_fill_pins_the_background_glass() {
        // Exact bytes: the bar must repeat the background glass, and no
        // formula or constructor change may drift them apart silently.
        assert_eq!(glass_fill(), egui::Color32::from_rgba_premultiplied(17, 17, 20, 217));
    }

    #[test]
    fn fold_waits_on_any_pending_wanted_seat() {
        let spread = Spread { first: 0, second: Some(1) };
        let both_decided = [Some(ready(2, 2)), Some(SeatView::Failed("x".into()))];
        assert!(!fold_status(spread, ViewMode::Comic, &both_decided).waiting);
        let right_pending = [Some(ready(2, 2)), Some(SeatView::Pending)];
        assert!(fold_status(spread, ViewMode::Comic, &right_pending).waiting);
        let left_pending = [Some(SeatView::Pending), Some(ready(2, 2))];
        assert!(fold_status(spread, ViewMode::Comic, &left_pending).waiting);
        // An unwanted seat is never waited on.
        let single_spread = Spread { first: 0, second: None };
        let single = [Some(ready(2, 2)), None];
        assert!(!fold_status(single_spread, ViewMode::Normal, &single).waiting);
    }

    #[test]
    fn fit_demands_refit_until_a_total_is_recorded() {
        let mut fit = Fit::default();
        assert!(fit.should_refit((100.0, 200.0)));
        fit.settle((100.0, 200.0));
        assert!(!fit.should_refit((100.0, 200.0)));
        assert!(fit.should_refit((100.0, 201.0)));
    }

    #[test]
    fn user_move_survives_spread_shape_changes() {
        let mut fit = Fit::default();
        fit.settle((100.0, 200.0));
        fit.note_user_move();
        assert!(fit.user_moved);
        // A late-decoded page changes the shape after the user moved.
        fit.settle((120.0, 200.0));
        assert!(fit.user_moved);
    }

    #[test]
    fn page_turn_discards_user_placement() {
        let mut fit = Fit::default();
        fit.settle((100.0, 200.0));
        fit.note_user_move();
        fit.page_turned();
        assert!(fit.should_refit((100.0, 200.0)));
        assert!(!fit.user_moved);
    }

    #[test]
    fn resize_discards_user_placement() {
        let mut fit = Fit::default();
        fit.settle((100.0, 200.0));
        fit.note_user_move();
        fit.resized();
        assert!(fit.should_refit((100.0, 200.0)));
        assert!(!fit.user_moved);
    }

    #[test]
    fn reset_view_discards_user_placement() {
        let mut fit = Fit::default();
        fit.settle((100.0, 200.0));
        fit.note_user_move();
        fit.reset_view();
        assert!(fit.should_refit((100.0, 200.0)));
        assert!(!fit.user_moved);
    }

    #[test]
    fn fold_reports_the_first_error_in_reading_order() {
        let spread = Spread { first: 0, second: Some(1) };
        let both_failed = [Some(SeatView::Failed("seat0".into())), Some(SeatView::Failed("seat1".into()))];
        assert_eq!(fold_status(spread, ViewMode::Comic, &both_failed).error.as_deref(), Some("seat0"));
        // Manga reads the right seat first, so its error wins.
        assert_eq!(fold_status(spread, ViewMode::Manga, &both_failed).error.as_deref(), Some("seat1"));
        let right_failed = [Some(ready(2, 2)), Some(SeatView::Failed("seat1".into()))];
        assert_eq!(fold_status(spread, ViewMode::Comic, &right_failed).error.as_deref(), Some("seat1"));
    }

    #[test]
    fn fold_takes_facts_from_the_lead_page() {
        let spread = Spread { first: 0, second: Some(1) };
        // In Manga the lead page sits in the right seat.
        let both = [Some(ready(800, 600)), Some(ready(640, 480))];
        assert_eq!(fold_status(spread, ViewMode::Manga, &both).page, Some((Format::Png, (640, 480))));
        assert_eq!(fold_status(spread, ViewMode::Comic, &both).page, Some((Format::Png, (800, 600))));
        // A still-deciding lead seat yields its place to the other page.
        let lead_pending = [Some(ready(800, 600)), Some(SeatView::Pending)];
        assert_eq!(fold_status(spread, ViewMode::Manga, &lead_pending).page, Some((Format::Png, (800, 600))));
        assert!(fold_status(spread, ViewMode::Manga, &lead_pending).waiting);
        // A failed lead stops the fold.
        let lead_failed = [Some(ready(800, 600)), Some(SeatView::Failed("seat1".into()))];
        assert_eq!(fold_status(spread, ViewMode::Manga, &lead_failed).page, None);
        assert_eq!(fold_status(spread, ViewMode::Manga, &lead_failed).error.as_deref(), Some("seat1"));
        // The partner's decode still counts as waiting even then.
        let failed_and_pending = [Some(SeatView::Pending), Some(SeatView::Failed("x".into()))];
        assert!(fold_status(spread, ViewMode::Manga, &failed_and_pending).waiting);
        assert_eq!(fold_status(spread, ViewMode::Manga, &failed_and_pending).page, None);
        // An empty seat never speaks.
        let only_right = [None, Some(ready(800, 600))];
        assert_eq!(fold_status(spread, ViewMode::Manga, &only_right).page, Some((Format::Png, (800, 600))));
    }
}
