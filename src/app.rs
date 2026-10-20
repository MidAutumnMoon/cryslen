//! App: winit event handling, state transitions, frame composition.
//!
//! State is a single plain struct. Every event maps to an `Action` first
//! (pure, testable), then mutates state.

use std::sync::Arc;
use std::time::{Duration, Instant};

use egui_winit::winit;

use crate::decode::Decoded;
use crate::keys::Action;
use crate::render::PageTextures;
use crate::source::Source;
use crate::view::{Spread, SpreadLayout, View};
use crate::window::Window;
use crate::worker::{SeatView, SpreadView, Worker};

/// User events that wake the parked loop.
pub enum UserEvent {
    /// egui wants a repaint after this delay. Arming only: `new_events`
    /// fires the redraw when the deadline is reached.
    Repaint(Duration),
    /// The decode book changed. Redraw to pick up newly decided pages.
    PagesUpdated,
}

/// Everything mode-ish about the viewing session. Plain data.
struct State {
    /// Index of the spread's first page (a spread start).
    index: usize,
    double: bool,
    rtl: bool,
    view: View,
    fullscreen: bool,
}

/// Baked-in defaults (no config file, ever). Manga defaults: RTL on.
const DEFAULT_STATE: State = State {
    index: 0,
    double: false,
    rtl: true,
    view: View { scale: 1.0, offset: (0.0, 0.0) },
    fullscreen: false,
};

/// Status facts folded over both seats of a spread.
struct FoldedStatus {
    /// A wanted seat is still undecided.
    waiting: bool,
    /// First failure among the seats, in seat order.
    error: Option<String>,
    /// Format of the first Ready seat, left preferred.
    format: Option<&'static str>,
    /// Dims of the first Ready seat, left preferred.
    dims: Option<(u32, u32)>,
}

/// Fold the seats' views into status facts. `views[i]` is the worker's
/// view for `wanted[i]`.
fn fold_status(wanted: [Option<usize>; 2], views: &[Option<SeatView>; 2]) -> FoldedStatus {
    let mut folded = FoldedStatus { waiting: false, error: None, format: None, dims: None };
    for (seat, view) in views.iter().enumerate() {
        if wanted[seat].is_some() && matches!(view, Some(SeatView::Pending)) {
            folded.waiting = true;
        }
        match view {
            Some(SeatView::Failed(message)) if folded.error.is_none() => {
                folded.error = Some(message.clone());
            }
            Some(SeatView::Ready(decoded)) if folded.format.is_none() => {
                folded.format = Some(decoded.format.label());
                folded.dims = Some((decoded.width, decoded.height));
            }
            _ => {}
        }
    }
    folded
}

/// The auto-fit protocol: which spread total the view was fitted to, and
/// whether the user has placed the view since.
#[derive(Default)]
struct Fit {
    /// Spread total the view was fitted to. None demands a refit.
    fitted_total: Option<(f32, f32)>,
    /// The user zoomed or panned since the last fit.
    user_moved: bool,
}

impl Fit {
    /// Whether the view must be fitted to `total` this frame.
    fn should_refit(&self, total: (f32, f32)) -> bool {
        self.fitted_total != Some(total)
    }

    /// Record the fit of `total`.
    fn fitted_to(&mut self, total: (f32, f32)) {
        self.fitted_total = Some(total);
    }

    /// User zoom or pan. Their placement wins over later fit triggers.
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
        self.fitted_total = None;
        self.user_moved = false;
    }
}

/// The GL bundle: everything that exists only between `resumed` and exit.
/// Field order is drop order: pages and egui are dropped while the window
/// (which owns the GL context) is still alive.
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
    /// GL exists only after `resumed` (winit requirement).
    gl: Option<Gl>,
    /// GL init failure from `resumed`, which cannot return a Result. Main
    /// reports it after the event loop exits.
    pub start_error: Option<eros::ErrorUnion>,
    /// What each seat holds, the last frame included. Holding the
    /// decoded page is the design: worker eviction can drop the book's
    /// entry, never the pixels on screen.
    held: [Option<Arc<Decoded>>; 2],
    /// The spread last sent to the worker. Frame composition pins
    /// against it: every spread change must retune.
    retuned: Spread,
    /// Window size the GL surface was last resized to (physical px).
    last_surface_size: (u32, u32),
    cursor: (f32, f32),
    dragging: bool,
    /// Set while any wanted seat is undecided. The last frame stays on
    /// screen (hold-last-frame rule) and the status line flags waiting
    /// after 150 ms. No spinner animation: it would force repaints.
    waiting_since: Option<Instant>,
    /// Layout of the spread currently on screen. Pan and zoom read this
    /// memo between frames.
    last_layout: SpreadLayout,
    fit: Fit,
    /// Proxy for egui's repaint bridge. winit 0.30 cannot create one from
    /// `ActiveEventLoop` (only `EventLoop::create_proxy`, before `run_app`).
    proxy: Option<winit::event_loop::EventLoopProxy<UserEvent>>,
}

impl App {
    pub fn new(
        source: Arc<Source>,
        worker: Worker,
        index: usize,
        proxy: winit::event_loop::EventLoopProxy<UserEvent>,
    ) -> App {
        let state = State { index, ..DEFAULT_STATE };
        let spread = crate::view::spread_for(state.index, state.double, source.len());
        worker.retune(spread);
        App {
            source,
            worker,
            state,
            gl: None,
            start_error: None,
            held: [None, None],
            retuned: spread,
            last_surface_size: (0, 0),
            cursor: (0.0, 0.0),
            dragging: false,
            waiting_since: None,
            last_layout: SpreadLayout::default(),
            fit: Fit::default(),
            proxy: Some(proxy),
        }
    }

    /// Spread for the current index/mode, clamped to the book.
    fn spread(&self) -> Spread {
        crate::view::spread_for(self.state.index, self.state.double, self.source.len())
    }

    /// Apply one frame's seat decisions: adopt each wanted seat's Ready
    /// page, hold undecided and failed seats on their last frame, and
    /// refresh the layout memo from the held frames. Returns the
    /// seat-ordered views for the status fold.
    fn display(&mut self, spread: Spread, view: &SpreadView) -> [Option<SeatView>; 2] {
        let views: [Option<SeatView>; 2] =
            view.seats(self.state.rtl).map(|seat| seat.cloned());
        let wanted = spread.seats(self.state.rtl);
        for seat in 0..2 {
            match (wanted[seat], views[seat].as_ref()) {
                (None, _) => {
                    self.held[seat] = None;
                    if let Some(gl) = self.gl.as_mut() {
                        gl.pages.clear(seat);
                    }
                }
                (Some(page), Some(SeatView::Ready(decoded))) => {
                    self.held[seat] = Some(Arc::clone(decoded));
                    if let Some(gl) = self.gl.as_mut() {
                        gl.pages.bind(seat, page, decoded);
                    }
                }
                // Undecided or failed: hold the seat's frame.
                (Some(_), Some(SeatView::Pending | SeatView::Failed(_))) => {}
                // A wanted seat always has a view.
                (Some(_), None) => {}
            }
        }
        let dims = self
            .held
            .each_ref()
            .map(|held| held.as_ref().map(|d| (d.width, d.height)));
        self.last_layout = crate::view::layout(dims);
        views
    }

    /// One frame: decide, upload, draw, present.
    fn frame(&mut self) {
        let size = match self.surface_size() {
            Some(size) => size,
            None => return,
        };
        let win = (size.width as f32, size.height as f32);

        let spread = self.spread();
        debug_assert_eq!(spread, self.retuned, "every spread change must retune the worker");
        let view = self.worker.snapshot(&spread);
        let views = self.display(spread, &view);

        // Refit when the composed spread changes shape, unless the user
        // has zoomed or panned: their view wins over geometry.
        let total = self.last_layout.total();
        if self.fit.should_refit(total) {
            if !self.fit.user_moved {
                self.state.view = crate::view::fit(total, win);
            }
            self.fit.fitted_to(total);
        }

        let folded = fold_status(spread.seats(self.state.rtl), &views);
        self.waiting_since = if folded.waiting {
            Some(self.waiting_since.unwrap_or_else(Instant::now))
        } else {
            None
        };
        // While waiting on a decode, wake up once when the status grace
        // expires so "decoding…" appears. Without this the loop stays
        // parked and the flag never shows.
        if let Some(since) = self.waiting_since {
            let elapsed = since.elapsed();
            if elapsed < WAIT_GRACE {
                if let Some(proxy) = &self.proxy {
                    let _ = proxy.send_event(UserEvent::Repaint(WAIT_GRACE - elapsed));
                }
            }
        }

        let status = crate::ui::StatusLine {
            total: self.source.len(),
            spread,
            double: self.state.double,
            rtl: self.state.rtl,
            source_label: self.source.label(),
            format: folded.format,
            dims: folded.dims,
            zoom_percent: self.state.view.scale * 100.0,
            waiting: self.waiting_since.is_some_and(|t| t.elapsed() >= WAIT_GRACE),
            error: folded.error,
        };

        let Some(gl) = self.gl.as_mut() else { return };
        gl.egui.painter.clear([size.width, size.height], [0.08, 0.08, 0.09, 1.0]);
        gl.pages.draw(win, &self.last_layout, &self.state.view);
        gl.egui.run(&gl.window.winit, |ui| crate::ui::show(ui, &status));
        gl.egui.paint(&gl.window.winit);
        gl.window.pre_present_and_swap();
        // Reveal only after the first frame is presented, so the window
        // never maps empty.
        gl.window.winit.set_visible(true);
    }

    /// Physical surface size, resizing the GL surface when it changed.
    /// None when there is no window yet or it is minimized.
    fn surface_size(&mut self) -> Option<winit::dpi::PhysicalSize<u32>> {
        let gl = self.gl.as_ref()?;
        let size = gl.window.winit.inner_size();
        if size.width == 0 || size.height == 0 {
            return None; // minimized
        }
        if (size.width, size.height) != self.last_surface_size {
            gl.window.resize(size);
        }
        self.last_surface_size = (size.width, size.height);
        Some(size)
    }

    fn request_redraw(&self) {
        if let Some(gl) = self.gl.as_ref() {
            gl.window.winit.request_redraw();
        }
    }

    /// (spread total, window size) in physical px, the coordinate frame that
    /// pan and zoom work in.
    fn geometry(&self) -> ((f32, f32), (f32, f32)) {
        let window = self.gl.as_ref().map(|gl| {
            let window_size = gl.window.winit.inner_size();
            (window_size.width as f32, window_size.height as f32)
        });
        (self.last_layout.total(), window.unwrap_or((0.0, 0.0)))
    }

    fn dispatch(&mut self, act: Action, event_loop: &winit::event_loop::ActiveEventLoop) {
        match act {
            Action::Quit => event_loop.exit(),
            Action::ToggleDouble => {
                self.state.double = !self.state.double;
                self.state.index = crate::view::spread_for(
                    self.state.index,
                    self.state.double,
                    self.source.len(),
                )
                .first;
                self.after_page_change();
            }
            Action::ToggleRtl => {
                self.state.rtl = !self.state.rtl;
                self.request_redraw(); // same pages, seats swap
            }
            Action::ToggleFullscreen => {
                self.state.fullscreen = !self.state.fullscreen;
                let fullscreen_request =
                    self.state.fullscreen.then_some(winit::window::Fullscreen::Borderless(None));
                if let Some(gl) = self.gl.as_ref() {
                    gl.window.winit.set_fullscreen(fullscreen_request);
                }
            }
            Action::ResetView => {
                self.fit.reset_view();
                self.request_redraw();
            }
            Action::Next | Action::Prev => {
                let forward = act.is_forward(self.state.rtl);
                self.state.index = crate::view::advance(
                    self.state.index,
                    forward,
                    self.state.double,
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
        self.retuned = spread;
        self.worker.retune(spread);
    }
}

impl winit::application::ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = match Window::new(event_loop) {
            Ok(window) => window,
            Err(e) => {
                self.start_error = Some(e);
                event_loop.exit();
                return;
            }
        };
        let gl = window.glow();
        let egui = egui_glow::EguiGlow::new(event_loop, Arc::clone(&gl), None, None, true);
        // Bridge egui repaint requests into the winit loop.
        if let Some(proxy) = self.proxy.clone() {
            let proxy = egui::mutex::Mutex::new(proxy);
            egui.egui_ctx.set_request_repaint_callback(move |info| {
                let _ = proxy.lock().send_event(UserEvent::Repaint(info.delay));
            });
        }
        let pages = PageTextures::new(gl);
        window.winit.request_redraw();
        self.gl = Some(Gl { pages, egui, window });
    }

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Repaint(delay) => {
                if let Some(deadline) = Instant::now().checked_add(delay) {
                    event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(deadline));
                }
            }
            UserEvent::PagesUpdated => self.request_redraw(),
        }
    }

    fn new_events(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        if let winit::event::StartCause::ResumeTimeReached { .. } = cause {
            self.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.gl = None;
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
        use winit::keyboard::PhysicalKey;

        if matches!(event, WindowEvent::CloseRequested | WindowEvent::Destroyed) {
            event_loop.exit();
        }
        if self.gl.is_none() {
            return;
        }

        // Give egui the event first. If it wants the pointer, we don't
        // pan or zoom.
        let egui_wants = {
            let Gl { egui, window, .. } =
                self.gl.as_mut().expect("gl is Some (returned early when absent)");
            let response = egui.on_window_event(&window.winit, &event);
            let wants = egui.egui_ctx.egui_wants_pointer_input();
            if response.repaint {
                window.winit.request_redraw();
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
                let wheel_delta_y = {
                    let Some(gl) = self.gl.as_ref() else { return };
                    match delta {
                        MouseScrollDelta::LineDelta(_, lines) => *lines,
                        MouseScrollDelta::PixelDelta(pixels) => {
                            pixels.y as f32
                                / (PIXELS_PER_WHEEL_LINE
                                    * gl.window.winit.scale_factor() as f32)
                        }
                    }
                };
                let (total, window) = self.geometry();
                // Scroll up (positive dy) zooms in.
                let zoom_factor = (wheel_delta_y * WHEEL_ZOOM_SENSITIVITY).exp();
                let zoomed = crate::view::zoom_at(self.state.view, self.cursor, zoom_factor);
                self.state.view = crate::view::clamp(zoomed, total, window);
                self.fit.note_user_move();
                self.request_redraw();
            }
            WindowEvent::MouseInput { state, button: winit::event::MouseButton::Left, .. } => {
                self.dragging = *state == ElementState::Pressed && !egui_wants;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let cursor_pos = (position.x as f32, position.y as f32);
                if self.dragging && !egui_wants {
                    let (total, window) = self.geometry();
                    let pan_delta =
                        (cursor_pos.0 - self.cursor.0, cursor_pos.1 - self.cursor.1);
                    let panned = crate::view::pan(self.state.view, pan_delta);
                    self.state.view = crate::view::clamp(panned, total, window);
                    self.fit.note_user_move();
                    self.request_redraw();
                }
                self.cursor = cursor_pos;
            }
            WindowEvent::RedrawRequested => {
                self.frame();
                // Park until the next input, so idle costs nothing.
                event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
            }
            _ => {}
        }
    }
}

/// How long a page may stay undecided before the status line says so.
const WAIT_GRACE: Duration = Duration::from_millis(150);

/// Wheel zoom sensitivity: e^(`WHEEL_ZOOM_SENSITIVITY` * lines) per event,
/// so one notch is ~×1.16 and three notches ~×1.6.
const WHEEL_ZOOM_SENSITIVITY: f32 = 0.15;

/// Assumed wheel resolution for high-resolution (pixel-mode) wheels: how
/// many physical px of scroll equal one "line", normalized by scale factor
/// so zoom speed does not vary with output scaling.
const PIXELS_PER_WHEEL_LINE: f32 = 40.0;

#[cfg(test)]
mod tests {
    use super::*;

    fn ready(w: u32, h: u32) -> SeatView {
        SeatView::Ready(Arc::new(
            crate::decode::Decoded::rgba(
                w,
                h,
                crate::decode::Format::Png,
                vec![0u8; w as usize * h as usize * 4].into_boxed_slice(),
            )
            .expect("valid rgba page"),
        ))
    }

    #[test]
    fn fold_waits_on_any_pending_wanted_seat() {
        let both_decided = [Some(ready(2, 2)), Some(SeatView::Failed("x".into()))];
        assert!(!fold_status([Some(0), Some(1)], &both_decided).waiting);
        let right_pending = [Some(ready(2, 2)), Some(SeatView::Pending)];
        assert!(fold_status([Some(0), Some(1)], &right_pending).waiting);
        let left_pending = [Some(SeatView::Pending), Some(ready(2, 2))];
        assert!(fold_status([Some(0), Some(1)], &left_pending).waiting);
        // An unwanted seat is never waited on.
        let single = [Some(ready(2, 2)), None];
        assert!(!fold_status([Some(0), None], &single).waiting);
    }

    #[test]
    fn fit_demands_refit_until_a_total_is_recorded() {
        let mut fit = Fit::default();
        assert!(fit.should_refit((100.0, 200.0)));
        fit.fitted_to((100.0, 200.0));
        assert!(!fit.should_refit((100.0, 200.0)));
        assert!(fit.should_refit((100.0, 201.0)));
    }

    #[test]
    fn user_move_survives_spread_shape_changes() {
        let mut fit = Fit::default();
        fit.fitted_to((100.0, 200.0));
        fit.note_user_move();
        assert!(fit.user_moved);
        // A late-decoded page changes the shape after the user moved.
        fit.fitted_to((120.0, 200.0));
        assert!(fit.user_moved);
    }

    #[test]
    fn page_turn_discards_user_placement() {
        let mut fit = Fit::default();
        fit.fitted_to((100.0, 200.0));
        fit.note_user_move();
        fit.page_turned();
        assert!(fit.should_refit((100.0, 200.0)));
        assert!(!fit.user_moved);
    }

    #[test]
    fn resize_discards_user_placement() {
        let mut fit = Fit::default();
        fit.fitted_to((100.0, 200.0));
        fit.note_user_move();
        fit.resized();
        assert!(fit.should_refit((100.0, 200.0)));
        assert!(!fit.user_moved);
    }

    #[test]
    fn reset_view_discards_user_placement() {
        let mut fit = Fit::default();
        fit.fitted_to((100.0, 200.0));
        fit.note_user_move();
        fit.reset_view();
        assert!(fit.should_refit((100.0, 200.0)));
        assert!(!fit.user_moved);
    }

    #[test]
    fn fold_reports_the_first_error_in_seat_order() {
        let both_failed =
            [Some(SeatView::Failed("left".into())), Some(SeatView::Failed("right".into()))];
        assert_eq!(fold_status([Some(0), Some(1)], &both_failed).error.as_deref(), Some("left"));
        let right_failed = [Some(ready(2, 2)), Some(SeatView::Failed("right".into()))];
        assert_eq!(fold_status([Some(0), Some(1)], &right_failed).error.as_deref(), Some("right"));
    }

    #[test]
    fn fold_takes_format_and_dims_from_the_leftmost_ready_seat() {
        let only_right = [None, Some(ready(800, 600))];
        let folded = fold_status([Some(0), Some(1)], &only_right);
        assert_eq!(folded.format, Some("png"));
        assert_eq!(folded.dims, Some((800, 600)));
        let both = [Some(ready(640, 480)), Some(ready(800, 600))];
        let folded = fold_status([Some(0), Some(1)], &both);
        assert_eq!(folded.dims, Some((640, 480)));
    }
}
