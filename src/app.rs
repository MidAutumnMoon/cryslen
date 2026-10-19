//! App: winit event handling, state transitions, frame composition.
//!
//! State is a single plain struct; every event maps to an `Action` first
//! (pure, testable), then mutates state.

use std::sync::Arc;
use std::time::{Duration, Instant};

use egui_winit::winit;
use winit::keyboard::KeyCode;

use crate::page::{Source, Spread};
use crate::render::{PageSlots, Window};
use crate::view::{SpreadLayout, View};
use crate::worker::{Page, Worker};

/// User events that wake the parked loop.
pub enum UserEvent {
    /// egui wants a repaint after this delay (arm-only; fired by new_events).
    Repaint(Duration),
    /// Decode cache changed; redraw to pick up newly decoded pages.
    PagesUpdated,
}

/// Everything mode-ish about the viewing session. Plain data.
pub struct State {
    /// Index of the spread's first page (a spread start).
    pub index: usize,
    pub double: bool,
    pub rtl: bool,
    pub view: View,
    pub fullscreen: bool,
}

/// Baked-in defaults (no config file, ever). Manga defaults: RTL on.
pub const START: State = State {
    index: 0,
    double: false,
    rtl: true,
    view: View { scale: 1.0, offset: (0.0, 0.0) },
    fullscreen: false,
};

/// Input actions, derived from physical keys (layout-independent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Next,
    Prev,
    Quit,
    ToggleDouble,
    ToggleRtl,
    ToggleFullscreen,
    ResetView,
}

/// Key map, baked in. Arrows are *directional*: Right=Next, Left=Prev; the
/// app swaps them when `rtl` is on (reading direction decides, not the key).
pub fn action(code: KeyCode) -> Option<Action> {
    use Action::*;
    match code {
        KeyCode::Escape | KeyCode::KeyQ => Some(Quit),
        KeyCode::KeyD => Some(ToggleDouble),
        KeyCode::KeyM => Some(ToggleRtl),
        KeyCode::KeyF => Some(ToggleFullscreen),
        KeyCode::KeyR => Some(ResetView),
        KeyCode::ArrowRight | KeyCode::Space | KeyCode::PageDown => Some(Next),
        KeyCode::ArrowLeft | KeyCode::PageUp => Some(Prev),
        _ => None,
    }
}

/// Whether an action advances reading order under the current direction.
pub fn is_forward(act: Action, rtl: bool) -> bool {
    match act {
        Action::Next => !rtl,
        Action::Prev => rtl,
        _ => false,
    }
}

/// The GL bundle: everything that exists only between `resumed` and exit.
/// Field order is drop order: slots and egui are dropped while the window
/// (which owns the GL context) is still alive.
pub struct Gl {
    pub slots: PageSlots,
    pub egui: egui_glow::EguiGlow,
    pub window: Window,
}

pub struct App {
    pub source: Arc<Source>,
    pub worker: Worker,
    pub state: State,
    /// GL exists only after `resumed` (winit requirement).
    pub gl: Option<Gl>,
    /// Window size the GL surface was last resized to (physical px).
    pub last_win: (u32, u32),
    pub cursor: (f32, f32),
    pub dragging: bool,
    /// Set when the current spread's first page is undecided (hold-last-frame
    /// rule: keep showing what's on screen; the status line flags waiting
    /// after 150 ms — no spinner animation that would force repaints).
    pub waiting_since: Option<Instant>,
    /// Layout of the spread currently on screen; the memo pan/zoom needs
    /// between frames.
    pub last_layout: SpreadLayout,
    /// Spread total the current view was fitted to; NaN forces a refit on
    /// the next frame (page turn, resize, decode arrival changes the shape).
    pub fitted_total: (f32, f32),
    /// The user zoomed or panned since the last fit; suppresses the
    /// auto-refit that fires when a late-decoded page changes the spread
    /// shape (their view wins over geometry).
    pub user_moved: bool,
    /// Proxy for egui's repaint bridge. winit 0.30 cannot create one from
    /// `ActiveEventLoop` (only `EventLoop::create_proxy`, before `run_app`).
    pub proxy: Option<winit::event_loop::EventLoopProxy<UserEvent>>,
}

impl App {
    pub fn new(
        source: Arc<Source>,
        worker: Worker,
        index: usize,
        proxy: winit::event_loop::EventLoopProxy<UserEvent>,
    ) -> App {
        App {
            source,
            worker,
            state: State { index, ..START },
            gl: None,
            last_win: (0, 0),
            cursor: (0.0, 0.0),
            dragging: false,
            waiting_since: None,
            last_layout: SpreadLayout::default(),
            fitted_total: (f32::NAN, f32::NAN),
            user_moved: false,
            proxy: Some(proxy),
        }
    }

    /// Spread for the current index/mode, clamped to the book.
    pub fn spread(&self) -> Spread {
        crate::page::spread_for(self.state.index, self.state.double, self.source.len())
    }

    /// Upload what changed: for each visual seat (left, right) either the
    /// freshly decoded page, or — when that page is undecided — nothing, so
    /// the slot keeps its last frame (hold-last-frame rule). Seats without a
    /// page in this spread are cleared.
    fn sync_slots(&mut self, spread: &Spread) {
        let seats = spread.seats(self.state.rtl);
        if let Some(gl) = self.gl.as_mut() {
            for (slot, &page) in seats.iter().enumerate() {
                let shown = page.and_then(|p| self.worker.page(p));
                match (page, shown) {
                    (Some(p), Some(Page::Ready(d))) => gl.slots.set(slot, p, &d),
                    (None, _) => gl.slots.clear(slot),
                    // Undecided or failed: keep whatever the slot shows.
                    _ => {}
                }
            }
        }
        let dims = match self.gl.as_ref() {
            Some(gl) => [gl.slots.size(0), gl.slots.size(1)],
            None => [None, None],
        };
        self.last_layout = crate::view::layout(dims);
    }

    /// One frame: decide, upload, draw, present.
    pub fn frame(&mut self) {
        let size = match self.surface_size() {
            Some(size) => size,
            None => return,
        };
        let win = (size.width as f32, size.height as f32);

        let spread = self.spread();
        let first = self.worker.page(spread.first);
        self.sync_slots(&spread);

        // Refit when the composed spread changed shape (page turn, resize,
        // newly decoded dims) — unless the user has zoomed/panned since the
        // last fit, in which case their view wins over geometry.
        let total = self.last_layout.total();
        if total != self.fitted_total {
            if !self.user_moved {
                self.state.view = crate::view::fit(total, win);
            }
            self.fitted_total = total;
        }

        self.waiting_since = match &first {
            Some(Page::Ready(_)) | Some(Page::Failed(_)) => None,
            None => Some(self.waiting_since.unwrap_or_else(Instant::now)),
        };
        // While waiting on a decode, wake up once when the 150 ms status
        // grace expires so "decoding…" appears. Without this the loop stays
        // parked and the flag never shows.
        if let Some(since) = self.waiting_since {
            let shown = since.elapsed();
            if shown < WAIT_GRACE {
                if let Some(proxy) = &self.proxy {
                    let _ = proxy.send_event(UserEvent::Repaint(WAIT_GRACE - shown));
                }
            }
        }

        let status = crate::ui::StatusLine {
            total: self.source.len(),
            spread,
            double: self.state.double,
            rtl: self.state.rtl,
            source_label: self.source.label(),
            format: match &first {
                Some(Page::Ready(d)) => Some(d.format.label()),
                _ => None,
            },
            dims: match &first {
                Some(Page::Ready(d)) => Some((d.width, d.height)),
                _ => None,
            },
            zoom_percent: self.state.view.scale * 100.0,
            waiting: self.waiting_since.is_some_and(|t| t.elapsed() >= WAIT_GRACE),
            error: match &first {
                Some(Page::Failed(e)) => Some(e.clone()),
                _ => None,
            },
        };

        let Some(gl) = self.gl.as_mut() else { return };
        gl.egui.painter.clear([size.width, size.height], [0.08, 0.08, 0.09, 1.0]);
        gl.slots.draw(win, &self.last_layout, &self.state.view);
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
        if (size.width, size.height) != self.last_win {
            gl.window.resize(size);
        }
        self.last_win = (size.width, size.height);
        Some(size)
    }

    /// Force a refit on the next frame and ask for one.
    fn needs_refit(&mut self) {
        self.fitted_total = (f32::NAN, f32::NAN);
        self.user_moved = false;
        self.request_redraw();
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
                self.state.index = crate::page::spread_for(
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
            Action::ResetView => self.needs_refit(),
            Action::Next | Action::Prev => {
                let forward = is_forward(act, self.state.rtl);
                self.state.index = crate::page::advance(
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
        self.needs_refit(); // fit to the new spread once its dims are known
        self.worker.focus(self.state.index); // cheap, never blocks
    }
}

impl winit::application::ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = Window::new(event_loop);
        let gl = window.glow();
        let egui = egui_glow::EguiGlow::new(event_loop, Arc::clone(&gl), None, None, true);
        // Bridge egui repaint requests into the winit loop (arm-only: the
        // deadline itself is fired by new_events -> ResumeTimeReached).
        if let Some(proxy) = self.proxy.clone() {
            let proxy = egui::mutex::Mutex::new(proxy);
            egui.egui_ctx.set_request_repaint_callback(move |info| {
                let _ = proxy.lock().send_event(UserEvent::Repaint(info.delay));
            });
        }
        let slots = PageSlots::new(gl);
        window.winit.request_redraw();
        self.gl = Some(Gl { slots, egui, window });
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
        // egui_glow logs a leak warning if destroy() never runs. Drop order is
        // Gl's documented invariant.
        if let Some(mut gl) = self.gl.take() {
            gl.egui.destroy();
        }
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

        // Give egui the event first; if it wants the pointer, we don't pan/zoom.
        let egui_wants = {
            let Gl { egui, window, .. } = self.gl.as_mut().unwrap();
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
                self.needs_refit();
            }
            WindowEvent::KeyboardInput { event: key, .. } => {
                // Auto-repeat must not re-toggle mode keys at ~30 Hz.
                if key.state == ElementState::Pressed && !key.repeat {
                    if let PhysicalKey::Code(code) = key.physical_key {
                        if let Some(act) = action(code) {
                            self.dispatch(act, event_loop);
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
                self.user_moved = true;
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
                    self.user_moved = true;
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

    #[test]
    fn key_map() {
        assert_eq!(action(KeyCode::Escape), Some(Action::Quit));
        assert_eq!(action(KeyCode::KeyQ), Some(Action::Quit));
        assert_eq!(action(KeyCode::KeyD), Some(Action::ToggleDouble));
        assert_eq!(action(KeyCode::KeyM), Some(Action::ToggleRtl));
        assert_eq!(action(KeyCode::KeyF), Some(Action::ToggleFullscreen));
        assert_eq!(action(KeyCode::KeyR), Some(Action::ResetView));
        assert_eq!(action(KeyCode::ArrowRight), Some(Action::Next));
        assert_eq!(action(KeyCode::Space), Some(Action::Next));
        assert_eq!(action(KeyCode::PageDown), Some(Action::Next));
        assert_eq!(action(KeyCode::ArrowLeft), Some(Action::Prev));
        assert_eq!(action(KeyCode::PageUp), Some(Action::Prev));
        assert_eq!(action(KeyCode::KeyA), None);
        assert_eq!(action(KeyCode::ArrowUp), None);
    }

    #[test]
    fn forward_follows_reading_direction() {
        // Western: Right/Space advances.
        assert!(is_forward(Action::Next, false));
        assert!(!is_forward(Action::Prev, false));
        // RTL manga: the same keys flip — Left-ward "next" reads forward.
        assert!(!is_forward(Action::Next, true));
        assert!(is_forward(Action::Prev, true));
    }

    #[test]
    fn page_turn_respects_rtl() {
        // advance() composes with is_forward: in RTL, "Next" from page 1
        // moves to page 0 (the cover), in western to page 2.
        let fwd = is_forward(Action::Next, true);
        assert_eq!(crate::page::advance(1, fwd, false, 10), 0);
        let fwd = is_forward(Action::Next, false);
        assert_eq!(crate::page::advance(1, fwd, false, 10), 2);
    }
}
