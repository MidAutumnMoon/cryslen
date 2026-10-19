//! App: winit event handling, state transitions, frame composition.
//!
//! State is a single plain struct; every event maps
//! to an `Action` first (pure, testable), then mutates state. The frame is:
//! upload what the cache has -> draw pages -> paint egui status -> present.

use std::sync::Arc;
use std::time::{Duration, Instant};

use egui_winit::winit;
use winit::keyboard::KeyCode;

use crate::page::{Source, Spread};
use crate::render::{Pages, Window};
use crate::view::{SpreadLayout, View};
use crate::worker::Worker;

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

/// Whether an action moves toward higher page indices: in RTL the "next"
/// key (Right/Space) is visually leftward, so the arrows swap.
pub fn forward_of(act: Action, rtl: bool) -> bool {
    matches!(act, Action::Next) != rtl
}

/// Grace period before the status line admits the spread is still decoding.
pub const WAIT_GRACE: Duration = Duration::from_millis(150);

pub struct App {
    pub source: Arc<Source>,
    pub worker: Worker,
    pub state: State,
    /// GL things exist only after `resumed` (winit requirement).
    // Field order is drop order: Pages' textures and egui's GL objects must
    // die while the context (owned by `window`) is still alive.
    pub pages: Option<Pages>,
    pub egui: Option<egui_glow::EguiGlow>,
    pub window: Option<Window>,
    pub cursor: (f32, f32),
    pub dragging: bool,
    /// Fatal-ish message (open/decode failure path); shown in the status line.
    pub error: Option<String>,
    /// Set when the current spread has undecoded pages (hold-last-frame rule:
    /// keep showing what's on screen; the status line flags waiting after
    /// 150 ms — no spinner animation that would force repaints).
    pub waiting_since: Option<Instant>,
    pub last_layout: SpreadLayout,
    /// Spread total the current view was fitted to; NaN forces a refit on
    /// the next frame (page turn, resize, decode arrival changes the shape).
    pub fitted_total: (f32, f32),
    /// Window size the GL surface was last resized to.
    pub last_win: (u32, u32),
    /// Page index currently held by each texture slot. Texture uploads are
    /// expensive (full re-upload + mipmaps); only upload on page change.
    pub uploaded: [Option<usize>; 2],
    /// The user zoomed or panned since the last fit; suppresses the
    /// auto-refit that fires when a late-decoded page changes the spread
    /// shape (their view wins over geometry).
    pub user_moved: bool,
    /// Proxy for egui's repaint bridge. winit 0.30 cannot create one from
    /// `ActiveEventLoop` (only `EventLoop::create_proxy`, before `run_app`),
    /// so main() must pass one via [`App::set_proxy`]; until then the bridge
    /// stays off (the status line never animates, so nothing is lost).
    pub proxy: Option<winit::event_loop::EventLoopProxy<UserEvent>>,
}

impl App {
    pub fn new(source: Arc<Source>, worker: Worker) -> App {
        App {
            source,
            worker,
            state: START,
            pages: None,
            egui: None,
            window: None,
            cursor: (0.0, 0.0),
            dragging: false,
            error: None,
            waiting_since: None,
            last_layout: SpreadLayout::default(),
            fitted_total: (f32::NAN, f32::NAN),
            last_win: (0, 0),
            uploaded: [None, None],
            user_moved: false,
            proxy: None,
        }
    }

    /// Hand over an event-loop proxy (cloneable, created in main before
    /// `run_app`) so egui repaint requests can wake the parked loop.
    pub fn set_proxy(&mut self, proxy: winit::event_loop::EventLoopProxy<UserEvent>) {
        self.proxy = Some(proxy);
    }

    /// Spread for the current index/mode, with the tail pair clamped to the
    /// book (a 6-page book ends on `5-` not `5-7`).
    pub fn spread(&self) -> Spread {
        let mut s = crate::page::spread_for(self.state.index, self.state.double);
        if s.second.is_some_and(|p| p >= self.source.len()) {
            s.second = None;
        }
        s
    }

    /// Feed the current spread to the renderer from the cache; returns the
    /// layout used (also stored in `last_layout` for the status line).
    pub fn sync_slots(&mut self) {
        let spread = self.spread();
        // Texture slots are visual (0 = left, 1 = right), matching
        // SpreadLayout::rects; RTL seats the spread's first page on the right.
        let left = if self.state.rtl { spread.second } else { Some(spread.first) };
        let right = if self.state.rtl { Some(spread.first) } else { spread.second };
        if let Some(pages) = self.pages.as_mut() {
            for (slot, page) in [(0_usize, left), (1_usize, right)] {
                match page.and_then(|p| self.worker.get(p)) {
                    Some(d) if self.uploaded[slot] != page => {
                        pages.set(slot, &d);
                        self.uploaded[slot] = page;
                    }
                    // Slot without a page in this spread: nothing is coming,
                    // drop the stale texture. Undecoded-but-wanted slots keep
                    // their last frame (hold-last-frame rule).
                    None if page.is_none() => {
                        pages.clear(slot);
                        self.uploaded[slot] = None;
                    }
                    _ => {}
                }
            }
        }
        let dims = [
            self.pages.as_ref().and_then(|p| p.size(0)),
            self.pages.as_ref().and_then(|p| p.size(1)),
        ];
        self.last_layout = crate::view::layout(&spread, dims, self.state.rtl);
        self.waiting_since = if self.worker.get(spread.first).is_some() {
            None
        } else {
            Some(self.waiting_since.unwrap_or_else(Instant::now))
        };
        // Decode failures surface in the status line (worker stderr has detail).
        if let Some(e) = self.worker.take_error() {
            self.error = Some(e);
        }
    }

    /// One frame: sync slots, draw pages, paint status, present.
    pub fn frame(&mut self) {
        let size = {
            let Some(window) = self.window.as_ref() else { return };
            let size = window.winit.inner_size();
            if size.width == 0 || size.height == 0 {
                return; // minimized
            }
            if (size.width, size.height) != self.last_win {
                window.resize(size);
            }
            size
        };
        self.last_win = (size.width, size.height);
        let win = (size.width as f32, size.height as f32);

        self.sync_slots();

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

        // While waiting on a decode, wake up once when the 150 ms status
        // grace expires so "decoding…" actually appears (loop parks otherwise).
        if let Some(since) = self.waiting_since {
            let shown = since.elapsed();
            if shown < WAIT_GRACE {
                if let Some(proxy) = &self.proxy {
                    let _ = proxy.send_event(UserEvent::Repaint(WAIT_GRACE - shown));
                }
            }
        }

        let spread = self.spread();
        let first = self.worker.get(spread.first);
        let status = crate::ui::StatusLine {
            page: self.state.index,
            total: self.source.len(),
            spread,
            double: self.state.double,
            rtl: self.state.rtl,
            source_label: self.source.label(),
            format: first.as_ref().map(|d| d.format.label()),
            dims: first.as_ref().map(|d| (d.width, d.height)),
            zoom_percent: self.state.view.scale * 100.0,
            waiting: self.waiting_since.is_some_and(|t| t.elapsed() >= WAIT_GRACE),
            error: self.error.clone(),
        };

        let (Some(window), Some(pages), Some(egui)) =
            (&self.window, &self.pages, self.egui.as_mut())
        else {
            return;
        };
        egui.painter.clear([size.width, size.height], [0.08, 0.08, 0.09, 1.0]);
        pages.draw(win, &self.last_layout, &self.state.view);
        egui.run(&window.winit, |ui| crate::ui::show(ui, &status));
        egui.paint(&window.winit);
        window.pre_present_and_swap();
        // First frame made it to the screen — no startup flash after this.
        window.winit.set_visible(true);
    }

    /// Force a refit on the next frame and ask for one.
    fn needs_refit(&mut self) {
        self.fitted_total = (f32::NAN, f32::NAN);
        self.user_moved = false;
        self.request_redraw();
    }

    fn request_redraw(&self) {
        if let Some(window) = self.window.as_ref() {
            window.winit.request_redraw();
        }
    }

    /// (spread total, window size), both in physical px — the pan/zoom math
    /// frame of reference.
    fn geometry(&self) -> ((f32, f32), (f32, f32)) {
        let win = self.window.as_ref().map(|w| {
            let s = w.winit.inner_size();
            (s.width as f32, s.height as f32)
        });
        (self.last_layout.total(), win.unwrap_or((0.0, 0.0)))
    }

    fn dispatch(&mut self, act: Action, event_loop: &winit::event_loop::ActiveEventLoop) {
        match act {
            Action::Quit => event_loop.exit(),
            Action::ToggleDouble => {
                self.state.double = !self.state.double;
                self.state.index =
                    crate::page::spread_for(self.state.index, self.state.double).first;
                self.after_page_change();
            }
            Action::ToggleRtl => {
                self.state.rtl = !self.state.rtl;
                self.request_redraw(); // same pages, seats swap
            }
            Action::ToggleFullscreen => {
                self.state.fullscreen = !self.state.fullscreen;
                let fs = self.state.fullscreen.then_some(
                    winit::window::Fullscreen::Borderless(None),
                );
                if let Some(window) = self.window.as_ref() {
                    window.winit.set_fullscreen(fs);
                }
            }
            Action::ResetView => self.needs_refit(),
            Action::Next | Action::Prev => {
                let forward = forward_of(act, self.state.rtl);
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
        self.error = None; // per-page errors do not outlive the page
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
        let pages = Pages::new(gl);
        // Fresh GL objects: forget what the slots held, or a second `resumed`
        // would skip uploads forever (slot tracking vs empty textures).
        self.uploaded = [None, None];
        window.winit.request_redraw();
        self.pages = Some(pages);
        self.egui = Some(egui);
        self.window = Some(window);
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
        // Teardown while the GL context still exists; egui_glow logs a leak
        // warning if destroy() never runs. Field order makes this the default
        // too, in case anything is dropped implicitly.
        self.pages = None;
        if let Some(egui) = self.egui.as_mut() {
            egui.destroy();
        }
        self.window = None;
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
        if self.window.is_none() {
            return;
        }

        // Give egui the event first; if it wants the pointer, we don't pan/zoom.
        let egui_wants = {
            let window = self.window.as_ref().unwrap();
            match self.egui.as_mut() {
                Some(egui) => {
                    let response = egui.on_window_event(&window.winit, &event);
                    let wants = egui.egui_ctx.egui_wants_pointer_input();
                    if response.repaint {
                        window.winit.request_redraw();
                    }
                    wants
                }
                None => false,
            }
        };

        match &event {
            WindowEvent::Resized(size) => {
                if let Some(window) = self.window.as_ref() {
                    window.resize(*size);
                }
                self.last_win = (size.width, size.height);
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
                let dy = {
                    let Some(window) = self.window.as_ref() else { return };
                    match delta {
                        MouseScrollDelta::LineDelta(_, y) => *y,
                        // PixelDelta is physical; normalize so zoom speed
                        // doesn't change with the output scale factor.
                        MouseScrollDelta::PixelDelta(p) => {
                            p.y as f32 / (40.0 * window.winit.scale_factor() as f32)
                        }
                    }
                };
                let (total, win) = self.geometry();
                let v = crate::view::zoom_at(self.state.view, self.cursor, (-(dy * 0.15)).exp());
                self.state.view = crate::view::clamp(v, total, win);
                self.user_moved = true;
                self.request_redraw();
            }
            WindowEvent::MouseInput { state, button: winit::event::MouseButton::Left, .. } => {
                self.dragging = *state == ElementState::Pressed && !egui_wants;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new = (position.x as f32, position.y as f32);
                if self.dragging && !egui_wants {
                    let (total, win) = self.geometry();
                    let delta = (new.0 - self.cursor.0, new.1 - self.cursor.1);
                    let v = crate::view::pan(self.state.view, delta);
                    self.state.view = crate::view::clamp(v, total, win);
                    self.user_moved = true;
                    self.request_redraw();
                }
                self.cursor = new;
            }
            WindowEvent::RedrawRequested => {
                self.frame();
                // Idle: park the loop — zero CPU until the next input.
                event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
            }
            _ => {}
        }
    }
}

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
        assert!(forward_of(Action::Next, false));
        assert!(!forward_of(Action::Prev, false));
        // RTL manga: the same keys flip — Left-ward "next" reads forward.
        assert!(!forward_of(Action::Next, true));
        assert!(forward_of(Action::Prev, true));
    }

    #[test]
    fn page_turn_respects_rtl() {
        // advance() composes with forward_of: in RTL, "Next" from page 1
        // moves to page 0 (the cover), in western to page 2.
        let fwd = forward_of(Action::Next, true);
        assert_eq!(crate::page::advance(1, fwd, false, 10), 0);
        let fwd = forward_of(Action::Next, false);
        assert_eq!(crate::page::advance(1, fwd, false, 10), 2);
    }
}
