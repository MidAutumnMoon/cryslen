//! App: winit event handling, state transitions, frame composition.

use std::sync::Arc;
use std::time::{Duration, Instant};

use egui_winit::winit;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::StartCause;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::ControlFlow;
use winit::event_loop::EventLoopProxy;
use winit::keyboard::ModifiersState;
use winit::window::WindowId;

use crate::keys::Action;
use crate::render::PageTextures;
use crate::screen::{self, Screen};
use crate::source::Source;
use crate::view;
use crate::view::{FitMode, Spread, ViewMode, Viewport};
use crate::window::Window;
use crate::worker::Worker;

/// User events that wake the parked loop.
pub enum UserEvent {
    /// Arms a wake; `new_events` requests the redraw at the deadline.
    Repaint(Duration),
    /// The worker decided `page`. The handler redraws only when the
    /// decision changes the screen or its staged prefetch.
    PagesUpdated(usize),
}

/// A failed send means the loop is gone, leaving the event with no consumer.
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is the crate's visibility style for crate-wide items, even inside private modules"
)]
pub(crate) fn post(proxy: &EventLoopProxy<UserEvent>, event: UserEvent) {
    let Ok(()) = proxy.send_event(event) else { return; };
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

pub struct App {
    source: Arc<Source>,
    worker: Worker,
    gl: Option<Gl>,
    fatal_error: Option<eros::ErrorUnion>,
    /// Reading position: the spread start index and the pairing mode.
    index: usize,
    mode: ViewMode,
    /// The spread's placement in the viewport, with its fit lifecycle.
    fit: view::Fit,
    /// What each seat shows, under the hold and commit policy.
    screen: Screen,
    /// The spread the worker was last tuned to; the screen's shown lags
    /// until the plan commits it.
    retuned_spread: Spread,
    /// Cursor position in viewport px, same as window px.
    cursor: (f32, f32),
    dragging: bool,
    modifiers: ModifiersState,
    waiting_since: Option<Instant>,
    mode_toast_since: Option<Instant>,
    /// The viewport the last frame painted into, read by input between frames.
    last_viewport: Viewport,
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
        let mode = ViewMode::Normal;
        let spread = view::spread_for(index, mode, source.len());
        Self {
            source,
            worker,
            gl: None,
            fatal_error: None,
            index,
            mode,
            fit: view::Fit::default(),
            screen: Screen::default(),
            retuned_spread: spread,
            cursor: (0.0, 0.0),
            dragging: false,
            modifiers: ModifiersState::empty(),
            waiting_since: None,
            mode_toast_since: None,
            last_viewport: Viewport::default(),
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
        view::spread_for(self.index, self.mode, self.source.len())
    }

    fn is_fullscreen(&self) -> bool {
        self.gl
            .as_ref()
            .is_some_and(|gl| gl.window.winit().fullscreen().is_some())
    }

    fn frame(&mut self) -> eros::Result<()> {
        use crate::decode;
        use crate::ui;
        use crate::ui::{PageFacts, StatusLine, Wait};

        let spread = self.spread();
        if spread != self.retuned_spread {
            self.retuned_spread = spread;
            self.worker.retune(spread);
        }

        let Some(size) = self.sync_surface_size() else {
            return Ok(());
        };

        let snapshot = self.worker.snapshot(&spread);
        if let Some(gl) = self.gl.as_mut() {
            let committed = self.screen.display(&mut gl.pages, spread, self.mode, &snapshot);
            self.screen.stage_forward(
                &mut gl.pages,
                spread,
                self.mode,
                self.source.len(),
                &self.worker,
            );
            if committed {
                self.fit.commit();
            }
        }

        // Refit before the status line reads the scale, so the bar does
        // not show a superseded zoom.
        let layout = view::layout(self.screen.seat_dims());
        let spread_size = layout.total();
        self.fit.resolve(spread_size, self.last_viewport);

        let folded = ui::fold_status(spread, &snapshot);
        self.waiting_since = folded
            .waiting
            .then(|| self.waiting_since.unwrap_or_else(Instant::now));
        // The parked loop otherwise never shows the wait flag.
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

        let speaking_page = folded.page.map_or(spread.first, |(page, _, _)| page);
        let page = folded.page.map(|(page, decoded, dims)| PageFacts {
            claimed: self.source.name(page).and_then(decode::format_from_name),
            decoded,
            dims,
        });
        let waiting = self
            .waiting_since
            .is_some_and(|since| since.elapsed() >= WAIT_GRACE)
            .then(|| if self.source.unpacking() { Wait::Unpacking } else { Wait::Decoding });
        let status = StatusLine {
            total: self.source.len(),
            spread,
            mode: self.mode,
            toast: self.mode_toast_since.is_some(),
            name: self.source.page_name(speaking_page),
            page,
            zoom: self.fit.scale(),
            waiting,
            error: folded.error.or_else(|| self.screen.error().map(str::to_owned)),
        };

        // `EguiGlow::run` only records shapes, so egui lays out before
        // pages are placed; `paint` later draws the bar above them.
        // Fullscreen is chrome-free: the bar hides and the viewport is
        // the whole window.
        let viewport = {
            let bar = !self.is_fullscreen();
            let Some(gl) = self.gl.as_mut() else { return Ok(()) };
            let mut content = egui::Rect::NOTHING;
            gl.egui.run(gl.window.winit(), |ui| content = ui::show(ui, &status, bar));
            content_viewport(size, content, gl.egui.egui_ctx.pixels_per_point())
        };
        // The bar's height changes without a winit resize, so diff every
        // frame; the status line used the old scale, so redraw once.
        if viewport != self.last_viewport {
            self.fit.refit();
            self.last_viewport = viewport;
            self.request_redraw();
        }
        self.fit.resolve(spread_size, viewport);

        let Some(gl) = self.gl.as_mut() else { return Ok(()) };

        gl.egui.painter.clear([size.width, size.height], clear_color(gl.window.translucent()));
        gl.pages.draw((size.width, size.height), viewport, &layout, &self.fit.placement());
        gl.egui.paint(gl.window.winit());
        gl.window.pre_present_and_swap()?;
        Ok(())
    }

    /// Bring the GL surface up to the window's current size and report
    /// it; None while the window is zero-sized (minimized).
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
        (view::layout(self.screen.seat_dims()).total(), self.last_viewport)
    }

    fn dispatch(&mut self, act: Action, event_loop: &ActiveEventLoop) {
        match act {
            Action::Quit => event_loop.exit(),
            Action::Escape => {
                // Esc steps out of fullscreen first; quitting is the
                // bare-window case.
                if self.is_fullscreen() {
                    if let Some(gl) = self.gl.as_ref() {
                        gl.window.winit().set_fullscreen(None);
                    }
                } else {
                    event_loop.exit();
                }
            }
            Action::SelectMode(mode) => {
                let switch = view::switch_mode(self.mode, mode, self.index, self.source.len());
                self.mode = mode;
                self.index = switch.index;
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
                    let request = (!self.is_fullscreen()).then_some(Fullscreen::Borderless(None));
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
                self.index = if matches!(act, Action::First) {
                    view::first_spread()
                } else {
                    view::last_spread(self.mode, self.source.len())
                };
                self.after_page_change();
            }
            Action::NextBook | Action::PrevBook => {
                self.switch_book(matches!(act, Action::NextBook));
            }
            Action::Right | Action::Left | Action::Forward | Action::Back => {
                let forward = act.is_forward(self.mode);
                self.index = view::advance(self.index, forward, self.mode, self.source.len());
                self.after_page_change();
            }
        }
    }

    /// Open the next (`forward`) or previous sibling book of the same
    /// kind in this book's parent directory, landing on its cover. A
    /// failed switch shows the error in the status line and keeps the
    /// current book.
    fn switch_book(&mut self, forward: bool) {
        let Some(path) = self.source.sibling(forward) else { return };
        let source = match Source::open(&path) {
            Ok(source) => Arc::new(source),
            Err(error) => {
                self.screen.note_error(error.to_string());
                self.request_redraw();
                return;
            }
        };
        let spread = view::spread_for(view::first_spread(), self.mode, source.len());
        let proxy = self.proxy.clone();
        let worker = match Worker::spawn(Arc::clone(&source), spread, move |page| {
            post(&proxy, UserEvent::PagesUpdated(page));
        }) {
            Ok(worker) => worker,
            Err(error) => {
                self.screen.note_error(error.to_string());
                self.request_redraw();
                return;
            }
        };
        // The old worker may sit mid-decode on the old book; detaching
        // keeps the switch instant.
        self.worker.detach();
        self.worker = worker;
        self.source = source;
        self.index = view::first_spread();
        self.retuned_spread = spread;
        self.screen.clear_book();
        if let Some(gl) = self.gl.as_mut() {
            gl.pages.clear_all();
        }
        self.after_page_change();
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
            UserEvent::PagesUpdated(page) => {
                if screen::page_watched(page, self.spread(), self.mode, self.source.len()) {
                    self.request_redraw();
                }
            }
        }
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, cause: StartCause) {
        if let StartCause::ResumeTimeReached { .. } = cause {
            self.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Quit must not wait out an in-flight decode; Drop would join it.
        self.worker.detach();
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

        let egui_wants_pointer = if matches!(event, WindowEvent::RedrawRequested) {
            // RedrawRequested carries nothing for egui, and egui-winit
            // answers it with `repaint: true`; honoring that answer would
            // re-request the redraw and self-sustain the loop at
            // compositor refresh while idle.
            gl.egui.egui_ctx.egui_wants_pointer_input()
        } else {
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
                self.sync_surface_size();
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
                self.fit.zoom(zoom_factor, self.cursor, total, viewport);
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
const GLASS_ALPHA: f32 = 0.7;

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
