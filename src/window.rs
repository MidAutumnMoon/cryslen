//! Window and GL context creation (glutin-on-winit). Creation returns
//! errors for main to report.

use core::num::NonZeroU32;
use std::sync::Arc;

use egui_winit::winit;
use winit::raw_window_handle::HasWindowHandle as _;

/// Window + EGL context + surface. vsync is DontWait, so the app must call
/// `pre_present_and_swap()` to present (Wayland frame pacing inside).
pub struct Window {
    pub winit: winit::window::Window,
    gl: Arc<glow::Context>,
    surface: glutin::surface::Surface<glutin::surface::WindowSurface>,
    context: glutin::context::PossiblyCurrentContext,
    /// Size the surface was last resized to (physical px). Creation
    /// counts as the first resize.
    resized_to: (u32, u32),
}

impl Window {
    pub fn new(event_loop: &winit::event_loop::ActiveEventLoop) -> eros::Result<Window> {
        unsafe { Self::create(event_loop) }
    }

    /// Prefer desktop GL, fall back to GLES. The swap is non-blocking.
    ///
    /// # Safety
    ///
    /// `event_loop` must be the loop this window runs on, because the raw
    /// display handle taken from it backs the GL display.
    unsafe fn create(event_loop: &winit::event_loop::ActiveEventLoop) -> eros::Result<Window> {
        use glutin::context::NotCurrentGlContext as _;
        use glutin::display::GlDisplay as _;
        use glutin::prelude::GlSurface as _;
        use winit::raw_window_handle::HasDisplayHandle as _;

        use winit::platform::wayland::WindowAttributesExtWayland as _;

        let attrs = winit::window::WindowAttributes::default()
            .with_name("cryslen", "cryslen")
            .with_resizable(true)
            .with_inner_size(winit::dpi::LogicalSize { width: INITIAL_WINDOW_WIDTH, height: INITIAL_WINDOW_HEIGHT })
            .with_title("cryslen")
            // Hidden at creation. app.rs reveals the window after the
            // first presented frame.
            .with_visible(false);

        // glutin-winit's DisplayBuilder picks a config through a closure
        // that cannot fail, so using it would force a panic on an empty
        // config list.
        let raw_display = event_loop.display_handle()?.as_raw();
        let gl_display = glutin::display::Display::new(
            raw_display,
            glutin::display::DisplayApiPreference::Egl,
        )?;
        let template = glutin::config::ConfigTemplateBuilder::new().build();
        let gl_config = gl_display
            .find_configs(template)?
            .next()
            .ok_or_else(|| eros::error!("no GL config matched the template"))?;

        let window = glutin_winit::finalize_window(event_loop, attrs, &gl_config)?;
        let raw_window = window.window_handle().expect("window handle").as_raw();
        let ctx_attrs =
            glutin::context::ContextAttributesBuilder::new().build(Some(raw_window));
        let fallback = glutin::context::ContextAttributesBuilder::new()
            .with_context_api(glutin::context::ContextApi::Gles(None))
            .build(Some(raw_window));
        let not_current = match gl_display.create_context(&gl_config, &ctx_attrs) {
            Ok(context) => context,
            Err(desktop) => gl_display
                .create_context(&gl_config, &fallback)
                .map_err(|gles| {
                    eros::error!("GL context creation failed. desktop: {desktop}. GLES: {gles}")
                })?,
        };

        let (w, h): (u32, u32) = window.inner_size().into();
        let surface_attrs =
            glutin::surface::SurfaceAttributesBuilder::<glutin::surface::WindowSurface>::new()
                .build(
                    raw_window,
                    NonZeroU32::new(w).expect("nonzero window width"),
                    NonZeroU32::new(h).expect("nonzero window height"),
                );
        let surface = gl_display.create_window_surface(&gl_config, &surface_attrs)?;
        let context = not_current.make_current(&surface)?;
        // Non-blocking present plus pre_present_and_swap's
        // window.pre_present_notify() gives Wayland frame-callback pacing
        // without a blocking eglSwapBuffers owning the event loop.
        surface
            .set_swap_interval(&context, glutin::surface::SwapInterval::DontWait)?;

        let gl = Arc::new(glow::Context::from_loader_function(|s| {
            let s = std::ffi::CString::new(s).expect("GL proc name without interior NUL");
            gl_display.get_proc_address(&s)
        }));
        Ok(Window { winit: window, gl, surface, context, resized_to: (w, h) })
    }

    pub fn glow(&self) -> Arc<glow::Context> {
        Arc::clone(&self.gl)
    }

    /// Resize the surface when the window size changed since the last
    /// call. No-op otherwise.
    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if (size.width, size.height) == self.resized_to {
            return;
        }
        use glutin::prelude::GlSurface as _;
        self.surface.resize(
            &self.context,
            size.width.try_into().expect("nonzero surface width"),
            size.height.try_into().expect("nonzero surface height"),
        );
        self.resized_to = (size.width, size.height);
    }

    /// Frame-callback notify then non-blocking swap.
    pub fn pre_present_and_swap(&self) {
        use glutin::prelude::GlSurface as _;
        self.winit.pre_present_notify();
        self.surface.swap_buffers(&self.context).expect("swap buffers (GL context lost)");
    }
}

/// Initial window size, logical px.
const INITIAL_WINDOW_WIDTH: f64 = 1000.0;
const INITIAL_WINDOW_HEIGHT: f64 = 750.0;
