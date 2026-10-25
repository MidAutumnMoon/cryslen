//! Window and GL context creation (glutin-on-winit).

use core::num::NonZeroU32;
use std::sync::Arc;

use egui_winit::winit;
use glutin::context::PossiblyCurrentContext;
use glutin::surface::Surface;
use glutin::surface::WindowSurface;
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::raw_window_handle::HasWindowHandle as _;
use winit::window::Window as WinitWindow;

/// Window + EGL context + surface. vsync is `DontWait`, so the app must call
/// `pre_present_and_swap()` to present.
pub struct Window {
    winit: WinitWindow,
    gl: Arc<glow::Context>,
    surface: Surface<WindowSurface>,
    context: PossiblyCurrentContext,
    /// True when the alpha-demanding EGL query matched. False runs
    /// fully opaque. The fallback config may still carry alpha bits,
    /// but the opaque hint and clear make that invisible.
    translucent: bool,
    /// Size the surface was last resized to (physical px). Creation
    /// counts as the first resize.
    resized_to: (u32, u32),
}

impl Window {
    /// Prefer desktop GL, fall back to GLES.
    ///
    /// # Safety
    ///
    /// The event loop must outlive the returned window: the platform
    /// display behind its raw handle backs every GL object created here.
    pub unsafe fn new(event_loop: &ActiveEventLoop) -> eros::Result<Self> {
        use glutin::config::ConfigTemplateBuilder;
        use glutin::context::ContextApi;
        use glutin::context::ContextAttributesBuilder;
        use glutin::context::NotCurrentGlContext as _;
        use glutin::display::Display;
        use glutin::display::DisplayApiPreference;
        use glutin::display::GlDisplay as _;
        use glutin::prelude::GlSurface as _;
        use glutin::surface::SurfaceAttributesBuilder;
        use glutin::surface::SwapInterval;
        use std::ffi::CString;
        use winit::dpi::LogicalSize;
        use winit::raw_window_handle::HasDisplayHandle as _;

        use winit::platform::wayland::WindowAttributesExtWayland as _;
        use winit::window::WindowAttributes;

        // glutin-winit's DisplayBuilder picks a config through a closure
        // that cannot fail, so using it would force a panic on an empty
        // config list.
        let raw_display = event_loop.display_handle()?.as_raw();
        let gl_display = Display::new(
            raw_display,
            DisplayApiPreference::Egl,
        )?;
        // Prefer a config with alpha so the background can show the
        // desktop through it. A machine with none runs fully opaque,
        // a mode, not an error.
        let translucent_template = ConfigTemplateBuilder::new().with_alpha_size(8).build();
        let (gl_config, translucent) =
            if let Some(config) = gl_display.find_configs(translucent_template)?.next() {
                (config, true)
            } else {
                // Alpha-agnostic second pass: the default template also
                // demands alpha, which would repeat the same empty query.
                let plain = ConfigTemplateBuilder::new().with_alpha_size(0).build();
                let config = gl_display
                    .find_configs(plain)?
                    .next()
                    .ok_or_else(|| eros::error!("no GL config matched the template"))?;
                (config, false)
            };

        let attrs = WindowAttributes::default()
            .with_name("cryslen", "cryslen")
            .with_resizable(true)
            .with_inner_size(LogicalSize { width: INITIAL_WINDOW_WIDTH, height: INITIAL_WINDOW_HEIGHT })
            .with_title("cryslen")
            // Without the transparent hint winit marks the whole surface
            // opaque, so compositors may skip blending and background
            // effects behind it even though the buffer has alpha.
            .with_transparent(translucent)
            // Hidden at creation, revealed after the first presented frame.
            .with_visible(false);

        let window = glutin_winit::finalize_window(event_loop, attrs, &gl_config)?;
        let raw_window = window.window_handle()?.as_raw();
        let ctx_attrs =
            ContextAttributesBuilder::new().build(Some(raw_window));
        let fallback = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::Gles(None))
            .build(Some(raw_window));
        let not_current = match gl_display.create_context(&gl_config, &ctx_attrs) {
            Ok(context) => context,
            Err(desktop) => gl_display
                .create_context(&gl_config, &fallback)
                .map_err(|gles| {
                    eros::error!("GL context creation failed. desktop: {desktop}. GLES: {gles}")
                })?,
        };

        let (width, height): (u32, u32) = window.inner_size().into();
        let surface_attrs =
            SurfaceAttributesBuilder::<WindowSurface>::new()
                .build(
                    raw_window,
                    NonZeroU32::new(width).ok_or_else(|| eros::error!("window width is zero"))?,
                    NonZeroU32::new(height).ok_or_else(|| eros::error!("window height is zero"))?,
                );
        let surface = gl_display.create_window_surface(&gl_config, &surface_attrs)?;
        let context = not_current.make_current(&surface)?;
        // Non-blocking present plus pre_present_and_swap's
        // window.pre_present_notify() gives Wayland frame-callback pacing
        // without a blocking eglSwapBuffers owning the event loop.
        surface
            .set_swap_interval(&context, SwapInterval::DontWait)?;

        let gl = Arc::new(glow::Context::from_loader_function(|symbol| {
            #[expect(
                clippy::expect_used,
                reason = "symbol names are static ASCII GL entry points, NUL-free by construction"
            )]
            let symbol = CString::new(symbol).expect("GL proc name without interior NUL");
            gl_display.get_proc_address(&symbol)
        }));
        Ok(Self { winit: window, gl, surface, context, translucent, resized_to: (width, height) })
    }

    pub fn glow(&self) -> Arc<glow::Context> {
        Arc::clone(&self.gl)
    }

    /// Lets a compositor that blurs translucent surfaces show the
    /// desktop through the window.
    pub fn translucent(&self) -> bool {
        self.translucent
    }

    pub(crate) fn winit(&self) -> &WinitWindow {
        &self.winit
    }

    /// Resize the surface when the window size changed since the last
    /// call. A zero size (minimized) and an unchanged size are no-ops.
    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        use glutin::prelude::GlSurface as _;

        let (Some(width), Some(height)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        else {
            return;
        };
        if (size.width, size.height) == self.resized_to {
            return;
        }
        self.surface.resize(&self.context, width, height);
        self.resized_to = (size.width, size.height);
    }

    pub fn pre_present_and_swap(&self) -> eros::Result<()> {
        use glutin::prelude::GlSurface as _;
        self.winit.pre_present_notify();
        self.surface.swap_buffers(&self.context)?;
        Ok(())
    }
}

/// Initial window size, logical px.
const INITIAL_WINDOW_WIDTH: f64 = 1000.0;
const INITIAL_WINDOW_HEIGHT: f64 = 750.0;
