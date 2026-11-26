//! Wayland window, EGL context, egui host, and page texture store.
//! Drops `egui` and `pages` while `context` is current, then GL/surface/winit.

use core::num::NonZeroU32;
use std::sync::Arc;

use egui::Color32;
use egui::Context as EguiContext;
use egui::Rect;
use egui::TexturesDelta;
use egui::Ui;
use egui::epaint::ClippedShape;
use egui_glow::Painter;
#[cfg(debug_assertions)]
use egui_glow::glow;
use egui_glow::glow::Context as GlowContext;
#[cfg(debug_assertions)]
use egui_glow::glow::HasContext as _;
use glutin::context::PossiblyCurrentContext;
use glutin::surface::Surface;
use glutin::surface::WindowSurface;
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::OwnedDisplayHandle;
use winit::window::Window as WinitWindow;

use crate::image::pixels::ImageLimits;
use crate::render::textures::DrawQuad;
use crate::render::textures::PageRenderer;

/// Initial window size, logical px.
const INITIAL_WINDOW_WIDTH: f64 = 1000.0;
const INITIAL_WINDOW_HEIGHT: f64 = 750.0;

const WAYLAND_APP_ID: &str = "cryslen";
const WINDOW_TITLE: &str = "cryslen";

const OPENGL_MAJOR_VERSION: u8 = 3;
const OPENGL_MINOR_VERSION: u8 = 3;
const TRANSLUCENT_CONFIG_ALPHA_BITS: u8 = 8;
const OPAQUE_CONFIG_ALPHA_BITS: u8 = 0;

const OPAQUE_BACKGROUND_COLOR: Color32 = Color32::from_rgb(20, 20, 23);
const TRANSLUCENT_BACKGROUND_ALPHA: u8 = 179;
const TRANSLUCENT_BACKGROUND_COLOR: Color32 =
    Color32::from_rgba_unmultiplied_const(
        OPAQUE_BACKGROUND_COLOR.r(),
        OPAQUE_BACKGROUND_COLOR.g(),
        OPAQUE_BACKGROUND_COLOR.b(),
        TRANSLUCENT_BACKGROUND_ALPHA,
    );

struct EguiFrame {
    shapes: Vec<ClippedShape>,
    textures_delta: TexturesDelta,
    pixels_per_point: f32,
    repaint_requested: bool,
}

struct Egui {
    context: EguiContext,
    painter: Painter,
}

impl Egui {
    fn new(
        gl: Arc<GlowContext>,
        panel_fill: Option<Color32>,
    ) -> eros::Result<Self> {
        use egui_system_font_provider::SystemFontProvider;

        let painter = Painter::new(gl, "", None, true)?;
        let context = EguiContext::default();
        // System fonts cover filename characters the built-in fonts cannot
        // render.
        context.add_font_provider(Arc::new(SystemFontProvider::new()));
        apply_dark_theme(&context, panel_fill);
        Ok(Self { context, painter })
    }

    fn layout<UiOutput>(
        &self,
        size: PhysicalSize<u32>,
        scale_factor: f64,
        mut run_ui: impl FnMut(&mut Ui) -> UiOutput,
    ) -> (EguiFrame, UiOutput) {
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
        let logical_size = size.to_logical::<f32>(scale_factor);
        let screen_rect = (logical_size.width > 0.0
            && logical_size.height > 0.0)
            .then(|| {
                Rect::from_min_size(
                    Pos2::ZERO,
                    egui::vec2(logical_size.width, logical_size.height),
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
        let mut ui_output: Option<UiOutput> = None;
        let output = self.context.run_ui(input, |ui| {
            ui_output = Some(run_ui(ui));
        });
        let repaint_requested = output
            .viewport_output
            .get(&ViewportId::ROOT)
            .is_some_and(|viewport| viewport.repaint_delay.is_zero());
        #[expect(
            clippy::expect_used,
            reason = "egui::Context::run_ui always invokes its ui closure once"
        )]
        let ui_output = ui_output.expect("egui ran the ui closure");
        (
            EguiFrame {
                shapes: output.shapes,
                textures_delta: output.textures_delta,
                pixels_per_point: output.pixels_per_point,
                repaint_requested,
            },
            ui_output,
        )
    }

    fn clear(&self, size: PhysicalSize<u32>, color: [f32; 4]) {
        self.painter.clear([size.width, size.height], color);
    }

    fn paint(&mut self, size: PhysicalSize<u32>, mut frame: EguiFrame) {
        let clipped_primitives = self
            .context
            .tessellate(frame.shapes, frame.pixels_per_point);
        self.painter.paint_and_update_textures(
            [size.width, size.height],
            frame.pixels_per_point,
            &clipped_primitives,
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

/// Window, EGL context, egui host, and page texture store. vsync is
/// `DontWait`, so the app calls `present()` to draw and swap buffers.
pub struct Window {
    // Drop order: `egui` and `pages` while `context` is current, `context`
    // before `surface`, `surface` before `winit`, and `_display` last.
    egui: Egui,
    pages: PageRenderer,
    _gl: Arc<GlowContext>,
    context: PossiblyCurrentContext,
    surface: Surface<WindowSurface>,
    winit: WinitWindow,
    _display: OwnedDisplayHandle,
    limits: ImageLimits,
    /// Alpha config matched. The fallback may still carry alpha bits; the
    /// opaque hint and clear hide them.
    translucent: bool,
    /// Size the surface was last resized to. Creation counts as the
    /// first resize.
    resized_to: (u32, u32),
}

impl Window {
    /// Create the window with a desktop OpenGL 3.3 core context.
    pub fn new(event_loop: &ActiveEventLoop) -> eros::Result<Self> {
        use std::ffi::CString;

        use glutin::config::Api as ConfigApi;
        use glutin::config::ConfigTemplateBuilder;
        use glutin::context::ContextApi;
        use glutin::context::ContextAttributesBuilder;
        use glutin::context::GlProfile;
        use glutin::context::NotCurrentGlContext as _;
        use glutin::context::Version;
        use glutin::display::Display;
        use glutin::display::DisplayApiPreference;
        use glutin::display::GlDisplay as _;
        use glutin::prelude::GlSurface as _;
        use glutin::surface::SurfaceAttributesBuilder;
        use glutin::surface::SwapInterval;
        use winit::dpi::LogicalSize;
        use winit::platform::wayland::WindowAttributesExtWayland as _;
        use winit::raw_window_handle::HasDisplayHandle as _;
        use winit::raw_window_handle::HasWindowHandle as _;
        use winit::window::WindowAttributes;

        let display = event_loop.owned_display_handle();
        let raw_display = display.display_handle()?.as_raw();
        // SAFETY: `display` outlives `gl_display` locally and `_display` drops
        // last in `Window`.
        let gl_display = unsafe {
            Display::new(raw_display, DisplayApiPreference::Egl)
        }?;
        // Prefer a config with alpha so the desktop shows through; a machine
        // without one runs opaque.
        let translucent_template = ConfigTemplateBuilder::new()
            .with_api(ConfigApi::OPENGL)
            .with_alpha_size(TRANSLUCENT_CONFIG_ALPHA_BITS)
            .build();
        // SAFETY: `translucent_template` sets no raw window handle.
        let mut configs =
            unsafe { gl_display.find_configs(translucent_template) }?;
        let (gl_config, translucent) = if let Some(config) = configs.next()
        {
            (config, true)
        } else {
            let opaque_template = ConfigTemplateBuilder::new()
                .with_api(ConfigApi::OPENGL)
                .with_alpha_size(OPAQUE_CONFIG_ALPHA_BITS)
                .build();
            // SAFETY: `opaque_template` sets no raw window handle.
            let config = unsafe {
                gl_display.find_configs(opaque_template)
            }?
            .next()
            .ok_or_else(|| {
                eros::error!(
                    "no desktop OpenGL 3.3 core config on this EGL display"
                )
            })?;
            (config, false)
        };

        let window_attributes = WindowAttributes::default()
            .with_name(WAYLAND_APP_ID, WAYLAND_APP_ID)
            .with_resizable(true)
            .with_inner_size(LogicalSize {
                width: INITIAL_WINDOW_WIDTH,
                height: INITIAL_WINDOW_HEIGHT,
            })
            .with_title(WINDOW_TITLE)
            .with_transparent(translucent);

        let window = glutin_winit::finalize_window(
            event_loop,
            window_attributes,
            &gl_config,
        )?;
        let raw_window = window.window_handle()?.as_raw();
        let context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(
                OPENGL_MAJOR_VERSION,
                OPENGL_MINOR_VERSION,
            ))))
            .with_profile(GlProfile::Core)
            .build(Some(raw_window));
        // SAFETY: `window` owns `raw_window` and outlives the context.
        let pending_context = unsafe {
            gl_display.create_context(&gl_config, &context_attributes)
        }?;

        let (width, height): (u32, u32) = window.inner_size().into();
        let surface_attributes =
            SurfaceAttributesBuilder::<WindowSurface>::new().build(
                raw_window,
                NonZeroU32::new(width)
                    .ok_or_else(|| eros::error!("window width is zero"))?,
                NonZeroU32::new(height).ok_or_else(|| {
                    eros::error!("window height is zero")
                })?,
            );
        // SAFETY: `window` owns `raw_window` and outlives `surface`.
        let surface = unsafe {
            gl_display
                .create_window_surface(&gl_config, &surface_attributes)
        }?;
        let context = pending_context.make_current(&surface)?;
        surface.set_swap_interval(&context, SwapInterval::DontWait)?;

        // SAFETY: `context` is current and `gl_display` is alive for lookups.
        let gl = unsafe {
            GlowContext::from_loader_function(|symbol| {
                #[expect(
                    clippy::expect_used,
                    reason = "symbol names are static ASCII GL entry points, NUL-free by construction"
                )]
                let symbol = CString::new(symbol)
                    .expect("GL proc name without interior NUL");
                gl_display.get_proc_address(&symbol)
            })
        };
        #[cfg(debug_assertions)]
        let mut gl = gl;
        #[cfg(debug_assertions)]
        if gl.supports_debug() {
            // SAFETY: the context is current, supports debug output, and the
            // callback only logs.
            unsafe {
                gl.debug_message_callback(
                    |source, kind, id, severity, message| {
                        if severity == glow::DEBUG_SEVERITY_NOTIFICATION {
                            return;
                        }
                        tracing::debug!(
                            source,
                            kind,
                            id,
                            severity,
                            message = message,
                            "GL debug message"
                        );
                    },
                );
            }
            // SAFETY: the context is current and debug output callback is installed.
            unsafe {
                gl.enable(glow::DEBUG_OUTPUT);
            }
        }
        let gl = Arc::new(gl);
        let pages = PageRenderer::new(Arc::clone(&gl))?;
        let limits = compute_image_limits(
            &window,
            event_loop,
            pages.max_texture_edge(),
        );
        let panel_fill =
            translucent.then_some(TRANSLUCENT_BACKGROUND_COLOR);
        let egui = Egui::new(Arc::clone(&gl), panel_fill)?;
        Ok(Self {
            egui,
            pages,
            _gl: gl,
            context,
            surface,
            winit: window,
            _display: display,
            limits,
            translucent,
            resized_to: (width, height),
        })
    }

    #[must_use]
    pub const fn limits(&self) -> ImageLimits {
        self.limits
    }

    #[must_use]
    pub const fn pages(&self) -> &PageRenderer {
        &self.pages
    }

    pub const fn pages_mut(&mut self) -> &mut PageRenderer {
        &mut self.pages
    }

    #[must_use]
    pub const fn winit(&self) -> &WinitWindow {
        &self.winit
    }

    /// Sync the EGL surface to the window's inner size; returns `None` if minimized.
    pub fn sync_surface_size(&mut self) -> Option<PhysicalSize<u32>> {
        let size = self.winit.inner_size();
        if size.width == 0 || size.height == 0 {
            return None;
        }
        self.resize_surface(size);
        Some(size)
    }

    /// Lay out UI, clear, draw page quads, paint UI, and swap buffers.
    /// Returns `Ok(true)` when egui requested an immediate repaint.
    pub fn present<State, UiOutput>(
        &mut self,
        state: &mut State,
        mut build_ui: impl FnMut(
            &mut Ui,
            &PageRenderer,
            &mut State,
        ) -> UiOutput,
        build_quads: impl FnOnce(
            PhysicalSize<u32>,
            f64,
            UiOutput,
            &mut State,
        ) -> ((u32, u32), Vec<DrawQuad>),
    ) -> eros::Result<bool> {
        use glutin::prelude::GlSurface as _;

        let Some(size) = self.sync_surface_size() else {
            return Ok(false);
        };
        let scale_factor = self.winit.scale_factor();
        let (frame, ui_output) =
            self.egui.layout(size, scale_factor, |ui| {
                build_ui(ui, &self.pages, state)
            });
        let repaint_requested = frame.repaint_requested;
        let (clip_size, quads) =
            build_quads(size, scale_factor, ui_output, state);
        self.egui.clear(size, clear_color(self.translucent));
        if !quads.is_empty() {
            self.pages
                .draw((size.width, size.height), clip_size, &quads);
        }
        self.egui.paint(size, frame);
        self.winit.pre_present_notify();
        self.surface.swap_buffers(&self.context)?;
        Ok(repaint_requested)
    }

    /// Resize the surface when the window size changed since the last
    /// call. A zero size (minimized) and an unchanged size are no-ops.
    fn resize_surface(&mut self, size: PhysicalSize<u32>) {
        use glutin::prelude::GlSurface as _;

        let (Some(width), Some(height)) =
            (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        else {
            return;
        };
        if (size.width, size.height) == self.resized_to {
            return;
        }
        self.surface.resize(&self.context, width, height);
        self.resized_to = (size.width, size.height);
    }
}

fn compute_image_limits(
    winit: &WinitWindow,
    event_loop: &ActiveEventLoop,
    max_texture_edge: u32,
) -> ImageLimits {
    let monitor_box = winit
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
    let window_size = winit.inner_size();
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

/// Framebuffer clear color, premultiplied because Wayland composites
/// ARGB buffers that way.
fn clear_color(translucent: bool) -> [f32; 4] {
    let color = if translucent {
        TRANSLUCENT_BACKGROUND_COLOR
    } else {
        OPAQUE_BACKGROUND_COLOR
    };
    color.to_normalized_gamma_f32()
}

/// A light desktop theme would otherwise pick dark status text over the
/// dark window.
fn apply_dark_theme(context: &EguiContext, panel_fill: Option<Color32>) {
    use egui::Theme;

    context.set_theme(Theme::Dark);
    if let Some(fill) = panel_fill {
        context.global_style_mut(|style| {
            style.visuals.panel_fill = fill;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn egui_stays_dark_when_the_desktop_is_light() {
        use egui::RawInput;
        use egui::Theme;

        let context = EguiContext::default();
        apply_dark_theme(&context, Some(TRANSLUCENT_BACKGROUND_COLOR));
        context
            .run_ui(
                RawInput {
                    system_theme: Some(Theme::Light),
                    ..Default::default()
                },
                |_| {},
            )
            .drop_without_applying_deltas();
        assert_eq!(context.theme(), Theme::Dark);
        assert_eq!(
            context.global_style().visuals.panel_fill,
            TRANSLUCENT_BACKGROUND_COLOR
        );
    }
}
