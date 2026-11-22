//! Geometry: the placement of a composed spread in the viewport, fit
//! modes, zoom, and pan. Pure data, no GL.

use strum::VariantArray;

/// Where pages live: the window area left after egui chrome, in physical
/// px, anchored at the window's top-left.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Viewport {
    size: (u32, u32),
}

impl Viewport {
    /// A viewport filling the window above a bottom chrome strip whose top
    /// edge sits `bar_top` physical px from the window top. The only sizing
    /// constructor, so every non-empty viewport is chrome-aware.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the clamped height is a non-negative integer within window.1 (u32)"
    )]
    pub fn above_bottom_bar(window: (u32, u32), bar_top: f64) -> Self {
        let height = if bar_top.is_nan() {
            0
        } else {
            // Nearest whole px matches egui's painter clip-rect rounding.
            bar_top.round().clamp(0.0, f64::from(window.1)) as u32
        };
        Self {
            size: (window.0, height),
        }
    }

    pub fn size(self) -> (u32, u32) {
        self.size
    }

    fn size_f64(self) -> (f64, f64) {
        (f64::from(self.size.0), f64::from(self.size.1))
    }
}

/// Placement of the composed spread: scale is screen px per image px, offset
/// is the position of the spread's top-left corner in viewport px.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    pub scale: f64,
    pub offset: (f64, f64),
}

impl Default for Placement {
    fn default() -> Self {
        Self {
            scale: 1.0,
            offset: (0.0, 0.0),
        }
    }
}

/// How a spread fits the viewport when the view is not user-placed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, VariantArray)]
pub enum FitMode {
    #[default]
    Page,
    Width,
    Height,
    Actual,
}

/// A page's rect inside the composed spread, in image-pixel space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageRect {
    pub pos: (f64, f64),
    pub size: (f64, f64),
}

/// Layout of a spread's visual seats. `rects` is always [left, right].
#[derive(Debug, Clone, Copy, Default)]
pub struct SpreadLayout {
    pub rects: [Option<PageRect>; 2],
}

impl SpreadLayout {
    /// Bounding size of the composed spread, in image pixels.
    pub fn total(&self) -> (f64, f64) {
        self.rects
            .iter()
            .flatten()
            .fold((0.0, 0.0), |total, page_rect| {
                (total.0 + page_rect.size.0, total.1.max(page_rect.size.1))
            })
    }
}

/// `dims[i]` is the page size in visual seat i, 0 = left, 1 = right.
pub fn layout(dims: [Option<(u32, u32)>; 2]) -> SpreadLayout {
    let mut next_seat_x = 0.0;
    SpreadLayout {
        rects: dims.map(|seat_dims| {
            seat_dims.map(|(width, height)| {
                let (width, height) =
                    (f64::from(width), f64::from(height));
                let rect = PageRect {
                    pos: (next_seat_x, 0.0),
                    size: (width, height),
                };
                next_seat_x += width;
                rect
            })
        }),
    }
}

/// Placement of one fit mode, centered on both axes. `Page` fits both
/// axes, `Width` and `Height` fill their axis, `Actual` maps one image px
/// to one screen px. Degenerate inputs are inert.
fn fit_as(
    mode: FitMode,
    total: (f64, f64),
    viewport: Viewport,
) -> Placement {
    let size = viewport.size_f64();
    if total.0 <= 0.0 || total.1 <= 0.0 || size.0 <= 0.0 || size.1 <= 0.0 {
        return Placement {
            scale: 1.0,
            offset: (0.0, 0.0),
        };
    }
    let scale = match mode {
        FitMode::Page => (size.0 / total.0).min(size.1 / total.1),
        FitMode::Width => size.0 / total.0,
        FitMode::Height => size.1 / total.1,
        FitMode::Actual => 1.0,
    };
    Placement {
        scale,
        offset: (
            total.0.mul_add(-scale, size.0) / 2.0,
            total.1.mul_add(-scale, size.1) / 2.0,
        ),
    }
}

/// Zoom scale bounds, in screen px per image px. Applied at zoom time
/// only; `fit_as` does not clamp.
const MIN_SCALE: f64 = 0.05;
const MAX_SCALE: f64 = 40.0;

/// Zoom by `factor`, anchoring the point under `cursor`; clamps to
/// [`MIN_SCALE`]..=[`MAX_SCALE`] with the applied factor.
fn zoom_at(
    mut placement: Placement,
    cursor: (f64, f64),
    factor: f64,
) -> Placement {
    let old = placement.scale;
    placement.scale = (old * factor).clamp(MIN_SCALE, MAX_SCALE);
    let applied = placement.scale / old;
    placement.offset = (
        (cursor.0 - placement.offset.0).mul_add(-applied, cursor.0),
        (cursor.1 - placement.offset.1).mul_add(-applied, cursor.1),
    );
    placement
}

fn pan(placement: Placement, delta: (f64, f64)) -> Placement {
    Placement {
        scale: placement.scale,
        offset: (
            placement.offset.0 + delta.0,
            placement.offset.1 + delta.1,
        ),
    }
}

/// Reachability margin for [`clamp`]: at most this many px of the spread
/// may hang off any viewport edge.
const EDGE_KEEP_PX: f64 = 64.0;

/// Clamp pan so the spread cannot be lost. An axis where the scaled
/// spread fits the viewport is pinned to the centered position.
fn clamp(
    placement: Placement,
    total: (f64, f64),
    viewport: Viewport,
) -> Placement {
    let size = viewport.size_f64();
    let scaled = (total.0 * placement.scale, total.1 * placement.scale);
    Placement {
        scale: placement.scale,
        offset: (
            clamp_axis(placement.offset.0, scaled.0, size.0),
            clamp_axis(placement.offset.1, scaled.1, size.1),
        ),
    }
}

fn clamp_axis(offset: f64, content: f64, size: f64) -> f64 {
    if content <= size {
        (size - content) / 2.0
    } else {
        offset.clamp(size - content - EDGE_KEEP_PX, EDGE_KEEP_PX)
    }
}

/// The current spread's fit: mode, live placement, and whether a commit
/// refits (a user move during the hold pins it).
#[derive(Debug, Clone, Copy, Default)]
pub struct Fit {
    mode: FitMode,
    placement: Placement,
    settled: Option<(f64, f64)>,
    user_moved: bool,
}

impl Fit {
    /// Fit `total` into `viewport` when due and return the placement. A
    /// user-placed view keeps its zoom and only clamps its pan into the
    /// new shape or viewport; otherwise the fit mode places the spread.
    pub fn resolve(
        &mut self,
        total: (f64, f64),
        viewport: Viewport,
    ) -> Placement {
        if self.settled == Some(total) {
            return self.placement;
        }
        self.placement = if self.user_moved {
            clamp(self.placement, total, viewport)
        } else {
            fit_as(self.mode, total, viewport)
        };
        self.settled = Some(total);
        self.placement
    }

    /// Zoom by `factor` around `cursor` (viewport px), clamped so the
    /// spread stays reachable.
    pub fn zoom(
        &mut self,
        factor: f64,
        cursor: (f64, f64),
        total: (f64, f64),
        viewport: Viewport,
    ) {
        self.placement = clamp(
            zoom_at(self.placement, cursor, factor),
            total,
            viewport,
        );
        self.user_moved = true;
    }

    /// Pan by `delta` viewport px, clamped so the spread stays reachable.
    pub fn pan(
        &mut self,
        delta: (f64, f64),
        total: (f64, f64),
        viewport: Viewport,
    ) {
        self.placement =
            clamp(pan(self.placement, delta), total, viewport);
        self.user_moved = true;
    }

    /// Select a fit mode and discard the placement memory, so the next
    /// `resolve` places the spread by it. Re-selecting the live mode
    /// resets the view.
    pub fn select(&mut self, mode: FitMode) {
        self.mode = mode;
        self.refit();
    }

    /// A turn or re-pairing mode switch: pre-turn user moves must not
    /// gate the next commit's fit.
    pub fn begin_turn(&mut self) {
        self.user_moved = false;
    }

    /// The app showed a new spread: the next `resolve` places it. Moves
    /// made during the hold carry into it.
    pub fn commit(&mut self) {
        self.settled = None;
    }

    /// Discard the placement memory: the next frame refits, and a user
    /// placement no longer pins the zoom.
    pub fn refit(&mut self) {
        self.settled = None;
        self.user_moved = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn above_bottom_bar_sizes_and_clamps() {
        let viewport = Viewport::above_bottom_bar((1000, 800), 740.0);
        assert_eq!(viewport.size(), (1000, 740));
        // A bar top past the window bottom degenerates to the window.
        // A negative or NaN one degenerates to nothing.
        let past = Viewport::above_bottom_bar((100, 50), 80.0);
        assert_eq!(past.size(), (100, 50));
        let negative = Viewport::above_bottom_bar((100, 50), -10.0);
        assert_eq!(negative.size(), (100, 0));
        let nan = Viewport::above_bottom_bar((100, 50), f64::NAN);
        assert_eq!(nan.size(), (100, 0));
    }

    #[test]
    fn layout_places_seats() {
        let left = (100_u32, 300);
        let right = (200, 150);
        let rect = |pos: (f64, f64), size: (f64, f64)| {
            Some(PageRect { pos, size })
        };
        let cases = [
            ([None, None], [None, None], (0.0, 0.0)),
            (
                [Some(left), None],
                [rect((0.0, 0.0), (100.0, 300.0)), None],
                (100.0, 300.0),
            ),
            (
                [None, Some(right)],
                [None, rect((0.0, 0.0), (200.0, 150.0))],
                (200.0, 150.0),
            ),
            (
                [Some(left), Some(right)],
                [
                    rect((0.0, 0.0), (100.0, 300.0)),
                    rect((100.0, 0.0), (200.0, 150.0)),
                ],
                (300.0, 300.0),
            ),
        ];
        for (dims, rects, total) in cases {
            let spread = layout(dims);
            assert_eq!(spread.rects, rects, "seats {dims:?}");
            assert_eq!(spread.total(), total, "seats {dims:?}");
        }

        // Dimensions above 2^24 keep exact integer precision in f64.
        let wide_strip = layout([Some((20_000_001, 1)), None]);
        assert_eq!(wide_strip.total(), (20_000_001.0, 1.0));
    }

    fn full_window(win: (u32, u32)) -> Viewport {
        Viewport::above_bottom_bar(win, f64::from(win.1))
    }

    #[test]
    fn fit_page_centers_on_the_min_axis() {
        let cases = [
            (
                (1000.0_f64, 2000.0),
                full_window((500, 500)),
                Placement {
                    scale: 0.25,
                    offset: (125.0, 0.0),
                },
            ),
            (
                (2000.0, 1000.0),
                full_window((500, 500)),
                Placement {
                    scale: 0.25,
                    offset: (0.0, 125.0),
                },
            ),
            // Bar-clamped viewport: centering uses the 600-px viewport,
            // not the 800-px window.
            (
                (500.0, 400.0),
                Viewport::above_bottom_bar((1000, 800), 600.0),
                Placement {
                    scale: 1.5,
                    offset: (125.0, 0.0),
                },
            ),
            // Degenerate total and empty viewport are inert.
            (
                (0.0, 100.0),
                full_window((500, 500)),
                Placement {
                    scale: 1.0,
                    offset: (0.0, 0.0),
                },
            ),
            (
                (100.0, 100.0),
                Viewport::default(),
                Placement {
                    scale: 1.0,
                    offset: (0.0, 0.0),
                },
            ),
        ];
        for (total, viewport, want) in cases {
            assert_eq!(
                fit_as(FitMode::Page, total, viewport),
                want,
                "total {total:?} size {:?}",
                viewport.size()
            );
        }
    }

    #[test]
    fn fit_modes_fill_their_axis() {
        let total = (2000.0_f64, 1000.0);
        let viewport = full_window((500, 500));
        let width = fit_as(FitMode::Width, total, viewport);
        assert_eq!(width.scale, 0.25);
        assert!(
            (width.offset.1 - 125.0).abs() < 1e-6,
            "the fitting height centers"
        );
        let height = fit_as(FitMode::Height, total, viewport);
        assert_eq!(height.scale, 0.5);
        assert!(
            (height.offset.0 + 250.0).abs() < 1e-6,
            "the oversized width centers"
        );
        let actual = fit_as(FitMode::Actual, total, viewport);
        assert_eq!(actual.scale, 1.0);
        assert!((actual.offset.0 + 750.0).abs() < 1e-6);
        assert!((actual.offset.1 + 250.0).abs() < 1e-6);
        for &mode in FitMode::VARIANTS {
            assert_eq!(
                fit_as(mode, (0.0, 100.0), viewport),
                Placement {
                    scale: 1.0,
                    offset: (0.0, 0.0)
                },
                "{mode:?}"
            );
        }
    }

    #[expect(
        clippy::float_cmp_const,
        reason = "clamping to the bound yields the bound exactly"
    )]
    #[test]
    fn zoom_at_anchors_and_clamps() {
        for (scale, factor) in [
            (2.0_f64, 2.0), // plain zoom
            (1.0, 0.5),     // plain shrink
            (30.0, 100.0),  // clamps at MAX_SCALE
            (0.1, 0.001),   // clamps at MIN_SCALE
        ] {
            for cursor in
                [(300.0_f64, 200.0), (0.0, 0.0), (1234.0, 4321.0)]
            {
                let before = Placement {
                    scale,
                    offset: (100.0, 50.0),
                };
                let after = zoom_at(before, cursor, factor);
                let under_cursor = |placement: Placement| {
                    (
                        (cursor.0 - placement.offset.0) / placement.scale,
                        (cursor.1 - placement.offset.1) / placement.scale,
                    )
                };
                let (want_x, want_y) = under_cursor(before);
                let (got_x, got_y) = under_cursor(after);
                // The anchor distance is measured in image pixels, whose
                // magnitude dwarfs the px tolerances elsewhere in the suite.
                assert!(
                    (got_x - want_x).abs() < 0.01,
                    "scale {scale} factor {factor} cursor {cursor:?}"
                );
                assert!(
                    (got_y - want_y).abs() < 0.01,
                    "scale {scale} factor {factor} cursor {cursor:?}"
                );
            }
        }
        assert_eq!(
            zoom_at(
                Placement {
                    scale: 30.0,
                    offset: (0.0, 0.0)
                },
                (0.0, 0.0),
                100.0
            )
            .scale,
            MAX_SCALE
        );
        assert_eq!(
            zoom_at(
                Placement {
                    scale: 0.1,
                    offset: (0.0, 0.0)
                },
                (0.0, 0.0),
                0.001
            )
            .scale,
            MIN_SCALE
        );
    }

    #[test]
    fn clamp_keeps_spread_reachable() {
        let viewport = full_window((500, 500));
        // 2000 content px at scale 1 in a 500 px viewport, dragged off both ends.
        let wide = clamp(
            Placement {
                scale: 1.0,
                offset: (-5000.0, 3000.0),
            },
            (2000.0, 400.0),
            viewport,
        );
        // x is oversized: some of the spread stays reachable from both edges
        assert!(
            wide.offset.0 >= 500.0 - 2000.0 - EDGE_KEEP_PX - 1e-4,
            "{}",
            wide.offset.0
        );
        assert!(wide.offset.0 <= EDGE_KEEP_PX + 1e-4);
        // y fits (400 < 500) so it is pinned centered regardless of the drag
        assert!((wide.offset.1 - 50.0).abs() < 1e-4);
        assert_eq!(wide.scale, 1.0, "clamping moves the offset only");

        let tall = clamp(
            Placement {
                scale: 1.0,
                offset: (3000.0, -5000.0),
            },
            (400.0, 2000.0),
            viewport,
        );
        assert!((tall.offset.0 - 50.0).abs() < 1e-4);
        assert!(
            tall.offset.1 >= 500.0 - 2000.0 - EDGE_KEEP_PX - 1e-4,
            "{}",
            tall.offset.1
        );
        assert!(tall.offset.1 <= EDGE_KEEP_PX + 1e-4);
        assert_eq!(tall.scale, 1.0);

        // A fully-fitted spread dragged away returns to its centered placement.
        let total = (400.0, 300.0);
        let fitted = fit_as(FitMode::Page, total, viewport);
        let dragged = pan(fitted, (300.0, -200.0));
        let recentered = clamp(dragged, total, viewport);
        assert!((recentered.offset.0 - fitted.offset.0).abs() < 1e-4);
        assert!((recentered.offset.1 - fitted.offset.1).abs() < 1e-4);
    }

    #[test]
    fn fit_lifecycle() {
        let viewport = full_window((500, 500));
        let total = (100.0, 200.0);
        let mut fit = Fit::default();

        let fitted = Placement {
            scale: 2.5,
            offset: (125.0, 0.0),
        };
        assert_eq!(
            fit.resolve(total, viewport),
            fitted,
            "a fresh fit places by the mode"
        );
        assert_eq!(
            fit.resolve(total, viewport),
            fitted,
            "the settled total keeps the placement"
        );

        fit.zoom(2.0, (10.0, 10.0), total, viewport);
        let zoomed = fit.placement.scale;
        assert!(zoomed > fitted.scale, "the zoom zooms in");
        fit.resolve((120.0, 200.0), viewport);
        assert_eq!(
            fit.placement.scale, zoomed,
            "a user zoom survives a later settle"
        );

        let wide = (2000.0, 1000.0);
        let mut zoomed_view = Fit::default();
        let wide_fitted = zoomed_view.resolve(wide, viewport);
        zoomed_view.zoom(4.0, (250.0, 250.0), wide, viewport);
        let before = zoomed_view.placement;
        zoomed_view.pan((-100.0, 0.0), wide, viewport);
        assert_eq!(
            zoomed_view.placement.scale, before.scale,
            "pan keeps the zoom"
        );
        assert!(
            zoomed_view.placement.offset.0 < before.offset.0,
            "pan moves the oversized spread"
        );
        assert!(wide_fitted.scale < before.scale, "the zoom zoomed in");

        let mut select_view = Fit::default();
        select_view.resolve(total, viewport);
        select_view.zoom(2.0, (10.0, 10.0), total, viewport);
        select_view.select(FitMode::Width);
        assert_eq!(
            select_view.resolve((250.0, 200.0), viewport),
            Placement {
                scale: 2.0,
                offset: (0.0, 50.0),
            },
            "select refits by the mode"
        );

        let mut turn_view = Fit::default();
        turn_view.resolve(total, viewport);
        turn_view.zoom(2.0, (0.0, 0.0), total, viewport);
        let held = turn_view.placement;
        turn_view.begin_turn();
        assert_eq!(
            turn_view.resolve(total, viewport),
            held,
            "begin_turn keeps the held frame frozen"
        );

        turn_view.zoom(2.0, (0.0, 0.0), total, viewport);
        let hold_zoom = turn_view.placement.scale;
        turn_view.commit();
        turn_view.resolve((200.0, 400.0), viewport);
        assert_eq!(
            turn_view.placement.scale, hold_zoom,
            "commit keeps hold-time moves"
        );

        turn_view.refit();
        assert_eq!(
            turn_view.resolve((200.0, 400.0), viewport),
            Placement {
                scale: 1.25,
                offset: (125.0, 0.0),
            },
            "refit discards the pin"
        );
    }
}
