//! Spread pairing, viewport geometry, fit modes, zoom, pan, and quad layout.

use std::num::NonZeroUsize;

use strum::Display;
use strum::VariantArray;
use winit::keyboard::KeyCode;

use crate::book::catalog::Direction;
use crate::render::textures::DrawQuad;

#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Display,
    VariantArray
)]
pub enum ReadingMode {
    #[default]
    Normal,
    Manga,
    Comic,
}

impl ReadingMode {
    #[must_use]
    pub const fn pairs(self) -> bool {
        matches!(self, Self::Manga | Self::Comic)
    }

    #[must_use]
    pub const fn rtl(self) -> bool {
        matches!(self, Self::Manga)
    }

    #[must_use]
    pub const fn key_code(self) -> KeyCode {
        match self {
            Self::Normal => KeyCode::KeyN,
            Self::Manga => KeyCode::KeyM,
            Self::Comic => KeyCode::KeyC,
        }
    }

    #[must_use]
    pub const fn shortcut(self) -> &'static str {
        match self {
            Self::Normal => "n",
            Self::Manga => "m",
            Self::Comic => "c",
        }
    }

    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Normal => "left to right, single page",
            Self::Manga => "right to left, double page",
            Self::Comic => "left to right, double page",
        }
    }
}

/// One screen's pages, 0-based: the cover alone, then pairs (`1+2`,
/// `3+4`, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spread {
    pub first: usize,
    pub second: Option<usize>,
}

impl Spread {
    pub const fn single(first: usize) -> Self {
        Self {
            first,
            second: None,
        }
    }

    /// Seat order as `[left, right]` under `mode`.
    pub const fn seats(self, mode: ReadingMode) -> [Option<usize>; 2] {
        if mode.rtl() {
            [self.second, Some(self.first)]
        } else {
            [Some(self.first), self.second]
        }
    }

    pub fn pages(self) -> impl Iterator<Item = usize> {
        [Some(self.first), self.second].into_iter().flatten()
    }

    pub const fn contains(self, page: usize) -> bool {
        self.first == page
            || matches!(self.second, Some(second) if second == page)
    }
}

/// Spread containing `page` under `mode`, clamped to the book. An
/// odd tail shows one page (`second: None`).
pub fn spread_for(
    page: usize,
    mode: ReadingMode,
    page_count: NonZeroUsize,
) -> Spread {
    let total_pages = page_count.get();
    let clamped_page = page.min(total_pages - 1);
    if !mode.pairs() {
        return Spread::single(clamped_page);
    }
    if clamped_page == 0 {
        return Spread::single(0);
    }
    let first = spread_start(spread_index(clamped_page, mode));
    Spread {
        first,
        second: (first + 1 < total_pages).then_some(first + 1),
    }
}

/// Next/previous spread start from `page`, wrapping at the ends.
/// Always returns a spread *start*.
pub fn advance(
    page: usize,
    direction: Direction,
    mode: ReadingMode,
    page_count: NonZeroUsize,
) -> usize {
    step(page, direction, mode, page_count).unwrap_or_else(|| {
        match direction {
            Direction::Forward => 0,
            Direction::Backward => last_spread(mode, page_count),
        }
    })
}

/// Next/previous spread start from `page`, or `None` at the book ends.
pub fn step(
    page: usize,
    direction: Direction,
    mode: ReadingMode,
    page_count: NonZeroUsize,
) -> Option<usize> {
    let total_pages = page_count.get();
    let clamped_page = page.min(total_pages - 1);
    if !mode.pairs() {
        return match direction {
            Direction::Forward => (clamped_page + 1 < total_pages)
                .then_some(clamped_page + 1),
            Direction::Backward => clamped_page.checked_sub(1),
        };
    }
    let current_spread = spread_index(clamped_page, mode);
    let next_spread = match direction {
        Direction::Forward => {
            let candidate = current_spread + 1;
            (candidate < spread_count(total_pages)).then_some(candidate)?
        }
        Direction::Backward => current_spread.checked_sub(1)?,
    };
    Some(spread_start(next_spread))
}

const fn spread_index(page: usize, mode: ReadingMode) -> usize {
    if mode.pairs() { page.div_ceil(2) } else { page }
}

const fn spread_start(spread_index: usize) -> usize {
    if spread_index > 0 {
        2 * spread_index - 1
    } else {
        0
    }
}

const fn spread_count(page_count: usize) -> usize {
    1 + (page_count - 1).div_ceil(2)
}

pub fn last_spread(mode: ReadingMode, page_count: NonZeroUsize) -> usize {
    if !mode.pairs() {
        return page_count.get() - 1;
    }
    spread_start(spread_count(page_count.get()) - 1)
}

pub fn spread_distance(
    from_page: usize,
    to_page: usize,
    mode: ReadingMode,
) -> usize {
    spread_index(from_page, mode).abs_diff(spread_index(to_page, mode))
}

/// Where pages live: the window area left after egui chrome, in physical
/// px, anchored at the window's top-left.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Viewport {
    size: (u32, u32),
}

impl Viewport {
    /// A viewport filling the window above a bottom chrome strip whose top
    /// edge sits `bar_top_px` physical px from the window top.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the clamped height is a non-negative integer within window_size.1 (u32)"
    )]
    pub fn above_bottom_bar(
        window_size: (u32, u32),
        bar_top_px: f64,
    ) -> Self {
        let height = if bar_top_px.is_nan() {
            0
        } else {
            // Nearest whole px matches egui's painter clip-rect rounding.
            bar_top_px.round().clamp(0.0, f64::from(window_size.1)) as u32
        };
        Self {
            size: (window_size.0, height),
        }
    }

    #[must_use]
    pub const fn size(self) -> (u32, u32) {
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

impl FitMode {
    #[must_use]
    pub const fn key_code(self) -> KeyCode {
        match self {
            Self::Page => KeyCode::KeyR,
            Self::Width => KeyCode::KeyW,
            Self::Height => KeyCode::KeyH,
            Self::Actual => KeyCode::Digit1,
        }
    }
}

/// A page's rect inside the composed spread, in image-pixel space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageRect {
    pub origin: (f64, f64),
    pub size: (f64, f64),
    pub dims: (u32, u32),
}

/// Layout of a spread's visual seats. `rects` is always [left, right].
#[derive(Debug, Clone, Copy, Default)]
pub struct SpreadGeometry {
    pub rects: [Option<PageRect>; 2],
}

impl SpreadGeometry {
    /// Bounding size of the composed spread, in image pixels.
    pub fn total(&self) -> (f64, f64) {
        self.rects.iter().flatten().fold(
            (0.0, 0.0),
            |spread_size, page_rect| {
                (
                    spread_size.0 + page_rect.size.0,
                    spread_size.1.max(page_rect.size.1),
                )
            },
        )
    }

    /// Screen quads for `seat_pages` under `placement` with whole-pixel
    /// origins so 1:1 nearest sampling stays even.
    pub fn draw_quads(
        &self,
        seat_pages: [Option<usize>; 2],
        placement: &Placement,
    ) -> Vec<DrawQuad> {
        self.rects
            .iter()
            .zip(seat_pages)
            .filter_map(|(page_rect, page)| {
                let page_rect = (*page_rect)?;
                let page = page?;
                Some(DrawQuad {
                    page,
                    origin: (
                        page_rect
                            .origin
                            .0
                            .mul_add(placement.scale, placement.offset.0)
                            .round(),
                        page_rect
                            .origin
                            .1
                            .mul_add(placement.scale, placement.offset.1)
                            .round(),
                    ),
                    size: (
                        page_rect.size.0 * placement.scale,
                        page_rect.size.1 * placement.scale,
                    ),
                    page_dims: page_rect.dims,
                })
            })
            .collect()
    }
}

/// `seat_dims[seat]` is the page size in visual seat `seat`: 0 = left, 1 = right.
pub fn layout(seat_dims: [Option<(u32, u32)>; 2]) -> SpreadGeometry {
    let mut next_seat_x = 0.0;
    SpreadGeometry {
        rects: seat_dims.map(|dims| {
            dims.map(|(width, height)| {
                let size = (f64::from(width), f64::from(height));
                let page_rect = PageRect {
                    origin: (next_seat_x, 0.0),
                    size,
                    dims: (width, height),
                };
                next_seat_x += size.0;
                page_rect
            })
        }),
    }
}

fn fit_as(
    mode: FitMode,
    spread_size: (f64, f64),
    viewport: Viewport,
) -> Placement {
    let viewport_size = viewport.size_f64();
    if spread_size.0 <= 0.0
        || spread_size.1 <= 0.0
        || viewport_size.0 <= 0.0
        || viewport_size.1 <= 0.0
    {
        return Placement {
            scale: 1.0,
            offset: (0.0, 0.0),
        };
    }
    let scale = match mode {
        FitMode::Page => (viewport_size.0 / spread_size.0)
            .min(viewport_size.1 / spread_size.1),
        FitMode::Width => viewport_size.0 / spread_size.0,
        FitMode::Height => viewport_size.1 / spread_size.1,
        FitMode::Actual => 1.0,
    };
    Placement {
        scale,
        offset: (
            spread_size.0.mul_add(-scale, viewport_size.0) / 2.0,
            spread_size.1.mul_add(-scale, viewport_size.1) / 2.0,
        ),
    }
}

const MIN_ZOOM_SCALE: f64 = 0.05;
const MAX_ZOOM_SCALE: f64 = 40.0;

fn zoom_at(
    mut placement: Placement,
    cursor: (f64, f64),
    factor: f64,
) -> Placement {
    let previous_scale = placement.scale;
    placement.scale =
        (previous_scale * factor).clamp(MIN_ZOOM_SCALE, MAX_ZOOM_SCALE);
    let applied_factor = placement.scale / previous_scale;
    placement.offset = (
        (cursor.0 - placement.offset.0).mul_add(-applied_factor, cursor.0),
        (cursor.1 - placement.offset.1).mul_add(-applied_factor, cursor.1),
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

const MIN_VISIBLE_EDGE_PX: f64 = 64.0;

fn clamp(
    placement: Placement,
    spread_size: (f64, f64),
    viewport: Viewport,
) -> Placement {
    let viewport_size = viewport.size_f64();
    let scaled_spread = (
        spread_size.0 * placement.scale,
        spread_size.1 * placement.scale,
    );
    Placement {
        scale: placement.scale,
        offset: (
            clamp_axis(
                placement.offset.0,
                scaled_spread.0,
                viewport_size.0,
            ),
            clamp_axis(
                placement.offset.1,
                scaled_spread.1,
                viewport_size.1,
            ),
        ),
    }
}

fn clamp_axis(offset: f64, content_span: f64, viewport_span: f64) -> f64 {
    if content_span <= viewport_span {
        (viewport_span - content_span) / 2.0
    } else {
        offset.clamp(
            viewport_span - content_span - MIN_VISIBLE_EDGE_PX,
            MIN_VISIBLE_EDGE_PX,
        )
    }
}

/// The current spread's fit: mode, live placement, and whether a commit
/// refits (a user move during the hold pins it).
#[derive(Debug, Clone, Copy, Default)]
pub struct Fit {
    mode: FitMode,
    placement: Placement,
    resolved_for: Option<((f64, f64), Viewport)>,
    is_user_pinned: bool,
}

impl Fit {
    #[must_use]
    pub const fn new(mode: FitMode) -> Self {
        Self {
            mode,
            placement: Placement {
                scale: 1.0,
                offset: (0.0, 0.0),
            },
            resolved_for: None,
            is_user_pinned: false,
        }
    }

    #[must_use]
    pub const fn mode(&self) -> FitMode {
        self.mode
    }

    pub fn resolve(
        &mut self,
        spread_size: (f64, f64),
        viewport: Viewport,
    ) -> Placement {
        if self.resolved_for == Some((spread_size, viewport)) {
            return self.placement;
        }
        self.placement = if self.is_user_pinned {
            clamp(self.placement, spread_size, viewport)
        } else {
            fit_as(self.mode, spread_size, viewport)
        };
        self.resolved_for = Some((spread_size, viewport));
        self.placement
    }

    pub fn zoom(
        &mut self,
        factor: f64,
        cursor: (f64, f64),
        spread_size: (f64, f64),
        viewport: Viewport,
    ) {
        self.placement = clamp(
            zoom_at(self.placement, cursor, factor),
            spread_size,
            viewport,
        );
        self.is_user_pinned = true;
    }

    pub fn pan(
        &mut self,
        delta: (f64, f64),
        spread_size: (f64, f64),
        viewport: Viewport,
    ) {
        self.placement =
            clamp(pan(self.placement, delta), spread_size, viewport);
        self.is_user_pinned = true;
    }

    pub fn select(&mut self, fit_mode: FitMode) {
        self.mode = fit_mode;
        self.refit();
    }

    pub fn begin_turn(&mut self) {
        self.is_user_pinned = false;
    }

    pub fn commit(&mut self) {
        self.resolved_for = None;
    }

    pub fn refit(&mut self) {
        self.resolved_for = None;
        self.is_user_pinned = false;
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;

    fn pages(count: usize) -> NonZeroUsize {
        NonZeroUsize::new(count).expect("test book is non-empty")
    }

    #[test]
    fn spread_structure() {
        assert_eq!(
            spread_for(1, ReadingMode::Manga, pages(10)),
            Spread {
                first: 1,
                second: Some(2)
            }
        );
        for &mode in ReadingMode::VARIANTS {
            for page_count in 1_usize..=10 {
                for page in 0..=(page_count + 2) {
                    let clamped = page.min(page_count - 1);
                    let spread = spread_for(page, mode, pages(page_count));
                    assert!(
                        spread.contains(clamped),
                        "mode {mode:?} page_count {page_count} page {page}"
                    );
                    assert_eq!(
                        spread_distance(clamped, spread.first, mode),
                        0,
                        "mode {mode:?} page_count {page_count} page {page}"
                    );
                    if mode.pairs() {
                        assert!(
                            spread.first == 0 || spread.first % 2 == 1,
                            "pairs start at 0 or an odd page: mode {mode:?} page_count {page_count} page {page}"
                        );
                        assert_eq!(
                            spread.second.is_none(),
                            spread.first == 0
                                || spread.first + 1 >= page_count,
                            "the cover is alone and an odd tail ends alone: mode {mode:?} page_count {page_count} page {page}"
                        );
                    } else {
                        assert_eq!(
                            spread.first, clamped,
                            "mode {mode:?} page_count {page_count} page {page}"
                        );
                        assert_eq!(
                            spread.second, None,
                            "mode {mode:?} page_count {page_count} page {page}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn advance_steps_between_spread_starts() {
        for page_count in [1_usize, 2, 3, 6, 9, 10, 11, 12] {
            for &mode in ReadingMode::VARIANTS {
                for page in 0..=page_count {
                    for &direction in Direction::VARIANTS {
                        let start =
                            spread_for(page, mode, pages(page_count))
                                .first;
                        let stepped = advance(
                            page,
                            direction,
                            mode,
                            pages(page_count),
                        );
                        assert_eq!(
                            spread_for(stepped, mode, pages(page_count))
                                .first,
                            stepped,
                            "stepped page must be a spread start: page_count {page_count} mode {mode:?} page {page} direction {direction:?}"
                        );
                        let back = direction.opposite();
                        assert_eq!(
                            advance(
                                stepped,
                                back,
                                mode,
                                pages(page_count)
                            ),
                            start,
                            "step then step back must round-trip: page_count {page_count} mode {mode:?} page {page} direction {direction:?}"
                        );
                        let at_end = match direction {
                            Direction::Forward => {
                                start
                                    == last_spread(mode, pages(page_count))
                            }
                            Direction::Backward => start == 0,
                        };
                        let bounded =
                            step(page, direction, mode, pages(page_count));
                        assert_eq!(
                            bounded,
                            (!at_end).then_some(stepped),
                            "bounded step must match advance except at book ends: page_count {page_count} mode {mode:?} page {page} direction {direction:?}"
                        );
                        if let Some(next) = bounded {
                            assert_eq!(
                                spread_distance(start, next, mode),
                                1,
                                "bounded step moves one spread: page_count {page_count} mode {mode:?} page {page} direction {direction:?}"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(
            advance(1, Direction::Forward, ReadingMode::Manga, pages(10)),
            3
        );
        assert_eq!(
            advance(4, Direction::Forward, ReadingMode::Normal, pages(10)),
            5
        );
        assert_eq!(
            advance(0, Direction::Forward, ReadingMode::Manga, pages(10)),
            1
        );
    }

    #[test]
    fn above_bottom_bar_sizes_and_clamps() {
        let viewport = Viewport::above_bottom_bar((1000, 800), 740.0);
        assert_eq!(viewport.size(), (1000, 740));
        let clamped_bar_top = Viewport::above_bottom_bar((100, 50), 80.0);
        assert_eq!(clamped_bar_top.size(), (100, 50));
        let negative = Viewport::above_bottom_bar((100, 50), -10.0);
        assert_eq!(negative.size(), (100, 0));
        let nan = Viewport::above_bottom_bar((100, 50), f64::NAN);
        assert_eq!(nan.size(), (100, 0));
    }

    #[test]
    fn layout_places_seats() {
        let left_dims = (100_u32, 300);
        let right_dims = (200, 150);
        let make_page_rect = |origin: (f64, f64), dims: (u32, u32)| {
            Some(PageRect {
                origin,
                size: (f64::from(dims.0), f64::from(dims.1)),
                dims,
            })
        };
        let cases = [
            ([None, None], [None, None], (0.0, 0.0)),
            (
                [Some(left_dims), None],
                [make_page_rect((0.0, 0.0), left_dims), None],
                (100.0, 300.0),
            ),
            (
                [None, Some(right_dims)],
                [None, make_page_rect((0.0, 0.0), right_dims)],
                (200.0, 150.0),
            ),
            (
                [Some(left_dims), Some(right_dims)],
                [
                    make_page_rect((0.0, 0.0), left_dims),
                    make_page_rect((100.0, 0.0), right_dims),
                ],
                (300.0, 300.0),
            ),
        ];
        for (seat_dims, seat_rects, spread_size) in cases {
            let spread = layout(seat_dims);
            assert_eq!(spread.rects, seat_rects, "seats {seat_dims:?}");
            assert_eq!(spread.total(), spread_size, "seats {seat_dims:?}");
        }

        let wide_strip = layout([Some((20_000_001, 1)), None]);
        assert_eq!(wide_strip.total(), (20_000_001.0, 1.0));
    }

    fn full_window(window_size: (u32, u32)) -> Viewport {
        Viewport::above_bottom_bar(window_size, f64::from(window_size.1))
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
            (
                (500.0, 400.0),
                Viewport::above_bottom_bar((1000, 800), 600.0),
                Placement {
                    scale: 1.5,
                    offset: (125.0, 0.0),
                },
            ),
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
        for (spread_size, viewport, expected) in cases {
            assert_eq!(
                fit_as(FitMode::Page, spread_size, viewport),
                expected,
                "spread_size {spread_size:?} size {:?}",
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
        assert!((width.offset.1 - 125.0).abs() < 1e-6);
        let height = fit_as(FitMode::Height, total, viewport);
        assert_eq!(height.scale, 0.5);
        assert!((height.offset.0 + 250.0).abs() < 1e-6);
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
        for (scale, factor) in
            [(2.0_f64, 2.0), (1.0, 0.5), (30.0, 100.0), (0.1, 0.001)]
        {
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
                let (expected_x, expected_y) = under_cursor(before);
                let (actual_x, actual_y) = under_cursor(after);
                assert!(
                    (actual_x - expected_x).abs() < 0.01,
                    "scale {scale} factor {factor} cursor {cursor:?}"
                );
                assert!(
                    (actual_y - expected_y).abs() < 0.01,
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
            MAX_ZOOM_SCALE
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
            MIN_ZOOM_SCALE
        );
    }

    #[test]
    fn clamp_keeps_spread_reachable() {
        let viewport = full_window((500, 500));
        let wide = clamp(
            Placement {
                scale: 1.0,
                offset: (-5000.0, 3000.0),
            },
            (2000.0, 400.0),
            viewport,
        );
        assert!(
            wide.offset.0 >= 500.0 - 2000.0 - MIN_VISIBLE_EDGE_PX - 1e-4
        );
        assert!(wide.offset.0 <= MIN_VISIBLE_EDGE_PX + 1e-4);
        assert!((wide.offset.1 - 50.0).abs() < 1e-4);
        assert_eq!(wide.scale, 1.0);

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
            tall.offset.1 >= 500.0 - 2000.0 - MIN_VISIBLE_EDGE_PX - 1e-4
        );
        assert!(tall.offset.1 <= MIN_VISIBLE_EDGE_PX + 1e-4);
        assert_eq!(tall.scale, 1.0);

        let spread_size = (400.0, 300.0);
        let fitted = fit_as(FitMode::Page, spread_size, viewport);
        let dragged = pan(fitted, (300.0, -200.0));
        let recentered = clamp(dragged, spread_size, viewport);
        assert!((recentered.offset.0 - fitted.offset.0).abs() < 1e-4);
        assert!((recentered.offset.1 - fitted.offset.1).abs() < 1e-4);
    }

    #[test]
    fn fit_lifecycle() {
        let viewport = full_window((500, 500));
        let spread_size = (100.0, 200.0);
        let mut fit = Fit::default();

        let fitted = Placement {
            scale: 2.5,
            offset: (125.0, 0.0),
        };
        assert_eq!(fit.resolve(spread_size, viewport), fitted);
        assert_eq!(fit.resolve(spread_size, viewport), fitted);

        fit.zoom(2.0, (10.0, 10.0), spread_size, viewport);
        let zoomed_scale = fit.placement.scale;
        assert!(zoomed_scale > fitted.scale);
        fit.resolve((120.0, 200.0), viewport);
        assert_eq!(fit.placement.scale, zoomed_scale);

        let wide_spread = (2000.0, 1000.0);
        let mut zoomed_view = Fit::default();
        let wide_fitted = zoomed_view.resolve(wide_spread, viewport);
        zoomed_view.zoom(4.0, (250.0, 250.0), wide_spread, viewport);
        let before = zoomed_view.placement;
        zoomed_view.pan((-100.0, 0.0), wide_spread, viewport);
        assert_eq!(zoomed_view.placement.scale, before.scale);
        assert!(zoomed_view.placement.offset.0 < before.offset.0);
        assert!(wide_fitted.scale < before.scale);

        let mut select_view = Fit::default();
        select_view.resolve(spread_size, viewport);
        select_view.zoom(2.0, (10.0, 10.0), spread_size, viewport);
        select_view.select(FitMode::Width);
        assert_eq!(
            select_view.resolve((250.0, 200.0), viewport),
            Placement {
                scale: 2.0,
                offset: (0.0, 50.0),
            }
        );

        let mut turn_view = Fit::default();
        turn_view.resolve(spread_size, viewport);
        turn_view.zoom(2.0, (0.0, 0.0), spread_size, viewport);
        let pinned_placement = turn_view.placement;
        turn_view.begin_turn();
        assert_eq!(
            turn_view.resolve(spread_size, viewport),
            pinned_placement
        );

        turn_view.zoom(2.0, (0.0, 0.0), spread_size, viewport);
        let pinned_scale = turn_view.placement.scale;
        turn_view.commit();
        turn_view.resolve((200.0, 400.0), viewport);
        assert_eq!(turn_view.placement.scale, pinned_scale);

        turn_view.refit();
        assert_eq!(
            turn_view.resolve((200.0, 400.0), viewport),
            Placement {
                scale: 1.25,
                offset: (125.0, 0.0),
            }
        );

        let mut initial = Fit::default();
        assert_eq!(
            initial.resolve(spread_size, Viewport::default()),
            Placement::default()
        );
        assert_eq!(initial.resolve(spread_size, viewport), fitted);
    }
}
