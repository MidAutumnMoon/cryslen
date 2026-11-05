//! View math: the view modes, the spread-pairing model, and the
//! placement of a composed spread in the viewport. Pure data, no GL.

use std::num::NonZeroUsize;

/// A book's page count. A live source lists at least one page, so the
/// count is nonzero by shape and the spread math never clamps it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageCount(NonZeroUsize);

impl PageCount {
    pub fn new(count: usize) -> Option<Self> {
        NonZeroUsize::new(count).map(Self)
    }

    pub fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    Normal,
    Manga,
    Comic,
}

impl ViewMode {
    pub const ALL: [Self; 3] = [Self::Normal, Self::Manga, Self::Comic];

    pub fn pairs(self) -> bool {
        matches!(self, Self::Manga | Self::Comic)
    }

    pub fn rtl(self) -> bool {
        self == Self::Manga
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Manga => "Manga",
            Self::Comic => "Comic",
        }
    }

    pub fn shortcut(self) -> &'static str {
        match self {
            Self::Normal => "n",
            Self::Manga => "m",
            Self::Comic => "c",
        }
    }

    pub fn effect(self) -> &'static str {
        match self {
            Self::Normal => "left to right, single page",
            Self::Manga => "right to left, double page",
            Self::Comic => "left to right, double page",
        }
    }
}

/// Which pages one screen shows. The cover is alone, then pairs,
/// 0-based: `0`, `1+2`, `3+4`, ...
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spread {
    pub first: usize,
    pub second: Option<usize>,
}

/// Spread containing `index` under `mode`, clamped to the book. An
/// odd tail shows one page (`second: None`).
pub fn spread_for(index: usize, mode: ViewMode, len: PageCount) -> Spread {
    let len = len.get();
    let index = index.min(len - 1);
    if !mode.pairs() {
        return Spread::lone(index);
    }
    if index == 0 {
        return Spread::lone(0);
    }
    let first = spread_start(spread_number(index));
    Spread {
        first,
        second: (first + 1 < len).then_some(first + 1),
    }
}

/// Next/previous spread start from `current`, wrapping at the ends.
/// Always returns a spread *start*.
pub fn advance(
    current: usize,
    forward: bool,
    mode: ViewMode,
    len: PageCount,
) -> usize {
    let len = len.get();
    let current = current.min(len - 1);
    if !mode.pairs() {
        let next = if forward {
            current + 1
        } else {
            current + len - 1
        };
        return next % len;
    }
    let spreads = spread_count(len);
    let number = spread_number(current);
    let next = if forward {
        (number + 1) % spreads
    } else {
        (number + spreads - 1) % spreads
    };
    spread_start(next)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModeSwitch {
    pub index: usize,
    pub pairing_changed: bool,
}

/// Where a switch from `old` to `new` lands. A pairing change moves
/// to the new spread's start. A direction-only change keeps the page.
pub fn switch_mode(
    old: ViewMode,
    new: ViewMode,
    index: usize,
    len: PageCount,
) -> ModeSwitch {
    if old.pairs() == new.pairs() {
        ModeSwitch {
            index,
            pairing_changed: false,
        }
    } else {
        ModeSwitch {
            index: spread_for(index, new, len).first,
            pairing_changed: true,
        }
    }
}

fn spread_number(page: usize) -> usize {
    page.div_ceil(2)
}

fn spread_start(number: usize) -> usize {
    if number > 0 {
        2 * number - 1
    } else {
        0
    }
}

fn spread_count(len: usize) -> usize {
    1 + (len - 1).div_ceil(2)
}

/// Seat order as [left, right] under `mode`. Western reads first-left.
/// Manga reads first-right, so a lone page sits on the right.
pub fn seating<T>(
    first: T,
    second: Option<T>,
    mode: ViewMode,
) -> [Option<T>; 2] {
    if mode.rtl() {
        [second, Some(first)]
    } else {
        [Some(first), second]
    }
}

impl Spread {
    pub const fn lone(first: usize) -> Self {
        Self {
            first,
            second: None,
        }
    }

    pub fn seats(&self, mode: ViewMode) -> [Option<usize>; 2] {
        seating(self.first, self.second, mode)
    }

    pub fn contains(&self, page: usize) -> bool {
        self.first == page || self.second == Some(page)
    }
}

/// Where pages live: the window area left after egui chrome, in physical
/// px, anchored at the window's top-left.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Viewport {
    size: (f32, f32),
}

impl Viewport {
    /// A viewport filling the window above a bottom chrome strip whose top
    /// edge sits `bar_top` physical px from the window top. The only sizing
    /// constructor, so every non-empty viewport is chrome-aware.
    pub fn above_bottom_bar(window: (f32, f32), bar_top: f32) -> Self {
        Self {
            size: (
                window.0.max(0.0),
                bar_top.clamp(0.0, window.1.max(0.0)),
            ),
        }
    }

    pub fn size(self) -> (f32, f32) {
        self.size
    }
}

/// Placement of the composed spread: scale is screen px per image px, offset
/// is the position of the spread's top-left corner in viewport px.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    pub scale: f32,
    pub offset: (f32, f32),
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
    pub pos: (f32, f32),
    pub size: (f32, f32),
}

/// Layout of a spread's visual seats. `rects` is always [left, right].
#[derive(Debug, Clone, Copy, Default)]
pub struct SpreadLayout {
    pub rects: [Option<PageRect>; 2],
}

impl SpreadLayout {
    /// Bounding size of the composed spread, in image pixels.
    pub fn total(&self) -> (f32, f32) {
        self.rects
            .iter()
            .flatten()
            .fold((0.0, 0.0), |total, page_rect| {
                (total.0 + page_rect.size.0, total.1.max(page_rect.size.1))
            })
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "page edges and window extents stay far below f32's exact-integer range 2^24"
)]
pub fn px_f32(dims: (u32, u32)) -> (f32, f32) {
    (dims.0 as f32, dims.1 as f32)
}

/// Compose one or two pages side by side, top-aligned. `dims[i]` is the
/// decoded size of visual seat i (0 = left, 1 = right). A still-decoding
/// seat is None and joins the layout when it arrives.
pub fn layout(dims: [Option<(u32, u32)>; 2]) -> SpreadLayout {
    let mut next_seat_x = 0.0;
    SpreadLayout {
        rects: dims.map(|seat_dims| {
            seat_dims.map(px_f32).map(|(width, height)| {
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
pub fn fit_as(
    mode: FitMode,
    total: (f32, f32),
    viewport: Viewport,
) -> Placement {
    let size = viewport.size();
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

pub fn first_spread() -> usize {
    0
}

pub fn last_spread(mode: ViewMode, len: PageCount) -> usize {
    if !mode.pairs() {
        return len.get() - 1;
    }
    spread_start(spread_count(len.get()) - 1)
}

/// Zoom scale bounds, in screen px per image px. Applied at zoom time
/// only; `fit_as` does not clamp.
const MIN_SCALE: f32 = 0.05;
const MAX_SCALE: f32 = 40.0;

/// Zoom by `factor`, keeping the image point under `cursor` (viewport px)
/// anchored. Clamps scale to [`MIN_SCALE`]..=[`MAX_SCALE`]. The anchor
/// uses the *actual* factor applied so hitting the clamp does not shift
/// the image.
pub fn zoom_at(
    mut placement: Placement,
    cursor: (f32, f32),
    factor: f32,
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

pub fn pan(placement: Placement, delta: (f32, f32)) -> Placement {
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
const EDGE_KEEP_PX: f32 = 64.0;

/// Clamp pan so the spread cannot be lost. An axis where the scaled
/// spread fits the viewport is pinned to the centered position.
pub fn clamp(
    placement: Placement,
    total: (f32, f32),
    viewport: Viewport,
) -> Placement {
    let size = viewport.size();
    let scaled = (total.0 * placement.scale, total.1 * placement.scale);
    Placement {
        scale: placement.scale,
        offset: (
            clamp_axis(placement.offset.0, scaled.0, size.0),
            clamp_axis(placement.offset.1, scaled.1, size.1),
        ),
    }
}

fn clamp_axis(offset: f32, content: f32, size: f32) -> f32 {
    if content <= size {
        (size - content) / 2.0
    } else {
        offset.clamp(size - content - EDGE_KEEP_PX, EDGE_KEEP_PX)
    }
}

/// The current spread's fit: the mode, the live placement, and whether
/// the placement is due for recomputation. A user zoom or pan keeps its
/// placement until a turn; a commit refits only if the user did not move
/// during the hold.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fit {
    mode: FitMode,
    placement: Placement,
    settled: Option<(f32, f32)>,
    user_moved: bool,
}

impl Fit {
    /// Fit `total` into `viewport` when due and return the placement. A
    /// user-placed view keeps its zoom and only clamps its pan into the
    /// new shape or viewport; otherwise the fit mode places the spread.
    pub fn resolve(
        &mut self,
        total: (f32, f32),
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
        factor: f32,
        cursor: (f32, f32),
        total: (f32, f32),
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
        delta: (f32, f32),
        total: (f32, f32),
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

    /// The screen committed a new spread: the next `resolve` places it.
    /// Moves made during the hold carry into it.
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
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;

    fn pages(count: usize) -> PageCount {
        PageCount::new(count).expect("test book is non-empty")
    }

    #[test]
    fn spread_structure() {
        for (mode, pairs, rtl) in [
            (ViewMode::Normal, false, false),
            (ViewMode::Manga, true, true),
            (ViewMode::Comic, true, false),
        ] {
            assert_eq!(mode.pairs(), pairs, "{mode:?}");
            assert_eq!(mode.rtl(), rtl, "{mode:?}");
        }
        // Anchor: page 1 opens the first pair.
        assert_eq!(
            spread_for(1, ViewMode::Manga, pages(10)),
            Spread {
                first: 1,
                second: Some(2)
            }
        );
        for &mode in &ViewMode::ALL {
            for len in 1_usize..=10 {
                for page in 0..=(len + 2) {
                    let clamped = page.min(len - 1);
                    let spread = spread_for(page, mode, pages(len));
                    assert!(
                        spread.contains(clamped),
                        "mode {mode:?} len {len} page {page}"
                    );
                    if mode.pairs() {
                        assert!(
                            spread.first == 0 || spread.first % 2 == 1,
                            "pairs start at 0 or an odd page: mode {mode:?} len {len} page {page}"
                        );
                        assert_eq!(
                            spread.second.is_none(),
                            spread.first == 0 || spread.first + 1 >= len,
                            "the cover is alone and an odd tail ends alone: mode {mode:?} len {len} page {page}"
                        );
                    } else {
                        assert_eq!(
                            spread.first, clamped,
                            "mode {mode:?} len {len} page {page}"
                        );
                        assert_eq!(
                            spread.second, None,
                            "mode {mode:?} len {len} page {page}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn advance_steps_between_spread_starts() {
        for len in [1_usize, 2, 3, 6, 9, 10] {
            for &mode in &ViewMode::ALL {
                for page in 0..=len {
                    for &forward in &[false, true] {
                        let stepped =
                            advance(page, forward, mode, pages(len));
                        assert_eq!(
                            spread_for(stepped, mode, pages(len)).first,
                            stepped,
                            "stepped page must be a spread start: len {len} mode {mode:?} page {page} forward {forward}"
                        );
                        assert_eq!(
                            advance(stepped, !forward, mode, pages(len)),
                            spread_for(page, mode, pages(len)).first,
                            "step then step back must round-trip: len {len} mode {mode:?} page {page} forward {forward}"
                        );
                    }
                }
            }
        }
        // Direction anchors: a mid-spread step lands on the neighbor spread's start.
        assert_eq!(advance(1, true, ViewMode::Manga, pages(10)), 3);
        assert_eq!(advance(4, true, ViewMode::Normal, pages(10)), 5);
        // The cover's next spread starts at 1, not 2.
        assert_eq!(advance(0, true, ViewMode::Manga, pages(10)), 1);
    }

    #[test]
    fn mode_switch_relands_only_on_pairing_change() {
        assert_eq!(
            switch_mode(ViewMode::Normal, ViewMode::Manga, 2, pages(10)),
            ModeSwitch {
                index: 1,
                pairing_changed: true
            }
        );
        assert_eq!(
            switch_mode(ViewMode::Normal, ViewMode::Comic, 0, pages(10)),
            ModeSwitch {
                index: 0,
                pairing_changed: true
            }
        );
        assert_eq!(
            switch_mode(ViewMode::Manga, ViewMode::Comic, 5, pages(10)),
            ModeSwitch {
                index: 5,
                pairing_changed: false
            }
        );
        assert_eq!(
            switch_mode(ViewMode::Manga, ViewMode::Manga, 5, pages(10)),
            ModeSwitch {
                index: 5,
                pairing_changed: false
            }
        );
    }

    #[test]
    fn seats_follow_reading_direction() {
        let spread = Spread {
            first: 1,
            second: Some(2),
        };
        assert_eq!(spread.seats(ViewMode::Normal), [Some(1), Some(2)]);
        assert_eq!(spread.seats(ViewMode::Comic), [Some(1), Some(2)]);
        assert_eq!(spread.seats(ViewMode::Manga), [Some(2), Some(1)]);
        let lone = Spread::lone(5);
        assert_eq!(lone.seats(ViewMode::Normal), [Some(5), None]);
        // A lone Comic page mirrors Normal, not the rtl modes.
        assert_eq!(lone.seats(ViewMode::Comic), [Some(5), None]);
        assert_eq!(lone.seats(ViewMode::Manga), [None, Some(5)]);
    }

    #[test]
    fn above_bottom_bar_sizes_and_clamps() {
        let viewport = Viewport::above_bottom_bar((1000.0, 800.0), 740.0);
        assert_eq!(viewport.size(), (1000.0, 740.0));
        // A bar top past the window bottom degenerates to the window.
        // A negative one degenerates to nothing.
        let past = Viewport::above_bottom_bar((100.0, 50.0), 80.0);
        assert_eq!(past.size(), (100.0, 50.0));
        let negative = Viewport::above_bottom_bar((100.0, 50.0), -10.0);
        assert_eq!(negative.size(), (100.0, 0.0));
    }

    #[test]
    fn layout_places_seats() {
        let left = (100_u32, 300);
        let right = (200, 150);
        let rect = |pos: (f32, f32), size: (f32, f32)| {
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
    }

    fn full_window(win: (f32, f32)) -> Viewport {
        Viewport::above_bottom_bar(win, win.1)
    }

    #[test]
    fn fit_page_centers_on_the_min_axis() {
        let cases = [
            (
                (1000.0_f32, 2000.0),
                full_window((500.0, 500.0)),
                Placement {
                    scale: 0.25,
                    offset: (125.0, 0.0),
                },
            ),
            (
                (2000.0, 1000.0),
                full_window((500.0, 500.0)),
                Placement {
                    scale: 0.25,
                    offset: (0.0, 125.0),
                },
            ),
            // Bar-clamped viewport: centering uses the 600-px viewport,
            // not the 800-px window.
            (
                (500.0, 400.0),
                Viewport::above_bottom_bar((1000.0, 800.0), 600.0),
                Placement {
                    scale: 1.5,
                    offset: (125.0, 0.0),
                },
            ),
            // Degenerate total and empty viewport are inert.
            (
                (0.0, 100.0),
                full_window((500.0, 500.0)),
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
        let total = (2000.0_f32, 1000.0);
        let viewport = full_window((500.0, 500.0));
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
        // Degenerate totals stay inert in every mode.
        for mode in [
            FitMode::Page,
            FitMode::Width,
            FitMode::Height,
            FitMode::Actual,
        ] {
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

    #[test]
    fn first_and_last_spreads() {
        // Single page: both ends are the cover.
        for mode in ViewMode::ALL {
            assert_eq!(last_spread(mode, pages(1)), 0);
        }
        // Pairs: the final spread starts on the last odd page.
        assert_eq!(last_spread(ViewMode::Manga, pages(10)), 9);
        assert_eq!(last_spread(ViewMode::Comic, pages(11)), 9);
        // An odd tail ends alone on the previous pair start.
        assert_eq!(last_spread(ViewMode::Manga, pages(12)), 11);
        // Normal: the last page.
        assert_eq!(last_spread(ViewMode::Normal, pages(10)), 9);
        for len in 1_usize..=12 {
            for &mode in &ViewMode::ALL {
                let last = last_spread(mode, pages(len));
                assert_eq!(
                    spread_for(last, mode, pages(len)).first,
                    last,
                    "len {len} {mode:?}"
                );
            }
        }
    }

    #[expect(
        clippy::float_cmp_const,
        reason = "clamping to the bound yields the bound exactly"
    )]
    #[test]
    fn zoom_at_anchors_and_clamps() {
        for (scale, factor) in [
            (2.0_f32, 2.0), // plain zoom
            (1.0, 0.5),     // plain shrink
            (30.0, 100.0),  // clamps at MAX_SCALE
            (0.1, 0.001),   // clamps at MIN_SCALE
        ] {
            for cursor in
                [(300.0_f32, 200.0), (0.0, 0.0), (1234.0, 4321.0)]
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
        let viewport = full_window((500.0, 500.0));
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
        let viewport = full_window((500.0, 500.0));
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
