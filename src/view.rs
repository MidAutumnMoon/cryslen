//! View math: the view modes, the spread-pairing model, and the
//! placement of a composed spread in the viewport. Pure data, no GL.

/// A way of laying out and reading pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    /// One page per screen, left to right.
    Normal,
    /// Spreads, right to left.
    Manga,
    /// Spreads, left to right.
    Comic,
}

impl ViewMode {
    /// The modes in status-bar display order.
    pub const ALL: [Self; 3] = [Self::Normal, Self::Manga, Self::Comic];

    pub fn pairs(self) -> bool {
        matches!(self, Self::Manga | Self::Comic)
    }

    pub fn rtl(self) -> bool {
        self == Self::Manga
    }

    /// Name shown in the status-bar group.
    pub fn label(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Manga => "Manga",
            Self::Comic => "Comic",
        }
    }

    /// The key that selects this mode.
    pub fn shortcut(self) -> &'static str {
        match self {
            Self::Normal => "n",
            Self::Manga => "m",
            Self::Comic => "c",
        }
    }

    /// One line stating the mode's effect, shown on switch.
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
pub fn spread_for(index: usize, mode: ViewMode, len: usize) -> Spread {
    let len = len.max(1);
    let index = index.min(len - 1);
    if !mode.pairs() {
        return Spread { first: index, second: None };
    }
    if index == 0 {
        return Spread { first: 0, second: None };
    }
    let first = spread_start(spread_number(index));
    Spread { first, second: (first + 1 < len).then_some(first + 1) }
}

/// Next/previous spread start from `current`, wrapping at the ends.
/// Always returns a spread *start*.
pub fn advance(current: usize, forward: bool, mode: ViewMode, len: usize) -> usize {
    let len = len.max(1);
    let current = current.min(len - 1);
    if !mode.pairs() {
        let next = if forward { current + 1 } else { current + len - 1 };
        return next % len;
    }
    // The double-mode cycle is the lone cover, then the pairs.
    let spreads = 1 + (len - 1).div_ceil(2);
    let number = spread_number(current);
    let next = if forward { (number + 1) % spreads } else { (number + spreads - 1) % spreads };
    spread_start(next)
}

/// The outcome of a mode switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModeSwitch {
    /// The spread start to land on.
    pub index: usize,
    /// Whether the pairing changed, demanding the page-turn treatment.
    pub pairing_changed: bool,
}

/// Where a switch from `old` to `new` lands. A pairing change moves
/// to the new spread's start. A direction-only change keeps the page.
pub fn switch_mode(old: ViewMode, new: ViewMode, index: usize, len: usize) -> ModeSwitch {
    if old.pairs() == new.pairs() {
        ModeSwitch { index, pairing_changed: false }
    } else {
        ModeSwitch { index: spread_for(index, new, len).first, pairing_changed: true }
    }
}

/// 0-based spread number of a page. The cover is 0, and pages 1 and 2
/// share 1.
fn spread_number(page: usize) -> usize {
    page.div_ceil(2)
}

/// First page of a double-mode spread number. Pairs start on odd pages.
fn spread_start(number: usize) -> usize {
    if number > 0 { 2 * number - 1 } else { 0 }
}

/// One of the two seats of an on-screen spread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seat {
    Left,
    Right,
}

impl Seat {
    /// The seats in on-screen order.
    pub const ALL: [Self; 2] = [Self::Left, Self::Right];
}

/// Seat order as [left, right] under `mode`. Western reads first-left.
/// Manga reads first-right, so a lone page sits on the right.
pub fn seating<T>(first: T, second: Option<T>, mode: ViewMode) -> [Option<T>; 2] {
    if mode.rtl() {
        [second, Some(first)]
    } else {
        [Some(first), second]
    }
}

impl Spread {
    /// Visual seating as [left, right] under `mode`.
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
    /// A viewport filling the window above a bottom chrome strip whose
    /// top edge sits `bar_top` physical px from the window top. The only
    /// sizing constructor, so every non-empty viewport is chrome-aware.
    /// A whole-window viewport is the visible, greppable
    /// `above_bottom_bar(win, win.1)`.
    pub fn above_bottom_bar(window: (f32, f32), bar_top: f32) -> Self {
        Self { size: (window.0.max(0.0), bar_top.clamp(0.0, window.1.max(0.0))) }
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
        let mut total_width: f32 = 0.0;
        let mut total_height: f32 = 0.0;
        for page_rect in self.rects.iter().flatten() {
            total_width += page_rect.size.0;
            total_height = total_height.max(page_rect.size.1);
        }
        (total_width, total_height)
    }
}

/// Pixel extents as floats: the view model's integer-to-float boundary.
#[expect(
    clippy::cast_precision_loss,
    reason = "page edges are capped at 16384 px and window extents are physical-device sized, both far below f32's exact-integer range 2^24"
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
        rects: dims.map(|seat_dims| seat_dims.map(px_f32).map(|(width, height)| {
            let rect = PageRect { pos: (next_seat_x, 0.0), size: (width, height) };
            next_seat_x += width;
            rect
        })),
    }
}

/// Placement that fits `total` into the viewport, centered.
pub fn fit(total: (f32, f32), viewport: Viewport) -> Placement {
    let size = viewport.size();
    if total.0 <= 0.0 || total.1 <= 0.0 || size.0 <= 0.0 || size.1 <= 0.0 {
        return Placement { scale: 1.0, offset: (0.0, 0.0) };
    }
    let scale = (size.0 / total.0).min(size.1 / total.1);
    Placement {
        scale,
        offset: (total.0.mul_add(-scale, size.0) / 2.0, total.1.mul_add(-scale, size.1) / 2.0),
    }
}

/// Zoom scale bounds on `scale`, which is screen px per image px.
const MIN_SCALE: f32 = 0.05;
const MAX_SCALE: f32 = 40.0;

/// Zoom by `factor`, keeping the image point under `cursor` (viewport px)
/// anchored. Clamps scale to [`MIN_SCALE`]..=[`MAX_SCALE`]. The anchor
/// uses the *actual* factor applied so hitting the clamp does not shift
/// the image.
pub fn zoom_at(mut placement: Placement, cursor: (f32, f32), factor: f32) -> Placement {
    let old = placement.scale;
    placement.scale = (old * factor).clamp(MIN_SCALE, MAX_SCALE);
    let applied = placement.scale / old;
    // keep the image pixel under the cursor fixed: off' = cursor - (cursor-off)*applied
    placement.offset = (
        (cursor.0 - placement.offset.0).mul_add(-applied, cursor.0),
        (cursor.1 - placement.offset.1).mul_add(-applied, cursor.1),
    );
    placement
}

/// Pan by `delta` viewport px.
pub fn pan(placement: Placement, delta: (f32, f32)) -> Placement {
    Placement {
        scale: placement.scale,
        offset: (placement.offset.0 + delta.0, placement.offset.1 + delta.1),
    }
}

/// Reachability margin for [`clamp`]: at most this many px of the spread
/// may hang off any viewport edge.
const EDGE_KEEP_PX: f32 = 64.0;

/// Clamp pan so the spread cannot be lost. An axis where the scaled
/// spread fits the viewport is pinned to the centered position.
pub fn clamp(placement: Placement, total: (f32, f32), viewport: Viewport) -> Placement {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_flags() {
        assert!(!ViewMode::Normal.pairs());
        assert!(!ViewMode::Normal.rtl());
        assert!(ViewMode::Manga.pairs());
        assert!(ViewMode::Manga.rtl());
        assert!(ViewMode::Comic.pairs());
        assert!(!ViewMode::Comic.rtl());
    }

    #[test]
    fn mode_table_strings() {
        assert_eq!(ViewMode::ALL, [ViewMode::Normal, ViewMode::Manga, ViewMode::Comic]);
        assert_eq!(ViewMode::Normal.effect(), "left to right, single page");
        assert_eq!(ViewMode::Manga.effect(), "right to left, double page");
        assert_eq!(ViewMode::Comic.effect(), "left to right, double page");
    }

    #[test]
    fn layout_places_seats_left_to_right() {
        let spread_layout = layout([Some((100, 200)), Some((100, 200))]);
        assert_eq!(spread_layout.rects[0], Some(PageRect { pos: (0.0, 0.0), size: (100.0, 200.0) }));
        assert_eq!(spread_layout.rects[1], Some(PageRect { pos: (100.0, 0.0), size: (100.0, 200.0) }));
        assert_eq!(spread_layout.total(), (200.0, 200.0));
    }

    #[test]
    fn layout_asymmetric_pair_top_aligned() {
        let spread_layout = layout([Some((100, 300)), Some((200, 150))]);
        assert_eq!(spread_layout.rects[0], Some(PageRect { pos: (0.0, 0.0), size: (100.0, 300.0) }));
        assert_eq!(spread_layout.rects[1], Some(PageRect { pos: (100.0, 0.0), size: (200.0, 150.0) }));
        assert_eq!(spread_layout.total(), (300.0, 300.0));
    }

    #[test]
    fn layout_skips_undecoded_seats() {
        let missing_right = layout([Some((100, 200)), None]);
        assert_eq!(missing_right.rects[1], None);
        assert_eq!(missing_right.total(), (100.0, 200.0));
        // an undecoded left seat leaves the right page starting at x=0
        let missing_left = layout([None, Some((100, 200))]);
        assert_eq!(missing_left.rects[0], None);
        assert_eq!(missing_left.rects[1], Some(PageRect { pos: (0.0, 0.0), size: (100.0, 200.0) }));
    }

    /// The whole window as a viewport, for tests with no bottom bar.
    fn full_window(win: (f32, f32)) -> Viewport {
        Viewport::above_bottom_bar(win, win.1)
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
    fn fit_scales_to_min_axis_and_centers() {
        let placement = fit((1000.0, 2000.0), full_window((500.0, 500.0)));
        assert!((placement.scale - 0.25).abs() < 1e-6);
        assert!((placement.offset.0 - 125.0).abs() < 1e-6);
        assert!((placement.offset.1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn fit_centers_within_the_viewport_not_the_window() {
        // Height-fitting image: the viewport is 600 px tall in an 800 px
        // window, so centering lands at 0, not at the window's 100.
        let viewport = Viewport::above_bottom_bar((1000.0, 800.0), 600.0);
        let placement = fit((500.0, 400.0), viewport);
        assert!((placement.scale - 1.5).abs() < 1e-6);
        assert!((placement.offset.1 - 0.0).abs() < 1e-6, "{}", placement.offset.1);
        assert!((placement.offset.0 - 125.0).abs() < 1e-6);
    }

    #[test]
    fn fit_degenerate_inputs_are_inert() {
        assert_eq!(
            fit((0.0, 0.0), full_window((500.0, 500.0))),
            Placement { scale: 1.0, offset: (0.0, 0.0) }
        );
        assert_eq!(
            fit((100.0, 100.0), Viewport::default()),
            Placement { scale: 1.0, offset: (0.0, 0.0) }
        );
    }

    #[test]
    fn zoom_keeps_cursor_anchored() {
        let placement = Placement { scale: 2.0, offset: (100.0, 50.0) };
        let placement = zoom_at(placement, (300.0, 200.0), 2.0);
        // The image pixel under the cursor before is (300-100)/2 = 100.
        // After the zoom it is (300 - offset)/scale.
        assert!(((300.0 - placement.offset.0) / placement.scale - 100.0).abs() < 0.01);
    }

    #[expect(
        clippy::float_cmp_const,
        reason = "clamping to the bound yields the bound exactly"
    )]
    #[test]
    fn zoom_clamps_scale() {
        assert_eq!(
            zoom_at(Placement { scale: 30.0, offset: (0.0, 0.0) }, (0.0, 0.0), 100.0).scale,
            MAX_SCALE
        );
        assert_eq!(
            zoom_at(Placement { scale: 0.1, offset: (0.0, 0.0) }, (0.0, 0.0), 0.001).scale,
            MIN_SCALE
        );
    }

    #[test]
    fn clamp_oversized_keeps_edge_reachable() {
        // 2000 content px at scale 1 in a 500 px viewport
        let viewport = full_window((500.0, 500.0));
        let placement = clamp(Placement { scale: 1.0, offset: (-5000.0, 0.0) }, (2000.0, 400.0), viewport);
        assert!(placement.offset.0 >= 500.0 - 2000.0 - EDGE_KEEP_PX - 1e-4, "{}", placement.offset.0);
        assert!(placement.offset.0 <= EDGE_KEEP_PX + 1e-4);
        // y fits (400 < 500) so it is centered
        assert!((placement.offset.1 - 50.0).abs() < 1e-4);
    }

    #[test]
    fn clamp_centers_fitted_axes() {
        let total = (400.0, 300.0);
        let viewport = full_window((1000.0, 800.0));
        let fitted = fit(total, viewport);
        let dragged = pan(fitted, (300.0, -200.0));
        let placement = clamp(dragged, total, viewport);
        assert!((placement.offset.0 - fitted.offset.0).abs() < 1e-4);
        assert!((placement.offset.1 - fitted.offset.1).abs() < 1e-4);
    }

    #[test]
    fn clamp_never_changes_scale() {
        let placement =
            clamp(Placement { scale: 7.0, offset: (1e9, -1e9) }, (100.0, 100.0), full_window((50.0, 50.0)));
        assert_eq!(placement.scale, 7.0);
    }

    #[test]
    fn cover_alone_then_pairs() {
        assert_eq!(spread_for(0, ViewMode::Manga, 10), Spread { first: 0, second: None });
        assert_eq!(spread_for(1, ViewMode::Manga, 10), Spread { first: 1, second: Some(2) });
        assert_eq!(spread_for(2, ViewMode::Manga, 10), Spread { first: 1, second: Some(2) });
        assert_eq!(spread_for(3, ViewMode::Manga, 10), Spread { first: 3, second: Some(4) });
        assert_eq!(spread_for(0, ViewMode::Normal, 10), Spread { first: 0, second: None });
        assert_eq!(spread_for(5, ViewMode::Normal, 10), Spread { first: 5, second: None });
    }

    #[test]
    fn odd_tail_clamps_at_source() {
        // 7 pages: index 6 lives in pair (5, 6).
        assert_eq!(spread_for(6, ViewMode::Manga, 7), Spread { first: 5, second: Some(6) });
        // 6 pages: the would-be pair (5, 6) shows page 5 alone.
        assert_eq!(spread_for(5, ViewMode::Manga, 6), Spread { first: 5, second: None });
        // Out-of-range index clamps to the last page.
        assert_eq!(spread_for(99, ViewMode::Manga, 6).first, 5);
    }

    #[test]
    fn advance_steps_and_wraps() {
        assert_eq!(advance(1, true, ViewMode::Manga, 10), 3);
        assert_eq!(advance(3, false, ViewMode::Manga, 10), 1);
        // 8 is the second page of the last spread (7, 8).
        assert_eq!(advance(8, true, ViewMode::Manga, 9), 0);
        assert_eq!(advance(1, false, ViewMode::Manga, 10), 0);
        assert_eq!(advance(4, true, ViewMode::Normal, 10), 5);
        // The cover's next spread starts at 1, not 2.
        assert_eq!(advance(0, true, ViewMode::Manga, 10), 1);
    }

    #[test]
    fn advance_wraps_at_both_ends() {
        // Single mode: last page to first and back.
        assert_eq!(advance(9, true, ViewMode::Normal, 10), 0);
        assert_eq!(advance(0, false, ViewMode::Normal, 10), 9);
        // Double mode with a lone tail page.
        assert_eq!(advance(9, true, ViewMode::Manga, 10), 0);
        assert_eq!(advance(0, false, ViewMode::Manga, 10), 9);
        // Double mode ending on a full pair.
        assert_eq!(advance(7, true, ViewMode::Manga, 9), 0);
        assert_eq!(advance(0, false, ViewMode::Manga, 9), 7);
        // Odd tail: 6 pages are 0 | 1,2 | 3,4 | 5.
        assert_eq!(advance(5, true, ViewMode::Manga, 6), 0);
        assert_eq!(advance(0, false, ViewMode::Manga, 6), 5);
    }

    #[test]
    fn advance_round_trips_through_spread_starts() {
        // A mid-spread page round-trips to its spread start, not to itself.
        for len in [1, 2, 3, 6, 9, 10] {
            for &mode in &ViewMode::ALL {
                for page in 0..len {
                    for &forward in &[false, true] {
                        let stepped = advance(page, forward, mode, len);
                        let back = advance(stepped, !forward, mode, len);
                        assert_eq!(back, spread_for(page, mode, len).first);
                    }
                }
            }
        }
    }

    #[test]
    fn advance_tiny_books() {
        assert_eq!(advance(0, true, ViewMode::Manga, 1), 0);
        assert_eq!(advance(0, false, ViewMode::Normal, 1), 0);
        // 2 pages, double mode: cover and one lone page cycle into each
        // other.
        assert_eq!(advance(0, true, ViewMode::Manga, 2), 1);
        assert_eq!(advance(1, true, ViewMode::Manga, 2), 0);
        // A zero-length book never exists, but the math must tolerate it.
        assert_eq!(advance(0, true, ViewMode::Manga, 0), 0);
    }

    #[test]
    fn mode_switch_reclamps_only_on_pairing_change() {
        assert_eq!(
            switch_mode(ViewMode::Normal, ViewMode::Manga, 2, 10),
            ModeSwitch { index: 1, pairing_changed: true }
        );
        // The cover stays the cover.
        assert_eq!(
            switch_mode(ViewMode::Normal, ViewMode::Comic, 0, 10),
            ModeSwitch { index: 0, pairing_changed: true }
        );
        assert_eq!(
            switch_mode(ViewMode::Manga, ViewMode::Comic, 5, 10),
            ModeSwitch { index: 5, pairing_changed: false }
        );
        // So does re-selecting the mode.
        assert_eq!(
            switch_mode(ViewMode::Manga, ViewMode::Manga, 5, 10),
            ModeSwitch { index: 5, pairing_changed: false }
        );
    }

    #[test]
    fn seats_follow_reading_direction() {
        let spread = Spread { first: 1, second: Some(2) };
        assert_eq!(spread.seats(ViewMode::Normal), [Some(1), Some(2)]);
        assert_eq!(spread.seats(ViewMode::Comic), [Some(1), Some(2)]);
        assert_eq!(spread.seats(ViewMode::Manga), [Some(2), Some(1)]);
        let lone = Spread { first: 5, second: None };
        assert_eq!(lone.seats(ViewMode::Normal), [Some(5), None]);
        assert_eq!(lone.seats(ViewMode::Manga), [None, Some(5)]);
    }
}
