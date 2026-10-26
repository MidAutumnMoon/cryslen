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

    /// `spread_for`: mode flags, the lone cover, odd-start pairs, clamping at the source.
    #[test]
    fn spread_for_structure() {
        // (mode, pairs, rtl): the pairing and direction flags per mode
        for (mode, pairs, rtl) in [
            (ViewMode::Normal, false, false),
            (ViewMode::Manga, true, true),
            (ViewMode::Comic, true, false),
        ] {
            assert_eq!(mode.pairs(), pairs, "{mode:?}");
            assert_eq!(mode.rtl(), rtl, "{mode:?}");
        }
        // Anchor: page 1 opens the first pair.
        assert_eq!(spread_for(1, ViewMode::Manga, 10), Spread { first: 1, second: Some(2) });
        for &mode in &ViewMode::ALL {
            for len in 0_usize..=10 {
                for page in 0..=(len + 2) {
                    let clamped = page.min(len.max(1) - 1);
                    let spread = spread_for(page, mode, len);
                    assert!(spread.contains(clamped), "mode {mode:?} len {len} page {page}");
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
                        assert_eq!(spread.first, clamped, "mode {mode:?} len {len} page {page}");
                        assert_eq!(spread.second, None, "mode {mode:?} len {len} page {page}");
                    }
                }
            }
        }
    }

    /// advance steps between spread starts and wraps; a step and its inverse return to the spread start.
    #[test]
    fn advance_steps_between_spread_starts() {
        for len in [0_usize, 1, 2, 3, 6, 9, 10] {
            for &mode in &ViewMode::ALL {
                for page in 0..=len {
                    for &forward in &[false, true] {
                        let stepped = advance(page, forward, mode, len);
                        assert_eq!(
                            spread_for(stepped, mode, len).first,
                            stepped,
                            "stepped page must be a spread start: len {len} mode {mode:?} page {page} forward {forward}"
                        );
                        assert_eq!(
                            advance(stepped, !forward, mode, len),
                            spread_for(page, mode, len).first,
                            "step then step back must round-trip: len {len} mode {mode:?} page {page} forward {forward}"
                        );
                    }
                }
            }
        }
        // Direction anchors: a mid-spread step lands on the neighbor spread's start.
        assert_eq!(advance(1, true, ViewMode::Manga, 10), 3);
        assert_eq!(advance(4, true, ViewMode::Normal, 10), 5);
        // The cover's next spread starts at 1, not 2.
        assert_eq!(advance(0, true, ViewMode::Manga, 10), 1);
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

    /// layout composes present seats side by side, top-aligned; undecoded seats join when they arrive.
    #[test]
    fn layout_places_seats() {
        let left = (100_u32, 300);
        let right = (200, 150);
        let absent = None::<(u32, u32)>;
        // Literal anchor, so a swapped u32->f32 mapping cannot pass.
        assert_eq!(
            layout([Some(left), Some(right)]).rects,
            [
                Some(PageRect { pos: (0.0, 0.0), size: (100.0, 300.0) }),
                Some(PageRect { pos: (100.0, 0.0), size: (200.0, 150.0) }),
            ]
        );
        for dims in [
            [absent, absent],            // empty spread
            [Some(left), absent],        // left only
            [absent, Some(right)],       // right only: right starts at x=0
            [Some(left), Some(right)],   // asymmetric pair
        ] {
            let spread = layout(dims);
            // rect present iff input Some; x accumulates over present seats, top-aligned
            let mut next_seat_x = 0.0_f32;
            for (rect, seat_dims) in spread.rects.iter().zip(dims.iter()) {
                match seat_dims {
                    Some(seat_dims) => {
                        let (width, height) = px_f32(*seat_dims);
                        assert_eq!(
                            *rect,
                            Some(PageRect { pos: (next_seat_x, 0.0), size: (width, height) }),
                            "seats {dims:?}"
                        );
                        next_seat_x += width;
                    }
                    _ => assert_eq!(rect, &None, "seats {dims:?}"),
                }
            }
            // total is the width sum by the height max
            let widths = dims.iter().flatten().map(|&seat| px_f32(seat).0).sum::<f32>();
            let heights = dims.iter().flatten().map(|&seat| px_f32(seat).1).fold(0.0_f32, f32::max);
            assert!((spread.total().0 - widths).abs() < 1e-4, "seats {dims:?}");
            assert!((spread.total().1 - heights).abs() < 1e-4, "seats {dims:?}");
        }
    }

    /// The whole window as a viewport, for tests with no bottom bar.
    fn full_window(win: (f32, f32)) -> Viewport {
        Viewport::above_bottom_bar(win, win.1)
    }

    /// fit scales to the smaller axis ratio and centers each axis in the viewport; degenerate inputs are inert.
    #[test]
    fn fit_centers_and_scales() {
        for (total, viewport) in [
            ((1000.0_f32, 2000.0), full_window((500.0, 500.0))), // height is the min axis
            ((2000.0, 1000.0), full_window((500.0, 500.0))),     // width is the min axis
            // bar-clamped viewport: centering uses the 600-px viewport, not the 800-px window
            ((500.0, 400.0), Viewport::above_bottom_bar((1000.0, 800.0), 600.0)),
            ((0.0, 100.0), full_window((500.0, 500.0))),         // degenerate total is inert
            ((100.0, 100.0), Viewport::default()),               // empty viewport is inert
        ] {
            let size = viewport.size();
            let got = fit(total, viewport);
            if total.0 <= 0.0 || total.1 <= 0.0 || size.0 <= 0.0 || size.1 <= 0.0 {
                assert_eq!(got, Placement { scale: 1.0, offset: (0.0, 0.0) }, "total {total:?} size {size:?}");
                continue;
            }
            let want_scale = (size.0 / total.0).min(size.1 / total.1);
            assert!((got.scale - want_scale).abs() < 1e-6, "total {total:?} size {size:?}");
            let want_offset = (
                total.0.mul_add(-want_scale, size.0) / 2.0,
                total.1.mul_add(-want_scale, size.1) / 2.0,
            );
            assert!((got.offset.0 - want_offset.0).abs() < 1e-6, "total {total:?} size {size:?}");
            assert!((got.offset.1 - want_offset.1).abs() < 1e-6, "total {total:?} size {size:?}");
        }
    }

    /// `zoom_at` anchors the image point under the cursor with the factor actually applied, and clamps the scale.
    #[expect(
        clippy::float_cmp_const,
        reason = "clamping to the bound yields the bound exactly"
    )]
    #[test]
    fn zoom_at_anchors_and_clamps() {
        for (scale, factor) in [
            (2.0_f32, 2.0),  // plain zoom
            (1.0, 0.5),      // plain shrink
            (30.0, 100.0),   // clamps at MAX_SCALE
            (0.1, 0.001),    // clamps at MIN_SCALE
        ] {
            for cursor in [(300.0_f32, 200.0), (0.0, 0.0), (1234.0, 4321.0)] {
                let before = Placement { scale, offset: (100.0, 50.0) };
                let after = zoom_at(before, cursor, factor);
                let under_cursor = |placement: Placement| {
                    ((cursor.0 - placement.offset.0) / placement.scale,
                     (cursor.1 - placement.offset.1) / placement.scale)
                };
                let (want_x, want_y) = under_cursor(before);
                let (got_x, got_y) = under_cursor(after);
                assert!((got_x - want_x).abs() < 0.01, "scale {scale} factor {factor} cursor {cursor:?}");
                assert!((got_y - want_y).abs() < 0.01, "scale {scale} factor {factor} cursor {cursor:?}");
            }
        }
        // Both clamp bounds land exactly.
        assert_eq!(zoom_at(Placement { scale: 30.0, offset: (0.0, 0.0) }, (0.0, 0.0), 100.0).scale, MAX_SCALE);
        assert_eq!(zoom_at(Placement { scale: 0.1, offset: (0.0, 0.0) }, (0.0, 0.0), 0.001).scale, MIN_SCALE);
    }

    /// clamp pins fitting axes centered and keeps every oversized axis edge-reachable.
    #[test]
    fn clamp_keeps_spread_reachable() {
        let viewport = full_window((500.0, 500.0));
        // 2000 content px at scale 1 in a 500 px viewport, dragged off both ends.
        let wide = clamp(Placement { scale: 1.0, offset: (-5000.0, 3000.0) }, (2000.0, 400.0), viewport);
        // x is oversized: some of the spread stays reachable from both edges
        assert!(wide.offset.0 >= 500.0 - 2000.0 - EDGE_KEEP_PX - 1e-4, "{}", wide.offset.0);
        assert!(wide.offset.0 <= EDGE_KEEP_PX + 1e-4);
        // y fits (400 < 500) so it is pinned centered regardless of the drag
        assert!((wide.offset.1 - 50.0).abs() < 1e-4);
        assert_eq!(wide.scale, 1.0, "clamping moves the offset only");

        let tall = clamp(Placement { scale: 1.0, offset: (3000.0, -5000.0) }, (400.0, 2000.0), viewport);
        assert!((tall.offset.0 - 50.0).abs() < 1e-4);
        assert!(tall.offset.1 >= 500.0 - 2000.0 - EDGE_KEEP_PX - 1e-4, "{}", tall.offset.1);
        assert!(tall.offset.1 <= EDGE_KEEP_PX + 1e-4);
        assert_eq!(tall.scale, 1.0);

        // A fully-fitted spread dragged away returns to its centered placement.
        let total = (400.0, 300.0);
        let fitted = fit(total, viewport);
        let dragged = pan(fitted, (300.0, -200.0));
        let recentered = clamp(dragged, total, viewport);
        assert!((recentered.offset.0 - fitted.offset.0).abs() < 1e-4);
        assert!((recentered.offset.1 - fitted.offset.1).abs() < 1e-4);
    }
}
