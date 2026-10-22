//! View math: the view modes, the spread-pairing model (cover alone,
//! then pairs), and how a composed spread (1–2 pages) is placed in the
//! window. Pure data, no GL.

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
    pub const ALL: [ViewMode; 3] = [ViewMode::Normal, ViewMode::Manga, ViewMode::Comic];

    /// Whether pages pair into spreads.
    pub fn double(self) -> bool {
        matches!(self, ViewMode::Manga | ViewMode::Comic)
    }

    /// Whether reading runs right to left.
    pub fn rtl(self) -> bool {
        self == ViewMode::Manga
    }

    /// Name shown in the status-bar group.
    pub fn label(self) -> &'static str {
        match self {
            ViewMode::Normal => "Normal",
            ViewMode::Manga => "Manga",
            ViewMode::Comic => "Comic",
        }
    }

    /// The key that selects this mode, shown next to the name. Must
    /// match the binding in `crate::keys::from_key`.
    pub fn shortcut(self) -> &'static str {
        match self {
            ViewMode::Normal => "n",
            ViewMode::Manga => "m",
            ViewMode::Comic => "c",
        }
    }

    /// One line stating the mode's effect, shown on switch.
    pub fn effect(self) -> &'static str {
        match self {
            ViewMode::Normal => "left to right · single page",
            ViewMode::Manga => "right to left · double page",
            ViewMode::Comic => "left to right · double page",
        }
    }
}

/// Which pages one screen shows. The cover is alone, afterwards pairs,
/// 0-based: `0`, `1+2`, `3+4`, ...
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spread {
    pub first: usize,
    pub second: Option<usize>,
}

/// Spread containing `index` under `mode`, clamped to the book: an
/// odd tail shows one page (`second: None`), never a phantom page.
pub fn spread_for(index: usize, mode: ViewMode, len: usize) -> Spread {
    let len = len.max(1);
    let index = index.min(len - 1);
    if !mode.double() {
        return Spread { first: index, second: None };
    }
    if index == 0 {
        return Spread { first: 0, second: None };
    }
    let first = spread_start(spread_number(index));
    Spread { first, second: (first + 1 < len).then_some(first + 1) }
}

/// Next/previous spread start from `current`, wrapping at the ends.
/// Forward from the last spread lands on the cover, and backward from
/// the cover on the last spread. Always returns a spread *start*.
pub fn advance(current: usize, forward: bool, mode: ViewMode, len: usize) -> usize {
    let len = len.max(1);
    let current = current.min(len - 1);
    if !mode.double() {
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
    if old.double() == new.double() {
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

impl Spread {
    /// Visual seating as [left, right] under `mode`. Western reads
    /// first-left. Manga reads first-right, so a lone page lands in
    /// the right seat.
    pub fn seats(&self, mode: ViewMode) -> [Option<usize>; 2] {
        if mode.rtl() {
            [self.second, Some(self.first)]
        } else {
            [Some(self.first), self.second]
        }
    }

    /// Whether `page` is one of this spread's pages.
    pub fn contains(&self, page: usize) -> bool {
        self.first == page || self.second == Some(page)
    }
}

/// Placement of the composed spread: scale is screen px per image px, offset
/// is the screen position of the composed spread's top-left corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View {
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
/// Seating (which page goes where) was decided before layout by
/// [`Spread::seats`]. A seat with no decoded page yet is `None`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpreadLayout {
    pub rects: [Option<PageRect>; 2],
}

impl SpreadLayout {
    /// Bounding size of the composed spread, in image pixels.
    pub fn total(&self) -> (f32, f32) {
        let mut total_width: f32 = 0.0;
        let mut total_height: f32 = 0.0;
        for r in self.rects.iter().flatten() {
            total_width += r.size.0;
            total_height = total_height.max(r.size.1);
        }
        (total_width, total_height)
    }
}

/// Compose one or two pages side by side, top-aligned. `dims[i]` is the
/// decoded size of visual seat i (0 = left, 1 = right). A still-decoding
/// seat is None and joins the layout when it arrives.
pub fn layout(dims: [Option<(u32, u32)>; 2]) -> SpreadLayout {
    let mut rects: [Option<PageRect>; 2] = [None, None];
    let mut next_seat_x = 0.0;
    for (seat, dims) in dims.iter().enumerate() {
        if let Some((width, height)) = *dims {
            rects[seat] =
                Some(PageRect { pos: (next_seat_x, 0.0), size: (width as f32, height as f32) });
            next_seat_x += width as f32;
        }
    }
    SpreadLayout { rects }
}

/// View that fits `total` into `win`, centered.
pub fn fit(total: (f32, f32), win: (f32, f32)) -> View {
    if total.0 <= 0.0 || total.1 <= 0.0 || win.0 <= 0.0 || win.1 <= 0.0 {
        return View { scale: 1.0, offset: (0.0, 0.0) };
    }
    let scale = (win.0 / total.0).min(win.1 / total.1);
    View {
        scale,
        offset: ((win.0 - total.0 * scale) / 2.0, (win.1 - total.1 * scale) / 2.0),
    }
}

/// Zoom scale bounds: `scale` is screen px per image px, so these cap how
/// far in (40 px per image px) and out (1 image px covers 20 screen px) a
/// page can be zoomed.
const MIN_SCALE: f32 = 0.05;
const MAX_SCALE: f32 = 40.0;

/// Zoom by `factor`, keeping the image point under `cursor` (screen px)
/// anchored. Clamps scale to [`MIN_SCALE`]..=[`MAX_SCALE`]. The anchor uses
/// the *actual* factor applied so hitting the clamp does not shift the image.
pub fn zoom_at(mut view: View, cursor: (f32, f32), factor: f32) -> View {
    let old = view.scale;
    view.scale = (old * factor).clamp(MIN_SCALE, MAX_SCALE);
    let applied = view.scale / old;
    // keep the image pixel under the cursor fixed: off' = cursor - (cursor-off)*applied
    view.offset = (cursor.0 - (cursor.0 - view.offset.0) * applied,
                   cursor.1 - (cursor.1 - view.offset.1) * applied);
    view
}

/// Pan by `delta` screen px.
pub fn pan(view: View, delta: (f32, f32)) -> View {
    View { scale: view.scale, offset: (view.offset.0 + delta.0, view.offset.1 + delta.1) }
}

/// Reachability margin for [`clamp`]: at most this many px of the spread may
/// hang off any window edge (and a fully-visible spread stays centered).
const EDGE_KEEP_PX: f32 = 64.0;

/// Clamp pan so the spread cannot be lost: at least `EDGE_KEEP_PX` px of it
/// stays inside the window. An axis where the scaled spread fits the window
/// is pinned to the centered position.
pub fn clamp(view: View, total: (f32, f32), window: (f32, f32)) -> View {
    let axis = |offset: f32, content: f32, window: f32| -> f32 {
        if content <= window {
            (window - content) / 2.0
        } else {
            offset.clamp(window - content - EDGE_KEEP_PX, EDGE_KEEP_PX)
        }
    };
    let scaled = (total.0 * view.scale, total.1 * view.scale);
    View {
        scale: view.scale,
        offset: (
            axis(view.offset.0, scaled.0, window.0),
            axis(view.offset.1, scaled.1, window.1),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_flags() {
        assert!(!ViewMode::Normal.double());
        assert!(!ViewMode::Normal.rtl());
        assert!(ViewMode::Manga.double());
        assert!(ViewMode::Manga.rtl());
        assert!(ViewMode::Comic.double());
        assert!(!ViewMode::Comic.rtl());
    }

    #[test]
    fn mode_table_strings() {
        assert_eq!(ViewMode::ALL, [ViewMode::Normal, ViewMode::Manga, ViewMode::Comic]);
        assert_eq!(ViewMode::Normal.effect(), "left to right · single page");
        assert_eq!(ViewMode::Manga.effect(), "right to left · double page");
        assert_eq!(ViewMode::Comic.effect(), "left to right · double page");
    }

    #[test]
    fn layout_places_seats_left_to_right() {
        let l = layout([Some((100, 200)), Some((100, 200))]);
        assert_eq!(l.rects[0], Some(PageRect { pos: (0.0, 0.0), size: (100.0, 200.0) }));
        assert_eq!(l.rects[1], Some(PageRect { pos: (100.0, 0.0), size: (100.0, 200.0) }));
        assert_eq!(l.total(), (200.0, 200.0));
    }

    #[test]
    fn layout_asymmetric_pair_top_aligned() {
        let l = layout([Some((100, 300)), Some((200, 150))]);
        assert_eq!(l.rects[0], Some(PageRect { pos: (0.0, 0.0), size: (100.0, 300.0) }));
        assert_eq!(l.rects[1], Some(PageRect { pos: (100.0, 0.0), size: (200.0, 150.0) }));
        assert_eq!(l.total(), (300.0, 300.0));
    }

    #[test]
    fn layout_skips_undecoded_seats() {
        let l = layout([Some((100, 200)), None]);
        assert_eq!(l.rects[1], None);
        assert_eq!(l.total(), (100.0, 200.0));
        // an undecoded left seat leaves the right page starting at x=0
        let l = layout([None, Some((100, 200))]);
        assert_eq!(l.rects[0], None);
        assert_eq!(l.rects[1], Some(PageRect { pos: (0.0, 0.0), size: (100.0, 200.0) }));
    }

    #[test]
    fn fit_scales_to_min_axis_and_centers() {
        let v = fit((1000.0, 2000.0), (500.0, 500.0));
        assert!((v.scale - 0.25).abs() < 1e-6);
        assert!((v.offset.0 - 125.0).abs() < 1e-6);
        assert!((v.offset.1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn fit_degenerate_inputs_are_inert() {
        let v = fit((0.0, 0.0), (500.0, 500.0));
        assert_eq!(v, View { scale: 1.0, offset: (0.0, 0.0) });
        let v = fit((100.0, 100.0), (0.0, 0.0));
        assert_eq!(v, View { scale: 1.0, offset: (0.0, 0.0) });
    }

    #[test]
    fn zoom_keeps_cursor_anchored() {
        let v = View { scale: 2.0, offset: (100.0, 50.0) };
        let v = zoom_at(v, (300.0, 200.0), 2.0);
        // image pixel under the cursor before: (300-100)/2 = 100; after: (300-offset)/4
        assert!((300.0 - v.offset.0) / v.scale - 100.0 < 0.01);
    }

    #[test]
    fn zoom_clamps_scale() {
        let v = zoom_at(View { scale: 30.0, offset: (0.0, 0.0) }, (0.0, 0.0), 100.0);
        assert_eq!(v.scale, MAX_SCALE);
        let v = zoom_at(View { scale: 0.1, offset: (0.0, 0.0) }, (0.0, 0.0), 0.001);
        assert_eq!(v.scale, MIN_SCALE);
    }

    #[test]
    fn clamp_oversized_keeps_edge_reachable() {
        // 2000 content px at scale 1 in a 500 px window
        let v = clamp(View { scale: 1.0, offset: (-5000.0, 0.0) }, (2000.0, 400.0), (500.0, 500.0));
        assert!(v.offset.0 >= 500.0 - 2000.0 - EDGE_KEEP_PX - 1e-4, "{}", v.offset.0);
        assert!(v.offset.0 <= EDGE_KEEP_PX + 1e-4);
        // y fits (400 < 500) so it is centered
        assert!((v.offset.1 - 50.0).abs() < 1e-4);
    }

    #[test]
    fn clamp_centers_fitted_axes() {
        let total = (400.0, 300.0);
        let win = (1000.0, 800.0);
        let fitted = fit(total, win);
        let dragged = pan(fitted, (300.0, -200.0));
        let c = clamp(dragged, total, win);
        assert!((c.offset.0 - fitted.offset.0).abs() < 1e-4);
        assert!((c.offset.1 - fitted.offset.1).abs() < 1e-4);
    }

    #[test]
    fn clamp_never_changes_scale() {
        let v = clamp(View { scale: 7.0, offset: (1e9, -1e9) }, (100.0, 100.0), (50.0, 50.0));
        assert_eq!(v.scale, 7.0);
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
        // 6 pages: the would-be pair (5, 6) shows page 5 alone, with no
        // phantom page.
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
        // The cover's next spread starts at 1, not 2 (status-line bug class).
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
        let s = switch_mode(ViewMode::Normal, ViewMode::Comic, 0, 10);
        assert_eq!(s.index, 0);
        assert!(s.pairing_changed);
        assert_eq!(
            switch_mode(ViewMode::Manga, ViewMode::Comic, 5, 10),
            ModeSwitch { index: 5, pairing_changed: false }
        );
        // So does re-selecting the mode.
        let s = switch_mode(ViewMode::Manga, ViewMode::Manga, 5, 10);
        assert_eq!(s.index, 5);
        assert!(!s.pairing_changed);
    }

    #[test]
    fn seats_follow_reading_direction() {
        let s = Spread { first: 1, second: Some(2) };
        assert_eq!(s.seats(ViewMode::Normal), [Some(1), Some(2)]);
        assert_eq!(s.seats(ViewMode::Comic), [Some(1), Some(2)]);
        assert_eq!(s.seats(ViewMode::Manga), [Some(2), Some(1)]);
        let lone = Spread { first: 5, second: None };
        assert_eq!(lone.seats(ViewMode::Normal), [Some(5), None]);
        assert_eq!(lone.seats(ViewMode::Manga), [None, Some(5)]);
    }
}
