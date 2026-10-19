//! View math: how a composed spread (1–2 pages) is placed in the window.
//! Pure data, no GL. Ported functions from
//! research/prototype-rawgl/src/main.rs (fit, zoom_at) and add spread layout.

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

/// Layout of a spread: which page index (if decoded and part of the spread)
/// occupies which rect. In RTL mode the *first* page of the pair goes on the
/// RIGHT (manga reading order); `rects` is always [left_slot, right_slot].
#[derive(Debug, Clone, Copy, Default)]
pub struct SpreadLayout {
    pub rects: [Option<(usize, PageRect)>; 2],
}

impl SpreadLayout {
    /// Bounding size of the composed spread, in image pixels.
    pub fn total(&self) -> (f32, f32) {
        let mut w: f32 = 0.0;
        let mut h: f32 = 0.0;
        for (_, r) in self.rects.iter().flatten() {
            w += r.size.0;
            h = h.max(r.size.1);
        }
        (w, h)
    }
}

/// Compose one or two pages side by side, top-aligned. `dims[i]` is the
/// decoded size of visual slot i (0 = left, 1 = right; None while still
/// decoding). RTL swaps slot assignment. Pages whose dims are unknown are
/// skipped (shown when ready).
pub fn layout(
    spread: &crate::page::Spread,
    dims: [Option<(u32, u32)>; 2],
    rtl: bool,
) -> SpreadLayout {
    // Which page sits left/right: western reads first-left; manga reads
    // first-right, so a lone page lands in the right slot.
    let left = if rtl { spread.second } else { Some(spread.first) };
    let right = if rtl { Some(spread.first) } else { spread.second };
    let live = |d: Option<(u32, u32)>| d.filter(|&(w, h)| w > 0 && h > 0);

    let mut rects: [Option<(usize, PageRect)>; 2] = [None, None];
    let mut x = 0.0;
    for (slot, page) in [(0, left), (1, right)] {
        if let (Some(p), Some((w, h))) = (page, live(dims[slot])) {
            rects[slot] =
                Some((p, PageRect { pos: (x, 0.0), size: (w as f32, h as f32) }));
            x += w as f32;
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

/// Zoom by `factor`, keeping the image point under `cursor` (screen px)
/// anchored. Clamps scale to 0.05..=40; the anchor uses the *actual* factor
/// applied so hitting the clamp does not shift the image.
pub fn zoom_at(mut view: View, cursor: (f32, f32), factor: f32) -> View {
    let old = view.scale;
    view.scale = (old * factor).clamp(0.05, 40.0);
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
pub const KEEP: f32 = 64.0;

/// Clamp pan so the spread cannot be lost: at least `KEEP` px of it stays
/// inside the window. An axis where the scaled spread fits the window is
/// pinned to the centered position. Simple and predictable, no rubber bands.
pub fn clamp(view: View, total: (f32, f32), win: (f32, f32)) -> View {
    let axis = |off: f32, content: f32, win: f32| -> f32 {
        if content <= win {
            (win - content) / 2.0
        } else {
            off.clamp(win - content - KEEP, KEEP)
        }
    };
    let scaled = (total.0 * view.scale, total.1 * view.scale);
    View {
        scale: view.scale,
        offset: (
            axis(view.offset.0, scaled.0, win.0),
            axis(view.offset.1, scaled.1, win.1),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(pos: (f32, f32), size: (f32, f32)) -> (usize, PageRect) {
        (0, PageRect { pos, size })
    }

    #[test]
    fn layout_symmetric_pair_western() {
        let spread = crate::page::Spread { first: 1, second: Some(2) };
        let l = layout(&spread, [Some((100, 200)), Some((100, 200))], false);
        assert_eq!(l.rects[0].unwrap().0, 1);
        assert_eq!(l.rects[0].unwrap().1, rect((0.0, 0.0), (100.0, 200.0)).1);
        assert_eq!(l.rects[1].unwrap().0, 2);
        assert_eq!(l.rects[1].unwrap().1, rect((100.0, 0.0), (100.0, 200.0)).1);
        assert_eq!(l.total(), (200.0, 200.0));
    }

    #[test]
    fn layout_symmetric_pair_rtl_swaps() {
        let spread = crate::page::Spread { first: 1, second: Some(2) };
        let l = layout(&spread, [Some((100, 200)), Some((100, 200))], true);
        // first page (1) on the RIGHT, second (2) on the LEFT
        assert_eq!(l.rects[0].unwrap().0, 2);
        assert_eq!(l.rects[1].unwrap().0, 1);
        assert_eq!(l.total(), (200.0, 200.0));
    }

    #[test]
    fn layout_asymmetric_pair_top_aligned() {
        let spread = crate::page::Spread { first: 1, second: Some(2) };
        // western: first (100x300) left, second (200x150) right
        let l = layout(&spread, [Some((100, 300)), Some((200, 150))], false);
        assert_eq!(l.rects[0].unwrap().1, rect((0.0, 0.0), (100.0, 300.0)).1);
        assert_eq!(l.rects[1].unwrap().1, rect((100.0, 0.0), (200.0, 150.0)).1);
        assert_eq!(l.total(), (300.0, 300.0)); // widths sum, heights max
    }

    #[test]
    fn layout_single_page_western_left_rtl_right() {
        let spread = crate::page::Spread { first: 5, second: None };
        let west = layout(&spread, [Some((80, 120)), None], false);
        assert_eq!(west.rects[0].unwrap().0, 5);
        assert_eq!(west.rects[1], None);
        let rtl = layout(&spread, [None, Some((80, 120))], true);
        assert_eq!(rtl.rects[0], None);
        assert_eq!(rtl.rects[1].unwrap().0, 5);
        assert_eq!(rtl.rects[1].unwrap().1, rect((0.0, 0.0), (80.0, 120.0)).1);
    }

    #[test]
    fn layout_skips_undecoded_second_slot() {
        let spread = crate::page::Spread { first: 1, second: Some(2) };
        let l = layout(&spread, [Some((100, 200)), None], false);
        assert_eq!(l.rects[1], None);
        assert_eq!(l.total(), (100.0, 200.0));
        // and an undecoded first slot leaves the second starting at x=0
        let l = layout(&spread, [None, Some((100, 200))], false);
        assert_eq!(l.rects[0], None);
        assert_eq!(l.rects[1].unwrap().1, rect((0.0, 0.0), (100.0, 200.0)).1);
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
        assert_eq!(v.scale, 40.0);
        let v = zoom_at(View { scale: 0.1, offset: (0.0, 0.0) }, (0.0, 0.0), 0.001);
        assert_eq!(v.scale, 0.05);
    }

    #[test]
    fn clamp_oversized_keeps_edge_reachable() {
        // 2000 content px at scale 1 in a 500 px window
        let v = clamp(View { scale: 1.0, offset: (-5000.0, 0.0) }, (2000.0, 400.0), (500.0, 500.0));
        assert!(v.offset.0 >= 500.0 - 2000.0 - KEEP - 1e-4, "{}", v.offset.0);
        assert!(v.offset.0 <= KEEP + 1e-4);
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
}
