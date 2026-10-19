//! View math: how a composed spread (1–2 pages) is placed in the window.
//! Pure data, no GL.
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

/// Layout of a spread's visual slots: `rects` is always [left, right]; slot
/// seating (which page goes where) was decided before layout by
/// [`crate::page::Spread::seats`]. A slot with no decoded page yet is `None`.
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
/// decoded size of visual slot i (0 = left, 1 = right; None while still
/// decoding — that page joins the layout when it arrives).
pub fn layout(dims: [Option<(u32, u32)>; 2]) -> SpreadLayout {
    let mut rects: [Option<PageRect>; 2] = [None, None];
    let mut next_slot_x = 0.0;
    for (slot, dims) in dims.iter().enumerate() {
        if let Some((width, height)) = *dims {
            rects[slot] =
                Some(PageRect { pos: (next_slot_x, 0.0), size: (width as f32, height as f32) });
            next_slot_x += width as f32;
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
pub const MIN_SCALE: f32 = 0.05;
pub const MAX_SCALE: f32 = 40.0;

/// Zoom by `factor`, keeping the image point under `cursor` (screen px)
/// anchored. Clamps scale to [`MIN_SCALE`]..=[`MAX_SCALE`]; the anchor uses
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
pub const KEEP: f32 = 64.0;

/// Clamp pan so the spread cannot be lost: at least `KEEP` px of it stays
/// inside the window. An axis where the scaled spread fits the window is
/// pinned to the centered position.
pub fn clamp(view: View, total: (f32, f32), window: (f32, f32)) -> View {
    let axis = |offset: f32, content: f32, window: f32| -> f32 {
        if content <= window {
            (window - content) / 2.0
        } else {
            offset.clamp(window - content - KEEP, KEEP)
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
    fn layout_places_slots_left_to_right() {
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
    fn layout_skips_undecoded_slots() {
        let l = layout([Some((100, 200)), None]);
        assert_eq!(l.rects[1], None);
        assert_eq!(l.total(), (100.0, 200.0));
        // an undecoded left slot leaves the right page starting at x=0
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
