//! Status overlay: one egui bottom panel, data in / text out.

use crate::view::{Spread, ViewMode};

/// Everything the status bar shows. Built by the app each frame from
/// plain state. No logic lives here.
pub struct StatusLine {
    pub total: usize,
    pub spread: Spread,
    pub mode: ViewMode,
    /// The mode-switch toast is showing.
    pub toast: bool,
    pub source_label: &'static str,
    pub format: Option<&'static str>,
    pub dims: Option<(u32, u32)>,
    pub zoom_percent: f32,
    pub waiting: bool,
    pub error: Option<String>,
}

/// Render the bottom status bar. Monospace, single row, no
/// interaction. Book facts on the left. The mode group on the right,
/// active mode colored. egui 0.37 panels show into a `&mut Ui` (what
/// `EguiGlow::run` hands over), not into the context. Example left
/// side: `12/214-13 · cbz · jxl 6000x4000 · 87% · decoding…`
pub fn show(ui: &mut egui::Ui, status: &StatusLine) {
    egui::Panel::bottom("status").show(ui, |ui| {
        // Sides measures the right end from what it renders, so the
        // facts truncate at the group's true edge. `shrink_left()` is
        // load-bearing. Without it the group parks beside the facts
        // instead of at the right edge.
        egui::Sides::new().shrink_left().show(
            ui,
            |ui| {
                ui.add(egui::Label::new(egui::RichText::new(line(status)).monospace()).truncate());
            },
            |ui| right_end(ui, status),
        );
    });
}

/// The right end of the bar. The mode group, with the active mode
/// colored and the rest gray. A showing toast sits left of it.
fn right_end(ui: &mut egui::Ui, status: &StatusLine) {
    for (i, &mode) in ViewMode::ALL.iter().enumerate().rev() {
        let text = egui::RichText::new(group_label(mode)).monospace();
        let text =
            if mode == status.mode { text.color(mode_color(mode)).strong() } else { text.weak() };
        ui.label(text);
        if i > 0 {
            ui.label(egui::RichText::new("/").weak().monospace());
        }
    }
    if status.toast {
        ui.label(egui::RichText::new(status.mode.effect()).monospace());
    }
}

/// One entry of the mode group, e.g. `Normal (n)`.
fn group_label(mode: ViewMode) -> String {
    format!("{} ({})", mode.label(), mode.shortcut())
}

/// A muted hue per mode, for the active entry of the group.
fn mode_color(mode: ViewMode) -> egui::Color32 {
    match mode {
        ViewMode::Normal => egui::Color32::from_rgb(0x87, 0xb3, 0x7d),
        ViewMode::Manga => egui::Color32::from_rgb(0xd2, 0x7b, 0x7b),
        ViewMode::Comic => egui::Color32::from_rgb(0x7b, 0x9f, 0xd2),
    }
}

/// The single status line, separate so it is directly testable.
fn line(status: &StatusLine) -> String {
    let mut head = format!("{}/{}", status.spread.first + 1, status.total);
    if let Some(second) = status.spread.second {
        head.push_str(&format!("-{}", second + 1));
    }
    let mut segs = vec![head, status.source_label.to_string()];
    if let (Some(format), Some((w, h))) = (status.format, status.dims) {
        segs.push(format!("{format} {w}x{h}"));
    }
    segs.push(format!("{:.0}%", status.zoom_percent));
    if status.waiting {
        segs.push("decoding…".into());
    }
    if let Some(error) = &status.error {
        segs.push(error.clone());
    }
    segs.join(" · ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status() -> StatusLine {
        StatusLine {
            total: 214,
            spread: Spread { first: 11, second: Some(12) },
            mode: ViewMode::Manga,
            toast: false,
            source_label: "cbz",
            format: Some("jxl"),
            dims: Some((6000, 4000)),
            zoom_percent: 86.6,
            waiting: false,
            error: None,
        }
    }

    #[test]
    fn full_line() {
        assert_eq!(line(&status()), "12/214-13 · cbz · jxl 6000x4000 · 87%");
    }

    #[test]
    fn waiting_and_error_append() {
        let mut s = status();
        s.waiting = true;
        s.error = Some("boom".into());
        assert!(line(&s).ends_with("· decoding… · boom"));
    }

    #[test]
    fn single_page_and_unknown_format() {
        let mut s = status();
        s.spread = Spread { first: 4, second: None };
        s.format = None;
        s.dims = None;
        s.zoom_percent = 100.0;
        assert_eq!(line(&s), "5/214 · cbz · 100%");
    }

    #[test]
    fn group_names_every_mode_with_its_key() {
        assert_eq!(group_label(ViewMode::Normal), "Normal (n)");
        assert_eq!(group_label(ViewMode::Manga), "Manga (m)");
        assert_eq!(group_label(ViewMode::Comic), "Comic (c)");
    }

    /// One painted text item: its glyph text, screen rect, elided flag.
    struct TextItem {
        text: String,
        rect: egui::Rect,
        elided: bool,
    }

    fn walk(shape: &egui::Shape, out: &mut Vec<TextItem>) {
        match shape {
            egui::Shape::Text(t) => out.push(TextItem {
                text: t.galley.rows.iter().map(|r| r.text()).collect::<String>(),
                rect: t.galley.rect.translate(t.pos.to_vec2()),
                elided: t.galley.elided,
            }),
            egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
            _ => {}
        }
    }

    /// Render the bar headlessly on a `width`-wide screen and return
    /// the text items of the settled frame. egui runs fully on the
    /// CPU here. Only the texture upload would need a GPU, and
    /// `drop_without_applying_deltas` skips it.
    fn render(width: f32, status: &StatusLine) -> Vec<TextItem> {
        let ctx = egui::Context::default();
        let input = || egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width, 600.0),
            )),
            ..Default::default()
        };
        ctx.run_ui(input(), |ui| show(ui, status)).drop_without_applying_deltas();
        let mut out = ctx.run_ui(input(), |ui| show(ui, status));
        let shapes = std::mem::take(&mut out.shapes);
        out.drop_without_applying_deltas();
        let mut items = Vec::new();
        shapes.iter().for_each(|cs| walk(&cs.shape, &mut items));
        items
    }

    fn find<'a>(items: &'a [TextItem], prefix: &str) -> &'a TextItem {
        items.iter().find(|t| t.text.starts_with(prefix)).expect(prefix)
    }

    /// The panel frame's inner margin is the flush-left and
    /// flush-right anchor. `ROUND` tolerates pixel rounding.
    const EDGE: f32 = 8.0;
    const ROUND: f32 = 0.51;
    const WIDE: f32 = 1280.0;

    #[test]
    fn facts_flush_left_and_never_shift() {
        let plain = render(WIDE, &status());
        let mut with_toast = status();
        with_toast.toast = true;
        let toast = render(WIDE, &with_toast);
        assert!((find(&plain, "12/214-13").rect.left() - EDGE).abs() < ROUND);
        assert_eq!(
            find(&plain, "12/214-13").rect.left(),
            find(&toast, "12/214-13").rect.left()
        );
    }

    #[test]
    fn group_anchored_right_regardless_of_toast() {
        let plain = render(WIDE, &status());
        let mut with_toast = status();
        with_toast.toast = true;
        let toast = render(WIDE, &with_toast);
        for items in [&plain, &toast] {
            let comic = find(items, "Comic (c)");
            let rightmost = items.iter().map(|t| t.rect.right()).fold(f32::MIN, f32::max);
            assert!((rightmost - comic.rect.right()).abs() < ROUND, "group must be rightmost");
            assert!((comic.rect.right() - (WIDE - EDGE)).abs() < ROUND, "group must be flush right");
        }
        assert_eq!(
            find(&plain, "Comic (c)").rect,
            find(&toast, "Comic (c)").rect,
            "the group must not move when the toast appears"
        );
    }

    #[test]
    fn toast_sits_left_of_the_group_only_when_shown() {
        let mut with_toast = status();
        with_toast.toast = true;
        let toast = render(WIDE, &with_toast);
        assert!(find(&toast, "right to left").rect.right() < find(&toast, "Comic (c)").rect.left());
        let plain = render(WIDE, &status());
        assert!(plain.iter().all(|i| !i.text.contains(status().mode.effect())));
    }

    #[test]
    fn facts_truncate_instead_of_pushing_the_group() {
        let narrow = render(480.0, &status());
        let facts = find(&narrow, "12/214");
        assert!(facts.elided, "facts must elide on a narrow screen");
        assert!(facts.text.ends_with('…'));
        assert!(facts.rect.right() <= find(&narrow, "Comic (c)").rect.left());
        let wide = render(WIDE, &status());
        assert!(!find(&wide, "12/214").elided);
    }
}
