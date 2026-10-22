//! Status overlay: one egui bottom panel, data in / text out.

use crate::decode::Format;
use crate::source::PageName;
use crate::view::{Spread, ViewMode};

/// Everything the status bar shows. Built by the app each frame. No
/// logic lives here beyond formatting.
pub struct StatusLine {
    pub total: usize,
    pub spread: Spread,
    pub mode: ViewMode,
    /// The mode-switch toast is showing.
    pub toast: bool,
    pub name: PageName,
    /// Facts of the page the line speaks for, once it has decoded.
    pub page: Option<PageFacts>,
    /// Placement scale, screen px per image px.
    pub zoom: f32,
    pub waiting: bool,
    pub error: Option<String>,
}

/// The decoded page as the status line sees it.
pub struct PageFacts {
    /// Format the decoder identified, the content truth.
    pub decoded: Format,
    /// Format the filename's extension claims. Listed pages always
    /// carry one.
    pub claimed: Option<Format>,
    pub dims: (u32, u32),
}

/// Render the status bar into `ui`. egui 0.37 panels show into the
/// `&mut Ui` that `EguiGlow::run` hands over, not into a context.
pub fn show(ui: &mut egui::Ui, status: &StatusLine) {
    egui::Panel::bottom("status").show(ui, |ui| {
        // Sides measures the right end from what it renders, so the
        // facts truncate at the group's true edge. Without
        // `shrink_left()` the group parks beside the facts.
        egui::Sides::new().shrink_left().show(
            ui,
            |ui| {
                let job = line(ui.style(), status);
                ui.add(egui::Label::new(job).truncate());
            },
            |ui| right_end(ui, status),
        );
    });
}

/// The mode group plus a showing toast to its left. The Sides right
/// side lays out right-to-left, so the display order iterates in
/// reverse.
fn right_end(ui: &mut egui::Ui, status: &StatusLine) {
    for (i, &mode) in ViewMode::ALL.iter().enumerate().rev() {
        let text = egui::RichText::new(group_label(mode)).monospace();
        let text =
            if mode == status.mode { text.color(mode_color(mode)).strong() } else { text.weak() };
        ui.label(text);
        if i > 0 {
            ui.label(egui::RichText::new("·").weak().monospace());
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

/// Muted hue for the spread range, quieter than the mode colors so the
/// group keeps the bar's strongest accent.
const HEAD_COLOR: egui::Color32 = egui::Color32::from_rgb(0xc2, 0x9e, 0x6f);

/// Muted hue for the current file's basename.
const NAME_COLOR: egui::Color32 = egui::Color32::from_rgb(0xa2, 0x92, 0xc5);

/// The left-side facts as one job of colored sections, separate so they
/// are directly testable. One job, one label: the whole line keeps
/// eliding together on narrow screens.
fn line(style: &egui::Style, status: &StatusLine) -> egui::text::LayoutJob {
    let font = egui::TextStyle::Monospace.resolve(style);
    let plain = style.visuals.text_color();
    let section = |job: &mut egui::text::LayoutJob, text: String, color: egui::Color32| {
        job.append(&text, 0.0, egui::TextFormat::simple(font.clone(), color));
    };
    let mut job = egui::text::LayoutJob::default();

    let mut head = format!("{}", status.spread.first + 1);
    if let Some(second) = status.spread.second {
        head.push_str(&format!("-{}", second + 1));
    }
    section(&mut job, head, HEAD_COLOR);
    section(&mut job, format!("/{}", status.total), plain);

    section(&mut job, " · ".into(), plain);
    if let Some(parent) = &status.name.parent {
        section(&mut job, parent.clone(), plain);
        section(&mut job, "/".into(), plain);
    }
    section(&mut job, status.name.file.clone(), NAME_COLOR);

    if let Some(facts) = &status.page {
        section(&mut job, " · ".into(), plain);
        // A format label shows only when the filename's extension lies.
        let (w, h) = facts.dims;
        let dims_text = if facts.claimed != Some(facts.decoded) {
            format!("{} {w}x{h}", facts.decoded.label())
        } else {
            format!("{w}x{h}")
        };
        section(&mut job, dims_text, plain);
    }
    section(&mut job, format!(" · {:.0}%", status.zoom * 100.0), plain);
    if status.waiting {
        section(&mut job, " · decoding…".into(), plain);
    }
    if let Some(error) = &status.error {
        section(&mut job, format!(" · {error}"), plain);
    }
    job
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
            name: PageName { parent: Some("b".into()), file: "010.png".into() },
            // The name claims png, the decoded page is jxl: the mismatch
            // fuses the format into the dims.
            page: Some(PageFacts {
                decoded: Format::Jxl,
                claimed: Some(Format::Png),
                dims: (6000, 4000),
            }),
            zoom: 0.866,
            waiting: false,
            error: None,
        }
    }

    /// Section colors in append order. Sections carry no text slices
    /// here; the text and the deterministic append order pin the mapping.
    fn section_colors(job: &egui::text::LayoutJob) -> Vec<egui::Color32> {
        job.sections.iter().map(|s| s.format.color).collect()
    }

    #[test]
    fn facts_text_and_section_colors() {
        let style = egui::Style::default();
        let plain = style.visuals.text_color();
        let job = line(&style, &status());
        assert_eq!(job.text, "12-13/214 · b/010.png · jxl 6000x4000 · 87%");
        assert_eq!(
            section_colors(&job),
            // Equal adjacent formats merge: head | range-total, parent |
            // basename | dims, zoom.
            vec![HEAD_COLOR, plain, NAME_COLOR, plain]
        );
    }

    #[test]
    fn matching_extension_hides_the_format() {
        let mut s = status();
        if let Some(page) = &mut s.page {
            page.claimed = Some(Format::Jxl);
        }
        let job = line(&egui::Style::default(), &s);
        assert_eq!(job.text, "12-13/214 · b/010.png · 6000x4000 · 87%");
    }

    #[test]
    fn single_page_head_and_uncolored_parent() {
        let style = egui::Style::default();
        let plain = style.visuals.text_color();
        let mut s = status();
        s.spread = Spread { first: 4, second: None };
        s.page = None;
        s.zoom = 1.0;
        let job = line(&style, &s);
        assert_eq!(job.text, "5/214 · b/010.png · 100%");
        assert_eq!(section_colors(&job), vec![HEAD_COLOR, plain, NAME_COLOR, plain]);
    }

    #[test]
    fn head_leads_with_the_spread_range() {
        let style = egui::Style::default();
        let mut s = status();
        s.spread = Spread { first: 49, second: Some(50) };
        s.total = 2146;
        let job = line(&style, &s);
        assert!(job.text.starts_with("50-51/2146 · "));
    }

    #[test]
    fn waiting_and_error_append() {
        let mut s = status();
        s.waiting = true;
        s.error = Some("boom".into());
        let job = line(&egui::Style::default(), &s);
        assert!(job.text.ends_with("· decoding… · boom"));
    }

    #[test]
    fn group_names_every_mode_with_its_key() {
        assert_eq!(group_label(ViewMode::Normal), "Normal (n)");
        assert_eq!(group_label(ViewMode::Manga), "Manga (m)");
        assert_eq!(group_label(ViewMode::Comic), "Comic (c)");
    }

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
    /// the text items of the settled frame. Only the texture upload
    /// needs a GPU, and `drop_without_applying_deltas` skips it.
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
    const PANEL_MARGIN: f32 = 8.0;
    const ROUND: f32 = 0.51;
    /// Screen widths: wide enough to show everything, narrow enough to
    /// force elision.
    const WIDE: f32 = 1280.0;
    const NARROW: f32 = 480.0;

    #[test]
    fn facts_flush_left_and_never_shift() {
        let plain = render(WIDE, &status());
        let mut with_toast = status();
        with_toast.toast = true;
        let toast = render(WIDE, &with_toast);
        assert!((find(&plain, "12-13/214").rect.left() - PANEL_MARGIN).abs() < ROUND);
        assert_eq!(
            find(&plain, "12-13/214").rect.left(),
            find(&toast, "12-13/214").rect.left()
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
            assert!(
                (comic.rect.right() - (WIDE - PANEL_MARGIN)).abs() < ROUND,
                "group must be flush right"
            );
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
        let narrow = render(NARROW, &status());
        let facts = find(&narrow, "12-13");
        assert!(facts.elided, "facts must elide on a narrow screen");
        assert!(facts.text.ends_with('…'));
        assert!(facts.rect.right() <= find(&narrow, "Comic (c)").rect.left());
        let wide = render(WIDE, &status());
        assert!(!find(&wide, "12-13").elided);
    }
}
