//! Status overlay: one egui bottom panel, data in / text out.

use crate::page::Spread;

/// Everything the status line shows. Built by the app each frame from plain
/// state. No logic lives here.
pub struct StatusLine {
    pub total: usize,
    pub spread: Spread,
    pub double: bool,
    pub rtl: bool,
    pub source_label: &'static str,
    pub format: Option<&'static str>,
    pub dims: Option<(u32, u32)>,
    pub zoom_percent: f32,
    pub waiting: bool,
    pub error: Option<String>,
}

/// Render the bottom status panel. Monospace, single line, no interaction.
/// egui 0.37 panels show into a `&mut Ui` (what `EguiGlow::run` hands over),
/// not into the context. Example output:
/// `12/214-13 · cbz · jxl 6000x4000 · 87% · rtl 2page · decoding…`
pub fn show(ui: &mut egui::Ui, status: &StatusLine) {
    egui::Panel::bottom("status").show(ui, |ui| {
        ui.label(egui::RichText::new(line(status)).monospace());
    });
}

/// The single status line, separate so it is directly testable.
pub fn line(status: &StatusLine) -> String {
    let mut head = format!("{}/{}", status.spread.first + 1, status.total);
    if let Some(second) = status.spread.second {
        head.push_str(&format!("-{}", second + 1));
    }
    let mut segs = vec![head, status.source_label.to_string()];
    if let (Some(format), Some((w, h))) = (status.format, status.dims) {
        segs.push(format!("{format} {w}x{h}"));
    }
    segs.push(format!("{:.0}%", status.zoom_percent));
    let mut modes = String::new();
    if status.rtl {
        modes.push_str("rtl");
    }
    if status.double {
        if !modes.is_empty() {
            modes.push(' ');
        }
        modes.push_str("2page");
    }
    if !modes.is_empty() {
        segs.push(modes);
    }
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
            double: true,
            rtl: true,
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
        assert_eq!(
            line(&status()),
            "12/214-13 · cbz · jxl 6000x4000 · 87% · rtl 2page"
        );
    }

    #[test]
    fn waiting_and_error_append() {
        let mut s = status();
        s.waiting = true;
        s.error = Some("boom".into());
        assert!(line(&s).ends_with("· rtl 2page · decoding… · boom"));
    }

    #[test]
    fn single_page_and_unknown_format() {
        let mut s = status();
        s.spread = Spread { first: 4, second: None };
        s.double = false;
        s.rtl = false;
        s.format = None;
        s.dims = None;
        s.zoom_percent = 100.0;
        assert_eq!(line(&s), "5/214 · cbz · 100%");
    }
}
