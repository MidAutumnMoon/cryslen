//! Status overlay: one egui bottom panel, data in / text out.

use std::sync::Arc;

use egui::text::LayoutJob;

use crate::decode::Format;
use crate::fault::PageError;
use crate::fault::StatusFault;
use crate::source::PageName;
use crate::view::Spread;
use crate::view::ViewMode;
use crate::worker::SeatView;
use crate::worker::SpreadView;

/// Everything the status bar shows, built by the app each frame.
pub struct StatusLine {
    pub page_count: usize,
    pub spread: Spread,
    pub mode: ViewMode,
    /// The mode-switch toast is showing.
    pub toast: bool,
    pub name: PageName,
    /// Facts of the page the line speaks for, once it has decoded.
    pub page: Option<PageFacts>,
    /// Placement scale, screen px per image px.
    pub zoom: f32,
    /// The wait flag, once a pending seat has outlived the grace.
    pub waiting: Option<Wait>,
    pub error: Option<StatusFault>,
}

/// What the status line says a still-pending page waits for.
#[derive(Clone, Copy)]
pub enum Wait {
    /// The archive's unpack pass has not ended.
    Unpacking,
    /// The source is not unpacking; the wait is the decode.
    Decoding,
}

impl Wait {
    fn label(self) -> &'static str {
        match self {
            Self::Unpacking => "unpacking…",
            Self::Decoding => "decoding…",
        }
    }
}

/// The status line's view of one spread's snapshot: whether it still
/// waits, the first failure in reading order, and the first decoded
/// page's facts.
pub struct FoldedStatus {
    pub waiting: bool,
    /// The first failed page in reading order, carrying the page so
    /// the status line can speak for the page that failed, not the
    /// first page that decoded.
    pub error: Option<(usize, Arc<PageError>)>,
    pub page: Option<(usize, Format, (u32, u32))>,
}

/// Fold the spread's pages into the status line's error and page facts
/// in reading order.
pub fn fold_status(spread: Spread, snapshot: &SpreadView) -> FoldedStatus {
    let mut folded = FoldedStatus {
        waiting: matches!(snapshot.first, SeatView::Pending)
            || matches!(snapshot.second.as_ref(), Some(SeatView::Pending)),
        error: None,
        page: None,
    };
    for (page, view) in snapshot.reading_order(spread) {
        match view {
            SeatView::Failed(error) => {
                folded.error = Some((page, Arc::clone(error)));
                break;
            }
            SeatView::Ready(decoded) if folded.page.is_none() => {
                folded.page = Some((
                    page,
                    decoded.format,
                    (decoded.width, decoded.height),
                ));
            }
            SeatView::Ready(_) | SeatView::Pending => {}
        }
    }
    folded
}

pub struct PageFacts {
    /// Format the decoder identified, the content truth.
    pub decoded: Format,
    /// Format the filename's extension claims. Listed pages always
    /// carry one.
    pub claimed: Option<Format>,
    pub dims: (u32, u32),
}

/// Render the status bar into `ui` and return the content rect: what
/// the chrome has not consumed, the only legal page viewport. A hidden
/// bar (`fullscreen`) consumes nothing, so the rect is the whole window.
/// egui 0.37 panels show into the `&mut Ui` that `EguiGlow::run` hands
/// over, not into a context.
pub fn show(
    ui: &mut egui::Ui,
    status: &StatusLine,
    show_bar: bool,
) -> egui::Rect {
    if !show_bar {
        return ui.available_rect_before_wrap();
    }
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
    ui.available_rect_before_wrap()
}

/// The mode group plus a showing toast to its left. The Sides right
/// side lays out right-to-left, so the display order iterates in
/// reverse.
fn right_end(ui: &mut egui::Ui, status: &StatusLine) {
    for (mode_index, &mode) in ViewMode::ALL.iter().enumerate().rev() {
        let text = egui::RichText::new(group_label(mode)).monospace();
        let text = if mode == status.mode {
            text.color(mode_color(mode)).strong()
        } else {
            text.weak()
        };
        ui.label(text);
        if mode_index > 0 {
            ui.label(egui::RichText::new("·").weak().monospace());
        }
    }
    if status.toast {
        ui.label(egui::RichText::new(status.mode.effect()).monospace());
    }
}

fn group_label(mode: ViewMode) -> String {
    format!("{} ({})", mode.label(), mode.shortcut())
}

fn mode_color(mode: ViewMode) -> egui::Color32 {
    match mode {
        ViewMode::Normal => egui::Color32::from_rgb(0x87, 0xb3, 0x7d),
        ViewMode::Manga => egui::Color32::from_rgb(0xd2, 0x7b, 0x7b),
        ViewMode::Comic => egui::Color32::from_rgb(0x7b, 0x9f, 0xd2),
    }
}

/// Muted hue for the spread range, quieter than the mode colors so the
/// group keeps the bar's strongest accent.
const HEAD_COLOR: egui::Color32 =
    egui::Color32::from_rgb(0xc2, 0x9e, 0x6f);

const NAME_COLOR: egui::Color32 =
    egui::Color32::from_rgb(0xa2, 0x92, 0xc5);

/// The left-side facts as one job of colored sections, separate so they
/// are directly testable. One job, one label: the whole line keeps
/// eliding together on narrow screens.
fn line(style: &egui::Style, status: &StatusLine) -> LayoutJob {
    let font = egui::TextStyle::Monospace.resolve(style);
    let plain = style.visuals.text_color();
    let section =
        |job: &mut LayoutJob, text: &str, color: egui::Color32| {
            job.append(
                text,
                0.0,
                egui::TextFormat::simple(font.clone(), color),
            );
        };
    let mut job = LayoutJob::default();

    let head = status.spread.second.map_or_else(
        || format!("{}", status.spread.first + 1),
        |second| format!("{}-{}", status.spread.first + 1, second + 1),
    );
    section(&mut job, &head, HEAD_COLOR);
    section(&mut job, &format!("/{}", status.page_count), plain);

    section(&mut job, " · ", plain);
    if let Some(parent) = &status.name.parent {
        section(&mut job, parent, plain);
        section(&mut job, "/", plain);
    }
    section(&mut job, &status.name.file, NAME_COLOR);

    if let Some(facts) = &status.page {
        section(&mut job, " · ", plain);
        let (width, height) = facts.dims;
        let dims_text = if facts.claimed == Some(facts.decoded) {
            format!("{width}x{height}")
        } else {
            format!("{} {width}x{height}", facts.decoded.label())
        };
        section(&mut job, &dims_text, plain);
    }
    section(&mut job, &format!(" · {:.0}%", status.zoom * 100.0), plain);
    if let Some(wait) = status.waiting {
        section(&mut job, &format!(" · {}", wait.label()), plain);
    }
    if let Some(error) = &status.error {
        section(&mut job, &format!(" · {error}"), plain);
    }
    job
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;
    use crate::testkit::failed_page;
    use crate::testkit::failed_seat;
    use crate::testkit::ready_seat;

    #[test]
    fn fold_status_table() {
        struct Expect {
            waiting: bool,
            error: Option<&'static str>,
            page: Option<(usize, Format, (u32, u32))>,
        }
        let pair = Spread {
            first: 0,
            second: Some(1),
        };
        let spread_view = |first: SeatView, second: Option<SeatView>| {
            SpreadView { first, second }
        };
        let rows = [
            (
                pair,
                spread_view(ready_seat(2, 2), Some(failed_seat("x"))),
                Expect {
                    waiting: false,
                    error: Some("x"),
                    page: Some((0, Format::Png, (2, 2))),
                },
            ),
            (
                pair,
                spread_view(SeatView::Pending, Some(ready_seat(2, 2))),
                Expect {
                    waiting: true,
                    error: None,
                    page: Some((1, Format::Png, (2, 2))),
                },
            ),
            (
                Spread::lone(0),
                spread_view(ready_seat(2, 2), None),
                Expect {
                    waiting: false,
                    error: None,
                    page: Some((0, Format::Png, (2, 2))),
                },
            ),
            (
                pair,
                spread_view(
                    failed_seat("first"),
                    Some(failed_seat("second")),
                ),
                Expect {
                    waiting: false,
                    error: Some("first"),
                    page: None,
                },
            ),
            (
                pair,
                spread_view(
                    ready_seat(800, 600),
                    Some(failed_seat("second")),
                ),
                Expect {
                    waiting: false,
                    error: Some("second"),
                    page: Some((0, Format::Png, (800, 600))),
                },
            ),
            (
                pair,
                spread_view(
                    failed_seat("first"),
                    Some(ready_seat(640, 480)),
                ),
                Expect {
                    waiting: false,
                    error: Some("first"),
                    page: None,
                },
            ),
            (
                pair,
                spread_view(SeatView::Pending, Some(failed_seat("x"))),
                Expect {
                    waiting: true,
                    error: Some("x"),
                    page: None,
                },
            ),
            (
                Spread::lone(0),
                spread_view(SeatView::Pending, None),
                Expect {
                    waiting: true,
                    error: None,
                    page: None,
                },
            ),
        ];
        for (i, (spread, snapshot, want)) in rows.iter().enumerate() {
            let folded = fold_status(*spread, snapshot);
            assert_eq!(folded.waiting, want.waiting, "row {i}: waiting");
            assert_eq!(
                folded
                    .error
                    .as_ref()
                    .map(|(_, error)| error.to_string())
                    .as_deref(),
                want.error,
                "row {i}: error"
            );
            assert_eq!(folded.page, want.page, "row {i}: page");
        }
    }

    fn status() -> StatusLine {
        StatusLine {
            page_count: 214,
            spread: Spread {
                first: 11,
                second: Some(12),
            },
            mode: ViewMode::Manga,
            toast: false,
            name: PageName {
                parent: Some("b".into()),
                file: "010.png".into(),
            },
            // The name claims png and the page decodes as jxl, so the
            // format label prefixes the dims.
            page: Some(PageFacts {
                decoded: Format::Jxl,
                claimed: Some(Format::Png),
                dims: (6000, 4000),
            }),
            zoom: 0.866,
            waiting: None,
            error: None,
        }
    }

    #[test]
    fn status_line_text() {
        let style = egui::Style::default();
        let plain = style.visuals.text_color();
        let base = status();
        // matching extension: the format label hides.
        let mut matching = status();
        if let Some(page) = &mut matching.page {
            page.claimed = Some(Format::Jxl);
        }
        // single page with no decoded facts yet.
        let mut single = status();
        single.spread = Spread::lone(4);
        single.page = None;
        single.zoom = 1.0;
        let mut appending = status();
        appending.waiting = Some(Wait::Decoding);
        appending.error = Some(StatusFault::Page(failed_page("boom")));
        let mut unpacking = status();
        unpacking.waiting = Some(Wait::Unpacking);
        let rows = [
            (&base, "12-13/214 · b/010.png · jxl 6000x4000 · 87%"),
            (&matching, "12-13/214 · b/010.png · 6000x4000 · 87%"),
            (&single, "5/214 · b/010.png · 100%"),
            (&appending, "12-13/214 · b/010.png · jxl 6000x4000 · 87% · decoding… · boom"),
            (&unpacking, "12-13/214 · b/010.png · jxl 6000x4000 · 87% · unpacking…"),
        ];
        for (i, (status_line, want)) in rows.iter().enumerate() {
            assert_eq!(line(&style, status_line).text, *want, "row {i}");
        }
        let section_colors = |job: &LayoutJob| -> Vec<egui::Color32> {
            job.sections
                .iter()
                .map(|section| section.format.color)
                .collect()
        };
        assert_eq!(
            section_colors(&line(&style, &base)),
            vec![HEAD_COLOR, plain, NAME_COLOR, plain]
        );
        assert_eq!(group_label(ViewMode::Normal), "Normal (n)");
        assert_eq!(group_label(ViewMode::Manga), "Manga (m)");
        assert_eq!(group_label(ViewMode::Comic), "Comic (c)");
    }

    struct TextItem {
        text: String,
        rect: egui::Rect,
        elided: bool,
    }

    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "egui's non-text shape variants carry no text items"
    )]
    fn walk(shape: &egui::Shape, out: &mut Vec<TextItem>) {
        match shape {
            egui::Shape::Text(text_shape) => out.push(TextItem {
                text: text_shape
                    .galley
                    .rows
                    .iter()
                    .map(|row| row.text())
                    .collect::<String>(),
                rect: text_shape
                    .galley
                    .rect
                    .translate(text_shape.pos.to_vec2()),
                elided: text_shape.galley.elided,
            }),
            egui::Shape::Vec(shapes) => {
                for child_shape in shapes {
                    walk(child_shape, out);
                }
            }
            _ => {}
        }
    }

    /// Render the bar headlessly on a `width`-wide screen and return
    /// the settled frame's text items and content rect. egui panel
    /// sizing settles one frame late, so a settle pass runs first.
    /// Only the texture upload needs a GPU, and
    /// `drop_without_applying_deltas` skips it.
    fn render(
        width: f32,
        status: &StatusLine,
    ) -> (Vec<TextItem>, egui::Rect) {
        use std::mem;

        let ctx = egui::Context::default();
        let input = || egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width, SCREEN_HEIGHT),
            )),
            ..Default::default()
        };
        ctx.run_ui(input(), |ui| {
            show(ui, status, true);
        })
        .drop_without_applying_deltas();
        let mut content = egui::Rect::NOTHING;
        let mut out =
            ctx.run_ui(input(), |ui| content = show(ui, status, true));
        let shapes = mem::take(&mut out.shapes);
        out.drop_without_applying_deltas();
        let mut items = Vec::new();
        for shape in &shapes {
            walk(&shape.shape, &mut items);
        }
        (items, content)
    }

    fn find<'items>(
        items: &'items [TextItem],
        prefix: &str,
    ) -> &'items TextItem {
        items
            .iter()
            .find(|item| item.text.starts_with(prefix))
            .expect(prefix)
    }

    /// The panel frame's inner margin is the flush-left and
    /// flush-right anchor. `ROUND` tolerates pixel rounding.
    const PANEL_MARGIN: f32 = 8.0;
    const ROUND: f32 = 0.51;
    const SCREEN_HEIGHT: f32 = 600.0;
    /// Screen widths: wide enough to show everything, narrow enough to
    /// force elision.
    const WIDE: f32 = 1280.0;
    const NARROW: f32 = 480.0;

    #[test]
    fn bar_geometry() {
        let (plain, wide_rect) = render(WIDE, &status());
        let mut with_toast = status();
        with_toast.toast = true;
        let (toast, _) = render(WIDE, &with_toast);
        let (narrow, narrow_rect) = render(NARROW, &status());

        assert!(wide_rect.min.x.abs() < ROUND, "{}", wide_rect.min.x);
        assert!(wide_rect.min.y.abs() < ROUND, "{}", wide_rect.min.y);
        assert!(
            (wide_rect.width() - WIDE).abs() < ROUND,
            "{}",
            wide_rect.width()
        );
        assert!(
            wide_rect.max.y > 400.0,
            "content must span most of the window: {}",
            wide_rect.max.y
        );
        assert!(
            wide_rect.max.y < SCREEN_HEIGHT,
            "content must exclude the bar strip: {}",
            wide_rect.max.y
        );

        assert!(
            (narrow_rect.width() - NARROW).abs() < ROUND,
            "{}",
            narrow_rect.width()
        );
        // 8.0 tolerates a text line of bar-height difference across widths,
        // unlike ROUND's sub-pixel rounding.
        assert!(
            (narrow_rect.max.y - wide_rect.max.y).abs() < 8.0,
            "{narrow_rect:?} vs {wide_rect:?}"
        );

        assert!(
            (find(&plain, "12-13/214").rect.left() - PANEL_MARGIN).abs()
                < ROUND
        );
        assert_eq!(
            find(&plain, "12-13/214").rect.left(),
            find(&toast, "12-13/214").rect.left()
        );

        for items in [&plain, &toast] {
            let comic = find(items, "Comic (c)");
            let rightmost = items
                .iter()
                .map(|item| item.rect.right())
                .fold(f32::MIN, f32::max);
            assert!(
                (rightmost - comic.rect.right()).abs() < ROUND,
                "group must be rightmost"
            );
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

        assert!(
            find(&toast, "right to left").rect.right()
                < find(&toast, "Comic (c)").rect.left()
        );
        assert!(plain
            .iter()
            .all(|item| !item.text.contains(status().mode.effect())));

        let facts = find(&narrow, "12-13");
        assert!(facts.elided, "facts must elide on a narrow screen");
        assert!(facts.text.ends_with('…'));
        assert!(
            facts.rect.right() <= find(&narrow, "Comic (c)").rect.left()
        );
        assert!(!find(&plain, "12-13").elided);
    }
}
