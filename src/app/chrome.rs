//! Status bar, status timers, and page-jump prompt rendering.

use std::time::Duration;
use std::time::Instant;

use egui::Color32;
use egui::Context;
use egui::Rect;
use egui::Style;
use egui::TextFormat;
use egui::Ui;
use egui::text::LayoutJob;

use crate::app::layout::ReadingMode;
use crate::app::layout::Spread;
use crate::book::catalog::PageName;
use crate::image::pixels::Format;

pub struct ReadingStatus {
    pub page_count: usize,
    pub spread: Spread,
    pub page_dims: Option<(u32, u32)>,
    /// Set only when the decoded format differs from the filename extension.
    pub actual_format: Option<Format>,
    /// Placement scale, screen px per image px.
    pub zoom_scale: f64,
}

pub struct StatusLine<'fault> {
    pub reading: Option<ReadingStatus>,
    pub mode: ReadingMode,
    pub show_mode_toast: bool,
    pub name: PageName,
    /// Whether to show `loading…`.
    pub show_loading: bool,
    pub fault: Option<&'fault str>,
}

#[derive(Clone, Copy, Default)]
enum WaitTimer {
    #[default]
    Idle,
    Grace(Instant),
    Shown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimerTick {
    pub next_deadline: Option<Instant>,
    pub changed: bool,
}

/// Status-line timers, pure in the clock the caller passes.
#[derive(Default)]
pub struct Timers {
    wait: WaitTimer,
    toast_until: Option<Instant>,
}

const WAIT_INDICATOR_GRACE: Duration = Duration::from_millis(150);

const MODE_TOAST_DURATION: Duration = Duration::from_millis(1500);

impl Timers {
    pub fn toast(&mut self, now: Instant) {
        self.toast_until = Some(now + MODE_TOAST_DURATION);
    }

    pub fn tick(&mut self, now: Instant, waiting: bool) -> TimerTick {
        let was_waiting = self.is_waiting();
        self.wait = if waiting {
            match self.wait {
                WaitTimer::Idle => {
                    WaitTimer::Grace(now + WAIT_INDICATOR_GRACE)
                }
                WaitTimer::Grace(deadline) if now >= deadline => {
                    WaitTimer::Shown
                }
                state @ (WaitTimer::Grace(_) | WaitTimer::Shown) => state,
            }
        } else {
            WaitTimer::Idle
        };
        let wait_deadline = match self.wait {
            WaitTimer::Grace(deadline) => Some(deadline),
            WaitTimer::Idle | WaitTimer::Shown => None,
        };

        let toast_expired =
            self.toast_until.is_some_and(|deadline| now >= deadline);
        if toast_expired {
            self.toast_until = None;
        }

        TimerTick {
            next_deadline: [wait_deadline, self.toast_until]
                .into_iter()
                .flatten()
                .min(),
            changed: self.is_waiting() != was_waiting || toast_expired,
        }
    }

    pub fn is_waiting(&self) -> bool {
        matches!(self.wait, WaitTimer::Shown)
    }

    pub fn is_toasting(&self) -> bool {
        self.toast_until.is_some()
    }
}

pub fn show(
    ui: &mut Ui,
    status: &StatusLine<'_>,
    show_status_bar: bool,
) -> Rect {
    use egui::Label;
    use egui::Panel;
    use egui::Sides;

    if show_status_bar {
        Panel::bottom("status").show(ui, |ui| {
            // Sides measures the right end from what it renders, so the
            // facts truncate at the group's true edge. Without
            // `shrink_left()` the group parks beside the facts.
            Sides::new().shrink_left().show(
                ui,
                |ui| {
                    let layout = status_line_layout(ui.style(), status);
                    ui.add(Label::new(layout).truncate());
                },
                |ui| mode_indicators(ui, status),
            );
        });
    }
    ui.available_rect_before_wrap()
}

const JUMP_BACKDROP_DIM_ALPHA: u8 = 150;
const NO_CORNER_RADIUS: f32 = 0.0;
const PROMPT_MARGIN: i8 = 28;
const PROMPT_MIN_WIDTH: f32 = 480.0;
const PROMPT_PREFIX_FONT_SIZE: f32 = 36.0;
const PROMPT_VALUE_FONT_SIZE: f32 = 44.0;
const PROMPT_PREFIX_GAP: f32 = 12.0;

/// Draws the prompt over `page_area`; callers pass the page surface, so
/// the status bar stays outside the dim.
pub fn page_jump(
    context: &Context,
    page_area: Rect,
    digits: &str,
    page_count: usize,
    digits_valid: bool,
) {
    use egui::Align2;
    use egui::Area;
    use egui::Frame;
    use egui::Id;
    use egui::LayerId;
    use egui::Margin;
    use egui::Order;

    let dim_layer =
        LayerId::new(Order::Middle, Id::unique("page-jump-dim"));
    context
        .layer_painter(dim_layer)
        .with_clip_rect(page_area)
        .rect_filled(
            page_area,
            NO_CORNER_RADIUS,
            Color32::from_black_alpha(JUMP_BACKDROP_DIM_ALPHA),
        );

    Area::new(Id::unique("page-jump"))
        .order(Order::Foreground)
        .interactable(false)
        .fade_in(false)
        .fixed_pos(page_area.center())
        .pivot(Align2::CENTER_CENTER)
        .show(context, |ui| {
            Frame::popup(ui.style())
                .inner_margin(Margin::same(PROMPT_MARGIN))
                .show(ui, |ui| {
                    ui.set_min_width(PROMPT_MIN_WIDTH);
                    ui.label(page_jump_text(
                        ui.style(),
                        digits,
                        page_count,
                        digits_valid,
                    ));
                });
        });
}

fn page_jump_text(
    style: &Style,
    digits: &str,
    page_count: usize,
    digits_valid: bool,
) -> LayoutJob {
    use egui::Align;
    use egui::FontId;

    let value_color = if digits.is_empty() {
        style.visuals.weak_text_color()
    } else if digits_valid {
        style.visuals.text_color()
    } else {
        style.visuals.error_fg_color
    };
    let placeholder = digits.is_empty().then(|| format!("1-{page_count}"));
    let value_text = placeholder.as_deref().unwrap_or(digits);

    let mut prefix_format = TextFormat::simple(
        FontId::monospace(PROMPT_PREFIX_FONT_SIZE),
        style.visuals.strong_text_color(),
    );
    prefix_format.valign = Align::Center;

    let mut layout = LayoutJob::default();
    layout.append(":", 0.0, prefix_format);
    layout.append(
        value_text,
        PROMPT_PREFIX_GAP,
        TextFormat::simple(
            FontId::monospace(PROMPT_VALUE_FONT_SIZE),
            value_color,
        ),
    );
    layout
}

/// `egui::Sides` lays out the right side right-to-left, so iterate in
/// reverse.
fn mode_indicators(ui: &mut Ui, status: &StatusLine<'_>) {
    use egui::RichText;
    use strum::VariantArray as _;

    for (mode_index, &mode) in
        ReadingMode::VARIANTS.iter().enumerate().rev()
    {
        let text = RichText::new(mode_label(mode)).monospace();
        let text = if mode == status.mode {
            text.color(mode_color(mode)).strong()
        } else {
            text.weak()
        };
        ui.label(text);
        if mode_index > 0 {
            ui.label(RichText::new("·").weak().monospace());
        }
    }
    if status.show_mode_toast {
        ui.label(RichText::new(status.mode.description()).monospace());
    }
}

fn mode_label(mode: ReadingMode) -> String {
    format!("{mode} ({})", mode.shortcut())
}

const NORMAL_MODE_COLOR: Color32 = Color32::from_rgb(0x87, 0xb3, 0x7d);
const MANGA_MODE_COLOR: Color32 = Color32::from_rgb(0xd2, 0x7b, 0x7b);
const COMIC_MODE_COLOR: Color32 = Color32::from_rgb(0x7b, 0x9f, 0xd2);

fn mode_color(mode: ReadingMode) -> Color32 {
    match mode {
        ReadingMode::Normal => NORMAL_MODE_COLOR,
        ReadingMode::Manga => MANGA_MODE_COLOR,
        ReadingMode::Comic => COMIC_MODE_COLOR,
    }
}

const PAGE_RANGE_COLOR: Color32 = Color32::from_rgb(0xc2, 0x9e, 0x6f);
const FILE_NAME_COLOR: Color32 = Color32::from_rgb(0xa2, 0x92, 0xc5);

fn status_line_layout(
    style: &Style,
    status: &StatusLine<'_>,
) -> LayoutJob {
    use egui::TextStyle;

    let font = TextStyle::Monospace.resolve(style);
    let plain = style.visuals.text_color();
    let section = |layout: &mut LayoutJob, text: &str, color: Color32| {
        layout.append(text, 0.0, TextFormat::simple(font.clone(), color));
    };
    let mut layout = LayoutJob::default();

    if let Some(reading) = &status.reading {
        let page_range = reading.spread.second.map_or_else(
            || format!("{}", reading.spread.first + 1),
            |second| {
                format!("{}-{}", reading.spread.first + 1, second + 1)
            },
        );
        section(&mut layout, &page_range, PAGE_RANGE_COLOR);
        section(&mut layout, &format!("/{}", reading.page_count), plain);
        section(&mut layout, " · ", plain);
    }

    if let Some(parent_dir) = &status.name.parent_dir {
        section(&mut layout, parent_dir, plain);
        section(&mut layout, "/", plain);
    }
    section(&mut layout, &status.name.file, FILE_NAME_COLOR);

    if let Some(reading) = &status.reading {
        if let Some((width, height)) = reading.page_dims {
            section(&mut layout, " · ", plain);
            let dims_text = reading.actual_format.map_or_else(
                || format!("{width}x{height}"),
                |format| format!("{format} {width}x{height}"),
            );
            section(&mut layout, &dims_text, plain);
        }
        section(
            &mut layout,
            &format!(" · {:.0}%", reading.zoom_scale * 100.0),
            plain,
        );
    }
    if status.show_loading {
        section(&mut layout, " · loading…", plain);
    }
    if let Some(fault) = status.fault {
        section(&mut layout, &format!(" · {fault}"), plain);
    }
    layout
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use egui::Shape;

    use super::*;

    #[test]
    fn timers_wait_and_toast_lifecycle() {
        let start = Instant::now();
        let mut timers = Timers::default();

        assert_eq!(
            timers.tick(start, true),
            TimerTick {
                next_deadline: Some(start + WAIT_INDICATOR_GRACE),
                changed: false,
            }
        );
        assert!(!timers.is_waiting());
        assert_eq!(
            timers.tick(start + WAIT_INDICATOR_GRACE, true),
            TimerTick {
                next_deadline: None,
                changed: true,
            }
        );
        assert!(timers.is_waiting());
        assert_eq!(
            timers.tick(start + WAIT_INDICATOR_GRACE * 2, false),
            TimerTick {
                next_deadline: None,
                changed: true,
            }
        );
        assert!(!timers.is_waiting());

        timers.toast(start);
        assert!(timers.is_toasting());
        assert_eq!(
            timers.tick(start, true),
            TimerTick {
                next_deadline: Some(start + WAIT_INDICATOR_GRACE),
                changed: false,
            }
        );
        assert_eq!(
            timers.tick(start, false),
            TimerTick {
                next_deadline: Some(start + MODE_TOAST_DURATION),
                changed: false,
            }
        );
        assert_eq!(
            timers.tick(start + MODE_TOAST_DURATION, false),
            TimerTick {
                next_deadline: None,
                changed: true,
            }
        );
        assert!(!timers.is_toasting());
    }

    fn sample_status() -> StatusLine<'static> {
        StatusLine {
            reading: Some(ReadingStatus {
                page_count: 214,
                spread: Spread {
                    first: 11,
                    second: Some(12),
                },
                page_dims: Some((6000, 4000)),
                actual_format: Some(Format::Jxl),
                zoom_scale: 0.866,
            }),
            mode: ReadingMode::Manga,
            show_mode_toast: false,
            name: PageName {
                parent_dir: Some("b".into()),
                file: "010.png".into(),
            },
            show_loading: false,
            fault: None,
        }
    }

    struct TextItem {
        text: String,
        bounds: Rect,
        elided: bool,
    }

    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "egui's non-text shape variants carry no text items"
    )]
    fn collect_text_items(shape: &Shape, items: &mut Vec<TextItem>) {
        match shape {
            Shape::Text(text_shape) => items.push(TextItem {
                text: text_shape
                    .galley
                    .rows
                    .iter()
                    .map(|row| row.text())
                    .collect::<String>(),
                bounds: text_shape
                    .galley
                    .rect
                    .translate(text_shape.pos.to_vec2()),
                elided: text_shape.galley.elided,
            }),
            Shape::Vec(shapes) => {
                for child_shape in shapes {
                    collect_text_items(child_shape, items);
                }
            }
            _ => {}
        }
    }

    fn render(
        width: f32,
        status: &StatusLine<'_>,
    ) -> (Vec<TextItem>, Rect) {
        use std::mem;

        use egui::Pos2;
        use egui::RawInput;

        let egui_context = Context::default();
        let input = || RawInput {
            screen_rect: Some(Rect::from_min_size(
                Pos2::ZERO,
                egui::vec2(width, SCREEN_HEIGHT),
            )),
            ..Default::default()
        };
        egui_context
            .run_ui(input(), |ui| {
                show(ui, status, true);
            })
            .drop_without_applying_deltas();
        let mut page_area = Rect::NOTHING;
        let mut output = egui_context.run_ui(input(), |ui| {
            page_area = show(ui, status, true);
        });
        let shapes = mem::take(&mut output.shapes);
        output.drop_without_applying_deltas();
        let mut items = Vec::new();
        for clipped in &shapes {
            collect_text_items(&clipped.shape, &mut items);
        }
        (items, page_area)
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

    const PANEL_MARGIN: f32 = 8.0;
    const LAYOUT_TOLERANCE: f32 = 0.51;
    const SCREEN_HEIGHT: f32 = 600.0;
    const WIDE_SCREEN_WIDTH: f32 = 1280.0;
    const NARROW_SCREEN_WIDTH: f32 = 480.0;
    const MIN_CANVAS_HEIGHT: f32 = 400.0;
    const MAX_BAR_HEIGHT_DRIFT: f32 = 8.0;

    #[test]
    fn bar_geometry() {
        let (plain_items, wide_area) =
            render(WIDE_SCREEN_WIDTH, &sample_status());
        let mut with_toast = sample_status();
        with_toast.show_mode_toast = true;
        let (toast_items, _) = render(WIDE_SCREEN_WIDTH, &with_toast);
        let (narrow_items, narrow_area) =
            render(NARROW_SCREEN_WIDTH, &sample_status());

        assert!(wide_area.min.x.abs() < LAYOUT_TOLERANCE);
        assert!(wide_area.min.y.abs() < LAYOUT_TOLERANCE);
        assert!(
            (wide_area.width() - WIDE_SCREEN_WIDTH).abs()
                < LAYOUT_TOLERANCE
        );
        assert!(wide_area.max.y > MIN_CANVAS_HEIGHT);
        assert!(wide_area.max.y < SCREEN_HEIGHT);
        assert!(
            (narrow_area.width() - NARROW_SCREEN_WIDTH).abs()
                < LAYOUT_TOLERANCE
        );
        assert!(
            (narrow_area.max.y - wide_area.max.y).abs()
                < MAX_BAR_HEIGHT_DRIFT
        );

        assert!(
            (find(&plain_items, "12-13/214").bounds.left() - PANEL_MARGIN)
                .abs()
                < LAYOUT_TOLERANCE
        );
        assert_eq!(
            find(&plain_items, "12-13/214").bounds.left(),
            find(&toast_items, "12-13/214").bounds.left()
        );

        for items in [&plain_items, &toast_items] {
            let comic = find(items, "Comic (c)");
            let rightmost = items
                .iter()
                .map(|item| item.bounds.right())
                .fold(f32::MIN, f32::max);
            assert!(
                (rightmost - comic.bounds.right()).abs()
                    < LAYOUT_TOLERANCE
            );
            assert!(
                (comic.bounds.right()
                    - (WIDE_SCREEN_WIDTH - PANEL_MARGIN))
                    .abs()
                    < LAYOUT_TOLERANCE
            );
        }
        assert_eq!(
            find(&plain_items, "Comic (c)").bounds,
            find(&toast_items, "Comic (c)").bounds,
        );

        assert!(
            find(&toast_items, "right to left").bounds.right()
                < find(&toast_items, "Comic (c)").bounds.left()
        );
        assert!(plain_items.iter().all(|item| {
            !item.text.contains(sample_status().mode.description())
        }));

        let left_summary = find(&narrow_items, "12-13");
        assert!(left_summary.elided);
        assert!(left_summary.text.ends_with('…'));
        assert!(
            left_summary.bounds.right()
                <= find(&narrow_items, "Comic (c)").bounds.left()
        );
        assert!(!find(&plain_items, "12-13").elided);
    }
}
