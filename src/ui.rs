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

use crate::browse::ReadingMode;
use crate::browse::Spread;
use crate::catalog::PageName;
use crate::decode::Format;

pub struct ReadingStatus {
    pub page_count: usize,
    pub spread: Spread,
    pub page_dims: Option<(u32, u32)>,
    /// Set only when the decoded format differs from the filename extension.
    pub actual_format: Option<Format>,
    /// Placement scale, screen px per image px.
    pub zoom: f64,
}

pub struct StatusLine<'fault> {
    pub reading: Option<ReadingStatus>,
    pub mode: ReadingMode,
    pub toast: bool,
    pub name: PageName,
    /// Whether to show `loading…`.
    pub waiting: bool,
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

const WAIT_GRACE: Duration = Duration::from_millis(150);

const MODE_TOAST: Duration = Duration::from_millis(1500);

impl Timers {
    pub fn toast(&mut self, now: Instant) {
        self.toast_until = Some(now + MODE_TOAST);
    }

    pub fn tick(&mut self, now: Instant, waiting: bool) -> TimerTick {
        let was_waiting = self.is_waiting();
        self.wait = if waiting {
            match self.wait {
                WaitTimer::Idle => WaitTimer::Grace(now + WAIT_GRACE),
                WaitTimer::Grace(due) if now >= due => WaitTimer::Shown,
                state @ (WaitTimer::Grace(_) | WaitTimer::Shown) => state,
            }
        } else {
            WaitTimer::Idle
        };
        let wait_deadline = match self.wait {
            WaitTimer::Grace(due) => Some(due),
            WaitTimer::Idle | WaitTimer::Shown => None,
        };

        let toast_expired = self.toast_until.is_some_and(|due| now >= due);
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
                    let job = line(ui.style(), status);
                    ui.add(Label::new(job).truncate());
                },
                |ui| right_end(ui, status),
            );
        });
    }
    ui.available_rect_before_wrap()
}

const PAGE_DIM_ALPHA: u8 = 150;
const PROMPT_MARGIN: i8 = 28;
const PROMPT_MIN_WIDTH: f32 = 480.0;
const PROMPT_PREFIX_SIZE: f32 = 36.0;
const PROMPT_VALUE_SIZE: f32 = 44.0;
const PROMPT_PREFIX_GAP: f32 = 12.0;

/// Draws the prompt over `page_rect`; callers pass the page surface, so
/// the status bar stays outside the dim.
pub fn page_jump(
    ctx: &Context,
    page_rect: Rect,
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

    // The prompt frame is Foreground; keep the dim one layer below it.
    let dim_layer =
        LayerId::new(Order::Middle, Id::unique("page-jump-dim"));
    ctx.layer_painter(dim_layer)
        .with_clip_rect(page_rect)
        .rect_filled(
            page_rect,
            0.0,
            Color32::from_black_alpha(PAGE_DIM_ALPHA),
        );

    Area::new(Id::unique("page-jump"))
        .order(Order::Foreground)
        .interactable(false)
        .fade_in(false)
        .fixed_pos(page_rect.center())
        .pivot(Align2::CENTER_CENTER)
        .show(ctx, |ui| {
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
        FontId::monospace(PROMPT_PREFIX_SIZE),
        style.visuals.strong_text_color(),
    );
    prefix_format.valign = Align::Center;

    let mut job = LayoutJob::default();
    job.append(":", 0.0, prefix_format);
    job.append(
        value_text,
        PROMPT_PREFIX_GAP,
        TextFormat::simple(
            FontId::monospace(PROMPT_VALUE_SIZE),
            value_color,
        ),
    );
    job
}

/// `egui::Sides` lays out the right side right-to-left, so iterate in
/// reverse.
fn right_end(ui: &mut Ui, status: &StatusLine<'_>) {
    use egui::RichText;
    use strum::VariantArray as _;

    for (mode_index, &mode) in
        ReadingMode::VARIANTS.iter().enumerate().rev()
    {
        let text = RichText::new(group_label(mode)).monospace();
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
    if status.toast {
        ui.label(RichText::new(status.mode.effect()).monospace());
    }
}

fn group_label(mode: ReadingMode) -> String {
    format!("{mode} ({})", mode.shortcut())
}

fn mode_color(mode: ReadingMode) -> Color32 {
    match mode {
        ReadingMode::Normal => Color32::from_rgb(0x87, 0xb3, 0x7d),
        ReadingMode::Manga => Color32::from_rgb(0xd2, 0x7b, 0x7b),
        ReadingMode::Comic => Color32::from_rgb(0x7b, 0x9f, 0xd2),
    }
}

/// Muted hue for the spread range, quieter than the mode colors so the
/// group keeps the bar's strongest accent.
const HEAD_COLOR: Color32 = Color32::from_rgb(0xc2, 0x9e, 0x6f);

const NAME_COLOR: Color32 = Color32::from_rgb(0xa2, 0x92, 0xc5);

/// The left-side facts as one job of colored sections, separate so they
/// are directly testable. One job, one label: the whole line keeps
/// eliding together on narrow screens.
fn line(style: &Style, status: &StatusLine<'_>) -> LayoutJob {
    use egui::TextStyle;

    let font = TextStyle::Monospace.resolve(style);
    let plain = style.visuals.text_color();
    let section = |job: &mut LayoutJob, text: &str, color: Color32| {
        job.append(text, 0.0, TextFormat::simple(font.clone(), color));
    };
    let mut job = LayoutJob::default();

    if let Some(reading) = &status.reading {
        let head = reading.spread.second.map_or_else(
            || format!("{}", reading.spread.first + 1),
            |second| {
                format!("{}-{}", reading.spread.first + 1, second + 1)
            },
        );
        section(&mut job, &head, HEAD_COLOR);
        section(&mut job, &format!("/{}", reading.page_count), plain);
        section(&mut job, " · ", plain);
    }

    if let Some(parent_dir) = &status.name.parent_dir {
        section(&mut job, parent_dir, plain);
        section(&mut job, "/", plain);
    }
    section(&mut job, &status.name.file, NAME_COLOR);

    if let Some(reading) = &status.reading {
        if let Some((width, height)) = reading.page_dims {
            section(&mut job, " · ", plain);
            let dims_text = reading.actual_format.map_or_else(
                || format!("{width}x{height}"),
                |format| format!("{format} {width}x{height}"),
            );
            section(&mut job, &dims_text, plain);
        }
        section(
            &mut job,
            &format!(" · {:.0}%", reading.zoom * 100.0),
            plain,
        );
    }
    if status.waiting {
        section(&mut job, " · loading…", plain);
    }
    if let Some(fault) = status.fault {
        section(&mut job, &format!(" · {fault}"), plain);
    }
    job
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
                next_deadline: Some(start + WAIT_GRACE),
                changed: false,
            }
        );
        assert!(!timers.is_waiting());
        assert_eq!(
            timers.tick(start + WAIT_GRACE, true),
            TimerTick {
                next_deadline: None,
                changed: true,
            }
        );
        assert!(timers.is_waiting());
        assert_eq!(
            timers.tick(start + WAIT_GRACE * 2, false),
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
                next_deadline: Some(start + WAIT_GRACE),
                changed: false,
            }
        );
        assert_eq!(
            timers.tick(start, false),
            TimerTick {
                next_deadline: Some(start + MODE_TOAST),
                changed: false,
            }
        );
        assert_eq!(
            timers.tick(start + MODE_TOAST, false),
            TimerTick {
                next_deadline: None,
                changed: true,
            }
        );
        assert!(!timers.is_toasting());
    }

    fn status() -> StatusLine<'static> {
        StatusLine {
            reading: Some(ReadingStatus {
                page_count: 214,
                spread: Spread {
                    first: 11,
                    second: Some(12),
                },
                // The name claims png and the page decodes as jxl, so the
                // format label prefixes the dims.
                page_dims: Some((6000, 4000)),
                actual_format: Some(Format::Jxl),
                zoom: 0.866,
            }),
            mode: ReadingMode::Manga,
            toast: false,
            name: PageName {
                parent_dir: Some("b".into()),
                file: "010.png".into(),
            },
            waiting: false,
            fault: None,
        }
    }

    struct TextItem {
        text: String,
        rect: Rect,
        elided: bool,
    }

    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "egui's non-text shape variants carry no text items"
    )]
    fn walk(shape: &Shape, out: &mut Vec<TextItem>) {
        match shape {
            Shape::Text(text_shape) => out.push(TextItem {
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
            Shape::Vec(shapes) => {
                for child_shape in shapes {
                    walk(child_shape, out);
                }
            }
            _ => {}
        }
    }

    /// Render the bar headlessly on a `width`-wide screen; egui panel sizing
    /// settles one frame late. Only the texture upload needs a GPU.
    fn render(
        width: f32,
        status: &StatusLine<'_>,
    ) -> (Vec<TextItem>, Rect) {
        use std::mem;

        use egui::Pos2;
        use egui::RawInput;

        let ctx = Context::default();
        let input = || RawInput {
            screen_rect: Some(Rect::from_min_size(
                Pos2::ZERO,
                egui::vec2(width, SCREEN_HEIGHT),
            )),
            ..Default::default()
        };
        ctx.run_ui(input(), |ui| {
            show(ui, status, true);
        })
        .drop_without_applying_deltas();
        let mut content = Rect::NOTHING;
        let mut out = ctx.run_ui(input(), |ui| {
            content = show(ui, status, true);
        });
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
        assert!(
            plain
                .iter()
                .all(|item| !item.text.contains(status().mode.effect()))
        );

        let facts = find(&narrow, "12-13");
        assert!(facts.elided, "facts must elide on a narrow screen");
        assert!(facts.text.ends_with('…'));
        assert!(
            facts.rect.right() <= find(&narrow, "Comic (c)").rect.left()
        );
        assert!(!find(&plain, "12-13").elided);
    }
}
