//! Status overlay: one egui bottom panel, data in / text out.

use std::fmt;
use std::iter;
use std::time::Duration;
use std::time::Instant;

use egui::text::LayoutJob;

use crate::browse::PageInfo;
use crate::browse::ReadingMode;
use crate::browse::Spread;
use crate::catalog::PageName;
use crate::decode::Format;
use crate::fault::BookFault;
use crate::fault::PageFault;

pub struct StatusLine<'fault> {
    pub page_count: usize,
    pub spread: Spread,
    pub mode: ReadingMode,
    pub toast: bool,
    pub name: PageName,
    pub page: Option<PageStatus>,
    /// Placement scale, screen px per image px.
    pub zoom: f32,
    /// The wait flag, once a waiting spread has outlived the grace.
    pub waiting: Option<Wait>,
    pub fault: Option<StatusFault<'fault>>,
}

/// What the status line blames: a page's own failure (decode, source, or
/// a texture the GPU refused), or the book's.
pub enum StatusFault<'fault> {
    Page(&'fault PageFault),
    Book(&'fault BookFault),
}

impl fmt::Display for StatusFault<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Page(fault) => fault.fmt(f),
            Self::Book(fault) => fault.fmt(f),
        }
    }
}

/// The one fault the line shows: the page's own failure first, then the
/// book's. Each layer's cell keeps its own retention; this is only the
/// ordering.
pub fn fault_to_show<'fault>(
    page: Option<&'fault PageFault>,
    book: Option<&'fault BookFault>,
) -> Option<StatusFault<'fault>> {
    page.map(StatusFault::Page)
        .or_else(|| book.map(StatusFault::Book))
}

/// Status-line timers, pure in the clock the caller passes.
#[derive(Default)]
pub struct Timers {
    waiting_since: Option<Instant>,
    toast_since: Option<Instant>,
}

const WAIT_GRACE: Duration = Duration::from_millis(150);

const MODE_TOAST: Duration = Duration::from_millis(1500);

impl Timers {
    pub fn toast(&mut self, now: Instant) {
        self.toast_since = Some(now);
    }

    /// Track the wait flag and return the grace's remaining delay for the
    /// parked loop to arm.
    pub fn tick_wait(
        &mut self,
        now: Instant,
        waiting: bool,
    ) -> Option<Duration> {
        self.waiting_since =
            waiting.then(|| self.waiting_since.unwrap_or(now));
        // The checked subtraction is the grace boundary itself: some
        // of the grace remains, or none does.
        let since = self.waiting_since?;
        WAIT_GRACE.checked_sub(now - since)
    }

    /// The wait word once the grace has elapsed; `unpacking` runs only
    /// when the word will show.
    pub fn wait_word(
        &self,
        now: Instant,
        unpacking: impl FnOnce() -> bool,
    ) -> Option<Wait> {
        let since = self.waiting_since?;
        if now - since < WAIT_GRACE {
            return None;
        }
        Some(if unpacking() {
            Wait::Unpacking
        } else {
            Wait::Decoding
        })
    }

    /// Expire the toast and return the delay until it does.
    pub fn tick_toast(&mut self, now: Instant) -> Option<Duration> {
        self.toast_since =
            self.toast_since.filter(|&since| now - since < MODE_TOAST);
        self.toast_since
            .map(|since| MODE_TOAST.saturating_sub(now - since))
    }

    pub fn toasting(&self) -> bool {
        self.toast_since.is_some()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wait {
    Unpacking,
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

/// The status line's view of one spread, in page order.
pub struct SpreadStatus<'fault> {
    pub waiting: bool,
    pub fault: Option<&'fault PageFault>,
    pub page: Option<(usize, PageInfo)>,
    /// The page the line names: the failed one if any page failed, else
    /// the first decoded page, else the spread's first page.
    pub named_page: usize,
}

pub fn summarize_spread<'fault>(
    spread: Spread,
    state: impl Fn(usize) -> (Option<PageInfo>, Option<&'fault PageFault>),
) -> SpreadStatus<'fault> {
    let mut has_info = false;
    let mut has_fault = false;
    let mut failure: Option<(usize, &'fault PageFault)> = None;
    let mut facts = None;
    let mut first_decoded = None;
    for page in iter::once(spread.first).chain(spread.second) {
        let (info, fault) = state(page);
        has_info |= info.is_some();
        has_fault |= fault.is_some();
        if failure.is_some() {
            continue;
        }
        if let Some(fault) = fault {
            failure = Some((page, fault));
            continue;
        }
        if let Some(info) = info {
            first_decoded.get_or_insert(page);
            facts.get_or_insert((page, info));
        }
    }
    let (named_page, fault) = match failure {
        Some((page, fault)) => (page, Some(fault)),
        None => (first_decoded.unwrap_or(spread.first), None),
    };
    SpreadStatus {
        waiting: !has_info && !has_fault,
        fault,
        page: facts,
        named_page,
    }
}

pub struct PageStatus {
    /// The page's own dims and decoded format.
    pub info: PageInfo,
    /// Format the filename's extension claims. Listed pages always
    /// carry one.
    pub claimed: Option<Format>,
}

/// Render the status bar into `ui` and return the content rect: what
/// the chrome leaves, the only legal page viewport. A hidden
/// bar (`fullscreen`) leaves the whole window as content.
pub fn show(
    ui: &mut egui::Ui,
    status: &StatusLine<'_>,
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
fn right_end(ui: &mut egui::Ui, status: &StatusLine<'_>) {
    for (mode_index, &mode) in ReadingMode::ALL.iter().enumerate().rev() {
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

fn group_label(mode: ReadingMode) -> String {
    format!("{} ({})", mode.label(), mode.shortcut())
}

fn mode_color(mode: ReadingMode) -> egui::Color32 {
    match mode {
        ReadingMode::Normal => egui::Color32::from_rgb(0x87, 0xb3, 0x7d),
        ReadingMode::Manga => egui::Color32::from_rgb(0xd2, 0x7b, 0x7b),
        ReadingMode::Comic => egui::Color32::from_rgb(0x7b, 0x9f, 0xd2),
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
fn line(style: &egui::Style, status: &StatusLine<'_>) -> LayoutJob {
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
    if let Some(parent_dir) = &status.name.parent_dir {
        section(&mut job, parent_dir, plain);
        section(&mut job, "/", plain);
    }
    section(&mut job, &status.name.file, NAME_COLOR);

    if let Some(facts) = &status.page {
        section(&mut job, " · ", plain);
        let (width, height) = facts.info.dims;
        let dims_text = if facts.claimed == Some(facts.info.format) {
            format!("{width}x{height}")
        } else {
            format!("{} {width}x{height}", facts.info.format.label())
        };
        section(&mut job, &dims_text, plain);
    }
    section(&mut job, &format!(" · {:.0}%", status.zoom * 100.0), plain);
    if let Some(wait) = status.waiting {
        section(&mut job, &format!(" · {}", wait.label()), plain);
    }
    if let Some(fault) = &status.fault {
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
    use super::*;

    #[test]
    fn timers_wait_and_toast_lifecycle() {
        let start = Instant::now();
        let mut timers = Timers::default();

        // A waiting spread arms the grace; the word appears only after
        // it elapses, and mid-grace the wake is what remains of it.
        assert_eq!(timers.tick_wait(start, true), Some(WAIT_GRACE));
        assert_eq!(
            timers.tick_wait(start + Duration::from_millis(100), true),
            Some(Duration::from_millis(50))
        );
        let almost_due = (start + WAIT_GRACE)
            .checked_sub(Duration::from_millis(1))
            .expect("time moves forward");
        assert_eq!(
            timers.wait_word(almost_due, || false),
            None,
            "the word appears only once the grace has fully elapsed"
        );
        // The grace boundary itself is due: the word shows and the
        // wake is immediate.
        assert_eq!(
            timers.tick_wait(start + WAIT_GRACE, true),
            Some(Duration::ZERO)
        );
        assert_eq!(
            timers.wait_word(start + WAIT_GRACE, || false),
            Some(Wait::Decoding)
        );
        assert_eq!(
            timers.wait_word(start + WAIT_GRACE * 2, || true),
            Some(Wait::Unpacking)
        );
        // A decided spread clears the wait and disarms its wake.
        assert_eq!(timers.tick_wait(start + WAIT_GRACE * 2, false), None);
        assert_eq!(
            timers.wait_word(start + WAIT_GRACE * 3, || false),
            None
        );

        // The toast expires on its own schedule.
        timers.toast(start);
        assert!(timers.toasting());
        let almost_expired = (start + MODE_TOAST)
            .checked_sub(Duration::from_millis(1))
            .expect("time moves forward");
        assert_eq!(
            timers.tick_toast(almost_expired),
            Some(Duration::from_millis(1))
        );
        assert!(timers.toasting());
        assert_eq!(timers.tick_toast(start + MODE_TOAST), None);
        assert!(!timers.toasting());
    }

    #[test]
    fn the_line_blames_the_page_before_the_book() {
        let page_fault = PageFault::new("page");
        let book_fault = BookFault::Timeout;
        assert!(fault_to_show(None, None).is_none());
        assert!(matches!(
            fault_to_show(None, Some(&book_fault)),
            Some(StatusFault::Book(_))
        ));
        assert!(matches!(
            fault_to_show(Some(&page_fault), None),
            Some(StatusFault::Page(_))
        ));
        assert!(matches!(
            fault_to_show(Some(&page_fault), Some(&book_fault)),
            Some(StatusFault::Page(_))
        ));
    }

    fn status() -> StatusLine<'static> {
        StatusLine {
            page_count: 214,
            spread: Spread {
                first: 11,
                second: Some(12),
            },
            mode: ReadingMode::Manga,
            toast: false,
            name: PageName {
                parent_dir: Some("b".into()),
                file: "010.png".into(),
            },
            // The name claims png and the page decodes as jxl, so the
            // format label prefixes the dims.
            page: Some(PageStatus {
                info: PageInfo {
                    dims: (6000, 4000),
                    format: Format::Jxl,
                },
                claimed: Some(Format::Png),
            }),
            zoom: 0.866,
            waiting: None,
            fault: None,
        }
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

    /// Render the bar headlessly on a `width`-wide screen; egui panel sizing
    /// settles one frame late. Only the texture upload needs a GPU.
    fn render(
        width: f32,
        status: &StatusLine<'_>,
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
