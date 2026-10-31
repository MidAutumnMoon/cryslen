//! Screen: what each seat shows. The commit policy holds a spread
//! change while any of its wanted seats is undecided and changes every
//! seat in one frame, stages the forward spread's textures ahead of a
//! turn, and owns the errors no page owns.

use std::sync::Arc;

use crate::decode::Decoded;
use crate::render::PageTextures;
use crate::view;
use crate::view::Seat;
use crate::view::Spread;
use crate::view::ViewMode;
use crate::worker::SeatView;
use crate::worker::SpreadView;
use crate::worker::Worker;

enum SeatStep<'view> {
    Show { page: usize, decoded: &'view Arc<Decoded> },
    Hold,
    Clear,
}

struct ScreenPlan<'view> {
    steps: [SeatStep<'view>; 2],
    shown_after: Option<Spread>,
    committed: bool,
    stage: Option<(usize, &'view Arc<Decoded>)>,
}

fn seat_step(seat: Option<(usize, &SeatView)>, bound: Option<usize>) -> SeatStep<'_> {
    match seat {
        Some((page, SeatView::Ready(decoded))) => SeatStep::Show { page, decoded },
        Some((page, SeatView::Pending)) if bound.is_none_or(|bound_page| bound_page == page) => {
            SeatStep::Hold
        }
        None | Some((_, SeatView::Failed(_) | SeatView::Pending)) => SeatStep::Clear,
    }
}

/// Before anything has been shown, ready seats appear as they arrive
/// instead of waiting for the whole spread. `bound[i]` is the page seat
/// i currently shows.
fn plan_screen(
    shown: Option<Spread>,
    wanted: Spread,
    mode: ViewMode,
    seat_views: [Option<&SeatView>; 2],
    bound: [Option<usize>; 2],
) -> ScreenPlan<'_> {
    let [wanted_left, wanted_right] = wanted.seats(mode);
    let [view_left, view_right] = seat_views;
    let [bound_left, bound_right] = bound;
    let seats = [wanted_left.zip(view_left), wanted_right.zip(view_right)];
    let [seat_left, seat_right] = seats;
    let undecided =
        seats.into_iter().any(|seat| matches!(seat, Some((_, SeatView::Pending))));
    if undecided && shown.is_some_and(|shown| shown != wanted) {
        let stage = seats.into_iter().find_map(|seat| match seat {
            Some((page, SeatView::Ready(decoded))) => Some((page, decoded)),
            _ => None,
        });
        return ScreenPlan {
            steps: [SeatStep::Hold, SeatStep::Hold],
            shown_after: shown,
            committed: false,
            stage,
        };
    }
    let steps = [seat_step(seat_left, bound_left), seat_step(seat_right, bound_right)];
    #[expect(
        clippy::rest_pattern_accessible_field,
        reason = "the matched fields are unused here, so naming them would trip unneeded_field_pattern"
    )]
    let any_shown = steps.iter().any(|step| matches!(step, SeatStep::Show { .. }));
    // An all-failed first spread shows nothing, so `shown` stays None.
    let shown_after = (shown.is_some() || any_shown).then_some(wanted);
    ScreenPlan {
        steps,
        shown_after,
        committed: shown_after != shown,
        stage: None,
    }
}

/// The spread one forward turn lands on: the only neighbor the frame
/// stages and watches, because reading runs forward.
fn forward_spread(spread: Spread, mode: ViewMode, len: usize) -> Spread {
    let next = view::advance(spread.first, true, mode, len);
    view::spread_for(next, mode, len)
}

/// Whether `page`'s decision changes the screen: a wanted seat's
/// arrival commits or ends a hold, and a next-spread page becomes
/// texture-staged. Everything else is prefetch the screen never shows.
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is the crate's visibility style for crate-wide items, even inside private modules"
)]
pub(crate) fn page_watched(page: usize, spread: Spread, mode: ViewMode, len: usize) -> bool {
    spread.contains(page) || forward_spread(spread, mode, len).contains(page)
}

#[derive(Clone, Copy)]
struct SeatShow {
    page: usize,
    dims: (u32, u32),
}

/// The hold and commit policy and the seat state it tracks have one
/// owner here, so they cannot drift apart.
#[derive(Default)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is the crate's visibility style for crate-wide items, even inside private modules"
)]
pub(crate) struct Screen {
    /// The last spread the plan committed; None before the first
    /// commit (cold start).
    shown: Option<Spread>,
    seats: [Option<SeatShow>; 2],
    /// A texture upload the GPU refused. A binding frame rewrites this:
    /// a refused bind retries every frame while its page stays wanted,
    /// so the error holds exactly that long.
    upload_error: Option<String>,
    /// A failure the app noted (a book switch that failed). No frame
    /// retries it, so it holds until a successful switch clears the
    /// book.
    noted_error: Option<String>,
}

impl Screen {
    /// Present `wanted` through `pages` per the plan. Returns whether
    /// the frame committed a new spread, which re-arms the caller's fit.
    pub(crate) fn display(
        &mut self,
        pages: &mut PageTextures,
        wanted: Spread,
        mode: ViewMode,
        snapshot: &SpreadView,
    ) -> bool {
        let bound = self.seats.each_ref().map(|seat| seat.map(|show| show.page));
        let plan = plan_screen(self.shown, wanted, mode, snapshot.seats(mode), bound);
        // The frame's binds are one outcome: a refused upload on either
        // seat must survive the partner's success, and a frame that
        // bound cleanly clears the upload error. A hold frame leaves it.
        let mut bind_error = None;
        let mut bound_cleanly = false;
        for ((seat, shown), step) in Seat::ALL
            .into_iter()
            .zip(self.seats.each_mut())
            .zip(plan.steps)
        {
            match step {
                SeatStep::Show { page, decoded } => {
                    match pages.bind(seat, page, decoded) {
                        Ok(()) => bound_cleanly = true,
                        Err(error) => bind_error = Some(error.to_string()),
                    }
                    // The seat keeps its page and dims even on a refused
                    // upload, so the layout and hold policy stay stable
                    // while the next frame retries the bind.
                    *shown = Some(SeatShow { page, dims: (decoded.width, decoded.height) });
                }
                SeatStep::Hold => {}
                SeatStep::Clear => {
                    *shown = None;
                    pages.clear(seat);
                }
            }
        }
        if bind_error.is_some() || bound_cleanly {
            self.upload_error = bind_error;
        }
        if let Some((page, decoded)) = plan.stage {
            self.stage(pages, page, decoded);
        }
        self.shown = plan.shown_after;
        plan.committed
    }

    /// Upload the forward spread's decided pages ahead of the turn, so
    /// an ordinary commit is a pure rebind: the multi-megabyte upload
    /// lands on this frame, which redraws anyway, instead of the turn's
    /// first visible one.
    pub(crate) fn stage_forward(
        &mut self,
        pages: &mut PageTextures,
        current: Spread,
        mode: ViewMode,
        len: usize,
        worker: &Worker,
    ) {
        let next = forward_spread(current, mode, len);
        if next == current {
            return;
        }
        let snapshot = worker.snapshot(&next);
        for (page, seat) in snapshot.reading_order(next) {
            if let SeatView::Ready(decoded) = seat {
                self.stage(pages, page, decoded);
            }
        }
    }

    /// Upload `page`'s textures without binding a seat. A refusal
    /// surfaces in the status line and repeats while staging retries
    /// each frame.
    fn stage(&mut self, pages: &mut PageTextures, page: usize, decoded: &Arc<Decoded>) {
        if let Err(error) = pages.stage(page, decoded) {
            self.upload_error = Some(error.to_string());
        }
    }

    pub(crate) fn seat_dims(&self) -> [Option<(u32, u32)>; 2] {
        self.seats.each_ref().map(|seat| seat.map(|show| show.dims))
    }

    /// An error the status line must show that no page owns.
    pub(crate) fn error(&self) -> Option<&str> {
        self.noted_error.as_deref().or(self.upload_error.as_deref())
    }

    pub(crate) fn note_error(&mut self, message: String) {
        self.noted_error = Some(message);
    }

    /// Drop every shown fact.
    pub(crate) fn clear_book(&mut self) {
        self.shown = None;
        self.seats = [None, None];
        self.upload_error = None;
        self.noted_error = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::ready_seat;
    use crate::view::ViewMode;

    #[test]
    fn plan_screen_table() {
        #[derive(Debug, PartialEq)]
        enum StepKind {
            Show,
            Hold,
            Clear,
        }
        type StepFacts = (StepKind, Option<(usize, (u32, u32))>);
        fn show(page: usize, dims: (u32, u32)) -> StepFacts {
            (StepKind::Show, Some((page, dims)))
        }
        const HOLD: StepFacts = (StepKind::Hold, None);
        const CLEAR: StepFacts = (StepKind::Clear, None);
        fn step_facts(step: &SeatStep<'_>) -> StepFacts {
            match step {
                SeatStep::Show { page, decoded } => {
                    (StepKind::Show, Some((*page, (decoded.width, decoded.height))))
                }
                SeatStep::Hold => (StepKind::Hold, None),
                SeatStep::Clear => (StepKind::Clear, None),
            }
        }
        struct Row {
            name: &'static str,
            shown: Option<Spread>,
            wanted: Spread,
            mode: ViewMode,
            views: [Option<SeatView>; 2],
            /// The page each seat currently shows, [left, right].
            bound: [Option<usize>; 2],
            steps: [StepFacts; 2],
            shown_after: Option<Spread>,
            committed: bool,
            stage: Option<(usize, (u32, u32))>,
        }
        let old = Spread { first: 3, second: Some(4) };
        let pair = Spread { first: 5, second: Some(6) };
        let failed = || Some(SeatView::Failed("x".into()));
        let pending = || Some(SeatView::Pending);
        // Manga seats a pair as [second, first]: left = page 6, right = 5.
        // Comic/Normal seat [first, second]: left = first.
        // `old` and `pair` were committed under Manga: bound [second, first].
        let rows = [
            Row {
                name: "a settled pair commits both seats in one frame (rtl seating)",
                shown: Some(old), wanted: pair, mode: ViewMode::Manga,
                views: [Some(ready_seat(600, 700)), Some(ready_seat(500, 700))],
                bound: [Some(4), Some(3)],
                steps: [show(6, (600, 700)), show(5, (500, 700))],
                shown_after: Some(pair), committed: true, stage: None,
            },
            Row {
                name: "an undecided left page holds every seat and stages the partner",
                shown: Some(old), wanted: pair, mode: ViewMode::Manga,
                views: [pending(), Some(ready_seat(500, 700))],
                bound: [Some(4), Some(3)],
                steps: [HOLD, HOLD],
                shown_after: Some(old), committed: false, stage: Some((5, (500, 700))),
            },
            Row {
                name: "an undecided right page holds every seat and stages the partner",
                shown: Some(old), wanted: pair, mode: ViewMode::Manga,
                views: [Some(ready_seat(600, 700)), pending()],
                bound: [Some(4), Some(3)],
                steps: [HOLD, HOLD],
                shown_after: Some(old), committed: false, stage: Some((6, (600, 700))),
            },
            Row {
                name: "a failed partner commits: the seat clears",
                shown: Some(old), wanted: pair, mode: ViewMode::Manga,
                views: [failed(), Some(ready_seat(500, 700))],
                bound: [Some(4), Some(3)],
                steps: [CLEAR, show(5, (500, 700))],
                shown_after: Some(pair), committed: true, stage: None,
            },
            Row {
                name: "both failed commits to background mid-session",
                shown: Some(old), wanted: pair, mode: ViewMode::Manga,
                views: [failed(), failed()],
                bound: [Some(4), Some(3)],
                steps: [CLEAR, CLEAR],
                shown_after: Some(pair), committed: true, stage: None,
            },
            Row {
                name: "a pending page behind a failure still holds",
                shown: Some(old), wanted: pair, mode: ViewMode::Manga,
                views: [failed(), pending()],
                bound: [Some(4), Some(3)],
                steps: [HOLD, HOLD],
                shown_after: Some(old), committed: false, stage: None,
            },
            Row {
                name: "the shown spread re-read as pending holds invisibly",
                shown: Some(pair), wanted: pair, mode: ViewMode::Manga,
                views: [pending(), Some(ready_seat(500, 700))],
                bound: [Some(6), Some(5)],
                steps: [HOLD, show(5, (500, 700))],
                shown_after: Some(pair), committed: false, stage: None,
            },
            Row {
                name: "a seat left on the wrong page by a mid-hold flip clears",
                shown: Some(pair), wanted: pair, mode: ViewMode::Manga,
                views: [pending(), Some(ready_seat(500, 700))],
                // bound is pair's Comic arrangement: left = 5, right = 6.
                bound: [Some(5), Some(6)],
                steps: [CLEAR, show(5, (500, 700))],
                shown_after: Some(pair), committed: false, stage: None,
            },
            Row {
                name: "a lone Manga page parks right and clears left",
                shown: Some(old), wanted: Spread::lone(7), mode: ViewMode::Manga,
                views: [None, Some(ready_seat(700, 700))],
                bound: [Some(4), Some(3)],
                steps: [CLEAR, show(7, (700, 700))],
                shown_after: Some(Spread::lone(7)), committed: true, stage: None,
            },
            Row {
                name: "ltr seating binds the first page left",
                shown: Some(old), wanted: pair, mode: ViewMode::Comic,
                views: [Some(ready_seat(500, 700)), Some(ready_seat(600, 700))],
                bound: [Some(4), Some(3)],
                steps: [show(5, (500, 700)), show(6, (600, 700))],
                shown_after: Some(pair), committed: true, stage: None,
            },
            Row {
                name: "a Normal turn holds the unwanted seat until commit",
                shown: Some(Spread::lone(2)), wanted: Spread::lone(3),
                mode: ViewMode::Normal,
                views: [pending(), None],
                bound: [Some(2), None],
                steps: [HOLD, HOLD],
                shown_after: Some(Spread::lone(2)), committed: false, stage: None,
            },
            Row {
                name: "a Normal commit clears the unwanted seat in the swap",
                shown: Some(Spread::lone(2)), wanted: Spread::lone(3),
                mode: ViewMode::Normal,
                views: [Some(ready_seat(300, 400)), None],
                bound: [Some(2), None],
                steps: [show(3, (300, 400)), CLEAR],
                shown_after: Some(Spread::lone(3)), committed: true, stage: None,
            },
            Row {
                name: "cold start with nothing decided keeps nothing",
                shown: None, wanted: pair, mode: ViewMode::Manga,
                views: [pending(), pending()],
                bound: [None, None],
                steps: [HOLD, HOLD],
                shown_after: None, committed: false, stage: None,
            },
            Row {
                name: "cold start shows a decided seat at once",
                shown: None, wanted: pair, mode: ViewMode::Manga,
                views: [Some(ready_seat(600, 700)), pending()],
                bound: [None, None],
                steps: [show(6, (600, 700)), HOLD],
                shown_after: Some(pair), committed: true, stage: None,
            },
            Row {
                name: "cold start survives an all-failed first spread",
                shown: None, wanted: pair, mode: ViewMode::Manga,
                views: [failed(), failed()],
                bound: [None, None],
                steps: [CLEAR, CLEAR],
                shown_after: None, committed: false, stage: None,
            },
            Row {
                name: "cold start mixes a cleared failure with a shown page",
                shown: None, wanted: pair, mode: ViewMode::Manga,
                views: [failed(), Some(ready_seat(500, 700))],
                bound: [None, None],
                steps: [CLEAR, show(5, (500, 700))],
                shown_after: Some(pair), committed: true, stage: None,
            },
            Row {
                name: "steady state re-shows the same seats",
                shown: Some(pair), wanted: pair, mode: ViewMode::Manga,
                views: [Some(ready_seat(600, 700)), Some(ready_seat(500, 700))],
                bound: [Some(6), Some(5)],
                steps: [show(6, (600, 700)), show(5, (500, 700))],
                shown_after: Some(pair), committed: false, stage: None,
            },
            Row {
                name: "a direction flip on a decided spread re-mirrors at once",
                shown: Some(pair), wanted: pair, mode: ViewMode::Comic,
                views: [Some(ready_seat(500, 700)), Some(ready_seat(600, 700))],
                bound: [Some(6), Some(5)],
                steps: [show(5, (500, 700)), show(6, (600, 700))],
                shown_after: Some(pair), committed: false, stage: None,
            },
            Row {
                name: "a direction flip mid-hold keeps the old arrangement",
                shown: Some(old), wanted: pair, mode: ViewMode::Comic,
                views: [Some(ready_seat(500, 700)), pending()],
                bound: [Some(4), Some(3)],
                steps: [HOLD, HOLD],
                shown_after: Some(old), committed: false, stage: Some((5, (500, 700))),
            },
        ];
        for Row { name, shown, wanted, mode, views, bound, steps, shown_after, committed, stage } in rows {
            let views = views.each_ref().map(Option::as_ref);
            let plan = plan_screen(shown, wanted, mode, views, bound);
            assert_eq!(plan.steps.each_ref().map(step_facts), steps, "{name}: steps");
            assert_eq!(plan.shown_after, shown_after, "{name}: shown");
            assert_eq!(plan.committed, committed, "{name}: committed");
            assert_eq!(
                plan.stage.map(|(page, decoded)| (page, (decoded.width, decoded.height))),
                stage,
                "{name}: stage"
            );
        }
    }
}
