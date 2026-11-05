//! Screen: what each seat shows. The commit policy holds a spread
//! change while any wanted seat is undecided (unless nothing has shown
//! yet), then changes every seat in one frame; it stages the forward
//! spread's textures ahead of a turn and owns the upload error that no
//! page owns.

use std::sync::Arc;

use crate::decode::Decoded;
use crate::fault::PageError;
use crate::fault::TextureError;
use crate::render::PageTextures;
use crate::view;
use crate::view::PageCount;
use crate::view::Spread;
use crate::view::ViewMode;
use crate::worker::SeatView;
use crate::worker::SpreadView;
use crate::worker::Worker;

/// Stage one page's textures, protecting `bound` (the pages the seats
/// draw) from eviction. `PageTextures` implements this against GL; tests
/// substitute a recorder.
pub trait PageStage {
    fn stage(
        &mut self,
        page: usize,
        decoded: &Decoded,
        bound: [Option<usize>; 2],
    ) -> Result<(), TextureError>;
}

impl PageStage for PageTextures {
    fn stage(
        &mut self,
        page: usize,
        decoded: &Decoded,
        bound: [Option<usize>; 2],
    ) -> Result<(), TextureError> {
        self.stage(page, decoded, bound)
    }
}

enum SeatStep<'view> {
    Show {
        page: usize,
        decoded: &'view Arc<Decoded>,
    },
    Hold,
    Clear,
}

struct ScreenPlan<'view> {
    steps: [SeatStep<'view>; 2],
    shown_after: Option<Spread>,
    committed: bool,
    stage: Option<(usize, &'view Arc<Decoded>)>,
}

fn seat_step(
    seat: Option<(usize, &SeatView)>,
    bound: Option<usize>,
) -> SeatStep<'_> {
    match seat {
        Some((page, SeatView::Ready(decoded))) => {
            SeatStep::Show { page, decoded }
        }
        Some((page, SeatView::Pending))
            if bound.is_none_or(|bound_page| bound_page == page) =>
        {
            SeatStep::Hold
        }
        None | Some((_, SeatView::Failed(_) | SeatView::Pending)) => {
            SeatStep::Clear
        }
    }
}

/// Build the frame's seat plan. `bound[i]` is the page seat i currently
/// shows.
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
    let undecided = seats
        .into_iter()
        .any(|seat| matches!(seat, Some((_, SeatView::Pending))));
    if undecided && shown.is_some_and(|shown| shown != wanted) {
        let stage = seats.into_iter().find_map(|seat| match seat {
            Some((page, SeatView::Ready(decoded))) => {
                Some((page, decoded))
            }
            _ => None,
        });
        return ScreenPlan {
            steps: [SeatStep::Hold, SeatStep::Hold],
            shown_after: shown,
            committed: false,
            stage,
        };
    }
    let steps = [
        seat_step(seat_left, bound_left),
        seat_step(seat_right, bound_right),
    ];
    #[expect(
        clippy::rest_pattern_accessible_field,
        reason = "the matched fields are unused here, so naming them would trip unneeded_field_pattern"
    )]
    let any_shown = steps
        .iter()
        .any(|step| matches!(step, SeatStep::Show { .. }));
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
fn forward_spread(
    spread: Spread,
    mode: ViewMode,
    len: PageCount,
) -> Spread {
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
pub(crate) fn page_watched(
    page: usize,
    spread: Spread,
    mode: ViewMode,
    len: PageCount,
) -> bool {
    spread.contains(page)
        || forward_spread(spread, mode, len).contains(page)
}

#[derive(Clone, Copy)]
struct SeatShow {
    page: usize,
    dims: (u32, u32),
}

#[derive(Default)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is the crate's visibility style for crate-wide items, even inside private modules"
)]
pub(crate) struct Screen {
    shown: Option<Spread>,
    seats: [Option<SeatShow>; 2],
    upload_error: Option<Arc<PageError>>,
}

impl Screen {
    /// Present `wanted` through `pages` per the plan. Returns whether
    /// the frame committed a new spread, which re-arms the caller's fit.
    #[expect(
        clippy::indexing_slicing,
        reason = "i enumerates two zipped two-seat arrays, so it always indexes within them"
    )]
    pub(crate) fn display(
        &mut self,
        pages: &mut impl PageStage,
        wanted: Spread,
        mode: ViewMode,
        snapshot: &SpreadView,
    ) -> bool {
        let mut bound = self.seat_pages();
        let plan = plan_screen(
            self.shown,
            wanted,
            mode,
            snapshot.seats(mode),
            bound,
        );
        // The frame's uploads are one outcome: a refusal on either seat
        // survives the partner's success, and a staging frame with no
        // refusal clears the error. A hold frame leaves it.
        let mut stage_error = None;
        let mut staged_cleanly = false;
        for (i, (shown, step)) in
            self.seats.iter_mut().zip(plan.steps).enumerate()
        {
            match step {
                SeatStep::Show { page, decoded } => {
                    // The incoming page joins the protected set before its upload.
                    bound[i] = Some(page);
                    match pages.stage(page, decoded, bound) {
                        Ok(()) => staged_cleanly = true,
                        Err(error) => {
                            stage_error = Some(Arc::new(error.widen()));
                        }
                    }
                    // The seat keeps its page and dims even on a refused
                    // upload, so the layout and hold policy stay stable
                    // while the next frame retries the stage.
                    *shown = Some(SeatShow {
                        page,
                        dims: (decoded.width, decoded.height),
                    });
                }
                SeatStep::Hold => {}
                SeatStep::Clear => {
                    bound[i] = None;
                    *shown = None;
                }
            }
        }
        if stage_error.is_some() || staged_cleanly {
            self.upload_error = stage_error;
        }
        if let Some((page, decoded)) = plan.stage {
            self.stage(pages, page, decoded);
        }
        self.shown = plan.shown_after;
        plan.committed
    }

    /// Upload the forward spread's decided pages ahead of the turn, so
    /// an ordinary commit finds its pages already cached: the
    /// multi-megabyte upload lands on this frame, which redraws
    /// anyway, instead of the turn's first visible one.
    pub(crate) fn stage_forward(
        &mut self,
        pages: &mut impl PageStage,
        current: Spread,
        mode: ViewMode,
        len: PageCount,
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

    pub(crate) fn seat_pages(&self) -> [Option<usize>; 2] {
        self.seats.each_ref().map(|seat| seat.map(|show| show.page))
    }

    pub(crate) fn seat_dims(&self) -> [Option<(u32, u32)>; 2] {
        self.seats.each_ref().map(|seat| seat.map(|show| show.dims))
    }

    pub(crate) fn upload_error(&self) -> Option<&Arc<PageError>> {
        self.upload_error.as_ref()
    }

    /// Upload `page`'s textures without showing them. A refusal
    /// surfaces in the status line and repeats while staging retries
    /// each frame.
    fn stage(
        &mut self,
        pages: &mut impl PageStage,
        page: usize,
        decoded: &Arc<Decoded>,
    ) {
        if let Err(error) = pages.stage(page, decoded, self.seat_pages()) {
            self.upload_error = Some(Arc::new(error.widen()));
        }
    }

    pub(crate) fn clear_book(&mut self) {
        self.shown = None;
        self.seats = [None, None];
        self.upload_error = None;
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;
    use crate::fault::TextureFault;
    use crate::testkit::failed_seat;
    use crate::testkit::ready_seat;
    use crate::view::ViewMode;
    use eros::ErrorUnion;

    fn pages(count: usize) -> PageCount {
        PageCount::new(count).expect("test book is non-empty")
    }

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
                SeatStep::Show { page, decoded } => (
                    StepKind::Show,
                    Some((*page, (decoded.width, decoded.height))),
                ),
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
            bound: [Option<usize>; 2],
            steps: [StepFacts; 2],
            shown_after: Option<Spread>,
            committed: bool,
            stage: Option<(usize, (u32, u32))>,
        }
        let old = Spread {
            first: 3,
            second: Some(4),
        };
        let pair = Spread {
            first: 5,
            second: Some(6),
        };
        let failed = || Some(failed_seat("x"));
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
        for Row {
            name,
            shown,
            wanted,
            mode,
            views,
            bound,
            steps,
            shown_after,
            committed,
            stage,
        } in rows
        {
            let views = views.each_ref().map(Option::as_ref);
            let plan = plan_screen(shown, wanted, mode, views, bound);
            assert_eq!(
                plan.steps.each_ref().map(step_facts),
                steps,
                "{name}: steps"
            );
            assert_eq!(plan.shown_after, shown_after, "{name}: shown");
            assert_eq!(plan.committed, committed, "{name}: committed");
            assert_eq!(
                plan.stage.map(|(page, decoded)| (
                    page,
                    (decoded.width, decoded.height)
                )),
                stage,
                "{name}: stage"
            );
        }
    }

    /// A `PageStage` that records every ask and refuses the pages it
    /// is told to, so `display`'s apply half runs headless.
    struct Recorder {
        refuse: Vec<usize>,
        log: Vec<(usize, [Option<usize>; 2])>,
    }

    impl PageStage for Recorder {
        fn stage(
            &mut self,
            page: usize,
            _decoded: &Decoded,
            bound: [Option<usize>; 2],
        ) -> Result<(), TextureError> {
            self.log.push((page, bound));
            if self.refuse.contains(&page) {
                Err(ErrorUnion::new(TextureFault {
                    dims: (1, 1),
                    max_edge: 1,
                }))
            } else {
                Ok(())
            }
        }
    }

    fn lone_view(seat: SeatView) -> SpreadView {
        SpreadView {
            first: seat,
            second: None,
        }
    }

    #[test]
    fn a_refused_upload_keeps_the_seat_and_retries() {
        let wanted = Spread::lone(5);
        let mut screen = Screen::default();
        let mut pages = Recorder {
            refuse: vec![5],
            log: Vec::new(),
        };

        // The upload is refused: the seat keeps its page and dims, the
        // commit happens, and the error surfaces.
        let committed = screen.display(
            &mut pages,
            wanted,
            ViewMode::Normal,
            &lone_view(ready_seat(600, 700)),
        );
        assert!(committed, "the refused page still commits the spread");
        assert_eq!(screen.seat_dims(), [Some((600, 700)), None]);
        assert!(screen.upload_error().is_some());

        // The next frame retries the same stage: the page is protected
        // in its own bound set even though the upload never landed.
        pages.refuse.clear();
        let retried = screen.display(
            &mut pages,
            wanted,
            ViewMode::Normal,
            &lone_view(ready_seat(600, 700)),
        );
        assert!(!retried, "the spread is already shown");
        assert_eq!(pages.log.last().copied(), Some((5, [Some(5), None])));
        assert!(
            screen.upload_error().is_none(),
            "a cleanly staged frame clears the error"
        );
    }

    #[test]
    fn a_refused_upload_survives_the_partners_success() {
        // Manga: left = second page 6, right = first page 5.
        let wanted = Spread {
            first: 5,
            second: Some(6),
        };
        let mut screen = Screen::default();
        let mut pages = Recorder {
            refuse: vec![6],
            log: Vec::new(),
        };
        screen.display(
            &mut pages,
            wanted,
            ViewMode::Manga,
            &SpreadView {
                first: ready_seat(500, 700),
                second: Some(ready_seat(600, 700)),
            },
        );
        assert!(screen.upload_error().is_some());
        // The partner's success must not clear the refused upload's
        // error: the frame is one outcome.
        assert_eq!(
            screen.seat_dims(),
            [Some((600, 700)), Some((500, 700))]
        );
        // The incoming pages were protected across both stages: the
        // second stage's bound set names both seats' new pages.
        assert_eq!(
            pages.log,
            vec![(6, [Some(6), None]), (5, [Some(6), Some(5)])]
        );
    }

    #[test]
    fn a_hold_frame_leaves_the_upload_error_alone() {
        let refused = Spread::lone(5);
        let wanted = Spread::lone(6);
        let mut screen = Screen::default();
        let mut pages = Recorder {
            refuse: vec![5],
            log: Vec::new(),
        };
        // Seed the upload error: page 5's stage is refused while its
        // seat keeps it.
        screen.display(
            &mut pages,
            refused,
            ViewMode::Normal,
            &lone_view(ready_seat(600, 700)),
        );
        assert!(screen.upload_error().is_some());
        // A hold frame plans no stages at all, so the refusal survives
        // it: only a staging frame rewrites the error.
        pages.log.clear();
        let committed = screen.display(
            &mut pages,
            wanted,
            ViewMode::Normal,
            &lone_view(SeatView::Pending),
        );
        assert!(!committed);
        assert!(pages.log.is_empty(), "a hold frame stages nothing");
        assert!(
            screen.upload_error().is_some(),
            "a hold frame leaves the upload error"
        );
    }

    #[test]
    fn a_clear_step_drops_the_seat_without_touching_pages() {
        let old = Spread::lone(5);
        let wanted = Spread::lone(6);
        let mut screen = Screen::default();
        let mut pages = Recorder {
            refuse: vec![],
            log: Vec::new(),
        };
        screen.display(
            &mut pages,
            old,
            ViewMode::Normal,
            &lone_view(ready_seat(600, 700)),
        );
        pages.log.clear();
        // A failed page clears its seat: pure bookkeeping, no stage ask.
        screen.display(
            &mut pages,
            wanted,
            ViewMode::Normal,
            &lone_view(failed_seat("boom")),
        );
        assert_eq!(screen.seat_dims(), [None, None]);
        assert!(pages.log.is_empty(), "clearing a seat stages nothing");
        assert_eq!(screen.seat_pages(), [None, None]);
    }

    #[test]
    fn page_watched_covers_the_shown_and_forward_spreads() {
        let len = pages(10);
        let shown = Spread {
            first: 3,
            second: Some(4),
        };
        for page in [3_usize, 4] {
            assert!(page_watched(page, shown, ViewMode::Manga, len));
        }
        // The forward spread under Manga from 3 is {5, 6}.
        for page in [5_usize, 6] {
            assert!(page_watched(page, shown, ViewMode::Manga, len));
        }
        assert!(!page_watched(0, shown, ViewMode::Manga, len));
        assert!(!page_watched(9, shown, ViewMode::Manga, len));
        // At the book's end the forward spread wraps to the cover.
        let last = Spread {
            first: 9,
            second: None,
        };
        assert!(page_watched(0, last, ViewMode::Manga, len));
    }
}
