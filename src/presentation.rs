//! Frame planning: the spread the screen presents and the artifacts the
//! frame keeps or asks for. Pure: no clocks, GL, or IO.

use std::collections::BTreeMap;

use crate::decode::Format;
use crate::fault::PageFault;
use crate::plan::Direction;
use crate::preview::Artifact;
use crate::preview::Artifacts;
use crate::preview::VIEW_EDGE;
use crate::view;
use crate::view::PageCount;
use crate::view::Spread;
use crate::view::ViewMode;
use crate::worker::Motion;
use crate::worker::Want;

/// The page's own dims and format, independent of the stored representation.
#[derive(Clone, Copy)]
pub struct CachedPage {
    pub dims: (u32, u32),
    pub format: Format,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentedSpread {
    pub mode: ViewMode,
    pub spread: Spread,
}

pub struct Navigation {
    target: usize,
    presented: Option<PresentedSpread>,
}

impl Navigation {
    pub fn new(target: usize) -> Self {
        Self {
            target,
            presented: None,
        }
    }

    pub fn target(&self) -> usize {
        self.target
    }

    pub fn presented(&self) -> Option<Spread> {
        self.presented.map(|presented| presented.spread)
    }

    pub fn presented_with_mode(&self) -> Option<PresentedSpread> {
        self.presented
    }

    pub fn set_target(&mut self, target: usize) {
        self.target = target;
    }

    pub fn record_presented(
        &mut self,
        mode: ViewMode,
        presented: Option<Spread>,
    ) {
        self.presented =
            presented.map(|spread| PresentedSpread { mode, spread });
    }

    pub fn commit_presented(&mut self, mode: ViewMode) -> bool {
        let Some(presented) = self.presented else {
            return false;
        };
        if presented.mode != mode || presented.spread.first == self.target
        {
            return false;
        }
        self.target = presented.spread.first;
        true
    }
}

pub struct FrameInput<'frame> {
    pub target: Spread,
    pub presented: Option<PresentedSpread>,
    pub mode: ViewMode,
    pub motion: Motion,
    pub direction: Direction,
    pub len: PageCount,
    pub facts: &'frame BTreeMap<usize, CachedPage>,
    pub faults: &'frame BTreeMap<usize, PageFault>,
    pub measurable: bool,
}

pub struct FramePlan {
    presented: Option<Spread>,
    demand: Spread,
    wants: Vec<Want>,
    keep: Keep,
}

struct Keep {
    mode: ViewMode,
    len: PageCount,
    target: Spread,
    anchor: Spread,
}

impl FramePlan {
    #[must_use]
    pub fn presented(&self) -> Option<Spread> {
        self.presented
    }

    /// The spread the frame treats as on screen: the presented one, or the
    /// target while nothing is presentable.
    #[must_use]
    pub fn anchor(&self) -> Spread {
        self.keep.anchor
    }

    #[must_use]
    pub fn demand(&self) -> Spread {
        self.demand
    }

    #[must_use]
    pub fn into_wants(self) -> Vec<Want> {
        self.wants
    }

    /// Every want, plus the role-bounded extras. It always covers `wants`.
    pub fn retains(&self, page: usize, kind: Artifact) -> bool {
        self.wants.contains(&Want::new(page, kind))
            || self.keep.retains(page, kind)
    }
}

impl Keep {
    fn retains(&self, page: usize, kind: Artifact) -> bool {
        match kind {
            Artifact::Thumb => true,
            Artifact::View => view_kept(
                page,
                self.anchor,
                self.target,
                self.mode,
                self.len,
            ),
            Artifact::Native => self.anchor.contains(page),
        }
    }
}

pub fn plan_frame(
    input: &FrameInput<'_>,
    artifacts: impl Fn(usize) -> Artifacts,
    scale_for: impl Fn(Spread) -> f32,
) -> FramePlan {
    let skimming = input.motion == Motion::Skimming;
    let presented = choose_presented(
        input.target,
        input.presented.map(|presented| presented.spread),
        input.mode,
        input.len,
        input.facts,
        input.direction == Direction::Forward,
        |page| {
            input.faults.contains_key(&page)
                || (input.facts.contains_key(&page)
                    && if skimming {
                        artifacts(page).any()
                    } else {
                        artifacts(page).readable()
                    })
        },
    );
    let anchor = presented.unwrap_or(input.target);
    let demand = demand_spread(
        input.target,
        anchor,
        input.motion,
        input.direction,
        input.mode,
        input.len,
    );
    let anchor_scale = scale_for(anchor);
    let wants = wants(
        demand,
        anchor,
        input.mode,
        input.motion == Motion::Still,
        // Natives key off the anchor, not the demand spread.
        |page| {
            input.measurable
                && input.facts.get(&page).is_some_and(|entry| {
                    native_wanted(anchor_scale, entry.dims)
                })
        },
        artifacts,
        |page| input.faults.contains_key(&page),
    );
    FramePlan {
        presented,
        demand,
        wants,
        keep: Keep {
            mode: input.mode,
            len: input.len,
            target: input.target,
            anchor,
        },
    }
}

/// Wants for a spread with no textures, before the frame can measure
/// placement.
pub fn cold_wants(spread: Spread, mode: ViewMode) -> Vec<Want> {
    wants(
        spread,
        spread,
        mode,
        true,
        |_| false,
        |_| Artifacts::default(),
        |_| false,
    )
}

#[expect(
    clippy::cast_precision_loss,
    reason = "page edges and the fixed view edge stay far below f32's exact-integer range, so the threshold comparison holds"
)]
fn native_wanted(scale: f32, page_dims: (u32, u32)) -> bool {
    let page_edge = page_dims.0.max(page_dims.1);
    page_edge > VIEW_EDGE && scale > VIEW_EDGE as f32 / page_edge as f32
}

/// While skimming, demand the next spread instead of the far target:
/// serving a far pair splits it before both pages arrive.
fn demand_spread(
    target: Spread,
    anchor: Spread,
    motion: Motion,
    direction: Direction,
    mode: ViewMode,
    len: PageCount,
) -> Spread {
    if motion == Motion::Still {
        return target;
    }
    let next_start = view::advance(
        anchor.first,
        direction == Direction::Forward,
        mode,
        len,
    );
    view::spread_for(next_start, mode, len)
}

fn wants(
    demand: Spread,
    anchor: Spread,
    mode: ViewMode,
    still: bool,
    upscaled: impl Fn(usize) -> bool,
    artifacts: impl Fn(usize) -> Artifacts,
    faulted: impl Fn(usize) -> bool,
) -> Vec<Want> {
    let mut wants = Vec::new();
    for spread in [demand, anchor] {
        for page in spread.seats(mode).into_iter().flatten() {
            if faulted(page) {
                continue;
            }
            let resident = artifacts(page);
            if still {
                if !resident.readable() {
                    push_want(&mut wants, page, Artifact::View);
                }
                if !resident.contains(Artifact::Thumb) {
                    push_want(&mut wants, page, Artifact::Thumb);
                }
                if anchor.contains(page)
                    && upscaled(page)
                    && !resident.contains(Artifact::Native)
                {
                    push_want(&mut wants, page, Artifact::Native);
                }
                continue;
            }
            if !resident.contains(Artifact::Thumb) {
                push_want(&mut wants, page, Artifact::Thumb);
            }
        }
    }
    wants
}

fn push_want(wants: &mut Vec<Want>, page: usize, kind: Artifact) {
    let want = Want { page, kind };
    if !wants.contains(&want) {
        wants.push(want);
    }
}

const VIEW_KEEP_SPREADS: usize = 2;

fn view_kept(
    page: usize,
    anchor: Spread,
    target: Spread,
    mode: ViewMode,
    len: PageCount,
) -> bool {
    let start = view::spread_for(page, mode, len).first;
    start.abs_diff(anchor.first) <= VIEW_KEEP_SPREADS
        || start.abs_diff(target.first) <= VIEW_KEEP_SPREADS
}

/// The target spread once every page clears `can_show`; otherwise the
/// nearest presentable start between the presented and target spreads, so
/// a skim advances monotonically; otherwise the presented spread.
fn choose_presented(
    target: Spread,
    presented: Option<Spread>,
    mode: ViewMode,
    len: PageCount,
    cached: &BTreeMap<usize, CachedPage>,
    forward: bool,
    can_show: impl Fn(usize) -> bool,
) -> Option<Spread> {
    let presentable = |spread: Spread| {
        spread.seats(mode).into_iter().flatten().all(&can_show)
    };
    if presentable(target) {
        return Some(target);
    }
    let (low, high) = presented.map_or((0, usize::MAX), |presented| {
        (
            presented.first.min(target.first),
            presented.first.max(target.first),
        )
    });
    let mut best: Option<Spread> = None;
    for &page in cached.keys() {
        let spread = view::spread_for(page, mode, len);
        if spread.first < low
            || spread.first > high
            || !presentable(spread)
        {
            continue;
        }
        let distance = spread.first.abs_diff(target.first);
        let closer = best.is_none_or(|current| {
            let current_distance = current.first.abs_diff(target.first);
            distance < current_distance
                || (distance == current_distance
                    && if forward {
                        spread.first > current.first
                    } else {
                        spread.first < current.first
                    })
        });
        if closer {
            best = Some(spread);
        }
    }
    best.or(presented)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;

    #[test]
    fn a_native_is_only_wanted_when_the_view_has_fewer_pixels() {
        assert!(!native_wanted(16.0, (1920, 1080)));
        assert!(!native_wanted(16.0, (VIEW_EDGE, 1440)));
        let four_k = (3840, 2160);
        assert!(!native_wanted(0.66, four_k));
        assert!(native_wanted(0.67, four_k));
        let ten_k = (10080, 5640);
        assert!(!native_wanted(0.25, ten_k));
        assert!(native_wanted(0.26, ten_k));
    }

    #[test]
    fn retention_keeps_views_within_the_role_bounds() {
        let len = PageCount::new(20).expect("the test book is not empty");
        let facts = BTreeMap::new();
        let faults = BTreeMap::new();
        let plan = plan_frame(
            &FrameInput {
                target: Spread::lone(10),
                presented: Some(PresentedSpread {
                    mode: ViewMode::Normal,
                    spread: Spread::lone(10),
                }),
                mode: ViewMode::Normal,
                motion: Motion::Still,
                direction: Direction::Forward,
                len,
                facts: &facts,
                faults: &faults,
                measurable: false,
            },
            |_| Artifacts::default(),
            |_| 1.0,
        );
        assert!(plan.retains(10, Artifact::Native));
        assert!(!plan.retains(12, Artifact::Native));
        assert!(plan.retains(12, Artifact::View));
        assert!(!plan.retains(13, Artifact::View));
        assert!(plan.retains(0, Artifact::Thumb));
    }

    #[test]
    fn native_wants_follow_the_anchor_scale() {
        let len = PageCount::new(12).expect("the test book is not empty");
        let entry = CachedPage {
            dims: (4000, 6000),
            format: Format::Png,
        };
        let facts: BTreeMap<usize, CachedPage> =
            [1, 2].into_iter().map(|page| (page, entry)).collect();
        let faults = BTreeMap::new();
        let plan = plan_frame(
            &FrameInput {
                target: Spread::lone(2),
                presented: Some(PresentedSpread {
                    mode: ViewMode::Normal,
                    spread: Spread::lone(1),
                }),
                mode: ViewMode::Normal,
                motion: Motion::Still,
                direction: Direction::Forward,
                len,
                facts: &facts,
                faults: &faults,
                measurable: true,
            },
            |_| Artifacts::default(),
            |spread| if spread.first == 1 { 1.0 } else { 0.1 },
        );
        assert_eq!(plan.anchor(), Spread::lone(1));
        assert!(
            plan.wants.contains(&Want::new(1, Artifact::Native)),
            "the on-screen page asks for its native at the anchor scale"
        );
        assert!(
            !plan.wants.contains(&Want::new(2, Artifact::Native)),
            "the target page is not upscaled"
        );
    }

    #[test]
    fn presentation_advances_through_presentable_spreads() {
        let len = PageCount::new(12).expect("the test book is not empty");
        let mode = ViewMode::Normal;
        let entry = CachedPage {
            dims: (2, 2),
            format: Format::Png,
        };
        let cached: BTreeMap<usize, CachedPage> =
            [0, 2, 4, 8].into_iter().map(|page| (page, entry)).collect();
        let can_show = |page: usize| matches!(page, 0 | 2 | 4);
        let lone = Spread::lone;

        // The target wins as soon as it has a pixel.
        assert_eq!(
            choose_presented(
                lone(4),
                Some(lone(0)),
                mode,
                len,
                &cached,
                true,
                can_show
            ),
            Some(lone(4))
        );
        // Otherwise the nearest presentable spread between the presented
        // and the target; page 8 is cached but has no pixels.
        assert_eq!(
            choose_presented(
                lone(9),
                Some(lone(0)),
                mode,
                len,
                &cached,
                true,
                can_show
            ),
            Some(lone(4))
        );
        // Nothing between the two: the presented spread keeps its pixels.
        assert_eq!(
            choose_presented(
                lone(9),
                Some(lone(2)),
                mode,
                len,
                &cached,
                true,
                |page| page == 0
            ),
            Some(lone(2))
        );
        // A cold start picks the nearest presentable spread.
        assert_eq!(
            choose_presented(
                lone(5),
                None,
                mode,
                len,
                &cached,
                true,
                can_show
            ),
            Some(lone(4))
        );
        // A cold start presents only decoded spreads.
        assert_eq!(
            choose_presented(
                lone(5),
                None,
                mode,
                len,
                &cached,
                true,
                |_| false
            ),
            None
        );
    }

    #[test]
    fn paired_spread_waits_for_every_seat() {
        let len = PageCount::new(12).expect("the test book is not empty");
        let mode = ViewMode::Manga;
        let entry = CachedPage {
            dims: (2, 2),
            format: Format::Png,
        };
        let cached: BTreeMap<usize, CachedPage> =
            [0, 2].into_iter().map(|page| (page, entry)).collect();
        let pair = Spread {
            first: 1,
            second: Some(2),
        };
        let presented = Spread::lone(0);

        assert_eq!(
            choose_presented(
                pair,
                Some(presented),
                mode,
                len,
                &cached,
                true,
                |page| page == 2
            ),
            Some(presented)
        );
        assert_eq!(
            choose_presented(
                pair,
                Some(presented),
                mode,
                len,
                &cached,
                true,
                |page| matches!(page, 1 | 2)
            ),
            Some(pair)
        );
    }

    #[test]
    fn commit_presented_requires_the_current_mode() {
        let mut navigation = Navigation::new(5);
        assert!(!navigation.commit_presented(ViewMode::Normal));
        assert_eq!(navigation.target(), 5);

        navigation
            .record_presented(ViewMode::Normal, Some(Spread::lone(4)));
        assert!(navigation.commit_presented(ViewMode::Normal));
        assert_eq!(navigation.target(), 4);
        assert!(!navigation.commit_presented(ViewMode::Normal));

        navigation.set_target(5);
        assert!(!navigation.commit_presented(ViewMode::Manga));
        assert_eq!(navigation.target(), 5);
    }

    #[test]
    fn skim_demands_the_next_spread() {
        let len = PageCount::new(40).expect("the test book is not empty");
        let facts = BTreeMap::new();
        let faults = BTreeMap::new();
        let plan = plan_frame(
            &FrameInput {
                target: Spread::lone(20),
                presented: Some(PresentedSpread {
                    mode: ViewMode::Normal,
                    spread: Spread::lone(2),
                }),
                mode: ViewMode::Normal,
                motion: Motion::Skimming,
                direction: Direction::Forward,
                len,
                facts: &facts,
                faults: &faults,
                measurable: true,
            },
            |_| Artifacts::default(),
            |_| 1.0,
        );
        assert_eq!(plan.presented, Some(Spread::lone(2)));
        assert_eq!(plan.demand, Spread::lone(3));
        assert_ne!(plan.demand, Spread::lone(20));
    }
}
