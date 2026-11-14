//! Frame planning: the spread the screen shows and the roles the frame
//! keeps or asks for. Pure: no clocks, GL, or IO.

use std::collections::BTreeMap;

use crate::browse;
use crate::browse::Direction;
use crate::browse::Motion;
use crate::browse::PageCount;
use crate::browse::PageInfo;
use crate::browse::ReadingMode;

use crate::browse::Request;
use crate::browse::Role;
use crate::browse::Roles;
use crate::browse::ShownSpread;
use crate::browse::Spread;
use crate::cache::ImageLimits;

pub struct FrameInput {
    pub target: Spread,
    pub shown: Option<ShownSpread>,
    pub mode: ReadingMode,
    pub motion: Motion,
    pub direction: Direction,
    pub len: PageCount,
    pub limits: ImageLimits,
    pub measurable: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PageSummary {
    pub info: Option<PageInfo>,
    pub faulted: bool,
}

pub struct Frame {
    shown: Option<Spread>,
    demand: Spread,
    requests: Vec<Request>,
    retention: Retention,
    playback: [Option<(usize, Role)>; 2],
}

struct Retention {
    mode: ReadingMode,
    len: PageCount,
    target: Spread,
    on_screen: Spread,
}

impl Frame {
    #[must_use]
    pub fn shown(&self) -> Option<Spread> {
        self.shown
    }

    /// The spread the frame treats as on screen: the shown one, or the
    /// target while nothing is presentable.
    #[must_use]
    pub fn on_screen(&self) -> Spread {
        self.retention.on_screen
    }

    #[must_use]
    pub fn demand(&self) -> Spread {
        self.demand
    }

    #[must_use]
    pub fn into_requests(self) -> Vec<Request> {
        self.requests
    }

    /// The pages a still frame animates.
    #[must_use]
    pub fn playback_targets(&self) -> [Option<(usize, Role)>; 2] {
        self.playback
    }

    /// Every request, plus the role-bounded extras. It always covers
    /// `requests`.
    pub fn retains(&self, page: usize, role: Role) -> bool {
        self.requests.contains(&Request::new(page, role))
            || self.retention.retains(page, role)
    }
}

impl Retention {
    fn retains(&self, page: usize, role: Role) -> bool {
        match role {
            Role::Thumb => true,
            Role::Fit => fit_kept(
                page,
                self.on_screen,
                self.target,
                self.mode,
                self.len,
            ),
            Role::Full => self.on_screen.contains(page),
        }
    }
}

pub fn build_frame<P>(
    input: &FrameInput,
    pages: &BTreeMap<usize, P>,
    summary: impl Fn(&P) -> PageSummary,
    roles: impl Fn(usize) -> Roles,
    scale_for: impl Fn(Spread) -> f32,
) -> Frame {
    let page = |index: usize| pages.get(&index).map_or_default(&summary);
    let shown = choose_presented(input, pages, &summary, &roles);
    let on_screen = shown.unwrap_or(input.target);
    let playback = if input.motion == Motion::Still {
        on_screen.seats(input.mode).map(|seat| {
            let candidate = seat?;
            let entry = page(candidate);
            if entry.faulted {
                return None;
            }
            let info = entry.info?;
            if !info.format.may_animate() {
                return None;
            }
            let role = roles(candidate).best()?;
            (role != Role::Thumb).then_some((candidate, role))
        })
    } else {
        [None, None]
    };
    let demand = demand_spread(
        input.target,
        on_screen,
        input.motion,
        input.direction,
        input.mode,
        input.len,
    );
    let on_screen_scale = scale_for(on_screen);
    let requests = request_roles(
        demand,
        on_screen,
        input.mode,
        input.motion == Motion::Still,
        // Full images key off the on-screen spread, not the demand spread.
        |candidate| {
            input.measurable
                && page(candidate).info.is_some_and(|entry| {
                    full_wanted(on_screen_scale, entry.dims, input.limits)
                })
        },
        &roles,
        |candidate| page(candidate).faulted,
    );
    Frame {
        shown,
        demand,
        requests,
        retention: Retention {
            mode: input.mode,
            len: input.len,
            target: input.target,
            on_screen,
        },
        playback,
    }
}

/// Requests for a spread with no textures, before the frame can measure
/// placement.
pub fn cold_requests(spread: Spread, mode: ReadingMode) -> Vec<Request> {
    request_roles(
        spread,
        spread,
        mode,
        true,
        |_| false,
        |_| Roles::default(),
        |_| false,
    )
}

#[expect(
    clippy::cast_precision_loss,
    reason = "page edges and image limits stay far below f32's exact-integer range, so the threshold comparison holds"
)]
fn full_wanted(
    scale: f32,
    page_dims: (u32, u32),
    limits: ImageLimits,
) -> bool {
    const ROUNDING_MARGIN_PX: f32 = 1.0;

    let page_edge = page_dims.0.max(page_dims.1);
    let full_edge = page_edge.min(limits.max_texture_edge());
    full_edge > limits.fit_edge()
        && scale * page_edge as f32
            > limits.fit_edge() as f32 + ROUNDING_MARGIN_PX
}

/// While skimming, demand the next spread instead of the far target:
/// serving a far pair splits it before both pages arrive.
fn demand_spread(
    target: Spread,
    on_screen: Spread,
    motion: Motion,
    direction: Direction,
    mode: ReadingMode,
    len: PageCount,
) -> Spread {
    if motion == Motion::Still {
        return target;
    }
    let next_start =
        browse::advance(on_screen.first, direction, mode, len);
    browse::spread_for(next_start, mode, len)
}

fn request_roles(
    demand: Spread,
    on_screen: Spread,
    mode: ReadingMode,
    still: bool,
    upscaled: impl Fn(usize) -> bool,
    roles: impl Fn(usize) -> Roles,
    faulted: impl Fn(usize) -> bool,
) -> Vec<Request> {
    let mut requests = Vec::new();
    for spread in [demand, on_screen] {
        for page in spread.seats(mode).into_iter().flatten() {
            if faulted(page) {
                continue;
            }
            let resident = roles(page);
            if still {
                if !resident.has_settled_image() {
                    push_request(&mut requests, page, Role::Fit);
                }
                if on_screen.contains(page)
                    && upscaled(page)
                    && !resident.contains(Role::Full)
                {
                    push_request(&mut requests, page, Role::Full);
                }
                continue;
            }
            if !resident.any() {
                push_request(&mut requests, page, Role::Thumb);
            }
        }
    }
    requests
}

fn push_request(requests: &mut Vec<Request>, page: usize, role: Role) {
    let request = Request { page, role };
    if !requests.contains(&request) {
        requests.push(request);
    }
}

const FIT_KEEP_SPREADS: usize = 2;

fn fit_kept(
    page: usize,
    on_screen: Spread,
    target: Spread,
    mode: ReadingMode,
    len: PageCount,
) -> bool {
    let start = browse::spread_for(page, mode, len).first;
    start.abs_diff(on_screen.first) <= FIT_KEEP_SPREADS
        || start.abs_diff(target.first) <= FIT_KEEP_SPREADS
}

/// The target spread once every page is presentable; otherwise the
/// nearest presentable start between the shown and target spreads, so
/// a skim advances monotonically; otherwise the shown spread.
fn choose_presented<P>(
    input: &FrameInput,
    pages: &BTreeMap<usize, P>,
    summary: impl Fn(&P) -> PageSummary,
    roles: impl Fn(usize) -> Roles,
) -> Option<Spread> {
    let forward = input.direction == Direction::Forward;
    let skimming = input.motion == Motion::Skimming;
    let shown = input.shown.map(|entry| entry.spread);
    let can_show = |candidate: usize| {
        let entry = pages.get(&candidate).map_or_default(&summary);
        entry.faulted
            || (entry.info.is_some()
                && if skimming {
                    roles(candidate).any()
                } else {
                    roles(candidate).has_settled_image()
                })
    };
    let presentable = |spread: Spread| {
        spread.seats(input.mode).into_iter().flatten().all(can_show)
    };
    if presentable(input.target) {
        return Some(input.target);
    }
    let (low, high) = shown.map_or((0, usize::MAX), |shown| {
        (
            shown.first.min(input.target.first),
            shown.first.max(input.target.first),
        )
    });
    let mut best: Option<Spread> = None;
    for (&page, entry) in pages.range(low..=high.saturating_add(1)) {
        if summary(entry).info.is_none() {
            continue;
        }
        let spread = browse::spread_for(page, input.mode, input.len);
        if spread.first < low
            || spread.first > high
            || !presentable(spread)
        {
            continue;
        }
        let distance = spread.first.abs_diff(input.target.first);
        let closer = best.is_none_or(|current| {
            let current_distance =
                current.first.abs_diff(input.target.first);
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
    best.or(shown)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;
    use crate::browse::ReadingPosition;
    use crate::decode::Format;

    fn test_limits() -> ImageLimits {
        const FIT_EDGE: u32 = 2560;
        const MAX_TEXTURE_EDGE: u32 = 16_384;
        ImageLimits::new(FIT_EDGE, MAX_TEXTURE_EDGE)
    }

    fn ready_page(info: PageInfo) -> PageSummary {
        PageSummary {
            info: Some(info),
            faulted: false,
        }
    }

    #[test]
    fn a_full_image_is_only_requested_when_the_fit_has_fewer_pixels() {
        let limits = test_limits();
        assert!(!full_wanted(16.0, (1920, 1080), limits));
        assert!(!full_wanted(16.0, (2560, 1440), limits));
        let four_k = (3840, 2160);
        assert!(!full_wanted(0.66, four_k, limits));
        assert!(full_wanted(0.67, four_k, limits));
        // Fractional Wayland scales can push fit rendering a sub-pixel past fit_edge.
        assert!(!full_wanted(2560.5 / 5000.0, (5000, 3000), limits));
        let ten_k = (10080, 5640);
        assert!(!full_wanted(0.25, ten_k, limits));
        assert!(full_wanted(0.26, ten_k, limits));
        assert!(!full_wanted(16.0, ten_k, ImageLimits::new(2560, 2560)));
    }

    #[test]
    fn retention_keeps_fit_images_within_the_role_bounds() {
        let len = PageCount::new(20).expect("the test book is not empty");
        let pages: BTreeMap<usize, PageSummary> = BTreeMap::new();
        let frame = build_frame(
            &FrameInput {
                target: Spread::lone(10),
                shown: Some(ShownSpread {
                    mode: ReadingMode::Normal,
                    spread: Spread::lone(10),
                }),
                mode: ReadingMode::Normal,
                motion: Motion::Still,
                direction: Direction::Forward,
                len,
                limits: test_limits(),
                measurable: false,
            },
            &pages,
            |summary| *summary,
            |_| Roles::default(),
            |_| 1.0,
        );
        assert!(frame.retains(10, Role::Full));
        assert!(!frame.retains(12, Role::Full));
        assert!(frame.retains(12, Role::Fit));
        assert!(!frame.retains(13, Role::Fit));
        assert!(frame.retains(0, Role::Thumb));
    }

    #[test]
    fn full_requests_follow_the_on_screen_scale() {
        let len = PageCount::new(12).expect("the test book is not empty");
        let entry = PageInfo {
            dims: (4000, 6000),
            format: Format::Png,
        };
        let pages: BTreeMap<usize, PageSummary> = [1, 2]
            .into_iter()
            .map(|page| (page, ready_page(entry)))
            .collect();
        let frame = build_frame(
            &FrameInput {
                target: Spread::lone(2),
                shown: Some(ShownSpread {
                    mode: ReadingMode::Normal,
                    spread: Spread::lone(1),
                }),
                mode: ReadingMode::Normal,
                motion: Motion::Still,
                direction: Direction::Forward,
                len,
                limits: test_limits(),
                measurable: true,
            },
            &pages,
            |summary| *summary,
            |_| Roles::default(),
            |spread| if spread.first == 1 { 1.0 } else { 0.1 },
        );
        assert_eq!(frame.on_screen(), Spread::lone(1));
        assert!(
            frame.requests.contains(&Request::new(1, Role::Full)),
            "the on-screen page asks for its full image at the on-screen scale"
        );
        assert!(
            !frame.requests.contains(&Request::new(2, Role::Full)),
            "the target page is not upscaled"
        );
    }

    #[test]
    fn presentation_advances_through_presentable_spreads() {
        let len = PageCount::new(12).expect("the test book is not empty");
        let mode = ReadingMode::Normal;
        let entry = PageInfo {
            dims: (2, 2),
            format: Format::Png,
        };
        let pages: BTreeMap<usize, PageSummary> = [0, 2, 4, 8]
            .into_iter()
            .map(|page| (page, ready_page(entry)))
            .collect();
        let fit_on = |fit_pages: &'static [usize]| {
            move |page: usize| {
                let mut roles = Roles::default();
                if fit_pages.contains(&page) {
                    roles.set(Role::Fit);
                }
                roles
            }
        };
        let input = |target, shown: Option<Spread>| FrameInput {
            target,
            shown: shown.map(|spread| ShownSpread { mode, spread }),
            mode,
            motion: Motion::Skimming,
            direction: Direction::Forward,
            len,
            limits: test_limits(),
            measurable: false,
        };
        let lone = Spread::lone;

        // The target wins as soon as it has a pixel.
        assert_eq!(
            choose_presented(
                &input(lone(4), Some(lone(0))),
                &pages,
                |summary| *summary,
                fit_on(&[0, 2, 4])
            ),
            Some(lone(4))
        );
        // Otherwise the nearest presentable spread between the shown
        // and the target; page 8 is known but has no pixels.
        assert_eq!(
            choose_presented(
                &input(lone(9), Some(lone(0))),
                &pages,
                |summary| *summary,
                fit_on(&[0, 2, 4])
            ),
            Some(lone(4))
        );
        // Nothing between the two: the shown spread keeps its pixels.
        assert_eq!(
            choose_presented(
                &input(lone(9), Some(lone(2))),
                &pages,
                |summary| *summary,
                fit_on(&[0])
            ),
            Some(lone(2))
        );
        // A cold start picks the nearest presentable spread.
        assert_eq!(
            choose_presented(
                &input(lone(5), None),
                &pages,
                |summary| *summary,
                fit_on(&[0, 2, 4])
            ),
            Some(lone(4))
        );
        // A cold start presents only decoded spreads.
        assert_eq!(
            choose_presented(
                &input(lone(5), None),
                &pages,
                |summary| *summary,
                |_| Roles::default()
            ),
            None
        );
    }

    #[test]
    fn paired_spread_waits_for_every_seat() {
        let len = PageCount::new(12).expect("the test book is not empty");
        let mode = ReadingMode::Manga;
        let entry = PageInfo {
            dims: (2, 2),
            format: Format::Png,
        };
        let mut pages: BTreeMap<usize, PageSummary> = [0, 2]
            .into_iter()
            .map(|page| (page, ready_page(entry)))
            .collect();
        let pair = Spread {
            first: 1,
            second: Some(2),
        };
        let shown = Spread::lone(0);
        let input = FrameInput {
            target: pair,
            shown: Some(ShownSpread {
                mode,
                spread: shown,
            }),
            mode,
            motion: Motion::Still,
            direction: Direction::Forward,
            len,
            limits: test_limits(),
            measurable: false,
        };
        let fit_on = |fit_pages: &'static [usize]| {
            move |page: usize| {
                let mut roles = Roles::default();
                if fit_pages.contains(&page) {
                    roles.set(Role::Fit);
                }
                roles
            }
        };

        assert_eq!(
            choose_presented(
                &input,
                &pages,
                |summary| *summary,
                fit_on(&[2])
            ),
            Some(shown)
        );
        pages.insert(1, ready_page(entry));
        assert_eq!(
            choose_presented(
                &input,
                &pages,
                |summary| *summary,
                fit_on(&[1, 2])
            ),
            Some(pair)
        );
    }

    #[test]
    fn commit_shown_requires_the_current_mode() {
        let mut position = ReadingPosition::new(5);
        assert!(!position.commit_shown(ReadingMode::Normal));
        assert_eq!(position.target(), 5);

        position.record_shown(ReadingMode::Normal, Some(Spread::lone(4)));
        assert!(position.commit_shown(ReadingMode::Normal));
        assert_eq!(position.target(), 4);
        assert!(!position.commit_shown(ReadingMode::Normal));

        position.set_target(5);
        assert!(!position.commit_shown(ReadingMode::Manga));
        assert_eq!(position.target(), 5);
    }

    #[test]
    fn skim_demands_the_next_spread() {
        let len = PageCount::new(40).expect("the test book is not empty");
        let pages: BTreeMap<usize, PageSummary> = BTreeMap::new();
        let frame = build_frame(
            &FrameInput {
                target: Spread::lone(20),
                shown: Some(ShownSpread {
                    mode: ReadingMode::Normal,
                    spread: Spread::lone(2),
                }),
                mode: ReadingMode::Normal,
                motion: Motion::Skimming,
                direction: Direction::Forward,
                len,
                limits: test_limits(),
                measurable: true,
            },
            &pages,
            |summary| *summary,
            |_| Roles::default(),
            |_| 1.0,
        );
        assert_eq!(frame.shown, Some(Spread::lone(2)));
        assert_eq!(frame.demand, Spread::lone(3));
        assert_ne!(frame.demand, Spread::lone(20));
    }

    #[test]
    fn playback_targets_need_an_animated_page_and_a_still_frame() {
        let len = PageCount::new(20).expect("the test book is not empty");
        let gif_page = PageInfo {
            dims: (2, 2),
            format: Format::Gif,
        };
        let still_image = PageInfo {
            dims: (2, 2),
            format: Format::Png,
        };
        let pages: BTreeMap<usize, PageSummary> = BTreeMap::from([
            (4, ready_page(gif_page)),
            (6, ready_page(gif_page)),
            (8, ready_page(still_image)),
            (
                10,
                PageSummary {
                    info: Some(gif_page),
                    faulted: true,
                },
            ),
            (12, ready_page(gif_page)),
        ]);
        let roles = |page: usize| {
            let mut page_roles = Roles::default();
            page_roles.set(if page == 6 {
                Role::Thumb
            } else {
                Role::Fit
            });
            page_roles
        };
        let frame_at = |page: usize, motion: Motion| {
            build_frame(
                &FrameInput {
                    target: Spread::lone(page),
                    shown: Some(ShownSpread {
                        mode: ReadingMode::Normal,
                        spread: Spread::lone(page),
                    }),
                    mode: ReadingMode::Normal,
                    motion,
                    direction: Direction::Forward,
                    len,
                    limits: test_limits(),
                    measurable: false,
                },
                &pages,
                |summary| *summary,
                roles,
                |_| 1.0,
            )
        };

        assert_eq!(
            frame_at(4, Motion::Still).playback_targets(),
            [Some((4, Role::Fit)), None]
        );
        assert_eq!(
            frame_at(6, Motion::Still).playback_targets(),
            [None, None],
            "a thumb-only page never plays"
        );
        assert_eq!(
            frame_at(8, Motion::Still).playback_targets(),
            [None, None],
            "a still image never plays"
        );
        assert_eq!(
            frame_at(10, Motion::Still).playback_targets(),
            [None, None],
            "a faulted page never plays"
        );
        assert_eq!(
            frame_at(12, Motion::Skimming).playback_targets(),
            [None, None],
            "skimming never plays"
        );
    }

    #[test]
    fn still_frames_never_request_thumbs() {
        let len = PageCount::new(10).expect("the test book is not empty");
        assert_eq!(
            cold_requests(Spread::lone(0), ReadingMode::Normal),
            vec![Request::new(0, Role::Fit)]
        );

        let entry = PageInfo {
            dims: (4000, 6000),
            format: Format::Png,
        };
        let pages = BTreeMap::from([(2, ready_page(entry))]);
        let frame = build_frame(
            &FrameInput {
                target: Spread::lone(2),
                shown: Some(ShownSpread {
                    mode: ReadingMode::Normal,
                    spread: Spread::lone(2),
                }),
                mode: ReadingMode::Normal,
                motion: Motion::Still,
                direction: Direction::Forward,
                len,
                limits: test_limits(),
                measurable: true,
            },
            &pages,
            |summary| *summary,
            |page| {
                let mut roles = Roles::default();
                if page == 2 {
                    roles.set(Role::Thumb);
                }
                roles
            },
            |_| 1.0,
        );
        assert_eq!(
            frame.into_requests(),
            vec![Request::new(2, Role::Fit), Request::new(2, Role::Full)]
        );
    }

    #[test]
    fn skim_skips_thumb_when_any_role_is_resident() {
        let len = PageCount::new(10).expect("the test book is not empty");
        let entry = PageInfo {
            dims: (800, 600),
            format: Format::Png,
        };
        let pages = BTreeMap::from([(4, ready_page(entry))]);
        let frame = build_frame(
            &FrameInput {
                target: Spread::lone(6),
                shown: Some(ShownSpread {
                    mode: ReadingMode::Normal,
                    spread: Spread::lone(4),
                }),
                mode: ReadingMode::Normal,
                motion: Motion::Skimming,
                direction: Direction::Forward,
                len,
                limits: test_limits(),
                measurable: true,
            },
            &pages,
            |summary| *summary,
            |page| {
                let mut roles = Roles::default();
                if page == 4 {
                    roles.set(Role::Fit);
                }
                roles
            },
            |_| 1.0,
        );
        assert_eq!(
            frame.into_requests(),
            vec![Request::new(5, Role::Thumb)]
        );
    }
}
