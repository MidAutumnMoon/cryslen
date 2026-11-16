//! Frame planning: the spread the screen shows and the roles the frame
//! keeps or asks for. Pure: no clocks, GL, or IO.

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
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PageSummary {
    pub info: Option<PageInfo>,
    pub faulted: bool,
    pub roles: Roles,
}

pub struct Frame {
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

/// Pass `None` for `on_screen_scale` before the viewport has a size; full
/// requests start once placement computes a scale.
pub fn build_frame(
    input: &FrameInput,
    on_screen: Spread,
    on_screen_scale: Option<f32>,
    page_state: impl Fn(usize) -> PageSummary,
) -> Frame {
    let playback = if input.motion == Motion::Still {
        on_screen.seats(input.mode).map(|seat| {
            let candidate = seat?;
            let entry = page_state(candidate);
            if entry.faulted {
                return None;
            }
            let info = entry.info?;
            if !info.format.may_animate() {
                return None;
            }
            let role = entry.roles.best()?;
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
    let requests = request_roles(
        demand,
        on_screen,
        input.mode,
        input.motion == Motion::Still,
        on_screen_scale,
        input.limits,
        &page_state,
    );
    Frame {
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

/// Initial `Fit` requests before textures or placement exist.
#[must_use]
pub fn cold_requests(spread: Spread, mode: ReadingMode) -> Vec<Request> {
    spread
        .seats(mode)
        .into_iter()
        .flatten()
        .map(|page| Request::new(page, Role::Fit))
        .collect()
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
    on_screen_scale: Option<f32>,
    limits: ImageLimits,
    page_state: impl Fn(usize) -> PageSummary,
) -> Vec<Request> {
    let mut requests = Vec::new();
    for spread in [demand, on_screen] {
        for page in spread.seats(mode).into_iter().flatten() {
            let entry = page_state(page);
            if entry.faulted {
                continue;
            }
            let resident = entry.roles;
            if still {
                if !resident.has_settled_image() {
                    push_request(&mut requests, page, Role::Fit);
                }
                if on_screen.contains(page)
                    && !resident.contains(Role::Full)
                    && on_screen_scale.is_some_and(|scale| {
                        entry.info.is_some_and(|info| {
                            full_wanted(scale, info.dims, limits)
                        })
                    })
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

/// Returns `input.target` once presentable, else the nearest presentable
/// start in `loaded_pages` between `input.shown` and `input.target`
/// (monotonic skim), else `input.shown`.
pub fn choose_presented(
    input: &FrameInput,
    loaded_pages: impl IntoIterator<Item = usize>,
    page_state: impl Fn(usize) -> PageSummary,
) -> Option<Spread> {
    let forward = input.direction == Direction::Forward;
    let skimming = input.motion == Motion::Skimming;
    let shown = input.shown.map(|entry| entry.spread);
    let can_show = |candidate: usize| {
        let entry = page_state(candidate);
        entry.faulted
            || (entry.info.is_some()
                && if skimming {
                    entry.roles.any()
                } else {
                    entry.roles.has_settled_image()
                })
    };
    let presentable = |spread: Spread| {
        spread.seats(input.mode).into_iter().flatten().all(can_show)
    };
    if presentable(input.target) {
        return Some(input.target);
    }
    let (low, high) = shown.map_or((0, usize::MAX), |previous| {
        (
            previous.first.min(input.target.first),
            previous.first.max(input.target.first),
        )
    });
    let mut best: Option<Spread> = None;
    for page in loaded_pages {
        // A paired spread starting at `high` may be visited via its second seat.
        if page < low || page > high.saturating_add(1) {
            continue;
        }
        if page_state(page).info.is_none() {
            continue;
        }
        let spread = browse::spread_for(page, input.mode, input.len);
        if spread.first < low || spread.first > high || !presentable(spread) {
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

    fn roles_with(role: Role) -> Roles {
        let mut roles = Roles::default();
        roles.set(role);
        roles
    }

    fn ready_page(info: PageInfo, roles: Roles) -> PageSummary {
        PageSummary {
            info: Some(info),
            faulted: false,
            roles,
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
            },
            Spread::lone(10),
            None,
            |_| PageSummary::default(),
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
        let input = FrameInput {
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
        };
        let page_state = |page| PageSummary {
            info: [1, 2].contains(&page).then_some(entry),
            ..Default::default()
        };
        let on_screen = choose_presented(&input, [1, 2], page_state)
            .unwrap_or(input.target);
        assert_eq!(on_screen, Spread::lone(1));
        let scale = if on_screen.first == 1 { 1.0 } else { 0.1 };
        let frame = build_frame(&input, on_screen, Some(scale), page_state);
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
        let known_pages = [0, 2, 4, 8];
        let state = |fit_pages: &'static [usize]| {
            move |page: usize| {
                let info = known_pages.contains(&page).then_some(entry);
                let roles = if fit_pages.contains(&page) {
                    roles_with(Role::Fit)
                } else {
                    Roles::default()
                };
                PageSummary {
                    info,
                    faulted: false,
                    roles,
                }
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
        };
        let lone = Spread::lone;

        // The target wins as soon as it has a pixel.
        assert_eq!(
            choose_presented(
                &input(lone(4), Some(lone(0))),
                known_pages,
                state(&[0, 2, 4]),
            ),
            Some(lone(4))
        );
        // Otherwise the nearest presentable spread between the shown
        // and the target; page 8 is known but has no pixels.
        assert_eq!(
            choose_presented(
                &input(lone(9), Some(lone(0))),
                known_pages,
                state(&[0, 2, 4]),
            ),
            Some(lone(4))
        );
        // Nothing between the two: the shown spread keeps its pixels.
        assert_eq!(
            choose_presented(
                &input(lone(9), Some(lone(2))),
                known_pages,
                state(&[0]),
            ),
            Some(lone(2))
        );
        // A cold start picks the nearest presentable spread.
        assert_eq!(
            choose_presented(
                &input(lone(5), None),
                known_pages,
                state(&[0, 2, 4]),
            ),
            Some(lone(4))
        );
        // A cold start presents only decoded spreads.
        assert_eq!(
            choose_presented(&input(lone(5), None), known_pages, state(&[])),
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
        };
        let state = |info_pages: &'static [usize], fit_pages: &'static [usize]| {
            move |page: usize| {
                let info = info_pages.contains(&page).then_some(entry);
                let roles = if fit_pages.contains(&page) {
                    roles_with(Role::Fit)
                } else {
                    Roles::default()
                };
                PageSummary {
                    info,
                    faulted: false,
                    roles,
                }
            }
        };

        assert_eq!(
            choose_presented(&input, [0, 2], state(&[0, 2], &[2])),
            Some(shown)
        );
        assert_eq!(
            choose_presented(&input, [0, 1, 2], state(&[0, 1, 2], &[1, 2])),
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
        let input = FrameInput {
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
        };
        let on_screen = choose_presented(&input, [], |_| PageSummary::default())
            .unwrap_or(input.target);
        assert_eq!(on_screen, Spread::lone(2));
        let frame =
            build_frame(&input, on_screen, Some(1.0), |_| PageSummary::default());
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
        let page_state = |page: usize| {
            let role = if page == 6 { Role::Thumb } else { Role::Fit };
            let roles = roles_with(role);
            match page {
                4 | 6 | 12 => ready_page(gif_page, roles),
                8 => ready_page(still_image, roles),
                10 => PageSummary {
                    info: Some(gif_page),
                    faulted: true,
                    roles,
                },
                _ => PageSummary::default(),
            }
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
                },
                Spread::lone(page),
                None,
                page_state,
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
            },
            Spread::lone(2),
            Some(1.0),
            |page| {
                if page == 2 {
                    ready_page(entry, roles_with(Role::Thumb))
                } else {
                    PageSummary::default()
                }
            },
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
            },
            Spread::lone(4),
            Some(1.0),
            |page| {
                if page == 4 {
                    ready_page(entry, roles_with(Role::Fit))
                } else {
                    PageSummary::default()
                }
            },
        );
        assert_eq!(
            frame.into_requests(),
            vec![Request::new(5, Role::Thumb)]
        );
    }
}
