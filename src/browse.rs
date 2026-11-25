use std::num::NonZeroUsize;
use std::ops::Deref;

use enum_map::Enum;
use enum_map::EnumMap;
use serde::Deserialize;
use serde::Serialize;
use serde::Serializer;
use strum::Display;
use strum::IntoStaticStr;
use strum::VariantArray;

/// Non-empty page names in reading order.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(try_from = "Vec<String>")]
pub struct PageNames {
    names: Vec<String>,
    count: NonZeroUsize,
}

impl PageNames {
    pub fn new(names: Vec<String>) -> Option<Self> {
        let count = NonZeroUsize::new(names.len())?;
        Some(Self { names, count })
    }

    pub fn count(&self) -> NonZeroUsize {
        self.count
    }
}

impl Deref for PageNames {
    type Target = [String];

    fn deref(&self) -> &Self::Target {
        &self.names
    }
}

impl<const N: usize> PartialEq<[&str; N]> for PageNames {
    fn eq(&self, other: &[&str; N]) -> bool {
        self.names == other
    }
}

impl TryFrom<Vec<String>> for PageNames {
    type Error = &'static str;

    fn try_from(names: Vec<String>) -> Result<Self, Self::Error> {
        Self::new(names).ok_or("the book has no pages")
    }
}

impl Serialize for PageNames {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.names.serialize(serializer)
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Display,
    VariantArray
)]
pub enum ReadingMode {
    #[default]
    Normal,
    Manga,
    Comic,
}

impl ReadingMode {
    #[must_use]
    pub fn pairs(self) -> bool {
        matches!(self, Self::Manga | Self::Comic)
    }

    #[must_use]
    pub fn rtl(self) -> bool {
        self == Self::Manga
    }

    #[must_use]
    pub fn shortcut(self) -> &'static str {
        match self {
            Self::Normal => "n",
            Self::Manga => "m",
            Self::Comic => "c",
        }
    }

    #[must_use]
    pub fn effect(self) -> &'static str {
        match self {
            Self::Normal => "left to right, single page",
            Self::Manga => "right to left, double page",
            Self::Comic => "left to right, double page",
        }
    }
}

/// One screen's pages, 0-based: the cover alone, then pairs (`1+2`,
/// `3+4`, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spread {
    pub first: usize,
    pub second: Option<usize>,
}

impl Spread {
    pub const fn lone(first: usize) -> Self {
        Self {
            first,
            second: None,
        }
    }

    /// Seat order as `[left, right]` under `mode`.
    pub fn seats(&self, mode: ReadingMode) -> [Option<usize>; 2] {
        if mode.rtl() {
            [self.second, Some(self.first)]
        } else {
            [Some(self.first), self.second]
        }
    }

    pub fn pages(self) -> impl Iterator<Item = usize> {
        [Some(self.first), self.second].into_iter().flatten()
    }

    pub fn contains(&self, page: usize) -> bool {
        self.first == page || self.second == Some(page)
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    VariantArray
)]
pub enum Direction {
    Forward,
    Backward,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Display,
    Enum,
    IntoStaticStr,
    VariantArray
)]
pub enum Role {
    #[strum(serialize = "thumb")]
    Thumb,
    #[strum(serialize = "view")]
    Fit,
    #[strum(serialize = "native")]
    Full,
}

/// Resolution tier produced by the child decode worker.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize, Display
)]
pub enum WorkRole {
    #[strum(serialize = "view")]
    Fit,
    #[strum(serialize = "native")]
    Full,
}

impl From<WorkRole> for Role {
    fn from(role: WorkRole) -> Self {
        match role {
            WorkRole::Fit => Self::Fit,
            WorkRole::Full => Self::Full,
        }
    }
}

/// The roles present for one page.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Roles {
    present: EnumMap<Role, bool>,
}

impl Roles {
    pub fn set(&mut self, role: Role) {
        self.present[role] = true;
    }

    #[must_use]
    pub fn contains(self, role: Role) -> bool {
        self.present[role]
    }

    #[must_use]
    pub fn any(self) -> bool {
        self.present.values().any(|present| *present)
    }

    /// A fit or full image can fill a settled seat.
    #[must_use]
    pub fn has_settled_image(self) -> bool {
        self.contains(Role::Fit) || self.contains(Role::Full)
    }

    /// The largest role present.
    #[must_use]
    pub fn best(self) -> Option<Role> {
        use strum::VariantArray as _;

        Role::VARIANTS
            .iter()
            .rev()
            .copied()
            .find(|&role| self.contains(role))
    }
}

/// One missing worker role. The app sends the whole priority plan and the
/// child sends no acknowledgement: an item leaving the plan means the app
/// no longer needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct Request {
    pub page: usize,
    pub role: WorkRole,
}

/// Spread containing `index` under `mode`, clamped to the book. An
/// odd tail shows one page (`second: None`).
pub fn spread_for(
    index: usize,
    mode: ReadingMode,
    len: NonZeroUsize,
) -> Spread {
    let len = len.get();
    let index = index.min(len - 1);
    if !mode.pairs() {
        return Spread::lone(index);
    }
    if index == 0 {
        return Spread::lone(0);
    }
    let first = spread_start(spread_number(index, mode));
    Spread {
        first,
        second: (first + 1 < len).then_some(first + 1),
    }
}

/// Next/previous spread start from `current`, wrapping at the ends.
/// Always returns a spread *start*.
pub fn advance(
    current: usize,
    direction: Direction,
    mode: ReadingMode,
    len: NonZeroUsize,
) -> usize {
    step(current, direction, mode, len).unwrap_or_else(
        || match direction {
            Direction::Forward => 0,
            Direction::Backward => last_spread(mode, len),
        },
    )
}

/// Next/previous spread start from `current`, or `None` at the book ends.
pub fn step(
    current: usize,
    direction: Direction,
    mode: ReadingMode,
    len: NonZeroUsize,
) -> Option<usize> {
    let len = len.get();
    let current = current.min(len - 1);
    if !mode.pairs() {
        return match direction {
            Direction::Forward => {
                (current + 1 < len).then_some(current + 1)
            }
            Direction::Backward => current.checked_sub(1),
        };
    }
    let number = spread_number(current, mode);
    let next = match direction {
        Direction::Forward => {
            let candidate = number + 1;
            (candidate < spread_count(len)).then_some(candidate)?
        }
        Direction::Backward => number.checked_sub(1)?,
    };
    Some(spread_start(next))
}

fn spread_number(page: usize, mode: ReadingMode) -> usize {
    if mode.pairs() { page.div_ceil(2) } else { page }
}

fn spread_start(number: usize) -> usize {
    if number > 0 { 2 * number - 1 } else { 0 }
}

fn spread_count(len: usize) -> usize {
    1 + (len - 1).div_ceil(2)
}

pub fn last_spread(mode: ReadingMode, len: NonZeroUsize) -> usize {
    if !mode.pairs() {
        return len.get() - 1;
    }
    spread_start(spread_count(len.get()) - 1)
}

pub fn spread_distance(
    from: usize,
    to: usize,
    mode: ReadingMode,
) -> usize {
    spread_number(from, mode).abs_diff(spread_number(to, mode))
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;

    fn pages(count: usize) -> NonZeroUsize {
        NonZeroUsize::new(count).expect("test book is non-empty")
    }

    #[test]
    fn spread_structure() {
        // Anchor: page 1 opens the first pair.
        assert_eq!(
            spread_for(1, ReadingMode::Manga, pages(10)),
            Spread {
                first: 1,
                second: Some(2)
            }
        );
        for &mode in ReadingMode::VARIANTS {
            for len in 1_usize..=10 {
                for page in 0..=(len + 2) {
                    let clamped = page.min(len - 1);
                    let spread = spread_for(page, mode, pages(len));
                    assert!(
                        spread.contains(clamped),
                        "mode {mode:?} len {len} page {page}"
                    );
                    assert_eq!(
                        spread_distance(clamped, spread.first, mode),
                        0,
                        "mode {mode:?} len {len} page {page}"
                    );
                    if mode.pairs() {
                        assert!(
                            spread.first == 0 || spread.first % 2 == 1,
                            "pairs start at 0 or an odd page: mode {mode:?} len {len} page {page}"
                        );
                        assert_eq!(
                            spread.second.is_none(),
                            spread.first == 0 || spread.first + 1 >= len,
                            "the cover is alone and an odd tail ends alone: mode {mode:?} len {len} page {page}"
                        );
                    } else {
                        assert_eq!(
                            spread.first, clamped,
                            "mode {mode:?} len {len} page {page}"
                        );
                        assert_eq!(
                            spread.second, None,
                            "mode {mode:?} len {len} page {page}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn advance_steps_between_spread_starts() {
        for len in [1_usize, 2, 3, 6, 9, 10] {
            for &mode in ReadingMode::VARIANTS {
                for page in 0..=len {
                    for &direction in Direction::VARIANTS {
                        let start =
                            spread_for(page, mode, pages(len)).first;
                        let stepped =
                            advance(page, direction, mode, pages(len));
                        assert_eq!(
                            spread_for(stepped, mode, pages(len)).first,
                            stepped,
                            "stepped page must be a spread start: len {len} mode {mode:?} page {page} direction {direction:?}"
                        );
                        let back = match direction {
                            Direction::Forward => Direction::Backward,
                            Direction::Backward => Direction::Forward,
                        };
                        assert_eq!(
                            advance(stepped, back, mode, pages(len)),
                            start,
                            "step then step back must round-trip: len {len} mode {mode:?} page {page} direction {direction:?}"
                        );
                        let at_end = match direction {
                            Direction::Forward => {
                                start == last_spread(mode, pages(len))
                            }
                            Direction::Backward => start == 0,
                        };
                        let bounded =
                            step(page, direction, mode, pages(len));
                        assert_eq!(
                            bounded,
                            (!at_end).then_some(stepped),
                            "bounded step must match advance except at book ends: len {len} mode {mode:?} page {page} direction {direction:?}"
                        );
                        if let Some(next) = bounded {
                            assert_eq!(
                                spread_distance(start, next, mode),
                                1,
                                "bounded step moves one spread: len {len} mode {mode:?} page {page} direction {direction:?}"
                            );
                        }
                    }
                }
            }
        }
        // Direction anchors: a mid-spread step lands on the neighbor spread's start.
        assert_eq!(
            advance(1, Direction::Forward, ReadingMode::Manga, pages(10)),
            3
        );
        assert_eq!(
            advance(4, Direction::Forward, ReadingMode::Normal, pages(10)),
            5
        );
        // The cover's next spread starts at 1, not 2.
        assert_eq!(
            advance(0, Direction::Forward, ReadingMode::Manga, pages(10)),
            1
        );
    }

    #[test]
    fn seats_follow_reading_direction() {
        let spread = Spread {
            first: 1,
            second: Some(2),
        };
        assert_eq!(spread.seats(ReadingMode::Normal), [Some(1), Some(2)]);
        assert_eq!(spread.seats(ReadingMode::Comic), [Some(1), Some(2)]);
        assert_eq!(spread.seats(ReadingMode::Manga), [Some(2), Some(1)]);
        let lone = Spread::lone(5);
        assert_eq!(lone.seats(ReadingMode::Normal), [Some(5), None]);
        // A lone Comic page mirrors Normal, not the rtl modes.
        assert_eq!(lone.seats(ReadingMode::Comic), [Some(5), None]);
        assert_eq!(lone.seats(ReadingMode::Manga), [None, Some(5)]);
    }

    #[test]
    fn first_and_last_spreads() {
        // Single page: both ends are the cover.
        for &mode in ReadingMode::VARIANTS {
            assert_eq!(last_spread(mode, pages(1)), 0);
        }
        // Pairs: the final spread starts on the last odd page.
        assert_eq!(last_spread(ReadingMode::Manga, pages(10)), 9);
        assert_eq!(last_spread(ReadingMode::Comic, pages(11)), 9);
        // An odd tail ends alone on the previous pair start.
        assert_eq!(last_spread(ReadingMode::Manga, pages(12)), 11);
        // Normal: the last page.
        assert_eq!(last_spread(ReadingMode::Normal, pages(10)), 9);
        for len in 1_usize..=12 {
            for &mode in ReadingMode::VARIANTS {
                let last = last_spread(mode, pages(len));
                assert_eq!(
                    spread_for(last, mode, pages(len)).first,
                    last,
                    "len {len} {mode:?}"
                );
            }
        }
    }

    #[test]
    fn roles_prefer_the_largest_role() {
        let mut roles = Roles::default();
        assert_eq!(roles.best(), None);
        assert!(!roles.any());

        roles.set(Role::Thumb);
        assert_eq!(roles.best(), Some(Role::Thumb));
        assert!(roles.contains(Role::Thumb));
        assert!(roles.any());
        assert!(!roles.has_settled_image());

        roles.set(Role::Fit);
        assert_eq!(roles.best(), Some(Role::Fit));
        assert!(roles.has_settled_image());

        roles.set(Role::Full);
        assert_eq!(roles.best(), Some(Role::Full));
        assert!(roles.contains(Role::Full));
    }
}
