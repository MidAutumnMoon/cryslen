//! The reading model shared by parent, child, and renderer: page counts,
//! pairing, direction, roles, requests, position, and spread arithmetic.
//! Pure: no clocks, IO, or GL.

use std::num::NonZeroUsize;

use serde::Deserialize;
use serde::Serialize;

use crate::decode::Format;

/// A live book's page count. Sources never return an empty page list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageCount(NonZeroUsize);

impl PageCount {
    pub fn new(count: usize) -> Option<Self> {
        NonZeroUsize::new(count).map(Self)
    }

    pub fn get(self) -> usize {
        self.0.get()
    }
}

/// Page pairing and reading direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum ReadingMode {
    Normal,
    Manga,
    Comic,
}

impl ReadingMode {
    pub const ALL: [Self; 3] = [Self::Normal, Self::Manga, Self::Comic];

    #[must_use]
    pub fn pairs(self) -> bool {
        matches!(self, Self::Manga | Self::Comic)
    }

    #[must_use]
    pub fn rtl(self) -> bool {
        self == Self::Manga
    }

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Manga => "Manga",
            Self::Comic => "Comic",
        }
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

    pub fn seats(&self, mode: ReadingMode) -> [Option<usize>; 2] {
        seating(self.first, self.second, mode)
    }

    pub fn contains(&self, page: usize) -> bool {
        self.first == page || self.second == Some(page)
    }
}

/// Seat order as `[left, right]` under `mode`.
pub fn seating<T>(
    first: T,
    second: Option<T>,
    mode: ReadingMode,
) -> [Option<T>; 2] {
    if mode.rtl() {
        [second, Some(first)]
    } else {
        [Some(first), second]
    }
}

/// Reading direction of a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Direction {
    Forward,
    Backward,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum Motion {
    /// Settled reading: the prefetch head is worth fit images.
    Still,
    /// Turning faster than a fit image can arrive: only thumbs can show.
    Skimming,
}

/// A page image: a thumb for skims, a fit image for settled reading, or
/// the page's own pixels for zoom. Only the first two enter the store.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize,
)]
#[repr(usize)]
pub enum Role {
    Thumb,
    Fit,
    Full,
}

impl Role {
    pub const COUNT: usize = 3;
    /// Ascending size; array storage keyed by [`Self::index`] follows this
    /// order.
    pub const ALL: [Self; Self::COUNT] = [Self::Thumb, Self::Fit, Self::Full];

    /// Slot index for role-keyed arrays.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The role's stable name. The store keeps `thumb` and `view` for
    /// existing entries; full images bypass the store.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Thumb => "thumb",
            Self::Fit => "view",
            Self::Full => "native",
        }
    }
}

#[expect(
    clippy::indexing_slicing,
    reason = "the loop bound keeps the index in range"
)]
const _: () = {
    let mut index = 0;
    while index < Role::COUNT {
        assert!(
            Role::ALL[index].index() == index,
            "Role::ALL order must match Role::index"
        );
        index += 1;
    }
};

/// The roles present for one page.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Roles {
    present: [bool; Role::COUNT],
}

impl Roles {
    #[expect(
        clippy::indexing_slicing,
        reason = "Role::index is bounded by Role::COUNT"
    )]
    fn slot(self, role: Role) -> bool {
        self.present[role.index()]
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "Role::index is bounded by Role::COUNT"
    )]
    fn slot_mut(&mut self, role: Role) -> &mut bool {
        &mut self.present[role.index()]
    }

    pub fn set(&mut self, role: Role) {
        *self.slot_mut(role) = true;
    }

    pub fn clear(&mut self, role: Role) {
        *self.slot_mut(role) = false;
    }

    #[must_use]
    pub fn contains(self, role: Role) -> bool {
        self.slot(role)
    }

    #[must_use]
    pub fn any(self) -> bool {
        self.present.iter().any(|present| *present)
    }

    /// A fit or full image can fill a still seat.
    #[must_use]
    pub fn has_settled_image(self) -> bool {
        self.contains(Role::Fit) || self.contains(Role::Full)
    }

    /// The largest role present: full, then fit, then thumb.
    #[must_use]
    pub fn best(self) -> Option<Role> {
        Role::ALL
            .into_iter()
            .rev()
            .find(|role| self.contains(*role))
    }
}

/// One missing role. The app sends the whole set and the child sends no
/// acknowledgement: a request leaving the set means the app got it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[non_exhaustive]
pub struct Request {
    pub page: usize,
    pub role: Role,
}

impl Request {
    #[must_use]
    pub fn new(page: usize, role: Role) -> Self {
        Self { page, role }
    }
}

/// The page's own dims and format, independent of stored representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageInfo {
    pub dims: (u32, u32),
    pub format: Format,
}

/// A spread as shown, with the mode it was drawn under. A mode switch
/// makes an older pairing uncommittable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShownSpread {
    pub mode: ReadingMode,
    pub spread: Spread,
}

/// The target page and the last spread shown.
pub struct ReadingPosition {
    target: usize,
    shown: Option<ShownSpread>,
}

impl ReadingPosition {
    pub fn new(target: usize) -> Self {
        Self {
            target,
            shown: None,
        }
    }

    pub fn target(&self) -> usize {
        self.target
    }

    pub fn shown(&self) -> Option<Spread> {
        self.shown.map(|shown| shown.spread)
    }

    pub fn shown_with_mode(&self) -> Option<ShownSpread> {
        self.shown
    }

    pub fn set_target(&mut self, target: usize) {
        self.target = target;
    }

    pub fn record_shown(&mut self, mode: ReadingMode, shown: Option<Spread>) {
        self.shown = shown.map(|spread| ShownSpread { mode, spread });
    }

    pub fn commit_shown(&mut self, mode: ReadingMode) -> bool {
        let Some(shown) = self.shown else {
            return false;
        };
        if shown.mode != mode || shown.spread.first == self.target {
            return false;
        }
        self.target = shown.spread.first;
        true
    }
}
/// Spread containing `index` under `mode`, clamped to the book. An
/// odd tail shows one page (`second: None`).
pub fn spread_for(index: usize, mode: ReadingMode, len: PageCount) -> Spread {
    let len = len.get();
    let index = index.min(len - 1);
    if !mode.pairs() {
        return Spread::lone(index);
    }
    if index == 0 {
        return Spread::lone(0);
    }
    let first = spread_start(spread_number(index));
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
    len: PageCount,
) -> usize {
    let forward = direction == Direction::Forward;
    let len = len.get();
    let current = current.min(len - 1);
    if !mode.pairs() {
        let next = if forward {
            current + 1
        } else {
            current + len - 1
        };
        return next % len;
    }
    let spreads = spread_count(len);
    let number = spread_number(current);
    let next = if forward {
        (number + 1) % spreads
    } else {
        (number + spreads - 1) % spreads
    };
    spread_start(next)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModeSwitch {
    pub index: usize,
    pub pairing_changed: bool,
}

/// Where a switch from `old` to `new` lands. A pairing change moves
/// to the new spread's start. A direction-only change keeps the page.
pub fn switch_mode(
    old: ReadingMode,
    new: ReadingMode,
    index: usize,
    len: PageCount,
) -> ModeSwitch {
    if old.pairs() == new.pairs() {
        ModeSwitch {
            index,
            pairing_changed: false,
        }
    } else {
        ModeSwitch {
            index: spread_for(index, new, len).first,
            pairing_changed: true,
        }
    }
}

fn spread_number(page: usize) -> usize {
    page.div_ceil(2)
}

fn spread_start(number: usize) -> usize {
    if number > 0 {
        2 * number - 1
    } else {
        0
    }
}

fn spread_count(len: usize) -> usize {
    1 + (len - 1).div_ceil(2)
}

pub fn first_spread() -> usize {
    0
}

pub fn last_spread(mode: ReadingMode, len: PageCount) -> usize {
    if !mode.pairs() {
        return len.get() - 1;
    }
    spread_start(spread_count(len.get()) - 1)
}


#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;

    fn pages(count: usize) -> PageCount {
        PageCount::new(count).expect("test book is non-empty")
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
        for &mode in &ReadingMode::ALL {
            for len in 1_usize..=10 {
                for page in 0..=(len + 2) {
                    let clamped = page.min(len - 1);
                    let spread = spread_for(page, mode, pages(len));
                    assert!(
                        spread.contains(clamped),
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
            for &mode in &ReadingMode::ALL {
                for page in 0..=len {
                    for &direction in &[Direction::Backward, Direction::Forward] {
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
                            spread_for(page, mode, pages(len)).first,
                            "step then step back must round-trip: len {len} mode {mode:?} page {page} direction {direction:?}"
                        );
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
    fn mode_switch_relands_only_on_pairing_change() {
        assert_eq!(
            switch_mode(ReadingMode::Normal, ReadingMode::Manga, 2, pages(10)),
            ModeSwitch {
                index: 1,
                pairing_changed: true
            }
        );
        assert_eq!(
            switch_mode(ReadingMode::Normal, ReadingMode::Comic, 0, pages(10)),
            ModeSwitch {
                index: 0,
                pairing_changed: true
            }
        );
        assert_eq!(
            switch_mode(ReadingMode::Manga, ReadingMode::Comic, 5, pages(10)),
            ModeSwitch {
                index: 5,
                pairing_changed: false
            }
        );
        assert_eq!(
            switch_mode(ReadingMode::Manga, ReadingMode::Manga, 5, pages(10)),
            ModeSwitch {
                index: 5,
                pairing_changed: false
            }
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
        for mode in ReadingMode::ALL {
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
            for &mode in &ReadingMode::ALL {
                let last = last_spread(mode, pages(len));
                assert_eq!(
                    spread_for(last, mode, pages(len)).first,
                    last,
                    "len {len} {mode:?}"
                );
            }
        }
    }
}
