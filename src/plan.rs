//! Prefetch planning: which spreads the worker should have ready.
//!
//! Decoding runs ahead of the reader, and the order matters: the next
//! spread is the one the reader is most likely to land on, so it goes
//! first, and the rest of the window is a hedge. [`Plan::build`] lists the
//! anchor, then both neighbors at distance one, then distance two, and so
//! on, alternating sides, with the direction of travel ahead of the
//! opposite side at equal distance. The walk never wraps, so a plan holds
//! each page at most once and stops at the book's ends.
//!
//! [`AHEAD_PAGES`] and [`BEHIND_PAGES`] bound the window asymmetrically
//! because readers move forward far more often than back: the direction
//! of travel gets the wide budget, the opposite side the narrow one. They
//! count page indices from the anchor's first page: a spread joins the
//! plan only if it starts within the budget, which keeps the plan's size
//! near the budget rather than doubling it for pair modes.
//!
//! The first [`HEAD_SPREADS`] spreads are [`Plan::head`]: the ones that
//! need a view image, while the tail is only worth a thumbnail. A short
//! book makes the head shorter, so it is never assumed to be full.

use crate::view::advance;
use crate::view::spread_for;
use crate::view::PageCount;
use crate::view::Spread;
use crate::view::ViewMode;

pub const AHEAD_PAGES: usize = 512;
pub const BEHIND_PAGES: usize = 64;
pub const HEAD_SPREADS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
}

/// An ordered prefetch list for one navigation step.
#[derive(Debug, Clone)]
pub struct Plan {
    spreads: Vec<Spread>,
    head: usize,
}

impl Plan {
    /// Plan around `anchor` for a step in `direction`.
    ///
    /// The anchor comes first, then neighbors by increasing page distance,
    /// the direction of travel's side first at equal distance, bounded to
    /// [`AHEAD_PAGES`] in the direction of travel and [`BEHIND_PAGES`]
    /// against it. The walk stops at the ends of the book instead of
    /// wrapping, so an end page has no neighbors on that side.
    pub fn build(
        anchor: Spread,
        direction: Direction,
        mode: ViewMode,
        len: PageCount,
    ) -> Self {
        let (ahead, behind) = match direction {
            Direction::Forward => (AHEAD_PAGES, BEHIND_PAGES),
            Direction::Backward => (BEHIND_PAGES, AHEAD_PAGES),
        };
        let forward = walk(anchor.first, true, mode, len, ahead);
        let backward = walk(anchor.first, false, mode, len, behind);
        let mut spreads =
            Vec::with_capacity(1 + forward.len() + backward.len());
        spreads.push(anchor);
        for distance in 0..forward.len().max(backward.len()) {
            let (first, second) = match direction {
                Direction::Forward => {
                    (forward.get(distance), backward.get(distance))
                }
                Direction::Backward => {
                    (backward.get(distance), forward.get(distance))
                }
            };
            spreads.extend(first);
            spreads.extend(second);
        }
        let head = spreads.len().min(HEAD_SPREADS);
        Self { spreads, head }
    }

    pub fn spreads(&self) -> &[Spread] {
        &self.spreads
    }

    /// The leading spreads that need a view image rather than a
    /// thumbnail.
    pub fn head(&self) -> &[Spread] {
        self.spreads.get(..self.head).unwrap_or_default()
    }
}

/// Spreads reachable from `start` in one direction, nearest first, stopping
/// before the walk wraps around the book or exceeds `limit` pages.
fn walk(
    start: usize,
    forward: bool,
    mode: ViewMode,
    len: PageCount,
    limit: usize,
) -> Vec<Spread> {
    let mut spreads = Vec::new();
    let mut current = start;
    loop {
        let next = advance(current, forward, mode, len);
        let wrapped = if forward {
            next <= current
        } else {
            next >= current
        };
        if wrapped {
            break;
        }
        let distance = if forward { next - start } else { start - next };
        if distance > limit {
            break;
        }
        spreads.push(spread_for(next, mode, len));
        current = next;
    }
    spreads
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn pages(count: usize) -> PageCount {
        PageCount::new(count).expect("test book is non-empty")
    }

    fn starts(plan: &Plan) -> Vec<usize> {
        plan.spreads().iter().map(|spread| spread.first).collect()
    }

    #[test]
    fn single_mode_walks_out_from_the_anchor() {
        let book = pages(10);
        let anchor = spread_for(4, ViewMode::Normal, book);
        let forward = Plan::build(
            anchor,
            Direction::Forward,
            ViewMode::Normal,
            book,
        );
        assert_eq!(starts(&forward), vec![4, 5, 3, 6, 2, 7, 1, 8, 0, 9]);
        assert!(forward
            .spreads()
            .iter()
            .all(|spread| spread.second.is_none()));

        let backward = Plan::build(
            anchor,
            Direction::Backward,
            ViewMode::Normal,
            book,
        );
        assert_eq!(starts(&backward), vec![4, 3, 5, 2, 6, 1, 7, 0, 8, 9]);
    }

    #[test]
    fn the_ends_do_not_wrap() {
        let long_book = pages(1200);
        let opening = spread_for(0, ViewMode::Normal, long_book);
        let before = Plan::build(
            opening,
            Direction::Backward,
            ViewMode::Normal,
            long_book,
        );
        assert_eq!(before.spreads().first(), Some(&opening));
        assert_eq!(before.spreads().len(), 1 + BEHIND_PAGES);
        assert!(before
            .spreads()
            .iter()
            .all(|spread| spread.first <= BEHIND_PAGES));

        let closing = spread_for(1199, ViewMode::Normal, long_book);
        let after = Plan::build(
            closing,
            Direction::Forward,
            ViewMode::Normal,
            long_book,
        );
        assert_eq!(after.spreads().first(), Some(&closing));
        assert_eq!(after.spreads().len(), 1 + BEHIND_PAGES);
        assert!(after
            .spreads()
            .iter()
            .all(|spread| spread.first >= 1199 - BEHIND_PAGES));

        let single = pages(1);
        let only = spread_for(0, ViewMode::Manga, single);
        for direction in [Direction::Forward, Direction::Backward] {
            let alone =
                Plan::build(only, direction, ViewMode::Manga, single);
            assert_eq!(alone.spreads(), &[only]);
            assert_eq!(alone.head(), &[only]);
        }
    }

    #[test]
    fn bounds_are_respected_and_unique() {
        let book = pages(1200);
        let anchor = spread_for(600, ViewMode::Normal, book);
        let plan = Plan::build(
            anchor,
            Direction::Forward,
            ViewMode::Normal,
            book,
        );
        assert_eq!(plan.spreads().len(), 1 + AHEAD_PAGES + BEHIND_PAGES);
        assert_eq!(plan.spreads().first(), Some(&anchor));
        assert_eq!(plan.head().len(), HEAD_SPREADS);
        assert!(plan.spreads().iter().all(|spread| {
            (600 - BEHIND_PAGES..=600 + AHEAD_PAGES)
                .contains(&spread.first)
        }));
        let unique: BTreeSet<usize> =
            plan.spreads().iter().map(|spread| spread.first).collect();
        assert_eq!(unique.len(), plan.spreads().len());
    }
}
