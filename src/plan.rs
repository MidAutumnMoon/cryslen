//! Prefetch planning: which spreads the worker should have ready.

use crate::view;
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
        let increasing = walk(anchor.first, true, mode, len, ahead);
        let decreasing = walk(anchor.first, false, mode, len, behind);
        let mut spreads =
            Vec::with_capacity(1 + increasing.len() + decreasing.len());
        spreads.push(anchor);
        for distance in 0..increasing.len().max(decreasing.len()) {
            let (travel, opposite) = match direction {
                Direction::Forward => {
                    (increasing.get(distance), decreasing.get(distance))
                }
                Direction::Backward => {
                    (decreasing.get(distance), increasing.get(distance))
                }
            };
            spreads.extend(travel);
            spreads.extend(opposite);
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
        let next = view::advance(current, forward, mode, len);
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
        spreads.push(view::spread_for(next, mode, len));
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
        let anchor = view::spread_for(4, ViewMode::Normal, book);
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
        let opening = view::spread_for(0, ViewMode::Normal, long_book);
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

        let closing = view::spread_for(1199, ViewMode::Normal, long_book);
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
        let only = view::spread_for(0, ViewMode::Manga, single);
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
        let anchor = view::spread_for(600, ViewMode::Normal, book);
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
