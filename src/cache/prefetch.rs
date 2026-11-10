//! Prefetch planning: which spreads the worker should have ready.

use crate::browse;
use crate::browse::Direction;
use crate::browse::ReadingMode;
use crate::browse::PageCount;
use crate::browse::Spread;

pub const AHEAD_PAGES: usize = 512;
pub const BEHIND_PAGES: usize = 64;
pub const HEAD_SPREADS: usize = 3;

/// An ordered prefetch list for one navigation step.
#[derive(Debug, Clone)]
pub struct Prefetch {
    spreads: Vec<Spread>,
    head: usize,
}

impl Prefetch {
    /// Prefetch around `origin` for a step in `direction`.
    pub fn build(
        origin: Spread,
        direction: Direction,
        mode: ReadingMode,
        len: PageCount,
    ) -> Self {
        let (ahead, behind) = match direction {
            Direction::Forward => (AHEAD_PAGES, BEHIND_PAGES),
            Direction::Backward => (BEHIND_PAGES, AHEAD_PAGES),
        };
        let increasing =
            walk(origin.first, Direction::Forward, mode, len, ahead);
        let decreasing =
            walk(origin.first, Direction::Backward, mode, len, behind);
        let mut spreads =
            Vec::with_capacity(1 + increasing.len() + decreasing.len());
        spreads.push(origin);
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

    /// The leading spreads that need a fit image rather than a
    /// thumbnail.
    pub fn head(&self) -> &[Spread] {
        self.spreads.get(..self.head).unwrap_or_default()
    }
}

fn walk(
    start: usize,
    direction: Direction,
    mode: ReadingMode,
    len: PageCount,
    limit: usize,
) -> Vec<Spread> {
    let mut spreads = Vec::new();
    let mut current = start;
    loop {
        let next = browse::advance(current, direction, mode, len);
        let forward = direction == Direction::Forward;
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
        spreads.push(browse::spread_for(next, mode, len));
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

    fn starts(prefetch: &Prefetch) -> Vec<usize> {
        prefetch.spreads().iter().map(|spread| spread.first).collect()
    }

    #[test]
    fn single_mode_walks_out_from_the_origin() {
        let book = pages(10);
        let origin = browse::spread_for(4, ReadingMode::Normal, book);
        let forward = Prefetch::build(
            origin,
            Direction::Forward,
            ReadingMode::Normal,
            book,
        );
        assert_eq!(starts(&forward), vec![4, 5, 3, 6, 2, 7, 1, 8, 0, 9]);
        assert!(forward
            .spreads()
            .iter()
            .all(|spread| spread.second.is_none()));

        let backward = Prefetch::build(
            origin,
            Direction::Backward,
            ReadingMode::Normal,
            book,
        );
        assert_eq!(starts(&backward), vec![4, 3, 5, 2, 6, 1, 7, 0, 8, 9]);
    }

    #[test]
    fn the_ends_do_not_wrap() {
        let long_book = pages(1200);
        let opening = browse::spread_for(0, ReadingMode::Normal, long_book);
        let before = Prefetch::build(
            opening,
            Direction::Backward,
            ReadingMode::Normal,
            long_book,
        );
        assert_eq!(before.spreads().first(), Some(&opening));
        assert_eq!(before.spreads().len(), 1 + BEHIND_PAGES);
        assert!(before
            .spreads()
            .iter()
            .all(|spread| spread.first <= BEHIND_PAGES));

        let closing = browse::spread_for(1199, ReadingMode::Normal, long_book);
        let after = Prefetch::build(
            closing,
            Direction::Forward,
            ReadingMode::Normal,
            long_book,
        );
        assert_eq!(after.spreads().first(), Some(&closing));
        assert_eq!(after.spreads().len(), 1 + BEHIND_PAGES);
        assert!(after
            .spreads()
            .iter()
            .all(|spread| spread.first >= 1199 - BEHIND_PAGES));

        let single = pages(1);
        let only = browse::spread_for(0, ReadingMode::Manga, single);
        for direction in [Direction::Forward, Direction::Backward] {
            let alone =
                Prefetch::build(only, direction, ReadingMode::Manga, single);
            assert_eq!(alone.spreads(), &[only]);
            assert_eq!(alone.head(), &[only]);
        }
    }

    #[test]
    fn bounds_are_respected_and_unique() {
        let book = pages(1200);
        let origin = browse::spread_for(600, ReadingMode::Normal, book);
        let prefetch = Prefetch::build(
            origin,
            Direction::Forward,
            ReadingMode::Normal,
            book,
        );
        assert_eq!(prefetch.spreads().len(), 1 + AHEAD_PAGES + BEHIND_PAGES);
        assert_eq!(prefetch.spreads().first(), Some(&origin));
        assert_eq!(prefetch.head().len(), HEAD_SPREADS);
        assert!(prefetch.spreads().iter().all(|spread| {
            (600 - BEHIND_PAGES..=600 + AHEAD_PAGES)
                .contains(&spread.first)
        }));
        let unique: BTreeSet<usize> =
            prefetch.spreads().iter().map(|spread| spread.first).collect();
        assert_eq!(unique.len(), prefetch.spreads().len());
    }
}
