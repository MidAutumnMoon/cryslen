//! Role-keyed VRAM cache with per-role byte budgets and priority eviction.

use std::collections::HashMap;

use enum_map::Enum;
use enum_map::EnumMap;
use strum::Display;
use strum::IntoStaticStr;
use strum::VariantArray;

use crate::image::pixels::PageInfo;

const ROLE_COVER_MARGIN_PX: f64 = 1.0;

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
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

impl Role {
    /// A fit or full image can fill a settled seat.
    #[must_use]
    pub const fn is_settled(self) -> bool {
        !matches!(self, Self::Thumb)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertOutcome {
    Kept,
    EvictedStaleThumb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EvictionPriority {
    pub spread_distance: usize,
    pub is_behind: bool,
}

pub trait Slot {
    fn dims(&self) -> (u32, u32);
    fn page_info(&self) -> PageInfo;
    fn bytes(&self) -> usize;
}

struct PageSlots<Item> {
    roles: EnumMap<Role, Option<Item>>,
    playback: Option<Item>,
}

impl<Item> Default for PageSlots<Item> {
    fn default() -> Self {
        Self {
            roles: EnumMap::default(),
            playback: None,
        }
    }
}

impl<Item: Slot> PageSlots<Item> {
    fn best_role(&self) -> Option<Role> {
        self.roles
            .iter()
            .rev()
            .find_map(|(role, slot)| slot.as_ref().map(|_| role))
    }

    fn has_any(&self) -> bool {
        self.playback.is_some() || self.roles.values().any(Option::is_some)
    }

    fn page_info(&self) -> Option<PageInfo> {
        if let Some(playback) = &self.playback {
            return Some(playback.page_info());
        }
        let best = self.best_role()?;
        self.roles[best].as_ref().map(Slot::page_info)
    }

    /// Keep `Fit` whenever it covers `quad_size_px` so zooming back out
    /// avoids re-filtering `Full`. A live playback frame takes precedence.
    fn best_slot(&self, quad_size_px: (f64, f64)) -> Option<&Item> {
        if let Some(playback) = self.playback.as_ref() {
            return Some(playback);
        }
        if let Some(fit) = self.roles[Role::Fit].as_ref() {
            let dims = fit.dims();
            if f64::from(dims.0) + ROLE_COVER_MARGIN_PX >= quad_size_px.0
                && f64::from(dims.1) + ROLE_COVER_MARGIN_PX
                    >= quad_size_px.1
            {
                return Some(fit);
            }
        }
        self.roles[self.best_role()?].as_ref()
    }
}

/// Pure role/budget cache parameterized by slot type (`PageTexture` in
/// production, lightweight test slots in unit tests).
pub struct SlotTable<Item> {
    pages: HashMap<usize, PageSlots<Item>>,
    role_bytes: EnumMap<Role, usize>,
}

impl<Item> Default for SlotTable<Item> {
    fn default() -> Self {
        Self {
            pages: HashMap::new(),
            role_bytes: EnumMap::default(),
        }
    }
}

impl<Item: Slot> SlotTable<Item> {
    #[must_use]
    pub fn best_role(&self, page: usize) -> Option<Role> {
        self.pages.get(&page)?.best_role()
    }

    #[must_use]
    pub fn has_settled_image(&self, page: usize) -> bool {
        self.best_role(page).is_some_and(Role::is_settled)
    }

    #[must_use]
    pub fn has(&self, page: usize, role: Role) -> bool {
        self.pages
            .get(&page)
            .is_some_and(|slots| slots.roles[role].is_some())
    }

    #[must_use]
    pub fn has_any(&self, page: usize) -> bool {
        self.pages.get(&page).is_some_and(PageSlots::has_any)
    }

    #[must_use]
    pub fn page_info(&self, page: usize) -> Option<PageInfo> {
        self.pages.get(&page)?.page_info()
    }

    #[must_use]
    pub fn best_slot(
        &self,
        page: usize,
        quad_size_px: (f64, f64),
    ) -> Option<&Item> {
        self.pages.get(&page)?.best_slot(quad_size_px)
    }

    #[must_use]
    pub fn playback(&self, page: usize) -> Option<&Item> {
        self.pages.get(&page)?.playback.as_ref()
    }

    #[must_use]
    pub fn average_role_bytes(&self, role: Role) -> Option<usize> {
        let count = self
            .pages
            .values()
            .filter(|slots| slots.roles[role].is_some())
            .count();
        self.role_bytes[role].checked_div(count)
    }

    /// Insert `slot` at `(page, role)`. When a fresh `Fit`/`Full` disagrees
    /// with a cached `Thumb` (e.g. an in-place directory edit), the stale
    /// `Thumb` is evicted so layout and status adopt the real dimensions.
    pub fn insert(
        &mut self,
        page: usize,
        role: Role,
        slot: Item,
    ) -> InsertOutcome {
        let incoming_info = slot.page_info();
        let incoming_bytes = slot.bytes();
        let entry = self.pages.entry(page).or_default();
        if let Some(previous) = entry.roles[role].replace(slot) {
            self.role_bytes[role] =
                self.role_bytes[role].saturating_sub(previous.bytes());
        }
        self.role_bytes[role] =
            self.role_bytes[role].saturating_add(incoming_bytes);

        if role != Role::Thumb
            && let Some(thumb) = &entry.roles[Role::Thumb]
        {
            let thumb_info = thumb.page_info();
            if (thumb_info.dims != incoming_info.dims
                || thumb_info.format != incoming_info.format)
                && let Some(removed) = entry.roles[Role::Thumb].take()
            {
                self.role_bytes[Role::Thumb] = self.role_bytes
                    [Role::Thumb]
                    .saturating_sub(removed.bytes());
                return InsertOutcome::EvictedStaleThumb;
            }
        }
        InsertOutcome::Kept
    }

    pub fn set_playback(&mut self, page: usize, slot: Item) {
        self.pages.entry(page).or_default().playback = Some(slot);
    }

    pub fn clear(&mut self) {
        self.pages.clear();
        self.role_bytes = EnumMap::default();
    }

    pub fn evict_to_budget(
        &mut self,
        role: Role,
        budget_bytes: usize,
        is_pinned: &dyn Fn(usize) -> bool,
        priority: &dyn Fn(usize) -> EvictionPriority,
    ) {
        while self.role_bytes[role] > budget_bytes {
            let victim = self
                .pages
                .iter()
                .filter(|&(&page, slots)| {
                    slots.roles[role].is_some() && !is_pinned(page)
                })
                .max_by_key(|&(&page, _)| (priority(page), page))
                .map(|(&page, _)| page);
            let Some(page) = victim else {
                break;
            };
            if let Some(slots) = self.pages.get_mut(&page) {
                if let Some(removed) = slots.roles[role].take() {
                    self.role_bytes[role] = self.role_bytes[role]
                        .saturating_sub(removed.bytes());
                }
                if !slots.has_any() {
                    self.pages.remove(&page);
                }
            }
        }
    }

    pub fn retain_playback(
        &mut self,
        keep_playback: &dyn Fn(usize) -> bool,
    ) {
        self.pages.retain(|&page, slots| {
            if slots.playback.is_some() && !keep_playback(page) {
                slots.playback = None;
            }
            slots.has_any()
        });
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;
    use crate::image::pixels::Format;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestSlot {
        dims: (u32, u32),
        info: PageInfo,
        bytes: usize,
        tag: &'static str,
    }

    impl Slot for TestSlot {
        fn dims(&self) -> (u32, u32) {
            self.dims
        }

        fn page_info(&self) -> PageInfo {
            self.info
        }

        fn bytes(&self) -> usize {
            self.bytes
        }
    }

    fn slot(
        dims: (u32, u32),
        page_dims: (u32, u32),
        bytes: usize,
        tag: &'static str,
    ) -> TestSlot {
        TestSlot {
            dims,
            info: PageInfo {
                dims: page_dims,
                format: Format::Png,
                animated: false,
            },
            bytes,
            tag,
        }
    }

    #[test]
    fn best_slot_and_eviction_honor_coverage_pinning_and_stale_thumbs() {
        let mut cache = SlotTable::default();
        cache.insert(
            0,
            Role::Thumb,
            slot((100, 100), (1000, 1000), 10, "t0"),
        );
        cache.insert(
            0,
            Role::Fit,
            slot((500, 500), (1000, 1000), 100, "f0"),
        );
        cache.insert(
            0,
            Role::Full,
            slot((1000, 1000), (1000, 1000), 400, "u0"),
        );

        assert_eq!(
            cache.best_slot(0, (400.0, 400.0)).expect("slot").tag,
            "f0",
            "Fit is preferred while covering screen_size"
        );
        assert_eq!(
            cache.best_slot(0, (700.0, 700.0)).expect("slot").tag,
            "u0",
            "Full takes over once screen_size exceeds Fit"
        );

        cache
            .set_playback(0, slot((1000, 1000), (1000, 1000), 400, "pb0"));
        assert_eq!(
            cache.best_slot(0, (400.0, 400.0)).expect("slot").tag,
            "pb0",
            "playback overrides static roles"
        );

        // Stale thumb with wrong page_dims gets evicted when Fit arrives.
        cache.insert(
            1,
            Role::Thumb,
            slot((100, 100), (500, 500), 10, "t1"),
        );
        assert_eq!(
            cache.insert(
                1,
                Role::Fit,
                slot((600, 600), (800, 800), 100, "f1"),
            ),
            InsertOutcome::EvictedStaleThumb
        );
        assert!(!cache.has(1, Role::Thumb));
        assert_eq!(cache.page_info(1).expect("info").dims, (800, 800));

        // Eviction skips pinned pages and evicts farthest priority first.
        cache.insert(
            2,
            Role::Fit,
            slot((500, 500), (500, 500), 100, "f2"),
        );
        cache.insert(
            5,
            Role::Fit,
            slot((500, 500), (500, 500), 100, "f5"),
        );
        cache.evict_to_budget(
            Role::Fit,
            200,
            &|page| page == 5,
            &|page| EvictionPriority {
                spread_distance: page,
                is_behind: false,
            },
        );
        assert!(cache.has(5, Role::Fit), "pinned page stays");
        assert!(!cache.has(2, Role::Fit), "unpinned farther page evicted");
        assert!(cache.has(0, Role::Fit), "closer page kept");
    }
}
