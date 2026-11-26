//! Current book reading state, spread/fit tracking, VRAM budgets, and load planning.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use crate::app::chrome::ReadingStatus;
use crate::app::chrome::StatusLine;
use crate::app::layout;
use crate::app::layout::Fit;
use crate::app::layout::FitMode;
use crate::app::layout::ReadingMode;
use crate::app::layout::Spread;
use crate::app::layout::Viewport;
use crate::book::catalog;
use crate::book::catalog::Direction;
use crate::book::catalog::PageName;
use crate::book::catalog::PageNames;
use crate::book::loader::Request;
use crate::book::loader::WorkPlan;
use crate::image::pixels::ImageLimits;
use crate::image::pixels::PageInfo;
use crate::image::pixels::TargetRole;
use crate::image::resample;
use crate::render::cache::EvictionPriority;
use crate::render::cache::Role;
use crate::render::cache::Slot;
use crate::render::cache::SlotTable;

const VRAM_FIT_BUDGET_BYTES: usize = 384 * 1024 * 1024;
const VRAM_THUMB_BUDGET_BYTES: usize = 64 * 1024 * 1024;
const VRAM_FULL_BUDGET_BYTES: usize = 192 * 1024 * 1024;

const THUMB_FALLBACK_GRACE: Duration = Duration::from_millis(50);
const LOOKAHEAD_FORWARD_DISTANCE: usize = 16;
const LOOKAHEAD_BACKWARD_DISTANCE: usize = 4;
const DEFAULT_FIT_PAGE_BYTES: usize = 8 * 1024 * 1024;
const LOOKAHEAD_VRAM_NUMERATOR: usize = 4;
const LOOKAHEAD_VRAM_DENOMINATOR: usize = 5;

#[derive(Clone, Copy, Default)]
enum WaitLogState {
    #[default]
    Idle,
    Since(Instant),
    Logged,
}

pub struct ViewStep {
    pub next_plan: Option<WorkPlan>,
    pub animated_pages: [Option<usize>; 2],
    pub is_waiting: bool,
    pub needs_redraw: bool,
    pub grace_deadline: Option<Instant>,
}

struct TargetView {
    page: usize,
    fit: Fit,
    direction: Direction,
    last_turn: Option<Instant>,
}

impl TargetView {
    const fn new(page: usize, fit_mode: FitMode) -> Self {
        Self {
            page,
            fit: Fit::new(fit_mode),
            direction: Direction::Forward,
            last_turn: None,
        }
    }

    fn move_to(
        &mut self,
        page: usize,
        direction: Direction,
        now: Instant,
    ) {
        self.page = page;
        self.direction = direction;
        self.last_turn = Some(now);
        self.fit.begin_turn();
        tracing::debug!(index = self.page, ?direction, "turn");
    }

    fn snap_for_mode(&mut self, page: usize) {
        self.page = page;
        self.fit.begin_turn();
        self.fit.commit();
        self.direction = Direction::Forward;
    }
}

pub struct Reading {
    path: PathBuf,
    is_archive: bool,
    directory_name: Option<String>,
    page_names: PageNames,
    mode: ReadingMode,
    target: TargetView,
    shown: Option<Spread>,
    /// Sticky page faults so later uploads cannot hide a decode/upload failure.
    faults: BTreeMap<usize, String>,
    wait_log: WaitLogState,
    last_work_plan: WorkPlan,
}

impl Reading {
    #[must_use]
    pub fn open(
        path: PathBuf,
        is_archive: bool,
        page_names: PageNames,
        land_on: Option<&str>,
        mode: ReadingMode,
        fit_mode: FitMode,
        has_thumbs: bool,
    ) -> (Self, WorkPlan) {
        let directory_name = (!is_archive)
            .then(|| {
                path.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
            })
            .flatten();
        let target_page =
            land_on.map_or(0, |name| catalog::landing(&page_names, name));
        let work = initial_work_plan(
            layout::spread_for(target_page, mode, page_names.count()),
            mode,
            has_thumbs,
        );
        let reading = Self {
            path,
            is_archive,
            directory_name,
            page_names,
            mode,
            target: TargetView::new(target_page, fit_mode),
            shown: None,
            faults: BTreeMap::new(),
            wait_log: WaitLogState::Idle,
            last_work_plan: work.clone(),
        };
        (reading, work)
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub const fn mode(&self) -> ReadingMode {
        self.mode
    }

    #[must_use]
    pub const fn fit_mode(&self) -> FitMode {
        self.target.fit.mode()
    }

    pub const fn fit_mut(&mut self) -> &mut Fit {
        &mut self.target.fit
    }

    pub fn zoom<Item>(
        &mut self,
        zoom_factor: f64,
        cursor: (f64, f64),
        viewport: Viewport,
        cache: &SlotTable<Item>,
    ) -> bool
    where
        Item: Slot,
    {
        if (zoom_factor - 1.0).abs() <= f64::EPSILON {
            return false;
        }
        self.adjust_fit(viewport, cache, |fit, spread_size, viewport| {
            fit.zoom(zoom_factor, cursor, spread_size, viewport);
        })
    }

    pub fn pan<Item>(
        &mut self,
        delta: (f64, f64),
        viewport: Viewport,
        cache: &SlotTable<Item>,
    ) -> bool
    where
        Item: Slot,
    {
        if delta == (0.0, 0.0) {
            return false;
        }
        self.adjust_fit(viewport, cache, |fit, spread_size, viewport| {
            fit.pan(delta, spread_size, viewport);
        })
    }

    fn adjust_fit<Item>(
        &mut self,
        viewport: Viewport,
        cache: &SlotTable<Item>,
        update: impl FnOnce(&mut Fit, (f64, f64), Viewport),
    ) -> bool
    where
        Item: Slot,
    {
        let (width, height) = viewport.size();
        if width == 0 || height == 0 {
            return false;
        }
        let spread_size = layout::layout(self.seat_dims(cache)).total();
        if spread_size.0 <= 0.0 || spread_size.1 <= 0.0 {
            return false;
        }
        self.target.fit.resolve(spread_size, viewport);
        update(&mut self.target.fit, spread_size, viewport);
        true
    }

    #[must_use]
    pub fn page_count(&self) -> NonZeroUsize {
        self.page_names.count()
    }

    #[must_use]
    pub const fn shown(&self) -> Option<Spread> {
        self.shown
    }

    #[must_use]
    pub fn target_spread(&self) -> Spread {
        layout::spread_for(self.target.page, self.mode, self.page_count())
    }

    #[must_use]
    pub fn on_screen(&self) -> Spread {
        self.shown.unwrap_or_else(|| self.target_spread())
    }

    #[must_use]
    pub fn is_waiting(&self) -> bool {
        self.shown != Some(self.target_spread())
    }

    #[must_use]
    pub fn is_faulted(&self, page: usize) -> bool {
        self.faults.contains_key(&page)
    }

    pub fn record_fault(&mut self, page: usize, fault: String) {
        self.faults.insert(page, fault);
    }

    #[must_use]
    pub fn sibling(&self, direction: Direction) -> Option<PathBuf> {
        catalog::sibling(&self.path, self.is_archive, direction)
    }

    fn page_file_name(&self, page: usize) -> Option<&str> {
        self.page_names.get(page).map(String::as_str)
    }

    fn page_name(&self, page: usize) -> PageName {
        let clamped_page =
            page.min(self.page_names.count().get().saturating_sub(1));
        catalog::page_name(
            &self.page_names,
            self.is_archive,
            self.directory_name.as_deref(),
            clamped_page,
        )
    }

    pub fn turn(
        &mut self,
        direction: Direction,
        is_repeat: bool,
        now: Instant,
    ) -> bool {
        let page_count = self.page_count();
        let next_page = layout::advance(
            self.target.page,
            direction,
            self.mode,
            page_count,
        );
        if is_repeat {
            let Some(spread) = self.shown else {
                return false;
            };
            // Pace held key repeats so target stays within one spread of screen.
            let shown_start =
                layout::spread_for(spread.first, self.mode, page_count)
                    .first;
            let target_start = layout::spread_for(
                self.target.page,
                self.mode,
                page_count,
            )
            .first;
            if target_start != shown_start && next_page != shown_start {
                return false;
            }
        }
        self.target.move_to(next_page, direction, now);
        true
    }

    pub fn jump_to_first(&mut self, now: Instant) {
        self.target.move_to(0, Direction::Forward, now);
    }

    pub fn jump_to_last(&mut self, now: Instant) {
        self.target.move_to(
            layout::last_spread(self.mode, self.page_count()),
            Direction::Backward,
            now,
        );
    }

    pub fn jump_to_page(
        &mut self,
        page_number: NonZeroUsize,
        now: Instant,
    ) -> bool {
        let page = page_number.get() - 1;
        let page_count = self.page_count();
        let target_start =
            layout::spread_for(page, self.mode, page_count).first;
        let current_start =
            layout::spread_for(self.target.page, self.mode, page_count)
                .first;
        if target_start == current_start {
            return false;
        }
        let direction = if target_start > current_start {
            Direction::Forward
        } else {
            Direction::Backward
        };
        self.target.move_to(target_start, direction, now);
        true
    }

    pub fn select_fit(&mut self, fit_mode: FitMode) {
        self.target.fit.select(fit_mode);
    }

    pub fn select_mode(&mut self, next: ReadingMode) -> bool {
        if self.mode == next {
            return false;
        }
        let pairing_changed = self.mode.pairs() != next.pairs();
        self.mode = next;
        if pairing_changed {
            let page = layout::spread_for(
                self.target.page,
                next,
                self.page_count(),
            )
            .first;
            self.target.snap_for_mode(page);
        } else {
            self.target.direction = Direction::Forward;
        }
        pairing_changed
    }

    #[must_use]
    pub fn thumbs_to_hydrate<Item>(
        &self,
        is_skimming: bool,
        cache: &SlotTable<Item>,
    ) -> Vec<usize>
    where
        Item: Slot,
    {
        let target_spread = self.target_spread();
        let next_spread = is_skimming
            .then(|| {
                layout::step(
                    target_spread.first,
                    self.target.direction,
                    self.mode,
                    self.page_count(),
                )
                .map(|start| {
                    layout::spread_for(start, self.mode, self.page_count())
                })
            })
            .flatten();
        let mut pages = Vec::new();
        for spread in
            [Some(target_spread), next_spread].into_iter().flatten()
        {
            for page in spread.seats(self.mode).into_iter().flatten() {
                if self.is_faulted(page)
                    || cache.has_any(page)
                    || pages.contains(&page)
                {
                    continue;
                }
                pages.push(page);
            }
        }
        pages
    }

    pub fn reconcile_view<Item>(
        &mut self,
        cache: &mut SlotTable<Item>,
        viewport: Viewport,
        limits: ImageLimits,
        is_skimming: bool,
        has_thumbs: bool,
        now: Instant,
    ) -> ViewStep
    where
        Item: Slot,
    {
        let (shown, shown_changed, grace_deadline) =
            self.resolve_shown(now, is_skimming, cache);
        if shown_changed {
            self.target.fit.commit();
        }
        let seat_dims = self.seat_dims(cache);
        let spread_layout = layout::layout(seat_dims);
        let spread_size = spread_layout.total();
        let placement = self.target.fit.resolve(spread_size, viewport);
        if shown_changed {
            let on_screen = self.on_screen();
            tracing::debug!(
                ?on_screen,
                ?shown,
                total = ?spread_size,
                ?seat_dims,
                scale = placement.scale,
                "placement"
            );
        }

        self.evict_vram(cache);

        let (viewport_width, viewport_height) = viewport.size();
        let zoom_scale = (viewport_width > 0 && viewport_height > 0)
            .then_some(placement.scale);
        let next_plan = self.update_work(
            is_skimming,
            zoom_scale,
            limits,
            has_thumbs,
            cache,
        );
        self.update_wait_log(now);
        let animated_pages = self.animated_targets(is_skimming, cache);
        let is_waiting = self.is_waiting();
        ViewStep {
            next_plan,
            animated_pages,
            is_waiting,
            needs_redraw: shown_changed,
            grace_deadline,
        }
    }

    fn evict_vram<Item>(&self, cache: &mut SlotTable<Item>)
    where
        Item: Slot,
    {
        let on_screen = self.on_screen();
        let target_spread = self.target_spread();
        let is_active = |page: usize| {
            on_screen.contains(page) || target_spread.contains(page)
        };
        let is_visible = |page: usize| on_screen.contains(page);
        let priority = |page: usize| self.vram_eviction_priority(page);
        cache.evict_to_budget(
            Role::Fit,
            VRAM_FIT_BUDGET_BYTES,
            &is_active,
            &priority,
        );
        cache.evict_to_budget(
            Role::Thumb,
            VRAM_THUMB_BUDGET_BYTES,
            &is_active,
            &priority,
        );
        cache.evict_to_budget(
            Role::Full,
            VRAM_FULL_BUDGET_BYTES,
            &is_visible,
            &priority,
        );
    }

    #[must_use]
    pub fn seat_dims<Item>(
        &self,
        cache: &SlotTable<Item>,
    ) -> [Option<(u32, u32)>; 2]
    where
        Item: Slot,
    {
        self.on_screen().seats(self.mode).map(|seat| {
            let page = seat?;
            cache.page_info(page).map(|info| info.dims)
        })
    }

    #[must_use]
    pub fn status_line<'fault, Item>(
        &'fault self,
        zoom_scale: f64,
        cache: &SlotTable<Item>,
    ) -> StatusLine<'fault>
    where
        Item: Slot,
    {
        use crate::image::decode::codec;

        let on_screen = self.on_screen();
        let mut failure: Option<(usize, &'fault str)> = None;
        let mut status_page: Option<(usize, PageInfo)> = None;
        for page in on_screen.pages() {
            if failure.is_some() {
                continue;
            }
            if let Some(fault) = self.faults.get(&page) {
                failure = Some((page, fault.as_str()));
                continue;
            }
            if cache.has_any(page)
                && let Some(info) = cache.page_info(page)
            {
                status_page.get_or_insert((page, info));
            }
        }
        let (named_page, fault) = match failure {
            Some((page, fault)) => (page, Some(fault)),
            None => (
                status_page.map_or(on_screen.first, |(page, _)| page),
                None,
            ),
        };
        let (page_dims, actual_format) = match status_page {
            Some((page, info)) => {
                let claimed_format = self
                    .page_file_name(page)
                    .and_then(codec::format_from_name);
                let actual_format = (claimed_format != Some(info.format))
                    .then_some(info.format);
                (Some(info.dims), actual_format)
            }
            None => (None, None),
        };
        StatusLine {
            reading: Some(ReadingStatus {
                page_count: self.page_count().get(),
                spread: on_screen,
                page_dims,
                actual_format,
                zoom_scale,
            }),
            mode: self.mode,
            show_mode_toast: false,
            name: self.page_name(named_page),
            show_loading: failure.is_none() && self.is_waiting(),
            fault,
        }
    }

    fn resolve_shown<Item>(
        &mut self,
        now: Instant,
        is_skimming: bool,
        cache: &SlotTable<Item>,
    ) -> (Option<Spread>, bool, Option<Instant>)
    where
        Item: Slot,
    {
        let (shown, deadline) =
            self.choose_presented(now, is_skimming, cache);
        let previous_shown = self.shown;
        let shown_changed = shown != previous_shown;
        if shown_changed {
            let seats = shown.map(|spread| {
                spread.seats(self.mode).map(|seat| {
                    seat.map_or("empty", |page| {
                        cache.best_role(page).map_or("none", Into::into)
                    })
                })
            });
            let turn_elapsed = self
                .target
                .last_turn
                .map(|last| now.saturating_duration_since(last));
            tracing::debug!(
                from = ?previous_shown,
                to = ?shown,
                is_skimming,
                ?seats,
                took = ?turn_elapsed,
                "shown"
            );
        }
        self.shown = shown;
        (shown, shown_changed, deadline)
    }

    fn choose_presented<Item>(
        &self,
        now: Instant,
        is_skimming: bool,
        cache: &SlotTable<Item>,
    ) -> (Option<Spread>, Option<Instant>)
    where
        Item: Slot,
    {
        let target_spread = self.target_spread();
        let shown = self.shown;
        let can_show = |candidate: usize, allow_thumb: bool| {
            if self.is_faulted(candidate) {
                return true;
            }
            if cache.page_info(candidate).is_none() {
                return false;
            }
            if allow_thumb {
                cache.has_any(candidate)
            } else {
                cache.has_settled_image(candidate)
            }
        };
        let seats = target_spread.seats(self.mode);
        if seats
            .into_iter()
            .flatten()
            .all(|page| can_show(page, false))
        {
            return (Some(target_spread), None);
        }
        if seats.into_iter().flatten().all(|page| can_show(page, true)) {
            let grace_elapsed =
                self.target.last_turn.is_none_or(|turned_at| {
                    now.saturating_duration_since(turned_at)
                        >= THUMB_FALLBACK_GRACE
                });
            if shown.is_none() || is_skimming || grace_elapsed {
                return (Some(target_spread), None);
            }
            return (
                shown,
                self.target
                    .last_turn
                    .map(|turned_at| turned_at + THUMB_FALLBACK_GRACE),
            );
        }
        (shown, None)
    }

    fn update_work<Item>(
        &mut self,
        is_skimming: bool,
        zoom_scale: Option<f64>,
        limits: ImageLimits,
        has_thumbs: bool,
        cache: &SlotTable<Item>,
    ) -> Option<WorkPlan>
    where
        Item: Slot,
    {
        let plan = self.build_work(
            is_skimming,
            zoom_scale,
            limits,
            has_thumbs,
            cache,
        );
        if plan == self.last_work_plan {
            return None;
        }
        tracing::debug!(
            current = plan.current.len(),
            lookahead = plan.lookahead.len(),
            thumb_anchor = ?plan.thumb_anchor,
            "sent work"
        );
        self.last_work_plan.clone_from(&plan);
        Some(plan)
    }

    fn build_work<Item>(
        &self,
        is_skimming: bool,
        zoom_scale: Option<f64>,
        limits: ImageLimits,
        has_thumbs: bool,
        cache: &SlotTable<Item>,
    ) -> WorkPlan
    where
        Item: Slot,
    {
        let target_spread = self.target_spread();
        let on_screen = self.on_screen();
        let page_count = self.page_count();
        let needs_fit = |page: usize| {
            !self.is_faulted(page) && !cache.has_settled_image(page)
        };

        if is_skimming {
            let target_has_preview =
                target_spread.seats(self.mode).into_iter().flatten().all(
                    |page| self.is_faulted(page) || cache.has_any(page),
                );
            if target_has_preview {
                return WorkPlan::default();
            }
            let mut current = Vec::new();
            for page in
                target_spread.seats(self.mode).into_iter().flatten()
            {
                if needs_fit(page) {
                    push_request(&mut current, page, TargetRole::Fit);
                }
            }
            return WorkPlan {
                current,
                lookahead: Vec::new(),
                thumb_anchor: None,
            };
        }

        let mut current = Vec::new();
        let mut budgeted_pages = Vec::new();
        for spread in [target_spread, on_screen] {
            for page in spread.seats(self.mode).into_iter().flatten() {
                if !budgeted_pages.contains(&page) {
                    budgeted_pages.push(page);
                }
                if needs_fit(page) {
                    push_request(&mut current, page, TargetRole::Fit);
                }
            }
        }
        for page in on_screen.seats(self.mode).into_iter().flatten() {
            if self.is_faulted(page) || cache.has(page, Role::Full) {
                continue;
            }
            let wants_full = zoom_scale.is_some_and(|scale| {
                cache.page_info(page).is_some_and(|info| {
                    full_wanted(scale, info.dims, limits)
                })
            });
            if wants_full {
                push_request(&mut current, page, TargetRole::Full);
            }
        }

        let fit_page_bytes = cache
            .average_role_bytes(Role::Fit)
            .unwrap_or(DEFAULT_FIT_PAGE_BYTES)
            .max(1);
        let lookahead_budget_bytes = VRAM_FIT_BUDGET_BYTES
            .checked_div(LOOKAHEAD_VRAM_DENOMINATOR)
            .map_or(VRAM_FIT_BUDGET_BYTES, |fifth| {
                fifth.saturating_mul(LOOKAHEAD_VRAM_NUMERATOR)
            });
        let ahead = walk_spreads(
            target_spread.first,
            self.target.direction,
            self.mode,
            page_count,
            LOOKAHEAD_FORWARD_DISTANCE,
        );
        let behind = walk_spreads(
            target_spread.first,
            self.target.direction.opposite(),
            self.mode,
            page_count,
            LOOKAHEAD_BACKWARD_DISTANCE,
        );
        let guaranteed_spreads = usize::from(!ahead.is_empty())
            + usize::from(!behind.is_empty());
        let prioritized_spreads = ahead
            .first()
            .into_iter()
            .chain(behind.first())
            .chain(ahead.get(1..).into_iter().flatten())
            .chain(behind.get(1..).into_iter().flatten());
        let mut lookahead = Vec::new();
        for (spread_index, spread) in prioritized_spreads.enumerate() {
            let new_pages = spread
                .seats(self.mode)
                .into_iter()
                .flatten()
                .filter(|page| !budgeted_pages.contains(page))
                .count();
            if spread_index >= guaranteed_spreads
                && budgeted_pages
                    .len()
                    .saturating_add(new_pages)
                    .saturating_mul(fit_page_bytes)
                    > lookahead_budget_bytes
            {
                break;
            }
            for page in spread.seats(self.mode).into_iter().flatten() {
                if !budgeted_pages.contains(&page) {
                    budgeted_pages.push(page);
                }
                if needs_fit(page)
                    && !current.iter().any(|request| request.page == page)
                    && !lookahead.contains(&page)
                {
                    lookahead.push(page);
                }
            }
        }

        WorkPlan {
            current,
            lookahead,
            thumb_anchor: has_thumbs
                .then_some((target_spread.first, self.target.direction)),
        }
    }

    fn update_wait_log(&mut self, now: Instant) {
        const HOLD_LOG_THRESHOLD: Duration = Duration::from_millis(250);

        let target = self.target_spread();
        let shown = self.shown;
        let holding = shown != Some(target);
        self.wait_log = match (self.wait_log, holding) {
            (_, false) => WaitLogState::Idle,
            (WaitLogState::Idle, true) => WaitLogState::Since(now),
            (WaitLogState::Since(since), true) => {
                let waited = now.saturating_duration_since(since);
                if waited >= HOLD_LOG_THRESHOLD {
                    tracing::info!(
                        ?target,
                        ?shown,
                        ?waited,
                        "holding for a fit image"
                    );
                    WaitLogState::Logged
                } else {
                    WaitLogState::Since(since)
                }
            }
            (WaitLogState::Logged, true) => WaitLogState::Logged,
        };
    }

    fn vram_eviction_priority(&self, page: usize) -> EvictionPriority {
        let target_first = self.target_spread().first;
        let spread_distance =
            layout::spread_distance(page, target_first, self.mode);
        let is_behind = match self.target.direction {
            Direction::Forward => page < target_first,
            Direction::Backward => page > target_first,
        };
        EvictionPriority {
            spread_distance,
            is_behind,
        }
    }

    fn animated_targets<Item>(
        &self,
        is_skimming: bool,
        cache: &SlotTable<Item>,
    ) -> [Option<usize>; 2]
    where
        Item: Slot,
    {
        let settled =
            !is_skimming && self.shown == Some(self.target_spread());
        if !settled {
            return [None, None];
        }
        self.on_screen().seats(self.mode).map(|seat| {
            let page = seat?;
            if self.is_faulted(page) {
                return None;
            }
            let info = cache.page_info(page)?;
            (info.animated && cache.has_settled_image(page))
                .then_some(page)
        })
    }
}

fn walk_spreads(
    start: usize,
    direction: Direction,
    mode: ReadingMode,
    page_count: NonZeroUsize,
    max_page_distance: usize,
) -> Vec<Spread> {
    let mut spreads = Vec::new();
    let mut current = start;
    while let Some(next) =
        layout::step(current, direction, mode, page_count)
    {
        if next.abs_diff(start) > max_page_distance {
            break;
        }
        spreads.push(layout::spread_for(next, mode, page_count));
        current = next;
    }
    spreads
}

fn initial_work_plan(
    spread: Spread,
    mode: ReadingMode,
    has_thumbs: bool,
) -> WorkPlan {
    let current = spread
        .seats(mode)
        .into_iter()
        .flatten()
        .map(|page| Request {
            page,
            role: TargetRole::Fit,
        })
        .collect();
    WorkPlan {
        current,
        lookahead: Vec::new(),
        thumb_anchor: has_thumbs
            .then_some((spread.first, Direction::Forward)),
    }
}

fn push_request(
    requests: &mut Vec<Request>,
    page: usize,
    role: TargetRole,
) {
    let request = Request { page, role };
    if !requests.contains(&request) {
        requests.push(request);
    }
}

fn full_wanted(
    zoom_scale: f64,
    page_dims: (u32, u32),
    limits: ImageLimits,
) -> bool {
    const ROUNDING_MARGIN_PX: f64 = 1.0;

    let fit_dims = resample::fit_dims(page_dims, limits.fit_box());
    let max_edge = limits.max_texture_edge();
    let full_dims = resample::fit_dims(page_dims, (max_edge, max_edge));
    full_dims != fit_dims
        && (zoom_scale * f64::from(page_dims.0)
            > f64::from(fit_dims.0) + ROUNDING_MARGIN_PX
            || zoom_scale * f64::from(page_dims.1)
                > f64::from(fit_dims.1) + ROUNDING_MARGIN_PX)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;
    use crate::image::pixels::Format;
    use crate::testkit;

    #[derive(Clone, Copy)]
    struct TestSlot {
        dims: (u32, u32),
        info: PageInfo,
        bytes: usize,
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

    fn slot(dims: (u32, u32), format: Format, animated: bool) -> TestSlot {
        TestSlot {
            dims,
            info: PageInfo {
                dims,
                format,
                animated,
            },
            bytes: 1024,
        }
    }

    fn slot_with_bytes(bytes: usize) -> TestSlot {
        TestSlot {
            dims: (800, 600),
            info: PageInfo {
                dims: (800, 600),
                format: Format::Png,
                animated: false,
            },
            bytes,
        }
    }

    fn reading_with_pages(
        count: usize,
        land_on: Option<&str>,
        mode: ReadingMode,
    ) -> (Reading, WorkPlan) {
        let names = (0..count)
            .map(|index| format!("{index:03}.png"))
            .collect::<Vec<_>>();
        Reading::open(
            PathBuf::from("/tmp/book"),
            false,
            PageNames::new(names).expect("non-empty test book"),
            land_on,
            mode,
            FitMode::Page,
            false,
        )
    }

    #[test]
    fn a_full_image_is_only_requested_when_the_fit_has_fewer_pixels() {
        let limits = testkit::limits();
        assert!(!full_wanted(16.0, (1920, 1080), limits));
        assert!(!full_wanted(16.0, (2560, 1440), limits));
        let portrait = (3628, 5148);
        assert!(!full_wanted(1440.0 / 5148.0, portrait, limits));
        assert!(full_wanted(0.4, portrait, limits));
        let four_k = (3840, 2160);
        assert!(!full_wanted(0.66, four_k, limits));
        assert!(full_wanted(0.67, four_k, limits));
        assert!(!full_wanted(1440.5 / 3000.0, (5000, 3000), limits));
        let ten_k = (10080, 5640);
        assert!(!full_wanted(0.25, ten_k, limits));
        assert!(full_wanted(0.26, ten_k, limits));
        assert!(!full_wanted(
            16.0,
            ten_k,
            ImageLimits::new((2560, 1440), 2560)
        ));
    }

    #[test]
    fn full_requests_follow_the_on_screen_scale() {
        let (mut reading, _) =
            reading_with_pages(12, Some("002.png"), ReadingMode::Normal);
        reading.shown = Some(Spread::single(1));
        let mut cache = SlotTable::default();
        cache.insert(1, Role::Fit, slot((4000, 6000), Format::Png, false));
        cache.insert(2, Role::Fit, slot((4000, 6000), Format::Png, false));

        assert_eq!(reading.on_screen(), Spread::single(1));
        let plan = reading
            .update_work(
                false,
                Some(1.0),
                testkit::limits(),
                false,
                &cache,
            )
            .expect("plan changed");
        assert!(
            plan.current.contains(&Request {
                page: 1,
                role: TargetRole::Full
            }),
            "on-screen page asks for full in current tier"
        );
        assert!(
            !plan.current.contains(&Request {
                page: 2,
                role: TargetRole::Full
            }),
            "target page is not upscaled"
        );
    }

    #[test]
    fn thumb_presentation_respects_grace_for_taps_and_shows_immediately_on_repeat()
     {
        let mode = ReadingMode::Normal;
        let (mut reading, _) = reading_with_pages(10, None, mode);
        let mut cache = SlotTable::default();
        cache.insert(0, Role::Fit, slot((100, 100), Format::Png, false));
        cache.insert(1, Role::Thumb, slot((100, 100), Format::Png, false));
        reading.shown = Some(Spread::single(0));

        let start = Instant::now();
        assert!(reading.turn(Direction::Forward, false, start));

        let (shown, _, deadline) =
            reading.resolve_shown(start, false, &cache);
        assert_eq!(shown, Some(Spread::single(0)));
        assert_eq!(deadline, Some(start + THUMB_FALLBACK_GRACE));
        assert!(reading.is_waiting());

        let (shown_while_skimming, _, deadline_while_skimming) =
            reading.resolve_shown(start, true, &cache);
        assert_eq!(shown_while_skimming, Some(Spread::single(1)));
        assert_eq!(deadline_while_skimming, None);

        reading.shown = Some(Spread::single(0));
        let (shown_after_grace, _, deadline_after_grace) = reading
            .resolve_shown(start + THUMB_FALLBACK_GRACE, false, &cache);
        assert_eq!(shown_after_grace, Some(Spread::single(1)));
        assert_eq!(deadline_after_grace, None);
    }

    #[test]
    fn repeat_turn_waits_for_shown_when_one_spread_ahead() {
        let mode = ReadingMode::Manga;
        let now = Instant::now();
        let (mut reading, _) =
            reading_with_pages(10, Some("001.png"), mode);

        assert!(!reading.turn(Direction::Forward, true, now));
        assert_eq!(reading.target.page, 1);

        assert!(reading.turn(Direction::Forward, false, now));
        assert_eq!(reading.target.page, 3);

        reading.shown =
            Some(layout::spread_for(3, mode, reading.page_count()));
        assert!(reading.turn(Direction::Forward, true, now));
        assert_eq!(reading.target.page, 5);
        assert!(!reading.turn(Direction::Forward, true, now));
        assert_eq!(reading.target.page, 5);

        assert!(reading.turn(Direction::Backward, true, now));
        assert_eq!(reading.target.page, 3);
    }

    #[test]
    fn animated_targets_require_a_settled_animated_page() {
        let mode = ReadingMode::Normal;
        let (mut reading, _) = reading_with_pages(20, None, mode);
        let mut cache = SlotTable::default();
        cache.insert(4, Role::Fit, slot((2, 2), Format::Gif, true));
        cache.insert(6, Role::Thumb, slot((2, 2), Format::Gif, true));
        cache.insert(8, Role::Fit, slot((2, 2), Format::Gif, false));
        cache.insert(10, Role::Fit, slot((2, 2), Format::Gif, true));
        cache.insert(12, Role::Fit, slot((2, 2), Format::Gif, true));
        reading.record_fault(10, String::from("bad page"));

        reading.target.page = 6;
        reading.shown = Some(Spread::single(6));
        assert_eq!(reading.animated_targets(false, &cache), [None, None]);

        reading.target.page = 8;
        reading.shown = Some(Spread::single(8));
        assert_eq!(reading.animated_targets(false, &cache), [None, None]);

        reading.target.page = 10;
        reading.shown = Some(Spread::single(10));
        assert_eq!(reading.animated_targets(false, &cache), [None, None]);

        reading.target.page = 12;
        reading.shown = Some(Spread::single(12));
        assert_eq!(reading.animated_targets(true, &cache), [None, None]);

        reading.target.page = 4;
        reading.shown = Some(Spread::single(4));
        assert_eq!(
            reading.animated_targets(false, &cache),
            [Some(4), None]
        );
    }

    #[test]
    fn thumbs_to_hydrate_picks_target_and_next_when_skimming() {
        let (mut reading, _) =
            reading_with_pages(4, None, ReadingMode::Normal);
        let mut cache = SlotTable::default();
        assert_eq!(reading.thumbs_to_hydrate(false, &cache), vec![0]);
        cache.insert(0, Role::Thumb, slot((10, 10), Format::Png, false));
        assert_eq!(reading.thumbs_to_hydrate(true, &cache), vec![1]);
        reading.record_fault(1, String::from("fault"));
        assert_eq!(
            reading.thumbs_to_hydrate(true, &cache),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn book_status_blames_page_and_reports_slow_turn_waiting() {
        let (mut reading, _) =
            reading_with_pages(10, Some("001.png"), ReadingMode::Manga);
        let spread = Spread {
            first: 1,
            second: Some(2),
        };
        reading.shown = Some(spread);

        let cache = SlotTable::<TestSlot>::default();
        let waiting = reading.status_line(1.0, &cache);
        assert_eq!(waiting.name.file, "001.png");
        assert_eq!(waiting.fault, None);
        assert!(!waiting.show_loading);

        let mut loaded_cache = SlotTable::default();
        loaded_cache.insert(
            1,
            Role::Fit,
            slot((800, 600), Format::Png, false),
        );
        let settled = reading.status_line(1.0, &loaded_cache);
        assert!(!settled.show_loading);
        assert_eq!(
            settled.reading.expect("reading status").page_dims,
            Some((800, 600))
        );

        // Slow turn to next spread keeps spread 1+2 shown while waiting.
        assert!(reading.turn(Direction::Forward, false, Instant::now()));
        assert!(reading.is_waiting());
        assert!(reading.status_line(1.0, &loaded_cache).show_loading);

        reading.target.page = 1;
        reading.record_fault(2, String::from("page"));
        let page_faulted = reading.status_line(1.0, &loaded_cache);
        assert!(!page_faulted.show_loading);
        assert_eq!(page_faulted.fault, Some("page"));
        assert_eq!(page_faulted.name.file, "002.png");
    }

    #[test]
    fn work_plan_bounds_lookahead_by_window_bytes() {
        let mode = ReadingMode::Normal;
        let (mut reading, _) =
            reading_with_pages(60, Some("010.png"), mode);
        reading.shown = Some(Spread::single(10));

        let mut cache = SlotTable::default();
        cache.insert(10, Role::Fit, slot_with_bytes(8 * 1024 * 1024));
        let normal_plan = reading.build_work(
            false,
            None,
            testkit::limits(),
            false,
            &cache,
        );
        assert_eq!(
            normal_plan.lookahead.get(..3),
            Some([11, 9, 12].as_slice())
        );
        assert!(
            normal_plan.lookahead.len() >= 18,
            "8 MiB/page allows lookahead window, got {}",
            normal_plan.lookahead.len()
        );

        let mut tight_cache = SlotTable::default();
        tight_cache.insert(
            10,
            Role::Fit,
            slot_with_bytes(80 * 1024 * 1024),
        );
        let tight_plan = reading.build_work(
            false,
            None,
            testkit::limits(),
            false,
            &tight_cache,
        );
        assert!(
            tight_plan.lookahead.len() < 8,
            "80 MiB/page caps lookahead early, got {}",
            tight_plan.lookahead.len()
        );
        for page in [11, 9] {
            assert!(
                tight_plan.lookahead.contains(&page),
                "+1 and -1 stay guaranteed"
            );
        }
    }

    #[test]
    fn work_plan_pauses_on_warm_skim_and_focuses_target_on_cold_skim() {
        let mode = ReadingMode::Manga;
        let (reading, _) = reading_with_pages(20, Some("005.png"), mode);
        let mut warm_cache = SlotTable::default();
        warm_cache.insert(
            5,
            Role::Thumb,
            slot((10, 10), Format::Png, false),
        );
        warm_cache.insert(
            6,
            Role::Thumb,
            slot((10, 10), Format::Png, false),
        );

        let warm_plan = reading.build_work(
            true,
            None,
            testkit::limits(),
            true,
            &warm_cache,
        );
        assert_eq!(warm_plan, WorkPlan::default());

        let cold_plan = reading.build_work(
            true,
            None,
            testkit::limits(),
            true,
            &SlotTable::<TestSlot>::default(),
        );
        assert_eq!(
            cold_plan.current,
            vec![
                Request {
                    page: 6,
                    role: TargetRole::Fit,
                },
                Request {
                    page: 5,
                    role: TargetRole::Fit,
                },
            ]
        );
        assert_eq!(cold_plan.lookahead, Vec::new());
        assert_eq!(cold_plan.thumb_anchor, None);
    }
}
