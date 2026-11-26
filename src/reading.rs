use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use crate::animation::Player;
use crate::book::PageEvent;
use crate::browse;
use crate::browse::Direction;
use crate::browse::PageNames;
use crate::browse::ReadingMode;
use crate::browse::Request;
use crate::browse::Role;
use crate::browse::Roles;
use crate::browse::Spread;
use crate::browse::WorkRole;
use crate::cache::ThumbImage;
use crate::cache::ThumbPack;
use crate::catalog;
use crate::catalog::PageName;
use crate::decode::Format;
use crate::geometry;
use crate::geometry::Fit;
use crate::geometry::Viewport;
use crate::protocol::BookInfo;
use crate::protocol::ImageLimits;
use crate::protocol::ParentMsg;
use crate::protocol::PlaybackCommand;
use crate::protocol::PlaybackEvent;
use crate::protocol::PlaybackId;
use crate::protocol::TextureFault;
use crate::protocol::ValidatedImage;
use crate::protocol::WorkPlan;
use crate::ui::ReadingStatus;

const VRAM_FIT_BUDGET_BYTES: usize = 384 * 1024 * 1024;
const VRAM_THUMB_BUDGET_BYTES: usize = 64 * 1024 * 1024;
const VRAM_FULL_BUDGET_BYTES: usize = 192 * 1024 * 1024;

pub trait PageStore {
    fn roles(&self, page: usize) -> Roles;
    fn has_any(&self, page: usize) -> bool;
    fn average_role_bytes(&self, role: Role) -> Option<usize>;
    fn insert_page(
        &mut self,
        page: usize,
        role: WorkRole,
        image: &ValidatedImage,
    ) -> Result<(), TextureFault>;
    fn insert_thumb(&mut self, page: usize, thumb: &ThumbImage);
    fn update_playback(
        &mut self,
        page: usize,
        image: &ValidatedImage,
    ) -> Result<(), TextureFault>;
    fn evict_to_budget(
        &mut self,
        role: Role,
        budget_bytes: usize,
        is_pinned: &dyn Fn(usize) -> bool,
        priority: &dyn Fn(usize) -> (usize, bool),
    );
    fn retain_playback(&mut self, keep_playback: &dyn Fn(usize) -> bool);
}

#[derive(Clone, Copy)]
pub struct ReconcileView {
    pub mode: ReadingMode,
    pub viewport: Viewport,
    pub limits: ImageLimits,
    pub is_skimming: bool,
    pub now: Instant,
}

pub struct ReconcileOutcome {
    pub messages: Vec<ParentMsg>,
    pub redraw: bool,
    pub waiting: bool,
    pub next_deadline: Option<Instant>,
}

const THUMB_GRACE: Duration = Duration::from_millis(50);
const LOOKAHEAD_AHEAD_PAGES: usize = 16;
const LOOKAHEAD_BEHIND_PAGES: usize = 4;
const THUMB_WARM_AHEAD_PAGES: usize = 256;
const THUMB_WARM_BEHIND_PAGES: usize = 32;
const DEFAULT_FIT_PAGE_BYTES: usize = 8 * 1024 * 1024;
const LOOKAHEAD_VRAM_NUMERATOR: usize = 4;
const LOOKAHEAD_VRAM_DENOMINATOR: usize = 5;

/// Native page dimensions and decoded format, fixed across roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PageInfo {
    dims: (u32, u32),
    format: Format,
}

#[derive(Default)]
enum PagePlayback {
    #[default]
    Idle,
    Playing(Player<Box<ValidatedImage>>),
    /// Sticky for the book so a stopped decoder is not retried on revisit.
    Refused,
}

#[derive(Default)]
struct PageState {
    info: Option<PageInfo>,
    /// Page faults stay sticky so a later upload or blip cannot erase them.
    fault: Option<String>,
    playback: PagePlayback,
}

impl PageState {
    fn player_mut(
        &mut self,
        id: PlaybackId,
    ) -> Option<&mut Player<Box<ValidatedImage>>> {
        match &mut self.playback {
            PagePlayback::Playing(player) if player.id() == id => {
                Some(player)
            }
            PagePlayback::Playing(_)
            | PagePlayback::Refused
            | PagePlayback::Idle => {
                tracing::debug!(?id, "stale playback update");
                None
            }
        }
    }

    fn stop_playback(&mut self) -> Option<PlaybackCommand> {
        let id = match &self.playback {
            PagePlayback::Playing(player) => player.id(),
            PagePlayback::Idle | PagePlayback::Refused => return None,
        };
        self.playback = PagePlayback::Idle;
        tracing::debug!(?id, "playback player stop");
        Some(PlaybackCommand::Stop { id })
    }

    fn refuse_playback(&mut self) -> Option<PlaybackCommand> {
        let stop = self.stop_playback();
        self.playback = PagePlayback::Refused;
        stop
    }
}

#[derive(Clone, Copy, Default)]
enum Hold {
    #[default]
    Idle,
    Since(Instant),
    Logged,
}

pub struct Reading {
    path: PathBuf,
    archive: bool,
    dir_name: Option<String>,
    names: PageNames,
    target: usize,
    shown: Option<Spread>,
    pages: BTreeMap<usize, PageState>,
    thumb_pack: Option<ThumbPack>,
    /// Drives child lookahead order.
    direction: Direction,
    last_turn: Option<Instant>,
    hold: Hold,
    sent_work: WorkPlan,
    next_generation: u32,
}

impl Reading {
    pub fn open(
        path: PathBuf,
        info: BookInfo,
        land_on: Option<&str>,
        mode: ReadingMode,
    ) -> (Self, WorkPlan) {
        let thumb_pack = ThumbPack::open(&path, info.archive, &info.names);
        Self::open_with_pack(path, info, land_on, mode, thumb_pack)
    }

    fn open_with_pack(
        path: PathBuf,
        info: BookInfo,
        land_on: Option<&str>,
        mode: ReadingMode,
        thumb_pack: Option<ThumbPack>,
    ) -> (Self, WorkPlan) {
        let archive = info.archive;
        let dir_name = if archive {
            None
        } else {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        };
        let target = land_on.map_or(0, |name| {
            catalog::landing(info.names.iter().map(String::as_str), name)
        });
        let work = cold_work(
            browse::spread_for(target, mode, info.names.count()),
            mode,
        );
        let reading = Self {
            path,
            archive,
            dir_name,
            names: info.names,
            target,
            shown: None,
            pages: BTreeMap::new(),
            thumb_pack,
            direction: Direction::Forward,
            last_turn: None,
            hold: Hold::Idle,
            sent_work: work.clone(),
            next_generation: 0,
        };
        (reading, work)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn page_count(&self) -> NonZeroUsize {
        self.names.count()
    }

    pub fn shown(&self) -> Option<Spread> {
        self.shown
    }

    fn is_faulted(&self, page: usize) -> bool {
        self.pages
            .get(&page)
            .is_some_and(|state| state.fault.is_some())
    }

    fn page_info(&self, page: usize) -> Option<PageInfo> {
        self.pages.get(&page)?.info
    }

    fn target_spread(&self, mode: ReadingMode) -> Spread {
        browse::spread_for(self.target, mode, self.page_count())
    }

    pub fn sibling(&self, direction: Direction) -> Option<PathBuf> {
        catalog::sibling(&self.path, self.archive, direction)
    }

    fn name(&self, index: usize) -> Option<&str> {
        self.names.get(index).map(String::as_str)
    }

    fn page_name(&self, index: usize) -> PageName {
        let index = index.min(self.names.count().get().saturating_sub(1));
        catalog::page_name(
            &self.names,
            self.archive,
            self.dir_name.as_deref(),
            index,
        )
    }

    fn set_turn(
        &mut self,
        target: usize,
        direction: Direction,
        now: Instant,
    ) {
        self.target = target;
        self.direction = direction;
        self.last_turn = Some(now);
        tracing::debug!(index = self.target, ?direction, "turn");
    }

    pub fn turn(
        &mut self,
        direction: Direction,
        mode: ReadingMode,
        is_repeat: bool,
        now: Instant,
    ) -> bool {
        let page_count = self.page_count();
        let next =
            browse::advance(self.target, direction, mode, page_count);
        if is_repeat {
            let Some(spread) = self.shown else {
                return false;
            };
            // Pace held key repeats so target stays within one spread of screen.
            let shown_start =
                browse::spread_for(spread.first, mode, page_count).first;
            let target_start =
                browse::spread_for(self.target, mode, page_count).first;
            if target_start != shown_start && next != shown_start {
                return false;
            }
        }
        self.set_turn(next, direction, now);
        true
    }

    pub fn jump_to_first(&mut self, now: Instant) {
        self.set_turn(0, Direction::Forward, now);
    }

    pub fn jump_to_last(&mut self, mode: ReadingMode, now: Instant) {
        self.set_turn(
            browse::last_spread(mode, self.page_count()),
            Direction::Backward,
            now,
        );
    }

    pub fn jump_to_page(
        &mut self,
        page_number: NonZeroUsize,
        mode: ReadingMode,
        now: Instant,
    ) -> bool {
        let index = page_number.get() - 1;
        let page_count = self.page_count();
        let target_start =
            browse::spread_for(index, mode, page_count).first;
        let current_start =
            browse::spread_for(self.target, mode, page_count).first;
        if target_start == current_start {
            return false;
        }
        let direction = if target_start > current_start {
            Direction::Forward
        } else {
            Direction::Backward
        };
        self.set_turn(target_start, direction, now);
        true
    }

    pub fn switch_mode(
        &mut self,
        previous: ReadingMode,
        next: ReadingMode,
    ) -> bool {
        let pairing_changed = previous.pairs() != next.pairs();
        if pairing_changed {
            self.target =
                browse::spread_for(self.target, next, self.page_count())
                    .first;
        }
        self.direction = Direction::Forward;
        pairing_changed
    }

    fn player_mut(
        &mut self,
        id: PlaybackId,
    ) -> Option<&mut Player<Box<ValidatedImage>>> {
        let Some(state) = self.pages.get_mut(&id.page) else {
            tracing::debug!(?id, "stale playback update");
            return None;
        };
        state.player_mut(id)
    }

    pub fn reconcile(
        &mut self,
        events: impl IntoIterator<Item = PageEvent>,
        fit: &mut Fit,
        view: ReconcileView,
        textures: &mut impl PageStore,
    ) -> ReconcileOutcome {
        let mut messages = Vec::new();
        let mut touched = Vec::new();
        let mut redraw = false;

        for event in events {
            self.apply_event(
                event,
                textures,
                &mut messages,
                &mut touched,
                &mut redraw,
            );
        }

        self.hydrate_thumbs(view.mode, view.is_skimming, textures, &mut touched);

        let (shown, changed, thumb_grace_deadline) = self.resolve_shown(
            view.mode,
            view.now,
            view.is_skimming,
            textures,
        );
        if changed {
            fit.commit();
            redraw = true;
        }
        let on_screen = self.on_screen(view.mode);
        if touched.iter().any(|&page| on_screen.contains(page)) {
            redraw = true;
        }
        let dims = self.seat_dims(view.mode);
        let layout = geometry::layout(dims);
        let spread_size = layout.total();
        let placement = fit.resolve(spread_size, view.viewport);
        if changed {
            tracing::debug!(
                ?on_screen,
                ?shown,
                total = ?spread_size,
                ?dims,
                scale = placement.scale,
                "placement"
            );
        }

        let target_spread = self.target_spread(view.mode);
        let is_active_page = |page: usize| {
            on_screen.contains(page) || target_spread.contains(page)
        };
        let priority = |page: usize| self.vram_evict_priority(view.mode, page);
        textures.evict_to_budget(
            Role::Fit,
            VRAM_FIT_BUDGET_BYTES,
            &is_active_page,
            &priority,
        );
        textures.evict_to_budget(
            Role::Thumb,
            VRAM_THUMB_BUDGET_BYTES,
            &is_active_page,
            &priority,
        );
        textures.evict_to_budget(
            Role::Full,
            VRAM_FULL_BUDGET_BYTES,
            &|page| on_screen.contains(page),
            &priority,
        );

        let (viewport_width, viewport_height) = view.viewport.size();
        let on_screen_scale = (viewport_width > 0 && viewport_height > 0)
            .then_some(placement.scale);
        if let Some(plan) = self.update_work(
            view.mode,
            view.is_skimming,
            on_screen_scale,
            view.limits,
            textures,
        ) {
            messages.push(ParentMsg::SetWork(plan));
        }
        self.note_hold(view.mode, view.now);

        for command in self.sync_playback(
            view.mode,
            view.is_skimming,
            textures,
        ) {
            messages.push(ParentMsg::Playback(command));
        }
        textures.retain_playback(&|page| self.retains_playback(page));

        let (playback_deadline, playback_updated) =
            self.tick_playback(view.now, textures, &mut messages);
        if playback_updated {
            redraw = true;
        }

        let waiting = self.is_waiting(view.mode, textures);
        let next_deadline = [playback_deadline, thumb_grace_deadline]
            .into_iter()
            .flatten()
            .min();
        ReconcileOutcome {
            messages,
            redraw,
            waiting,
            next_deadline,
        }
    }

    fn apply_event(
        &mut self,
        event: PageEvent,
        textures: &mut impl PageStore,
        messages: &mut Vec<ParentMsg>,
        touched: &mut Vec<usize>,
        redraw: &mut bool,
    ) {
        match event {
            PageEvent::Ready {
                page,
                role,
                image,
                thumb,
            } => {
                touched.push(page);
                match textures.insert_page(page, role, &image) {
                    Ok(()) => {
                        self.record_page_info(
                            page,
                            image.meta.page_dims,
                            image.meta.format,
                        );
                        if let Some(thumb) = thumb
                            && let Some(pack) = &mut self.thumb_pack
                        {
                            pack.record(page, thumb);
                        }
                    }
                    Err(fault) => {
                        tracing::warn!(
                            page,
                            ?role,
                            %fault,
                            "texture upload refused"
                        );
                        if !textures.roles(page).has_settled_image() {
                            self.pages.entry(page).or_default().fault =
                                Some(fault.to_string());
                        }
                    }
                }
            }
            PageEvent::Thumb { page, thumb } => {
                if let Some(pack) = &mut self.thumb_pack {
                    pack.record(page, thumb);
                }
            }
            PageEvent::Failed { page, fault } => {
                let state = self.pages.entry(page).or_default();
                state.fault = Some(fault);
                if let Some(command) = state.stop_playback() {
                    messages.push(ParentMsg::Playback(command));
                }
                *redraw = true;
            }
            PageEvent::Playback(playback) => match playback {
                PlaybackEvent::Frame { id, delay, image } => {
                    if let Some(player) = self.player_mut(id)
                        && !player.on_frame(delay, image)
                    {
                        tracing::debug!(?id, "unrequested playback frame");
                    }
                }
                PlaybackEvent::Finished { id } => {
                    if let Some(player) = self.player_mut(id)
                        && player.on_finished()
                    {
                        tracing::debug!(?id, "playback finished");
                    }
                }
                PlaybackEvent::Failed { id } => {
                    let Some(state) = self.pages.get_mut(&id.page) else {
                        tracing::debug!(?id, "stale playback update");
                        return;
                    };
                    if state.player_mut(id).is_some() {
                        tracing::warn!(?id, "playback failed");
                        if let Some(command) = state.refuse_playback() {
                            messages.push(ParentMsg::Playback(command));
                        }
                        *redraw = true;
                    }
                }
            },
        }
    }

    fn record_page_info(
        &mut self,
        page: usize,
        dims: (u32, u32),
        format: Format,
    ) {
        let state = self.pages.entry(page).or_default();
        if let Some(existing) = state.info {
            debug_assert_eq!(
                (existing.dims, existing.format),
                (dims, format),
                "page facts are fixed at the first upload"
            );
        } else {
            state.info = Some(PageInfo { dims, format });
        }
    }

    /// Load warm thumbnails from the local book thumb pack for `target`
    /// (and one spread ahead while skimming) if not yet in VRAM.
    fn hydrate_thumbs(
        &mut self,
        mode: ReadingMode,
        is_skimming: bool,
        textures: &mut impl PageStore,
        touched: &mut Vec<usize>,
    ) {
        let target = self.target_spread(mode);
        let next_spread = is_skimming
            .then(|| {
                browse::step(
                    target.first,
                    self.direction,
                    mode,
                    self.page_count(),
                )
                .map(|start| browse::spread_for(start, mode, self.page_count()))
            })
            .flatten();
        for spread in [Some(target), next_spread].into_iter().flatten() {
            for page in spread.seats(mode).into_iter().flatten() {
                if touched.contains(&page)
                    || self.is_faulted(page)
                    || textures.roles(page).any()
                {
                    continue;
                }
                let Some(thumb) =
                    self.thumb_pack.as_ref().and_then(|pack| pack.load(page))
                else {
                    continue;
                };
                textures.insert_thumb(page, &thumb);
                self.record_page_info(page, thumb.page_dims(), thumb.format());
                touched.push(page);
            }
        }
    }

    fn resolve_shown(
        &mut self,
        mode: ReadingMode,
        now: Instant,
        is_skimming: bool,
        textures: &impl PageStore,
    ) -> (Option<Spread>, bool, Option<Instant>) {
        let (shown, deadline) =
            self.choose_presented(mode, now, is_skimming, textures);
        let previous_shown = self.shown();
        let changed = shown != previous_shown;
        if changed {
            let seats = shown.map(|spread| {
                spread.seats(mode).map(|seat| {
                    seat.map_or("empty", |page| {
                        textures.roles(page).best().map_or("none", Into::into)
                    })
                })
            });
            let took = self
                .last_turn
                .map(|last| now.saturating_duration_since(last));
            tracing::debug!(
                from = ?previous_shown,
                to = ?shown,
                is_skimming,
                ?seats,
                ?took,
                "shown"
            );
        }
        self.shown = shown;
        (shown, changed, deadline)
    }

    fn choose_presented(
        &self,
        mode: ReadingMode,
        now: Instant,
        is_skimming: bool,
        textures: &impl PageStore,
    ) -> (Option<Spread>, Option<Instant>) {
        let target = self.target_spread(mode);
        let shown = self.shown();
        let can_show = |candidate: usize, allow_thumb: bool| {
            if self.is_faulted(candidate) {
                return true;
            }
            if self.page_info(candidate).is_none() {
                return false;
            }
            let page_roles = textures.roles(candidate);
            if allow_thumb {
                page_roles.any()
            } else {
                page_roles.has_settled_image()
            }
        };
        let seats = target.seats(mode);
        if seats.into_iter().flatten().all(|page| can_show(page, false)) {
            return (Some(target), None);
        }
        if seats.into_iter().flatten().all(|page| can_show(page, true)) {
            let grace_elapsed = self.last_turn.is_none_or(|turned_at| {
                now.saturating_duration_since(turned_at) >= THUMB_GRACE
            });
            if shown.is_none() || is_skimming || grace_elapsed {
                return (Some(target), None);
            }
            return (
                shown,
                self.last_turn.map(|turned_at| turned_at + THUMB_GRACE),
            );
        }
        (shown, None)
    }

    fn on_screen(&self, mode: ReadingMode) -> Spread {
        self.shown().unwrap_or_else(|| self.target_spread(mode))
    }

    pub fn seat_dims(&self, mode: ReadingMode) -> [Option<(u32, u32)>; 2] {
        self.on_screen(mode).seats(mode).map(|seat| {
            let page = seat?;
            self.page_info(page).map(|info| info.dims)
        })
    }

    fn update_work(
        &mut self,
        mode: ReadingMode,
        is_skimming: bool,
        on_screen_scale: Option<f64>,
        limits: ImageLimits,
        textures: &impl PageStore,
    ) -> Option<WorkPlan> {
        let plan = self.build_work(
            mode,
            is_skimming,
            on_screen_scale,
            limits,
            textures,
        );
        if plan == self.sent_work {
            return None;
        }
        tracing::debug!(
            current = plan.current.len(),
            lookahead = plan.lookahead.len(),
            thumbs = plan.thumbs.len(),
            "sent work"
        );
        self.sent_work.clone_from(&plan);
        Some(plan)
    }

    fn build_work(
        &self,
        mode: ReadingMode,
        is_skimming: bool,
        on_screen_scale: Option<f64>,
        limits: ImageLimits,
        textures: &impl PageStore,
    ) -> WorkPlan {
        let target = self.target_spread(mode);
        let on_screen = self.on_screen(mode);
        let page_count = self.page_count();
        let needs_fit = |page: usize| {
            !self.is_faulted(page) && !textures.roles(page).has_settled_image()
        };

        if is_skimming {
            let target_has_preview =
                target.seats(mode).into_iter().flatten().all(|page| {
                    self.is_faulted(page) || textures.roles(page).any()
                });
            if target_has_preview {
                return WorkPlan::default();
            }
            let mut current = Vec::new();
            for page in target.seats(mode).into_iter().flatten() {
                if needs_fit(page) {
                    push_request(&mut current, page, WorkRole::Fit);
                }
            }
            return WorkPlan {
                current,
                lookahead: Vec::new(),
                thumbs: Vec::new(),
            };
        }

        let mut current = Vec::new();
        let mut window_pages = Vec::new();
        for spread in [target, on_screen] {
            for page in spread.seats(mode).into_iter().flatten() {
                if !window_pages.contains(&page) {
                    window_pages.push(page);
                }
                if needs_fit(page) {
                    push_request(&mut current, page, WorkRole::Fit);
                }
            }
        }
        for page in on_screen.seats(mode).into_iter().flatten() {
            if self.is_faulted(page) || textures.roles(page).contains(Role::Full)
            {
                continue;
            }
            let wants_full = on_screen_scale.is_some_and(|scale| {
                self.page_info(page)
                    .is_some_and(|info| full_wanted(scale, info.dims, limits))
            });
            if wants_full {
                push_request(&mut current, page, WorkRole::Full);
            }
        }

        let fit_page_bytes = textures
            .average_role_bytes(Role::Fit)
            .unwrap_or(DEFAULT_FIT_PAGE_BYTES)
            .max(1);
        let window_cap_bytes = VRAM_FIT_BUDGET_BYTES
            .checked_div(LOOKAHEAD_VRAM_DENOMINATOR)
            .map_or(VRAM_FIT_BUDGET_BYTES, |step| {
                step.saturating_mul(LOOKAHEAD_VRAM_NUMERATOR)
            });
        let opposite = match self.direction {
            Direction::Forward => Direction::Backward,
            Direction::Backward => Direction::Forward,
        };
        let ahead = walk_spreads(
            target.first,
            self.direction,
            mode,
            page_count,
            LOOKAHEAD_AHEAD_PAGES,
        );
        let behind = walk_spreads(
            target.first,
            opposite,
            mode,
            page_count,
            LOOKAHEAD_BEHIND_PAGES,
        );
        let guaranteed_spreads =
            usize::from(!ahead.is_empty()) + usize::from(!behind.is_empty());
        let ordered = ahead
            .first()
            .into_iter()
            .chain(behind.first())
            .chain(ahead.get(1..).into_iter().flatten())
            .chain(behind.get(1..).into_iter().flatten());
        let mut lookahead = Vec::new();
        for (spread_index, spread) in ordered.enumerate() {
            let new_pages = spread
                .seats(mode)
                .into_iter()
                .flatten()
                .filter(|page| !window_pages.contains(page))
                .count();
            if spread_index >= guaranteed_spreads
                && window_pages
                    .len()
                    .saturating_add(new_pages)
                    .saturating_mul(fit_page_bytes)
                    > window_cap_bytes
            {
                break;
            }
            for page in spread.seats(mode).into_iter().flatten() {
                if !window_pages.contains(&page) {
                    window_pages.push(page);
                }
                if needs_fit(page)
                    && !current.iter().any(|request| request.page == page)
                    && !lookahead.contains(&page)
                {
                    lookahead.push(page);
                }
            }
        }

        let mut thumbs = Vec::new();
        if let Some(pack) = &self.thumb_pack {
            let warm_ahead = walk_spreads(
                target.first,
                self.direction,
                mode,
                page_count,
                THUMB_WARM_AHEAD_PAGES,
            );
            let warm_behind = walk_spreads(
                target.first,
                opposite,
                mode,
                page_count,
                THUMB_WARM_BEHIND_PAGES,
            );
            for spread in warm_ahead.into_iter().chain(warm_behind) {
                for page in spread.seats(mode).into_iter().flatten() {
                    if self.is_faulted(page)
                        || pack.has(page)
                        || current.iter().any(|request| request.page == page)
                        || lookahead.contains(&page)
                        || thumbs.contains(&page)
                    {
                        continue;
                    }
                    thumbs.push(page);
                }
            }
        }

        WorkPlan {
            current,
            lookahead,
            thumbs,
        }
    }

    fn note_hold(&mut self, mode: ReadingMode, now: Instant) {
        const HOLD_LOG_THRESHOLD: Duration = Duration::from_millis(250);

        let target = self.target_spread(mode);
        let shown = self.shown();
        let holding = shown != Some(target);
        self.hold = match (self.hold, holding) {
            (_, false) => Hold::Idle,
            (Hold::Idle, true) => Hold::Since(now),
            (Hold::Since(since), true) => {
                let waited = now.saturating_duration_since(since);
                if waited >= HOLD_LOG_THRESHOLD {
                    tracing::info!(
                        ?target,
                        ?shown,
                        ?waited,
                        "holding for a fit image"
                    );
                    Hold::Logged
                } else {
                    Hold::Since(since)
                }
            }
            (Hold::Logged, true) => Hold::Logged,
        };
    }

    fn vram_evict_priority(
        &self,
        mode: ReadingMode,
        page: usize,
    ) -> (usize, bool) {
        let target_first = self.target_spread(mode).first;
        let distance = browse::spread_distance(page, target_first, mode);
        let is_behind = match self.direction {
            Direction::Forward => page < target_first,
            Direction::Backward => page > target_first,
        };
        (distance, is_behind)
    }

    fn retains_playback(&self, page: usize) -> bool {
        self.pages.get(&page).is_some_and(|state| {
            matches!(state.playback, PagePlayback::Playing(_))
        })
    }

    /// Starts a live playback stream when a settled animated seat is shown.
    fn sync_playback(
        &mut self,
        mode: ReadingMode,
        is_skimming: bool,
        textures: &impl PageStore,
    ) -> Vec<PlaybackCommand> {
        let on_screen = self.on_screen(mode);
        let settled =
            !is_skimming && self.shown() == Some(self.target_spread(mode));
        let targets = if settled {
            on_screen.seats(mode).map(|seat| {
                let page = seat?;
                if self.is_faulted(page) {
                    return None;
                }
                let info = self.page_info(page)?;
                if !info.format.may_animate()
                    || !textures.roles(page).has_settled_image()
                {
                    return None;
                }
                Some(page)
            })
        } else {
            [None, None]
        };
        let mut commands = Vec::new();
        for (&page, state) in &mut self.pages {
            let keep = match &state.playback {
                PagePlayback::Playing(_) => {
                    state.fault.is_none()
                        && targets
                            .iter()
                            .flatten()
                            .any(|&want_page| want_page == page)
                }
                PagePlayback::Idle | PagePlayback::Refused => true,
            };
            if !keep && let Some(command) = state.stop_playback() {
                commands.push(command);
            }
        }
        for &page in targets.iter().flatten() {
            let Some(state) = self.pages.get_mut(&page) else {
                continue;
            };
            if state.info.is_none()
                || state.fault.is_some()
                || !matches!(state.playback, PagePlayback::Idle)
            {
                continue;
            }
            let id = PlaybackId {
                page,
                generation: self.next_generation,
            };
            self.next_generation = id.generation.wrapping_add(1);
            state.playback = PagePlayback::Playing(Player::new(id));
            commands.push(PlaybackCommand::Start { id });
            tracing::debug!(?id, "playback player start");
        }
        commands
    }

    fn tick_playback(
        &mut self,
        now: Instant,
        textures: &mut impl PageStore,
        messages: &mut Vec<ParentMsg>,
    ) -> (Option<Instant>, bool) {
        use crate::animation::Due;

        let mut next_deadline: Option<Instant> = None;
        let mut updated = false;
        for (&page, state) in &mut self.pages {
            let PagePlayback::Playing(player) = &mut state.playback else {
                continue;
            };
            let id = player.id();
            match player.tick(now) {
                Due::Show { image, next_due } => {
                    if let Err(error) = textures.update_playback(page, &image) {
                        tracing::warn!(
                            page,
                            %error,
                            "playback frame upload failed"
                        );
                        if let Some(command) = state.refuse_playback() {
                            messages.push(ParentMsg::Playback(command));
                        }
                        continue;
                    }
                    tracing::debug!(
                        page,
                        delay = ?next_due.saturating_duration_since(now),
                        "playback frame shown"
                    );
                    updated = true;
                    next_deadline =
                        Some(min_deadline(next_deadline, next_due));
                }
                Due::Wait(next_due) => {
                    next_deadline =
                        Some(min_deadline(next_deadline, next_due));
                }
                Due::Idle => {}
            }
            if player.take_next() {
                messages.push(ParentMsg::Playback(PlaybackCommand::Next { id }));
            }
        }
        (next_deadline, updated)
    }

    fn is_waiting(&self, mode: ReadingMode, textures: &impl PageStore) -> bool {
        self.on_screen(mode).pages().all(|page| {
            let decoded =
                self.page_info(page).is_some() && textures.has_any(page);
            !decoded && !self.is_faulted(page)
        })
    }

    pub fn book_status<'fault>(
        &'fault self,
        mode: ReadingMode,
        zoom: f64,
        textures: &impl PageStore,
    ) -> BookStatus<'fault> {
        use crate::decode;

        let on_screen = self.on_screen(mode);
        let mut failure: Option<(usize, &'fault str)> = None;
        let mut first_decoded: Option<(usize, PageInfo)> = None;
        for page in on_screen.pages() {
            let state = self.pages.get(&page);
            let info = state
                .and_then(|page_state| page_state.info)
                .filter(|_| textures.has_any(page));
            let fault =
                state.and_then(|page_state| page_state.fault.as_deref());
            if failure.is_some() {
                continue;
            }
            if let Some(fault) = fault {
                failure = Some((page, fault));
                continue;
            }
            if let Some(info) = info {
                first_decoded.get_or_insert((page, info));
            }
        }
        let (named_page, fault) = match failure {
            Some((page, fault)) => (page, Some(fault)),
            None => (
                first_decoded.map_or(on_screen.first, |(page, _)| page),
                None,
            ),
        };
        let (page_dims, actual_format) = match first_decoded {
            Some((page, info)) => {
                let claimed_format =
                    self.name(page).and_then(decode::format_from_name);
                let actual_format = (claimed_format != Some(info.format))
                    .then_some(info.format);
                (Some(info.dims), actual_format)
            }
            None => (None, None),
        };
        BookStatus {
            reading: ReadingStatus {
                page_count: self.page_count().get(),
                spread: on_screen,
                page_dims,
                actual_format,
                zoom,
            },
            name: self.page_name(named_page),
            waiting: failure.is_none() && first_decoded.is_none(),
            fault,
        }
    }
}

pub struct BookStatus<'fault> {
    pub reading: ReadingStatus,
    pub name: PageName,
    pub waiting: bool,
    pub fault: Option<&'fault str>,
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
    while let Some(next) = browse::step(current, direction, mode, page_count) {
        if next.abs_diff(start) > max_page_distance {
            break;
        }
        spreads.push(browse::spread_for(next, mode, page_count));
        current = next;
    }
    spreads
}

fn cold_work(spread: Spread, mode: ReadingMode) -> WorkPlan {
    let current: Vec<Request> = spread
        .seats(mode)
        .into_iter()
        .flatten()
        .map(|page| Request {
            page,
            role: WorkRole::Fit,
        })
        .collect();
    WorkPlan {
        current,
        lookahead: Vec::new(),
        thumbs: Vec::new(),
    }
}

fn push_request(requests: &mut Vec<Request>, page: usize, role: WorkRole) {
    let request = Request { page, role };
    if !requests.contains(&request) {
        requests.push(request);
    }
}

fn min_deadline(deadline: Option<Instant>, next_due: Instant) -> Instant {
    deadline.map_or(next_due, |current| current.min(next_due))
}

fn full_wanted(
    scale: f64,
    page_dims: (u32, u32),
    limits: ImageLimits,
) -> bool {
    use crate::cache;

    const ROUNDING_MARGIN_PX: f64 = 1.0;

    let fit = cache::fit_dims(page_dims, limits.fit_box());
    let max_edge = limits.max_texture_edge();
    let full = cache::fit_dims(page_dims, (max_edge, max_edge));
    full != fit
        && (scale * f64::from(page_dims.0)
            > f64::from(fit.0) + ROUNDING_MARGIN_PX
            || scale * f64::from(page_dims.1)
                > f64::from(fit.1) + ROUNDING_MARGIN_PX)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;
    use crate::testkit;

    #[derive(Default)]
    struct FakeStore {
        roles: BTreeMap<usize, Roles>,
        average_fit_bytes: Option<usize>,
        refuse_page: bool,
    }

    impl FakeStore {
        fn from_roles(items: impl IntoIterator<Item = (usize, Role)>) -> Self {
            let mut store = Self::default();
            for (page, role) in items {
                store.roles.entry(page).or_default().set(role);
            }
            store
        }
    }

    impl PageStore for FakeStore {
        fn roles(&self, page: usize) -> Roles {
            self.roles.get(&page).copied().unwrap_or_default()
        }

        fn has_any(&self, page: usize) -> bool {
            self.roles(page).any()
        }

        fn average_role_bytes(&self, role: Role) -> Option<usize> {
            (role == Role::Fit).then_some(self.average_fit_bytes).flatten()
        }

        fn insert_page(
            &mut self,
            page: usize,
            role: WorkRole,
            _image: &ValidatedImage,
        ) -> Result<(), TextureFault> {
            if self.refuse_page {
                return Err(TextureFault {
                    dims: (200, 200),
                    max_edge: 100,
                });
            }
            self.roles.entry(page).or_default().set(Role::from(role));
            Ok(())
        }

        fn insert_thumb(&mut self, page: usize, _thumb: &ThumbImage) {
            self.roles.entry(page).or_default().set(Role::Thumb);
        }

        fn update_playback(
            &mut self,
            _page: usize,
            _image: &ValidatedImage,
        ) -> Result<(), TextureFault> {
            Ok(())
        }

        fn evict_to_budget(
            &mut self,
            _role: Role,
            _budget_bytes: usize,
            _is_pinned: &dyn Fn(usize) -> bool,
            _priority: &dyn Fn(usize) -> (usize, bool),
        ) {
        }

        fn retain_playback(&mut self, _keep_playback: &dyn Fn(usize) -> bool) {
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
        let info = BookInfo {
            names: PageNames::new(names).expect("non-empty test book"),
            archive: false,
        };
        Reading::open_with_pack(
            PathBuf::from("/tmp/book"),
            info,
            land_on,
            mode,
            None,
        )
    }

    fn seed_page(
        reading: &mut Reading,
        page: usize,
        dims: (u32, u32),
        format: Format,
    ) {
        reading.pages.entry(page).or_default().info =
            Some(PageInfo { dims, format });
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
        reading.shown = Some(Spread::lone(1));
        seed_page(&mut reading, 1, (4000, 6000), Format::Png);
        seed_page(&mut reading, 2, (4000, 6000), Format::Png);

        assert_eq!(reading.on_screen(ReadingMode::Normal), Spread::lone(1));
        let plan = reading
            .update_work(
                ReadingMode::Normal,
                false,
                Some(1.0),
                testkit::limits(),
                &FakeStore::default(),
            )
            .expect("plan changed");
        assert!(
            plan.current.contains(&Request {
                page: 1,
                role: WorkRole::Full
            }),
            "on-screen page asks for full in current tier"
        );
        assert!(
            !plan.current.contains(&Request {
                page: 2,
                role: WorkRole::Full
            }),
            "target page is not upscaled"
        );
    }

    #[test]
    fn thumb_presentation_respects_grace_for_taps_and_shows_immediately_on_repeat() {
        let mode = ReadingMode::Normal;
        let (mut reading, _) = reading_with_pages(10, None, mode);
        seed_page(&mut reading, 0, (100, 100), Format::Png);
        seed_page(&mut reading, 1, (100, 100), Format::Png);
        reading.shown = Some(Spread::lone(0));

        let start = Instant::now();
        assert!(reading.turn(Direction::Forward, mode, false, start));
        let store = FakeStore::from_roles([(0, Role::Fit), (1, Role::Thumb)]);

        // Inside 50ms grace on a discrete tap: keep page 0 and arm deadline.
        let (shown, _, deadline) =
            reading.resolve_shown(mode, start, false, &store);
        assert_eq!(shown, Some(Spread::lone(0)));
        assert_eq!(deadline, Some(start + THUMB_GRACE));

        // While holding turn key: show warm thumb immediately.
        let (shown_while_skimming, _, deadline_while_skimming) =
            reading.resolve_shown(mode, start, true, &store);
        assert_eq!(shown_while_skimming, Some(Spread::lone(1)));
        assert_eq!(deadline_while_skimming, None);

        // Reset shown to page 0 and let grace expire.
        reading.shown = Some(Spread::lone(0));
        let (shown_after_grace, _, deadline_after_grace) =
            reading.resolve_shown(mode, start + THUMB_GRACE, false, &store);
        assert_eq!(shown_after_grace, Some(Spread::lone(1)));
        assert_eq!(deadline_after_grace, None);
    }

    #[test]
    fn repeat_turn_waits_for_shown_when_one_spread_ahead() {
        let mode = ReadingMode::Manga;
        let now = Instant::now();
        let (mut reading, _) =
            reading_with_pages(10, Some("001.png"), mode);

        // Repeat without a shown spread is dropped.
        assert!(!reading.turn(Direction::Forward, mode, true, now));
        assert_eq!(reading.target, 1);

        // Discrete tap always advances.
        assert!(reading.turn(Direction::Forward, mode, false, now));
        assert_eq!(reading.target, 3);

        reading.shown =
            Some(browse::spread_for(3, mode, reading.page_count()));
        assert!(reading.turn(Direction::Forward, mode, true, now));
        assert_eq!(reading.target, 5);
        assert!(!reading.turn(Direction::Forward, mode, true, now));
        assert_eq!(reading.target, 5);

        assert!(reading.turn(Direction::Backward, mode, true, now));
        assert_eq!(reading.target, 3);
    }

    #[test]
    fn playback_targets_need_an_animated_page_and_a_settled_frame() {
        let mode = ReadingMode::Normal;
        let (mut reading, _) = reading_with_pages(20, None, mode);
        for page in [4, 6, 10, 12] {
            seed_page(&mut reading, page, (2, 2), Format::Gif);
        }
        seed_page(&mut reading, 8, (2, 2), Format::Png);
        reading.pages.entry(10).or_default().fault =
            Some(String::from("bad page"));

        let store = FakeStore::from_roles([
            (4, Role::Fit),
            (6, Role::Thumb),
            (8, Role::Fit),
            (10, Role::Fit),
            (12, Role::Fit),
        ]);

        reading.target = 6;
        reading.shown = Some(Spread::lone(6));
        assert_eq!(reading.sync_playback(mode, false, &store), []);

        reading.target = 8;
        reading.shown = Some(Spread::lone(8));
        assert_eq!(reading.sync_playback(mode, false, &store), []);

        reading.target = 10;
        reading.shown = Some(Spread::lone(10));
        assert_eq!(reading.sync_playback(mode, false, &store), []);

        reading.target = 12;
        reading.shown = Some(Spread::lone(12));
        assert_eq!(reading.sync_playback(mode, true, &store), []);

        reading.target = 4;
        reading.shown = Some(Spread::lone(4));
        assert_eq!(
            reading.sync_playback(mode, false, &store),
            vec![PlaybackCommand::Start {
                id: PlaybackId {
                    page: 4,
                    generation: 0
                },
            }]
        );
    }

    #[test]
    fn thumb_pack_hydrates_warm_pages_in_reading() {
        use tempfile::TempDir;

        use crate::cache;
        use crate::decode::Decoded;
        use crate::decode::Transfer;

        let dir = TempDir::new().expect("temp dir");
        let root = dir.path().join("thumbs");
        let mut pack = ThumbPack::open_at(&root, [42; 16], 4).expect("pack");
        let thumb = cache::derive_thumb(
            &Decoded::rgba(
                10,
                10,
                Format::Png,
                Transfer::Srgb,
                vec![128_u8; 400].into_boxed_slice(),
            )
            .expect("decoded")
            .with_page_dims((200, 200)),
            1,
        )
        .expect("thumb");
        pack.record(0, thumb.clone());
        pack.record(1, thumb);
        pack.flush();

        let info = BookInfo {
            names: PageNames::new(
                (0..4).map(|index| format!("{index}.png")).collect(),
            )
            .expect("names"),
            archive: false,
        };
        let (mut reading, _) = Reading::open_with_pack(
            PathBuf::from("/tmp/book"),
            info,
            None,
            ReadingMode::Normal,
            Some(pack),
        );
        let mut store = FakeStore::default();
        let mut touched = Vec::new();
        reading.hydrate_thumbs(ReadingMode::Normal, false, &mut store, &mut touched);
        assert_eq!(touched, vec![0]);
        assert!(store.roles(0).contains(Role::Thumb));
        assert!(!store.has_any(1));
        assert_eq!(
            reading.page_info(0),
            Some(PageInfo {
                dims: (200, 200),
                format: Format::Png,
            })
        );

        touched.clear();
        reading.hydrate_thumbs(ReadingMode::Normal, true, &mut store, &mut touched);
        assert_eq!(touched, vec![1]);
        assert!(store.roles(1).contains(Role::Thumb));
    }

    #[test]
    fn book_status_blames_page_and_summarizes_spread() {
        let (mut reading, _) =
            reading_with_pages(10, None, ReadingMode::Manga);
        let spread = Spread {
            first: 1,
            second: Some(2),
        };
        reading.shown = Some(spread);

        let waiting = reading.book_status(
            ReadingMode::Manga,
            1.0,
            &FakeStore::default(),
        );
        assert!(waiting.waiting);
        assert_eq!(waiting.fault, None);
        assert_eq!(waiting.name.file, "001.png");

        seed_page(&mut reading, 1, (800, 600), Format::Png);
        let mut store = FakeStore::from_roles([(1, Role::Fit)]);
        let settled = reading.book_status(ReadingMode::Manga, 1.0, &store);
        assert!(!settled.waiting);
        assert_eq!(settled.reading.page_dims, Some((800, 600)));

        let mut messages = Vec::new();
        let mut touched = Vec::new();
        let mut redraw = false;
        reading.apply_event(
            PageEvent::Failed {
                page: 2,
                fault: String::from("page"),
            },
            &mut store,
            &mut messages,
            &mut touched,
            &mut redraw,
        );
        assert!(redraw);
        let page_faulted =
            reading.book_status(ReadingMode::Manga, 1.0, &store);
        assert!(!page_faulted.waiting);
        assert_eq!(page_faulted.fault, Some("page"));
        assert_eq!(page_faulted.name.file, "002.png");
    }

    #[test]
    fn refused_upload_faults_only_unsettled_pages() {
        use crate::decode::Decoded;
        use crate::decode::Transfer;
        use crate::protocol::WireImage;

        let image = WireImage::from_decoded(
            &Decoded::rgba(
                10,
                10,
                Format::Png,
                Transfer::Srgb,
                vec![128_u8; 400].into_boxed_slice(),
            )
            .expect("decoded")
            .with_page_dims((200, 200)),
        )
        .validate()
        .expect("valid image");

        let (mut reading, _) =
            reading_with_pages(4, None, ReadingMode::Normal);
        let mut store = FakeStore {
            refuse_page: true,
            ..FakeStore::default()
        };
        let mut messages = Vec::new();
        let mut touched = Vec::new();
        let mut redraw = false;

        reading.apply_event(
            PageEvent::Ready {
                page: 0,
                role: WorkRole::Fit,
                image: Box::new(image.clone()),
                thumb: None,
            },
            &mut store,
            &mut messages,
            &mut touched,
            &mut redraw,
        );
        assert!(reading.is_faulted(0));

        store.roles.entry(1).or_default().set(Role::Fit);
        reading.apply_event(
            PageEvent::Ready {
                page: 1,
                role: WorkRole::Full,
                image: Box::new(image),
                thumb: None,
            },
            &mut store,
            &mut messages,
            &mut touched,
            &mut redraw,
        );
        assert!(!reading.is_faulted(1));
    }

    #[test]
    fn work_plan_bounds_lookahead_by_window_bytes() {
        let mode = ReadingMode::Normal;
        let (mut reading, _) =
            reading_with_pages(60, Some("010.png"), mode);
        reading.shown = Some(Spread::lone(10));

        let normal_plan = reading.build_work(
            mode,
            false,
            None,
            testkit::limits(),
            &FakeStore {
                average_fit_bytes: Some(8 * 1024 * 1024),
                ..FakeStore::default()
            },
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

        let tight_plan = reading.build_work(
            mode,
            false,
            None,
            testkit::limits(),
            &FakeStore {
                average_fit_bytes: Some(80 * 1024 * 1024),
                ..FakeStore::default()
            },
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

        let warm_plan = reading.build_work(
            mode,
            true,
            None,
            testkit::limits(),
            &FakeStore::from_roles([(5, Role::Thumb), (6, Role::Thumb)]),
        );
        assert_eq!(warm_plan, WorkPlan::default());

        let cold_plan = reading.build_work(
            mode,
            true,
            None,
            testkit::limits(),
            &FakeStore::default(),
        );
        assert_eq!(
            cold_plan.current,
            vec![
                Request {
                    page: 6,
                    role: WorkRole::Fit,
                },
                Request {
                    page: 5,
                    role: WorkRole::Fit,
                },
            ]
        );
        assert_eq!(cold_plan.lookahead, Vec::new());
        assert_eq!(cold_plan.thumbs, Vec::new());
    }
}
