use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use crate::animation::Player;
use crate::book::BookEvent;
use crate::browse;
use crate::browse::Direction;
use crate::browse::Motion;
use crate::browse::PageNames;
use crate::browse::ReadingMode;
use crate::browse::Request;
use crate::browse::Role;
use crate::browse::Roles;
use crate::browse::Spread;
use crate::catalog;
use crate::catalog::PageName;
use crate::decode::Format;
use crate::protocol::BookInfo;
use crate::protocol::ImageLimits;
use crate::protocol::PlaybackCommand;
use crate::protocol::PlaybackEvent;
use crate::protocol::PlaybackId;
use crate::protocol::TextureFault;
use crate::protocol::ValidatedImage;
use crate::ui::StatusLine;

const FIT_KEEP_SPREADS: usize = 2;

/// Native page dimensions and decoded format, fixed across roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PageInfo {
    dims: (u32, u32),
    format: Format,
}

pub struct PendingUpload {
    pub page: usize,
    pub role: Role,
    pub image: Box<ValidatedImage>,
}

enum PagePlayback {
    Playing(Player<Box<ValidatedImage>>),
    /// Sticky for the book so a stopped decoder is not retried on revisit.
    Refused,
}

#[derive(Default)]
struct PageState {
    info: Option<PageInfo>,
    /// Page faults stay sticky so a later upload or blip cannot erase them.
    fault: Option<String>,
    playback: Option<PagePlayback>,
}

impl PageState {
    fn player_mut(
        &mut self,
        id: PlaybackId,
    ) -> Option<&mut Player<Box<ValidatedImage>>> {
        match &mut self.playback {
            Some(PagePlayback::Playing(player)) if player.id() == id => {
                Some(player)
            }
            Some(PagePlayback::Playing(_) | PagePlayback::Refused)
            | None => {
                tracing::debug!(?id, "stale playback update");
                None
            }
        }
    }

    fn stop_playback(&mut self) -> Option<PlaybackCommand> {
        if !matches!(self.playback, Some(PagePlayback::Playing(_))) {
            return None;
        }
        let Some(PagePlayback::Playing(player)) = self.playback.take()
        else {
            return None;
        };
        let id = player.id();
        let role = player.role();
        tracing::debug!(?id, ?role, "playback player stop");
        Some(PlaybackCommand::Stop { id })
    }

    fn refuse_playback(&mut self) -> Option<PlaybackCommand> {
        let stop = self.stop_playback();
        self.playback = Some(PagePlayback::Refused);
        stop
    }
}

pub struct Reading {
    path: PathBuf,
    archive: bool,
    dir_name: Option<String>,
    names: PageNames,
    target: usize,
    shown: Option<(ReadingMode, Spread)>,
    pages: BTreeMap<usize, PageState>,
    pending_uploads: VecDeque<PendingUpload>,
    motion: Motion,
    /// Drives child prefetch order and skim tie-breaking.
    direction: Direction,
    last_turn: Option<Instant>,
    hold_since: Option<Instant>,
    hold_logged: bool,
    sent_requests: Vec<Request>,
    next_generation: u32,
}

impl Reading {
    pub fn open(
        path: PathBuf,
        info: BookInfo,
        land_on: Option<&str>,
        mode: ReadingMode,
    ) -> (Self, Vec<Request>) {
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
        let requests = cold_requests(
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
            pending_uploads: VecDeque::new(),
            motion: Motion::Still,
            direction: Direction::Forward,
            last_turn: None,
            hold_since: None,
            hold_logged: false,
            sent_requests: requests.clone(),
            next_generation: 0,
        };
        (reading, requests)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn page_count(&self) -> NonZeroUsize {
        self.names.count()
    }

    pub fn target(&self) -> usize {
        self.target
    }

    pub fn shown(&self) -> Option<Spread> {
        self.shown.map(|(_, spread)| spread)
    }

    pub fn direction(&self) -> Direction {
        self.direction
    }

    pub fn take_pending_uploads(&mut self) -> VecDeque<PendingUpload> {
        use std::mem;

        mem::take(&mut self.pending_uploads)
    }

    pub fn sibling(&self, forward: bool) -> Option<PathBuf> {
        catalog::sibling(&self.path, self.archive, forward)
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

    fn target_spread(&self, mode: ReadingMode) -> Spread {
        browse::spread_for(self.target, mode, self.page_count())
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
        tracing::debug!(
            index = self.target,
            ?direction,
            motion = ?self.motion,
            "turn"
        );
    }

    fn commit_shown(&mut self, mode: ReadingMode) -> bool {
        let Some((shown_mode, spread)) = self.shown else {
            return false;
        };
        let target = self.target;
        if shown_mode != mode || spread.first == target {
            return false;
        }
        self.target = spread.first;
        true
    }

    pub fn turn(
        &mut self,
        direction: Direction,
        mode: ReadingMode,
        now: Instant,
    ) -> bool {
        let page_count = self.page_count();
        let next =
            browse::advance(self.target, direction, mode, page_count);
        if self.motion == Motion::Skimming {
            let Some((_, spread)) = self.shown else {
                return false;
            };
            // Keep skims within one spread of what is on screen.
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
        if self.motion == Motion::Skimming {
            // The new pairing is a different coordinate system; commit the
            // shown spread before switching.
            let from = self.target;
            if self.commit_shown(previous) {
                tracing::debug!(
                    from,
                    to = self.target,
                    "mode switch commits shown"
                );
            }
        }
        let pairing_changed = previous.pairs() != next.pairs();
        if pairing_changed {
            self.target =
                browse::spread_for(self.target, next, self.page_count())
                    .first;
        }
        self.direction = Direction::Forward;
        pairing_changed
    }

    pub fn set_motion(
        &mut self,
        motion: Motion,
        mode: ReadingMode,
    ) -> bool {
        if motion == self.motion {
            return false;
        }
        if motion == Motion::Still {
            // A turn may not have reached the screen; commit the shown spread
            // so its thumb promotes in place.
            let from = self.target;
            if self.commit_shown(mode) {
                tracing::debug!(from, to = self.target, "commit shown");
            }
        }
        tracing::debug!(
            ?motion,
            target = self.target,
            shown = ?self.shown(),
            "motion"
        );
        self.motion = motion;
        true
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

    pub fn apply_event(
        &mut self,
        event: BookEvent,
    ) -> Option<PlaybackCommand> {
        match event {
            BookEvent::Ready { page, role, image } => {
                self.pending_uploads.push_back(PendingUpload {
                    page,
                    role,
                    image,
                });
                None
            }
            BookEvent::Failed { page, fault } => {
                let state = self.pages.entry(page).or_default();
                state.fault = Some(fault);
                state.stop_playback()
            }
            BookEvent::Playback(playback) => match playback {
                PlaybackEvent::Frame { id, delay, image } => {
                    if let Some(player) = self.player_mut(id)
                        && !player.on_frame(delay, image)
                    {
                        tracing::debug!(?id, "unrequested playback frame");
                    }
                    None
                }
                PlaybackEvent::Finished { id } => {
                    if let Some(player) = self.player_mut(id)
                        && player.on_finished()
                    {
                        tracing::debug!(?id, "playback finished");
                    }
                    None
                }
                PlaybackEvent::Failed { id } => {
                    let Some(state) = self.pages.get_mut(&id.page) else {
                        tracing::debug!(?id, "stale playback update");
                        return None;
                    };
                    if state.player_mut(id).is_some() {
                        tracing::warn!(?id, "playback failed");
                        return state.refuse_playback();
                    }
                    None
                }
            },
            BookEvent::Died(_) => None,
        }
    }

    pub fn record_uploaded(
        &mut self,
        page: usize,
        image: &ValidatedImage,
    ) {
        let dims = image.meta.page_dims;
        let format = image.meta.format;
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

    pub fn record_upload_refused(
        &mut self,
        page: usize,
        role: Role,
        has_settled_image: bool,
        fault: TextureFault,
    ) {
        if role != Role::Thumb && !has_settled_image {
            self.pages.entry(page).or_default().fault =
                Some(fault.to_string());
        }
    }

    pub fn resolve_shown(
        &mut self,
        mode: ReadingMode,
        now: Instant,
        roles: impl Fn(usize) -> Roles,
    ) -> (Option<Spread>, bool) {
        let shown = self.choose_presented(mode, &roles);
        let previous_shown = self.shown();
        let changed = shown != previous_shown;
        if changed {
            let seats = shown.map(|spread| {
                spread.seats(mode).map(|seat| {
                    seat.map_or("empty", |page| {
                        roles(page).best().map_or("none", Into::into)
                    })
                })
            });
            let took = self
                .last_turn
                .map(|last| now.saturating_duration_since(last));
            tracing::debug!(
                from = ?previous_shown,
                to = ?shown,
                motion = ?self.motion,
                ?seats,
                ?took,
                "shown"
            );
        }
        // Record every frame so a mode switch retags an unchanged spread.
        self.shown = shown.map(|spread| (mode, spread));
        (shown, changed)
    }

    /// Target once presentable, else the nearest presentable start in
    /// `pages` between `shown` and `target` (monotonic skim), else `shown`.
    fn choose_presented(
        &self,
        mode: ReadingMode,
        roles: impl Fn(usize) -> Roles,
    ) -> Option<Spread> {
        let target = self.target_spread(mode);
        let forward = self.direction == Direction::Forward;
        let skimming = self.motion == Motion::Skimming;
        let shown = self.shown();
        let can_show = |candidate: usize| {
            let state = self.pages.get(&candidate);
            let faulted =
                state.is_some_and(|page_state| page_state.fault.is_some());
            if faulted {
                return true;
            }
            if state.and_then(|page_state| page_state.info).is_none() {
                return false;
            }
            let page_roles = roles(candidate);
            if skimming {
                page_roles.any()
            } else {
                page_roles.has_settled_image()
            }
        };
        let presentable = |spread: Spread| {
            spread.seats(mode).into_iter().flatten().all(can_show)
        };
        if presentable(target) {
            return Some(target);
        }
        if !skimming {
            return shown;
        }
        let (low, high) = match shown {
            None => (0, usize::MAX),
            Some(previous) => {
                let (low, high) = if forward {
                    (previous.first, target.first)
                } else {
                    (target.first, previous.first)
                };
                if low > high {
                    return shown;
                }
                (low, high)
            }
        };
        let page_count = self.page_count();
        let mut best: Option<Spread> = None;
        for &page in self.pages.keys() {
            // A paired spread starting at `high` may be visited via its second seat.
            if page < low || page > high.saturating_add(1) {
                continue;
            }
            if self
                .pages
                .get(&page)
                .and_then(|page_state| page_state.info)
                .is_none()
            {
                continue;
            }
            let spread = browse::spread_for(page, mode, page_count);
            if spread.first < low
                || spread.first > high
                || !presentable(spread)
            {
                continue;
            }
            let distance = spread.first.abs_diff(target.first);
            let closer = best.is_none_or(|current| {
                let current_distance =
                    current.first.abs_diff(target.first);
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

    pub fn on_screen(&self, mode: ReadingMode) -> Spread {
        self.shown().unwrap_or_else(|| self.target_spread(mode))
    }

    pub fn seat_dims(
        &self,
        mode: ReadingMode,
        spread: Spread,
    ) -> [Option<(u32, u32)>; 2] {
        spread.seats(mode).map(|seat| {
            let page = seat?;
            self.pages.get(&page)?.info.map(|info| info.dims)
        })
    }

    pub fn update_requests(
        &mut self,
        mode: ReadingMode,
        on_screen: Spread,
        on_screen_scale: Option<f64>,
        limits: ImageLimits,
        roles: impl Fn(usize) -> Roles,
    ) -> Option<Vec<Request>> {
        let demand = self.demand_spread(on_screen, mode);
        let requests = self.request_roles(
            demand,
            on_screen,
            mode,
            on_screen_scale,
            limits,
            roles,
        );
        if requests == self.sent_requests {
            return None;
        }
        tracing::debug!(?requests, ?demand, "sent requests");
        self.sent_requests.clone_from(&requests);
        Some(requests)
    }

    /// While skimming, demand the adjacent spread so paired pages load
    /// in step with display.
    fn demand_spread(
        &self,
        on_screen: Spread,
        mode: ReadingMode,
    ) -> Spread {
        if self.motion == Motion::Still {
            return self.target_spread(mode);
        }
        let page_count = self.page_count();
        let next_start = browse::advance(
            on_screen.first,
            self.direction,
            mode,
            page_count,
        );
        browse::spread_for(next_start, mode, page_count)
    }

    fn preupload_spread(&self, mode: ReadingMode) -> Option<Spread> {
        if self.motion != Motion::Still {
            return None;
        }
        let target = self.target_spread(mode);
        if self.shown() != Some(target) {
            return None;
        }
        let page_count = self.page_count();
        let next_start =
            browse::step(target.first, self.direction, mode, page_count)?;
        Some(browse::spread_for(next_start, mode, page_count))
    }

    fn request_roles(
        &self,
        demand: Spread,
        on_screen: Spread,
        mode: ReadingMode,
        on_screen_scale: Option<f64>,
        limits: ImageLimits,
        roles: impl Fn(usize) -> Roles,
    ) -> Vec<Request> {
        let still = self.motion == Motion::Still;
        let mut requests = Vec::new();
        for spread in
            [Some(demand), Some(on_screen), self.preupload_spread(mode)]
                .into_iter()
                .flatten()
        {
            for page in spread.seats(mode).into_iter().flatten() {
                let state = self.pages.get(&page);
                if state
                    .is_some_and(|page_state| page_state.fault.is_some())
                {
                    continue;
                }
                let resident = roles(page);
                if still {
                    if !resident.has_settled_image() {
                        push_request(&mut requests, page, Role::Fit);
                    }
                    if on_screen.contains(page)
                        && !resident.contains(Role::Full)
                        && on_screen_scale.is_some_and(|scale| {
                            state
                                .and_then(|page_state| page_state.info)
                                .is_some_and(|info| {
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

    pub fn note_hold(&mut self, mode: ReadingMode, now: Instant) {
        use std::time::Duration;

        const HOLD_LOG_THRESHOLD: Duration = Duration::from_millis(250);

        let target = self.target_spread(mode);
        let shown = self.shown();
        if self.motion == Motion::Still && shown != Some(target) {
            let since = *self.hold_since.get_or_insert(now);
            let waited = now.saturating_duration_since(since);
            if !self.hold_logged && waited >= HOLD_LOG_THRESHOLD {
                self.hold_logged = true;
                tracing::info!(
                    ?target,
                    ?shown,
                    ?waited,
                    "holding for a fit image"
                );
            }
        } else {
            self.hold_since = None;
            self.hold_logged = false;
        }
    }

    pub fn retains_texture(
        &self,
        mode: ReadingMode,
        on_screen: Spread,
        page: usize,
        role: Role,
    ) -> bool {
        if self.sent_requests.contains(&Request { page, role }) {
            return true;
        }
        let target = self.target_spread(mode);
        match role {
            Role::Thumb => true,
            Role::Fit => {
                let start =
                    browse::spread_for(page, mode, self.page_count())
                        .first;
                start.abs_diff(on_screen.first) <= FIT_KEEP_SPREADS
                    || start.abs_diff(target.first) <= FIT_KEEP_SPREADS
            }
            Role::Full => on_screen.contains(page),
        }
    }

    /// A player follows the role its seat draws; a role change restarts it.
    pub fn sync_playback(
        &mut self,
        mode: ReadingMode,
        on_screen: Spread,
        roles: impl Fn(usize) -> Roles,
    ) -> Vec<PlaybackCommand> {
        let targets = if self.motion == Motion::Still {
            on_screen.seats(mode).map(|seat| {
                let page = seat?;
                let state = self.pages.get(&page)?;
                if state.fault.is_some() {
                    return None;
                }
                let info = state.info?;
                if !info.format.may_animate() {
                    return None;
                }
                let role = roles(page).best()?;
                (role != Role::Thumb).then_some((page, role))
            })
        } else {
            [None, None]
        };
        let mut commands = Vec::new();
        for (&page, state) in &mut self.pages {
            let keep = match &state.playback {
                Some(PagePlayback::Playing(player)) => {
                    state.fault.is_none()
                        && targets.iter().flatten().any(
                            |&(want_page, want_role)| {
                                want_page == page
                                    && want_role == player.role()
                            },
                        )
                }
                Some(PagePlayback::Refused) | None => true,
            };
            if !keep && let Some(command) = state.stop_playback() {
                commands.push(command);
            }
        }
        for &(page, role) in targets.iter().flatten() {
            let Some(state) = self.pages.get_mut(&page) else {
                continue;
            };
            if state.info.is_none()
                || state.fault.is_some()
                || state.playback.is_some()
            {
                continue;
            }
            let id = PlaybackId {
                page,
                generation: self.next_generation,
            };
            self.next_generation = id.generation.wrapping_add(1);
            state.playback =
                Some(PagePlayback::Playing(Player::new(id, role)));
            commands.push(PlaybackCommand::Start { id, role });
            tracing::debug!(?id, ?role, "playback player start");
        }
        commands
    }

    pub fn tick_playback(
        &mut self,
        now: Instant,
        mut upload: impl FnMut(
            usize,
            Role,
            &ValidatedImage,
        ) -> Result<(), TextureFault>,
    ) -> PlaybackTick {
        use crate::animation::Due;

        let mut next_deadline: Option<Instant> = None;
        let mut commands = Vec::new();
        let mut updated = false;
        for (&page, state) in &mut self.pages {
            let Some(PagePlayback::Playing(player)) = &mut state.playback
            else {
                continue;
            };
            let id = player.id();
            let role = player.role();
            match player.tick(now) {
                Due::Show { image, next_due } => {
                    if let Err(error) = upload(page, role, &image) {
                        tracing::warn!(
                            page,
                            ?role,
                            %error,
                            "playback frame upload failed"
                        );
                        if let Some(command) = state.refuse_playback() {
                            commands.push(command);
                        }
                        continue;
                    }
                    tracing::debug!(
                        page,
                        ?role,
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
                commands.push(PlaybackCommand::Next { id });
            }
        }
        PlaybackTick {
            next_deadline,
            commands,
            updated,
        }
    }

    pub fn is_waiting(
        &self,
        on_screen: Spread,
        has_texture: impl Fn(usize) -> bool,
    ) -> bool {
        use std::iter;

        iter::once(on_screen.first)
            .chain(on_screen.second)
            .all(|page| {
                let state = self.pages.get(&page);
                let decoded = state
                    .and_then(|page_state| page_state.info)
                    .is_some_and(|_| has_texture(page));
                let faulted = state
                    .and_then(|page_state| page_state.fault.as_deref())
                    .is_some();
                !decoded && !faulted
            })
    }

    pub fn status_line<'fault>(
        &'fault self,
        on_screen: Spread,
        mode: ReadingMode,
        zoom: f64,
        book_fault: Option<&'fault str>,
        has_texture: impl Fn(usize) -> bool,
    ) -> StatusLine<'fault> {
        use crate::decode;
        use crate::ui::ReadingStatus;
        use std::iter;

        let mut failure: Option<(usize, &'fault str)> = None;
        let mut first_decoded: Option<(usize, PageInfo)> = None;
        for page in iter::once(on_screen.first).chain(on_screen.second) {
            let state = self.pages.get(&page);
            let info = state
                .and_then(|page_state| page_state.info)
                .filter(|_| has_texture(page));
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
        let (named_page, page_fault) = match failure {
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
        StatusLine {
            reading: Some(ReadingStatus {
                page_count: self.page_count().get(),
                spread: on_screen,
                page_dims,
                actual_format,
                zoom,
            }),
            mode,
            toast: false,
            name: self.page_name(named_page),
            waiting: failure.is_none() && first_decoded.is_none(),
            fault: page_fault.or(book_fault),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct PlaybackTick {
    pub next_deadline: Option<Instant>,
    pub commands: Vec<PlaybackCommand>,
    pub updated: bool,
}

fn cold_requests(spread: Spread, mode: ReadingMode) -> Vec<Request> {
    spread
        .seats(mode)
        .into_iter()
        .flatten()
        .map(|page| Request {
            page,
            role: Role::Fit,
        })
        .collect()
}

fn push_request(requests: &mut Vec<Request>, page: usize, role: Role) {
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

    fn test_limits() -> ImageLimits {
        const FIT_BOX: (u32, u32) = (2560, 1440);
        const MAX_TEXTURE_EDGE: u32 = 16_384;
        ImageLimits::new(FIT_BOX, MAX_TEXTURE_EDGE)
    }

    fn roles_with(role: Role) -> Roles {
        let mut roles = Roles::default();
        roles.set(role);
        roles
    }

    fn reading_with_pages(
        count: usize,
        land_on: Option<&str>,
        mode: ReadingMode,
    ) -> (Reading, Vec<Request>) {
        let names = (0..count)
            .map(|index| format!("{index:03}.png"))
            .collect::<Vec<_>>();
        let info = BookInfo {
            names: PageNames::new(names).expect("non-empty test book"),
            archive: false,
        };
        Reading::open(PathBuf::from("/tmp/book"), info, land_on, mode)
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

    fn wire_image(
        width: u32,
        height: u32,
        format: Format,
    ) -> Box<ValidatedImage> {
        use crate::decode::Decoded;
        use crate::decode::Transfer;
        use crate::protocol::WireImage;

        const RGBA_BYTES_PER_PIXEL: usize = 4;
        let bytes =
            vec![
                0_u8;
                width as usize * height as usize * RGBA_BYTES_PER_PIXEL
            ];
        let decoded = Decoded::rgba(
            width,
            height,
            format,
            Transfer::Srgb,
            bytes.into_boxed_slice(),
        )
        .expect("valid rgba test image");
        Box::new(
            WireImage::from_decoded(&decoded)
                .validate()
                .expect("valid test wire image"),
        )
    }

    fn ready_event(
        page: usize,
        role: Role,
        width: u32,
        height: u32,
        format: Format,
    ) -> BookEvent {
        BookEvent::Ready {
            page,
            role,
            image: wire_image(width, height, format),
        }
    }

    fn check_texture_edge(
        image: &ValidatedImage,
        max_edge: u32,
    ) -> Result<(), TextureFault> {
        if image.meta.width > max_edge || image.meta.height > max_edge {
            return Err(TextureFault {
                dims: (image.meta.width, image.meta.height),
                max_edge,
            });
        }
        Ok(())
    }

    #[test]
    fn a_full_image_is_only_requested_when_the_fit_has_fewer_pixels() {
        let limits = test_limits();
        assert!(!full_wanted(16.0, (1920, 1080), limits));
        assert!(!full_wanted(16.0, (2560, 1440), limits));
        let portrait = (3628, 5148);
        assert!(!full_wanted(1440.0 / 5148.0, portrait, limits));
        assert!(full_wanted(0.4, portrait, limits));
        let four_k = (3840, 2160);
        assert!(!full_wanted(0.66, four_k, limits));
        assert!(full_wanted(0.67, four_k, limits));
        // Fractional Wayland scales can push fit rendering a sub-pixel past fit_box.
        assert!(!full_wanted(1440.5 / 3000.0, (5000, 3000), limits));
        let ten_k = (10080, 5640);
        assert!(!full_wanted(0.25, ten_k, limits));
        assert!(full_wanted(0.26, ten_k, limits));
        assert!(!full_wanted(
            16.0,
            ten_k,
            ImageLimits::new((2560, 1440), 2560)
        ));
        // Dimensions above f32's exact-integer limit (2^24) keep sub-pixel precision.
        let wide_strip = (20_000_001, 1);
        assert!(!full_wanted(2560.5 / 20_000_001.0, wide_strip, limits));
        assert!(full_wanted(2561.5 / 20_000_001.0, wide_strip, limits));
    }

    #[test]
    fn retention_keeps_fit_images_within_the_role_bounds() {
        let (mut reading, _) =
            reading_with_pages(20, Some("010.png"), ReadingMode::Normal);
        reading.shown = Some((ReadingMode::Normal, Spread::lone(10)));
        assert!(reading.retains_texture(
            ReadingMode::Normal,
            Spread::lone(10),
            10,
            Role::Full,
        ));
        assert!(!reading.retains_texture(
            ReadingMode::Normal,
            Spread::lone(10),
            12,
            Role::Full,
        ));
        assert!(reading.retains_texture(
            ReadingMode::Normal,
            Spread::lone(10),
            12,
            Role::Fit,
        ));
        assert!(!reading.retains_texture(
            ReadingMode::Normal,
            Spread::lone(10),
            13,
            Role::Fit,
        ));
        assert!(reading.retains_texture(
            ReadingMode::Normal,
            Spread::lone(10),
            0,
            Role::Thumb,
        ));
    }

    #[test]
    fn full_requests_follow_the_on_screen_scale() {
        let (mut reading, _) =
            reading_with_pages(12, Some("002.png"), ReadingMode::Normal);
        reading.shown = Some((ReadingMode::Normal, Spread::lone(1)));
        seed_page(&mut reading, 1, (4000, 6000), Format::Png);
        seed_page(&mut reading, 2, (4000, 6000), Format::Png);

        let on_screen = reading.on_screen(ReadingMode::Normal);
        assert_eq!(on_screen, Spread::lone(1));
        let requests = reading
            .update_requests(
                ReadingMode::Normal,
                on_screen,
                Some(1.0),
                test_limits(),
                |_| Roles::default(),
            )
            .expect("requests changed");
        assert!(
            requests.contains(&Request {
                page: 1,
                role: Role::Full
            }),
            "the on-screen page asks for its full image at the on-screen scale"
        );
        assert!(
            !requests.contains(&Request {
                page: 2,
                role: Role::Full
            }),
            "the target page is not upscaled"
        );
    }

    #[test]
    fn presentation_advances_through_presentable_spreads() {
        let mode = ReadingMode::Normal;
        let (mut reading, _) = reading_with_pages(12, None, mode);
        for page in [0, 2, 4, 8] {
            seed_page(&mut reading, page, (2, 2), Format::Png);
        }
        let roles = |fit_pages: &'static [usize]| {
            move |page: usize| {
                if fit_pages.contains(&page) {
                    roles_with(Role::Fit)
                } else {
                    Roles::default()
                }
            }
        };
        let lone = Spread::lone;
        reading.motion = Motion::Skimming;

        // The target wins as soon as it has a pixel.
        reading.target = 4;
        reading.shown = Some((mode, lone(0)));
        assert_eq!(
            reading.choose_presented(mode, roles(&[0, 2, 4])),
            Some(lone(4))
        );

        // Otherwise the nearest presentable spread between the shown and the
        // target; page 8 is known but has no pixels.
        reading.target = 9;
        reading.shown = Some((mode, lone(0)));
        assert_eq!(
            reading.choose_presented(mode, roles(&[0, 2, 4])),
            Some(lone(4))
        );

        // Nothing between the two: the shown spread keeps its pixels.
        reading.target = 9;
        reading.shown = Some((mode, lone(2)));
        assert_eq!(
            reading.choose_presented(mode, roles(&[0])),
            Some(lone(2))
        );

        // A cold start picks the nearest presentable spread.
        reading.target = 5;
        reading.shown = None;
        assert_eq!(
            reading.choose_presented(mode, roles(&[0, 2, 4])),
            Some(lone(4))
        );

        // A cold start presents only decoded spreads.
        reading.target = 5;
        reading.shown = None;
        assert_eq!(reading.choose_presented(mode, roles(&[])), None);

        // Skimming across a wrap holds the shown spread until target lands.
        reading.target = 0;
        reading.shown = Some((mode, lone(4)));
        assert_eq!(
            reading.choose_presented(mode, roles(&[2, 4])),
            Some(lone(4))
        );

        // Still mode holds the shown spread on jumps until target lands.
        reading.motion = Motion::Still;
        reading.target = 9;
        reading.shown = Some((mode, lone(0)));
        assert_eq!(
            reading.choose_presented(mode, roles(&[0, 4])),
            Some(lone(0))
        );
    }

    #[test]
    fn paired_spread_waits_for_every_seat() {
        let mode = ReadingMode::Manga;
        let pair = Spread {
            first: 1,
            second: Some(2),
        };
        let shown = Spread::lone(0);

        let (mut reading, _) =
            reading_with_pages(12, Some("001.png"), mode);
        reading.shown = Some((mode, shown));
        seed_page(&mut reading, 0, (2, 2), Format::Png);
        seed_page(&mut reading, 2, (2, 2), Format::Png);
        assert_eq!(
            reading.choose_presented(mode, |page| {
                if page == 2 {
                    roles_with(Role::Fit)
                } else {
                    Roles::default()
                }
            }),
            Some(shown)
        );

        seed_page(&mut reading, 1, (2, 2), Format::Png);
        assert_eq!(
            reading.choose_presented(mode, |page| {
                if [1, 2].contains(&page) {
                    roles_with(Role::Fit)
                } else {
                    Roles::default()
                }
            }),
            Some(pair)
        );
    }

    #[test]
    fn mode_switch_relands_only_on_pairing_change() {
        let (mut reading, _) =
            reading_with_pages(10, Some("002.png"), ReadingMode::Normal);
        assert!(
            reading.switch_mode(ReadingMode::Normal, ReadingMode::Manga)
        );
        assert_eq!(reading.target(), 1);

        let (mut cover, _) =
            reading_with_pages(10, None, ReadingMode::Normal);
        assert!(
            cover.switch_mode(ReadingMode::Normal, ReadingMode::Comic)
        );
        assert_eq!(cover.target(), 0);

        let (mut pair_to_pair, _) =
            reading_with_pages(10, Some("005.png"), ReadingMode::Manga);
        assert!(
            !pair_to_pair
                .switch_mode(ReadingMode::Manga, ReadingMode::Comic)
        );
        assert_eq!(pair_to_pair.target(), 5);

        assert!(
            !pair_to_pair
                .switch_mode(ReadingMode::Manga, ReadingMode::Manga)
        );
        assert_eq!(pair_to_pair.target(), 5);

        // Switching while skimming commits the shown spread under the previous
        // mode before mapping into the new pairing.
        pair_to_pair.shown = Some((
            ReadingMode::Comic,
            Spread {
                first: 3,
                second: Some(4),
            },
        ));
        pair_to_pair.motion = Motion::Skimming;
        pair_to_pair.target = 7;
        assert!(
            pair_to_pair
                .switch_mode(ReadingMode::Comic, ReadingMode::Normal)
        );
        assert_eq!(pair_to_pair.target(), 3);
    }

    #[test]
    fn skimming_turn_waits_for_shown_when_one_spread_ahead() {
        let mode = ReadingMode::Manga;
        let now = Instant::now();
        let (mut reading, _) =
            reading_with_pages(10, Some("001.png"), mode);

        reading.motion = Motion::Skimming;
        assert!(!reading.turn(Direction::Forward, mode, now));
        assert_eq!(reading.target(), 1);

        reading.motion = Motion::Still;
        assert!(reading.turn(Direction::Forward, mode, now));
        assert_eq!(reading.target(), 3);

        reading.shown = Some((
            mode,
            browse::spread_for(3, mode, reading.page_count()),
        ));
        reading.motion = Motion::Skimming;
        assert!(reading.turn(Direction::Forward, mode, now));
        assert_eq!(reading.target(), 5);
        assert!(!reading.turn(Direction::Forward, mode, now));
        assert_eq!(reading.target(), 5);

        assert!(reading.turn(Direction::Backward, mode, now));
        assert_eq!(reading.target(), 3);
    }

    #[test]
    fn playback_targets_need_an_animated_page_and_a_still_frame() {
        let mode = ReadingMode::Normal;
        let (mut reading, _) = reading_with_pages(20, None, mode);
        for page in [4, 6, 10, 12] {
            seed_page(&mut reading, page, (2, 2), Format::Gif);
        }
        seed_page(&mut reading, 8, (2, 2), Format::Png);
        reading.pages.entry(10).or_default().fault =
            Some(String::from("bad page"));

        let roles = |page: usize| {
            roles_with(if page == 6 { Role::Thumb } else { Role::Fit })
        };

        assert_eq!(
            reading.sync_playback(mode, Spread::lone(6), roles),
            [],
            "a thumb-only page never plays"
        );
        assert_eq!(
            reading.sync_playback(mode, Spread::lone(8), roles),
            [],
            "a still image never plays"
        );
        assert_eq!(
            reading.sync_playback(mode, Spread::lone(10), roles),
            [],
            "a faulted page never plays"
        );
        reading.motion = Motion::Skimming;
        assert_eq!(
            reading.sync_playback(mode, Spread::lone(12), roles),
            [],
            "skimming never plays"
        );
        reading.motion = Motion::Still;
        assert_eq!(
            reading.sync_playback(mode, Spread::lone(4), roles),
            vec![PlaybackCommand::Start {
                id: PlaybackId {
                    page: 4,
                    generation: 0
                },
                role: Role::Fit,
            }]
        );
    }

    #[test]
    fn still_frames_never_request_thumbs() {
        let (_, opening_requests) =
            reading_with_pages(10, None, ReadingMode::Normal);
        assert_eq!(
            opening_requests,
            vec![Request {
                page: 0,
                role: Role::Fit
            }]
        );

        let (mut reading, _) =
            reading_with_pages(10, Some("002.png"), ReadingMode::Normal);
        seed_page(&mut reading, 2, (4000, 6000), Format::Png);
        let requests = reading
            .update_requests(
                ReadingMode::Normal,
                Spread::lone(2),
                Some(1.0),
                test_limits(),
                |page| {
                    if page == 2 {
                        roles_with(Role::Thumb)
                    } else {
                        Roles::default()
                    }
                },
            )
            .expect("requests updated");
        assert_eq!(
            requests,
            vec![
                Request {
                    page: 2,
                    role: Role::Fit
                },
                Request {
                    page: 2,
                    role: Role::Full
                }
            ]
        );
    }

    #[test]
    fn skim_skips_thumb_when_any_role_is_resident() {
        let (mut reading, _) =
            reading_with_pages(10, Some("009.png"), ReadingMode::Normal);
        reading.shown = Some((ReadingMode::Normal, Spread::lone(4)));
        reading.motion = Motion::Skimming;
        seed_page(&mut reading, 4, (800, 600), Format::Png);

        let requests = reading
            .update_requests(
                ReadingMode::Normal,
                Spread::lone(4),
                Some(1.0),
                test_limits(),
                |page| {
                    if page == 4 {
                        roles_with(Role::Fit)
                    } else {
                        Roles::default()
                    }
                },
            )
            .expect("requests updated");
        assert_eq!(
            requests,
            vec![Request {
                page: 5,
                role: Role::Thumb
            }]
        );
    }

    #[test]
    fn status_line_blames_page_before_book_and_summarizes_spread() {
        let (mut reading, _) =
            reading_with_pages(10, None, ReadingMode::Manga);
        let spread = Spread {
            first: 1,
            second: Some(2),
        };
        let book_fault = "the book process did not answer in time";

        let waiting = reading.status_line(
            spread,
            ReadingMode::Manga,
            1.0,
            None,
            |_| false,
        );
        assert!(waiting.waiting);
        assert_eq!(waiting.fault, None);
        assert_eq!(waiting.name.file, "001.png");
        assert!(
            waiting
                .reading
                .is_some_and(|status| status.page_dims.is_none())
        );

        let book_only = reading.status_line(
            spread,
            ReadingMode::Manga,
            1.0,
            Some(book_fault),
            |_| false,
        );
        assert_eq!(book_only.fault, Some(book_fault));

        seed_page(&mut reading, 1, (800, 600), Format::Png);
        let settled = reading.status_line(
            spread,
            ReadingMode::Manga,
            1.0,
            None,
            |page| page == 1,
        );
        assert!(!settled.waiting);
        assert_eq!(settled.name.file, "001.png");
        let settled_reading = settled.reading.expect("reading status");
        assert_eq!(settled_reading.page_dims, Some((800, 600)));
        assert_eq!(settled_reading.actual_format, None);

        seed_page(&mut reading, 2, (1200, 900), Format::Jxl);
        let single_mismatch = reading.status_line(
            Spread::lone(2),
            ReadingMode::Normal,
            0.5,
            None,
            |page| page == 2,
        );
        assert_eq!(
            single_mismatch
                .reading
                .expect("reading status")
                .actual_format,
            Some(Format::Jxl)
        );

        reading.apply_event(BookEvent::Failed {
            page: 2,
            fault: String::from("page"),
        });
        let page_faulted = reading.status_line(
            spread,
            ReadingMode::Manga,
            1.0,
            Some(book_fault),
            |page| page == 1,
        );
        assert!(!page_faulted.waiting);
        assert_eq!(page_faulted.fault, Some("page"));
        assert_eq!(page_faulted.name.file, "002.png");
    }

    #[test]
    fn refused_upload_faults_only_unsettled_non_thumb_roles() {
        let fault = || TextureFault {
            dims: (200, 200),
            max_edge: 100,
        };
        let (mut reading, _) =
            reading_with_pages(4, None, ReadingMode::Normal);

        reading.record_upload_refused(0, Role::Thumb, false, fault());
        assert!(
            reading
                .pages
                .get(&0)
                .is_none_or(|state| state.fault.is_none())
        );

        reading.record_upload_refused(0, Role::Fit, false, fault());
        assert!(
            reading
                .pages
                .get(&0)
                .and_then(|state| state.fault.as_ref())
                .is_some()
        );

        reading.record_upload_refused(1, Role::Full, true, fault());
        assert!(
            reading
                .pages
                .get(&1)
                .is_none_or(|state| state.fault.is_none())
        );
    }

    #[test]
    fn sync_motion_commits_shown_spread_when_returning_to_still() {
        let (mut reading, _) =
            reading_with_pages(20, Some("010.png"), ReadingMode::Normal);
        reading.shown = Some((ReadingMode::Normal, Spread::lone(4)));

        assert!(reading.set_motion(Motion::Skimming, ReadingMode::Normal));
        assert_eq!(reading.target(), 10);

        assert!(reading.set_motion(Motion::Still, ReadingMode::Normal));
        assert_eq!(reading.target(), 4);
        assert!(!reading.set_motion(Motion::Still, ReadingMode::Normal));

        // A spread shown under another mode is never committed when settling.
        reading.target = 10;
        assert!(reading.set_motion(Motion::Skimming, ReadingMode::Manga));
        assert!(reading.set_motion(Motion::Still, ReadingMode::Manga));
        assert_eq!(reading.target(), 10);
    }

    #[test]
    fn playback_lifecycle_handles_restarts_refusals_and_page_faults() {
        use std::time::Duration;

        const MAX_EDGE: u32 = 100;
        const FITTING_EDGE: u32 = 20;
        const OVERSIZED_EDGE: u32 = 200;
        const FRAME_DELAY: Duration = Duration::from_millis(50);

        let playback_id =
            |page, generation| PlaybackId { page, generation };
        let frame_event = |target_id, edge| {
            BookEvent::Playback(PlaybackEvent::Frame {
                id: target_id,
                delay: FRAME_DELAY,
                image: wire_image(edge, edge, Format::Gif),
            })
        };

        let mode = ReadingMode::Normal;
        let (mut reading, _) = reading_with_pages(5, None, mode);
        for page in [0, 1, 2] {
            let event = ready_event(
                page,
                Role::Fit,
                FITTING_EDGE,
                FITTING_EDGE,
                Format::Gif,
            );
            assert_eq!(reading.apply_event(event), None);
        }
        for upload in reading.take_pending_uploads() {
            reading.record_uploaded(upload.page, &upload.image);
        }

        let fit_playback = playback_id(0, 0);
        let full_playback = playback_id(0, 1);
        assert_eq!(
            reading.sync_playback(mode, Spread::lone(0), |_| {
                roles_with(Role::Fit)
            }),
            vec![PlaybackCommand::Start {
                id: fit_playback,
                role: Role::Fit
            }]
        );
        assert_eq!(
            reading.sync_playback(mode, Spread::lone(0), |_| {
                roles_with(Role::Full)
            }),
            vec![
                PlaybackCommand::Stop { id: fit_playback },
                PlaybackCommand::Start {
                    id: full_playback,
                    role: Role::Full
                },
            ]
        );

        let now = Instant::now();
        assert_eq!(
            reading.tick_playback(now, |_, _, image| {
                check_texture_edge(image, MAX_EDGE)
            }),
            PlaybackTick {
                next_deadline: None,
                commands: vec![PlaybackCommand::Next {
                    id: full_playback
                }],
                updated: false,
            }
        );

        assert_eq!(
            reading.apply_event(frame_event(fit_playback, FITTING_EDGE)),
            None
        );
        assert_eq!(
            reading.apply_event(frame_event(full_playback, FITTING_EDGE)),
            None
        );
        assert_eq!(
            reading.tick_playback(now, |_, _, image| {
                check_texture_edge(image, MAX_EDGE)
            }),
            PlaybackTick {
                next_deadline: Some(now + FRAME_DELAY),
                commands: vec![PlaybackCommand::Next {
                    id: full_playback
                }],
                updated: true,
            }
        );
        assert_eq!(
            reading.apply_event(BookEvent::Playback(
                PlaybackEvent::Failed { id: fit_playback }
            )),
            None
        );
        assert_eq!(
            reading.apply_event(BookEvent::Playback(
                PlaybackEvent::Failed { id: full_playback }
            )),
            Some(PlaybackCommand::Stop { id: full_playback })
        );
        assert_eq!(
            reading.sync_playback(mode, Spread::lone(0), |_| {
                roles_with(Role::Full)
            }),
            []
        );

        let faulted_playback = playback_id(1, 2);
        assert_eq!(
            reading.sync_playback(mode, Spread::lone(1), |_| {
                roles_with(Role::Fit)
            }),
            vec![PlaybackCommand::Start {
                id: faulted_playback,
                role: Role::Fit
            }]
        );
        assert_eq!(
            reading.apply_event(BookEvent::Failed {
                page: 1,
                fault: String::from("corrupt"),
            }),
            Some(PlaybackCommand::Stop {
                id: faulted_playback
            })
        );
        assert!(
            reading.pages.get(&1).and_then(|state| state.info).is_some()
        );
        assert_eq!(
            reading.sync_playback(mode, Spread::lone(1), |_| {
                roles_with(Role::Fit)
            }),
            []
        );

        let oversized_playback = playback_id(2, 3);
        assert_eq!(
            reading.sync_playback(mode, Spread::lone(2), |_| {
                roles_with(Role::Fit)
            }),
            vec![PlaybackCommand::Start {
                id: oversized_playback,
                role: Role::Fit
            }]
        );
        assert_eq!(
            reading.tick_playback(now, |_, _, image| {
                check_texture_edge(image, MAX_EDGE)
            }),
            PlaybackTick {
                next_deadline: None,
                commands: vec![PlaybackCommand::Next {
                    id: oversized_playback
                }],
                updated: false,
            }
        );
        assert_eq!(
            reading.apply_event(frame_event(
                oversized_playback,
                OVERSIZED_EDGE,
            )),
            None
        );
        assert_eq!(
            reading.tick_playback(now, |_, _, image| {
                check_texture_edge(image, MAX_EDGE)
            }),
            PlaybackTick {
                next_deadline: None,
                commands: vec![PlaybackCommand::Stop {
                    id: oversized_playback
                }],
                updated: false,
            }
        );
        assert_eq!(
            reading.sync_playback(mode, Spread::lone(2), |_| {
                roles_with(Role::Fit)
            }),
            []
        );
    }

    #[test]
    fn still_shown_spread_preuploads_next_spread_without_wrapping() {
        let mode = ReadingMode::Normal;
        let (mut reading, _) = reading_with_pages(6, None, mode);
        let roles = |ready: usize| {
            move |page: usize| {
                if page == ready {
                    roles_with(Role::Fit)
                } else {
                    Roles::default()
                }
            }
        };

        reading.shown = Some((mode, Spread::lone(0)));
        assert!(reading.turn(Direction::Forward, mode, Instant::now()));
        assert_eq!(
            reading.update_requests(
                mode,
                Spread::lone(0),
                None,
                test_limits(),
                roles(0),
            ),
            Some(vec![Request {
                page: 1,
                role: Role::Fit
            }])
        );

        reading.shown = Some((mode, Spread::lone(1)));
        assert_eq!(
            reading.update_requests(
                mode,
                Spread::lone(1),
                None,
                test_limits(),
                roles(1),
            ),
            Some(vec![Request {
                page: 2,
                role: Role::Fit
            }])
        );

        reading.target = 5;
        reading.shown = Some((mode, Spread::lone(5)));
        assert_eq!(
            reading.update_requests(
                mode,
                Spread::lone(5),
                None,
                test_limits(),
                roles(5),
            ),
            Some(Vec::new())
        );
    }
}
