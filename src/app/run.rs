//! Windowed application runner, CLI entry, event loop, and book switching.

use std::collections::BTreeSet;
use std::collections::HashSet;
use std::ffi::OsString;
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::TryRecvError;
use egui::Rect;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::ElementState;
use winit::event::KeyEvent;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::EventLoopProxy;
use winit::keyboard::KeyCode;
use winit::keyboard::ModifiersState;
use winit::keyboard::PhysicalKey;
use winit::window::WindowId;

use crate::app::chrome;
use crate::app::chrome::StatusLine;
use crate::app::chrome::Timers;
use crate::app::input;
use crate::app::input::Action;
use crate::app::input::CommandMode;
use crate::app::input::JumpOutcome;
use crate::app::input::PointerState;
use crate::app::layout;
use crate::app::layout::FitMode;
use crate::app::layout::ReadingMode;
use crate::app::layout::Viewport;
use crate::app::reading::Reading;
use crate::book::catalog;
use crate::book::catalog::PageName;
use crate::book::loader::DecodePool;
use crate::book::loader::PageEvent;
use crate::book::playback::GifPlayer;
use crate::book::playback::GifTick;
use crate::book::source::ListedBook;
use crate::book::source::Source;
use crate::book::thumbs::ThumbHandle;
use crate::book::thumbs::ThumbStore;
use crate::image::pixels::ImageLimits;
use crate::image::resample;
use crate::render::cache::InsertOutcome;
use crate::render::textures::PageRenderer;
use crate::render::window::Window;

#[derive(Debug, Clone, Copy)]
enum Wake {
    Notify,
}

fn create_wake_notifier(
    proxy: &EventLoopProxy<Wake>,
) -> Arc<dyn Fn() + Send + Sync> {
    let proxy = proxy.clone();
    Arc::new(move || {
        if proxy.send_event(Wake::Notify).is_err() {
            tracing::trace!("event loop closed");
        }
    })
}

type OpenResult = eros::Result<(ListedBook, Option<ThumbHandle>)>;

struct PendingOpen {
    path: PathBuf,
    land_on: Option<String>,
    result_receiver: Receiver<OpenResult>,
}

struct PagePlayer {
    page: usize,
    player: GifPlayer,
}

struct BookStep {
    needs_redraw: bool,
    is_waiting: bool,
    next_deadline: Option<Instant>,
}

struct LoadedBook {
    source: Arc<Source>,
    thumb_handle: Option<Arc<ThumbHandle>>,
    reading: Reading,
    loader: DecodePool,
    players: [Option<PagePlayer>; 2],
    refused_playback: BTreeSet<usize>,
}

impl LoadedBook {
    fn open(
        path: PathBuf,
        land_on: Option<&str>,
        listed: ListedBook,
        book_thumbs: Option<ThumbHandle>,
        limits: ImageLimits,
        (mode, fit_mode): (ReadingMode, FitMode),
        wake: &Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        let is_archive = listed.is_archive();
        let names = listed.names().clone();
        let source = Arc::new(listed.start());
        let thumb_handle = book_thumbs.map(Arc::new);
        let (reading, initial_plan) = Reading::open(
            path,
            is_archive,
            names,
            land_on,
            mode,
            fit_mode,
            thumb_handle.is_some(),
        );
        let loader = DecodePool::start(
            Arc::clone(&source),
            thumb_handle.clone(),
            limits,
            initial_plan,
            Arc::clone(wake),
        );
        tracing::debug!(
            path = %reading.path().display(),
            count = reading.page_count().get(),
            "book adopted"
        );
        Self {
            source,
            thumb_handle,
            reading,
            loader,
            players: [None, None],
            refused_playback: BTreeSet::new(),
        }
    }

    fn reconcile(
        &mut self,
        pages: &mut PageRenderer,
        viewport: Viewport,
        limits: ImageLimits,
        is_skimming: bool,
        wake: &Arc<dyn Fn() + Send + Sync>,
        now: Instant,
    ) -> BookStep {
        let mut needs_redraw = false;
        let mut updated_pages = Vec::new();
        for event in self.loader.drain() {
            match event {
                PageEvent::Ready { page, role, image } => {
                    const STALE_THUMB_THREADS: usize = 1;

                    updated_pages.push(page);
                    match pages.insert_image(page, role, &image) {
                        Ok(outcome) => {
                            if outcome == InsertOutcome::EvictedStaleThumb
                            {
                                tracing::debug!(
                                    page,
                                    ?role,
                                    "evicted stale cached thumb"
                                );
                                if let Some(thumb_handle) =
                                    &self.thumb_handle
                                {
                                    match resample::derive_thumb(
                                        &image,
                                        STALE_THUMB_THREADS,
                                    ) {
                                        Ok(encoded_thumb) => {
                                            thumb_handle.record_overwrite(
                                                page,
                                                encoded_thumb,
                                            );
                                        }
                                        Err(error) => {
                                            tracing::debug!(
                                                page,
                                                %error,
                                                "stale thumb derive failed"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        Err(fault) => {
                            tracing::warn!(
                                page,
                                ?role,
                                %fault,
                                "texture upload refused"
                            );
                            if !pages.cache().has_settled_image(page) {
                                self.reading
                                    .record_fault(page, fault.to_string());
                            }
                        }
                    }
                }
                PageEvent::Failed { page, fault } => {
                    self.reading.record_fault(page, fault);
                    self.refused_playback.insert(page);
                    needs_redraw = true;
                }
            }
        }

        if let Some(thumb_handle) = &self.thumb_handle {
            let pages_to_hydrate =
                self.reading.thumbs_to_hydrate(is_skimming, pages.cache());
            for page in pages_to_hydrate {
                if let Some(thumb) = thumb_handle.load(page) {
                    pages.insert_thumb(page, &thumb);
                    updated_pages.push(page);
                }
            }
        }

        let view_step = self.reading.reconcile_view(
            pages.cache_mut(),
            viewport,
            limits,
            is_skimming,
            self.thumb_handle.is_some(),
            now,
        );
        if view_step.needs_redraw
            || updated_pages
                .iter()
                .any(|&page| self.reading.on_screen().contains(page))
        {
            needs_redraw = true;
        }
        if let Some(plan) = view_step.next_plan {
            self.loader.set_work(plan);
        }

        self.sync_playback(view_step.animated_pages, limits, wake);
        pages.cache_mut().retain_playback(&|page| {
            self.players
                .iter()
                .flatten()
                .any(|player| player.page == page)
        });
        let (playback_deadline, playback_updated) =
            self.tick_playback(now, pages);
        if playback_updated {
            needs_redraw = true;
        }
        let next_deadline = [playback_deadline, view_step.grace_deadline]
            .into_iter()
            .flatten()
            .min();
        BookStep {
            needs_redraw,
            is_waiting: view_step.is_waiting,
            next_deadline,
        }
    }

    fn sync_playback(
        &mut self,
        animated_pages: [Option<usize>; 2],
        limits: ImageLimits,
        wake: &Arc<dyn Fn() + Send + Sync>,
    ) {
        let mut next_players: [Option<PagePlayer>; 2] = [None, None];
        for (seat_index, (animated_page, player_slot)) in animated_pages
            .into_iter()
            .zip(&mut next_players)
            .enumerate()
        {
            let Some(page) = animated_page else {
                continue;
            };
            if self.refused_playback.contains(&page) {
                continue;
            }
            if let Some(existing) =
                self.players.iter_mut().find_map(|entry| {
                    entry.take_if(|player| player.page == page)
                })
            {
                *player_slot = Some(existing);
                continue;
            }
            let Ok(bytes) = self.source.bytes(page) else {
                self.refused_playback.insert(page);
                continue;
            };
            tracing::debug!(
                page,
                seat = seat_index,
                "playback player start"
            );
            *player_slot = Some(PagePlayer {
                page,
                player: GifPlayer::start(bytes, limits, Arc::clone(wake)),
            });
        }
        self.players = next_players;
    }

    fn tick_playback(
        &mut self,
        now: Instant,
        pages: &mut PageRenderer,
    ) -> (Option<Instant>, bool) {
        let mut next_deadline: Option<Instant> = None;
        let mut updated = false;
        for player_slot in &mut self.players {
            let Some(page_player) = player_slot else {
                continue;
            };
            let page = page_player.page;
            match page_player.player.tick(now) {
                GifTick::Show { image, next_due } => {
                    if let Err(error) = pages.update_playback(page, &image)
                    {
                        tracing::warn!(
                            page,
                            %error,
                            "playback frame upload failed"
                        );
                        self.refused_playback.insert(page);
                        *player_slot = None;
                        continue;
                    }
                    updated = true;
                    next_deadline =
                        Some(min_deadline(next_deadline, next_due));
                }
                GifTick::Wait(next_due) => {
                    next_deadline =
                        Some(min_deadline(next_deadline, next_due));
                }
                GifTick::Idle => {}
                GifTick::Failed => {
                    tracing::warn!(page, "playback failed");
                    self.refused_playback.insert(page);
                    *player_slot = None;
                }
            }
        }
        (next_deadline, updated)
    }

    fn stop(mut self) {
        self.players = [None, None];
        self.loader.stop();
        self.source.cancel();
    }

    fn stop_and_join(mut self) {
        self.players = [None, None];
        self.source.cancel();
        self.loader.stop_and_join();
    }
}

enum ActiveBook {
    Initial {
        name: PageName,
        mode: ReadingMode,
        fit_mode: FitMode,
    },
    Loaded(Box<LoadedBook>),
}

impl ActiveBook {
    fn mode(&self) -> ReadingMode {
        match self {
            Self::Initial { mode, .. } => *mode,
            Self::Loaded(book) => book.reading.mode(),
        }
    }

    fn fit_mode(&self) -> FitMode {
        match self {
            Self::Initial { fit_mode, .. } => *fit_mode,
            Self::Loaded(book) => book.reading.fit_mode(),
        }
    }

    fn select_mode(&mut self, next: ReadingMode) {
        match self {
            Self::Initial { mode, .. } => *mode = next,
            Self::Loaded(book) => {
                book.reading.select_mode(next);
            }
        }
    }

    fn select_fit(&mut self, next: FitMode) {
        match self {
            Self::Initial { fit_mode, .. } => *fit_mode = next,
            Self::Loaded(book) => book.reading.select_fit(next),
        }
    }

    fn reading(&self) -> Option<&Reading> {
        match self {
            Self::Initial { .. } => None,
            Self::Loaded(book) => Some(&book.reading),
        }
    }

    fn reading_mut(&mut self) -> Option<&mut Reading> {
        match self {
            Self::Initial { .. } => None,
            Self::Loaded(book) => Some(&mut book.reading),
        }
    }
}

struct App {
    initial_path: PathBuf,
    initial_land_on: Option<String>,
    wake: Arc<dyn Fn() + Send + Sync>,
    window: Option<Window>,
    thumb_store: Option<ThumbStore>,
    book: ActiveBook,
    pending_open: Option<PendingOpen>,
    open_fault: Option<String>,
    viewport: Viewport,
    timers: Timers,
    modifiers: ModifiersState,
    pointer: PointerState,
    command_mode: CommandMode,
    skimming_turn_keys: HashSet<KeyCode>,
    needs_redraw: bool,
    fatal_error: Option<eros::ErrorUnion>,
}

impl App {
    fn new(
        open_path: &Path,
        land_on: Option<&str>,
        proxy: &EventLoopProxy<Wake>,
    ) -> Self {
        Self {
            initial_path: open_path.to_path_buf(),
            initial_land_on: land_on.map(str::to_owned),
            wake: create_wake_notifier(proxy),
            window: None,
            thumb_store: None,
            book: ActiveBook::Initial {
                name: catalog::opening_name(open_path, land_on),
                mode: ReadingMode::default(),
                fit_mode: FitMode::default(),
            },
            pending_open: None,
            open_fault: None,
            viewport: Viewport::default(),
            timers: Timers::default(),
            modifiers: ModifiersState::empty(),
            pointer: PointerState::default(),
            command_mode: CommandMode::default(),
            skimming_turn_keys: HashSet::new(),
            needs_redraw: false,
            fatal_error: None,
        }
    }

    fn take_fatal_error(&mut self) -> Option<eros::ErrorUnion> {
        self.fatal_error.take()
    }

    fn fail(
        &mut self,
        error: eros::ErrorUnion,
        event_loop: &ActiveEventLoop,
    ) {
        self.fatal_error = Some(error);
        event_loop.exit();
    }

    fn start_open(&mut self, path: &Path, land_on: Option<&str>) {
        self.open_fault = None;
        let path_owned = path.to_path_buf();
        let path_for_thread = path_owned.clone();
        let thumb_store = self.thumb_store.clone();
        let wake = Arc::clone(&self.wake);
        let reclaimable_bytes = match &self.book {
            ActiveBook::Initial { .. } => 0,
            ActiveBook::Loaded(book) => {
                book.source.archive_resident_bytes()
            }
        };
        let (result_sender, result_receiver) =
            crossbeam_channel::bounded(1);
        let spawned = thread::Builder::new()
            .name(String::from("cy-open"))
            .spawn(move || {
                let result = Source::open_listed(
                    &path_for_thread,
                    reclaimable_bytes,
                )
                .map(|listed| {
                    let book_thumbs = thumb_store.and_then(|store| {
                        store.open_book(
                            &path_for_thread,
                            listed.is_archive(),
                            listed.names(),
                        )
                    });
                    (listed, book_thumbs)
                });
                if result_sender.send(result).is_err() {
                    tracing::debug!("book opener receiver dropped");
                }
                wake();
            });
        if let Err(error) = spawned {
            self.open_fault =
                Some(format!("could not spawn book opener: {error}"));
            self.needs_redraw = true;
            return;
        }
        self.pending_open = Some(PendingOpen {
            path: path_owned,
            land_on: land_on.map(str::to_owned),
            result_receiver,
        });
    }

    fn poll_open(&mut self) {
        let Some(pending) = &self.pending_open else {
            return;
        };
        let result = match pending.result_receiver.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => {
                Err(eros::error!("book opener terminated unexpectedly"))
            }
        };
        let Some(opened) = self.pending_open.take() else {
            return;
        };
        let Some(limits) = self.window.as_ref().map(Window::limits) else {
            return;
        };
        match result {
            Ok((listed, book_thumbs)) => {
                self.adopt_book(
                    opened.path,
                    opened.land_on.as_deref(),
                    listed,
                    book_thumbs,
                    limits,
                );
            }
            Err(fault) => {
                tracing::warn!(%fault, "book open failed");
                self.open_fault = Some(fault.to_string());
                self.needs_redraw = true;
            }
        }
    }

    fn adopt_book(
        &mut self,
        path: PathBuf,
        land_on: Option<&str>,
        listed: ListedBook,
        book_thumbs: Option<ThumbHandle>,
        limits: ImageLimits,
    ) {
        use std::mem;

        let mode = self.book.mode();
        let fit_mode = self.book.fit_mode();
        let previous = mem::replace(
            &mut self.book,
            ActiveBook::Initial {
                name: catalog::opening_name(&path, land_on),
                mode,
                fit_mode,
            },
        );
        if let ActiveBook::Loaded(previous_book) = previous {
            previous_book.stop();
        }
        if let Some(window) = &mut self.window {
            window.pages_mut().cache_mut().clear();
        }

        let loaded = LoadedBook::open(
            path,
            land_on,
            listed,
            book_thumbs,
            limits,
            (mode, fit_mode),
            &self.wake,
        );
        self.open_fault = None;
        self.skimming_turn_keys.clear();
        self.set_dragging(false);
        self.command_mode = CommandMode::Browse;
        self.book = ActiveBook::Loaded(Box::new(loaded));
        self.needs_redraw = true;
    }

    fn is_fullscreen(&self) -> bool {
        self.window
            .as_ref()
            .is_some_and(|window| window.winit().fullscreen().is_some())
    }

    fn set_dragging(&mut self, dragging: bool) {
        use winit::window::CursorIcon;

        if self.pointer.dragging == dragging {
            return;
        }
        self.pointer.dragging = dragging;
        if let Some(window) = self.window.as_ref() {
            let icon = if dragging {
                CursorIcon::Move
            } else {
                CursorIcon::Default
            };
            window.winit().set_cursor(icon);
        }
    }

    fn cursor_in_viewport(&self) -> Option<(f64, f64)> {
        let cursor = self.pointer.position?;
        let (width, height) = self.viewport.size();
        (width > 0
            && height > 0
            && cursor.0 >= 0.0
            && cursor.1 >= 0.0
            && cursor.0 < f64::from(width)
            && cursor.1 < f64::from(height))
        .then_some(cursor)
    }

    fn browsing_cursor(&self) -> Option<(f64, f64)> {
        matches!(self.command_mode, CommandMode::Browse)
            .then(|| self.cursor_in_viewport())
            .flatten()
    }

    fn zoom_at(&mut self, factor: f64, cursor: (f64, f64)) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let Some(reading) = self.book.reading_mut() else {
            return;
        };
        if reading.zoom(
            factor,
            cursor,
            self.viewport,
            window.pages().cache(),
        ) {
            self.needs_redraw = true;
        }
    }

    fn pan_by(&mut self, delta: (f64, f64)) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let Some(reading) = self.book.reading_mut() else {
            return;
        };
        if reading.pan(delta, self.viewport, window.pages().cache()) {
            self.needs_redraw = true;
        }
    }

    fn reconcile(&mut self, event_loop: &ActiveEventLoop) {
        use winit::event_loop::ControlFlow;

        if self.window.is_none() {
            return;
        }
        self.poll_open();
        let now = Instant::now();
        let Self {
            book,
            window,
            viewport,
            timers,
            skimming_turn_keys,
            needs_redraw,
            open_fault,
            pending_open,
            wake,
            ..
        } = self;
        let Some(window) = window.as_mut() else {
            return;
        };
        let limits = window.limits();
        let is_skimming = !skimming_turn_keys.is_empty();

        let (waiting, reconcile_deadline) = match book {
            ActiveBook::Initial { .. } => {
                (pending_open.is_some() && open_fault.is_none(), None)
            }
            ActiveBook::Loaded(loaded_book) => {
                let step = loaded_book.reconcile(
                    window.pages_mut(),
                    *viewport,
                    limits,
                    is_skimming,
                    wake,
                    now,
                );
                if step.needs_redraw {
                    *needs_redraw = true;
                }
                (step.is_waiting, step.next_deadline)
            }
        };

        let timer_tick = timers.tick(now, waiting);
        if timer_tick.changed {
            *needs_redraw = true;
        }
        let deadline = [reconcile_deadline, timer_tick.next_deadline]
            .into_iter()
            .flatten()
            .min();

        if *needs_redraw {
            window.winit().request_redraw();
        }
        event_loop.set_control_flow(
            deadline.map_or(ControlFlow::Wait, ControlFlow::WaitUntil),
        );
    }

    fn draw_frame(&mut self) -> eros::Result<()> {
        use std::time::Duration;

        const SLOW_FRAME_THRESHOLD: Duration = Duration::from_millis(50);

        struct FrameContext<'app> {
            book: &'app mut ActiveBook,
            viewport: &'app mut Viewport,
            open_fault: &'app Option<String>,
            command_mode: &'app CommandMode,
            timers: &'app Timers,
            show_status_bar: bool,
            viewport_changed: bool,
        }

        self.needs_redraw = false;
        let frame_started = Instant::now();
        let Self {
            book,
            window,
            viewport,
            timers,
            command_mode,
            open_fault,
            needs_redraw,
            ..
        } = self;
        let Some(window) = window.as_mut() else {
            return Ok(());
        };
        let show_status_bar = window.winit().fullscreen().is_none()
            || timers.is_toasting()
            || open_fault.is_some();
        let mut frame_context = FrameContext {
            book,
            viewport,
            open_fault,
            command_mode,
            timers,
            show_status_bar,
            viewport_changed: false,
        };

        let repaint_requested = window.present(
            &mut frame_context,
            |ui, pages, context| match context.book {
                ActiveBook::Initial { name, mode, .. } => {
                    let status = StatusLine {
                        reading: None,
                        mode: *mode,
                        show_mode_toast: context.timers.is_toasting(),
                        name: name.clone(),
                        show_loading: context.timers.is_waiting()
                            && context.open_fault.is_none(),
                        fault: context.open_fault.as_deref(),
                    };
                    let _page_area =
                        chrome::show(ui, &status, context.show_status_bar);
                    None
                }
                ActiveBook::Loaded(loaded_book) => {
                    let cache = pages.cache();
                    let seat_dims = loaded_book.reading.seat_dims(cache);
                    let spread_layout = layout::layout(seat_dims);
                    let spread_size = spread_layout.total();
                    let placement = loaded_book
                        .reading
                        .fit_mut()
                        .resolve(spread_size, *context.viewport);
                    let mut status = loaded_book
                        .reading
                        .status_line(placement.scale, cache);
                    status.show_mode_toast = context.timers.is_toasting();
                    status.show_loading =
                        status.show_loading && context.timers.is_waiting();
                    if let Some(fault) = context.open_fault.as_deref() {
                        status.fault = Some(fault);
                    }
                    let bar_visible =
                        context.show_status_bar || status.fault.is_some();
                    let page_count =
                        loaded_book.reading.page_count().get();
                    let page_area = chrome::show(ui, &status, bar_visible);
                    if let Some(digits) =
                        context.command_mode.jump_digits()
                    {
                        let valid =
                            input::parse_page_number(digits, page_count)
                                .is_some();
                        chrome::page_jump(
                            ui.ctx(),
                            page_area,
                            digits,
                            page_count,
                            valid,
                        );
                    }
                    Some((page_area, spread_layout, spread_size))
                }
            },
            |window_size, scale_factor, ui_output, context| {
                let Some((page_area, spread_layout, spread_size)) =
                    ui_output
                else {
                    return (context.viewport.size(), Vec::new());
                };
                let ActiveBook::Loaded(loaded_book) = &mut *context.book
                else {
                    return (context.viewport.size(), Vec::new());
                };
                let measured_viewport =
                    content_viewport(window_size, page_area, scale_factor);
                if measured_viewport != *context.viewport {
                    *context.viewport = measured_viewport;
                    context.viewport_changed = true;
                }
                let placement = loaded_book
                    .reading
                    .fit_mut()
                    .resolve(spread_size, *context.viewport);
                let quads = loaded_book.reading.shown().map_or_else(
                    Vec::new,
                    |spread| {
                        spread_layout.draw_quads(
                            spread.seats(loaded_book.reading.mode()),
                            &placement,
                        )
                    },
                );
                (context.viewport.size(), quads)
            },
        )?;
        if repaint_requested || frame_context.viewport_changed {
            *needs_redraw = true;
        }
        let elapsed = frame_started.elapsed();
        if elapsed >= SLOW_FRAME_THRESHOLD {
            tracing::info!(
                total = ?elapsed,
                shown = ?self.book.reading().and_then(Reading::shown),
                "slow frame"
            );
        }
        Ok(())
    }

    fn on_key(
        &mut self,
        key: &KeyEvent,
        is_synthetic: bool,
        event_loop: &ActiveEventLoop,
    ) {
        if is_synthetic && key.state == ElementState::Pressed {
            return;
        }
        match key.state {
            ElementState::Pressed => {
                if let CommandMode::Jump(digits) = &mut self.command_mode {
                    let Some(page_count) = self
                        .book
                        .reading()
                        .map(|reading| reading.page_count().get())
                    else {
                        return;
                    };
                    match input::handle_jump_key(
                        digits,
                        key,
                        self.modifiers,
                        page_count,
                    ) {
                        JumpOutcome::Ignored => {}
                        JumpOutcome::Edited => self.needs_redraw = true,
                        JumpOutcome::Canceled => {
                            self.command_mode = CommandMode::Browse;
                            self.needs_redraw = true;
                        }
                        JumpOutcome::Committed(target_page) => {
                            self.command_mode = CommandMode::Browse;
                            self.open_fault = None;
                            if let Some(reading) = self.book.reading_mut()
                            {
                                reading.jump_to_page(
                                    target_page,
                                    Instant::now(),
                                );
                            }
                            self.needs_redraw = true;
                        }
                    }
                    return;
                }
                if input::is_jump_trigger(
                    key.text.as_deref(),
                    key.repeat,
                    self.modifiers,
                ) {
                    if self.book.reading().is_some() {
                        self.skimming_turn_keys.clear();
                        self.set_dragging(false);
                        self.command_mode =
                            CommandMode::Jump(String::new());
                        self.needs_redraw = true;
                    }
                    return;
                }
                let PhysicalKey::Code(code) = key.physical_key else {
                    return;
                };
                let Some(action) = Action::from_key(code, self.modifiers)
                else {
                    return;
                };
                if key.repeat && !matches!(action, Action::Turn(_)) {
                    return;
                }
                if key.repeat && matches!(action, Action::Turn(_)) {
                    self.skimming_turn_keys.insert(code);
                }
                self.dispatch(action, key.repeat, event_loop);
            }
            ElementState::Released => {
                let PhysicalKey::Code(code) = key.physical_key else {
                    return;
                };
                self.skimming_turn_keys.remove(&code);
            }
        }
    }

    fn dispatch(
        &mut self,
        action: Action,
        is_repeat: bool,
        event_loop: &ActiveEventLoop,
    ) {
        match action {
            Action::Quit => event_loop.exit(),
            Action::Escape => {
                if self.is_fullscreen() {
                    if let Some(window) = self.window.as_ref() {
                        window.winit().set_fullscreen(None);
                    }
                    self.needs_redraw = true;
                } else {
                    event_loop.exit();
                }
            }
            Action::ToggleFullscreen => {
                use winit::window::Fullscreen;

                if let Some(window) = self.window.as_ref() {
                    let fullscreen_target = (!self.is_fullscreen())
                        .then_some(Fullscreen::Borderless(None));
                    window.winit().set_fullscreen(fullscreen_target);
                }
                self.needs_redraw = true;
            }
            Action::SelectFit(fit_mode) => {
                self.book.select_fit(fit_mode);
                self.needs_redraw = true;
            }
            Action::SelectMode(mode) => {
                self.open_fault = None;
                self.timers.toast(Instant::now());
                self.book.select_mode(mode);
                self.needs_redraw = true;
            }
            Action::First => {
                self.open_fault = None;
                if let Some(reading) = self.book.reading_mut() {
                    reading.jump_to_first(Instant::now());
                }
                self.needs_redraw = true;
            }
            Action::Last => {
                self.open_fault = None;
                if let Some(reading) = self.book.reading_mut() {
                    reading.jump_to_last(Instant::now());
                }
                self.needs_redraw = true;
            }
            Action::SwitchBook(direction) => {
                if self.pending_open.is_some() {
                    return;
                }
                let Some(path) = self
                    .book
                    .reading()
                    .and_then(|reading| reading.sibling(direction))
                else {
                    return;
                };
                tracing::debug!(path = %path.display(), "book switch");
                self.start_open(&path, None);
            }
            Action::Turn(turn) => {
                self.open_fault = None;
                let direction = turn.direction(self.book.mode());
                let Some(reading) = self.book.reading_mut() else {
                    return;
                };
                if reading.turn(direction, is_repeat, Instant::now()) {
                    self.needs_redraw = true;
                }
            }
        }
    }
}

impl ApplicationHandler<Wake> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = match Window::new(event_loop) {
            Ok(window) => window,
            Err(error) => {
                self.fail(error, event_loop);
                return;
            }
        };
        self.thumb_store = ThumbStore::open();
        self.window = Some(window);
        self.needs_redraw = true;
        let initial_path = self.initial_path.clone();
        let initial_land_on = self.initial_land_on.clone();
        self.start_open(&initial_path, initial_land_on.as_deref());
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: Wake) {
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.reconcile(event_loop);
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        use std::mem;

        self.pending_open = None;
        let (mode, fit_mode) = (self.book.mode(), self.book.fit_mode());
        let previous = mem::replace(
            &mut self.book,
            ActiveBook::Initial {
                name: catalog::opening_name(
                    &self.initial_path,
                    self.initial_land_on.as_deref(),
                ),
                mode,
                fit_mode,
            },
        );
        if let ActiveBook::Loaded(loaded_book) = previous {
            loaded_book.stop_and_join();
        }
        self.window = None;
    }

    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "winit's unhandled WindowEvent variants need no work"
    )]
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        use winit::event::MouseButton;

        match event {
            WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                if let Some(window) = &mut self.window {
                    let _size = window.sync_surface_size();
                }
                self.needs_redraw = true;
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                self.needs_redraw = true;
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
            }
            WindowEvent::KeyboardInput {
                event: key,
                is_synthetic,
                ..
            } => {
                self.on_key(&key, is_synthetic, event_loop);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current = (position.x, position.y);
                if let Some(previous) =
                    self.pointer.position.replace(current)
                    && self.pointer.dragging
                    && matches!(self.command_mode, CommandMode::Browse)
                {
                    self.pan_by((
                        current.0 - previous.0,
                        current.1 - previous.1,
                    ));
                }
            }
            WindowEvent::CursorLeft { .. } => {
                self.pointer.position = None;
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => match state {
                ElementState::Pressed => {
                    if self.book.reading().is_some()
                        && self.browsing_cursor().is_some()
                    {
                        self.set_dragging(true);
                    }
                }
                ElementState::Released => self.set_dragging(false),
            },
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(cursor) = self.browsing_cursor()
                    && let Some(window) = self.window.as_ref()
                {
                    let factor = input::wheel_zoom_factor(
                        delta,
                        window.winit().scale_factor(),
                    );
                    self.zoom_at(factor, cursor);
                }
            }
            WindowEvent::PinchGesture { delta, .. } => {
                if let Some(cursor) = self.browsing_cursor() {
                    self.zoom_at(delta.exp(), cursor);
                }
            }
            WindowEvent::Focused(false) => {
                self.skimming_turn_keys.clear();
                self.set_dragging(false);
                self.modifiers = ModifiersState::empty();
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.draw_frame() {
                    self.fail(error, event_loop);
                }
            }
            _ => {}
        }
    }
}

/// The page viewport from the chrome egui just laid out: the full-width
/// window area above the status bar, in physical px.
fn content_viewport(
    window: PhysicalSize<u32>,
    page_area: Rect,
    scale_factor: f64,
) -> Viewport {
    let bar_top_px = f64::from(page_area.max.y) * scale_factor;
    Viewport::above_bottom_bar((window.width, window.height), bar_top_px)
}

fn min_deadline(current: Option<Instant>, candidate: Instant) -> Instant {
    current.map_or(candidate, |deadline| deadline.min(candidate))
}

#[expect(
    clippy::use_debug,
    reason = "the fatal exit prints the full error report"
)]
pub fn main(args: &[OsString]) -> ExitCode {
    const USAGE_EXIT_CODE: u8 = 2;

    let Some(argument) = args.get(1) else {
        eprintln!("usage: cy <dir | archive | image-file>");
        return ExitCode::from(USAGE_EXIT_CODE);
    };
    let (open_path, land_on) =
        match resolve_source_target(Path::new(argument)) {
            Ok(target) => target,
            Err(error) => {
                eprintln!("cy: {error}");
                return ExitCode::FAILURE;
            }
        };

    match run(&open_path, land_on.as_deref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // `RUST_BACKTRACE` arms the backtrace in the Debug report.
            eprintln!("cy: {error:?}");
            ExitCode::FAILURE
        }
    }
}

fn resolve_source_target(
    path: &Path,
) -> eros::Result<(PathBuf, Option<String>)> {
    use crate::image::decode::codec;

    let normalized_path = catalog::normalize_path(path);
    let Some(name) = normalized_path.file_name() else {
        return Ok((normalized_path, None));
    };
    if !normalized_path.is_file() || catalog::is_archive(&normalized_path)
    {
        return Ok((normalized_path, None));
    }
    let name = name.to_string_lossy().into_owned();
    if codec::format_from_name(&name).is_none() {
        eros::bail!("unsupported source: {}", normalized_path.display());
    }
    let parent_directory = normalized_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok((parent_directory.to_path_buf(), Some(name)))
}

fn run(open_path: &Path, land_on: Option<&str>) -> eros::Result<()> {
    use winit::event_loop::EventLoop;

    let event_loop = EventLoop::<Wake>::with_user_event().build()?;
    let mut app = App::new(open_path, land_on, &event_loop.create_proxy());
    event_loop.run_app(&mut app)?;
    if let Some(error) = app.take_fatal_error() {
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;

    #[test]
    fn classify_argument_kinds() {
        use std::env;
        use std::fs;
        use std::os::unix::fs as unix_fs;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        fs::write(dir.path().join("page.png"), b"x")
            .expect("write page.png");
        fs::write(dir.path().join("book.cbz"), b"not really a zip")
            .expect("write book.cbz");
        fs::write(dir.path().join("notes.txt"), b"not a page")
            .expect("write notes.txt");
        fs::create_dir_all(dir.path().join("sub")).expect("mkdir sub");
        unix_fs::symlink(
            dir.path().join("page.png"),
            dir.path().join("link.png"),
        )
        .expect("symlink page");
        unix_fs::symlink(
            dir.path().join("sub"),
            dir.path().join("link_dir"),
        )
        .expect("symlink dir");
        unix_fs::symlink(
            dir.path().join("missing.png"),
            dir.path().join("broken.png"),
        )
        .expect("broken symlink");

        let cases = [
            (
                dir.path().join("page.png"),
                (dir.path().to_path_buf(), Some(String::from("page.png"))),
            ),
            (
                dir.path().join("link.png"),
                (dir.path().to_path_buf(), Some(String::from("link.png"))),
            ),
            (
                dir.path().join("book.cbz"),
                (dir.path().join("book.cbz"), None),
            ),
            (dir.path().to_path_buf(), (dir.path().to_path_buf(), None)),
            (
                dir.path().join("link_dir/."),
                (dir.path().join("link_dir"), None),
            ),
            (
                dir.path().join("broken.png"),
                (dir.path().join("broken.png"), None),
            ),
            (
                PathBuf::from("."),
                (env::current_dir().expect("current dir"), None),
            ),
        ];
        for (path, expected) in cases {
            assert_eq!(
                resolve_source_target(&path).ok(),
                Some(expected),
                "{}",
                path.display()
            );
        }
        let unsupported =
            resolve_source_target(&dir.path().join("notes.txt"))
                .expect_err("non-image file must refuse");
        assert!(unsupported.to_string().contains("unsupported source"));
    }

    #[test]
    fn content_viewport_subtracts_the_bar_in_physical_px() {
        use egui::Pos2;

        let page_area =
            Rect::from_min_max(Pos2::ZERO, Pos2::new(500.0, 360.25));
        let viewport =
            content_viewport(PhysicalSize::new(1000, 750), page_area, 2.0);
        assert_eq!(viewport.size(), (1000, 721));
    }
}
