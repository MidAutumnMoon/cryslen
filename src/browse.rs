//! Browsing vocabulary shared by parent, child, and renderer. Pure data:
//! no clocks, IO, or GL.

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum Role {
    Thumb,
    Fit,
    Full,
}

impl Role {
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

/// The roles present for one page.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Roles {
    thumb: bool,
    fit: bool,
    full: bool,
}

impl Roles {
    pub fn set(&mut self, role: Role) {
        match role {
            Role::Thumb => self.thumb = true,
            Role::Fit => self.fit = true,
            Role::Full => self.full = true,
        }
    }

    pub fn clear(&mut self, role: Role) {
        match role {
            Role::Thumb => self.thumb = false,
            Role::Fit => self.fit = false,
            Role::Full => self.full = false,
        }
    }

    #[must_use]
    pub fn contains(self, role: Role) -> bool {
        match role {
            Role::Thumb => self.thumb,
            Role::Fit => self.fit,
            Role::Full => self.full,
        }
    }

    #[must_use]
    pub fn any(self) -> bool {
        self.thumb || self.fit || self.full
    }

    /// A fit or full image can fill a still seat.
    #[must_use]
    pub fn has_settled_image(self) -> bool {
        self.fit || self.full
    }

    /// The largest role present: full, then fit, then thumb.
    #[must_use]
    pub fn best(self) -> Option<Role> {
        if self.full {
            Some(Role::Full)
        } else if self.fit {
            Some(Role::Fit)
        } else if self.thumb {
            Some(Role::Thumb)
        } else {
            None
        }
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