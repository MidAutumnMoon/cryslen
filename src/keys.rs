//! The baked-in key map: physical keys to `Action`s.

use crate::view::ViewMode;
use winit::keyboard::KeyCode;

/// Input actions, derived from physical keys (layout-independent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    Quit,
    SelectMode(ViewMode),
    ToggleFullscreen,
    ResetView,
    /// Page one spread toward the screen right.
    Right,
    /// Page one spread toward the screen left.
    Left,
    /// Page one spread forward in reading order.
    Forward,
    /// Page one spread backward in reading order.
    Back,
}

impl Action {
    /// Key map, baked in. Arrows page by screen direction. Space and
    /// the Page keys page by reading order.
    pub(crate) fn from_key(code: KeyCode) -> Option<Self> {
        use Action::*;
        match code {
            KeyCode::Escape | KeyCode::KeyQ => Some(Quit),
            KeyCode::KeyN => Some(SelectMode(ViewMode::Normal)),
            KeyCode::KeyM => Some(SelectMode(ViewMode::Manga)),
            KeyCode::KeyC => Some(SelectMode(ViewMode::Comic)),
            KeyCode::KeyF => Some(ToggleFullscreen),
            KeyCode::KeyR => Some(ResetView),
            KeyCode::ArrowRight => Some(Right),
            KeyCode::ArrowLeft => Some(Left),
            KeyCode::Space | KeyCode::PageDown => Some(Forward),
            KeyCode::PageUp => Some(Back),
            _ => None,
        }
    }

    /// Whether a held key keeps firing this action.
    pub(crate) fn repeats(self) -> bool {
        matches!(self, Self::Right | Self::Left | Self::Forward | Self::Back)
    }

    /// Whether this action advances reading order in `mode`. The
    /// arrows follow the reading direction. The reading-order keys do
    /// not.
    pub(crate) fn is_forward(self, mode: ViewMode) -> bool {
        match self {
            Self::Forward => true,
            Self::Back => false,
            Self::Right => !mode.rtl(),
            Self::Left => mode.rtl(),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_map() {
        assert_eq!(Action::from_key(KeyCode::Escape), Some(Action::Quit));
        assert_eq!(Action::from_key(KeyCode::KeyQ), Some(Action::Quit));
        assert_eq!(
            Action::from_key(KeyCode::KeyN),
            Some(Action::SelectMode(ViewMode::Normal))
        );
        assert_eq!(
            Action::from_key(KeyCode::KeyM),
            Some(Action::SelectMode(ViewMode::Manga))
        );
        assert_eq!(
            Action::from_key(KeyCode::KeyC),
            Some(Action::SelectMode(ViewMode::Comic))
        );
        assert_eq!(Action::from_key(KeyCode::KeyF), Some(Action::ToggleFullscreen));
        assert_eq!(Action::from_key(KeyCode::KeyR), Some(Action::ResetView));
        assert_eq!(Action::from_key(KeyCode::ArrowRight), Some(Action::Right));
        assert_eq!(Action::from_key(KeyCode::ArrowLeft), Some(Action::Left));
        assert_eq!(Action::from_key(KeyCode::Space), Some(Action::Forward));
        assert_eq!(Action::from_key(KeyCode::PageDown), Some(Action::Forward));
        assert_eq!(Action::from_key(KeyCode::PageUp), Some(Action::Back));
        assert_eq!(Action::from_key(KeyCode::KeyA), None);
        assert_eq!(Action::from_key(KeyCode::KeyD), None);
        assert_eq!(Action::from_key(KeyCode::ArrowUp), None);
    }

    #[test]
    fn only_paging_repeats_on_hold() {
        assert!(Action::Right.repeats());
        assert!(Action::Left.repeats());
        assert!(Action::Forward.repeats());
        assert!(Action::Back.repeats());
        let one_shot = [
            Action::Quit,
            Action::SelectMode(ViewMode::Normal),
            Action::SelectMode(ViewMode::Manga),
            Action::SelectMode(ViewMode::Comic),
            Action::ToggleFullscreen,
            Action::ResetView,
        ];
        for act in one_shot {
            assert!(!act.repeats());
        }
    }

    #[test]
    fn forward_follows_reading_direction() {
        // Left to right, the rightward arrow advances.
        assert!(Action::Right.is_forward(ViewMode::Normal));
        assert!(!Action::Left.is_forward(ViewMode::Normal));
        // Manga flips the arrows but never the reading-order keys.
        assert!(!Action::Right.is_forward(ViewMode::Manga));
        assert!(Action::Left.is_forward(ViewMode::Manga));
        assert!(Action::Forward.is_forward(ViewMode::Manga));
        assert!(!Action::Back.is_forward(ViewMode::Manga));
    }

    #[test]
    fn page_turn_respects_rtl() {
        use crate::view;

        // advance() composes with is_forward. In manga, rightward from
        // page 1 moves to the cover while forward moves deeper in.
        assert_eq!(view::advance(1, Action::Right.is_forward(ViewMode::Manga), ViewMode::Manga, 10), 0);
        assert_eq!(view::advance(1, Action::Forward.is_forward(ViewMode::Manga), ViewMode::Manga, 10), 3);
        assert_eq!(view::advance(1, Action::Right.is_forward(ViewMode::Normal), ViewMode::Normal, 10), 2);
    }

    #[test]
    fn mode_keys_match_their_shortcut_letters() {
        let bindings = [
            (KeyCode::KeyN, "n", ViewMode::Normal),
            (KeyCode::KeyM, "m", ViewMode::Manga),
            (KeyCode::KeyC, "c", ViewMode::Comic),
        ];
        for (code, letter, mode) in bindings {
            assert_eq!(Action::from_key(code), Some(Action::SelectMode(mode)));
            assert_eq!(mode.shortcut(), letter);
        }
    }
}
