//! The baked-in key map: physical keys to `Action`s.

use crate::view::ViewMode;
use egui_winit::winit;
use winit::keyboard::{KeyCode, ModifiersState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is the crate's visibility style for crate-wide items, even inside private modules"
)]
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
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "winit's unmapped KeyCode variants map to no action"
    )]
    pub(crate) fn from_key(code: KeyCode, modifiers: ModifiersState) -> Option<Self> {
        match code {
            KeyCode::Escape | KeyCode::KeyQ => Some(Self::Quit),
            KeyCode::KeyN => Some(Self::SelectMode(ViewMode::Normal)),
            KeyCode::KeyM => Some(Self::SelectMode(ViewMode::Manga)),
            KeyCode::KeyC => Some(Self::SelectMode(ViewMode::Comic)),
            KeyCode::KeyF => Some(Self::ToggleFullscreen),
            KeyCode::KeyR => Some(Self::ResetView),
            KeyCode::ArrowRight => Some(Self::Right),
            KeyCode::ArrowLeft => Some(Self::Left),
            KeyCode::Space if modifiers.shift_key() => Some(Self::Back),
            KeyCode::Space | KeyCode::PageDown => Some(Self::Forward),
            KeyCode::PageUp => Some(Self::Back),
            _ => None,
        }
    }

    /// Whether a held key keeps firing this action.
    pub(crate) fn repeats(self) -> bool {
        matches!(self, Self::Right | Self::Left | Self::Forward | Self::Back)
    }

    /// Whether this action advances reading order in `mode`. Manga
    /// flips the arrows (Left advances). The reading-order keys never
    /// flip.
    pub(crate) fn is_forward(self, mode: ViewMode) -> bool {
        match self {
            Self::Forward => true,
            Self::Right => !mode.rtl(),
            Self::Left => mode.rtl(),
            Self::Back
            | Self::Quit
            | Self::SelectMode(_)
            | Self::ToggleFullscreen
            | Self::ResetView => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The baked key map: every bound physical key maps to its action, nothing else does.
    #[test]
    fn key_map() {
        let bindings: &[(KeyCode, Option<Action>)] = &[
            (KeyCode::Escape, Some(Action::Quit)),
            (KeyCode::KeyQ, Some(Action::Quit)),
            (KeyCode::KeyN, Some(Action::SelectMode(ViewMode::Normal))),
            (KeyCode::KeyM, Some(Action::SelectMode(ViewMode::Manga))),
            (KeyCode::KeyC, Some(Action::SelectMode(ViewMode::Comic))),
            (KeyCode::KeyF, Some(Action::ToggleFullscreen)),
            (KeyCode::KeyR, Some(Action::ResetView)),
            (KeyCode::ArrowRight, Some(Action::Right)),
            (KeyCode::ArrowLeft, Some(Action::Left)),
            (KeyCode::Space, Some(Action::Forward)),
            (KeyCode::PageDown, Some(Action::Forward)),
            (KeyCode::PageUp, Some(Action::Back)),
            (KeyCode::KeyA, None),
            (KeyCode::KeyD, None),
            (KeyCode::ArrowUp, None),
        ];
        for (code, action) in bindings {
            assert_eq!(Action::from_key(*code, ModifiersState::empty()), *action, "{code:?}");
        }
    }

    /// Any modifier state holding Shift reverses Space. Every other
    /// binding ignores modifiers.
    #[test]
    fn modifier_interactions() {
        let bindings: &[(KeyCode, ModifiersState, Option<Action>)] = &[
            (KeyCode::Space, ModifiersState::SHIFT, Some(Action::Back)),
            // Shift wins even combined with other modifiers.
            (
                KeyCode::Space,
                ModifiersState::SHIFT | ModifiersState::CONTROL,
                Some(Action::Back),
            ),
            (KeyCode::Space, ModifiersState::CONTROL, Some(Action::Forward)),
            (KeyCode::PageDown, ModifiersState::SHIFT, Some(Action::Forward)),
            (KeyCode::PageUp, ModifiersState::SHIFT, Some(Action::Back)),
            (KeyCode::KeyQ, ModifiersState::CONTROL, Some(Action::Quit)),
        ];
        for (code, modifiers, action) in bindings {
            assert_eq!(Action::from_key(*code, *modifiers), *action, "{code:?} {modifiers:?}");
        }
    }

    /// Only the paging actions repeat on hold; forward follows the reading direction.
    #[test]
    fn action_semantics() {
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
            assert!(!act.repeats(), "{act:?} is one-shot");
        }
        let forwards: &[(Action, ViewMode, bool)] = &[
            // Left to right, the rightward arrow advances. The
            // reading-order keys never flip.
            (Action::Right, ViewMode::Normal, true),
            (Action::Left, ViewMode::Normal, false),
            (Action::Forward, ViewMode::Normal, true),
            (Action::Back, ViewMode::Normal, false),
            // Manga flips the arrows but never the reading-order keys.
            (Action::Right, ViewMode::Manga, false),
            (Action::Left, ViewMode::Manga, true),
            (Action::Forward, ViewMode::Manga, true),
            (Action::Back, ViewMode::Manga, false),
            // Comic reads like Normal.
            (Action::Right, ViewMode::Comic, true),
            (Action::Left, ViewMode::Comic, false),
            (Action::Forward, ViewMode::Comic, true),
            (Action::Back, ViewMode::Comic, false),
        ];
        for &(act, mode, want) in forwards {
            assert_eq!(act.is_forward(mode), want, "{act:?} in {mode:?}");
        }
    }
}
