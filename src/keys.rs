//! The baked-in key map: physical keys to `Action`s.

use crate::geometry::FitMode;
use crate::browse::Direction;
use crate::browse::ReadingMode;
use egui_winit::winit;
use winit::keyboard::KeyCode;
use winit::keyboard::ModifiersState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Quit,
    /// Leave fullscreen when fullscreen, quit otherwise.
    Escape,
    SelectMode(ReadingMode),
    ToggleFullscreen,
    ResetView,
    SelectFit(FitMode),
    /// The book's first spread.
    First,
    /// The book's last spread.
    Last,
    /// The next sibling book in the parent directory.
    NextBook,
    /// The previous sibling book in the parent directory.
    PrevBook,
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
    /// Bindings answer only to bare keys, `Shift+Space` and `Shift+B`
    /// excepted; every other modifier combo (anything the desktop owns)
    /// stays with the desktop.
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "winit's unmapped KeyCode variants map to no action"
    )]
    pub fn from_key(
        code: KeyCode,
        modifiers: ModifiersState,
    ) -> Option<Self> {
        let bare_shift = modifiers.shift_key()
            && !modifiers.control_key()
            && !modifiers.alt_key()
            && !modifiers.super_key();
        if bare_shift {
            return match code {
                KeyCode::Space => Some(Self::Back),
                KeyCode::KeyB => Some(Self::PrevBook),
                _ => None,
            };
        }
        if !modifiers.is_empty() {
            return None;
        }
        #[expect(
            clippy::wildcard_enum_match_arm,
            reason = "winit's unmapped KeyCode variants map to no action"
        )]
        match code {
            KeyCode::KeyQ => Some(Self::Quit),
            KeyCode::Escape => Some(Self::Escape),
            KeyCode::KeyN => Some(Self::SelectMode(ReadingMode::Normal)),
            KeyCode::KeyM => Some(Self::SelectMode(ReadingMode::Manga)),
            KeyCode::KeyC => Some(Self::SelectMode(ReadingMode::Comic)),
            KeyCode::KeyF => Some(Self::ToggleFullscreen),
            KeyCode::KeyR => Some(Self::ResetView),
            KeyCode::KeyW => Some(Self::SelectFit(FitMode::Width)),
            KeyCode::KeyH => Some(Self::SelectFit(FitMode::Height)),
            KeyCode::Digit1 => Some(Self::SelectFit(FitMode::Actual)),
            KeyCode::KeyB => Some(Self::NextBook),
            KeyCode::Home => Some(Self::First),
            KeyCode::End => Some(Self::Last),
            KeyCode::ArrowRight => Some(Self::Right),
            KeyCode::ArrowLeft => Some(Self::Left),
            KeyCode::Space | KeyCode::PageDown => Some(Self::Forward),
            KeyCode::PageUp => Some(Self::Back),
            _ => None,
        }
    }

    /// Whether a held key keeps firing this action.
    pub fn repeats(self) -> bool {
        matches!(
            self,
            Self::Right | Self::Left | Self::Forward | Self::Back
        )
    }

    /// The reading direction this action advances in `mode`, if it turns
    /// pages. Manga flips the arrows (Left advances). The reading-order
    /// keys never flip.
    pub fn direction(self, mode: ReadingMode) -> Option<Direction> {
        match self {
            Self::Forward => Some(Direction::Forward),
            Self::Right => Some(if mode.rtl() {
                Direction::Backward
            } else {
                Direction::Forward
            }),
            Self::Left => Some(if mode.rtl() {
                Direction::Forward
            } else {
                Direction::Backward
            }),
            Self::Back => Some(Direction::Backward),
            Self::Quit
            | Self::Escape
            | Self::SelectMode(_)
            | Self::ToggleFullscreen
            | Self::ResetView
            | Self::SelectFit(_)
            | Self::First
            | Self::Last
            | Self::NextBook
            | Self::PrevBook => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_map() {
        let bindings: &[(KeyCode, Option<Action>)] = &[
            (KeyCode::Escape, Some(Action::Escape)),
            (KeyCode::KeyQ, Some(Action::Quit)),
            (KeyCode::KeyN, Some(Action::SelectMode(ReadingMode::Normal))),
            (KeyCode::KeyM, Some(Action::SelectMode(ReadingMode::Manga))),
            (KeyCode::KeyC, Some(Action::SelectMode(ReadingMode::Comic))),
            (KeyCode::KeyF, Some(Action::ToggleFullscreen)),
            (KeyCode::KeyR, Some(Action::ResetView)),
            (KeyCode::KeyW, Some(Action::SelectFit(FitMode::Width))),
            (KeyCode::KeyH, Some(Action::SelectFit(FitMode::Height))),
            (KeyCode::Digit1, Some(Action::SelectFit(FitMode::Actual))),
            (KeyCode::KeyB, Some(Action::NextBook)),
            (KeyCode::ShiftLeft, None), // the Shift+Space pairing lives on Space
            (KeyCode::Home, Some(Action::First)),
            (KeyCode::End, Some(Action::Last)),
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
            assert_eq!(
                Action::from_key(*code, ModifiersState::empty()),
                *action,
                "{code:?}"
            );
        }
    }

    #[test]
    fn modifier_interactions() {
        let bindings: &[(KeyCode, ModifiersState, Option<Action>)] = &[
            (KeyCode::Space, ModifiersState::SHIFT, Some(Action::Back)),
            (KeyCode::KeyB, ModifiersState::SHIFT, Some(Action::PrevBook)),
            (KeyCode::KeyQ, ModifiersState::CONTROL, None),
            (KeyCode::KeyQ, ModifiersState::SUPER, None),
            (KeyCode::KeyC, ModifiersState::CONTROL, None),
            (KeyCode::Space, ModifiersState::CONTROL, None),
            (KeyCode::ArrowRight, ModifiersState::ALT, None),
            (
                KeyCode::Space,
                ModifiersState::SHIFT | ModifiersState::CONTROL,
                None,
            ),
            (KeyCode::PageDown, ModifiersState::SHIFT, None),
            (KeyCode::PageUp, ModifiersState::SHIFT, None),
        ];
        for (code, modifiers, action) in bindings {
            assert_eq!(
                Action::from_key(*code, *modifiers),
                *action,
                "{code:?} {modifiers:?}"
            );
        }
    }

    #[test]
    fn action_semantics() {
        assert!(Action::Right.repeats());
        assert!(Action::Left.repeats());
        assert!(Action::Forward.repeats());
        assert!(Action::Back.repeats());
        let one_shot = [
            Action::Quit,
            Action::Escape,
            Action::SelectMode(ReadingMode::Normal),
            Action::SelectMode(ReadingMode::Manga),
            Action::SelectMode(ReadingMode::Comic),
            Action::ToggleFullscreen,
            Action::ResetView,
            Action::SelectFit(FitMode::Page),
            Action::SelectFit(FitMode::Width),
            Action::SelectFit(FitMode::Height),
            Action::SelectFit(FitMode::Actual),
            Action::First,
            Action::Last,
            Action::NextBook,
            Action::PrevBook,
        ];
        for action in one_shot {
            assert!(!action.repeats(), "{action:?} is one-shot");
        }
        let forwards: &[(Action, ReadingMode, Direction)] = &[
            (Action::Right, ReadingMode::Normal, Direction::Forward),
            (Action::Left, ReadingMode::Normal, Direction::Backward),
            (Action::Forward, ReadingMode::Normal, Direction::Forward),
            (Action::Back, ReadingMode::Normal, Direction::Backward),
            (Action::Right, ReadingMode::Manga, Direction::Backward),
            (Action::Left, ReadingMode::Manga, Direction::Forward),
            (Action::Forward, ReadingMode::Manga, Direction::Forward),
            (Action::Back, ReadingMode::Manga, Direction::Backward),
            (Action::Right, ReadingMode::Comic, Direction::Forward),
            (Action::Left, ReadingMode::Comic, Direction::Backward),
            (Action::Forward, ReadingMode::Comic, Direction::Forward),
            (Action::Back, ReadingMode::Comic, Direction::Backward),
        ];
        for &(action, mode, want) in forwards {
            assert_eq!(
                action.direction(mode),
                Some(want),
                "{action:?} in {mode:?}"
            );
        }
    }
}
