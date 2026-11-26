//! Keyboard bindings, page-jump input stepping, pointer tracking, and wheel zoom.

use std::num::NonZeroUsize;

use winit::event::KeyEvent;
use winit::event::MouseScrollDelta;
use winit::keyboard::KeyCode;
use winit::keyboard::ModifiersState;
use winit::keyboard::PhysicalKey;

use crate::app::layout::FitMode;
use crate::app::layout::ReadingMode;
use crate::book::catalog::Direction;

#[derive(Default)]
pub enum CommandMode {
    #[default]
    Browse,
    Jump(String),
}

impl CommandMode {
    #[must_use]
    pub fn jump_digits(&self) -> Option<&str> {
        match self {
            Self::Browse => None,
            Self::Jump(digits) => Some(digits.as_str()),
        }
    }
}

#[derive(Default)]
pub struct PointerState {
    pub position: Option<(f64, f64)>,
    pub dragging: bool,
}

pub fn parse_page_number(
    digits: &str,
    page_count: usize,
) -> Option<NonZeroUsize> {
    digits
        .parse::<NonZeroUsize>()
        .ok()
        .filter(|page| page.get() <= page_count)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Turn {
    /// Page one spread toward the screen right.
    Right,
    /// Page one spread toward the screen left.
    Left,
    /// Page one spread in reading order.
    ReadingOrder(Direction),
}

impl Turn {
    /// Manga flips the arrows (Left advances); reading-order keys never flip.
    #[must_use]
    pub fn direction(self, mode: ReadingMode) -> Direction {
        match self {
            Self::ReadingOrder(direction) => direction,
            Self::Right => {
                if mode.rtl() {
                    Direction::Backward
                } else {
                    Direction::Forward
                }
            }
            Self::Left => {
                if mode.rtl() {
                    Direction::Forward
                } else {
                    Direction::Backward
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Quit,
    /// Leave fullscreen when fullscreen, quit otherwise.
    Escape,
    SelectMode(ReadingMode),
    ToggleFullscreen,
    SelectFit(FitMode),
    /// The book's first spread.
    First,
    /// The book's last spread.
    Last,
    /// The next or previous sibling book in the parent directory.
    SwitchBook(Direction),
    Turn(Turn),
}

fn has_desktop_modifier(modifiers: ModifiersState) -> bool {
    modifiers.control_key() || modifiers.alt_key() || modifiers.super_key()
}

pub fn is_jump_trigger(
    text: Option<&str>,
    is_repeat: bool,
    modifiers: ModifiersState,
) -> bool {
    !is_repeat && !has_desktop_modifier(modifiers) && text == Some(":")
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
        key_code: KeyCode,
        modifiers: ModifiersState,
    ) -> Option<Self> {
        use strum::VariantArray as _;

        if modifiers.shift_key() && !has_desktop_modifier(modifiers) {
            return match key_code {
                KeyCode::Space => Some(Self::Turn(Turn::ReadingOrder(
                    Direction::Backward,
                ))),
                KeyCode::KeyB => {
                    Some(Self::SwitchBook(Direction::Backward))
                }
                _ => None,
            };
        }
        if !modifiers.is_empty() {
            return None;
        }
        if let Some(&mode) = ReadingMode::VARIANTS
            .iter()
            .find(|mode| mode.key_code() == key_code)
        {
            return Some(Self::SelectMode(mode));
        }
        if let Some(&fit_mode) = FitMode::VARIANTS
            .iter()
            .find(|fit_mode| fit_mode.key_code() == key_code)
        {
            return Some(Self::SelectFit(fit_mode));
        }
        match key_code {
            KeyCode::KeyQ => Some(Self::Quit),
            KeyCode::Escape => Some(Self::Escape),
            KeyCode::KeyF => Some(Self::ToggleFullscreen),
            KeyCode::KeyB => Some(Self::SwitchBook(Direction::Forward)),
            KeyCode::Home => Some(Self::First),
            KeyCode::End => Some(Self::Last),
            KeyCode::ArrowRight => Some(Self::Turn(Turn::Right)),
            KeyCode::ArrowLeft => Some(Self::Turn(Turn::Left)),
            KeyCode::Space | KeyCode::PageDown => {
                Some(Self::Turn(Turn::ReadingOrder(Direction::Forward)))
            }
            KeyCode::PageUp => {
                Some(Self::Turn(Turn::ReadingOrder(Direction::Backward)))
            }
            _ => None,
        }
    }
}

pub enum JumpOutcome {
    Ignored,
    Edited,
    Canceled,
    Committed(NonZeroUsize),
}

pub fn handle_jump_key(
    digits: &mut String,
    key: &KeyEvent,
    modifiers: ModifiersState,
    page_count: usize,
) -> JumpOutcome {
    if has_desktop_modifier(modifiers) {
        return JumpOutcome::Ignored;
    }

    // Digits and Backspace repeat at the compositor's rate.
    let digit = key.text.as_deref().and_then(|text| {
        let [byte] = text.as_bytes() else {
            return None;
        };
        byte.is_ascii_digit().then(|| char::from(*byte))
    });
    if let Some(digit) = digit {
        digits.push(digit);
        return JumpOutcome::Edited;
    }

    let PhysicalKey::Code(key_code) = key.physical_key else {
        return JumpOutcome::Ignored;
    };
    if key_code == KeyCode::Backspace {
        digits.pop();
        return JumpOutcome::Edited;
    }
    if key.repeat {
        return JumpOutcome::Ignored;
    }
    if matches!(key_code, KeyCode::Enter | KeyCode::NumpadEnter) {
        if digits.is_empty() {
            JumpOutcome::Canceled
        } else if let Some(target) = parse_page_number(digits, page_count)
        {
            JumpOutcome::Committed(target)
        } else {
            JumpOutcome::Ignored
        }
    } else if key_code == KeyCode::Escape {
        JumpOutcome::Canceled
    } else {
        JumpOutcome::Ignored
    }
}

pub fn wheel_zoom_factor(
    delta: MouseScrollDelta,
    display_scale_factor: f64,
) -> f64 {
    const LOGICAL_PX_PER_LINE: f64 = 40.0;
    const LOG_ZOOM_PER_LINE: f64 = 0.2;

    let wheel_lines = match delta {
        MouseScrollDelta::LineDelta(_, vertical_lines) => {
            f64::from(vertical_lines)
        }
        MouseScrollDelta::PixelDelta(pixels) => {
            pixels.y / display_scale_factor / LOGICAL_PX_PER_LINE
        }
    };
    (wheel_lines * LOG_ZOOM_PER_LINE).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modifier_interactions() {
        let bindings: &[(KeyCode, ModifiersState, Option<Action>)] = &[
            (
                KeyCode::Space,
                ModifiersState::SHIFT,
                Some(Action::Turn(Turn::ReadingOrder(
                    Direction::Backward,
                ))),
            ),
            (
                KeyCode::KeyB,
                ModifiersState::SHIFT,
                Some(Action::SwitchBook(Direction::Backward)),
            ),
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
        for (key_code, modifiers, action) in bindings {
            assert_eq!(
                Action::from_key(*key_code, *modifiers),
                *action,
                "{key_code:?} {modifiers:?}"
            );
        }
        assert!(super::parse_page_number("12", 20).is_some());
        assert!(super::parse_page_number("21", 20).is_none());
        assert!(super::is_jump_trigger(
            Some(":"),
            false,
            ModifiersState::SHIFT
        ));
        assert!(!super::is_jump_trigger(
            Some(":"),
            true,
            ModifiersState::SHIFT
        ));
        assert!(!super::is_jump_trigger(
            Some(":"),
            false,
            ModifiersState::CONTROL
        ));
    }
}
