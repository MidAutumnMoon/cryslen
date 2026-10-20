//! The baked-in key map: physical keys to `Action`s.

use winit::keyboard::KeyCode;

/// Input actions, derived from physical keys (layout-independent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Next,
    Prev,
    Quit,
    ToggleDouble,
    ToggleRtl,
    ToggleFullscreen,
    ResetView,
}

impl Action {
    /// Whether a held key keeps firing this action. Only paging repeats.
    pub(crate) fn repeats(self) -> bool {
        matches!(self, Action::Next | Action::Prev)
    }
}

/// Key map, baked in. Arrows are directional: Right means Next, Left means
/// Prev. The app swaps them when `rtl` is on, because the reading direction
/// decides, not the key label.
pub fn action(code: KeyCode) -> Option<Action> {
    use Action::*;
    match code {
        KeyCode::Escape | KeyCode::KeyQ => Some(Quit),
        KeyCode::KeyD => Some(ToggleDouble),
        KeyCode::KeyM => Some(ToggleRtl),
        KeyCode::KeyF => Some(ToggleFullscreen),
        KeyCode::KeyR => Some(ResetView),
        KeyCode::ArrowRight | KeyCode::Space | KeyCode::PageDown => Some(Next),
        KeyCode::ArrowLeft | KeyCode::PageUp => Some(Prev),
        _ => None,
    }
}

/// Whether an action advances reading order under the current direction.
pub fn is_forward(act: Action, rtl: bool) -> bool {
    match act {
        Action::Next => !rtl,
        Action::Prev => rtl,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_map() {
        assert_eq!(action(KeyCode::Escape), Some(Action::Quit));
        assert_eq!(action(KeyCode::KeyQ), Some(Action::Quit));
        assert_eq!(action(KeyCode::KeyD), Some(Action::ToggleDouble));
        assert_eq!(action(KeyCode::KeyM), Some(Action::ToggleRtl));
        assert_eq!(action(KeyCode::KeyF), Some(Action::ToggleFullscreen));
        assert_eq!(action(KeyCode::KeyR), Some(Action::ResetView));
        assert_eq!(action(KeyCode::ArrowRight), Some(Action::Next));
        assert_eq!(action(KeyCode::Space), Some(Action::Next));
        assert_eq!(action(KeyCode::PageDown), Some(Action::Next));
        assert_eq!(action(KeyCode::ArrowLeft), Some(Action::Prev));
        assert_eq!(action(KeyCode::PageUp), Some(Action::Prev));
        assert_eq!(action(KeyCode::KeyA), None);
        assert_eq!(action(KeyCode::ArrowUp), None);
    }

    #[test]
    fn only_paging_repeats_on_hold() {
        assert!(Action::Next.repeats());
        assert!(Action::Prev.repeats());
        let one_shot = [
            Action::Quit,
            Action::ToggleDouble,
            Action::ToggleRtl,
            Action::ToggleFullscreen,
            Action::ResetView,
        ];
        for act in one_shot {
            assert!(!act.repeats());
        }
    }

    #[test]
    fn forward_follows_reading_direction() {
        // Western: Right/Space advances.
        assert!(is_forward(Action::Next, false));
        assert!(!is_forward(Action::Prev, false));
        // RTL manga: the same keys flip, so Left-ward "next" reads forward.
        assert!(!is_forward(Action::Next, true));
        assert!(is_forward(Action::Prev, true));
    }

    #[test]
    fn page_turn_respects_rtl() {
        // advance() composes with is_forward: in RTL, "Next" from page 1
        // moves to page 0 (the cover), in western to page 2.
        let fwd = is_forward(Action::Next, true);
        assert_eq!(crate::view::advance(1, fwd, false, 10), 0);
        let fwd = is_forward(Action::Next, false);
        assert_eq!(crate::view::advance(1, fwd, false, 10), 2);
    }
}
