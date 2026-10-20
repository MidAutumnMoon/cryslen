//! The baked-in key map: physical keys to `Action`s.

use winit::keyboard::KeyCode;

/// Input actions, derived from physical keys (layout-independent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    Next,
    Prev,
    Quit,
    ToggleDouble,
    ToggleRtl,
    ToggleFullscreen,
    ResetView,
}

impl Action {
    /// Key map, baked in. Arrows are directional: Right means Next, Left means
    /// Prev. The app swaps them when `rtl` is on, because the reading direction
    /// decides, not the key label.
    pub(crate) fn from_key(code: KeyCode) -> Option<Action> {
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

    /// Whether a held key keeps firing this action. Only paging repeats.
    pub(crate) fn repeats(self) -> bool {
        matches!(self, Action::Next | Action::Prev)
    }

    /// Whether an action advances reading order under the current direction.
    pub(crate) fn is_forward(self, rtl: bool) -> bool {
        match self {
            Action::Next => !rtl,
            Action::Prev => rtl,
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
        assert_eq!(Action::from_key(KeyCode::KeyD), Some(Action::ToggleDouble));
        assert_eq!(Action::from_key(KeyCode::KeyM), Some(Action::ToggleRtl));
        assert_eq!(Action::from_key(KeyCode::KeyF), Some(Action::ToggleFullscreen));
        assert_eq!(Action::from_key(KeyCode::KeyR), Some(Action::ResetView));
        assert_eq!(Action::from_key(KeyCode::ArrowRight), Some(Action::Next));
        assert_eq!(Action::from_key(KeyCode::Space), Some(Action::Next));
        assert_eq!(Action::from_key(KeyCode::PageDown), Some(Action::Next));
        assert_eq!(Action::from_key(KeyCode::ArrowLeft), Some(Action::Prev));
        assert_eq!(Action::from_key(KeyCode::PageUp), Some(Action::Prev));
        assert_eq!(Action::from_key(KeyCode::KeyA), None);
        assert_eq!(Action::from_key(KeyCode::ArrowUp), None);
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
        assert!(Action::Next.is_forward(false));
        assert!(!Action::Prev.is_forward(false));
        // RTL manga: the same keys flip, so Left-ward "next" reads forward.
        assert!(!Action::Next.is_forward(true));
        assert!(Action::Prev.is_forward(true));
    }

    #[test]
    fn page_turn_respects_rtl() {
        // advance() composes with is_forward: in RTL, "Next" from page 1
        // moves to page 0 (the cover), in western to page 2.
        let fwd = Action::Next.is_forward(true);
        assert_eq!(crate::view::advance(1, fwd, false, 10), 0);
        let fwd = Action::Next.is_forward(false);
        assert_eq!(crate::view::advance(1, fwd, false, 10), 2);
    }
}
