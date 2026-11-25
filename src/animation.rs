//! Parent-side frame timing; the child owns the decoder cursor and loop policy.

use std::collections::VecDeque;
use std::time::Duration;
use std::time::Instant;

use crate::protocol::PlaybackId;

/// Below this a GIF delay is a stuck frame.
const MIN_FRAME_DELAY: Duration = Duration::from_millis(20);
/// The web convention plays zero and one-centisecond delays at 100 ms.
const FALLBACK_FRAME_DELAY: Duration = Duration::from_millis(100);

const LOOKAHEAD: usize = 2;

struct BufferedFrame<Image> {
    image: Image,
    delay: Duration,
}

#[derive(Clone, Copy)]
enum RequestState {
    Idle,
    Queued,
    Sent,
    Finished,
}

pub struct Player<Image> {
    id: PlaybackId,
    buffered: VecDeque<BufferedFrame<Image>>,
    request: RequestState,
    next_due: Option<Instant>,
}

pub enum Due<Image> {
    Show { image: Image, next_due: Instant },
    Wait(Instant),
    Idle,
}

impl<Image> Player<Image> {
    pub fn new(id: PlaybackId) -> Self {
        Self {
            id,
            buffered: VecDeque::new(),
            request: RequestState::Queued,
            next_due: None,
        }
    }

    #[must_use]
    pub fn id(&self) -> PlaybackId {
        self.id
    }

    /// Buffer a frame the child answered. A frame without a sent request
    /// is stale.
    pub fn on_frame(&mut self, delay: Duration, image: Image) -> bool {
        if !matches!(self.request, RequestState::Sent) {
            return false;
        }
        self.buffered.push_back(BufferedFrame {
            image,
            delay: frame_delay(delay),
        });
        self.request = RequestState::Idle;
        self.replenish();
        true
    }

    /// The child finished every allowed pass. Buffered frames still play.
    pub fn on_finished(&mut self) -> bool {
        if !matches!(self.request, RequestState::Sent) {
            return false;
        }
        self.request = RequestState::Finished;
        true
    }

    /// Claim the next request the app still has to send, if any.
    pub fn take_next(&mut self) -> bool {
        match self.request {
            RequestState::Queued => {
                self.request = RequestState::Sent;
                true
            }
            RequestState::Idle
            | RequestState::Sent
            | RequestState::Finished => false,
        }
    }

    /// A late frame shows on arrival and its delay starts then.
    pub fn tick(&mut self, now: Instant) -> Due<Image> {
        if matches!(self.request, RequestState::Finished)
            && self.buffered.is_empty()
        {
            return Due::Idle;
        }
        if let Some(due) = self.next_due
            && now < due
        {
            return Due::Wait(due);
        }
        if let Some(frame) = self.buffered.pop_front() {
            return self.show(frame, now);
        }
        Due::Idle
    }

    fn show(
        &mut self,
        frame: BufferedFrame<Image>,
        now: Instant,
    ) -> Due<Image> {
        let next_due = now + frame.delay;
        self.next_due = Some(next_due);
        self.replenish();
        Due::Show {
            image: frame.image,
            next_due,
        }
    }

    /// Keeps one request in flight.
    fn replenish(&mut self) {
        if !matches!(self.request, RequestState::Idle) {
            return;
        }
        if self.buffered.len() < LOOKAHEAD {
            self.request = RequestState::Queued;
        }
    }
}

fn frame_delay(delay: Duration) -> Duration {
    if delay < MIN_FRAME_DELAY {
        FALLBACK_FRAME_DELAY
    } else {
        delay
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn player(generation: u32) -> Player<u8> {
        Player::new(PlaybackId {
            page: 0,
            generation,
        })
    }

    fn shown_delay<Image>(
        due: &Due<Image>,
        now: Instant,
    ) -> Option<Duration> {
        match due {
            Due::Show { next_due, .. } => {
                Some(next_due.saturating_duration_since(now))
            }
            Due::Wait(_) | Due::Idle => None,
        }
    }

    #[test]
    fn frames_play_in_order_and_finish() {
        let mut player = player(7);
        assert!(player.take_next());

        let start = Instant::now();
        assert!(player.on_frame(Duration::from_millis(50), 1));
        assert_eq!(
            shown_delay(&player.tick(start), start),
            Some(Duration::from_millis(50))
        );
        assert!(player.take_next());

        assert!(matches!(
            player.tick(start + Duration::from_millis(20)),
            Due::Wait(_)
        ));
        assert!(player.on_frame(Duration::from_millis(50), 2));
        let second_due = start + Duration::from_millis(50);
        assert_eq!(
            shown_delay(&player.tick(second_due), second_due),
            Some(Duration::from_millis(50))
        );
        assert!(player.take_next());

        assert!(player.on_finished());
        assert!(matches!(
            player.tick(start + Duration::from_millis(100)),
            Due::Idle
        ));
        assert!(!player.take_next());
    }

    #[test]
    fn tiny_delays_are_clamped() {
        let mut player = player(1);
        let start = Instant::now();
        assert!(player.take_next());
        assert!(player.on_frame(Duration::ZERO, 1));
        assert_eq!(
            shown_delay(&player.tick(start), start),
            Some(FALLBACK_FRAME_DELAY)
        );
    }

    #[test]
    fn frames_need_a_request() {
        let mut player = player(1);
        assert!(!player.on_frame(Duration::from_millis(20), 1));
        assert!(!player.on_finished());
        assert!(player.take_next());
        assert!(player.on_frame(Duration::from_millis(20), 1));
        assert!(!player.on_frame(Duration::from_millis(20), 1));
        assert!(!player.on_finished());
    }

    #[test]
    fn finished_still_drains_buffered_frames() {
        let mut player = player(1);
        let start = Instant::now();
        assert!(player.take_next());
        assert!(player.on_frame(Duration::from_millis(50), 1));
        assert!(player.take_next());
        assert!(player.on_finished());

        assert_eq!(
            shown_delay(&player.tick(start), start),
            Some(Duration::from_millis(50))
        );
        assert!(matches!(
            player.tick(start + Duration::from_millis(50)),
            Due::Idle
        ));
    }
}
