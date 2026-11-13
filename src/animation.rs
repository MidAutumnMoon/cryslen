//! Playback vocabulary and the parent's frame pipeline: no IO, GL, or
//! decode. The child owns the decoder cursor and the loop policy; this
//! decides when a delivered frame is due.

use std::collections::VecDeque;
use std::time::Duration;
use std::time::Instant;

use serde::Deserialize;
use serde::Serialize;

use crate::browse::Role;
use crate::decode::Decoded;

/// Below this a GIF delay is a stuck frame.
const MIN_FRAME_DELAY: Duration = Duration::from_millis(20);
/// The web convention plays zero and one-centisecond delays at 100 ms.
const FALLBACK_FRAME_DELAY: Duration = Duration::from_millis(100);

const LOOKAHEAD: usize = 2;

/// `generation` changes on every restart, so a stopped session's frames
/// cannot reach the new one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[non_exhaustive]
pub struct PlaybackId {
    pub page: usize,
    pub generation: u32,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub enum PlaybackCommand {
    /// Open the page's stream and fix its role.
    Start { id: PlaybackId, role: Role },
    /// Produce the next frame. A pass boundary is invisible to the parent.
    Next { id: PlaybackId },
    /// Drop the session. The child answers nothing.
    Stop { id: PlaybackId },
}

/// One child answer about a session. Frames arrive in order.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[non_exhaustive]
pub enum PlaybackEvent<Image> {
    /// One composited frame, already sized for the session's role.
    Frame {
        id: PlaybackId,
        delay: Duration,
        image: Image,
    },
    /// Every allowed pass has been played.
    Finished { id: PlaybackId },
    /// Playback failed. The page stays valid.
    Failed { id: PlaybackId },
}

pub type DecodedEvent = PlaybackEvent<Box<Decoded>>;

impl<Image> PlaybackEvent<Image> {
    #[must_use]
    pub fn map_image<Next>(
        self,
        map: impl FnOnce(Image) -> Next,
    ) -> PlaybackEvent<Next> {
        match self {
            Self::Frame { id, delay, image } => PlaybackEvent::Frame {
                id,
                delay,
                image: map(image),
            },
            Self::Finished { id } => PlaybackEvent::Finished { id },
            Self::Failed { id } => PlaybackEvent::Failed { id },
        }
    }
}

struct BufferedFrame {
    image: Box<Decoded>,
    delay: Duration,
}

struct ShownFrame {
    delay: Duration,
    shown_at: Instant,
}

#[derive(Clone, Copy)]
enum RequestState {
    Idle,
    Queued,
    Sent,
}

pub struct Player {
    id: PlaybackId,
    role: Role,
    buffered: VecDeque<BufferedFrame>,
    request: RequestState,
    shown: Option<ShownFrame>,
    finished: bool,
}

pub enum Due {
    Show {
        image: Box<Decoded>,
        delay: Duration,
    },
    Wait(Duration),
    Idle,
}

impl Player {
    pub fn new(id: PlaybackId, role: Role) -> Self {
        Self {
            id,
            role,
            buffered: VecDeque::new(),
            request: RequestState::Queued,
            shown: None,
            finished: false,
        }
    }

    #[must_use]
    pub fn id(&self) -> PlaybackId {
        self.id
    }

    #[must_use]
    pub fn role(&self) -> Role {
        self.role
    }

    /// Buffer a frame the child answered. A frame without a sent request
    /// is stale.
    pub fn on_frame(
        &mut self,
        delay: Duration,
        image: Box<Decoded>,
    ) -> bool {
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
        self.finished = true;
        self.request = RequestState::Idle;
        true
    }

    /// Claim the next request the app still has to send, if any.
    pub fn take_next(&mut self) -> bool {
        match self.request {
            RequestState::Queued => {
                self.request = RequestState::Sent;
                true
            }
            RequestState::Idle | RequestState::Sent => false,
        }
    }

    /// A late frame shows on arrival and its delay starts then.
    pub fn tick(&mut self, now: Instant) -> Due {
        if self.finished && self.buffered.is_empty() {
            return Due::Idle;
        }
        if let Some(shown) = &self.shown {
            let due = shown.shown_at + shown.delay;
            if now < due {
                return Due::Wait(due - now);
            }
        }
        if let Some(frame) = self.buffered.pop_front() {
            return self.show(frame, now);
        }
        Due::Idle
    }

    fn show(&mut self, frame: BufferedFrame, now: Instant) -> Due {
        self.shown = Some(ShownFrame {
            delay: frame.delay,
            shown_at: now,
        });
        self.replenish();
        Due::Show {
            image: frame.image,
            delay: frame.delay,
        }
    }

    /// Keeps one request in flight.
    fn replenish(&mut self) {
        if self.finished || !matches!(self.request, RequestState::Idle) {
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
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;
    use crate::decode::Format;
    use crate::decode::Transfer;

    fn player(generation: u32) -> Player {
        Player::new(
            PlaybackId {
                page: 0,
                generation,
            },
            Role::Fit,
        )
    }

    fn frame(shade: u8) -> Decoded {
        Decoded::rgba(
            1,
            1,
            Format::Gif,
            Transfer::Srgb,
            vec![shade, 0, 0, 255].into_boxed_slice(),
        )
        .expect("test frame constructs")
    }

    fn shown_delay(due: &Due) -> Option<Duration> {
        match due {
            Due::Show { delay, .. } => Some(*delay),
            Due::Wait(_) | Due::Idle => None,
        }
    }

    #[test]
    fn frames_play_in_order_and_finish() {
        let mut player = player(7);
        assert!(player.take_next());

        let start = Instant::now();
        assert!(
            player.on_frame(Duration::from_millis(50), Box::new(frame(1)))
        );
        assert_eq!(
            shown_delay(&player.tick(start)),
            Some(Duration::from_millis(50))
        );
        assert!(player.take_next());

        assert!(matches!(
            player.tick(start + Duration::from_millis(20)),
            Due::Wait(_)
        ));
        assert!(
            player.on_frame(Duration::from_millis(50), Box::new(frame(2)))
        );
        assert_eq!(
            shown_delay(&player.tick(start + Duration::from_millis(50))),
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
        assert!(player.take_next());
        assert!(player.on_frame(Duration::ZERO, Box::new(frame(1))));
        assert_eq!(
            shown_delay(&player.tick(Instant::now())),
            Some(FALLBACK_FRAME_DELAY)
        );
    }

    #[test]
    fn frames_need_a_request() {
        let mut player = player(1);
        assert!(!player
            .on_frame(Duration::from_millis(20), Box::new(frame(1))));
        assert!(!player.on_finished());
        assert!(player.take_next());
        assert!(
            player.on_frame(Duration::from_millis(20), Box::new(frame(1)))
        );
        assert!(!player
            .on_frame(Duration::from_millis(20), Box::new(frame(1))));
        assert!(!player.on_finished());
    }

    #[test]
    fn finished_still_drains_buffered_frames() {
        let mut player = player(1);
        let start = Instant::now();
        assert!(player.take_next());
        assert!(
            player.on_frame(Duration::from_millis(50), Box::new(frame(1)))
        );
        assert!(player.take_next());
        assert!(player.on_finished());

        assert_eq!(
            shown_delay(&player.tick(start)),
            Some(Duration::from_millis(50))
        );
        assert!(matches!(
            player.tick(start + Duration::from_millis(50)),
            Due::Idle
        ));
    }
}
