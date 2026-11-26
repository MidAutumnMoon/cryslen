//! In-process GIF playback stream over a 1-frame bounded channel.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread::Builder;
use std::time::Duration;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::TryRecvError;

use crate::book::source;
use crate::image::decode::gif;
use crate::image::decode::gif::Animation;
use crate::image::decode::gif::Frame;
use crate::image::decode::gif::Loops;
use crate::image::pixels::Image;
use crate::image::pixels::ImageLimits;
use crate::image::pixels::TargetRole;
use crate::image::resample;

/// Below this a GIF delay is a stuck frame.
const MIN_FRAME_DELAY: Duration = Duration::from_millis(20);
/// The web convention plays zero and one-centisecond delays at 100 ms.
const FALLBACK_FRAME_DELAY: Duration = Duration::from_millis(100);

pub enum GifTick {
    Show { image: Image, next_due: Instant },
    Wait(Instant),
    Idle,
    Failed,
}

enum ProducerMessage {
    Frame(Frame),
    Finished,
    Failed,
}

pub struct GifPlayer {
    receiver: Receiver<ProducerMessage>,
    waiting_wake: Arc<AtomicBool>,
    next_due: Option<Instant>,
    finished: bool,
}

impl GifPlayer {
    pub fn start(
        bytes: Arc<[u8]>,
        limits: ImageLimits,
        wake: Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        let (sender, receiver) = crossbeam_channel::bounded(1);
        let waiting_wake = Arc::new(AtomicBool::new(false));
        let producer_wake = Arc::clone(&waiting_wake);
        let spawned =
            Builder::new().name(String::from("cy-gif")).spawn(move || {
                run_producer(
                    &bytes,
                    limits,
                    &sender,
                    &producer_wake,
                    &*wake,
                );
            });
        if let Err(error) = spawned {
            tracing::warn!(%error, "could not spawn gif player");
            return Self {
                receiver,
                waiting_wake,
                next_due: None,
                finished: true,
            };
        }
        Self {
            receiver,
            waiting_wake,
            next_due: None,
            finished: false,
        }
    }

    pub fn tick(&mut self, now: Instant) -> GifTick {
        if self.finished {
            return GifTick::Idle;
        }
        if let Some(due) = self.next_due
            && now < due
        {
            return GifTick::Wait(due);
        }
        if let Some(tick) = self.try_pop_frame(now) {
            return tick;
        }
        // Arm wakeup before re-checking so a concurrent send cannot be missed.
        self.waiting_wake.store(true, Ordering::Release);
        if let Some(tick) = self.try_pop_frame(now) {
            self.waiting_wake.store(false, Ordering::Relaxed);
            return tick;
        }
        GifTick::Idle
    }

    fn try_pop_frame(&mut self, now: Instant) -> Option<GifTick> {
        match self.receiver.try_recv() {
            Ok(ProducerMessage::Frame(frame)) => {
                let next_due = now + frame_delay(frame.delay);
                self.next_due = Some(next_due);
                Some(GifTick::Show {
                    image: frame.image,
                    next_due,
                })
            }
            Ok(ProducerMessage::Finished) => {
                self.finished = true;
                Some(GifTick::Idle)
            }
            Ok(ProducerMessage::Failed)
            | Err(TryRecvError::Disconnected) => {
                self.finished = true;
                Some(GifTick::Failed)
            }
            Err(TryRecvError::Empty) => None,
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

fn run_producer(
    bytes: &Arc<[u8]>,
    limits: ImageLimits,
    sender: &crossbeam_channel::Sender<ProducerMessage>,
    waiting_wake: &AtomicBool,
    wake: &dyn Fn(),
) {
    const PLAYBACK_RESAMPLE_THREADS: usize = 1;

    let emit = |message: ProducerMessage| -> bool {
        let sent = sender.send(message).is_ok();
        if waiting_wake.swap(false, Ordering::AcqRel) {
            wake();
        }
        sent
    };

    let mut session = match source::catch_panic("playback open", || {
        gif::open_animation(Arc::clone(bytes))
    }) {
        Ok(animation) => PlaybackSession {
            animation,
            passes: 0,
        },
        Err(error) => {
            tracing::warn!(%error, "playback open failed");
            let _kept = emit(ProducerMessage::Failed);
            return;
        }
    };

    loop {
        let message = match source::catch_panic("playback frame", || {
            session.advance()
        }) {
            Ok(Some(mut frame)) => {
                match resample::derive(
                    &frame.image,
                    TargetRole::Full,
                    limits,
                    PLAYBACK_RESAMPLE_THREADS,
                ) {
                    Ok(derived) => {
                        frame.image = derived.into_owned();
                        ProducerMessage::Frame(frame)
                    }
                    Err(fault) => {
                        tracing::warn!(
                            %fault,
                            "playback frame exceeds texture limit"
                        );
                        ProducerMessage::Failed
                    }
                }
            }
            Ok(None) => ProducerMessage::Finished,
            Err(error) => {
                tracing::warn!(%error, "playback frame failed");
                ProducerMessage::Failed
            }
        };
        let done = !matches!(message, ProducerMessage::Frame(_));
        if !emit(message) || done {
            return;
        }
    }
}

struct PlaybackSession {
    animation: Animation,
    passes: u32,
}

impl PlaybackSession {
    fn advance(&mut self) -> eros::Result<Option<Frame>> {
        if let Some(frame) = self.animation.next_frame().transpose()? {
            return Ok(Some(frame));
        }
        self.passes = self.passes.saturating_add(1);
        if !self.should_repeat() {
            return Ok(None);
        }
        self.animation.rewind()?;
        self.animation.next_frame().transpose()
    }

    fn should_repeat(&self) -> bool {
        match self.animation.loops() {
            Loops::Infinite => true,
            Loops::Finite(total_passes) => self.passes < total_passes,
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use image::codecs::gif::Repeat;

    use super::*;
    use crate::image::pixels::Pixels;
    use crate::testkit;

    fn first_pixel(image: &Image) -> [u8; 4] {
        let Pixels::Rgba8(bytes) = image.pixels() else {
            panic!("a gif frame decodes to rgba");
        };
        let mut pixel = [0; 4];
        pixel.copy_from_slice(bytes.get(..4).expect("one pixel"));
        pixel
    }

    fn next_shown(player: &mut GifPlayer, now: Instant) -> Image {
        testkit::eventually("no animation frame", || {
            match player.tick(now) {
                GifTick::Show { image, .. } => Some(image),
                GifTick::Wait(_) | GifTick::Idle => None,
                GifTick::Failed => panic!("playback failed"),
            }
        })
    }

    #[test]
    fn animation_frames_stream_in_order_and_loop() {
        let colors =
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
        let bytes: Arc<[u8]> = Arc::from(testkit::tiny_animated_gif(
            8,
            8,
            &colors,
            50,
            Repeat::Infinite,
        ));
        let mut player =
            GifPlayer::start(bytes, testkit::limits(), Arc::new(|| {}));
        let mut now = Instant::now();
        for color in colors {
            let image = next_shown(&mut player, now);
            assert_eq!((image.width(), image.height()), (8, 8));
            assert_eq!(first_pixel(&image), color);
            assert!(matches!(
                player.tick(now + Duration::from_millis(20)),
                GifTick::Wait(_)
            ));
            now += Duration::from_millis(50);
        }
        let looped = next_shown(&mut player, now);
        assert_eq!(first_pixel(&looped), colors[0]);
    }

    #[test]
    fn a_finite_animation_finishes_after_its_last_pass() {
        let colors =
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
        let bytes: Arc<[u8]> = Arc::from(testkit::tiny_animated_gif(
            8,
            8,
            &colors,
            50,
            Repeat::Finite(0),
        ));
        let mut player =
            GifPlayer::start(bytes, testkit::limits(), Arc::new(|| {}));
        let mut now = Instant::now();
        for color in colors {
            let image = next_shown(&mut player, now);
            assert_eq!(first_pixel(&image), color);
            now += Duration::from_millis(50);
        }
        testkit::eventually("player never finished", || {
            player.tick(now);
            player.finished.then_some(())
        });
        assert!(matches!(player.tick(now), GifTick::Idle));
    }
}
