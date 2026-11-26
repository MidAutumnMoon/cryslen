//! JPEG XL decode through `jxl` with cooperative cancellation.

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use jxl::api::JxlParallelRunner;
use jxl::api::JxlParallelRunnerFun;
use jxl::error::Error as JxlError;
use jxl::error::Result as JxlResult;

use crate::image::pixels;
use crate::image::pixels::Image;

use super::codec::DecodeOutcome;

const MAX_JXL_THREADS: usize = 16;

pub(super) fn decode(
    bytes: &[u8],
    max_threads: usize,
    cancel: &AtomicBool,
) -> eros::Result<DecodeOutcome> {
    use std::slice;

    use jxl::api::Event;
    use jxl::api::ExtraChannel;
    use jxl::api::JxlColorType;
    use jxl::api::JxlDataFormat;
    use jxl::api::JxlDecoder;
    use jxl::api::JxlDecoderOptions;
    use jxl::api::JxlOutputBuffer;
    use jxl::api::JxlPixelFormat;

    use crate::image::pixels::Depth;
    use crate::image::pixels::Format;
    use crate::image::pixels::Transfer;

    if cancel.load(Ordering::Relaxed) {
        return Ok(DecodeOutcome::Canceled);
    }

    let mut runner = Pool {
        threads: max_threads.clamp(1, MAX_JXL_THREADS),
        cancel,
    };
    // The whole file is in memory, so `NeedMoreInput` means the file is
    // truncated.
    let mut input: &[u8] = bytes;
    let mut decoder = JxlDecoder::new(JxlDecoderOptions::default());

    loop {
        let event =
            match decoder.process(&mut input, None, Some(&mut runner)) {
                Ok(event) => event,
                Err(_) if cancel.load(Ordering::Relaxed) => {
                    return Ok(DecodeOutcome::Canceled);
                }
                Err(error) => return Err(error.into()),
            };
        if event == Event::BasicInfo {
            break;
        }
        if let Event::NeedMoreInput { size_hint } = event {
            tracing::debug!(size_hint, "truncated jxl header");
            eros::bail!("jxl: truncated header");
        }
        // A fresh decoder emits BasicInfo or NeedMoreInput before all else.
    }

    let info = decoder
        .basic_info()
        .ok_or_else(|| eros::error!("jxl: missing basic info"))?;
    let (Ok(width), Ok(height)) =
        (u32::try_from(info.size.0), u32::try_from(info.size.1))
    else {
        eros::bail!("jxl: bad dimensions");
    };
    let pixel_count =
        pixels::admit_dims(width, height, "jxl: bad dimensions")?;

    let extra_channels = info.extra_channels.len();
    let has_alpha = info
        .extra_channels
        .iter()
        .any(|channel| channel.ec_type == ExtraChannel::Alpha);
    // `BasicInfo` seeds `current_pixel_format()` with the source color type.
    let mono = !has_alpha
        && decoder
            .current_pixel_format()
            .is_some_and(|format| format.color_type.is_grayscale());
    let pixel_format = if mono {
        JxlPixelFormat {
            color_type: JxlColorType::Grayscale,
            color_data_format: Some(JxlDataFormat::U8 {
                bit_depth: Depth::D8.bits(),
            }),
            extra_channel_format: vec![None; extra_channels],
        }
    } else {
        JxlPixelFormat::rgba8(extra_channels)
    };
    let samples_per_pixel = pixel_format.color_type.samples_per_pixel();
    let stride = width as usize * samples_per_pixel;
    decoder.set_pixel_format(pixel_format)?;

    let mut buffer = vec![0_u8; pixel_count as usize * samples_per_pixel];
    loop {
        let mut output =
            JxlOutputBuffer::new(&mut buffer, height as usize, stride);
        let event = match decoder.process(
            &mut input,
            Some(slice::from_mut(&mut output)),
            Some(&mut runner),
        ) {
            Ok(event) => event,
            Err(_) if cancel.load(Ordering::Relaxed) => {
                return Ok(DecodeOutcome::Canceled);
            }
            Err(error) => return Err(error.into()),
        };
        match event {
            Event::FrameComplete { has_more_frames } => {
                if has_more_frames {
                    tracing::debug!(
                        "jxl animation: later frames not decoded"
                    );
                }
                break;
            }
            Event::NeedMoreInput { size_hint } => {
                tracing::debug!(size_hint, "truncated jxl image");
                eros::bail!("jxl: truncated image");
            }
            Event::BasicInfo | Event::FrameHeader | Event::Complete => {}
        }
    }

    // The crate either converts to sRGB or emits simple PQ values its
    // API does not distinguish, so jxl pages ride the sRGB path.
    let page = if mono {
        Image::gray_u8(
            width,
            height,
            Format::Jxl,
            Transfer::Srgb,
            buffer.into_boxed_slice(),
        )?
    } else {
        Image::rgba(
            width,
            height,
            Format::Jxl,
            Transfer::Srgb,
            buffer.into_boxed_slice(),
        )?
    };
    Ok(DecodeOutcome::Ready(page))
}

fn canceled_error() -> JxlError {
    use std::io;
    use std::io::ErrorKind;

    JxlError::IOError(io::Error::new(ErrorKind::Interrupted, "canceled"))
}

/// Scoped-thread runner. A failing task fails the whole run, which is
/// how the crate's driver expects a broken codestream to be reported.
struct Pool<'cancel> {
    threads: usize,
    cancel: &'cancel AtomicBool,
}

impl JxlParallelRunner for Pool<'_> {
    fn run(
        &mut self,
        num: usize,
        fun: &JxlParallelRunnerFun<'_>,
    ) -> JxlResult<()> {
        use std::sync::OnceLock;
        use std::thread;

        if self.cancel.load(Ordering::Relaxed) {
            return Err(canceled_error());
        }
        let thread_count = self.threads.min(num);
        if thread_count <= 1 {
            for task_index in 0..num {
                if self.cancel.load(Ordering::Relaxed) {
                    return Err(canceled_error());
                }
                fun(task_index)?;
            }
            return Ok(());
        }
        let tasks_per_thread = num.div_ceil(thread_count);
        let failure: OnceLock<JxlError> = OnceLock::new();
        thread::scope(|scope| {
            for thread_index in 0..thread_count {
                let run_task: &JxlParallelRunnerFun = fun;
                let failure = &failure;
                let cancel = self.cancel;
                scope.spawn(move || {
                    for task_index in thread_index * tasks_per_thread
                        ..((thread_index + 1) * tasks_per_thread).min(num)
                    {
                        if cancel.load(Ordering::Relaxed) {
                            failure.get_or_init(canceled_error);
                            break;
                        }
                        if let Err(task_error) = run_task(task_index) {
                            // The first failure wins; a later one has
                            // nothing to add.
                            failure.get_or_init(|| task_error);
                            break;
                        }
                    }
                });
            }
        });
        failure.into_inner().map_or(Ok(()), Err)
    }

    fn run_ordered(
        &mut self,
        num: usize,
        max_threads: Option<usize>,
        fun: &JxlParallelRunnerFun<'_>,
    ) -> JxlResult<()> {
        use std::sync::atomic::AtomicUsize;

        let max_threads = max_threads
            .unwrap_or(usize::MAX)
            .min(self.num_threads())
            .min(num);
        if max_threads <= 1 {
            for task_index in 0..num {
                if self.cancel.load(Ordering::Relaxed) {
                    return Err(canceled_error());
                }
                fun(task_index)?;
            }
            return Ok(());
        }
        let next_index = AtomicUsize::new(0);
        let cancel = self.cancel;
        self.run(max_threads, &|_| loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(canceled_error());
            }
            let task_index = next_index.fetch_add(1, Ordering::Relaxed);
            if task_index >= num {
                return Ok(());
            }
            fun(task_index)?;
        })
    }

    fn num_threads(&self) -> usize {
        self.threads
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;
    use crate::image::decode::codec;
    use crate::image::pixels::Format;
    use crate::image::pixels::Pixels;
    use crate::image::pixels::SamplePlanes;

    // wide.jxl and animated.jxl are verbatim copies of exif.jxl and
    // 5_frames_numbered_jxli.jxl from jxl 0.8.0's tests/testdata.
    #[test]
    fn jxl_fixtures_decode() {
        let gray = codec::decode(
            include_bytes!("../../../tests/fixtures/gray.jxl"),
            "a.jxl",
        )
        .expect("gray jxl decodes");
        assert_eq!((gray.width(), gray.height()), (8, 8));
        assert_eq!(gray.format(), Format::Jxl);
        match gray.pixels() {
            Pixels::Gray {
                full_range,
                planes:
                    SamplePlanes::U8 {
                        planes: [luma],
                        alpha,
                    },
            } => {
                assert!(*full_range);
                assert!(alpha.is_none());
                assert_eq!(luma.first().copied(), Some(0x20));
                assert_eq!(luma.get(32).copied(), Some(0xE0));
            }
            Pixels::Gray { .. }
            | Pixels::Rgba8(_)
            | Pixels::Yuv { .. } => {
                panic!("opaque 8-bit gray jxl must decode to U8 gray")
            }
        }

        let alpha = codec::decode(
            include_bytes!("../../../tests/fixtures/alpha.jxl"),
            "a.jxl",
        )
        .expect("alpha jxl decodes");
        match alpha.pixels() {
            Pixels::Rgba8(bytes) => {
                assert_eq!(
                    bytes.get(..4),
                    Some(&[0x40, 0x80, 0xC0, 0xFF][..])
                );
                assert_eq!(
                    bytes.get(128..132),
                    Some(&[0x40, 0x80, 0xC0, 0x80][..])
                );
            }
            Pixels::Gray { .. } | Pixels::Yuv { .. } => {
                panic!("jxl must decode to rgba")
            }
        }

        let wide = codec::decode(
            include_bytes!("../../../tests/fixtures/wide.jxl"),
            "a.jxl",
        )
        .expect("wide jxl decodes");
        assert_eq!((wide.width(), wide.height()), (256, 128));
        let Pixels::Rgba8(wide_rgba) = wide.pixels() else {
            panic!("jxl must decode to rgba")
        };
        assert!(
            wide_rgba.as_chunks::<4>().0.iter().all(|px| px[3] == 0xFF)
        );

        let animated = codec::decode(
            include_bytes!("../../../tests/fixtures/animated.jxl"),
            "a.jxl",
        )
        .expect("animated jxl decodes");
        assert_eq!((animated.width(), animated.height()), (256, 256));
        let Pixels::Rgba8(animated_rgba) = animated.pixels() else {
            panic!("jxl must decode to rgba")
        };
        assert!(
            animated_rgba
                .as_chunks::<4>()
                .0
                .iter()
                .any(|px| px[0] != 0 || px[1] != 0 || px[2] != 0)
        );
    }

    #[test]
    fn truncated_jxl_ends_the_decode() {
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let fixture = include_bytes!("../../../tests/fixtures/gray.jxl");
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            for cut in [1_usize, 2, 8, 35] {
                let truncated =
                    fixture.get(..cut).expect("cut inside the fixture");
                let result = codec::decode(truncated, "a.jxl");
                if sender.send((cut, result)).is_err() {
                    return;
                }
            }
        });
        for _ in 0..4 {
            let (cut, result) = receiver
                .recv_timeout(Duration::from_secs(5))
                .expect("truncated jxl decode finishes");
            assert!(result.is_err(), "cut {cut} did not fail");
        }
    }

    // The fixtures are single-group, so only this test exercises the
    // runner's parallel path.
    #[test]
    fn pool_reports_the_first_task_error() {
        use std::sync::atomic::AtomicUsize;

        use jxl::api::JxlParallelRunner as _;

        const TOTAL_TASKS: usize = 100;

        let never_cancel = AtomicBool::new(false);
        let mut pool = Pool {
            threads: 4,
            cancel: &never_cancel,
        };
        let failing = |task: usize| {
            if task == 1 {
                Err(JxlError::InvalidSignature)
            } else {
                Ok(())
            }
        };
        assert!(matches!(
            pool.run(4, &failing),
            Err(JxlError::InvalidSignature)
        ));

        let passing = |_: usize| Ok(());
        pool.run(4, &passing).expect("every passing task succeeds");

        let mut two_threads = Pool {
            threads: 2,
            cancel: &never_cancel,
        };
        let failing_last = |task: usize| {
            if task == 3 {
                Err(JxlError::InvalidSignature)
            } else {
                Ok(())
            }
        };
        assert!(matches!(
            two_threads.run(4, &failing_last),
            Err(JxlError::InvalidSignature)
        ));

        let cancel = AtomicBool::new(false);
        let completed_tasks = AtomicUsize::new(0);
        let mut cancellable = Pool {
            threads: 2,
            cancel: &cancel,
        };
        let abort_after_first = |_: usize| {
            completed_tasks.fetch_add(1, Ordering::Relaxed);
            cancel.store(true, Ordering::Relaxed);
            Ok(())
        };
        assert!(matches!(
            cancellable.run_ordered(TOTAL_TASKS, None, &abort_after_first),
            Err(JxlError::IOError(_))
        ));
        assert!(completed_tasks.load(Ordering::Relaxed) < TOTAL_TASKS);
    }
}
