use jxl::api::Event;
use jxl::api::JxlDecoder;
use jxl::api::JxlDecoderOptions;
use jxl::api::JxlOutputBuffer;
use jxl::api::JxlParallelRunner;
use jxl::api::JxlParallelRunnerFun;
use jxl::api::JxlPixelFormat;
use jxl::error::Error as JxlError;
use jxl::error::Result as JxlResult;

use eros::ErrorUnion;
use eros::IntoUnion as _;

use crate::fault::DecodeError;
use crate::fault::DecodeRefusal;

use super::Decoded;
use super::Format;
use super::Transfer;

const MAX_THREADS: usize = 16;

fn bad_dimensions() -> DecodeError {
    ErrorUnion::new(DecodeRefusal::Corrupt("jxl: bad dimensions"))
}

pub(super) fn decode(bytes: &[u8]) -> Result<Decoded, DecodeError> {
    use std::slice;

    let mut runner = Pool(threads());
    // The whole file is in memory, so `NeedMoreInput` means the file is
    // truncated.
    let mut input: &[u8] = bytes;
    let mut decoder = JxlDecoder::new(JxlDecoderOptions::default());

    loop {
        let event = decoder
            .process(&mut input, None, Some(&mut runner))
            .union()?;
        if event == Event::BasicInfo {
            break;
        }
        if let Event::NeedMoreInput { size_hint } = event {
            tracing::debug!(size_hint, "truncated jxl header");
            return Err(ErrorUnion::new(DecodeRefusal::Corrupt(
                "jxl: truncated header",
            )));
        }
        // A fresh decoder emits BasicInfo or NeedMoreInput before all else.
    }

    let info = decoder.basic_info().ok_or_else(|| {
        ErrorUnion::new(DecodeRefusal::Corrupt("jxl: missing basic info"))
    })?;
    #[expect(
        clippy::map_err_ignore,
        reason = "the conversion error only says the value does not fit, which the message already states"
    )]
    let (width, height) = (
        u32::try_from(info.size.0).map_err(|_| bad_dimensions())?,
        u32::try_from(info.size.1).map_err(|_| bad_dimensions())?,
    );
    let pixel_count =
        super::checked_pixels(width, height, "jxl: bad dimensions")
            .union()?;
    super::check_dims(width, height).union()?;

    let extra_channels = info.extra_channels.len();
    decoder
        .set_pixel_format(JxlPixelFormat::rgba8(extra_channels))
        .union()?;

    let mut rgba = vec![0_u8; pixel_count as usize * 4];
    loop {
        let mut output = JxlOutputBuffer::new(
            &mut rgba,
            height as usize,
            width as usize * 4,
        );
        match decoder
            .process(
                &mut input,
                Some(slice::from_mut(&mut output)),
                Some(&mut runner),
            )
            .union()?
        {
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
                return Err(ErrorUnion::new(DecodeRefusal::Corrupt(
                    "jxl: truncated image",
                )));
            }
            Event::BasicInfo | Event::FrameHeader | Event::Complete => {}
        }
    }

    // The crate either converts to sRGB or emits simple PQ values its
    // API does not distinguish, so jxl pages ride the sRGB path.
    Decoded::rgba(
        width,
        height,
        Format::Jxl,
        Transfer::Srgb,
        rgba.into_boxed_slice(),
    )
    .union()
}

fn threads() -> usize {
    use std::num::NonZeroUsize;
    use std::thread;

    thread::available_parallelism()
        .map_or(1, NonZeroUsize::get)
        .min(MAX_THREADS)
}

/// Scoped-thread runner. A failing task fails the whole run, which is
/// how the crate's driver expects a broken codestream to be reported.
/// The driver invokes this only with two or more threads, so no
/// sequential path is needed.
struct Pool(usize);

impl JxlParallelRunner for Pool {
    fn run(
        &mut self,
        num: usize,
        fun: &JxlParallelRunnerFun<'_>,
    ) -> JxlResult<()> {
        use std::sync::OnceLock;
        use std::thread;

        let thread_count = self.0.min(num);
        let chunk = num.div_ceil(thread_count);
        let failure: OnceLock<JxlError> = OnceLock::new();
        thread::scope(|scope| {
            for thread_index in 0..thread_count {
                let fun: &JxlParallelRunnerFun = fun;
                let failure = &failure;
                scope.spawn(move || {
                    for task_index in thread_index * chunk
                        ..((thread_index + 1) * chunk).min(num)
                    {
                        if let Err(task_error) = fun(task_index) {
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
    fn num_threads(&self) -> usize {
        self.0
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::Pool;

    use jxl::api::JxlParallelRunner as _;
    use jxl::error::Error as JxlError;

    // The fixtures are single-group, so only this test exercises the
    // runner's parallel path.
    #[test]
    fn pool_reports_the_first_task_error() {
        let mut pool = Pool(4);
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

        // Two threads over four tasks pin the chunk partition. A range
        // that drops the last task would leave the failure unreported.
        let mut two_threads = Pool(2);
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
    }
}
