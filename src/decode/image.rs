use eros::ErrorUnion;
use eros::IntoUnion as _;

use crate::fault::DecodeError;
use crate::fault::DecodeRefusal;

use super::check_dims;
use super::Decoded;
use super::Format;
use super::Transfer;

/// image-crate path: decode with a concrete decoder so EXIF orientation is
/// readable, then apply it.
#[expect(
    clippy::wildcard_enum_match_arm,
    reason = "image's other ImageFormat variants never arrive: decode dispatches only the six formats the crate enables"
)]
pub(super) fn decode_image(
    bytes: &[u8],
    image_format: image::ImageFormat,
) -> Result<Decoded, DecodeError> {
    use image::ImageDecoder as _;
    use std::io::Cursor;

    let format = match image_format {
        image::ImageFormat::Jpeg => Format::Jpeg,
        image::ImageFormat::Png => Format::Png,
        image::ImageFormat::Gif => Format::Gif,
        image::ImageFormat::WebP => Format::Webp,
        image::ImageFormat::Bmp => Format::Bmp,
        image::ImageFormat::Tiff => Format::Tiff,
        _ => {
            return Err(ErrorUnion::new(DecodeRefusal::Unsupported(
                "internal dispatch",
            )))
        }
    };
    let reader =
        image::ImageReader::with_format(Cursor::new(bytes), image_format);
    let mut decoder = reader.into_decoder().union()?;
    // Orientation only swaps w and h, and the cap checks both, so this
    // pre-orientation check catches rotated pages too.
    let (width, height) = decoder.dimensions();
    check_dims(width, height).union()?;
    let orientation = decoder.orientation().union()?;
    let mut oriented =
        image::DynamicImage::from_decoder(decoder).union()?;
    oriented.apply_orientation(orientation);
    let rgba = oriented.into_rgba8();
    Decoded::rgba(
        rgba.width(),
        rgba.height(),
        format,
        Transfer::Srgb,
        rgba.into_raw().into_boxed_slice(),
    )
    .union()
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::super::decode;
    use super::super::Pixels;
    use super::*;

    use crate::testkit;

    // The encoder cannot write EXIF orientation tags, so orientation
    // handling has no testable fixture.
    #[test]
    fn encoder_roundtrips() {
        let png = testkit::tiny_png(4, 3, [10, 20, 30, 255]);
        let png_page = decode(&png, "p.png").expect("png decodes");
        assert_eq!((png_page.width, png_page.height), (4, 3));
        assert_eq!(png_page.format, Format::Png);
        match png_page.pixels {
            Pixels::Rgba8(bytes) => {
                assert_eq!(bytes.len(), 4 * 4 * 3);
                assert_eq!(bytes.get(..4), Some(&[10, 20, 30, 255][..]));
            }
            Pixels::Yuv(_) => panic!("png must decode to rgba"),
        }

        let jpeg = testkit::tiny_jpeg(5, 2, [200, 100, 50]);
        let jpeg_page = decode(&jpeg, "p.jpg").expect("jpeg decodes");
        assert_eq!(jpeg_page.format, Format::Jpeg);
        assert_eq!((jpeg_page.width, jpeg_page.height), (5, 2));

        let gif = testkit::tiny_gif(3, 2, [200, 100, 50]);
        let gif_page = decode(&gif, "p.gif").expect("gif decodes");
        assert_eq!(gif_page.format, Format::Gif);
        assert_eq!((gif_page.width, gif_page.height), (3, 2));

        let bmp = testkit::tiny_bmp(2, 3, [200, 100, 50]);
        let bmp_page = decode(&bmp, "p.bmp").expect("bmp decodes");
        assert_eq!(bmp_page.format, Format::Bmp);
        assert_eq!((bmp_page.width, bmp_page.height), (2, 3));

        let tiff = testkit::tiny_tiff(3, 1, [200, 100, 50]);
        let tiff_page = decode(&tiff, "p.tiff").expect("tiff decodes");
        assert_eq!(tiff_page.format, Format::Tiff);
        assert_eq!((tiff_page.width, tiff_page.height), (3, 1));
    }
}
