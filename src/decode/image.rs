use super::Decoded;
use super::Format;
use super::Transfer;

/// Decode with a concrete decoder so EXIF orientation is readable, then
/// apply it.
pub(super) fn decode(bytes: &[u8], format: Format) -> eros::Result<Decoded> {
    use image::DynamicImage;
    use image::ImageDecoder as _;
    use image::ImageFormat;
    use image::ImageReader;
    use std::io::Cursor;

    let image_format = match format {
        Format::Jpeg => ImageFormat::Jpeg,
        Format::Png => ImageFormat::Png,
        Format::Gif => ImageFormat::Gif,
        Format::Webp => ImageFormat::WebP,
        Format::Bmp => ImageFormat::Bmp,
        Format::Tiff => ImageFormat::Tiff,
        Format::Avif | Format::Jxl => {
            eros::bail!("unsupported internal dispatch");
        }
    };
    let reader =
        ImageReader::with_format(Cursor::new(bytes), image_format);
    let mut decoder = reader.into_decoder()?;
    // Orientation only swaps w and h, and the cap checks both, so this
    // pre-orientation check catches rotated pages too.
    let (width, height) = decoder.dimensions();
    super::admit_dims(width, height, "bad dimensions")?;
    let orientation = decoder.orientation()?;
    let mut oriented = DynamicImage::from_decoder(decoder)?;
    oriented.apply_orientation(orientation);
    let rgba = oriented.into_rgba8();
    Ok(Decoded::rgba(
        rgba.width(),
        rgba.height(),
        format,
        Transfer::Srgb,
        rgba.into_raw().into_boxed_slice(),
    )?)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use super::*;

    use crate::decode;
    use crate::decode::Pixels;
    use crate::testkit;

    // The encoder cannot write EXIF orientation tags, so orientation
    // handling has no testable fixture.
    #[test]
    fn encoder_roundtrips() {
        let png = testkit::tiny_png(4, 3, [10, 20, 30, 255]);
        let png_page = decode::decode(&png, "p.png").expect("png decodes");
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
        let jpeg_page =
            decode::decode(&jpeg, "p.jpg").expect("jpeg decodes");
        assert_eq!(jpeg_page.format, Format::Jpeg);
        assert_eq!((jpeg_page.width, jpeg_page.height), (5, 2));

        let gif = testkit::tiny_gif(3, 2, [200, 100, 50]);
        let gif_page = decode::decode(&gif, "p.gif").expect("gif decodes");
        assert_eq!(gif_page.format, Format::Gif);
        assert_eq!((gif_page.width, gif_page.height), (3, 2));

        let bmp = testkit::tiny_bmp(2, 3, [200, 100, 50]);
        let bmp_page = decode::decode(&bmp, "p.bmp").expect("bmp decodes");
        assert_eq!(bmp_page.format, Format::Bmp);
        assert_eq!((bmp_page.width, bmp_page.height), (2, 3));

        let tiff = testkit::tiny_tiff(3, 1, [200, 100, 50]);
        let tiff_page =
            decode::decode(&tiff, "p.tiff").expect("tiff decodes");
        assert_eq!(tiff_page.format, Format::Tiff);
        assert_eq!((tiff_page.width, tiff_page.height), (3, 1));
    }
}
