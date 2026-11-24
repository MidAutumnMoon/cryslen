use super::Decoded;
use super::Format;

/// Decode with a concrete decoder so EXIF orientation is readable, then
/// apply it.
pub(super) fn decode(
    bytes: &[u8],
    format: Format,
) -> eros::Result<Decoded> {
    use super::Depth;
    use super::Plane;
    use super::Transfer;
    use image::DynamicImage;
    use image::ImageDecoder as _;
    use image::ImageFormat;
    use image::ImageReader;
    use std::io::Cursor;

    let image_format = match format {
        Format::Jpeg => ImageFormat::Jpeg,
        Format::Png => ImageFormat::Png,
        Format::Gif => ImageFormat::Gif,
        Format::Bmp => ImageFormat::Bmp,
        Format::Tiff => ImageFormat::Tiff,
        Format::Avif | Format::Jxl | Format::Webp => {
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
    Ok(if let DynamicImage::ImageLuma8(gray) = oriented {
        Decoded::mono(
            gray.width(),
            gray.height(),
            format,
            Transfer::Srgb,
            Depth::D8,
            Plane::U8(gray.into_raw().into_boxed_slice()),
        )?
    } else if let DynamicImage::ImageLuma16(gray) = oriented {
        Decoded::mono(
            gray.width(),
            gray.height(),
            format,
            Transfer::Srgb,
            Depth::D16,
            Plane::U16(gray.into_raw().into_boxed_slice()),
        )?
    } else {
        let rgba = oriented.into_rgba8();
        Decoded::rgba(
            rgba.width(),
            rgba.height(),
            format,
            Transfer::Srgb,
            rgba.into_raw().into_boxed_slice(),
        )?
    })
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

    // The encoder cannot write EXIF orientation tags, so orientation
    // handling has no testable fixture.
    #[test]
    fn encoder_roundtrips() {
        use crate::testkit;

        let png = testkit::tiny_png(4, 3, [10, 20, 30, 255]);
        let png_page = decode::decode(&png, "p.png").expect("png decodes");
        assert_eq!((png_page.width(), png_page.height()), (4, 3));
        assert_eq!(png_page.format(), Format::Png);
        match png_page.pixels() {
            Pixels::Rgba8(bytes) => {
                assert_eq!(bytes.len(), 4 * 4 * 3);
                assert_eq!(bytes.get(..4), Some(&[10, 20, 30, 255][..]));
            }
            Pixels::Yuv(_) => panic!("png must decode to rgba"),
        }

        let jpeg = testkit::tiny_jpeg(5, 2, [200, 100, 50]);
        let jpeg_page =
            decode::decode(&jpeg, "p.jpg").expect("jpeg decodes");
        assert_eq!(jpeg_page.format(), Format::Jpeg);
        assert_eq!((jpeg_page.width(), jpeg_page.height()), (5, 2));

        let gif = testkit::tiny_gif(3, 2, [200, 100, 50]);
        let gif_page = decode::decode(&gif, "p.gif").expect("gif decodes");
        assert_eq!(gif_page.format(), Format::Gif);
        assert_eq!((gif_page.width(), gif_page.height()), (3, 2));

        let bmp = testkit::tiny_bmp(2, 3, [200, 100, 50]);
        let bmp_page = decode::decode(&bmp, "p.bmp").expect("bmp decodes");
        assert_eq!(bmp_page.format(), Format::Bmp);
        assert_eq!((bmp_page.width(), bmp_page.height()), (2, 3));

        let tiff = testkit::tiny_tiff(3, 1, [200, 100, 50]);
        let tiff_page =
            decode::decode(&tiff, "p.tiff").expect("tiff decodes");
        assert_eq!(tiff_page.format(), Format::Tiff);
        assert_eq!((tiff_page.width(), tiff_page.height()), (3, 1));
    }

    #[test]
    fn opaque_grayscale_png_decodes_to_mono() {
        use crate::decode::Depth;
        use crate::decode::Plane;
        use crate::decode::Subsampling;
        use image::DynamicImage;
        use image::GrayImage;
        use image::ImageBuffer;
        use image::ImageFormat;
        use image::Luma;
        use std::io::Cursor;

        let mut png8 = Vec::new();
        let gray8 =
            GrayImage::from_raw(3, 2, vec![0, 40, 80, 120, 160, 200])
                .expect("gray8 dims");
        DynamicImage::ImageLuma8(gray8)
            .write_to(&mut Cursor::new(&mut png8), ImageFormat::Png)
            .expect("encode 8-bit grayscale png");
        let page8 =
            decode::decode(&png8, "g8.png").expect("luma8 decodes");
        let Pixels::Yuv(yuv8) = page8.pixels() else {
            panic!("opaque luma8 png must decode to mono");
        };
        assert_eq!(
            (yuv8.spec.depth, yuv8.spec.subsampling),
            (Depth::D8, Subsampling::Mono)
        );
        assert!(matches!(
            &yuv8.luma,
            Plane::U8(luma) if **luma == [0, 40, 80, 120, 160, 200]
        ));

        let mut png16 = Vec::new();
        let gray16 = ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(
            2,
            2,
            vec![0, 15000, 30000, 45000],
        )
        .expect("gray16 dims");
        DynamicImage::ImageLuma16(gray16)
            .write_to(&mut Cursor::new(&mut png16), ImageFormat::Png)
            .expect("encode 16-bit grayscale png");
        let page16 =
            decode::decode(&png16, "g16.png").expect("luma16 decodes");
        let Pixels::Yuv(yuv16) = page16.pixels() else {
            panic!("opaque luma16 png must decode to mono");
        };
        assert_eq!(
            (yuv16.spec.depth, yuv16.spec.subsampling),
            (Depth::D16, Subsampling::Mono)
        );
        assert!(matches!(
            &yuv16.luma,
            Plane::U16(luma) if **luma == [0, 15000, 30000, 45000]
        ));
    }
}
