//! Raster still decoders via the `image` crate (JPEG, PNG, BMP, TIFF).

use crate::image::pixels;
use crate::image::pixels::Format;
use crate::image::pixels::Image;

/// Decode with a concrete decoder so EXIF orientation is readable, then
/// apply it.
pub(super) fn decode(bytes: &[u8], format: Format) -> eros::Result<Image> {
    use std::io::Cursor;

    use image::DynamicImage;
    use image::ImageDecoder as _;
    use image::ImageFormat;
    use image::ImageReader;

    use crate::image::pixels::Transfer;
    use crate::image::pixels::WideDepth;

    let image_format = match format {
        Format::Jpeg => ImageFormat::Jpeg,
        Format::Png => ImageFormat::Png,
        Format::Bmp => ImageFormat::Bmp,
        Format::Tiff => ImageFormat::Tiff,
        Format::Gif | Format::Avif | Format::Jxl | Format::Webp => {
            eros::bail!("unsupported internal dispatch");
        }
    };
    let reader =
        ImageReader::with_format(Cursor::new(bytes), image_format);
    let mut decoder = reader.into_decoder()?;
    // Orientation only swaps width and height, and the cap checks both, so
    // this pre-orientation check catches rotated pages too.
    let (width, height) = decoder.dimensions();
    pixels::admit_dims(width, height, "bad dimensions")?;
    let orientation = decoder.orientation()?;
    let mut oriented = DynamicImage::from_decoder(decoder)?;
    oriented.apply_orientation(orientation);
    Ok(if let DynamicImage::ImageLuma8(gray) = oriented {
        Image::gray_u8(
            gray.width(),
            gray.height(),
            format,
            Transfer::Srgb,
            gray.into_raw().into_boxed_slice(),
        )?
    } else if let DynamicImage::ImageLuma16(gray) = oriented {
        Image::gray_u16(
            gray.width(),
            gray.height(),
            format,
            Transfer::Srgb,
            WideDepth::D16,
            gray.into_raw().into_boxed_slice(),
        )?
    } else {
        let rgba = oriented.into_rgba8();
        Image::rgba(
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
    use crate::image::decode::codec;
    use crate::image::pixels::Pixels;
    use crate::image::pixels::SamplePlanes;

    #[test]
    fn encoder_roundtrips() {
        use crate::testkit;

        let png = testkit::tiny_png(4, 3, [10, 20, 30, 255]);
        let png_page = codec::decode(&png, "p.png").expect("png decodes");
        assert_eq!((png_page.width(), png_page.height()), (4, 3));
        assert_eq!(png_page.format(), Format::Png);
        match png_page.pixels() {
            Pixels::Rgba8(bytes) => {
                assert_eq!(bytes.len(), 4 * 4 * 3);
                assert_eq!(bytes.get(..4), Some(&[10, 20, 30, 255][..]));
            }
            Pixels::Gray { .. } | Pixels::Yuv { .. } => {
                panic!("png must decode to rgba")
            }
        }

        let jpeg = testkit::tiny_jpeg(5, 2, [200, 100, 50]);
        let jpeg_page =
            codec::decode(&jpeg, "p.jpg").expect("jpeg decodes");
        assert_eq!(jpeg_page.format(), Format::Jpeg);
        assert_eq!((jpeg_page.width(), jpeg_page.height()), (5, 2));

        let gif = testkit::tiny_gif(3, 2, [200, 100, 50]);
        let gif_page = codec::decode(&gif, "p.gif").expect("gif decodes");
        assert_eq!(gif_page.format(), Format::Gif);
        assert_eq!((gif_page.width(), gif_page.height()), (3, 2));

        let bmp = testkit::tiny_bmp(2, 3, [200, 100, 50]);
        let bmp_page = codec::decode(&bmp, "p.bmp").expect("bmp decodes");
        assert_eq!(bmp_page.format(), Format::Bmp);
        assert_eq!((bmp_page.width(), bmp_page.height()), (2, 3));

        let tiff = testkit::tiny_tiff(3, 1, [200, 100, 50]);
        let tiff_page =
            codec::decode(&tiff, "p.tiff").expect("tiff decodes");
        assert_eq!(tiff_page.format(), Format::Tiff);
        assert_eq!((tiff_page.width(), tiff_page.height()), (3, 1));
    }

    #[test]
    fn opaque_grayscale_png_decodes_to_gray() {
        use std::io::Cursor;

        use image::DynamicImage;
        use image::GrayImage;
        use image::ImageBuffer;
        use image::ImageFormat;
        use image::Luma;

        {
            let mut png = Vec::new();
            let image =
                GrayImage::from_raw(3, 2, vec![0, 40, 80, 120, 160, 200])
                    .expect("gray8 dims");
            DynamicImage::ImageLuma8(image)
                .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
                .expect("encode 8-bit grayscale png");
            let page =
                codec::decode(&png, "g8.png").expect("luma8 decodes");
            let Pixels::Gray {
                full_range,
                planes:
                    SamplePlanes::U8 {
                        planes: [luma],
                        alpha,
                    },
            } = page.pixels()
            else {
                panic!("opaque luma8 png must decode to U8 gray");
            };
            assert!(*full_range);
            assert!(alpha.is_none());
            assert_eq!(&**luma, [0, 40, 80, 120, 160, 200]);
        }

        {
            use crate::image::pixels::WideDepth;

            let mut png = Vec::new();
            let image = ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(
                2,
                2,
                vec![0, 15000, 30000, 45000],
            )
            .expect("gray16 dims");
            DynamicImage::ImageLuma16(image)
                .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
                .expect("encode 16-bit grayscale png");

            let page =
                codec::decode(&png, "g16.png").expect("luma16 decodes");
            let Pixels::Gray {
                full_range,
                planes:
                    SamplePlanes::U16 {
                        depth,
                        planes: [luma],
                        alpha,
                    },
            } = page.pixels()
            else {
                panic!("opaque luma16 png must decode to U16 gray");
            };
            assert!(*full_range);
            assert_eq!(*depth, WideDepth::D16);
            assert!(alpha.is_none());
            assert_eq!(&**luma, [0, 15000, 30000, 45000]);
        }
    }
}
