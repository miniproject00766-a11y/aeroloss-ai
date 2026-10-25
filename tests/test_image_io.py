import io
import unittest
from unittest.mock import patch

from PIL import Image, UnidentifiedImageError

from src.image_io import load_rgb_image


class TestImageIO(unittest.TestCase):
    def encode(self, image, image_format):
        buffer = io.BytesIO()
        image.save(buffer, format=image_format)
        return buffer.getvalue()

    def test_supported_formats_return_rgb_images(self):
        image = Image.new('RGB', (8, 6), color='white')
        for image_format in ('JPEG', 'PNG', 'WEBP'):
            with self.subTest(image_format=image_format):
                decoded = load_rgb_image(self.encode(image, image_format))
                self.assertEqual(decoded.mode, 'RGB')
                self.assertEqual(decoded.size, (8, 6))

    def test_rejects_empty_and_corrupt_bytes(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            load_rgb_image(b'')
        with self.assertRaises(UnidentifiedImageError):
            load_rgb_image(b'not an image')

    def test_rejects_unsupported_format(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported image format'):
            load_rgb_image(self.encode(Image.new('RGB', (2, 2)), 'TIFF'))

    def test_rejects_compressed_size_over_limit(self):
        with patch('src.image_io.MAX_IMAGE_BYTES', 4):
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                load_rgb_image(b'12345')

    def test_rejects_pixel_count_over_limit(self):
        image = Image.new('RGB', (4, 3), color='white')
        with patch('src.image_io.MAX_IMAGE_PIXELS', 11):
            with self.assertRaisesRegex(ValueError, '40-megapixel'):
                load_rgb_image(self.encode(image, 'PNG'))

    def test_applies_exif_orientation(self):
        image = Image.new('RGB', (2, 3), color='white')
        exif = Image.Exif()
        exif[274] = 6
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', exif=exif)

        decoded = load_rgb_image(buffer.getvalue())
        self.assertEqual(decoded.size, (3, 2))


if __name__ == '__main__':
    unittest.main()
