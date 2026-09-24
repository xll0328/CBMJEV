import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageOps

from cbmjev.runtime import load_payload


class PayloadCompressionTests(unittest.TestCase):
    def test_lossless_equivalence_including_orientation_and_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            for mode in ("RGB", "RGBA", "L", "P"):
                image = Image.new(mode, (37, 23))
                for x in range(37):
                    for y in range(23):
                        v = (x * 17 + y * 13) % 256
                        image.putpixel((x, y), ((v, 255-v, x*y % 256, 123)
                                                if mode == "RGBA" else
                                                (v, 255-v, x*y % 256) if mode == "RGB" else v))
                exif = Image.Exif()
                exif[274] = 6
                path = Path(folder) / (mode + ".png")
                image.save(path, exif=exif)
                record = {"modality": "image", "image_paths": [path.name]}
                fast = load_payload(record, folder)
                legacy = load_payload(record, folder, png_compress_level=6)
                self.assertNotEqual(fast.images, legacy.images)
                with Image.open(path) as source:
                    expected = ImageOps.exif_transpose(source).convert("RGB")
                    for payload in (fast, legacy):
                        with Image.open(io.BytesIO(payload.images[0])) as actual:
                            self.assertEqual(actual.mode, "RGB")
                            self.assertEqual(actual.size, expected.size)
                            self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_bounds_and_path_safety_remain_enforced(self):
        record = {"modality": "image", "image_paths": ["../escape.png"]}
        for value in (-1, 10, True, .5):
            with self.assertRaises(ValueError):
                load_payload(record, ".", png_compress_level=value)
        with self.assertRaises(ValueError):
            load_payload(record, ".")


if __name__ == "__main__":
    unittest.main()
