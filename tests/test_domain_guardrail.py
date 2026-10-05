import os
import sys
import unittest
from PIL import Image, ImageDraw
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


from src.pipeline import AeroLossPipeline

class TestDomainGuardrail(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = AeroLossPipeline()

    def test_authentic_blade_image_passes(self):
        """Verifies that an authentic wind turbine blade inspection photo is accepted."""
        real_patch = os.path.join(BASE_DIR, "data", "processed", "patches", "crack", "patch_00058.jpg")
        self.assertTrue(os.path.exists(real_patch), f"Missing patch file: {real_patch}")

        res = self.pipeline.analyze_image(real_patch)
        self.assertTrue(res.get('is_valid_blade_image'), "Real blade image was incorrectly rejected!")
        self.assertEqual(res.get('status'), 'SUCCESS')
        self.assertIn('visual_detection', res)
        self.assertIn('physics_aerodynamics', res)

    def test_dog_or_colorful_non_blade_image_rejected(self):
        """Verifies that high-saturation / unfamiliar objects (dog, nature, colorful patterns) are intercepted and rejected."""
        # Create non-blade colorful image simulating an animal / household scene
        non_blade = Image.new('RGB', (224, 224), color=(255, 69, 0)) # Vivid orange-red
        draw = ImageDraw.Draw(non_blade)
        draw.ellipse([40, 40, 180, 180], fill=(50, 205, 50)) # Bright lime green
        draw.rectangle([10, 10, 90, 90], fill=(30, 144, 255)) # Dodger blue

        res = self.pipeline.analyze_image(non_blade)
        self.assertFalse(res.get('is_valid_blade_image'), "Non-blade image was incorrectly accepted!")
        self.assertEqual(res.get('status'), 'REJECTED_OUT_OF_DOMAIN')
        self.assertIn('rejection_reason', res)
        self.assertNotIn('physics_aerodynamics', res, "Physics engine should NOT run on invalid images!")

if __name__ == '__main__':
    unittest.main()
