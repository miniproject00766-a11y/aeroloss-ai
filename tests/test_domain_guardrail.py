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

    def test_multiple_authentic_blade_images_pass(self):
        """Verifies that authentic wind turbine blade inspection photos are accepted (0% false rejection)."""
        raw_dir = os.path.join(BASE_DIR, "data", "raw", "images")
        sample_files = ['1.jpg', '1001.jpg', '1013.jpg', '0.jpg', '10.jpg']
        for fname in sample_files:
            fpath = os.path.join(raw_dir, fname)
            if os.path.exists(fpath):
                res = self.pipeline.analyze_image(fpath)
                self.assertTrue(res.get('is_valid_blade_image'), f"Authentic turbine photo '{fname}' was falsely rejected: {res.get('rejection_reason')}")
                self.assertEqual(res.get('status'), 'SUCCESS')

    def test_dog_or_colorful_non_blade_image_rejected(self):
        """Verifies that synthetic non-blade objects (colorful patterns, household scenes) are intercepted and rejected."""
        non_blade = Image.new('RGB', (224, 224), color=(255, 69, 0)) # Vivid orange-red
        draw = ImageDraw.Draw(non_blade)
        draw.ellipse([40, 40, 180, 180], fill=(50, 205, 50)) # Bright lime green
        draw.rectangle([10, 10, 90, 90], fill=(30, 144, 255)) # Dodger blue

        res = self.pipeline.analyze_image(non_blade)
        self.assertFalse(res.get('is_valid_blade_image'), "Non-blade image was incorrectly accepted!")
        self.assertEqual(res.get('status'), 'REJECTED_OUT_OF_DOMAIN')
        self.assertIn('rejection_reason', res)
        self.assertNotIn('physics_aerodynamics', res, "Physics engine should NOT run on invalid images!")

    def test_cracked_and_yellow_tip_blades_pass_food_filter(self):
        """Verifies that authentic blades with cracks, dark fissures, or yellow hazard tips are not falsely rejected as food."""
        samples = [
            os.path.join(BASE_DIR, "data", "raw", "images", "860.jpg"),
            os.path.join(BASE_DIR, "test_uploads", "01_drone_blade_erosion_crack.jpg"),
        ]
        for fpath in samples:
            if os.path.exists(fpath):
                res = self.pipeline.analyze_image(fpath)
                self.assertTrue(res.get('is_valid_blade_image'), f"Blade {fpath} was falsely rejected: {res.get('rejection_reason')}")
                self.assertNotEqual(res.get('status'), 'REJECTED_OUT_OF_DOMAIN')

    def test_genuine_food_and_ood_images_strictly_rejected(self):
        """Verifies that genuine food produce and animals are rejected by domain guardrail."""
        ood_samples = [
            os.path.join(BASE_DIR, "test_uploads", "ood_test_suite", "pizza.jpg"),
            os.path.join(BASE_DIR, "test_uploads", "ood_test_suite", "bird.jpg"),
        ]
        for fpath in ood_samples:
            if os.path.exists(fpath):
                res = self.pipeline.analyze_image(fpath)
                self.assertFalse(res.get('is_valid_blade_image'), f"OOD file {fpath} was incorrectly accepted!")
                self.assertEqual(res.get('status'), 'REJECTED_OUT_OF_DOMAIN')

if __name__ == '__main__':
    unittest.main()
