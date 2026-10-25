"""
AeroLoss AI — Independent Model 1 and Model 2 Verification Test Suite
Target: Validate separate execution of Model 1 (Guardrail) and Model 2 (YOLOv8 Detection)
"""

import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model1.image_quality_validator import ImageQualityValidator, validate_inspection_image
from ultralytics import YOLO

CHECKPOINT_PATH = REPO_ROOT / "models" / "yolov8s_blade_damage_best.pt"
CANONICAL_CLASSES = ["corrosion", "crack", "craze", "hide_craze", "surface_injure", "thunderstrike"]


class TestIndependentModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_img_dir = REPO_ROOT / "data" / "raw" / "images"
        cls.neg_img_dir = REPO_ROOT / "data" / "negative_evaluation_samples"
        cls.validator = ImageQualityValidator()
        cls.checkpoint_path = CHECKPOINT_PATH
        
        # Verify checkpoint exists
        cls.checkpoint_exists = cls.checkpoint_path.exists()
        if cls.checkpoint_exists:
            cls.detector = YOLO(str(cls.checkpoint_path))

    def test_checkpoint_verification(self):
        """Verify newly trained checkpoint exists and contains 6 canonical classes."""
        self.assertTrue(self.checkpoint_exists, f"Checkpoint missing at {self.checkpoint_path}")
        self.assertGreater(self.checkpoint_path.stat().st_size, 20 * 1024 * 1024, "Checkpoint size too small")
        
        class_mapping = list(self.detector.names.values())
        self.assertEqual(len(class_mapping), 6, "Expected exactly 6 classes in checkpoint")
        self.assertEqual(class_mapping, CANONICAL_CLASSES, "Checkpoint class mapping mismatch")

    def test_model1_independent_valid_blade(self):
        """Model 1 test: valid blade image evaluated without invoking YOLOv8."""
        test_img = self.raw_img_dir / "0.jpg"
        self.assertTrue(test_img.exists(), "Sample 0.jpg not found")
        
        res = self.validator.validate(test_img)
        self.assertEqual(res["decode_status"], "OK")
        self.assertEqual(res["quality_decision"], "USABLE")
        self.assertEqual(res["relevance_decision"], "BLADE")
        self.assertFalse(res["requires_manual_review"])
        self.assertIn("dimensions", res)
        self.assertIn("metrics", res)
        # Ensure YOLOv8 detection keys are strictly absent from Model 1 output
        self.assertNotIn("detections", res)
        self.assertNotIn("confidence", res)

    def test_model1_independent_non_blade_chart(self):
        """Model 1 test: non-blade digital chart evaluated independently."""
        test_img = self.neg_img_dir / "chart_confusion_matrix_03.png"
        if not test_img.exists():
            self.skipTest(f"Sample {test_img} not found")
            
        res = self.validator.validate(test_img)
        self.assertEqual(res["decode_status"], "OK")
        self.assertEqual(res["relevance_decision"], "NON_BLADE")
        self.assertIn("dimensions", res)

    def test_model1_independent_noise_image(self):
        """Model 1 test: synthetic noise rejected as UNUSABLE."""
        test_img = self.neg_img_dir / "synthetic_gaussian_noise.jpg"
        if not test_img.exists():
            self.skipTest(f"Sample {test_img} not found")
            
        res = self.validator.validate(test_img)
        self.assertEqual(res["decode_status"], "OK")
        self.assertEqual(res["quality_decision"], "UNUSABLE")

    def test_model2_independent_valid_blade_detections(self):
        """Model 2 test: YOLOv8 executed directly without applying Model 1 guardrail."""
        if not self.checkpoint_exists:
            self.skipTest("Checkpoint missing")
            
        test_img = self.raw_img_dir / "0.jpg"
        results = self.detector(str(test_img), conf=0.20, verbose=False)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        
        boxes = results[0].boxes
        # Verify bounding box attributes
        self.assertGreaterEqual(len(boxes), 1, "Expected detections on primary blade 0.jpg")
        for b in boxes:
            cls_id = int(b.cls[0])
            self.assertIn(cls_id, range(6))
            conf = float(b.conf[0])
            self.assertGreaterEqual(conf, 0.20)
            xyxy = b.xyxy[0].tolist()
            self.assertEqual(len(xyxy), 4)

    def test_model2_independent_non_blade_image(self):
        """Model 2 test: YOLOv8 executed on non-blade image without prior gating."""
        if not self.checkpoint_exists:
            self.skipTest("Checkpoint missing")
            
        test_img = self.neg_img_dir / "chart_confusion_matrix_03.png"
        if not test_img.exists():
            self.skipTest(f"Sample {test_img} not found")
            
        # Model 2 runs without Model 1 filtering
        results = self.detector(str(test_img), conf=0.25, verbose=False)
        self.assertIsNotNone(results)
        # Bounding boxes should simply be returned as-is (often 0 on clean charts)
        boxes = results[0].boxes
        self.assertIsInstance(len(boxes), int)

    def test_model2_handles_invalid_file_safely(self):
        """Model 2 test: handles non-existent or corrupted file gracefully."""
        if not self.checkpoint_exists:
            self.skipTest("Checkpoint missing")
            
        with self.assertRaises(Exception):
            self.detector("non_existent_image_12345.jpg", verbose=False)


if __name__ == "__main__":
    unittest.main()
