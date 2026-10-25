"""
AeroLoss AI — Integrated Model 1 & YOLOv8 Detection Pipeline Test Suite
Target: src.pipeline.aeroloss_detection_pipeline

Tests:
1. Valid blade image passing guardrail and invoking YOLOv8 detector.
2. Technically unusable image (blackout/blank) stopped at Stage 1, detector skipped.
3. Out-of-domain non-blade image (digital chart) stopped at Stage 1, detector skipped.
4. Ambiguous/uncertain image (pasture cow with distant turbine) routed to review state,
   detector skipped by default to prevent treating unverified imagery as valid blade data.
5. Review passthrough flag allows provisional inference with explicit warnings.
6. Graceful behavior when detector weights are unavailable (no crash, explicit status).
7. Clean blade image handling (valid blade, zero defects detected).
8. Strict separation of quality and relevance dimensions in structured response.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.aeroloss_detection_pipeline import (
    AeroLossDetectionPipeline,
    run_aeroloss_detection_pipeline,
    CANONICAL_DAMAGE_CLASSES
)


class TestAeroLossDetectionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = AeroLossDetectionPipeline()
        cls.raw_img_dir = REPO_ROOT / "data" / "raw" / "images"
        cls.neg_img_dir = REPO_ROOT / "data" / "negative_evaluation_samples"

    def test_pipeline_initialization(self):
        """Test pipeline initialization and class name definitions."""
        self.assertIsNotNone(self.pipeline)
        self.assertEqual(len(self.pipeline.class_names), 6)
        self.assertIn("corrosion", self.pipeline.class_names)
        self.assertIn("surface_injure", self.pipeline.class_names)

    def test_valid_blade_image_with_detections(self):
        """Test 1: Valid primary blade image passes guardrail and executes YOLO detection."""
        test_img = self.raw_img_dir / "0.jpg"
        self.assertTrue(test_img.exists(), f"Image not found: {test_img}")

        res = self.pipeline.predict(test_img)

        # Dimension separation assertions
        self.assertEqual(res["quality_decision"], "USABLE")
        self.assertEqual(res["relevance_decision"], "BLADE")
        self.assertFalse(res["requires_manual_review"])

        # Pipeline gating assertions
        self.assertEqual(res["pipeline_status"], "PASSED_GUARDRAIL")
        if self.pipeline.detector_loaded:
            self.assertEqual(res["damage_detection_status"], "SUCCESS")
            self.assertGreater(res["detections_count"], 0)
            det = res["detections"][0]
            self.assertIn("class_name", det)
            self.assertIn("confidence", det)
            self.assertIn("bbox_xyxy", det)
            self.assertIn("area_fraction", det)
            self.assertGreater(det["confidence"], 0.20)
        else:
            self.assertEqual(res["damage_detection_status"], "DETECTOR_MODEL_UNAVAILABLE")

    def test_technically_unusable_image_stopped(self):
        """Test 2: Technically unusable image (severe blur) is stopped at Stage 1; detector is skipped."""
        import cv2
        test_img = self.raw_img_dir / "0.jpg"
        blade_bgr = cv2.imread(str(test_img))
        # Synthesize heavy optical defocus / motion blur on genuine blade
        blurred_blade = cv2.GaussianBlur(blade_bgr, (65, 65), sigmaX=30)

        res = self.pipeline.predict(blurred_blade)

        self.assertEqual(res["quality_decision"], "UNUSABLE")
        self.assertEqual(res["pipeline_status"], "STOPPED_UNUSABLE_QUALITY")
        self.assertEqual(res["damage_detection_status"], "SKIPPED_QUALITY_DEFECT")
        self.assertEqual(len(res["detections"]), 0)
        self.assertIn("replacement inspection capture is required", res["summary"])

    def test_clear_non_blade_image_stopped(self):
        """Test 3: Clear non-blade image (digital chart) is rejected at Stage 1; detector skipped."""
        chart_path = self.neg_img_dir / "chart_airfoil_polar_02.png"
        if not chart_path.exists():
            self.skipTest("Negative sample chart_airfoil_polar_02.png not found")

        res = self.pipeline.predict(chart_path)

        self.assertEqual(res["relevance_decision"], "NON_BLADE")
        self.assertEqual(res["pipeline_status"], "STOPPED_NON_BLADE_DOMAIN")
        self.assertEqual(res["damage_detection_status"], "SKIPPED_DOMAIN_REJECTION")
        self.assertEqual(len(res["detections"]), 0)
        self.assertIn("NON_BLADE", res["summary"])

    def test_ambiguous_image_routes_to_manual_review(self):
        """Test 4: Ambiguous pasture cow with distant turbine routes to review, not silent pass."""
        cow_path = self.neg_img_dir / "ambiguous_pasture_cow_distant_turbine_01.jpg"
        if not cow_path.exists():
            self.skipTest("Negative sample ambiguous_pasture_cow_distant_turbine_01.jpg not found")

        # Default safe mode: allow_review_passthrough=False
        res = self.pipeline.predict(cow_path, allow_review_passthrough=False)

        self.assertTrue(res["requires_manual_review"])
        self.assertEqual(res["pipeline_status"], "REVIEW_REQUIRED")
        self.assertEqual(res["damage_detection_status"], "PENDING_OPERATOR_REVIEW")
        self.assertEqual(len(res["detections"]), 0)
        self.assertIn("manual review", res["summary"].lower())

    def test_review_passthrough_attaches_prominent_warning(self):
        """Test 5: Explicit review passthrough allows provisional detection with warning."""
        cow_path = self.neg_img_dir / "ambiguous_pasture_cow_distant_turbine_01.jpg"
        if not cow_path.exists():
            self.skipTest("Negative sample ambiguous_pasture_cow_distant_turbine_01.jpg not found")

        # Explicit passthrough enabled
        res = self.pipeline.predict(cow_path, allow_review_passthrough=True)

        self.assertTrue(res["requires_manual_review"])
        self.assertEqual(res["pipeline_status"], "REVIEW_REQUIRED")
        self.assertTrue(any("PROVISIONAL INFERENCE" in w for w in res["warnings"]))

    def test_unavailable_model_handling(self):
        """Test 6: Graceful fallback when model weights path is invalid."""
        fake_pipeline = AeroLossDetectionPipeline(weights_path=REPO_ROOT / "models" / "non_existent_weights_123.pt")
        # Ensure detector is marked unavailable
        fake_pipeline.model = None
        fake_pipeline.detector_loaded = False

        test_img = self.raw_img_dir / "0.jpg"
        res = fake_pipeline.predict(test_img)

        self.assertEqual(res["quality_decision"], "USABLE")
        self.assertEqual(res["relevance_decision"], "BLADE")
        self.assertEqual(res["pipeline_status"], "PASSED_GUARDRAIL")
        self.assertEqual(res["damage_detection_status"], "DETECTOR_MODEL_UNAVAILABLE")
        self.assertEqual(len(res["detections"]), 0)
        self.assertTrue(any("weights are not loaded" in w for w in res["warnings"]))

    def test_clean_blade_zero_detections(self):
        """Test 7: Clean blade handling (real inspection image where no damage is detected)."""
        clean_img_path = self.raw_img_dir / "1042.jpg"
        if not clean_img_path.exists():
            self.skipTest("Clean blade sample 1042.jpg not found")

        res = self.pipeline.predict(clean_img_path)
        self.assertEqual(res["quality_decision"], "USABLE")
        self.assertEqual(res["relevance_decision"], "BLADE")
        if self.pipeline.detector_loaded:
            self.assertEqual(res["damage_detection_status"], "SUCCESS")
            # 1042.jpg has zero detected defect instances with the current weights
            self.assertEqual(res["detections_count"], 0)
            self.assertIn("clean blade", res["summary"].lower())

    def test_convenience_function(self):
        """Test 8: Standalone convenience function run_aeroloss_detection_pipeline."""
        test_img = self.raw_img_dir / "0.jpg"
        res = run_aeroloss_detection_pipeline(test_img)
        self.assertIsInstance(res, dict)
        self.assertIn("pipeline_status", res)
        self.assertIn("quality_decision", res)
        self.assertIn("relevance_decision", res)
        self.assertIn("detections", res)


if __name__ == "__main__":
    unittest.main()
