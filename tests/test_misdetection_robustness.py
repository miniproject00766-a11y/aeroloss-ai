import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
from PIL import Image

from services.yolo_service import YoloService
from services.damage_service import DamageService
from services.decision_service import DecisionService
from src.decision.decision_engine import DecisionEngine

class TestMisdetectionRobustness(unittest.TestCase):
    def setUp(self):
        self.yolo_service = YoloService()
        self.damage_service = DamageService()
        self.decision_service = DecisionService()
        self.decision_engine = DecisionEngine()

    def test_low_confidence_proposals_not_discarded(self):
        """Proposals with confidence < 0.25 must be preserved, flagged low-confidence, and escalated to UNCERTAIN_NEEDS_REVIEW."""
        detections = [{
            'damage_type': 'corrosion',
            'confidence': 0.18,
            'confidence_pct': '18.0%',
            'is_low_confidence': True,
            'confidence_grade': 'LOW',
            'bounding_box': [100.0, 100.0, 300.0, 300.0],
            'damage_area_percent': 3.8,
            'r_over_R': 0.85,
            'detection_method': 'yolov8',
            'is_synthetic': False,
            'is_heuristic_fallback': False,
            'crosscheck': {
                'status': 'EVALUATED',
                'models_agree': None,
                'classification_disagreement': False,
                'yolo_class': 'corrosion',
                'yolo_confidence': 0.18,
                'mobilenet_class': None,
                'decision_rule_applied': 'YOLO_ONLY'
            }
        }]

        char = self.damage_service.characterize_detections(detections, 1024, 1024)
        self.assertEqual(char['analysis_state'], 'UNCERTAIN_NEEDS_REVIEW')
        self.assertTrue(char['is_low_confidence'])
        self.assertEqual(char['confidence_grade'], 'LOW')
        self.assertFalse(char['health_confirmed'])
        self.assertNotEqual(char['health_status_label'], 'Healthy')

        dec = self.decision_service.evaluate_decision(
            defect_class=char['primary_defect'],
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['damage_area_pct'],
            daily_loss_inr=500.0,
            payback_days=80.0,
            confidence=char['confidence'],
            char=char
        )
        self.assertIn(dec['decision'], ['ENGINEERING_ASSESSMENT', 'REPAIR'])
        if dec['decision'] == 'ENGINEERING_ASSESSMENT':
            self.assertIn(dec['urgency'], ['CRITICAL', 'HIGH'])

    def test_unlocalized_defect_escalation(self):
        """When 0 YOLO boxes exist but whole-image model sees damage, system must not claim 'Healthy' or 'No Defect'."""
        unlocalized_dets = [{
            'damage_type': 'crack',
            'confidence': 0.72,
            'confidence_pct': '72.0%',
            'is_low_confidence': True,
            'confidence_grade': 'LOW',
            'bounding_box': None,
            'damage_area_percent': 0.0,
            'r_over_R': None,
            'detection_method': 'mobilenet_whole_image_unlocalized',
            'is_synthetic': False,
            'is_heuristic_fallback': False,
            'is_unlocalized': True,
            'crosscheck': {
                'status': 'WHOLE_IMAGE_UNLOCALIZED',
                'models_agree': None,
                'classification_disagreement': True,
                'yolo_class': None,
                'yolo_confidence': None,
                'mobilenet_class': 'crack',
                'mobilenet_confidence': 0.72,
                'decision_rule_applied': 'UNLOCALIZED_DEFECT_DETECTION_NEEDS_REVIEW'
            }
        }]

        char = self.damage_service.characterize_detections(unlocalized_dets, 1024, 1024)
        self.assertEqual(char['analysis_state'], 'UNCERTAIN_NEEDS_REVIEW')
        self.assertTrue(char['is_unlocalized'])
        self.assertEqual(char['loss_status'], 'UNAVAILABLE_NEEDS_REVIEW')
        self.assertTrue(char['structural_safety_override'])
        self.assertEqual(char['severity_proxy'], 5)

        dec = self.decision_engine.evaluate_action(
            defect_class=char['primary_defect'],
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['damage_area_pct'],
            daily_loss_inr=0.0,
            payback_days=999.0,
            confidence=char['confidence'],
            image_frame_coverage_pct=0.0,
            is_structural_override=char['structural_safety_override']
        )
        self.assertEqual(dec['decision'], 'ENGINEERING_ASSESSMENT')
        self.assertEqual(dec['urgency'], 'CRITICAL')
        self.assertTrue(dec['safety_override_active'])

    def test_no_damage_detected_baseline_semantics(self):
        """When 0 defects are detected across models, state is NO_DAMAGE_DETECTED with Nominal (Unconfirmed), baseline 0 loss, and honest disclaimers."""
        char = self.damage_service.characterize_detections([], 1024, 1024)
        self.assertEqual(char['analysis_state'], 'NO_DAMAGE_DETECTED')
        self.assertEqual(char['detection_status'], 'NO_DEFECT_DETECTED')
        self.assertEqual(char['severity_proxy'], 0)
        self.assertFalse(char['health_confirmed'])
        self.assertEqual(char['health_status_label'], 'Nominal (Unconfirmed)')
        self.assertEqual(char['loss_status'], 'BASELINE_ZERO_ESTIMATE')

        dec = self.decision_engine.evaluate_action(
            defect_class=char['primary_defect'],
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['damage_area_pct'],
            daily_loss_inr=0.0,
            payback_days=None,
            confidence=1.0,
            image_frame_coverage_pct=0.0,
            is_structural_override=False
        )
        self.assertEqual(dec['decision'], 'MONITOR')
        self.assertEqual(dec['urgency'], 'LOW')
        reasons_text = " ".join(dec['reasons'])
        self.assertIn("UNCONFIRMED HEALTH DISCLAIMER", reasons_text)
        # Must not claim 999 days payback or aerodynamic degradation on clean blade
        self.assertNotIn("Extended payback period", reasons_text)
        self.assertNotIn("Minor aerodynamic degradation", reasons_text)

    def test_real_test_image_end_to_end(self):
        """Run on 01_drone_blade_erosion_crack.jpg to verify it is NEVER misdetected as NO DEFECT."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        img_path = os.path.join(base_dir, "test_uploads", "01_drone_blade_erosion_crack.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            dets, draw_img, is_valid, rej_reason, guard_status = self.yolo_service.detect_defects(img)
            self.assertTrue(is_valid)
            self.assertGreater(len(dets), 0, "Visibly damaged blade image must produce candidate detections or unlocalized signal, never empty [].")
            char = self.damage_service.characterize_detections(dets, img.width, img.height)
            self.assertIn(char['analysis_state'], ['DAMAGE_DETECTED', 'UNCERTAIN_NEEDS_REVIEW'])
            self.assertNotEqual(char['primary_defect'], 'No Defect')
            self.assertGreater(char['severity_proxy'], 0)

if __name__ == '__main__':
    unittest.main()
