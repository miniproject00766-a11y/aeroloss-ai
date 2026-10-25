import os
import sys
import unittest
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from services.yolo_service import YoloService
from services.damage_service import DamageService
from services.physics_service import PhysicsService
from services.financial_service import FinancialService
from services.decision_service import DecisionService
from src.pipeline import AeroLossPipeline

class TestInspectionIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = AeroLossPipeline()
        cls.yolo_service = YoloService(pipeline=cls.pipeline)
        cls.damage_service = DamageService()
        cls.physics_service = PhysicsService()
        cls.financial_service = FinancialService()
        cls.decision_service = DecisionService()

    def test_class_names_concordance(self):
        expected_classes = ['corrosion', 'crack', 'craze', 'hide_craze', 'surface_injure', 'thunderstrike']
        self.assertEqual(self.yolo_service.class_names, expected_classes)
        self.assertEqual(self.pipeline.class_names, expected_classes)
        if self.yolo_service.model is not None and hasattr(self.yolo_service.model, 'names'):
            model_names = [self.yolo_service.model.names[i] for i in sorted(self.yolo_service.model.names.keys())]
            self.assertEqual(model_names, expected_classes)

    def test_craze_with_large_box_triggers_conservative_assessment(self):
        """
        A craze-labelled box covering ~62.81% of a 1024x1024 frame is a large abnormal region.
        High confidence / model agreement must NOT cap this as cosmetic Level 2 MONITOR.
        Classification remains craze; safety handling is ENGINEERING_ASSESSMENT / CRITICAL.
        Image analysis must not be treated as confirmed internal structural failure.
        """
        mock_dets_low_conf = [{
            'damage_type': 'craze',
            'confidence': 0.1738,
            'confidence_pct': '17.4%',
            'is_low_confidence': True,
            'bounding_box': [10.0, 10.0, 900.0, 750.0],
            'damage_area_percent': 99.0,
            'r_over_R': 0.618
        }]
        char_low = self.damage_service.characterize_detections(mock_dets_low_conf, 1024, 1024)
        self.assertEqual(char_low['primary_defect'], 'craze')
        self.assertAlmostEqual(char_low['union_frame_coverage_pct'], 62.81, places=1)
        self.assertAlmostEqual(char_low['blade_damage_area_proxy_pct'], 5.0, places=1)
        self.assertTrue(char_low['is_uncertain_extensive_damage'])
        self.assertTrue(char_low['structural_safety_override'])
        self.assertEqual(char_low['severity_proxy'], 4)
        self.assertEqual(char_low['confidence_grade'], 'LOW')

        dec_low = self.decision_service.evaluate_decision(
            defect_class=char_low['primary_defect'],
            severity=char_low['severity_proxy'],
            r_R=char_low['spanwise_position_r_R'],
            area_pct=char_low['damage_area_pct'],
            daily_loss_inr=279.94,
            payback_days=142.9,
            confidence=char_low['confidence'],
            image_frame_coverage_pct=char_low['union_frame_coverage_pct'],
            char=char_low
        )
        self.assertEqual(dec_low['decision'], 'ENGINEERING_ASSESSMENT')
        self.assertEqual(dec_low['urgency'], 'CRITICAL')
        self.assertTrue(dec_low['safety_override_active'])
        self.assertIn('does not confirm internal structural failure', ' '.join(dec_low['reasons']).lower())

        mock_dets_high_conf = [{
            'damage_type': 'craze',
            'confidence': 0.85,
            'confidence_pct': '85.0%',
            'is_low_confidence': False,
            'models_agree': True,
            'bounding_box': [10.0, 10.0, 900.0, 750.0],
            'r_over_R': 0.618
        }]
        char_high = self.damage_service.characterize_detections(mock_dets_high_conf, 1024, 1024)
        self.assertEqual(char_high['primary_defect'], 'craze')
        self.assertEqual(char_high['severity_proxy'], 4)
        self.assertTrue(char_high['structural_safety_override'])
        self.assertTrue(char_high['is_uncertain_extensive_damage'])

    def test_low_confidence_detection_does_not_trigger_mandatory_safety_override(self):
        """
        Verify that bounding box [372.0, 5.0, 759.0, 135.0] on 1024x1024 derives 4.80% coverage
        (not an arbitrary placeholder), retains Level 1 severity, and does not trigger Critical Safety Override.
        """
        mock_dets = [{
            'damage_type': 'craze',
            'confidence': 0.1008,
            'confidence_pct': '10.1%',
            'is_low_confidence': True,
            'bounding_box': [372.0, 5.0, 759.0, 135.0],
            'damage_area_percent': 99.0,  # Contradictory placeholder to verify derivation
            'r_over_R': 0.576
        }]
        char = self.damage_service.characterize_detections(mock_dets, 1024, 1024)
        self.assertEqual(char['severity_proxy'], 1)
        # Exactly derived: (759-372) * (135-5) / (1024*1024) * 100 = 50310 / 1048576 * 100 = 4.80%
        self.assertAlmostEqual(char['union_frame_coverage_pct'], 4.80, places=1)
        self.assertAlmostEqual(char['blade_damage_area_proxy_pct'], 0.48, places=2)

        phys = self.physics_service.compute_aerodynamics(
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['damage_area_pct'],
            wind_speed=7.56,
            rated_kw=3600.0
        )
        fin = self.financial_service.compute_financials(
            daily_energy_loss_kwh=phys['daily_energy_loss_kwh'],
            tariff=4.50,
            repair_cost=40000.0
        )
        dec = self.decision_service.evaluate_decision(
            defect_class=char['primary_defect'],
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['damage_area_pct'],
            daily_loss_inr=fin['daily_loss_inr'],
            payback_days=fin['payback_days'],
            confidence=char['confidence'],
            wind_speed=7.56
        )
        self.assertNotEqual(dec['urgency'], 'CRITICAL')
        self.assertFalse(dec.get('safety_override_active', False))
        self.assertIn(dec['decision'], ['MONITOR', 'ENGINEERING_ASSESSMENT'])

    def test_inconsistent_input_area_strictly_overridden_by_geometry(self):
        """Verify that any conflicting damage_area_percent in input is overridden by geometric coordinates."""
        mock_dets = [{
            'damage_type': 'corrosion',
            'confidence': 0.80,
            'bounding_box': [0.0, 0.0, 100.0, 100.0],
            'damage_area_percent': 99.9  # Erroneous input
        }]
        char = self.damage_service.characterize_detections(mock_dets, 1000, 1000)
        # 100 * 100 / (1000 * 1000) * 100 = 1.0%
        self.assertEqual(char['union_frame_coverage_pct'], 1.0)
        self.assertEqual(char['blade_damage_area_proxy_pct'], 0.1)
        self.assertEqual(mock_dets[0]['damage_area_percent'], 1.0)

    def test_clipped_and_inverted_coordinates_handling(self):
        """Verify that out-of-boundary and inverted coordinates are safely clipped and sorted."""
        # Out-of-bounds box: [-50, -50, 1200, 1200] on 1000x1000 should clip to [0, 0, 1000, 1000] (100%)
        cov_clipped = self.damage_service.compute_box_coverage([-50.0, -50.0, 1200.0, 1200.0], 1000, 1000)
        self.assertEqual(cov_clipped, 100.0)

        # Inverted coordinates: [500, 400, 100, 200] on 1000x1000 -> w=400, h=200 -> 8.0%
        cov_inv = self.damage_service.compute_box_coverage([500.0, 400.0, 100.0, 200.0], 1000, 1000)
        self.assertEqual(cov_inv, 8.0)

    def test_different_image_dimensions(self):
        """Verify coverage calculation on non-square image dimensions (e.g. 870x288 and 548x217)."""
        # Box [0.0, 222.4, 326.9, 284.7] on 870x288
        # bw = 326.9, bh = 62.3, area = 20365.87, total = 250560 -> 8.13%
        cov1 = self.damage_service.compute_box_coverage([0.0, 222.4, 326.9, 284.7], 870, 288)
        self.assertEqual(cov1, 8.13)

        # Box [6.9, 5.4, 441.8, 217.0] on 548x217
        # bw = 434.9, bh = 211.6, area = 92024.84, total = 118916 -> 77.39%
        cov2 = self.damage_service.compute_box_coverage([6.9, 5.4, 441.8, 217.0], 548, 217)
        self.assertEqual(cov2, 77.39)

    def test_overlapping_boxes_sweep_line_union_coverage(self):
        """
        Verify that multiple overlapping bounding boxes compute exact union area
        and do NOT double-count the overlapping intersection.
        """
        # Box A: [0, 0, 500, 500] (25% on 1000x1000)
        # Box B: [250, 250, 750, 750] (25% on 1000x1000)
        # Intersection: [250, 250, 500, 500] (area 62,500, 6.25%)
        # Exact Union: 250,000 + 250,000 - 62,500 = 437,500 -> 43.75%
        boxes = [
            [0.0, 0.0, 500.0, 500.0],
            [250.0, 250.0, 750.0, 750.0]
        ]
        union_cov = self.damage_service.compute_union_coverage(boxes, 1000, 1000)
        self.assertEqual(union_cov, 43.75)

        dets = [
            {'damage_type': 'craze', 'confidence': 0.70, 'bounding_box': boxes[0]},
            {'damage_type': 'craze', 'confidence': 0.65, 'bounding_box': boxes[1]}
        ]
        char = self.damage_service.characterize_detections(dets, 1000, 1000)
        self.assertEqual(char['union_frame_coverage_pct'], 43.75)
        self.assertEqual(char['blade_damage_area_proxy_pct'], 4.375)
        self.assertEqual(char['union_bbox_area_pct'], 43.75)
        self.assertEqual(char['primary_bbox_area_pct'], 25.0)

    def test_multi_box_detections_preserved_and_prioritized(self):
        """Verify that multiple bounding boxes are preserved and primary defect selection is correct."""
        mock_dets = [
            {'damage_type': 'craze', 'confidence': 0.1065, 'bounding_box': [372.4, 5.5, 759.0, 135.5]},
            {'damage_type': 'hide_craze', 'confidence': 0.2304, 'bounding_box': [0.0, 222.4, 326.9, 284.7]}
        ]
        char = self.damage_service.characterize_detections(mock_dets, 870, 288)
        self.assertEqual(char['defect_count'], 2)
        # hide_craze has higher base severity (3) than craze (1)
        self.assertEqual(char['primary_defect'], 'hide_craze')
        self.assertEqual(char['severity_proxy'], 3)
        # Union coverage of disjoint boxes: 20.058% + 8.128% = 28.186% -> 28.19%
        self.assertEqual(char['union_frame_coverage_pct'], 28.19)
        self.assertEqual(char['blade_damage_area_proxy_pct'], 2.819)

    def test_crack_and_thunderstrike_retain_high_severity(self):
        """Verify that authentic structural defects (crack, thunderstrike) retain appropriate high severity."""
        crack_det = [{'damage_type': 'crack', 'confidence': 0.92, 'bounding_box': [100.0, 100.0, 450.0, 450.0], 'r_over_R': 0.88}]
        char_crack = self.damage_service.characterize_detections(crack_det, 1000, 1000)
        self.assertEqual(char_crack['severity_proxy'], 5)

        strike_det = [{'damage_type': 'thunderstrike', 'confidence': 0.85, 'bounding_box': [100.0, 100.0, 300.0, 250.0], 'r_over_R': 0.90}]
        char_strike = self.damage_service.characterize_detections(strike_det, 1000, 1000)
        self.assertEqual(char_strike['severity_proxy'], 5)

if __name__ == '__main__':
    unittest.main()
