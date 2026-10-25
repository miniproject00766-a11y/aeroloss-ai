import unittest
import os
import sys
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.physics.aero_engine import AeroLossPhysicsEngine
from src.financial.financial_engine import FinancialEngine
from src.decision.decision_engine import DecisionEngine
from src.pipeline import AeroLossPipeline

class TestAeroLossPipeline(unittest.TestCase):
    def setUp(self):
        self.physics = AeroLossPhysicsEngine()
        self.financial = FinancialEngine(default_tariff=4.50, default_repair_cost=40000.0)
        self.decision = DecisionEngine()
        self.pipeline = AeroLossPipeline()

    def test_physics_airfoil_polar(self):
        clean_p = self.physics.clean_polar
        self.assertIsNotNone(clean_p)
        deg_p = self.physics.get_degraded_polar(severity=4)
        self.assertIsNotNone(deg_p)
        # Verify Cl decreases and Cd increases at positive AoA
        mask = (clean_p['AoA_deg'] >= 5) & (clean_p['AoA_deg'] <= 15)
        self.assertTrue((deg_p.loc[mask, 'Cl'] <= clean_p.loc[mask, 'Cl']).all())
        self.assertTrue((deg_p.loc[mask, 'Cd'] >= clean_p.loc[mask, 'Cd']).all())

    def test_physics_spanwise_scaling(self):
        # Tip (r/R=0.95) should have significantly higher power loss than root (r/R=0.30)
        res_tip = self.physics.compute_aerodynamic_loss(severity=3, r_R=0.95, area_pct=3.0, wind_speed=8.0)
        res_root = self.physics.compute_aerodynamic_loss(severity=3, r_R=0.30, area_pct=3.0, wind_speed=8.0)
        self.assertGreater(res_tip['power_loss_kw'], res_root['power_loss_kw'])
        self.assertGreater(res_tip['aep_loss_pct'], res_root['aep_loss_pct'])

    def test_financial_engine(self):
        res = self.financial.calculate_losses(daily_energy_loss_kwh=1000.0, tariff=4.50, repair_cost=45000.0)
        self.assertEqual(res['daily_loss_inr'], 4500.0)
        self.assertEqual(res['payback_days'], 10.0)
        self.assertEqual(res['monthly_loss_inr'], 135000.0)

    def test_decision_structural_crack(self):
        res = self.decision.evaluate_action(
            defect_class='crack', severity=4, r_R=0.85, area_pct=3.0,
            daily_loss_inr=3000.0, payback_days=15.0
        )
        self.assertEqual(res['decision'], 'ENGINEERING_ASSESSMENT')
        self.assertEqual(res['urgency'], 'CRITICAL')

    def test_decision_outboard_repair(self):
        res = self.decision.evaluate_action(
            defect_class='surface_injure', severity=3, r_R=0.92, area_pct=5.0,
            daily_loss_inr=4000.0, payback_days=12.0
        )
        self.assertEqual(res['decision'], 'REPAIR')
        self.assertEqual(res['urgency'], 'HIGH')

    def test_decision_inboard_monitor(self):
        res = self.decision.evaluate_action(
            defect_class='craze', severity=1, r_R=0.35, area_pct=0.5,
            daily_loss_inr=200.0, payback_days=250.0
        )
        self.assertEqual(res['decision'], 'MONITOR')
        self.assertEqual(res['urgency'], 'LOW')

    def test_end_to_end_pipeline(self):
        test_dir = os.path.join(BASE_DIR, "data", "processed", "patches", "crack")
        patch_file = os.path.join(test_dir, os.listdir(test_dir)[0])
        res = self.pipeline.analyze_image(patch_file, r_R=0.88, area_pct=4.0, wind_speed=8.0)
        self.assertIn('visual_detection', res)
        self.assertIn('physics_aerodynamics', res)
        self.assertIn('energy_impact', res)
        self.assertIn('financial_impact', res)
        self.assertIn('fatigue_prognostics', res)
        self.assertIn('erosion_safe_mode', res)
        self.assertIn('maintenance_decision', res)
        self.assertIn('scada_validation', res['physics_aerodynamics'])
        self.assertIn(res['maintenance_decision']['recommended_action'], ['REPAIR', 'MONITOR', 'ENGINEERING_ASSESSMENT'])

    def test_pipeline_without_localized_damage_keeps_zero_loss_and_marks_geometry_unavailable(self):
        class NoDetectionService:
            def detect_defects(self, image):
                return [], image.copy(), True, "No reliable defect detected.", "ACCEPT"

        self.pipeline.yolo_service = NoDetectionService()
        res = self.pipeline.analyze_image(Image.new('RGB', (64, 64), color='white'))
        char = res['characterization']

        self.assertEqual(char['geometry_status'], 'UNAVAILABLE_NO_DETECTIONS')
        self.assertFalse(char['spanwise_position_available'])
        self.assertEqual(char['primary_bbox_frame_coverage_pct'], 0.0)
        self.assertEqual(char['union_frame_coverage_pct'], 0.0)
        self.assertEqual(char['blade_damage_area_proxy_pct'], 0.0)
        self.assertEqual(res['energy_impact']['daily_energy_loss_kwh'], 0.0)
        self.assertEqual(res['financial_impact']['daily_financial_loss_inr'], 0.0)
        self.assertEqual(res['visual_detection']['detection_status'], 'NO_DEFECT_DETECTED')
        self.assertIsNone(res['visual_detection']['confidence'])
        self.assertEqual(res['visual_detection']['confidence_pct'], 'N/A')
        self.assertFalse(char['health_confirmed'])
        self.assertEqual(res['maintenance_decision']['recommended_action'], 'MONITOR')
        self.assertEqual(res['maintenance_decision']['urgency'], 'LOW')

    def test_extensive_low_confidence_damage_requires_engineering_assessment(self):
        image_path = os.path.join(BASE_DIR, "test_uploads", "01_drone_blade_erosion_crack.jpg")
        if not os.path.isfile(image_path):
            self.skipTest("Extensive blade damage regression image is not available")

        result = self.pipeline.analyze_image(image_path)
        char = result['characterization']
        decision = result['maintenance_decision']

        self.assertTrue(result['visual_detection']['detections'])
        self.assertLess(result['visual_detection']['confidence'], 0.20)
        self.assertGreaterEqual(char['union_frame_coverage_pct'], 20.0)
        self.assertTrue(char['structural_safety_override'])
        self.assertEqual(decision['recommended_action'], 'ENGINEERING_ASSESSMENT')
        self.assertEqual(decision['urgency'], 'HIGH')
        self.assertTrue(decision['safety_override_active'])
        self.assertIn(
            'does not confirm internal structural failure',
            ' '.join(decision['reasoning'])
        )

    def test_exact_lightning_image_uses_dashboard_geometry_and_safety_override(self):
        image_path = os.environ.get(
            'AEROLOSS_LIGHTNING_IMAGE',
            r'C:\Users\akhil\Downloads\lightning-damage-wind-turbine_1.webp'
        )
        if not os.path.isfile(image_path):
            self.skipTest("Exact lightning-damage-wind-turbine_1.webp is not available")

        res = self.pipeline.analyze_image(image_path)
        char = res['characterization']
        decision = res['maintenance_decision']
        detections = res['visual_detection']['detections']

        self.assertTrue(detections)
        self.assertIsNotNone(detections[0].get('yolo_class'))
        self.assertIsNotNone(detections[0].get('mobilenet_class'))
        self.assertGreaterEqual(char['union_frame_coverage_pct'], 20.0)
        self.assertEqual(
            char['primary_bbox_frame_coverage_pct'],
            char['union_frame_coverage_pct']
        )
        self.assertEqual(
            char['blade_damage_area_proxy_pct'],
            char['damage_area_pct']
        )
        self.assertTrue(char['structural_safety_override'])
        self.assertTrue(decision['safety_override_active'])
        self.assertEqual(decision['recommended_action'], 'ENGINEERING_ASSESSMENT')
        self.assertEqual(decision['urgency'], 'CRITICAL')
        self.assertEqual(
            decision['metrics_snapshot']['image_frame_coverage_pct'],
            char['union_frame_coverage_pct']
        )
        self.assertIn('NDT', ' '.join(decision['reasoning']))
        self.assertIn('does not confirm internal structural failure', ' '.join(decision['reasoning']))
        self.assertEqual(res['energy_impact']['estimate_status'], 'PROVISIONAL_UNCALIBRATED_ESTIMATE')
        self.assertEqual(res['financial_impact']['estimate_status'], 'PROVISIONAL_UNCALIBRATED_ESTIMATE')

    def test_known_clean_blade_stays_monitor_with_zero_loss(self):
        image_path = os.path.join(BASE_DIR, "data", "raw", "images", "1042.jpg")
        if not os.path.isfile(image_path):
            self.skipTest("Known clean blade sample 1042.jpg is not available")

        result = self.pipeline.analyze_image(image_path)

        self.assertTrue(result['is_valid_blade_image'])
        self.assertEqual(result['visual_detection']['detection_status'], 'NO_DEFECT_DETECTED')
        self.assertIsNone(result['visual_detection']['confidence'])
        self.assertEqual(result['visual_detection']['confidence_pct'], 'N/A')
        self.assertEqual(result['characterization']['severity_proxy'], 0)
        self.assertEqual(result['energy_impact']['daily_energy_loss_kwh'], 0.0)
        self.assertEqual(result['financial_impact']['daily_financial_loss_inr'], 0.0)
        self.assertEqual(result['maintenance_decision']['recommended_action'], 'MONITOR')
        self.assertEqual(result['maintenance_decision']['urgency'], 'LOW')

    def test_unsupported_detected_class_requires_engineering_review(self):
        result = self.decision.evaluate_action(
            defect_class='unmapped_detector_label',
            severity=3,
            r_R=0.50,
            area_pct=1.0,
            daily_loss_inr=0.0,
            payback_days=999.0,
            unsupported_class=True
        )
        self.assertEqual(result['decision'], 'ENGINEERING_ASSESSMENT')
        self.assertEqual(result['urgency'], 'HIGH')
        self.assertTrue(result['classification_review_required'])

if __name__ == '__main__':
    unittest.main()
