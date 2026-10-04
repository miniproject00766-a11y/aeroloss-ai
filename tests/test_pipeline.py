import unittest
import os
import sys
import numpy as np

BASE_DIR = r"C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai"
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
        self.assertIn('maintenance_decision', res)
        self.assertIn(res['maintenance_decision']['recommended_action'], ['REPAIR', 'MONITOR', 'ENGINEERING_ASSESSMENT'])

if __name__ == '__main__':
    unittest.main()
