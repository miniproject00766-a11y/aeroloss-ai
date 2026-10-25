import os
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from src.physics.fatigue_engine import DropletFatigueEngine

class TestFatigueModel(unittest.TestCase):
    def setUp(self):
        self.engine = DropletFatigueEngine()

    def test_relative_velocity_radial_growth(self):
        v_root = self.engine.calculate_relative_velocity(r_R=0.20)
        v_mid = self.engine.calculate_relative_velocity(r_R=0.50)
        v_tip = self.engine.calculate_relative_velocity(r_R=0.95)
        self.assertLess(v_root, v_mid)
        self.assertLess(v_mid, v_tip)

    def test_water_hammer_pressure_magnitude(self):
        v_tip = self.engine.calculate_relative_velocity(r_R=0.95)
        p_wh = self.engine.water_hammer_pressure(v_tip)
        # For tip speed ~75 m/s, water hammer pressure is approximately 80 - 150 MPa
        self.assertGreater(p_wh, 40.0)
        self.assertLess(p_wh, 250.0)

    def test_monotonic_cost_of_inaction(self):
        prog = self.engine.forecast_progression(initial_severity=2, r_R=0.90)
        costs = prog['cost_of_inaction_inr']
        self.assertGreaterEqual(costs[-1], costs[0])
        self.assertGreaterEqual(prog['cost_escalation_ratio'], 1.0)

    def test_dynamic_esm_location_awareness(self):
        # Outboard defect in rain triggers derating setpoint
        esm_tip = self.engine.compute_erosion_safe_mode(rain_intensity_mmh=5.0, r_R=0.92, severity=3)
        self.assertTrue(esm_tip['esm_active'])
        self.assertLess(esm_tip['derated_rpm'], 14.0)

        # Inboard defect is spared from curtailment, preserving generation
        esm_root = self.engine.compute_erosion_safe_mode(rain_intensity_mmh=5.0, r_R=0.40, severity=3)
        self.assertFalse(esm_root['esm_active'])
        self.assertEqual(esm_root['derated_rpm'], 14.0)
        self.assertGreater(esm_root['aep_saved_vs_blind_derate_pct'], 0.0)

if __name__ == '__main__':
    unittest.main()