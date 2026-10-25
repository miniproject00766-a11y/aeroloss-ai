"""
AeroLoss AI — Physics & Financial Consistency Regression Suite
Tests:
1. Units and formula consistency (power deficit, AEP loss %, daily/annual energy conversion).
2. Financial chain arithmetic (daily loss = daily kWh * tariff, annual = daily * 365, payback = repair / daily).
3. Zero and negative loss edge cases (nominal clean blade returns 0 loss and None payback; negative inputs rejected).
4. Bounded and clearly labelled estimates (2D image frame coverage vs 3D blade damage proxy).
5. Preservation of mandatory structural safety overrides (crack and thunderstrike).
6. Capacity factor propagation across turbine configurations.
7. Solver transparency and scientific limitations disclosure.
"""

import os
import sys
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.physics.aero_engine import AeroLossPhysicsEngine
from services.physics_service import PhysicsService
from src.financial.financial_engine import FinancialEngine
from services.financial_service import FinancialService
from services.damage_service import DamageService
from services.decision_service import DecisionService


class TestPhysicsFinancialConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aero_engine = AeroLossPhysicsEngine()
        cls.physics_svc = PhysicsService()
        cls.fin_engine = FinancialEngine(default_tariff=4.50, default_repair_cost=40000.0)
        cls.fin_svc = FinancialService()
        cls.damage_svc = DamageService()
        cls.decision_svc = DecisionService()

    def test_units_and_formula_consistency(self):
        """Verifies mathematical consistency of daily energy loss derived from rated power, CF, and AEP loss %."""
        rated_kw = 3600.0
        cf = 0.36
        wind_speed = 7.56
        res = self.aero_engine.compute_aerodynamic_loss(
            severity=2, r_R=0.685, area_pct=0.29, wind_speed=wind_speed,
            rated_power_kw=rated_kw, capacity_factor=cf
        )
        expected_daily_baseline = rated_kw * 24.0 * cf  # 31,104.0 kWh/day
        self.assertAlmostEqual(res['daily_baseline_energy_kwh'], expected_daily_baseline, places=1)

        expected_daily_loss = round(expected_daily_baseline * (res['aep_loss_pct'] / 100.0), 2)
        self.assertAlmostEqual(res['daily_energy_loss_kwh'], expected_daily_loss, places=2)

        expected_annual_loss = round(expected_daily_loss * 365.0, 2)
        self.assertAlmostEqual(res['annual_energy_loss_kwh'], expected_annual_loss, places=1)
        self.assertEqual(res['wind_speed_ms'], round(wind_speed, 2))
        self.assertEqual(res['capacity_factor'], cf)

    def test_financial_chain_arithmetic(self):
        """Verifies daily loss = kWh * tariff, annual = daily * 365, and simple payback = cost / daily."""
        daily_kwh = 120.0
        tariff = 5.00
        repair_cost = 60000.0
        res = self.fin_engine.calculate_losses(
            daily_energy_loss_kwh=daily_kwh, tariff=tariff, repair_cost=repair_cost
        )
        self.assertEqual(res['daily_loss_inr'], 600.0)  # 120 * 5.00
        self.assertEqual(res['annual_loss_inr'], 219000.0)  # 600.0 * 365
        self.assertEqual(res['monthly_loss_inr'], 18000.0)  # 600.0 * 30
        self.assertEqual(res['payback_days'], 100.0)  # 60000 / 600
        self.assertAlmostEqual(res['payback_years'], 0.27, places=2)

    def test_zero_defect_clean_blade_edge_case(self):
        """Verifies that a clean turbine blade (severity=0, area=0) produces strictly zero loss and None payback."""
        res_aero = self.aero_engine.compute_aerodynamic_loss(
            severity=0, r_R=0.88, area_pct=0.0, wind_speed=8.0, rated_power_kw=3600.0, capacity_factor=0.36
        )
        self.assertEqual(res_aero['power_loss_kw'], 0.0)
        self.assertEqual(res_aero['aep_loss_pct'], 0.0)
        self.assertEqual(res_aero['daily_energy_loss_kwh'], 0.0)
        self.assertEqual(res_aero['annual_energy_loss_kwh'], 0.0)
        self.assertEqual(res_aero['aerodynamic_status'], 'CLEAN_NOMINAL_SURFACE')

        res_fin = self.fin_engine.calculate_losses(
            daily_energy_loss_kwh=res_aero['daily_energy_loss_kwh'], tariff=4.50, repair_cost=40000.0
        )
        self.assertEqual(res_fin['daily_loss_inr'], 0.0)
        self.assertEqual(res_fin['annual_loss_inr'], 0.0)
        self.assertIsNone(res_fin['payback_days'])
        self.assertIsNone(res_fin['payback_years'])
        self.assertEqual(res_fin['sensitivity_range_inr']['daily_loss_p10'], 0.0)
        self.assertEqual(res_fin['sensitivity_range_inr']['daily_loss_p90'], 0.0)

    def test_negative_inputs_raise_validation_error(self):
        """Verifies defensive input validation rejects negative rated power or negative area."""
        with self.assertRaises(ValueError):
            self.aero_engine.compute_aerodynamic_loss(severity=2, r_R=0.5, area_pct=-1.0)
        with self.assertRaises(ValueError):
            self.aero_engine.compute_aerodynamic_loss(severity=2, r_R=0.5, area_pct=1.0, rated_power_kw=-1000.0)

    def test_2d_frame_coverage_vs_3d_blade_damage_proxy(self):
        """Verifies that 2D photographic frame coverage is clearly distinguished from 3D blade damage proxy."""
        # Mock detection occupying 20% of the photographic frame
        mock_dets = [{
            'damage_type': 'surface_injure',
            'confidence': 0.85,
            'bounding_box': [0.0, 0.0, 400.0, 512.0],  # 400*512 / (1000*1024) = 20.0%
            'r_over_R': 0.75
        }]
        char = self.damage_svc.characterize_detections(mock_dets, img_width=1000, img_height=1024)
        self.assertEqual(char['image_frame_coverage_pct'], 20.0)
        # Conservative proxy scales by 0.10 -> 2.0%
        self.assertEqual(char['blade_damage_area_proxy_pct'], 2.0)
        self.assertEqual(char['damage_area_pct'], 2.0)
        self.assertIn('scale_proxy_method', char['provenance'])

    def test_capacity_factor_scaling(self):
        """Verifies that turbine-specific capacity factors (e.g. 0.42 vs 0.36) proportionately scale daily baseline."""
        res_36 = self.aero_engine.compute_aerodynamic_loss(severity=3, r_R=0.8, area_pct=1.0, capacity_factor=0.36)
        res_42 = self.aero_engine.compute_aerodynamic_loss(severity=3, r_R=0.8, area_pct=1.0, capacity_factor=0.42)
        ratio = res_42['daily_energy_loss_kwh'] / res_36['daily_energy_loss_kwh']
        self.assertAlmostEqual(ratio, 0.42 / 0.36, places=2)

    def test_preservation_of_structural_safety_overrides(self):
        """Verifies that structural cracks and lightning strikes trigger mandatory engineering assessment with critical urgency."""
        dec_crack = self.decision_svc.evaluate_decision(
            defect_class='crack', severity=5, r_R=0.612, area_pct=0.869,
            daily_loss_inr=50.0, payback_days=800.0, confidence=0.85
        )
        self.assertEqual(dec_crack['decision'], 'ENGINEERING_ASSESSMENT')
        self.assertEqual(dec_crack['urgency'], 'CRITICAL')
        self.assertTrue(dec_crack['safety_override_active'])

        dec_thunder = self.decision_svc.evaluate_decision(
            defect_class='thunderstrike', severity=5, r_R=0.50, area_pct=1.0,
            daily_loss_inr=20.0, payback_days=2000.0, confidence=0.90
        )
        self.assertEqual(dec_thunder['decision'], 'ENGINEERING_ASSESSMENT')
        self.assertEqual(dec_thunder['urgency'], 'CRITICAL')
        self.assertTrue(dec_thunder['safety_override_active'])

    def test_solver_transparency_and_disclaimers(self):
        """Verifies that solver method and scientific limitations transparently document the empirical scaling nature."""
        res = self.physics_svc.compute_aerodynamics(severity=3, r_R=0.85, area_pct=1.0)
        self.assertEqual(res['solver_method'], 'empirical_polar_lut_interpolation')
        self.assertEqual(res['aerodynamic_model_type'], 'empirical_scaling_proxy')
        self.assertIn('scientific_limitations', res)
        limitations = res['scientific_limitations']
        self.assertIn('bem_solver_status', limitations)
        self.assertIn('optical_scale_status', limitations)
        self.assertIn('not an iterative dynamic blade element momentum (BEM)', limitations['bem_solver_status'])


if __name__ == '__main__':
    unittest.main()
