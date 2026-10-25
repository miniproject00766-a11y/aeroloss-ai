"""
AeroLoss AI — Phase 1 Regression Tests for Physics Engine & LUT Lookup
Tests:
1. Matching LUT rows across multiple severities and locations.
2. Non-matching LUT rows (out-of-envelope queries falling back to exact calculation).
3. Multiple wind speeds across Region 2 and Region 3.
4. Input validation (negative rated power, out-of-range r/R, negative area, non-numeric).
5. Output concordance between LUT and exact physics calculation.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is available
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from services.physics_service import PhysicsService
from src.physics.aero_engine import AeroLossPhysicsEngine


class TestPhysicsLUTRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = PhysicsService()
        cls.engine = AeroLossPhysicsEngine()

    # -------------------------------------------------------------------------
    # 1. Matching LUT Rows (Fix for the Boolean idxmin Bug)
    # -------------------------------------------------------------------------
    def test_lut_matching_row_severity_4_outboard(self):
        """Verify that querying high severity outboard defect returns correct LUT row, NOT row 0."""
        res = self.service.compute_aerodynamics(
            severity=4,
            r_R=0.90,
            area_pct=5.0,
            wind_speed=10.0,
            rated_kw=3600.0,
            use_lut=True
        )
        self.assertEqual(res['computation_mode'], 'LUT_fast')
        # In the old bug, this returned severity 1 and power_loss_kw = 0.01 kW
        # The correct LUT row has power_loss_kw = 158.08 kW and aep_loss_pct = 4.27%
        self.assertGreater(res['power_loss_kw'], 50.0, "Severity 4 at r/R=0.90 must not return row 0 (0.01 kW)")
        self.assertAlmostEqual(res['aep_loss_pct'], 4.27, delta=0.5)
        self.assertAlmostEqual(res['daily_energy_loss_kwh'], 1327.48, delta=10.0)

    def test_lut_matching_row_severity_5_tip(self):
        """Verify that Severity 5 tip defect at 15 m/s matches the high severity LUT entry."""
        res = self.service.compute_aerodynamics(
            severity=5,
            r_R=0.98,
            area_pct=10.0,
            wind_speed=15.0,
            rated_kw=3600.0,
            use_lut=True
        )
        self.assertEqual(res['computation_mode'], 'LUT_fast')
        self.assertGreater(res['aep_loss_pct'], 8.0, "Severity 5 tip AEP loss should be >= 8%")
        self.assertGreater(res['daily_energy_loss_kwh'], 2500.0)

    def test_lut_matching_row_severity_1_inboard(self):
        """Verify that Severity 1 inboard defect returns low aerodynamic loss."""
        res = self.service.compute_aerodynamics(
            severity=1,
            r_R=0.30,
            area_pct=0.5,
            wind_speed=4.0,
            rated_kw=3600.0,
            use_lut=True
        )
        self.assertEqual(res['computation_mode'], 'LUT_fast')
        self.assertLessEqual(res['power_loss_kw'], 0.10)
        self.assertAlmostEqual(res['aep_loss_pct'], 0.20, delta=0.05)

    # -------------------------------------------------------------------------
    # 2. Non-Matching LUT Rows (Fallback to Exact Calculation)
    # -------------------------------------------------------------------------
    def test_lut_non_matching_high_wind_speed_fallback(self):
        """Query with wind speed outside LUT envelope (v=22 m/s) gracefully falls back to exact calculation."""
        res = self.service.compute_aerodynamics(
            severity=3,
            r_R=0.85,
            area_pct=3.0,
            wind_speed=22.0,
            rated_kw=3600.0,
            use_lut=True
        )
        # Should gracefully fall back to BEM_exact
        self.assertEqual(res['computation_mode'], 'BEM_exact')
        self.assertEqual(res['airfoil_reference'], 'FFA-W3-241 (Re=1e7 BEM)')
        self.assertGreater(res['power_loss_kw'], 0.0)

    def test_lut_non_matching_low_wind_speed_fallback(self):
        """Query below cut-in (v=2.0 m/s) falls back to exact calculation with zero power."""
        res = self.service.compute_aerodynamics(
            severity=2,
            r_R=0.50,
            area_pct=2.0,
            wind_speed=2.0,
            rated_kw=3600.0,
            use_lut=True
        )
        self.assertEqual(res['computation_mode'], 'BEM_exact')
        self.assertEqual(res['power_baseline_kw'], 0.0)
        self.assertEqual(res['power_loss_kw'], 0.0)

    # -------------------------------------------------------------------------
    # 3. Wind Speed Sensitivity and Power Curve Regions
    # -------------------------------------------------------------------------
    def test_power_loss_monotonic_in_region_2(self):
        """In Region 2 (3 <= v < 11.5 m/s), power loss must increase monotonically with wind speed."""
        ws_list = [4.0, 6.0, 8.0, 10.0]
        losses = []
        for ws in ws_list:
            res = self.engine.compute_aerodynamic_loss(
                severity=3, r_R=0.85, area_pct=3.0, wind_speed=ws, rated_power_kw=3600.0
            )
            losses.append(res['power_loss_kw'])

        for i in range(len(losses) - 1):
            self.assertLess(losses[i], losses[i + 1], f"Power loss at {ws_list[i]} m/s should be < loss at {ws_list[i+1]} m/s")

    def test_region_3_rated_pitch_mitigation(self):
        """In Region 3 (v >= 11.5 m/s), baseline power is capped at rated_power_kw."""
        res_12 = self.engine.compute_aerodynamic_loss(
            severity=3, r_R=0.85, area_pct=3.0, wind_speed=12.0, rated_power_kw=3600.0
        )
        res_15 = self.engine.compute_aerodynamic_loss(
            severity=3, r_R=0.85, area_pct=3.0, wind_speed=15.0, rated_power_kw=3600.0
        )
        self.assertEqual(res_12['power_baseline_kw'], 3600.0)
        self.assertEqual(res_15['power_baseline_kw'], 3600.0)

    # -------------------------------------------------------------------------
    # 4. Input Validation & Error Handling
    # -------------------------------------------------------------------------
    def test_invalid_negative_rated_power_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.service.compute_aerodynamics(
                severity=3, r_R=0.85, area_pct=2.0, wind_speed=8.0, rated_kw=-3600.0
            )

    def test_invalid_out_of_range_r_R_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.service.compute_aerodynamics(
                severity=3, r_R=1.50, area_pct=2.0, wind_speed=8.0, rated_kw=3600.0
            )

    def test_invalid_negative_area_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.service.compute_aerodynamics(
                severity=3, r_R=0.85, area_pct=-5.0, wind_speed=8.0, rated_kw=3600.0
            )

    def test_non_numeric_input_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.service.compute_aerodynamics(
                severity="critical", r_R=0.85, area_pct=2.0, wind_speed=8.0, rated_kw=3600.0
            )

    # -------------------------------------------------------------------------
    # 5. Concordance between LUT and Live Physics Calculation
    # -------------------------------------------------------------------------
    def test_lut_and_live_concordance_on_grid_point(self):
        """On an exact grid point, LUT and live calculation should have consistent AEP loss."""
        live_res = self.service.compute_aerodynamics(
            severity=3, r_R=0.70, area_pct=2.0, wind_speed=8.0, rated_kw=3600.0, use_lut=False
        )
        lut_res = self.service.compute_aerodynamics(
            severity=3, r_R=0.70, area_pct=2.0, wind_speed=8.0, rated_kw=3600.0, use_lut=True
        )
        self.assertAlmostEqual(live_res['aep_loss_pct'], lut_res['aep_loss_pct'], delta=0.1)
        self.assertAlmostEqual(live_res['power_loss_kw'], lut_res['power_loss_kw'], delta=0.5)
        self.assertAlmostEqual(live_res['daily_energy_loss_kwh'], lut_res['daily_energy_loss_kwh'], delta=1.0)


if __name__ == '__main__':
    unittest.main()
