"""
AeroLoss AI — Energy-Loss Calibration, SCADA Schema & Physical Boundary Test Suite
Validates:
1. Boundary conditions: zero damage yields zero aerodynamic deficit.
2. Inflow wind speed regions: cut-in, sub-rated cubic power, rated power, and storm cut-out.
3. Monotonic physical progression with severity and outboard spanwise location (r/R).
4. Defect calibration interface disclosure and experimental references (Campobasso et al., 2022).
5. Multi-modal matched SCADA and inspection telemetry schema completeness.
6. SCADA underperformance attribution caveats (cannot attribute solely to blade damage).
7. Reproducibility of evaluation metrics from data/manifests/reproducible_evaluation_metrics.json.
8. Financial calculations across dynamic tariffs and repair costs (no hardcoding).
"""

import os
import sys
import json
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.physics.aero_engine import AeroLossPhysicsEngine
from services.physics_service import PhysicsService
from src.financial.financial_engine import FinancialEngine


class TestEnergyLossCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = AeroLossPhysicsEngine()
        cls.service = PhysicsService()
        cls.fin_engine = FinancialEngine()

    def test_healthy_blade_zero_loss_boundary(self):
        """Zero defect or zero damage area proxy must return identically zero power and energy loss."""
        # Severity 0
        res0 = self.engine.compute_aerodynamic_loss(severity=0, r_R=0.75, area_pct=0.0, wind_speed=8.0)
        self.assertEqual(res0['power_loss_kw'], 0.0)
        self.assertEqual(res0['aep_loss_pct'], 0.0)
        self.assertEqual(res0['daily_energy_loss_kwh'], 0.0)
        self.assertEqual(res0['annual_energy_loss_kwh'], 0.0)
        self.assertEqual(res0['aerodynamic_status'], 'CLEAN_NOMINAL_SURFACE')

        # Area 0.0 with severity > 0
        res_area0 = self.engine.compute_aerodynamic_loss(severity=3, r_R=0.85, area_pct=0.0, wind_speed=8.0)
        self.assertEqual(res_area0['power_loss_kw'], 0.0)
        self.assertEqual(res_area0['aep_loss_pct'], 0.0)
        self.assertEqual(res_area0['daily_energy_loss_kwh'], 0.0)

    def test_wind_speed_operational_regions(self):
        """Verify baseline power curve across 4 IEC operating regions."""
        rated_kw = 3600.0

        # Region 1: Below cut-in (< 3.0 m/s) -> 0 kW
        r1 = self.engine.compute_aerodynamic_loss(severity=2, r_R=0.6, area_pct=1.0, wind_speed=2.5, rated_power_kw=rated_kw)
        self.assertEqual(r1['power_baseline_kw'], 0.0)
        self.assertEqual(r1['power_loss_kw'], 0.0)

        # Region 2: Sub-rated (3.0 <= v < 11.5 m/s) -> cubic curve
        r2 = self.engine.compute_aerodynamic_loss(severity=2, r_R=0.6, area_pct=1.0, wind_speed=7.5, rated_power_kw=rated_kw)
        expected_cubic = rated_kw * np.power((7.5 - 3.0) / (11.5 - 3.0), 3.0)
        self.assertAlmostEqual(r2['power_baseline_kw'], round(expected_cubic, 2), delta=0.5)

        # Region 3: Rated power (11.5 <= v <= 25.0 m/s) -> 3600 kW
        r3 = self.engine.compute_aerodynamic_loss(severity=2, r_R=0.6, area_pct=1.0, wind_speed=14.0, rated_power_kw=rated_kw)
        self.assertEqual(r3['power_baseline_kw'], rated_kw)

        # Region 4: Storm cut-out (> 25.0 m/s) -> 0 kW
        r4 = self.engine.compute_aerodynamic_loss(severity=2, r_R=0.6, area_pct=1.0, wind_speed=26.0, rated_power_kw=rated_kw)
        self.assertEqual(r4['power_baseline_kw'], 0.0)
        self.assertEqual(r4['power_loss_kw'], 0.0)

    def test_aerodynamic_loss_physical_monotonicity(self):
        """Increasing severity and outboard location must monotonically increase aerodynamic loss."""
        # Monotonicity with severity (r/R fixed at 0.80)
        losses_by_sev = []
        for sev in range(1, 6):
            res = self.engine.compute_aerodynamic_loss(severity=sev, r_R=0.80, area_pct=2.0, wind_speed=8.0)
            losses_by_sev.append(res['aep_loss_pct'])
        
        for i in range(len(losses_by_sev) - 1):
            self.assertLess(losses_by_sev[i], losses_by_sev[i + 1], f"Severity {i+1} loss not strictly less than severity {i+2}")

        # Monotonicity with spanwise position r/R across active aerodynamic span
        losses_by_span = []
        for r in [0.50, 0.65, 0.80, 0.95]:
            res = self.engine.compute_aerodynamic_loss(severity=3, r_R=r, area_pct=2.0, wind_speed=8.0)
            losses_by_span.append(res['aep_loss_pct'])

        for i in range(len(losses_by_span) - 1):
            self.assertLess(losses_by_span[i], losses_by_span[i + 1], f"Span position {i} loss not strictly less than {i+1}")

    def test_aerodynamic_calibration_interface_structure(self):
        """Verifies presence and schema of the aerodynamic calibration interface."""
        schema = self.engine.get_aerodynamic_calibration_interface()
        self.assertIn('interface_version', schema)
        self.assertIn('required_airfoil_parameters', schema)
        self.assertIn('defect_geometry_parameters', schema)
        self.assertIn('output_polar_degradation', schema)
        self.assertIn('scientific_reference', schema)
        self.assertIn('Campobasso et al. (2022)', schema['scientific_reference'])
        self.assertIn('10.1002/we.2666', schema['scientific_reference'])

    def test_linked_scada_inspection_schema_structure(self):
        """Verifies presence and completeness of multi-modal matched telemetry schema."""
        schema = self.engine.get_linked_scada_inspection_schema()
        self.assertIn('schema_version', schema)
        self.assertIn('turbine_metadata', schema)
        self.assertIn('inspection_telemetry', schema)
        self.assertIn('synchronized_scada_telemetry', schema)
        self.assertIn('matched_attribution_criteria', schema)
        self.assertIn('turbine_match', schema['matched_attribution_criteria'])
        self.assertIn('temporal_proximity', schema['matched_attribution_criteria'])

    def test_scada_attribution_caveats(self):
        """SCADA cross-validation must disclose operational attribution limitations."""
        res = self.engine.compute_aerodynamic_loss(severity=3, r_R=0.80, area_pct=2.0, wind_speed=8.0)
        scada_val = res.get('scada_validation', {})
        self.assertTrue(scada_val.get('scada_cross_validation_active', False))
        self.assertIn('attribution_caveat', scada_val)
        self.assertIn('wake', scada_val['attribution_caveat'])
        self.assertIn('yaw misalignment', scada_val['attribution_caveat'])

    def test_dynamic_financial_calculations_no_hardcoding(self):
        """Financial loss and payback must calculate dynamically across varied tariffs and repair costs."""
        daily_kwh = 500.0

        # Tariff = 3.50, Repair = 35000 -> Daily loss = 1750, Payback = 20.0 days
        fin1 = self.fin_engine.calculate_losses(daily_kwh, tariff=3.50, repair_cost=35000.0)
        self.assertEqual(fin1['daily_loss_inr'], 1750.0)
        self.assertEqual(fin1['payback_days'], 20.0)
        self.assertEqual(fin1['annual_loss_inr'], round(1750.0 * 365, 2))

        # Tariff = 6.00, Repair = 90000 -> Daily loss = 3000, Payback = 30.0 days
        fin2 = self.fin_engine.calculate_losses(daily_kwh, tariff=6.00, repair_cost=90000.0)
        self.assertEqual(fin2['daily_loss_inr'], 3000.0)
        self.assertEqual(fin2['payback_days'], 30.0)

        # Zero energy loss -> None payback
        fin0 = self.fin_engine.calculate_losses(0.0, tariff=4.50, repair_cost=40000.0)
        self.assertEqual(fin0['daily_loss_inr'], 0.0)
        self.assertIsNone(fin0['payback_days'])

    def test_reproducible_evaluation_metrics_manifest(self):
        """Verifies existence and schema of reproducible evaluation metrics manifest."""
        manifest_path = os.path.join(BASE_DIR, "data", "manifests", "reproducible_evaluation_metrics.json")
        self.assertTrue(os.path.exists(manifest_path), "reproducible_evaluation_metrics.json must exist")
        with open(manifest_path) as f:
            metrics = json.load(f)
        self.assertIn('val', metrics)
        val_01 = metrics['val'].get('0.1', {})
        self.assertIn('checkpoint', val_01)
        self.assertIn('mAP50', val_01)
        self.assertIn('mean_precision', val_01)
        self.assertIn('mean_recall', val_01)
        self.assertIn('per_class', val_01)


if __name__ == '__main__':
    unittest.main()
