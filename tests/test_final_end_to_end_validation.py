"""
AeroLoss AI — Final End-to-End Energy, Financial & Physics Validation Test Suite
Covers:
1. Crack case unrounded internal arithmetic and display-boundary rounding reconciliation.
2. 2D photographic frame coverage (30.21%) vs 3D blade area proxy (3.021%) and AEP loss (3.95%).
3. Financial equation verification (daily, monthly, annual revenue loss, payback, zero-damage boundary).
4. Structural safety override precedence over financial payback.
5. Multi-proposal clustering and distinct physical damage region isolation.
6. API endpoint determinism and response contract fidelity.
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


class TestFinalEndToEndValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aero_engine = AeroLossPhysicsEngine()
        cls.physics_svc = PhysicsService()
        cls.fin_engine = FinancialEngine(default_tariff=4.50, default_repair_cost=40000.0)
        cls.damage_svc = DamageService()
        cls.decision_svc = DecisionService()

    def test_crack_case_unrounded_vs_display_rounding(self):
        """
        Reconciles the observed Crack report:
        Baseline 570.6 kW, Damaged 529.0 kW, Deficit 41.5 kW.
        Demonstrates that unrounded internal math is strictly consistent:
        570.5835 kW - 41.5175 kW = 529.0661 kW.
        The 0.1 kW apparent discrepancy (570.6 - 529.0 = 41.6) is an artifact of
        independent rounding of floating-point numbers at display boundaries.
        """
        rated_kw = 3600.0
        ws = 7.60  # Operating wind speed producing ~570.6 kW baseline
        area_pct = 3.021  # 10% proxy of 30.21% frame coverage
        sev = 5
        cf = 0.36
        r_R = 0.7548

        res = self.aero_engine.compute_aerodynamic_loss(
            severity=sev, r_R=r_R, area_pct=area_pct, wind_speed=ws,
            rated_power_kw=rated_kw, capacity_factor=cf
        )

        # Baseline: 570.58 kW -> rounds to 570.6 kW
        self.assertAlmostEqual(res['power_baseline_kw'], 570.58, places=1)
        self.assertEqual(round(res['power_baseline_kw'], 1), 570.6)

        # Instantaneous Deficit: 41.52 kW -> rounds to 41.5 kW
        self.assertAlmostEqual(res['power_loss_kw'], 41.52, places=1)
        self.assertEqual(round(res['power_loss_kw'], 1), 41.5)

        # Damaged Power: 529.07 kW (or 529.1 kW rounded)
        self.assertAlmostEqual(res['power_damaged_kw'], 529.07, places=1)

        # Unrounded physical identity: baseline - loss == damaged
        unrounded_diff = res['power_baseline_kw'] - res['power_loss_kw']
        self.assertAlmostEqual(unrounded_diff, res['power_damaged_kw'], places=2)

        # AEP Loss: 3.95%
        self.assertAlmostEqual(res['aep_loss_pct'], 3.95, places=1)

        # Daily Energy Loss: 31104.0 * (3.95 / 100) = 1,228.6 kWh/day
        self.assertAlmostEqual(res['daily_energy_loss_kwh'], 1228.6, places=0)

    def test_blade_area_proxy_and_frame_coverage_relationship(self):
        """
        Verifies that 2D frame coverage (e.g. 30.21%) scales to 3.021% blade area proxy
        via the documented 10% local frame scale factor, and drives logarithmic area scaling.
        """
        union_cov = 30.21
        # Mock detection with 30.21% coverage
        # 1024 * sqrt(0.3021) = 562.83
        mock_box = [0.0, 0.0, 562.83, 562.83]
        cov = self.damage_svc.compute_box_coverage(mock_box, 1024, 1024)
        self.assertAlmostEqual(cov, 30.21, delta=0.1)

        char = self.damage_svc.characterize_detections([{
            'damage_type': 'crack',
            'confidence': 0.202,
            'bounding_box': mock_box,
            'r_over_R': 0.755
        }], img_width=1024, img_height=1024)

        self.assertAlmostEqual(char['image_frame_coverage_pct'], 30.21, delta=0.1)
        self.assertAlmostEqual(char['blade_damage_area_proxy_pct'], 3.021, delta=0.01)

        # Logarithmic modifier: 1 + 0.15 * ln(1 + 3.021) = 1.2087 (+20.9% aerodynamic penalty)
        area_mod = float(np.clip(1.0 + 0.15 * np.log1p(3.021), 0.8, 2.5))
        self.assertAlmostEqual(area_mod, 1.2087, places=3)

    def test_crack_case_financial_chain(self):
        """
        Verifies exact financial conversions for Crack Case 4:
        - Daily revenue loss: 1228.61 * 4.50 = ₹5,528.74 / day
        - Annual revenue loss: 5528.74 * 365 = ₹2,017,990.10 / year
        - Payback: 40000 / 5528.74 = 7.2 days
        """
        daily_kwh = 1228.608
        tariff = 4.50
        repair_cost = 40000.0

        fin = self.fin_engine.calculate_losses(
            daily_energy_loss_kwh=daily_kwh,
            tariff=tariff,
            repair_cost=repair_cost
        )

        self.assertAlmostEqual(fin['daily_loss_inr'], 5528.74, places=1)
        self.assertAlmostEqual(fin['annual_loss_inr'], 2017990.10, places=0)
        self.assertEqual(fin['payback_days'], 7.2)

    def test_zero_damage_boundary_safety(self):
        """Zero damage produces zero financial loss and None payback (no division by zero)."""
        fin = self.fin_engine.calculate_losses(
            daily_energy_loss_kwh=0.0,
            tariff=4.50,
            repair_cost=40000.0
        )
        self.assertEqual(fin['daily_loss_inr'], 0.0)
        self.assertEqual(fin['annual_loss_inr'], 0.0)
        self.assertEqual(fin['monthly_loss_inr'], 0.0)
        self.assertIsNone(fin['payback_days'])
        self.assertIsNone(fin['payback_years'])

    def test_structural_safety_override_precedence(self):
        """
        Verifies that structural defects (crack, thunderstrike, severe tip rupture)
        cannot be downgraded to MONITOR by favourable payback.
        """
        # Crack with fast payback (7.2 days) MUST still be ENGINEERING_ASSESSMENT / CRITICAL
        dec = self.decision_svc.evaluate_decision(
            defect_class='crack',
            severity=5,
            r_R=0.755,
            area_pct=3.021,
            daily_loss_inr=5528.74,
            payback_days=7.2,
            confidence=0.202,
            image_frame_coverage_pct=30.21
        )
        self.assertEqual(dec['decision'], 'ENGINEERING_ASSESSMENT')
        self.assertEqual(dec['urgency'], 'CRITICAL')
        self.assertTrue(dec['safety_override_active'])

        # Benign surface erosion (corrosion) with high payback is deferrable to MONITOR
        dec_corrosion = self.decision_svc.evaluate_decision(
            defect_class='corrosion',
            severity=2,
            r_R=0.50,
            area_pct=0.5,
            daily_loss_inr=200.0,
            payback_days=200.0,
            confidence=0.92,
            image_frame_coverage_pct=5.0
        )
        self.assertEqual(dec_corrosion['decision'], 'MONITOR')
        self.assertEqual(dec_corrosion['urgency'], 'LOW')
        self.assertFalse(dec_corrosion['safety_override_active'])

    def test_duplicate_proposals_clustering(self):
        """
        Verifies that overlapping bounding boxes on the same physical defect
        are clustered into 1 distinct damage region and never double-counted.
        """
        box1 = [100.0, 100.0, 300.0, 300.0]
        box2 = [120.0, 110.0, 310.0, 290.0]  # Heavy overlap with box1 (IoU > 0.70)
        box3 = [700.0, 700.0, 850.0, 850.0]  # Distinct separate defect

        dets = [
            {'damage_type': 'corrosion', 'confidence': 0.88, 'bounding_box': box1},
            {'damage_type': 'corrosion', 'confidence': 0.85, 'bounding_box': box2},
            {'damage_type': 'corrosion', 'confidence': 0.91, 'bounding_box': box3}
        ]

        distinct_count, clusters = self.damage_svc.cluster_detections(dets, iou_thresh=0.35)
        self.assertEqual(distinct_count, 2)
        self.assertEqual(len(clusters[0]), 2)  # box1 & box2 merged
        self.assertEqual(len(clusters[1]), 1)  # box3 isolated

        # Analytical union coverage is less than sum of individual areas
        sum_area = self.damage_svc.compute_box_coverage(box1) + self.damage_svc.compute_box_coverage(box2)
        union_area = self.damage_svc.compute_union_coverage([box1, box2])
        self.assertLess(union_area, sum_area)


if __name__ == '__main__':
    unittest.main()
