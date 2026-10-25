import os
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from services.decision_service import DecisionService
from src.decision.decision_engine import DecisionEngine
from services.damage_service import DamageService
from src.physics.aero_engine import AeroLossPhysicsEngine
from src.financial.financial_engine import FinancialEngine

class TestSafetyOverrides(unittest.TestCase):
    def setUp(self):
        self.engine = DecisionEngine()
        self.service = DecisionService()
        self.damage_svc = DamageService()
        self.aero_engine = AeroLossPhysicsEngine()
        self.fin_engine = FinancialEngine()

    def test_1_severe_fracture_misclassified_as_craze(self):
        """
        REQUIRED TEST 1:
        A visibly severe blade-tip fracture misclassified as CRAZE with low confidence
        must trigger structural safety override (Severity Level 5/5, CRITICAL urgency, ENGINEERING_ASSESSMENT).
        """
        # Replicating the exact problem case: images (3).jpg
        # Image size 277x480, box [43.5, 291.4, 173.4, 424.2] -> 12.97% union frame coverage, r/R = 0.781
        mock_dets = [{
            'damage_type': 'craze',
            'confidence': 0.19,  # Low confidence
            'yolo_class': 'craze',
            'yolo_confidence': 0.19,
            'mobilenet_class': 'craze',
            'mobilenet_confidence': 0.96,
            'bounding_box': [43.5, 291.4, 173.4, 424.2],
            'r_over_R': 0.781
        }]
        char = self.damage_svc.characterize_detections(mock_dets, img_width=277, img_height=480)
        
        # Verify severity is separated from cosmetic 'craze' label
        self.assertEqual(char['severity_proxy'], 5, "Extensive outboard tip anomaly must be escalated to Severity 5")
        self.assertTrue(char['structural_safety_override'])
        self.assertEqual(char['spanwise_position_r_R'], 0.781)
        self.assertAlmostEqual(char['union_frame_coverage_pct'], 12.97, places=1)

        # Verify decision engine output
        res = self.service.evaluate_decision(
            defect_class=char['primary_defect'],
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['damage_area_pct'],
            daily_loss_inr=420.0,
            payback_days=95.3,  # Payback must NOT downgrade decision
            confidence=char['confidence'],
            image_frame_coverage_pct=char['image_frame_coverage_pct'],
            char=char
        )
        self.assertEqual(res['decision'], 'ENGINEERING_ASSESSMENT')
        self.assertEqual(res['urgency'], 'CRITICAL')
        self.assertTrue(res['safety_override_active'])

    def test_2_severe_damage_with_low_model_confidence(self):
        """
        REQUIRED TEST 2:
        Severe visual damage with low model confidence must not be dismissed or downgraded.
        """
        mock_dets = [{
            'damage_type': 'surface_injure',
            'confidence': 0.22,  # Low confidence
            'bounding_box': [50.0, 50.0, 350.0, 400.0],  # ~10.5% coverage
            'r_over_R': 0.72
        }]
        char = self.damage_svc.characterize_detections(mock_dets, img_width=1000, img_height=1000)
        self.assertGreaterEqual(char['severity_proxy'], 4)
        self.assertTrue(char['structural_safety_override'])

        res = self.service.evaluate_decision(
            defect_class=char['primary_defect'],
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['damage_area_pct'],
            daily_loss_inr=350.0,
            payback_days=114.0,
            confidence=char['confidence'],
            image_frame_coverage_pct=char['image_frame_coverage_pct'],
            char=char
        )
        self.assertEqual(res['decision'], 'ENGINEERING_ASSESSMENT')
        self.assertIn(res['urgency'], ['HIGH', 'CRITICAL'])
        self.assertTrue(res['safety_override_active'])

    def test_3_classifier_disagreement_on_potential_structural_defect(self):
        """
        REQUIRED TEST 3:
        When YOLO and MobileNet disagree on a potentially serious defect (e.g. YOLO says craze, MobileNet says crack),
        the safety override must trigger conservative assessment.
        """
        mock_dets = [{
            'damage_type': 'craze',
            'confidence': 0.45,
            'yolo_class': 'craze',
            'yolo_confidence': 0.45,
            'mobilenet_class': 'crack',
            'mobilenet_confidence': 0.52,
            'classification_disagreement': True,
            'bounding_box': [100.0, 100.0, 300.0, 300.0],
            'r_over_R': 0.65,
            'crosscheck': {
                'yolo_class': 'craze',
                'mobilenet_class': 'crack',
                'mobilenet_probabilities': {'crack': 0.52, 'craze': 0.45, 'corrosion': 0.03}
            }
        }]
        char = self.damage_svc.characterize_detections(mock_dets, img_width=1000, img_height=1000)
        self.assertGreaterEqual(char['severity_proxy'], 4)
        self.assertTrue(char['structural_safety_override'])

        res = self.service.evaluate_decision(
            defect_class=char['primary_defect'],
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['damage_area_pct'],
            daily_loss_inr=500.0,
            payback_days=80.0,
            classification_disagreement=True,
            crosscheck=mock_dets[0]['crosscheck'],
            image_frame_coverage_pct=char['image_frame_coverage_pct'],
            char=char
        )
        self.assertEqual(res['decision'], 'ENGINEERING_ASSESSMENT')
        self.assertTrue(res['safety_override_active'])

    def test_4_large_full_frame_coverage_despite_small_individual_boxes(self):
        """
        REQUIRED TEST 4:
        Multiple small bounding boxes whose union covers a large fraction of the frame
        must trigger macro-damage safety override using the union metric, NOT individual box sizes.
        """
        # 5 separate boxes of ~2.5% each distributed across the blade, total union = ~12.5%
        mock_dets = [
            {'damage_type': 'craze', 'confidence': 0.5, 'bounding_box': [10.0, 10.0, 60.0, 500.0], 'r_over_R': 0.75}, # 2.5%
            {'damage_type': 'craze', 'confidence': 0.5, 'bounding_box': [80.0, 10.0, 130.0, 500.0], 'r_over_R': 0.75}, # 2.5%
            {'damage_type': 'craze', 'confidence': 0.5, 'bounding_box': [150.0, 10.0, 200.0, 500.0], 'r_over_R': 0.75}, # 2.5%
            {'damage_type': 'craze', 'confidence': 0.5, 'bounding_box': [220.0, 10.0, 270.0, 500.0], 'r_over_R': 0.75}, # 2.5%
            {'damage_type': 'craze', 'confidence': 0.5, 'bounding_box': [290.0, 10.0, 340.0, 500.0], 'r_over_R': 0.75}, # 2.5%
        ]
        char = self.damage_svc.characterize_detections(mock_dets, img_width=1000, img_height=1000)
        
        # Verify individual box is small (~2.5%) but union is large (~12.5%)
        self.assertLess(char['primary_bbox_frame_coverage_pct'], 3.0)
        self.assertGreaterEqual(char['union_frame_coverage_pct'], 12.0)
        # Verify capped blade damage proxy (10% scale -> ~1.25%) is NOT confused with full-frame metric
        self.assertLess(char['blade_damage_area_proxy_pct'], 2.0)
        
        # In outboard region (r/R=0.75) with 12.5% union coverage, must trigger structural safety override
        self.assertEqual(char['severity_proxy'], 5)
        self.assertTrue(char['structural_safety_override'])

    def test_5_healthy_blade_produces_zero_damage_related_loss(self):
        """
        REQUIRED TEST 5:
        A healthy blade with zero detections must produce strictly zero loss, Severity 0, and MONITOR (LOW).
        """
        char = self.damage_svc.characterize_detections([], img_width=1000, img_height=1000)
        self.assertEqual(char['severity_proxy'], 0)
        self.assertEqual(char['primary_defect'], 'No Defect')
        self.assertEqual(char['damage_area_pct'], 0.0)
        self.assertEqual(char['image_frame_coverage_pct'], 0.0)

        # Physics check
        res_aero = self.aero_engine.compute_aerodynamic_loss(
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['damage_area_pct'],
            wind_speed=7.56,
            rated_power_kw=3600.0,
            capacity_factor=0.36
        )
        self.assertEqual(res_aero['power_loss_kw'], 0.0)
        self.assertEqual(res_aero['daily_energy_loss_kwh'], 0.0)
        self.assertEqual(res_aero['aep_loss_pct'], 0.0)

        # Financial check
        res_fin = self.fin_engine.calculate_losses(
            daily_energy_loss_kwh=res_aero['daily_energy_loss_kwh'],
            tariff=4.50,
            repair_cost=40000.0
        )
        self.assertEqual(res_fin['daily_loss_inr'], 0.0)
        self.assertIsNone(res_fin['payback_days'])

        # Decision check
        res_dec = self.service.evaluate_decision(
            defect_class=char['primary_defect'],
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['damage_area_pct'],
            daily_loss_inr=res_fin['daily_loss_inr'],
            payback_days=999.0,
            char=char
        )
        self.assertEqual(res_dec['decision'], 'MONITOR')
        self.assertEqual(res_dec['urgency'], 'LOW')
        self.assertFalse(res_dec['safety_override_active'])

    def test_6_safety_override_never_downgraded_by_payback(self):
        """
        REQUIRED TEST 6:
        A structural safety override must NEVER be downgraded by favourable or long payback.
        """
        # Case A: Favourable payback (10 days) on structural crack
        res_fav = self.service.evaluate_decision(
            defect_class='crack', severity=5, r_R=0.85, area_pct=2.0,
            daily_loss_inr=4000.0, payback_days=10.0, confidence=0.85
        )
        self.assertEqual(res_fav['decision'], 'ENGINEERING_ASSESSMENT')
        self.assertEqual(res_fav['urgency'], 'CRITICAL')
        self.assertTrue(res_fav['safety_override_active'])

        # Case B: Unfavourable/extended payback (300 days) on structural crack
        res_unfav = self.service.evaluate_decision(
            defect_class='crack', severity=5, r_R=0.85, area_pct=2.0,
            daily_loss_inr=133.0, payback_days=300.0, confidence=0.85
        )
        self.assertEqual(res_unfav['decision'], 'ENGINEERING_ASSESSMENT')
        self.assertEqual(res_unfav['urgency'], 'CRITICAL')
        self.assertTrue(res_unfav['safety_override_active'])

    def test_7_existing_physics_and_financial_formulas_unchanged(self):
        """
        REQUIRED TEST 7:
        Core physics and financial formulas remain identical, unit-consistent, and non-inflated.
        """
        # Verify baseline daily energy = rated * 24 * CF
        rated = 3600.0
        cf = 0.36
        expected_baseline_kwh = rated * 24.0 * cf  # 31,104 kWh/day
        res_aero = self.aero_engine.compute_aerodynamic_loss(
            severity=5, r_R=0.781, area_pct=1.3, wind_speed=7.56,
            rated_power_kw=rated, capacity_factor=cf
        )
        self.assertEqual(res_aero['daily_baseline_energy_kwh'], expected_baseline_kwh)
        
        # Daily loss = baseline * AEP loss % / 100
        expected_daily_loss = round(expected_baseline_kwh * (res_aero['aep_loss_pct'] / 100.0), 2)
        self.assertAlmostEqual(res_aero['daily_energy_loss_kwh'], expected_daily_loss, places=2)

        # Financial daily loss = daily loss * tariff
        tariff = 4.50
        res_fin = self.fin_engine.calculate_losses(
            daily_energy_loss_kwh=res_aero['daily_energy_loss_kwh'],
            tariff=tariff,
            repair_cost=40000.0
        )
        expected_daily_inr = round(res_aero['daily_energy_loss_kwh'] * tariff, 2)
        self.assertEqual(res_fin['daily_loss_inr'], expected_daily_inr)

        # Annual = daily * 365
        self.assertEqual(res_fin['annual_loss_inr'], round(expected_daily_inr * 365.0, 2))

    def test_8_spatial_clustering_overlapping_boxes(self):
        """
        REQUIRED: Spatially overlapping bounding boxes must be clustered into distinct physical regions.
        Prevents multiple YOLO proposals on the same defect from being reported as multiple independent defects.
        """
        # 3 overlapping boxes on the same leading edge damage
        mock_dets = [
            {'damage_type': 'corrosion', 'confidence': 0.85, 'bounding_box': [100.0, 100.0, 200.0, 300.0]},
            {'damage_type': 'corrosion', 'confidence': 0.80, 'bounding_box': [110.0, 120.0, 210.0, 310.0]},
            {'damage_type': 'corrosion', 'confidence': 0.75, 'bounding_box': [105.0, 110.0, 195.0, 290.0]}
        ]
        distinct_count, clusters = self.damage_svc.cluster_detections(mock_dets, iou_thresh=0.35)
        self.assertEqual(distinct_count, 1)
        self.assertEqual(len(clusters[0]), 3)

        char = self.damage_svc.characterize_detections(mock_dets, 1000, 1000)
        self.assertEqual(char['distinct_damage_regions'], 1)
        self.assertEqual(char['candidate_detections_count'], 3)
        self.assertTrue(char['has_overlapping_detections'])

    def test_9_images_5_vertical_erosion_streak(self):
        """
        REQUIRED OBSERVED CASE 1: images (5).jpg
        4 boxes confirmed as surface_injure/corrosion (MobileNet 99.5%), 1 box low-confidence hide_craze (21.7%).
        Primary defect must NOT be hijacked by 21.7% hide_craze.
        Must be categorized as surface erosion (Severity Level 3/5, IEA Task 46 Category 3), NOT Level 5/5 CRITICAL.
        """
        mock_dets = [
            {
                'damage_type': 'surface_injure', 'confidence': 0.52, 'yolo_class': 'surface_injure',
                'bounding_box': [245.0, 48.0, 305.0, 295.0],
                'crosscheck': {'mobilenet_class': 'surface_injure', 'mobilenet_confidence': 0.995},
                'models_agree': True
            },
            {
                'damage_type': 'corrosion', 'confidence': 0.48, 'yolo_class': 'corrosion',
                'bounding_box': [250.0, 50.0, 300.0, 290.0],
                'crosscheck': {'mobilenet_class': 'corrosion', 'mobilenet_confidence': 0.991},
                'models_agree': True
            },
            {
                'damage_type': 'surface_injure', 'confidence': 0.45, 'yolo_class': 'surface_injure',
                'bounding_box': [240.0, 45.0, 310.0, 300.0],
                'crosscheck': {'mobilenet_class': 'surface_injure', 'mobilenet_confidence': 0.985},
                'models_agree': True
            },
            {
                'damage_type': 'surface_injure', 'confidence': 0.40, 'yolo_class': 'surface_injure',
                'bounding_box': [248.0, 55.0, 298.0, 285.0],
                'crosscheck': {'mobilenet_class': 'surface_injure', 'mobilenet_confidence': 0.978},
                'models_agree': True
            },
            {
                'damage_type': 'hide_craze', 'confidence': 0.217, 'yolo_class': 'hide_craze',
                'bounding_box': [242.0, 46.0, 308.0, 298.0],
                'crosscheck': {'mobilenet_class': 'surface_injure', 'mobilenet_confidence': 0.992},
                'models_agree': False,
                'classification_disagreement': True
            }
        ]
        char = self.damage_svc.characterize_detections(mock_dets, img_width=442, img_height=377)
        # Primary must be surface defect, not 21.7% hide_craze
        self.assertIn(char['primary_defect'], ['surface_injure', 'corrosion'])
        self.assertNotEqual(char['primary_defect'], 'hide_craze')
        # All 5 boxes cluster into 1 distinct damage region
        self.assertEqual(char['distinct_damage_regions'], 1)
        self.assertEqual(char['candidate_detections_count'], 5)
        self.assertTrue(char['has_overlapping_detections'])
        # Surface erosion is Severity Level 3/5, not Level 5/5
        self.assertEqual(char['severity_proxy'], 3)
        self.assertFalse(char['structural_safety_override'])

    def test_10_example_source_14_extensive_leading_edge_erosion(self):
        """
        REQUIRED OBSERVED CASE 2: Example-of-leading-edge-erosion-Source-14.jpg
        Extensive leading-edge surface erosion covering 36.4% of frame.
        Box 1: corrosion (MobileNet 99.2%), Box 2: corrosion (MobileNet 86.8%), Box 0: hide_craze (22.2%).
        Primary defect must be corrosion (NOT hide_craze).
        Severity must be Level 4/5 (IEA Task 46 Category 4 Severe Leading Edge Erosion),
        with REPAIR decision, NOT Level 5/5 CRITICAL.
        """
        mock_dets = [
            {
                'damage_type': 'corrosion', 'confidence': 0.88, 'yolo_class': 'corrosion',
                'bounding_box': [50.0, 100.0, 600.0, 250.0],
                'crosscheck': {'mobilenet_class': 'corrosion', 'mobilenet_confidence': 0.992},
                'models_agree': True
            },
            {
                'damage_type': 'corrosion', 'confidence': 0.72, 'yolo_class': 'corrosion',
                'bounding_box': [100.0, 120.0, 550.0, 240.0],
                'crosscheck': {'mobilenet_class': 'corrosion', 'mobilenet_confidence': 0.868},
                'models_agree': True
            },
            {
                'damage_type': 'hide_craze', 'confidence': 0.222, 'yolo_class': 'hide_craze',
                'bounding_box': [60.0, 110.0, 580.0, 245.0],
                'crosscheck': {'mobilenet_class': 'corrosion', 'mobilenet_confidence': 0.950},
                'models_agree': False,
                'classification_disagreement': True
            }
        ]
        char = self.damage_svc.characterize_detections(mock_dets, img_width=678, img_height=351)
        self.assertEqual(char['primary_defect'], 'corrosion')
        self.assertGreaterEqual(char['union_frame_coverage_pct'], 20.0)
        # Extensive leading-edge erosion is Severity Level 4/5 (IEA Category 4), NOT Level 5/5
        self.assertEqual(char['severity_proxy'], 4)

        # Decision engine check: scheduled REPAIR with HIGH urgency
        res = self.service.evaluate_decision(
            defect_class=char['primary_defect'],
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['damage_area_pct'],
            daily_loss_inr=4353.0,
            payback_days=9.2,
            confidence=char['confidence'],
            image_frame_coverage_pct=char['union_frame_coverage_pct'],
            char=char
        )
        self.assertEqual(res['decision'], 'REPAIR')
        self.assertEqual(res['urgency'], 'HIGH')

    def test_11_images_4_subsurface_delamination(self):
        """
        REQUIRED OBSERVED CASE 3: images (4).jpg
        Subsurface delamination confirmed as hide_craze (MobileNet 99.8%) with 31.2% frame coverage.
        Extensive delamination represents serious structural hazard -> Severity Level 5/5, CRITICAL.
        """
        mock_dets = [
            {
                'damage_type': 'hide_craze', 'confidence': 0.998, 'yolo_class': 'hide_craze',
                'bounding_box': [50.0, 30.0, 550.0, 220.0],
                'crosscheck': {'mobilenet_class': 'hide_craze', 'mobilenet_confidence': 0.998},
                'models_agree': True
            },
            {
                'damage_type': 'surface_injure', 'confidence': 0.85, 'yolo_class': 'surface_injure',
                'bounding_box': [100.0, 50.0, 500.0, 200.0],
                'crosscheck': {'mobilenet_class': 'surface_injure', 'mobilenet_confidence': 0.998},
                'models_agree': True
            }
        ]
        char = self.damage_svc.characterize_detections(mock_dets, img_width=600, img_height=333)
        self.assertEqual(char['primary_defect'], 'hide_craze')
        self.assertEqual(char['severity_proxy'], 5)
        self.assertTrue(char['structural_safety_override'])

        res = self.service.evaluate_decision(
            defect_class=char['primary_defect'],
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['damage_area_pct'],
            daily_loss_inr=3765.0,
            payback_days=10.6,
            confidence=char['confidence'],
            image_frame_coverage_pct=char['union_frame_coverage_pct'],
            char=char
        )
        self.assertEqual(res['decision'], 'ENGINEERING_ASSESSMENT')
        self.assertEqual(res['urgency'], 'CRITICAL')
        self.assertTrue(res['safety_override_active'])

    def test_12_blade_defects_2_crazing(self):
        """
        REQUIRED OBSERVED CASE 4: blade_defects-2.webp
        1 detection: craze with low YOLO confidence (21.7%) and 13-14% frame coverage.
        Uncertain craze anomaly -> Level 4/5, ENGINEERING_ASSESSMENT, HIGH urgency.
        """
        mock_dets = [{
            'damage_type': 'craze', 'confidence': 0.217, 'yolo_class': 'craze',
            'yolo_confidence': 0.217,
            'bounding_box': [200.0, 100.0, 1200.0, 370.0],
            'r_over_R': 0.50,
            'crosscheck': {'mobilenet_class': 'craze', 'mobilenet_confidence': 0.731},
            'models_agree': True
        }]
        char = self.damage_svc.characterize_detections(mock_dets, img_width=1920, img_height=1080)
        self.assertEqual(char['primary_defect'], 'craze')
        self.assertEqual(char['severity_proxy'], 4)
        self.assertTrue(char['structural_safety_override'])

        res = self.service.evaluate_decision(
            defect_class=char['primary_defect'],
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['damage_area_pct'],
            daily_loss_inr=1470.0,
            payback_days=27.2,
            confidence=char['confidence'],
            image_frame_coverage_pct=char['union_frame_coverage_pct'],
            char=char
        )
        self.assertEqual(res['decision'], 'ENGINEERING_ASSESSMENT')
        self.assertEqual(res['urgency'], 'HIGH')
        self.assertTrue(res['safety_override_active'])

    def test_13_lightning_damage_misclassified_as_confident_craze(self):
        """
        Observed failure: lightning-damage-wind-turbine_1.webp
        YOLO CRAZE 67.9%, MobileNet ~78% agreement, 63.43% frame coverage, 5% blade-area proxy,
        mid-span r/R from close-up y-heuristic, small financial loss, long payback.
        Must NOT return severity 2 / MONITOR / LOW.
        Must return conservative ENGINEERING_ASSESSMENT / CRITICAL without relabelling as confirmed thunderstrike.
        """
        mock_dets = [{
            'damage_type': 'craze',
            'confidence': 0.679,
            'yolo_class': 'craze',
            'yolo_confidence': 0.679,
            'mobilenet_class': 'craze',
            'mobilenet_confidence': 0.78,
            'models_agree': True,
            'classification_disagreement': False,
            'bounding_box': [0.0, 0.0, 796.43, 796.43],
            'r_over_R': 0.59,
            'crosscheck': {
                'yolo_class': 'craze',
                'yolo_confidence': 0.679,
                'mobilenet_class': 'craze',
                'mobilenet_confidence': 0.78,
                'models_agree': True,
                'mobilenet_probabilities': {
                    'craze': 0.78, 'crack': 0.08, 'thunderstrike': 0.06,
                    'hide_craze': 0.04, 'corrosion': 0.02, 'surface_injure': 0.02
                }
            }
        }]
        char = self.damage_svc.characterize_detections(mock_dets, img_width=1000, img_height=1000)
        self.assertEqual(char['primary_defect'], 'craze')
        self.assertAlmostEqual(char['union_frame_coverage_pct'], 63.43, places=1)
        self.assertEqual(char['blade_damage_area_proxy_pct'], 5.0)
        self.assertNotEqual(char['union_frame_coverage_pct'], char['blade_damage_area_proxy_pct'])
        self.assertGreaterEqual(char['severity_proxy'], 4)
        self.assertTrue(char['structural_safety_override'])
        self.assertIn('does not confirm internal structural failure', char['structural_safety_reason'].lower())

        aero = self.aero_engine.compute_aerodynamic_loss(
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['blade_damage_area_proxy_pct'],
            wind_speed=7.56,
            rated_power_kw=3600.0,
            capacity_factor=0.36
        )
        fin = self.fin_engine.calculate_losses(
            daily_energy_loss_kwh=aero['daily_energy_loss_kwh'],
            tariff=4.50,
            repair_cost=40000.0
        )
        res = self.service.evaluate_decision(
            defect_class=char['primary_defect'],
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['damage_area_pct'],
            daily_loss_inr=fin['daily_loss_inr'],
            payback_days=fin['payback_days'] if fin['payback_days'] is not None else 999.0,
            confidence=char['confidence'],
            image_frame_coverage_pct=char['union_frame_coverage_pct'],
            char=char
        )
        self.assertEqual(res['decision'], 'ENGINEERING_ASSESSMENT')
        self.assertEqual(res['urgency'], 'CRITICAL')
        self.assertTrue(res['safety_override_active'])
        self.assertNotEqual(res['decision'], 'MONITOR')

    def test_14_ordinary_small_craze_is_not_escalated(self):
        """Ordinary small gelcoat craze must remain low severity MONITOR, not CRITICAL."""
        mock_dets = [{
            'damage_type': 'craze',
            'confidence': 0.85,
            'models_agree': True,
            'bounding_box': [100.0, 100.0, 180.0, 160.0],  # 0.48% of 1000x1000
            'r_over_R': 0.55
        }]
        char = self.damage_svc.characterize_detections(mock_dets, img_width=1000, img_height=1000)
        self.assertEqual(char['primary_defect'], 'craze')
        self.assertLess(char['union_frame_coverage_pct'], 1.0)
        self.assertEqual(char['severity_proxy'], 1)
        self.assertFalse(char['structural_safety_override'])

        res = self.service.evaluate_decision(
            defect_class=char['primary_defect'],
            severity=char['severity_proxy'],
            r_R=char['spanwise_position_r_R'],
            area_pct=char['damage_area_pct'],
            daily_loss_inr=40.0,
            payback_days=1000.0,
            confidence=char['confidence'],
            image_frame_coverage_pct=char['union_frame_coverage_pct'],
            char=char
        )
        self.assertEqual(res['decision'], 'MONITOR')
        self.assertEqual(res['urgency'], 'LOW')
        self.assertFalse(res['safety_override_active'])

if __name__ == '__main__':
    unittest.main()
