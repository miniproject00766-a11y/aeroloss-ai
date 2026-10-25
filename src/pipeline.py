import os
import sys
import json
import joblib
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# Set project base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.physics.aero_engine import AeroLossPhysicsEngine
from src.physics.fatigue_engine import DropletFatigueEngine
from src.financial.financial_engine import FinancialEngine
from src.decision.decision_engine import DecisionEngine
from src.domain_guardrail import RobustDomainGuardrail
from src.image_io import load_rgb_image
from src.model1.image_quality_validator import ImageQualityValidator
from services.damage_service import DamageService

MODELS_DIR = os.path.join(BASE_DIR, "models")

class AeroLossPipeline:
    """
    End-to-end AeroLoss AI Inference Orchestrator.
    Image -> Defect Detection -> Characterization (r/R, Severity)
          -> Physics (Airfoil + BEM) -> Energy Loss (kWh/day)
          -> Financial Loss (INR/day) -> Decision (REPAIR / MONITOR / ASSESSMENT)
          -> Droplet Fatigue & Erosion-Safe Mode (ESM)
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.physics_engine = AeroLossPhysicsEngine()
        self.fatigue_engine = DropletFatigueEngine()
        self.financial_engine = FinancialEngine()
        self.decision_engine = DecisionEngine()
        self.damage_service = DamageService()
        self.image_quality_validator = ImageQualityValidator()
        self.yolo_service = None
        
        # Load Vision Classifier (6-class defect classifier)
        vision_ckpt_path = os.path.join(MODELS_DIR, "blade_defect_classifier.pth")
        if os.path.exists(vision_ckpt_path):
            ckpt = torch.load(vision_ckpt_path, map_location=self.device)
            self.class_names = ckpt.get('class_names', ['corrosion', 'crack', 'craze', 'hide_craze', 'surface_injure', 'thunderstrike'])
            self.temperature = float(ckpt.get('temperature', 1.4722))
            self.vision_model = models.mobilenet_v3_small(weights=None)
            self.vision_model.classifier[3] = nn.Linear(self.vision_model.classifier[3].in_features, len(self.class_names))
            self.vision_model.load_state_dict(ckpt['model_state_dict'])
            self.vision_model.to(self.device)
            self.vision_model.eval()
        else:
            self.vision_model = None
            self.temperature = 1.0
            self.class_names = ['corrosion', 'crack', 'craze', 'hide_craze', 'surface_injure', 'thunderstrike']

        # Vision Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize Robust Multi-Signal Domain Guardrail
        self.guardrail = RobustDomainGuardrail(device=self.device)

        # Load ML Decision Model (prioritize realistic calibrated model pipeline)
        decision_path_realistic = os.path.join(MODELS_DIR, "decision_classifier_realistic.joblib")
        decision_path = os.path.join(MODELS_DIR, "decision_classifier.joblib")
        if os.path.exists(decision_path_realistic):
            try:
                self.ml_decision_model = joblib.load(decision_path_realistic)
            except Exception:
                self.ml_decision_model = None
        elif os.path.exists(decision_path):
            try:
                self.ml_decision_model = joblib.load(decision_path)
            except Exception:
                self.ml_decision_model = None
        else:
            self.ml_decision_model = None

    def validate_blade_domain(self, pil_img, probs=None, confidence=None, threshold=0.45, top2_threshold=0.75):
        """
        Out-of-Distribution (OOD) Domain Guardrail using RobustDomainGuardrail.
        Returns:
            is_valid: bool (True for ACCEPT and REVIEW_REQUIRED, False for REJECT_OUT_OF_DOMAIN)
            reason: str
            status: 'ACCEPT' | 'REJECT_OUT_OF_DOMAIN' | 'REVIEW_REQUIRED'
            guardrail_metrics: dict
        """
        status, conf, reason, metrics = self.guardrail.evaluate(pil_img, classifier_probs=probs)
        # In binary legacy contexts, ACCEPT and REVIEW_REQUIRED allow downstream processing
        is_valid = (status != 'REJECT_OUT_OF_DOMAIN')
        return is_valid, reason, status, metrics

    def analyze_image(self, image_input, r_R=None, area_pct=None, wind_speed=7.56, rated_kw=3600.0, tariff=4.50, repair_cost=40000.0):
        """
        Runs the full 7-stage AeroLoss pipeline on a blade defect image.
        """
        # 1. Load image and run the same localization/classification path as the API.
        pil_img = load_rgb_image(image_input)

        if self.yolo_service is None:
            from services.yolo_service import YoloService
            self.yolo_service = YoloService(pipeline=self)

        detections, annotated_img, is_valid, validation_msg, guard_status = self.yolo_service.detect_defects(pil_img)

        if not is_valid:
            return {
                'is_valid_blade_image': False,
                'status': 'REJECTED_OUT_OF_DOMAIN',
                'guardrail_decision': 'REJECT_OUT_OF_DOMAIN',
                'error': 'Uploaded image is not recognized as a wind turbine blade.',
                'rejection_reason': validation_msg,
                'guardrail_metrics': {},
                'advice': 'Please upload an authentic, clear drone or borescope photograph of an industrial wind turbine blade surface.'
            }

        # 2. Characterize detector boxes once; frame geometry and blade-area proxy
        # remain separate and carry the same provenance as the API response.
        char = self.damage_service.characterize_detections(
            detections,
            img_width=pil_img.width,
            img_height=pil_img.height
        )
        pred_class = char['primary_defect']
        conf_raw = char.get('confidence')
        confidence = float(conf_raw) if conf_raw is not None else 1.0
        severity = int(char['severity_proxy'])

        # Explicit user overrides affect model inputs only; they do not replace the
        # measured detector geometry or the coverage used by the safety rule.
        geometry_r_R = char.get('spanwise_position_r_R')
        geometry_r_R_available = bool(char.get('spanwise_position_available', geometry_r_R is not None))
        effective_r_R = float(r_R) if r_R is not None else (
            float(geometry_r_R) if geometry_r_R_available else 0.88
        )
        effective_area = float(area_pct) if area_pct is not None else float(char['damage_area_pct'])
        decision_r_R = float(r_R) if r_R is not None else (
            float(geometry_r_R) if geometry_r_R_available else 0.50
        )
        all_probs = char.get('crosscheck', {}).get('mobilenet_probabilities', {})

        # 3. Aerodynamic & Physics Engine. Values are provisional estimates and
        # do not establish calibrated blade surface area or defect-to-energy loss.
        aero_results = self.physics_engine.compute_aerodynamic_loss(
            severity=severity,
            r_R=effective_r_R,
            area_pct=effective_area,
            wind_speed=wind_speed,
            rated_power_kw=rated_kw
        )

        # 4. Energy Analytics
        aep_loss_pct = aero_results['aep_loss_pct']
        daily_energy_loss_kwh = aero_results['daily_energy_loss_kwh']
        power_loss_kw = aero_results['power_loss_kw']

        # 5. Financial Analytics
        fin_results = self.financial_engine.calculate_losses(
            daily_energy_loss_kwh=daily_energy_loss_kwh,
            tariff=tariff,
            repair_cost=repair_cost
        )

        # 6. Safety decision precedes financial/payback rules.
        rule_decision = self.decision_engine.evaluate_action(
            defect_class=pred_class,
            severity=severity,
            r_R=decision_r_R,
            area_pct=effective_area,
            daily_loss_inr=fin_results['daily_loss_inr'],
            payback_days=fin_results['payback_days'] if fin_results['payback_days'] is not None else 999.0,
            confidence=confidence,
            image_frame_coverage_pct=char.get('union_frame_coverage_pct', 0.0),
            is_structural_override=char.get('structural_safety_override', False),
            unsupported_class=char.get('unsupported_class', False)
        )

        # Cross-validate with trained ML Decision model
        ml_decision_pred = None
        if self.ml_decision_model is not None and not rule_decision.get('safety_override_active', False):
            feat_df = pd.DataFrame([{
                'defect_class': pred_class,
                'severity': severity,
                'r_R': effective_r_R,
                'area_pct': effective_area,
                'wind_speed': wind_speed,
                'aep_loss_pct': aep_loss_pct,
                'daily_loss_inr': fin_results['daily_loss_inr'],
                'repair_cost_inr': repair_cost,
                'payback_days': fin_results['payback_days'],
                'confidence': confidence
            }])
            ml_decision_pred = str(self.ml_decision_model.predict(feat_df)[0])

        final_decision = rule_decision['decision']

        # 7. Springer Droplet Fatigue & Dynamic Erosion-Safe Mode (ESM)
        fatigue_forecast = self.fatigue_engine.forecast_progression(
            initial_severity=severity,
            r_R=effective_r_R,
            wind_speed=wind_speed
        )
        esm_status = self.fatigue_engine.compute_erosion_safe_mode(
            rain_intensity_mmh=4.0,
            r_R=effective_r_R,
            severity=severity
        )

        return {
            'is_valid_blade_image': True,
            'status': 'SUCCESS' if guard_status == 'ACCEPT' else 'REVIEW_REQUIRED',
            'guardrail_decision': guard_status,
            'guardrail_metrics': {},
            'visual_detection': {
                'detected_class': pred_class,
                'confidence': round(confidence, 4) if detections else None,
                'confidence_pct': f"{confidence*100:.1f}%" if detections else 'N/A',
                'class_probabilities': all_probs,
                'temperature_calibration_applied': True,
                'detection_status': 'DETECTIONS_FOUND' if detections else 'NO_DEFECT_DETECTED',
                'crosscheck': char.get('crosscheck', {}),
                'detections': detections
            },
            'characterization': {
                **char,
                'health_confirmed': False,
                'geometry_status': char.get(
                    'geometry_status',
                    'PROVISIONAL_IMAGE_GEOMETRY' if detections else 'UNAVAILABLE_NO_DETECTIONS'
                ),
                'spanwise_position_source': char.get('provenance', {}).get(
                    'spanwise_position_source',
                    char.get('spanwise_position_source', 'UNAVAILABLE_NO_DETECTIONS')
                ),
                'model_inputs': {
                    'spanwise_position_r_R': round(effective_r_R, 3),
                    'spanwise_position_source': (
                        'USER_OVERRIDE' if r_R is not None else
                        ('DETECTION_ESTIMATE' if geometry_r_R_available else 'PROVISIONAL_COMPUTATIONAL_DEFAULT')
                    ),
                    'damage_area_pct': round(effective_area, 3),
                    'damage_area_source': (
                        'USER_OVERRIDE' if area_pct is not None else
                        char.get('provenance', {}).get('damage_area_source', 'UNAVAILABLE_NO_DETECTIONS')
                    )
                }
            },
            'physics_aerodynamics': {
                'airfoil_reference': 'FFA-W3-241 (Re=1e7)',
                'wind_speed_ms': round(wind_speed, 2),
                'baseline_power_kw': aero_results['power_baseline_kw'],
                'damaged_power_kw': aero_results['power_damaged_kw'],
                'estimated_power_loss_kw': power_loss_kw,
                'spanwise_weight_factor': aero_results['spanwise_factor'],
                'scada_validation': aero_results.get('scada_validation', {}),
                'estimate_status': 'PROVISIONAL_UNCALIBRATED_ESTIMATE'
            },
            'energy_impact': {
                'aep_loss_pct': aep_loss_pct,
                'daily_energy_loss_kwh': daily_energy_loss_kwh,
                'monthly_energy_loss_kwh': round(daily_energy_loss_kwh * 30.0, 1),
                'annual_energy_loss_kwh': round(daily_energy_loss_kwh * 365.0, 1),
                'estimate_status': 'PROVISIONAL_UNCALIBRATED_ESTIMATE'
            },
            'financial_impact': {
                'tariff_inr_kwh': tariff,
                'repair_cost_inr': repair_cost,
                'daily_financial_loss_inr': fin_results['daily_loss_inr'],
                'monthly_financial_loss_inr': fin_results['monthly_loss_inr'],
                'annual_financial_loss_inr': fin_results['annual_loss_inr'],
                'payback_days': fin_results['payback_days'],
                'estimate_status': 'PROVISIONAL_UNCALIBRATED_ESTIMATE'
            },
            'fatigue_prognostics': {
                'model': 'George Springer Droplet Impingement Fatigue (1976)',
                'relative_velocity_ms': fatigue_forecast['relative_velocity_ms'],
                'water_hammer_pressure_mpa': fatigue_forecast['water_hammer_pressure_mpa'],
                'days_to_next_severity_level': fatigue_forecast['days_to_next_severity_level'],
                'projected_severity_180d': fatigue_forecast['projected_severity'][-1],
                'cost_day_0_inr': fatigue_forecast['cost_day_0_inr'],
                'cost_day_180_inr': fatigue_forecast['cost_day_180_inr'],
                'cost_escalation_ratio': fatigue_forecast['cost_escalation_ratio'],
                'trajectory_time_days': fatigue_forecast['time_points_days'],
                'trajectory_cost_inr': fatigue_forecast['cost_of_inaction_inr']
            },
            'erosion_safe_mode': esm_status,
            'maintenance_decision': {
                'recommended_action': final_decision,
                'urgency': rule_decision['urgency'],
                'safety_override_active': rule_decision.get('safety_override_active', False),
                'metrics_snapshot': rule_decision.get('metrics_snapshot', {}),
                'ml_classifier_concurrence': ml_decision_pred,
                'reasoning': rule_decision['reasons']
            }
        }

if __name__ == "__main__":
    pipeline = AeroLossPipeline()
    # Test on a real test patch
    test_patch = os.path.join(BASE_DIR, "data", "processed", "patches", "crack", os.listdir(os.path.join(BASE_DIR, "data", "processed", "patches", "crack"))[0])
    print(f"Testing pipeline on patch: {test_patch}")
    res = pipeline.analyze_image(test_patch, r_R=0.89, area_pct=4.5, wind_speed=8.5, tariff=4.50, repair_cost=40000.0)
    print(json.dumps(res, indent=2))
