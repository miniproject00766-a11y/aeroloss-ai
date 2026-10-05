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
from src.financial.financial_engine import FinancialEngine
from src.decision.decision_engine import DecisionEngine

MODELS_DIR = os.path.join(BASE_DIR, "models")

class AeroLossPipeline:
    """
    End-to-end AeroLoss AI Inference Orchestrator.
    Image -> Defect Detection -> Characterization (r/R, Severity)
          -> Physics (Airfoil + BEM) -> Energy Loss (kWh/day)
          -> Financial Loss (INR/day) -> Decision (REPAIR / MONITOR / ASSESSMENT)
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.physics_engine = AeroLossPhysicsEngine()
        self.financial_engine = FinancialEngine()
        self.decision_engine = DecisionEngine()
        
        # Load Vision Classifier
        vision_ckpt_path = os.path.join(MODELS_DIR, "blade_defect_classifier.pth")
        if os.path.exists(vision_ckpt_path):
            ckpt = torch.load(vision_ckpt_path, map_location=self.device)
            self.class_names = ckpt['class_names']
            self.vision_model = models.mobilenet_v3_small(weights=None)
            self.vision_model.classifier[3] = nn.Linear(self.vision_model.classifier[3].in_features, len(self.class_names))
            self.vision_model.load_state_dict(ckpt['model_state_dict'])
            self.vision_model.to(self.device)
            self.vision_model.eval()
        else:
            self.vision_model = None
            self.class_names = ['corrosion', 'crack', 'craze', 'hide_craze', 'surface_injure', 'thunderstrike']

        # Vision Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load ML Decision Model (prioritize realistic calibrated model)
        decision_path_realistic = os.path.join(MODELS_DIR, "decision_classifier_realistic.joblib")
        decision_path = os.path.join(MODELS_DIR, "decision_classifier.joblib")
        if os.path.exists(decision_path_realistic):
            self.ml_decision_model = joblib.load(decision_path_realistic)
        elif os.path.exists(decision_path):
            self.ml_decision_model = joblib.load(decision_path)
        else:
            self.ml_decision_model = None

    def analyze_image(self, image_input, r_R=None, area_pct=None, wind_speed=7.56, rated_kw=3600.0, tariff=4.50, repair_cost=40000.0):
        """
        Runs the full 7-stage AeroLoss pipeline on a blade defect image.
        """
        # 1. Load image
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert('RGB')
        else:
            raise ValueError("image_input must be file path or PIL Image")

        # 2. Vision Inference
        if self.vision_model is not None:
            tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.vision_model(tensor_img)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                pred_idx = int(np.argmax(probs))
                pred_class = self.class_names[pred_idx]
                confidence = float(probs[pred_idx])
                all_probs = {self.class_names[i]: round(float(probs[i]), 4) for i in range(len(self.class_names))}
        else:
            pred_class = 'surface_injure'
            confidence = 0.90
            all_probs = {pred_class: confidence}

        # 3. Damage Characterization Proxy
        base_sev_map = {
            'craze': 1, 'surface_injure': 2, 'corrosion': 2,
            'hide_craze': 3, 'crack': 4, 'thunderstrike': 5
        }
        base_sev = base_sev_map.get(pred_class, 2)
        
        # Default r/R and area if not explicitly supplied
        effective_r_R = float(r_R) if r_R is not None else 0.88
        effective_area = float(area_pct) if area_pct is not None else 4.2

        if effective_area > 5.0:
            severity = min(5, base_sev + 1)
        elif effective_area < 0.5:
            severity = max(1, base_sev - 1)
        else:
            severity = base_sev

        # 4. Aerodynamic & Physics Engine
        aero_results = self.physics_engine.compute_aerodynamic_loss(
            severity=severity,
            r_R=effective_r_R,
            area_pct=effective_area,
            wind_speed=wind_speed,
            rated_power_kw=rated_kw
        )

        # 5. Energy Analytics
        aep_loss_pct = aero_results['aep_loss_pct']
        daily_energy_loss_kwh = aero_results['daily_energy_loss_kwh']
        power_loss_kw = aero_results['power_loss_kw']

        # 6. Financial Analytics
        fin_results = self.financial_engine.calculate_losses(
            daily_energy_loss_kwh=daily_energy_loss_kwh,
            tariff=tariff,
            repair_cost=repair_cost
        )

        # 7. Maintenance Decision Support
        rule_decision = self.decision_engine.evaluate_action(
            defect_class=pred_class,
            severity=severity,
            r_R=effective_r_R,
            area_pct=effective_area,
            daily_loss_inr=fin_results['daily_loss_inr'],
            payback_days=fin_results['payback_days'],
            confidence=confidence
        )

        # Cross-validate with trained ML Decision model
        ml_decision_pred = None
        if self.ml_decision_model is not None:
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

        return {
            'visual_detection': {
                'detected_class': pred_class,
                'confidence': round(confidence, 4),
                'confidence_pct': f"{confidence*100:.1f}%",
                'class_probabilities': all_probs
            },
            'characterization': {
                'severity_proxy': severity,
                'spanwise_position_r_R': round(effective_r_R, 3),
                'damage_area_pct': round(effective_area, 2),
                'blade_region': 'Outboard (High Speed)' if effective_r_R >= 0.80 else ('Mid-Span' if effective_r_R >= 0.50 else 'Root/Inboard')
            },
            'physics_aerodynamics': {
                'airfoil_reference': 'FFA-W3-241 (Re=1e7)',
                'wind_speed_ms': round(wind_speed, 2),
                'baseline_power_kw': aero_results['power_baseline_kw'],
                'damaged_power_kw': aero_results['power_damaged_kw'],
                'estimated_power_loss_kw': power_loss_kw,
                'spanwise_weight_factor': aero_results['spanwise_factor']
            },
            'energy_impact': {
                'aep_loss_pct': aep_loss_pct,
                'daily_energy_loss_kwh': daily_energy_loss_kwh,
                'monthly_energy_loss_kwh': round(daily_energy_loss_kwh * 30.0, 1),
                'annual_energy_loss_kwh': round(daily_energy_loss_kwh * 365.0, 1)
            },
            'financial_impact': {
                'tariff_inr_kwh': tariff,
                'repair_cost_inr': repair_cost,
                'daily_financial_loss_inr': fin_results['daily_loss_inr'],
                'monthly_financial_loss_inr': fin_results['monthly_loss_inr'],
                'annual_financial_loss_inr': fin_results['annual_loss_inr'],
                'payback_days': fin_results['payback_days']
            },
            'maintenance_decision': {
                'recommended_action': final_decision,
                'urgency': rule_decision['urgency'],
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
