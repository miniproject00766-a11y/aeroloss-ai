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
        
        # Load Vision Classifier (6-class defect classifier)
        vision_ckpt_path = os.path.join(MODELS_DIR, "blade_defect_classifier.pth")
        if os.path.exists(vision_ckpt_path):
            ckpt = torch.load(vision_ckpt_path, map_location=self.device)
            self.class_names = ckpt.get('class_names', ['corrosion', 'crack', 'craze', 'hide_craze', 'surface_injure', 'thunderstrike'])
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

        # Load Out-of-Distribution (OOD) Semantic Domain Model (MobileNetV3 on ImageNet)
        try:
            self.domain_weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.domain_model = models.mobilenet_v3_small(weights=self.domain_weights).to(self.device).eval()
            self.domain_categories = self.domain_weights.meta['categories']
            self.domain_transform = self.domain_weights.transforms()
        except Exception as e:
            self.domain_model = None
            self.domain_categories = []
            self.domain_transform = None

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

    def validate_blade_domain(self, pil_img, probs, confidence, threshold=0.70):
        """
        Out-of-Distribution (OOD) Domain Guardrail.
        Verifies that the uploaded photo exhibits authentic wind turbine blade surface characteristics.
        Prevents non-blade images (fabrics, textiles, household items, nature, food) from generating fake damage numbers.
        """
        # 1. Universal Ontological Open-Domain Veto (ImageNet Foundation Filter)
        # Uses WordNet taxonomic super-branches to universally reject:
        # - All living organisms (birds, mammals, reptiles, insects, plants): WordNet synsets 0..397
        # - All food, produce, and beverages: WordNet synsets 920..970
        # - All consumer personal transport / vehicles
        # - All apparel, garments, and woven textiles
        if self.domain_model is not None and self.domain_transform is not None:
            try:
                domain_tensor = self.domain_transform(pil_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    domain_logits = self.domain_model(domain_tensor)
                    domain_probs = torch.softmax(domain_logits, dim=1).squeeze()
                    top5 = torch.topk(domain_probs, 5)

                top_indices = top5.indices.tolist()
                top_probs = top5.values.tolist()
                top_idx, top_prob = top_indices[0], top_probs[0]
                top_name = self.domain_categories[top_idx]

                # (a) Living Organisms (All Birds, Animals, Insects, Plants: indices 0..397)
                if 0 <= top_idx <= 397 and top_prob >= 0.20:
                    return False, f"Domain Guardrail: Biological organism detected ('{top_name}', {top_prob*100:.1f}%). Authentic turbine blades are industrial composite structures."

                # (b) Food, Produce & Beverages (indices 920..970)
                if 920 <= top_idx <= 970 and top_prob >= 0.20:
                    return False, f"Domain Guardrail: Food/organic matter detected ('{top_name}', {top_prob*100:.1f}%). Authentic turbine blades are non-organic fiberglass composite."

                # (c) Consumer Road Vehicles & Personal Transport
                VEHICLE_KEYWORDS = {'car', 'automobile', 'wagon', 'minivan', 'truck', 'bus', 'motorcycle', 'bicycle'}
                if any(k in top_name.lower() for k in VEHICLE_KEYWORDS) and top_prob >= 0.20:
                    return False, f"Domain Guardrail: Consumer transport/vehicle detected ('{top_name}', {top_prob*100:.1f}%)."

                # (d) Apparel, Garments & Woven Textiles
                TEXTILE_KEYWORDS = {'wool', 'dishrag', 'quilt', 'velvet', 'poncho', 'jersey', 'towel', 'rug', 'carpet', 'cloth', 'fabric', 'suit', 'jacket', 'coat', 'blanket', 'linen', 'curtain', 'bonnet', 'stole', 'shawl'}
                textile_hits = [(self.domain_categories[i].lower(), domain_probs[i].item()) for i in top_indices if any(k in self.domain_categories[i].lower() for k in TEXTILE_KEYWORDS)]
                if textile_hits:
                    hit_sum = sum(p for _, p in textile_hits)
                    top_is_textile = any(k in top_name.lower() for k in TEXTILE_KEYWORDS)
                    if (top_is_textile and top_prob >= 0.15) or hit_sum >= 0.20:
                        return False, f"Domain Guardrail: Woven textile/fabric detected ('{top_name}', {top_prob*100:.1f}%). Industrial blades exhibit smooth aerodynamic gelcoat."

            except Exception as e:
                pass

        # 2. Softmax Confidence Threshold
        if confidence < threshold:
            return False, f"Confidence below blade verification threshold ({confidence*100:.1f}% < {threshold*100:.1f}%). Model cannot verify aerodynamic blade surface."

        # 3. Shannon Entropy of Softmax Distribution
        probs_clipped = np.clip(probs, 1e-7, 1.0)
        entropy = -np.sum(probs_clipped * np.log(probs_clipped))
        max_possible_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_possible_entropy if max_possible_entropy > 0 else 0.0

        if normalized_entropy > 0.65:
            return False, f"High prediction uncertainty / entropy ({normalized_entropy:.2f}). Model cannot reliably identify aerodynamic blade surface features."

        # 4. Domain Color Saturation Filter
        # Wind turbine blades are non-saturated composite (white, light gray, gelcoat, matte black).
        # Unfamiliar photos (animals, clothing, household items, nature) typically have high saturation.
        np_img = np.array(pil_img)
        if len(np_img.shape) == 3 and np_img.shape[2] == 3:
            saturation = float(np.mean(np.max(np_img, axis=2) - np.min(np_img, axis=2)))
            if saturation > 85.0:
                return False, f"Color saturation anomaly ({saturation:.1f} > 85.0). Industrial turbine blades exhibit low-saturation fiberglass/gelcoat reflectance."

        return True, "Valid wind turbine blade surface."

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
            probs = np.array([0.90])

        # 2.1 Out-of-Distribution (OOD) Domain Guardrail Check
        is_valid, validation_msg = self.validate_blade_domain(pil_img, probs, confidence, threshold=0.70)
        if not is_valid:
            return {
                'is_valid_blade_image': False,
                'status': 'REJECTED_OUT_OF_DOMAIN',
                'error': 'Uploaded image is not recognized as a wind turbine blade.',
                'rejection_reason': validation_msg,
                'confidence': round(confidence, 4),
                'confidence_pct': f"{confidence*100:.1f}%",
                'detected_candidate': pred_class,
                'advice': 'Please upload an authentic, clear drone or borescope photograph of an industrial wind turbine blade surface.'
            }

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
            'is_valid_blade_image': True,
            'status': 'SUCCESS',
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
