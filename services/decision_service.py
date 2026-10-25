import os
import sys
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.decision.decision_engine import DecisionEngine

class DecisionService:
    """
    Maintenance Decision Support Service.
    Evaluates defect severity, outboard location (r/R), daily financial loss,
    and payback period to recommend REPAIR, MONITOR, or ENGINEERING ASSESSMENT
    using both rule-based physics and fine-tuned scikit-learn ML Decision Models.
    """
    def __init__(self):
        self.engine = DecisionEngine()
        models_dir = os.path.join(BASE_DIR, "models")
        real_path = os.path.join(models_dir, "decision_classifier_realistic.joblib")
        base_path = os.path.join(models_dir, "decision_classifier.joblib")

        self.ml_model = None
        if os.path.exists(real_path):
            try:
                self.ml_model = joblib.load(real_path)
            except Exception as e:
                print(f"Warning loading realistic decision model: {e}")
        
        if self.ml_model is None and os.path.exists(base_path):
            try:
                self.ml_model = joblib.load(base_path)
            except Exception as e:
                print(f"Warning loading base decision model: {e}")

    def evaluate_decision(
        self,
        defect_class,
        severity,
        r_R,
        area_pct,
        daily_loss_inr,
        payback_days,
        confidence=0.90,
        wind_speed=7.56,
        repair_cost=40000.0,
        aep_loss_pct=2.5,
        classification_disagreement=False,
        crosscheck=None,
        image_frame_coverage_pct=0.0,
        char=None
    ):
        if char and isinstance(char, dict):
            if image_frame_coverage_pct == 0.0:
                image_frame_coverage_pct = float(char.get('image_frame_coverage_pct', 0.0))
            is_structural_override = bool(char.get('structural_safety_override', False) or char.get('is_macro_anomaly_uncertainty', False))
        else:
            is_structural_override = False

        models_disagree_on_structural = False
        if classification_disagreement and crosscheck:
            y_cls = crosscheck.get('yolo_class', defect_class)
            m_cls = crosscheck.get('mobilenet_class', '')
            probs = crosscheck.get('mobilenet_probabilities', {})
            struct_prob = (
                float(probs.get('crack', 0.0)) +
                float(probs.get('thunderstrike', 0.0)) +
                float(probs.get('hide_craze', 0.0))
            )
            if y_cls in ['crack', 'thunderstrike', 'hide_craze'] or m_cls in ['crack', 'thunderstrike', 'hide_craze'] or struct_prob >= 0.15:
                models_disagree_on_structural = True

        res = self.engine.evaluate_action(
            defect_class=defect_class,
            severity=severity,
            r_R=r_R,
            area_pct=area_pct,
            daily_loss_inr=daily_loss_inr,
            payback_days=payback_days,
            confidence=confidence,
            image_frame_coverage_pct=image_frame_coverage_pct,
            is_structural_override=is_structural_override,
            models_disagree_on_structural=models_disagree_on_structural,
            unsupported_class=bool(char.get('unsupported_class', False)) if isinstance(char, dict) else False
        )

        ml_prediction = None
        if self.ml_model is not None and not res.get('safety_override_active', False):
            try:
                feat_df = pd.DataFrame([{
                    'defect_class': defect_class,
                    'severity': severity,
                    'r_R': r_R,
                    'area_pct': area_pct,
                    'wind_speed': wind_speed,
                    'aep_loss_pct': aep_loss_pct,
                    'daily_loss_inr': daily_loss_inr,
                    'repair_cost_inr': repair_cost,
                    'payback_days': payback_days,
                    'confidence': confidence
                }])
                ml_prediction = str(self.ml_model.predict(feat_df)[0])
            except Exception as e:
                print(f"ML Decision Model inference error: {e}")

        res['classification_disagreement'] = bool(classification_disagreement)
        if classification_disagreement and crosscheck:
            y_cls = crosscheck.get('yolo_class', defect_class)
            y_conf = crosscheck.get('yolo_confidence', confidence)
            m_cls = crosscheck.get('mobilenet_class', 'unknown')
            m_conf = crosscheck.get('mobilenet_confidence', 0.0)
            rule = crosscheck.get('decision_rule_applied', 'YOLO_PRIMARY')
            
            y_conf_str = f"{y_conf*100:.1f}%" if y_conf is not None else "N/A"
            m_conf_str = f"{m_conf*100:.1f}%" if m_conf is not None else "N/A"
            advisory = (
                f"Cross-Model Classification Disagreement: YOLOv8 predicted '{y_cls}' ({y_conf_str}) "
                f"while crop MobileNetV3 predicted '{m_cls}' ({m_conf_str}). "
                f"Primary classification governed by: {rule}. Engineering review advisory flagged."
            )
            res['reasons'].append(advisory)

        res['ml_classifier_concurrence'] = ml_prediction
        return res
