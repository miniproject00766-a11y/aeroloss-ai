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

    def evaluate_decision(self, defect_class, severity, r_R, area_pct, daily_loss_inr, payback_days, confidence=0.90, wind_speed=7.56, repair_cost=40000.0, aep_loss_pct=2.5):
        res = self.engine.evaluate_action(
            defect_class=defect_class,
            severity=severity,
            r_R=r_R,
            area_pct=area_pct,
            daily_loss_inr=daily_loss_inr,
            payback_days=payback_days,
            confidence=confidence
        )

        ml_prediction = None
        if self.ml_model is not None:
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

        final_decision = ml_prediction if ml_prediction else res['decision']
        
        res['decision'] = final_decision
        res['ml_classifier_concurrence'] = ml_prediction
        return res
