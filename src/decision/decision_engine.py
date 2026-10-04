class DecisionEngine:
    """
    Translates defect characteristics, physics impacts, and economic payback
    into actionable maintenance decision support categories:
    - REPAIR
    - MONITOR
    - ENGINEERING_ASSESSMENT
    """
    def __init__(self, payback_threshold_days=90.0, outboard_threshold_rR=0.80):
        self.payback_threshold_days = payback_threshold_days
        self.outboard_threshold_rR = outboard_threshold_rR

    def evaluate_action(self, defect_class, severity, r_R, area_pct, daily_loss_inr, payback_days, confidence=0.90):
        reasons = []

        # 1. Structural / Safety Priority Check:
        # Cracks and lightning/thunderstrike damages compromise structural blade integrity
        # regardless of immediate aerodynamic loss.
        if defect_class in ['crack', 'thunderstrike']:
            category = 'ENGINEERING_ASSESSMENT'
            reasons.append(f"Structural defect '{defect_class}' poses catastrophic failure risk.")
            if severity >= 4:
                reasons.append("High defect severity requires immediate NDT or blade specialist review.")
            urgency = 'CRITICAL'

        # 2. Economic & Aerodynamic Repair Threshold
        elif (r_R >= self.outboard_threshold_rR and severity >= 3 and payback_days <= self.payback_threshold_days) or \
             (daily_loss_inr >= 3000.0 and payback_days <= 60.0):
            category = 'REPAIR'
            reasons.append(f"High outboard aerodynamic penalty at r/R={r_R:.2f} with severity {severity}.")
            reasons.append(f"Rapid repair payback of {payback_days:.1f} days justifies scheduled intervention.")
            urgency = 'HIGH'

        # 3. Uncertain / Borderline Cases
        elif confidence < 0.65 or (severity == 3 and 90.0 < payback_days <= 150.0):
            category = 'ENGINEERING_ASSESSMENT'
            reasons.append("Moderate aerodynamic impact with borderline repair economics or lower visual confidence.")
            reasons.append("Recommend detailed inspection before scheduling rope-access crew.")
            urgency = 'MEDIUM'

        # 4. Inboard or Low Severity Monitor Threshold
        else:
            category = 'MONITOR'
            reasons.append(f"Minor aerodynamic degradation (r/R={r_R:.2f}, severity={severity}).")
            reasons.append(f"Extended payback period ({payback_days:.1f} days) suggests deferring repair to next scheduled outage.")
            urgency = 'LOW'

        return {
            'decision': category,
            'urgency': urgency,
            'reasons': reasons,
            'metrics_snapshot': {
                'defect_class': defect_class,
                'severity': severity,
                'r_R': r_R,
                'area_pct': area_pct,
                'daily_loss_inr': daily_loss_inr,
                'payback_days': payback_days
            }
        }
