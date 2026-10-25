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

    def evaluate_action(
        self,
        defect_class,
        severity,
        r_R,
        area_pct,
        daily_loss_inr,
        payback_days,
        confidence=0.90,
        image_frame_coverage_pct=0.0,
        is_structural_override=False,
        models_disagree_on_structural=False,
        unsupported_class=False
    ):
        reasons = []

        # -------------------------------------------------------------
        # STEP 0: NO DAMAGE DETECTED BASELINE
        # -------------------------------------------------------------
        is_no_defect = bool(
            int(severity) == 0 or
            defect_class in ['No Defect', 'No Supported Defect Detected', 'nominal', 'clean']
        )
        if is_no_defect and not is_structural_override:
            return {
                'decision': 'MONITOR',
                'urgency': 'LOW',
                'reasons': [
                    "No supported optical blade surface defects detected in the inspection image.",
                    "Baseline aerodynamic power loss is estimated at 0.0 kW (Model Baseline - Physical Loss Unmeasured).",
                    "UNCONFIRMED HEALTH DISCLAIMER: Non-detection of optical surface defects does not confirm subsurface, structural, or full-blade health. Standard periodic inspection schedule should be maintained."
                ],
                'safety_override_active': False,
                'metrics_snapshot': {
                    'defect_class': defect_class,
                    'severity': 0,
                    'r_R': r_R,
                    'area_pct': 0.0,
                    'image_frame_coverage_pct': 0.0,
                    'daily_loss_inr': 0.0,
                    'payback_days': None
                }
            }

        # -------------------------------------------------------------
        # STEP 1: CONSERVATIVE STRUCTURAL-DAMAGE SAFETY OVERRIDE
        # APPLIED FIRST, BEFORE ANY ECONOMIC / PAYBACK LOGIC
        # Under IEC 61400-23 / DNV-GL blade safety guidelines, structural blade
        # integrity takes absolute precedence over repair payback or economic optimization.
        # Differentiates structural shell rupture from aerodynamic surface erosion (IEA Task 46).
        # -------------------------------------------------------------
        is_outboard_tip = bool(float(r_R) >= 0.70)
        is_macro_hazard = bool(
            is_structural_override or
            (float(image_frame_coverage_pct) >= 20.0 and defect_class in ['crack', 'thunderstrike', 'hide_craze', 'craze']) or
            (float(image_frame_coverage_pct) >= 8.0 and is_outboard_tip and float(confidence) < 0.40) or
            (float(image_frame_coverage_pct) >= 5.0 and is_outboard_tip and float(confidence) < 0.35) or
            models_disagree_on_structural
        )

        is_structural_hazard = bool(
            defect_class in ['crack', 'thunderstrike'] or
            int(severity) >= 5 or
            (int(severity) >= 4 and defect_class not in ['corrosion', 'surface_injure']) or
            is_macro_hazard
        )

        if is_structural_hazard:
            category = 'ENGINEERING_ASSESSMENT'
            reasons.append("MANDATORY SAFETY OVERRIDE (IEC 61400 / DNV-GL): Structural blade integrity takes precedence over economic/statistical models.")

            if defect_class in ['crack', 'thunderstrike']:
                reasons.append(f"Structural defect '{defect_class}' poses catastrophic blade failure hazard.")
            elif float(image_frame_coverage_pct) >= 8.0 and is_outboard_tip:
                reasons.append(
                    f"Extensive blade-tip anomaly ({float(image_frame_coverage_pct):.1f}% frame coverage at r/R={float(r_R):.2f}) "
                    f"indicates severe shell fracture, delamination, or tip split under high aeroelastic dynamic pressure."
                )
            elif float(image_frame_coverage_pct) >= 20.0:
                reasons.append(
                    f"Extensive photographic frame anomaly ({float(image_frame_coverage_pct):.1f}% coverage) "
                    f"exceeds physical plausibility for superficial weathering. "
                    f"Classification '{defect_class}' is not treated as proof of a benign gelcoat craze. "
                    f"Recommend qualified engineering/NDT inspection; image analysis does not confirm internal structural failure."
                )
            elif models_disagree_on_structural:
                reasons.append("Cross-model classification disagreement on potential structural defect; physical engineering review required.")
            else:
                reasons.append(f"High defect severity (Level {severity}/5) requires structural engineering assessment.")

            # Urgency Determination: Level 5, crack/thunderstrike, outboard tip anomaly, or extensive coverage is CRITICAL
            if int(severity) >= 5 or defect_class in ['crack', 'thunderstrike'] or (float(image_frame_coverage_pct) >= 8.0 and is_outboard_tip) or (float(image_frame_coverage_pct) >= 20.0 and defect_class not in ['corrosion', 'surface_injure']):
                urgency = 'CRITICAL'
                reasons.append("Critical structural risk requires immediate NDT or blade specialist review.")
            else:
                urgency = 'HIGH'

            # RETURN IMMEDIATELY: Structural safety override cannot be downgraded by payback!
            return {
                'decision': category,
                'urgency': urgency,
                'reasons': reasons,
                'safety_override_active': True,
                'metrics_snapshot': {
                    'defect_class': defect_class,
                    'severity': severity,
                    'r_R': r_R,
                    'area_pct': area_pct,
                    'image_frame_coverage_pct': image_frame_coverage_pct,
                    'daily_loss_inr': daily_loss_inr,
                    'payback_days': payback_days
                }
            }

        if unsupported_class:
            return {
                'decision': 'ENGINEERING_ASSESSMENT',
                'urgency': 'HIGH',
                'reasons': [
                    f"Detected class '{defect_class}' is not in the supported taxonomy.",
                    "Classification is uncertain; obtain qualified engineering review before relying on economic estimates."
                ],
                'safety_override_active': False,
                'classification_review_required': True,
                'metrics_snapshot': {
                    'defect_class': defect_class,
                    'severity': severity,
                    'r_R': r_R,
                    'area_pct': area_pct,
                    'image_frame_coverage_pct': image_frame_coverage_pct,
                    'daily_loss_inr': daily_loss_inr,
                    'payback_days': payback_days
                }
            }

        # -------------------------------------------------------------
        # STEP 2: ECONOMIC & AERODYNAMIC REPAIR THRESHOLD (Only for non-structural defects)
        # -------------------------------------------------------------
        if (r_R >= self.outboard_threshold_rR and severity >= 3 and payback_days <= self.payback_threshold_days) or \
           (daily_loss_inr >= 3000.0 and payback_days <= 60.0) or \
           (severity >= 4 and payback_days <= self.payback_threshold_days):
            category = 'REPAIR'
            reasons.append(f"Significant aerodynamic degradation (severity Level {severity}/5, r/R={r_R:.2f}).")
            reasons.append(f"Rapid repair payback of {payback_days:.1f} days justifies scheduled surface restoration.")
            urgency = 'HIGH'

        # -------------------------------------------------------------
        # STEP 3: UNCERTAIN / BORDERLINE CASES
        # -------------------------------------------------------------
        elif confidence < 0.65 or (severity == 3 and 90.0 < payback_days <= 150.0):
            category = 'ENGINEERING_ASSESSMENT'
            reasons.append("Moderate aerodynamic impact with borderline repair economics or lower visual confidence.")
            reasons.append("Recommend detailed inspection before scheduling rope-access crew.")
            urgency = 'HIGH' if confidence < 0.25 else 'MEDIUM'

        # -------------------------------------------------------------
        # STEP 4: INBOARD OR LOW SEVERITY MONITOR THRESHOLD
        # -------------------------------------------------------------
        else:
            category = 'MONITOR'
            reasons.append(f"Minor aerodynamic degradation (r/R={r_R:.2f}, severity={severity}).")
            reasons.append(f"Extended payback period ({payback_days:.1f} days) suggests deferring repair to next scheduled outage.")
            urgency = 'LOW'

        return {
            'decision': category,
            'urgency': urgency,
            'reasons': reasons,
            'safety_override_active': False,
            'metrics_snapshot': {
                'defect_class': defect_class,
                'severity': severity,
                'r_R': r_R,
                'area_pct': area_pct,
                'image_frame_coverage_pct': image_frame_coverage_pct,
                'daily_loss_inr': daily_loss_inr,
                'payback_days': payback_days
            }
        }
