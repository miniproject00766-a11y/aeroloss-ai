import numpy as np

class DamageService:
    """
    Damage Characterization Service:
    Translates YOLO bounding boxes into normalized spanwise position (r/R),
    damage area percentage (proxy), and configurable severity grade (1-5).
    """
    def __init__(self):
        self.base_severity_map = {
            'craze': 1,
            'surface_injure': 2,
            'corrosion': 2,
            'hide_craze': 3,
            'crack': 4,
            'thunderstrike': 5
        }

    def characterize_detections(self, detections, img_width=1024, img_height=1024):
        if not detections:
            return {
                'primary_defect': 'No Defect',
                'severity_proxy': 0,
                'spanwise_position_r_R': 0.88,
                'damage_area_pct': 0.0,
                'blade_region': 'Outboard Tip',
                'defect_count': 0
            }

        # Select highest severity / highest confidence defect as primary
        sorted_dets = sorted(
            detections,
            key=lambda d: (self.base_severity_map.get(d['damage_type'], 2), d['confidence']),
            reverse=True
        )
        primary = sorted_dets[0]

        primary_class = primary['damage_type']
        confidence = primary['confidence']
        area_pct = primary.get('damage_area_percent', 4.2)
        r_R = primary.get('r_over_R', 0.88)

        base_sev = self.base_severity_map.get(primary_class, 2)
        if area_pct > 6.0:
            severity = min(5, base_sev + 1)
        elif area_pct < 0.8:
            severity = max(1, base_sev - 1)
        else:
            severity = base_sev

        blade_region = 'Outboard (High Speed)' if r_R >= 0.80 else ('Mid-Span' if r_R >= 0.50 else 'Root/Inboard')

        return {
            'primary_defect': primary_class,
            'confidence': confidence,
            'confidence_pct': f"{confidence*100:.1f}%",
            'severity_proxy': severity,
            'spanwise_position_r_R': round(r_R, 3),
            'damage_area_pct': round(area_pct, 2),
            'blade_region': blade_region,
            'defect_count': len(detections)
        }
