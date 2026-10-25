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

    def compute_box_coverage(self, bbox, img_width=1024, img_height=1024):
        """
        Computes rectangular bounding-box coverage percentage on an image of size (W, H):
        coverage = 100 * max(0, x2 - x1) * max(0, y2 - y1) / (W * H)
        Coordinates are validated and clipped strictly against image boundaries [0, W] x [0, H].
        """
        if not bbox or len(bbox) < 4 or img_width <= 0 or img_height <= 0:
            return 0.0
        try:
            raw_x1, raw_y1, raw_x2, raw_y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        except (ValueError, TypeError):
            return 0.0

        x_min = min(raw_x1, raw_x2)
        x_max = max(raw_x1, raw_x2)
        y_min = min(raw_y1, raw_y2)
        y_max = max(raw_y1, raw_y2)

        cx1 = max(0.0, min(float(img_width), x_min))
        cx2 = max(0.0, min(float(img_width), x_max))
        cy1 = max(0.0, min(float(img_height), y_min))
        cy2 = max(0.0, min(float(img_height), y_max))

        box_w = max(0.0, cx2 - cx1)
        box_h = max(0.0, cy2 - cy1)
        total_img_area = float(img_width) * float(img_height)
        return round((box_w * box_h / total_img_area) * 100.0, 2)

    def compute_union_coverage(self, bboxes, img_width=1024, img_height=1024):
        """
        Computes exact analytical union coverage of multiple bounding boxes via 2D sweep-line.
        Overlapping rectangular regions are merged so area is never double-counted.
        """
        if not bboxes or img_width <= 0 or img_height <= 0:
            return 0.0

        clipped_boxes = []
        for b in bboxes:
            if not b or len(b) < 4:
                continue
            try:
                rx1, ry1, rx2, ry2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                cx1 = max(0.0, min(float(img_width), min(rx1, rx2)))
                cx2 = max(0.0, min(float(img_width), max(rx1, rx2)))
                cy1 = max(0.0, min(float(img_height), min(ry1, ry2)))
                cy2 = max(0.0, min(float(img_height), max(ry1, ry2)))
                if cx2 > cx1 and cy2 > cy1:
                    clipped_boxes.append((cx1, cy1, cx2, cy2))
            except (ValueError, TypeError):
                continue

        if not clipped_boxes:
            return 0.0

        # Sweep-line over unique x-coordinates
        x_coords = sorted(list(set([bx[0] for bx in clipped_boxes] + [bx[2] for bx in clipped_boxes])))
        total_union_area = 0.0

        for i in range(len(x_coords) - 1):
            xl = x_coords[i]
            xr = x_coords[i + 1]
            dx = xr - xl
            if dx <= 0.0:
                continue

            y_intervals = []
            for x1, y1, x2, y2 in clipped_boxes:
                if x1 <= xl and x2 >= xr:
                    y_intervals.append((y1, y2))

            if not y_intervals:
                continue

            y_intervals.sort(key=lambda seg: seg[0])
            merged_dy = 0.0
            cur_y1, cur_y2 = y_intervals[0]
            for ny1, ny2 in y_intervals[1:]:
                if ny1 <= cur_y2:
                    cur_y2 = max(cur_y2, ny2)
                else:
                    merged_dy += (cur_y2 - cur_y1)
                    cur_y1, cur_y2 = ny1, ny2
            merged_dy += (cur_y2 - cur_y1)
            total_union_area += dx * merged_dy

        total_img_area = float(img_width) * float(img_height)
        return round(min(100.0, max(0.0, (total_union_area / total_img_area) * 100.0)), 2)

    def compute_box_iou(self, boxA, boxB):
        """Computes IoU and containment ratio between two bounding boxes [x1, y1, x2, y2]."""
        try:
            ax1 = min(float(boxA[0]), float(boxA[2]))
            ay1 = min(float(boxA[1]), float(boxA[3]))
            ax2 = max(float(boxA[0]), float(boxA[2]))
            ay2 = max(float(boxA[1]), float(boxA[3]))

            bx1 = min(float(boxB[0]), float(boxB[2]))
            by1 = min(float(boxB[1]), float(boxB[3]))
            bx2 = max(float(boxB[0]), float(boxB[2]))
            by2 = max(float(boxB[1]), float(boxB[3]))
        except (ValueError, TypeError, IndexError):
            return 0.0, 0.0

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter_area = iw * ih

        areaA = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        areaB = max(0.0, (bx2 - bx1) * (by2 - by1))
        union_area = areaA + areaB - inter_area
        iou = (inter_area / union_area) if union_area > 0 else 0.0
        iomin = (inter_area / min(areaA, areaB)) if min(areaA, areaB) > 0 else 0.0
        return iou, iomin

    def cluster_detections(self, detections, iou_thresh=0.35, iomin_thresh=0.60):
        """
        Clusters spatially overlapping bounding boxes into distinct physical damage regions.
        Prevents duplicate/overlapping YOLO proposals on the same physical defect from being
        misleadingly presented as multiple independent blade damage occurrences.
        """
        if not detections:
            return 0, []

        n = len(detections)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        for i in range(n):
            box_i = detections[i].get('bounding_box')
            if not box_i:
                continue
            for j in range(i + 1, n):
                box_j = detections[j].get('bounding_box')
                if not box_j:
                    continue
                iou, iomin = self.compute_box_iou(box_i, box_j)
                if iou >= iou_thresh or iomin >= iomin_thresh:
                    union(i, j)

        clusters = {}
        for idx in range(n):
            root = find(idx)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(detections[idx])

        return len(clusters), list(clusters.values())

    def detection_evidence_score(self, d):
        """
        Computes evidence-weighted score to select the primary defect.
        Prevents low-confidence noise proposals (e.g. 21% hide_craze) from overriding
        highly confident, consensus-verified surface erosion defects (e.g. 99.5% corrosion).
        """
        damage_type = d.get('damage_type', 'surface_injure')
        conf = float(d.get('confidence', 0.5))
        crosscheck = d.get('crosscheck', {})
        mnet_conf = crosscheck.get('mobilenet_confidence')
        mnet_cls = crosscheck.get('mobilenet_class')
        models_agree = d.get('models_agree')

        # Base hazard score
        if damage_type in ['crack', 'thunderstrike']:
            score = 100.0 + conf * 20.0
        elif damage_type == 'hide_craze':
            # Subsurface delamination: prioritised if confident or MobileNet confirmed
            if conf >= 0.35 or (mnet_cls == 'hide_craze' and mnet_conf is not None and mnet_conf >= 0.50):
                score = 60.0 + conf * 15.0
            else:
                score = 25.0 + conf * 10.0
        elif damage_type in ['corrosion', 'surface_injure']:
            score = 30.0 + conf * 15.0
        elif damage_type not in self.base_severity_map:
            score = 50.0 + conf * 10.0
        else: # craze
            score = 10.0 + conf * 10.0

        # Confirmation bonus
        if models_agree is True:
            score += 15.0
        elif mnet_conf is not None and mnet_conf >= 0.70 and mnet_cls == damage_type:
            score += 10.0

        # Penalize noisy low-confidence proposals when better candidates exist
        if conf < 0.25 and (models_agree is False or (mnet_cls and mnet_cls not in ['crack', 'thunderstrike'])):
            score -= 15.0

        return score

    def characterize_detections(self, detections, img_width=1024, img_height=1024):
        """
        Translates detected bounding boxes into aerodynamic damage parameters.
        Distinguishes 2D photographic frame coverage from 3D blade damage area proxy:
        - image_frame_coverage_pct: Exact geometric area % of the 2D photographic frame occupied by bounding boxes.
        - blade_damage_area_proxy_pct: Documented conservative proxy (10% local frame scale, capped at 5.0%)
          estimating the fractional damage over the entire 3D blade surface until physical photogrammetry is calibrated.
        """
        if not detections:
            return {
                'primary_defect': 'No Defect',
                'detection_status': 'NO_DEFECT_DETECTED',
                'analysis_state': 'NO_DAMAGE_DETECTED',
                'health_confirmed': False,
                'health_status_label': 'Nominal (Unconfirmed)',
                'unsupported_class': False,
                'severity_proxy': 0,
                'confidence': None,
                'confidence_pct': 'N/A',
                'is_low_confidence': False,
                'confidence_grade': 'NOMINAL_BASELINE',
                'spanwise_position_r_R': 0.88,
                'spanwise_position_available': False,
                'spanwise_position_source': 'PROVISIONAL_NOMINAL_DEFAULT_NOT_MEASURED',
                'geometry_status': 'UNAVAILABLE_NO_DETECTIONS',
                'damage_area_pct': 0.0,
                'blade_damage_area_proxy_pct': 0.0,
                'image_frame_coverage_pct': 0.0,
                'primary_bbox_frame_coverage_pct': 0.0,
                'union_frame_coverage_pct': 0.0,
                'bbox_image_area_pct': 0.0,
                'primary_bbox_area_pct': 0.0,
                'union_bbox_area_pct': 0.0,
                'blade_region': 'Nominal Surface (Unconfirmed)',
                'defect_count': 0,
                'loss_status': 'BASELINE_ZERO_ESTIMATE',
                'structural_safety_override': False,
                'structural_safety_reason': None,
                'is_uncertain_extensive_damage': False,
                'is_macro_anomaly_uncertainty': False,
                'uncertainty_state': 'NO_DEFECT_BASELINE',
                'provenance': {
                    'spanwise_position_source': 'PROVISIONAL_NOMINAL_DEFAULT_NOT_MEASURED',
                    'damage_area_source': 'NO_DETECTIONS_RECORDED',
                    'image_frame_coverage_pct': 0.0,
                    'blade_damage_area_proxy_pct': 0.0,
                    'uncertainty_flag': 'NOMINAL_NO_DEFECT',
                    'provenance_note': (
                        'No supported optical bounding boxes detected. Frame coverage is zero detected-box coverage; '
                        'baseline physical loss is estimated at zero. Non-detection of optical defects is not physical proof of blade health.'
                    )
                }
            }

        # Check for unlocalized defect detection from whole-image classifier
        if detections and detections[0].get('is_unlocalized'):
            primary = detections[0]
            primary_class = primary.get('damage_type', 'unknown')
            confidence = float(primary.get('confidence', 0.5))
            severity = 5 if primary_class in ['crack', 'thunderstrike'] else (4 if primary_class == 'hide_craze' else 3)
            override_reason = (
                f"Unlocalized defect signal '{primary_class}' detected by deep learning classifier "
                f"({confidence*100:.1f}% confidence) without localized bounding box. "
                f"Qualified physical engineering assessment required."
            )
            return {
                'primary_defect': primary_class,
                'detection_status': 'UNLOCALIZED_DEFECT_SIGNAL',
                'analysis_state': 'UNCERTAIN_NEEDS_REVIEW',
                'health_confirmed': False,
                'health_status_label': 'Critical' if severity >= 5 else 'Uncertain (Review Required)',
                'unsupported_class': primary_class not in self.base_severity_map,
                'detection_method': 'mobilenet_whole_image_unlocalized',
                'is_synthetic': False,
                'is_heuristic_fallback': False,
                'is_unlocalized': True,
                'confidence': confidence,
                'confidence_pct': f"{confidence*100:.1f}%",
                'is_low_confidence': True,
                'confidence_grade': 'LOW',
                'geometry_status': 'UNAVAILABLE_UNLOCALIZED',
                'spanwise_position_available': False,
                'spanwise_position_source': 'PROVISIONAL_NOMINAL_DEFAULT_NOT_MEASURED',
                'structural_safety_override': True,
                'structural_safety_reason': override_reason,
                'is_uncertain_extensive_damage': True,
                'is_macro_anomaly_uncertainty': True,
                'uncertainty_state': 'UNCERTAIN_UNLOCALIZED_SIGNAL',
                'macro_anomaly_note': override_reason,
                'severity_proxy': severity,
                'spanwise_position_r_R': 0.70,
                'primary_bbox_frame_coverage_pct': 0.0,
                'primary_bbox_area_pct': 0.0,
                'union_frame_coverage_pct': 0.0,
                'image_frame_coverage_pct': 0.0,
                'union_bbox_area_pct': 0.0,
                'blade_damage_area_proxy_pct': 0.0,
                'damage_area_pct': 0.0,
                'distinct_damage_regions': 0,
                'candidate_detections_count': 0,
                'has_overlapping_detections': False,
                'detection_clustering_note': 'No bounding box localized; defect signal detected at whole-image level.',
                'blade_region': 'Spanwise Location Unresolved',
                'defect_count': 1,
                'loss_status': 'UNAVAILABLE_NEEDS_REVIEW',
                'crosscheck': primary.get('crosscheck', {}),
                'models_agree': None,
                'classification_disagreement': True,
                'yolo_class': None,
                'yolo_confidence': None,
                'mobilenet_class': primary_class,
                'mobilenet_confidence': confidence,
                'decision_rule_applied': 'UNLOCALIZED_DEFECT_DETECTION_NEEDS_REVIEW',
                'provenance': {
                    'spanwise_position_source': 'UNRESOLVED_UNLOCALIZED_DEFECT',
                    'damage_area_source': 'UNAVAILABLE_UNLOCALIZED_DEFECT',
                    'primary_bbox_frame_coverage_pct': 0.0,
                    'union_frame_coverage_pct': 0.0,
                    'image_frame_coverage_pct': 0.0,
                    'blade_damage_area_proxy_pct': 0.0,
                    'uncertainty_flag': 'UNLOCALIZED_DEFECT_SIGNAL',
                    'provenance_note': override_reason
                }
            }

        # 1. Derive bounding-box coverage from coordinates
        bboxes_list = []
        for d in detections:
            if 'bounding_box' in d and d['bounding_box'] is not None:
                derived_pct = self.compute_box_coverage(d['bounding_box'], img_width, img_height)
                d['damage_area_percent'] = derived_pct
                bboxes_list.append(d['bounding_box'])
            elif 'damage_area_percent' not in d:
                d['damage_area_percent'] = 0.0

        # Compute combined union coverage on the photographic frame
        if bboxes_list:
            union_cov_pct = self.compute_union_coverage(bboxes_list, img_width, img_height)
        else:
            union_cov_pct = max(0.0, min(100.0, float(detections[0].get('damage_area_percent', 0.0))))

        # 2. Photographic Frame Coverage vs. Actual 3D Blade Damage Area Proxy
        # 2D bounding-box frame coverage reflects camera proximity, framing, and zoom.
        # An inspection frame typically captures a localized section (~2-4 m of span) out of a ~50-60 m blade (~200 m^2 total blade area).
        # A conservative proxy of 10% local frame scale is applied to estimate the blade surface damage fraction.
        image_frame_coverage_pct = round(union_cov_pct, 2)
        blade_damage_area_proxy_pct = round(float(np.clip(union_cov_pct * 0.10, 0.0, 5.0)), 3) if union_cov_pct > 0 else 0.0

        # 3. Spatial Clustering of Overlapping Proposals
        distinct_regions, clusters = self.cluster_detections(detections, iou_thresh=0.35, iomin_thresh=0.60)
        has_overlapping = bool(distinct_regions < len(detections))
        if has_overlapping:
            clustering_note = (
                f"{len(detections)} candidate bounding box proposals clustered into "
                f"{distinct_regions} distinct physical damage region(s) (IoU >= 0.35 / containment >= 0.60)."
            )
        else:
            clustering_note = (
                f"{len(detections)} candidate detection(s) occupy distinct physical blade locations."
            )

        # 4. Evidence-Weighted Primary Defect Selection
        # Prevents low-confidence noise proposals (e.g. 21% hide_craze) from overriding
        # highly confident consensus surface erosion detections (e.g. 99.5% corrosion).
        sorted_dets = sorted(
            detections,
            key=lambda d: self.detection_evidence_score(d),
            reverse=True
        )
        primary = sorted_dets[0]

        primary_class = primary['damage_type']
        unsupported_class = primary_class not in self.base_severity_map
        confidence = float(primary.get('confidence', 0.5))
        primary_box_cov = float(primary.get('damage_area_percent', union_cov_pct))
        has_spanwise_position = primary.get('r_over_R') is not None
        r_R = float(primary['r_over_R']) if has_spanwise_position else 0.88
        spanwise_position_source = (
            'INFERRED_IMAGE_Y_COORDINATE_HEURISTIC'
            if has_spanwise_position else
            'PROVISIONAL_NOMINAL_DEFAULT_NOT_MEASURED'
        )
        geometry_status = (
            'PROVISIONAL_IMAGE_GEOMETRY'
            if bboxes_list and has_spanwise_position else
            'PARTIAL_GEOMETRY_UNAVAILABLE'
        )

        # -------------------------------------------------------------
        # Multi-Factor Physical & Structural Severity Determination:
        # Visual classification alone (e.g. CRAZE, CRACK, CORROSION, SURFACE_INJURE)
        # MUST NOT alone determine physical severity (IEC 61400 / DNV-GL).
        # Surface erosion (IEA Wind Task 46 Cat 1-4) is differentiated from catastrophic structural fracture.
        # Evaluates full-frame coverage, primary bounding-box coverage, spanwise position (r/R),
        # model confidence, and cross-model agreement signals.
        # -------------------------------------------------------------
        is_low_conf = bool(confidence < 0.35)
        conf_grade = 'LOW' if confidence < 0.35 else ('MEDIUM' if confidence < 0.65 else 'HIGH')
        is_outboard_tip = bool(r_R >= 0.70)
        is_uncertain_extensive_damage = bool(is_low_conf and union_cov_pct >= 8.0)

        # Cross-model structural signals:
        mnet_probs = primary.get('crosscheck', {}).get('mobilenet_probabilities', {})
        struct_prob = (
            float(mnet_probs.get('crack', 0.0)) +
            float(mnet_probs.get('thunderstrike', 0.0)) +
            float(mnet_probs.get('hide_craze', 0.0))
        )
        has_structural_disagreement = bool(
            primary.get('classification_disagreement', False) and
            (primary.get('yolo_class') in ['crack', 'thunderstrike', 'hide_craze'] or
             primary.get('mobilenet_class') in ['crack', 'thunderstrike', 'hide_craze'] or
             struct_prob >= 0.15)
        )

        structural_safety_override = False
        override_reason = None

        # --- SEVERITY & SAFETY HIERARCHY ---
        if (
            union_cov_pct >= 20.0
            and is_low_conf
            and primary_class in ['corrosion', 'surface_injure']
        ):
            structural_safety_override = True
            is_uncertain_extensive_damage = True
            severity = 4
            override_reason = (
                f"Extensive low-confidence blade anomaly ({union_cov_pct:.1f}% frame coverage) "
                f"was localized as '{primary_class}', but the class is provisional. "
                f"Qualified engineering/NDT assessment is required; image analysis does not "
                f"confirm internal structural failure."
            )

        # Branch 1: Confirmed Catastrophic Structural Defects (crack, thunderstrike)
        elif primary_class in ['crack', 'thunderstrike']:
            structural_safety_override = True
            if primary_class == 'thunderstrike':
                severity = 5
            elif union_cov_pct >= 3.0 or is_outboard_tip:
                severity = 5
            else:
                severity = 4
            override_reason = f"Confirmed structural defect '{primary_class}' poses catastrophic blade failure hazard."

        # Branch 2: Severe Outboard Blade-Tip Rupture / Tip Fracture (e.g. images (3).jpg)
        # Under extreme centrifugal tip speed and dynamic aeroelastic pressure, an extensive tip defect
        # that is not verified benign surface weathering represents blade-tip fracture, shell rupture, or tip split.
        elif is_outboard_tip and union_cov_pct >= 8.0 and primary_class == 'craze' and (is_low_conf or union_cov_pct >= 10.0):
            structural_safety_override = True
            severity = 5
            override_reason = (
                f"Extensive blade-tip anomaly ({union_cov_pct:.1f}% frame coverage at r/R={r_R:.2f}) "
                f"under high centrifugal dynamic pressure indicates severe shell fracture, delamination, or tip split."
            )

        # Branch 3: Subsurface Delamination (hide_craze)
        elif primary_class == 'hide_craze':
            if (union_cov_pct >= 20.0 or (union_cov_pct >= 8.0 and is_outboard_tip)) and not is_low_conf:
                structural_safety_override = True
                severity = 5
                override_reason = (
                    f"Extensive subsurface composite delamination ({union_cov_pct:.1f}% frame coverage) "
                    f"under aerodynamic dynamic pressure risks skin detachment."
                )
            elif union_cov_pct >= 8.0:
                if is_low_conf:
                    severity = 3
                    is_uncertain_extensive_damage = True
                    override_reason = (
                        f"Uncertain extensive anomaly ({union_cov_pct:.1f}% frame coverage, confidence {confidence*100:.1f}%) "
                        f"with suspected subsurface delamination; provisional advisory Level 3."
                    )
                else:
                    structural_safety_override = True
                    severity = 4
                    override_reason = (
                        f"Subsurface composite delamination ({union_cov_pct:.1f}% frame coverage) "
                        f"requires specialist physical engineering assessment."
                    )
            elif union_cov_pct < 1.0:
                severity = 2
            else:
                severity = 3

        # Branch 4: Cross-Model Disagreement on Potential Structural Defect
        elif has_structural_disagreement:
            structural_safety_override = True
            severity = 5 if (is_outboard_tip or union_cov_pct >= 10.0) else 4
            override_reason = (
                f"Cross-model classification disagreement on potential structural defect "
                f"(YOLO: {primary.get('yolo_class')}, MobileNet: {primary.get('mobilenet_class')}). "
                f"Escalated for specialist physical NDT review."
            )

        # Branch 5: Leading-Edge Surface Erosion & Weathering (corrosion, surface_injure)
        # Complies with IEA Wind Task 46: surface erosion spans wide portions of the leading edge
        # (Category 1 to 4) causing substantial aerodynamic drag, but is NOT a catastrophic structural fracture (Level 5).
        elif primary_class in ['corrosion', 'surface_injure']:
            if union_cov_pct >= 20.0:
                severity = 4  # IEA Task 46 Category 4: Deep gouging / extensive leading-edge erosion
                override_reason = (
                    f"Extensive leading-edge surface erosion ({union_cov_pct:.1f}% frame coverage, IEA Task 46 Category 4) "
                    f"requires scheduled leading-edge surface restoration."
                )
            elif union_cov_pct >= 8.0:
                if is_outboard_tip and is_low_conf:
                    severity = 4
                    structural_safety_override = True
                    override_reason = (
                        f"Uncertain outboard anomaly ({union_cov_pct:.1f}% frame coverage at r/R={r_R:.2f}) "
                        f"with low classification confidence ({confidence*100:.1f}%); specialist structural review required."
                    )
                else:
                    severity = 3  # IEA Task 46 Category 3: Moderate-to-severe erosion along leading edge
            elif union_cov_pct >= 2.0:
                severity = 2  # IEA Task 46 Category 2: Moderate erosion / coat breakdown
            else:
                severity = 2

        # Branch 6: Cosmetic Crazing & Hairline Anomalies (craze)
        # A large, visibly abnormal region must NOT be treated as low risk solely because
        # the closed-set classifier labels it CRAZE, the models agree, or confidence is high.
        # Close-up photographs often map to mid-span r/R via the uncalibrated y-heuristic,
        # which previously skipped the outboard tip-fracture rule (Branch 2).
        elif primary_class == 'craze':
            if union_cov_pct >= 20.0:
                structural_safety_override = True
                is_uncertain_extensive_damage = True
                severity = 4
                agreement_note = (
                    " and cross-model agreement"
                    if primary.get('models_agree') is True
                    else ""
                )
                override_reason = (
                    f"Extensive optical anomaly occupies {union_cov_pct:.1f}% of the inspection frame. "
                    f"Model class is '{primary_class}' with {confidence*100:.1f}% confidence"
                    f"{agreement_note}; this is a classification label, not confirmation of defect type "
                    f"or internal blade structure. A large abnormal damage region requires conservative "
                    f"ENGINEERING_ASSESSMENT and qualified engineering/NDT inspection. "
                    f"Image analysis does not confirm internal structural failure."
                )
            elif union_cov_pct >= 8.0:
                if is_outboard_tip or is_low_conf:
                    severity = 4
                    structural_safety_override = True
                    override_reason = (
                        f"Uncertain extensive anomaly ({union_cov_pct:.1f}% frame coverage) "
                        f"exceeds physical bounds for superficial craze; structural review recommended."
                    )
                else:
                    severity = 2
            elif union_cov_pct < 1.0:
                severity = 1
            else:
                severity = 1

        elif unsupported_class:
            severity = 3

        # Branch 7: Fallback / Default
        else:
            base_sev = self.base_severity_map.get(primary_class, 2)
            severity = max(1, min(5, base_sev))

        is_macro_anomaly_uncertainty = structural_safety_override

        blade_region = 'Outboard (High Speed)' if r_R >= 0.80 else ('Mid-Span' if r_R >= 0.50 else 'Root/Inboard')

        y_conf_val = primary.get('yolo_confidence')
        y_conf_pct = f"{float(y_conf_val)*100:.1f}%" if y_conf_val is not None else None
        m_conf_val = primary.get('mobilenet_confidence')
        m_conf_pct = f"{float(m_conf_val)*100:.1f}%" if m_conf_val is not None else None

        crosscheck_data = primary.get('crosscheck', {
            'status': 'EVALUATED' if primary.get('mobilenet_class') else 'NOT_AVAILABLE',
            'models_agree': primary.get('models_agree', None),
            'classification_disagreement': primary.get('classification_disagreement', False),
            'yolo_class': primary.get('yolo_class', primary_class),
            'yolo_confidence': y_conf_val,
            'yolo_confidence_pct': y_conf_pct,
            'mobilenet_class': primary.get('mobilenet_class'),
            'mobilenet_confidence': m_conf_val,
            'mobilenet_confidence_pct': m_conf_pct,
            'decision_rule_applied': primary.get('decision_rule_applied', 'NONE')
        })

        return {
            'primary_defect': primary_class,
            'detection_status': 'DETECTIONS_FOUND',
            'analysis_state': 'UNCERTAIN_NEEDS_REVIEW' if (is_low_conf or is_uncertain_extensive_damage or has_structural_disagreement or structural_safety_override) else 'DAMAGE_DETECTED',
            'loss_status': 'CALCULATED_MODEL_ESTIMATE',
            'health_status_label': 'Critical' if severity >= 4 else ('Monitor' if severity <= 1 else 'High Risk'),
            'health_confirmed': False,
            'unsupported_class': unsupported_class,
            'detection_method': primary.get('detection_method', 'yolov8'),
            'is_synthetic': primary.get('is_synthetic', False),
            'is_heuristic_fallback': primary.get('is_heuristic_fallback', False),
            'confidence': confidence,
            'confidence_pct': f"{confidence*100:.1f}%",
            'is_low_confidence': is_low_conf,
            'confidence_grade': conf_grade,
            'geometry_status': geometry_status,
            'spanwise_position_available': has_spanwise_position,
            'spanwise_position_source': spanwise_position_source,
            'structural_safety_override': structural_safety_override,
            'structural_safety_reason': override_reason,
            'is_uncertain_extensive_damage': is_uncertain_extensive_damage,
            'is_macro_anomaly_uncertainty': is_macro_anomaly_uncertainty,
            'uncertainty_state': (
                'STRUCTURAL_SAFETY_OVERRIDE' if structural_safety_override else
                ('CLASSIFICATION_UNCERTAIN_EXTENSIVE_DAMAGE' if is_uncertain_extensive_damage else
                 ('LOW_CONFIDENCE_BORDERLINE' if is_low_conf else 'NOMINAL'))
            ),
            'macro_anomaly_note': override_reason,
            'severity_proxy': severity,
            'spanwise_position_r_R': round(r_R, 3),

            # --- Disambiguated Coverage Metrics ---
            # 1. Primary Bounding-Box Frame Coverage (% of 2D image frame)
            'primary_bbox_frame_coverage_pct': round(primary_box_cov, 2),
            'primary_bbox_area_pct': round(primary_box_cov, 2),
            # 2. Analytical Sweep-Line Union Frame Coverage (% of 2D image frame)
            'union_frame_coverage_pct': image_frame_coverage_pct,
            'image_frame_coverage_pct': image_frame_coverage_pct,
            'union_bbox_area_pct': image_frame_coverage_pct,
            # 3. 3D Blade Damage Area Proxy (10% local frame scale, capped at 5.0% for BEM solver)
            'blade_damage_area_proxy_pct': blade_damage_area_proxy_pct,
            'damage_area_pct': blade_damage_area_proxy_pct,

            # --- Spatial Clustering of Detections ---
            'distinct_damage_regions': distinct_regions,
            'candidate_detections_count': len(detections),
            'has_overlapping_detections': has_overlapping,
            'detection_clustering_note': clustering_note,

            'blade_region': blade_region,
            'defect_count': len(detections),
            'crosscheck': crosscheck_data,
            'models_agree': primary.get('models_agree', None),
            'classification_disagreement': primary.get('classification_disagreement', False),
            'yolo_class': primary.get('yolo_class', primary_class),
            'yolo_confidence': primary.get('yolo_confidence', confidence),
            'mobilenet_class': primary.get('mobilenet_class'),
            'mobilenet_confidence': primary.get('mobilenet_confidence'),
            'decision_rule_applied': primary.get('decision_rule_applied', 'NONE'),
            'provenance': {
                'spanwise_position_source': spanwise_position_source,
                'damage_area_source': 'CONSERVATIVE_LOCAL_FRAME_BLADE_AREA_PROXY',
                'primary_bbox_frame_coverage_pct': round(primary_box_cov, 2),
                'union_frame_coverage_pct': image_frame_coverage_pct,
                'image_frame_coverage_pct': image_frame_coverage_pct,
                'blade_damage_area_proxy_pct': blade_damage_area_proxy_pct,
                'scale_proxy_method': 'CONSERVATIVE_LOCAL_FRAME_PROXY_0.10',
                'distinct_damage_regions': distinct_regions,
                'candidate_detections_count': len(detections),
                'uncertainty_flag': (
                    'STRUCTURAL_SAFETY_OVERRIDE' if structural_safety_override else
                    ('CLASSIFICATION_UNCERTAIN_EXTENSIVE_DAMAGE' if is_uncertain_extensive_damage else
                     'PROVISIONAL_OPTICAL_HEURISTIC')
                ),
                'provenance_note': (
                    override_reason
                    if structural_safety_override else
                    ('Extensive optical anomaly with low model classification confidence. Superficial craze cap overridden to prevent masking potential structural composite delamination; requires specialist physical inspection.'
                     if is_uncertain_extensive_damage else
                     'UNVALIDATED PROVISIONAL ASSUMPTION: union_frame_coverage_pct represents 2D photographic bounding-box area % of image frame. '
                     'blade_damage_area_proxy_pct is an uncalibrated provisional heuristic proxy (10% local frame scale capped at 5.0%) estimating blade surface fraction without empirical photogrammetric calibration or drone flight test evidence.')
                )
            }
        }
