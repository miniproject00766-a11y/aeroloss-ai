import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from src.image_io import load_rgb_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YOLO_CONFIDENCE_THRESHOLD = 0.10
YOLO_CONFIDENT_ANCHOR_THRESHOLD = 0.20

class YoloService:
    """
    Hybrid Vision & Detection Service for Wind Turbine Blade Defect Classification & Localization.
    Integrates fine-tuned PyTorch MobileNetV3 deep learning classifier, OOD Domain Guardrail,
    and YOLOv8 object detection overlay.
    """
    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        # Exactly match the 6 classes trained in models/best.pt and models/blade_defect_classifier.pth
        self.class_names = ['corrosion', 'crack', 'craze', 'hide_craze', 'surface_injure', 'thunderstrike']
        self.model = None

        # Check for fine-tuned YOLO model if present
        weights_paths = [
            os.path.join(MODELS_DIR, "best.pt"),
            os.path.join(MODELS_DIR, "yolov8_damage.pt"),
            os.path.join(BASE_DIR, "runs", "aeroloss_yolov8", "weights", "best.pt")
        ]
        for wp in weights_paths:
            if os.path.exists(wp):
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(wp)
                    if hasattr(self.model, 'names') and isinstance(self.model.names, dict):
                        # Ensure class names match model names exactly
                        self.class_names = [self.model.names[i] for i in sorted(self.model.names.keys())]
                    print(f"YoloService: Loaded fine-tuned YOLOv8 model from {wp} with classes {self.class_names}")
                    break
                except Exception as e:
                    print(f"YoloService: YOLO load error: {e}")

    def detect_defects(self, image_input, target_class=None):
        """
        Runs defect detection, OOD guardrail, and classification inference.
        Returns:
            detections: list of dicts with bbox, class, confidence, area_pct, r_over_R
            annotated_image: PIL Image with rendered visual bounding box overlay
            is_valid: bool
            rejection_reason: str
            guard_status: str
        """
        if isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            pil_img = load_rgb_image(image_input)

        w, h = pil_img.size

        # 1. Run MobileNetV3 Deep Learning Inference & Guardrail
        pred_class = target_class if target_class in self.class_names else None
        confidence = 0.92
        is_valid = True
        rejection_reason = "Valid wind turbine blade surface."
        probs = np.array([0.90])

        if self.pipeline and self.pipeline.vision_model is not None:
            tensor_img = self.pipeline.transform(pil_img).unsqueeze(0).to(self.pipeline.device)
            with torch.no_grad():
                logits = self.pipeline.vision_model(tensor_img)
                t = getattr(self.pipeline, 'temperature', 1.0)
                calibrated_logits = logits / max(0.1, t)
                probs = torch.softmax(calibrated_logits, dim=1).squeeze().cpu().numpy()
                pred_idx = int(np.argmax(probs))
                cnn_class = self.pipeline.class_names[pred_idx]
                cnn_conf = float(probs[pred_idx])

                # Check OOD Guardrail
                guard_res = self.pipeline.validate_blade_domain(pil_img, probs, cnn_conf)
                is_valid = guard_res[0]
                rejection_reason = guard_res[1]
                guard_status = guard_res[2] if len(guard_res) > 2 else ('ACCEPT' if is_valid else 'REJECT_OUT_OF_DOMAIN')
                
                if not pred_class:
                    pred_class = cnn_class
                    confidence = cnn_conf
                else:
                    confidence = cnn_conf
        else:
            guard_status = 'ACCEPT'

        if not pred_class:
            pred_class = 'unknown'
            confidence = 0.0

        draw_img = pil_img.copy()
        draw = ImageDraw.Draw(draw_img)

        # 2. If OOD rejected, draw warning banner and return
        if not is_valid:
            banner_h = max(70, int(h * 0.22))
            draw.rectangle([0, h//2 - banner_h//2, w, h//2 + banner_h//2], fill=(220, 38, 38), outline=(255, 255, 255), width=3)
            draw.text((w * 0.05, h//2 - 20), "🛑 REJECTED: OUT-OF-DOMAIN / NON-BLADE IMAGE", fill="white")
            detail_msg = rejection_reason[:65] if len(rejection_reason) > 65 else rejection_reason
            draw.text((w * 0.05, h//2 + 5), detail_msg, fill="white")
            return [], draw_img, False, rejection_reason, guard_status

        # 3. If YOLO model is available, run YOLO
        if self.model is not None:
            try:
                results = self.model(pil_img, conf=0.03, verbose=False)[0]
                raw_boxes = results.boxes
                detections = []
                if raw_boxes is not None and len(raw_boxes) > 0:
                    cand_list = []
                    for box in raw_boxes:
                        coords = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_idx = int(box.cls[0].cpu().numpy())
                        xmin, ymin, xmax, ymax = coords
                        bw = max(1.0, xmax - xmin)
                        bh = max(1.0, ymax - ymin)
                        box_aspect = max(bw / max(1.0, bh), bh / max(1.0, bw))
                        if box_aspect >= 10.0 and conf < 0.20:
                            continue
                        cand_list.append((box, coords, conf, cls_idx, bw, bh))

                    anchors = [c for c in cand_list if c[2] >= YOLO_CONFIDENCE_THRESHOLD]
                    selected_boxes = []
                    if anchors:
                        selected_boxes.extend(anchors)
                        for c in cand_list:
                            if c[2] < YOLO_CONFIDENCE_THRESHOLD:
                                for a in anchors:
                                    ix1 = max(c[1][0], a[1][0])
                                    iy1 = max(c[1][1], a[1][1])
                                    ix2 = min(c[1][2], a[1][2])
                                    iy2 = min(c[1][3], a[1][3])
                                    iw = max(0.0, ix2 - ix1)
                                    ih = max(0.0, iy2 - iy1)
                                    inter = iw * ih
                                    a_area = a[4] * a[5]
                                    if a_area > 0 and (inter / a_area) >= 0.75:
                                        c_area = c[4] * c[5]
                                        if c_area >= 1.25 * a_area:
                                            selected_boxes.append(c)
                                            break

                    for item in selected_boxes:
                        box, coords, conf, cls_idx, bw, bh = item
                        if hasattr(self.model, 'names') and cls_idx in self.model.names:
                            det_cls = self.model.names[cls_idx]
                        elif cls_idx < len(self.class_names):
                            det_cls = self.class_names[cls_idx]
                        else:
                            det_cls = 'unknown'

                        xmin, ymin, xmax, ymax = coords
                        area_pct = round(((bw * bh) / (w * h)) * 100.0, 2)
                        yc = (ymin + ymax) / 2.0
                        r_R = round(float(np.clip(0.20 + 0.78 * (yc / float(h)), 0.15, 0.98)), 3)

                        is_low_conf = bool(conf < 0.25)
                        conf_grade = 'LOW' if conf < 0.25 else ('MEDIUM' if conf < 0.60 else 'HIGH')

                        # --- Crop-Level MobileNetV3 Cross-Check (Phase 3) ---
                        c_xmin = max(0, min(w - 1, int(np.floor(xmin))))
                        c_ymin = max(0, min(h - 1, int(np.floor(ymin))))
                        c_xmax = max(0, min(w, int(np.ceil(xmax))))
                        c_ymax = max(0, min(h, int(np.ceil(ymax))))
                        crop_w = c_xmax - c_xmin
                        crop_h = c_ymax - c_ymin

                        mobilenet_cls = None
                        mobilenet_conf = None
                        mobilenet_probs_dict = {}
                        crosscheck_status = "NOT_EVALUATED"

                        if crop_w < 4 or crop_h < 4:
                            crosscheck_status = "INVALID_EMPTY_CROP"
                        elif self.pipeline and getattr(self.pipeline, 'vision_model', None) is not None:
                            try:
                                crop_pil = pil_img.crop((c_xmin, c_ymin, c_xmax, c_ymax))
                                t_crop = self.pipeline.transform(crop_pil).unsqueeze(0).to(self.pipeline.device)
                                with torch.no_grad():
                                    c_logits = self.pipeline.vision_model(t_crop)
                                    t = getattr(self.pipeline, 'temperature', 1.0)
                                    calib_logits = c_logits / max(0.1, t)
                                    probs = torch.softmax(calib_logits, dim=1).squeeze().cpu().numpy()
                                    mb_idx = int(np.argmax(probs))
                                    mobilenet_cls = self.pipeline.class_names[mb_idx]
                                    mobilenet_conf = round(float(probs[mb_idx]), 4)
                                    mobilenet_probs_dict = {
                                        self.pipeline.class_names[i]: round(float(probs[i]), 4)
                                        for i in range(len(self.pipeline.class_names))
                                    }
                                    crosscheck_status = "EVALUATED"
                            except Exception as e:
                                print(f"MobileNetV3 crop crosscheck error: {e}")
                                mobilenet_cls = None
                                mobilenet_conf = None
                                crosscheck_status = f"ERROR: {str(e)}"

                        # Explicit, conservative decision rule:
                        # 1. Consensus: if both agree, preserve consensus
                        # 2. Safety Override: if either model detects structural defect (crack/thunderstrike), prioritize structural safety
                        # 3. Non-structural Disagreement: keep YOLO's localized class as primary, flag for engineering review
                        # 4. Never blend or manufacture combined confidence
                        if mobilenet_cls is None:
                            primary_cls = det_cls
                            primary_conf = round(conf, 4)
                            is_agree = None
                            disagreement = False
                            decision_rule = f"YOLO_ONLY_{crosscheck_status}"
                        elif det_cls == mobilenet_cls:
                            primary_cls = det_cls
                            primary_conf = round(conf, 4)
                            is_agree = True
                            disagreement = False
                            decision_rule = "CONSENSUS_AGREEMENT"
                        else:
                            is_agree = False
                            disagreement = True
                            if det_cls in ['crack', 'thunderstrike']:
                                primary_cls = det_cls
                                primary_conf = round(conf, 4)
                                decision_rule = "SAFETY_OVERRIDE_YOLO_STRUCTURAL"
                            elif mobilenet_cls in ['crack', 'thunderstrike'] and mobilenet_conf >= 0.35:
                                primary_cls = mobilenet_cls
                                primary_conf = mobilenet_conf
                                decision_rule = "SAFETY_OVERRIDE_MOBILENET_STRUCTURAL_PROMOTION"
                            else:
                                primary_cls = det_cls
                                primary_conf = round(conf, 4)
                                decision_rule = "YOLO_LOCALIZATION_PRIMARY_DISAGREEMENT_REVIEW"

                        detections.append({
                            'damage_type': primary_cls,
                            'confidence': primary_conf,
                            'confidence_pct': f"{primary_conf*100:.1f}%",
                            'is_low_confidence': is_low_conf,
                            'confidence_grade': conf_grade,
                            'bounding_box': [round(c, 1) for c in coords],
                            'damage_area_percent': area_pct,
                            'r_over_R': r_R,
                            'detection_method': 'yolov8',
                            'is_synthetic': False,
                            'is_heuristic_fallback': False,
                            'crosscheck': {
                                'status': crosscheck_status,
                                'models_agree': is_agree,
                                'classification_disagreement': disagreement,
                                'yolo_class': det_cls,
                                'yolo_confidence': round(conf, 4),
                                'yolo_confidence_pct': f"{conf*100:.1f}%",
                                'mobilenet_class': mobilenet_cls,
                                'mobilenet_confidence': mobilenet_conf,
                                'mobilenet_confidence_pct': f"{mobilenet_conf*100:.1f}%" if mobilenet_conf is not None else None,
                                'mobilenet_probabilities': mobilenet_probs_dict,
                                'decision_rule_applied': decision_rule
                            },
                            'models_agree': is_agree,
                            'classification_disagreement': disagreement,
                            'yolo_class': det_cls,
                            'yolo_confidence': round(conf, 4),
                            'mobilenet_class': mobilenet_cls,
                            'mobilenet_confidence': mobilenet_conf,
                            'decision_rule_applied': decision_rule
                        })
                        color = "#ef4444" if primary_cls in ['crack', 'thunderstrike'] else ("#f59e0b" if primary_cls in ['corrosion', 'hide_craze'] else "#3b82f6")
                        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)
                        
                        if is_agree is True:
                            label_text = f"{primary_cls.upper()} {conf*100:.0f}% [MNet: {mobilenet_conf*100:.0f}%]"
                        elif disagreement is True and mobilenet_cls:
                            label_text = f"{primary_cls.upper()} {conf*100:.0f}% [!= {mobilenet_cls.upper()}]"
                        else:
                            label_suffix = " (Low Conf)" if is_low_conf else ""
                            label_text = f"{primary_cls.upper()} {conf*100:.1f}%{label_suffix}"

                        tag_w = len(label_text) * 8 + 14
                        draw.rectangle([xmin, max(0, ymin - 22), xmin + tag_w, ymin], fill=color)
                        draw.text((xmin + 4, max(0, ymin - 20)), label_text, fill="white")
                if detections:
                    return detections, draw_img, True, rejection_reason, guard_status
            except Exception as e:
                print(f"YOLO inference error: {e}")

                # 4. Multi-Cue Blade Defect Saliency & Localization
        np_arr = np.array(pil_img)
        img_bgr = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 4.1 Segment Blade Body (mask out sky and ground vegetation)
        is_sky = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 140) & (hsv[:, :, 1] > 25)
        is_green = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 25)
        blade_mask = ~(is_sky | is_green)

        # 4.2 Defect Feature Maps
        # (a) Dark fissures & crack shadows inside composite shell
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        dark_cracks = (gray < 110) & blade_mask

        # (b) Exposed foam core / laminate delamination (yellowish, brownish, dark carbon)
        core_damage = (hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] > 25) & blade_mask

        # (c) High-gradient fiber fracture edges
        edges = cv2.Canny(gray, 40, 130) & blade_mask

        # Combine saliency energy
        defect_energy = (blackhat.astype(np.float32) * 2.5) + (dark_cracks.astype(np.float32) * 60.0) + (core_damage.astype(np.float32) * 50.0) + (edges.astype(np.float32) * 1.5)
        defect_smooth = cv2.GaussianBlur(defect_energy, (19, 19), 0)

        # Check absolute physical energy across blade surface
        max_blade_energy = float(np.max(defect_smooth[blade_mask])) if np.any(blade_mask) else 0.0
        mean_blade_energy = float(np.mean(defect_smooth[blade_mask])) if np.any(blade_mask) else 0.0

        # Threshold top energy
        p92 = np.percentile(defect_smooth[blade_mask], 92) if np.any(blade_mask) else 30
        thresh = (defect_smooth > max(25.0, p92)).astype(np.uint8) * 255

        # Merge connected crack and fracture components
        kernel_merge = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_merge)

        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_box = None
        best_contour_energy = 0.0
        if contours:
            c_candidates = []
            for c in contours:
                cx, cy, cbw, cbh = cv2.boundingRect(c)
                aspect_ratio = max(cbw / max(1.0, float(cbh)), cbh / max(1.0, float(cbw)))
                # Bounding box must be substantial but not the entire frame or blade silhouette boundary
                if 400 <= cbw * cbh <= int(0.70 * w * h) and aspect_ratio < 6.0:
                    c_roi = defect_smooth[cy:cy+cbh, cx:cx+cbw]
                    c_sum = float(np.sum(c_roi))
                    c_max = float(np.max(c_roi)) if c_roi.size > 0 else 0.0
                    c_density = c_sum / float(cbw * cbh)
                    c_candidates.append((c, cx, cy, cbw, cbh, c_sum, c_max, c_density))

            if c_candidates:
                # Rank candidates by integrated defect energy
                best_c, x, y, bw, bh, best_sum, best_max, best_density = max(c_candidates, key=lambda item: item[5])
                
                # Physical validation evidence floor:
                # Authentic surface defects (cracks, leading edge erosion, oxidation) produce concentrated peak energy >= 180.0
                # Regions with low peak energy or density are merely gelcoat lighting variations on nominal clean blade surfaces
                if best_max >= 180.0 and best_density >= 2.5:
                    pad_x = int(bw * 0.15)
                    pad_y = int(bh * 0.15)
                    xmin = max(0, x - pad_x)
                    ymin = max(0, y - pad_y)
                    xmax = min(w, x + bw + pad_x)
                    ymax = min(h, y + bh + pad_y)
                    best_box = [xmin, ymin, xmax, ymax]
                    best_contour_energy = best_max

        # If no significant localized defect contour exists, return clean baseline status with 0 detections
        if not best_box:
            draw_img = pil_img.copy()
            draw = ImageDraw.Draw(draw_img)
            clean_label = "NOMINAL BLADE SURFACE [No Defect Detected]"
            draw.text((max(10, int(w * 0.04)), max(10, int(h * 0.04))), clean_label, fill=(34, 197, 94))
            draw.text((max(10, int(w * 0.04)), max(32, int(h * 0.08))), "Unconfirmed Baseline: Absence of optical surface defect is not proof of full-blade health.", fill=(180, 180, 180))
            return [], draw_img, True, "No reliable defect detected on turbine blade surface. Baseline zero loss; unconfirmed blade health.", guard_status

        xmin, ymin, xmax, ymax = best_box
        bw = max(1.0, xmax - xmin)
        bh = max(1.0, ymax - ymin)
        area_pct = round(((bw * bh) / (w * h)) * 100.0, 1)
        yc = (ymin + ymax) / 2.0
        r_R = round(float(np.clip(0.20 + 0.78 * (yc / float(h)), 0.15, 0.98)), 3)

        # 4.3 Crop-Level Defect Re-Classification
        # Feed the localized defect box to MobileNetV3 for conditional classification
        crop = img_bgr[ymin:ymax, xmin:xmax]
        crop_class_val = pred_class
        crop_conf_raw = 0.50
        crop_probs_dict = {}

        if crop.size > 0 and self.pipeline and self.pipeline.vision_model is not None:
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            t_crop = self.pipeline.transform(pil_crop).unsqueeze(0).to(self.pipeline.device)
            with torch.no_grad():
                c_logits = self.pipeline.vision_model(t_crop)
                t = getattr(self.pipeline, 'temperature', 1.0)
                calib_logits = c_logits / max(0.1, t)
                crop_probs = torch.softmax(calib_logits, dim=1).squeeze().cpu().numpy()
            crop_idx = int(np.argmax(crop_probs))
            crop_class_val = self.pipeline.class_names[crop_idx]
            crop_conf_raw = float(crop_probs[crop_idx])
            crop_probs_dict = {
                self.pipeline.class_names[i]: round(float(crop_probs[i]), 4)
                for i in range(len(self.pipeline.class_names))
            }

            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            dark_fissure_ratio = float(np.mean(gray_crop < 80))

            # Severe structural crack promotion: if macroscopic fracture or dark fissure exists
            if (crop_class_val == 'craze' or pred_class == 'craze') and (dark_fissure_ratio > 0.04 or area_pct > 10.0):
                crop_class_val = 'crack'

        # CRITICAL SAFEGUARD: Never present a closed-set classifier's conditional softmax confidence as proof that a defect exists!
        # Softmax in a closed 6-class model sums to 1.0 even on clean surfaces.
        # Heuristic detection confidence is calibrated by physical energy and strictly capped at provisional low-confidence (< 0.25).
        heuristic_conf = round(float(np.clip(best_contour_energy / 2000.0, 0.10, 0.24)), 4)

        detections = [{
            'damage_type': crop_class_val,
            'confidence': heuristic_conf,
            'confidence_pct': f"{heuristic_conf*100:.1f}%",
            'is_low_confidence': True,
            'confidence_grade': 'LOW',
            'bounding_box': [xmin, ymin, xmax, ymax],
            'damage_area_percent': area_pct,
            'r_over_R': r_R,
            'detection_method': 'opencv_heuristic_fallback',
            'is_synthetic': True,
            'is_heuristic_fallback': True,
            'crosscheck': {
                'status': 'HEURISTIC_FALLBACK_PROVISIONAL',
                'models_agree': None,
                'classification_disagreement': False,
                'yolo_class': None,
                'yolo_confidence': None,
                'yolo_confidence_pct': None,
                'mobilenet_class': crop_class_val,
                'mobilenet_confidence': heuristic_conf,
                'mobilenet_confidence_pct': f"{heuristic_conf*100:.1f}%",
                'mobilenet_conditional_softmax': round(crop_conf_raw, 4),
                'mobilenet_probabilities': crop_probs_dict,
                'decision_rule_applied': 'OPENCV_SALIENCY_FALLBACK'
            },
            'models_agree': None,
            'classification_disagreement': False,
            'yolo_class': None,
            'yolo_confidence': None,
            'mobilenet_class': crop_class_val,
            'mobilenet_confidence': heuristic_conf,
            'decision_rule_applied': 'OPENCV_SALIENCY_FALLBACK'
        }]

        color = "#ef4444" if crop_class_val in ['crack', 'thunderstrike'] else ("#f59e0b" if crop_class_val in ['corrosion', 'hide_craze'] else "#3b82f6")
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)
        label_text = f"{crop_class_val.upper()} {heuristic_conf*100:.1f}% [Heuristic Advisory]"
        tag_w = len(label_text) * 8 + 14
        draw.rectangle([xmin, max(0, ymin - 24), xmin + tag_w, ymin], fill=color)
        draw.text((xmin + 5, max(0, ymin - 22)), label_text, fill="white")

        return detections, draw_img, True, rejection_reason, guard_status
