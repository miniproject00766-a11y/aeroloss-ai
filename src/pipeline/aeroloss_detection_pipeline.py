"""
AeroLoss AI — Integrated Model 1 & YOLOv8 Blade Damage Detection Pipeline
Module: src.pipeline.aeroloss_detection_pipeline

Orchestrates:
1. Model 1 Technical Quality Validation (decoding, blur, exposure, contrast).
2. Model 1 Domain Relevance Guardrail (blank, chart, noise, non-blade rejection).
3. Supervisory Routing: Stops unusable images; routes ambiguous cases to review.
4. Downstream YOLOv8 Damage Detection: Executed ONLY when guardrail criteria are met.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
from PIL import Image
import cv2

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model1.image_quality_validator import ImageQualityValidator, validate_inspection_image

# Canonical 6 blade damage classes matching Pascal VOC XML and YOLO training
CANONICAL_DAMAGE_CLASSES = [
    "corrosion",
    "crack",
    "craze",
    "hide_craze",
    "surface_injure",
    "thunderstrike"
]


class AeroLossDetectionPipeline:
    """
    Two-stage inspection pipeline:
    Stage 1: Deterministic Image Quality & Domain Relevance Guardrail (Model 1)
    Stage 2: YOLOv8 Blade Defect Localization and Classification
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        quality_config: Optional[Dict[str, Any]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Initialize the integrated detection pipeline.

        Args:
            weights_path: Path to trained YOLOv8 weights (.pt). If None, searches standard paths.
            quality_config: Optional custom threshold dict or path for Model 1 validator.
            conf_threshold: Default confidence threshold for damage detection.
            iou_threshold: Default NMS IoU threshold for damage detection.
            device: Compute device ('cpu', 'cuda', or None for auto).
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.class_names = list(CANONICAL_DAMAGE_CLASSES)

        # Initialize Model 1 Guardrail
        self.validator = ImageQualityValidator(config=quality_config)

        # Locate and load YOLOv8 detector
        self.model = None
        self.weights_path = None
        self.detector_loaded = False

        candidate_weights = []
        if weights_path is not None:
            candidate_weights.append(Path(weights_path))
        candidate_weights.extend([
            PROJECT_ROOT / "models" / "detection" / "yolov8_defect" / "weights" / "best.pt",
            PROJECT_ROOT / "models" / "best.pt",
            PROJECT_ROOT / "models" / "yolov8_damage.pt"
        ])

        for cp in candidate_weights:
            if cp.exists() and cp.is_file():
                try:
                    from ultralytics import YOLO
                    loaded = YOLO(str(cp))
                    # Verify model class compatibility
                    if hasattr(loaded, "names") and isinstance(loaded.names, dict):
                        # Verify whether it matches blade damage classes
                        loaded_classes = [loaded.names[i] for i in sorted(loaded.names.keys())]
                        # Accept if class names overlap with canonical classes
                        self.class_names = loaded_classes
                    self.model = loaded
                    self.weights_path = str(cp)
                    self.detector_loaded = True
                    break
                except Exception as e:
                    # Model failed to load (e.g. incompatible checkpoint or ultralytics error)
                    self.model = None
                    self.detector_loaded = False

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        allow_review_passthrough: bool = False,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute the full two-stage pipeline on an inspection image.

        Args:
            image_input: Path to image file, PIL Image, or BGR/RGB numpy array.
            allow_review_passthrough: If True, executes downstream YOLO detection on
                images flagged as REVIEW_REQUIRED, attaching a provisional warning.
                If False (default, recommended), halts at review state to prevent
                unverified non-blade imagery from receiving automated defect scores.
            conf_threshold: Optional override for detector confidence threshold.
            iou_threshold: Optional override for detector NMS IoU threshold.

        Returns:
            Structured dictionary containing Model 1 decisions, YOLOv8 detections,
            quality metrics, supervisory routing, and comprehensive warnings.
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        # -------------------------------------------------------------
        # Stage 1: Model 1 Quality and Relevance Validation
        # -------------------------------------------------------------
        guardrail_result = self.validator.validate(image_input)

        quality_decision = guardrail_result["quality_decision"]      # USABLE | REVIEW_REQUIRED | UNUSABLE
        relevance_decision = guardrail_result["relevance_decision"]  # BLADE | UNCERTAIN | NON_BLADE
        requires_manual_review = guardrail_result["requires_manual_review"]
        reasons = list(guardrail_result["reasons"])
        metrics = guardrail_result["metrics"]
        dimensions = guardrail_result["dimensions"]

        warnings: List[str] = []
        detections: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # Stage 2: Guardrail Decision & Routing
        # -------------------------------------------------------------

        # Branch 1: Definitive Non-Blade Out-of-Domain Rejection (Chart, Blank, Synthetic Noise)
        if relevance_decision == "NON_BLADE":
            pipeline_status = "STOPPED_NON_BLADE_DOMAIN"
            damage_detection_status = "SKIPPED_DOMAIN_REJECTION"
            summary = (
                f"Image rejected at Stage 1: Subject identified as NON_BLADE content. "
                f"Reasons: {'; '.join(reasons)}. "
                f"Inspection aborted; out-of-domain asset."
            )
            return self._build_response(
                pipeline_status=pipeline_status,
                quality_decision=quality_decision,
                relevance_decision=relevance_decision,
                requires_manual_review=requires_manual_review,
                damage_detection_status=damage_detection_status,
                reasons=reasons,
                metrics=metrics,
                dimensions=dimensions,
                detections=detections,
                warnings=warnings,
                summary=summary
            )

        # Branch 2: Critical Image Quality Failure (Defocus, Blackout, Whiteout, Corrupt)
        if quality_decision == "UNUSABLE":
            pipeline_status = "STOPPED_UNUSABLE_QUALITY"
            damage_detection_status = "SKIPPED_QUALITY_DEFECT"
            summary = (
                f"Image rejected at Stage 1: Image is technically UNUSABLE. "
                f"Issues: {'; '.join(reasons)}. "
                f"A replacement inspection capture is required before damage detection can proceed."
            )
            return self._build_response(
                pipeline_status=pipeline_status,
                quality_decision=quality_decision,
                relevance_decision=relevance_decision,
                requires_manual_review=requires_manual_review,
                damage_detection_status=damage_detection_status,
                reasons=reasons,
                metrics=metrics,
                dimensions=dimensions,
                detections=detections,
                warnings=warnings,
                summary=summary
            )

        # Branch 3: Ambiguous or Review-Required State
        if requires_manual_review:
            pipeline_status = "REVIEW_REQUIRED"
            warnings.append(
                "Image exhibits borderline technical quality or ambiguous blade relevance. "
                "Human operator review is required to verify blade surface context."
            )

            if not allow_review_passthrough:
                damage_detection_status = "PENDING_OPERATOR_REVIEW"
                summary = (
                    f"Inspection held at Stage 1 for manual review ({'; '.join(reasons)}). "
                    f"Downstream automated damage detection was not executed to prevent false positives."
                )
                return self._build_response(
                    pipeline_status=pipeline_status,
                    quality_decision=quality_decision,
                    relevance_decision=relevance_decision,
                    requires_manual_review=requires_manual_review,
                    damage_detection_status=damage_detection_status,
                    reasons=reasons,
                    metrics=metrics,
                    dimensions=dimensions,
                    detections=detections,
                    warnings=warnings,
                    summary=summary
                )
            else:
                warnings.append(
                    "PROVISIONAL INFERENCE: allow_review_passthrough is enabled. "
                    "Downstream detections MUST NOT be treated as verified blade damage without human confirmation."
                )

        # -------------------------------------------------------------
        # Stage 3: YOLOv8 Blade Damage Detection Execution
        # -------------------------------------------------------------
        if not self.detector_loaded or self.model is None:
            pipeline_status = "PASSED_GUARDRAIL" if not requires_manual_review else "REVIEW_REQUIRED"
            damage_detection_status = "DETECTOR_MODEL_UNAVAILABLE"
            warnings.append("YOLOv8 detector model weights are not loaded. Defect detection could not be performed.")
            summary = "Model 1 guardrail passed; damage detection pending trained YOLOv8 model weights."
            return self._build_response(
                pipeline_status=pipeline_status,
                quality_decision=quality_decision,
                relevance_decision=relevance_decision,
                requires_manual_review=requires_manual_review,
                damage_detection_status=damage_detection_status,
                reasons=reasons,
                metrics=metrics,
                dimensions=dimensions,
                detections=detections,
                warnings=warnings,
                summary=summary
            )

        # Prepare image for YOLO inference
        if isinstance(image_input, (str, Path)):
            yolo_input = str(image_input)
        elif isinstance(image_input, Image.Image):
            yolo_input = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            yolo_input = image_input
        else:
            yolo_input = str(image_input)

        try:
            results = self.model.predict(
                source=yolo_input,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False
            )
            yolo_res = results[0]
            boxes = yolo_res.boxes
            img_w = dimensions.get("width", 1024) or 1024
            img_h = dimensions.get("height", 1024) or 1024

            if boxes is not None and len(boxes) > 0:
                for idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    xywh_norm = box.xywhn[0].tolist() if hasattr(box, "xywhn") else [
                        (xyxy[0] + xyxy[2]) / (2.0 * img_w),
                        (xyxy[1] + xyxy[3]) / (2.0 * img_h),
                        (xyxy[2] - xyxy[0]) / img_w,
                        (xyxy[3] - xyxy[1]) / img_h
                    ]

                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                    box_area = max(0.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
                    area_fraction = round(box_area / (img_w * img_h), 6)

                    detections.append({
                        "box_id": idx + 1,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(score, 4),
                        "bbox_xyxy": [round(c, 2) for c in xyxy],
                        "bbox_xywh_norm": [round(c, 6) for c in xywh_norm],
                        "area_fraction": area_fraction
                    })

            damage_detection_status = "SUCCESS"
            pipeline_status = "PASSED_GUARDRAIL" if not requires_manual_review else "REVIEW_REQUIRED"

            if len(detections) == 0:
                summary = "Inspection completed: Image passed guardrail. No blade surface damage detected (clean blade)."
            else:
                defect_counts = {}
                for d in detections:
                    cname = d["class_name"]
                    defect_counts[cname] = defect_counts.get(cname, 0) + 1
                counts_str = ", ".join(f"{cnt}x {cn}" for cn, cnt in defect_counts.items())
                summary = f"Inspection completed: Detected {len(detections)} defect instance(s) ({counts_str})."

        except Exception as e:
            damage_detection_status = "DETECTION_ERROR"
            warnings.append(f"YOLOv8 inference exception: {str(e)}")
            summary = f"Model 1 guardrail passed, but YOLOv8 inference failed: {str(e)}"
            pipeline_status = "PASSED_GUARDRAIL" if not requires_manual_review else "REVIEW_REQUIRED"

        return self._build_response(
            pipeline_status=pipeline_status,
            quality_decision=quality_decision,
            relevance_decision=relevance_decision,
            requires_manual_review=requires_manual_review,
            damage_detection_status=damage_detection_status,
            reasons=reasons,
            metrics=metrics,
            dimensions=dimensions,
            detections=detections,
            warnings=warnings,
            summary=summary
        )

    def _build_response(
        self,
        pipeline_status: str,
        quality_decision: str,
        relevance_decision: str,
        requires_manual_review: bool,
        damage_detection_status: str,
        reasons: List[str],
        metrics: Dict[str, Any],
        dimensions: Dict[str, Any],
        detections: List[Dict[str, Any]],
        warnings: List[str],
        summary: str
    ) -> Dict[str, Any]:
        """Construct the canonical structured response dictionary."""
        return {
            "pipeline_version": "1.0.0",
            "pipeline_status": pipeline_status,
            "quality_decision": quality_decision,
            "relevance_decision": relevance_decision,
            "requires_manual_review": requires_manual_review,
            "damage_detection_status": damage_detection_status,
            "detector_model_loaded": self.detector_loaded,
            "detector_weights_path": self.weights_path,
            "guardrail_reasons": reasons,
            "quality_metrics": metrics,
            "dimensions": dimensions,
            "detections_count": len(detections),
            "detections": detections,
            "warnings": warnings,
            "summary": summary
        }


def run_aeroloss_detection_pipeline(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    weights_path: Optional[Union[str, Path]] = None,
    quality_config: Optional[Dict[str, Any]] = None,
    allow_review_passthrough: bool = False,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Dict[str, Any]:
    """
    Convenience function to run the complete AeroLoss Model 1 + YOLOv8 pipeline.
    """
    pipeline = AeroLossDetectionPipeline(
        weights_path=weights_path,
        quality_config=quality_config,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )
    return pipeline.predict(image_input, allow_review_passthrough=allow_review_passthrough)


if __name__ == "__main__":
    print("Testing AeroLossDetectionPipeline self-test...")
    pipe = AeroLossDetectionPipeline()
    print("Pipeline initialized. Detector loaded:", pipe.detector_loaded, pipe.weights_path)

    # Test blank black array
    blank = np.zeros((512, 512, 3), dtype=np.uint8)
    res = pipe.predict(blank)
    print("Blank array test status:", res["pipeline_status"], res["damage_detection_status"])
    print("Summary:", res["summary"])
