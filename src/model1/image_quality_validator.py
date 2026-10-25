"""
AeroLoss AI — Model 1: Image Quality Validation and Blade-Relevance Guardrail
Version: 1.0.0-provisional
Author: AeroLoss AI Senior Engineering Team

This module provides a deterministic, transparent, and explainable first-stage guardrail
for inspecting drone and field photographs before they reach the Model 2 YOLOv8 detector.
It evaluates technical usability (Dimension B) and blade relevance (Dimension A) as
strictly independent and orthogonal outputs.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import yaml
import numpy as np
from PIL import Image
import cv2

# Default fallback thresholds if YAML config is not found
DEFAULT_CONFIG = {
    "version": "1.0.0-provisional",
    "status": "PROVISIONAL_CALIBRATED",
    "resolution": {
        "min_width_usable": 512,
        "min_height_usable": 512,
        "min_width_unusable": 256,
        "min_height_unusable": 256,
        "aspect_ratio_min_usable": 0.33,
        "aspect_ratio_max_usable": 3.00,
    },
    "blur": {
        "laplacian_var_unusable": 1.0,
        "laplacian_var_review": 5.0,
        "laplacian_var_noise_cutoff": 10000.0,
    },
    "exposure": {
        "dark_pixel_threshold_unusable": 0.60,
        "dark_pixel_threshold_review": 0.25,
        "min_mean_brightness_unusable": 15.0,
        "min_mean_brightness_review": 30.0,
        "bright_pixel_threshold_unusable": 0.85,
        "bright_pixel_threshold_review": 0.35,
        "max_mean_brightness_unusable": 252.0,
        "max_mean_brightness_review": 240.0,
    },
    "contrast": {
        "min_std_unusable": 2.5,
        "min_std_review": 10.0,
        "min_entropy_unusable": 0.50,
        "min_entropy_review": 2.50,
    },
    "relevance_heuristics": {
        "chart_bright_fraction_threshold": 0.70,
        "chart_entropy_max": 2.5,
        "chart_min_blur_score": 1000.0,
        "chart_min_line_count": 50,
        "noise_entropy_min": 4.5,
        "noise_blur_min": 10000.0,
    },
    "policy": {
        "allow_manual_override": True,
        "conservative_mode": True,
        "separate_quality_and_relevance": True,
    }
}


def load_validator_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or return robust default."""
    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        return loaded
            except Exception:
                pass

    # Try default workspace config path
    repo_root = Path(__file__).resolve().parent.parent.parent
    std_config = repo_root / "configs" / "model1_quality_thresholds.yaml"
    if std_config.exists():
        try:
            with open(std_config, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    return loaded
        except Exception:
            pass

    return DEFAULT_CONFIG.copy()


class ImageQualityValidator:
    """Deterministic image quality and blade relevance validator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[Union[str, Path]] = None):
        if config is not None:
            self.config = config
        else:
            self.config = load_validator_config(config_path)

    def validate(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Validate an inspection image against technical quality and blade relevance checks.

        Returns a structured dictionary with separate quality_decision and relevance_decision.
        """
        reasons: List[str] = []
        quality_decision = "USABLE"
        relevance_decision = "BLADE"
        requires_manual_review = False

        pil_img: Optional[Image.Image] = None
        cv_img_bgr: Optional[np.ndarray] = None
        source_name = "in_memory"

        # -------------------------------------------------------------
        # 1. Decoding and Readability Check
        # -------------------------------------------------------------
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            source_name = p.name
            if not p.exists():
                return self._create_failure_response(
                    decode_status=f"FILE_NOT_FOUND: {p}",
                    reasons=[f"File does not exist: {p}"],
                    source_name=source_name
                )
            if p.stat().st_size == 0:
                return self._create_failure_response(
                    decode_status="ZERO_BYTE_FILE",
                    reasons=["File is empty (0 bytes)"],
                    source_name=source_name
                )
            try:
                with Image.open(p) as img_test:
                    img_test.verify()
                pil_img = Image.open(p)
                pil_img.load()
            except Exception as e:
                return self._create_failure_response(
                    decode_status=f"DECODE_ERROR: {str(e)}",
                    reasons=[f"Failed to decode image file: {str(e)}"],
                    source_name=source_name
                )
            try:
                cv_img_bgr = cv2.imread(str(p))
                if cv_img_bgr is None:
                    # Fallback via PIL RGB array
                    cv_img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception:
                cv_img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

        elif isinstance(image_input, Image.Image):
            pil_img = image_input.copy()
            cv_img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            cv_img_bgr = image_input.copy()
            if len(cv_img_bgr.shape) == 2:
                pil_img = Image.fromarray(cv_img_bgr)
                cv_img_bgr = cv2.cvtColor(cv_img_bgr, cv2.COLOR_GRAY2BGR)
            else:
                pil_img = Image.fromarray(cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB))
        else:
            return self._create_failure_response(
                decode_status="UNSUPPORTED_INPUT_TYPE",
                reasons=[f"Unsupported image input type: {type(image_input)}"],
                source_name=source_name
            )

        # -------------------------------------------------------------
        # 2. Dimensions and Aspect Ratio Check
        # -------------------------------------------------------------
        w, h = pil_img.size
        aspect_ratio = round(float(w) / float(h), 4) if h > 0 else 0.0
        total_pixels = w * h

        res_cfg = self.config.get("resolution", DEFAULT_CONFIG["resolution"])
        if w < res_cfg["min_width_unusable"] or h < res_cfg["min_height_unusable"]:
            quality_decision = "UNUSABLE"
            reasons.append(f"Sub-critical resolution: {w}x{h} (minimum usable is {res_cfg['min_width_unusable']}x{res_cfg['min_height_unusable']})")
        elif w < res_cfg["min_width_usable"] or h < res_cfg["min_height_usable"]:
            if quality_decision != "UNUSABLE":
                quality_decision = "REVIEW_REQUIRED"
            reasons.append(f"Borderline resolution: {w}x{h} (target is at least {res_cfg['min_width_usable']}x{res_cfg['min_height_usable']})")

        if aspect_ratio < res_cfg["aspect_ratio_min_usable"] or aspect_ratio > res_cfg["aspect_ratio_max_usable"]:
            if quality_decision != "UNUSABLE":
                quality_decision = "REVIEW_REQUIRED"
            reasons.append(f"Extreme aspect ratio: {aspect_ratio:.2f} (outside standard envelope [{res_cfg['aspect_ratio_min_usable']}, {res_cfg['aspect_ratio_max_usable']}])")

        # -------------------------------------------------------------
        # 3. Grayscale, Exposure, and Contrast Metrics
        # -------------------------------------------------------------
        gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
        gray_f32 = gray.astype(np.float32)

        mean_brightness = round(float(np.mean(gray_f32)), 2)
        std_brightness = round(float(np.std(gray_f32)), 2)
        dark_frac = round(float(np.sum(gray <= 15) / total_pixels), 4)
        bright_frac = round(float(np.sum(gray >= 245) / total_pixels), 4)

        # Shannon Entropy
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy = round(float(-np.sum(hist * np.log2(hist))), 3)

        # Contrast checks
        cnt_cfg = self.config.get("contrast", DEFAULT_CONFIG["contrast"])
        if std_brightness < cnt_cfg["min_std_unusable"] or entropy < cnt_cfg["min_entropy_unusable"]:
            quality_decision = "UNUSABLE"
            reasons.append(f"Flat dynamic range: std={std_brightness:.2f}, entropy={entropy:.2f} bits (solid/uniform content)")
        elif std_brightness < cnt_cfg["min_std_review"] or entropy < cnt_cfg["min_entropy_review"]:
            if quality_decision != "UNUSABLE":
                quality_decision = "REVIEW_REQUIRED"
            reasons.append(f"Low contrast: std={std_brightness:.2f}, entropy={entropy:.2f} bits")

        # Exposure checks
        exp_cfg = self.config.get("exposure", DEFAULT_CONFIG["exposure"])
        if dark_frac > exp_cfg["dark_pixel_threshold_unusable"] or mean_brightness < exp_cfg["min_mean_brightness_unusable"]:
            quality_decision = "UNUSABLE"
            reasons.append(f"Severe underexposure: dark_fraction={dark_frac*100:.1f}%, mean_brightness={mean_brightness:.1f}")
        elif dark_frac > exp_cfg["dark_pixel_threshold_review"] or mean_brightness < exp_cfg["min_mean_brightness_review"]:
            if quality_decision != "UNUSABLE":
                quality_decision = "REVIEW_REQUIRED"
            reasons.append(f"Heavy shadow / low illumination: dark_fraction={dark_frac*100:.1f}%, mean={mean_brightness:.1f}")

        if bright_frac > exp_cfg["bright_pixel_threshold_unusable"] or mean_brightness > exp_cfg["max_mean_brightness_unusable"]:
            quality_decision = "UNUSABLE"
            reasons.append(f"Severe overexposure / whiteout: bright_fraction={bright_frac*100:.1f}%, mean_brightness={mean_brightness:.1f}")
        elif bright_frac > exp_cfg["bright_pixel_threshold_review"] or mean_brightness > exp_cfg["max_mean_brightness_review"]:
            if quality_decision != "UNUSABLE":
                quality_decision = "REVIEW_REQUIRED"
            reasons.append(f"Intense solar glint or washed-out sky: bright_fraction={bright_frac*100:.1f}%, mean={mean_brightness:.1f}")

        # -------------------------------------------------------------
        # 4. Focus and Blur Metric (Variance of Laplacian)
        # -------------------------------------------------------------
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = round(float(lap.var()), 2)

        blur_cfg = self.config.get("blur", DEFAULT_CONFIG["blur"])
        is_synthetic_noise = False
        if blur_score > blur_cfg["laplacian_var_noise_cutoff"]:
            is_synthetic_noise = True
            reasons.append(f"Anomalous high-frequency variance: blur_score={blur_score:.1f} (indicates synthetic/sensor noise)")
        elif blur_score < blur_cfg["laplacian_var_unusable"]:
            quality_decision = "UNUSABLE"
            reasons.append(f"Severe optical defocus / motion blur: blur_score={blur_score:.2f} (cutoff is {blur_cfg['laplacian_var_unusable']})")
        elif blur_score < blur_cfg["laplacian_var_review"]:
            if quality_decision != "UNUSABLE":
                quality_decision = "REVIEW_REQUIRED"
            reasons.append(f"Moderate softness / weak focus: blur_score={blur_score:.2f} (review threshold is {blur_cfg['laplacian_var_review']})")

        # -------------------------------------------------------------
        # 5. Blade Relevance Baseline (Conservative Visual Heuristics)
        # -------------------------------------------------------------
        # Compute HSV chromaticity and Canny edge statistics
        hsv = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0
        mean_saturation = round(float(np.mean(sat)), 3)
        mean_value = round(float(np.mean(val)), 3)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = round(float(np.sum(edges > 0) / total_pixels), 4)

        # Detect linear structures (dominant turbine blade edges)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min(w, h) // 8, maxLineGap=20)
        line_count = len(lines) if lines is not None else 0

        # Heuristic 5a: Synthetic noise rejection
        rel_cfg = self.config.get("relevance_heuristics", DEFAULT_CONFIG["relevance_heuristics"])
        if is_synthetic_noise and entropy > rel_cfg["noise_entropy_min"]:
            relevance_decision = "NON_BLADE"
            if quality_decision != "UNUSABLE":
                quality_decision = "UNUSABLE"
            reasons.append("Subject classified as synthetic noise artifact")

        # Heuristic 5b: Blank or uniform frame rejection
        elif std_brightness < cnt_cfg["min_std_unusable"]:
            relevance_decision = "NON_BLADE"
            reasons.append("Subject has zero discernible visual structure (solid/blank)")

        # Heuristic 5c: Digital chart / diagram rejection
        # Requires high white background, low entropy, AND verified artificial sharpness (high blur variance & sharp line/grid markers).
        # Overcast sky and solar glint blade photos do NOT exhibit digital chart sharpness/grid lines and are NOT rejected as NON_BLADE.
        elif (bright_frac > rel_cfg.get("chart_bright_fraction_threshold", 0.70) and 
              entropy < rel_cfg.get("chart_entropy_max", 2.5) and 
              blur_score > rel_cfg.get("chart_min_blur_score", 1000.0) and 
              line_count >= rel_cfg.get("chart_min_line_count", 50)):
            relevance_decision = "NON_BLADE"
            reasons.append(f"Subject exhibits digital document / chart properties (bright_fraction={bright_frac:.2f}, entropy={entropy:.2f}, blur={blur_score:.1f}, lines={line_count})")

        # Heuristic 5d: Distant turbine / pasture / landscape ambiguity
        # E.g. pasture with cow: vegetation color present without prominent blade structure
        green_mask = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 90) & (sat > 0.15)
        green_fraction = round(float(np.sum(green_mask) / total_pixels), 3)

        # Line angle distribution for structural orientation
        line_angles = []
        if lines is not None and len(lines) > 0:
            for line in lines:
                pts = line.ravel()
                ang = np.degrees(np.arctan2(pts[3] - pts[1], pts[2] - pts[0])) % 180
                line_angles.append(ang)
        line_angle_std = round(float(np.std(line_angles)), 2) if len(line_angles) >= 5 else 0.0

        if green_fraction > 0.20 and line_count < 10:
            relevance_decision = "UNCERTAIN"
            if quality_decision == "USABLE":
                quality_decision = "REVIEW_REQUIRED"
            reasons.append(f"Pasture/landscape vegetation prominent ({green_fraction*100:.1f}% green hue); blade not primary foreground")

        # Heuristic 5e: Ambiguous organic texture / high chromaticity
        # Note: Turbine blade defects frequently exhibit orange/rust corrosion, composite peeling, or paint degradation.
        # Rule: Do NOT use color saturation, edge density, or texture alone as definitive NON_BLADE evidence.
        # Ambiguous cases route to UNCERTAIN / REVIEW_REQUIRED rather than false rejection.
        elif mean_saturation > 0.40 and edge_density > 0.04 and (line_angle_std > 40.0 or line_count == 0):
            relevance_decision = "UNCERTAIN"
            if quality_decision == "USABLE":
                quality_decision = "REVIEW_REQUIRED"
            reasons.append(f"Atypical surface chromaticity or multi-directional edge texture (mean_saturation={mean_saturation:.2f}, line_angle_std={line_angle_std:.1f} deg); routed to manual review")

        # -------------------------------------------------------------
        # 6. Overall Aggregation
        # -------------------------------------------------------------
        if quality_decision == "REVIEW_REQUIRED" or relevance_decision == "UNCERTAIN":
            requires_manual_review = True

        if not reasons:
            reasons.append("Image passes all technical usability criteria; valid blade inspection framing")

        return {
            "source_name": source_name,
            "decode_status": "OK",
            "quality_decision": quality_decision,
            "relevance_decision": relevance_decision,
            "requires_manual_review": requires_manual_review,
            "reasons": reasons,
            "dimensions": {
                "width": w,
                "height": h,
                "aspect_ratio": aspect_ratio,
                "total_pixels": total_pixels
            },
            "metrics": {
                "laplacian_blur_score": blur_score,
                "brightness_mean": mean_brightness,
                "brightness_std": std_brightness,
                "dark_pixel_fraction": dark_frac,
                "clipped_bright_fraction": bright_frac,
                "entropy": entropy,
                "edge_density": edge_density,
                "mean_saturation": mean_saturation,
                "prominent_line_count": line_count
            },
            "provisional_thresholds_applied": {
                "laplacian_unusable": blur_cfg["laplacian_var_unusable"],
                "laplacian_review": blur_cfg["laplacian_var_review"],
                "dark_pixel_unusable": exp_cfg["dark_pixel_threshold_unusable"],
                "bright_pixel_unusable": exp_cfg["bright_pixel_threshold_unusable"],
                "min_contrast_std_unusable": cnt_cfg["min_std_unusable"]
            }
        }

    def _create_failure_response(self, decode_status: str, reasons: List[str], source_name: str) -> Dict[str, Any]:
        """Generate structured response for unreadable or non-existent files."""
        return {
            "source_name": source_name,
            "decode_status": decode_status,
            "quality_decision": "UNUSABLE",
            "relevance_decision": "UNCERTAIN",
            "requires_manual_review": False,
            "reasons": reasons,
            "dimensions": {
                "width": 0,
                "height": 0,
                "aspect_ratio": 0.0,
                "total_pixels": 0
            },
            "metrics": {
                "laplacian_blur_score": 0.0,
                "brightness_mean": 0.0,
                "brightness_std": 0.0,
                "dark_pixel_fraction": 0.0,
                "clipped_bright_fraction": 0.0,
                "entropy": 0.0,
                "edge_density": 0.0,
                "mean_saturation": 0.0,
                "prominent_line_count": 0
            },
            "provisional_thresholds_applied": {}
        }


def validate_inspection_image(image_input: Union[str, Path, np.ndarray, Image.Image], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function for validating an inspection image.
    Independent of YOLOv8; returns structured dictionary with decisions and metrics.
    """
    validator = ImageQualityValidator(config=config)
    return validator.validate(image_input)


if __name__ == "__main__":
    # Self-test if executed directly
    print("Testing ImageQualityValidator self-test...")
    validator = ImageQualityValidator()
    # Test blank black array
    blank_black = np.zeros((512, 512, 3), dtype=np.uint8)
    res = validator.validate(blank_black)
    print("Blank black test:", res["quality_decision"], res["relevance_decision"], res["reasons"])
    print("Validator module initialized successfully.")
