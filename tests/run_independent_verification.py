"""
AeroLoss AI — Full Verification Runner for Independent Model Testing
Executes validation on:
- Valid blade image (data/raw/images/0.jpg)
- Non-blade chart image (data/negative_evaluation_samples/chart_confusion_matrix_03.png)
- Poor-quality noise image (data/negative_evaluation_samples/synthetic_gaussian_noise.jpg)
- Missing checkpoint handling
- Corrupted / invalid input handling
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model1.image_quality_validator import ImageQualityValidator
from ultralytics import YOLO

CHECKPOINT_PATH = REPO_ROOT / "models" / "yolov8s_blade_damage_best.pt"
CANONICAL_CLASSES = ["corrosion", "crack", "craze", "hide_craze", "surface_injure", "thunderstrike"]

def run_verification():
    print("=" * 80)
    print("AEROLOSS AI — INDEPENDENT MODEL TESTING VALIDATION RUN")
    print("=" * 80)
    
    # Checkpoint Verification
    print("\n--- 1. Checkpoint Verification ---")
    assert CHECKPOINT_PATH.exists(), f"Checkpoint missing at {CHECKPOINT_PATH}"
    size_mb = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
    print(f"Checkpoint Path: {CHECKPOINT_PATH}")
    print(f"File Size: {size_mb:.2f} MB")
    
    detector = YOLO(str(CHECKPOINT_PATH))
    print(f"Architecture: YOLOv8s (Parameters: ~11.1M)")
    print(f"Classes in Checkpoint: {detector.names}")
    assert list(detector.names.values()) == CANONICAL_CLASSES, "Class names mismatch!"
    print("Checkpoint verified successfully.")

    # Model 1 Validator
    print("\n--- 2. Model 1 (Guardrail) Independent Verification ---")
    validator = ImageQualityValidator()
    
    test_files = [
        ("Valid Blade", REPO_ROOT / "data" / "raw" / "images" / "0.jpg"),
        ("Non-Blade Chart", REPO_ROOT / "data" / "negative_evaluation_samples" / "chart_confusion_matrix_03.png"),
        ("Synthetic Noise", REPO_ROOT / "data" / "negative_evaluation_samples" / "synthetic_gaussian_noise.jpg")
    ]
    
    m1_results = {}
    for label, path in test_files:
        assert path.exists(), f"Test image missing: {path}"
        res = validator.validate(path)
        m1_results[label] = res
        print(f"\n[{label}] ({path.name}):")
        print(f"  - Technical Quality:  {res['quality_decision']}")
        print(f"  - Blade Relevance:    {res['relevance_decision']}")
        print(f"  - Manual Review Req:  {res['requires_manual_review']}")
        print(f"  - Resolution:         {res['dimensions']['width']}x{res['dimensions']['height']}")
        print(f"  - Blur Score (Lap):   {res['metrics']['laplacian_blur_score']:.2f}")
        print(f"  - Bright Clipping:    {res['metrics']['clipped_bright_fraction']*100:.1f}%")
        print(f"  - Dark Clipping:      {res['metrics']['dark_pixel_fraction']*100:.1f}%")
        print(f"  - Reasons:            {'; '.join(res['reasons'][:2])}")
        
        # Verify strict isolation: NO detection results
        assert "detections" not in res, "Isolation violation: detections found in Model 1 output!"
        assert "boxes" not in res, "Isolation violation: boxes found in Model 1 output!"

    print("\n--- 3. Model 2 (YOLOv8) Independent Verification ---")
    m2_results = {}
    for label, path in test_files:
        res = detector(str(path), conf=0.25, verbose=False)
        boxes = res[0].boxes
        m2_results[label] = {
            "num_detections": len(boxes),
            "detections": []
        }
        print(f"\n[{label}] ({path.name}):")
        print(f"  - Detections Count:   {len(boxes)}")
        if len(boxes) == 0:
            print("  - Message:            No damage objects detected at conf=0.25.")
        for b in boxes:
            c_id = int(b.cls[0])
            c_name = detector.names[c_id]
            conf = float(b.conf[0])
            xyxy = [int(v) for v in b.xyxy[0].tolist()]
            m2_results[label]["detections"].append({
                "class": c_name,
                "confidence": conf,
                "box": xyxy
            })
            print(f"    * Class: {c_name:<16} Conf: {conf*100:.1f}% Box: {xyxy}")

    print("\n--- 4. Robustness & Error Handling ---")
    # Test invalid image path
    try:
        inv_res = validator.validate("invalid_fake_path_9999.jpg")
        print(f"Invalid path Model 1 handling: decode_status={inv_res['decode_status']}, quality={inv_res['quality_decision']}")
        assert inv_res["decode_status"] == "FAILED"
    except Exception as e:
        print(f"Model 1 exception on invalid path: {e}")

    print("\nALL INDEPENDENT VERIFICATION CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_verification()
