"""
AeroLoss AI — Model 1 Unit Test Suite
Target: src.model1.image_quality_validator
Tests deterministic image quality checks and blade-relevance guardrails.
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model1.image_quality_validator import ImageQualityValidator, validate_inspection_image


@pytest.fixture
def validator():
    return ImageQualityValidator()


def test_valid_readable_blade_image(validator):
    """Test 1: Valid readable blade image from primary raw dataset."""
    blade_path = REPO_ROOT / "data" / "raw" / "images" / "0.jpg"
    assert blade_path.exists(), f"Primary image not found: {blade_path}"

    res = validator.validate(blade_path)
    assert res["decode_status"] == "OK"
    assert res["dimensions"]["width"] == 1024
    assert res["dimensions"]["height"] == 1024
    assert res["quality_decision"] == "USABLE"
    assert res["relevance_decision"] == "BLADE"
    assert res["requires_manual_review"] is False
    assert res["metrics"]["laplacian_blur_score"] > 5.0


def test_blurry_image(validator):
    """Test 2: Real and synthetic blurry image detection."""
    # Test real blurry example from repository
    blurry_path = REPO_ROOT / "data" / "negative_evaluation_samples" / "blurred_low_quality_indoor_01.jpg"
    if blurry_path.exists():
        res = validator.validate(blurry_path)
        assert res["quality_decision"] == "UNUSABLE"
        assert any("blur" in r.lower() or "resolution" in r.lower() for r in res["reasons"])

    # Test software unit fixture with synthetic Gaussian defocus
    sharp_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    import cv2
    blurred_array = cv2.GaussianBlur(sharp_array, (51, 51), sigmaX=20)
    blurred_res = validator.validate(blurred_array)
    assert blurred_res["quality_decision"] in ["UNUSABLE", "REVIEW_REQUIRED"]


def test_extremely_dark_image(validator):
    """Test 3: Detection of severe underexposure / blackout."""
    black_path = REPO_ROOT / "data" / "negative_evaluation_samples" / "synthetic_blank_black.jpg"
    if black_path.exists():
        res = validator.validate(black_path)
        assert res["quality_decision"] == "UNUSABLE"
        assert any("underexposure" in r.lower() or "dynamic range" in r.lower() for r in res["reasons"])

    # Synthetic software unit test fixture
    dark_array = np.zeros((512, 512, 3), dtype=np.uint8)
    dark_res = validator.validate(dark_array)
    assert dark_res["quality_decision"] == "UNUSABLE"
    assert dark_res["metrics"]["dark_pixel_fraction"] == 1.0


def test_overexposed_image(validator):
    """Test 4: Detection of severe overexposure / whiteout."""
    white_path = REPO_ROOT / "data" / "negative_evaluation_samples" / "synthetic_blank_white.jpg"
    if white_path.exists():
        res = validator.validate(white_path)
        assert res["quality_decision"] == "UNUSABLE"
        assert any("overexposure" in r.lower() or "dynamic range" in r.lower() for r in res["reasons"])

    # Synthetic software unit test fixture
    white_array = np.full((512, 512, 3), 255, dtype=np.uint8)
    white_res = validator.validate(white_array)
    assert white_res["quality_decision"] == "UNUSABLE"
    assert white_res["metrics"]["clipped_bright_fraction"] == 1.0


def test_very_low_resolution_image(validator):
    """Test 5: Detection of sub-critical image dimensions."""
    # Create 128x128 synthetic test fixture
    low_res_img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
    res = validator.validate(low_res_img)
    assert res["quality_decision"] == "UNUSABLE"
    assert any("sub-critical resolution" in r.lower() for r in res["reasons"])


def test_corrupted_or_unsupported_image(validator):
    """Test 6: Handling of non-existent or corrupted image files."""
    # Non-existent file
    non_existent = REPO_ROOT / "data" / "non_existent_image_12345.jpg"
    res_missing = validator.validate(non_existent)
    assert res_missing["quality_decision"] == "UNUSABLE"
    assert "FILE_NOT_FOUND" in res_missing["decode_status"]

    # Corrupted header file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_RANDOM_CORRUPTION_BYTES_12345")
        corrupted_path = Path(f.name)

    try:
        res_corrupt = validator.validate(corrupted_path)
        assert res_corrupt["quality_decision"] == "UNUSABLE"
        assert "DECODE_ERROR" in res_corrupt["decode_status"]
    finally:
        if corrupted_path.exists():
            corrupted_path.unlink()


def test_clear_non_blade_image(validator):
    """Test 7: Detection of clear non-blade images and routing of ambiguous out-of-domain objects."""
    # 1. Definitive non-blade document / chart verification
    chart_path = REPO_ROOT / "data" / "negative_evaluation_samples" / "chart_airfoil_polar_02.png"
    if chart_path.exists():
        res = validator.validate(chart_path)
        assert res["relevance_decision"] == "NON_BLADE"
        assert any("chart" in r.lower() or "document" in r.lower() for r in res["reasons"])

    # 2. Ambiguous out-of-domain natural object (pizza / organic texture)
    # Must NOT be classified as blade; must require manual review (not definitive false rejection)
    pizza_path = REPO_ROOT / "data" / "negative_evaluation_samples" / "food_pizza_01.jpg"
    if pizza_path.exists():
        res = validator.validate(pizza_path)
        assert res["relevance_decision"] in ["NON_BLADE", "UNCERTAIN"]
        assert res["requires_manual_review"] is True
        assert any("chromaticity" in r.lower() or "texture" in r.lower() for r in res["reasons"])


def test_ambiguous_image_triggers_manual_review(validator):
    """Test 8: Ambiguous pasture cow with distant turbine triggers review."""
    ambig_path = REPO_ROOT / "data" / "negative_evaluation_samples" / "ambiguous_pasture_cow_distant_turbine_01.jpg"
    if ambig_path.exists():
        res = validator.validate(ambig_path)
        assert res["requires_manual_review"] is True
        assert res["relevance_decision"] == "UNCERTAIN"
        assert any("pasture" in r.lower() or "vegetation" in r.lower() for r in res["reasons"])


def test_convenience_function():
    """Test 9: Convenience function validate_inspection_image."""
    test_array = np.full((600, 600, 3), 180, dtype=np.uint8)
    res = validate_inspection_image(test_array)
    assert isinstance(res, dict)
    assert "quality_decision" in res
    assert "relevance_decision" in res
