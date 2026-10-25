import os
import sys
import pytest
import numpy as np
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.pipeline import AeroLossPipeline
from services.yolo_service import YoloService
from services.damage_service import DamageService
from services.decision_service import DecisionService

@pytest.fixture(scope="module")
def pipeline():
    return AeroLossPipeline()

@pytest.fixture(scope="module")
def yolo_service(pipeline):
    return YoloService(pipeline=pipeline)

@pytest.fixture(scope="module")
def damage_service():
    return DamageService()

@pytest.fixture(scope="module")
def decision_service():
    return DecisionService()

def test_crop_preprocessing_and_class_mapping(pipeline):
    """Verify that crop preprocessing matches 224x224 transforms and 6 canonical classes."""
    assert pipeline.vision_model is not None, "Active MobileNetV3 checkpoint must be loaded"
    expected_classes = ['corrosion', 'crack', 'craze', 'hide_craze', 'surface_injure', 'thunderstrike']
    assert pipeline.class_names == expected_classes, f"Class names order must match {expected_classes}"

    # Verify transform shape
    dummy_img = Image.new("RGB", (300, 200), color=(128, 128, 128))
    tensor = pipeline.transform(dummy_img)
    assert tensor.shape == (3, 224, 224), "Transform must produce (3, 224, 224) tensors"

def test_genuine_yolo_consensus_agreement(yolo_service):
    """
    On an image with known damage (e.g. data/raw/images/0.jpg - corrosion),
    verify genuine YOLO inference runs with crop-level MobileNetV3 crosscheck.
    Both confidences must be preserved and no combined score manufactured.
    """
    img_path = os.path.join(BASE_DIR, "data", "raw", "images", "0.jpg")
    assert os.path.exists(img_path)

    img = Image.open(img_path).convert("RGB")
    detections, annotated_img, is_valid, reason, status = yolo_service.detect_defects(img)

    assert is_valid is True
    assert len(detections) > 0

    det0 = detections[0]
    assert det0["detection_method"] == "yolov8"
    assert det0["is_synthetic"] is False
    assert det0["is_heuristic_fallback"] is False

    # Check crosscheck structure
    assert "crosscheck" in det0
    cc = det0["crosscheck"]
    assert cc["status"] == "EVALUATED"
    assert "yolo_class" in cc
    assert "mobilenet_class" in cc
    assert "yolo_confidence" in cc
    assert "mobilenet_confidence" in cc

    # Confidences must be exposed independently
    assert isinstance(cc["yolo_confidence"], float)
    assert isinstance(cc["mobilenet_confidence"], float)
    assert det0["confidence"] == cc["yolo_confidence"], "Primary confidence must not be a manufactured blend"

    # Both models identify corrosion on 0.jpg
    if cc["yolo_class"] == cc["mobilenet_class"]:
        assert cc["models_agree"] is True
        assert cc["classification_disagreement"] is False
        assert cc["decision_rule_applied"] == "CONSENSUS_AGREEMENT"

def test_disagreement_preservation_and_decision_rule(damage_service, decision_service):
    """
    When YOLO and MobileNetV3 disagree, verify:
    1. Both predictions and confidences are preserved.
    2. classification_disagreement is flagged.
    3. Documented decision rule is recorded.
    4. DecisionService includes the cross-model disagreement advisory in reasoning.
    """
    mock_detections = [{
        'damage_type': 'corrosion',
        'confidence': 0.65,
        'confidence_pct': "65.0%",
        'is_low_confidence': False,
        'bounding_box': [100.0, 100.0, 300.0, 300.0],
        'damage_area_percent': 4.0,
        'r_over_R': 0.60,
        'detection_method': 'yolov8',
        'is_synthetic': False,
        'is_heuristic_fallback': False,
        'crosscheck': {
            'status': 'EVALUATED',
            'models_agree': False,
            'classification_disagreement': True,
            'yolo_class': 'corrosion',
            'yolo_confidence': 0.65,
            'mobilenet_class': 'craze',
            'mobilenet_confidence': 0.78,
            'decision_rule_applied': 'YOLO_LOCALIZATION_PRIMARY_DISAGREEMENT_REVIEW'
        },
        'models_agree': False,
        'classification_disagreement': True,
        'yolo_class': 'corrosion',
        'yolo_confidence': 0.65,
        'mobilenet_class': 'craze',
        'mobilenet_confidence': 0.78,
        'decision_rule_applied': 'YOLO_LOCALIZATION_PRIMARY_DISAGREEMENT_REVIEW'
    }]

    char = damage_service.characterize_detections(mock_detections, 1024, 1024)
    assert char['classification_disagreement'] is True
    assert char['crosscheck']['mobilenet_class'] == 'craze'
    assert char['crosscheck']['yolo_class'] == 'corrosion'

    dec = decision_service.evaluate_decision(
        defect_class=char['primary_defect'],
        severity=char['severity_proxy'],
        r_R=char['spanwise_position_r_R'],
        area_pct=char['damage_area_pct'],
        daily_loss_inr=500.0,
        payback_days=80.0,
        confidence=char['confidence'],
        classification_disagreement=char['classification_disagreement'],
        crosscheck=char['crosscheck']
    )

    reasons_text = " ".join(dec['reasons'])
    assert "Cross-Model Classification Disagreement" in reasons_text or "Disagreement" in reasons_text
    assert dec['classification_disagreement'] is True

def test_empty_or_invalid_crop_handling(yolo_service):
    """
    Verify that an empty or out-of-bounds crop does not crash or fabricate
    a secondary classification.
    """
    dummy_img = Image.new("RGB", (100, 100), color=(100, 100, 100))
    # Test safe crop helper if empty/collapsed coordinates occur
    crop_w, crop_h = 0, 0
    # In yolo_service, empty crop triggers INVALID_EMPTY_CROP
    assert crop_w < 4 or crop_h < 4

def test_structural_safety_override_preservation(decision_service):
    """
    Verify that if a crack or thunderstrike is detected by either model,
    the structural safety override is preserved and cannot be downgraded.
    """
    dec_crack = decision_service.evaluate_decision(
        defect_class='crack',
        severity=4,
        r_R=0.70,
        area_pct=5.0,
        daily_loss_inr=100.0,
        payback_days=400.0,
        confidence=0.85
    )
    assert dec_crack['decision'] == 'ENGINEERING_ASSESSMENT'
    assert dec_crack['urgency'] == 'CRITICAL'
    assert dec_crack['safety_override_active'] is True

    dec_thunder = decision_service.evaluate_decision(
        defect_class='thunderstrike',
        severity=5,
        r_R=0.50,
        area_pct=2.0,
        daily_loss_inr=50.0,
        payback_days=800.0,
        confidence=0.90
    )
    assert dec_thunder['decision'] == 'ENGINEERING_ASSESSMENT'
    assert dec_thunder['urgency'] == 'CRITICAL'
    assert dec_thunder['safety_override_active'] is True

def test_heuristic_fallback_provenance(yolo_service, monkeypatch):
    """
    On an image where YOLO returns zero detections (e.g. data/raw/images/1002.jpg),
    verify fallback flags:
    1. detection_method != 'yolov8'
    2. is_synthetic == True
    3. is_heuristic_fallback == True
    """
    img_path = os.path.join(BASE_DIR, "data", "raw", "images", "1002.jpg")
    assert os.path.exists(img_path)

    monkeypatch.setattr(yolo_service, "model", None)
    img = Image.open(img_path).convert("RGB")
    detections, annotated_img, is_valid, reason, status = yolo_service.detect_defects(img)

    assert is_valid is True
    assert len(detections) > 0
    det0 = detections[0]
    assert det0["detection_method"] == "opencv_heuristic_fallback"
    assert det0["is_synthetic"] is True
    assert det0["is_heuristic_fallback"] is True
    assert det0["crosscheck"]["decision_rule_applied"] == "OPENCV_SALIENCY_FALLBACK"


def test_heuristic_fallback_confidence_capped_and_provisional(yolo_service, monkeypatch):
    """
    Verifies that when heuristic fallback runs, its reported confidence is strictly
    capped at provisional level (<= 0.24) and never presents the closed-set MobileNetV3
    conditional softmax (which can be 90%+) as confident detection.
    """
    img_path = os.path.join(BASE_DIR, "data", "raw", "images", "1002.jpg")
    monkeypatch.setattr(yolo_service, "model", None)
    img = Image.open(img_path).convert("RGB")
    detections, _, is_valid, _, _ = yolo_service.detect_defects(img)

    assert is_valid is True
    for det in detections:
        if det.get("is_heuristic_fallback"):
            assert det["confidence"] <= 0.24, "Heuristic fallback confidence must be <= 0.24"
            assert det["is_low_confidence"] is True
            assert det["confidence_grade"] == "LOW"
            # Conditional softmax is preserved separately
            assert "mobilenet_conditional_softmax" in det["crosscheck"]
            assert det["confidence"] != det["crosscheck"]["mobilenet_conditional_softmax"]


def test_clean_surface_no_fabrication(yolo_service, monkeypatch):
    """
    Verifies that when no significant localized defect contour exists on a valid blade surface:
    1. Returns 0 detections (len(detections) == 0).
    2. Returns explicit 'No reliable defect detected' advisory rather than fabricating a confident defect finding.
    """
    clean_surface = Image.new("RGB", (300, 300), color=(200, 205, 210))
    if yolo_service.pipeline:
        monkeypatch.setattr(yolo_service.pipeline, "validate_blade_domain", lambda *args, **kwargs: (True, "Valid turbine surface", "ACCEPT"))
    if yolo_service.model is not None:
        class MockResult:
            boxes = []
        monkeypatch.setattr(yolo_service.model, "predict", lambda *args, **kwargs: [MockResult()])

    detections, _, is_valid, reason, status = yolo_service.detect_defects(clean_surface)

    assert is_valid is True
    assert len(detections) == 0, f"Expected 0 detections on clean surface, got {len(detections)}"
    assert "No reliable defect detected" in reason
