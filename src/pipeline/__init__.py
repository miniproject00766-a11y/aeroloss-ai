"""
AeroLoss AI Pipeline Package.
Provides end-to-end inference and integrated Model 1 + YOLOv8 damage detection.
"""

import sys
import importlib.util
from pathlib import Path

# Re-export integrated detection pipeline
from src.pipeline.aeroloss_detection_pipeline import (
    AeroLossDetectionPipeline,
    run_aeroloss_detection_pipeline
)

# Load and re-export legacy AeroLossPipeline to preserve full backwards compatibility
_legacy_pipeline_path = Path(__file__).resolve().parent.parent / "pipeline.py"
if _legacy_pipeline_path.exists():
    _spec = importlib.util.spec_from_file_location("src._legacy_pipeline_file", str(_legacy_pipeline_path))
    if _spec and _spec.loader:
        _legacy_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_legacy_mod)
        if hasattr(_legacy_mod, "AeroLossPipeline"):
            AeroLossPipeline = _legacy_mod.AeroLossPipeline

__all__ = [
    "AeroLossDetectionPipeline",
    "run_aeroloss_detection_pipeline",
    "AeroLossPipeline"
]
