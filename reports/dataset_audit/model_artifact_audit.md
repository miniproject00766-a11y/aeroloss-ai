# AeroLoss AI — Model Artifact, Checkpoint, and Configuration Audit

**Audit Date:** 2026-10-20  
**Auditor:** Senior Deep-Learning Engineer & Computer-Vision Data Engineer  
**Audit Scope:** `models/`, `runs/`, root checkpoints, training arguments, evaluation logs, and model pipelines  
**Audit Status:** Fully Audited (Read-Only Inspection; No Execution of Untrusted Serialized Weights)

---

## 1. Inventory of Existing Checkpoints and Model Weights

| Artifact Path | Artifact Type | File Size | Exists? | Inspected? | Verification Status | Notes & Verification Findings |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8n.pt` | PyTorch Ultralytics Weights | 6,549,796 B (6.25 MB) | **YES** | **YES** | **VERIFIED** | Baseline COCO pre-trained YOLOv8 nano model used as transfer learning backbone. |
| `models/best.pt` | PyTorch Ultralytics Weights | 6,258,424 B (5.97 MB) | **YES** | **YES** | **VERIFIED** | Primary fine-tuned defect detector on 6-class dataset. Identical copy of `models/detection/yolov8_defect/weights/best.pt`. |
| `models/detection/yolov8_defect/weights/best.pt` | PyTorch Weights | 6,258,424 B (5.97 MB) | **YES** | **YES** | **VERIFIED** | Trained for 15 epochs on `data/yolo_dataset`. Evaluated on 160 test images ($m\text{AP}_{50} = 0.6935$). |
| `models/detection/yolov8_defect/weights/last.pt` | PyTorch Weights | 6,258,424 B (5.97 MB) | **YES** | **YES** | **VERIFIED** | Final epoch (15/15) checkpoint from same training run. |
| `models/blade_defect_classifier.pth` | PyTorch State Dict | 8,769,141 B (8.36 MB) | **YES** | **YES** | **CONDITIONALLY VERIFIED** | MobileNetV3-Small classifier trained on 1,584 patch crops (`data/processed/patches/`). Safe inspection via metadata. |
| `models/blade_defect_classifier_large_scale.pth` | PyTorch State Dict | 8,769,141 B (8.36 MB) | **YES** | **YES** | **CONDITIONALLY VERIFIED** | Re-trained classifier checkpoint evaluated on scaled patch data. Safe state-dict schema matches 6 classes. |
| `models/physics_regressors.joblib` | Serialized Scikit-Learn | 4,281,950 B (4.08 MB) | **YES** | **YES** | **UNVERIFIED AS REAL PHYSICS** | GradientBoosting regressors for `power_loss_kw`, `aep_loss_pct`, `daily_energy_loss_kwh`. Trained exclusively on synthetic data. |
| `models/decision_classifier.joblib` | Serialized Scikit-Learn | 1,234,510 B (1.18 MB) | **YES** | **YES** | **UNVERIFIED AS REAL DECISION** | Decision tree / ensemble trained on synthetic business and maintenance heuristics. |
| `models/domain_classifier/` | Checkpoints & Metrics | Subfolder | **YES** | **YES** | **UNVERIFIED ROBUSTNESS** | Heuristic OOD guardrail based on aspect-ratio, Laplacian blur variance, and color histograms. |

---

## 2. Training Configuration Audit (`args.yaml` and `dataset.yaml`)

### 2.1 Full YOLO Training Arguments (`models/detection/yolov8_defect/args.yaml`)
- **Base Model:** `yolov8n.pt`
- **Data Configuration:** `C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai\data\yolo_dataset\dataset.yaml`
- **Epochs:** `15` (Relatively low; typical convergence for 1,000 images is 50–100 epochs)
- **Batch Size:** `16`
- **Image Size (`imgsz`):** `512` (Downscaled from original $1024 \times 1024$; compromises small crack/craze resolution)
- **Optimizer:** `auto` (resolved to AdamW in log metrics)
- **Hardware Device:** `cpu` (indicates local CPU training run without GPU acceleration)
- **Augmentations Enabled:** `close_mosaic: 10`, `augment: false` (Augmentations were disabled in training arguments!)
- **Class Remap:** `true`

### 2.2 Dataset Partition Schema (`data/yolo_dataset/dataset.yaml`)
- Class Index to Name Mapping:
  - `0: corrosion`
  - `1: crack`
  - `2: craze`
  - `3: hide_craze`
  - `4: surface_injure`
  - `5: thunderstrike`
- Splits: `train: images/train` (745 images), `val: images/val` (160 images), `test: images/test` (160 images).

---

## 3. Metrics Files and Supported Claims

### 3.1 Detector Metrics (`models/model_metrics.json` & `yolo_evaluation_test.json`)

| Metric | Recorded Value | Evaluation Dataset | Verification Status |
| :--- | :--- | :--- | :--- |
| **Precision** | `0.7673` (76.7%) | Independent 160 test images | **VERIFIED** by `yolo_evaluation_test.json` |
| **Recall** | `0.5724` (57.2%) | Independent 160 test images | **VERIFIED** (Notice low recall due to small subtle defects) |
| **$F_1$ Score** | `0.6557` (65.6%) | Harmonic mean of P & R | **VERIFIED** |
| **$m\text{AP}_{50}$** | `0.6935` (69.4%) | Independent 160 test images | **VERIFIED** by test evaluation json |
| **$m\text{AP}_{50-95}$** | `0.4009` (40.1%) | Standard COCO IoU threshold sweep | **VERIFIED** |
| **Per-Class $AP_{50}$:** | | | |
| - `crack` | `0.881` | 38 ground-truth test boxes | **VERIFIED** |
| - `surface_injure`| `0.787` | 63 ground-truth test boxes | **VERIFIED** |
| - `craze` | `0.774` | 37 ground-truth test boxes | **VERIFIED** |
| - `thunderstrike` | `0.668` | 13 ground-truth test boxes | **VERIFIED** |
| - `hide_craze` | `0.621` | 54 ground-truth test boxes | **VERIFIED** |
| - `corrosion` | `0.430` | 36 ground-truth test boxes | **VERIFIED** (Lowest performer due to diffuse boundary texture) |

*Critical Finding on Evaluation Leakage:* As discovered during our cross-dataset audit, twin image `640.jpg` is in `train` while `547.jpg` is in `val`, and near-duplicate `189.jpg` is in `train` while `172.jpg` is in `val`. While test-split direct duplicates were minimal, the test metrics are slightly inflated by near-duplicate viewpoint capture.

### 3.2 Patch Classifier Metrics (`models/vision_metrics.json`)
- **Architecture:** MobileNetV3-Small
- **Test Accuracy:** `88.38%`
- **Weighted $F_1$:** `0.8794`
- **Expected Calibration Error (ECE):** `0.0275` (after temperature scaling $T = 1.4722$)
- **Status:** **VERIFIED** against `data/processed/test_patches_group_strict.csv`.

### 3.3 Physics Surrogate Regressors (`models/physics_metrics.json`)
- **Target 1:** `power_loss_kw` ($R^2 = 0.9793$, $\text{MAE} = 2.31\text{ kW}$)
- **Target 2:** `aep_loss_pct` ($R^2 = 0.9995$, $\text{MAE} = 0.015\%$)
- **Target 3:** `daily_energy_loss_kwh` ($R^2 = 0.9983$, $\text{MAE} = 7.24\text{ kWh}$)
- **CRITICAL AUDIT VERDICT: UNVERIFIED AS PHYSICAL GROUND TRUTH.**
  - Inspection of `src/physics/train_physics_regressor.py` confirms these models were fitted on `data/synthetic/aeroloss_multimodal_dataset.csv`.
  - In `src/data/synthetic_generator.py`, synthetic samples were created by passing randomized uniform distributions into `AeroLossPhysicsEngine()`.
  - The surrogate model merely learned an analytical formula ($R^2 \approx 0.999$ is a model-fitting tautology, not physical empirical validation).

---

## 4. Reusability Assessment

| Model / Artifact | Reusable? | Technical Conditions for Reuse |
| :--- | :--- | :--- |
| `models/best.pt` (YOLOv8) | **CONDITIONALLY REUSABLE** | Can serve as an initial warm-start detector for Model 2 prototyping, but must be retrained at full resolution ($1024 \times 1024$), with mosaic/flip augmentations and deduplicated split boundaries. |
| `blade_defect_classifier.pth` | **CONDITIONALLY REUSABLE** | Accurate patch classifier for 6 classes, but redundant if full-image YOLOv8 operates at $1024 \times 1024$ with sufficient class head capacity. |
| `physics_regressors.joblib` | **NOT REUSABLE IN PRODUCTION** | Must be replaced by direct, verifiable BEM blade calculation or calibrated against empirical SCADA power-curve shifts. |
| `decision_classifier.joblib`| **NOT REUSABLE IN PRODUCTION** | Hardcoded synthetic decision trees should be replaced by a transparent, auditable engineering rules engine (ISO 10816-21 / DNV-ST-0361). |
| `models/aero_lookup_table.csv`| **REUSABLE** | Pre-computed polar degradation table based on FFA-W3-241 airfoil. |
