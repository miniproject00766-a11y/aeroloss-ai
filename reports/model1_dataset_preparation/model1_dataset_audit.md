# AeroLoss AI — Model 1 Dataset Audit and Benchmark Preparation Report

**Document Location:** `reports/model1_dataset_preparation/model1_dataset_audit.md`  
**Date:** 2026-10-20  
**Target Component:** Model 1 — Image Quality and Blade-Relevance Guardrail  
**Lead Roles:** Senior Deep-Learning Engineer, Computer-Vision Data Engineer, ML Dataset Quality Specialist  
**Project Root:** `C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai`  
**Status:** Audit Completed; Portable Manifests & Guidelines Compiled  

---

## 1. Executive Summary

This audit establishes the empirical visual baseline for **Model 1 (Image Quality and Blade-Relevance Guardrail)**, the first gatekeeper component of the modular AeroLoss AI platform. Model 1 is tasked with determining whether an incoming photograph is:
1. Technically usable (free of debilitating blur, extreme underexposure, overexposure, or corruption), and
2. Contextually relevant to wind turbine blade inspection (depicting authentic aerodynamic composite surfaces rather than out-of-domain distractor objects or digital artifacts).

### Core Findings
1. **Total Candidate Images Processed:** Exactly **1,095 images** were discovered, decoded, and programmatically analyzed:
   - **Primary Benchmark:** 1,065 high-resolution blade inspection photographs (`data/raw/images/`).
   - **Negative / Out-of-Domain (OOD) Set:** 30 evaluation images (`data/negative_evaluation_samples/`).
2. **Decoding and Format Integrity:** Exactly **1,095 / 1,095 images (100.0%)** decoded successfully using Pillow and OpenCV. Zero corrupted files or unreadable headers were detected.
3. **Exact Duplicates Identified:**
   - 5 distinct exact duplicate groups encompassing 10 files:
     - 2 pairs in the primary blade dataset (`547.jpg` $\equiv$ `640.jpg`; `565.jpg` $\equiv$ `668.jpg`).
     - 3 pairs in the negative evaluation dataset (`animal_bird_01.jpg` $\equiv$ `animal_bird_coucal_02.jpg`; `chart_confusion_matrix_03.png` $\equiv$ `chart_decision_matrix_04.png`; `chart_feature_importance_01.png` $\equiv$ `chart_feature_importance_02.png`).
4. **Perceptual Near-Duplicates Clustered:**
   - Pairwise difference hashing ($dHash$) across all 1,095 images identified **308 near-duplicate pairs** (Hamming distance $\le 2$).
   - Connected-component clustering resolved these into **78 distinct multi-image clusters** containing 263 total files, corresponding to consecutive drone inspection flight captures of the same blade region.
5. **Benchmark Readiness & Deficit Analysis:**
   - **Status:** **`CONDITIONALLY_READY`**.
   - While primary blade imagery is abundant (1,065 images), the repository suffers from a **severe shortage of verified negative evaluation images** (only 27 unique negative images exist, leaving a net deficit of 148 images against the 175-image negative target).
   - All 1,065 primary blade images have verified `BLADE` relevance provenance via Pascal VOC XML annotations, but their technical `quality_label` remains objectively marked as `UNLABELED` pending human calibration.

---

## 2. Source Datasets Inspected

| Source Directory | File Count | Formats | Inferred Dimension A (Relevance) | Quality Label State | Provenance / Annotation Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `data/raw/images/` | `1,065` | JPEG (100%) | `BLADE` (1,065) | `UNLABELED` (1,065) | Backed by 1,065 Pascal VOC XML files in `data/raw/annotations/` (1,584 total defect boxes across 6 classes). |
| `data/negative_evaluation_samples/` | `30` | JPEG (24), PNG (6) | `NON_BLADE` (29), `UNCERTAIN` (1) | `USABLE` (18), `UNUSABLE` (11), `REVIEW_REQUIRED` (1) | Human-reviewed specifically for this audit; covers animals, apparel, food, charts, diagrams, and synthetic noise. |
| `data/processed/patches/` | `1,584` | JPEG (100%) | Excluded from full-frame relevance benchmark | N/A (Defect crops) | Resized $224 \times 224$ crops mapped to raw images via `defect_features.csv`. Retained for defect classifier study, not Model 1 guardrail training. |

---

## 3. Physical Dimensions and Quality Measurement Distributions

Across all 1,095 processed images, objective low-level computer vision metrics were computed:

| Metric | Minimum | 25th Percentile ($Q_1$) | Median ($Q_2$) | Mean | 75th Percentile ($Q_3$) | Maximum |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Width (px)** | `224` | `1,024` | `1,024` | `1,024.1` | `1,024` | `2,400` |
| **Height (px)** | `210` | `1,024` | `1,024` | `1,023.7` | `1,024` | `1,200` |
| **File Size (KB)** | `1.38` | `83.1` | `127.4` | `147.2` | `194.2` | `786.1` |
| **Brightness Mean ($\mu_{\text{gray}}$)** | `0.00` | `127.3` | `148.9` | `146.4` | `168.2` | `255.0` |
| **Brightness Std ($\sigma_{\text{gray}}$)** | `0.00` | `29.4` | `39.8` | `41.2` | `51.7` | `102.3` |
| **Dark Pixel Fraction ($I \le 15$)** | `0.000` | `0.000` | `0.001` | `0.018` | `0.008` | `1.000` |
| **Clipped Bright Fraction ($I \ge 245$)**| `0.000` | `0.000` | `0.008` | `0.038` | `0.046` | `1.000` |
| **Laplacian Blur Variance ($\sigma^2_{\text{Lap}}$)**| `0.00` | `58.2` | `125.8` | `234.6` | `268.4` | `11,438.4` |
| **Shannon Entropy (bits)** | `0.000` | `5.942` | `6.731` | `6.485` | `7.218` | `7.982` |
| **Canny Edge Density** | `0.000` | `0.011` | `0.024` | `0.031` | `0.044` | `0.187` |

### Key Diagnostic Observations:
1. **Resolution Uniformity:** 1,064 out of 1,065 primary blade images are exactly $1024 \times 1024$ pixels (1 image is $788 \times 788$). Negative evaluation samples vary from $224 \times 224$ (synthetic noise) to $2400 \times 1000$ (airfoil polar chart).
2. **Dynamic Range Spread:**
   - 28 primary images exhibit high dark clipping ($> 15\%$ pixels $\le 15$), primarily caused by deep blade chord shadows or dark ground backgrounds.
   - 94 primary images exhibit high bright clipping ($> 15\%$ pixels $\ge 245$), caused by intense solar glint or washed-out sky backdrops.
3. **Blur Variance Distribution:**
   - Median Laplacian variance across blade images is `125.8`.
   - 75 images have blur variance $< 40.0$, representing candidates for optical defocus or motion blur.
   - 642 images have blur variance $\ge 100.0$, representing crisp, high-fidelity blade captures.

---

## 4. Class and Label Distribution (Orthogonal Dimensions)

As documented in `model1_class_distribution.csv`, the dataset reflects strict separation between relevance and quality:

### Dimension A: Blade Relevance
- **`BLADE`**: `1,065 images` (97.26%) — Supported by verified Pascal VOC XML annotations.
- **`NON_BLADE`**: `29 images` (2.65%) — Verified natural photography, civil infrastructure, charts, diagrams, and synthetic noise.
- **`UNCERTAIN`**: `1 image` (0.09%) — `ambiguous_pasture_cow_distant_turbine_01.jpg` (pasture cow foreground with distant horizon turbine).

### Dimension B: Technical Quality
- **`UNLABELED`**: `1,065 images` (97.26%) — Primary raw blade images remain unreviewed to prevent guessed quality ground truth.
- **`USABLE`**: `18 images` (1.64%) — Verified high-fidelity negative real photos and diagrams.
- **`UNUSABLE`**: `11 images` (1.00%) — Pure synthetic noise, solid fills, and severe motion blur (`blurred_low_quality_indoor_01.jpg`).
- **`REVIEW_REQUIRED`**: `1 image` (0.09%) — Borderline framing / low resolution.

---

## 5. Duplicate Groups and Cross-Split Contamination

### 5.1 Exact Duplicates (MD5 Hash)
Programmatic file hashing identified 5 exact duplicate groups across 10 files (detailed in `model1_duplicate_groups.csv`):
- `EXACT_DUP_01`: `data/raw/images/547.jpg` $\leftrightarrow$ `data/raw/images/640.jpg`
- `EXACT_DUP_02`: `data/raw/images/565.jpg` $\leftrightarrow$ `data/raw/images/668.jpg`
- `EXACT_DUP_03`: `data/negative_evaluation_samples/animal_bird_01.jpg` $\leftrightarrow$ `animal_bird_coucal_02.jpg`
- `EXACT_DUP_04`: `data/negative_evaluation_samples/chart_confusion_matrix_03.png` $\leftrightarrow$ `chart_decision_matrix_04.png`
- `EXACT_DUP_05`: `data/negative_evaluation_samples/chart_feature_importance_01.png` $\leftrightarrow$ `chart_feature_importance_02.png`

### 5.2 Perceptual Near-Duplicates (dHash Distance $\le 2$)
- 78 connected near-duplicate clusters (`NEAR_DUP_01` through `NEAR_DUP_78`) containing 263 primary blade images were identified.
- *Root Cause:* Drone automated flight paths take continuous video frames or high-frequency bursts, capturing consecutive viewpoints of the same defect from slightly shifted coordinates (e.g. `172.jpg` vs `189.jpg`).
- *Partitioning Rule:* All members of each near-duplicate cluster must be assigned to the **same split** (all Train, all Val, or all Test) to eliminate data leakage.

---

## 6. Gap Analysis & Missing Categories

To meet the 400-image benchmark target defined in `model1_benchmark_plan.md`:
1. **Real-World Non-Blade Photographs:**
   - Target: 100 images.
   - Available: 13 unique images.
   - **Deficit:** **87 images needed** (animals, vehicles, urban structures, powerlines, solar arrays).
2. **Digital Graphics & Synthetic Distractors:**
   - Target: 75 images.
   - Available: 14 unique images.
   - **Deficit:** **61 images needed** (technical CAD drawings, UI dashboards, weather radar maps, synthetic text).
3. **Blade Quality Stratification:**
   - 225 representative blade inspection images (150 normal + 75 difficult) can be sampled directly from `data/raw/images/` using the computed blur and exposure percentiles.

---

## 7. Deliverables Generated

The following files have been generated, validated, and placed in `reports/model1_dataset_preparation/`:

1. [`model1_dataset_audit.md`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/reports/model1_dataset_preparation/model1_dataset_audit.md) — Comprehensive technical audit report (this document).
2. [`model1_image_manifest.csv`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/reports/model1_dataset_preparation/model1_image_manifest.csv) — Portable per-image manifest containing all 1,095 rows, metrics, and orthogonal labels.
3. [`model1_labeling_guidelines.md`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/reports/model1_dataset_preparation/model1_labeling_guidelines.md) — Rigorous labeling protocols for Relevance (Dimension A) and Quality (Dimension B).
4. [`model1_class_distribution.csv`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/reports/model1_dataset_preparation/model1_class_distribution.csv) — Verified frequency breakdown for both label dimensions and review statuses.
5. [`model1_duplicate_groups.csv`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/reports/model1_dataset_preparation/model1_duplicate_groups.csv) — 5 exact duplicate groups and 78 perceptual near-duplicate clusters with canonical identifiers.
6. [`model1_benchmark_plan.md`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/reports/model1_dataset_preparation/model1_benchmark_plan.md) — 400-image sampling target, gap analysis, and cluster-level train/val/test splitting strategy.
7. [`model1_preparation_summary.json`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/reports/model1_dataset_preparation/model1_preparation_summary.json) — Valid machine-readable JSON summarizing counts, metrics, and next actions.

---

## 8. Prioritized Next Steps

- **Next Task:** Proceed to **Prompt 2: Image-Quality Validation**, where the computed metrics (Laplacian blur variance, brightness histograms, and HSV chromaticity) will be calibrated against human-reviewed sample tiers to establish deterministic guardrail rejection thresholds.
