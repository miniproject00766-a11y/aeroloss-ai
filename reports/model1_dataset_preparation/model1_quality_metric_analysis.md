# AeroLoss AI — Model 1 Image Quality Metric Analysis and Threshold Calibration Report

**Document Location:** `reports/model1_dataset_preparation/model1_quality_metric_analysis.md`  
**Date:** 2026-10-20  
**Target Component:** Model 1 — Image Quality Validation and Blade-Relevance Guardrail  
**Author:** Senior Deep-Learning Engineer & Computer-Vision Data Engineer  
**Status:** Calibrated Against Empirical Manifest; Provisional Thresholds Documented  

---

## 1. Executive Summary

This report establishes the empirical justification for the provisional thresholds configured in [`configs/model1_quality_thresholds.yaml`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/configs/model1_quality_thresholds.yaml).

Rather than selecting arbitrary thresholds or tuning them to enforce an artificial pass/fail quota, each threshold is derived from empirical statistical distributions computed across all 1,065 primary blade inspection photographs (`data/raw/images/`) and 30 negative evaluation control images (`data/negative_evaluation_samples/`).

> [!IMPORTANT]
> **Provisional Nature of Thresholds:**  
> Because the quality labels for all 1,065 primary blade images remain strictly `UNLABELED` (pending dedicated human review), these thresholds are designated as **provisional baselines**. They are designed to safely catch gross technical failures (severe defocus, complete blackout, whiteout clipping, sub-critical resolution) while routing borderline cases to `REVIEW_REQUIRED`.

---

## 2. Empirical Metric Distributions (Primary Blade Dataset, $N = 1,065$)

Programmatic quantile analysis across the primary blade dataset yields:

| Quality Metric | Minimum | $p_{01}$ (1st %) | $p_{05}$ (5th %) | Median ($p_{50}$) | $p_{95}$ (95th %) | $p_{99}$ (99th %) | Maximum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Laplacian Blur Variance ($\sigma^2_{\text{Lap}}$)** | `0.16` | `0.63` | `1.80` | `21.79` | `269.89` | `635.25` | `2,879.96` |
| **Dark Pixel Fraction ($I \le 15$)** | `0.000` | `0.000` | `0.000` | `0.0001` | `0.0332` | `0.1611` | `0.3195` |
| **Clipped Bright Fraction ($I \ge 245$)**| `0.000` | `0.000` | `0.000` | `0.0007` | `0.7918` | `0.9854` | `0.9978` |
| **Mean Luminance ($\mu_{\text{gray}}$)** | `22.59` | `34.41` | `79.74` | `151.36` | `240.69` | `253.35` | `254.65` |
| **Luminance Std Dev ($\sigma_{\text{gray}}$)** | `3.02` | `5.24` | `11.55` | `38.65` | `70.27` | `86.80` | `111.68` |
| **Shannon Entropy ($H$, bits)** | `0.081` | `0.426` | `2.756` | `5.985` | `7.343` | `7.557` | `7.757` |
| **Canny Edge Density** | `0.000` | `0.000` | `0.0005` | `0.0078` | `0.0524` | `0.1123` | `0.1594` |

---

## 3. Justification for Calibrated Provisional Thresholds

### 3.1 Blur and Defocus Thresholds
- **The "Naive Threshold" Failure:**  
  In general computer vision literature, a rule-of-thumb threshold of $\sigma^2_{\text{Lap}} < 100$ is frequently cited for detecting blur in urban or portrait photography. **Applying $\sigma^2_{\text{Lap}} < 100$ to wind turbine blade inspections would catastrophically reject over 70% of valid blade images.**  
  *Physical Cause:* Industrial wind turbine blades possess vast swaths of smooth, untextured composite gelcoat and clear sky background, naturally yielding low high-frequency energy.
- **Empirically Calibrated Blur Thresholds:**
  - `laplacian_var_unusable: 1.0` &rarr; Rejects only catastrophic optical defocus (e.g. `blurred_low_quality_indoor_01.jpg` with score $0.31$, and the bottom $1\%$ of raw images).
  - `laplacian_var_review: 5.0` &rarr; Flags borderline soft captures ($p_{05} = 1.80$) for human verification without automatic rejection.
  - `laplacian_var_noise_cutoff: 10000.0` &rarr; Pure synthetic random noise exhibits massive artificial variance ($\sigma^2_{\text{Lap}} = 14,742$ to $51,509$); this upper ceiling rejects high-frequency digital noise.

### 3.2 Exposure and Dynamic Range Thresholds
- **Dark Clipping & Underexposure:**
  - In normal blade inspections, median dark clipping ($I \le 15$) is nearly zero ($0.01\%$). Even deep trailing-edge shadows reach only $16.1\%$ at $p_{99}$.
  - `dark_pixel_threshold_unusable: 0.60` &rarr; Rejects frames where $>60\%$ of pixels are clamped to black (e.g. `synthetic_blank_black.jpg` with $100\%$ dark pixels).
  - `dark_pixel_threshold_review: 0.25` &rarr; Flags severe shadow occlusions.
- **Bright Clipping & Overexposure:**
  - Because drone photographs frequently include bright overcast or sunny sky backgrounds, bright clipping can reach up to $79.2\%$ at $p_{95}$ without invalidating the blade section visible in the foreground.
  - `bright_pixel_threshold_unusable: 0.85` and `max_mean_brightness_unusable: 252.0` &rarr; Rejects complete whiteout frames (e.g. `synthetic_blank_white.jpg` with $100\%$ bright pixels).
  - `bright_pixel_threshold_review: 0.35` &rarr; Routes intense solar glint to operator review.
- **Dynamic Range / Contrast:**
  - `min_std_unusable: 2.5` &rarr; Rejects completely flat, solid-color frames (e.g. `synthetic_blank_gray.jpg` with $\sigma_{\text{gray}} = 0.0$).
  - `min_std_review: 10.0` &rarr; Flags low-contrast, washed-out imagery ($p_{05} = 11.55$).

---

## 4. Relevance Baseline: Strengths & Fundamental Limitations

Model 1 implements an explainable, deterministic baseline for blade relevance:
1. **Digital Graphic / Document Rejection:** Digital charts and confusion matrices exhibit large white backgrounds ($I_{\ge 245} > 0.70$) with sharp, low-entropy text ($H < 2.5$). Model 1 reliably filters these without deep learning.
2. **Synthetic Noise & Solid Fill Rejection:** Extreme Laplacian variance ($> 10,000$) or zero variance ($< 2.5$) unequivocally flags synthetic artifacts.
3. **Distant Landscape / Pasture Ambiguity:** When vegetation color dominates ($> 20\%$ green hue) and dominant collinear blade structural edges are absent (e.g. `ambiguous_pasture_cow_distant_turbine_01.jpg`), Model 1 assigns `UNCERTAIN` and routes to `REVIEW_REQUIRED`.

### The Inherent Boundary of Heuristic Relevance
> [!WARNING]
> **Why Heuristics Cannot Replace Deep Learning for Open-Set Relevance:**
> Simple color and edge statistics cannot reliably verify that an image contains a wind turbine blade. For instance, extreme close-up photographs of severe rust/corrosion on blades (e.g. `1.jpg`) exhibit high orange-red chromatic saturation and irregular jagged edges, overlapping with some non-industrial scenes.
> Therefore, Model 1's heuristic guardrail is strictly a **filter for gross non-blade distractors and technical defects**, not an authoritative classifier. A deep-learning feature extractor will be required in later stages once balanced training data is acquired.
