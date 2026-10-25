# AeroLoss AI — Model 1 Pre-Training Readiness and Dataset Integrity Audit

**Document Location:** `reports/model1_dataset_preparation/model1_pretraining_readiness_audit.md`  
**Audit Date:** 2026-10-20  
**Target Component:** Model 1 — Image Quality and Blade-Relevance Guardrail  
**Lead Roles:** Senior Deep-Learning Engineer, Computer-Vision Data Engineer, ML Dataset Quality Specialist  
**Project Root:** `C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai`  
**Audit Status:** Complete Rigorous Audit; No Original Data or Annotations Modified  

---

## 1. Executive Summary & Audit Checklist

Before initiating Model 1 development, this focused pre-training audit examined the perceptual duplicate clustering mechanism, investigated the origin of the reported 252-image benchmark, and assessed evaluation readiness given the known shortage of negative samples.

### Summary Checklist

| Audit Check Item | Scope & Criteria | Evaluation Result | Status |
| :--- | :--- | :--- | :--- |
| **Check 1: Near-Duplicate Clustering** | Code inspection of dHash & connected components; cluster counts; false-positive identification. | Identified 105 clusters (263 files). Uncovered severe false-positive collapse in `NEAR_DUP_02` due to zero-hash collisions on blank/uniform images. | **NEEDS REVIEW** |
| **Check 2: 252-Image Benchmark Origin** | Trace exact selection procedure; verify manifest rows, human labels, and source counts. | The 252 number was an arithmetic planning target ($225\text{ blade} + 27\text{ unique negatives}$), NOT an existing materialized or human-labeled subset. Manifest actually holds 1,095 images. | **NEEDS REVIEW (Corrected)** |
| **Check 3: Negative Sample Readiness** | Assess sample size, diversity, and statistical power of out-of-domain evaluation set. | Only 27 unique negative samples exist in the repository. Insufficient for training or statistical ROC calibration; adequate only for deterministic sanity checks. | **NEEDS REVIEW** |
| **Check 4: Primary Blade Quality Labels** | Verify all 1,065 primary blade images remain `quality_label = UNLABELED`. | Programmatically verified: exactly 1,065 / 1,065 blade images are `UNLABELED`. Zero fabricated quality ground truth. | **PASS** |
| **Check 5: Exact Duplicate Isolation** | Confirm detection of 2 primary blade duplicate pairs and 3 negative duplicate pairs. | Exactly 5 groups (10 files) identified; original files fully preserved without deletion or moving. | **PASS** |
| **Check 6: Path Portability & Execution** | Ensure script runs cleanly with exit code 0 using relative paths. | `curate_model1_dataset.py` refactored to use dynamic relative paths; executed cleanly with exit code 0. | **PASS** |

### Overall Recommendation: **PROCEED WITH A LIMITED HEURISTIC PILOT (DO NOT TRAIN A DL CLASSIFIER YET)**
- **Do NOT train a deep-learning classifier** (e.g. CNN/ViT) on the current dataset. With only 13 natural non-blade photographs, any learned model will severely overfit and fail to generalize to real-world out-of-domain images.
- **Proceed with a deterministic, rule-based pilot** (Prompt 2) to calibrate Laplacian blur variance, histogram exposure clipping, and aspect ratio thresholds on a curated sample of blade images and the 27 available negative controls.

---

## 2. In-Depth Audit of Near-Duplicate Clustering

### 2.1 Inspection of Clustering Algorithm in `curate_model1_dataset.py`
The curation script employs a two-stage near-duplicate discovery pipeline:
1. **Perceptual Difference Hashing (dHash):** Each image is resized to $9 \times 8$ grayscale, and adjacent horizontal pixels are compared:
   $$\text{diff}_{i, j} = (I_{i, j+1} > I_{i, j})$$
   producing a 64-bit binary integer.
2. **Pairwise Hamming Distance Matrix:** Pairwise distance $D_H(A, B)$ is computed via bitwise XOR. Pairs with $D_H \le 2$ bits are flagged as candidate near-duplicates.
3. **Connected-Component Graph Clustering:** A disjoint-set (Union-Find) algorithm links any two images sharing $D_H \le 2$ into an equivalence class.

### 2.2 Why the Duplicate CSV Contains Exactly 105 Clusters
The clustering generated **105 perceptual near-duplicate groups** (`NEAR_DUP_01` through `NEAR_DUP_105`), encompassing **263 total files**:

| Cluster Size | Number of Clusters | Total Images Involved | Typical Composition |
| :---: | :---: | :---: | :--- |
| **Size 2** | `84` | `168` | Direct pairwise drone burst frames (e.g. `1007.jpg` and `1008.jpg`, $D_H = 2$). |
| **Size 3** | `9` | `27` | Short consecutive drone inspection sequences (e.g. `135.jpg`, `175.jpg`, `180.jpg`). |
| **Size 4** | `7` | `28` | Multi-angle defect inspections (e.g. `100.jpg`, `191.jpg`, `744.jpg`, `794.jpg`). |
| **Size 5** | `4` | `20` | Extended stationary hovering sequences along a single blade chord. |
| **Size 20** | `1` | `20` | **MEGA-CLUSTER (`NEAR_DUP_02`) — Confirmed False-Positive Collapse.** |
| **Total** | **`105`** | **`263`** | — |

### 2.3 False-Positive Breakdown: The Collapse of `NEAR_DUP_02`
Inspection of `NEAR_DUP_02` revealed a severe algorithmic failure mode:

```text
NEAR_DUP_02 Members (20 files):
- Negative Samples (4): synthetic_blank_black.jpg, synthetic_blank_gray.jpg, synthetic_blank_white.jpg, synthetic_neon_banner.jpg
- Primary Blade Images (16): 1003.jpg, 13.jpg, 281.jpg, 285.jpg, 472.jpg, 651.jpg, 666.jpg, 711.jpg, 788.jpg, 802.jpg, 83.jpg, 839.jpg, 873.jpg, 879.jpg, 901.jpg, 938.jpg
```

#### Why Did Blade Images Cluster with Blank Synthetic Files?
1. **Zero-Hash Collision on Low-Texture Surfaces:**  
   In solid blank files (`synthetic_blank_black.jpg`, `synthetic_blank_gray.jpg`, `synthetic_blank_white.jpg`), all adjacent pixel differences are strictly zero. The resulting 64-bit dHash is identically:
   $$\text{dHash} = \texttt{0x0000000000000000} \quad (0 \text{ bits set})$$
2. **Smooth Blade Surface Hashes:**  
   Blade inspection images with large areas of smooth, untextured white gelcoat or uniform overcast sky also produce dHashes with very few bits set (e.g. `281.jpg` has $\text{dHash} = \texttt{0x0}$, `1003.jpg` has 1 bit set, `13.jpg` has 2 bits set).
3. **Transitive Chaining Error:**  
   Because $D_H(\texttt{0x0}, \texttt{0x0}) = 0 \le 2$, Union-Find linked the blank images with `281.jpg` and `83.jpg`. Then, because `1003.jpg` was distance 1 from `0x0`, and `13.jpg` was distance 2, the graph connected component merged all 16 blade images and 4 synthetic images into one single 20-image cluster!

### 2.4 Recommendations for Duplicate Clustering
1. **Domain Isolation:** Primary blade images (`data/raw/images/`) and negative samples (`data/negative_evaluation_samples/`) must **never be clustered together**. Perceptual hashing should only be executed within homogeneous domain partitions.
2. **Entropy Pre-Filter for Hashing:** Images with Shannon entropy $H < 1.0\text{ bits}$ or Laplacian variance $\sigma^2_{\text{Lap}} < 5.0$ must be flagged as untextured/blank and excluded from dHash graph union to prevent transitive mega-clusters.
3. **Split Policy for Genuine Clusters:** For the remaining 104 genuine blade clusters (such as `NEAR_DUP_01`), the grouping rule stands: **all images within a cluster must be assigned to the same split (all Train, all Val, or all Test)** to prevent flight-sequence leakage.

---

## 3. Deconstruction of the Reported "252-Image Benchmark"

### 3.1 Trace of the 252-Image Figure
Section 2 of `model1_benchmark_plan.md` presented a table showing:
$$\text{"Available in Repository: 225 + 27 = 252"}$$
This number was derived as follows:
- **Planned In-Domain Blade Strata:** 150 standard sharp blades + 75 degraded/difficult blades = **225 planned blade samples**.
- **Available Unique Negative Samples:** 30 negative files minus 3 exact duplicate files = **27 unique negative samples**.
- **Sum:** $225 + 27 = \mathbf{252}$.

### 3.2 What Actually Exists in the Master Manifest (`model1_image_manifest.csv`)
A strict programmatic audit of `model1_image_manifest.csv` confirms:
1. **The manifest contains all 1,095 images**, NOT 252 images.
2. **There is NO materialized 225-blade subset in the repository.** The 225 blade count was purely a theoretical sampling quota, not an existing filtered file collection.
3. **Human Review Coverage:** Exactly **30 images** have verified human review records (`label_source = HUMAN_REVIEW`). All 30 are from `data/negative_evaluation_samples/`.
4. **Primary Blade Annotations:** Exactly **1,065 images** have `label_source = EXISTING_VERIFIED_ANNOTATION` for `relevance_label = BLADE` (derived from Pascal VOC XMLs). Exactly **0 primary blade images have verified human quality labels**.

### 3.3 Authoritative Status of the 1,095 Candidate Images

| Source Dataset | Total Files | Unique Files | Relevance Label | Quality Label | Label Source | Human-Verified Status |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| **`primary_benchmark_images`** | `1,065` | `1,063` | `BLADE` (1,065) | `UNLABELED` (1,065) | `EXISTING_VERIFIED_ANNOTATION` | **UNREVIEWED** for quality; relevance backed by XML defect bounding boxes. |
| **`negative_evaluation_samples` (Real)** | `15` | `13` | `NON_BLADE` (14), `UNCERTAIN` (1) | `USABLE` (12), `UNUSABLE` (1), `REVIEW_REQUIRED` (1) | `HUMAN_REVIEW` | **HUMAN-REVIEWED** |
| **`negative_evaluation_samples` (Graphics/Noise)** | `15` | `14` | `NON_BLADE` (15) | `USABLE` (6), `UNUSABLE` (9) | `HUMAN_REVIEW` | **HUMAN-REVIEWED** |
| **Total** | **`1,095`** | **`1,090`** | — | — | — | **30 Reviewed / 1,065 Unreviewed** |

---

## 4. Evaluation Readiness & Negative Sample Deficit

### 4.1 Quantitative Shortage of Out-of-Domain Data
The repository contains only **27 unique negative samples** (13 natural/industrial scenes, 14 digital graphics/synthetic noise). Against the industry standard planning target of 175–200 negative evaluation images:
- **Net Deficit:** **148 images missing**.
- **Category Coverage:** Entire real-world domains (e.g. open ocean water, drone rotors in frame, cloudy skies without blades, wind farm roads, service technician harness gear) are completely unrepresented.

### 4.2 What the Current Data CAN Support
1. **Deterministic Quality Guardrail Calibration (Prompt 2):**  
   The primary dataset possesses rich diversity in focus, lighting, and contrast (Laplacian blur variance ranges from $0.0$ to $51,509$, and bright pixel clipping ranges from $0.0\%$ to $99.8\%$). This is fully sufficient to calibrate rule-based thresholds for:
   - Severe blur / defocus rejection ($\sigma^2_{\text{Lap}} < 40$).
   - Extreme overexposure / underexposure rejection ($I_{\text{clipped}} > 35\%$).
   - Aspect ratio anomaly flagging ($AR < 0.5$ or $AR > 2.5$).
2. **Diagnostic Pipeline Smoke Testing:**  
   The 27 negative samples serve as an effective regression suite to verify that Model 1 immediately rejects obvious digital noise, solid black frames, and pure graphics.

### 4.3 What the Current Data CANNOT Support
1. **Deep-Learning Binary Classifier Training:**  
   Training a neural network (e.g. MobileNet/ResNet) to classify `BLADE` vs `NON_BLADE` on 1,065 positive images and only 13 natural negative images will result in catastrophic failure on out-of-distribution inputs due to sample scarcity.
2. **Statistical ROC / AUC Metrics:**  
   With only 13 natural negative samples, a single classification mistake alters the False Positive Rate ($FPR$) by $7.7\%$, rendering any claimed specificity or $F_1$ score statistically uninformative.
3. **Automated Quality Filtering Ground Truth:**  
   Because all 1,065 blade images are `quality_label = UNLABELED`, automated quality models cannot be trained via supervised learning until human ground truth is annotated.

---

## 5. Strategic Recommendations Before Prompt 2

1. **Adopt a Two-Tier Architecture for Model 1:**
   - **Tier 1 (Deterministic Physics/Image-Processing Guardrail):** Implement Laplacian variance blur detection, histogram dynamic range checks, and HSV chromaticity filters. Calibrate these thresholds using human-verified boundary examples in Prompt 2.
   - **Tier 2 (Learned Feature Embeddings):** Defer training any deep-learning classifier until a balanced out-of-domain dataset (at least 150 diverse non-blade photographs) is formally curated.
2. **Fix `NEAR_DUP_02` in Duplicate Manifest:**
   - Disconnect the 4 synthetic blank files from the 16 genuine blade images in `model1_duplicate_groups.csv` by restricting dHash grouping to intra-dataset pairs.
3. **Proceed to Prompt 2 with Full Transparency:**
   - Acknowledge that the benchmark evaluation is a **pilot evaluation** focused on image-quality boundary calibration, not an end-to-end commercial OOD guarantee.
