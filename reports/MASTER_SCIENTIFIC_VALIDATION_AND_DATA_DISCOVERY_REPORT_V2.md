# AeroLoss AI — Master Data Discovery, Scientific Validation & Independent Audit Report (Version 2.0 — Final Project Freeze)

**Date:** October 24, 2026  
**Auditor Roles:** Senior Wind-Energy Researcher, SCADA Data Engineer, Aerodynamic Specialist, ML Engineer & Independent Software Auditor  
**Workspace:** `C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai`  
**Active Environment:** Python 3.13.7 | PyTorch 2.6.0+cpu | YOLOv8n (`models/best.pt`) | MobileNetV3-Large (`models/blade_defect_classifier.pth`)  
**Previous Baseline Document:** [`reports/MASTER_SCIENTIFIC_VALIDATION_AND_DATA_DISCOVERY_REPORT.md`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/reports/MASTER_SCIENTIFIC_VALIDATION_AND_DATA_DISCOVERY_REPORT.md) (Preserved without modification)  

---

## 1. Executive Summary

This final audit report (Version 2.0) delivers a rigorous, transparent, and corrected scientific validation of **AeroLoss AI**, an end-to-end multi-modal software prototype designed to detect wind turbine blade defects from inspection imagery, estimate aerodynamic power degradation, and compute economic repair payback.

### Core Audit Verdict & Corrected Scientific Findings
1. **Computer Vision (Claim A):** **VALIDATED strictly within the reported held-out image test set ($N=213$).** The fine-tuned MobileNetV3 classifier achieves **83.1% test accuracy** (Macro-F1: 0.840, Weighted-F1: 0.832) on a test partition grouped strictly by parent image with **zero image-level and zero parent-image leakage** across 745 train, 160 val, and 160 test parent drone photographs. This accuracy applies solely to the benchmark image test set and cannot be extrapolated to unconstrained field inspections.
2. **Defect Severity & Localisation (Claim B):** **PARTIALLY VALIDATED.** YOLOv8 successfully localises defect bounding boxes and extracts spanwise radial positions ($r/R$). However, $r/R$ is inferred from vertical image frame coordinates ($yc/h$), an uncalibrated proxy for true 3D blade spanwise coordinates.
3. **Physical Damage Area Conversion (Claim C):** **NOT VALIDATED (Data Missing).** The inspection imagery completely lacks camera distance, lens focal length, Ground Sampling Distance (GSD), and metric calibration targets. The 10% frame coverage factor is an unvalidated provisional rule of thumb to prevent close-ups from claiming $>10\text{ m}^2$ fractures.
4. **Aerodynamic Degradation & AEP Loss (Claim D):** **PARTIALLY VALIDATED against Published Benchmarks.** The analytical scaling formula exhibits general agreement with published IEA Wind Task 46 & Sareen et al. (2014) reference intervals for Categories 2 through 5. **Correction:** For Category 1 (light pitting), the model predicts an AEP loss of **0.62%**, which lies **outside the published 0.1–0.5% reference interval** by $+0.12$ percentage points. This mismatch is an acknowledged limitation of the empirical proxy, not full concordance. Furthermore, the engine is an **empirical degradation proxy**, not a converged dynamic Blade Element Momentum (BEM) solver.
5. **SCADA Telemetry & Power Performance:** Real operational SCADA telemetry (`T1.csv`, 50,530 rows) confirms an empirical operating deficit of **$-176.22\text{ kW}$ ($-4.89\%$ of rated power)** relative to manufacturer theoretical power. **Crucial Disclaimers:** 
   - **The observed SCADA power deficit cannot be attributed specifically to blade damage** because the inspection images and SCADA data originate from completely separate wind farms and are not linked to the same turbine or event.
   - Real-world operational SCADA deficits reflect the combined confounding effects of wake deficits, yaw misalignment, and atmospheric turbulence.
6. **Financial Impact & Safety Overrides (Claims E & F):** **Claim E is PARTIALLY VALIDATED strictly as mathematically and unit-consistent model estimates**, NOT as independently validated real-world financial losses. Payback crossover periods are modeled hypothetical projections based on assumed tariffs and repair costs, not metered billing guarantees. **Claim F is VALIDATED:** Critical defects (`crack`, `thunderstrike`) trigger mandatory IEC 61400 / DNV-GL safety overrides (`ENGINEERING_ASSESSMENT`, `CRITICAL`), completely insulating structural integrity from economic payback.
7. **System Health:** 28 targeted regression, physics consistency, and guardrail tests pass with 0 failures in 19.33s. Deterministic repeated endpoint runs on `104.jpg` and `860.jpg` produce 0 diffs.

---

## 2. Local Dataset Inventory & Verification

Every local file and archive was directly verified on disk:

| Dataset / File | File Type & Location | Records / Size | Columns & Variables | Sampling & Span | Linked to Turbines / Photos? | Quality & Leakage Findings |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`T1.csv`** | CSV ([`data/raw/T1.csv`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/data/raw/T1.csv)) | 50,530 rows (3.79 MB) | `Date/Time`, `LV ActivePower (kW)`, `Wind Speed (m/s)`, `Theoretical_Power_Curve (KWh)`, `Wind Direction (°)` | 10-minute intervals; Full year 2018 (2018-01-01 to 2018-12-31) | **NO.** Only labeled "T1". No turbine serial, pitch, RPM, or fault log. | 0 missing, 0 duplicates. 77.7% generating, 15.3% below cut-in, 7.0% curtailment/downtime. Cannot be linked to inspection images. |
| **`FFA_W3_241_polar.csv`** | CSV ([`data/raw/FFA_W3_241_polar.csv`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/data/raw/FFA_W3_241_polar.csv)) | 120 rows (10.38 KB) | `Airfoil`, `AoA_deg`, `Cl`, `Cd`, `Cm`, `Re` | Steady-state polar at $Re = 10^7$; $\alpha \in [-180^\circ, +180^\circ]$ | **NO.** Generic IEA 15-MW reference airfoil specification. | Cl: -1.145 to 1.927; Cd: 0.0081 to 1.500. Pure 2D polar; lacks rotor chord, twist, and multi-station schedules for full BEM. |
| **`Dataset.zip`** (Local Archive) | ZIP archive (`C:\Users\akhil\Downloads\Dataset.zip`) | 434.31 MB (3,232 files) | Contains `Dataset/T1.csv`, `WT blade defect dataset/` (1,065 XMLs + images), and nested `fuhrlander-master/` | Varies | **NO.** Pure compilation package. | Discovered in Downloads folder; contains all source raw materials. |
| **`WT blade defect dataset`** | XML/JPEG (`data/raw/annotations`, `data/raw/images`) | 1,065 images, 1,065 XMLs, 1,584 bounding boxes | `folder`, `filename`, `path`, `object/name`, `bndbox [xmin, ymin, xmax, ymax]` | Static optical drone photo patches | **NO.** XML paths show student machine roots (`E:\py--study\blade_new\...`). | No timestamps, no turbine IDs, no GPS coordinates, no camera distance/GSD. Strictly an optical defect benchmark. |
| **`fuhrlander-master.zip`** | ZIP archive (`C:\Users\akhil\Downloads\fuhrlander-master.zip`) | 353.02 MB (6 compressed data files) | 5 turbines (`turbine_80` to `turbine_84`), 78 sensors (312 variables), alarm logbooks | 5-minute intervals (300s); 3 full years (2012–2014); 215,460 rows/turbine | **NO.** European Fuhrländer FL2500 2.5 MW fleet. | High-quality mechanical/electrical SCADA with alarms. Zero blade photos or drone damage records. |
| **`IEA_Wind_Task_46_WP3.1`** | PDF ([`data/raw/IEA_Wind_Task_46_WP3.1_AEP_Loss-1.pdf`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/data/raw/IEA_Wind_Task_46_WP3.1_AEP_Loss-1.pdf)) | 3.09 MB (Academic report) | Leading-edge erosion modeling and AEP loss methodology | Theoretical reference document | Reference only | Primary reference establishing damage categorization (Categories 1–5). |
| **`aeroloss.db`** | SQLite ([`database/aeroloss.db`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/database/aeroloss.db)) | 8 turbine rows, 30+ inspection logs | Full schema with turbine parameters, inspection results, financial loss | Relational store | Synthetic turbine fleet linking simulated drone cases | Seeds demo turbines with $P_{\text{rated}}=3600\text{ kW}$, $CF=0.42/0.36$, repair cost ₹40,000. |

---

## 3. External Datasets Discovery & Suitability Evaluation

A systematic literature and repository search was performed across **Zenodo, NREL, DTU, OpenAIRE, and Kaggle**:

| Candidate Dataset | Source Organisation & DOI / URL | Variables & Sampling Frequency | Turbine Type & Capacity | Download Status | Can It Link Blade Damage to SCADA Power? | License & Usability Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Kelmarsh Wind Farm Data** | Cubico Sustainable Investments via Zenodo (DOI: `10.5281/zenodo.5841834`) | 10-minute SCADA, wind speed, power, pitch, yaw, temperatures, status logs | 6 $\times$ Senvion MM92 (2.05 MW) | Verified accessible via Zenodo API | **NO.** Contains SCADA and event logs, but **no drone damage photographs or defect annotations**. | CC BY 4.0; Open access. Excellent for multi-turbine power baseline validation. |
| **Penmanshiel Wind Farm Data** | Cubico Sustainable Investments via Zenodo (DOI: `10.5281/zenodo.5946808`) | 10-minute SCADA, wind speed, active power, pitch angle, substation meter | 14 $\times$ Senvion MM82 (2.05 MW) | Verified accessible via Zenodo API | **NO.** Event logs track mechanical/grid stoppages, not quantified blade surface damage. | CC BY 4.0; Open access. |
| **CARE to Compare Benchmark** | Zenodo / arXiv (Zenodo Record: 7083049) | Multi-year SCADA time series across multiple wind farms with fault logbooks | Commercial multi-MW utility turbines | Verified accessible | **NO.** Contains gearbox, generator, and bearing failure logs, but **no blade defect imagery**. | Open benchmark for SCADA anomaly detection. |
| **DTU Blade Damage Thermography** | DTU Wind Energy (Mendeley Data / AQUADA) | Thermal images of blade internal delamination and fatigue debonding | Full-scale test blades in DTU blade test facility | Documented in literature | **NO.** Internal non-destructive thermal testing, not operational drone RGB photos linked to SCADA. | Academic research use. |
| **Multiclass Drone Blade Defect Dataset** | Figshare / MDPI (doi:10.3390/app10144933) | High-resolution UAV RGB images annotated with surface cracks, erosion | Drone inspection images of various operating blades | Referenced in literature | **NO.** Vision-only benchmark; **no SCADA power telemetry recorded**. | Academic citation. |

### Decisive Scientific Finding on Data Linkage
> [!IMPORTANT]
> **No public dataset currently exists that bundles simultaneous operational SCADA power logs with drone RGB defect photographs of the same turbine.**
> State-of-the-art wind energy literature confirms that aerodynamic degradation from blade surface defects is universally studied either via numerical simulation (CFD / dynamic BEM coupling wind-tunnel polars to synthetic wind fields), wind-tunnel testing of artificially roughened or 3D-scanned eroded airfoils, or disjoint multi-modal modeling. AeroLoss AI correctly implements disjoint multi-modal modeling and must not falsely claim that local images and SCADA originate from the same turbine.

---

## 4. Scientific Claims Definition & Evaluation Matrix

| Claim | Current Method & Assumptions | Available Supporting Data | Required Ground Truth | Validation Metric | Final Scientific Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Defect Detection & Classification** | YOLOv8n localization + crop-level MobileNetV3 classification with consensus rule | 1,065 drone images, 1,584 annotated patches across 6 canonical classes | Expert manual bounding-box annotations | Accuracy, Per-class Precision / Recall / F1, Macro-F1 | **VALIDATED strictly within held-out benchmark scope ($N=213$)** ($83.1\%$ accuracy, $0.840$ Macro-F1, 0 parent leakage). |
| **B. Defect Severity & Localisation** | Vertical centroid projection: $r/R = 0.20 + 0.78(yc/h)$; class-based severity ceilings (1–5) | Image pixel coordinates in XML annotations | Measured blade coordinate from root flange ($m$) or drone telemetry | Radial distance error ($\Delta r/R$), Root Mean Square Error | **PARTIALLY VALIDATED** (Consistent geometric heuristic, but uncalibrated to true 3D blade coordinates). |
| **C. Physical Blade Damage Area** | Conservative heuristic proxy: $\text{area\_proxy} = \min(5.0, 0.10 \times \text{frame\_cov})$ | 2D bounding-box pixel areas | Laser rangefinder telemetry, camera GSD ($\text{mm/px}$), or physical ruler | Measured metric damage area ($m^2$ or $\text{cm}^2$) | **NOT VALIDATED (Data Missing)**. Unvalidated provisional assumption to prevent close-up distortion. |
| **D. Aerodynamic Degradation & AEP Loss** | Analytical scaling proxy based on IEA Task 46 reference categories & Sandia spanwise weighting | Single 2D polar (`FFA_W3_241_polar.csv`) and IEA Task 46 reference definitions | Multi-station wind tunnel polars or multi-MW turbine CFD simulations | AEP loss percentage concordance against published wind-tunnel trials | **PARTIALLY VALIDATED against Published Benchmarks** (Categories 2–5 concordant; Category 1 is a known overestimation mismatch; not a dynamic BEM solver). |
| **E. Daily Energy Loss & Financial Impact** | Unrounded float arithmetic: $\Delta E = P_{\text{rated}} \times 24 \times CF \times (\Delta \text{AEP}/100)$; Revenue loss $= \Delta E \times \text{tariff}$ | Commercial operational SCADA (`T1.csv`, 50,530 rows) for fleet bounds | Calibrated billing meter revenue loss | Mathematical unit consistency, IEC 61400-12-1 baseline MAE/RMSE | **PARTIALLY VALIDATED strictly as mathematically/unit-consistent model estimates**, NOT as real-world financial losses. |
| **F. Maintenance Recommendation & Safety** | Decision matrix with mandatory IEC 61400 / DNV-GL safety override for cracks & lightning strikes | Domain decision rules in `services/decision_service.py` | Certified wind turbine blade structural integrity standards | Concurrence rate with NDT safety requirements | **VALIDATED** (Safety rules reliably override economic payback on all critical structural defects). |

---

## 5. Completed Validation Experiments & Results

### Experiment 1: Vision Model & Zero-Leakage Audit
- **Dataset Partitioning:** 1,065 raw images partitioned strictly by parent `image_id`:
  - Training: 745 parent images, 1,087 patches
  - Validation: 160 parent images, 284 patches
  - Testing: 160 parent images, 213 patches
  - **Parent-image overlap between all sets: 0 (Zero Data Leakage).**
- **Test Set Performance ($N = 213$):**
  - **Overall Accuracy:** **83.1%** (177 correct predictions out of 213)
  - **Macro-Averaged F1:** **0.840**
  - **Weighted-Averaged F1:** **0.832**
- **Confusion Matrix:**
  $$\begin{bmatrix}
  28 & 0 & 3 & 2 & 8 & 1 \\
  0 & 22 & 1 & 0 & 4 & 0 \\
  0 & 1 & 32 & 1 & 4 & 0 \\
  0 & 1 & 1 & 41 & 3 & 0 \\
  0 & 0 & 2 & 2 & 41 & 0 \\
  0 & 0 & 1 & 1 & 0 & 13
  \end{bmatrix}$$
- **Per-Class Metrics:**
  - `corrosion`: Precision: $1.000$, Recall: $0.667$, F1: $0.800$ (Support: 42)
  - `crack`: Precision: $0.917$, Recall: $0.815$, F1: $0.863$ (Support: 27)
  - `craze`: Precision: $0.800$, Recall: $0.842$, F1: $0.821$ (Support: 38)
  - `hide_craze`: Precision: $0.872$, Recall: $0.891$, F1: $0.882$ (Support: 46)
  - `surface_injure`: Precision: $0.683$, Recall: $0.911$, F1: $0.781$ (Support: 45)
  - `thunderstrike`: Precision: $0.929$, Recall: $0.867$, F1: $0.897$ (Support: 15)

---

### Experiment 2: SCADA Operational Baseline & Attribution Audit (`T1.csv`)
- **Operational Data Partitioning:**
  - Total records: 50,530 rows (10-minute intervals across 2018).
  - Clean generating points ($3.0 \le v \le 25.0\text{ m/s}$, $P > 0$): 39,266 records (77.7%).
  - Cut-in and zero generation: 7,749 points ($v < 3.0$) + 3,514 points (curtailment/downtime).
- **Time-Ordered Split (80% Train: Jan–Oct 2018 / 20% Held-Out Test: Oct–Dec 2018):**
  - Evaluated test generating points: $7,948$ samples.
- **IEC 61400-12-1 Method of Bins ($0.5\text{ m/s}$ bins) Baseline Evaluation:**
  - **MAE:** **$177.43\text{ kW}$** ($4.93\%$ of rated power)
  - **RMSE:** **$338.27\text{ kW}$** ($9.40\%$ of rated power)
  - **Mean Bias:** **$+3.62\text{ kW}$** ($+0.10\%$ of rated power)
- **Manufacturer Theoretical Curve Deficit Analysis:**
  - Evaluating operational test data against `Theoretical_Power_Curve (KWh)`:
  - **Mean Observed Deficit:** **$-176.22\text{ kW}$ ($-4.89\%$ of rated power)**.
- **Scientific Attribution Finding:** **The observed SCADA power deficit cannot be attributed specifically to blade damage.** Because the drone inspection images and SCADA records are not linked to the same turbine or event, the empirical $\sim 4.9\%$ power underperformance in `T1.csv` represents fleet operational losses (wake interference, yaw misalignment, and atmospheric turbulence). Attributing this deficit to blade defects would be a severe scientific error.

---

### Experiment 3: Aerodynamic Scaling Comparison against Published Literature
AeroLoss empirical scaling formulas were compared against published wind-tunnel and simulation intervals from IEA Wind Task 46 (Maniaci et al., 2020) and Sareen et al. (2014) at blade tip ($r/R = 0.92$, $\text{area\_proxy} = 1.0\%$):

| Damage Level / Category | Physical Definition | AeroLoss Model Output | Published IEA Task 46 Reference Interval | Concordance Status & Audit Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Level 1** | Light pitting / pinholes | **0.62% AEP loss** | $0.1\% - 0.5\%$ | **Outside Reference Interval (Mismatch of $+0.12\%$).** The model overestimates minor pitting compared to published literature. |
| **Level 2** | Small patches / coating loss | **1.24% AEP loss** | $0.5\% - 1.5\%$ | **Concordant** (falls within published reference interval). |
| **Level 3** | Large gouges / coating loss | **2.26% AEP loss** | $1.5\% - 2.8\%$ | **Concordant** (falls within published reference interval). |
| **Level 4** | Laminate erosion / delamination | **3.97% AEP loss** | $2.5\% - 4.5\%$ | **Concordant** (falls within published reference interval). |
| **Level 5** | Severe core structural fracture | **6.53% AEP loss** | $4.5\% - 7.0\%+$ | **Concordant** (falls within published reference interval). |

---

### Experiment 4: Physical Damage Area Recovery Audit
- Inspected 1,065 Pascal VOC XML files for physical scale metadata (`scale`, `focal`, `distance`, `gsd`, `meter`, `mm`).
- **Discovered tags:** **0 (`set()`)**.
- **Finding:** Neither real-world physical area ($m^2$) nor metric crack length can be derived from the available evidence without unverified assumptions. The system transparently flags this as an **unvalidated provisional assumption** in both the API provenance metadata and UI disclaimers.

---

### Experiment 5: Financial Chain & Safety Override Verification
- **Financial Calculations as Model Estimates:** Energy and financial losses are calculated from unrounded intermediate values ($\Delta E = P_{\text{rated}} \times 24 \times CF \times (\Delta \text{AEP}/100)$). These metrics represent **mathematically and unit-consistent model estimates**, NOT independently validated real-world financial losses. Payback crossover periods are modeled hypothetical projections based on assumed tariffs (₹4.50/kWh) and repair mobilization costs (₹40,000), not metered billing guarantees.
- **Zero Damage Edge Case:** A defect severity of 0 produces:
  - Daily energy loss: **0.0 kWh/day**
  - Daily financial loss: **₹0.00/day**
  - Payback: **`null`** (`"N/A (No Loss)"`)
- **Safety Override Intactness:** Evaluated on a micro-crack with trivial financial loss (₹10/day, payback > 4000 days):
  - Result: `decision = "ENGINEERING_ASSESSMENT"`, `urgency = "CRITICAL"`, `safety_override_active = True`.
  - Confirms structural safety logic cannot be compromised by low economic losses.

---

## 6. Software Changes & Before/After Comparison

Only minimal, targeted fixes were applied during this cycle:

| Area | Before Modification | After Modification | Justification |
| :--- | :--- | :--- | :--- |
| **[`server.py`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/server.py#L286-L301) Parameter Precedence** | Database row unconditionally overwrote client payload parameters (`repair_cost`, `tariff`, `rated_kw`). | Client request parameters take explicit precedence (`data.get(...) or t_dict[...]`), falling back to DB defaults. | Eliminates silent discarding of user-adjusted sliders on the dashboard. |
| **[`server.py`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/server.py#L484) Upload Persistence** | Hardcoded `0.36` saved into database for all photo uploads regardless of turbine $CF$. | Replaced hardcoded float with active `capacity_factor`. | Guarantees database inspection records match runtime physics inputs. |
| **[`src/physics/aero_engine.py`](file:///C:/Users/akhil/.gemini/antigravity/scratch/aeroloss_ai/src/physics/aero_engine.py#L114) Float Auditability** | Returned only rounded `aep_loss_pct` (2 decimal places). | Exposes continuous unrounded float `'unrounded_aep_loss_pct': round(base, 6)`. | Enables external verification and audits of intermediate float propagation. |
| **Limitations Transparency** | Generic disclaimer strings. | Explicitly labeled as: `"UNVALIDATED PROVISIONAL ASSUMPTION: 2D image-frame coverage is NOT 3D blade surface area..."` | Strictly enforces truth in advertising for academic review. |

---

## 7. Remaining Scientific Limitations & Why They Remain

1. **Absence of Co-Located Image-SCADA Telemetry:** No dataset connects drone damage photos directly to simultaneous SCADA power measurements on the same turbine.
   * *Why it remains:* Industry operators tightly guard proprietary maintenance logs and SCADA time series. Public research benchmarks treat drone inspection and SCADA monitoring as separate disciplines.
2. **Empirical Proxy vs Dynamic BEM Solver:** The aerodynamic model uses analytical scaling calibrated to IEA Task 46 rather than a runtime BEM streamtube solver.
   * *Why it remains:* A true BEM solver requires complete 3D blade aerodynamic schedules (chord distribution, aerodynamic twist, multi-station airfoils FFA-W3-500 through FFA-W3-211) which are not present in the workspace.
3. **Category 1 AEP Loss Overestimation:** The analytical scaling model predicts 0.62% for Category 1 pinholes, exceeding the published 0.1–0.5% reference interval.
   * *Why it remains:* The analytical power-law curve $(0.015/0.070 \times 3.8)$ imposes a continuous scaling floor that is slightly elevated at the lowest damage tier.
4. **Uncalibrated Optical Area Proxy:** The 10% local frame scale factor is an uncalibrated heuristic rule of thumb.
   * *Why it remains:* Drone photographs lack camera focal length, sensor pixel dimensions, and flight distance telemetry.
5. **Attribution of SCADA Deficits:** Fleet operational deficits in `T1.csv` (~4.9%) cannot be isolated to blade damage without blade cleaning/repair intervention logs.

---

## 8. Exact Additional Data Needed for Full Real-World Validation

To advance AeroLoss AI from a defensible academic prototype to a fully validated industrial tool, the following empirical datasets must be acquired:
1. **Calibrated Drone Inspection Metadata:** High-resolution RGB drone imagery with paired telemetry (camera sensor dimensions, focal length, drone-to-blade laser rangefinder distance, or stereoscopic photogrammetric point clouds) to convert 2D bounding boxes into true metric surface area ($m^2$).
2. **Pre- and Post-Repair SCADA Records:** Continuous 10-minute SCADA data for a turbine spanning at least 30 days before and 30 days after a documented blade repair or leading-edge recoating campaign, enabling controlled regression of power curves.
3. **Full 3D Rotor Blade Geometry:** Spanwise chord, aerodynamic twist, and multi-airfoil polar families (FFA-W3-500 to FFA-W3-211) across multiple Reynolds numbers ($10^6$ to $1.5 \times 10^7$) to run a true runtime BEM streamtube solver.
4. **Site-Specific Tariff & O&M Contracts:** Actual power purchase agreement (PPA) tariffs and rope-access contractor mobilization costs for real-world payback validation.

---

## 9. Software Audit & Test Results

```bash
python -m pytest tests/test_physics_financial_consistency.py tests/test_physics_lut_regression.py tests/test_safety_overrides.py tests/test_domain_guardrail.py -v
```

* **Total Tests Executed:** **28**
* **Passed:** **28**
* **Failed:** **0**
* **Skipped:** **0**
* **Execution Time:** **19.33 seconds**
* **Deterministic Repeatability:** Verified with 0 diffs across identical dual runs on `104.jpg` and `860.jpg`.
* **Git Status:** Clean of staged commits; active model weights and legacy checkpoints preserved.

---

## 10. Recommendation for Academic Paper & Demonstration

### **Recommendation: GO for Academic Research Prototype & Methodological Demonstration**

AeroLoss AI is **ready to freeze and present** under the following defensible framing:

* **What is proven:**
  1. Automated drone inspection defect classification (83.1% accuracy, 0.840 Macro-F1) strictly on the held-out image test set ($N=213$) with zero parent-image leakage.
  2. Multi-model consensus cross-checking and robust out-of-domain blade guardrails.
  3. Strict structural safety overrides conforming to IEC 61400 / DNV-GL requirements.
  4. Mathematical and unit consistency across the energy-financial calculation chain.
  5. Close numerical concordance of analytical AEP loss scaling for Categories 2–5 with published IEA Wind Task 46 reference intervals.
* **What is transparently documented:**
  1. The aerodynamic degradation engine is an analytical empirical proxy, not a dynamic BEM solver.
  2. Category 1 AEP loss (0.62%) is an acknowledged model overestimation exceeding the 0.1–0.5% literature range.
  3. Physical damage area is a provisional uncalibrated proxy due to lack of camera GSD telemetry.
  4. Energy-loss and payback calculations are mathematically consistent model estimates, not turbine-metered billing guarantees.
  5. Local inspection images and SCADA records originate from separate benchmark sources and are not co-located; the observed SCADA deficit cannot be attributed to blade damage.
