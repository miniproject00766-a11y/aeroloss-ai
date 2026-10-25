# AeroLoss AI — Staged Preprocessing and Dataset Readiness Plan

**Document Path:** `reports/dataset_audit/recommended_preprocessing_plan.md`  
**Target Architecture:** AeroLoss AI Modular Reconstruction Pipeline (Models 1 through 5)  
**Author:** Senior Deep-Learning Engineer & Computer-Vision Data Engineer  

---

## 1. Component-by-Component Staged Preprocessing Plan

### Stage 1: Model 1 — Image Quality & Out-of-Domain (OOD) Guardrail

#### Current Deficiencies
- The existing OOD evaluation folder (`data/negative_evaluation_samples/`) contains only 30 ad-hoc images.
- Hardcoded heuristics (e.g. Laplacian blur variance $< 100$) risk rejecting valid high-contrast drone photographs or accepting non-blade industrial structures.

#### Actionable Preprocessing Steps
1. **Curate a 3-Tier Quality Calibration Benchmark:**
   - **Usable:** Sharp drone blade inspection photos with clear surface texture and distinct blade edges.
   - **Review-Required:** Drone photos with motion blur ($\text{Var}(\text{Laplacian}) \in [50, 100]$), glare/sun reflections, extreme shadow contrast, or distant framing ($< 15\%$ blade pixel occupancy).
   - **Unusable:** Severely out-of-focus captures, complete whiteout/blackout frames, synthetic noise, non-turbine subjects (animals, buildings, vehicles).
2. **Deterministic Metric Calibration:**
   - *Blur Metric:* Modified Laplacian Variance ($\sigma^2_{\text{Lap}}$) normalized by image diagonal.
   - *Exposure Metric:* Percentage of clipped pixels ($I \le 5$ and $I \ge 250$).
   - *Color Domain Check:* Blade surface chromaticity in HSV space (composite resin/gelcoat typically occupies low-saturation gray-white-beige regions).
   - *Blade Presence Check:* Edge density and Hough transform / line segment detector to verify long linear blade edge geometry before passing to defect detectors.
3. **Guardrail Golden Test Set:**
   - Build a formal evaluation manifest of 200 curated in-domain turbine blade images and 200 diverse out-of-domain distractor images to calibrate false-positive rejection rates ($FPR < 1.0\%$).

---

### Stage 2: Model 2 — Full-Image Defect Detection (YOLOv8)

#### Current Deficiencies
- Exact duplicate pair leakage across splits (`640.jpg` in train, `547.jpg` in val; `565.jpg` in train, `668.jpg` in val).
- Near-duplicate captures of the same blade defect leaking across train and val (`189.jpg` vs `172.jpg`).
- Model downscaled to $512 \times 512$, discarding fine structural features of narrow cracks.
- Severe class imbalance (thunderstrike: 92 boxes vs surface injure: 412 boxes).

#### Actionable Preprocessing Steps
1. **Deduplication and Flight-Grouped Partitioning:**
   - Remove one instance from each exact duplicate pair (`640` and `668`).
   - Cluster near-duplicate images (perceptual dHash Hamming distance $\le 2$) into grouped flight clusters using connected components.
   - Perform **GroupKFold** or **GroupStratifiedSplit** at the flight-cluster level (80% train / 10% val / 10% test), guaranteeing that images of the same defect from slightly different drone angles never appear across different splits.
2. **Native Full-Resolution Conversion:**
   - Convert all 1,065 Pascal VOC XML files into standard YOLO normalized `.txt` bounding boxes:
     $$x_{\text{center}} = \frac{x_{\min} + x_{\max}}{2 \cdot W}, \quad y_{\text{center}} = \frac{y_{\min} + y_{\max}}{2 \cdot H}, \quad w = \frac{x_{\max} - x_{\min}}{W}, \quad h = \frac{y_{\max} - y_{\min}}{H}$$
   - Retain full native $1024 \times 1024$ training resolution (`imgsz=1024`) on the T4 GPU.
3. **Class Balancing Strategy:**
   - Apply targeted copy-paste augmentation for underrepresented classes (`thunderstrike` and `crack`) during training.
   - Use class-weighted focal loss ($\gamma = 1.5$) to compensate for the 4.5:1 object count ratio between `surface_injure` and `thunderstrike`.
4. **Geometrically Safe Augmentation Pipeline:**
   - *Permitted:* Random horizontal/vertical flip, random scale ($0.8 - 1.2$), mild perspective shift ($\pm 5^\circ$), Albumentations CLAHE (contrast enhancement), random shadow / lighting adjustments.
   - *Forbidden:* Aggressive cropping that removes defect context; extreme shear that distorts physical aspect ratio ($L/W$).

---

### Stage 3: Model 3 — Composite Damage Characterization

#### Actionable Preprocessing Steps
1. **Exact Multi-Box Coverage via Sweep-Line 2D Interval Union:**
   - Maintain the sweep-line 2D interval algorithm (`SweepLineUnion`) to prevent overlapping bounding boxes from exceeding 100% surface area.
2. **Spanwise Coordinate ($r/R$) Estimation:**
   - In drone inspection imagery, the turbine root or tip is not always in frame. Implement viewport chord tapering analysis: blade chord width decreases monotonically from max chord ($r/R \approx 0.20$) to tip ($r/R = 1.0$).
   - When absolute root/tip markers are missing, assign spanwise bounds with explicit uncertainty intervals (e.g. $r/R = 0.70 \pm 0.15$).
3. **Aspect Ratio and Severity Grading:**
   - Compute bounding box aspect ratio $AR = \frac{\max(w, h)}{\min(w, h)}$ to differentiate compact pitting/spallation from propagating longitudinal cracks ($AR > 5.0$).

---

### Stage 4: Model 4 — Aerodynamic & Energy Loss Modeling

#### Actionable Preprocessing Steps
1. **Reconcile Turbine Architecture:**
   - Discontinue the practice of evaluating 3.6-MW SCADA data with unscaled 15-MW BEM equations.
   - Formulate turbine scaling parameters:
     - Model A: 3.6-MW baseline (calibrated against `T1.csv` SCADA).
     - Model B: 15-MW reference (calibrated against IEA Wind Task 46 report and OpenFAST).
2. **SCADA Cleaning Pipeline (`T1.csv`):**
   - *Step 1 (Timestamp & Grid Verification):* Verify continuous 10-minute cadence; flag missing gaps ($> 20\text{ min}$).
   - *Step 2 (Operational Clamping):* Clamp negative active power values ($-2.47\text{ kW} \le P < 0$) during zero/sub-cut-in wind ($V < 3.0\text{ m/s}$) to $0.0\text{ kW}$ (standby power).
   - *Step 3 (Curtailment Filtering):* Exclude rows where $V > 10\text{ m/s}$ and $P < 100\text{ kW}$ as non-aerodynamic curtailment/outage events.
   - *Step 4 (IEC 61400-12-1 Binning):* Segment clean operational records into $0.5\text{ m/s}$ wind-speed bins; compute empirical mean power and standard deviation to form the benchmark baseline power curve.
3. **Elimination of Synthetic Regressors:**
   - Replace the synthetic GradientBoosting surrogate with direct BEM lookup or analytical boundary layer displacement models grounded in `FFA_W3_241_polar.csv`.

---

### Stage 5: Model 5 — Financial & Maintenance Decision Support

#### Actionable Preprocessing Steps
1. **Standardized Maintenance Taxonomy:**
   - Base maintenance decisions on verifiable international standards:
     - **IEC 61400-26-1** (Time-based and production-based availability).
     - **DNV-ST-0361** (Blade inspection and damage categorization: Category 1 cosmetic to Category 5 catastrophic).
2. **Engineering Threshold Calibration:**
   - Categorize defects into explicit actionable bins:
     - `MONITOR`: Superficial craze / minor surface injure ($r/R < 0.60$, area $< 1.0\%$, severity $\le 2$).
     - `REPAIR`: Leading-edge erosion / corrosion ($r/R > 0.75$, $C_l/C_d$ degradation $> 15\%$, payback $< 180\text{ days}$).
     - `ENGINEERING ASSESSMENT`: Propagating longitudinal cracks ($AR > 5$), lightning strikes (potential structural spar core delamination).
3. **Deterministic Economic Parameters:**
   - Maintain configurable financial parameters: Electricity tariff ($\text{INR/kWh}$ or $\text{USD/MWh}$), offshore/onshore mobilization cost, weather repair window feasibility.
