# AeroLoss AI — Model 1 Labeling Guidelines and Annotation Protocol
**Document Version:** 1.0.0  
**Target Component:** Model 1 — Image Quality and Blade-Relevance Guardrail  
**Author:** Senior Deep-Learning Engineer & Computer-Vision Data Engineer  
**Date:** 2026-10-20  

---

## 1. Objective and Architectural Role of Model 1

In the rebuilt modular **AeroLoss AI** system, **Model 1** operates as the initial gatekeeper (guardrail) for all uploaded visual data before images reach downstream defect detection (Model 2 YOLOv8), damage characterization (Model 3), or aerodynamic loss modeling (Model 4).

Model 1 must evaluate every input along two **strictly orthogonal and independent dimensions**:
1. **Dimension A: Blade Relevance** — Does the photograph depict a wind turbine blade or blade-inspection context suitable for structural/aerodynamic evaluation?
2. **Dimension B: Image Quality** — Does the photograph possess sufficient technical fidelity (focus, exposure, resolution, contrast) to enable reliable downstream computer vision processing?

> [!IMPORTANT]
> **Orthogonality Principle:**  
> Quality and Relevance must never be conflated into a single composite label. A razor-sharp photograph of an eagle is `NON_BLADE` and `USABLE`. Conversely, an authentic drone photograph of a turbine blade taken in heavy fog or with severe motion blur is `BLADE` and `UNUSABLE`.

---

## 2. Dimension A: Blade Relevance Taxonomy

| Label | Definition & Visual Scope | Inclusion Criteria | Exclusion Criteria | Exemplar Cases |
| :--- | :--- | :--- | :--- | :--- |
| **`BLADE`** | Image clearly depicts a wind turbine blade, blade section, leading/trailing edge, root, or aerodynamic surface. | 1. Aerodynamic airfoil profile, spar cap, leading edge, trailing edge, or blade surface texture.<br>2. Drone or ground-based blade inspection viewpoint.<br>3. Wind turbine blade is the primary subject ($>20\%$ pixel occupancy). | 1. Distant wind farm panoramas where individual blades are $< 5\%$ of image width.<br>2. Non-turbine objects or structures without visible blade elements. | `data/raw/images/*.jpg` (1,065 verified inspection photos) |
| **`NON_BLADE`** | Image depicts scenes, objects, or patterns entirely unrelated to wind turbine blade inspection. | 1. Animals, humans, clothing, food, motor vehicles.<br>2. Non-turbine civil or electrical infrastructure (transmission lines, substations, solar panels, buildings).<br>3. Software screenshots, diagrams, charts, graphs, CAD renders.<br>4. Synthetic noise patterns, uniform blank frames. | Any image showing an authentic wind turbine blade as a prominent subject. | `animal_bird_01.jpg`, `food_pizza_01.jpg`, `chart_confusion_matrix_03.png`, `synthetic_blank_black.jpg` |
| **`UNCERTAIN`** | Image is ambiguous, possesses mixed domain cues, or cannot be confidently classified without manual operator inspection. | 1. Distant landscape or pasture where a wind turbine is visible on the far horizon but not framed for blade inspection.<br>2. Extreme micro-closeups showing unrecognizable fiber/gelcoat texture without blade geometry context.<br>3. Ambiguous composite components (e.g. nacelle cowling, spinner dome, tower segment) lacking definitive blade identifiers. | Definitively identifiable blades or definitively unrelated subjects. | `ambiguous_pasture_cow_distant_turbine_01.jpg` |

---

## 3. Dimension B: Image Quality Taxonomy

| Label | Definition & Technical Usability | Technical Criteria | Downstream Implication | Exemplar Cases |
| :--- | :--- | :--- | :--- | :--- |
| **`USABLE`** | High technical fidelity; sharp edge boundaries; well-distributed dynamic range; sufficient resolution for small defect localization. | 1. Laplacian blur variance $\sigma^2_{\text{Lap}} \ge 100$ (indicates sharp high-frequency focus).<br>2. Pixel dimensions $\ge 512 \times 512$.<br>3. Clipped dark pixels ($I \le 15$) $< 10\%$.<br>4. Clipped bright pixels ($I \ge 245$) $< 15\%$.<br>5. Key surface textures, hairline cracks, and surface pitting clearly distinguishable. | Image proceeds directly to Model 2 (YOLOv8) defect detection without intervention. | `0.jpg`, `1.jpg`, `animal_bird_01.jpg`, `diagram_infographic_02.jpg` |
| **`REVIEW_REQUIRED`** | Borderline technical quality; partial degradation that may compromise subtle defect detection but retains macroscopic visibility. | 1. Moderate motion blur or optical defocus ($\sigma^2_{\text{Lap}} \in [40, 100)$).<br>2. Significant sun glare, backlight silhouetting, or deep cast shadows ($15\% - 35\%$ clipped pixels).<br>3. Unfavorable framing (blade occupies only $10\% - 20\%$ of frame).<br>4. Mild compression artifacts or sensor noise. | Pipeline flags image for human supervisory review; user prompted to confirm or re-upload. | `ambiguous_pasture_cow_distant_turbine_01.jpg`, blade images taken against blinding sun glare |
| **`UNUSABLE`** | Severe degradation rendering image technically incapable of reliable defect detection or classification. | 1. Severe optical blur or camera shake ($\sigma^2_{\text{Lap}} < 40$).<br>2. Complete underexposure (blackout, dark fraction $> 50\%$) or severe overexposure (whiteout, bright fraction $> 50\%$).<br>3. Sub-standard resolution ($< 256 \times 256$).<br>4. Corrupted file header or unreadable decoding.<br>5. Pure synthetic noise, solid color fill, or uninformative patterns. | Pipeline immediately rejects upload with specific diagnostic error code; prevents false defect alarms. | `blurred_low_quality_indoor_01.jpg`, `synthetic_blank_black.jpg`, `synthetic_gaussian_noise.jpg` |

---

## 4. Special Borderline Protocols & Disambiguation Rules

### Rule 1: The "Distant Turbine" Dilemma
- **Scenario:** A drone photograph captures a cow grazing in a meadow, with a commercial wind turbine visible 800 meters away on the horizon (`ambiguous_pasture_cow_distant_turbine_01.jpg`).
- **Policy:**
  - *Relevance:* Assign **`UNCERTAIN`** (or `NON_BLADE` under strict inspection mode). Even though a turbine exists in the frame, it is entirely unsuitable for blade surface defect inspection.
  - *Quality:* Assign **`REVIEW_REQUIRED`** or `UNUSABLE` relative to blade inspection.

### Rule 2: Extreme Close-Ups and Cropped Defect Patches
- **Scenario:** A $224 \times 224$ crop depicts a severe composite delamination or leading-edge crack, but the blade's outer aerodynamic contour is completely out of frame.
- **Policy:**
  - If verified by field metadata or companion full-frame context, label **`BLADE`**.
  - If presented in isolation without provenance, label **`UNCERTAIN`** because identical textures occur on civil concrete fractures, fiberglass boat hulls, or aircraft fairings.

### Rule 3: Quality Assessment of Non-Blade Images
- **Scenario:** Assessing an image labeled `NON_BLADE`.
- **Policy:** Quality must evaluate the technical photographic properties of the file, not its relevance.
  - `animal_bird_01.jpg`: `NON_BLADE` + `USABLE` (well-focused, balanced exposure).
  - `blurred_low_quality_indoor_01.jpg`: `NON_BLADE` + `UNUSABLE` (unfocused, severe motion blur).
  - `synthetic_uniform_noise.jpg`: `NON_BLADE` + `UNUSABLE` (zero photographic content).

### Rule 4: Clean Blades vs Defective Blades
- **Scenario:** An inspector uploads a pristine, newly manufactured turbine blade with zero cracks or erosion.
- **Policy:**
  - Relevance is **`BLADE`**. Quality is **`USABLE`**.
  - Reviewers and algorithms must **never reject clean blades as non-blade or unusable**. The downstream YOLOv8 detector should output zero detections, triggering a clean operational report (`ACTION: MONITOR`).

---

## 5. Reviewer Workflow and Provenance Rules

1. **Deterministic Traceability:** Every reviewed label must record `label_source`:
   - `HUMAN_REVIEW`: Assigned after explicit visual inspection by an engineer.
   - `EXISTING_VERIFIED_ANNOTATION`: Inherited from confirmed external ground truth (e.g. primary benchmark XML annotations).
   - `UNLABELED`: Assigned when no human or verified source has reviewed that specific dimension.
2. **Conservative Default:** If a reviewer is in doubt between `USABLE` and `UNUSABLE`, assign `REVIEW_REQUIRED`.
3. **No Guessing:** Unreviewed images in the raw benchmark must remain `quality_label = UNLABELED` until explicitly audited.
