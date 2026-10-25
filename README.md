# AeroLoss AI 🌪️⚡

> **Edge-Accelerated Computer Vision & Aerodynamic Surrogate Framework for Defect-Informed Wind Turbine Derating and Maintenance Timing**

---

## 📌 Executive Summary

**AeroLoss AI** is a prototype decision-support application for wind-turbine blade image inspection. It combines image classification and bounding-box detection with transparent frame-coverage metrics, a heuristic blade-area proxy, provisional energy and financial estimates, and rule-based maintenance guidance.

Image-derived spanwise position is an uncalibrated estimate, and the capped blade-area proxy is not a physical surface-area measurement. The aerodynamic calculation uses empirical polar degradation and scaling; it is not a complete BEM/CFD solver and has not been field-calibrated. Energy, revenue-loss, fatigue, and payback outputs are therefore provisional estimates—not guaranteed operating outcomes. An engineering assessment recommendation is a conservative image-based escalation, not confirmation of internal structural failure. No latency or model-performance claim is made here.

---

## ⚠️ Capabilities and Limitations

- The vision pipeline recognizes a fixed set of trained defect labels and localizes candidate regions. A missing detection means “no defect detected,” not that the blade has been confirmed healthy.
- Uploaded inspection photos pass a technical-quality preflight. Unusable images are stopped before detection; borderline images may be analyzed only as provisional evidence and require operator review.
- Primary-box and union-box coverage are percentages of the photograph frame. Spanwise position and blade-area values are explicitly provisional image proxies; perspective, framing, and unknown blade boundaries limit their physical interpretation.
- Large abnormal regions and selected structural-risk cases are escalated for qualified engineering/NDT assessment before financial payback is considered. Image classification alone does not diagnose internal damage.
- Energy, financial, and fatigue outputs are exploratory estimates. Validate and calibrate them against site-specific geometry, operating data, inspections, and repair records before operational or investment use.
- Model metrics in the repository are evaluation artifacts; their sample sizes and provenance should be checked before using them as general performance claims.

---

## 🏗️ System Architecture

```
 ┌────────────────────────┐         HTTP REST (JSON)         ┌────────────────────────┐
 │        FRONTEND        │ ◄──────────────────────────────► │        BACKEND         │
 │   (User Interface)     │     e.g., fetch('/api/analyze')  │   (AI & Physics Engine)│
 │   static/index.html    │                                  │   server.py + src/     │
 └────────────────────────┘                                  └───────────┬────────────┘
                                                                         │
                                                             Read/Write  │ SQL / CSV / Models
                                                                         ▼
                                                             ┌────────────────────────┐
                                                             │     STORAGE / MODELS   │
                                                             │   - MobileNetV3 (.pth) │
                                                             │   - GBDT Surrogates    │
                                                             │   - Airfoil Polars     │
                                                             └────────────────────────┘
```

### Component Breakdown
* **`src/vision/train_classifier.py`**: MobileNetV3-Small classifier trained on drone inspection patches (corrosion, crack, craze, hide_craze, surface_injure, thunderstrike).
* **`src/physics/aero_engine.py`**: Provisional aerodynamic estimate using bundled polar data and empirical severity/spanwise scaling; not a validated BEM or site-calibrated model.
* **`src/physics/train_physics_regressor.py`**: GBDT regressor training script predicting power loss, AEP loss, and daily energy loss.
* **`src/financial/financial_engine.py`**: Financial loss calculator (daily ₹/day, annual ₹ Lakhs/yr, and payback period).
* **`src/decision/decision_engine.py`**: Operational decision engine recommending `REPAIR`, `MONITOR`, or `ENGINEERING_ASSESSMENT`.
* **`src/pipeline.py`**: End-to-end inference orchestrator connecting vision, provisional characterization, aerodynamic/financial estimates, and safety-first decisions.
* **`server.py`**: Python HTTP server hosting REST APIs and serving the web dashboard.
* **`static/index.html`**: Interactive web dashboard featuring real-time $r/R$ slider, preset selection, and dual financial metric cards.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
* Python 3.9+
* PyTorch
* Scikit-Learn
* Pandas, NumPy, Pillow

```bash
pip install torch torchvision scikit-learn pandas numpy pillow
```

### 2. Run the Dashboard Server
```bash
python server.py
```
Open your browser and navigate to:
```
http://localhost:8080
```

### 3. Run Pipeline Inference Test
```bash
python src/pipeline.py
```

---

## 📊 Illustrative Response Shape

The following values only illustrate the response schema; they are not a calibrated or field-validated prediction.

```json
{
  "visual_detection": {
    "detected_class": "crack",
    "confidence": 1.0
  },
  "characterization": {
    "severity_proxy": 4,
    "spanwise_position_r_R": 0.89,
    "damage_area_pct": 4.5
  },
  "physics_aerodynamics": {
    "airfoil_reference": "FFA-W3-241 (Re=1e7)",
    "wind_speed_ms": 8.5,
    "baseline_power_kw": 975.29,
    "damaged_power_kw": 938.69,
    "estimated_power_loss_kw": 36.6,
    "spanwise_weight_factor": 0.4581
  },
  "financial_impact": {
    "tariff_inr_kwh": 4.5,
    "repair_cost_inr": 40000.0,
    "daily_financial_loss_inr": 3076.74,
    "annual_financial_loss_inr": 1123010.1,
    "payback_days": 13.0
  },
  "maintenance_decision": {
    "recommended_action": "ENGINEERING_ASSESSMENT",
    "urgency": "CRITICAL"
  }
}
```

---

## 📄 License
MIT License.
