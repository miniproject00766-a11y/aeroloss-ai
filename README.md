# AeroLoss AI 🌪️⚡

> **Edge-Accelerated Computer Vision & Aerodynamic Surrogate Framework for Defect-Informed Wind Turbine Derating and Maintenance Timing**

---

## 📌 Executive Summary

**AeroLoss AI** bridges the critical gap between drone-based blade visual inspection and physical aerodynamic revenue loss. 

While existing drone inspection tools stop at drawing 2D bounding boxes (e.g. *"Category 3 erosion detected"*), and traditional CFD / BEM solvers take hours to compute energy losses, **AeroLoss AI** directly predicts aerodynamic polar degradation ($\Delta C_l, \Delta C_d$), turbine annual energy production (AEP) loss, daily revenue loss (₹/day), and optimal repair payback timelines in **under 50 milliseconds** on a standard CPU.

---

## 🚀 Key Innovations & Novelties

1. **Photo-to-Physics Polar Inference:**  
   Direct optical mapping from 2D surface erosion photographs to FFA-W3-241 airfoil aerodynamic polars at high Reynolds numbers ($Re \approx 10^7$) without 3D geometric meshing or CFD.
2. **Defect-Informed Erosion-Safe Mode (ESM):**  
   Dynamic turbine supervisory derating setpoints ($\Omega_{\text{derated}}$) during precipitation based on optically detected defect spanwise position ($r/R$), eliminating unnecessary power curtailment for inboard defects.
3. **Droplet-Fatigue "Cost of Inaction" Progression:**  
   Physics-backed repair countdown combining Springer's liquid impingement fatigue model ($N_c \propto V_{\text{rel}}^{-2}$) with regional rainfall data to forecast damage escalation (e.g. ₹40,000 preventive tape repair vs. ₹3,50,000 structural repair).
4. **Sub-50ms Edge Physics Surrogate:**  
   Multi-target Gradient-Boosted Decision Tree (GBDT) surrogate ($R^2 > 0.99$) executing on standard CPUs in $<50\text{ ms}$, enabling real-time browser slider interaction.
5. **Calibrated Spanwise Sensitivity Scaling Law:**  
   Empirical-theoretical formulation demonstrating that damage at the blade tip ($r/R = 0.95$) is **over $20\times$ more financially damaging** than the identical defect at mid-span ($r/R = 0.50$):
   $$\Delta P(r/R) \propto \left(\frac{r}{R}\right)^{6.7}$$

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
* **`src/physics/aero_engine.py`**: Aerodynamic engine implementing IEA Wind Task 46 & Sandia National Labs polar degradation on FFA-W3-241 airfoil ($Re = 10^7$).
* **`src/physics/train_physics_regressor.py`**: GBDT regressor training script predicting power loss, AEP loss, and daily energy loss.
* **`src/financial/financial_engine.py`**: Financial loss calculator (daily ₹/day, annual ₹ Lakhs/yr, and payback period).
* **`src/decision/decision_engine.py`**: Operational decision engine recommending `REPAIR`, `MONITOR`, or `ENGINEERING_ASSESSMENT`.
* **`src/pipeline.py`**: End-to-end inference orchestrator connecting Vision $\to$ Physics $\to$ Financials $\to$ Decision.
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

## 📊 Sample Inference Output

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
