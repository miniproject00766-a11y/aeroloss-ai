# AeroLoss AI — Airfoil Polar Data Quality Report (`FFA_W3_241_polar.csv`)

**Audit Date:** 2026-10-20  
**Auditor:** Senior Deep-Learning Engineer & Computer-Vision Data Engineer  
**File Path:** `data/raw/FFA_W3_241_polar.csv`  
**File Size:** 10,631 bytes (~10.38 KB)  
**Audit Status:** Fully Audited (Read-Only Programmatic Inspection)

---

## 1. Schema, Dimensions, and Missingness

The file `data/raw/FFA_W3_241_polar.csv` contains steady-state 2D aerodynamic section characteristics for the Swedish Aeronautical Research Institute FFA-W3-241 airfoil (24.1% relative thickness), which serves as the primary aerodynamic profile along the mid-to-outer span ($r/R \approx 0.60 - 0.75$) of the IEA 15-MW reference wind turbine blade.

| Metric | Measured Value | Validation Note |
| :--- | :--- | :--- |
| **Row Count** | `120` | Full $360^\circ$ deep-stall polar representation |
| **Column Count** | `6` | Standard aerodynamic coefficient format |
| **File Size** | `10,631 bytes` | Clean tabular CSV |
| **Missing Values** | `0` across all columns | No NaNs, nulls, or empty entries |
| **Non-Finite Values**| `0` | No `inf` or `-inf` values |

### Column Schema and Types

| Column Name | Inferred Type | Description | Minimum | Maximum | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Airfoil` | String (Categorical) | Airfoil cross-section identifier | `FFA-W3-241` | `FFA-W3-241` | Constant (`FFA-W3-241`) |
| `AoA_deg` | Float64 | Angle of attack ($\alpha$) in degrees | `-180.00` | `+180.00` | `0.5250` |
| `Cl` | Float64 | Section Lift Coefficient ($C_l$) | `-1.1448` | `+1.9272` | `0.1790` |
| `Cd` | Float64 | Section Drag Coefficient ($C_d$) | `0.00808` | `1.5000` | `0.5452` |
| `Cm` | Float64 | Section Pitching Moment ($C_m$) | `-0.4814` | `+0.4770` | `-0.0290` |
| `Re` | Float64 / Int64 | Chord-based Reynolds Number | `10,000,000` | `10,000,000` | Constant ($10^7$) |

---

## 2. Angle of Attack ($\alpha$) Range and Monotonicity

- **Angular Coverage:** Extends across the complete $360^\circ$ envelope from $-180.0^\circ$ to $+180.0^\circ$, enabling full deep-stall aeroelastic simulation (Viterna / Montgomerie extrapolation).
- **Monotonicity:** Programmatic inspection confirms `AoA_deg.is_monotonic_increasing == True`.
- **Duplicates:** Exactly `0` duplicate angle entries.
- **Angular Resolution:**
  - Attached flow regime ($-10^\circ \le \alpha \le +20^\circ$): fine step resolution ($\sim 1.0^\circ$ increments).
  - Deep stall and reversed flow regimes ($|\alpha| > 30^\circ$): broader empirical increments ($\sim 2.2^\circ - 4.1^\circ$).

---

## 3. Aerodynamic Coefficient Sanity and Stall Physics

1. **Linear Attached Lift Regime:**
   - Zero-lift angle of attack ($\alpha_{L=0}$): $\approx -2.5^\circ$, consistent with moderately cambered wind turbine airfoils.
   - Linear lift slope ($\frac{dC_l}{d\alpha}$): $\approx 0.108\text{ deg}^{-1}$ ($6.18\text{ rad}^{-1}$), matching thin-airfoil theory ($2\pi \approx 6.28$).
2. **Maximum Lift & Stall Behavior:**
   - $C_{l,\max} = 1.9272$ occurring at $\alpha_{\text{stall}} \approx 11.5^\circ$.
   - Post-stall lift shedding transitions smoothly into deep-stall flat-plate cross-flow drag ($C_{d,\max} = 1.50$ at $\alpha \approx 90^\circ$).
3. **Minimum Drag:**
   - Minimum section drag $C_{d,\min} = 0.008082$ occurs at $\alpha \approx 2.0^\circ$, corresponding to the laminar drag bucket.
   - Maximum aerodynamic glide ratio $\left(\frac{C_l}{C_d}\right)_{\max} \approx 141.2$ at $\alpha \approx 6.0^\circ$.

---

## 4. Sufficiency Assessment for BEM Calculations

> [!IMPORTANT]
> **Data Sufficiency Finding:**  
> The file `FFA_W3_241_polar.csv` is **NOT** sufficient on its own to perform a complete Blade Element Momentum (BEM) turbine calculation.

To calculate blade aerodynamic power ($P$), annual energy production ($AEP$), and localized energy loss ($\Delta P_{\text{aero}}$), the following additional inputs and aerodynamic tables are strictly required:

| Missing Dependency | Description | Current State in Repository |
| :--- | :--- | :--- |
| **Radial Blade Geometry** | Chord distribution $c(r)$, twist distribution $\theta(r)$, and relative thickness $t/c(r)$ from hub ($r = 0$) to tip ($r = R$). | Hardcoded in `src/physics/aero_engine.py` as parametric approximations rather than loaded from verified OpenFAST `.dat` blade geometry files. |
| **Multi-Station Polars** | A typical modern blade utilizes 5–8 distinct airfoil profiles across the span (e.g. FFA-W3-500 at root, FFA-W3-360 at mid-root, FFA-W3-301, FFA-W3-241 at mid-span, FFA-W3-211 at tip). | Only `FFA_W3_241_polar.csv` is provided in `data/raw/`. Other blade stations must be interpolated or scaled. |
| **Reynolds Number Polars** | Operational blade chord Re varies from $3 \times 10^6$ (near root / lower wind) to $1.2 \times 10^7$ (outer span at rated tip speed). | Only $Re = 10^7$ is provided. |
| **Degraded Polars (Roughness / Defect Polars)** | BEM cannot predict defect aerodynamic penalty without comparative degraded polars representing surface roughness (leading-edge erosion, grit, pitting). | Degraded polars are approximated analytically in `src/physics/aero_engine.py` by attenuating $C_l$ and increasing $C_d$ via heuristic severity multipliers. |
| **Aeroelastic Induction Corrections** | Prandtl tip/hub loss factors ($F$) and Glauert empirical correction for high axial induction ($a > 0.4$). | Partially implemented as analytic equations in `aero_engine.py`. |

---

## 5. Verification of External Authority Claims

- The file originates directly from the IEA Wind Task 46 reference study (`IEA_Wind_Task_46_WP3.1_AEP_Loss-1.pdf`), where DTU Wind Energy and NREL established baseline aerodynamic polars for the IEA 15-MW turbine.
- The clean polar data is verified and physically sound for section FFA-W3-241 at $Re = 10^7$.
- However, claims that AeroLoss AI provides a "fully certified standalone BEM solver" are unverified because the codebase lacks multi-station input decks and certified AeroDyn / OpenFAST input verification.
