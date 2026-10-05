import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
POLAR_FILE = os.path.join(RAW_DIR, "FFA_W3_241_polar.csv")

class AeroLossPhysicsEngine:
    """
    Physics-informed aerodynamic degradation and power loss engine.
    Implements IEA Wind Task 46 & Sandia National Labs formulations
    with FFA-W3-241 airfoil polar data.
    """
    def __init__(self, polar_csv=POLAR_FILE):
        self.polar_csv = polar_csv
        self.clean_polar = pd.read_csv(polar_csv) if os.path.exists(polar_csv) else None
        self._precompute_lookup_table()

    def get_degraded_polar(self, severity=3):
        """
        Applies IEA Task 46 Category 1-5 polar degradation to clean FFA-W3-241 polar.
        Category 4 baseline: 12% loss in lift at design, 19% at ClMax, 77% drag increase at design, 141% at ClMax.
        Scaled linearly for categories 1 through 5.
        """
        if self.clean_polar is None:
            return None
        
        df = self.clean_polar.copy()
        # Scale factor relative to Category 4 (cat 4 = 1.0)
        # Cat 1: 0.25, Cat 2: 0.50, Cat 3: 0.75, Cat 4: 1.0, Cat 5: 1.30
        scale = {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00, 5: 1.35}.get(severity, 0.50)

        delta_cl_factor = 1.0 - (0.19 * scale)
        delta_cd_factor = 1.0 + (1.41 * scale)

        # Cl degradation primarily in positive lift region (AoA 0 to 20 deg)
        mask_lift = (df['AoA_deg'] >= 0) & (df['AoA_deg'] <= 25)
        df.loc[mask_lift, 'Cl'] = df.loc[mask_lift, 'Cl'] * delta_cl_factor
        df.loc[mask_lift, 'Cd'] = df.loc[mask_lift, 'Cd'] * delta_cd_factor

        # Post-stall drag increase
        mask_post_stall = (df['AoA_deg'] > 25) & (df['AoA_deg'] <= 90)
        df.loc[mask_post_stall, 'Cd'] = df.loc[mask_post_stall, 'Cd'] * (1.0 + 0.30 * scale)

        df['L_D'] = df['Cl'] / np.maximum(df['Cd'], 1e-4)
        df['severity'] = severity
        return df

    def compute_aerodynamic_loss(self, severity, r_R, area_pct=1.0, wind_speed=7.5, rated_power_kw=3600.0):
        """
        Calculates delta power (kW), AEP loss %, and daily energy loss (kWh/day)
        based on spanwise position (r/R), damage severity (1-5), and wind speed.
        Uses IEA Task 46 spanwise scaling (r/R)^6.7.
        """
        # Spanwise velocity weighting: (r/R)^6.7
        # Local relative kinetic energy and dynamic pressure increase steeply towards tip
        span_factor = float(np.power(np.clip(r_R, 0.1, 1.0), 6.7))

        # Severity aerodynamic penalty (0.01 to 0.08 max delta Cp)
        sev_multiplier = {1: 0.012, 2: 0.025, 3: 0.045, 4: 0.070, 5: 0.095}.get(int(severity), 0.03)

        # Area modifier: defect area % scaled
        area_mod = float(np.clip(1.0 + 0.10 * np.log1p(area_pct), 0.8, 1.8))

        # Operational region power response:
        # Region 2 (3 m/s <= v < 11.5 m/s): Aerodynamic loss directly reduces power ~ v^3
        # Region 3 (v >= 11.5 m/s): Turbine is rated power limited, pitch control mitigates bulk power loss
        if wind_speed < 3.0:
            power_baseline = 0.0
            power_loss = 0.0
        elif 3.0 <= wind_speed < 11.5:
            # Region 2 cubic power curve up to rated
            power_baseline = rated_power_kw * np.power((wind_speed - 3.0) / (11.5 - 3.0), 3.0)
            # Power loss fraction is highest near rated (8 - 11 m/s)
            loss_fraction = sev_multiplier * span_factor * area_mod
            power_loss = power_baseline * loss_fraction
        elif 11.5 <= wind_speed <= 25.0:
            # Region 3 rated power with slight control margin penalty
            power_baseline = rated_power_kw
            power_loss = rated_power_kw * (0.008 * sev_multiplier * span_factor)
        else:
            power_baseline = 0.0
            power_loss = 0.0

        # Estimated AEP Loss % (annualized across Rayleigh/Weibull mean wind 7.5 m/s)
        # Empirical mapping matching IEA Task 46 benchmark table (0.5% - 5.5%)
        base_aep_loss_pct = (sev_multiplier / 0.070) * 4.1 * span_factor * area_mod
        aep_loss_pct = float(np.clip(base_aep_loss_pct, 0.10, 8.50))

        # Daily Energy Loss (kWh/day)
        # Derived from average capacity factor ~0.36 on a 3.6 MW turbine (or user-specified rated)
        # Daily turbine production = rated_power_kw * 24h * CF (~31,100 kWh/day)
        daily_baseline_kwh = rated_power_kw * 24.0 * 0.36
        daily_energy_loss_kwh = round(daily_baseline_kwh * (aep_loss_pct / 100.0), 2)

        return {
            'power_baseline_kw': round(float(power_baseline), 2),
            'power_loss_kw': round(float(power_loss), 2),
            'power_damaged_kw': round(float(max(0.0, power_baseline - power_loss)), 2),
            'aep_loss_pct': round(float(aep_loss_pct), 2),
            'daily_energy_loss_kwh': daily_energy_loss_kwh,
            'spanwise_factor': round(span_factor, 4)
        }

    def _precompute_lookup_table(self):
        """
        Builds the fast Precomputed Aerodynamic Impact Lookup Table (LUT)
        as described in FR-09 of the PRD.
        """
        lut_rows = []
        severities = [1, 2, 3, 4, 5]
        r_R_vals = [0.30, 0.50, 0.70, 0.85, 0.90, 0.95, 0.98]
        wind_speeds = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0]
        area_pcts = [0.5, 2.0, 5.0, 10.0]

        for s in severities:
            for r in r_R_vals:
                for v in wind_speeds:
                    for a in area_pcts:
                        res = self.compute_aerodynamic_loss(s, r, a, v, 3600.0)
                        lut_rows.append({
                            'severity': s,
                            'r_R': r,
                            'wind_speed': v,
                            'area_pct': a,
                            'power_loss_kw': res['power_loss_kw'],
                            'aep_loss_pct': res['aep_loss_pct'],
                            'daily_energy_loss_kwh': res['daily_energy_loss_kwh']
                        })

        self.lut_df = pd.DataFrame(lut_rows)
        lut_path = os.path.join(MODELS_DIR, "aero_lookup_table.csv")
        self.lut_df.to_csv(lut_path, index=False)

if __name__ == "__main__":
    engine = AeroLossPhysicsEngine()
    print("Testing Physics Engine with illustrative PRD Case (Severity=3, r/R=0.90, area=4.5%):")
    res = engine.compute_aerodynamic_loss(severity=3, r_R=0.90, area_pct=4.5, wind_speed=8.5, rated_power_kw=3600.0)
    for k, v in res.items():
        print(f"  {k}: {v}")
