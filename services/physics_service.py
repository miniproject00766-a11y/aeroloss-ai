import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.physics.aero_engine import AeroLossPhysicsEngine
from src.physics.fatigue_engine import DropletFatigueEngine

class PhysicsService:
    """
    Physics Service wrapping AeroLoss BEM + Airfoil Polar Degradation engine,
    George Springer Droplet Fatigue model, and LUT caching.
    """
    def __init__(self):
        self.engine = AeroLossPhysicsEngine()
        self.fatigue_engine = DropletFatigueEngine()
        self.lut_path = os.path.join(BASE_DIR, "models", "aero_lookup_table.csv")
        self.lut_df = pd.read_csv(self.lut_path) if os.path.exists(self.lut_path) else None

    def compute_aerodynamics(self, severity, r_R, area_pct, wind_speed=7.56, rated_kw=3600.0, capacity_factor=0.36, use_lut=False):
        """
        Calculates baseline power, damaged power, power degradation, AEP loss %,
        Springer droplet fatigue projection, and dynamic Erosion-Safe Mode setpoints.

        Parameters:
            severity (int/float): Defect severity grade (0-5).
            r_R (float): Spanwise location r/R (0.0 root - 1.0 tip).
            area_pct (float): Damage area percentage proxy of blade surface (%).
            wind_speed (float): Inflow hub wind speed (m/s).
            rated_kw (float): Turbine rated generator power in kW.
            capacity_factor (float): Annual turbine capacity factor (default 0.36).
            use_lut (bool): If True and query is in LUT envelope, uses nearest precomputed row.
        """
        # 1. Defensive input validation
        try:
            sev_num = float(severity)
            r_R_num = float(r_R)
            area_num = float(area_pct)
            ws_num = float(wind_speed)
            r_kw = float(rated_kw)
            cf_num = float(capacity_factor if capacity_factor is not None else 0.36)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Physics inputs must be numeric: {e}")

        if r_kw <= 0.0:
            raise ValueError(f"rated_kw must be positive, got {r_kw}")
        if r_R_num < 0.0 or r_R_num > 1.2:
            raise ValueError(f"r_R out of aerodynamic blade range [0.0, 1.2], got {r_R_num}")
        if area_num < 0.0:
            raise ValueError(f"area_pct cannot be negative, got {area_num}")
        if cf_num <= 0.0 or cf_num > 1.0:
            cf_num = 0.36

        # Zero-defect clean blade shortcut
        if sev_num <= 0 or area_num <= 0.0:
            res = self.engine.compute_aerodynamic_loss(
                severity=0,
                r_R=r_R_num,
                area_pct=0.0,
                wind_speed=ws_num,
                rated_power_kw=r_kw,
                capacity_factor=cf_num
            )
            res['baseline_power_kw'] = res['power_baseline_kw']
            res['damaged_power_kw'] = res['power_damaged_kw']
            res['estimated_power_loss_kw'] = 0.0
            res['wind_speed_ms'] = round(ws_num, 2)
            res['capacity_factor'] = round(cf_num, 3)
            res['airfoil_reference'] = 'FFA-W3-241 (Re=1e7 BEM)'
            res['aerodynamic_model_type'] = 'empirical_scaling_proxy'
            res['solver_method'] = 'empirical_polar_lut_interpolation'
            res['scientific_limitations'] = {
                'bem_solver_status': 'Analytical empirical degradation proxy using IEA Task 46 & Sandia scaling; not an iterative dynamic BEM streamtube solver.',
                'optical_scale_status': (
                    'UNVALIDATED PROVISIONAL ASSUMPTION: 2D image-frame coverage is NOT 3D blade surface area. '
                    'The 10% local frame scale proxy is an uncalibrated heuristic assumption with no empirical photogrammetric calibration or drone flight test evidence. '
                    'It is applied strictly as a provisional guardrail to prevent photographic close-ups from claiming gross overestimation of localized defects.'
                ),
                'aerodynamic_polar_basis': 'FFA-W3-241 airfoil polar baseline at Re=1e7 from IEA Task 46 reference definitions.',
                'fatigue_mechanics_basis': 'George Springer (1976) droplet acoustic shock model applicable strictly to liquid droplet incubation; not a Paris-law cyclic bending fatigue solver.'
            }
            res['fatigue_prognostics'] = self.fatigue_engine.forecast_progression(0, r_R_num, wind_speed=ws_num)
            res['erosion_safe_mode'] = self.fatigue_engine.compute_erosion_safe_mode(4.0, r_R_num, 0)
            res['computation_mode'] = 'BEM_exact'
            return res

        # 2. Fast Precomputed LUT lookup if requested
        if use_lut and self.lut_df is not None and not self.lut_df.empty:
            sev_int = int(np.clip(round(sev_num), 1, 5))
            in_envelope = (
                (1.0 <= sev_num <= 5.0) and
                (0.20 <= r_R_num <= 1.02) and
                (3.0 <= ws_num <= 17.0) and
                (0.0 <= area_num <= 15.0)
            )
            if in_envelope:
                cand = self.lut_df[self.lut_df['severity'] == sev_int]
                if not cand.empty:
                    d_r = (cand['r_R'] - r_R_num) / 0.70
                    d_w = (cand['wind_speed'] - ws_num) / 11.0
                    d_a = (cand['area_pct'] - area_num) / 10.0
                    dist_sq = d_r**2 + d_w**2 + d_a**2
                    best_idx = dist_sq.idxmin()
                    row = cand.loc[best_idx]

                    if (abs(row['r_R'] - r_R_num) <= 0.20 and
                        abs(row['wind_speed'] - ws_num) <= 3.0 and
                        abs(row['area_pct'] - area_num) <= 6.0):

                        if ws_num < 3.0 or ws_num > 25.0:
                            base_kw = 0.0
                        elif 3.0 <= ws_num < 11.5:
                            base_kw = r_kw * np.power((ws_num - 3.0) / (11.5 - 3.0), 3.0)
                        else:
                            base_kw = r_kw

                        scale = r_kw / 3600.0
                        p_loss = round(float(row['power_loss_kw']) * scale, 2)
                        p_damaged = round(float(max(0.0, base_kw - p_loss)), 2)
                        daily_baseline_kwh = r_kw * 24.0 * cf_num
                        daily_loss = round(daily_baseline_kwh * (float(row['aep_loss_pct']) / 100.0), 2)

                        fatigue = self.fatigue_engine.forecast_progression(sev_int, r_R_num, wind_speed=ws_num)
                        esm = self.fatigue_engine.compute_erosion_safe_mode(4.0, r_R_num, sev_int)
                        span_weight = round(float(np.power(r_R_num, 4.5 if sev_int < 4 else 3.0)), 4)

                        return {
                            'airfoil_reference': 'FFA-W3-241 (Re=1e7 LUT)',
                            'aerodynamic_model_type': 'empirical_scaling_proxy',
                            'wind_speed_ms': round(ws_num, 2),
                            'capacity_factor': round(cf_num, 3),
                            'power_baseline_kw': round(float(base_kw), 2),
                            'baseline_power_kw': round(float(base_kw), 2),
                            'power_loss_kw': p_loss,
                            'estimated_power_loss_kw': p_loss,
                            'power_damaged_kw': p_damaged,
                            'damaged_power_kw': p_damaged,
                            'aep_loss_pct': round(float(row['aep_loss_pct']), 2),
                            'unrounded_aep_loss_pct': round(float(row['aep_loss_pct']), 6),
                            'daily_energy_loss_kwh': daily_loss,
                            'daily_baseline_energy_kwh': round(daily_baseline_kwh, 2),
                            'annual_energy_loss_kwh': round(daily_loss * 365.0, 2),
                            'spanwise_factor': span_weight,
                            'spanwise_weight_factor': span_weight,
                            'fatigue_prognostics': fatigue,
                            'erosion_safe_mode': esm,
                            'computation_mode': 'LUT_fast'
                        }

        # 3. Exact calculation with SCADA empirical validation
        res = self.engine.compute_aerodynamic_loss(
            severity=sev_num,
            r_R=r_R_num,
            area_pct=area_num,
            wind_speed=ws_num,
            rated_power_kw=r_kw,
            capacity_factor=cf_num
        )
        res['baseline_power_kw'] = res['power_baseline_kw']
        res['damaged_power_kw'] = res['power_damaged_kw']
        res['estimated_power_loss_kw'] = res['power_loss_kw']
        res['wind_speed_ms'] = round(ws_num, 2)
        res['capacity_factor'] = round(cf_num, 3)
        res['airfoil_reference'] = 'FFA-W3-241 (Re=1e7 BEM)'
        res['aerodynamic_model_type'] = 'empirical_scaling_proxy'
        res['solver_method'] = 'empirical_polar_lut_interpolation'
        res['scientific_limitations'] = {
            'bem_solver_status': (
                'Analytical empirical degradation proxy using IEA Wind Task 46 & Sandia spanwise scaling; '
                'not an iterative dynamic blade element momentum (BEM) streamtube solver. '
                'FFA-W3-241 airfoil polar data is stored as a 2D baseline polar reference rather than dynamic runtime streamtube integration.'
            ),
            'optical_scale_status': (
                'UNVALIDATED PROVISIONAL ASSUMPTION: 2D image-frame coverage is NOT 3D blade surface area. '
                'The 10% local frame scale proxy is an uncalibrated heuristic assumption with no empirical photogrammetric calibration or drone flight test evidence. '
                'It is applied strictly as a provisional guardrail to prevent photographic close-ups from claiming gross overestimation of localized defects.'
            ),
            'aerodynamic_polar_basis': 'FFA-W3-241 airfoil polar baseline at Re=1e7 from IEA Task 46 reference definitions.',
            'fatigue_mechanics_basis': 'George Springer (1976) droplet acoustic shock model applicable strictly to liquid droplet incubation; not a Paris-law cyclic bending fatigue solver.'
        }
        res['fatigue_prognostics'] = self.fatigue_engine.forecast_progression(sev_num, r_R_num, wind_speed=ws_num)
        res['erosion_safe_mode'] = self.fatigue_engine.compute_erosion_safe_mode(4.0, r_R_num, sev_num)
        res['computation_mode'] = 'BEM_exact'
        return res
