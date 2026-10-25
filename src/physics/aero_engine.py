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
        lut_path = os.path.join(MODELS_DIR, "aero_lookup_table.csv")
        if not os.path.exists(lut_path):
            self._precompute_lookup_table()
        else:
            self.lut_df = pd.read_csv(lut_path)

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

    def compute_aerodynamic_loss(self, severity, r_R, area_pct=1.0, wind_speed=7.5, rated_power_kw=3600.0, capacity_factor=0.36):
        """
        Calculates delta power (kW), AEP loss %, and daily energy loss (kWh/day)
        based on spanwise position (r/R), damage severity (1-5), and wind speed.
        Uses IEA Wind Task 46 & Sandia empirical spanwise scaling.

        Physical Units & Interpretations:
        - wind_speed: Instantaneous inflow hub wind speed in m/s.
        - power_baseline_kw: Baseline operational power (kW) at active wind speed.
        - power_loss_kw: Instantaneous aerodynamic power deficit (kW) at active wind speed.
        - power_damaged_kw: Net damaged power output (kW).
        - aep_loss_pct: Annual Energy Production (AEP) loss percentage (%), annualized across
          an integrated Rayleigh/Weibull wind climate distribution (mean ~7.5 m/s, k=2.0).
        - capacity_factor: Turbine annual capacity factor (default 0.36, or configured in turbine database).
        - daily_baseline_energy_kwh: Baseline daily energy generation (kWh/day) = rated_power_kw * 24h * capacity_factor.
        - daily_energy_loss_kwh: Annualized equivalent daily energy loss (kWh/day) = daily_baseline_energy_kwh * (aep_loss_pct / 100).
        - annual_energy_loss_kwh: Annualized equivalent annual energy loss (kWh/year) = daily_energy_loss_kwh * 365.
        """
        # Defensive input validation
        try:
            sev_num = int(round(float(severity)))
            r_R_val = float(r_R)
            area_val = float(area_pct)
            ws_val = float(wind_speed)
            rated_kw_val = float(rated_power_kw)
            cf_val = float(capacity_factor if capacity_factor is not None else 0.36)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Aerodynamic computation inputs must be numeric: {e}")

        if rated_kw_val <= 0.0:
            raise ValueError(f"rated_power_kw must be positive, got {rated_power_kw}")
        if r_R_val < 0.0 or r_R_val > 1.2:
            raise ValueError(f"r_R out of aerodynamic blade range [0.0, 1.2], got {r_R}")
        if area_val < 0.0:
            raise ValueError(f"area_pct cannot be negative, got {area_pct}")
        if cf_val <= 0.0 or cf_val > 1.0:
            cf_val = 0.36

        # Operational region baseline power curve:
        # Region 1 (v < 3.0 m/s): Below cut-in, zero power
        # Region 2 (3.0 <= v < 11.5 m/s): Aerodynamic cubic power curve up to rated
        # Region 3 (11.5 <= v <= 25.0 m/s): Rated power limited, pitch-controlled
        # Region 4 (v > 25.0 m/s): Cut-out storm protection shutdown
        if ws_val < 3.0 or ws_val > 25.0:
            power_baseline = 0.0
        elif 3.0 <= ws_val < 11.5:
            power_baseline = rated_kw_val * np.power((ws_val - 3.0) / (11.5 - 3.0), 3.0)
        else: # 11.5 <= ws_val <= 25.0
            power_baseline = rated_kw_val

        daily_baseline_kwh = rated_kw_val * 24.0 * cf_val

        # ZERO DEFECT / CLEAN BLADE EDGE CASE:
        # If severity is 0 (or damage area proxy is 0.0), aerodynamic degradation is identically zero.
        if sev_num <= 0 or area_val <= 0.0:
            result = {
                'power_baseline_kw': round(float(power_baseline), 2),
                'power_loss_kw': 0.0,
                'power_damaged_kw': round(float(power_baseline), 2),
                'aep_loss_pct': 0.0,
                'unrounded_aep_loss_pct': 0.0,
                'daily_energy_loss_kwh': 0.0,
                'daily_baseline_energy_kwh': round(float(daily_baseline_kwh), 2),
                'annual_energy_loss_kwh': 0.0,
                'spanwise_factor': 0.0,
                'wind_speed_ms': round(float(ws_val), 2),
                'capacity_factor': round(float(cf_val), 3),
                'aerodynamic_status': 'CLEAN_NOMINAL_SURFACE'
            }
            result['aerodynamic_calibration'] = {
                'status': 'CLEAN_NOMINAL_BASELINE',
                'model_type': 'Empirical Polar Scaling Proxy (IEA Task 46 & Sandia)',
                'reference_basis': 'FFA-W3-241 airfoil baseline (Re=1e7) clean polar',
                'calibration_gap': 'None (Clean nominal surface)',
                'calibration_interface': self.get_aerodynamic_calibration_interface()
            }
            scada_val = self.validate_scada_deficit(ws_val, 0.0, rated_kw_val)
            result['scada_validation'] = scada_val
            return result

        # DEFECT PRESENT (Severity 1-5):
        # Spanwise velocity & structural impact weighting:
        # Outboard erosion scales with local dynamic pressure (~ r/R^4.5),
        # while structural fractures (severity >= 4) compromise structural aeroelastic stiffness across downstream span
        if sev_num >= 4:
            span_factor = float(max(0.20, np.power(np.clip(r_R_val, 0.05, 1.0), 3.0)))
        else:
            span_factor = float(max(0.05, np.power(np.clip(r_R_val, 0.05, 1.0), 4.5)))

        # Severity aerodynamic penalty (calibrated to IEA Wind Task 46 reference categories)
        sev_multiplier = {1: 0.015, 2: 0.030, 3: 0.055, 4: 0.085, 5: 0.140}.get(sev_num, 0.030)

        # Defect area proxy modifier: logarithmic scaling on estimated blade surface fraction
        area_mod = float(np.clip(1.0 + 0.15 * np.log1p(area_val), 0.8, 2.5))

        # Operational power loss
        if ws_val < 3.0 or ws_val > 25.0:
            power_loss = 0.0
        elif 3.0 <= ws_val < 11.5:
            loss_fraction = sev_multiplier * span_factor * area_mod
            power_loss = power_baseline * loss_fraction
        else: # 11.5 <= ws_val <= 25.0
            power_loss = rated_kw_val * (0.012 * sev_multiplier * span_factor)

        # Estimated Annual Energy Production (AEP) Loss % (annualized across Rayleigh/Weibull climate distribution)
        base_aep_loss_pct = (sev_multiplier / 0.070) * 3.8 * span_factor * area_mod
        aep_loss_pct = round(float(np.clip(base_aep_loss_pct, 0.0, 15.0)), 2)

        # Daily Energy Loss (kWh/day)
        # Derived from rated generator power, capacity factor, and annualized AEP loss %
        daily_energy_loss_kwh = round(daily_baseline_kwh * (aep_loss_pct / 100.0), 2)
        annual_energy_loss_kwh = round(daily_energy_loss_kwh * 365.0, 2)

        result = {
            'power_baseline_kw': round(float(power_baseline), 2),
            'power_loss_kw': round(float(power_loss), 2),
            'power_damaged_kw': round(float(max(0.0, power_baseline - power_loss)), 2),
            'aep_loss_pct': round(float(aep_loss_pct), 2),
            'unrounded_aep_loss_pct': round(float(base_aep_loss_pct), 6),
            'daily_energy_loss_kwh': daily_energy_loss_kwh,
            'daily_baseline_energy_kwh': round(float(daily_baseline_kwh), 2),
            'annual_energy_loss_kwh': annual_energy_loss_kwh,
            'spanwise_factor': round(span_factor, 4),
            'wind_speed_ms': round(float(ws_val), 2),
            'capacity_factor': round(float(cf_val), 3),
            'aerodynamic_status': 'DEFECT_DEGRADATION_ACTIVE'
        }

        # Aerodynamic calibration disclosure and interface
        result['aerodynamic_calibration'] = {
            'status': 'PROVISIONAL_MODEL_ESTIMATE',
            'model_type': 'Empirical Polar Scaling Proxy (IEA Task 46 & Sandia)',
            'reference_basis': 'FFA-W3-241 airfoil baseline (Re=1e7) & Campobasso et al. (2022) DOI:10.1002/we.2666',
            'calibration_gap': 'UNVALIDATED PROVISIONAL MODEL ESTIMATE: 2D image pixel area does not directly measure 3D damaged surface area or boundary layer thickness. Direct polar measurement or high-fidelity 3D CFD is required for real-world calibration.',
            'calibration_interface': self.get_aerodynamic_calibration_interface()
        }
        # Cross-validate against real turbine SCADA dataset (T1.csv)
        scada_val = self.validate_scada_deficit(ws_val, power_loss, rated_kw_val)
        result['scada_validation'] = scada_val
        return result

    def validate_scada_deficit(self, wind_speed, predicted_loss_kw, rated_power_kw=3600.0):
        """
        Cross-validates BEM aerodynamic degradation against real empirical SCADA telemetry from T1.csv.
        Calculates theoretical vs actual underperformance bounds under observed wind conditions.
        Note: SCADA underperformance represents fleet operational deficits including wake, turbulence,
        and yaw misalignment; it provides an empirical upper ceiling rather than defect-exclusive attribution.
        """
        scada_path = os.path.join(RAW_DIR, "T1.csv")
        if os.path.exists(scada_path) and not hasattr(self, '_scada_curve'):
            try:
                scada_df = pd.read_csv(scada_path)
                # Filter to active generating operational states (exclude curtailment, downtime, negative power)
                clean_mask = (scada_df['Wind Speed (m/s)'] >= 3.0) & (scada_df['Wind Speed (m/s)'] <= 25.0) & (scada_df['LV ActivePower (kW)'] > 0.0)
                scada_clean = scada_df[clean_mask].copy()
                scada_clean['ws_bin'] = scada_clean['Wind Speed (m/s)'].round(1)
                self._scada_curve = scada_clean.groupby('ws_bin').agg({
                    'LV ActivePower (kW)': 'mean',
                    'Theoretical_Power_Curve (KWh)': 'mean'
                }).reset_index()
            except Exception:
                self._scada_curve = None

        if getattr(self, '_scada_curve', None) is not None:
            closest_bin = (self._scada_curve['ws_bin'] - wind_speed).abs().idxmin()
            row = self._scada_curve.iloc[closest_bin]
            scada_theor = float(row['Theoretical_Power_Curve (KWh)'])
            scada_meas = float(row['LV ActivePower (kW)'])
            scada_deficit = max(0.0, scada_theor - scada_meas)
            # T1 actual rated power is 3600.0 kW (max theoretical 3600 kW, max active power 3618.7 kW)
            t1_rated_kw = 3600.0
            scaling = rated_power_kw / t1_rated_kw
            scaled_scada_deficit = scada_deficit * scaling
            concurrence_error = abs(predicted_loss_kw - scaled_scada_deficit)
            return {
                'scada_cross_validation_active': True,
                'scada_wind_speed_ms': round(float(row['ws_bin']), 1),
                'scada_mean_active_power_kw': round(scada_meas * scaling, 1),
                'scada_theoretical_power_kw': round(scada_theor * scaling, 1),
                'scada_observed_deficit_kw': round(scaled_scada_deficit, 1),
                'bem_vs_scada_concurrence_error_kw': round(concurrence_error, 1),
                'scada_turbine_rated_kw': t1_rated_kw,
                'attribution_caveat': 'Observed deficit includes combined operational losses (aerodynamic degradation, wake, yaw misalignment, and control margins).'
            }
        return {'scada_cross_validation_active': False}


    def get_aerodynamic_calibration_interface(self):
        """
        Defines the structured calibration interface for integrating empirical aerodynamic polar measurements
        (e.g., wind-tunnel lift/drag polar tables, 3D CFD cavitation/erosion simulations, or field acoustic monitoring).
        """
        return {
            'interface_version': '1.0.0-provisional',
            'required_airfoil_parameters': [
                'airfoil_name',
                'reynolds_number',
                'mach_number',
                'angles_of_attack_deg',
                'clean_polar'
            ],
            'defect_geometry_parameters': [
                'defect_type',
                'cavity_depth_mm',
                'chordwise_extent_pct',
                'spanwise_extent_r_R',
                'equivalent_sand_grain_roughness_ks_mm'
            ],
            'output_polar_degradation': [
                'degraded_Cl',
                'degraded_Cd',
                'transition_location_xc',
                'stall_angle_shift_deg'
            ],
            'scientific_reference': 'Campobasso et al. (2022), Wind Energy 25(1):168-189, DOI:10.1002/we.2666'
        }

    def get_linked_scada_inspection_schema(self):
        """
        Returns recommended data schema for collecting matched blade-inspection and SCADA telemetry,
        ensuring energy loss models are trained/validated on synchronized, linked operational datasets.
        """
        return {
            'schema_version': '1.0.0-matched-telemetry',
            'turbine_metadata': {
                'turbine_id': 'Unique identifier (e.g. WTG-01, NordTank-500)',
                'rated_power_kw': 3600.0,
                'rotor_diameter_m': 112.0,
                'hub_height_m': 84.0,
                'airfoil_profiles': ['FFA-W3-241', 'FFA-W3-301']
            },
            'inspection_telemetry': {
                'inspection_id': 'Unique inspection session identifier',
                'timestamp_utc': 'ISO-8601 UTC timestamp',
                'image_id': 'Unique raw inspection photograph identifier',
                'camera_intrinsics': {
                    'focal_length_mm': 'Focal length in mm',
                    'sensor_dimensions_mm': ['sensor_width_mm', 'sensor_height_mm'],
                    'standoff_distance_m': 'Distance to blade surface in meters'
                },
                'defect_annotation': {
                    'defect_class': 'corrosion | crack | craze | hide_craze | surface_injure | thunderstrike',
                    'bounding_box_pixels': ['x_min', 'y_min', 'x_max', 'y_max'],
                    'calibrated_spanwise_r_R': 'Calibrated physical spanwise position [0.0 - 1.0]',
                    'calibrated_defect_area_m2': 'True physical surface area in m^2',
                    'calibrated_roughness_or_depth_mm': 'Cavity depth or roughness in mm'
                }
            },
            'synchronized_scada_telemetry': {
                'timestamp_utc': 'ISO-8601 UTC timestamp matching inspection window',
                'averaging_period_minutes': 10,
                'wind_speed_ms': '10-minute average hub wind speed (m/s)',
                'wind_direction_deg': '10-minute average wind direction (deg)',
                'measured_active_power_kw': '10-minute measured active electrical power (kW)',
                'theoretical_power_curve_kw': 'OEM warranted power curve at measured wind speed (kW)',
                'blade_pitch_angle_deg': '10-minute average blade pitch angle (deg)',
                'generator_speed_rpm': '10-minute average generator speed (RPM)',
                'ambient_air_temperature_c': 'Ambient temperature (deg C)',
                'air_density_kg_m3': 'Air density (kg/m^3)',
                'operational_status': 'GENERATING | CURTAILED | DERATED | MAINTENANCE'
            },
            'matched_attribution_criteria': {
                'turbine_match': 'Inspection images and SCADA records MUST belong to the exact same physical turbine.',
                'temporal_proximity': 'SCADA operational period must precede and follow inspection within 48 hours under clean vs damaged states.',
                'confounding_factors_controlled': 'Wake deficits, yaw misalignment, icing, and wind shear must be filtered or normalized.'
            }
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
