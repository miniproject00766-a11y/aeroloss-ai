import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.physics.aero_engine import AeroLossPhysicsEngine

class PhysicsService:
    """
    Physics Service wrapping AeroLoss BEM + Airfoil Polar Degradation engine
    and LUT caching for fast repeated inference.
    """
    def __init__(self):
        self.engine = AeroLossPhysicsEngine()
        self.lut_path = os.path.join(BASE_DIR, "models", "aero_lookup_table.csv")
        self.lut_df = pd.read_csv(self.lut_path) if os.path.exists(self.lut_path) else None

    def compute_aerodynamics(self, severity, r_R, area_pct, wind_speed=7.56, rated_kw=3600.0, use_lut=False):
        """
        Calculates baseline power, damaged power, power degradation, and AEP loss %.
        """
        if use_lut and self.lut_df is not None:
            try:
                # Find closest match in LUT
                idx = ((self.lut_df['severity'] == int(severity)) &
                       (np.abs(self.lut_df['r_R'] - r_R) < 0.15) &
                       (np.abs(self.lut_df['wind_speed'] - wind_speed) < 2.0)).idxmin()
                row = self.lut_df.iloc[idx]
                return {
                    'airfoil_reference': 'FFA-W3-241 (Re=1e7 LUT)',
                    'wind_speed_ms': round(float(wind_speed), 2),
                    'baseline_power_kw': round(float(rated_kw * 0.40), 2),
                    'damaged_power_kw': round(float(rated_kw * 0.40 - row['power_loss_kw']), 2),
                    'estimated_power_loss_kw': round(float(row['power_loss_kw']), 2),
                    'aep_loss_pct': round(float(row['aep_loss_pct']), 2),
                    'daily_energy_loss_kwh': round(float(row['daily_energy_loss_kwh']), 2),
                    'spanwise_weight_factor': round(float(np.power(r_R, 6.7)), 4),
                    'computation_mode': 'LUT_fast'
                }
            except Exception:
                pass

        # Full BEM calculation
        res = self.engine.compute_aerodynamic_loss(
            severity=severity,
            r_R=r_R,
            area_pct=area_pct,
            wind_speed=wind_speed,
            rated_power_kw=rated_kw
        )
        res['baseline_power_kw'] = res['power_baseline_kw']
        res['damaged_power_kw'] = res['power_damaged_kw']
        res['estimated_power_loss_kw'] = res['power_loss_kw']
        res['airfoil_reference'] = 'FFA-W3-241 (Re=1e7 BEM)'
        res['computation_mode'] = 'BEM_exact'
        return res

