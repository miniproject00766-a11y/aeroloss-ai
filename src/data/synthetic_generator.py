import os
import sys
import pandas as pd
import numpy as np

# Ensure parent directory is in path
sys.path.insert(0, r"C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai")

from src.physics.aero_engine import AeroLossPhysicsEngine
from src.financial.financial_engine import FinancialEngine
from src.decision.decision_engine import DecisionEngine

PROCESSED_DIR = r"C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai\data\processed"
SYNTHETIC_DIR = r"C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai\data\synthetic"
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

def generate_multimodal_dataset(target_samples_per_decision=1200, random_seed=42):
    np.random.seed(random_seed)
    
    real_csv = os.path.join(PROCESSED_DIR, "defect_features.csv")
    real_df = pd.read_csv(real_csv) if os.path.exists(real_csv) else None
    
    physics_engine = AeroLossPhysicsEngine()
    financial_engine = FinancialEngine()
    decision_engine = DecisionEngine()
    
    classes = ['surface_injure', 'hide_craze', 'craze', 'corrosion', 'crack', 'thunderstrike']
    
    records = []
    counts = {'REPAIR': 0, 'MONITOR': 0, 'ENGINEERING_ASSESSMENT': 0}
    
    max_trials = 50000
    trial = 0
    
    while any(counts[d] < target_samples_per_decision for d in counts) and trial < max_trials:
        trial += 1
        cls_name = np.random.choice(classes)
        
        # Decide regime bias to encourage target balance
        deficit = [d for d in counts if counts[d] < target_samples_per_decision]
        target_focus = np.random.choice(deficit) if deficit else 'MONITOR'
        
        if target_focus == 'REPAIR':
            # Outboard severe erosion/corrosion/wear with high tariff or reasonable repair cost
            cls_name = np.random.choice(['surface_injure', 'corrosion', 'hide_craze', 'craze'])
            r_R = float(np.random.uniform(0.80, 0.98))
            severity = int(np.random.choice([3, 4, 5]))
            area_pct = float(np.random.uniform(2.5, 12.0))
            wind_speed = float(np.random.uniform(7.0, 11.5))
            tariff = float(np.random.uniform(4.0, 6.5))
            repair_cost = float(np.random.choice([25000.0, 35000.0, 45000.0, 55000.0]))
        elif target_focus == 'ENGINEERING_ASSESSMENT':
            # Structural defects or high uncertainty
            if np.random.rand() < 0.8:
                cls_name = np.random.choice(['crack', 'thunderstrike'])
                severity = int(np.random.choice([3, 4, 5]))
            else:
                cls_name = np.random.choice(classes)
                severity = 3
            r_R = float(np.random.uniform(0.30, 0.95))
            area_pct = float(np.random.uniform(1.0, 8.0))
            wind_speed = float(np.random.uniform(5.0, 15.0))
            tariff = float(np.random.uniform(3.5, 5.5))
            repair_cost = float(np.random.choice([40000.0, 60000.0, 80000.0]))
        else: # MONITOR
            cls_name = np.random.choice(['craze', 'surface_injure', 'corrosion', 'hide_craze'])
            r_R = float(np.random.uniform(0.20, 0.75))
            severity = int(np.random.choice([1, 2]))
            area_pct = float(np.random.uniform(0.2, 3.0))
            wind_speed = float(np.random.uniform(4.0, 12.0))
            tariff = float(np.random.uniform(3.5, 5.0))
            repair_cost = float(np.random.choice([50000.0, 75000.0, 90000.0]))
            
        aspect_ratio = round(float(np.random.uniform(0.5, 3.5)), 3)
        rated_kw = float(np.random.choice([2000.0, 2500.0, 3000.0, 3600.0, 4200.0]))
        confidence = round(float(np.random.uniform(0.75, 0.98)), 3)
        
        # Physics calculation
        aero_res = physics_engine.compute_aerodynamic_loss(
            severity=severity,
            r_R=r_R,
            area_pct=area_pct,
            wind_speed=wind_speed,
            rated_power_kw=rated_kw
        )
        
        # Financial calculation
        fin_res = financial_engine.calculate_losses(
            daily_energy_loss_kwh=aero_res['daily_energy_loss_kwh'],
            tariff=tariff,
            repair_cost=repair_cost
        )
        
        # Decision calculation
        dec_res = decision_engine.evaluate_action(
            defect_class=cls_name,
            severity=severity,
            r_R=r_R,
            area_pct=area_pct,
            daily_loss_inr=fin_res['daily_loss_inr'],
            payback_days=fin_res['payback_days'],
            confidence=confidence
        )
        
        assigned_decision = dec_res['decision']
        if counts[assigned_decision] < target_samples_per_decision:
            counts[assigned_decision] += 1
            records.append({
                'sample_id': f"synth_{len(records):06d}",
                'defect_class': cls_name,
                'severity': severity,
                'r_R': round(r_R, 3),
                'area_pct': round(area_pct, 3),
                'aspect_ratio': aspect_ratio,
                'wind_speed': round(wind_speed, 2),
                'rated_power_kw': rated_kw,
                'power_baseline_kw': aero_res['power_baseline_kw'],
                'power_loss_kw': aero_res['power_loss_kw'],
                'power_damaged_kw': aero_res['power_damaged_kw'],
                'aep_loss_pct': aero_res['aep_loss_pct'],
                'daily_energy_loss_kwh': aero_res['daily_energy_loss_kwh'],
                'tariff_inr': round(tariff, 2),
                'daily_loss_inr': fin_res['daily_loss_inr'],
                'repair_cost_inr': repair_cost,
                'payback_days': fin_res['payback_days'],
                'confidence': confidence,
                'decision': assigned_decision,
                'urgency': dec_res['urgency']
            })
            
    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    out_path = os.path.join(SYNTHETIC_DIR, "aeroloss_multimodal_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} perfectly balanced multimodal samples at {out_path}")
    print("\nDecision distribution:\n", df['decision'].value_counts())
    print("\nDefect class distribution:\n", df['defect_class'].value_counts())
    return df

if __name__ == "__main__":
    generate_multimodal_dataset(target_samples_per_decision=1200)
