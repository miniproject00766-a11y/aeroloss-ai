import os
import sys
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import AeroLossPipeline

p = AeroLossPipeline()

print("--- TESTING USER FABRIC ---")
res_fabric = p.analyze_image("test_uploads/user_fabric_test.jpg")
print("Status:", res_fabric.get('status'))
print("Is Valid:", res_fabric.get('is_valid_blade_image'))
print("Rejection Reason:", res_fabric.get('rejection_reason'))

print("\n--- TESTING AUTHENTIC BLADE DEFECT IMAGES ---")
for blade_file in sorted(glob.glob("test_uploads/0[1-5]*.jpg")):
    res = p.analyze_image(blade_file)
    print(f"{os.path.basename(blade_file)} -> Status: {res.get('status', 'SUCCESS')}, Detected: {res.get('damage_type')}, Severity: {res.get('severity_level')}, Loss: {res.get('daily_energy_loss_kwh')} kWh/d")
