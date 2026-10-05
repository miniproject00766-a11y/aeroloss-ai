import requests
import base64
import json

# Read user fabric image as base64
with open("test_uploads/user_fabric_test.jpg", "rb") as f:
    b64_fabric = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "turbine_id": "WTG-04B",
    "wind_speed": 7.56,
    "image_base64": b64_fabric
}

resp = requests.post("http://localhost:8080/api/analyze", json=payload, timeout=20)
print("--- FABRIC TEST ---")
print("HTTP Status Code:", resp.status_code)
data = resp.json()
print("is_valid_blade_image:", data.get("is_valid_blade_image"))
print("status:", data.get("status"))
print("rejection_reason:", data.get("rejection_reason"))
print("maintenance_decision:", data.get("maintenance_decision", {}).get("recommended_action"))
print("daily_loss_inr:", data.get("financial_impact", {}).get("daily_loss_inr"))

with open("test_uploads/02_real_dataset_crack_patch_00155.jpg", "rb") as f:
    b64_blade = base64.b64encode(f.read()).decode("utf-8")

payload_blade = {
    "turbine_id": "WTG-04B",
    "wind_speed": 7.56,
    "image_base64": b64_blade
}

resp_blade = requests.post("http://localhost:8080/api/analyze", json=payload_blade, timeout=20)
print("\n--- AUTHENTIC BLADE TEST ---")
print("HTTP Status Code:", resp_blade.status_code)
data_blade = resp_blade.json()
print("is_valid_blade_image:", data_blade.get("is_valid_blade_image"))
print("status:", data_blade.get("status"))
print("detected_class:", data_blade.get("visual_detection", {}).get("detected_class"))
print("confidence_pct:", data_blade.get("visual_detection", {}).get("confidence_pct"))
print("maintenance_decision:", data_blade.get("maintenance_decision", {}).get("recommended_action"))
print("daily_loss_inr: Rs.", data_blade.get("financial_impact", {}).get("daily_financial_loss_inr"))
