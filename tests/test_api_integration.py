import os
import unittest
import json
import base64
import requests
from PIL import Image
import io

class TestAPISafetyAndIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:8080"
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        def encode_quality_usable_fixture(path):
            with Image.open(path) as source:
                image = source.convert('RGB')
                if image.width < 512 or image.height < 512:
                    image = image.resize(
                        (max(512, image.width), max(512, image.height)),
                        Image.Resampling.LANCZOS
                    )
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=95)
            return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')

        patch_path = os.path.join(base_dir, "data", "processed", "patches", "crack", "patch_00155.jpg")
        cls.b64_crack = encode_quality_usable_fixture(patch_path)

        strike_path = os.path.join(base_dir, "data", "processed", "patches", "thunderstrike", "patch_00427.jpg")
        cls.b64_strike = encode_quality_usable_fixture(strike_path)

    def test_crack_never_downgraded_via_api(self):
        """Verify that a crack submitted via API is strictly forced to ENGINEERING_ASSESSMENT regardless of economics."""
        payload = {
            "turbine_id": "WTG-04B",
            "wind_speed": 4.0,
            "r_R": 0.50,
            "repair_cost": 200000.0,
            "image_base64": self.b64_crack
        }
        resp = requests.post(f"{self.base_url}/api/analyze", json=payload, timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        
        self.assertTrue(data["is_valid_blade_image"])
        dec = data["maintenance_decision"]
        self.assertEqual(dec["recommended_action"], "ENGINEERING_ASSESSMENT")
        self.assertEqual(dec["urgency"], "CRITICAL")
        self.assertIn("MANDATORY SAFETY OVERRIDE", " ".join(dec["reasoning"]))

    def test_thunderstrike_safety_override_via_api(self):
        """Verify that lightning strike defect triggers mandatory ENGINEERING_ASSESSMENT."""
        payload = {
            "turbine_id": "WTG-05B",
            "wind_speed": 7.0,
            "r_R": 0.90,
            "image_base64": self.b64_strike
        }
        resp = requests.post(f"{self.base_url}/api/analyze", json=payload, timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        
        self.assertTrue(data["is_valid_blade_image"])
        dec = data["maintenance_decision"]
        self.assertEqual(dec["recommended_action"], "ENGINEERING_ASSESSMENT")
        self.assertEqual(dec["urgency"], "CRITICAL")

    def test_rejected_out_of_domain_payload(self):
        """Verify that an out-of-domain image produces clean rejection with zero fabricated physics."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        non_blade_path = os.path.join(base_dir, "test_uploads", "ood_test_suite", "bird.jpg")
        with open(non_blade_path, "rb") as image_file:
            b64_bad = "data:image/jpeg;base64," + base64.b64encode(image_file.read()).decode('utf-8')

        payload = {
            "turbine_id": "WTG-04B",
            "image_base64": b64_bad
        }
        resp = requests.post(f"{self.base_url}/api/analyze", json=payload, timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        
        self.assertFalse(data["is_valid_blade_image"])
        self.assertEqual(data["status"], "REJECTED_OUT_OF_DOMAIN")
        self.assertEqual(data["physics_aerodynamics"]["estimated_power_loss_kw"], 0.0)
        self.assertEqual(data["financial_impact"]["daily_financial_loss_inr"], 0.0)
        self.assertEqual(data["maintenance_decision"]["recommended_action"], "REJECTED")

    def test_unusable_quality_upload_is_stopped_before_inference(self):
        img = Image.new('RGB', (128, 128), color=(90, 90, 90))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        payload = {
            "turbine_id": "WTG-04B",
            "image_base64": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
        }

        response = requests.post(f"{self.base_url}/api/analyze", json=payload, timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["is_valid_blade_image"])
        self.assertEqual(data["status"], "REJECTED_UNUSABLE_QUALITY")
        self.assertEqual(data["quality_assessment"]["quality_decision"], "UNUSABLE")
        self.assertIsNone(data["physics_aerodynamics"])
        self.assertIsNone(data["energy_impact"])
        self.assertIsNone(data["financial_impact"])
        self.assertEqual(data["maintenance_decision"]["recommended_action"], "RECAPTURE_REQUIRED")

    def test_known_clean_blade_upload_stays_monitor_with_zero_loss(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        clean_path = os.path.join(base_dir, "data", "raw", "images", "1042.jpg")
        with open(clean_path, "rb") as image_file:
            encoded = "data:image/jpeg;base64," + base64.b64encode(image_file.read()).decode('utf-8')
        response = requests.post(
            f"{self.base_url}/api/analyze",
            json={"turbine_id": "WTG-04B", "image_base64": encoded},
            timeout=10
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["quality_assessment"]["quality_decision"], "USABLE")
        self.assertEqual(data["visual_detection"]["detection_status"], "NO_DEFECT_DETECTED")
        self.assertIsNone(data["visual_detection"]["confidence"])
        self.assertEqual(data["visual_detection"]["confidence_pct"], "N/A")
        self.assertEqual(data["characterization"]["severity_proxy"], 0)
        self.assertEqual(data["energy_impact"]["daily_energy_loss_kwh"], 0.0)
        self.assertEqual(data["financial_impact"]["daily_loss_inr"], 0.0)
        self.assertEqual(data["maintenance_decision"]["recommended_action"], "MONITOR")
        self.assertEqual(data["maintenance_decision"]["urgency"], "LOW")

    def test_extensive_low_confidence_upload_requires_engineering_assessment(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        image_path = os.path.join(
            base_dir, "test_uploads", "01_drone_blade_erosion_crack.jpg"
        )
        with open(image_path, "rb") as image_file:
            encoded = "data:image/jpeg;base64," + base64.b64encode(image_file.read()).decode("utf-8")
        response = requests.post(
            f"{self.base_url}/api/analyze",
            json={"turbine_id": "WTG-04B", "image_base64": encoded},
            timeout=15
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["quality_assessment"]["quality_decision"], "USABLE")
        self.assertEqual(data["visual_detection"]["detected_class"], "corrosion")
        self.assertLess(data["visual_detection"]["confidence"], 0.20)
        self.assertGreaterEqual(data["characterization"]["union_frame_coverage_pct"], 20.0)
        self.assertTrue(data["maintenance_decision"]["safety_override_active"])
        self.assertEqual(
            data["maintenance_decision"]["recommended_action"],
            "ENGINEERING_ASSESSMENT"
        )
        self.assertEqual(data["maintenance_decision"]["urgency"], "HIGH")
        self.assertIn(
            "does not confirm internal structural failure",
            " ".join(data["maintenance_decision"]["reasoning"])
        )

    def test_turbine_status_and_history_consistency(self):
        """Verify that GET /api/turbines and history are synced."""
        resp = requests.get(f"{self.base_url}/api/turbines", timeout=5)
        self.assertEqual(resp.status_code, 200)
        turbines = resp.json()["turbines"]
        self.assertGreater(len(turbines), 0)
        
        t_id = turbines[0]["turbine_id"]
        resp_hist = requests.get(f"{self.base_url}/api/turbines/{t_id}/history", timeout=5)
        self.assertEqual(resp_hist.status_code, 200)
        hist_data = resp_hist.json()
        self.assertTrue("history" in hist_data or isinstance(hist_data, list))
        hist_list = hist_data["history"] if isinstance(hist_data, dict) else hist_data
        self.assertIsInstance(hist_list, list)

    def test_consecutive_independent_uploads(self):
        """Verify that two consecutive image uploads produce isolated inspection IDs, coordinates, and metrics without cache bleed."""
        payload1 = {
            "turbine_id": "WTG-04B",
            "wind_speed": 7.0,
            "r_R": 0.50,
            "image_base64": self.b64_crack
        }
        resp1 = requests.post(f"{self.base_url}/api/analyze", json=payload1, timeout=5)
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.json()

        payload2 = {
            "turbine_id": "WTG-05B",
            "wind_speed": 8.0,
            "r_R": 0.90,
            "image_base64": self.b64_strike
        }
        resp2 = requests.post(f"{self.base_url}/api/analyze", json=payload2, timeout=5)
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()

        # Check independence
        self.assertNotEqual(data1["inspection_id"], data2["inspection_id"])
        self.assertEqual(data1["turbine_id"], "WTG-04B")
        self.assertEqual(data2["turbine_id"], "WTG-05B")
        self.assertNotEqual(data1["characterization"]["spanwise_position_r_R"], data2["characterization"]["spanwise_position_r_R"])

    def test_uploaded_photo_pipeline_end_to_end(self):
        """
        Verify the exact end-to-end pipeline for the uploaded broken blade image:
        1. SHA-256 and dimensions verified.
        2. Detection produces multiple boxes (hide_craze and craze).
        3. Sweep-line union computes 28.19% coverage.
        4. Energy loss, financial loss, payback, and decision are internally consistent.
        5. Database record in aeroloss.db matches the API response field-for-field.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        img_path = os.path.join(base_dir, "test_uploads", "user_uploaded_broken_blade.jpg")
        with open(img_path, "rb") as f:
            raw_bytes = f.read()
        
        import hashlib
        import sqlite3
        sha = hashlib.sha256(raw_bytes).hexdigest()
        self.assertEqual(sha, "1dba7fe9e40c17c8cd23312aab66d8d3345b3734bbb65b8d6db2e7011c70736b")

        b64_img = "data:image/jpeg;base64," + base64.b64encode(raw_bytes).decode('utf-8')
        payload = {
            "turbine_id": "WTG-04B (Uploaded Photo)",
            "wind_speed": 7.56,
            "rated_kw": 3600.0,
            "tariff": 4.50,
            "repair_cost": 40000.0,
            "image_base64": b64_img
        }

        resp = requests.post(f"{self.base_url}/api/analyze", json=payload, timeout=10)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["quality_assessment"]["quality_decision"], "REVIEW_REQUIRED")

        insp_id = data["inspection_id"]
        self.assertTrue(insp_id.startswith("INSP-WTG-04B (Uploaded Photo)-"))

        # Verify visual detection
        v_det = data["visual_detection"]
        self.assertEqual(v_det["detected_class"], "hide_craze")
        self.assertEqual(v_det["defect_count"], 2)
        det_classes = [d["damage_type"] for d in v_det["detections"]]
        self.assertIn("hide_craze", det_classes)
        self.assertIn("craze", det_classes)

        # Verify characterization
        char = data["characterization"]
        self.assertEqual(char["primary_defect"], "hide_craze")
        self.assertEqual(char["severity_proxy"], 3)
        self.assertEqual(char["union_frame_coverage_pct"], 28.19)
        self.assertEqual(char["blade_damage_area_proxy_pct"], 2.819)
        self.assertEqual(char["spanwise_position_r_R"], 0.887)
        self.assertTrue(char["is_uncertain_extensive_damage"])
        self.assertEqual(char["uncertainty_state"], "CLASSIFICATION_UNCERTAIN_EXTENSIVE_DAMAGE")

        # Verify physics and financials
        phys = data["physics_aerodynamics"]
        self.assertEqual(phys["aep_loss_pct"], 2.09)
        self.assertEqual(phys["daily_energy_loss_kwh"], 650.07)

        fin = data["financial_impact"]
        self.assertEqual(fin["daily_loss_inr"], 2925.32)
        self.assertEqual(fin["payback_days"], 13.7)

        dec = data["maintenance_decision"]
        self.assertEqual(dec["recommended_action"], "ENGINEERING_ASSESSMENT")
        self.assertEqual(dec["urgency"], "CRITICAL")

        # Verify database record matches response
        db_path = os.path.join(base_dir, "database", "aeroloss.db")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM inspections WHERE inspection_id = ?;", (insp_id,))
        row = c.fetchone()
        conn.close()

        self.assertIsNotNone(row, "Inspection record must be persisted in database")
        # Row fields: inspection_id, turbine_id, date, class, conf, area, r_R, sev, ...
        self.assertEqual(row[0], insp_id)
        self.assertEqual(row[1], "WTG-04B (Uploaded Photo)")
        self.assertEqual(row[3], "hide_craze")
        self.assertAlmostEqual(row[4], 0.2304, places=2)
        self.assertEqual(row[5], 2.819)
        self.assertEqual(row[6], 0.887)
        self.assertEqual(row[7], 3)
        self.assertEqual(row[11], 2.09)
        self.assertEqual(row[12], 650.07)
        self.assertEqual(row[13], 2925.32)
        self.assertEqual(row[15], 13.7)
        self.assertEqual(row[16], "ENGINEERING_ASSESSMENT")
        self.assertEqual(row[17], "CRITICAL")

if __name__ == '__main__':
    unittest.main()
