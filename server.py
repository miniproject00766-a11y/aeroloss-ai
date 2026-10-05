import os
import sys
import json
import base64
import io
import mimetypes
import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from database.database import get_db_connection, init_db
from services.yolo_service import YoloService
from services.damage_service import DamageService
from services.physics_service import PhysicsService
from services.financial_service import FinancialService
from services.decision_service import DecisionService
from src.pipeline import AeroLossPipeline

# Ensure DB initialized
init_db()

# Initialize Service Layer
yolo_service = YoloService()
damage_service = DamageService()
physics_service = PhysicsService()
financial_service = FinancialService()
decision_service = DecisionService()
pipeline = AeroLossPipeline()

PATCH_DIR = os.path.join(BASE_DIR, "data", "processed", "patches")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODELS_DIR = os.path.join(BASE_DIR, "models")

class AeroLossAPIHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, filepath, content_type=None):
        if not os.path.exists(filepath):
            self.send_error(404, "File not found")
            return
        if not content_type:
            content_type, _ = mimetypes.guess_type(filepath)
            if not content_type:
                content_type = 'application/octet-stream'
        with open(filepath, 'rb') as f:
            data = f.read()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == '/' or path == '/index.html':
            self._send_file(os.path.join(STATIC_DIR, "index.html"), "text/html")

        # 1. GET /api/turbines - List all turbines with live status
        elif path == '/api/turbines':
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM turbines ORDER BY turbine_id ASC;")
            rows = cursor.fetchall()
            conn.close()

            turbines = [dict(r) for r in rows]
            self._send_json({"turbines": turbines, "total_count": len(turbines)})

        # 2. GET /api/turbines/{turbine_id} - Fetch single turbine
        elif path.startswith('/api/turbines/') and not path.endswith('/history') and not path.endswith('/analysis') and not path.endswith('/inspect'):
            turb_id = path.replace('/api/turbines/', '').strip()
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM turbines WHERE turbine_id = ?;", (turb_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                self._send_json(dict(row))
            else:
                self._send_json({"error": f"Turbine {turb_id} not found"}, status=404)

        # 3. GET /api/turbines/{turbine_id}/history - Fetch inspection history
        elif path.startswith('/api/turbines/') and path.endswith('/history'):
            parts = path.split('/')
            turb_id = parts[3]
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM inspections WHERE turbine_id = ? ORDER BY inspection_date ASC;", (turb_id,))
            rows = cursor.fetchall()
            conn.close()

            history = [dict(r) for r in rows]
            self._send_json({"turbine_id": turb_id, "history": history, "inspection_count": len(history)})

        # 4. GET /api/turbines/{turbine_id}/analysis - Fetch latest analysis
        elif path.startswith('/api/turbines/') and path.endswith('/analysis'):
            parts = path.split('/')
            turb_id = parts[3]
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM turbines WHERE turbine_id = ?;", (turb_id,))
            turb_row = cursor.fetchone()
            
            if not turb_row:
                conn.close()
                self._send_json({"error": f"Turbine {turb_id} not found"}, status=404)
                return

            cursor.execute("SELECT * FROM inspections WHERE turbine_id = ? ORDER BY inspection_date DESC LIMIT 1;", (turb_id,))
            insp_row = cursor.fetchone()
            conn.close()

            if insp_row:
                self._send_json({"turbine": dict(turb_row), "latest_analysis": dict(insp_row)})
            else:
                self._send_json({"turbine": dict(turb_row), "latest_analysis": None})

        # 5. GET /api/model/metrics - Fetch actual YOLOv8 evaluation metrics
        elif path == '/api/model/metrics':
            metrics_file = os.path.join(MODELS_DIR, "model_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file) as f:
                    data = json.load(f)
                self._send_json(data)
            else:
                self._send_json({
                    "status": "Model evaluation pending",
                    "model_name": "YOLOv8n",
                    "version": "v2.4",
                    "metrics": {
                        "precision": "Pending",
                        "recall": "Pending",
                        "f1_score": "Pending",
                        "mAP_50": "Pending",
                        "mAP_50_95": "Pending"
                    }
                })

        # 6. GET /api/model/status
        elif path == '/api/model/status':
            weights_file = os.path.join(MODELS_DIR, "best.pt")
            is_ready = os.path.exists(weights_file)
            self._send_json({
                "model": "YOLOv8",
                "framework": "PyTorch / Ultralytics",
                "weights_found": is_ready,
                "weights_path": weights_file if is_ready else "Not loaded",
                "physics_airfoil": "FFA-W3-241 (Re=1e7)",
                "status": "Production Active" if is_ready else "Ready (Fallback mode)"
            })

        # 7. GET /api/samples
        elif path == '/api/samples':
            curated_list = [
                {"class_name": "corrosion", "filename": "patch_00384.jpg", "label": "Corrosion (Tip)", "badge": "r/R 0.88 • Tip", "zone": "Outboard Tip"},
                {"class_name": "surface_injure", "filename": "patch_00003.jpg", "label": "Surface Injure (Tip)", "badge": "r/R 0.93 • Tip", "zone": "Outboard Tip"},
                {"class_name": "thunderstrike", "filename": "patch_00427.jpg", "label": "Strike (Tip)", "badge": "r/R 0.90 • Tip", "zone": "Outboard Tip"},
                {"class_name": "crack", "filename": "patch_00155.jpg", "label": "Crack (Outboard)", "badge": "r/R 0.76 • Outboard", "zone": "Outboard Transition"},
                {"class_name": "corrosion", "filename": "patch_00000.jpg", "label": "Corrosion (Mid)", "badge": "r/R 0.59 • Mid", "zone": "Mid-Span"},
                {"class_name": "crack", "filename": "patch_00058.jpg", "label": "Crack (Mid)", "badge": "r/R 0.61 • Mid", "zone": "Mid-Span"}
            ]
            samples = []
            for item in curated_list:
                c = item["class_name"]
                fname = item["filename"]
                fpath = os.path.join(PATCH_DIR, c, fname)
                if os.path.exists(fpath):
                    samples.append({
                        "class_name": c,
                        "filename": fname,
                        "preview_url": f"/api/sample_image/{c}/{fname}",
                        "label": item["label"],
                        "badge": item["badge"],
                        "zone": item["zone"]
                    })
            self._send_json({"samples": samples})

        elif path.startswith('/api/sample_image/'):
            parts = path.replace('/api/sample_image/', '').split('/')
            if len(parts) == 2:
                self._send_file(os.path.join(PATCH_DIR, parts[0], parts[1]))
            else:
                self.send_error(400, "Invalid image URL")

        else:
            local_path = os.path.join(STATIC_DIR, path.lstrip('/'))
            if os.path.exists(local_path):
                self._send_file(local_path)
            else:
                self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # 1. POST /api/analyze - Run full ML + Physics + Financial Pipeline
        if path == '/api/analyze':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                raw_data = self.rfile.read(content_length)
                data = json.loads(raw_data.decode('utf-8'))

                turbine_id = data.get('turbine_id', 'WTG-04B')
                wind_speed = float(data.get('wind_speed', 7.56))
                sample_class = data.get('sample_class', 'crack')
                sample_file = data.get('sample_file', '')
                image_b64 = data.get('image_base64', None) or data.get('image_b64', None)

                # Load turbine config from database
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM turbines WHERE turbine_id = ?;", (turbine_id,))
                turb_row = cursor.fetchone()
                
                if turb_row:
                    t_dict = dict(turb_row)
                    rated_kw = float(t_dict['rated_power_kw'])
                    tariff = float(t_dict['tariff'])
                    repair_cost = float(t_dict['repair_cost'])
                    if not data.get('sample_class') and t_dict.get('sample_class'):
                        sample_class = t_dict['sample_class']
                    if not data.get('sample_file') and t_dict.get('sample_file'):
                        sample_file = t_dict['sample_file']
                else:
                    rated_kw = float(data.get('rated_kw', 3600.0))
                    tariff = float(data.get('tariff', 4.50))
                    repair_cost = float(data.get('repair_cost', 40000.0))

                # Load image
                img = None
                if image_b64:
                    if ',' in image_b64:
                        image_b64 = image_b64.split(',', 1)[1]
                    img_bytes = base64.b64decode(image_b64)
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                elif sample_class:
                    c_dir = os.path.join(PATCH_DIR, sample_class)
                    if os.path.exists(c_dir):
                        if sample_file and os.path.exists(os.path.join(c_dir, sample_file)):
                            spath = os.path.join(c_dir, sample_file)
                        else:
                            files = [f for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                            spath = os.path.join(c_dir, files[0]) if files else None
                        if spath and os.path.exists(spath):
                            img = Image.open(spath).convert('RGB')

                if img is None:
                    c_dir = os.path.join(PATCH_DIR, "crack")
                    files = [f for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    img = Image.open(os.path.join(c_dir, files[0])).convert('RGB')

                # Run YOLOv8 detection (autonomous detection if custom image was uploaded)
                effective_target_class = None if image_b64 else sample_class
                detections, annotated_img = yolo_service.detect_defects(img, target_class=effective_target_class)
                
                # Convert annotated image to base64 overlay
                buffered = io.BytesIO()
                annotated_img.save(buffered, format="JPEG")
                overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Characterization
                char = damage_service.characterize_detections(detections, img.width, img.height)

                # Override r_R if explicitly provided in request body (e.g. from spanwise slider)
                r_R_custom = data.get('r_R', None)
                if r_R_custom is not None:
                    char['spanwise_position_r_R'] = float(r_R_custom)
                    char['blade_region'] = 'Outboard (High Speed)' if float(r_R_custom) >= 0.80 else ('Mid-Span' if float(r_R_custom) >= 0.50 else 'Root/Inboard')

                # Physics BEM
                phys = physics_service.compute_aerodynamics(
                    severity=char['severity_proxy'],
                    r_R=char['spanwise_position_r_R'],
                    area_pct=char['damage_area_pct'],
                    wind_speed=wind_speed,
                    rated_kw=rated_kw
                )

                # Financial Engine
                fin = financial_service.compute_financials(
                    daily_energy_loss_kwh=phys['daily_energy_loss_kwh'],
                    tariff=tariff,
                    repair_cost=repair_cost
                )

                # Decision Engine
                dec = decision_service.evaluate_decision(
                    defect_class=char['primary_defect'],
                    severity=char['severity_proxy'],
                    r_R=char['spanwise_position_r_R'],
                    area_pct=char['damage_area_pct'],
                    daily_loss_inr=fin['daily_loss_inr'],
                    payback_days=fin['payback_days'],
                    confidence=char.get('confidence', 0.90),
                    wind_speed=wind_speed,
                    repair_cost=repair_cost,
                    aep_loss_pct=phys['aep_loss_pct']
                )

                # Save inspection record to database
                insp_id = f"INSP-{turbine_id}-{int(np.random.randint(1000, 9999))}"
                cursor.execute("""
                INSERT OR REPLACE INTO inspections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
                """, (
                    insp_id, turbine_id, "2026-10-04", char['primary_defect'], char.get('confidence', 0.90),
                    char['damage_area_pct'], char['spanwise_position_r_R'], char['severity_proxy'],
                    phys['power_baseline_kw'], phys['power_damaged_kw'], phys['power_loss_kw'], phys['aep_loss_pct'],
                    phys['daily_energy_loss_kwh'], fin['daily_loss_inr'], fin['repair_cost_inr'], fin['payback_days'],
                    dec['decision'], dec['urgency'], json.dumps(dec['reasons'])
                ))
                
                # Update turbine live status in database
                health_map = {'REPAIR': 'Critical' if char['severity_proxy'] >= 4 else 'High Risk', 'MONITOR': 'Monitor', 'ENGINEERING_ASSESSMENT': 'Monitor'}
                cursor.execute("""
                UPDATE turbines SET health_status = ?, severity_grade = ?, defect_summary = ?, explainability = ? WHERE turbine_id = ?;
                """, (
                    health_map.get(dec['decision'], 'Monitor'),
                    f"Level {char['severity_proxy']} / 5",
                    char['primary_defect'].replace('_', ' ').title(),
                    " ".join(dec['reasons']),
                    turbine_id
                ))
                conn.commit()
                conn.close()

                response_payload = {
                    "turbine_id": turbine_id,
                    "inspection_id": insp_id,
                    "image_overlay_b64": overlay_b64,
                    "visual_detection": {
                        "detected_class": char['primary_defect'],
                        "confidence": char.get('confidence', 0.90),
                        "confidence_pct": char.get('confidence_pct', '90.0%'),
                        "defect_count": char['defect_count'],
                        "detections": detections
                    },
                    "characterization": char,
                    "physics_aerodynamics": phys,
                    "energy_impact": {
                        "aep_loss_pct": phys['aep_loss_pct'],
                        "daily_energy_loss_kwh": phys['daily_energy_loss_kwh']
                    },
                    "financial_impact": fin,
                    "maintenance_decision": {
                        "recommended_action": dec['decision'],
                        "urgency": dec['urgency'],
                        "reasoning": dec['reasons']
                    },
                    "uncertainty": {
                        "mode": "SCADA-calibrated & Physics BEM",
                        "confidence_note": "AeroLoss AI decision-support system. Derived under configured aerodynamic polars."
                    }
                }

                self._send_json(response_payload)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        # 2. POST /api/turbines - Register new turbine
        elif path == '/api/turbines':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                data = json.loads(self.rfile.read(content_length).decode('utf-8'))

                tid = data.get('turbine_id', f"WTG-0{np.random.randint(7, 99)}A")
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO turbines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
                """, (
                    tid, data.get('turbine_name', f"Turbine {tid}"), data.get('turbine_model', 'Offshore 3.6 MW'),
                    float(data.get('rated_power_kw', 3600.0)), float(data.get('rotor_diameter', 107.0)), 3,
                    float(data.get('hub_height', 90.0)), float(data.get('blade_length', 52.0)), "FFA-W3-241",
                    float(data.get('tariff', 4.50)), 0.42, float(data.get('repair_cost', 40000.0)),
                    data.get('location', 'Offshore Cluster A'), "2026-10-04", "Healthy", "Level 0 / 5",
                    "No Defect", "0.88 (Tip)", "corrosion", "patch_00384.jpg", "Registered turbine unit."
                ))
                conn.commit()
                conn.close()
                self._send_json({"status": "Registered", "turbine_id": tid})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        else:
            self.send_error(404, "API endpoint not found")

def run(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, AeroLossAPIHandler)
    print(f"AeroLoss AI Multi-Turbine REST Backend & Dashboard live on http://localhost:{port}", flush=True)
    httpd.serve_forever()

if __name__ == '__main__':
    run(8080)
