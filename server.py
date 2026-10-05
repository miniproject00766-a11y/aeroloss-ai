import os
import sys
import json
import base64
import io
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


from src.pipeline import AeroLossPipeline
from src.physics.aero_engine import AeroLossPhysicsEngine
import pandas as pd

PATCH_DIR = os.path.join(BASE_DIR, "data", "processed", "patches")
STATIC_DIR = os.path.join(BASE_DIR, "static")
FEATURES_CSV = os.path.join(BASE_DIR, "data", "processed", "defect_features.csv")

# Load real defect features metadata index from annotations
defect_meta_dict = {}
if os.path.exists(FEATURES_CSV):
    try:
        df_feat = pd.read_csv(FEATURES_CSV)
        for _, row in df_feat.iterrows():
            fname = os.path.basename(str(row['patch_path']))
            defect_meta_dict[fname] = {
                'r_R': float(row['r_R']),
                'area_pct': float(row['area_pct']),
                'severity_proxy': int(row['severity_proxy']),
                'defect_class': str(row['defect_class'])
            }
        print(f"Loaded ground-truth metadata for {len(defect_meta_dict)} defect patches.")
    except Exception as e:
        print(f"Warning: could not index defect_features.csv: {e}")

pipeline = AeroLossPipeline()
physics_engine = AeroLossPhysicsEngine()

class AeroLossStitchHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
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

        if path == '/' or path == '/index.html':
            self._send_file(os.path.join(STATIC_DIR, "index.html"), "text/html")
        elif path == '/api/samples':
            curated_list = [
                {
                    "class_name": "corrosion",
                    "filename": "patch_00384.jpg",
                    "label": "Corrosion (Tip)",
                    "badge": "r/R 0.88 • Tip",
                    "zone": "Outboard Tip"
                },
                {
                    "class_name": "surface_injure",
                    "filename": "patch_00003.jpg",
                    "label": "Surface Injure (Tip)",
                    "badge": "r/R 0.93 • Tip",
                    "zone": "Outboard Tip"
                },
                {
                    "class_name": "thunderstrike",
                    "filename": "patch_00427.jpg",
                    "label": "Strike (Tip)",
                    "badge": "r/R 0.90 • Tip",
                    "zone": "Outboard Tip"
                },
                {
                    "class_name": "crack",
                    "filename": "patch_00155.jpg",
                    "label": "Crack (Outboard)",
                    "badge": "r/R 0.76 • Outboard",
                    "zone": "Outboard Transition"
                },
                {
                    "class_name": "corrosion",
                    "filename": "patch_00000.jpg",
                    "label": "Corrosion (Mid)",
                    "badge": "r/R 0.59 • Mid",
                    "zone": "Mid-Span"
                },
                {
                    "class_name": "crack",
                    "filename": "patch_00058.jpg",
                    "label": "Crack (Mid)",
                    "badge": "r/R 0.61 • Mid",
                    "zone": "Mid-Span"
                }
            ]
            samples = []
            for item in curated_list:
                c = item["class_name"]
                fname = item["filename"]
                fpath = os.path.join(PATCH_DIR, c, fname)
                if os.path.exists(fpath):
                    meta = defect_meta_dict.get(fname, {})
                    samples.append({
                        "class_name": c,
                        "filename": fname,
                        "preview_url": f"/api/sample_image/{c}/{fname}",
                        "label": item["label"],
                        "badge": item["badge"],
                        "zone": item["zone"],
                        "r_R": meta.get('r_R', 0.88),
                        "area_pct": meta.get('area_pct', 4.2),
                        "ground_truth_severity": meta.get('severity_proxy', 2)
                    })
            # Fallback if any file wasn't found
            if not samples and os.path.exists(PATCH_DIR):
                for c in sorted(os.listdir(PATCH_DIR)):
                    c_dir = os.path.join(PATCH_DIR, c)
                    if os.path.isdir(c_dir):
                        files = [f for f in os.listdir(c_dir) if f.endswith(('.jpg', '.png'))]
                        if files:
                            fname = files[0]
                            meta = defect_meta_dict.get(fname, {})
                            samples.append({
                                "class_name": c,
                                "filename": fname,
                                "preview_url": f"/api/sample_image/{c}/{fname}",
                                "label": c.replace('_', ' ').title(),
                                "badge": f"r/R {meta.get('r_R', 0.88):.2f}",
                                "zone": "Default",
                                "r_R": meta.get('r_R', 0.88),
                                "area_pct": meta.get('area_pct', 4.2),
                                "ground_truth_severity": meta.get('severity_proxy', 2)
                            })
            self._send_json({"samples": samples})
        elif path.startswith('/api/sample_image/'):
            parts = path.replace('/api/sample_image/', '').split('/')
            if len(parts) == 2:
                c, f = parts
                fpath = os.path.join(PATCH_DIR, c, f)
                self._send_file(fpath)
            else:
                self.send_error(400, "Invalid image URL")
        elif path == '/api/metrics':
            summary_path = os.path.join(BASE_DIR, "models", "final_evaluation_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    self._send_json(json.load(f))
            else:
                self._send_json({"status": "Metrics not found"})
        else:
            local_path = os.path.join(STATIC_DIR, path.lstrip('/'))
            if os.path.exists(local_path):
                self._send_file(local_path)
            else:
                self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/analyze':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                raw_data = self.rfile.read(content_length)
                data = json.loads(raw_data.decode('utf-8'))

                wind_speed = float(data.get('wind_speed', 7.56))
                rated_kw = float(data.get('rated_kw', 3600.0))
                tariff = float(data.get('tariff', 4.50))
                repair_cost = float(data.get('repair_cost', 40000.0))
                sample_class = data.get('sample_class', 'crack')
                sample_file = data.get('sample_file', '')
                image_b64 = data.get('image_base64', None)
                r_R = data.get('r_R', None)
                if r_R is not None and r_R != "":
                    r_R = float(r_R)
                else:
                    r_R = None

                area_pct = data.get('area_pct', None)
                if area_pct is not None and area_pct != "":
                    area_pct = float(area_pct)
                else:
                    area_pct = None

                # Look up real photogrammetric ground truth if available
                if (r_R is None or area_pct is None) and sample_file in defect_meta_dict:
                    meta = defect_meta_dict[sample_file]
                    if r_R is None:
                        r_R = float(meta.get('r_R', 0.88))
                    if area_pct is None:
                        area_pct = float(meta.get('area_pct', 4.2))

                img = None
                if image_b64:
                    if ',' in image_b64:
                        image_b64 = image_b64.split(',', 1)[1]
                    img_bytes = base64.b64decode(image_b64)
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                elif sample_class and sample_file:
                    path = os.path.join(PATCH_DIR, sample_class, sample_file)
                    if os.path.exists(path):
                        img = Image.open(path).convert('RGB')

                if img is None:
                    c_dir = os.path.join(PATCH_DIR, "crack")
                    files = os.listdir(c_dir)
                    img = Image.open(os.path.join(c_dir, files[0])).convert('RGB')

                res = pipeline.analyze_image(
                    img,
                    r_R=r_R,
                    area_pct=area_pct,
                    wind_speed=wind_speed,
                    rated_kw=rated_kw,
                    tariff=tariff,
                    repair_cost=repair_cost
                )
                self._send_json(res)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
        else:
            self.send_error(404, "API endpoint not found")

def run(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, AeroLossStitchHandler)
    print(f"AeroLoss AI Stitch Dashboard live on http://localhost:{port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run(8080)
