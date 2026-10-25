import json

with open("data/manifests/split_map.json", "r", encoding="utf-8") as f:
    split_json = f.read().strip()

template = """import os
import shutil
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import yaml

REPO_DIR = Path("/content/aeroloss_repo")
RAW_IMG_DIR = REPO_DIR / "data" / "raw" / "images"
RAW_ANN_DIR = REPO_DIR / "data" / "raw" / "annotations"
OUTPUT_YOLO_DIR = Path("/content/yolo_dataset")

CLASS_NAMES = ["corrosion", "crack", "craze", "hide_craze", "surface_injure", "thunderstrike"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

split_map = SPLIT_MAP_PLACEHOLDER

# Clean and recreate directory layout
for folder in ["images", "labels"]:
    target_dir = OUTPUT_YOLO_DIR / folder
    if target_dir.exists():
        shutil.rmtree(target_dir)
    for split in ["train", "val", "test"]:
        (target_dir / split).mkdir(parents=True, exist_ok=True)

img_files = sorted(list(RAW_IMG_DIR.glob("*.jpg")))
xml_files = sorted(list(RAW_ANN_DIR.glob("*.xml")))
assert len(img_files) == 1065, f"Expected 1065 images, found {len(img_files)}"
assert len(xml_files) == 1065, f"Expected 1065 XMLs, found {len(xml_files)}"

split_counts = {"train": 0, "val": 0, "test": 0}
split_box_counts = {s: Counter() for s in ["train", "val", "test"]}
total_boxes = 0

for img_p in img_files:
    fn = img_p.name
    stem = img_p.stem
    xml_p = RAW_ANN_DIR / f"{stem}.xml"
    split = split_map[fn]
    split_counts[split] += 1
    
    dest_img = OUTPUT_YOLO_DIR / "images" / split / fn
    dest_lbl = OUTPUT_YOLO_DIR / "labels" / split / f"{stem}.txt"
    shutil.copy2(img_p, dest_img)
    
    tree = ET.parse(xml_p)
    root = tree.getroot()
    yolo_lines = []
    
    for obj in root.findall("object"):
        raw_cls = obj.find("name").text.strip().lower()
        if raw_cls not in CLASS_TO_IDX:
            continue
        cls_idx = CLASS_TO_IDX[raw_cls]
        bndbox = obj.find("bndbox")
        xmin = max(0.0, min(1024.0, float(bndbox.find("xmin").text)))
        ymin = max(0.0, min(1024.0, float(bndbox.find("ymin").text)))
        xmax = max(0.0, min(1024.0, float(bndbox.find("xmax").text)))
        ymax = max(0.0, min(1024.0, float(bndbox.find("ymax").text)))
        
        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 0 or bh <= 0:
            continue
            
        xc = (xmin + xmax) / (2.0 * 1024.0)
        yc = (ymin + ymax) / (2.0 * 1024.0)
        wn = bw / 1024.0
        hn = bh / 1024.0
        
        yolo_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        split_box_counts[split][raw_cls] += 1
        total_boxes += 1
        
    with open(dest_lbl, "w", encoding="utf-8") as f:
        f.write("\\n".join(yolo_lines) + ("\\n" if yolo_lines else ""))

# Generate dataset.yaml
dataset_cfg = {
    "path": str(OUTPUT_YOLO_DIR.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {idx: name for idx, name in enumerate(CLASS_NAMES)}
}
with open(OUTPUT_YOLO_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
    yaml.dump(dataset_cfg, f, sort_keys=False)

print("=== DATASET EXTRACTION AND CONVERSION SUMMARY ===")
print("Total images converted:", len(img_files))
print("Total boxes converted:", total_boxes)
for s in ["train", "val", "test"]:
    print(f"[{s.upper()}] Images: {split_counts[s]} | Boxes: {sum(split_box_counts[s].values())}")
    for c in CLASS_NAMES:
        print(f"   - {c:<16}: {split_box_counts[s][c]}")

# Also zip to /content/aeroloss_yolo_dataset.zip for verification
shutil.make_archive("/content/aeroloss_yolo_dataset", "zip", "/content", "yolo_dataset")
zip_size_mb = os.path.getsize("/content/aeroloss_yolo_dataset.zip") / (1024*1024)
print(f"Packaged ZIP archive at /content/aeroloss_yolo_dataset.zip ({zip_size_mb:.2f} MB)")
"""

script = template.replace("SPLIT_MAP_PLACEHOLDER", split_json)
with open("colab_cell_code.py", "w", encoding="utf-8") as f:
    f.write(script)
print("Saved colab_cell_code.py, length:", len(script))
