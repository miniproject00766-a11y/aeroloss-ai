import os
import sys
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
YOLO_DIR = os.path.join(BASE_DIR, "data", "yolo_dataset")

CLASSES = ['surface_injure', 'hide_craze', 'corrosion', 'craze', 'crack', 'thunderstrike']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def convert_bbox_to_yolo(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x_center * dw, y_center * dh, w * dw, h * dh)

def prepare_dataset():
    print("Preparing YOLOv8 Dataset structure...")
    img_dir = os.path.join(RAW_DIR, "images")
    ann_dir = os.path.join(RAW_DIR, "annotations")

    if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
        print(f"Error: Raw dataset directories not found: {img_dir}, {ann_dir}")
        return

    # Clean / Create YOLO directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(YOLO_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, "labels", split), exist_ok=True)

    xml_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.xml')])
    valid_samples = []

    for xml_file in xml_files:
        base_id = os.path.splitext(xml_file)[0]
        img_name = base_id + ".jpg"
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            img_name = base_id + ".png"
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue
        valid_samples.append((base_id, img_path, os.path.join(ann_dir, xml_file)))

    print(f"Found {len(valid_samples)} valid image-annotation pairs.")

    # Train (70%), Val (15%), Test (15%) split by image ID
    train_samples, temp_samples = train_test_split(valid_samples, test_size=0.30, random_state=42)
    val_samples, test_samples = train_test_split(temp_samples, test_size=0.50, random_state=42)

    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }

    stats = {split: {'images': len(samples), 'boxes': 0, 'per_class': {c: 0 for c in CLASSES}} for split, samples in splits.items()}

    for split, samples in splits.items():
        split_img_dir = os.path.join(YOLO_DIR, "images", split)
        split_lbl_dir = os.path.join(YOLO_DIR, "labels", split)

        for base_id, img_path, xml_path in samples:
            # Copy image
            dest_img_path = os.path.join(split_img_dir, os.path.basename(img_path))
            shutil.copy2(img_path, dest_img_path)

            # Parse XML
            tree = ET.parse(xml_path)
            root = tree.getroot()

            size_elem = root.find('size')
            if size_elem is not None:
                w = int(size_elem.find('width').text)
                h = int(size_elem.find('height').text)
            else:
                try:
                    with Image.open(img_path) as im:
                        w, h = im.size
                except:
                    w, h = 1024, 1024

            if w == 0 or h == 0:
                w, h = 1024, 1024

            yolo_lines = []
            for obj in root.findall('object'):
                raw_cls = obj.find('name').text.strip().lower()
                if raw_cls not in CLASS_TO_IDX:
                    continue

                cls_idx = CLASS_TO_IDX[raw_cls]
                bndbox = obj.find('bndbox')
                xmin = max(0.0, float(bndbox.find('xmin').text))
                ymin = max(0.0, float(bndbox.find('ymin').text))
                xmax = min(float(w), float(bndbox.find('xmax').text))
                ymax = min(float(h), float(bndbox.find('ymax').text))

                if xmax <= xmin or ymax <= ymin:
                    continue

                yolo_bbox = convert_bbox_to_yolo((w, h), (xmin, ymin, xmax, ymax))
                yolo_lines.append(f"{cls_idx} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")

                stats[split]['boxes'] += 1
                stats[split]['per_class'][raw_cls] += 1

            # Save label txt
            lbl_file = os.path.join(split_lbl_dir, base_id + ".txt")
            with open(lbl_file, 'w') as f:
                f.write('\n'.join(yolo_lines) + '\n')

    # Create dataset.yaml
    yaml_content = f"""# AeroLoss AI Wind Turbine Blade Defect Dataset
path: {os.path.abspath(YOLO_DIR).replace('\\', '/')}
train: images/train
val: images/val
test: images/test

names:
  0: surface_injure
  1: hide_craze
  2: corrosion
  3: craze
  4: crack
  5: thunderstrike
"""
    yaml_path = os.path.join(YOLO_DIR, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print("\n--- DATASET GENERATION SUMMARY ---")
    print(f"Dataset YAML created at: {yaml_path}")
    for split in ['train', 'val', 'test']:
        print(f"[{split.upper()}] Images: {stats[split]['images']} | Bounding Boxes: {stats[split]['boxes']}")
        for c in CLASSES:
            print(f"   - {c}: {stats[split]['per_class'][c]}")

if __name__ == '__main__':
    prepare_dataset()
