import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PATCH_DIR = os.path.join(PROCESSED_DIR, "patches")

os.makedirs(PATCH_DIR, exist_ok=True)

CLASS_MAP = {
    'surface_injure': 'surface_injure',
    'craze': 'craze',
    'hide_craze': 'hide_craze',
    'crack': 'crack',
    'corrosion': 'corrosion',
    'thunderstrike': 'thunderstrike'
}

BASE_SEVERITY = {
    'craze': 1,
    'surface_injure': 2,
    'corrosion': 2,
    'hide_craze': 3,
    'crack': 4,
    'thunderstrike': 5
}

def parse_annotations():
    img_dir = os.path.join(RAW_DIR, "images")
    ann_dir = os.path.join(RAW_DIR, "annotations")
    
    records = []
    patch_counter = 0

    xml_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.xml')])
    print(f"Processing {len(xml_files)} annotation files...")

    for xml_file in xml_files:
        base_id = os.path.splitext(xml_file)[0]
        img_file = base_id + ".jpg"
        img_path = os.path.join(img_dir, img_file)
        if not os.path.exists(img_path):
            # try png
            img_file = base_id + ".png"
            img_path = os.path.join(img_dir, img_file)
            if not os.path.exists(img_path):
                continue

        tree = ET.parse(os.path.join(ann_dir, xml_file))
        root = tree.getroot()

        size_elem = root.find('size')
        if size_elem is not None:
            w = int(size_elem.find('width').text)
            h = int(size_elem.find('height').text)
        else:
            w, h = 1024, 1024

        # Lazy open image only if objects exist
        objs = root.findall('object')
        if not objs:
            continue

        try:
            im = Image.open(img_path).convert('RGB')
        except Exception as e:
            continue

        for idx, obj in enumerate(objs):
            raw_cls = obj.find('name').text.strip().lower()
            if raw_cls not in CLASS_MAP:
                continue
            cls_name = CLASS_MAP[raw_cls]

            bndbox = obj.find('bndbox')
            xmin = max(0, int(float(bndbox.find('xmin').text)))
            ymin = max(0, int(float(bndbox.find('ymin').text)))
            xmax = min(w, int(float(bndbox.find('xmax').text)))
            ymax = min(h, int(float(bndbox.find('ymax').text)))

            bw = max(1, xmax - xmin)
            bh = max(1, ymax - ymin)
            area_px = bw * bh
            area_pct = round((area_px / (w * h)) * 100, 4)
            aspect_ratio = round(bh / bw, 3)

            # Spanwise location proxy r/R:
            # Turbine blade span in aerial inspection is primarily oriented along the vertical or major axis.
            # Normalizing the centroid along the blade length creates a continuous proxy r/R in [0.20, 0.98]
            yc = (ymin + ymax) / 2.0
            xc = (xmin + xmax) / 2.0
            # Normalized position with natural variance
            r_R = round(float(np.clip(0.20 + 0.78 * (yc / float(h)), 0.15, 0.98)), 3)

            # Image-derived severity proxy (1 to 5)
            base_sev = BASE_SEVERITY.get(cls_name, 2)
            # Area-based severity escalation
            if area_pct > 5.0:
                sev = min(5, base_sev + 1)
            elif area_pct < 0.5:
                sev = max(1, base_sev - 1)
            else:
                sev = base_sev

            # Crop patch with 10% padding
            pad_x = int(bw * 0.10)
            pad_y = int(bh * 0.10)
            crop_xmin = max(0, xmin - pad_x)
            crop_ymin = max(0, ymin - pad_y)
            crop_xmax = min(w, xmax + pad_x)
            crop_ymax = min(h, ymax + pad_y)

            patch = im.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax)).resize((224, 224), Image.Resampling.BILINEAR)
            patch_id = f"patch_{patch_counter:05d}"
            cls_dir = os.path.join(PATCH_DIR, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            patch_filename = f"{patch_id}.jpg"
            patch_path = os.path.join(cls_dir, patch_filename)
            rel_patch_path = os.path.join("patches", cls_name, patch_filename)
            patch.save(patch_path, quality=90)

            records.append({
                'patch_id': patch_id,
                'image_id': base_id,
                'defect_class': cls_name,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'bbox_width': bw,
                'bbox_height': bh,
                'area_pct': area_pct,
                'aspect_ratio': aspect_ratio,
                'r_R': r_R,
                'severity_proxy': sev,
                'patch_path': rel_patch_path,
                'abs_patch_path': patch_path
            })
            patch_counter += 1

    df = pd.DataFrame(records)
    csv_path = os.path.join(PROCESSED_DIR, "defect_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"Extracted {len(df)} defect patches across {len(CLASS_MAP)} classes.")
    print("Class breakdown:\n", df['defect_class'].value_counts())

    # Leakage-Free Image-Grouped Stratified Train/Val/Test Split (70/15/15)
    # Patches from the same physical photo/blade are grouped strictly within a single split.
    priority = {'thunderstrike': 0, 'crack': 1, 'corrosion': 2, 'craze': 3, 'hide_craze': 4, 'surface_injure': 5}
    img_classes = df.groupby('image_id')['defect_class'].apply(
        lambda s: sorted(s, key=lambda c: priority.get(c, 99))[0]
    ).reset_index()

    img_tr, img_te = train_test_split(img_classes, test_size=0.30, random_state=42, stratify=img_classes['defect_class'])
    img_val, img_te = train_test_split(img_te, test_size=0.50, random_state=42, stratify=img_te['defect_class'])

    tr_ids = set(img_tr['image_id'])
    val_ids = set(img_val['image_id'])
    te_ids = set(img_te['image_id'])

    train_df = df[df['image_id'].isin(tr_ids)].copy()
    val_df = df[df['image_id'].isin(val_ids)].copy()
    test_df = df[df['image_id'].isin(te_ids)].copy()

    train_df.to_csv(os.path.join(PROCESSED_DIR, "train_patches.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val_patches.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test_patches.csv"), index=False)

    print(f"Leakage-Free Grouped Split: Train={len(train_df)} ({len(tr_ids)} imgs), Val={len(val_df)} ({len(val_ids)} imgs), Test={len(test_df)} ({len(te_ids)} imgs)")

if __name__ == "__main__":
    parse_annotations()
