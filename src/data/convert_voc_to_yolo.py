"""
AeroLoss AI — Reproducible Pascal VOC XML to YOLOv8 Dataset Converter & Leakage-Free Splitter
Script: src.data.convert_voc_to_yolo

Key Features:
1. Audits and parses 1,065 Pascal VOC XML annotations into normalized YOLO txt format (cls x_c y_c w h).
2. Uses group-aware splitting to strictly prevent duplicate and near-duplicate leakage across train, val, and test splits.
3. Maps classes canonically:
     0: corrosion
     1: crack
     2: craze
     3: hide_craze
     4: surface_injure
     5: thunderstrike
4. Preserves all raw images and XML annotations unchanged.
5. Emits a verifiable split manifest: data/manifests/yolo_split_manifest.csv
6. Emits portable YOLO dataset configuration: configs/yolo_blade_damage.yaml
"""

import os
import sys
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_IMG_DIR = PROJECT_ROOT / "data" / "raw" / "images"
RAW_ANN_DIR = PROJECT_ROOT / "data" / "raw" / "annotations"
DUP_GROUPS_CSV = PROJECT_ROOT / "reports" / "model1_dataset_preparation" / "model1_duplicate_groups.csv"

OUTPUT_YOLO_DIR = PROJECT_ROOT / "data" / "yolo_dataset"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Canonical alphabetical class mapping
CLASS_NAMES = [
    "corrosion",
    "crack",
    "craze",
    "hide_craze",
    "surface_injure",
    "thunderstrike"
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def convert_bbox_to_yolo(img_w: float, img_h: float, xmin: float, ymin: float, xmax: float, ymax: float) -> Tuple[float, float, float, float]:
    """Convert Pascal VOC pixel bounding box to normalized YOLO format (x_center, y_center, width, height)."""
    # Clamp coordinates to image boundaries
    xmin = max(0.0, min(float(xmin), float(img_w)))
    ymin = max(0.0, min(float(ymin), float(img_h)))
    xmax = max(0.0, min(float(xmax), float(img_w)))
    ymax = max(0.0, min(float(ymax), float(img_h)))

    box_w = xmax - xmin
    box_h = ymax - ymin

    if box_w <= 0 or box_h <= 0:
        return (0.0, 0.0, 0.0, 0.0)

    x_center = (xmin + xmax) / (2.0 * img_w)
    y_center = (ymin + ymax) / (2.0 * img_h)
    w_norm = box_w / img_w
    h_norm = box_h / img_h

    return (
        round(float(x_center), 6),
        round(float(y_center), 6),
        round(float(w_norm), 6),
        round(float(h_norm), 6)
    )


def build_group_aware_splits(
    image_filenames: List[str],
    dup_df: pd.DataFrame,
    seed: int = 42,
    target_train_ratio: float = 0.70,
    target_val_ratio: float = 0.15,
    target_test_ratio: float = 0.15
) -> Dict[str, str]:
    """
    Assign images to train, val, or test splits such that NO duplicate or
    near-duplicate group is split across partition boundaries.
    """
    # Filter out quarantined synthetic images that are not in primary dataset
    valid_filenames = set(image_filenames)
    primary_dups = dup_df[dup_df["filename"].isin(valid_filenames)]

    img_to_group: Dict[str, str] = {}
    group_to_imgs: Dict[str, List[str]] = defaultdict(list)

    for _, row in primary_dups.iterrows():
        fn = str(row["filename"])
        gid = str(row["group_id"])
        img_to_group[fn] = gid
        group_to_imgs[gid].append(fn)

    # Any image not in a duplicate group becomes its own singleton group
    for fn in image_filenames:
        if fn not in img_to_group:
            gid = f"SINGLE_{fn}"
            img_to_group[fn] = gid
            group_to_imgs[gid].append(fn)

    unique_groups = sorted(list(group_to_imgs.keys()))
    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    # Sort groups so larger groups are allocated first to balance splits
    unique_groups.sort(key=lambda g: len(group_to_imgs[g]), reverse=True)

    n_total_images = len(image_filenames)
    target_val = int(round(n_total_images * target_val_ratio))
    target_test = int(round(n_total_images * target_test_ratio))

    train_imgs: List[str] = []
    val_imgs: List[str] = []
    test_imgs: List[str] = []

    for gid in unique_groups:
        cluster = group_to_imgs[gid]
        c_size = len(cluster)

        # Fill validation first until target is reached
        if len(val_imgs) + c_size <= target_val:
            val_imgs.extend(cluster)
        # Fill test next until target is reached
        elif len(test_imgs) + c_size <= target_test:
            test_imgs.extend(cluster)
        # Remaining groups go to train
        else:
            train_imgs.extend(cluster)

    split_assignment: Dict[str, str] = {}
    for fn in train_imgs:
        split_assignment[fn] = "train"
    for fn in val_imgs:
        split_assignment[fn] = "val"
    for fn in test_imgs:
        split_assignment[fn] = "test"

    return split_assignment


def run_conversion():
    print("=== AeroLoss AI: Pascal VOC XML to YOLOv8 Dataset Conversion ===")
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Discover raw images and annotations
    img_files = sorted(list(RAW_IMG_DIR.glob("*.jpg")))
    xml_files = sorted(list(RAW_ANN_DIR.glob("*.xml")))
    print(f"Discovered {len(img_files)} images and {len(xml_files)} XML annotations.")
    assert len(img_files) == 1065, f"Expected 1065 images, found {len(img_files)}"
    assert len(xml_files) == 1065, f"Expected 1065 XML files, found {len(xml_files)}"

    image_filenames = [f.name for f in img_files]

    # 2. Load duplicate groups for leakage-free partition
    if DUP_GROUPS_CSV.exists():
        dup_df = pd.read_csv(DUP_GROUPS_CSV)
        print(f"Loaded duplicate groups manifest ({len(dup_df)} total group entries).")
    else:
        print("Warning: Duplicate groups CSV not found; falling back to singleton groups.")
        dup_df = pd.DataFrame(columns=["group_id", "filename", "group_type"])

    # 3. Compute group-aware splits
    split_map = build_group_aware_splits(image_filenames, dup_df, seed=42)

    train_count = sum(1 for s in split_map.values() if s == "train")
    val_count = sum(1 for s in split_map.values() if s == "val")
    test_count = sum(1 for s in split_map.values() if s == "test")
    print(f"\nGroup-aware split distribution (N={len(split_map)}):")
    print(f"  Train: {train_count} images ({train_count/len(split_map)*100:.1f}%)")
    print(f"  Val:   {val_count} images ({val_count/len(split_map)*100:.1f}%)")
    print(f"  Test:  {test_count} images ({test_count/len(split_map)*100:.1f}%)")

    # 4. Clean and recreate directory layout for fresh leakage-free export
    for folder in ["images", "labels"]:
        target_dir = OUTPUT_YOLO_DIR / folder
        if target_dir.exists():
            shutil.rmtree(target_dir)
        for split in ["train", "val", "test"]:
            (target_dir / split).mkdir(parents=True, exist_ok=True)

    manifest_records = []
    split_box_counts = {s: Counter() for s in ["train", "val", "test"]}
    total_converted_boxes = 0

    # 5. Process each image and annotation
    for img_path in img_files:
        stem = img_path.stem
        fn = img_path.name
        xml_path = RAW_ANN_DIR / f"{stem}.xml"
        split = split_map[fn]

        dest_img_path = OUTPUT_YOLO_DIR / "images" / split / fn
        dest_lbl_path = OUTPUT_YOLO_DIR / "labels" / split / f"{stem}.txt"

        # Copy image file
        shutil.copy2(img_path, dest_img_path)

        # Parse VOC XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size_elem = root.find("size")
        img_w = float(size_elem.find("width").text) if size_elem is not None and size_elem.find("width") is not None else 1024.0
        img_h = float(size_elem.find("height").text) if size_elem is not None and size_elem.find("height") is not None else 1024.0

        yolo_lines = []
        image_boxes = 0

        for obj in root.findall("object"):
            raw_cls = obj.find("name").text.strip().lower()
            if raw_cls not in CLASS_TO_IDX:
                print(f"Warning: Unexpected class '{raw_cls}' in {xml_path.name}")
                continue

            cls_idx = CLASS_TO_IDX[raw_cls]
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            xc, yc, w_norm, h_norm = convert_bbox_to_yolo(img_w, img_h, xmin, ymin, xmax, ymax)
            if w_norm > 0 and h_norm > 0:
                yolo_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")
                split_box_counts[split][raw_cls] += 1
                image_boxes += 1
                total_converted_boxes += 1

        with open(dest_lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

        manifest_records.append({
            "image_id": stem,
            "filename": fn,
            "split": split,
            "relative_image_path": f"data/yolo_dataset/images/{split}/{fn}",
            "relative_label_path": f"data/yolo_dataset/labels/{split}/{stem}.txt",
            "box_count": image_boxes,
            "classes": list(set(obj.find("name").text.strip() for obj in root.findall("object")))
        })

    # 6. Save split manifest
    manifest_df = pd.DataFrame(manifest_records)
    manifest_csv_path = MANIFESTS_DIR / "yolo_split_manifest.csv"
    manifest_df.to_csv(manifest_csv_path, index=False)
    print(f"\nSaved split manifest to: {manifest_csv_path}")

    # 7. Write documented, portable YOLO dataset YAML
    yaml_content = f"""# AeroLoss AI — YOLOv8 Wind Turbine Blade Damage Detection Dataset Configuration
# Generated by: src/data/convert_voc_to_yolo.py (Reproducible Split Seed: 42)
# Leakage Prevention: Zero duplicate or near-duplicate leakage across splits.

# Dataset root directory (relative to project root or overridden in Colab)
path: data/yolo_dataset

# Subdirectory splits
train: images/train
val: images/val
test: images/test

# Canonical Class Indices and Names (N=6)
names:
  0: corrosion
  1: crack
  2: craze
  3: hide_craze
  4: surface_injure
  5: thunderstrike
"""
    yaml_config_path = CONFIGS_DIR / "yolo_blade_damage.yaml"
    with open(yaml_config_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # Also keep a copy at data/yolo_dataset/dataset.yaml for local Ultralytics defaults
    with open(OUTPUT_YOLO_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Saved dataset configurations to:")
    print(f"  - {yaml_config_path}")
    print(f"  - {OUTPUT_YOLO_DIR / 'dataset.yaml'}")

    print(f"\nTotal bounding boxes converted: {total_converted_boxes}")
    for split in ["train", "val", "test"]:
        print(f"\n[{split.upper()}] Box count: {sum(split_box_counts[split].values())}")
        for cls_name in CLASS_NAMES:
            cnt = split_box_counts[split][cls_name]
            print(f"  - {cls_name:<16}: {cnt:>4}")

    print("\nConversion and leakage-free partition complete successfully.")


if __name__ == "__main__":
    run_conversion()
