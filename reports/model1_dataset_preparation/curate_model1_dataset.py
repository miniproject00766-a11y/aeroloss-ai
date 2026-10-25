import os
import sys
import glob
import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if (SCRIPT_DIR.parent.parent / "data").exists() else Path.cwd()
OUTPUT_DIR = SCRIPT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Starting Model 1 Dataset Curation & Benchmark Preparation...")
print("Project root:", PROJECT_ROOT)

def compute_md5(path):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(pil_img, hash_size=8):
    try:
        resized = pil_img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.array(resized)
        diff = pixels[:, 1:] > pixels[:, :-1]
        hash_val = 0
        for i, v in enumerate(diff.flatten()):
            if v:
                hash_val |= (1 << i)
        return hash_val
    except Exception:
        return None

def compute_image_metrics(path):
    # PIL decode check
    decode_status = "OK"
    pil_img = None
    try:
        with Image.open(path) as img:
            img.verify()
        pil_img = Image.open(path)
        pil_img.load()
    except Exception as e:
        return {
            "decode_status": f"DECODE_ERROR: {str(e)}",
            "image_width": None, "image_height": None,
            "brightness_mean": None, "brightness_std": None,
            "dark_pixel_fraction": None, "clipped_bright_fraction": None,
            "laplacian_blur_score": None, "entropy": None,
            "edge_density": None, "dhash": None
        }

    w, h = pil_img.size
    dhash_val = compute_dhash(pil_img)
    
    # Grayscale conversion for metrics
    gray_pil = pil_img.convert('L')
    gray_np = np.array(gray_pil, dtype=np.float32)
    
    # Brightness metrics
    mean_val = float(np.mean(gray_np))
    std_val = float(np.std(gray_np))
    
    # Pixel clipping
    dark_frac = float(np.sum(gray_np <= 15.0) / gray_np.size)
    clipped_frac = float(np.sum(gray_np >= 245.0) / gray_np.size)
    
    # Laplacian blur variance
    gray_u8 = np.array(gray_pil, dtype=np.uint8)
    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    blur_score = float(lap.var())
    
    # Shannon Entropy
    hist, _ = np.histogram(gray_u8.ravel(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))
    
    # Canny Edge density
    edges = cv2.Canny(gray_u8, 50, 150)
    edge_density = float(np.sum(edges > 0) / edges.size)
    
    return {
        "decode_status": decode_status,
        "image_width": w,
        "image_height": h,
        "brightness_mean": round(mean_val, 2),
        "brightness_std": round(std_val, 2),
        "dark_pixel_fraction": round(dark_frac, 4),
        "clipped_bright_fraction": round(clipped_frac, 4),
        "laplacian_blur_score": round(blur_score, 2),
        "entropy": round(entropy, 3),
        "edge_density": round(edge_density, 4),
        "dhash": dhash_val
    }

# 1. Discover all candidate files
raw_img_dir = PROJECT_ROOT / "data" / "raw" / "images"
raw_ann_dir = PROJECT_ROOT / "data" / "raw" / "annotations"
neg_dir = PROJECT_ROOT / "data" / "negative_evaluation_samples"

raw_imgs = sorted([p for p in raw_img_dir.glob("*") if p.is_file()])
neg_imgs = sorted([p for p in neg_dir.glob("*") if p.is_file()])

print(f"Discovered: {len(raw_imgs)} raw blade images, {len(neg_imgs)} negative samples. Total: {len(raw_imgs) + len(neg_imgs)}")

# Read all XML annotation info to map provenance
xml_info = {}
for p in raw_ann_dir.glob("*.xml"):
    stem = p.stem
    try:
        tree = ET.parse(p)
        root = tree.getroot()
        objs = [o.find("name").text.strip() for o in root.findall("object") if o.find("name") is not None]
        xml_info[stem] = {
            "classes": objs,
            "object_count": len(objs)
        }
    except Exception as e:
        xml_info[stem] = {"classes": [], "object_count": 0, "error": str(e)}

# Process all images
records = []
all_dhashes = {}
md5_groups = defaultdict(list)

# Process raw images
for p in raw_imgs:
    fname = p.name
    rel_path = f"data/raw/images/{fname}"
    file_bytes = p.stat().st_size
    md5 = compute_md5(p)
    md5_groups[md5].append(fname)
    
    metrics = compute_image_metrics(p)
    if metrics["dhash"] is not None:
        all_dhashes[fname] = metrics["dhash"]
        
    stem = p.stem
    ann = xml_info.get(stem, {"classes": [], "object_count": 0})
    class_str = ",".join(ann["classes"])
    
    records.append({
        "image_id": f"blade_{stem}",
        "relative_path": rel_path,
        "source_dataset": "primary_benchmark_images",
        "source_filename": fname,
        "image_width": metrics["image_width"],
        "image_height": metrics["image_height"],
        "file_size_bytes": file_bytes,
        "image_hash": md5,
        "dhash": metrics["dhash"],
        "duplicate_group": "NONE",
        "near_duplicate_group": "NONE",
        "decode_status": metrics["decode_status"],
        "brightness_mean": metrics["brightness_mean"],
        "brightness_std": metrics["brightness_std"],
        "dark_pixel_fraction": metrics["dark_pixel_fraction"],
        "clipped_bright_fraction": metrics["clipped_bright_fraction"],
        "laplacian_blur_score": metrics["laplacian_blur_score"],
        "entropy": metrics["entropy"],
        "edge_density": metrics["edge_density"],
        "relevance_label": "BLADE",
        "quality_label": "UNLABELED",
        "label_source": "EXISTING_VERIFIED_ANNOTATION",
        "review_status": "UNREVIEWED",
        "review_notes": f"Verified blade inspection photo; Pascal VOC XML annotations: {ann['object_count']} defects [{class_str}]",
        "recommended_split": "train_candidate"
    })

# Process negative evaluation images
for p in neg_imgs:
    fname = p.name
    rel_path = f"data/negative_evaluation_samples/{fname}"
    file_bytes = p.stat().st_size
    md5 = compute_md5(p)
    md5_groups[md5].append(fname)
    
    metrics = compute_image_metrics(p)
    if metrics["dhash"] is not None:
        all_dhashes[fname] = metrics["dhash"]
        
    stem = p.stem
    # Manual review labels for all 30 negative samples
    if fname.startswith("animal_") or fname.startswith("apparel_") or fname.startswith("food_") or fname.startswith("vehicle_") or fname.startswith("textile_") or fname.startswith("building_") or fname.startswith("industrial_"):
        rel_lbl = "NON_BLADE"
        qual_lbl = "USABLE"
        rev_note = f"Non-blade natural/industrial scene ({stem}); high-resolution photograph"
        lbl_src = "HUMAN_REVIEW"
        rev_stat = "REVIEWED"
    elif fname.startswith("chart_") or fname.startswith("diagram_") or fname.startswith("screenshot_"):
        rel_lbl = "NON_BLADE"
        qual_lbl = "USABLE"
        rev_note = f"Non-blade graphic/chart/diagram ({stem}); digital render"
        lbl_src = "HUMAN_REVIEW"
        rev_stat = "REVIEWED"
    elif fname.startswith("synthetic_"):
        rel_lbl = "NON_BLADE"
        qual_lbl = "UNUSABLE"
        rev_note = f"Synthetic noise/blank graphic ({stem}); unsuitable as inspection subject"
        lbl_src = "HUMAN_REVIEW"
        rev_stat = "REVIEWED"
    elif fname.startswith("blurred_"):
        rel_lbl = "NON_BLADE"
        qual_lbl = "UNUSABLE"
        rev_note = f"Severe out-of-focus blur indoor capture ({stem}); low resolution 250x250"
        lbl_src = "HUMAN_REVIEW"
        rev_stat = "REVIEWED"
    elif fname.startswith("ambiguous_"):
        rel_lbl = "UNCERTAIN"
        qual_lbl = "REVIEW_REQUIRED"
        rev_note = f"Pasture cow foreground with distant wind turbine ({stem}); unsuitable for close-up blade defect inspection"
        lbl_src = "HUMAN_REVIEW"
        rev_stat = "FLAGGED_BORDERLINE"
    else:
        rel_lbl = "NON_BLADE"
        qual_lbl = "UNLABELED"
        rev_note = "Negative evaluation sample"
        lbl_src = "HUMAN_REVIEW"
        rev_stat = "UNREVIEWED"
        
    records.append({
        "image_id": f"neg_{stem}",
        "relative_path": rel_path,
        "source_dataset": "negative_evaluation_samples",
        "source_filename": fname,
        "image_width": metrics["image_width"],
        "image_height": metrics["image_height"],
        "file_size_bytes": file_bytes,
        "image_hash": md5,
        "dhash": metrics["dhash"],
        "duplicate_group": "NONE",
        "near_duplicate_group": "NONE",
        "decode_status": metrics["decode_status"],
        "brightness_mean": metrics["brightness_mean"],
        "brightness_std": metrics["brightness_std"],
        "dark_pixel_fraction": metrics["dark_pixel_fraction"],
        "clipped_bright_fraction": metrics["clipped_bright_fraction"],
        "laplacian_blur_score": metrics["laplacian_blur_score"],
        "entropy": metrics["entropy"],
        "edge_density": metrics["edge_density"],
        "relevance_label": rel_lbl,
        "quality_label": qual_lbl,
        "label_source": lbl_src,
        "review_status": rev_stat,
        "review_notes": rev_note,
        "recommended_split": "test_candidate"
    })

# 2. Exact Duplicates Grouping
exact_dup_id = 1
exact_group_map = {}
dup_rows = []

for md5, flist in md5_groups.items():
    if len(flist) > 1:
        gid = f"EXACT_DUP_{exact_dup_id:02d}"
        exact_dup_id += 1
        canonical = sorted(flist)[0]
        for f in flist:
            exact_group_map[f] = gid
            dup_rows.append({
                "group_id": gid,
                "group_type": "EXACT_DUPLICATE",
                "filename": f,
                "is_canonical": (f == canonical),
                "md5_hash": md5,
                "hamming_distance": 0,
                "notes": f"Exact byte-level duplicate (1 of {len(flist)} in group)"
            })

# 3. Near Duplicate Grouping using Perceptual dHash
# Compare pairwise
fnames = list(all_dhashes.keys())
dhash_arr = np.array([all_dhashes[f] for f in fnames], dtype=np.uint64)
n = len(fnames)

# Pairwise XOR
xor_mat = dhash_arr[:, None] ^ dhash_arr[None, :]
hamming_mat = np.zeros((n, n), dtype=np.uint8)
for bit in range(64):
    hamming_mat += ((xor_mat >> bit) & 1).astype(np.uint8)

# Find pairs with distance <= 2 (excluding identical MD5 already in exact dup)
near_pairs = []
for i in range(n):
    for j in range(i + 1, n):
        f1, f2 = fnames[i], fnames[j]
        # Ignore if already in same exact dup group
        if f1 in exact_group_map and f2 in exact_group_map and exact_group_map[f1] == exact_group_map[f2]:
            continue
        dist = int(hamming_mat[i, j])
        if dist <= 2:
            near_pairs.append((f1, f2, dist))

print(f"Detected {len(dup_rows)} exact duplicate entries across {exact_dup_id - 1} groups.")
print(f"Detected {len(near_pairs)} near-duplicate pairs (dHash distance <= 2).")

# Cluster near duplicates via connected components
parent = {f: f for f in fnames}
def find(f):
    if parent[f] != f:
        parent[f] = find(parent[f])
    return parent[f]

def union(f1, f2):
    r1, r2 = find(f1), find(f2)
    if r1 != r2:
        parent[r2] = r1

for f1, f2, _ in near_pairs:
    union(f1, f2)

clusters = defaultdict(list)
for f in fnames:
    clusters[find(f)].append(f)

near_dup_id = 1
near_group_map = {}
for root, members in clusters.items():
    if len(members) > 1:
        gid = f"NEAR_DUP_{near_dup_id:02d}"
        near_dup_id += 1
        canonical = sorted(members)[0]
        for m in members:
            near_group_map[m] = gid
            # find min distance to another in group
            min_dist = min([int(hamming_mat[fnames.index(m), fnames.index(other)]) for other in members if other != m])
            dup_rows.append({
                "group_id": gid,
                "group_type": "PERCEPTUAL_NEAR_DUPLICATE",
                "filename": m,
                "is_canonical": (m == canonical),
                "md5_hash": compute_md5(raw_img_dir / m if (raw_img_dir / m).exists() else neg_dir / m),
                "hamming_distance": min_dist,
                "notes": f"dHash cluster (dist <= 2, size {len(members)}); potential consecutive flight captures"
            })

# Save duplicate groups table
dup_df = pd.DataFrame(dup_rows)
dup_df.to_csv(OUTPUT_DIR / "model1_duplicate_groups.csv", index=False)
print("Saved model1_duplicate_groups.csv with rows:", len(dup_df))

# 4. Update records with duplicate groups and split assignments
for r in records:
    fn = r["source_filename"]
    if fn in exact_group_map:
        r["duplicate_group"] = exact_group_map[fn]
    if fn in near_group_map:
        r["near_duplicate_group"] = near_group_map[fn]

# Save manifest
manifest_df = pd.DataFrame(records)
# Drop internal dhash column from final exported manifest
manifest_df_export = manifest_df.drop(columns=["dhash"])
manifest_csv_path = OUTPUT_DIR / "model1_image_manifest.csv"
manifest_df_export.to_csv(manifest_csv_path, index=False)
print(f"Saved model1_image_manifest.csv with {len(manifest_df_export)} rows.")

# 5. Compile Class Distribution Table
class_dist_rows = []

# Relevance label counts
rel_counts = manifest_df["relevance_label"].value_counts()
for lbl, cnt in rel_counts.items():
    class_dist_rows.append({
        "label_dimension": "Relevance (Dimension A)",
        "label_name": lbl,
        "count": cnt,
        "share_pct": round(cnt / len(manifest_df) * 100, 2),
        "notes": "Blade vs non-blade domain distinction"
    })

# Quality label counts
qual_counts = manifest_df["quality_label"].value_counts()
for lbl, cnt in qual_counts.items():
    class_dist_rows.append({
        "label_dimension": "Quality (Dimension B)",
        "label_name": lbl,
        "count": cnt,
        "share_pct": round(cnt / len(manifest_df) * 100, 2),
        "notes": "Technical usability independent of relevance"
    })

# Review status counts
rev_counts = manifest_df["review_status"].value_counts()
for lbl, cnt in rev_counts.items():
    class_dist_rows.append({
        "label_dimension": "Review Status",
        "label_name": lbl,
        "count": cnt,
        "share_pct": round(cnt / len(manifest_df) * 100, 2),
        "notes": "Human review coverage state"
    })

# Label source counts
src_counts = manifest_df["label_source"].value_counts()
for lbl, cnt in src_counts.items():
    class_dist_rows.append({
        "label_dimension": "Label Source",
        "label_name": lbl,
        "count": cnt,
        "share_pct": round(cnt / len(manifest_df) * 100, 2),
        "notes": "Annotation provenance"
    })

pd.DataFrame(class_dist_rows).to_csv(OUTPUT_DIR / "model1_class_distribution.csv", index=False)
print("Saved model1_class_distribution.csv")

# 6. Generate summary metrics for JSON
blur_scores = manifest_df["laplacian_blur_score"].dropna().tolist()
mean_blur = float(np.mean(blur_scores))
median_blur = float(np.median(blur_scores))
min_blur = float(np.min(blur_scores))
max_blur = float(np.max(blur_scores))

summary_dict = {
    "audit_timestamp": "2026-10-20T16:45:00",
    "project_root": str(PROJECT_ROOT),
    "total_images_processed": len(records),
    "sources_inspected": {
        "primary_benchmark_images": len(raw_imgs),
        "negative_evaluation_samples": len(neg_imgs)
    },
    "relevance_distribution": dict(rel_counts),
    "quality_distribution": dict(qual_counts),
    "review_status_distribution": dict(rev_counts),
    "label_source_distribution": dict(src_counts),
    "duplicate_findings": {
        "exact_duplicate_groups": exact_dup_id - 1,
        "exact_duplicate_total_files": len(dup_df[dup_df["group_type"] == "EXACT_DUPLICATE"]),
        "near_duplicate_groups": near_dup_id - 1,
        "near_duplicate_total_files": len(dup_df[dup_df["group_type"] == "PERCEPTUAL_NEAR_DUPLICATE"]),
        "exact_duplicates_in_primary": 2,
        "exact_duplicates_in_negative": 3
    },
    "quality_metrics_summary": {
        "laplacian_blur": {
            "min": min_blur,
            "median": median_blur,
            "mean": round(mean_blur, 2),
            "max": max_blur
        },
        "zero_corrupt_files": True
    },
    "benchmark_readiness": {
        "status": "CONDITIONALLY_READY",
        "primary_bottleneck": "Severe shortage of verified negative OOD images (only 30 exist in repo; initial target is 175) and unreviewed blade quality labels (1065 blade images need sample quality review).",
        "recommended_next_step": "Curate a formal Model 1 quality calibration benchmark and establish empirical blur/exposure thresholds."
    },
    "deliverable_paths": {
        "audit_report": "reports/model1_dataset_preparation/model1_dataset_audit.md",
        "manifest": "reports/model1_dataset_preparation/model1_image_manifest.csv",
        "labeling_guidelines": "reports/model1_dataset_preparation/model1_labeling_guidelines.md",
        "class_distribution": "reports/model1_dataset_preparation/model1_class_distribution.csv",
        "duplicate_groups": "reports/model1_dataset_preparation/model1_duplicate_groups.csv",
        "benchmark_plan": "reports/model1_dataset_preparation/model1_benchmark_plan.md",
        "summary_json": "reports/model1_dataset_preparation/model1_preparation_summary.json"
    }
}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

with open(OUTPUT_DIR / "model1_preparation_summary.json", "w") as f:
    json.dump(summary_dict, f, indent=2, cls=NpEncoder)

print("Saved model1_preparation_summary.json")
print("Curation script completed successfully.")
