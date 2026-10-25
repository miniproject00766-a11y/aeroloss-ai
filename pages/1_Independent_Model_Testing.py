"""
AeroLoss AI — Independent Model Testing Interface
Target: Dedicated interface for testing Model 1 and Model 2 independently.
"""

import os
import sys
import tempfile
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2

# Project root resilient resolution
current_dir = Path(__file__).resolve().parent
if current_dir.name == "pages":
    PROJECT_ROOT = current_dir.parent
else:
    PROJECT_ROOT = current_dir

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model1.image_quality_validator import ImageQualityValidator, validate_inspection_image
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="Independent Model Testing — AeroLoss AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .page-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .page-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .badge-usable {
        background-color: #059669;
        color: white;
        padding: 6px 16px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-review {
        background-color: #d97706;
        color: white;
        padding: 6px 16px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-unusable {
        background-color: #dc2626;
        color: white;
        padding: 6px 16px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-blade {
        background-color: #2563eb;
        color: white;
        padding: 6px 16px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-uncertain {
        background-color: #d97706;
        color: white;
        padding: 6px 16px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-nonblade {
        background-color: #dc2626;
        color: white;
        padding: 6px 16px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title & Description
st.markdown('<div class="page-title">🧪 AeroLoss AI — Independent Model Testing</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Separate, isolated upload and evaluation testing for <b>Model 1</b> '
    '(Image Quality & Blade-Relevance Guardrail) and <b>Model 2</b> (Trained YOLOv8 Defect Detector). '
    'Each model operates strictly independently without pipeline coupling.</div>',
    unsafe_allow_html=True
)

# Sidebar Info
st.sidebar.title("Model Artifacts & Specs")
st.sidebar.markdown("""
**Model 1 (Guardrail):**
- Engine: Deterministic Computer Vision
- Outputs: Image Quality & Blade Relevance
- Thresholds: `configs/model1_quality_thresholds.yaml`

**Model 2 (Defect Detector):**
- Model: YOLOv8 Small (`yolov8s.pt`)
- Checkpoint: `models/yolov8s_blade_damage_best.pt`
- Classes (6): `corrosion`, `crack`, `craze`, `hide_craze`, `surface_injure`, `thunderstrike`
""")

# Model Checkpoint Verification
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "yolov8s_blade_damage_best.pt"
CANONICAL_CLASSES = ["corrosion", "crack", "craze", "hide_craze", "surface_injure", "thunderstrike"]

@st.cache_resource
def load_yolo_detector(weights_path: Path):
    if not weights_path.exists():
        return None, f"Checkpoint file missing at: {weights_path}"
    try:
        model = YOLO(str(weights_path))
        class_names = list(model.names.values())
        if class_names != CANONICAL_CLASSES:
            return None, f"Class mapping mismatch in checkpoint: {model.names}"
        return model, None
    except Exception as e:
        return None, f"Failed to load YOLO model: {e}"

# Tabs for separate model testing
tab_model1, tab_model2 = st.tabs([
    "🛡️ MODEL 1 TEST: Image Quality & Blade Relevance",
    "🎯 MODEL 2 TEST: YOLOv8 Blade Damage Detection"
])

# ==============================================================================
# TAB 1: MODEL 1 INDEPENDENT TESTING
# ==============================================================================
with tab_model1:
    st.subheader("Model 1 Guardrail Evaluation (Isolated)")
    st.caption("Evaluates technical usability and blade relevance using deterministic image processing. **Does NOT run YOLOv8.**")

    # Sample Quick Select
    st.markdown("**Quick Test Samples:**")
    sample_col1, sample_col2, sample_col3, _ = st.columns([1, 1, 1, 2])
    sample_img_path = None
    
    with sample_col1:
        if st.button("📄 Load Valid Blade (0.jpg)"):
            sample_img_path = PROJECT_ROOT / "data" / "raw" / "images" / "0.jpg"
    with sample_col2:
        if st.button("📊 Load Non-Blade (Chart)"):
            sample_img_path = PROJECT_ROOT / "data" / "negative_evaluation_samples" / "chart_confusion_matrix_03.png"
    with sample_col3:
        if st.button("🌫️ Load Noisy / Blur Sample"):
            sample_img_path = PROJECT_ROOT / "data" / "negative_evaluation_samples" / "synthetic_gaussian_noise.jpg"

    # Upload Widget
    uploaded_file_m1 = st.file_uploader(
        "Upload an image for Model 1 validation (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="uploader_model1"
    )

    image_to_process_m1 = None
    source_label_m1 = ""

    if uploaded_file_m1 is not None:
        try:
            image_to_process_m1 = Image.open(uploaded_file_m1)
            source_label_m1 = uploaded_file_m1.name
        except Exception as e:
            st.error(f"Failed to read uploaded image: {e}")
    elif sample_img_path and sample_img_path.exists():
        image_to_process_m1 = Image.open(sample_img_path)
        source_label_m1 = sample_img_path.name

    if image_to_process_m1 is not None:
        st.markdown("---")
        col_img, col_diag = st.columns([1.1, 1.4], gap="large")

        with col_img:
            st.markdown(f"**Source:** `{source_label_m1}`")
            st.image(image_to_process_m1, use_container_width=True, caption=f"Uploaded Image: {source_label_m1}")

        # Run Model 1 Validator
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            # Save in RGB mode
            image_to_process_m1.convert("RGB").save(tmp_path, format="JPEG")

        try:
            val_result = validate_inspection_image(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        with col_diag:
            st.markdown("### Guardrail Diagnostic Output")
            
            # Status Badges
            q_dec = val_result.get("quality_decision", "UNUSABLE")
            r_dec = val_result.get("relevance_decision", "NON_BLADE")
            req_review = val_result.get("requires_manual_review", False)

            q_badge_cls = "badge-usable" if q_dec == "USABLE" else ("badge-review" if q_dec == "REVIEW_REQUIRED" else "badge-unusable")
            r_badge_cls = "badge-blade" if r_dec == "BLADE" else ("badge-uncertain" if r_dec == "UNCERTAIN" else "badge-nonblade")

            badge_col1, badge_col2, badge_col3 = st.columns(3)
            with badge_col1:
                st.markdown("**Technical Quality:**")
                st.markdown(f'<div class="{q_badge_cls}">{q_dec}</div>', unsafe_allow_html=True)
            with badge_col2:
                st.markdown("**Blade Relevance:**")
                st.markdown(f'<div class="{r_badge_cls}">{r_dec}</div>', unsafe_allow_html=True)
            with badge_col3:
                st.markdown("**Supervisory Action:**")
                if req_review:
                    st.markdown('<div class="badge-review">MANUAL REVIEW</div>', unsafe_allow_html=True)
                elif q_dec == "USABLE" and r_dec == "BLADE":
                    st.markdown('<div class="badge-usable">AUTO-PASS</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-unusable">REJECTED</div>', unsafe_allow_html=True)

            if req_review:
                st.warning("⚠️ **Supervisory Routing:** This image requires human-in-the-loop review before downstream automated processing.")

            # Metrics Breakdown
            st.markdown("#### Technical Measurements")
            dims = val_result.get("dimensions", {})
            metrics = val_result.get("metrics", {})

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Resolution", f"{dims.get('width', 0)} × {dims.get('height', 0)}")
                st.metric("Aspect Ratio", f"{dims.get('aspect_ratio', 0.0):.2f}")
            with m_col2:
                blur_val = metrics.get('laplacian_blur_score', metrics.get('blur_score', 0.0))
                b_mean = metrics.get('brightness_mean', metrics.get('mean_brightness', 0.0))
                st.metric("Blur Score (Laplacian)", f"{blur_val:.2f}")
                st.metric("Mean Brightness", f"{b_mean:.1f} / 255")
            with m_col3:
                br_clip = metrics.get('clipped_bright_fraction', 0.0)
                dk_clip = metrics.get('dark_pixel_fraction', metrics.get('clipped_dark_fraction', 0.0))
                st.metric("Bright Clipping", f"{br_clip*100:.1f}%")
                st.metric("Dark Clipping", f"{dk_clip*100:.1f}%")

            # Diagnostic Reasons List
            st.markdown("#### Decision Triggers & Reasons:")
            for reason in val_result.get("reasons", []):
                st.markdown(f"- {reason}")

            st.caption("ℹ️ *Quality decisions are computed by deterministic computer-vision heuristics and have not been validated against human quality labels.*")


# ==============================================================================
# TAB 2: MODEL 2 INDEPENDENT TESTING
# ==============================================================================
with tab_model2:
    st.subheader("Model 2 YOLOv8 Defect Detection (Isolated)")
    st.caption("Runs the fine-tuned YOLOv8 damage detector directly on uploaded images. **Does NOT apply Model 1 guardrails.**")

    # Checkpoint Verification Status
    detector, load_error = load_yolo_detector(CHECKPOINT_PATH)
    
    if load_error or detector is None:
        st.error(f"❌ **Model Checkpoint Error:** {load_error}")
        st.info(f"Expected verified checkpoint at: `{CHECKPOINT_PATH}`")
    else:
        st.success(f"✅ **Verified Checkpoint Active:** `yolov8s_blade_damage_best.pt` ({CHECKPOINT_PATH.stat().st_size / (1024*1024):.2f} MB)")
        
        # Inference Settings
        set_col1, set_col2, _ = st.columns([1, 1, 2])
        with set_col1:
            conf_threshold = st.slider("Confidence Threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
        with set_col2:
            iou_threshold = st.slider("NMS IoU Threshold", min_value=0.10, max_value=0.90, value=0.45, step=0.05)

        # Quick Test Samples for Model 2
        st.markdown("**Quick Test Samples:**")
        s2_col1, s2_col2, s2_col3, _ = st.columns([1, 1, 1, 2])
        sample_img_path_m2 = None

        with s2_col1:
            if st.button("🔍 Primary Blade #1 (0.jpg)", key="btn_m2_0"):
                sample_img_path_m2 = PROJECT_ROOT / "data" / "raw" / "images" / "0.jpg"
        with s2_col2:
            if st.button("🔍 Primary Blade #2 (1.jpg)", key="btn_m2_1"):
                sample_img_path_m2 = PROJECT_ROOT / "data" / "raw" / "images" / "1.jpg"
        with s2_col3:
            if st.button("📊 Non-Blade Chart", key="btn_m2_chart"):
                sample_img_path_m2 = PROJECT_ROOT / "data" / "negative_evaluation_samples" / "chart_confusion_matrix_03.png"

        # Upload Widget for Model 2
        uploaded_file_m2 = st.file_uploader(
            "Upload an image for Model 2 defect detection (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="uploader_model2"
        )

        image_to_process_m2 = None
        source_label_m2 = ""

        if uploaded_file_m2 is not None:
            try:
                image_to_process_m2 = Image.open(uploaded_file_m2)
                source_label_m2 = uploaded_file_m2.name
            except Exception as e:
                st.error(f"Failed to read uploaded image: {e}")
        elif sample_img_path_m2 and sample_img_path_m2.exists():
            image_to_process_m2 = Image.open(sample_img_path_m2)
            source_label_m2 = sample_img_path_m2.name

        if image_to_process_m2 is not None:
            st.markdown("---")
            
            # Execute YOLOv8 prediction
            np_img = np.array(image_to_process_m2.convert("RGB"))
            
            with st.spinner("Running YOLOv8 blade damage detection..."):
                results = detector.predict(np_img, conf=conf_threshold, iou=iou_threshold, verbose=False)
            
            det_result = results[0]
            boxes = det_result.boxes
            num_detections = len(boxes)

            # Columns for Image and Detections
            res_col_img, res_col_table = st.columns([1.2, 1.3], gap="large")

            with res_col_img:
                st.markdown(f"**Annotated Detections:** `{source_label_m2}`")
                # Plot bounding boxes on image
                annotated_bgr = det_result.plot()
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True, caption=f"YOLOv8s Predictions (conf ≥ {conf_threshold:.2f})")

            with res_col_table:
                st.markdown("### Detection Results Table")
                
                if num_detections == 0:
                    st.info(f"ℹ️ **No defect objects detected** at confidence threshold **{conf_threshold:.2f}**.\n\n"
                            f"Try lowering the confidence slider if fine, faint, or low-contrast defects are expected.")
                else:
                    st.success(f"🎯 **Detected {num_detections} damage object(s)** in inspection image.")
                    
                    # Compile detection details table
                    det_records = []
                    img_h, img_w = np_img.shape[:2]
                    total_area_px = img_w * img_h

                    for idx, b in enumerate(boxes):
                        cls_id = int(b.cls[0])
                        cls_name = detector.names[cls_id]
                        conf = float(b.conf[0])
                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        bw = x2 - x1
                        bh = y2 - y1
                        box_area_px = bw * bh
                        area_pct = (box_area_px / total_area_px) * 100.0

                        det_records.append({
                            "#": idx + 1,
                            "Damage Class": cls_name,
                            "Confidence": f"{conf * 100:.1f}%",
                            "BBox (x1, y1, x2, y2)": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]",
                            "Box Width": f"{int(bw)} px",
                            "Box Height": f"{int(bh)} px",
                            "Area Share": f"{area_pct:.2f}%"
                        })

                    df_detections = pd.DataFrame(det_records)
                    st.dataframe(df_detections, use_container_width=True, hide_index=True)

                    # Class summary metrics
                    st.markdown("#### Detected Classes Summary:")
                    class_counts = df_detections["Damage Class"].value_counts()
                    summary_cols = st.columns(len(class_counts))
                    for idx, (cname, cnt) in enumerate(class_counts.items()):
                        with summary_cols[idx]:
                            st.metric(cname, f"{cnt} bbox")

                st.caption("ℹ️ *Detections are statistical model inferences and do not represent validated ground truth.*")
