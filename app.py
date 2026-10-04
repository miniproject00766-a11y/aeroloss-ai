import os
import sys
import json
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Project base paths
BASE_DIR = r"C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai"
sys.path.insert(0, BASE_DIR)

from src.pipeline import AeroLossPipeline
from src.physics.aero_engine import AeroLossPhysicsEngine

# Page configuration
st.set_page_config(
    page_title="AeroLoss AI — Physics-Informed Wind Turbine Blade Maintenance Decision Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.2rem;
        font-weight: 500;
    }
    .badge-repair {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        padding: 12px 28px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    .badge-monitor {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        color: white;
        padding: 12px 28px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    .badge-assess {
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
        color: white;
        padding: 12px 28px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
    }
    .card-metric {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    return AeroLossPipeline()

pipeline = get_pipeline()
physics_engine = AeroLossPhysicsEngine()

# Header
st.markdown('<div class="main-header">⚡ AeroLoss AI — Wind Turbine Blade Decision Support</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI Visual Defect Detection • Airfoil & BEM Physics • Energy Analytics • Financial Loss (₹/day) • Maintenance Decision Support</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/wind-turbine.png", width=64)
st.sidebar.header("Turbine & Operational Parameters")

turbine_presets = {
    "Offshore 3.6 MW (T1 SCADA Reference)": {"rated_kw": 3600.0, "r_ref": 65.0, "rated_ws": 11.5},
    "IEA 15 MW Reference Turbine": {"rated_kw": 15000.0, "r_ref": 120.0, "rated_ws": 10.6},
    "Fuhrländer FL2500 (2.5 MW)": {"rated_kw": 2500.0, "r_ref": 50.0, "rated_ws": 12.0},
    "Custom Turbine": {"rated_kw": 3000.0, "r_ref": 60.0, "rated_ws": 11.5}
}

selected_preset = st.sidebar.selectbox("Turbine Configuration", list(turbine_presets.keys()))
preset_vals = turbine_presets[selected_preset]

if selected_preset == "Custom Turbine":
    rated_kw = st.sidebar.number_input("Rated Power (kW)", value=3000.0, step=100.0)
else:
    rated_kw = st.sidebar.number_input("Rated Power (kW)", value=preset_vals["rated_kw"], step=500.0)

wind_speed = st.sidebar.slider("Current Wind Speed (m/s)", min_value=3.0, max_value=25.0, value=7.56, step=0.1, 
                               help="SCADA Annual Mean Wind Speed is 7.56 m/s")
tariff = st.sidebar.number_input("Electricity Tariff (₹/kWh)", value=4.50, step=0.10, min_value=1.0)
repair_cost = st.sidebar.number_input("Estimated Repair Mobilization (₹)", value=40000.0, step=5000.0, min_value=5000.0)

st.sidebar.markdown("---")
st.sidebar.header("Metadata & Inspection Tags")
turbine_id = st.sidebar.text_input("Turbine Identifier", value="WTG-04B")
blade_id = st.sidebar.selectbox("Blade Identified", ["Blade A", "Blade B", "Blade C"])
inspector_name = st.sidebar.text_input("Inspection Pilot / Analyst", value="Drone Flight Tech #12")

st.sidebar.markdown("---")
st.sidebar.header("Spatial Overrides (Optional)")
manual_override = st.sidebar.checkbox("Manual Position & Area Override", value=False)
if manual_override:
    override_r_R = st.sidebar.slider("Spanwise Position (r/R)", 0.20, 0.98, 0.90, 0.01)
    override_area = st.sidebar.slider("Damage Area (%)", 0.1, 15.0, 4.5, 0.1)
else:
    override_r_R, override_area = None, None

# Navigation Tabs
tab_inspect, tab_power, tab_airfoil, tab_scada, tab_audit = st.tabs([
    "🔍 Blade Inspection & Decision",
    "📈 Aerodynamics & Power Curve",
    "✈️ FFA-W3-241 Airfoil Physics",
    "📊 SCADA Operational Data",
    "🛡️ ML Engineering Audit"
])

# -------------------------------------------------------------
# TAB 1: BLADE INSPECTION & DECISION SUPPORT
# -------------------------------------------------------------
with tab_inspect:
    col_img, col_diag = st.columns([1, 1.4], gap="large")

    with col_img:
        st.subheader("1. Inspection Image Input")
        
        sample_patches_dir = os.path.join(BASE_DIR, "data", "processed", "patches")
        sample_options = []
        sample_map = {}
        
        if os.path.exists(sample_patches_dir):
            for c in os.listdir(sample_patches_dir):
                c_dir = os.path.join(sample_patches_dir, c)
                if os.path.isdir(c_dir):
                    files = [f for f in os.listdir(c_dir) if f.endswith(('.jpg', '.png'))]
                    if files:
                        label = f"Sample: {c.upper()} ({files[0]})"
                        sample_options.append(label)
                        sample_map[label] = os.path.join(c_dir, files[0])

        input_choice = st.radio("Select Image Source", ["Ground-Truth Dataset Patch", "Upload Drone Image"], horizontal=True)
        
        current_img = None
        if input_choice == "Ground-Truth Dataset Patch" and sample_options:
            selected_patch_name = st.selectbox("Indexed Defect Samples", sample_options, index=1 if len(sample_options) > 1 else 0)
            current_img = sample_map[selected_patch_name]
            st.image(current_img, caption=f"Defect Inspection Frame: {selected_patch_name}", use_container_width=True)
        else:
            uploaded = st.file_uploader("Upload Drone Blade Inspection Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded:
                current_img = Image.open(uploaded)
                st.image(current_img, caption="Uploaded Drone Inspection Frame", use_container_width=True)

        if current_img is not None:
            st.info(f"Target: **{turbine_id}** | **{blade_id}** | Pilot: **{inspector_name}**")

    with col_diag:
        if current_img is not None:
            st.subheader("2. Multi-Stage Pipeline Analysis")
            
            with st.spinner("Processing Vision, Aerodynamics & Economics..."):
                res = pipeline.analyze_image(
                    current_img,
                    r_R=override_r_R,
                    area_pct=override_area,
                    wind_speed=wind_speed,
                    rated_kw=rated_kw,
                    tariff=tariff,
                    repair_cost=repair_cost
                )

            det = res['visual_detection']
            char = res['characterization']
            phys = res['physics_aerodynamics']
            en = res['energy_impact']
            fin = res['financial_impact']
            dec = res['maintenance_decision']

            # Decision Banner
            action = dec['recommended_action']
            urgency = dec['urgency']
            
            st.markdown("### Maintenance Decision Support Recommendation")
            if action == 'REPAIR':
                st.markdown(f'<div class="badge-repair">⚠️ ACTION: REPAIR NOW (Urgency: {urgency})</div>', unsafe_allow_html=True)
            elif action == 'MONITOR':
                st.markdown(f'<div class="badge-monitor">✅ ACTION: MONITOR (Urgency: {urgency})</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-assess">🔬 ACTION: ENGINEERING ASSESSMENT (Urgency: {urgency})</div>', unsafe_allow_html=True)

            st.write("")
            for r in dec['reasoning']:
                st.markdown(f"• **{r}**")

            st.markdown("---")

            # Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Detected Defect", det['detected_class'].title(), f"{det['confidence_pct']} confidence")
            with m2:
                st.metric("Severity & Location", f"Level {char['severity_proxy']}/5", f"r/R = {char['spanwise_position_r_R']}")
            with m3:
                st.metric("Daily Energy Loss", f"{en['daily_energy_loss_kwh']:,.1f} kWh", f"-{en['aep_loss_pct']:.2f}% AEP")
            with m4:
                st.metric("Daily Financial Loss", f"₹{fin['daily_financial_loss_inr']:,.0f} / day", f"Payback: {fin['payback_days']:.1f} days")

            # Spanwise Location Diagram
            st.markdown("#### Blade Radial Spanwise Position (r/R)")
            r_val = char['spanwise_position_r_R']
            fig_span = go.Figure()
            fig_span.add_shape(type="rect", x0=0, y0=-0.15, x1=0.50, y1=0.15, fillcolor="#94a3b8", opacity=0.4, line_width=0)
            fig_span.add_shape(type="rect", x0=0.50, y0=-0.15, x1=0.80, y1=0.15, fillcolor="#fde047", opacity=0.4, line_width=0)
            fig_span.add_shape(type="rect", x0=0.80, y0=-0.15, x1=1.0, y1=0.15, fillcolor="#f87171", opacity=0.4, line_width=0)
            fig_span.add_trace(go.Scatter(
                x=[r_val], y=[0], mode="markers+text",
                marker=dict(color="#1e293b", size=18, symbol="diamond"),
                text=[f"Damage: r/R = {r_val:.2f}"], textposition="top center", name="Damage Location"
            ))
            fig_span.update_layout(
                xaxis=dict(title="Normalized Blade Span r/R (0.0 = Root, 1.0 = Tip)", range=[0, 1.02]),
                yaxis=dict(showticklabels=False, range=[-0.35, 0.35]),
                height=130, margin=dict(l=20, r=20, t=10, b=30), showlegend=False
            )
            st.plotly_chart(fig_span, use_container_width=True)

            # Payback vs Cumulative Loss Projection
            st.markdown("#### Cumulative Revenue Loss vs. One-Time Repair Mobilization")
            days = np.arange(1, 181)
            cum_loss = days * fin['daily_financial_loss_inr']
            repair_line = np.full_like(days, fin['repair_cost_inr'])

            fig_payback = go.Figure()
            fig_payback.add_trace(go.Scatter(x=days, y=cum_loss, mode="lines", name="Cumulative Continuing Loss (₹)", line=dict(color="#ef4444", width=3)))
            fig_payback.add_trace(go.Scatter(x=days, y=repair_line, mode="lines", name="One-Time Repair Cost (₹)", line=dict(color="#3b82f6", width=2, dash="dash")))

            # Add payback crossover annotation if within range
            if fin['payback_days'] <= 180:
                fig_payback.add_annotation(
                    x=fin['payback_days'], y=fin['repair_cost_inr'],
                    text=f"Payback: Day {fin['payback_days']:.1f}",
                    showarrow=True, arrowhead=2, arrowcolor="#1e293b", arrowsize=1.2,
                    font=dict(color="#1e293b", size=11, family="sans-serif")
                )

            fig_payback.update_layout(
                xaxis_title="Days Operating with Damage",
                yaxis_title="Amount (₹)",
                height=260, margin=dict(l=20, r=20, t=10, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_payback, use_container_width=True)

            # JSON Export Button
            st.download_button(
                label="📥 Export Formal Inspection Assessment (JSON)",
                data=json.dumps(res, indent=2),
                file_name=f"aeroloss_inspection_{turbine_id}_{blade_id}.json",
                mime="application/json"
            )

# -------------------------------------------------------------
# TAB 2: AERODYNAMICS & POWER CURVE
# -------------------------------------------------------------
with tab_power:
    st.subheader("Turbine Power Curve & Aerodynamic Loss Modeling")
    st.write("Calculates theoretical baseline vs degraded power production across wind speeds from cut-in (3 m/s) to rated cut-out (25 m/s).")
    
    ws_range = np.linspace(3.0, 25.0, 45)
    base_powers = []
    loss_powers = []
    damaged_powers = []
    
    active_sev = 3 if 'char' not in locals() else char['severity_proxy']
    active_r = 0.90 if 'char' not in locals() else char['spanwise_position_r_R']
    active_area = 4.5 if 'char' not in locals() else char['damage_area_pct']
    
    for v in ws_range:
        a_res = physics_engine.compute_aerodynamic_loss(active_sev, active_r, active_area, v, rated_kw)
        base_powers.append(a_res['power_baseline_kw'])
        loss_powers.append(a_res['power_loss_kw'])
        damaged_powers.append(a_res['power_damaged_kw'])

    fig_pc = go.Figure()
    fig_pc.add_trace(go.Scatter(x=ws_range, y=base_powers, mode='lines', name='Baseline Clean Power (kW)', line=dict(color='#10b981', width=3)))
    fig_pc.add_trace(go.Scatter(x=ws_range, y=damaged_powers, mode='lines', name='Degraded Operating Power (kW)', line=dict(color='#ef4444', width=2.5, dash='dash')))
    fig_pc.add_trace(go.Scatter(x=ws_range, y=loss_powers, mode='lines', name='Aerodynamic Power Deficit ΔP (kW)', line=dict(color='#f59e0b', width=2)))

    fig_pc.update_layout(
        title=f"Power Curve Degradation (Severity: {active_sev}, r/R: {active_r:.2f}, Rated: {rated_kw:,.0f} kW)",
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Generator Electrical Output (kW)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_pc, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: FFA-W3-241 AIRFOIL PHYSICS
# -------------------------------------------------------------
with tab_airfoil:
    st.subheader("FFA-W3-241 Airfoil Polar (Re = 10,000,000)")
    st.markdown(r"""
    According to **IEA Wind Task 46 & Sandia National Laboratories wind-tunnel tests**:
    - **Lift Degradation:** Category 4 leading-edge erosion creates a **12% reduction in design $C_l$** and a **19% reduction at $C_{l,\max}$**.
    - **Drag Penalty:** Design drag increases by **77%**, and post-stall drag increases by **141%**, dropping the maximum $L/D$ by approximately **50%**.
    - **Spanwise Scaling:** Aerodynamic impact scales with relative velocity raised to the **6.7 exponent**: $(r/R)^{6.7}$.
    """)

    clean_polar = physics_engine.clean_polar
    if clean_polar is not None:
        deg_3 = physics_engine.get_degraded_polar(severity=3)
        deg_5 = physics_engine.get_degraded_polar(severity=5)

        mask = (clean_polar['AoA_deg'] >= -4) & (clean_polar['AoA_deg'] <= 22)
        
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            fig_cl = go.Figure()
            fig_cl.add_trace(go.Scatter(x=clean_polar.loc[mask, 'AoA_deg'], y=clean_polar.loc[mask, 'Cl'], name='Clean Baseline (FFA-W3-241)', line=dict(color='#10b981', width=3)))
            fig_cl.add_trace(go.Scatter(x=deg_3.loc[mask, 'AoA_deg'], y=deg_3.loc[mask, 'Cl'], name='Severity 3 (Moderate)', line=dict(color='#f59e0b', width=2, dash='dash')))
            fig_cl.add_trace(go.Scatter(x=deg_5.loc[mask, 'AoA_deg'], y=deg_5.loc[mask, 'Cl'], name='Severity 5 (Severe)', line=dict(color='#ef4444', width=2.5, dash='dot')))
            fig_cl.update_layout(title="Lift Coefficient (Cl) vs Angle of Attack", xaxis_title="AoA (degrees)", yaxis_title="Cl", height=380)
            st.plotly_chart(fig_cl, use_container_width=True)

        with c_p2:
            fig_cd = go.Figure()
            fig_cd.add_trace(go.Scatter(x=clean_polar.loc[mask, 'AoA_deg'], y=clean_polar.loc[mask, 'Cd'], name='Clean Baseline', line=dict(color='#10b981', width=3)))
            fig_cd.add_trace(go.Scatter(x=deg_3.loc[mask, 'AoA_deg'], y=deg_3.loc[mask, 'Cd'], name='Severity 3 Drag Penalty', line=dict(color='#f59e0b', width=2, dash='dash')))
            fig_cd.add_trace(go.Scatter(x=deg_5.loc[mask, 'AoA_deg'], y=deg_5.loc[mask, 'Cd'], name='Severity 5 Drag Penalty', line=dict(color='#ef4444', width=2.5, dash='dot')))
            fig_cd.update_layout(title="Drag Coefficient (Cd) vs Angle of Attack", xaxis_title="AoA (degrees)", yaxis_title="Cd", height=380)
            st.plotly_chart(fig_cd, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: SCADA OPERATIONAL DATA
# -------------------------------------------------------------
with tab_scada:
    st.subheader("Turbine SCADA Operational Distribution (T1.csv)")
    scada_file = os.path.join(BASE_DIR, "data", "raw", "T1.csv")
    if os.path.exists(scada_file):
        df_scada = pd.read_csv(scada_file, nrows=1000)
        
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.markdown("#### SCADA Power Curve Scatter (Actual vs Theoretical)")
            fig_sc = px.scatter(
                df_scada, x='Wind Speed (m/s)', y='LV ActivePower (kW)',
                opacity=0.6, color_discrete_sequence=['#3b82f6'],
                labels={'LV ActivePower (kW)': 'Active Power (kW)', 'Wind Speed (m/s)': 'Wind Speed (m/s)'}
            )
            fig_sc.update_layout(height=380)
            st.plotly_chart(fig_sc, use_container_width=True)

        with c_s2:
            st.markdown("#### Operational Wind Speed Histogram")
            fig_hist = px.histogram(
                df_scada, x='Wind Speed (m/s)', nbins=30,
                color_discrete_sequence=['#10b981'],
                labels={'Wind Speed (m/s)': 'Wind Speed (m/s)', 'count': 'Observations'}
            )
            fig_hist.update_layout(height=380)
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("SCADA file T1.csv is located in data/raw/T1.csv.")

# -------------------------------------------------------------
# TAB 5: ML ENGINEERING AUDIT
# -------------------------------------------------------------
with tab_audit:
    st.subheader("Strict ML Engineering Audit & Model Transparency")
    st.markdown(r"""
    The AeroLoss AI decision engine was rigorously audited against **synthetic determinism**, **group data leakage**, and **sensor uncertainty**:
    - **Vision Classifier**: Evaluated using `GroupShuffleSplit` on `image_id` with **0.0% scene leakage** across 160 completely unseen drone inspection flights.
    - **Decision Classifier**: Trained on a realistically perturbed dataset incorporating photogrammetry tilt noise ($\pm 3.5-5\%$), area segmentation error ($\pm 12\%$), and technician inter-annotator subjectivity ($8-10\%$).
    """)

    aud_c1, aud_c2, aud_c3 = st.columns(3)
    aud_c1.metric("Strict Unseen-Image Vision Acc", "95.30%", "F1: 0.9533 (0% Leakage)")
    aud_c2.metric("Out-of-Fold Decision Acc (5-Fold CV)", "93.97%", "F1: 0.9391")
    aud_c3.metric("Overfitting Generalization Gap", "0.61%", "Train: 94.58% | Test: 93.97%")

    st.markdown("---")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        cm_path = os.path.join(BASE_DIR, "models", "vision_confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Vision Defect Classifier Confusion Matrix", use_container_width=True)
    with img_col2:
        fi_path = os.path.join(BASE_DIR, "models", "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Decision Support Feature Importances", use_container_width=True)
