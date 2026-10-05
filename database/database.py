import os
import sqlite3
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "aeroloss.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Turbines Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS turbines (
        turbine_id TEXT PRIMARY KEY,
        turbine_name TEXT NOT NULL,
        turbine_model TEXT NOT NULL,
        rated_power_kw REAL NOT NULL,
        rotor_diameter REAL NOT NULL,
        blade_count INTEGER NOT NULL DEFAULT 3,
        hub_height REAL NOT NULL,
        blade_length REAL NOT NULL,
        airfoil_mapping TEXT NOT NULL,
        tariff REAL NOT NULL,
        capacity_factor REAL NOT NULL,
        repair_cost REAL NOT NULL,
        location TEXT NOT NULL,
        inspection_date TEXT NOT NULL,
        health_status TEXT NOT NULL,
        severity_grade TEXT NOT NULL,
        defect_summary TEXT NOT NULL,
        r_R_summary TEXT NOT NULL,
        sample_class TEXT NOT NULL,
        sample_file TEXT NOT NULL,
        explainability TEXT NOT NULL
    );
    """)

    # 2. Inspections & Analyses Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inspections (
        inspection_id TEXT PRIMARY KEY,
        turbine_id TEXT NOT NULL,
        inspection_date TEXT NOT NULL,
        detected_class TEXT NOT NULL,
        confidence REAL NOT NULL,
        damage_area_pct REAL NOT NULL,
        r_R REAL NOT NULL,
        severity INTEGER NOT NULL,
        baseline_power_kw REAL NOT NULL,
        damaged_power_kw REAL NOT NULL,
        power_loss_kw REAL NOT NULL,
        aep_loss_pct REAL NOT NULL,
        daily_energy_loss_kwh REAL NOT NULL,
        daily_financial_loss_inr REAL NOT NULL,
        repair_cost_inr REAL NOT NULL,
        payback_days REAL NOT NULL,
        recommended_action TEXT NOT NULL,
        urgency TEXT NOT NULL,
        reasoning TEXT NOT NULL,
        FOREIGN KEY (turbine_id) REFERENCES turbines (turbine_id)
    );
    """)

    # 3. Model Metrics Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_metrics (
        model_id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        version TEXT NOT NULL,
        precision REAL NOT NULL,
        recall REAL NOT NULL,
        f1_score REAL NOT NULL,
        mAP_50 REAL NOT NULL,
        mAP_50_95 REAL NOT NULL,
        metrics_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    conn.commit()

    # Seed Initial Turbine Database if empty
    cursor.execute("SELECT COUNT(*) FROM turbines;")
    count = cursor.fetchone()[0]
    if count == 0:
        seed_turbines(cursor)
        conn.commit()

    conn.close()
    print("SQLite Database initialized at:", DB_PATH)

def seed_turbines(cursor):
    initial_turbines = [
        (
            "WTG-04B", "Offshore Turbine Unit 04B", "Offshore 3.6 MW (T1 SCADA Reference)", 3600.0, 107.0, 3, 90.0, 52.0,
            "FFA-W3-241 (Re=1e7)", 4.50, 0.42, 40000.0, "Offshore Cluster A - Bay 4", "2026-10-04",
            "Critical", "Level 4 / 5", "Leading-Edge Damage", "0.91 (Tip)", "crack", "patch_00155.jpg",
            "Damage located at r/R = 0.91 near the blade tip, where tip rotational speed exceeds 250 km/h, amplifying profile drag and producing maximum daily loss."
        ),
        (
            "WTG-06A", "Offshore Turbine Unit 06A", "Offshore 3.6 MW (T1 SCADA Reference)", 3600.0, 107.0, 3, 90.0, 52.0,
            "FFA-W3-241 (Re=1e7)", 4.50, 0.42, 40000.0, "Offshore Cluster A - Bay 6", "2026-10-04",
            "High Risk", "Level 3 / 5", "Crack", "0.84 (Outboard)", "crack", "patch_00058.jpg",
            "Outboard structural crack at r/R = 0.84. Boundary layer flow detachment creates steady torque reduction resulting in high daily loss."
        ),
        (
            "WTG-03B", "Offshore Turbine Unit 03B", "Offshore 3.6 MW (T1 SCADA Reference)", 3600.0, 107.0, 3, 90.0, 52.0,
            "FFA-W3-241 (Re=1e7)", 4.50, 0.42, 40000.0, "Offshore Cluster A - Bay 3", "2026-10-04",
            "Monitor", "Level 2 / 5", "Surface Injury", "0.76 (Outboard)", "surface_injure", "patch_00003.jpg",
            "Moderate surface pitting at r/R = 0.76. Extended payback period indicates repair should be scheduled during routine outage."
        ),
        (
            "WTG-02A", "Offshore Turbine Unit 02A", "Offshore 3.6 MW (T1 SCADA Reference)", 3600.0, 107.0, 3, 90.0, 52.0,
            "FFA-W3-241 (Re=1e7)", 4.50, 0.42, 40000.0, "Offshore Cluster A - Bay 2", "2026-10-04",
            "Monitor", "Level 1 / 5", "Surface Defect", "0.62 (Mid-span)", "corrosion", "patch_00000.jpg",
            "Mid-span surface roughness at r/R = 0.62. Low rotational velocity limits total financial impact."
        ),
        (
            "WTG-01A", "Offshore Turbine Unit 01A", "Offshore 3.6 MW (T1 SCADA Reference)", 3600.0, 107.0, 3, 90.0, 52.0,
            "FFA-W3-241 (Re=1e7)", 4.50, 0.42, 40000.0, "Offshore Cluster A - Bay 1", "2026-10-04",
            "Healthy", "Level 0 / 5", "No Defect", "0.88 (Tip)", "corrosion", "patch_00384.jpg",
            "Normal clean blade condition. Aerodynamic lift and drag coefficients remain at baseline reference polars."
        ),
        (
            "WTG-05B", "Offshore Turbine Unit 05B", "Offshore 3.6 MW (T1 SCADA Reference)", 3600.0, 107.0, 3, 90.0, 52.0,
            "FFA-W3-241 (Re=1e7)", 4.50, 0.42, 40000.0, "Offshore Cluster A - Bay 5", "2026-10-04",
            "Healthy", "Level 0 / 5", "No Defect", "0.90 (Tip)", "thunderstrike", "patch_00427.jpg",
            "Clean operating profile with minor non-degrading surface discoloration. Zero AEP penalty."
        )
    ]

    cursor.executemany("""
    INSERT OR REPLACE INTO turbines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
    """, initial_turbines)

    # Seed initial inspection progression history for each turbine
    seed_history(cursor)

def seed_history(cursor):
    # Historical inspections over past 3 months
    history_records = [
        ("INSP-04B-01", "WTG-04B", "2026-08-01", "crack", 0.92, 1.8, 0.90, 2, 1450.0, 1420.0, 30.0, 1.2, 360.0, 1620.0, 40000.0, 24.7, "MONITOR", "Low", "Early minor micro-crack observed at r/R=0.90."),
        ("INSP-04B-02", "WTG-04B", "2026-09-01", "crack", 0.95, 3.2, 0.91, 3, 1450.0, 1400.0, 50.0, 2.5, 600.0, 2700.0, 40000.0, 14.8, "MONITOR", "Medium", "Crack expanded toward leading edge."),
        ("INSP-04B-03", "WTG-04B", "2026-10-04", "crack", 0.98, 4.5, 0.91, 4, 1450.0, 1355.0, 94.4, 3.8, 944.0, 4250.0, 40000.0, 9.4, "REPAIR", "Critical", "Leading-edge erosion severe near tip. Rapid payback crossover."),

        ("INSP-06A-01", "WTG-06A", "2026-08-15", "crack", 0.90, 2.1, 0.84, 2, 1450.0, 1425.0, 25.0, 1.0, 300.0, 1350.0, 40000.0, 29.6, "MONITOR", "Low", "Minor structural flaw at r/R=0.84."),
        ("INSP-06A-02", "WTG-06A", "2026-10-04", "crack", 0.94, 3.8, 0.84, 3, 1450.0, 1388.0, 62.0, 2.1, 592.0, 2665.0, 40000.0, 15.0, "REPAIR", "High", "Outboard crack expanded. Boundary layer detachment creates torque penalty."),

        ("INSP-03B-01", "WTG-03B", "2026-10-04", "surface_injure", 0.89, 2.4, 0.76, 2, 1450.0, 1427.0, 23.0, 0.7, 219.0, 985.0, 40000.0, 40.6, "MONITOR", "Medium", "Surface pitting at mid-outboard span."),
        ("INSP-02A-01", "WTG-02A", "2026-10-04", "corrosion", 0.85, 1.5, 0.62, 1, 1450.0, 1438.0, 12.0, 0.4, 95.0, 428.0, 40000.0, 93.4, "MONITOR", "Medium", "Mid-span surface roughness at r/R=0.62."),
        ("INSP-01A-01", "WTG-01A", "2026-10-04", "clean", 0.99, 0.0, 0.88, 0, 1450.0, 1450.0, 0.0, 0.0, 0.0, 0.0, 40000.0, 0.0, "CLEAN", "Low", "Normal clean blade condition. Aerodynamic lift and drag coefficients remain at baseline reference polars."),
        ("INSP-05B-01", "WTG-05B", "2026-10-04", "clean", 0.99, 0.0, 0.90, 0, 1450.0, 1450.0, 0.0, 0.0, 0.0, 0.0, 40000.0, 0.0, "CLEAN", "Low", "Clean operating profile with minor non-degrading surface discoloration. Zero AEP penalty.")
    ]

    cursor.executemany("""
    INSERT OR REPLACE INTO inspections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
    """, history_records)

if __name__ == '__main__':
    init_db()
