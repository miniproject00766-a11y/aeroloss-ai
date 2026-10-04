import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SYNTHETIC_DIR = r"C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai\data\synthetic"
MODELS_DIR = r"C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai\models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_physics_models():
    data_path = os.path.join(SYNTHETIC_DIR, "aeroloss_multimodal_dataset.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    features = ['defect_class', 'severity', 'r_R', 'area_pct', 'aspect_ratio', 'wind_speed', 'rated_power_kw']
    targets = ['power_loss_kw', 'aep_loss_pct', 'daily_energy_loss_kwh']

    X = df[features]
    y = df[targets]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    cat_cols = ['defect_class']
    num_cols = ['severity', 'r_R', 'area_pct', 'aspect_ratio', 'wind_speed', 'rated_power_kw']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ])

    models = {}
    metrics = {}

    for target in targets:
        print(f"\nTraining Regressor for target: '{target}'...")
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42))
        ])

        pipe.fit(X_train, y_train[target])
        y_pred = pipe.predict(X_test)

        r2 = float(r2_score(y_test[target], y_pred))
        mae = float(mean_absolute_error(y_test[target], y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test[target], y_pred)))
        mean_actual = float(y_test[target].mean())

        print(f"  Target: {target}")
        print(f"  R^2 Score: {r2:.4f}")
        print(f"  MAE:       {mae:.4f}")
        print(f"  RMSE:      {rmse:.4f}")
        print(f"  Mean Val:  {mean_actual:.4f}")

        models[target] = pipe
        metrics[target] = {
            'r2_score': round(r2, 4),
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'mean_actual': round(mean_actual, 4)
        }

    # Save trained pipelines
    bundle_path = os.path.join(MODELS_DIR, "physics_regressors.joblib")
    joblib.dump(models, bundle_path)
    print(f"\nSaved trained physics models bundle to {bundle_path}")

    # Save metrics JSON
    metrics_path = os.path.join(MODELS_DIR, "physics_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    train_physics_models()
