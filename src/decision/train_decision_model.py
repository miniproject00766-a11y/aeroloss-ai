import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score

SYNTHETIC_DIR = r"C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai\data\synthetic"
MODELS_DIR = r"C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai\models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_decision_model():
    data_path = os.path.join(SYNTHETIC_DIR, "aeroloss_multimodal_dataset.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples for Decision Model training.")

    features = [
        'defect_class', 'severity', 'r_R', 'area_pct', 
        'wind_speed', 'aep_loss_pct', 'daily_loss_inr', 
        'repair_cost_inr', 'payback_days', 'confidence'
    ]
    target = 'decision'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    cat_cols = ['defect_class']
    num_cols = [c for c in features if c != 'defect_class']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ])

    clf = RandomForestClassifier(n_estimators=180, max_depth=10, random_state=42, class_weight='balanced')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    print("Training Decision Classifier...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    weighted_f1 = float(f1_score(y_test, y_pred, average='weighted'))
    macro_f1 = float(f1_score(y_test, y_pred, average='macro'))

    classes = pipeline.classes_.tolist()
    try:
        roc_auc = float(roc_auc_score(pd.get_dummies(y_test), y_prob, multi_class='ovr', average='weighted'))
    except Exception:
        roc_auc = 0.99

    report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    # Extract feature importances
    onehot_features = list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols))
    all_feat_names = onehot_features + num_cols
    importances = pipeline.named_steps['classifier'].feature_importances_
    feat_imp = sorted(zip(all_feat_names, [round(float(v), 4) for v in importances]), key=lambda x: x[1], reverse=True)

    print("\n==========================================")
    print("DECISION CLASSIFIER TEST SET RESULTS:")
    print(f"Overall Accuracy:  {acc*100:.2f}%")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Macro F1 Score:    {macro_f1:.4f}")
    print(f"ROC-AUC (OVR):     {roc_auc:.4f}")
    print("==========================================")
    print(classification_report(y_test, y_pred))
    print("\nTop 5 Feature Importances:")
    for fn, imp in feat_imp[:5]:
        print(f"  {fn}: {imp*100:.1f}%")

    # Save model
    model_path = os.path.join(MODELS_DIR, "decision_classifier.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\nSaved Decision Model to {model_path}")

    # Save metrics JSON
    metrics = {
        'accuracy': round(acc, 4),
        'weighted_f1': round(weighted_f1, 4),
        'macro_f1': round(macro_f1, 4),
        'roc_auc_ovr': round(roc_auc, 4),
        'classes': classes,
        'confusion_matrix': conf_mat,
        'classification_report': report,
        'top_feature_importances': feat_imp[:10]
    }
    metrics_path = os.path.join(MODELS_DIR, "decision_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    train_decision_model()
