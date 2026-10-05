import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.physics.aero_engine import AeroLossPhysicsEngine

MODELS_DIR = os.path.join(BASE_DIR, "models")

def generate_evaluation_visuals():
    print("Generating comprehensive evaluation plots and metrics summary...")
    
    # 1. Vision Confusion Matrix Plot
    vision_metrics_file = os.path.join(MODELS_DIR, "vision_metrics.json")
    if os.path.exists(vision_metrics_file):
        with open(vision_metrics_file) as f:
            v_data = json.load(f)
        classes = v_data['class_names']
        cm = np.array(v_data['confusion_matrix'])

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)
        fig.colorbar(cax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), va='center', ha='center', fontsize=11, 
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=35, ha='left')
        ax.set_yticklabels(classes)
        ax.set_xlabel('Predicted Label', fontweight='bold', labelpad=10)
        ax.set_ylabel('Ground Truth Label', fontweight='bold', labelpad=10)
        ax.set_title(f'AeroLoss AI Vision Defect Classifier Confusion Matrix\nAccuracy: {v_data["test_accuracy"]*100:.2f}% | F1: {v_data["test_weighted_f1"]:.4f}', pad=20, fontweight='bold')
        plt.tight_layout()
        cm_path = os.path.join(MODELS_DIR, "vision_confusion_matrix.png")
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"Saved vision confusion matrix plot to {cm_path}")

    # 2. Decision Confusion Matrix Plot
    decision_metrics_file = os.path.join(MODELS_DIR, "decision_metrics.json")
    if os.path.exists(decision_metrics_file):
        with open(decision_metrics_file) as f:
            d_data = json.load(f)
        d_classes = d_data['classes']
        d_cm = np.array(d_data['confusion_matrix'])

        fig, ax = plt.subplots(figsize=(7, 5.5))
        cax = ax.matshow(d_cm, cmap=plt.cm.Greens, alpha=0.85)
        fig.colorbar(cax)
        for i in range(d_cm.shape[0]):
            for j in range(d_cm.shape[1]):
                ax.text(j, i, str(d_cm[i, j]), va='center', ha='center', fontsize=12,
                        color='white' if d_cm[i, j] > d_cm.max() / 2 else 'black')
        ax.set_xticks(range(len(d_classes)))
        ax.set_yticks(range(len(d_classes)))
        ax.set_xticklabels([c.replace('_', '\n') for c in d_classes], rotation=0)
        ax.set_yticklabels([c.replace('_', '\n') for c in d_classes])
        ax.set_xlabel('Predicted Maintenance Decision', fontweight='bold', labelpad=10)
        ax.set_ylabel('Ground Truth Action', fontweight='bold', labelpad=10)
        ax.set_title(f'AeroLoss AI Decision Engine Confusion Matrix\nAccuracy: {d_data["accuracy"]*100:.2f}% | F1: {d_data["weighted_f1"]:.4f}', pad=20, fontweight='bold')
        plt.tight_layout()
        dcm_path = os.path.join(MODELS_DIR, "decision_confusion_matrix.png")
        plt.savefig(dcm_path, dpi=200)
        plt.close()
        print(f"Saved decision confusion matrix plot to {dcm_path}")

    # 3. Decision Feature Importance Bar Chart
    if os.path.exists(decision_metrics_file):
        feat_imp = d_data['top_feature_importances']
        names = [item[0].replace('defect_class_', 'class:').replace('_', ' ') for item in feat_imp]
        vals = [item[1] * 100 for item in feat_imp]

        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, vals, align='center', color='#1f77b4', edgecolor='black', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('Relative Importance (%)', fontweight='bold')
        ax.set_title('AeroLoss AI Maintenance Decision Feature Importance', fontweight='bold')
        plt.tight_layout()
        fi_path = os.path.join(MODELS_DIR, "feature_importance.png")
        plt.savefig(fi_path, dpi=200)
        plt.close()
        print(f"Saved feature importance plot to {fi_path}")

    # 4. FFA-W3-241 Airfoil Polar Degradation Plot
    engine = AeroLossPhysicsEngine()
    clean_p = engine.clean_polar
    if clean_p is not None:
        deg_p3 = engine.get_degraded_polar(severity=3)
        deg_p5 = engine.get_degraded_polar(severity=5)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # AoA between -10 and 30 deg for standard aerodynamic envelope
        mask_c = (clean_p['AoA_deg'] >= -5) & (clean_p['AoA_deg'] <= 25)

        ax1.plot(clean_p.loc[mask_c, 'AoA_deg'], clean_p.loc[mask_c, 'Cl'], label='Clean Airfoil (Baseline)', color='green', lw=2.2)
        ax1.plot(deg_p3.loc[mask_c, 'AoA_deg'], deg_p3.loc[mask_c, 'Cl'], label='Severity 3 (Moderate)', color='orange', ls='--', lw=2.0)
        ax1.plot(deg_p5.loc[mask_c, 'AoA_deg'], deg_p5.loc[mask_c, 'Cl'], label='Severity 5 (Severe)', color='red', ls=':', lw=2.2)
        ax1.set_xlabel('Angle of Attack (deg)', fontweight='bold')
        ax1.set_ylabel('Lift Coefficient (Cl)', fontweight='bold')
        ax1.set_title('FFA-W3-241 (Re=1e7) Lift Curve Degradation', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(clean_p.loc[mask_c, 'AoA_deg'], clean_p.loc[mask_c, 'Cd'], label='Clean Airfoil', color='green', lw=2.2)
        ax2.plot(deg_p3.loc[mask_c, 'AoA_deg'], deg_p3.loc[mask_c, 'Cd'], label='Severity 3', color='orange', ls='--', lw=2.0)
        ax2.plot(deg_p5.loc[mask_c, 'AoA_deg'], deg_p5.loc[mask_c, 'Cd'], label='Severity 5', color='red', ls=':', lw=2.2)
        ax2.set_xlabel('Angle of Attack (deg)', fontweight='bold')
        ax2.set_ylabel('Drag Coefficient (Cd)', fontweight='bold')
        ax2.set_title('FFA-W3-241 (Re=1e7) Drag Polar Penalty', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        polar_plot_path = os.path.join(MODELS_DIR, "airfoil_polar_degradation.png")
        plt.savefig(polar_plot_path, dpi=200)
        plt.close()
        print(f"Saved airfoil degradation plot to {polar_plot_path}")

    # Compile overall evaluation summary JSON
    summary = {
        'vision_model': {
            'architecture': 'MobileNetV3-Small (Transfer Learning with Data Augmentation)',
            'test_accuracy': v_data['test_accuracy'],
            'test_weighted_f1': v_data['test_weighted_f1'],
            'classes': classes,
            'per_class_f1': {c: v_data['classification_report'][c]['f1-score'] for c in classes}
        },
        'physics_models': json.load(open(os.path.join(MODELS_DIR, "physics_metrics.json"))),
        'decision_model': {
            'algorithm': 'Random Forest Multi-Class Decision Support Classifier',
            'accuracy': d_data['accuracy'],
            'weighted_f1': d_data['weighted_f1'],
            'roc_auc_ovr': d_data['roc_auc_ovr'],
            'decision_classes': d_classes
        }
    }
    with open(os.path.join(MODELS_DIR, "final_evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print("ALL EVALUATIONS AND METRICS GENERATED SUCCESSFULLY!")
    print("=======================================================")

if __name__ == "__main__":
    generate_evaluation_visuals()
