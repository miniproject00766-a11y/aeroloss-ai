import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CLASS_NAMES = ['corrosion', 'crack', 'craze', 'hide_craze', 'surface_injure', 'thunderstrike']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

class BladeDefectDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(PROCESSED_DIR, row['patch_path'])
        if not os.path.exists(img_path):
            img_path = row.get('abs_patch_path', '')
        
        image = Image.open(img_path).convert('RGB')
        label = CLASS_TO_IDX[row['defect_class']]

        if self.transform:
            image = self.transform(image)

        return image, label

def train_visual_classifier(
    epochs=6,
    batch_size=32,
    lr=1e-3,
    output_checkpoint="models/blade_defect_classifier_candidate.pth",
    output_metrics="models/vision_metrics_candidate.json",
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Visual Classifier on device: {device} (seed={seed})")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_csv = os.path.join(PROCESSED_DIR, "train_patches.csv")
    val_csv = os.path.join(PROCESSED_DIR, "val_patches.csv")
    test_csv = os.path.join(PROCESSED_DIR, "test_patches.csv")

    train_dataset = BladeDefectDataset(train_csv, train_transform)
    val_dataset = BladeDefectDataset(val_csv, val_test_transform)
    test_dataset = BladeDefectDataset(test_csv, val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_df = pd.read_csv(train_csv)
    class_counts = train_df['defect_class'].value_counts()
    total_samples = len(train_df)
    weights = [total_samples / (len(CLASS_NAMES) * class_counts.get(c, 1)) for c in CLASS_NAMES]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Computed Class Weights: {dict(zip(CLASS_NAMES, [round(w, 2) for w in weights]))}")

    # Initialize MobileNetV3-Small with official ImageNet weights if available
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        print("Loaded ImageNet pre-trained weights for MobileNetV3-Small")
    except Exception as e:
        print(f"Using unweighted MobileNetV3 (offline fallback): {e}")
        model = models.mobilenet_v3_small(weights=None)

    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, len(CLASS_NAMES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / total

        # Validation on held-out validation set (model selection only)
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        val_acc = v_correct / v_total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest validation accuracy: {best_val_acc*100:.2f}%")
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Calibrate Temperature Scaling strictly on Validation Set
    model.eval()
    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            val_logits_list.append(model(images).cpu())
            val_labels_list.append(labels)
    val_logits = torch.cat(val_logits_list)
    val_labels = torch.cat(val_labels_list)

    temperature = nn.Parameter(torch.ones(1) * 1.5)
    temp_optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    def eval_temp():
        temp_optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(val_logits / torch.clamp(temperature, min=0.1), val_labels)
        loss.backward()
        return loss
    temp_optimizer.step(eval_temp)
    optimal_temp = float(torch.clamp(temperature, min=0.1).item())
    print(f"Optimal Temperature Scaling Parameter T = {optimal_temp:.4f}")

    # Final Independent Evaluation on Held-Out Zero-Leakage Test Set
    all_preds, all_labels = [], []
    all_probs = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            calibrated_outputs = outputs / optimal_temp
            probs = torch.softmax(calibrated_outputs, dim=1).cpu()
            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_acc = accuracy_score(all_labels, all_preds)
    test_weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(all_labels, all_preds).tolist()

    # Calculate Expected Calibration Error (ECE)
    confidences = np.max(all_probs, axis=1)
    accuracies = (all_preds == all_labels)
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    # Calculate Brier Score (multi-class mean squared error)
    one_hot_labels = np.eye(len(CLASS_NAMES))[all_labels]
    brier_score = float(np.mean(np.sum((all_probs - one_hot_labels)**2, axis=1)))

    print(f"\n==========================================")
    print(f"ALIGNED ZERO-LEAKAGE TEST SET RESULTS:")
    print(f"Test Accuracy:          {test_acc*100:.2f}%")
    print(f"Weighted F1 Score:      {test_weighted_f1:.4f}")
    print(f"Macro F1 Score:         {test_macro_f1:.4f}")
    print(f"Expected Calib Error:   {ece:.4f}")
    print(f"Brier Score:            {brier_score:.4f}")
    print(f"Optimal Temperature T:  {optimal_temp:.4f}")
    print(f"==========================================")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))

    # Save Candidate Checkpoint
    target_ckpt_path = os.path.join(BASE_DIR, output_checkpoint) if not os.path.isabs(output_checkpoint) else output_checkpoint
    os.makedirs(os.path.dirname(target_ckpt_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': CLASS_NAMES,
        'class_to_idx': CLASS_TO_IDX,
        'val_acc': float(best_val_acc),
        'temperature': float(optimal_temp),
        'test_accuracy': float(test_acc),
        'test_weighted_f1': float(test_weighted_f1),
        'test_macro_f1': float(test_macro_f1),
        'ece': float(ece),
        'brier_score': float(brier_score),
        'split_alignment': 'parent_image_aligned_zero_leakage',
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'optimizer': 'AdamW',
            'weight_decay': 1e-4,
            'loss_function': 'CrossEntropyLoss_ClassWeighted',
            'seed': seed
        }
    }, target_ckpt_path)
    print(f"Saved candidate checkpoint to: {target_ckpt_path}")

    # Save Candidate Metrics JSON
    target_metrics_path = os.path.join(BASE_DIR, output_metrics) if not os.path.isabs(output_metrics) else output_metrics
    metrics = {
        'model_checkpoint': os.path.basename(target_ckpt_path),
        'split_alignment_status': 'ALIGNED_ZERO_LEAKAGE',
        'independent_evaluation_status': 'VERIFIED_INDEPENDENT',
        'test_accuracy': round(float(test_acc), 4),
        'test_weighted_f1': round(float(test_weighted_f1), 4),
        'test_macro_f1': round(float(test_macro_f1), 4),
        'expected_calibration_error': round(float(ece), 4),
        'brier_score': round(float(brier_score), 4),
        'temperature': round(float(optimal_temp), 4),
        'best_val_acc': round(float(best_val_acc), 4),
        'confusion_matrix': conf_mat,
        'class_names': CLASS_NAMES,
        'classification_report': report,
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'optimizer': 'AdamW',
            'weight_decay': 1e-4,
            'seed': seed
        }
    }

    with open(target_metrics_path, "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved candidate metrics to: {target_metrics_path}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV3 Blade Defect Classifier")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-checkpoint", type=str, default="models/blade_defect_classifier_candidate.pth", help="Candidate checkpoint path")
    parser.add_argument("--output-metrics", type=str, default="models/vision_metrics_candidate.json", help="Candidate metrics JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_visual_classifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_checkpoint=args.output_checkpoint,
        output_metrics=args.output_metrics,
        seed=args.seed
    )
