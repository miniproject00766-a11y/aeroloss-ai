import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

PROCESSED_DIR = r"C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai\data\processed"
MODELS_DIR = r"C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai\models"
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
        img_path = row['abs_patch_path']
        if not os.path.exists(img_path):
            img_path = os.path.join(PROCESSED_DIR, row['patch_path'])
        
        image = Image.open(img_path).convert('RGB')
        label = CLASS_TO_IDX[row['defect_class']]

        if self.transform:
            image = self.transform(image)

        return image, label

def train_visual_classifier(epochs=6, batch_size=32, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Visual Classifier on device: {device}")

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

    train_loader = DataLoader(BladeDefectDataset(train_csv, train_transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(BladeDefectDataset(val_csv, val_test_transform), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(BladeDefectDataset(test_csv, val_test_transform), batch_size=batch_size, shuffle=False)

    # Initialize MobileNetV3-Small
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        print("Using unweighted MobileNetV3 (offline fallback)")
        model = models.mobilenet_v3_small(weights=None)

    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, len(CLASS_NAMES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, "blade_defect_classifier.pth")

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

        # Validation
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': CLASS_NAMES,
                'class_to_idx': CLASS_TO_IDX,
                'val_acc': val_acc
            }, best_model_path)

    print(f"\nBest validation accuracy: {best_val_acc*100:.2f}% (Saved to {best_model_path})")

    # Final Evaluation on Held-Out Test Set
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    conf_mat = confusion_matrix(all_labels, all_preds).tolist()

    metrics = {
        'test_accuracy': round(float(test_acc), 4),
        'test_weighted_f1': round(float(test_f1), 4),
        'confusion_matrix': conf_mat,
        'class_names': CLASS_NAMES,
        'classification_report': report
    }

    metrics_path = os.path.join(MODELS_DIR, "vision_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n==========================================")
    print(f"VISION CLASSIFIER TEST SET RESULTS:")
    print(f"Overall Accuracy: {test_acc*100:.2f}%")
    print(f"Weighted F1 Score: {test_f1:.4f}")
    print(f"==========================================")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

if __name__ == "__main__":
    train_visual_classifier(epochs=6, batch_size=32)
