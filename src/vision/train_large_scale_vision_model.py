import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Optimize CPU multi-threading
torch.set_num_threads(8)

BASE_DIR = r"C:\Users\akhil\.gemini\antigravity\scratch\aeroloss_ai"
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
KAGGLE_DIR = os.path.join(DATA_DIR, "external", "kaggle_13000_surface_damage", "NordTank586x371")
os.makedirs(MODELS_DIR, exist_ok=True)

class WindBladeUnifiedDataset(Dataset):
    def __init__(self, samples_df, transform=None):
        self.df = samples_df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = int(row['label_idx'])

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback blank image if corrupted
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label

def prepare_unified_dataset():
    print("=" * 75)
    print("INDEXING & CONSOLIDATING ALL IMAGE ASSETS FOR VISION MODEL TRAINING")
    print("=" * 75)

    samples = []

    # 1. Ingest Baseline Defect Patches (1,584 patches)
    baseline_csv = os.path.join(PROCESSED_DIR, "defect_features.csv")
    if os.path.exists(baseline_csv):
        df_base = pd.read_csv(baseline_csv)
        for _, row in df_base.iterrows():
            p_path = row['abs_patch_path']
            if not os.path.exists(p_path):
                p_path = os.path.join(PROCESSED_DIR, row['patch_path'])
            if os.path.exists(p_path):
                # Map specific damage to binary category (0: surface_wear/dirt, 1: structural_damage/erosion)
                # Or keep full defect labels. For the unified model: 1 = Damage, 0 = Clean/Dirt
                cls_name = str(row['defect_class'])
                samples.append({
                    'image_path': p_path,
                    'class_name': 'blade_damage',
                    'label_idx': 1,
                    'source': 'baseline_patches'
                })
        print(f"  - Ingested {len(samples):,} verified baseline defect patches")

    # 2. Ingest Kaggle 13k Images & YOLO Labels
    kaggle_imgs_dir = os.path.join(KAGGLE_DIR, "images")
    kaggle_lbls_dir = os.path.join(KAGGLE_DIR, "labels")

    if os.path.exists(kaggle_lbls_dir) and os.path.exists(kaggle_imgs_dir):
        label_files = [f for f in os.listdir(kaggle_lbls_dir) if f.endswith('.txt')]
        damage_count = 0
        dirt_count = 0

        for lf in label_files:
            img_name = lf.replace('.txt', '.png')
            img_path = os.path.join(kaggle_imgs_dir, img_name)
            
            if os.path.exists(img_path):
                txt_path = os.path.join(kaggle_lbls_dir, lf)
                try:
                    with open(txt_path, 'r') as fh:
                        classes_in_file = [line.strip().split()[0] for line in fh if line.strip()]
                    
                    # 1 = damage, 0 = dirt
                    if '1' in classes_in_file:
                        samples.append({
                            'image_path': img_path,
                            'class_name': 'blade_damage',
                            'label_idx': 1,
                            'source': 'kaggle_damage'
                        })
                        damage_count += 1
                    elif '0' in classes_in_file:
                        samples.append({
                            'image_path': img_path,
                            'class_name': 'surface_dirt',
                            'label_idx': 0,
                            'source': 'kaggle_dirt'
                        })
                        dirt_count += 1
                except Exception:
                    pass

        print(f"  - Ingested {damage_count:,} Kaggle damage images (Class: blade_damage)")
        print(f"  - Ingested {dirt_count:,} Kaggle dirt/surface wear images (Class: surface_dirt)")

    # 3. Add clean blade images from the unannotated pool
    all_kaggle_images = [os.path.join(kaggle_imgs_dir, f) for f in os.listdir(kaggle_imgs_dir) if f.endswith('.png')]
    labeled_img_names = set(os.path.basename(s['image_path']) for s in samples)
    unlabeled_images = [p for p in all_kaggle_images if os.path.basename(p) not in labeled_img_names]

    np.random.seed(42)
    # Sample up to 1,500 clean blade surface photos for balanced negative class
    clean_sample_size = min(1500, len(unlabeled_images))
    selected_clean = np.random.choice(unlabeled_images, size=clean_sample_size, replace=False)

    for cp in selected_clean:
        samples.append({
            'image_path': cp,
            'class_name': 'clean_blade_surface',
            'label_idx': 0,
            'source': 'kaggle_clean'
        })
    print(f"  - Ingested {clean_sample_size:,} pristine/clean blade surface photos (Class: clean/normal)")

    full_df = pd.DataFrame(samples)
    print("\nUNIFIED DATASET CLASS BREAKDOWN:")
    print(full_df['class_name'].value_counts())
    print(f"TOTAL CONSOLIDATED IMAGES READY FOR TRAINING: {len(full_df):,}")

    return full_df

def train_large_scale_model(epochs=4, batch_size=48, lr=1e-3):
    df_all = prepare_unified_dataset()

    # Stratified Train/Val/Test Split (70% / 15% / 15%)
    train_df, temp_df = train_test_split(df_all, test_size=0.30, random_state=42, stratify=df_all['label_idx'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label_idx'])

    print(f"\nTrain Set: {len(train_df):,} | Val Set: {len(val_df):,} | Test Set: {len(test_df):,}")

    # Advanced Data Augmentations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(WindBladeUnifiedDataset(train_df, train_transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindBladeUnifiedDataset(val_df, eval_transform), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WindBladeUnifiedDataset(test_df, eval_transform), batch_size=batch_size, shuffle=False)

    # Initialize MobileNetV3-Small
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nInitializing MobileNetV3-Small on: {device}")

    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        model = models.mobilenet_v3_small(weights=None)

    num_features = model.classifier[3].in_features
    # 2 output classes: 0 = Normal / Surface Wear, 1 = Active Blade Damage / Erosion
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 2)
    )
    model = model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    checkpoint_path = os.path.join(MODELS_DIR, "blade_defect_classifier_large_scale.pth")

    print("\n" + "=" * 75)
    print("STARTING LARGE-SCALE PYTORCH CONVOLUTIONAL TRAINING LOOP")
    print("=" * 75)

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total
        avg_train_loss = train_loss / total

        # Validation Phase
        model.eval()
        v_correct, v_total, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        val_acc = v_correct / v_total
        avg_val_loss = v_loss / v_total
        elapsed = time.time() - epoch_start

        print(f"Epoch [{epoch+1}/{epochs}] ({elapsed:.1f}s) | Train Loss: {avg_train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': ['normal_surface_dirt', 'blade_damage_erosion']
            }, checkpoint_path)
            print(f"  --> Saved new best checkpoint (Val Acc: {val_acc*100:.2f}%)")

    # =========================================================================
    # FINAL EVALUATION ON HELD-OUT TEST SET
    # =========================================================================
    print("\n" + "=" * 75)
    print("FINAL EVALUATION ON INDEPENDENT HELD-OUT TEST SET")
    print("=" * 75)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
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
    report = classification_report(all_labels, all_preds, target_names=['Normal/Dirt', 'Blade Damage'], output_dict=True)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    print(f"Final Test Accuracy:  {test_acc * 100:.2f}%")
    print(f"Final Weighted F1:    {test_f1:.4f}")
    print("\nDetailed Test Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal/Dirt', 'Blade Damage']))

    # Save metrics JSON
    metrics = {
        'training_scope': 'LARGE_SCALE_CONSOLIDATED',
        'total_images_indexed': len(df_all),
        'test_accuracy': round(float(test_acc), 4),
        'test_weighted_f1': round(float(test_f1), 4),
        'confusion_matrix': cm,
        'classification_report': report,
        'checkpoint': checkpoint_path
    }

    metrics_out = os.path.join(MODELS_DIR, "vision_large_scale_evaluation.json")
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved metrics to: {metrics_out}")
    print("=" * 75)
    print("LARGE-SCALE TRAINING COMPLETE!")
    print("=" * 75)

if __name__ == "__main__":
    train_large_scale_model(epochs=4, batch_size=48, lr=1e-3)
