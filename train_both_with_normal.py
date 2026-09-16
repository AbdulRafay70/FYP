"""
Train both ResNet50 and EfficientNetV2 on all Benign, Malignant, and Normal images.
Normal images are labeled as Class 0 (Benign / Non-cancerous).
"""

import os
import sys
import glob
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

sys.path.append(r"E:\FYP\training")
from data.transforms import get_classification_val_transforms
from models.classifier_heads import ResNet50Classifier, EfficientNetV2B0Classifier

device = torch.device("cpu")
print(f"[Training] Using device: {device}")

# -------------------------------------------------------------
# 1. Collect all dataset samples including NORMAL images
# -------------------------------------------------------------
samples = [] # (image_numpy, label: 0=Non-cancerous/Benign/Normal, 1=Malignant)

# A. BUSI Benign & Malignant crops
busi_dir = r"E:\FYP\datasets\BUSI\Dataset_BUSI_with_GT"
ben_masks = glob.glob(f"{busi_dir}/benign/*_mask.png")
mal_masks = glob.glob(f"{busi_dir}/malignant/*_mask.png")

print(f"Loading BUSI lesion crops: {len(ben_masks)} Benign masks, {len(mal_masks)} Malignant masks...")

for m_path in ben_masks:
    img_path = m_path.replace("_mask.png", ".png")
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        continue
    y_idx, x_idx = np.where(mask > 0)
    if len(y_idx) == 0:
        continue
    y1, y2 = y_idx.min(), y_idx.max()
    x1, x2 = x_idx.min(), x_idx.max()
    h, w = img.shape[:2]
    pad_x = int((x2 - x1) * 0.15)
    pad_y = int((y2 - y1) * 0.15)
    roi = img[max(0, y1-pad_y):min(h, y2+pad_y), max(0, x1-pad_x):min(w, x2+pad_x)]
    samples.append((roi, 0))

for m_path in mal_masks:
    img_path = m_path.replace("_mask.png", ".png")
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        continue
    y_idx, x_idx = np.where(mask > 0)
    if len(y_idx) == 0:
        continue
    y1, y2 = y_idx.min(), y_idx.max()
    x1, x2 = x_idx.min(), x_idx.max()
    h, w = img.shape[:2]
    pad_x = int((x2 - x1) * 0.15)
    pad_y = int((y2 - y1) * 0.15)
    roi = img[max(0, y1-pad_y):min(h, y2+pad_y), max(0, x1-pad_x):min(w, x2+pad_x)]
    samples.append((roi, 1))

# B. BUSI Normal Images (Labeled 0: Non-cancerous / Healthy)
norm_files = glob.glob(f"{busi_dir}/normal/*.png")
norm_files = [f for f in norm_files if "_mask" not in f]
print(f"Loading BUSI Normal healthy images: {len(norm_files)} scans...")

for p in norm_files:
    img = cv2.imread(p)
    if img is not None:
        h, w = img.shape[:2]
        # 1. Full scan
        samples.append((img, 0))
        # 2. Central crop
        samples.append((img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)], 0))
        # 3. Top-left region
        samples.append((img[0:int(h*0.65), 0:int(w*0.65)], 0))
        # 4. Top-right region
        samples.append((img[0:int(h*0.65), int(w*0.35):w], 0))
        # 5. Bottom-left region
        samples.append((img[int(h*0.35):h, 0:int(w*0.65)], 0))
        # 6. Bottom-right region
        samples.append((img[int(h*0.35):h, int(w*0.35):w], 0))
        # 7. Simulated YOLO bounding box crops (random boxes)
        for _ in range(2):
            crop_h = random.randint(int(h * 0.3), int(h * 0.7))
            crop_w = random.randint(int(w * 0.3), int(w * 0.7))
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            samples.append((img[top:top+crop_h, left:left+crop_w], 0))

# C. BUS_UC (Archive)
uc_ben = glob.glob(r"E:\FYP\datasets\archive\BUS_UC_classification\BUS_UC_classification\Benign\*.png")
uc_mal = glob.glob(r"E:\FYP\datasets\archive\BUS_UC_classification\BUS_UC_classification\Malignant\*.png")
for p in uc_ben:
    img = cv2.imread(p)
    if img is not None:
        samples.append((img, 0))
for p in uc_mal:
    img = cv2.imread(p)
    if img is not None:
        samples.append((img, 1))

total_0 = sum(1 for _, l in samples if l == 0)
total_1 = sum(1 for _, l in samples if l == 1)
print(f"\nTotal Dataset: {len(samples)} samples (Class 0 [Benign/Normal]: {total_0}, Class 1 [Malignant]: {total_1})")

# -------------------------------------------------------------
# 2. Stratified Train / Validation Split (80% / 20%)
# -------------------------------------------------------------
random.seed(42)
c0_samples = [s for s in samples if s[1] == 0]
c1_samples = [s for s in samples if s[1] == 1]
random.shuffle(c0_samples)
random.shuffle(c1_samples)

n_t0 = int(0.8 * len(c0_samples))
n_t1 = int(0.8 * len(c1_samples))

train_samples = c0_samples[:n_t0] + c1_samples[:n_t1]
val_samples = c0_samples[n_t0:] + c1_samples[n_t1:]
random.shuffle(train_samples)
random.shuffle(val_samples)

print(f"Train set: {len(train_samples)} | Validation set: {len(val_samples)}")

transform = get_classification_val_transforms(224)

# -------------------------------------------------------------
# 3. Train ResNet50 Classifier
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("  TRAINING MODEL 1: ResNet50 (with LayerNorm)")
print("=" * 60)

resnet_model = ResNet50Classifier(num_classes=2)
res_extractor = resnet_model.get_feature_extractor()
res_extractor.eval()

t0 = time.time()
def extract_resnet_feats(data_list, desc="Set"):
    feats, labels = [], []
    for i, (img, lbl) in enumerate(data_list):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(image=img_rgb)["image"].unsqueeze(0)
        with torch.no_grad():
            f = res_extractor(tensor)
            feats.append(torch.flatten(f, 1).squeeze(0))
            labels.append(lbl)
        if (i + 1) % 300 == 0 or (i + 1) == len(data_list):
            print(f"  [{desc}] {i+1}/{len(data_list)} ({time.time()-t0:.1f}s)")
    return torch.stack(feats), torch.tensor(labels, dtype=torch.long)

res_train_X, train_y = extract_resnet_feats(train_samples, "Train")
res_val_X, val_y = extract_resnet_feats(val_samples, "Val")

# ResNet Head
res_head = resnet_model.backbone.fc
w0 = len(train_y) / (2.0 * (train_y == 0).sum().item())
w1 = len(train_y) / (2.0 * (train_y == 1).sum().item())
criterion = nn.CrossEntropyLoss(weight=torch.tensor([w0, w1], dtype=torch.float32))
res_opt = torch.optim.AdamW(res_head.parameters(), lr=1e-3, weight_decay=1e-4)

best_res_f1 = 0.0
best_res_state = None
n_train = res_train_X.size(0)

for epoch in range(1, 101):
    res_head.train()
    perm = torch.randperm(n_train)
    for i in range(0, n_train, 32):
        idx = perm[i:i+32]
        bx, by = res_train_X[idx], train_y[idx]
        res_opt.zero_grad()
        loss = criterion(res_head(bx), by)
        loss.backward()
        res_opt.step()

    res_head.eval()
    with torch.no_grad():
        val_logits = res_head(res_val_X)
        val_preds = val_logits.argmax(dim=1)
        tp = ((val_preds == 1) & (val_y == 1)).sum().item()
        fp = ((val_preds == 1) & (val_y == 0)).sum().item()
        fn = ((val_preds == 0) & (val_y == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        acc = (val_preds == val_y).float().mean().item()

    if f1 > best_res_f1:
        best_res_f1 = f1
        best_res_state = {k: v.cpu().clone() for k, v in resnet_model.state_dict().items()}
        print(f"  Epoch {epoch:3d} | Acc: {acc*100:.1f}% | Sens: {recall*100:.1f}% | F1: {f1:.4f} -> BEST")

resnet_weights_path = r"E:\FYP\backend\weights\resnet50_breast.pth"
torch.save(best_res_state, resnet_weights_path)
print(f"ResNet50 Saved! Best Validation F1: {best_res_f1:.4f}")

# -------------------------------------------------------------
# 4. Train EfficientNetV2 Classifier
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("  TRAINING MODEL 2: EfficientNetV2-s (with LayerNorm)")
print("=" * 60)

eff_model = EfficientNetV2B0Classifier(num_classes=2)
# Feature extractor is backbone features + avgpool
class EffFeatureExtractor(nn.Module):
    def __init__(self, eff_backbone):
        super().__init__()
        self.features = eff_backbone.features
        self.avgpool = eff_backbone.avgpool
    def forward(self, x):
        return torch.flatten(self.avgpool(self.features(x)), 1)

eff_extractor = EffFeatureExtractor(eff_model.backbone)
eff_extractor.eval()

t1 = time.time()
def extract_eff_feats(data_list, desc="Set"):
    feats = []
    for i, (img, _) in enumerate(data_list):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(image=img_rgb)["image"].unsqueeze(0)
        with torch.no_grad():
            f = eff_extractor(tensor)
            feats.append(f.squeeze(0))
        if (i + 1) % 300 == 0 or (i + 1) == len(data_list):
            print(f"  [{desc}] {i+1}/{len(data_list)} ({time.time()-t1:.1f}s)")
    return torch.stack(feats)

eff_train_X = extract_eff_feats(train_samples, "Train")
eff_val_X = extract_eff_feats(val_samples, "Val")

eff_head = eff_model.backbone.classifier
eff_opt = torch.optim.AdamW(eff_head.parameters(), lr=1e-3, weight_decay=1e-4)

best_eff_f1 = 0.0
best_eff_state = None

for epoch in range(1, 101):
    eff_head.train()
    perm = torch.randperm(n_train)
    for i in range(0, n_train, 32):
        idx = perm[i:i+32]
        bx, by = eff_train_X[idx], train_y[idx]
        eff_opt.zero_grad()
        loss = criterion(eff_head(bx), by)
        loss.backward()
        eff_opt.step()

    eff_head.eval()
    with torch.no_grad():
        val_logits = eff_head(eff_val_X)
        val_preds = val_logits.argmax(dim=1)
        tp = ((val_preds == 1) & (val_y == 1)).sum().item()
        fp = ((val_preds == 1) & (val_y == 0)).sum().item()
        fn = ((val_preds == 0) & (val_y == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        acc = (val_preds == val_y).float().mean().item()

    if f1 > best_eff_f1:
        best_eff_f1 = f1
        best_eff_state = {k: v.cpu().clone() for k, v in eff_model.state_dict().items()}
        print(f"  Epoch {epoch:3d} | Acc: {acc*100:.1f}% | Sens: {recall*100:.1f}% | F1: {f1:.4f} -> BEST")

eff_weights_path = r"E:\FYP\backend\weights\efficientnetv2_breast.pth"
# Backup old efficientnet weights
if not os.path.exists(r"E:\FYP\backend\weights\efficientnetv2_breast_backup.pth"):
    import shutil
    shutil.copyfile(eff_weights_path, r"E:\FYP\backend\weights\efficientnetv2_breast_backup.pth")

torch.save(best_eff_state, eff_weights_path)
print(f"EfficientNetV2 Saved! Best Validation F1: {best_eff_f1:.4f}")

# -------------------------------------------------------------
# 5. Test Verification on Malignant, Benign, and Normal
# -------------------------------------------------------------
resnet_model.load_state_dict(best_res_state)
resnet_model.eval()

eff_model.load_state_dict(best_eff_state)
eff_model.eval()

def test_image(path, name):
    img = cv2.imread(path)
    tensor = transform(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"].unsqueeze(0)
    with torch.no_grad():
        r_logits = resnet_model(tensor)
        e_logits = eff_model(tensor)
        r_probs = torch.softmax(r_logits, dim=1)
        e_probs = torch.softmax(e_logits, dim=1)
        # 60/40 ensemble
        ens_probs = 0.40 * r_probs + 0.60 * e_probs
        diag = "Malignant" if ens_probs.argmax(1).item() == 1 else "Benign"
        conf = ens_probs.max().item() * 100
        r_pred = "Malignant" if r_logits.argmax(1).item() == 1 else "Benign"
        e_pred = "Malignant" if e_logits.argmax(1).item() == 1 else "Benign"
    print(f"  {name:20s} -> Final: {diag:9s} ({conf:5.1f}%) | ResNet: {r_pred:9s} ({r_probs.max().item()*100:5.1f}%) | EffNet: {e_pred:9s} ({e_probs.max().item()*100:5.1f}%)")

print("\n" + "=" * 60)
print("TEST VERIFICATION ACROSS ALL DIAGNOSTIC CLASSES:")
print("=" * 60)
test_image(r"E:\FYP\datasets\BUSI\Dataset_BUSI_with_GT\malignant\malignant (16).png", "malignant (16)")
test_image(r"E:\FYP\datasets\BUSI\Dataset_BUSI_with_GT\benign\benign (1).png", "benign (1)")
test_image(r"E:\FYP\datasets\BUSI\Dataset_BUSI_with_GT\normal\normal (1).png", "normal (1)")
test_image(r"E:\FYP\datasets\BUSI\Dataset_BUSI_with_GT\normal\normal (2).png", "normal (2)")
test_image(r"E:\FYP\datasets\BUSI\Dataset_BUSI_with_GT\normal\normal (10).png", "normal (10)")
print("=" * 60)
