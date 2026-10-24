# 🧬 UltraXAI — Explainable AI for Breast Cancer Detection from Ultrasound

**👨‍💻 Developed by: Abdul Rafay** — the complete Final Year Project (research, models, backend, frontend and documentation) was designed and developed by Abdul Rafay.

> **Final Year Project**: Automated Multi-Model Hybrid Ensemble Architecture for Breast Cancer Detection and Classification

## Overview

UltraXAI is a comprehensive AI-powered diagnostic system that detects, segments, classifies, and explains breast cancer findings from ultrasound images. The system features a **dual-track architecture** for both clinical deployment and academic benchmarking.

### Pipeline Flow
```
Upload Ultrasound Image → Preprocessing → Detection (YOLO) → Segmentation (Attention U-Net)
→ Classification (ResNet50 + EfficientNetV2B0 Ensemble) → Explainability (Grad-CAM++ & SHAP)
→ Clinical Dashboard with Diagnosis
```

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **AI/ML** | PyTorch, PyTorch Lightning, Ultralytics (YOLOv8) |
| **Backend** | FastAPI, WebSocket |
| **Frontend** | React 18, Vite, Framer Motion |
| **Preprocessing** | OpenCV, Albumentations, CLAHE |
| **XAI** | Grad-CAM++, SHAP, Captum |
| **Training** | Google Colab Pro (GPU) |

## Project Structure

```
FYP/
├── backend/          # FastAPI model serving
├── frontend/         # React clinical dashboard
├── training/         # PyTorch training pipeline
├── notebooks/        # Google Colab notebooks
├── evaluation/       # Benchmarking & evaluation
├── datasets/         # Training datasets
└── documents/        # Research & methodology docs
```

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 🔁 Data Augmentation & Dataset Expansion

Breast ultrasound datasets are small, so UltraXAI uses two levels of augmentation:

1. **Offline expansion** (`training/data/augment_offline.py`) — writes *N* augmented copies of every
   image (and its mask) to disk, multiplying dataset volume (e.g. `--copies 5` → **6× data**).
   Ultrasound-safe ops: flips, shift/scale/rotate, elastic & grid distortion, speckle
   (multiplicative) noise, Gaussian noise/blur, brightness/contrast, gamma, CLAHE, sharpening.
2. **Online augmentation** (`training/data/transforms.py`) — random transforms every epoch, plus
   coarse dropout, mixup and label smoothing during training.

```bash
cd training
# Expand ONLY the training data (never val/test, to avoid leakage)
python -m data.augment_offline --input_dir ../datasets/BUSI_train --output_dir ../datasets/BUSI_aug --copies 5
```

### ⚡ One-click training (local GPU)

```bash
pip install -r backend/requirements.txt albumentations scikit-learn
python download_busi.py            # or place BUSI in datasets/BUSI
python run_full_training.py --data_dir datasets/BUSI --copies 5 --epochs 80
```
Splits 70/15/15 → augments train only → trains both classifiers → saves weights to `backend/weights/`.

## 📈 Accuracy Improvements

`training/train_classifiers.py` now includes:

| Technique | Benefit |
|-----------|---------|
| Stratified train/val split | Fixes val-equals-train leakage → honest metrics |
| Offline-augmented data (`--aug_dirs`) | Larger, more diverse training set |
| Label smoothing (`--label_smoothing 0.1`) | Better calibration, less over-confidence |
| Mixup (`--mixup_alpha 0.2`) | Stronger regularisation on small data |
| Warmup + cosine LR schedule | Stable convergence of pretrained backbones |
| AdamW weight decay `1e-4` + grad clipping | Less overfitting, stable training |
| Mixed precision (AMP) on GPU | Faster training, larger batches |
| Flip test-time augmentation (TTA) | More robust validation predictions |

```bash
python train_classifiers.py --data_dirs ../datasets/BUSI_train --aug_dirs ../datasets/BUSI_aug \
    --epochs 80 --batch_size 16 --lr 1e-4
```

## 📝 Changelog — Latest Update

| Area | Change |
|------|--------|
| Data | Offline augmentation script (`training/data/augment_offline.py`) — up to 6× dataset volume |
| Data | New ultrasound augmentations: speckle noise, gamma, sharpen |
| Training | Fixed validation leakage (val was identical to train) with stratified split |
| Training | Label smoothing, mixup, warmup + cosine LR, AdamW, grad clipping, AMP, flip TTA |
| Training | Fixed bug where a model was never saved if its first F1 was 0 |
| Compatibility | Updated to Albumentations 2.x API (`Affine`, `GaussNoise(std_range)`, `CoarseDropout` ranges) |
| Pipeline | `run_full_training.py` — one command: split 70/15/15 → augment → train → save weights |
| Data | BUSI training set expanded **452 → 2,712 images (6×)**: 305→1,830 benign, 147→882 malignant; val 97 + test 98 kept unseen |
| Training | Uses the separate `val/` split (`--val_dirs`) so no augmented copy of a validation image is trained on |
| Docs | `documents/UltraXAI_Update_Report.pdf` summarising all updates |

## Datasets

| Dataset | Images | Download |
|---------|--------|----------|
| BUSI | 780 | [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
| BUS-BRA | 1,875 | [Zenodo](https://zenodo.org/records/8231412) |
| UDIAT | 163 | [Mendeley](https://data.mendeley.com/datasets/wmy84gzngw/1) |
| STU | 756 | [Kaggle](https://www.kaggle.com/datasets/mohammedtgadallah/stu-dataset) |
| OASBUD | 200 | [Zenodo](https://zenodo.org/records/545928) |

## License

This project is for academic purposes (Final Year Project).

---

**Developed by Abdul Rafay** — UltraXAI Final Year Project (FYP).
