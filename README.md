# 🧬 UltraXAI — Explainable AI for Breast Cancer Detection from Ultrasound

**👨‍💻 Developed by: Abdul Rafay** — the complete Final Year Project (research, models, backend, frontend and documentation) was designed and developed by Abdul Rafay.

> **Final Year Project**: Automated Multi-Model Hybrid Ensemble Architecture for Breast Cancer Detection and Classification

## Overview

UltraXAI is a comprehensive AI-powered diagnostic system that detects, segments, classifies, and explains breast cancer findings from ultrasound images. The system features a **dual-track architecture** for both clinical deployment and academic benchmarking.

### Pipeline Flow
```
Upload Ultrasound Image → Preprocessing → Detection (YOLO) → Segmentation (Attention U-Net)
→ Classification (ResNet50 + EfficientNetV2-S, weighted soft voting 0.40/0.60)
→ Explainability (ensemble Grad-CAM++) → Clinical Dashboard
```

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **AI/ML** | PyTorch, torchvision, Ultralytics (YOLOv8) — plain PyTorch training loops (no PyTorch Lightning) |
| **Backend** | FastAPI, Uvicorn, WebSocket (`/api/v1/ws/progress` streams real per-phase progress) |
| **Frontend** | React 19.1, Vite 6, Framer Motion 12, Axios (REST; the UI does not use the WebSocket yet) |
| **Preprocessing** | OpenCV, Albumentations 2.x |
| **XAI** | Grad-CAM++ (own implementation, `training/utils/explainability.py`). SHAP and Captum are **not** used (SHAP = future work) |
| **Training** | Local GPU via `run_full_training.py` (the exact GPU is recorded in `*_config.json` of each run) |

### Models (as implemented)

| Phase | Model | Input | Checkpoint (`backend/weights/`) |
|---|---|---|---|
| Detection | YOLOv8m-seg (Ultralytics, COCO-pretrained) | 640×640 | `yolov8m_seg_breast.pt` |
| Segmentation | Attention U-Net (1→1 ch, features 64-128-256-512, bottleneck 1024) | 256×256 grayscale, whole image | `attention_unet_breast.pth` |
| Classification | ResNet50 (ImageNet1K_V2, conv1…layer2 frozen) | 224×224 RGB lesion ROI | `resnet50_breast.pth` |
| Classification | **EfficientNetV2-S** (`torchvision.efficientnet_v2_s`, ImageNet1K_V1, features[0:5] frozen) | 224×224 RGB lesion ROI | `efficientnetv2_breast.pth` |
| Ensemble | Weighted soft voting: 0.40 × ResNet50 + 0.60 × EfficientNetV2-S probabilities | — | — |
| Explainability | Grad-CAM++ on ResNet50 `layer4[-1]` and EfficientNetV2-S `features[-1]`, fused with the ensemble weights, for the ensemble's predicted class | — | — |

> Earlier code and docs called the second classifier "EfficientNetV2-B0". The network has always been
> **EfficientNetV2-S**; the class is now `EfficientNetV2SClassifier` (old name kept as an alias).

### Preprocessing (identical in training and inference)
Grayscale decode → (classifiers only) lesion ROI crop = mask/box + 15 % padding → resize (bilinear) to the
model input size → `Normalize` (U-Net: mean 0.485 / std 0.229; classifiers: ImageNet mean/std on the
grayscale image replicated to 3 channels). CLAHE and denoising in `backend/app/preprocessing/image_pipeline.py`
are **not** applied at inference because the models were not trained on enhanced images; CLAHE is only used as a
random training augmentation.

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
   image (and its mask) to disk, multiplying dataset volume (e.g. `--copies 5` → 5 augmented copies + the original = **6× data**).
   Ultrasound-safe ops: flips, shift/scale/rotate, elastic & grid distortion, speckle
   (multiplicative) noise, Gaussian noise/blur, brightness/contrast, gamma, CLAHE, sharpening.
2. **Online augmentation** (`training/data/transforms.py`) — random transforms every epoch on the training
   set only (validation/test get resize + normalise only), plus mixup (α = 0.2) and label smoothing (0.1)
   in the classifier loss.

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
Splits the **original** images 70/15/15 (stratified, seed 42) → writes `datasets/prepared/split_manifest.csv/.json`
→ augments the **train** split only → verifies that no image or augmented variant is shared between
train/val/test (`training/data/manifest.py`, by source id and pixel hash; training aborts on a leak) →
trains both classifiers on lesion ROI crops → (`--train_unet`) trains the Attention U-Net on the same split.
Each model writes `<model>_config.json` (hyper-parameters, seed, environment, GPU) and `<model>_log.csv`
(epoch-wise train/val metrics) next to its checkpoint.

### 🧪 Test-set evaluation (Chapter 4 evidence)

```bash
python evaluation/evaluate_test_set.py --test_dir datasets/prepared/test --weights_dir backend/weights \
    --gradcam_samples 6 [--yolo_data datasets/yolo_format/data.yaml]
```
Evaluates ResNet50, EfficientNetV2-S and the ensemble on the **same held-out test images**, writes raw
predictions, TP/TN/FP/FN, accuracy/precision/sensitivity/specificity/F1/NPV/ROC-AUC, one confusion matrix
per model, ROC curves, U-Net Dice/IoU/pixel accuracy/Hausdorff on the test masks, optional YOLO
precision/recall/mAP@0.5/mAP@0.5:0.95, Grad-CAM++ sample figures and a run record (git commit, checkpoint
SHA-256, split-manifest SHA-256). `/api/v1/benchmark` serves these numbers; nothing is hard-coded.

> ⚠️ `confusion_matrix_binary.png` / `confusion_matrix_3class.png` (250 images, incl. 50 Normal) and the
> 96.4 % / 96.5 % / 99.30 % figures in older reports come from earlier experiments (`train_both_with_normal.py`,
> `eval_individual.py` — the latter evaluates on the whole dataset, i.e. with leakage). They are **not** final
> test results. Report only the output of `evaluate_test_set.py`.

### 📊 Dataset Volume (BUSI, actual run)

| | Benign | Malignant | Total |
|---|---|---|---|
| Original BUSI | 437 | 210 | 647 |
| Train split (original) | 305 | 147 | 452 |
| + Augmented copies (5 per image) | 1,525 | 735 | 2,260 |
| **Total training images** | **1,830** | **882** | **2,712** |
| Validation (unseen) | 66 | 31 | 97 |
| Test (unseen) | 66 | 32 | 98 |

Training data grew **452 → 2,712 images (6×, +2,260)**. Validation and test images are never augmented or trained on.

### ✅ Test-set results (98 held-out images, run of 18 Nov 2026)

Copied unrounded from [`evaluation/results/metrics.json`](evaluation/results/metrics.json). Positive class = Malignant.

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUC-ROC | TP / TN / FP / FN |
|---|---|---|---|---|---|---|---|
| ResNet50 | 0.8571428571428571 | 0.90625 | 0.8333333333333334 | 0.725 | 0.8055555555555556 | 0.9337121212121212 | 29 / 55 / 11 / 3 |
| EfficientNetV2-S | 0.9183673469387755 | 0.90625 | 0.9242424242424242 | 0.8529411764705882 | 0.8787878787878788 | 0.9592803030303031 | 29 / 61 / 5 / 3 |
| Ensemble (0.40 / 0.60) | 0.8979591836734694 | 0.90625 | 0.8939393939393939 | 0.8055555555555556 | 0.8529411764705882 | 0.9734848484848485 | 29 / 59 / 7 / 3 |

| Segmentation (Attention U-Net, epoch 57) | Dice | IoU | Pixel accuracy | Hausdorff |
|---|---|---|---|---|
| Test masks | 0.7316451072692871 | 0.6352987885475159 | 0.9486442194094045 | 105.04971736215042 |

YOLOv8m-Seg detection was not quantitatively evaluated (no `data.yaml` for the test split).

| Training (Quadro T1000, PyTorch 2.6.0, CUDA 12.4) | Precision | Epochs run | Best validation epoch |
|---|---|---|---|
| ResNet50 | AMP (float16) | 26 (early stop, patience 15) | 11: F1 0.84375, AUC 0.936461 |
| EfficientNetV2-S | float32 (float16 gave NaN loss) | 53 (early stop, patience 15) | 38: F1 0.870968, AUC 0.912512 |
| Attention U-Net | float32, 452 original train images | 74 (patience 20, max 150) | 57: Dice 0.67948776, IoU 0.5943793 |

Train a single classifier or disable AMP with `--models efficientnetv2 --no_amp efficientnetv2`.

## 📈 Accuracy Improvements

`training/train_classifiers.py` now includes:

| Technique | Benefit |
|-----------|---------|
| Separate train / val / test partitions | Fixes val-equals-train leakage; test is never seen in training |
| Lesion ROI crops (`roi_crop`) | Classifiers train on the same kind of input the backend gives them |
| Offline-augmented data (`--aug_dirs`) | Larger, more diverse training set |
| Label smoothing (`--label_smoothing 0.1`) | Better calibration, less over-confidence |
| Mixup (`--mixup_alpha 0.2`) | Stronger regularisation on small data |
| Warmup + cosine LR schedule | Stable convergence of pretrained backbones |
| AdamW weight decay `1e-4` (classifiers) + grad clipping 1.0 | Less overfitting, stable training |
| Mixed precision (AMP) on GPU | Faster training, larger batches |
| Flip TTA (validation only) | Used for model selection; test evaluation and the backend use no TTA |

```bash
python train_classifiers.py --data_dirs ../datasets/prepared/train --aug_dirs ../datasets/prepared/train_aug \
    --val_dirs ../datasets/prepared/val --epochs 80 --batch_size 16 --lr 1e-4
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
| Thesis corrections | Split manifest + automated leakage check; U-Net uses real train/val partitions; raw-data training holds out the test split |
| Thesis corrections | Classifiers train on lesion ROI crops (same as backend); U-Net runs on whole images (same as training) |
| Thesis corrections | EfficientNetV2-B0 → EfficientNetV2-S naming; ensemble-weighted Grad-CAM++; SHAP/Captum/Lightning claims removed |
| Thesis corrections | "No lesion detected" instead of 100 % Benign on empty masks; honest model status; real WebSocket progress |
| Thesis corrections | `evaluation/evaluate_test_set.py` + per-run config/epoch logs; see `documents/UltraXAI_Thesis_Correction_Report.docx` |

### Backend behaviour notes
* If neither YOLO nor the U-Net finds a lesion, the classifier is **not** run and the API returns
  `"No lesion detected"` (trust `N/A`) — the binary classifier was never trained on Normal images, so it
  must not output a confident "Benign".
* `/api/v1/models/status` returns `ready` / `degraded` / `no_models_loaded` and lists missing checkpoints.
* Track B (MedSAM + Swin-V2/ConvNeXt-V2) is **not implemented**; `/diagnose/track-b` is a placeholder.

## Datasets

| Dataset | Images | Download |
|---------|--------|----------|
| BUSI | 780 | [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
| BUS-BRA | 1,875 | [Zenodo](https://zenodo.org/records/8231412) |
| UDIAT | 163 | [Mendeley](https://data.mendeley.com/datasets/wmy84gzngw/1) |
| STU | 756 | [Kaggle](https://www.kaggle.com/datasets/mohammedtgadallah/stu-dataset) |
| OASBUD | 200 | [Zenodo](https://zenodo.org/records/545928) |
| **Total (raw)** | **3,774** | |

Only **BUSI** (Al-Dhabyani et al., 2020) is used in the experiments. BUS-BRA, UDIAT, STU and OASBUD are
listed as candidate datasets for future external validation and were **not** used.

**Currently used for training:** BUSI only: 647 benign + malignant images (the 133 normal images are excluded). After augmentation the model trains on **2,712 images**.

**If all 5 datasets are used** (estimate): about 2,640 training images (70% of 3,774) × 6 with augmentation ≈ **15,800 training images**.

> **Note:** 3,774 is the raw total of all 5 datasets **before** augmentation. It is **not** the training count.
>
> | | Images |
> |---|---|
> | Raw total, all 5 datasets (before augmentation) | 3,774 |
> | Current training images after augmentation (BUSI only: 452 + 2,260 copies) | **2,712** |
> | Estimated training images after augmentation if all 5 datasets are used (~2,640 × 6) | **~15,800** |

## License

This project is for academic purposes (Final Year Project).

---

**Developed by Abdul Rafay** — UltraXAI Final Year Project (FYP).
