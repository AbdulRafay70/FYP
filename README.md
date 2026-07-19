# 🧬 UltraXAI — Explainable AI for Breast Cancer Detection from Ultrasound

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
