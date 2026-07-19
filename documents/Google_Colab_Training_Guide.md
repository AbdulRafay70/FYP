# ☁️ Google Colab Training Guide — UltraXAI

Since you are not using local hardware accelerators, Google Colab is the recommended environment for training the UltraXAI models (YOLOv8-Seg, Attention U-Net, and the Ensembles).

This guide explains how to upload your code to Colab and execute the training scripts we've created.

## Prerequisites
1. A Google Account with Google Drive access.
2. The `e:\FYP\training` directory uploaded to your Google Drive.
3. Your datasets downloaded and uploaded to Google Drive.

---

## Step 1: Upload Your Project to Google Drive

1. Open Google Drive.
2. Create a new folder named `UltraXAI`.
3. Upload the entire `training` folder from your PC (`e:\FYP\training`) into the `UltraXAI` folder.
4. Upload your datasets into `UltraXAI/datasets/`. For example, `UltraXAI/datasets/BUSI`.

## Step 2: Create a Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **New Notebook**.
3. In the top menu, go to **Runtime > Change runtime type**.
4. Select **T4 GPU** or **A100 GPU** (if you have Colab Pro) and click Save.

## Step 3: Mount Google Drive

In your first notebook cell, run this code to mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 4: Install Dependencies

In the next cell, install the required libraries:

```bash
!pip install torch torchvision
!pip install ultralytics albumentations opencv-python-headless scikit-learn
```

## Step 5: Run the Training Scripts

You can train each phase by running the Python scripts directly from your Google Drive.

### 🟢 Phase 1: Train YOLOv8-Seg (Detection)

First, let the script convert your dataset to YOLO format and train:

```bash
# Navigate to the training directory
%cd /content/drive/MyDrive/UltraXAI/training

# Run the training script (point it to your dataset)
# The script will auto-generate the data.yaml file and YOLO format
!python train_yolo.py --data datasets/yolo_format/data.yaml --epochs 100
```
*Note: Ensure you update the path if your dataset is located elsewhere.*

### 🔵 Phase 2: Train Attention U-Net (Segmentation)

```bash
%cd /content/drive/MyDrive/UltraXAI/training

# Provide the paths to the dataset folders (e.g., BUSI, BUS-BRA)
!python train_attention_unet.py \
    --data_dirs datasets/BUSI datasets/BUS-BRA \
    --epochs 150 \
    --batch_size 16 \
    --lr 1e-4 \
    --output_dir /content/drive/MyDrive/UltraXAI/weights
```

### 🟣 Phase 3: Train Classifier Ensemble (ResNet50 + EfficientNetV2)

```bash
%cd /content/drive/MyDrive/UltraXAI/training

# Train both ResNet50 and EfficientNetV2-B0
!python train_classifiers.py \
    --data_dirs datasets/BUSI datasets/BUS-BRA \
    --epochs 80 \
    --batch_size 32 \
    --lr 1e-4 \
    --output_dir /content/drive/MyDrive/UltraXAI/weights
```

---

## Step 6: Download the Weights

Once the training scripts complete, the model weights will be saved in your Google Drive under `UltraXAI/weights/`. 

You will see these files:
- `yolov8m_seg_breast.pt` (usually inside a `runs/segment/.../weights/best.pt` folder for YOLO)
- `attention_unet_breast.pth`
- `resnet50_breast.pth`
- `efficientnetv2_breast.pth`

**Download these `.pt` and `.pth` files and place them in your local backend directory:**
`e:\FYP\backend\weights\`

After placing the weights there, the FastAPI backend will automatically load them on startup!
