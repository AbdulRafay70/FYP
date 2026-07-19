# Recommended Laptop Specifications for Developing and Running the FYP Pipeline

This document details the exact hardware specifications required to train, run, and display the complete multi-model AI/ML pipeline on a single laptop without needing external edge hardware.

---

## 💻 1. Optimal Laptop Specifications (Recommended)

To run **YOLOv8**, **Attention U-Net**, and **Ensembles** smoothly with sub-second inference times and fast training epochs, aim for a gaming or high-end creator laptop with the following specs:

| Component | Minimum Required Specs | Optimal / Recommended Specs | Why it is Needed |
| :--- | :--- | :--- | :--- |
| **Graphics Card (GPU)** | **NVIDIA RTX 3060 / 4050** (6 GB VRAM) | **NVIDIA RTX 4060 / 4070 / 4080** (8 GB or 12 GB VRAM) | **Critical for AI.** Enables CUDA acceleration for training models and sub-second inference times. |
| **System Memory (RAM)** | 16 GB DDR4/DDR5 | **32 GB DDR5** | To load entire datasets and run multiple neural networks simultaneously without crashing. |
| **Processor (CPU)** | Intel Core i7 or AMD Ryzen 7 | **Intel Core i9 (13th/14th Gen) or AMD Ryzen 9** | For rapid preprocessing, file parsing, and CPU-to-GPU data pipelines. |
| **Storage** | 512 GB SSD | **1 TB or 2 TB NVMe M.2 SSD** | High-speed data loading for large medical images. |
| **Operating System** | Windows 11 (64-bit) | Windows 11 or Ubuntu Linux | For seamless Python 3.11/3.12 and PyTorch integration. |

---

## 🛍️ 2. Top Recommended Laptop Models in the Market

If you are buying or selecting a laptop for your project, look for the following creator and gaming series:

### 1. High-Performance / Premium Option
- **Lenovo Legion Pro 5i / 7i** (with Intel i9, 32GB RAM, NVIDIA RTX 4070 or 4080).
- **ASUS ROG Zephyrus G16** (with AMD Ryzen 9, 32GB RAM, NVIDIA RTX 4070).

### 2. Balanced / Cost-Effective Option
- **Acer Predator Helios 16** (with Intel i7, 16GB or 32GB RAM, NVIDIA RTX 4060).
- **HP Omen 16** (with AMD Ryzen 7, 16GB RAM, NVIDIA RTX 4060).

---

## ☁️ 3. Alternative: Running on Cloud GPUs (Kaggle / Google Colab)

If your laptop does not have a dedicated NVIDIA GPU, you can still develop your project using cloud alternatives:
1. **Model Training**: Use the **Free GPU environment** inside **Kaggle Notebooks (NVIDIA P100 / T4)** or **Google Colab (T4 GPU)** to train your models.
2. **Inference & UI (Local Laptop)**: Download the trained model weights (`.pt` or `.pth` files) onto your local laptop. Your laptop can run the evaluation and the web interface using just its CPU.
