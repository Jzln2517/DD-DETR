# DD-DETR  
**Enhanced Object Detection in UAV Imagery via Dual-Frequency and Dynamic Attention Mechanisms**

## 📌 Introduction

Object detection in UAV imagery faces significant challenges, including:

- Large scale variations  
- Complex backgrounds  
- Low-light and low-contrast conditions  

To address these issues, we propose **DD-DETR**, a transformer-based detection framework integrating:

- **Low-light Feature Integration (LFI)**  
- **Dual-Frequency Enhancement Module (DFEM)**  
- **Dynamic Sparse Attention (DSA)**  

The proposed design improves small-object perception, foreground-background separation, and computational efficiency for UAV detection scenarios.

---

## 🧠 Framework Overview

DD-DETR introduces three core components:

### 1️⃣ Low-light Feature Integration (LFI)
Enhances edge and texture representation under adverse illumination.

### 2️⃣ Dual-Frequency Enhancement Module (DFEM)
Decomposes feature maps into complementary frequency components to enhance foreground-background discrimination.

### 3️⃣ Dynamic Sparse Attention (DSA)
Selectively models informative regions to reduce redundancy and improve efficiency.

---

## 📂 Project Structure
DD-DETR/main  
│  
├── configs/           # Training configuration files  
├── dataset/           # Dataset processing scripts  
├── models/            # Model architecture (LFI, DFEM, DSA)  
├── utils/             # Utilities (metrics, visualization, tools)  
├── train.py           # Training entry point  
├── val.py             # Validation and inference entry point  
├── requirements.txt   # Dependency list  


---

## ⚙️ Requirements

- Python ≥ 3.9  
- PyTorch ≥ 1.12  
- TorchVision ≥ 0.13  
- CUDA ≥ 11.x  
- pycocotools  
- timm  
- einops  

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Datasets

We evaluate DD-DETR on the following UAV datasets:

- **VisDrone2019**
- **UAVDT**

Please download the datasets from their official websites.

### Dataset Directory Structure

```
data/
├── VisDrone/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
└── UAVDT/
    ├── images/
    │   ├── train/
    │   ├── val/
    │
    └── labels/
        ├── train/
        ├── val/
```

After downloading, modify the dataset path in the corresponding config file if necessary.

---

## 🚀 Training

```bash
python train.py 
```
The network architecture is initialized from the structured configuration file without loading any macro-dataset pretrained weights, and then runs a training-from-scratch pipeline.
---

## 🔎 Validation and Inference

```bash
python val.py 
```
Detection results will be saved in:
```
runs/detect/
```

---

## 📈 Experimental Results

DD-DETR achieves:

- Improved AP and AP50 on VisDrone2019  
- Significant gains in small-object detection  
- Competitive real-time inference performance  

Detailed quantitative comparisons are provided in the paper.

---

## 🧪 Reproducibility

To reproduce the reported results:

1. Install the required dependencies  
2. Prepare datasets following the structure above  
3. Download pretrained weights (see Releases section)  
4. Run validation using the provided config files  

All hyperparameters and training settings are included in the configuration files.

---

## 📦 Pretrained Models

Pretrained weights will be released in the **Releases** section of this repository.

For long-term accessibility, a DOI link (via Zenodo) will also be provided.

---

## 📜 Citation

If you use this code in your research, please cite:

```bibtex
@article{jing2026dd-detr,
  title={Enhanced Object Detection in UAV Imagery via Dual-Frequency and Dynamic Attention Mechanisms},
  author={Jing, Zihao and Wang, Shaoqing and Tang, Weiyan and Zhu, Zhuangrui and Sun, Fuzhen},
  journal={The Visual Computer},
  year={2026}
}
```

---

## 🔗 Resources

- Code: https://github.com/Jzln2517/DD-DETR  
- DOI: To be added  

---

## 📧 Contact

Zihao Jing  
Email: (jzh2391310907@qq.com)
  
Shaoqing Wang  
Email: (wsq0533@163.com)

