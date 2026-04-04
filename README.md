# SHBT261 Mini-Project 2 — Semantic Segmentation on Pascal VOC 2007

Semantic segmentation with three models: **U-Net (ResNet-18)**, **U-Net (ResNet-50)**, and **DeepLabV3 (ResNet-50)**, trained on Pascal VOC 2007.

---

## Requirements

```bash
pip install torch torchvision          # PyTorch (CUDA recommended)
pip install albumentations             # data augmentation
pip install scipy scikit-learn tqdm    # metrics and utilities
pip install matplotlib                 # visualisation
pip install "numpy<2"                  # required for matplotlib compatibility
```

Tested on: Python 3.9, PyTorch 2.8.0+cu128, CUDA 12.8, NVIDIA RTX 5070 Ti.

---

## 1. Download the Dataset

Download the Pascal VOC 2007 dataset from Kaggle:

**URL:** https://www.kaggle.com/datasets/zaraks/pascal-voc-2007

After downloading, extract so that the folder structure looks like this:

```
SHBT261-miniProject2/
├── VOCtrainval_06-Nov-2007/
│   └── VOCdevkit/
│       └── VOC2007/
│           ├── JPEGImages/
│           ├── SegmentationClass/
│           └── ImageSets/
│               └── Segmentation/
│                   ├── train.txt
│                   └── val.txt
├── VOCtest_06-Nov-2007/      ← not used, can be ignored
├── main.py
├── ...
```

> The dataset path is configured in `config.py` as `DATA_ROOT = "./VOCtrainval_06-Nov-2007"`. If you place the data elsewhere, update that line.

---

## 2. (Optional) SAM Model

To use the SAM-based model, install the package and download the ViT-B checkpoint:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Then download `sam_vit_b_01ec64.pth` and place it in the project root:

```
SHBT261-miniProject2/
└── sam_vit_b_01ec64.pth
```

> SAM checkpoint path is set in `config.py` as `SAM_CHECKPOINT = "./sam_vit_b_01ec64.pth"`.

---

## 3. File Overview

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters, paths, class names, VOC colormap |
| `dataset.py` | Dataset loader with albumentations augmentation |
| `models.py` | U-Net (ResNet-18/50), DeepLabV3, SAM model definitions |
| `losses.py` | CrossEntropy, Dice, and Combined loss |
| `train.py` | Training loop, evaluation, checkpoint saving |
| `metrics.py` | mIoU, mDice, pixel accuracy, HD95, confusion matrix |
| `visualize.py` | Mosaic, best/worst, confusion matrix, training curve plots |
| `ablation.py` | Ablation study runners (backbone, augmentation, loss, pre-training) |
| `main.py` | Main CLI entry point |
| `voc2007.py` | Standalone dataset exploration script (from course materials) |

---

## 4. Reproduce Results

### Step 1 — Train all three models (60 epochs each)

```bash
python main.py --models unet18 unet50 deeplabv3plus --epochs 60
```

This trains all three models sequentially and:
- Saves best checkpoints to `checkpoints/<model>_best.pth`
- Saves final checkpoints to `checkpoints/<model>_final.pth`
- Saves visualisations to `results/`

Expected training time on RTX 5070 Ti: ~5 min per model (60 epochs, 209 images).

---

### Step 2 — Evaluate and generate all figures

```bash
python main.py --models unet18 unet50 deeplabv3plus --eval-only
```

Loads best checkpoints and produces:
- Full metrics (mIoU, mDice, pixel accuracy, HD95, per-class IoU)
- `results/unet18_mosaic.png`, `results/unet50_mosaic.png`, `results/deeplabv3plus_mosaic.png`
- `results/<model>_best3.png`, `results/<model>_worst3.png` (ranked by person IoU)
- `results/<model>_confusion.png` (normalised confusion matrix)
- `results/per_class_iou_comparison.png` (grouped bar chart across models)
- `results/training_curves.png` (loss / mIoU / mDice curves)

---

### Step 3 — Run ablation studies (30 epochs each)

```bash
python ablation.py --ablations all --epochs 30
```

Runs four ablations and saves bar charts + training curves to `results/`:

| Ablation | What it compares |
|----------|-----------------|
| `backbone` | U-Net ResNet-18 vs ResNet-50 |
| `augment` | Training with vs without data augmentation |
| `loss` | Cross-entropy vs Dice vs Combined loss |
| `pretrain` | ImageNet pretrained vs random initialization |

To run a single ablation:

```bash
python ablation.py --ablations backbone --epochs 30
python ablation.py --ablations augment  --epochs 30
python ablation.py --ablations loss     --epochs 30
python ablation.py --ablations pretrain --epochs 30
```

---

## 5. Additional Options

```bash
# Change loss function for main training
python main.py --models unet18 --epochs 60 --loss ce        # cross-entropy only
python main.py --models unet18 --epochs 60 --loss dice      # dice only
python main.py --models unet18 --epochs 60 --loss combined  # default

# Disable data augmentation
python main.py --models unet18 --epochs 60 --no-augment

# Include SAM model (requires checkpoint + package)
python main.py --models unet18 deeplabv3plus sam --epochs 60

# Run ablations as part of main pipeline
python main.py --models unet18 --epochs 60 --ablations all --ablation-epochs 30
```

---

## 6. Output Structure

After running Steps 1–3, your directory will look like:

```
SHBT261-miniProject2/
├── checkpoints/
│   ├── unet18_best.pth
│   ├── unet18_final.pth
│   ├── unet50_best.pth
│   ├── unet50_final.pth
│   ├── deeplabv3plus_best.pth
│   └── deeplabv3plus_final.pth
├── results/
│   ├── training_curves.png
│   ├── per_class_iou_comparison.png
│   ├── unet18_mosaic.png
│   ├── unet18_best3.png
│   ├── unet18_worst3.png
│   ├── unet18_confusion.png
│   ├── unet50_*.png
│   ├── deeplabv3plus_*.png
│   ├── ablation_backbone_miou.png
│   ├── ablation_backbone_curves.png
│   ├── ablation_aug_miou.png
│   ├── ablation_aug_curves.png
│   ├── ablation_loss_miou.png
│   ├── ablation_loss_curves.png
│   ├── ablation_pretrain_miou.png
│   └── ablation_pretrain_curves.png
```

> `checkpoints/` and `results/` are git-ignored. Re-run the steps above to regenerate them.

---

## 7. Key Results (Reproduced)

| Model | mIoU | mDice | Pixel Acc | HD95 |
|-------|------|-------|-----------|------|
| U-Net ResNet-18 | 0.3435 | 0.4510 | 85.5% | 25.15 |
| U-Net ResNet-50 | 0.2571 | 0.3408 | 84.3% | 25.92 |
| DeepLabV3 ResNet-50 | **0.5657** | **0.7028** | **90.2%** | 26.83 |

See `report.md` for full analysis, per-class breakdown, and ablation results.
