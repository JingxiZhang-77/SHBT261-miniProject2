# Mini-Project 2: Semantic Segmentation on Pascal VOC 2007

**Course:** SHBT 261  
**Dataset:** Pascal VOC 2007 (Segmentation)  
https://github.com/JingxiZhang-77/SHBT261-miniProject2

## 1. Introduction

Semantic segmentation assigns a class label to every pixel in an image. This project trains and evaluates three segmentation models on the Pascal VOC 2007 dataset, which contains 21 classes (20 objects + background) across 209 training and 213 validation images. We use the validation set as our test set throughout. The three models compared are a U-Net with ResNet-18 encoder (14.5M parameters), a U-Net with ResNet-50 encoder (32.7M parameters), and DeepLabV3 with ResNet-50 backbone (42.0M parameters). A SAM-based model (`SAMSegmentation`) with a frozen ViT-B encoder and lightweight trainable decoder was also implemented but not benchmarked quantitatively due to a separate checkpoint dependency.

## 2. Methods

### 2.1 Dataset and Preprocessing

The Pascal VOC 2007 segmentation split was loaded via `torchvision.datasets.VOCSegmentation`. Mask pixel values range from 0–20 (class index) and 255 (boundary/ignore); boundary pixels are excluded from all loss and metric computations (`ignore_index=255`). Training augmentations included `RandomResizedCrop(256×256, scale=(0.4, 1.0))`, `HorizontalFlip(p=0.5)`, `ColorJitter(brightness/contrast/saturation=0.3, hue=0.1, p=0.6)`, `GaussianBlur(p=0.2)`, and ImageNet normalisation. Validation used only resize and normalise. The DataLoader used batch size 4 with `drop_last=True` to avoid BatchNorm failures in ASPP layers.

### 2.2 Model Architectures

**U-Net (ResNet-18 / ResNet-50).** A classic encoder-decoder with skip connections. The ResNet backbone is split into five stages (enc0–enc4) producing feature maps at strides /2 through /32, with channel widths of 64/64/128/256/512 (ResNet-18) or 64/256/512/1024/2048 (ResNet-50). The decoder uses bilinear upsampling and double conv blocks with skip concatenation at each scale; a final `1×1` conv produces 21-channel logits.

**DeepLabV3 (ResNet-50).** Uses `torchvision.models.segmentation.deeplabv3_resnet50` with ImageNet-pretrained weights. The COCO classifier head is replaced with `Conv2d(256 → 21, 1×1)`. Atrous convolutions and ASPP capture multi-scale context without resolution loss; logits are bilinearly upsampled to input size.

**SAM-based (Not Benchmarked).** Frozen ViT-B encoder produces `(B, 256, 64, 64)` features. A trainable decoder upsamples via two bilinear ×2 conv-BN-ReLU stages to `(B, 21, 256, 256)` logits. Only decoder weights are optimised.

### 2.3 Training Setup

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| LR schedule | CosineAnnealingLR (η_min = 1e-6) |
| Epochs | 60 |
| Loss | Combined (0.5 × CE + 0.5 × Dice) |
| Mixed precision | AMP (`torch.amp.GradScaler`) |
| Device | NVIDIA RTX 5070 Ti (17 GB VRAM) |

Three loss functions were evaluated: pixel-wise cross-entropy (CE), soft Dice averaged over classes present in the batch, and a combined equal-weight sum of both. Checkpoints are saved on validation mIoU improvement; final evaluation uses the best checkpoint.

## 3. Evaluation Metrics

Five metrics are reported: mean Intersection-over-Union (mIoU) across 21 classes (NaN-excluded), mean Dice coefficient (mDice), pixel accuracy, 95th-percentile Hausdorff distance (HD95) averaged over samples and classes, and per-class IoU.

## 4. Results

### 4.1 Overall Performance

| Model | Params | mIoU ↑ | mDice ↑ | Pixel Acc ↑ | HD95 ↓ |
|-------|--------|--------|---------|------------|--------|
| U-Net ResNet-18 | 14.5M | 0.3435 | 0.4510 | 0.8549 | **25.15** |
| U-Net ResNet-50 | 32.7M | 0.2571 | 0.3408 | 0.8427 | 25.92 |
| **DeepLabV3 ResNet-50** | **42.0M** | **0.5657** | **0.7028** | **0.9016** | 26.83 |

DeepLabV3 outperforms both U-Net variants by a large margin (+0.222 mIoU over U-Net-18). Notably, the smaller U-Net ResNet-18 outperforms the larger ResNet-50 variant; this overfitting effect is examined in Section 5.

### 4.2 Training Curves

<img src="results/training_curves.png" width="50%">

*Figure 1: Training and validation loss (left), validation mIoU (center), and validation mDice (right) over 60 epochs for DeepLabV3.*

Both training and validation losses decrease steeply in the first ~15 epochs and plateau near 0.3–0.4, tracking closely and indicating no severe overfitting. Validation mIoU rises steadily to ~0.55+ by epoch 60, with mDice reaching ~0.70, consistent with the final reported values. The sustained improvement confirms 60 epochs was an appropriate budget.

### 4.3 Per-class IoU

<img src="results/per_class_iou_comparison.png" width="50%">

*Figure 2: Per-class IoU for all 21 VOC classes across all three models (blue = U-Net-18, brown = U-Net-50, cyan = DeepLabV3).*

DeepLabV3 dominates almost every class. Both U-Nets score zero on bicycle, bottle, pottedplant, and sheep — small or infrequent classes where DeepLabV3's ASPP multi-scale context allows it to break through. The single exception is `cat`, where U-Net-50 (0.84) marginally outperforms DeepLabV3 (0.77), likely because close-up texture-rich images suit ResNet-50's deeper features. `sofa` is the hardest class across all models (max IoU 0.20), consistently confused with chair and diningtable.

| Class | U-Net 18 | U-Net 50 | DeepLabV3 |
|-------|----------|----------|-----------|
| background | 0.8859 | 0.9046 | **0.9096** |
| aeroplane | 0.5605 | 0.0687 | **0.7224** |
| bicycle | 0.0000 | 0.0000 | **0.2935** |
| bird | 0.5649 | 0.2635 | **0.6428** |
| boat | 0.0210 | 0.0017 | **0.3950** |
| bottle | 0.0000 | 0.0000 | **0.6174** |
| bus | 0.5329 | 0.3427 | **0.8042** |
| car | 0.4638 | 0.0000 | **0.6587** |
| cat | 0.7671 | **0.8383** | 0.7747 |
| chair | 0.0973 | 0.0001 | **0.3393** |
| cow | 0.2386 | 0.2794 | **0.4131** |
| diningtable | 0.3288 | 0.1934 | **0.4923** |
| dog | 0.3741 | 0.3909 | **0.4103** |
| horse | 0.3770 | 0.4868 | **0.5819** |
| motorbike | 0.2266 | 0.2180 | **0.7032** |
| person | 0.7100 | 0.7078 | **0.7556** |
| pottedplant | 0.0000 | 0.0000 | **0.3434** |
| sheep | 0.0000 | 0.0000 | **0.4490** |
| sofa | 0.1811 | 0.1829 | **0.1964** |
| train | 0.2949 | 0.0833 | **0.7478** |
| tvmonitor | 0.5891 | 0.4367 | **0.6289** |

### 4.4 Qualitative Results

<img src="results/deeplabv3plus_mosaic.png" width="50%">

*Figure 3: DeepLabV3 predictions on 4 validation images (rows: input, prediction, ground truth). DeepLabV3 produces sharp, accurate boundaries across diverse scene types.*

<img src="results/deeplabv3plus_best3.png" width="50%">

*Figure 4: Top-3 DeepLabV3 predictions by person-class IoU (0.934, 0.885, 0.852). All three feature large, well-lit persons occupying most of the frame.*

<img src="results/deeplabv3plus_worst3.png" width="50%">

*Figure 5: Bottom-3 DeepLabV3 predictions by person-class IoU (all 0.000). In each case the ground-truth person region is a tiny sliver — an arm near a parrot, a figure beside a cow — effectively sub-pixel at 256×256 resolution.*

The best predictions share a common trait: persons are large and centrally positioned. The worst cases illustrate that an IoU of 0.0 is triggered whenever a person exists in the ground truth but is not predicted at all, disproportionately affecting images where persons are very small.

### 4.5 Confusion Matrix

<img src="results/deeplabv3plus_confusion.png" width="50%">

*Figure 6: Normalised confusion matrix for DeepLabV3. Rows = ground truth, columns = predicted.*

The matrix shows a strong diagonal across all classes. Notable off-diagonal patterns include mild person→background confusion (expected for partially occluded persons), quadruped inter-confusion (dog/cat/cow/horse share shape and texture), and sofa confused with chair and diningtable — consistent with its low IoU.

## 5. Ablation Studies

All ablations use U-Net ResNet-18 as the base and train for 30 epochs with combined loss (unless noted). The goal is comparative, not absolute.

### A1 — Backbone Size: ResNet-18 vs ResNet-50

<img src="results/ablation_backbone_miou.png" width="50%">

*Figure 7: A1 — mIoU after 30 epochs: ResNet-18 (0.2380) vs ResNet-50 (0.1339).*

| Backbone | mIoU ↑ | mDice ↑ | Pixel Acc ↑ | HD95 ↓ | Params |
|----------|--------|---------|------------|--------|--------|
| **ResNet-18** | **0.2380** | **0.3332** | **0.8173** | **30.62** | 14.5M |
| ResNet-50 | 0.1339 | 0.1837 | 0.7981 | 33.18 | 32.7M |

ResNet-18 achieves nearly double the mIoU of ResNet-50 after 30 epochs. With only 209 training images, ResNet-50's 32.7M parameters are severely over-parameterised, causing memorisation rather than generalisation. This pattern persists at 60 epochs (0.3435 vs 0.2571), confirming a structural effect rather than slow convergence.

### A2 — Data Augmentation: With vs Without

<img src="results/ablation_aug_miou.png" width="50%">

*Figure 8: A2 — mIoU after 30 epochs: augmentation OFF (0.2853) vs ON (0.2103).*

| Augmentation | mIoU ↑ | mDice ↑ | Pixel Acc ↑ | HD95 ↓ |
|-------------|--------|---------|------------|--------|
| OFF | **0.2853** | **0.3980** | **0.8370** | 33.20 |
| ON  | 0.2103 | 0.2944 | 0.8125 | **29.56** |

Augmentation-OFF wins at 30 epochs because aggressive augmentation creates challenging views that require more epochs to learn from. However, augmentation-ON already achieves a lower HD95 (29.56 vs 33.20), indicating better boundary precision. At 60 epochs augmented U-Net-18 reaches mIoU 0.3435, confirming augmentation pays off with sufficient training time.

### A3 — Loss Function: CE vs Dice vs Combined

<img src="results/ablation_loss_miou.png" width="50%">

*Figure 9: A3 — mIoU after 30 epochs: CE (0.2281), Dice (0.2148), Combined (0.2411).*

| Loss | mIoU ↑ | mDice ↑ | Pixel Acc ↑ | HD95 ↓ |
|------|--------|---------|------------|--------|
| Cross-Entropy | 0.2281 | 0.3148 | **0.8216** | **29.31** |
| Dice | 0.2148 | 0.2928 | 0.7986 | 34.82 |
| **Combined** | **0.2411** | **0.3299** | 0.8105 | 32.32 |

Differences are modest (~0.026 range) but consistent. Pure Dice has the worst HD95, reflecting spatially coarse predictions. Pure CE achieves the best pixel accuracy and HD95 by penalising each misclassified pixel uniformly. Combined loss inherits CE's spatial precision and Dice's class-balance sensitivity, yielding the best mIoU and mDice overall.

### A4 — Pre-training: ImageNet vs From Scratch

<img src="results/ablation_pretrain_miou.png" width="50%">

*Figure 10: A4 — mIoU after 30 epochs: ImageNet pretrained (0.2525) vs from scratch (0.0446).*

| Pre-training | mIoU ↑ | mDice ↑ | Pixel Acc ↑ | HD95 ↓ |
|-------------|--------|---------|------------|--------|
| **ImageNet pretrained** | **0.2525** | **0.3458** | **0.8272** | **35.15** |
| From scratch | 0.0446 | 0.0558 | 0.7221 | 36.61 |

This is the largest effect across all ablations: the pretrained model is 5.7× higher in mIoU. Training from scratch on 209 images is nearly futile — the model fails to learn meaningful boundaries without prior knowledge. ImageNet features (edges, textures, object parts) transfer directly to segmentation. This finding strongly motivates transfer learning in low-data regimes, particularly in medical imaging where annotations are expensive.

## 6. Discussion

DeepLabV3's dominant performance stems from two factors: its ASPP module captures multi-scale context simultaneously, providing global scene understanding that U-Net cannot match through successive pooling; and it starts from a COCO-pretrained segmentation head rather than ImageNet alone, giving it a much stronger prior. The result is a 22-point mIoU gap over U-Net-18 despite sharing the same ResNet-50 backbone. The ablation studies yield four actionable lessons: architecture fit matters more than model size in low-data regimes; pre-training is by far the most impactful factor; combined CE+Dice loss outperforms either alone; and augmentation requires a sufficient epoch budget to overcome its initial learning penalty.

The primary limitation of this study is dataset size — only 209 training images. Results on VOC 2012 or COCO would differ substantially. HD95 values of 25–37 pixels on 256×256 images (~10–14% of image width) reflect class-boundary confusion rather than fine-grain misalignment. The SAM-based model, while architecturally promising for few-shot transfer, was not benchmarked due to checkpoint availability.

## 7. Summary

| Criterion | Best Model | Value |
|-----------|-----------|-------|
| mIoU | DeepLabV3 | 0.5657 |
| mDice | DeepLabV3 | 0.7028 |
| Pixel Accuracy | DeepLabV3 | 90.16% |
| Best HD95 | U-Net ResNet-18 | 25.15 px |
| Most parameter-efficient | U-Net ResNet-18 | 14.5M, mIoU 0.34 |

DeepLabV3 is the clear winner on all primary segmentation metrics. Among the U-Net variants, the smaller ResNet-18 backbone is the better choice in this low-data setting. Pre-training is essential, combined loss is the best default, and augmentation should be paired with an adequate training budget.

*All results generated on NVIDIA RTX 5070 Ti (17 GB VRAM), PyTorch 2.8.0+cu128, Python 3.9.*
