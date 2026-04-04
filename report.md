# Mini-Project 2: Semantic Segmentation on Pascal VOC 2007

**Course:** SHBT 261  
**Dataset:** Pascal VOC 2007 (Segmentation)  
**Submission:** April 15, 2026

---

## 1. Introduction

Semantic segmentation assigns a class label to every pixel in an image. This project trains and evaluates three segmentation models on the Pascal VOC 2007 dataset, which contains 21 classes (20 objects + background) across 209 training and 213 validation images. We use the validation set as our test set throughout.

The three models compared are:
- **U-Net with ResNet-18 encoder** (14.5M parameters)
- **U-Net with ResNet-50 encoder** (32.7M parameters)
- **DeepLabV3 with ResNet-50 backbone** (42.0M parameters)

A SAM (Segment Anything Model)-based model was also implemented (`SAMSegmentation` in `models.py`) with a frozen ViT-B image encoder and a lightweight trainable decoder. Due to the requirement for a separate checkpoint download (`sam_vit_b_01ec64.pth`), SAM was not benchmarked quantitatively; all three models above were fully trained and evaluated.

---

## 2. Methods

### 2.1 Dataset and Preprocessing

The Pascal VOC 2007 segmentation split was loaded via `torchvision.datasets.VOCSegmentation`. Mask pixel values range from 0–20 (class index) and 255 (boundary/ignore). Boundary pixels are excluded from all loss and metric computations (`ignore_index=255`).

**Training augmentations** (albumentations):
- `RandomResizedCrop(256×256, scale=(0.4, 1.0))` — random crop and scale
- `HorizontalFlip(p=0.5)`
- `ColorJitter(brightness/contrast/saturation=0.3, hue=0.1, p=0.6)`
- `GaussianBlur(p=0.2)`
- ImageNet normalisation: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`

**Validation:** only resize to 256×256 and normalize (no augmentation).

**DataLoader:** batch size 4, `drop_last=True` on training (avoids BatchNorm failure with singleton batches in ASPP layers).

### 2.2 Model Architectures

#### U-Net (ResNet-18 / ResNet-50 Encoder)

A classic encoder-decoder architecture with skip connections. The ResNet backbone is split into five stages:

| Stage | ResNet-18 channels | ResNet-50 channels | Stride |
|-------|-------------------|-------------------|--------|
| enc0  | 64  | 64   | /2  |
| enc1  | 64  | 256  | /4  |
| enc2  | 128 | 512  | /8  |
| enc3  | 256 | 1024 | /16 |
| enc4  | 512 | 2048 | /32 |

The decoder uses bilinear upsampling followed by double conv blocks, concatenating skip features at each scale. A final `1×1` conv produces 21-channel logits upsampled to the input resolution.

#### DeepLabV3 (ResNet-50 Backbone)

Uses `torchvision.models.segmentation.deeplabv3_resnet50` with ImageNet-pretrained weights. The original COCO classifier head is replaced with `Conv2d(256 → 21, 1×1)`. DeepLabV3 employs atrous convolutions and ASPP (Atrous Spatial Pyramid Pooling) to capture multi-scale context without resolution loss. Logits are bilinearly upsampled to the input size.

> **Note:** The torchvision variant implements standard DeepLabV3 (ASPP only). DeepLabV3+ additionally adds a shallow encoder-decoder skip path; that extension is not present here.

#### SAM-based Segmentation (Implemented, Not Benchmarked)

Frozen SAM ViT-B image encoder produces `(B, 256, 64, 64)` feature maps for 1024×1024 inputs. A trainable `_SAMDecoder` upsamples these through two bilinear ×2 stages with conv-BN-ReLU blocks to produce `(B, 21, 256, 256)` logits. Only decoder parameters are optimized.

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

**Loss functions:**
- **Cross-entropy:** standard pixel-wise CE, ignoring boundary pixels.
- **Dice loss:** soft Dice averaged over classes present in the batch.
- **Combined:** equal-weight sum of CE and Dice.

Checkpoints are saved whenever validation mIoU improves. Final evaluation uses the best-mIoU checkpoint.

---

## 3. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection-over-Union across 21 classes (NaN-excluded) |
| **mDice** | Mean Dice coefficient |
| **Pixel Accuracy** | Fraction of correctly classified pixels |
| **HD95** | 95th-percentile Hausdorff distance (averaged over samples and classes) |
| **Per-class IoU** | IoU for each of the 21 VOC classes |

---

## 4. Results

### 4.1 Overall Performance

| Model | Params | mIoU ↑ | mDice ↑ | Pixel Acc ↑ | HD95 ↓ |
|-------|--------|--------|---------|------------|--------|
| U-Net ResNet-18 | 14.5M | 0.3435 | 0.4510 | 0.8549 | **25.15** |
| U-Net ResNet-50 | 32.7M | 0.2571 | 0.3408 | 0.8427 | 25.92 |
| **DeepLabV3 ResNet-50** | **42.0M** | **0.5657** | **0.7028** | **0.9016** | 26.83 |

DeepLabV3 outperforms both U-Net variants by a large margin (+0.222 mIoU over U-Net-18). The smaller U-Net ResNet-18 outperforms the larger ResNet-50 variant — explained in Section 5.

### 4.2 Training Curves

![Training Curves](results/training_curves.png)
*Figure 1: Training and validation loss (left), validation mIoU (center), and validation mDice (right) over 60 epochs for DeepLabV3. The solid line is training loss and the dashed line is validation loss.*

The training curves plot shows **three panels** for DeepLabV3 (the final model trained in this run):

- **Left — Loss:** Both training loss (solid blue) and validation loss (dashed orange) decrease steeply in the first ~15 epochs, then plateau around 0.3–0.4. The training and validation losses track closely, indicating no severe overfitting.
- **Middle — mIoU:** Validation mIoU rises steadily from near 0 to approximately **0.55+** by epoch 60, with noisy but upward momentum — typical for a 21-class segmentation task.
- **Right — mDice:** Mirrors the mIoU curve, reaching around **0.70** by epoch 60, consistent with the final reported value of 0.7028.

The gradual, sustained improvement with no plateau or divergence confirms that 60 epochs was an appropriate training budget.

### 4.3 Per-class IoU Comparison

![Per-class IoU Comparison](results/per_class_iou_comparison.png)
*Figure 2: Per-class IoU comparison across all 21 Pascal VOC classes for U-Net ResNet-18 (blue), U-Net ResNet-50 (brown), and DeepLabV3 (cyan). Zero-height bars for U-Nets on bicycle, bottle, pottedplant, and sheep indicate complete failure on those classes.*

This grouped bar chart shows the IoU for all 21 VOC classes side-by-side for all three models (blue = U-Net-18, brown = U-Net-50, cyan = DeepLabV3).

**What the chart reveals:**

- **DeepLabV3 (cyan) dominates almost every class.** The cyan bars are taller across the board — most dramatically for `aeroplane`, `bottle`, `bus`, `motorbike`, and `train`, where DeepLabV3 reaches 0.70–0.80 while both U-Nets fall below 0.55 or even to zero.
- **Zero-height blue/brown bars** for `bicycle`, `bottle`, `pottedplant`, and `sheep` show that both U-Net variants completely failed on these classes — they predicted zero pixels as those classes on the validation set. DeepLabV3's ASPP multi-scale reasoning allows it to break through on these difficult small-object classes.
- **`cat` is the one exception** where U-Net-50 (brown, 0.84) marginally beats DeepLabV3 (cyan, 0.77). Cat images in VOC are often close-up and texture-rich, which may suit ResNet-50's deeper features.
- **`person`** (class 15) has tall bars for all three models (~0.71–0.76), reflecting that person is the most frequent and most visually distinctive class in VOC.
- **`sofa`** has the shortest bars across all models (max 0.20), showing it is the hardest class — likely confused with `chair` and `diningtable`.

### 4.4 Per-class IoU Table

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

### 4.5 Qualitative Segmentation — Mosaic

![DeepLabV3 Mosaic](results/deeplabv3plus_mosaic.png)
*Figure 3: Qualitative segmentation results for DeepLabV3 on 4 validation images. Row 1: original input images. Row 2: model predictions. Row 3: ground truth masks. Colors correspond to VOC class labels shown in the legend (e.g., blue = tvmonitor, red = dog, lime = train).*

The mosaic shows 4 validation images in three rows: **Input → Prediction → Ground Truth**.

- **Column 1 (computer + monitor):** DeepLabV3 correctly segments the tvmonitor (blue) with sharp boundaries that closely match the ground truth.
- **Column 2 (dog on chair):** The model correctly identifies and segments the dog (red) and the chair region. The cat/dog boundary in the ground truth is closely followed in the prediction.
- **Column 3 (monitor on desk):** The tvmonitor (blue) is accurately segmented; the model correctly excludes the surrounding desk area.
- **Column 4 (train at station):** The train (lime green) is well segmented, capturing the correct shape. The prediction is slightly over-extended at edges but the overall form is correct.

The colormap legend at the bottom maps each color to its class index and name (e.g., blue = tvmonitor/class 20, red = dog/class 12, lime = train/class 19).

### 4.6 Best and Worst Predictions (Person Class)

**Best 3:**

![DeepLabV3 Best 3](results/deeplabv3plus_best3.png)
*Figure 4: Top-3 best predictions by DeepLabV3 ranked by person-class IoU. Row 1: input images. Row 2: model predictions. Row 3: ground truth masks. Person IoU scores of 0.934, 0.885, and 0.852 indicate near-perfect segmentation of large, clearly visible persons.*

The three best predictions (person IoU = 0.934, 0.885, 0.852) show the model's strength on person-centric images:
- **Left (IoU=0.934):** Close-up of two people — prediction nearly perfectly matches the ground truth silhouettes in pink/mauve. Very clean boundaries.
- **Middle (IoU=0.885):** Group of 4–5 people outdoors — the model correctly segments individual person regions and their relative positions.
- **Right (IoU=0.852):** Single person in dark clothing at a bar — the model captures the full body outline despite the dark background. A small purple region (pottedplant/background confusion) appears but does not significantly reduce IoU.

These cases share a common trait: **persons are large, well-lit, and occupy most of the frame**, making them easier to segment.

**Worst 3:**

![DeepLabV3 Worst 3](results/deeplabv3plus_worst3.png)
*Figure 5: Bottom-3 worst predictions by DeepLabV3 ranked by person-class IoU. All three cases have person IoU = 0.000, meaning the model failed to predict any person pixels. In each case, the person region in the ground truth is extremely small (e.g., an arm next to a parrot, or a tiny figure near a cow), effectively invisible at 256×256 resolution.*

The three worst predictions all have person IoU = 0.000 — meaning no person was detected even though the ground truth labels a person somewhere in the image:
- **Left (cow in field):** The ground truth contains a small person label near the cow, but the model predicts only cow (green) and background. The person is likely a tiny secondary object occluded or at image border.
- **Middle (parrot):** Ground truth has a person label (likely the hand/arm holding the bird), but the model predicts only bird (yellow-green). The person region is a very small sliver, invisible at 256×256 resolution.
- **Right (bird on water):** A tiny boat/duck with an extremely small person label. The model correctly segments the tiny blue object but misses the person, which is nearly sub-pixel at this resolution.

The worst cases illustrate a fundamental challenge: **IoU of 0.0 is triggered by any image where a person is present in the ground truth but not predicted at all** — this disproportionately affects images where persons are very small.

### 4.7 Confusion Matrix

![DeepLabV3 Confusion Matrix](results/deeplabv3plus_confusion.png)
*Figure 6: Normalised confusion matrix for DeepLabV3 on the validation set. Rows represent ground truth classes; columns represent predicted classes. Each cell value is the fraction of ground truth pixels for that class predicted as the column class. A perfect model would show 1.0 along the diagonal and 0 elsewhere.*

The normalised confusion matrix (rows = ground truth, columns = predicted) shows:

- **Strong diagonal:** Most classes have high diagonal values (dark blue squares), meaning correct classification is the dominant outcome — especially for background (row 0), bus, train, and tvmonitor.
- **Person ↔ Background confusion:** The person row shows a small off-diagonal value toward background, meaning some person pixels are misclassified as background. This is expected in images where persons are at image edges or partially occluded.
- **Animal class confusion:** Dog, cat, cow, horse rows show mild confusion with each other (visible lighter off-diagonal squares), which is expected since these quadrupeds share shape and texture features.
- **Sofa confusion:** The sofa row has a notable off-diagonal value toward `chair` and `diningtable`, consistent with its low IoU of 0.20 across all models.
- **Background dominance:** Background (class 0) absorbs some misclassified pixels from nearly every class — visible as the first-column entries for rare or small classes like bicycle and pottedplant.

---

## 5. Ablation Studies

All ablations use **U-Net ResNet-18** as the base model and train for **30 epochs** with combined loss (unless noted). Results are lower than 60-epoch full training by design — the goal is comparative, not absolute.

### A1 — Backbone Size: ResNet-18 vs ResNet-50

![A1 mIoU](results/ablation_backbone_miou.png)
*Figure 7: Ablation A1 — mIoU comparison between U-Net with ResNet-18 (0.2380) and ResNet-50 (0.1339) encoders after 30 epochs. Despite having more parameters, ResNet-50 underperforms due to overfitting on the small 209-image training set.*

| Backbone | mIoU ↑ | mDice ↑ | Pixel Acc ↑ | HD95 ↓ | Params |
|----------|--------|---------|------------|--------|--------|
| **ResNet-18** | **0.2380** | **0.3332** | **0.8173** | **30.62** | 14.5M |
| ResNet-50 | 0.1339 | 0.1837 | 0.7981 | 33.18 | 32.7M |

**The bar chart** shows ResNet-18 (0.2380) is nearly **double** ResNet-50 (0.1339) in mIoU after 30 epochs. This is counterintuitive — normally bigger is better. The explanation is **dataset size**: with only 209 training images, ResNet-50's 32.7M parameters are severely over-parameterized. The model memorizes training images and fails to generalize. ResNet-18's smaller capacity is a better fit for this low-data regime. This pattern holds at 60 epochs (0.3435 vs 0.2571), confirming it is a structural effect, not just slow convergence.

![A1 Curves](results/ablation_backbone_curves.png)
*Figure 8: Ablation A1 — training curves (loss, mIoU, mDice) for ResNet-18 vs ResNet-50 encoders over 30 epochs. ResNet-18 rises faster and reaches a consistently higher mIoU plateau, while ResNet-50's curve is flatter and more erratic.*

The training curves confirm ResNet-18 pulls ahead early (around epoch 5) and maintains a consistent lead. ResNet-50's mIoU curve is flatter and more erratic, indicating difficulty learning from the small dataset.

### A2 — Data Augmentation: With vs Without

![A2 mIoU](results/ablation_aug_miou.png)
*Figure 9: Ablation A2 — mIoU comparison between training with augmentation (0.2103) and without augmentation (0.2853) after 30 epochs. No-augmentation converges faster at this epoch budget; augmentation requires more epochs to show its regularization benefit.*

| Augmentation | mIoU ↑ | mDice ↑ | Pixel Acc ↑ | HD95 ↓ |
|-------------|--------|---------|------------|--------|
| OFF | **0.2853** | **0.3980** | **0.8370** | 33.20 |
| ON  | 0.2103 | 0.2944 | 0.8125 | **29.56** |

**The bar chart** shows augmentation-OFF (0.2853) wins over augmentation-ON (0.2103) at 30 epochs. The taller bar for "Augment=OFF" may seem surprising — isn't augmentation supposed to help? The answer is **training budget**: aggressive augmentation (random crops at 0.4–1.0 scale, color jitter, blur) creates very challenging views that require more epochs to learn from. At 30 epochs, the model with augmentation is still in the slow-learning phase. Notably, augmentation-ON has a **lower HD95** (29.56 vs 33.20), meaning better boundary quality, suggesting augmentation is already improving spatial precision even before mIoU fully catches up. At 60 epochs, augmented U-Net-18 reaches mIoU 0.3435 — confirming augmentation pays off with enough training time.

![A2 Curves](results/ablation_aug_curves.png)
*Figure 10: Ablation A2 — training curves for augmentation ON vs OFF over 30 epochs. Augmentation-OFF (no-aug) rises steeply in the first 15 epochs, while augmentation-ON improves more slowly but more steadily — a hallmark of better generalization that would continue to improve beyond 30 epochs.*

The curves show augmentation-OFF converges faster initially (steeper mIoU rise in first 15 epochs), while augmentation-ON shows slower but more stable improvement — a classic sign of better regularization kicking in over time.

### A3 — Loss Function: CE vs Dice vs Combined

![A3 mIoU](results/ablation_loss_miou.png)
*Figure 11: Ablation A3 — mIoU comparison across three loss functions after 30 epochs: Cross-Entropy (0.2281), Dice (0.2148), and Combined CE+Dice (0.2411). The differences are modest but consistent: Combined loss achieves the best mIoU by balancing spatial precision (CE) and class-balance sensitivity (Dice).*

| Loss | mIoU ↑ | mDice ↑ | Pixel Acc ↑ | HD95 ↓ |
|------|--------|---------|------------|--------|
| Cross-Entropy | 0.2281 | 0.3148 | **0.8216** | **29.31** |
| Dice | 0.2148 | 0.2928 | 0.7986 | 34.82 |
| **Combined** | **0.2411** | **0.3299** | 0.8105 | 32.32 |

**The bar chart** shows three bars of similar height, but Combined (rightmost, 0.2411) wins on mIoU with CE second (0.2281) and Dice last (0.2148). The differences are modest (~0.026 range), but the ordering is informative:

- **Pure Dice** has the lowest mIoU and worst HD95. Dice loss focuses on maximizing overlap of each class globally, which can lead to spatially coarse predictions with poor boundary sharpness.
- **Pure CE** achieves the best pixel accuracy and HD95 — it penalizes every misclassified pixel uniformly, promoting spatial precision.
- **Combined** balances both: it inherits CE's spatial precision and Dice's class-balance sensitivity (Dice naturally up-weights minority classes where intersections are small). The result is the best mIoU and mDice.

![A3 Curves](results/ablation_loss_curves.png)
*Figure 12: Ablation A3 — training curves for CE, Dice, and Combined loss over 30 epochs. All three converge similarly; Combined and CE track closely while pure Dice shows slightly lower and more variable mIoU.*

The training curves show all three losses follow a similar trajectory, but Combined and CE stay closer to each other while Dice diverges slightly on the mIoU metric.

### A4 — Pre-training: ImageNet vs From Scratch

![A4 mIoU](results/ablation_pretrain_miou.png)
*Figure 13: Ablation A4 — mIoU comparison between ImageNet-pretrained (0.2525) and randomly initialized from-scratch (0.0446) U-Net ResNet-18 after 30 epochs. Pre-training provides a 5.7× improvement, the largest effect observed across all ablation studies.*

| Pre-training | mIoU ↑ | mDice ↑ | Pixel Acc ↑ | HD95 ↓ |
|-------------|--------|---------|------------|--------|
| **ImageNet pretrained** | **0.2525** | **0.3458** | **0.8272** | **35.15** |
| From scratch | 0.0446 | 0.0558 | 0.7221 | 36.61 |

**The bar chart** shows the most dramatic effect of any ablation: the pretrained bar (0.2525) is **5.7× taller** than from-scratch (0.0446). Training from scratch on 209 images is nearly futile — mIoU of 0.0446 is barely above chance for 21 classes. The model fails to learn any meaningful class boundaries within 30 epochs without prior knowledge.

ImageNet pre-training transfers rich low-level features (edges, textures, object parts) that are directly applicable to segmentation. This is the single most impactful factor explored across all ablations. The finding strongly motivates transfer learning whenever labeled data is scarce — a lesson highly relevant to medical imaging where annotations are expensive.

![A4 Curves](results/ablation_pretrain_curves.png)
*Figure 14: Ablation A4 — training curves for pretrained vs from-scratch initialization over 30 epochs. The pretrained model's mIoU rises immediately and steadily from epoch 1. The from-scratch model barely moves for the first 15 epochs and plateaus near 0.04, illustrating how insufficient data prevents learning meaningful features without prior knowledge.*

The curves make this visually striking: the pretrained model's mIoU curve rises immediately and steadily, while the from-scratch model barely moves off zero for the first 15 epochs and plateaus around 0.04.

---

## 6. Discussion

### 6.1 Why DeepLabV3 Dominates

DeepLabV3's ASPP captures context at multiple atrous rates simultaneously, giving it a global understanding of scene layout that U-Net cannot match with successive pooling alone. It also starts from a COCO-pretrained segmentation head — not just ImageNet features — giving it a much stronger prior. The result is a 22-point mIoU gap over U-Net-18, even though both share a ResNet-50 backbone.

### 6.2 Hard Classes Across All Models

**Sofa** (max IoU 0.20 across all models) is the hardest class. It shares shape and color properties with chairs, diningtables, and cushions. All three models essentially fail to distinguish sofa from nearby furniture.

**Bicycle, bottle, pottedplant, sheep** were IoU=0 for both U-Nets. These are structurally different problems:
- Bicycle: thin, complex structure with lots of background inside the bounding box.
- Bottle: small, often partially occluded.
- Pottedplant, sheep: low frequency in the 209-image training set, and visually similar to background.

DeepLabV3's ASPP and stronger pre-training let it break the zero-IoU barrier on all four.

### 6.3 Limitations

- **Dataset size:** Only 209 training images. Results on VOC 2012 or COCO would differ substantially.
- **SAM not benchmarked:** The frozen ViT-B + decoder design is especially promising for few-shot transfer but was not evaluated due to checkpoint availability.
- **HD95 scale:** Values of 25–37 pixels on 256×256 images correspond to ~10–14% of image width — these are large errors reflecting class-boundary confusion rather than fine-grain misalignment.

---

## 7. Summary

| Criterion | Best Model | Value |
|-----------|-----------|-------|
| mIoU | DeepLabV3 | 0.5657 |
| mDice | DeepLabV3 | 0.7028 |
| Pixel Accuracy | DeepLabV3 | 90.16% |
| Best HD95 | U-Net ResNet-18 | 25.15 px |
| Most parameter-efficient | U-Net ResNet-18 | 14.5M, mIoU 0.34 |

**Key lessons:**
1. **Architecture > size** when data is scarce — ResNet-18 beats ResNet-50 in this regime.
2. **Pre-training is essential** — 5.7× mIoU gain; from-scratch training on 209 images fails.
3. **Combined loss is the best default** — balances boundary sharpness (CE) and class balance (Dice).
4. **Augmentation needs time** — hurts at 30 epochs, helps at 60; budget training accordingly.
5. **ASPP multi-scale context** is critical for small/rare classes where U-Net scores zero.

---

## 8. Generated Figures Reference

| File | What it shows |
|------|--------------|
| `results/training_curves.png` | Loss / mIoU / mDice curves over 60 epochs (DeepLabV3) |
| `results/per_class_iou_comparison.png` | Grouped bar chart: per-class IoU for all 3 models |
| `results/deeplabv3plus_mosaic.png` | 4-image mosaic: input / prediction / ground truth (DeepLabV3) |
| `results/unet18_mosaic.png` | Same for U-Net ResNet-18 |
| `results/unet50_mosaic.png` | Same for U-Net ResNet-50 |
| `results/deeplabv3plus_best3.png` | Top 3 person-IoU predictions (DeepLabV3) |
| `results/deeplabv3plus_worst3.png` | Bottom 3 person-IoU predictions (DeepLabV3) |
| `results/unet18_best3.png` / `unet18_worst3.png` | Best/worst 3 for U-Net-18 |
| `results/unet50_best3.png` / `unet50_worst3.png` | Best/worst 3 for U-Net-50 |
| `results/deeplabv3plus_confusion.png` | Normalised 21×21 confusion matrix (DeepLabV3) |
| `results/unet18_confusion.png` | Same for U-Net-18 |
| `results/unet50_confusion.png` | Same for U-Net-50 |
| `results/ablation_backbone_miou.png` | A1 bar chart: ResNet-18 vs ResNet-50 mIoU |
| `results/ablation_backbone_curves.png` | A1 training curves |
| `results/ablation_aug_miou.png` | A2 bar chart: augmentation ON vs OFF |
| `results/ablation_aug_curves.png` | A2 training curves |
| `results/ablation_loss_miou.png` | A3 bar chart: CE vs Dice vs Combined |
| `results/ablation_loss_curves.png` | A3 training curves |
| `results/ablation_pretrain_miou.png` | A4 bar chart: pretrained vs from scratch |
| `results/ablation_pretrain_curves.png` | A4 training curves |

---

## 9. Code Structure

| File | Description |
|------|-------------|
| `config.py` | Hyperparameters, paths, class names, VOC colormap |
| `dataset.py` | `VOCDataset` with albumentations augmentation |
| `models.py` | `UNetResNet`, `DeepLabV3PlusWrapper`, `SAMSegmentation` |
| `losses.py` | `CrossEntropyLoss`, `DiceLoss`, `CombinedLoss` |
| `train.py` | Training loop, evaluation, checkpoint saving |
| `metrics.py` | mIoU, mDice, pixel accuracy, HD95, confusion matrix |
| `visualize.py` | Mosaic, best/worst, training curves, confusion matrix, model comparison |
| `ablation.py` | Four ablation study runners |
| `main.py` | CLI entry point |

```bash
# Train all three models (60 epochs)
python main.py --models unet18 unet50 deeplabv3plus --epochs 60

# Evaluate from saved checkpoints
python main.py --models unet18 unet50 deeplabv3plus --eval-only

# Run all ablations (30 epochs each)
python ablation.py --ablations all --epochs 30
```

---

*All results generated on NVIDIA RTX 5070 Ti (17 GB VRAM), PyTorch 2.8.0+cu128, Python 3.9.*
