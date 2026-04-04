# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 09:17:22 2025

@author: mohae
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 1. Define the 21 classes
# -------------------------
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
NUM_CLASSES = len(VOC_CLASSES)
print("Number of classes:", NUM_CLASSES)
print("Classes:", VOC_CLASSES)


# ---------------------------------------
# Class mapping dictionary
# ---------------------------------------
class_mapping = {i: cls for i, cls in enumerate(VOC_CLASSES)}

print("📊 Pascal VOC 2007 Class Mapping:")
for idx, name in class_mapping.items():
    print(f"{idx:2d} → {name}")
    
    

# -----------------------------------
# 2. Define transforms for the images
# -----------------------------------
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),           # resize for speed
    transforms.ToTensor(),                   # convert to tensor (C,H,W)
])

transform_target = transforms.Compose([
    transforms.Resize((256, 256)),           # resize mask too
    transforms.PILToTensor()                 # keep as tensor (H,W)
])

# --------------------------------------
# 3. Load the Pascal VOC 2007 Segmentation Dataset
# --------------------------------------
train_dataset = VOCSegmentation(
    root="./VOCtrainval_06-Nov-2007",
    year="2007",
    image_set="train",
    download=False,
    transform=transform_img,
    target_transform=transform_target
)

val_dataset = VOCSegmentation(
    root="./VOCtrainval_06-Nov-2007",
    year="2007",
    image_set="val",
    download=False,
    transform=transform_img,
    target_transform=transform_target
)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

# -------------------------
# 4. Create DataLoaders
# -------------------------
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# ---------------------------------------
# 5. Inspect a single batch (inputs/outputs)
# ---------------------------------------
images, masks = next(iter(train_loader))
print("Image batch shape:", images.shape)    # (B, 3, 256, 256)
print("Mask batch shape:", masks.shape)      # (B, 1, 256, 256)

# ---------------------------------------
# 6. Visualize one image and mask
# ---------------------------------------
def show_sample(img, mask):
    img = img.permute(1, 2, 0).numpy()      # C,H,W -> H,W,C
    mask = mask.squeeze().numpy().copy()    # 1,H,W -> H,W

    # Clean mask: convert all values > 20 (like 255) to 0 for visualization
    mask[mask > 20] = 0

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    seg_map = plt.imshow(mask, cmap="tab20", vmin=0, vmax=20)
    plt.title("Segmentation Mask (cleaned)")
    plt.axis("off")

    # ✅ Custom colorbar with class numbers and names
    cbar = plt.colorbar(seg_map, ticks=range(21))
    tick_labels = [f"{i} – {VOC_CLASSES[i]}" for i in range(21)]
    cbar.ax.set_yticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=8)  # smaller font if needed

    plt.tight_layout()
    plt.show()

# Show first sample
show_sample(images[0], masks[0])
print("Classes in this mask:", np.unique(masks[0].numpy()))



