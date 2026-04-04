# -*- coding: utf-8 -*-
"""
Global configuration for Mini-Project 2: Semantic Segmentation on Pascal VOC 2007.
"""

import torch
import os
import platform

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_ROOT       = "./VOCtrainval_06-Nov-2007"
CHECKPOINT_DIR  = "./checkpoints"
RESULTS_DIR     = "./results"
SAM_CHECKPOINT  = "./sam_vit_b_01ec64.pth"   # download separately (see README below)

# ─── Dataset ─────────────────────────────────────────────────────────────────
IMG_SIZE     = 256
NUM_CLASSES  = 21
IGNORE_INDEX = 255

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Official Pascal VOC colormap (RGB, 0-255)
VOC_COLORMAP = [
    [  0,   0,   0], [128,   0,   0], [  0, 128,   0], [128, 128,   0],
    [  0,   0, 128], [128,   0, 128], [  0, 128, 128], [128, 128, 128],
    [ 64,   0,   0], [192,   0,   0], [ 64, 128,   0], [192, 128,   0],
    [ 64,   0, 128], [192,   0, 128], [ 64, 128, 128], [192, 128, 128],
    [  0,  64,   0], [128,  64,   0], [  0, 192,   0], [128, 192,   0],
    [  0,  64, 128],
]

# ─── Training ────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# Windows multiprocessing with DataLoader requires num_workers=0
NUM_WORKERS  = 0 if platform.system() == "Windows" else 4
BATCH_SIZE   = 4
EPOCHS       = 60
LR           = 1e-4
WEIGHT_DECAY = 1e-4

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── SAM Setup Instructions ──────────────────────────────────────────────────
"""
To use the SAM-based model, install the package and download the ViT-B checkpoint:

  pip install git+https://github.com/facebookresearch/segment-anything.git
  # Then download:
  #   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  # and place it at ./sam_vit_b_01ec64.pth

Required packages (all models):
  pip install albumentations scipy scikit-learn tqdm
"""
