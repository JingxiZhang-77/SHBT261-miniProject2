# -*- coding: utf-8 -*-
"""
Pascal VOC 2007 dataset loader with optional albumentations augmentation.
Masks: pixel values 0-20 = class index, 255 = boundary/ignore.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCSegmentation

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import DATA_ROOT, IMG_SIZE, NUM_WORKERS, BATCH_SIZE


def _build_transforms(img_size: int, augment: bool) -> A.Compose:
    if augment:
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size),
                                scale=(0.4, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.1, p=0.6),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class VOCDataset(Dataset):
    """
    Wraps torchvision VOCSegmentation and applies albumentations transforms.
    Returns:
        image : FloatTensor  (3, H, W)  ImageNet-normalised
        mask  : LongTensor   (H, W)     class indices; 255 = ignore
    """

    def __init__(self, root=DATA_ROOT, image_set="train",
                 img_size=IMG_SIZE, augment=False):
        self.base = VOCSegmentation(
            root=root, year="2007", image_set=image_set, download=False
        )
        self.tfm = _build_transforms(img_size, augment)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img_pil, mask_pil = self.base[idx]
        img  = np.array(img_pil,  dtype=np.uint8)
        mask = np.array(mask_pil, dtype=np.int32)

        out  = self.tfm(image=img, mask=mask)
        return out["image"].float(), out["mask"].long()


def get_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                augment=True, img_size=IMG_SIZE, root=DATA_ROOT):
    train_ds = VOCDataset(root=root, image_set="train",
                          img_size=img_size, augment=augment)
    val_ds   = VOCDataset(root=root, image_set="val",
                          img_size=img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
