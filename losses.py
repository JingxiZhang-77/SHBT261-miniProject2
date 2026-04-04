# -*- coding: utf-8 -*-
"""
Loss functions for semantic segmentation:
  - CrossEntropyLoss  : standard pixel-wise CE (ignores index 255)
  - DiceLoss          : soft Dice computed per class
  - CombinedLoss      : 0.5 * CE + 0.5 * Dice
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import IGNORE_INDEX, NUM_CLASSES


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = IGNORE_INDEX, weight=None):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)


class DiceLoss(nn.Module):
    """
    Soft Dice loss averaged over classes that are present in the batch.
    Ignores pixels with label == ignore_index.
    """

    def __init__(self, smooth: float = 1.0,
                 ignore_index: int = IGNORE_INDEX,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index
        self.num_classes  = num_classes

    def forward(self, logits, targets):
        # logits  : (B, C, H, W)
        # targets : (B, H, W)
        probs = F.softmax(logits, dim=1)
        valid = (targets != self.ignore_index).float()   # (B, H, W)

        total, n = 0.0, 0
        for cls in range(self.num_classes):
            gt   = (targets == cls).float() * valid      # (B, H, W)
            if gt.sum() == 0:
                continue
            pred = probs[:, cls] * valid                 # (B, H, W)
            inter = (pred * gt).sum()
            denom = pred.sum() + gt.sum() + self.smooth
            total += 1 - (2 * inter + self.smooth) / denom
            n += 1

        return total / max(n, 1)


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5,
                 ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.ce      = CrossEntropyLoss(ignore_index=ignore_index)
        self.dice    = DiceLoss(ignore_index=ignore_index)
        self.ce_w    = ce_weight
        self.dice_w  = dice_weight

    def forward(self, logits, targets):
        return self.ce_w * self.ce(logits, targets) + \
               self.dice_w * self.dice(logits, targets)


def get_loss(loss_type: str = "combined") -> nn.Module:
    if loss_type == "ce":
        return CrossEntropyLoss()
    if loss_type == "dice":
        return DiceLoss()
    if loss_type == "combined":
        return CombinedLoss()
    raise ValueError(f"Unknown loss: {loss_type}. Choose from: ce, dice, combined")
