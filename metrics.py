# -*- coding: utf-8 -*-
"""
Segmentation evaluation metrics:
  - mIoU (mean Intersection-over-Union)
  - mDice (mean Dice coefficient)
  - Pixel Accuracy
  - Per-class IoU and Accuracy
  - HD95 (95th-percentile Hausdorff distance)
  - Confusion Matrix

All functions accept numpy arrays.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import confusion_matrix as sk_cm

from config import NUM_CLASSES, IGNORE_INDEX, VOC_CLASSES


# ─── Per-sample HD95 ─────────────────────────────────────────────────────────

def _hd95_sample(pred: np.ndarray, target: np.ndarray,
                 num_classes: int = NUM_CLASSES,
                 ignore_index: int = IGNORE_INDEX):
    """
    Compute mean HD95 across all classes present in `target` for one sample.
    Returns None if no valid class pair found.
    """
    classes = np.unique(target[target != ignore_index])
    vals = []

    for cls in classes:
        p_mask = (pred   == cls).astype(np.uint8)
        g_mask = (target == cls).astype(np.uint8)

        if p_mask.sum() == 0 or g_mask.sum() == 0:
            continue

        dt_g = distance_transform_edt(1 - g_mask)
        dt_p = distance_transform_edt(1 - p_mask)

        d1 = dt_g[p_mask == 1]
        d2 = dt_p[g_mask == 1]

        hd95 = max(np.percentile(d1, 95), np.percentile(d2, 95))
        vals.append(hd95)

    return float(np.mean(vals)) if vals else None


# ─── Aggregate metrics ────────────────────────────────────────────────────────

def compute_metrics(all_preds: list, all_targets: list,
                    num_classes: int = NUM_CLASSES,
                    ignore_index: int = IGNORE_INDEX,
                    compute_hd95: bool = True,
                    hd95_max_samples: int = 100) -> dict:
    """
    Args:
        all_preds   : list of (H, W) np.ndarray – predicted class indices
        all_targets : list of (H, W) np.ndarray – ground-truth class indices
        compute_hd95: if False, skip the expensive HD95 computation
        hd95_max_samples: limit HD95 to first N samples for speed

    Returns:
        dict with keys: mIoU, mDice, pixel_acc, HD95,
                        iou_per_class, dice_per_class, acc_per_class,
                        confusion_matrix
    """
    # ── build flat arrays ignoring boundary pixels ────────────────────────
    flat_pred   = np.concatenate([p.flatten() for p in all_preds])
    flat_target = np.concatenate([t.flatten() for t in all_targets])

    valid        = flat_target != ignore_index
    flat_pred    = flat_pred[valid]
    flat_target  = flat_target[valid]

    # ── confusion matrix ─────────────────────────────────────────────────
    cm = sk_cm(flat_target, flat_pred, labels=list(range(num_classes)))

    # ── IoU ──────────────────────────────────────────────────────────────
    tp   = np.diag(cm)
    fp   = cm.sum(axis=0) - tp
    fn   = cm.sum(axis=1) - tp
    union = tp + fp + fn
    iou_per_class = np.where(union > 0, tp / union, np.nan)
    miou  = float(np.nanmean(iou_per_class))

    # ── Dice ─────────────────────────────────────────────────────────────
    denom = 2 * tp + fp + fn
    dice_per_class = np.where(denom > 0, 2 * tp / denom, np.nan)
    mdice = float(np.nanmean(dice_per_class))

    # ── Pixel Accuracy ────────────────────────────────────────────────────
    pixel_acc = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0

    # ── Per-class Accuracy ────────────────────────────────────────────────
    row_sum = cm.sum(axis=1)
    acc_per_class = np.where(row_sum > 0, np.diag(cm) / row_sum, np.nan)

    # ── HD95 ─────────────────────────────────────────────────────────────
    mean_hd95 = float("nan")
    if compute_hd95:
        hd_vals = []
        for pred, target in zip(all_preds[:hd95_max_samples],
                                all_targets[:hd95_max_samples]):
            v = _hd95_sample(pred, target, num_classes, ignore_index)
            if v is not None:
                hd_vals.append(v)
        mean_hd95 = float(np.mean(hd_vals)) if hd_vals else float("nan")

    return {
        "mIoU":           miou,
        "mDice":          mdice,
        "pixel_acc":      pixel_acc,
        "HD95":           mean_hd95,
        "iou_per_class":  iou_per_class,
        "dice_per_class": dice_per_class,
        "acc_per_class":  acc_per_class,
        "confusion_matrix": cm,
    }


def print_metrics(metrics: dict, class_names: list = VOC_CLASSES,
                  model_name: str = ""):
    sep = "=" * 62
    print(f"\n{sep}")
    if model_name:
        print(f"  Model : {model_name}")
    print(f"  mIoU        : {metrics['mIoU']:.4f}")
    print(f"  mDice       : {metrics['mDice']:.4f}")
    print(f"  Pixel Acc   : {metrics['pixel_acc']:.4f}")
    print(f"  HD95        : {metrics['HD95']:.2f}")
    print(f"\n  Per-class IoU:")
    for i, name in enumerate(class_names):
        v = metrics["iou_per_class"][i]
        tag = f"{v:.4f}" if not np.isnan(v) else "  N/A"
        print(f"    {i:2d}  {name:<15s} : {tag}")
    print(sep)
