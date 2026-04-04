# -*- coding: utf-8 -*-
"""
Visualisation utilities:
  - decode_segmap   : class-index mask → RGB using VOC colormap
  - mosaic          : grid of (image | pred | gt) triples
  - best_worst      : top-3 best and worst predictions by IoU on 'person' class
  - plot_history    : training curves
  - plot_confusion  : confusion-matrix heatmap
  - compare_models  : bar chart of per-class IoU across multiple models
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; change to "TkAgg" for interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from config import VOC_CLASSES, VOC_COLORMAP, RESULTS_DIR, NUM_CLASSES

# Pre-build colormap arrays once
_CMAP_ARRAY = np.array(VOC_COLORMAP, dtype=np.uint8)  # (21, 3)
_MPL_CMAP   = ListedColormap([np.array(c) / 255.0 for c in VOC_COLORMAP])

PERSON_CLS = VOC_CLASSES.index("person")   # = 15


# ─── Helpers ─────────────────────────────────────────────────────────────────

def decode_segmap(mask: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W) class-index mask to an (H, W, 3) uint8 RGB image.
    Values > 20 (e.g. 255 boundary) are shown as black.
    """
    mask = mask.copy()
    mask[mask > 20] = 0
    return _CMAP_ARRAY[mask]


def _unnorm(tensor) -> np.ndarray:
    """ImageNet-unnormalise a (3, H, W) float tensor → uint8 (H, W, 3)."""
    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).cpu().numpy()
    img  = img * STD + MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _legend():
    """Return a list of matplotlib Patch handles for the VOC legend."""
    return [
        mpatches.Patch(color=np.array(c) / 255.0, label=f"{i} {VOC_CLASSES[i]}")
        for i, c in enumerate(VOC_COLORMAP)
    ]


# ─── Mosaic ───────────────────────────────────────────────────────────────────

def save_mosaic(images, preds, targets, title="mosaic",
                n_cols=4, save_dir=RESULTS_DIR):
    """
    images  : list of (3,H,W) tensors (normalised)
    preds   : list of (H,W) np.ndarray  – predicted class idx
    targets : list of (H,W) np.ndarray  – gt class idx

    Each group of 3 columns = [Input | Prediction | Ground Truth]
    """
    n = min(len(images), n_cols)
    fig, axes = plt.subplots(3, n, figsize=(n * 3.5, 10))
    if n == 1:
        axes = axes[:, np.newaxis]

    row_labels = ["Input", "Prediction", "Ground Truth"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=4)

    for i in range(n):
        axes[0, i].imshow(_unnorm(images[i]))
        axes[0, i].axis("off")

        axes[1, i].imshow(decode_segmap(preds[i]))
        axes[1, i].axis("off")

        axes[2, i].imshow(decode_segmap(targets[i]))
        axes[2, i].axis("off")

    fig.suptitle(title, fontsize=14, y=1.01)
    # compact legend below figure
    fig.legend(handles=_legend(), loc="lower center",
               ncol=7, fontsize=7, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] mosaic saved → {path}")


# ─── Best / Worst ─────────────────────────────────────────────────────────────

def _person_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute IoU for the 'person' class between one pred/target pair."""
    p_mask = pred   == PERSON_CLS
    g_mask = target == PERSON_CLS
    inter  = (p_mask & g_mask).sum()
    union  = (p_mask | g_mask).sum()
    return inter / union if union > 0 else float("nan")


def save_best_worst(images, preds, targets, model_name="model",
                    topk=3, save_dir=RESULTS_DIR):
    """
    Show top-3 best and top-3 worst predictions ranked by person-class IoU.
    Saves two figures: {model_name}_best3.png  and  {model_name}_worst3.png
    """
    scores = []
    for i, (p, t) in enumerate(zip(preds, targets)):
        v = _person_iou(p, t)
        if not np.isnan(v):
            scores.append((v, i))

    if not scores:
        print("  [viz] no person class found in this batch – skipping best/worst.")
        return

    scores.sort(key=lambda x: x[0])

    for tag, indices in [("worst", scores[:topk]),
                         ("best",  scores[-topk:][::-1])]:
        k = len(indices)
        fig, axes = plt.subplots(3, k, figsize=(k * 3.5, 10))
        if k == 1:
            axes = axes[:, np.newaxis]

        for col, (score, idx) in enumerate(indices):
            axes[0, col].imshow(_unnorm(images[idx]))
            axes[0, col].set_title(f"person IoU={score:.3f}", fontsize=9)
            axes[0, col].axis("off")
            axes[1, col].imshow(decode_segmap(preds[idx]))
            axes[1, col].axis("off")
            axes[2, col].imshow(decode_segmap(targets[idx]))
            axes[2, col].axis("off")

        axes[0, 0].set_ylabel("Input",       fontsize=11)
        axes[1, 0].set_ylabel("Prediction",  fontsize=11)
        axes[2, 0].set_ylabel("Ground Truth",fontsize=11)

        fig.suptitle(f"{model_name} – {tag.upper()} {topk} (person IoU)", fontsize=13)
        fig.legend(handles=_legend(), loc="lower center",
                   ncol=7, fontsize=7, bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()
        path = os.path.join(save_dir, f"{model_name}_{tag}{topk}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] {tag} results saved → {path}")


# ─── Training curves ─────────────────────────────────────────────────────────

def plot_history(histories: dict, save_dir=RESULTS_DIR,
                 save_name="training_curves.png"):
    """
    histories : { model_name: { 'train_loss', 'val_loss', 'miou', 'mdice' } }
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for name, h in histories.items():
        ep = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], label=f"{name} train")
        axes[0].plot(ep, h["val_loss"],   label=f"{name} val", linestyle="--")
        axes[1].plot(ep, h["miou"],        label=name)
        axes[2].plot(ep, h["mdice"],       label=name)

    axes[0].set_title("Loss");          axes[0].set_xlabel("Epoch"); axes[0].legend(fontsize=8)
    axes[1].set_title("mIoU");          axes[1].set_xlabel("Epoch"); axes[1].legend(fontsize=8)
    axes[2].set_title("mDice");         axes[2].set_xlabel("Epoch"); axes[2].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, save_name)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [viz] training curves → {path}")


# ─── Confusion matrix ────────────────────────────────────────────────────────

def plot_confusion(cm: np.ndarray, model_name="model", save_dir=RESULTS_DIR):
    """Normalised (row-wise) confusion matrix heatmap."""
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm / row_sum, 0.0)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(VOC_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(VOC_CLASSES, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix (normalised) – {model_name}")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm_norm[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v > 0.5 else "black")

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_confusion.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  [viz] confusion matrix → {path}")


# ─── Model comparison bar chart ──────────────────────────────────────────────

def compare_models(model_metrics: dict, save_dir=RESULTS_DIR):
    """
    model_metrics : { model_name: metrics_dict }
    Plots per-class IoU as grouped bars + summary table.
    """
    names  = list(model_metrics.keys())
    n_m    = len(names)
    x      = np.arange(NUM_CLASSES)
    width  = 0.8 / n_m
    colors = plt.cm.tab10(np.linspace(0, 1, n_m))

    fig, ax = plt.subplots(figsize=(18, 5))
    for i, (name, m) in enumerate(model_metrics.items()):
        iou = m["iou_per_class"]
        iou = np.where(np.isnan(iou), 0, iou)
        ax.bar(x + i * width, iou, width, label=name, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width * (n_m - 1) / 2)
    ax.set_xticklabels(VOC_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("IoU"); ax.set_ylim(0, 1)
    ax.set_title("Per-class IoU comparison")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "per_class_iou_comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [viz] per-class IoU comparison → {path}")

    # Summary table
    header = f"{'Model':<22}  {'mIoU':>6}  {'mDice':>6}  {'pixAcc':>7}  {'HD95':>7}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for name, m in model_metrics.items():
        print(f"  {name:<20}  {m['mIoU']:>6.4f}  {m['mDice']:>6.4f}"
              f"  {m['pixel_acc']:>7.4f}  {m['HD95']:>7.2f}")
    print("─" * len(header))
