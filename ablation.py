# -*- coding: utf-8 -*-
"""
Ablation study runner.

Four ablations (run any subset via command-line flags):
  A1 – Backbone size     : UNet-ResNet18 vs UNet-ResNet50
  A2 – Data augmentation : with augmentation vs without
  A3 – Loss function     : CE vs Dice vs Combined
  A4 – Pre-training      : ImageNet pretrained vs from scratch

Results are printed and saved to results/ablation_*.png
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from config import DEVICE, RESULTS_DIR, CHECKPOINT_DIR
from dataset import get_loaders
from models  import build_model
from losses  import get_loss
from train   import train_model, evaluate_loader
from metrics import compute_metrics, print_metrics
from visualize import plot_history


# ─── Shared mini-train helper ─────────────────────────────────────────────────

def _run(tag, model, train_loader, val_loader, criterion,
         epochs=30, lr=1e-4, only_decoder=False):
    """Train model and return (history, best_miou, final_metrics)."""
    history, best_miou = train_model(
        model, train_loader, val_loader, criterion,
        name=tag, epochs=epochs, lr=lr,
        only_decoder_params=only_decoder,
    )
    # Reload best checkpoint for final evaluation
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f"{tag}_best.pth"),
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    _, m = evaluate_loader(model, val_loader, criterion,
                           device=DEVICE, compute_hd95=True)
    return history, best_miou, m


# ─── Ablation helpers ────────────────────────────────────────────────────────

def _bar_chart(results: dict, metric: str, title: str, filename: str):
    labels = list(results.keys())
    values = [results[k][metric] for k in labels]
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.5), 4))
    bars = ax.bar(labels, values, color=plt.cm.tab10(np.linspace(0, 1, len(labels))))
    ax.set_ylabel(metric); ax.set_title(title)
    ax.set_ylim(0, min(1, max(values) * 1.25))
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  [ablation] chart saved → {path}")


# ─── A1: Backbone size ───────────────────────────────────────────────────────

def ablation_backbone(epochs=30, augment=True):
    print("\n" + "=" * 60)
    print("  ABLATION A1 – Backbone: ResNet-18 vs ResNet-50")
    print("=" * 60)
    train_loader, val_loader = get_loaders(augment=augment)
    criterion = get_loss("combined")
    results   = {}
    histories = {}

    for backbone in ("resnet18", "resnet50"):
        tag  = f"ablation_backbone_{backbone}"
        name = f"UNet-{backbone}"
        model = build_model("unet18" if backbone == "resnet18" else "unet50")
        h, best, m = _run(tag, model, train_loader, val_loader, criterion, epochs)
        results[name]   = m
        histories[name] = h
        print_metrics(m, model_name=name)

    _bar_chart(results, "mIoU",  "A1 – mIoU: Backbone Size",  "ablation_backbone_miou.png")
    _bar_chart(results, "mDice", "A1 – mDice: Backbone Size", "ablation_backbone_mdice.png")
    plot_history(histories, save_name="ablation_backbone_curves.png")
    return results


# ─── A2: Data augmentation ───────────────────────────────────────────────────

def ablation_augmentation(epochs=30):
    print("\n" + "=" * 60)
    print("  ABLATION A2 – Data Augmentation: With vs Without")
    print("=" * 60)
    criterion = get_loss("combined")
    results   = {}
    histories = {}

    for aug in (True, False):
        train_loader, val_loader = get_loaders(augment=aug)
        tag  = f"ablation_aug_{'on' if aug else 'off'}"
        name = f"Augment={'ON' if aug else 'OFF'}"
        model = build_model("unet18")
        h, best, m = _run(tag, model, train_loader, val_loader, criterion, epochs)
        results[name]   = m
        histories[name] = h
        print_metrics(m, model_name=name)

    _bar_chart(results, "mIoU",  "A2 – mIoU: Augmentation",  "ablation_aug_miou.png")
    _bar_chart(results, "mDice", "A2 – mDice: Augmentation", "ablation_aug_mdice.png")
    plot_history(histories, save_name="ablation_aug_curves.png")
    return results


# ─── A3: Loss function ───────────────────────────────────────────────────────

def ablation_loss(epochs=30, augment=True):
    print("\n" + "=" * 60)
    print("  ABLATION A3 – Loss Function: CE vs Dice vs Combined")
    print("=" * 60)
    train_loader, val_loader = get_loaders(augment=augment)
    results   = {}
    histories = {}

    for loss_type in ("ce", "dice", "combined"):
        tag  = f"ablation_loss_{loss_type}"
        name = f"Loss={loss_type.upper()}"
        model     = build_model("unet18")
        criterion = get_loss(loss_type)
        h, best, m = _run(tag, model, train_loader, val_loader, criterion, epochs)
        results[name]   = m
        histories[name] = h
        print_metrics(m, model_name=name)

    _bar_chart(results, "mIoU",  "A3 – mIoU: Loss Function",  "ablation_loss_miou.png")
    _bar_chart(results, "mDice", "A3 – mDice: Loss Function", "ablation_loss_mdice.png")
    plot_history(histories, save_name="ablation_loss_curves.png")
    return results


# ─── A4: Pre-training ────────────────────────────────────────────────────────

def ablation_pretrain(epochs=30, augment=True):
    print("\n" + "=" * 60)
    print("  ABLATION A4 – Pre-training: Pretrained vs From Scratch")
    print("=" * 60)
    train_loader, val_loader = get_loaders(augment=augment)
    criterion = get_loss("combined")
    results   = {}
    histories = {}

    for pretrained in (True, False):
        tag  = f"ablation_pretrain_{'yes' if pretrained else 'no'}"
        name = f"Pretrained={'YES' if pretrained else 'NO'}"
        model = build_model("unet18", pretrained=pretrained)
        h, best, m = _run(tag, model, train_loader, val_loader, criterion, epochs)
        results[name]   = m
        histories[name] = h
        print_metrics(m, model_name=name)

    _bar_chart(results, "mIoU",  "A4 – mIoU: Pre-training",  "ablation_pretrain_miou.png")
    _bar_chart(results, "mDice", "A4 – mDice: Pre-training", "ablation_pretrain_mdice.png")
    plot_history(histories, save_name="ablation_pretrain_curves.png")
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--ablations", nargs="+",
                        choices=["backbone", "augment", "loss", "pretrain", "all"],
                        default=["all"],
                        help="Which ablation(s) to run")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs per ablation run")
    args = parser.parse_args()

    to_run = set(args.ablations)
    run_all = "all" in to_run

    if run_all or "backbone" in to_run:
        ablation_backbone(epochs=args.epochs)
    if run_all or "augment" in to_run:
        ablation_augmentation(epochs=args.epochs)
    if run_all or "loss" in to_run:
        ablation_loss(epochs=args.epochs)
    if run_all or "pretrain" in to_run:
        ablation_pretrain(epochs=args.epochs)
