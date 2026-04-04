# -*- coding: utf-8 -*-
"""
Mini-Project 2 – Main entry point.

Usage examples
--------------
# Train all three models (60 epochs each):
    python main.py --models unet18 unet50 deeplabv3plus --epochs 60

# Also train SAM-based model (requires checkpoint + package):
    python main.py --models unet18 deeplabv3plus sam --epochs 60

# Skip training, load checkpoints, run evaluation + visualisation only:
    python main.py --models unet18 unet50 deeplabv3plus --eval-only

# Run ablation studies (30 epochs per run):
    python main.py --ablations all --ablation-epochs 30
"""

import argparse
import os

import numpy as np
import torch

from config  import DEVICE, EPOCHS, LR, RESULTS_DIR
from dataset import get_loaders
from losses  import get_loss
from models  import build_model
from train   import train_model, evaluate_loader
from metrics import compute_metrics, print_metrics
from visualize import (
    save_mosaic, save_best_worst, plot_history,
    plot_confusion, compare_models,
)


# ─── Collect all predictions for vis ─────────────────────────────────────────

@torch.no_grad()
def _collect_preds(model, loader, device=DEVICE, n_samples=None):
    """Return (images_list, preds_list, targets_list) from val loader."""
    model.eval()
    images_all, preds_all, targets_all = [], [], []

    for imgs, masks in loader:
        imgs  = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        if logits.shape[2:] != masks.shape[1:]:
            import torch.nn.functional as F
            logits = F.interpolate(logits, size=masks.shape[1:],
                                   mode="bilinear", align_corners=False)
        preds = logits.argmax(dim=1).cpu().numpy()
        images_all.extend([imgs[i].cpu() for i in range(imgs.size(0))])
        preds_all.extend(preds)
        targets_all.extend(masks.numpy())

        if n_samples and len(images_all) >= n_samples:
            break

    return images_all, preds_all, targets_all


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=["unet18", "unet50", "deeplabv3plus", "sam"],
                        default=["unet18", "unet50", "deeplabv3plus"],
                        help="Models to train/evaluate")
    parser.add_argument("--epochs",    type=int,   default=EPOCHS)
    parser.add_argument("--lr",        type=float, default=LR)
    parser.add_argument("--loss",      type=str,   default="combined",
                        choices=["ce", "dice", "combined"])
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training; load best checkpoints and evaluate")
    parser.add_argument("--ablations", nargs="*",
                        choices=["backbone", "augment", "loss", "pretrain", "all"],
                        default=None,
                        help="Run ablation studies after main training")
    parser.add_argument("--ablation-epochs", type=int, default=30)
    args = parser.parse_args()

    # ── Data ─────────────────────────────────────────────────────────────────
    augment = not args.no_augment
    train_loader, val_loader = get_loaders(augment=augment)
    criterion = get_loss(args.loss)

    print(f"\nDevice  : {DEVICE}")
    print(f"Models  : {args.models}")
    print(f"Epochs  : {args.epochs}")
    print(f"Loss    : {args.loss}")
    print(f"Augment : {augment}")

    # ── Train / load ─────────────────────────────────────────────────────────
    all_histories = {}
    all_metrics   = {}

    for model_name in args.models:
        is_sam = (model_name == "sam")

        # Build
        try:
            model = build_model(model_name)
        except Exception as e:
            print(f"\n  [WARN] Could not build {model_name}: {e}\n  Skipping.")
            continue

        ckpt_path = os.path.join("checkpoints", f"{model_name}_best.pth")

        if args.eval_only:
            if not os.path.exists(ckpt_path):
                print(f"  [WARN] No checkpoint found for {model_name} – skipping.")
                continue
            print(f"\n  Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model.to(DEVICE)
        else:
            history, _ = train_model(
                model, train_loader, val_loader, criterion,
                name=model_name,
                epochs=args.epochs,
                lr=args.lr,
                only_decoder_params=is_sam,
            )
            all_histories[model_name] = history

            # Reload best
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model.to(DEVICE)

        # ── Evaluate ──────────────────────────────────────────────────────
        print(f"\n  Evaluating {model_name} on validation set (with HD95)…")
        _, m = evaluate_loader(model, val_loader, criterion,
                               device=DEVICE, compute_hd95=True)
        all_metrics[model_name] = m
        print_metrics(m, model_name=model_name)

        # ── Visualise ─────────────────────────────────────────────────────
        print(f"  Generating visualisations for {model_name}…")
        imgs, preds, targets = _collect_preds(model, val_loader,
                                              device=DEVICE, n_samples=64)

        save_mosaic(imgs[:8], preds[:8], targets[:8],
                    title=f"{model_name}_mosaic", n_cols=4)

        save_best_worst(imgs, preds, targets, model_name=model_name, topk=3)

        plot_confusion(m["confusion_matrix"], model_name=model_name)

    # ── Cross-model comparison ────────────────────────────────────────────
    if len(all_metrics) >= 2:
        compare_models(all_metrics)

    if all_histories:
        plot_history(all_histories)

    # ── Ablation studies ─────────────────────────────────────────────────
    if args.ablations:
        from ablation import (
            ablation_backbone, ablation_augmentation,
            ablation_loss, ablation_pretrain,
        )
        to_run = set(args.ablations)
        run_all = "all" in to_run
        if run_all or "backbone" in to_run:
            ablation_backbone(epochs=args.ablation_epochs)
        if run_all or "augment" in to_run:
            ablation_augmentation(epochs=args.ablation_epochs)
        if run_all or "loss" in to_run:
            ablation_loss(epochs=args.ablation_epochs)
        if run_all or "pretrain" in to_run:
            ablation_pretrain(epochs=args.ablation_epochs)

    print(f"\n  All results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
