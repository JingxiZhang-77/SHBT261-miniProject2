# -*- coding: utf-8 -*-
"""
Training and evaluation loop utilities.
"""

import os
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import DEVICE, EPOCHS, LR, WEIGHT_DECAY, CHECKPOINT_DIR
from metrics import compute_metrics


# ─── Single epoch ────────────────────────────────────────────────────────────

def _resize_logits(logits, target_size):
    if logits.shape[2:] != target_size:
        logits = F.interpolate(logits, size=target_size,
                               mode="bilinear", align_corners=False)
    return logits


def train_one_epoch(model, loader, optimizer, criterion, device=DEVICE,
                    scaler=None):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                logits = _resize_logits(logits, masks.shape[1:])
                loss   = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            logits = _resize_logits(logits, masks.shape[1:])
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device=DEVICE,
                    compute_hd95=False):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, masks in tqdm(loader, desc="  eval ", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        logits = model(images)
        logits = _resize_logits(logits, masks.shape[1:])

        running_loss += criterion(logits, masks).item()

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(masks.cpu().numpy())

    metrics = compute_metrics(all_preds, all_targets, compute_hd95=compute_hd95)
    return running_loss / len(loader), metrics


# ─── Full training loop ───────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, criterion,
                name: str = "model",
                epochs: int = EPOCHS,
                lr: float = LR,
                weight_decay: float = WEIGHT_DECAY,
                device: str = DEVICE,
                only_decoder_params: bool = False):
    """
    Train model and save best checkpoint by mIoU.

    Args:
        only_decoder_params: if True, only optimise parameters with
                             requires_grad=True (used for SAM frozen encoder).
    Returns:
        (history dict, best_miou)
    """
    model = model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad] \
                if only_decoder_params else list(model.parameters())

    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler    = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best_miou = 0.0
    history   = {"train_loss": [], "val_loss": [], "miou": [], "mdice": []}

    print(f"\n{'='*64}")
    print(f"  Model : {name}")
    print(f"  Params: {sum(p.numel() for p in trainable) / 1e6:.1f}M  "
          f"| Device: {device} | Epochs: {epochs}")
    print(f"{'='*64}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                  device, scaler)
        val_loss, m = evaluate_loader(model, val_loader, criterion, device,
                                      compute_hd95=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["miou"].append(m["mIoU"])
        history["mdice"].append(m["mDice"])

        elapsed = time.time() - t0
        print(f"  [{epoch:3d}/{epochs}] "
              f"loss {tr_loss:.4f}/{val_loss:.4f}  "
              f"mIoU {m['mIoU']:.4f}  "
              f"mDice {m['mDice']:.4f}  "
              f"pixAcc {m['pixel_acc']:.4f}  "
              f"{elapsed:.0f}s")

        if m["mIoU"] > best_miou:
            best_miou = m["mIoU"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "metrics": m, "history": history},
                       os.path.join(CHECKPOINT_DIR, f"{name}_best.pth"))
            print(f"    --> saved (mIoU {best_miou:.4f})")

    # Always save final state
    torch.save({"epoch": epochs, "model_state": model.state_dict(),
                "history": history},
               os.path.join(CHECKPOINT_DIR, f"{name}_final.pth"))

    print(f"\n  Best mIoU [{name}]: {best_miou:.4f}")
    return history, best_miou


def load_best(model, name: str, device: str = DEVICE):
    """Load best checkpoint weights into model and return it."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    return model
