# -*- coding: utf-8 -*-
"""
Segmentation model definitions:
  1. UNetResNet  – U-Net with ResNet-18 or ResNet-50 encoder (from torchvision)
  2. DeepLabV3PlusWrapper – torchvision DeepLabV3 with modified head
  3. SAMSegmentation – SAM ViT-B image encoder (frozen) + lightweight decoder

All models accept ImageNet-normalised inputs of shape (B, 3, H, W) and return
logits of shape (B, NUM_CLASSES, H, W).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.segmentation import (
    deeplabv3_resnet50, DeepLabV3_ResNet50_Weights,
)

from config import NUM_CLASSES


# ─── U-Net helpers ────────────────────────────────────────────────────────────

class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _Up(nn.Module):
    """Bilinear upsample + DoubleConv with skip connection."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ─── U-Net with ResNet encoder ────────────────────────────────────────────────

class UNetResNet(nn.Module):
    """
    U-Net using a ResNet-18 or ResNet-50 encoder from torchvision.

    Encoder stage outputs:
      ResNet-18: 64, 64, 128, 256, 512
      ResNet-50: 64, 256, 512, 1024, 2048
    """

    def __init__(self, backbone: str = "resnet18",
                 num_classes: int = NUM_CLASSES,
                 pretrained: bool = True):
        super().__init__()
        assert backbone in ("resnet18", "resnet50"), f"Unknown backbone: {backbone}"

        if backbone == "resnet18":
            base = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            c = [64, 64, 128, 256, 512]
        else:
            base = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            c = [64, 256, 512, 1024, 2048]

        # Encoder (5 stages)
        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)  # /2
        self.pool  = base.maxpool                                     # /4
        self.enc1  = base.layer1   # /4
        self.enc2  = base.layer2   # /8
        self.enc3  = base.layer3   # /16
        self.enc4  = base.layer4   # /32

        # Decoder
        self.up4 = _Up(c[4], c[3], 256)
        self.up3 = _Up(256,  c[2], 128)
        self.up2 = _Up(128,  c[1],  64)
        self.up1 = _Up( 64,  c[0],  64)
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _DoubleConv(64, 64),
        )
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d = self.up4(e4, e3)
        d = self.up3(d,  e2)
        d = self.up2(d,  e1)
        d = self.up1(d,  e0)
        d = self.up0(d)
        return self.head(d)


# ─── DeepLabV3+ (torchvision) ─────────────────────────────────────────────────

class DeepLabV3PlusWrapper(nn.Module):
    """
    torchvision DeepLabV3 with ResNet-50 backbone.
    Replaces the final classifier head to output NUM_CLASSES channels.
    Returns plain logits tensor (no 'out' dict).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.model = deeplabv3_resnet50(weights=weights)
        # Swap classification head
        self.model.classifier[4]     = nn.Conv2d(256, num_classes, 1)
        self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        out    = self.model(x)
        logits = out["out"]
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:],
                                   mode="bilinear", align_corners=False)
        return logits


# ─── SAM-based segmentation model ────────────────────────────────────────────

class _SAMDecoder(nn.Module):
    """
    Upsamples SAM neck output (B, 256, 64, 64) → (B, num_classes, 256, 256).
    Two 2x bilinear upsample steps with conv layers.
    """

    def __init__(self, in_channels: int = 256, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            # 64 → 128
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # 128 → 256
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # final head
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        return self.net(x)


class SAMSegmentation(nn.Module):
    """
    SAM ViT-B image encoder (frozen by default) + trainable decoder.

    The encoder outputs (B, 256, 64, 64) for 1024x1024 input.
    We resize our images to 1024x1024 before passing to the encoder.

    Note: SAM uses the same ImageNet normalisation as our dataset pipeline,
    so no re-normalisation is needed — normalised tensors can be passed directly.

    Requires:
        pip install git+https://github.com/facebookresearch/segment-anything.git
        Checkpoint: sam_vit_b_01ec64.pth  (placed at config.SAM_CHECKPOINT)
    """

    def __init__(self, sam_checkpoint: str,
                 model_type: str = "vit_b",
                 num_classes: int = NUM_CLASSES,
                 freeze_encoder: bool = True):
        super().__init__()

        try:
            from segment_anything import sam_model_registry
        except ImportError:
            raise ImportError(
                "segment-anything not installed. Run:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.image_encoder = sam.image_encoder

        if freeze_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)

        self.decoder = _SAMDecoder(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        B, C, H, W = x.shape

        # Resize to 1024x1024 (SAM's expected input size)
        x_sam = F.interpolate(x, size=(1024, 1024),
                              mode="bilinear", align_corners=False)

        if self.image_encoder.training or not all(
            not p.requires_grad for p in self.image_encoder.parameters()
        ):
            feats = self.image_encoder(x_sam)         # (B, 256, 64, 64)
        else:
            with torch.no_grad():
                feats = self.image_encoder(x_sam)

        logits = self.decoder(feats)                  # (B, C, 256, 256)
        logits = F.interpolate(logits, size=(H, W),
                               mode="bilinear", align_corners=False)
        return logits


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_model(name: str, **kwargs) -> nn.Module:
    """
    name: one of 'unet18', 'unet50', 'deeplabv3plus', 'sam'
    kwargs forwarded to the constructor.
    """
    if name == "unet18":
        return UNetResNet(backbone="resnet18", **kwargs)
    if name == "unet50":
        return UNetResNet(backbone="resnet50", **kwargs)
    if name == "deeplabv3plus":
        return DeepLabV3PlusWrapper(**kwargs)
    if name == "sam":
        from config import SAM_CHECKPOINT
        return SAMSegmentation(sam_checkpoint=SAM_CHECKPOINT, **kwargs)
    raise ValueError(f"Unknown model name: {name}")
