# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
MobileNet V3 Feature Extractor.

This module provides a feature extractor using MobileNet V3 Large with ImageNet pretrained weights.
Uses FP16 for faster inference and resizes input to 224x224.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from isaaclab.utils import configclass


class MobileNetFeatureExtractorNetwork(nn.Module):
    """Feature extractor using MobileNet V3 Large backbone.
    
    Uses ImageNet pretrained weights and freezes the backbone by default.
    Input is resized to 224x224 and processed in FP16 for faster inference.
    
    Args:
        embedding_dim: Output embedding dimension. Default is 128.
        freeze_backbone: If True, freezes the backbone weights. Default is True.
        use_fp16: If True, uses FP16 for inference. Default is True.
    """

    def __init__(
        self, 
        embedding_dim: int = 128, 
        freeze_backbone: bool = True,
        use_fp16: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone
        self.use_fp16 = use_fp16
        
        # Load pretrained MobileNet V3 Large
        weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.backbone = torchvision.models.mobilenet_v3_large(weights=weights)
        
        # MobileNet V3 Large outputs 960 features before classifier
        backbone_out_features = 960
        
        # Remove the classifier (keep features only)
        self.backbone.classifier = nn.Identity()
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        # Convert backbone to FP16 if requested
        if use_fp16:
            self.backbone = self.backbone.half()
        
        # Projection layer to desired embedding dimension (keep in FP32 for stability)
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_features, embedding_dim),
        )
        
        # ImageNet normalization
        self.register_buffer(
            'mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', 
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
            """Forward pass with automatic Aspect-Ratio preserving resize and padding.
            
            Args:
                x: RGB image tensor. Shape: (N, H, W, 3), values in [0, 255].
            """
            # 1. 轉置維度: (N, H, W, C) -> (N, C, H, W)
            x = x.permute(0, 3, 1, 2).float().contiguous()
            
            # 2. 取得原始尺寸
            h, w = x.shape[2], x.shape[3]
            target_size = 224
            
            # 3. 計算縮放比例 (以長邊為準)
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # 4. 等比例縮放
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            # 5. 計算需要補黑邊的量 (Padding)
            pad_top = (target_size - new_h) // 2
            pad_bottom = target_size - new_h - pad_top
            pad_left = (target_size - new_w) // 2
            pad_right = target_size - new_w - pad_left
            
            # 6. 填補黑邊: F.pad 參數順序是 (left, right, top, bottom)
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0)
            
            # 7. 標準化與半精度轉換
            x = x / 255.0
            x = (x - self.mean) / self.std
            
            if self.use_fp16:
                x = x.half()
                
            # 8. 特徵提取 (凍結狀態)
            with torch.no_grad():
                features = self.backbone(x)
                features = features.float() # 轉回 FP32 確保 Projection 穩定
                
            return self.projection(features)
    
    def train(self, mode: bool = True):
        """Override train to keep backbone in eval mode if frozen."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


@configclass
class MobileNetFeatureExtractorCfg:
    """Configuration for the MobileNet feature extractor model."""

    freeze_backbone: bool = True
    """If True, freezes the backbone weights. Default is True."""

    embedding_dim: int = 128
    """Output embedding dimension. Default is 128."""

    use_fp16: bool = True
    """If True, uses FP16 for inference. Default is True."""


class MobileNetFeatureExtractor:
    """Class for extracting features from RGB images using MobileNet V3 Large.

    Uses ImageNet pretrained weights. The backbone is frozen by default,
    so no CNN training is needed during RL. Uses FP16 for faster inference.
    """

    def __init__(self, cfg: MobileNetFeatureExtractorCfg, device: str):
        """Initialize the MobileNet feature extractor.

        Args:
            cfg: Configuration for the feature extractor model.
            device: Device to run the model on.
        """
        self.cfg = cfg
        self.device = device

        # Create the MobileNet network
        self.feature_extractor = MobileNetFeatureExtractorNetwork(
            embedding_dim=cfg.embedding_dim,
            freeze_backbone=cfg.freeze_backbone,
            use_fp16=cfg.use_fp16,
        )
        self.feature_extractor.to(self.device)
        
        # Always in eval mode since we're not training the backbone
        if cfg.freeze_backbone:
            self.feature_extractor.eval()
        
        print(f"[INFO]: MobileNet V3 Large loaded (FP16={cfg.use_fp16}, embedding_dim={cfg.embedding_dim})")

    def step(self, rgb_img: torch.Tensor) -> torch.Tensor:
        """Extract features from RGB image.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3), values in [0, 255].

        Returns:
            torch.Tensor: Feature embedding of shape (N, embedding_dim).
        """
        # Forward through the network
        with torch.no_grad():
            embeddings = self.feature_extractor(rgb_img)
        
        return embeddings.clone().detach()
