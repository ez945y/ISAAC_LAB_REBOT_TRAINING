# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Pretrained Feature Extractor using ResNet backbone.

This module provides a feature extractor that uses pre-trained ImageNet weights,
avoiding the need to train CNN from scratch during RL training.
"""

import torch
import torch.nn as nn
import torchvision

from isaaclab.utils import configclass


class PretrainedFeatureExtractorNetwork(nn.Module):
    """Feature extractor using pre-trained CNN backbone (ResNet).
    
    Uses ImageNet pre-trained weights and freezes the backbone by default.
    Only the final projection layer is trainable (if needed).
    
    Args:
        embedding_dim: Output embedding dimension. Default is 128.
        freeze_backbone: If True, freezes the backbone weights. Default is True.
        backbone: Backbone model name. Options: 'resnet18', 'resnet34', 'resnet50'. Default is 'resnet18'.
    """

    def __init__(
        self, 
        embedding_dim: int = 128, 
        freeze_backbone: bool = True,
        backbone: str = "resnet18",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone
        
        # Load pre-trained backbone
        if backbone == "resnet18":
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = torchvision.models.resnet18(weights=weights)
            backbone_out_features = 512
        elif backbone == "resnet34":
            weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            self.backbone = torchvision.models.resnet34(weights=weights)
            backbone_out_features = 512
        elif backbone == "resnet50":
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            self.backbone = torchvision.models.resnet50(weights=weights)
            backbone_out_features = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Options: 'resnet18', 'resnet34', 'resnet50'")
        
        # Remove the final fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        # Projection layer to desired embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_features, embedding_dim),
        )
        
        # ImageNet normalization
        self.data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: RGB image tensor. Shape: (N, H, W, 3), values in [0, 255].
            
        Returns:
            torch.Tensor: Feature embedding of shape (N, embedding_dim).
        """
        # Clone and permute: (N, H, W, C) -> (N, C, H, W)
        x = x.clone().permute(0, 3, 1, 2)
        
        # Normalize to [0, 1] and apply ImageNet normalization
        x = x / 255.0
        x = self.data_transforms(x)
        
        # Forward through backbone
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        
        # Project to embedding dimension
        out = self.projection(features)
        return out
    
    def train(self, mode: bool = True):
        """Override train to keep backbone in eval mode if frozen."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


@configclass
class PretrainedFeatureExtractorCfg:
    """Configuration for the pretrained feature extractor model."""

    backbone: str = "resnet18"
    """Backbone model. Options: 'resnet18', 'resnet34', 'resnet50'. Default is 'resnet18'."""

    freeze_backbone: bool = True
    """If True, freezes the backbone weights. Default is True."""

    embedding_dim: int = 128
    """Output embedding dimension. Default is 128."""

    write_image_to_file: bool = False
    """If True, the images from the camera sensor are written to file. Default is False."""


class PretrainedFeatureExtractor:
    """Class for extracting features from RGB images using pre-trained CNN.

    Uses a pre-trained ResNet backbone (ImageNet weights) to extract visual features.
    The backbone is frozen by default, so no CNN training is needed during RL.
    """

    def __init__(self, cfg: PretrainedFeatureExtractorCfg, device: str):
        """Initialize the pretrained feature extractor.

        Args:
            cfg: Configuration for the feature extractor model.
            device: Device to run the model on.
        """
        self.cfg = cfg
        self.device = device

        # Create the pretrained network
        self.feature_extractor = PretrainedFeatureExtractorNetwork(
            embedding_dim=cfg.embedding_dim,
            freeze_backbone=cfg.freeze_backbone,
            backbone=cfg.backbone,
        )
        self.feature_extractor.to(self.device)
        
        # Always in eval mode since we're not training the backbone
        if cfg.freeze_backbone:
            self.feature_extractor.eval()

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
