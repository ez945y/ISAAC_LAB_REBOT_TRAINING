# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import torch
import torch.nn as nn
import torchvision

from isaaclab.sensors import save_images_to_file
from isaaclab.utils import configclass


class FeatureExtractorNetwork(nn.Module):
    """CNN architecture used to regress keypoint positions from image data.
    
    Supports variable input sizes (e.g., 640x480, 120x120) by using BatchNorm2d 
    and AdaptiveAvgPool2d instead of fixed LayerNorm and AvgPool2d.
    
    Args:
        rgb_only: If True, the network only uses RGB images (3 channels).
                  If False, uses RGB + depth + segmentation (7 channels).
        embedding_dim: Output embedding dimension. Default is 27.
    """

    def __init__(self, rgb_only: bool = False, embedding_dim: int = 27):
        super().__init__()
        self.rgb_only = rgb_only
        num_channel = 3 if rgb_only else 7
        
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d(1),  # Output: (batch, 128, 1, 1) regardless of input size
        )

        self.linear = nn.Sequential(
            nn.Linear(128, embedding_dim),
        )

        self.data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        # Clone to avoid inplace operations on inference tensors
        x = x.clone().permute(0, 3, 1, 2).contiguous()
        
        # Normalize RGB channels (0:3)
        rgb_normalized = self.data_transforms(x[:, 0:3, :, :])
        
        if self.rgb_only:
            # RGB only mode: just use normalized RGB
            x = rgb_normalized
        else:
            # Full mode: normalize segmentation channels (4:7) and concatenate
            seg_normalized = self.data_transforms(x[:, 4:7, :, :])
            x = torch.cat([rgb_normalized, x[:, 3:4, :, :], seg_normalized], dim=1)
        
        cnn_x = self.cnn(x)
        out = self.linear(cnn_x.view(-1, 128))
        return out


@configclass
class FeatureExtractorCfg:
    """Configuration for the feature extractor model."""

    rgb_only: bool = False
    """If True, uses only RGB images (3 channels). If False, uses RGB + depth + segmentation (7 channels). Default is False for backward compatibility."""

    train: bool = True
    """If True, the feature extractor model is trained during the rollout process. Default is False."""

    load_checkpoint: bool = False
    """If True, the feature extractor model is loaded from a checkpoint. Default is False."""

    write_image_to_file: bool = False
    """If True, the images from the camera sensor are written to file. Default is False."""


class FeatureExtractor:
    """Class for extracting features from image data.

    It uses a CNN to regress keypoint positions from normalized RGB, depth, and segmentation images.
    If the train flag is set to True, the CNN is trained during the rollout process.
    """

    def __init__(self, cfg: FeatureExtractorCfg, device: str, log_dir: str | None = None):
        """Initialize the feature extractor model.

        Args:
            cfg: Configuration for the feature extractor model.
            device: Device to run the model on.
            log_dir: Directory to save checkpoints. If None, uses local "logs" folder resolved with respect to this file.
        """

        self.cfg = cfg
        self.device = device

        # Feature extractor model
        self.feature_extractor = FeatureExtractorNetwork(rgb_only=cfg.rgb_only)
        self.feature_extractor.to(self.device)

        self.step_count = 0
        if log_dir is not None:
            self.log_dir = log_dir
        else:
            self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.cfg.load_checkpoint:
            list_of_files = glob.glob(self.log_dir + "/*.pth")
            latest_file = max(list_of_files, key=os.path.getctime)
            checkpoint = os.path.join(self.log_dir, latest_file)
            print(f"[INFO]: Loading feature extractor checkpoint from {checkpoint}")
            self.feature_extractor.load_state_dict(torch.load(checkpoint, weights_only=True))

        if self.cfg.train:
            self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
            self.l2_loss = nn.MSELoss()
            self.feature_extractor.train()
        else:
            self.feature_extractor.eval()

    def _preprocess_images(
        self,
        rgb_img: torch.Tensor,
        depth_img: torch.Tensor | None = None,
        segmentation_img: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Preprocesses the input images.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor | None): Depth image tensor. Shape: (N, H, W, 1). Optional in rgb_only mode.
            segmentation_img (torch.Tensor | None): Segmentation image tensor. Shape: (N, H, W, 3). Optional in rgb_only mode.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]: Preprocessed RGB, depth, and segmentation
        """
        rgb_img = rgb_img / 255.0
        
        # Only process depth and segmentation if not in rgb_only mode
        if not self.cfg.rgb_only and depth_img is not None and segmentation_img is not None:
            # process depth image
            depth_img[depth_img == float("inf")] = 0
            depth_img /= 5.0
            max_depth = torch.max(depth_img)
            if max_depth > 0:
                depth_img /= max_depth
            # process segmentation image
            segmentation_img = segmentation_img / 255.0
            mean_tensor = torch.mean(segmentation_img, dim=(1, 2), keepdim=True)
            segmentation_img -= mean_tensor
        else:
            depth_img = None
            segmentation_img = None
            
        return rgb_img, depth_img, segmentation_img

    def _save_images(
        self,
        rgb_img: torch.Tensor,
        depth_img: torch.Tensor | None = None,
        segmentation_img: torch.Tensor | None = None,
    ):
        """Writes image buffers to file.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor | None): Depth image tensor. Shape: (N, H, W, 1).
            segmentation_img (torch.Tensor | None): Segmentation image tensor. Shape: (N, H, W, 3).
        """
        save_images_to_file(rgb_img, "shadow_hand_rgb.png")
        if depth_img is not None:
            save_images_to_file(depth_img, "shadow_hand_depth.png")
        if segmentation_img is not None:
            save_images_to_file(segmentation_img, "shadow_hand_segmentation.png")

    def step(
        self,
        rgb_img: torch.Tensor,
        depth_img: torch.Tensor | None = None,
        segmentation_img: torch.Tensor | None = None,
        gt_pose: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Extracts the features using the images and trains the model if the train flag is set to True.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor | None): Depth image tensor. Shape: (N, H, W, 1). Required if rgb_only=False.
            segmentation_img (torch.Tensor | None): Segmentation image tensor. Shape: (N, H, W, 3). Required if rgb_only=False.
            gt_pose (torch.Tensor | None): Ground truth pose tensor (position and corners). Shape: (N, 27). Required if train=True.

        Returns:
            tuple[torch.Tensor | None, torch.Tensor]: Pose loss and predicted pose.
        """
        rgb_img, depth_img, segmentation_img = self._preprocess_images(rgb_img, depth_img, segmentation_img)

        if self.cfg.write_image_to_file:
            self._save_images(rgb_img, depth_img, segmentation_img)

        # Build input tensor based on mode
        if self.cfg.rgb_only:
            img_input = rgb_img
        else:
            img_input = torch.cat((rgb_img, depth_img, segmentation_img), dim=-1)

        if self.cfg.train:
            with torch.enable_grad():
                with torch.inference_mode(False):
                    self.optimizer.zero_grad()

                    predicted_pose = self.feature_extractor(img_input)
                    pose_loss = self.l2_loss(predicted_pose, gt_pose.clone()) * 100

                    pose_loss.backward()
                    self.optimizer.step()

                    if self.step_count % 50000 == 0:
                        torch.save(
                            self.feature_extractor.state_dict(),
                            os.path.join(self.log_dir, f"cnn_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth"),
                        )

                    self.step_count += 1

                    return pose_loss, predicted_pose
        else:
            predicted_pose = self.feature_extractor(img_input)
            return None, predicted_pose
