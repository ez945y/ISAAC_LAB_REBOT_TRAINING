# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData
from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rel_ee_object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    object_data: ArticulationData = env.scene["object"].data

    return object_data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


def rel_ee_drawer_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the current target drawer handle."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    cabinet_tf_data: FrameTransformerData = env.scene["cabinet_frame"].data
    
    cmd = env.command_manager.get_term("drawer_task")
    batch_idx = torch.arange(env.num_envs, device=cmd.current_frame_idx.device)
    
    handle_pos = cabinet_tf_data.target_pos_w[batch_idx, cmd.current_frame_idx, :]
    return handle_pos - ee_tf_data.target_pos_w[..., 0, :]


def current_drawer_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The current target drawer index.
    
    Returns:
        torch.Tensor: Shape (num_envs, 1), values 0 (bottom) or 1 (top)
    """
    return env.command_manager.get_command("drawer_task")


def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the fingertips relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    fingertips_pos = ee_tf_data.target_pos_w[..., 1:, :] - env.scene.env_origins.unsqueeze(1)

    return fingertips_pos.view(env.num_envs, -1)


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

    return ee_pos


def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat


def wrist_camera_rgb(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    """Get RGB image from wrist camera, flattened to vector."""
    camera = env.scene[asset_cfg.name]
    rgb = camera.data.output["rgb"]           # shape: (num_envs, h, w, 4) or (num_envs, h, w, 3)
    rgb = rgb[..., :3]                        # remove alpha channel if present
    # Use contiguous().reshape() instead of view() for non-contiguous tensors
    return rgb.contiguous().reshape(env.num_envs, -1)  # → (num_envs, h*w*3)


def front_camera_rgb(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    """Get RGB image from front camera, flattened to vector."""
    camera = env.scene[asset_cfg.name]
    rgb = camera.data.output["rgb"]
    rgb = rgb[..., :3]
    # Use contiguous().reshape() instead of view() for non-contiguous tensors
    return rgb.contiguous().reshape(env.num_envs, -1)


def wrist_camera_embedding(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    """Get CNN embedding from wrist camera using FeatureExtractor.
    
    The FeatureExtractor must be initialized and stored in env.feature_extractor.
    If rgb_only=False, the camera must output rgb, depth, and semantic_segmentation data types.
    If rgb_only=True, only rgb is required.
    
    Returns:
        torch.Tensor: CNN embedding of shape (num_envs, embedding_dim)
    """
    camera = env.scene[asset_cfg.name]
    
    # Get RGB from camera (always required)
    rgb = camera.data.output["rgb"][..., :3]  # (num_envs, h, w, 3)
    
    # Get depth and segmentation only if not in rgb_only mode
    if env.feature_extractor.cfg.rgb_only:
        depth = None
        segmentation = None
    else:
        depth = camera.data.output["depth"]  # (num_envs, h, w, 1)
        segmentation = camera.data.output["semantic_segmentation"][..., :3]  # (num_envs, h, w, 3)
    
    # Use feature extractor to get embedding
    # Note: gt_pose is not used for inference, pass zeros
    gt_pose = torch.zeros(env.num_envs, 27, device=env.device)
    _, embeddings = env.feature_extractor.step(rgb, depth, segmentation, gt_pose)
    
    return embeddings.clone().detach()


def front_camera_embedding(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    """Get CNN embedding from front camera using FeatureExtractor.
    
    The FeatureExtractor must be initialized and stored in env.feature_extractor.
    If rgb_only=False, the camera must output rgb, depth, and semantic_segmentation data types.
    If rgb_only=True, only rgb is required.
    
    Returns:
        torch.Tensor: CNN embedding of shape (num_envs, embedding_dim)
    """
    camera = env.scene[asset_cfg.name]
    
    # Get RGB from camera (always required)
    rgb = camera.data.output["rgb"][..., :3]  # (num_envs, h, w, 3)
    
    # Get depth and segmentation only if not in rgb_only mode
    if env.feature_extractor.cfg.rgb_only:
        depth = None
        segmentation = None
    else:
        depth = camera.data.output["depth"]  # (num_envs, h, w, 1)
        segmentation = camera.data.output["semantic_segmentation"][..., :3]  # (num_envs, h, w, 3)
    
    # Use feature extractor to get embedding
    # Note: gt_pose is not used for inference, pass zeros
    gt_pose = torch.zeros(env.num_envs, 27, device=env.device)
    _, embeddings = env.feature_extractor.step(rgb, depth, segmentation, gt_pose)
    
    return embeddings.clone().detach()


def wrist_camera_pretrained_embedding(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    camera = env.scene[asset_cfg.name]
    
    # Get RGB from camera
    rgb = camera.data.output["rgb"][..., :3]  # (num_envs, h, w, 3)
    
    # Use pretrained feature extractor to get embedding
    embeddings = env.feature_extractor.step(rgb)
    
    return embeddings


def front_camera_pretrained_embedding(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    camera = env.scene[asset_cfg.name]
    
    # Get RGB from camera
    rgb = camera.data.output["rgb"][..., :3]  # (num_envs, h, w, 3)
    
    # Use pretrained feature extractor to get embedding
    embeddings = env.feature_extractor.step(rgb)
    
    return embeddings

