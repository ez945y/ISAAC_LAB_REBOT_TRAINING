# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for SO-ARM-101 single-finger gripper robot."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Gripper joint limits from URDF (Jaw joint)
# lower: -0.174533 (open), upper: 1.74533 (closed)
GRIPPER_LOWER = -0.174533
GRIPPER_UPPER = 1.74533
GRIPPER_RANGE = GRIPPER_UPPER - GRIPPER_LOWER


def gripper_pos(env) -> torch.Tensor:
    """Gripper joint position for single-finger gripper (SO-ARM-101).
    
    Returns the raw gripper joint position in radians.
    """
    robot = env.scene["robot"]
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    raw_pos = robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
    return raw_pos


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.02,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos + torch.tensor([0.0, 0.0, 0.1], device=object_pos.device), dim=1)
    # if object_cfg.name == "cube_2":
    #     print(object_pos - end_effector_pos + torch.tensor([0.0, 0.0, 0.1], device=object_pos.device))
    
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        grasped = torch.logical_and(
            pose_diff < diff_threshold,
            torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[0]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            )
            > env.cfg.gripper_threshold,
        )
    else:
        raise ValueError("No gripper_joint_names found in environment config")
        

    return grasped


def object_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
) -> torch.Tensor:
    """Check if an object is stacked by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    upper_object: RigidObject = env.scene[upper_object_cfg.name]
    lower_object: RigidObject = env.scene[lower_object_cfg.name]

    pos_diff = upper_object.data.root_pos_w - lower_object.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    stacked = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        stacked = torch.logical_and(
            torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[0]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            )
            < env.cfg.open_threshold,
            stacked,
        )
    else:
        raise ValueError("No gripper_joint_names found in environment config")

    return stacked


def wrist_camera_rgb(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """取得 wrist camera 的 RGB 影像，展平為 vector"""
    camera = env.scene[asset_cfg.name]
    rgb = camera.data.output["rgb"]           # shape: (num_envs, h, w, 4) or (num_envs, h, w, 3)
    rgb = rgb[..., :3]                        # 拿掉 alpha channel（如果有）
    return rgb.view(env.num_envs, -1)         # → (num_envs, h*w*3)


def front_camera_rgb(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """取得 front camera 的 RGB 影像，展平為 vector"""
    camera = env.scene[asset_cfg.name]
    rgb = camera.data.output["rgb"]
    rgb = rgb[..., :3]
    return rgb.view(env.num_envs, -1)