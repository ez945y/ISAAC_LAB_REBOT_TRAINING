# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation functions for SO-ARM-101 single-cube (move) task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Single-cube world-frame observations ───────────────────────


def cube_position_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """Position of a single cube in the world frame. Shape: [N, 3]."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_pos_w


def cube_orientation_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """Quaternion orientation of a single cube in the world frame. Shape: [N, 4]."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_quat_w


def object_obs(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Single-cube object observations (world frame).

    Returns [N, 10]:
        cube pos (3), cube quat (4), gripper-to-cube (3)
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = cube.data.root_pos_w
    cube_quat_w = cube.data.root_quat_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    gripper_to_cube = cube_pos_w - ee_pos_w

    return torch.cat(
        (
            cube_pos_w - env.scene.env_origins,
            cube_quat_w,
            gripper_to_cube,
        ),
        dim=1,
    )


# ── EE frame observations ─────────────────────────────────────


def ee_frame_pos(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector position in world frame (env-origin relative). Shape: [N, 3]."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]


def ee_frame_quat(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector quaternion in world frame. Shape: [N, 4]."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[:, 0, :]


# ── Gripper observations ──────────────────────────────────────


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gripper joint position for single-finger gripper (SO-ARM-101).

    Returns the raw gripper joint position in radians. Shape: [N, 1].
    """
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    return robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)


# ── Grasp detection ────────────────────────────────────────────


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Check if a single object is grasped by the gripper.

    Grasped = EE close to object AND gripper is NOT in open position.
    Supports single-finger gripper (SO-ARM-101).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    obj_pos = obj.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)

    # Check proximity
    is_close = pose_diff < diff_threshold

    # Check gripper is closed (not in open position)
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        gripper_not_open = (
            torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[0]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            )
            > env.cfg.gripper_threshold
        )
        grasped = torch.logical_and(is_close, gripper_not_open)
    else:
        raise ValueError("No gripper_joint_names in env config")

    return grasped
