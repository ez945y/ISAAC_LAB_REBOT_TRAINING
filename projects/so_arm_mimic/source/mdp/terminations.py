# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Termination functions for SO-ARM-101 single-cube (move) task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_moved(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    target_y_min: float = 0.05,
    upright_threshold: float = 0.90,
):
    """Check if cube has been moved to the left side and is upright with gripper open.

    Success conditions:
    1. Cube Y position > target_y_min (moved to left side)
    2. Cube is upright (z-axis of cube aligned with world z-axis)
    3. Gripper is open

    Supports single-finger gripper (SO-ARM-101).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]

    cube_pos = cube.data.root_pos_w     # [N, 3]
    cube_quat = cube.data.root_quat_w   # [N, 4] w,x,y,z

    # 1. Cube is on the left side (positive Y)
    on_left = cube_pos[:, 1] > target_y_min

    # 2. Cube is upright: check that the z-axis of cube frame
    #    is aligned with world z-axis. For quaternion (w,x,y,z):
    #    z_body_in_world = [2(xz+wy), 2(yz-wx), 1-2(x²+y²)]
    #    We check if z_body_in_world · [0,0,1] > threshold
    w, x, y, z = cube_quat[:, 0], cube_quat[:, 1], cube_quat[:, 2], cube_quat[:, 3]
    z_component = 1.0 - 2.0 * (x * x + y * y)  # dot product with world z
    is_upright = z_component > upright_threshold

    # 3. Gripper is open
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        is_gripper_open = (
            torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[0]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            )
            < env.cfg.open_threshold
        )
    else:
        raise ValueError("No gripper_joint_names found in environment config")

    success = torch.logical_and(on_left, is_upright)
    success = torch.logical_and(success, is_gripper_open)

    return success
