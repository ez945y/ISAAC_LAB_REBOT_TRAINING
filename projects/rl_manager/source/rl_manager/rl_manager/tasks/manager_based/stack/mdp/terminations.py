# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common termination functions for SO-ARM-101 single-finger gripper robot."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cubes_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.04,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    atol=0.0001,
    rtol=0.0001,
):
    """Check if all cubes are stacked and gripper is open.
    
    Supports single-finger gripper (SO-ARM-101).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
    pos_diff_c23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w

    # Compute cube position difference in x-y plane
    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
    xy_dist_c23 = torch.norm(pos_diff_c23[:, :2], dim=1)

    # Compute cube height difference
    h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)
    h_dist_c23 = torch.norm(pos_diff_c23[:, 2:], dim=1)

    # Check cube positions
    stacked = torch.logical_and(xy_dist_c12 < xy_threshold, xy_dist_c23 < xy_threshold)
    stacked = torch.logical_and(h_dist_c12 - height_diff < height_threshold, stacked)
    stacked = torch.logical_and(pos_diff_c12[:, 2] < 0.0, stacked)
    stacked = torch.logical_and(h_dist_c23 - height_diff < height_threshold, stacked)
    stacked = torch.logical_and(pos_diff_c23[:, 2] < 0.0, stacked)

    # Check gripper positions
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        # Check first gripper joint is open
        threshold = env.cfg.gripper_open_val - env.cfg.gripper_threshold 
        is_gripper_open = robot.data.joint_pos[:, gripper_joint_ids[0]] <= threshold
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
