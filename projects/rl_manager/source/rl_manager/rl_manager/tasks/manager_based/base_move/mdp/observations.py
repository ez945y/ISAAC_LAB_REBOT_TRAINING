# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for base_move task.

IMPORTANT: Only proprioceptive observations that can be measured on real robot.
NO ground truth (target positions, phases) in observations for sim-to-real.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_position_in_robot_frame(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector position relative to robot base frame.
    
    This is measurable via forward kinematics on real robot.
    
    Args:
        env: The environment instance.
        ee_frame_cfg: Configuration for the EE frame sensor.
    
    Returns:
        EE position tensor of shape (num_envs, 3).
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    
    # Get EE world position
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]  # (num_envs, 3)
    
    # Get robot base position
    robot = env.scene["robot"]
    base_pos_w = robot.data.root_pos_w  # (num_envs, 3)
    
    # EE position relative to base
    ee_pos_rel = ee_pos_w - base_pos_w
    
    return ee_pos_rel
