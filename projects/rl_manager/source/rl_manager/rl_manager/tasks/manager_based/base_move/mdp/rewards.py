# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for base_move task.

All rewards use internal command state - observations remain sim-to-real compatible.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def approach_phase_target(
    env: "ManagerBasedRLEnv",
    command_name: str = "movement_phase",
    std: float = 0.02,  # 2cm std for small movements
) -> torch.Tensor:
    """Exponential reward for approaching current phase target.
    
    Uses command's internal target position (not exposed to observations).
    
    Args:
        env: The environment instance.
        command_name: Name of the movement phase command.
        std: Standard deviation for exponential reward shaping.
    
    Returns:
        Reward tensor of shape (num_envs,).
    """
    command = env.command_manager.get_term(command_name)
    
    # Get current EE position
    ee_frame = env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]  # (num_envs, 3)
    
    # Get target position from command
    target_pos = command.get_current_target()
    
    # Calculate distance
    distance = torch.norm(ee_pos - target_pos, dim=-1)
    
    # Exponential reward: higher when closer to target
    reward = torch.exp(-distance / std)
    
    return reward


def velocity_towards_target(
    env: "ManagerBasedRLEnv",
    command_name: str = "movement_phase",
) -> torch.Tensor:
    """Reward for moving towards target (encourages active movement).
    
    This prevents the robot from staying still by rewarding velocity
    in the direction of the target.
    
    Args:
        env: The environment instance.
        command_name: Name of the movement phase command.
    
    Returns:
        Reward tensor of shape (num_envs,).
    """
    command = env.command_manager.get_term(command_name)
    
    # Get current EE position and velocity
    ee_frame = env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]  # (num_envs, 3)
    
    # Get robot base velocity (as proxy for EE velocity)
    robot = env.scene["robot"]
    # Use root linear velocity as approximate EE velocity direction
    ee_vel = robot.data.root_lin_vel_w  # (num_envs, 3)
    
    # Get target position from command
    target_pos = command.get_current_target()
    
    # Direction to target (normalized)
    direction_to_target = target_pos - ee_pos
    distance = torch.norm(direction_to_target, dim=-1, keepdim=True)
    direction_to_target = direction_to_target / (distance + 1e-6)
    
    # Dot product of velocity with target direction
    # Positive when moving towards target
    velocity_towards = torch.sum(ee_vel * direction_to_target, dim=-1)
    
    # Only reward positive movement (moving towards target)
    reward = torch.clamp(velocity_towards, min=0.0)
    
    return reward


def phase_completed_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str = "movement_phase",
) -> torch.Tensor:
    """Sparse bonus reward when a phase is completed.
    
    Args:
        env: The environment instance.
        command_name: Name of the movement phase command.
    
    Returns:
        Reward tensor of shape (num_envs,) with 1.0 for completed phases.
    """
    command = env.command_manager.get_term(command_name)
    return command.phase_just_completed.float()


def staying_alive(
    env: "ManagerBasedRLEnv",
) -> torch.Tensor:
    """Small constant reward for staying alive (encourages longer episodes).
    
    Args:
        env: The environment instance.
    
    Returns:
        Reward tensor of shape (num_envs,).
    """
    return torch.ones(env.num_envs, device=env.device)


def smooth_motion(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for high joint velocities to encourage smooth motion.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
    
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    robot = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel
    penalty = torch.sum(joint_vel ** 2, dim=-1)
    return penalty


def height_constraint_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "movement_phase",
) -> torch.Tensor:
    """Penalty for EE height deviation from target.
    
    Keeps EE at constant height as specified by the command's target.
    
    Args:
        env: The environment instance.
        command_name: Name of the movement phase command.
    
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    # Get current EE position in world frame
    ee_frame = env.scene["ee_frame"]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]  # (num_envs, 3)
    
    # Get target position from command
    command = env.command_manager.get_term(command_name)
    target_pos_w = command.get_current_target()
    
    # Deviation from target height (Z direction)
    height_error = (ee_pos_w[:, 2] - target_pos_w[:, 2]) ** 2
    
    return height_error


def velocity_limit_penalty(
    env: "ManagerBasedRLEnv",
    max_velocity: float = 0.02,  # 2cm/s
) -> torch.Tensor:
    """Penalty for moving faster than a specified limit.
    
    Args:
        env: The environment instance.
        max_velocity: Maximum allowed velocity in m/s (default 2cm/s).
    
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    robot = env.scene["robot"]
    # Use root linear velocity as proxy for EE velocity
    # (absolute velocity in world frame)
    ee_vel = torch.norm(robot.data.root_lin_vel_w, dim=-1)
    
    # Penalize only if velocity exceeds limit
    excess_vel = torch.clamp(ee_vel - max_velocity, min=0.0)
    
    return excess_vel ** 2
