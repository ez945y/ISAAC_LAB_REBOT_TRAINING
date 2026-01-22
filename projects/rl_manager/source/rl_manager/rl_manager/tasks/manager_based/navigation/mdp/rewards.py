# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, wrap_to_pi
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def alive_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """獎勵機器人保持存活。"""
    return (~env.termination_manager.terminated).float()

def goal_distance(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    計算機器人距離當前目標點的純量距離。
    
    返回:
        一個形狀為 (num_envs, 1) 的張量。
    """
    robot_pos_xy = env.scene["robot"].data.root_pos_w[:, :2]
    command = env.command_manager.get_command(command_name)

    target_pos_xy = command[:, :2]
    
    distance = torch.norm(target_pos_xy - robot_pos_xy, dim=1)
    
    return distance.unsqueeze(1)


def goal_heading_error_sin_cos(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    計算機器人朝向與目標朝向的誤差，並回傳其 cos 和 sin 值。
    
    返回:
        一個形狀為 (num_envs, 2) 的張量，包含 [朝向誤差cos, 朝向誤差sin]。
    """
    robot_pos_xy = env.scene["robot"].data.root_pos_w[:, :2]
    robot_quat_w = env.scene["robot"].data.root_quat_w
    command = env.command_manager.get_command(command_name)

    target_pos_xy = command[:, :2]

    # 計算機器人當前的 yaw 朝向
    robot_yaw_w = torch.atan2(2.0 * (robot_quat_w[:, 3] * robot_quat_w[:, 0] + robot_quat_w[:, 1] * robot_quat_w[:, 2]), 
                              1.0 - 2.0 * (robot_quat_w[:, 2] * robot_quat_w[:, 2] + robot_quat_w[:, 3] * robot_quat_w[:, 3]))
    
    # 計算指向目標點的理想 yaw 朝向
    target_yaw_w = torch.atan2(target_pos_xy[:, 1] - robot_pos_xy[:, 1],
                              target_pos_xy[:, 0] - robot_pos_xy[:, 0])
                              
    # 計算誤差並正規化到 [-pi, pi]
    heading_error = wrap_to_pi(target_yaw_w - robot_yaw_w)
    
    # 回傳 cos 和 sin 值
    return torch.stack([torch.cos(heading_error), torch.sin(heading_error)], dim=-1)

def feet_air_time(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]  # type: ignore[index]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    return reward

def track_desired_velocity_exp(
    env: ManagerBasedRLEnv, command_name: str, desired_speed: float, std: float
) -> torch.Tensor:
    """獎勵機器人追蹤一個動態計算的、朝向當前目標點的速度。"""
    asset: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    target_pos_xy = command[:, :2]
    current_pos_xy = asset.data.root_pos_w[:, :2]
    current_vel_xy = asset.data.root_lin_vel_w[:, :2]

    direction_vec = target_pos_xy - current_pos_xy
    direction_unit_vec = direction_vec / (torch.norm(direction_vec, dim=1, keepdim=True) + 1e-6)
    desired_velocity_xy = direction_unit_vec * desired_speed
    
    velocity_error_sq = torch.sum(torch.square(desired_velocity_xy - current_vel_xy), dim=1)
    reward = torch.exp(-velocity_error_sq / (std**2))
    
    return reward

# --- 輔助函式 ---
def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw

def position_progress_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    獎勵機器人靠近目標點的進展 (距離的減少量)。
    
    對應您要求的： position_progress_rew = self._previous_position_error - self._position_error
    """
    asset: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)
    
    robot_pos_xy = asset.data.root_pos_w[:, :2]
    target_pos_xy = command[:, :2]

    # 計算當前到目標的距離
    current_dist = torch.norm(target_pos_xy - robot_pos_xy, dim=1)
    
    # 從 command term 獲取上一步的距離
    get_dist_fn = getattr(command_term, "get_last_dist_to_goal", None)
    if callable(get_dist_fn):
        last_dist = get_dist_fn()
        # 確保 last_dist 是 torch.Tensor
        if not isinstance(last_dist, torch.Tensor):
            last_dist = current_dist.clone()
    else:
        # Fallback: use current distance if method not available
        last_dist = current_dist.clone()

    # 計算進展：如果距離變小，這是一個正數（獎勵）
    progress = last_dist - current_dist
    
    update_dist_fn = getattr(command_term, "update_last_dist_to_goal", None)
    if callable(update_dist_fn):
        update_dist_fn(current_dist)
    
    return progress


def heading_alignment_reward(env: ManagerBasedRLEnv, command_name: str, heading_coefficient: float) -> torch.Tensor:
    """
    獎勵機器人的朝向與目標方向對齊。
    
    對應您要求的： target_heading_rew = torch.exp(-torch.abs(self.target_heading_error) / self.heading_coefficient)
    """
    asset: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)

    target_pos_xy = command[:, :2]
    current_pos_xy = asset.data.root_pos_w[:, :2]
    current_yaw = quat_to_yaw(asset.data.root_quat_w) # 假設 quat_to_yaw 已定義

    # 計算從機器人指向目標的向量
    delta_pos = target_pos_xy - current_pos_xy
    # 計算理想的朝向
    desired_heading = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])
    
    # 計算朝向誤差
    heading_error = wrap_to_pi(desired_heading - current_yaw)
    
    # 這個獎勵在 (0, 1] 之間，對得越準，獎勵越接近 1
    reward = torch.exp(-torch.abs(heading_error) / heading_coefficient)
    
    return reward


def waypoint_reached_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """當到達一個航點時給予稀疏獎勵，並在到達時切換到下一個航點並更新指令。"""
    # 當前機器人位置
    robot_pos_xy = env.scene["robot"].data.root_pos_w[:, :2]
    # 取得當前指令(其中前2維是當前目標XY)
    command_term = env.command_manager.get_term(command_name)
    command = command_term.command
    target_pos_xy = command[:, :2]
    # 計算是否到達 (距離小於容差)
    tolerance = float(getattr(getattr(command_term, "cfg", object()), "tolerance", 0.3))
    dist = torch.norm(target_pos_xy - robot_pos_xy, dim=1)
    reached_mask = dist < tolerance

    if torch.any(reached_mask):
        reached_ids = torch.nonzero(reached_mask, as_tuple=False).squeeze(-1)
        advance_fn = getattr(command_term, "advance_waypoints", None)
        if callable(advance_fn):
            advance_fn(env_ids=reached_ids)

    return reached_mask.float()

def reset_waypoint_navigation(env: ManagerBasedRLEnv, env_ids: torch.Tensor, command_name: str):
    """為指定的環境重置航點導航邏輯，使用 GoalPoseCommandCfg 內的航點設定。"""
    command_term = env.command_manager.get_term(command_name)
    reset_fn = getattr(command_term, "reset_waypoints", None)
    if callable(reset_fn):
        reset_fn(env_ids)


def height_l2_penalty(env: ManagerBasedRLEnv, target_height: float) -> torch.Tensor:
    """
    懲罰機器人偏離期望的高度。
    
    這個函式使用二次方 (L2) 來計算懲罰。
    當機器人處於目標高度時，懲罰為 0；誤差越大，懲罰的負值越大。
    """
    asset: Articulation = env.scene["robot"]
    current_height = asset.data.root_pos_w[:, 2]
    
    height_error_sq = torch.square(current_height - target_height)
    
    return height_error_sq