# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_drawer_command(env: ManagerBasedRLEnv):
    """Get the drawer task command term."""
    return env.command_manager.get_term("drawer_task")


def _get_handle_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd = _get_drawer_command(env)
    batch_idx = torch.arange(env.num_envs, device=cmd.current_frame_idx.device)
    return env.scene["cabinet_frame"].data.target_pos_w[batch_idx, cmd.current_frame_idx, :]


def _get_handle_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd = _get_drawer_command(env)
    batch_idx = torch.arange(env.num_envs, device=cmd.current_frame_idx.device)
    return env.scene["cabinet_frame"].data.target_quat_w[batch_idx, cmd.current_frame_idx, :]


def _get_drawer_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd = _get_drawer_command(env)
    batch_idx = torch.arange(env.num_envs, device=cmd.current_joint_id.device)
    return env.scene[asset_cfg.name].data.joint_pos[batch_idx, cmd.current_joint_id]


def _get_drawer_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd = _get_drawer_command(env)
    batch_idx = torch.arange(env.num_envs, device=cmd.current_joint_id.device)
    return env.scene[asset_cfg.name].data.joint_vel[batch_idx, cmd.current_joint_id]


def approach_ee_handle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    r"""Reward the robot for reaching the drawer handle using inverse-square law.

    It uses a piecewise function to reward the robot for reaching the handle.

    .. math::

        reward = \begin{cases}
            2 * (1 / (1 + distance^2))^2 & \text{if } distance \leq threshold \\
            (1 / (1 + distance^2))^2 & \text{otherwise}
        \end{cases}

    """
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = _get_handle_pos(env, asset_cfg)

    # Compute the distance of the end-effector to the handle
    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
    

    # Reward the robot for reaching the handle
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    return torch.where(distance <= threshold, 2 * reward, reward)


def align_ee_handle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for aligning the end-effector with the handle.

    The reward is based on the alignment of the gripper with the handle. It is computed as follows:

    .. math::

        reward = 0.5 * (align_z^2 + align_x^2)

    where :math:`align_z` is the dot product of the z direction of the gripper and the -x direction of the handle
    and :math:`align_x` is the dot product of the x direction of the gripper and the -y direction of the handle.
    """
    ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    handle_quat = _get_handle_quat(env, asset_cfg)

    ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
    handle_mat = matrix_from_quat(handle_quat)

    # get current x and y direction of the handle
    handle_x, handle_y = handle_mat[..., 0], handle_mat[..., 1]
    # get current x and z direction of the gripper
    ee_tcp_x, ee_tcp_z = ee_tcp_rot_mat[..., 0], ee_tcp_rot_mat[..., 2]

    # make sure gripper aligns with the handle
    # in this case, the z direction of the gripper should be close to the -x direction of the handle
    # and the x direction of the gripper should be close to the -y direction of the handle
    # dot product of z and x should be large
    align_z = torch.bmm(ee_tcp_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_tcp_x.unsqueeze(1), handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


def align_grasp_around_handle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    handle_pos = _get_handle_pos(env, asset_cfg)
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # Soft alignment score: 0~1
    l_above = torch.clamp((lfinger_pos[:, 2] - handle_pos[:, 2]) / 0.04 + 0.5, 0.0, 1.0)  # 理想: +0.02 ~ +0.04 以上滿分
    r_below = torch.clamp((handle_pos[:, 2] - rfinger_pos[:, 2]) / 0.04 + 0.5, 0.0, 1.0)
    height_diff_bonus = torch.clamp((lfinger_pos[:, 2] - rfinger_pos[:, 2]) / 0.06, 0.0, 1.0)  # 高度差越大越好，但 cap at 1

    grasp_score = (l_above * r_below * height_diff_bonus) ** 0.5
    return grasp_score


def approach_gripper_handle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, offset: float = 0.04) -> torch.Tensor:
    """Reward the robot's gripper reaching the drawer handle with the right pose.

    This function returns the distance of fingertips to the handle when the fingers are in a grasping orientation
    (i.e., the left finger is above the handle and the right finger is below the handle). Otherwise, it returns zero.
    """
    # Target object position: (num_envs, 3)
    handle_pos = _get_handle_pos(env, asset_cfg)
    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # Compute the distance of each finger from the handle
    l_score = torch.clamp((offset - torch.abs(lfinger_pos[:, 2] - handle_pos[:, 2])) / offset, 0.0, 1.0)
    r_score = torch.clamp((offset - torch.abs(rfinger_pos[:, 2] - handle_pos[:, 2])) / offset, 0.0, 1.0)

    # Check if hand is in a graspable pose
    is_graspable = align_grasp_around_handle(env, asset_cfg)

    return is_graspable * (l_score + r_score)


def grasp_handle(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float, open_joint_pos: float
) -> torch.Tensor:
    """Reward for closing the fingers when being close to the handle."""
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = _get_handle_pos(env, asset_cfg)
    gripper_joint_pos = env.scene[robot_cfg.name].data.joint_pos[:, robot_cfg.joint_ids]
    
    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
    is_close = (distance <= threshold).float()
    
    # 夾爪關閉程度：current_pos / open_joint_pos
    # open (pos=1.74): close_ratio = 1 - 1.74/1.74 = 0
    # closed (pos=0): close_ratio = 1 - 0/1.74 = 1
    current_pos = gripper_joint_pos.sum(dim=-1).clamp(min=0.0)  # clamp 避免負值
    close_ratio = torch.clamp(1.0 - current_pos / open_joint_pos, 0.0, 1.0)
    
    is_graspable = align_grasp_around_handle(env, asset_cfg)
    
    # 獎勵 = 靠近 * 關閉程度 * 抓握姿勢對齊
    return is_close * close_ratio * is_graspable

def open_drawer_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Bonus for opening the drawer given by the joint position of the drawer."""
    cmd = _get_drawer_command(env)
    drawer_pos = _get_drawer_pos(env, asset_cfg)
    is_graspable = align_grasp_around_handle(env, asset_cfg).float()
    result = (is_graspable * 1.5 + 1.0) * drawer_pos
    
    # DEBUG: print env 0 every 100 steps
    if env.common_step_counter % 100 == 0:
        print(f"[open_drawer_bonus] env0: drawer_pos={drawer_pos[0].item():.4f}, "
              f"is_graspable={is_graspable[0].item():.4f}, result={result[0].item():.4f}, "
              f"joint_id={cmd.current_joint_id[0].item()}")
    
    return result


def drawer_completion_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """稀疏獎勵：完成一個抽屜時給大獎勵"""
    cmd = _get_drawer_command(env)
    return cmd.just_completed.float() * 100.0


def multi_stage_open_drawer(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Multi-stage bonus for opening the current target drawer."""
    cmd = _get_drawer_command(env)
    drawer_pos = _get_drawer_pos(env, asset_cfg)
    is_graspable = align_grasp_around_handle(env, asset_cfg).float()

    # Stage rewards (max drawer pos ~0.25)
    open_medium = (drawer_pos > 0.15).float()
    open_hard = (drawer_pos > 0.22).float()
    result = (open_medium * 0.3 + open_hard * 0.7) * is_graspable
    
    # DEBUG: print env 0 every 100 steps
    if env.common_step_counter % 100 == 0:
        print(f"[multi_stage] env0: drawer_pos={drawer_pos[0].item():.4f}, "
              f"is_graspable={is_graspable[0].item():.4f}, result={result[0].item():.4f}, "
              f"target={cmd.current_target[0].item()}, completed={cmd.completed_count[0].item()}")

    return result

def punish_drawers_not_open(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """懲罰當前目標抽屜沒開，完成後（>=0.25）不懲罰"""
    drawer_pos = _get_drawer_pos(env, asset_cfg)
    done_threshold = 0.25
    # 線性衰減：pos=0 時懲罰=1，pos>=done_threshold 時懲罰=0
    return torch.clamp(1.0 - drawer_pos / done_threshold, min=0.0, max=1.0)

def punish_open_without_grasp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """懲罰用勾的方式開抽屜：抽屜 > 0.2 但沒有正確抓握時給懲罰"""
    drawer_pos = _get_drawer_pos(env, asset_cfg)
    is_graspable = align_grasp_around_handle(env, asset_cfg).float()
    
    # 只在抽屜開超過 0.2 時懲罰
    drawer_opened = (drawer_pos > 0.2).float()
    not_grasping = 1.0 - is_graspable
    
    return drawer_opened * not_grasping


def reward_drawer_movement(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """獎勵抽屜移動：當靠近把手且抽屜在動時給獎勵，鼓勵拉開動作"""
    handle_pos = _get_handle_pos(env, asset_cfg)
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    drawer_vel = _get_drawer_vel(env, asset_cfg)
    is_graspable = align_grasp_around_handle(env, asset_cfg).float()
    
    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1)
    is_close = (distance < 0.02).float()
    
    # 正方向速度越大越好（拉開）
    positive_vel = torch.clamp(drawer_vel, min=0.0)
    
    return is_close * is_graspable * positive_vel
    
def punish_idle_near_handle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    handle_pos = _get_handle_pos(env, asset_cfg)
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    drawer_vel = _get_drawer_vel(env, asset_cfg).abs()
    drawer_pos = _get_drawer_pos(env, asset_cfg)
    
    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1)
    is_near = distance < 0.04
    is_idle = drawer_vel < 0.01
    
    return is_near & is_idle


def punish_ee_yz_deviation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """懲罰 ee 在 y, z 軸的大幅度偏移，與抽屜速度成正比（拉動時才懲罰）"""
    handle_pos = _get_handle_pos(env, asset_cfg)
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    drawer_vel = _get_drawer_vel(env, asset_cfg).abs()
    
    # 計算 y, z 軸的偏離量
    diff = ee_tcp_pos - handle_pos
    yz_deviation = torch.abs(diff[:, 1]) + torch.abs(diff[:, 2])
    result = yz_deviation * (0.2 + 10.0 * drawer_vel)
    
    # DEBUG: print env 0 every 100 steps
    if env.common_step_counter % 100 == 0:
        print(f"[ee_yz_dev] env0: drawer_vel={drawer_vel[0].item():.4f}, "
              f"yz_dev={yz_deviation[0].item():.4f}, result={result[0].item():.6f}")

    return result