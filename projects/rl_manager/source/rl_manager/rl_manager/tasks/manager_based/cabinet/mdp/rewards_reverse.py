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

def _get_process(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    drawer_pos_left = joint_pos[:, 0]
    drawer_pos_right    = joint_pos[:, 1]
    drawer_pos_bottom   = joint_pos[:, 2]
    
    stage_one   = (drawer_pos_left < -1.5)
    stage_two   = (drawer_pos_right > 1.5) & stage_one
    stage_three = (drawer_pos_bottom > 0.2) & stage_two
    
    process = torch.zeros_like(drawer_pos_left, dtype=torch.long)
    process = torch.where(stage_three, 3, process)
    process = torch.where(stage_two & ~stage_three, 2, process)
    process = torch.where(stage_one & ~stage_two, 1, process)
    
    return process


def _get_handle_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    process_idx = _get_process(env, asset_cfg)
    batch_idx = torch.arange(
        env.scene["cabinet_frame"].data.target_pos_w.shape[0],
        device=process_idx.device
    )
    return env.scene["cabinet_frame"].data.target_pos_w[batch_idx, process_idx, :]


def _get_handle_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    process_idx = _get_process(env, asset_cfg)
    batch_idx = torch.arange(
        env.scene["cabinet_frame"].data.target_quat_w.shape[0],
        device=process_idx.device
    )
    return env.scene["cabinet_frame"].data.target_quat_w[batch_idx, process_idx, :]


def _get_drawer_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    process_idx = _get_process(env, asset_cfg)
    batch_idx = torch.arange(
        env.scene[asset_cfg.name].data.joint_pos.shape[0],
        device=process_idx.device
    )
    return env.scene[asset_cfg.name].data.joint_pos[batch_idx, process_idx]


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
    align_x = torch.bmm(ee_tcp_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


def align_grasp_around_handle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Bonus for correct hand orientation around the handle.

    The correct hand orientation is when the left finger is above the handle and the right finger is below the handle.
    """
    # Target object position: (num_envs, 3)
    handle_pos = _get_handle_pos(env, asset_cfg)
    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # Check if hand is in a graspable pose
    is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])

    # bonus if left finger is above the drawer handle and right below
    return is_graspable


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
    lfinger_dist = torch.abs(lfinger_pos[:, 2] - handle_pos[:, 2])
    rfinger_dist = torch.abs(rfinger_pos[:, 2] - handle_pos[:, 2])

    # Check if hand is in a graspable pose
    is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])

    return is_graspable * ((offset - lfinger_dist) + (offset - rfinger_dist))


def grasp_handle(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float, open_joint_pos: float
) -> torch.Tensor:
    """Reward for closing the fingers when being close to the handle.

    The :attr:`threshold` is the distance from the handle at which the fingers should be closed.
    The :attr:`open_joint_pos` is the joint position when the fingers are open.

    Note:
        It is assumed that zero joint position corresponds to the fingers being closed.
    """
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = _get_handle_pos(env, asset_cfg)
    gripper_joint_pos = env.scene[robot_cfg.name].data.joint_pos[:, robot_cfg.joint_ids]

    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
    closeness = 1.0 / (1.0 + distance ** 2)
    closeness = torch.pow(closeness, 2)
    closeness = torch.where(distance > threshold, torch.zeros_like(closeness), closeness)

    closing_amount = torch.sum(open_joint_pos - gripper_joint_pos, dim=-1).clamp(0.0, open_joint_pos)
    is_graspable = align_grasp_around_handle(env, asset_cfg).float()
    return closeness * closing_amount * is_graspable


def open_drawer_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Bonus for opening the drawer given by the joint position of the drawer.

    The bonus is given when the drawer is open. If the grasp is around the handle, the bonus is doubled.
    """
    drawer_pos = _get_drawer_pos(env, asset_cfg)
    is_graspable = align_grasp_around_handle(env, asset_cfg).float()

    return (is_graspable + 1.0) * drawer_pos


def multi_stage_open_drawer(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Multi-stage bonus for opening the drawer.

    Depending on the drawer's position, the reward is given in three stages: easy, medium, and hard.
    This helps the agent to learn to open the drawer in a controlled manner.
    """
    joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    drawer_pos_left = joint_pos[:, 0]
    drawer_pos_right = joint_pos[:, 1]
    drawer_pos_bottom = joint_pos[:, 2]
    drawer_pos_top = joint_pos[:, 3]

    
    is_graspable = align_grasp_around_handle(env, asset_cfg).float()
    
    open_easy_left = (drawer_pos_left < -0.01) * 0.5
    open_medium_left = (drawer_pos_left < -0.2) * is_graspable
    open_hard_left= (drawer_pos_left < -0.5) * is_graspable
    open_easy_right = (drawer_pos_right > 0.01) * 0.5 * open_hard_left
    open_medium_right = (drawer_pos_right > 0.2) * is_graspable * open_hard_left
    open_hard_right= (drawer_pos_right > 0.5) * is_graspable * open_hard_left
    open_both = (drawer_pos_left < -1.0) * (drawer_pos_right > 1.0) * is_graspable * 0.5

    second_open_easy = (drawer_pos_bottom > 0.01) * open_both * 0.5
    second_open_medium = (drawer_pos_bottom > 0.2) * open_both * is_graspable
    second_open_hard = (drawer_pos_bottom > 0.3) * open_both * is_graspable


    return open_easy_left + open_medium_left + open_hard_left + open_easy_right + open_medium_right + open_hard_right + open_both + second_open_easy + second_open_medium + second_open_hard


# def penalize_early_release(
#     env: ManagerBasedRLEnv,
#     asset_cfg_robot: SceneEntityCfg,
#     asset_cfg_cabinet: SceneEntityCfg,
#     open_joint_threshold: float = 0.4,
#     threshold: float = 0.3,
# ) -> torch.Tensor:
#     drawer_pos = env.scene[asset_cfg_cabinet.name].data.joint_pos[:, asset_cfg_cabinet.joint_ids[0]]
#     gripper_pos = env.scene[asset_cfg_robot.name].data.joint_pos[:, asset_cfg_robot.joint_ids[0]]

#     is_released = gripper_pos > open_joint_threshold
#     is_grasp_aligned = align_grasp_around_handle(env, asset_cfg_cabinet).float()

#     not_opened = drawer_pos < threshold
#     penalize_release = not_opened * (is_released + (1.0 - is_grasp_aligned))

#     return penalize_release


# def penalize_ee_x_exceed_handle(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
# ) -> torch.Tensor:
#     drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
#     ee_tcp_x = env.scene["ee_frame"].data.target_pos_w[:, 0, 0]     # ee_tcp 的 x
#     handle_x  = env.scene["cabinet_frame"].data.target_pos_w[:, 0, 0]  # handle 的 x
#     exceed_amount = ee_tcp_x - handle_x

#     penalty = torch.clamp(exceed_amount, min=0.0)
#     return penalty ** 100