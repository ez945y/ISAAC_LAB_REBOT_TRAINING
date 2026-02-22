# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Termination functions for SO-ARM-101 single-cube (move) task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_on_platform(
    env: ManagerBasedRLEnv,
    cube_frame_name: str = "cube_1_frame",
    platform_frame_name: str = "platform_frame",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_tolerance: float = 0.12,
    min_z_above: float = 0.015,
    min_gripper_open: float = 0.7,
    max_z_above: float = 0.10,
):
    """Check if cube has been placed on the platform.

    Reads the ``target_pos_w`` from the FrameTransformer sensors.
    Platform center Z is at the **bottom** of the platform, so cube Z must
    be above it by ``min_z_above``.
    """
    # 1. Get FrameTransformer data
    # Note: Using the first target frame (index 0) defined in each transformer
    cube_tf: FrameTransformer = env.scene[cube_frame_name]
    plat_tf: FrameTransformer = env.scene[platform_frame_name]

    # target_pos_w is [N, num_targets, 3]
    cube_center = cube_tf.data.target_pos_w[:, 0, :]
    plat_center = plat_tf.data.target_pos_w[:, 0, :]

    # 2. Subtract env origins to get relative position (for debugging/sanity)
    # Actually get_world_poses/target_pos_w are already World including env_origin.
    # Isaac Lab terminations usually work in world coords.

    # 3. XY within tolerance of platform center
    dx = torch.abs(cube_center[:, 0] - plat_center[:, 0])
    dy = torch.abs(cube_center[:, 1] - plat_center[:, 1])
    xy_ok = (dx < xy_tolerance) & (dy < xy_tolerance)

    # 4. Z above platform surface
    z_ok = (cube_center[:, 2] > (plat_center[:, 2] + min_z_above)) & (cube_center[:, 2] < (plat_center[:, 2] + max_z_above))

    # Debug: print positions and check results
    # Subtracting env_origins for easier reading in debug
    origin = env.scene.env_origins[0]
    cube_rel = (cube_center[0] - origin).tolist()
    plat_rel = (plat_center[0] - origin).tolist()
    
    # 5. Gripper must be open (released the cube)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_joint_ids, _ = robot.find_joints(["gripper"])
    gripper_pos = robot.data.joint_pos[:, gripper_joint_ids[0]]
    gripper_open = gripper_pos >= min_gripper_open

    result = xy_ok & z_ok & gripper_open
    _Y = "\033[93m"  # bright yellow
    _G = "\033[92m"  # bright green
    _R = "\033[91m"  # bright red
    _E = "\033[0m"   # reset
    tag = f"{_G}SUCCESS{_E}" if result[0].item() else f"{_R}FAIL{_E}"
    # print(robot.data.joint_pos)
    # print(f"{_Y}[TERM]{_E} cube={[f'{v:.3f}' for v in cube_rel]}  plat={[f'{v:.3f}' for v in plat_rel]}  dx={dx[0].item():.4f} dy={dy[0].item():.4f} z_ok={z_ok[0].item()} grip={gripper_pos[0].item():.2f}  [{tag}]")

    return result



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
