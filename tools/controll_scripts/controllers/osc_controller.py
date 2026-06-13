# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Operational Space 控制器"""

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg

from .base import BaseController
from ..configs.base import BaseRobotConfig
from ..utils import physx_to_torch


class OSCController(BaseController):
    """Operational Space 控制器包裝
    
    使用 Isaac Lab 的 OperationalSpaceController，
    輸出關節力矩（effort）。
    """
    
    def __init__(
        self,
        robot: Articulation,
        robot_config: BaseRobotConfig,
        device: str,
        num_envs: int = 1,
    ):
        super().__init__(robot, robot_config, device, num_envs)
        
        # 創建 OSC 控制器
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            motion_control_axes_task=(1, 1, 1, 1, 1, 1),
            motion_stiffness_task=robot_config.osc_motion_stiffness,
            motion_damping_ratio_task=robot_config.osc_motion_damping_ratio,
            inertial_dynamics_decoupling=True,
            gravity_compensation=True,
        )
        self._osc_controller = OperationalSpaceController(
            osc_cfg, num_envs=num_envs, device=device
        )
        
        # 初始化
        self._osc_controller.reset()
    
    def compute(self, target_pose: torch.Tensor, gripper_pos: float) -> None:
        """計算並應用 OSC 控制
        
        Args:
            target_pose: 絕對目標姿態 [pos(3) + quat(4)] 在 base frame
            gripper_pos: 夾爪位置 [0, 1]
        """
        # 設定 OSC 指令
        self._osc_controller.set_command(target_pose)
        
        # 獲取當前狀態
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self._robot.data.body_quat_w[:, self._ee_body_idx]
        ee_vel_w = self._robot.data.body_vel_w[:, self._ee_body_idx, :]
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        root_vel_w = self._robot.data.root_vel_w
        
        # 轉換到基座座標系
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        
        # 計算相對速度並轉換到基座座標系
        relative_vel_w = ee_vel_w - root_vel_w
        ee_lin_vel_b = math_utils.quat_apply_inverse(root_quat_w, relative_vel_w[:, 0:3])
        ee_ang_vel_b = math_utils.quat_apply_inverse(root_quat_w, relative_vel_w[:, 3:6])
        ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)
        
        # 當前 EE 姿態
        current_ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)
        
        # 獲取 Jacobian（Kit 110 回傳 warp array，需轉 torch）
        jacobian_w = physx_to_torch(self._robot.root_physx_view.get_jacobians())[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ]
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])
        
        # 獲取動力學信息（Kit 110 回傳 warp array，需轉 torch）
        mass_matrix_full = physx_to_torch(self._robot.root_physx_view.get_generalized_mass_matrices())
        mass_matrix = mass_matrix_full[:, self._arm_joint_ids, :][:, :, self._arm_joint_ids]
        gravity = physx_to_torch(self._robot.root_physx_view.get_gravity_compensation_forces())[:, self._arm_joint_ids]
        
        joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._arm_joint_ids]
        
        # 計算 OSC 力矩
        joint_efforts = self._osc_controller.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=current_ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            mass_matrix=mass_matrix,
            gravity=gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
        )
        
        # 應用關節力矩
        self._robot.set_joint_effort_target(joint_efforts, self._arm_joint_ids)
        
        # 應用夾爪
        self._apply_gripper(gripper_pos)
        
        # 寫入模擬
        self._robot.write_data_to_sim()
    
    def reset(self) -> None:
        """重置控制器"""
        self._osc_controller.reset()
        
        # 重置機器人到初始位置
        initial_pos = self.initial_joint_pos
        self._robot.write_joint_state_to_sim(initial_pos, torch.zeros_like(initial_pos))
        self._robot.reset()
        
        # 重新設定 OSC 指令為當前姿態
        self._osc_controller.set_command(self.current_ee_pose)
