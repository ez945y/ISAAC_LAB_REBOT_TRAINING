# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Differential IK 控制器（加權版本，支持 5-DOF 機器人）"""

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from .base import BaseController
from ..configs.base import BaseRobotConfig
from ..utils import physx_to_torch


class IKController(BaseController):
    """Differential IK 控制器包裝
    
    使用 Isaac Lab 的 DifferentialIKController，
    輸出關節位置目標。
    
    對於 5-DOF 機器人，使用加權 pose 控制：
    - 位置誤差：高權重（優先追蹤）
    - 姿態誤差：低權重（允許一定偏差）
    """
    
    def __init__(
        self,
        robot: Articulation,
        robot_config: BaseRobotConfig,
        device: str,
        num_envs: int = 1,
    ):
        super().__init__(robot, robot_config, device, num_envs)
        
        # 創建 IK 控制器（使用 pose 控制以支持旋轉）
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": robot_config.ik_lambda_val},
        )
        self._ik_controller = DifferentialIKController(
            ik_cfg, num_envs=num_envs, device=device
        )
        
        # 姿態權重（對於 5-DOF 機器人，降低姿態權重讓位置優先）
        # 5 DOF = 只有 5 個關節控制 6 維姿態，需要取捨
        self._orientation_weight = robot_config.ik_orientation_weight
        
        # 初始化
        self._ik_controller.reset()
    
    def compute(self, target_pose: torch.Tensor, gripper_pos: float) -> None:
        """計算並應用 IK 控制
        
        Args:
            target_pose: 絕對目標姿態 [pos(3) + quat(4)] 在 base frame
            gripper_pos: 夾爪位置 [0, 1]
        """
        # 獲取當前狀態（Kit 110 的 robot.data.* 為 warp ProxyArray，轉成 torch）
        ee_pos_w = physx_to_torch(self._robot.data.body_pos_w)[:, self._ee_body_idx]
        ee_quat_w = physx_to_torch(self._robot.data.body_quat_w)[:, self._ee_body_idx]
        root_pos_w = physx_to_torch(self._robot.data.root_pos_w)
        root_quat_w = physx_to_torch(self._robot.data.root_quat_w)
        
        # 轉換到基座座標系
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        
        # 創建加權目標姿態：
        # 位置目標：直接使用目標
        # 姿態目標：混合當前姿態和目標姿態（降低姿態追蹤激進程度）
        weighted_target = target_pose.clone()
        if self._orientation_weight < 1.0:
            # 使用 slerp 在當前姿態和目標姿態之間插值
            weighted_target[:, 3:7] = self._slerp_quat(
                ee_quat_b, target_pose[:, 3:7], self._orientation_weight
            )
        
        # 設定 IK 指令
        self._ik_controller.set_command(weighted_target)
        
        # 獲取 Jacobian（Kit 110 回傳 warp array，需轉 torch）
        jacobian_w = physx_to_torch(self._robot.root_physx_view.get_jacobians())[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ]
        
        # 轉換 Jacobian 到基座座標系
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])
        
        # 獲取當前關節位置
        joint_pos = physx_to_torch(self._robot.data.joint_pos)[:, self._arm_joint_ids]
        
        # 計算 IK
        joint_pos_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos)
        
        # 應用關節目標
        self._robot.set_joint_position_target_index(joint_pos_des, self._arm_joint_ids)
        
        # 應用夾爪
        self._apply_gripper(gripper_pos)
        
        # 寫入模擬
        self._robot.write_data_to_sim()
    
    def _slerp_quat(self, q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
        """球面線性插值（SLERP）
        
        Args:
            q0: 起始四元數 (N, 4) [w, x, y, z]
            q1: 目標四元數 (N, 4) [w, x, y, z]
            t: 插值因子 [0, 1]
            
        Returns:
            插值後的四元數 (N, 4)
        """
        # 簡化版本：線性插值後歸一化（對於小角度足夠好）
        q = (1 - t) * q0 + t * q1
        return q / torch.norm(q, dim=-1, keepdim=True)
    
    def reset(self) -> None:
        """重置控制器"""
        self._ik_controller.reset()
        
        # 重置機器人到初始位置
        initial_pos = self.initial_joint_pos
        self._robot.write_joint_state_to_sim(initial_pos, torch.zeros_like(initial_pos))
        self._robot.reset()
        
        # 重新設定 IK 指令為當前姿態
        self._ik_controller.set_command(self.current_ee_pose)
