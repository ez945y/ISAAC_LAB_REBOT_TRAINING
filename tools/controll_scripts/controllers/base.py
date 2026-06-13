# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""控制器抽象基類"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from isaaclab.assets import Articulation

from ..configs.base import BaseRobotConfig
from ..utils import physx_to_torch


class BaseController(ABC):
    """控制器抽象基類
    
    定義統一的控制器介面，輸入為絕對姿態。
    支持 leader arm 等外部設備直接輸入目標姿態。
    """
    
    def __init__(
        self,
        robot: Articulation,
        robot_config: BaseRobotConfig,
        device: str,
        num_envs: int = 1,
    ):
        """初始化控制器
        
        Args:
            robot: Isaac Lab Articulation 對象
            robot_config: 機器人配置
            device: 計算設備 (e.g., "cuda:0")
            num_envs: 環境數量
        """
        self._robot = robot
        self._config = robot_config
        self._device = device
        self._num_envs = num_envs
        
        # 獲取索引
        self._setup_indices()
        
        # 讀取關節限制
        self._read_joint_limits()
        
        # 保存初始狀態（在控制器創建時）
        self._save_initial_state()
    
    def _setup_indices(self) -> None:
        """設定關節和末端執行器索引"""
        # 末端執行器
        body_ids, body_names = self._robot.find_bodies(self._config.ee_body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Cannot find EE body: {self._config.ee_body_name}")
        self._ee_body_idx = body_ids[0]
        
        # 手臂關節
        self._arm_joint_ids, self._arm_joint_names = self._robot.find_joints(
            self._config.arm_joint_names
        )
        
        # 夾爪關節
        self._gripper_joint_ids = self._robot.find_joints([self._config.gripper_joint_name])[0]
        
        # Jacobian 索引（固定基座需要 -1）
        if self._robot.is_fixed_base:
            self._jacobi_body_idx = self._ee_body_idx - 1
            self._jacobi_joint_ids = self._arm_joint_ids
        else:
            self._jacobi_body_idx = self._ee_body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._arm_joint_ids]
    
    def _read_joint_limits(self) -> None:
        """從機器人讀取關節限制"""
        # 從 PhysX view 讀取關節限制（Kit 110 回傳 warp array，需轉 torch）
        joint_limits = physx_to_torch(self._robot.root_physx_view.get_dof_limits())

        # 手臂關節限制
        self._arm_joint_lower = joint_limits[0, self._arm_joint_ids, 0]
        self._arm_joint_upper = joint_limits[0, self._arm_joint_ids, 1]
        
        # 夾爪關節限制
        self._gripper_lower = joint_limits[0, self._gripper_joint_ids, 0].item()
        self._gripper_upper = joint_limits[0, self._gripper_joint_ids, 1].item()
        
        print(f"[{self.__class__.__name__}] 讀取關節限制:")
        for i, name in enumerate(self._arm_joint_names):
            print(f"  {name}: [{self._arm_joint_lower[i]:.2f}, {self._arm_joint_upper[i]:.2f}]")
        print(f"  gripper: [{self._gripper_lower:.2f}, {self._gripper_upper:.2f}]")
    
    @abstractmethod
    def compute(self, target_pose: torch.Tensor, gripper_pos: float) -> None:
        """計算並應用控制指令
        
        Args:
            target_pose: 絕對目標姿態 [pos(3) + quat(4)] 在 base frame
                         Shape: (num_envs, 7)
            gripper_pos: 夾爪位置 [0, 1] (0=關閉, 1=全開)
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """重置控制器到初始狀態"""
        pass
    
    @property
    def current_ee_pose(self) -> torch.Tensor:
        """獲取當前末端執行器姿態 (base frame)
        
        Returns:
            torch.Tensor: 姿態 [pos(3) + quat(4)], Shape: (num_envs, 7)
        """
        import isaaclab.utils.math as math_utils

        # Kit 110 的 robot.data.* 為 warp ProxyArray，轉成 torch 再做數學運算
        ee_pos_w = physx_to_torch(self._robot.data.body_pos_w)[:, self._ee_body_idx]
        ee_quat_w = physx_to_torch(self._robot.data.body_quat_w)[:, self._ee_body_idx]
        root_pos_w = physx_to_torch(self._robot.data.root_pos_w)
        root_quat_w = physx_to_torch(self._robot.data.root_quat_w)

        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        
        return torch.cat([ee_pos_b, ee_quat_b], dim=-1)
    
    def _save_initial_state(self) -> None:
        """保存初始狀態（在控制器創建時的實際姿態）"""
        # 保存當前關節位置
        self._initial_joint_pos = physx_to_torch(self._robot.data.joint_pos).clone()
        # 保存當前 EE 姿態
        self._initial_ee_pose = self.current_ee_pose.clone()
        print(f"[{self.__class__.__name__}] 保存初始狀態:")
        print(f"  EE 位置: {self._initial_ee_pose[0, :3].cpu().numpy()}")
    
    @property
    def initial_joint_pos(self) -> torch.Tensor:
        """獲取初始關節位置（控制器創建時的實際位置）"""
        return self._initial_joint_pos.clone()
    
    @property
    def initial_ee_pose(self) -> torch.Tensor:
        """獲取初始 EE 姿態（控制器創建時的實際姿態）"""
        return self._initial_ee_pose.clone()
    
    @property
    def robot(self) -> Articulation:
        """獲取機器人對象"""
        return self._robot
    
    @property
    def config(self) -> BaseRobotConfig:
        """獲取機器人配置"""
        return self._config
    
    @property
    def device(self) -> str:
        """獲取計算設備"""
        return self._device
    
    def _apply_gripper(self, gripper_pos: float) -> None:
        """應用夾爪位置
        
        Args:
            gripper_pos: 夾爪位置 [0, 1] (0=關閉, 1=全開)
        """
        # 將 [0, 1] 映射到實際關節限制
        actual_pos = self._gripper_lower + gripper_pos * (self._gripper_upper - self._gripper_lower)
        jaw_pos = torch.tensor([[actual_pos]], device=self._device)
        self._robot.set_joint_position_target_index(jaw_pos, self._gripper_joint_ids)
