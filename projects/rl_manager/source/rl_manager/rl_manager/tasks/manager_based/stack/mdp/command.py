# Copyright (c) 2022-2025, The Isaac Lab Project Developers. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, List, Tuple
from dataclasses import field

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.envs import ManagerBasedRLEnv

# ----------------------------------------------------------------
# 3. StackProgressTracker: 追蹤所有環境的堆疊進度和完成率
# ----------------------------------------------------------------
class StackProgressTracker(CommandTerm):
    """
    追蹤所有環境的堆疊進度和完成率，供 DifficultyScheduler 使用。
    
    此類別會：
    1. 計算當前的 stack_progress (0~2)
    2. 追蹤每個進度等級的完成次數
    3. 計算各進度等級的完成率
    4. 提供 get_completion_rate(target_progress) 讓 DifficultyScheduler 判斷是否升級
    
    完成率計算：達到 target_progress 的環境數 / 總評估數
    """
    cfg: "StackProgressTrackerCfg"

    def __init__(self, cfg: "StackProgressTrackerCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        
        # 當前進度（每個 env）
        self.current_progress = torch.zeros(self.num_envs, device=self.device)
        
        # 完成次數統計（每個進度等級）
        self.progress_counts = torch.zeros(3, device=self.device)  # 進度 0, 1, 2
        self.total_samples: int = 0
        
        # 設定 metrics - 只用標量避免 CUDA 錯誤
        self.metrics["completion_rate_1"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["completion_rate_2"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """回傳當前進度作為 command (用於兼容性)。Shape is (num_envs, 1)."""
        return self.current_progress.unsqueeze(1)

    def _update_command(self) -> None:
        """
        在每個環境 step 時被呼叫，更新進度追蹤。
        """
        # 計算當前進度
        self.current_progress = stack_progress(
            env=self._env,
            robot_cfg=SceneEntityCfg(self.cfg.robot_name),
            cube_1_cfg=SceneEntityCfg(self.cfg.cube_1_name),
            cube_2_cfg=SceneEntityCfg(self.cfg.cube_2_name),
            xy_threshold=self.cfg.xy_threshold,
            height_threshold=self.cfg.height_threshold,
            height_diff=self.cfg.height_diff,
            grasp_diff_threshold=self.cfg.grasp_diff_threshold,
        )
        
        # 更新各進度等級的計數
        for level in range(3):  # 0, 1, 2
            count = (self.current_progress >= level).sum().item()
            self.progress_counts[level] += count
        self.total_samples += self.num_envs
        
        # 更新 metrics（用於 logging）
        if self.total_samples > 0:
            rate_1 = self.progress_counts[1].item() / self.total_samples
            rate_2 = self.progress_counts[2].item() / self.total_samples
            self.metrics["completion_rate_1"][:] = rate_1
            self.metrics["completion_rate_2"][:] = rate_2

    def _update_metrics(self) -> None:
        """更新 metrics (已在 _update_command 中處理)。"""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """當環境被 reset 時，不做任何事。"""
        pass

    def reset_stats(self):
        """重置統計（升級難度後使用）。"""
        self.progress_counts.zero_()
        self.total_samples = 0

    def get_completion_rate(self, target_progress: int = 1) -> float:
        """
        取得達到指定進度的完成率。
        
        Args:
            target_progress: 目標進度等級 (1 = 抓到方塊, 2 = 完成堆疊)
            
        Returns:
            完成率 (0.0 ~ 1.0)
        """
        if self.total_samples == 0:
            return 0.0
        return self.progress_counts[target_progress].item() / self.total_samples

    def get_total_samples(self) -> int:
        """取得總樣本數。"""
        return self.total_samples

    def should_promote(self, target_progress: int = 2, success_rate: float = 0.8, min_samples: int = 1000) -> bool:
        """
        判斷是否應該升級難度。
        
        Args:
            target_progress: 目標進度等級
            success_rate: 需要達到的完成率門檻 (預設 80%)
            min_samples: 最少樣本數才能評估
            
        Returns:
            是否應該升級
        """
        if self.total_samples < min_samples:
            return False
        return self.get_completion_rate(target_progress) >= success_rate




# ----------------------------------------------------------------
# 4. StackProgressTrackerCfg: StackProgressTracker 的配置
# ----------------------------------------------------------------
@configclass
class StackProgressTrackerCfg(CommandTermCfg):
    """Configuration for the stack progress tracker."""
    
    class_type: type[CommandTerm] = StackProgressTracker
    
    # 場景中的物件名稱
    robot_name: str = "robot"
    cube_1_name: str = "cube_1"
    cube_2_name: str = "cube_2"
    # cube_3_name: str = "cube_3"  # 暫時不用
    
    # stack_progress 參數
    xy_threshold: float = 0.04
    height_threshold: float = 0.005
    height_diff: float = 0.039
    grasp_diff_threshold: float = 0.02
    
    # 難度升級參數
    progress_threshold: float = 1.0  # 平均進度門檻 (1.0 = 完成第一次堆疊)
    min_steps_for_promotion: int = 100  # 最少步數才能判斷
    
    # CommandTermCfg 必要參數 - 使用很大的數字代替 inf，因為 PyTorch uniform_() 不支援 inf
    resampling_time_range: tuple[float, float] = (1e9, 1e9)  # 不自動 resample
    debug_vis: bool = False


def stack_progress(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    # cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),  # 暫時不用
    xy_threshold: float = 0.04,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    grasp_diff_threshold: float = 0.02,
    gripper_open_val: float = None,      # 從 env.cfg 取
    gripper_threshold: float = None,
    open_threshold: float = None,
) -> torch.Tensor:
    """
    回傳每個 env 的堆疊進度等級 (0~2)：
    0: 還沒抓到 cube_2
    1: 抓到 cube_2，但還沒放到 cube_1 上
    2: 成功把 cube_2 放到 cube_1 上 (完成!)
    
    # 以下暫時不用：
    # 3: 抓到 cube_3
    # 4: 成功把 cube_3 放到 cube_2 上（完整塔）
    """
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    # cube_3: RigidObject = env.scene[cube_3_cfg.name]  # 暫時不用

    # 抓取判斷參數（從 cfg 取，或用預設）
    gripper_open_val = gripper_open_val or env.cfg.gripper_open_val
    gripper_threshold = gripper_threshold or env.cfg.gripper_threshold
    open_threshold = open_threshold or env.cfg.open_threshold

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    gripper_pos = robot.data.joint_pos[:, gripper_joint_ids[0]]

    # 輔助函式：判斷是否抓到某個 cube
    def is_grasped(cube: RigidObject, diff_threshold: float = grasp_diff_threshold) -> torch.Tensor:
        obj_pos = cube.data.root_pos_w
        ee_pos = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
        pose_diff = torch.norm(obj_pos - ee_pos + torch.tensor([0.0, 0.0, 0.1], device=obj_pos.device), dim=1)
        closed_enough = torch.abs(gripper_pos - gripper_open_val) > gripper_threshold
        return torch.logical_and(pose_diff < diff_threshold, closed_enough)

    # 輔助函式：判斷 upper 是否疊在 lower 上
    def is_stacked(upper: RigidObject, lower: RigidObject) -> torch.Tensor:
        pos_diff = upper.data.root_pos_w - lower.data.root_pos_w
        xy_dist = torch.norm(pos_diff[:, :2], dim=1)
        h_dist = torch.norm(pos_diff[:, 2:], dim=1)
        stacked = torch.logical_and(xy_dist < xy_threshold, torch.abs(h_dist - height_diff) < height_threshold)
        stacked = torch.logical_and(pos_diff[:, 2] < 0.0, stacked)  # upper 在 lower 上方
        return stacked

    # 進度計算（從 0 到 2）
    progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # 進度 1：抓到 cube_2
    grasped_cube2 = is_grasped(cube_2)
    progress = torch.where(grasped_cube2, torch.ones_like(progress) * 1, progress)

    # 進度 2：cube_2 疊在 cube_1 上 (完成!)
    stacked_2_on_1 = is_stacked(cube_2, cube_1)
    progress = torch.where(stacked_2_on_1, torch.ones_like(progress) * 2, progress)

    # # 進度 3：抓到 cube_3（即使 2 on 1 失敗，也可進階）- 暫時不用
    # grasped_cube3 = is_grasped(cube_3)
    # progress = torch.where(grasped_cube3, torch.maximum(progress, torch.ones_like(progress) * 3), progress)

    # # 進度 4：cube_3 疊在 cube_2 上（完整塔）- 暫時不用
    # stacked_3_on_2 = is_stacked(cube_3, cube_2)
    # progress = torch.where(stacked_3_on_2, torch.ones_like(progress) * 4, progress)

    # 額外：gripper 開啟確認（可選）
    gripper_open = torch.abs(gripper_pos - gripper_open_val) < open_threshold
    progress = torch.where(gripper_open & (progress >= 2), progress, progress)  # 確保放開後才算成功

    return progress.float()  # 返回 (num_envs,) 的浮點 tensor，值 0~2