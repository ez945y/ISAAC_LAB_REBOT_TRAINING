# Copyright (c) 2022-2025, The Isaac Lab Project Developers. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class DrawerTaskCommand(CommandTerm):
    """隨機抽屜任務：完成一個抽屜後重置並抽新任務。
    
    - 每個 episode 開始時隨機抽一個抽屜當目標
    - 完成後 (>=done_threshold) 給稀疏獎勵，重置抽屜，再抽新目標
    - 沒有固定順序
    """
    
    cfg: "DrawerTaskCommandCfg"
    
    def __init__(self, cfg: "DrawerTaskCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        
        # 獲取實際的 joint_ids
        cabinet = env.scene[cfg.cabinet_cfg.name]
        joint_ids, joint_names = cabinet.find_joints(cfg.cabinet_cfg.joint_names)
        self.joint_ids = list(joint_ids)
        
        print(f"[DrawerTaskCommand] Resolved joint_ids={self.joint_ids}, names={joint_names}")
        
        # 當前目標 (0=bottom, 1=top)
        self.current_target = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 當前目標的實際 joint_id
        self.current_joint_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 當前目標的 frame index
        self.current_frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 剛完成的標記 (用於稀疏獎勵)
        self.just_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 完成計數
        self.completed_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self._resample_command(torch.arange(self.num_envs, device=self.device))
    
    @property
    def command(self) -> torch.Tensor:
        """返回當前目標 (0=bottom, 1=top)"""
        return self.current_target.unsqueeze(-1).float()
    
    def _update_indices(self, env_ids: torch.Tensor):
        """根據 current_target 更新 joint_id 和 frame_idx"""
        joint_ids_tensor = torch.tensor(self.joint_ids, device=self.device, dtype=torch.long)
        self.current_joint_id[env_ids] = joint_ids_tensor[self.current_target[env_ids]]
        self.current_frame_idx[env_ids] = self.current_target[env_ids].clone()
    
    def _update_command(self) -> None:
        # Reset just_completed flag
        self.just_completed.fill_(False)
        
        cabinet = self._env.scene[self.cfg.cabinet_cfg.name]
        joint_pos = cabinet.data.joint_pos
        
        # 獲取當前目標抽屜的位置
        batch_idx = torch.arange(self.num_envs, device=self.device)
        current_drawer_pos = joint_pos[batch_idx, self.current_joint_id]
        
        # 檢查哪些環境完成了當前任務
        completed_mask = current_drawer_pos >= self.cfg.done_threshold
        completed_ids = torch.where(completed_mask)[0]
        
        if len(completed_ids) > 0:
            # 標記剛完成
            self.just_completed[completed_ids] = True
            self.completed_count[completed_ids] += 1
            
            # 重置完成的抽屜回 0
            self._reset_drawer(completed_ids)
            
            # 抽新任務
            self._sample_new_target(completed_ids)
    
    def _reset_drawer(self, env_ids: torch.Tensor):
        """重置指定環境中已完成的抽屜"""
        cabinet = self._env.scene[self.cfg.cabinet_cfg.name]
        
        # 獲取需要重置的 joint indices
        reset_joint_ids = self.current_joint_id[env_ids]
        
        # 重置關節位置和速度（向量化操作）
        joint_pos = cabinet.data.joint_pos.clone()
        joint_vel = cabinet.data.joint_vel.clone()
        
        # 使用向量化索引
        joint_pos[env_ids, reset_joint_ids] = 0.0
        joint_vel[env_ids, reset_joint_ids] = 0.0
        
        # 寫回模擬
        cabinet.write_joint_state_to_sim(joint_pos, joint_vel)
    
    def _sample_new_target(self, env_ids: torch.Tensor):
        """隨機抽新目標"""
        self.current_target[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device, dtype=torch.long)
        self._update_indices(env_ids)
    
    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Episode 重置時調用"""
        env_ids = torch.as_tensor(env_ids, device=self.device)
        self.current_target[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device, dtype=torch.long)
        self.completed_count[env_ids] = 0
        self.just_completed[env_ids] = False
        self._update_indices(env_ids)

    def _update_metrics(self) -> None:
        pass


@configclass
class DrawerTaskCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = DrawerTaskCommand
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint", "drawer_top_joint"])
    done_threshold: float = 0.27
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    debug_vis: bool = False
