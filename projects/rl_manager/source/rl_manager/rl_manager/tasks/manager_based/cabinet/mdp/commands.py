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


class DrawerOrderCommand(CommandTerm):
    """隨機決定每個環境的抽屜開啟順序。
    
    order 0: 先下後上 (bottom -> top)
    order 1: 先上後下 (top -> bottom)
    
    存儲：
    - drawer_order: 初始順序 (0 或 1)
    - current_target: 當前目標的相對索引 (0 或 1)
    - current_joint_id: 當前目標的實際 joint_id
    - current_frame_idx: 當前目標的 frame index (用於 cabinet_frame)
    """
    
    cfg: "DrawerOrderCommandCfg"
    
    def __init__(self, cfg: "DrawerOrderCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        
        # 獲取實際的 joint_ids
        cabinet = env.scene[cfg.cabinet_cfg.name]
        self.joint_ids = cabinet.find_joints(cfg.cabinet_cfg.joint_names)[0]
        # joint_ids[0] = bottom drawer, joint_ids[1] = top drawer
        
        self.drawer_order = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_target = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_joint_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self._resample_command(torch.arange(self.num_envs, device=self.device))
    
    @property
    def command(self) -> torch.Tensor:
        """返回當前目標的相對索引 (0=bottom, 1=top)"""
        return self.current_target.unsqueeze(-1).float()
    
    def _update_indices(self):
        """根據 current_target 更新 joint_id 和 frame_idx"""
        # current_target: 0=bottom, 1=top
        # joint_ids[0]=bottom_joint_id, joint_ids[1]=top_joint_id
        joint_ids_tensor = torch.tensor(self.joint_ids, device=self.device, dtype=torch.long)
        self.current_joint_id = joint_ids_tensor[self.current_target]
        # frame_idx 和 current_target 相同 (cabinet_frame 的順序是 bottom=0, top=1)
        self.current_frame_idx = self.current_target.clone()
    
    def _update_command(self) -> None:
        cabinet = self._env.scene[self.cfg.cabinet_cfg.name]
        joint_pos = cabinet.data.joint_pos[:, self.joint_ids]
        
        bottom_pos = joint_pos[:, 0]
        top_pos = joint_pos[:, 1]
        
        first_drawer_pos = torch.where(self.drawer_order == 0, bottom_pos, top_pos)
        first_done = first_drawer_pos >= self.cfg.done_threshold
        self.current_target = torch.where(first_done, 1 - self.drawer_order, self.drawer_order)
        
        self._update_indices()
    
    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device)
        self.drawer_order[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device, dtype=torch.long)
        self.current_target[env_ids] = self.drawer_order[env_ids]
        self._update_indices()

    def _update_metrics(self) -> None:
        pass


@configclass
class DrawerOrderCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = DrawerOrderCommand
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint", "drawer_top_joint"])
    done_threshold: float = 0.3
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    debug_vis: bool = False
