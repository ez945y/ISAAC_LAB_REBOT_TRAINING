# Copyright (c) 2022-2025, The Isaac Lab Project Developers. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, List, Tuple
from dataclasses import field

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ----------------------------------------------------------------
# 1. 先定義「實作」類別：GoalPoseCommand
# ----------------------------------------------------------------
class GoalPoseCommand(CommandTerm):
    """
    一個指令生成器，用於設定和提供一個固定的或可由外部更新的目標姿態。
    """
    cfg: "GoalPoseCommandCfg"

    def __init__(self, cfg: "GoalPoseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.goal_poses_buf = torch.tensor(cfg.default_goal, device=self.device).repeat(self.num_envs, 1)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        # Waypoint state per env (managed internally to decouple MDP from indexing details)
        self.waypoint_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Distance tracking for position progress reward
        self.last_dist_to_goal = torch.zeros(self.num_envs, device=self.device)
        # 在初始化時就執行一次 resample，以設定初始指令
        self._resample_command(torch.arange(self.num_envs, device=self.device).tolist())
        # 如果有航點配置，設定第一個航點為初始目標
        if hasattr(self.cfg, "waypoints") and self.cfg.waypoints:
            self.reset_waypoints()

    @property
    def command(self) -> torch.Tensor:
        """The desired pose in world frame. Shape is (num_envs, 7)."""
        return torch.cat([self.pos_command_w, self.quat_command_w], dim=1)

    # -- 補上缺少的函式 --
    def _update_command(self) -> None:
        """
        這個函式在每個環境 step 的最後被呼叫，用來更新內部狀態或指標。
        """
        self._update_metrics()

    def _update_metrics(self) -> None:
        """計算目前的指令和機器人狀態之間的誤差。"""
        # 位置誤差 (歐幾里德距離)
        self.metrics["error_pos"] = torch.norm(
            self.pos_command_w - self.robot.data.root_pos_w, dim=1
        )
        # 航向誤差 (簡化計算)
        # 這裡僅計算四元數的差異，更精確的作法是計算旋轉角度差
        self.metrics["error_heading"] = torch.norm(
            self.quat_command_w - self.robot.data.root_quat_w, dim=1
        )
    # -- 補上缺少的函式 (結束) --

    def _resample_command(self, env_ids: Sequence[int]):
        """為指定的環境重新採樣指令 (在這裡是直接載入目標)。"""
        self.pos_command_w[env_ids] = self.goal_poses_buf[env_ids, :3]
        self.quat_command_w[env_ids] = self.goal_poses_buf[env_ids, 3:]

    def update_goal(self, goal_poses: torch.Tensor, env_ids: torch.Tensor | None = None):
        """從外部更新指定環境的目標姿態。"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 更新儲存的目標
        self.goal_poses_buf[env_ids] = goal_poses
        # 立即更新當前的指令
        self.pos_command_w[env_ids] = goal_poses[:, :3]
        self.quat_command_w[env_ids] = goal_poses[:, 3:]

    # --------------------------
    # Waypoint control (decoupled)
    # --------------------------
    def reset_waypoints(self, env_ids: torch.Tensor | None = None):
        """Reset per-env waypoint index to 0 and set command to first waypoint."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.waypoint_indices[env_ids] = 0
        self._apply_waypoint_indices(env_ids)
        # Reset distance tracking for these environments
        self.reset_last_dist_to_goal(env_ids)

    def advance_waypoints(self, env_ids: torch.Tensor):
        """Advance per-env waypoint index by 1 (cyclic) and set command to new waypoint."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        waypoints = getattr(self.cfg, "waypoints", None)
        if not waypoints:
            return
        num_waypoints = len(waypoints)
        if num_waypoints == 0:
            return
        self.waypoint_indices[env_ids] = (self.waypoint_indices[env_ids] + 1) % num_waypoints
        self._apply_waypoint_indices(env_ids)
        # Reset distance tracking for these environments after waypoint change
        self.reset_last_dist_to_goal(env_ids)

    def _apply_waypoint_indices(self, env_ids: torch.Tensor):
        """Build and apply goals from current indices for given env_ids."""
        waypoints = getattr(self.cfg, "waypoints", None)
        if not waypoints:
            return
        num_waypoints = len(waypoints)
        if num_waypoints == 0:
            return
        # split combined (x, y, yaw_deg) into tensors
        xy = [(w[0], w[1]) for w in waypoints]
        yaws_deg = [w[2] if len(w) > 2 else None for w in waypoints]
        wp_tensor = torch.tensor(xy, dtype=torch.float32, device=self.device)
        indices = self.waypoint_indices[env_ids]
        # ensure indices are LongTensor for indexing
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        else:
            indices = indices.to(dtype=torch.long)
        targets_xy = wp_tensor.index_select(0, indices)
        default_goal_tensor = torch.tensor(self.cfg.default_goal, device=self.device)
        default_z = default_goal_tensor[2]
        default_quat = default_goal_tensor[3:]
        # collect yaw per index if available in combined config
        yaw_list = yaws_deg
        new_goals = torch.zeros(len(env_ids), 7, device=self.device)
        new_goals[:, 0:2] = targets_xy
        new_goals[:, 2] = default_z
        # if any yaw is provided, apply per-env yaw; else keep default quaternion
        if any(y is not None for y in yaw_list):
            yaw_vals = []
            for i in indices:
                y = yaw_list[int(i.item()) % len(yaw_list)]
                if y is None:
                    yaw_vals.append(None)
                else:
                    yaw_vals.append(float(y))
            # build quaternion row-wise
            # fill defaults first
            new_goals[:, 3:] = default_quat.repeat(len(env_ids), 1)
            # overwrite rows that have yaw
            mask = [v is not None for v in yaw_vals]
            if any(mask):
                yaw_deg_tensor = torch.tensor([v for v in yaw_vals if v is not None], device=self.device)
                yaw_rad = -torch.deg2rad(yaw_deg_tensor)
                half = 0.5 * yaw_rad
                cosv = torch.cos(half)
                sinv = torch.sin(half)
                # indices in env_ids that have yaw
                idxs = torch.tensor([k for k, v in enumerate(mask) if v], dtype=torch.long, device=self.device)
                new_goals[idxs, 3] = cosv
                new_goals[idxs, 4] = 0.0
                new_goals[idxs, 5] = 0.0
                new_goals[idxs, 6] = sinv
        else:
            new_goals[:, 3:] = default_quat.repeat(len(env_ids), 1)
        self.update_goal(new_goals, env_ids=env_ids)

    def get_last_dist_to_goal(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get last distance to goal for specified environments."""
        if env_ids is None:
            return self.last_dist_to_goal
        return self.last_dist_to_goal[env_ids]

    def update_last_dist_to_goal(self, new_dist: torch.Tensor, env_ids: torch.Tensor | None = None):
        """Update last distance to goal for specified environments."""
        if env_ids is None:
            self.last_dist_to_goal = new_dist.clone()
        else:
            self.last_dist_to_goal[env_ids] = new_dist

    def reset_last_dist_to_goal(self, env_ids: torch.Tensor | None = None):
        """Reset last distance to goal for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Calculate current distance to goal
        robot_pos_xy = self.robot.data.root_pos_w[env_ids, :2]
        command = self.command[env_ids]
        target_pos_xy = command[:, :2]
        current_dist = torch.norm(target_pos_xy - robot_pos_xy, dim=1)
        
        self.last_dist_to_goal[env_ids] = current_dist


# ----------------------------------------------------------------
# 2. 再定義「設定」類別：GoalPoseCommandCfg
# ----------------------------------------------------------------
@configclass
class GoalPoseCommandCfg(CommandTermCfg):
    """Configuration for the goal pose command generator with waypoint management."""
    
    class_type: type[CommandTerm] = GoalPoseCommand
    asset_name: str = "robot"
    default_goal: list[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    debug_vis: bool = True
    
    # --- Waypoints: each as (x, y, yaw_deg). yaw_deg optional, CW positive by user convention ---
    waypoints: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [(2.8, 0.0, 0.0), (2.8, -4.3, -90.0), (-1.6, -4.3, -180.0), (-1.6, 0.0, -270.0)]
    )
    tolerance: float = 0.5
    command_name: str = "base_velocity"