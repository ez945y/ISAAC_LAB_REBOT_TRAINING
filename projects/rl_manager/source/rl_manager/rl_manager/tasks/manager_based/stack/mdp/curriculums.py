# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def initial_final_interpolate_fn(env: ManagerBasedRLEnv, env_id, data, initial_value, final_value, difficulty_term_str):
    """
    Interpolate between initial value iv and final value fv, for any arbitrarily
    nested structure of lists/tuples in 'data'. Scalars (int/float) are handled
    at the leaves.
    """
    # get the fraction scalar on the device
    difficulty_term: DifficultyScheduler = getattr(env.curriculum_manager.cfg, difficulty_term_str).func
    frac = difficulty_term.difficulty_frac
    if frac < 0.1:
        # no-op during start, since the difficulty fraction near 0 is wasting of resource.
        return mdp.modify_env_param.NO_CHANGE

    # convert iv/fv to tensors, but we'll peel them apart in recursion
    initial_value_tensor = torch.tensor(initial_value, device=env.device)
    final_value_tensor = torch.tensor(final_value, device=env.device)

    return _recurse(initial_value_tensor.tolist(), final_value_tensor.tolist(), data, frac)


def _recurse(iv_elem, fv_elem, data_elem, frac):
    # If it's a sequence, rebuild the same type with each element recursed
    if isinstance(data_elem, Sequence) and not isinstance(data_elem, (str, bytes)):
        # Note: we assume initial value element and final value element have the same structure as data
        return type(data_elem)(_recurse(iv_e, fv_e, d_e, frac) for iv_e, fv_e, d_e in zip(iv_elem, fv_elem, data_elem))
    # Otherwise it's a leaf scalar: do the interpolation
    new_val = frac * (fv_elem - iv_elem) + iv_elem
    if isinstance(data_elem, int):
        return int(new_val.item())
    else:
        # cast floats or any numeric
        return new_val.item()


class DifficultyScheduler(ManagerTermBase):
    """Adaptive difficulty scheduler for curriculum learning.

    使用 StackProgressTracker 的完成率來決定是否升級難度。
    
    升級條件：達到目標進度的環境比例 >= success_rate (預設 80%)
    
    Args:
        cfg: Configuration object specifying scheduler parameters.
        env: The manager-based RL environment.

    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        init_difficulty = self.cfg.params.get("init_difficulty", 0)
        # 全局難度（所有環境共享）
        self.current_difficulty = init_difficulty
        # 保持 per-env tensor 以兼容現有代碼
        self.current_adr_difficulties = torch.ones(env.num_envs, device=env.device) * init_difficulty
        self.difficulty_frac = 0.0

    def get_state(self):
        return self.current_adr_difficulties

    def set_state(self, state: torch.Tensor):
        self.current_adr_difficulties = state.clone().to(self._env.device)
        self.current_difficulty = self.current_adr_difficulties[0].item()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        target_progress: int = 2,           # 目標進度等級 (2 = 完成堆疊)
        success_rate: float = 0.8,          # 完成率門檻 (80%)
        min_samples: int = 1000,            # 最少樣本數
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 50,
        promotion_only: bool = True,        # 只升級不降級
    ):
        """
        根據完成率調整難度。
        
        Args:
            env: 環境
            env_ids: 被 reset 的環境 ID（不使用）
            target_progress: 目標進度等級 (1=抓到, 2=堆疊完成)
            success_rate: 需要達到的完成率門檻 (預設 80%)
            min_samples: 最少樣本數才能評估
            init_difficulty: 初始難度
            min_difficulty: 最小難度
            max_difficulty: 最大難度
            promotion_only: 只升級不降級 (預設 True)
        """
        # 取得 StackProgressTracker
        progress_tracker = env.command_manager.get_term("stack_progress")
        
        # 判斷是否應該升級
        should_promote = progress_tracker.should_promote(
            target_progress=target_progress,
            success_rate=success_rate,
            min_samples=min_samples,
        )
        
        # 升級難度
        if should_promote:
            self.current_difficulty = min(self.current_difficulty + 1, max_difficulty)
            # 重置統計，開始新一輪評估
            progress_tracker.reset_stats()
        
        # 同步所有環境的難度
        self.current_adr_difficulties[:] = self.current_difficulty
        
        # 計算難度分數 (0.0 ~ 1.0)
        self.difficulty_frac = self.current_difficulty / max(max_difficulty, 1)
        
        return self.difficulty_frac
