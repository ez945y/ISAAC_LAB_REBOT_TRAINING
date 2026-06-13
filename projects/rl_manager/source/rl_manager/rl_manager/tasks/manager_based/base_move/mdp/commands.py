# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Movement phase command for base_move task.

Pattern: Forward → Center → Left → Center (repeat)
Each single move is move_distance (e.g. 3cm), return to center each time.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MovementPhaseCommand(CommandTerm):
    """Movement phase command - Forward → Center → Left → Center cycle.
    
    Phase 0: Forward (+X from center)
    Phase 1: Return to center
    Phase 2: Left (+Y from center)
    Phase 3: Return to center
    (repeat)
    """
    
    cfg: "MovementPhaseCommandCfg"
    
    def __init__(self, cfg: "MovementPhaseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        
        # Current phase (0~3)
        self.current_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Flag: just completed a phase this step (for sparse reward)
        self.phase_just_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Counter: total completed phases (optional metric)
        self.completed_phases = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Center position recorded at episode start (world frame)
        self.center_pos = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Relative offsets for each phase (only XY, Z=0)
        d = cfg.move_distance
        self.phase_offsets = torch.tensor([
            [+d,  0.0, 0.0],   # 0: Forward (+X)
            [0.0, 0.0, 0.0],   # 1: Center
            [0.0, +d,  0.0],   # 2: Left (+Y)
            [0.0, 0.0, 0.0],   # 3: Center
        ], device=self.device)
        
        print(f"[MovementPhaseCommand] Initialized with move_distance={d:.4f}m, tolerance={cfg.tolerance:.4f}m")
        print(f"[MovementPhaseCommand] Pattern: Forward → Center → Left → Center (cycle)")
    
    @property
    def command(self) -> torch.Tensor:
        """Current phase index (for internal use)."""
        return self.current_phase.unsqueeze(-1).float()
    
    def get_current_target(self) -> torch.Tensor:
        """Get current target position in world frame for each env.
        
        Returns:
            (num_envs, 3) tensor
        """
        offsets = self.phase_offsets[self.current_phase]  # (num_envs, 3)
        target_pos = self.center_pos + offsets
        
        # Optional: force fixed height above robot base
        if self.cfg.absolute_height is not None:
            robot = self._env.scene[self.cfg.asset_name]
            base_z = robot.data.root_pos_w[:, 2]  # (num_envs,)
            target_pos[:, 2] = base_z + self.cfg.absolute_height
        
        return target_pos
    
    def _update_command(self) -> None:
        """Check if current target reached → advance phase if yes."""
        self.phase_just_completed.fill_(False)
        
        # Current EE position (world frame)
        ee_frame = self._env.scene["ee_frame"]
        ee_pos = ee_frame.data.target_pos_w[:, 0, :]  # (num_envs, 3)
        
        # Current target
        target_pos = self.get_current_target()
        
        # Distance (only XY plane for phase advance, ignore Z)
        xy_distance = torch.norm(ee_pos[:, :2] - target_pos[:, :2], dim=-1)
        reached_mask = xy_distance < self.cfg.tolerance
        
        if reached_mask.any():
            self.phase_just_completed[reached_mask] = True
            self.completed_phases[reached_mask] += 1
            # Cycle to next phase: 0→1→2→3→0...
            self.current_phase[reached_mask] = (self.current_phase[reached_mask] + 1) % 4
    
    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Reset command state when environments are reset."""
        env_ids = torch.as_tensor(env_ids, device=self.device)
        
        # Reset to phase 0 (start with forward)
        self.current_phase[env_ids] = 0
        self.phase_just_completed[env_ids] = False
        self.completed_phases[env_ids] = 0
        
        # Record current EE position as center for this episode
        ee_frame = self._env.scene["ee_frame"]
        self.center_pos[env_ids] = ee_frame.data.target_pos_w[env_ids, 0, :].clone()
    
    def _update_metrics(self) -> None:
        """Optional: can add logging or metrics here if needed."""
        pass


@configclass
class MovementPhaseCommandCfg(CommandTermCfg):
    """Configuration for the movement phase command."""
    
    class_type: type[CommandTerm] = MovementPhaseCommand
    
    # Robot asset name (for base Z if absolute_height used)
    asset_name: str = "robot"
    
    # Single direction move distance (meters), e.g. 3cm
    move_distance: float = 0.03
    
    # Optional: fix EE height relative to robot base (meters)
    # If None, target Z follows initial EE height
    absolute_height: float | None = 0.02  # 建議設 2cm，避免上下亂晃
    
    # Tolerance to consider phase reached (meters, XY only)
    tolerance: float = 0.008  # 0.8cm，比 move_distance 小一點
    
    # Required fields (not really used here)
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    debug_vis: bool = False