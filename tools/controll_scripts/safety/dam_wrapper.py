# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""DAM SafetyGuard wrapper for Isaac Lab control loops.

Bridges the local DAM SafetyGuard with Isaac Lab's torch.Tensor-based control
pipeline. Handles joint ordering, unit conversion, and per-step guard result
tracking.

Usage:
    from controll_scripts.safety import DAMSafetyWrapper

    wrapper = DAMSafetyWrapper(
        stackfile="controll_scripts/safety/soarm_isaac_safety.yaml",
        robot_config=robot_config,
        device=sim.device,
    )

    # In control loop — filter joint targets before sending to sim
    safe_targets = wrapper.filter(joint_pos_des, current_joint_pos)
    robot.set_joint_position_target(safe_targets, joint_ids)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..configs.base import BaseRobotConfig


class DAMSafetyWrapper:
    """Wraps DAM SafetyGuard for Isaac Lab controllers.

    Accepts and returns torch.Tensor on the simulation device.
    Tracks guard results per step for visualization and logging.
    """

    def __init__(
        self,
        stackfile: str,
        robot_config: "BaseRobotConfig",
        device: str,
        *,
        task: str = "default",
    ) -> None:
        try:
            import dam
        except ImportError as exc:
            raise ImportError(
                "DAMSafetyWrapper requires the local DAM package that exposes "
                "dam.SafetyGuard. Install that package or add it to PYTHONPATH "
                "before running the Isaac demos."
            ) from exc
        if not hasattr(dam, "SafetyGuard"):
            raise ImportError(
                "Imported a 'dam' package, but it does not expose SafetyGuard. "
                "Check that the local DAM package is installed instead of the "
                "unrelated PyPI 'dam' package."
            )

        # Resolve stackfile path relative to this file's directory
        if not os.path.isabs(stackfile):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            stackfile = os.path.join(base_dir, stackfile)

        # Build joint_names: arm joints + gripper (matches DAM preset order)
        joint_names = list(robot_config.arm_joint_names) + [robot_config.gripper_joint_name]

        self._guard = dam.SafetyGuard(
            stackfile,
            task=task,
            joint_names=joint_names,
            degrees_mode=False,  # Isaac Sim uses radians
        )
        self._device = device
        self._n_arm = len(robot_config.arm_joint_names)
        self._last_clamped = False
        self._last_decision = "PASS"
        self._step_count = 0
        self._clamp_count = 0

    def filter(
        self,
        action: torch.Tensor,
        obs: torch.Tensor,
        gripper_action: float | torch.Tensor | None = None,
        gripper_obs: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Filter joint position targets through DAM safety guard.

        Args:
            action: Target arm joint positions, shape (1, N_arm) or (N_arm,).
                    Radians, on sim device.
            obs: Current arm joint positions, same shape. Radians.
            gripper_action: Optional gripper target (scalar, radians).
                           If None, appends 0.0.
            gripper_obs: Optional current gripper position (scalar, radians).
                        If None, appends 0.0.

        Returns:
            Safe arm joint targets, same shape as input action.
            (Gripper is passed through DAM but only arm joints are returned.)
        """
        squeeze = action.dim() == 1
        if squeeze:
            action = action.unsqueeze(0)
            obs = obs.unsqueeze(0)

        # Append gripper to form full joint vector
        g_act = self._to_scalar(gripper_action, 0.0)
        g_obs = self._to_scalar(gripper_obs, 0.0)

        full_action = torch.cat([
            action[0],
            torch.tensor([g_act], device=self._device),
        ])
        full_obs = torch.cat([
            obs[0],
            torch.tensor([g_obs], device=self._device),
        ])

        # DAM accepts torch.Tensor directly (preserves device/dtype)
        safe_full = self._guard(full_action, full_obs)

        # Track results
        self._step_count += 1
        results = self._guard.last_results
        self._last_clamped = any(
            r.decision.name == "CLAMP" for r in results
        )
        if self._last_clamped:
            self._clamp_count += 1
            self._last_decision = "CLAMP"
        elif any(r.decision.name == "REJECT" for r in results):
            self._last_decision = "REJECT"
        else:
            self._last_decision = "PASS"

        # Return only arm joints (drop gripper)
        safe_arm = safe_full[: self._n_arm].unsqueeze(0)
        if squeeze:
            safe_arm = safe_arm.squeeze(0)
        return safe_arm

    # -- Properties -----------------------------------------------------------

    @property
    def last_clamped(self) -> bool:
        """True if the last action was clamped."""
        return self._last_clamped

    @property
    def last_decision(self) -> str:
        """Most restrictive decision from last call: PASS / CLAMP / REJECT."""
        return self._last_decision

    @property
    def clamp_rate(self) -> float:
        """Fraction of steps that were clamped so far."""
        if self._step_count == 0:
            return 0.0
        return self._clamp_count / self._step_count

    @property
    def step_count(self) -> int:
        """Total number of filter() calls."""
        return self._step_count

    def close(self) -> None:
        """Stop MCAP recording if active."""
        self._guard.close()

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _to_scalar(val: float | torch.Tensor | None, default: float) -> float:
        if val is None:
            return default
        if isinstance(val, torch.Tensor):
            return val.item()
        return float(val)
