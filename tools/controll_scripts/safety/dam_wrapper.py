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

    # Joint-space control — filter joint targets before sending to sim
    safe_targets = wrapper.filter(joint_pos_des, current_joint_pos)
    robot.set_joint_position_target(safe_targets, joint_ids)

    # EE-space control — attach the live Isaac controller once, then filter poses
    wrapper.attach_isaac_controller(robot, controller, robot_config)
    safe_targets = wrapper.filter_ee(target_pose, current_joint_pos)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .isaac_resolver import IsaacControllerKinematicsResolver, isaac_wxyz_to_dam_xyzw

if TYPE_CHECKING:
    from ..configs.base import BaseRobotConfig
    from ..controllers.base import BaseController
    from isaaclab.assets import Articulation


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

        stackfile = self._resolve_stackfile(stackfile)

        # Build joint_names: arm joints + gripper (matches DAM preset order)
        joint_names = list(robot_config.arm_joint_names) + [robot_config.gripper_joint_name]

        self._dam = dam
        self._stackfile = stackfile
        self._task = task
        self._joint_names = joint_names
        self._guard = dam.SafetyGuard(
            stackfile,
            task=task,
            joint_names=joint_names,
            degrees_mode=False,  # Isaac Sim uses radians
        )
        self._ee_guard = None
        self._ee_resolver: IsaacControllerKinematicsResolver | None = None
        self._device = device
        self._n_arm = len(robot_config.arm_joint_names)
        self._last_clamped = False
        self._last_decision = "PASS"
        self._step_count = 0
        self._clamp_count = 0
        self._last_safe_gripper = 0.0

    def attach_isaac_controller(
        self,
        robot: "Articulation",
        controller: "BaseController",
        robot_config: "BaseRobotConfig",
        *,
        urdf_path: str | None = None,
        ee_frame_name: str | None = None,
    ) -> None:
        """Enable EE-space filtering with the live Isaac robot/controller state."""
        self._ee_resolver = IsaacControllerKinematicsResolver(
            robot=robot,
            controller=controller,
            robot_config=robot_config,
            device=self._device,
            urdf_path=urdf_path,
            ee_frame_name=ee_frame_name,
        )
        self._ee_guard = self._dam.SafetyGuard(
            self._stackfile,
            task=self._task,
            joint_names=self._joint_names,
            degrees_mode=False,
            input_space="ee",
            kinematics_resolver=self._ee_resolver,
        )

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
        self._require_single_env(action, obs)
        self._require_width(action, self._n_arm, "action")
        self._require_width(obs, self._n_arm, "obs")

        # Append gripper to form full joint vector
        g_act = self._to_scalar(gripper_action, 0.0)
        g_obs = self._to_scalar(gripper_obs, 0.0)

        full_action = torch.cat([
            action[0],
            torch.tensor([g_act], dtype=action.dtype, device=action.device),
        ])
        full_obs = torch.cat([
            obs[0],
            torch.tensor([g_obs], dtype=obs.dtype, device=obs.device),
        ])

        # DAM accepts torch.Tensor directly (preserves device/dtype)
        safe_full = self._guard(full_action, full_obs)
        self._last_safe_gripper = float(safe_full[self._n_arm].item())

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

    def filter_ee(
        self,
        target_pose: torch.Tensor,
        obs: torch.Tensor,
        gripper_action: float | torch.Tensor | None = None,
        gripper_obs: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Filter an Isaac EE pose target and return safe arm joint targets.

        Args:
            target_pose: Isaac pose [x,y,z,qw,qx,qy,qz], shape (1, 7) or (7,).
            obs: Current arm joint positions, shape (1, N_arm) or (N_arm,).
            gripper_action: Optional gripper target. EE IK preserves it as the
                gripper joint target while DAM validates the full joint vector.
            gripper_obs: Optional current gripper position.
        """
        if self._ee_guard is None or self._ee_resolver is None:
            raise RuntimeError("Call attach_isaac_controller() before filter_ee().")

        squeeze = target_pose.dim() == 1
        if squeeze:
            target_pose = target_pose.unsqueeze(0)
            obs = obs.unsqueeze(0)
        self._require_single_env(target_pose, obs)
        self._require_width(target_pose, 7, "target_pose")
        self._require_width(obs, self._n_arm, "obs")

        g_obs = self._to_scalar(gripper_obs, 0.0)
        g_act = self._to_scalar(gripper_action, g_obs)
        self._ee_resolver.set_gripper_target(g_act)
        full_obs = torch.cat([
            obs[0],
            torch.tensor([g_obs], dtype=obs.dtype, device=obs.device),
        ])

        dam_target_pose = torch.as_tensor(
            isaac_wxyz_to_dam_xyzw(target_pose[0]),
            dtype=target_pose.dtype,
            device=target_pose.device,
        )
        self._ee_guard.set_ee_pose(self._ee_resolver.current_ee_pose_dam)
        _ = self._ee_guard(dam_target_pose, full_obs)

        if self._ee_resolver.last_safe_joint_positions is None:
            raise RuntimeError("DAM EE guard did not produce validated joint positions")

        safe_full = torch.as_tensor(
            self._ee_resolver.last_safe_joint_positions,
            dtype=obs.dtype,
            device=obs.device,
        )
        self._last_safe_gripper = float(safe_full[self._n_arm].item())
        safe_arm = safe_full[: self._n_arm].unsqueeze(0)

        self._step_count += 1
        results = self._ee_guard.last_results
        self._last_clamped = any(r.decision.name == "CLAMP" for r in results)
        if self._last_clamped:
            self._clamp_count += 1
            self._last_decision = "CLAMP"
        elif any(r.decision.name == "REJECT" for r in results):
            self._last_decision = "REJECT"
        else:
            self._last_decision = "PASS"

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

    @property
    def last_safe_gripper(self) -> float:
        """Validated gripper joint target from the most recent filter call."""
        return self._last_safe_gripper

    def close(self) -> None:
        """Stop MCAP recording if active."""
        self._guard.close()
        if self._ee_guard is not None:
            self._ee_guard.close()

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _resolve_stackfile(stackfile: str) -> str:
        path = Path(stackfile)
        if path.is_absolute():
            return str(path)
        if path.exists():
            return str(path)

        safety_dir = Path(__file__).resolve().parent
        bundled_path = safety_dir / path
        if bundled_path.exists():
            return str(bundled_path)

        bundled_name = safety_dir / path.name
        if bundled_name.exists():
            return str(bundled_name)

        return str(bundled_path)

    @staticmethod
    def _to_scalar(val: float | torch.Tensor | None, default: float) -> float:
        if val is None:
            return default
        if isinstance(val, torch.Tensor):
            return val.item()
        return float(val)

    @staticmethod
    def _require_single_env(action: torch.Tensor, obs: torch.Tensor) -> None:
        if action.dim() != 2 or obs.dim() != 2:
            raise ValueError(
                "DAMSafetyWrapper expects tensors shaped (N,) or (1, N); "
                f"got action {tuple(action.shape)} and obs {tuple(obs.shape)}"
            )
        if action.shape[0] != 1 or obs.shape[0] != 1:
            raise ValueError(
                "DAMSafetyWrapper currently supports one Isaac environment at a time. "
                f"Got action batch {action.shape[0]} and obs batch {obs.shape[0]}."
            )

    @staticmethod
    def _require_width(tensor: torch.Tensor, expected: int, name: str) -> None:
        if tensor.shape[1] != expected:
            raise ValueError(
                f"DAMSafetyWrapper expected {name} width {expected}, "
                f"got shape {tuple(tensor.shape)}."
            )
