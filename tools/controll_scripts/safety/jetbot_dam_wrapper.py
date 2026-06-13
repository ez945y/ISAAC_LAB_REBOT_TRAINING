# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""DAM SafetyGuard wrapper for Jetbot 2D position targets."""

from __future__ import annotations

from pathlib import Path

import torch

JETBOT_TARGET_PRESET = "jetbot_target_2d"
JETBOT_BOUNDARY_CALLBACK = "jetbot_target_inside_safe_region"
JETBOT_TARGET_NAMES = ["x_target", "y_target"]
_DAM_JETBOT_REGISTERED = False


class JetbotDAMWrapper:
    """Filter Jetbot 2D target positions through DAM.

    The action space is ``[x_target, y_target]`` in world meters. The demo then
    converts the validated target into wheel velocities. Keeping the action as a
    position target makes the visual story concrete: DAM clamps commands away
    from the forbidden left, right, and bottom boundary bands.
    """

    def __init__(
        self,
        stackfile: str,
        device: str,
        *,
        task: str = "default",
        joint_names: list[str] | None = None,
    ) -> None:
        try:
            import dam
        except ImportError as exc:
            raise ImportError(
                "JetbotDAMWrapper requires robot-dam (import name: dam). "
                "Install it before running the Jetbot DAM demo."
            ) from exc
        if not hasattr(dam, "SafetyGuard"):
            raise ImportError(
                "Imported a 'dam' package, but it does not expose SafetyGuard. "
                "Install the robot-dam package from https://github.com/ez945y/DAM."
            )

        _register_jetbot_target_api(dam)

        self._guard = dam.SafetyGuard(
            self._resolve_stackfile(stackfile),
            task=task,
            joint_names=joint_names or JETBOT_TARGET_NAMES,
            degrees_mode=False,
        )
        self._device = device
        self._last_decision = "PASS"
        self._last_delta = 0.0
        self._step_count = 0
        self._intervention_count = 0

    def filter(self, target_xy: torch.Tensor, current_xy: torch.Tensor) -> torch.Tensor:
        """Return the DAM-validated 2D target."""
        squeeze = target_xy.dim() == 1
        if squeeze:
            target_xy = target_xy.unsqueeze(0)
            current_xy = current_xy.unsqueeze(0)
        self._require_shape(target_xy, "target_xy")
        self._require_shape(current_xy, "current_xy")

        raw = target_xy[0]
        obs = current_xy[0]
        safe = torch.as_tensor(
            self._guard(raw, obs),
            dtype=raw.dtype,
            device=raw.device,
        ).reshape(-1)
        if safe.shape[0] != 2:
            raise RuntimeError(f"DAM returned {safe.shape[0]} target values; expected 2.")

        self._step_count += 1
        self._last_delta = torch.max(torch.abs(safe - raw)).item()
        self._record_results(getattr(self._guard, "last_results", None), self._last_delta)
        if self._last_decision in {"FAULT", "REJECT", "CLAMP"}:
            self._intervention_count += 1

        safe = safe.unsqueeze(0)
        return safe.squeeze(0) if squeeze else safe

    @property
    def last_decision(self) -> str:
        return self._last_decision

    @property
    def last_delta(self) -> float:
        return self._last_delta

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def intervention_rate(self) -> float:
        return self._intervention_count / self._step_count if self._step_count else 0.0

    def close(self) -> None:
        self._guard.close()

    @staticmethod
    def _resolve_stackfile(stackfile: str) -> str:
        path = Path(stackfile)
        if path.is_absolute() or path.exists():
            return str(path)
        safety_dir = Path(__file__).resolve().parent
        bundled = safety_dir / path
        if bundled.exists():
            return str(bundled)
        bundled_name = safety_dir / path.name
        if bundled_name.exists():
            return str(bundled_name)
        return str(bundled)

    @staticmethod
    def _require_shape(tensor: torch.Tensor, name: str) -> None:
        if tensor.dim() != 2 or tensor.shape != (1, 2):
            raise ValueError(f"JetbotDAMWrapper expected {name} shape (1, 2), got {tuple(tensor.shape)}.")

    def _record_results(self, results, delta: float) -> None:
        decision_names = []
        if results is not None:
            decision_names = [
                getattr(getattr(result, "decision", None), "name", None)
                for result in results
            ]
            decision_names = [name for name in decision_names if name]
        for decision in ("FAULT", "REJECT", "CLAMP"):
            if decision in decision_names:
                self._last_decision = decision
                return
        self._last_decision = "CLAMP" if delta > 1e-5 else "PASS"


def _register_jetbot_target_api(dam) -> None:
    """Register the Jetbot target preset and boundary callback with DAM."""
    global _DAM_JETBOT_REGISTERED
    if _DAM_JETBOT_REGISTERED:
        return
    missing = [name for name in ("register_preset", "register_callback") if not hasattr(dam, name)]
    if missing:
        raise ImportError(
            "JetbotDAMWrapper needs the DAM runtime with register_preset/register_callback. "
            f"Missing: {', '.join(missing)}."
        )

    # This is a semantic action preset, not a robot-arm hardware preset.
    _register_preset_once(dam)

    @dam.register_callback(JETBOT_BOUNDARY_CALLBACK, layer="L1", category="execution")
    def jetbot_target_inside_safe_region(
        *,
        obs,
        action,
        x_min=-0.28,
        x_max=1.20,
        y_abs_max=0.24,
    ):
        target = _as_flat(action)
        x_target = float(target[0])
        y_target = float(target[1])
        return x_min <= x_target <= x_max and abs(y_target) <= y_abs_max

    _DAM_JETBOT_REGISTERED = True


def _as_flat(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if hasattr(value, "detach") and hasattr(value.detach(), "cpu"):
        return value.detach().cpu().reshape(-1).tolist()
    if hasattr(value, "reshape"):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    raise TypeError(f"Cannot interpret DAM action/obs value as a flat vector: {type(value)!r}")


def _register_preset_once(dam) -> None:
    try:
        dam.register_preset(
            JETBOT_TARGET_PRESET,
            joint_names=JETBOT_TARGET_NAMES,
            degrees_mode=False,
            urdf_path=None,
        )
    except TypeError:
        try:
            dam.register_preset(
                JETBOT_TARGET_PRESET,
                joint_names=JETBOT_TARGET_NAMES,
                degrees_mode=False,
            )
        except ValueError as exc:
            if "already" not in str(exc).lower() and "exists" not in str(exc).lower():
                raise
    except ValueError as exc:
        if "already" not in str(exc).lower() and "exists" not in str(exc).lower():
            raise
