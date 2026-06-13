# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""DAM SafetyGuard wrapper for Jetbot wheel-velocity commands."""

from __future__ import annotations

from pathlib import Path

import torch


class JetbotDAMWrapper:
    """Filter Jetbot wheel velocity targets through DAM.

    The action space is ``[left_wheel_velocity, right_wheel_velocity]`` in
    radians/second. The wrapper is intentionally thin: all action changes come
    from ``dam.SafetyGuard`` rather than demo-side obstacle heuristics.
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

        self._device = device
        self._joint_names = joint_names or ["left_wheel", "right_wheel"]
        self._guard = dam.SafetyGuard(
            self._resolve_stackfile(stackfile),
            task=task,
            joint_names=self._joint_names,
            degrees_mode=False,
        )
        self._last_decision = "PASS"
        self._last_delta = 0.0
        self._step_count = 0
        self._intervention_count = 0

    def filter(self, action: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Return the DAM-validated wheel velocity target."""
        squeeze = action.dim() == 1
        if squeeze:
            action = action.unsqueeze(0)
            obs = obs.unsqueeze(0)
        self._require_shape(action, "action")
        self._require_shape(obs, "obs")

        raw = action[0]
        current = obs[0]
        safe = torch.as_tensor(
            self._guard(raw, current),
            dtype=raw.dtype,
            device=raw.device,
        ).reshape(-1)
        if safe.shape[0] != 2:
            raise RuntimeError(f"DAM returned {safe.shape[0]} wheel targets; expected 2.")

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
