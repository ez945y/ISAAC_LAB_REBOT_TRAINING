# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""DAM SafetyGuard wrapper for Jetbot differential-drive commands."""

from __future__ import annotations

from pathlib import Path

import torch

from .ackermann_solver import AckermannSolver

JETBOT_PRESET = "jetbot_diff_drive"
JETBOT_BOUNDARY_CALLBACK = "jetbot_rollout_inside_safe_region"
JETBOT_COMMAND_NAMES = ["v", "omega"]
_DAM_JETBOT_REGISTERED = False


class JetbotDAMWrapper:
    """Filter Jetbot ``[v, omega]`` commands through DAM.

    The wrapper owns an :class:`AckermannSolver` for rollout and wheel conversion.
    It does not need a Jetbot URDF: this safety layer only reasons about planar
    base state and command geometry.
    """

    def __init__(
        self,
        stackfile: str,
        device: str,
        *,
        task: str = "default",
        solver: AckermannSolver | None = None,
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

        self.solver = solver or AckermannSolver()
        _register_jetbot_api(dam, self.solver)

        self._guard = dam.SafetyGuard(
            self._resolve_stackfile(stackfile),
            task=task,
            input_space="ackermann",
            solvers={"base": self.solver},
        )
        self._device = device
        self._last_decision = "PASS"
        self._last_delta = 0.0
        self._step_count = 0
        self._intervention_count = 0

    def filter(self, command: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Return the DAM-validated ``[v, omega]`` command."""
        squeeze = command.dim() == 1
        if squeeze:
            command = command.unsqueeze(0)
            state = state.unsqueeze(0)
        self._require_command(command)
        self._require_state(state)

        raw = command[0]
        obs = state[0]
        safe = torch.as_tensor(
            self._guard(raw, obs),
            dtype=raw.dtype,
            device=raw.device,
        ).reshape(-1)
        if safe.shape[0] != 2:
            raise RuntimeError(f"DAM returned {safe.shape[0]} command values; expected 2.")

        self._step_count += 1
        self._last_delta = torch.max(torch.abs(safe - raw)).item()
        self._record_results(getattr(self._guard, "last_results", None), self._last_delta)
        if self._last_decision in {"FAULT", "REJECT", "CLAMP"}:
            self._intervention_count += 1

        safe = safe.unsqueeze(0)
        return safe.squeeze(0) if squeeze else safe

    def command_to_wheels(self, command: torch.Tensor) -> torch.Tensor:
        wheels = [
            self.solver.command_to_wheels(row).to(dtype=command.dtype, device=command.device)
            for row in command.reshape(-1, 2)
        ]
        return torch.stack(wheels, dim=0)

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
    def _require_command(command: torch.Tensor) -> None:
        if command.dim() != 2 or command.shape != (1, 2):
            raise ValueError(f"JetbotDAMWrapper expected command shape (1, 2), got {tuple(command.shape)}.")

    @staticmethod
    def _require_state(state: torch.Tensor) -> None:
        if state.dim() != 2 or state.shape[0] != 1 or state.shape[1] < 3:
            raise ValueError(f"JetbotDAMWrapper expected state shape (1, >=3), got {tuple(state.shape)}.")

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


def _register_jetbot_api(dam, solver: AckermannSolver) -> None:
    """Register the Jetbot diff-drive preset, callback, and solver with DAM."""
    global _DAM_JETBOT_REGISTERED
    if _DAM_JETBOT_REGISTERED:
        return
    missing = [
        name
        for name in ("register_preset", "register_callback", "register_solver")
        if not hasattr(dam, name)
    ]
    if missing:
        raise ImportError(
            "JetbotDAMWrapper needs the DAM runtime with register_preset/register_callback/register_solver. "
            f"Missing: {', '.join(missing)}."
        )

    _register_preset_once(dam)
    _register_solver_once(dam, solver)
    _register_solver_factory_once(dam)

    @dam.register_callback(
        JETBOT_BOUNDARY_CALLBACK,
        layer="L1",
        category="execution",
        description="Rolls out a diff-drive command and checks the next state stays in the safe arena.",
        params={
            "x_min": "Minimum safe local x",
            "x_max": "Maximum safe local x",
            "y_abs_max": "Maximum safe absolute local y",
            "dt": "Rollout horizon in seconds",
        },
    )
    def jetbot_rollout_inside_safe_region(
        *,
        obs,
        action,
        x_min=-0.28,
        x_max=1.20,
        y_abs_max=0.24,
        dt=1.0 / 15.0,
    ):
        state = _as_flat(obs)
        command = _as_flat(action)
        next_state = solver.rollout(state, command, dt=dt)
        x_next = float(next_state[0])
        y_next = float(next_state[1])
        return x_min <= x_next <= x_max and abs(y_next) <= y_abs_max

    _DAM_JETBOT_REGISTERED = True


def _register_preset_once(dam) -> None:
    kwargs = {
        "joint_names": JETBOT_COMMAND_NAMES,
        "degrees_mode": False,
        "assets": {},
        "solvers": {
            "base": {
                "type": "ackermann_solver",
                "capabilities": ["base", "rollout"],
                "params": {
                    "track_width": 0.12,
                    "wheel_radius": 1.0,
                    "default_dt": 1.0 / 60.0,
                    "max_v": 1.2,
                    "max_omega": 4.0,
                },
            }
        },
        "chains": {},
    }
    try:
        dam.register_preset(JETBOT_PRESET, **kwargs)
    except TypeError:
        minimal = {
            "joint_names": JETBOT_COMMAND_NAMES,
            "degrees_mode": False,
        }
        try:
            dam.register_preset(JETBOT_PRESET, **minimal)
        except ValueError as exc:
            if "already" not in str(exc).lower() and "exists" not in str(exc).lower():
                raise
    except ValueError as exc:
        if "already" not in str(exc).lower() and "exists" not in str(exc).lower():
            raise


def _register_solver_once(dam, solver: AckermannSolver) -> None:
    try:
        dam.register_solver("jetbot_ackermann", solver, capabilities=["base", "rollout"])
    except ValueError as exc:
        if "already" not in str(exc).lower() and "exists" not in str(exc).lower():
            raise


def _register_solver_factory_once(dam) -> None:
    if not hasattr(dam, "register_solver_factory"):
        return

    def make_ackermann_solver(params):
        params = params or {}
        return AckermannSolver(
            track_width=float(params.get("track_width", 0.12)),
            wheel_radius=float(params.get("wheel_radius", 1.0)),
            default_dt=float(params.get("default_dt", 1.0 / 60.0)),
            max_v=float(params.get("max_v", 1.2)),
            max_omega=float(params.get("max_omega", 4.0)),
        )

    try:
        dam.register_solver_factory(
            "ackermann_solver",
            make_ackermann_solver,
            capabilities=["base", "rollout"],
        )
    except ValueError as exc:
        if "already" not in str(exc).lower() and "exists" not in str(exc).lower():
            raise


def _as_flat(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if hasattr(value, "detach") and hasattr(value.detach(), "cpu"):
        return value.detach().cpu().reshape(-1).tolist()
    if hasattr(value, "reshape"):
        return value.reshape(-1).tolist()
    if isinstance(value, dict):
        return [value["x"], value["y"], value["yaw"]]
    if all(hasattr(value, name) for name in ("x", "y", "yaw")):
        return [value.x, value.y, value.yaw]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise TypeError(f"Cannot interpret DAM value as a flat vector: {type(value)!r}")
