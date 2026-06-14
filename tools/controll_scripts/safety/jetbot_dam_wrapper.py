# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""DAM SafetyGuard wrapper for Jetbot differential-drive commands.

The full observation/action vector is 5D: ``[x, y, yaw, v, omega]``.

- **pose** ``[x, y, yaw]`` — current planar pose (indices 0-2)
- **command** ``[v, omega]``  — velocity command (indices 3-4)

The wrapper constructs:

    obs    = [x, y, yaw, 0, 0]              # current pose, zero velocity
    action = [x, y, yaw, v_cmd, omega_cmd]   # echo pose + requested velocity

On REJECT the SafetyGuard fallback returns ``obs.joint_positions``, which has
``v=0, omega=0`` — a natural hold-position / full stop for a velocity-controlled
robot.

The preset (``jetbot_diff_drive``) and solver factory (``ackermann_solver``)
are defined in the DAM project and referenced by the stackfile.  This wrapper
only registers application-specific **callbacks** that are not part of DAM
core.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import osqp
import torch
from scipy import sparse

from .ackermann_solver import AckermannSolver

_CALLBACKS_REGISTERED = False


def _register_jetbot_api(solver: AckermannSolver) -> None:
    """Register solver factory + boundary callbacks referenced by the stackfile.

    The stackfile references these by name (``type: ackermann_solver``,
    ``callback: jetbot_rollout_inside_safe_region``).  The actual
    implementations must be registered via the API before SafetyGuard
    loads the stackfile.
    """
    global _CALLBACKS_REGISTERED
    if _CALLBACKS_REGISTERED:
        return

    import dam

    # -- Solver factory (preset references type: ackermann_solver) -----------

    @dam.register_solver_factory("ackermann_solver", capabilities=["base", "rollout"], replace=True)
    def make_ackermann_solver(params):
        params = params or {}
        return AckermannSolver(
            track_width=float(params.get("track_width", 0.12)),
            wheel_radius=float(params.get("wheel_radius", 1.0)),
            default_dt=float(params.get("default_dt", 1.0 / 60.0)),
            max_v=float(params.get("max_v", 1.8)),
            max_omega=float(params.get("max_omega", 8.0)),
        )

    # -- Boundary callback ---------------------------------------------------

    @dam.register_callback(
        "jetbot_rollout_inside_safe_region",
        layer="L1",
        category="execution",
        description="Rolls out a diff-drive command and checks the predicted state stays in the safe arena.",
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
        # obs.joint_positions            = [x, y, yaw, 0, 0]
        # action.target_joint_positions  = [x, y, yaw, v, omega]
        state = list(obs.joint_positions[:3])               # [x, y, yaw]
        command = list(action.target_joint_positions[3:5])   # [v, omega]

        # 1. Check if the proposed command is already safe
        next_state = solver.rollout(state, command, dt=dt)
        x_next = float(next_state[0])
        y_next = float(next_state[1])
        if x_min <= x_next <= x_max and abs(y_next) <= y_abs_max:
            return True

        # 2. If unsafe, solve QP to correct it
        # Clamp command to actuator limits to avoid zero gradients during linearization
        v_clamped = max(-solver.max_v, min(solver.max_v, command[0]))
        omega_clamped = max(-solver.max_omega, min(solver.max_omega, command[1]))

        state_t = torch.tensor(state, dtype=torch.float32)
        cmd_t = torch.tensor([v_clamped, omega_clamped], dtype=torch.float32, requires_grad=True)

        # Rollout & compute gradients via Autograd
        next_state_t = solver.rollout(state_t, cmd_t, dt=dt)
        x_next_val = next_state_t[0].item()
        y_next_val = next_state_t[1].item()

        next_state_t[0].backward(retain_graph=True)
        grad_x = cmd_t.grad.clone().numpy()

        cmd_t.grad.zero_()
        next_state_t[1].backward()
        grad_y = cmd_t.grad.clone().numpy()

        # Set up OSQP
        v_cmd, omega_cmd = command[0], command[1]
        v_max, omega_max = solver.max_v, solver.max_omega

        # Constraint: l <= A * delta_u <= u
        A_np = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [grad_x[0], grad_x[1]],
            [grad_y[0], grad_y[1]]
        ])
        A_csc = sparse.csc_matrix(A_np)

        l_np = np.array([
            -v_max - v_cmd,
            -omega_max - omega_cmd,
            x_min - x_next_val,
            -y_abs_max - y_next_val
        ])
        u_np = np.array([
            v_max - v_cmd,
            omega_max - omega_cmd,
            x_max - x_next_val,
            y_abs_max - y_next_val
        ])

        P_csc = sparse.csc_matrix(np.diag([1.0, 0.05])) # Prioritize keeping the turning command (smaller penalty on delta_omega)
        q_np = np.zeros(2)

        prob = osqp.OSQP()
        prob.setup(P_csc, q_np, A_csc, l_np, u_np, verbose=False, eps_abs=1e-5, eps_rel=1e-5)
        res = prob.solve()

        if res.info.status == 'solved':
            delta_v, delta_omega = res.x[0], res.x[1]
            safe_v = float(v_cmd + delta_v)
            safe_omega = float(omega_cmd + delta_omega)
        else:
            # Fallback on solver failure: stop
            safe_v = 0.0
            safe_omega = 0.0

        # Modify the proposed action in-place
        action.target_joint_positions[3] = safe_v
        action.target_joint_positions[4] = safe_omega
        return True

    _CALLBACKS_REGISTERED = True


class JetbotDAMWrapper:
    """Filter Jetbot ``[v, omega]`` commands through DAM.

    The wrapper owns an :class:`AckermannSolver` for rollout and wheel
    conversion.  It does not need a Jetbot URDF: this safety layer only
    reasons about planar base state and command geometry.
    """

    def __init__(
        self,
        stackfile: str,
        device: str,
        *,
        task: str = "default",
        solver: AckermannSolver | None = None,
    ) -> None:
        import dam

        self.solver = solver or AckermannSolver()
        _register_jetbot_api(self.solver)

        self._guard = dam.SafetyGuard(
            self._resolve_stackfile(stackfile),
            task=task,
        )
        self._device = device
        self._last_decision = "PASS"
        self._last_delta = 0.0
        self._step_count = 0
        self._intervention_count = 0

    def filter(self, command: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Return the DAM-validated ``[v, omega]`` command.

        Args:
            command: ``(1, 2)`` tensor — ``[v, omega]``.
            state:   ``(1, >=3)`` tensor — ``[x, y, yaw, ...]``.

        Returns:
            ``(1, 2)`` tensor — safe ``[v, omega]``.
        """
        squeeze = command.dim() == 1
        if squeeze:
            command = command.unsqueeze(0)
            state = state.unsqueeze(0)
        self._require_command(command)
        self._require_state(state)

        raw = command[0]       # [v, omega]
        pose = state[0, :3]    # [x, y, yaw]

        # Build 5D vectors: obs = [x,y,yaw,0,0], action = [x,y,yaw,v,omega]
        zeros = torch.zeros(2, dtype=pose.dtype, device=pose.device)
        obs_5 = torch.cat([pose, zeros])
        action_5 = torch.cat([pose, raw])

        safe_5 = torch.as_tensor(
            self._guard(action_5, obs_5),
            dtype=raw.dtype,
            device=raw.device,
        ).reshape(-1)

        safe = safe_5[3:5]     # Extract [v_safe, omega_safe]

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
            raise ValueError(
                f"JetbotDAMWrapper expected command shape (1, 2), got {tuple(command.shape)}."
            )

    @staticmethod
    def _require_state(state: torch.Tensor) -> None:
        if state.dim() != 2 or state.shape[0] != 1 or state.shape[1] < 3:
            raise ValueError(
                f"JetbotDAMWrapper expected state shape (1, >=3), got {tuple(state.shape)}."
            )

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
