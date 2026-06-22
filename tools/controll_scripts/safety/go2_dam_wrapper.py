# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""DAM SafetyGuard wrapper for the Go2 squad: inter-dog distance band.

Each dog is a holonomic planar base; its command is ``[vx, vy, omega]`` (body
frame). The single L1 boundary keeps a dog's NEAREST-neighbour distance inside a
band ``[min_dist, max_dist]``:

- ``nearest < min_dist`` -> too close (collision risk) -> push apart
- ``nearest > max_dist`` -> drifting out of the squad   -> pull back in

Because ``nearest = min`` over all neighbours, ``nearest >= min_dist`` already
implies *every* pair is clear, so the single nearest-distance band enforces both
"don't collide" and "don't break formation" at once.

Full obs/action vector is 6D ``[x, y, yaw, vx, vy, omega]`` (pose [0:3] +
command [3:6]). Neighbour positions change every call, so they are injected into
the guard through the ``solvers`` mapping (DAM 0.6.0 style) via a small mutable
holder — the boundary callback reads them alongside the rollout solver.

The correction is a small QP on ``[vx, vy, omega]``: linearise the distance to the
offending neighbour through the differentiable :class:`HolonomicSolver` rollout,
then OSQP for the minimal command change that puts the predicted distance back in
band. Yaw change is penalised more than linear, so the guard prefers a sidestep
(the whole point of using a holonomic model over a unicycle).
"""

from __future__ import annotations

import math
from pathlib import Path

import dam
import numpy as np
import osqp
import torch
from scipy import sparse

from .holonomic_solver import HolonomicSolver

# Shared with the stackfile (``callback: go2_min_max_separation``) and the
# solver/neighbour injection keys.
_SOLVER_KEY = "holonomic"
_NEIGHBOR_KEY = "neighbors"


class _NeighborHolder:
    """Mutable carrier so per-call neighbour positions reach the boundary callback
    through the guard's ``solvers`` injection (the solver itself is frozen)."""

    __slots__ = ("points",)

    def __init__(self) -> None:
        self.points: list[tuple[float, float]] = []


@dam.register_callback(
    "go2_min_max_separation",
    layer="L1",
    category="execution",
    description="Keep a dog's nearest-neighbour distance inside [min_dist, max_dist].",
    params={
        "min_dist": "Minimum allowed nearest-neighbour distance (m) — collision floor.",
        "max_dist": "Maximum allowed nearest-neighbour distance (m) — cohesion ceiling.",
        "dt": "Rollout horizon in seconds.",
    },
)
def go2_min_max_separation(
    *,
    obs,
    action,
    solvers,
    min_dist: float = 0.8,
    max_dist: float = 4.0,
    dt: float = 0.2,
    **_kwargs,
):
    """L1 boundary: predicted nearest-neighbour distance must stay in the band."""
    solver = solvers[_SOLVER_KEY]
    holder = solvers.get(_NEIGHBOR_KEY)
    neighbors = list(getattr(holder, "points", []) or [])
    if not neighbors:
        return True  # a lone dog has no separation to enforce

    # obs.joint_positions           = [x, y, yaw, 0, 0, 0]
    # action.target_joint_positions = [x, y, yaw, vx, vy, omega]
    state = list(obs.joint_positions[:3])
    command = list(action.target_joint_positions[3:6])

    nxt = solver.rollout(state, command, dt=dt)
    nx, ny = float(nxt[0]), float(nxt[1])
    dists = [math.hypot(nx - px, ny - py) for px, py in neighbors]
    j = min(range(len(dists)), key=lambda k: dists[k])
    nearest = dists[j]
    if min_dist <= nearest <= max_dist:
        return True  # already in band — pass the command through unchanged

    # --- correction: linearise distance to the offending neighbour, then QP ---
    px, py = neighbors[j]
    state_t = torch.tensor(state, dtype=torch.float32)
    cmd_t = torch.tensor(command, dtype=torch.float32, requires_grad=True)
    nxt_t = solver.rollout(state_t, cmd_t, dt=dt)
    dist_t = torch.sqrt((nxt_t[0] - px) ** 2 + (nxt_t[1] - py) ** 2 + 1e-9)
    d_val = float(dist_t.item())
    dist_t.backward()
    grad = cmd_t.grad.detach().numpy()  # d(dist)/d[vx, vy, omega]

    vx, vy, om = command
    # Soft (CBF-style) constraint with a slack variable so the QP is ALWAYS feasible:
    # reaching the band in one dt is usually impossible (a dog only moves ~v*dt), so a
    # hard constraint would be infeasible and leave the command uncorrected. The slack
    # absorbs the unreachable part and, being heavily penalised, drives the command to
    # the actuator limit in the band-improving direction (max sidestep/approach).
    # Variables: [du_vx, du_vy, du_omega, slack].
    slack_w = 1.0e3
    A_rows = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]]  # slack >= 0
    lo = [-solver.max_v - vx, -solver.max_v - vy, -solver.max_omega - om, 0.0]
    hi = [solver.max_v - vx, solver.max_v - vy, solver.max_omega - om, np.inf]
    if nearest < min_dist:
        # push apart: grad.du + slack >= min_dist - d_val  (distance must GROW)
        A_rows.append([grad[0], grad[1], grad[2], 1.0])
        lo.append(min_dist - d_val)
        hi.append(np.inf)
    else:
        # pull in: grad.du - slack <= max_dist - d_val  (distance must SHRINK)
        A_rows.append([grad[0], grad[1], grad[2], -1.0])
        lo.append(-np.inf)
        hi.append(max_dist - d_val)

    P = sparse.csc_matrix(np.diag([1.0, 1.0, 3.0, slack_w]))  # penalise yaw -> prefer sidestep
    q = np.zeros(4)
    A = sparse.csc_matrix(np.array(A_rows, dtype=float))
    prob = osqp.OSQP()
    prob.setup(P, q, A, np.array(lo), np.array(hi), verbose=False, eps_abs=1e-5, eps_rel=1e-5)
    res = prob.solve()

    if res.info.status == "solved":
        action.target_joint_positions[3] = float(vx + res.x[0])
        action.target_joint_positions[4] = float(vy + res.x[1])
        action.target_joint_positions[5] = float(om + res.x[2])
    elif nearest < min_dist:
        # Degenerate solver failure on the collision case must fail safe: stop.
        action.target_joint_positions[3] = 0.0
        action.target_joint_positions[4] = 0.0
        action.target_joint_positions[5] = 0.0
    return True


class Go2DAMWrapper:
    """Filter a Go2 ``[vx, vy, omega]`` command through the inter-dog distance guard."""

    def __init__(
        self,
        stackfile: str,
        device: str = "cpu",
        *,
        task: str = "default",
        solver: HolonomicSolver | None = None,
    ) -> None:
        self.solver = solver or HolonomicSolver()
        self._neighbors = _NeighborHolder()
        self._guard = dam.SafetyGuard(
            self._resolve_stackfile(stackfile),
            task=task,
            degrees_mode=False,  # our 6D "joints" are pose + body velocity, not motor degrees
            solvers={_SOLVER_KEY: self.solver, _NEIGHBOR_KEY: self._neighbors},
        )
        self._device = device
        self._last_decision = "PASS"
        self._last_delta = 0.0
        self._step_count = 0
        self._intervention_count = 0

    def filter(
        self,
        command: torch.Tensor,
        state: torch.Tensor,
        neighbors,
    ) -> torch.Tensor:
        """Return the DAM-validated ``[vx, vy, omega]`` command.

        Args:
            command:   ``(1, 3)`` tensor — ``[vx, vy, omega]``.
            state:     ``(1, >=3)`` tensor — ``[x, y, yaw, ...]``.
            neighbors: iterable of ``(x, y)`` positions of the OTHER dogs.

        Returns:
            ``(1, 3)`` tensor — safe ``[vx, vy, omega]``.
        """
        squeeze = command.dim() == 1
        if squeeze:
            command = command.unsqueeze(0)
            state = state.unsqueeze(0)
        self._require_command(command)
        self._require_state(state)

        self._neighbors.points = [(float(p[0]), float(p[1])) for p in neighbors]

        raw = command[0]       # [vx, vy, omega]
        pose = state[0, :3]    # [x, y, yaw]
        zeros = torch.zeros(3, dtype=pose.dtype, device=pose.device)
        obs_6 = torch.cat([pose, zeros])
        action_6 = torch.cat([pose, raw])

        safe_6 = torch.as_tensor(
            self._guard(action_6, obs_6), dtype=raw.dtype, device=raw.device
        ).reshape(-1)
        safe = safe_6[3:6]

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
        for cand in (safety_dir / path, safety_dir / path.name):
            if cand.exists():
                return str(cand)
        return str(safety_dir / path)

    @staticmethod
    def _require_command(command: torch.Tensor) -> None:
        if command.dim() != 2 or command.shape != (1, 3):
            raise ValueError(
                f"Go2DAMWrapper expected command shape (1, 3), got {tuple(command.shape)}."
            )

    @staticmethod
    def _require_state(state: torch.Tensor) -> None:
        if state.dim() != 2 or state.shape[0] != 1 or state.shape[1] < 3:
            raise ValueError(
                f"Go2DAMWrapper expected state shape (1, >=3), got {tuple(state.shape)}."
            )

    def _record_results(self, results, delta: float) -> None:
        decision_names = []
        if results is not None:
            decision_names = [
                getattr(getattr(result, "decision", None), "name", None) for result in results
            ]
            decision_names = [name for name in decision_names if name]
        for decision in ("FAULT", "REJECT", "CLAMP"):
            if decision in decision_names:
                self._last_decision = decision
                return
        self._last_decision = "CLAMP" if delta > 1e-5 else "PASS"
