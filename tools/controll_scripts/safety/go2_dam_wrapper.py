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
    """Mutable carrier so per-call neighbour positions + priorities reach the
    boundary callback through the guard's ``solvers`` injection (the solver is
    frozen). ``points`` is a list of ``(x, y, priority)``; ``self_priority`` is the
    acting dog's priority — together they set the responsibility split."""

    __slots__ = ("points", "self_priority")

    def __init__(self) -> None:
        self.points: list[tuple[float, float, float]] = []
        self.self_priority: float = 1.0


@dam.register_callback(
    "go2_min_max_separation",
    layer="L1",
    category="execution",
    description="Priority-weighted inter-dog distance band: each dog keeps every "
    "neighbour >= min_dist (collision, share scaled by priority) and its nearest "
    "<= max_dist (cohesion). Yielding emerges from the responsibility split.",
    params={
        "min_dist": "Collision floor (m): predicted distance to any neighbour stays above this.",
        "max_dist": "Cohesion ceiling (m): the nearest neighbour stays within this.",
        "dt": "Rollout horizon in seconds.",
        "gamma": "Discrete CBF rate in (0,1] — fraction of the margin a dog may close per step.",
        "influence": "Only neighbours within min_dist+influence (m) raise a collision constraint.",
        "lam_min": "Minimum avoidance responsibility a dog keeps even at top priority (safety floor).",
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
    gamma: float = 0.5,
    influence: float = 2.0,
    lam_min: float = 0.1,
    **_kwargs,
):
    """L1 boundary with priority-weighted responsibility (emergent yielding).

    For each near neighbour the discrete CBF asks the dog to keep its predicted
    distance from closing faster than ``gamma`` allows; the *amount* it must
    correct is scaled by its responsibility ``lambda = p_j / (p_self + p_j)``
    (floored at ``lam_min``). A high-priority dog gets ``lambda -> 0`` so its QP
    barely deviates (keeps course), while the low-priority dog gets ``lambda -> 1``
    and does the avoiding — so it *looks* like it yields, with no yield() logic.
    Cohesion (nearest > max_dist) is symmetric (no priority). One slack per
    constraint keeps the QP always feasible; yaw is penalised so it prefers a
    strafe.
    """
    solver = solvers[_SOLVER_KEY]
    holder = solvers.get(_NEIGHBOR_KEY)
    neighbors = list(getattr(holder, "points", []) or [])  # each (x, y, priority)
    if not neighbors:
        return True
    p_self = float(getattr(holder, "self_priority", 1.0))

    state = list(obs.joint_positions[:3])              # [x, y, yaw]
    command = list(action.target_joint_positions[3:6])  # [vx, vy, omega]
    vx, vy, om = command
    cx, cy = state[0], state[1]

    nxt0 = solver.rollout(state, command, dt=dt)
    nx0, ny0 = float(nxt0[0]), float(nxt0[1])
    state_t = torch.tensor(state, dtype=torch.float32)
    cmd_t = torch.tensor(command, dtype=torch.float32, requires_grad=True)
    nxt_t = solver.rollout(state_t, cmd_t, dt=dt)  # one graph, reused per neighbour

    def _grad_to(px: float, py: float):
        dist_t = torch.sqrt((nxt_t[0] - px) ** 2 + (nxt_t[1] - py) ** 2 + 1e-9)
        return torch.autograd.grad(dist_t, cmd_t, retain_graph=True)[0].detach().numpy()

    # Each entry: (grad(3,), rhs, sign)  with constraint  grad.du + sign*slack {>= or <=} rhs.
    # sign=+1 / lower-bound rhs => "push apart"; sign=-1 / upper-bound rhs => "pull in".
    push: list = []
    pull: list = []

    # --- collision: priority-weighted, per near neighbour ---
    for px, py, p_j in neighbors:
        dist_now = math.hypot(cx - px, cy - py)
        if dist_now - min_dist > influence:
            continue
        d_pred = math.hypot(nx0 - px, ny0 - py)
        # discrete CBF: dist(next) >= dist_now - gamma*(dist_now - min_dist)
        required = (dist_now - gamma * (dist_now - min_dist)) - d_pred
        if required <= 1e-4:
            continue  # not closing faster than allowed -> no action needed
        lam = max(lam_min, p_j / (p_self + p_j))
        push.append((_grad_to(px, py), lam * required))

    # --- cohesion: nearest neighbour, symmetric (no priority) ---
    npx, npy = min(((px, py) for px, py, _ in neighbors),
                   key=lambda p: math.hypot(cx - p[0], cy - p[1]))
    nd = math.hypot(cx - npx, cy - npy)
    if nd > max_dist:
        target = max_dist + (1.0 - gamma) * (nd - max_dist)  # allowed predicted distance
        d_pred = math.hypot(nx0 - npx, ny0 - npy)
        if d_pred > target:
            pull.append((_grad_to(npx, npy), target - d_pred))

    if not push and not pull:
        return True

    # --- assemble the slack QP: vars = [du_vx, du_vy, du_omega, s_0 ... s_{m-1}] ---
    m = len(push) + len(pull)
    n = 3 + m
    slack_w = 1.0e3
    rows, lo, hi = [], [], []

    def _row(vals):
        r = [0.0] * n
        for idx, v in vals:
            r[idx] = v
        return r

    rows += [_row([(0, 1.0)]), _row([(1, 1.0)]), _row([(2, 1.0)])]  # actuator deltas
    lo += [-solver.max_v - vx, -solver.max_v - vy, -solver.max_omega - om]
    hi += [solver.max_v - vx, solver.max_v - vy, solver.max_omega - om]
    for k in range(m):                                              # slacks >= 0
        rows.append(_row([(3 + k, 1.0)])); lo.append(0.0); hi.append(np.inf)

    k = 0
    for grad, rhs in push:   # grad.du + s >= rhs
        rows.append(_row([(0, grad[0]), (1, grad[1]), (2, grad[2]), (3 + k, 1.0)]))
        lo.append(rhs); hi.append(np.inf); k += 1
    for grad, rhs in pull:   # grad.du - s <= rhs
        rows.append(_row([(0, grad[0]), (1, grad[1]), (2, grad[2]), (3 + k, -1.0)]))
        lo.append(-np.inf); hi.append(rhs); k += 1

    P = sparse.csc_matrix(np.diag([1.0, 1.0, 3.0] + [slack_w] * m))  # penalise yaw -> sidestep
    q = np.zeros(n)
    A = sparse.csc_matrix(np.array(rows, dtype=float))
    prob = osqp.OSQP()
    prob.setup(P, q, A, np.array(lo), np.array(hi), verbose=False, eps_abs=1e-5, eps_rel=1e-5)
    res = prob.solve()

    if res.info.status == "solved":
        action.target_joint_positions[3] = float(vx + res.x[0])
        action.target_joint_positions[4] = float(vy + res.x[1])
        action.target_joint_positions[5] = float(om + res.x[2])
    elif push:
        # Degenerate solver failure with a collision constraint must fail safe: stop.
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
        *,
        self_priority: float = 1.0,
    ) -> torch.Tensor:
        """Return the DAM-validated ``[vx, vy, omega]`` command.

        Args:
            command:   ``(1, 3)`` tensor — ``[vx, vy, omega]``.
            state:     ``(1, >=3)`` tensor — ``[x, y, yaw, ...]``.
            neighbors: iterable of ``(x, y)`` OR ``(x, y, priority)`` for the OTHER
                dogs. Priority defaults to 1.0 (symmetric — everyone shares avoidance).
            self_priority: this dog's priority. Higher -> it yields less (keeps course).

        Returns:
            ``(1, 3)`` tensor — safe ``[vx, vy, omega]``.
        """
        squeeze = command.dim() == 1
        if squeeze:
            command = command.unsqueeze(0)
            state = state.unsqueeze(0)
        self._require_command(command)
        self._require_state(state)

        self._neighbors.points = [
            (float(p[0]), float(p[1]), float(p[2]) if len(p) >= 3 else 1.0) for p in neighbors
        ]
        self._neighbors.self_priority = float(self_priority)

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
