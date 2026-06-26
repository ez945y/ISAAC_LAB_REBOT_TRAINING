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
        self.points: list[tuple] = []  # (x, y, priority, same_group, vx, vy, yaw)
        self.self_priority: float = 1.0


@dam.register_callback(
    "go2_min_max_separation",
    layer="L1",
    category="execution",
    description="Priority-weighted inter-dog distance band: each dog keeps every "
    "neighbour >= min_dist (collision, share scaled by priority) and its nearest "
    "<= max_dist (cohesion). Yielding emerges from the responsibility split.",
    params={
        "min_dist": "HARD collision floor (m): absolute, symmetric — never crossed by anyone.",
        "comfort_dist": "SOFT comfort distance (m): preferred spacing; lower-priority gives it up.",
        "max_dist": "Cohesion ceiling (m): the nearest groupmate stays within this.",
        "dt": "Rollout horizon in seconds.",
        "gamma": "Discrete CBF rate in (0,1] — fraction of the margin a dog may close per step.",
        "influence": "Only neighbours within min_dist+influence (m) raise a constraint.",
        "lam_min": "Minimum comfort-yield share a dog keeps even at top priority.",
        "w_hard": "Slack penalty on the hard floor (near-inviolable).",
        "w_soft": "Slack penalty on the soft comfort/cohesion (gives way under pressure).",
        "capsule_half": "Half-length (m) of each dog's body capsule (0 = point/disc model).",
    },
)
def go2_min_max_separation(
    *,
    obs,
    action,
    solvers,
    min_dist: float = 1.0,
    comfort_dist: float = 2.0,
    max_dist: float = 4.0,
    dt: float = 0.2,
    gamma: float = 0.4,
    influence: float = 2.5,
    lam_min: float = 0.15,
    w_hard: float = 1.0e3,
    w_soft: float = 30.0,
    capsule_half: float = 0.0,
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
    neighbors = list(getattr(holder, "points", []) or [])  # (x,y,prio,same_group,vx,vy,yaw)
    if not neighbors:
        return True
    p_self = float(getattr(holder, "self_priority", 1.0))

    state = list(obs.joint_positions[:3])              # [x, y, yaw]
    command = list(action.target_joint_positions[3:6])  # [vx, vy, omega]
    vx, vy, om = command
    cx, cy = state[0], state[1]

    nxt0 = solver.rollout(state, command, dt=dt)
    nx0, ny0, nyaw0 = float(nxt0[0]), float(nxt0[1]), float(nxt0[2])
    syaw = state[2]
    cs, sn = math.cos(syaw), math.sin(syaw)     # current heading
    cp, sp = math.cos(nyaw0), math.sin(nyaw0)   # predicted heading

    def _grad_pt(nrx: float, nry: float, o: float) -> np.ndarray:
        """ANALYTIC d(projection on normal (nrx,nry))/d[vx,vy,omega] for a body point at
        offset ``o`` along the heading. Closed-form from the holonomic rollout -- no
        autograd, so nothing is retained between calls (retain_graph across reused
        guard calls was corrupting the QP into a spurious full stop)."""
        return np.array([
            (nrx * cs + nry * sn) * dt,                    # d/d vx
            (-nrx * sn + nry * cs) * dt,                   # d/d vy
            o * dt * (-nrx * sp + nry * cp),               # d/d omega (rotates the offset)
        ])

    # CAPSULE body model: approximate each dog's elongated body by spheres along its
    # heading at offsets {-h, 0, h} (h = capsule_half; h=0 -> the old point/disc). The
    # inter-dog distance is the closest spine-point pair, so head-on a pair reacts on
    # nose-to-nose distance (keeps centres further apart) while side-by-side dogs can
    # pack to body width -- exactly why capsules beat a circle for a long body.
    offs = (-capsule_half, 0.0, capsule_half) if capsule_half > 1e-6 else (0.0,)
    self_now = [(cx + o * cs, cy + o * sn) for o in offs]
    self_pred = [(nx0 + o * cp, ny0 + o * sp) for o in offs]

    def _capsule(px, py, vpx, vpy, nyaw):
        """(current_dist, predicted_separation, grad) between the two body capsules.

        The gradient is along the CURRENT closest-pair separation direction (not through
        the predicted point): a fast step can overshoot/pass through a very near
        neighbour, and a raw distance gradient would then flip sign (tell the dog to
        speed up *through* it). Projecting on the current normal is overshoot-safe.
        """
        ndx, ndy = math.cos(nyaw), math.sin(nyaw)
        nb_now = [(px + o * ndx, py + o * ndy) for o in offs]
        best, bi, bj = math.inf, 0, 0
        for i, a in enumerate(self_now):
            for j, b in enumerate(nb_now):
                dd = math.hypot(a[0] - b[0], a[1] - b[1])
                if dd < best:
                    best, bi, bj = dd, i, j
        dist_now = best
        ax0, ay0 = self_now[bi]
        bx0, by0 = nb_now[bj]
        sep = max(best, 1e-6)
        nrx, nry = (ax0 - bx0) / sep, (ay0 - by0) / sep   # unit normal, neighbour -> self
        nbx = px + vpx * dt + offs[bj] * ndx               # neighbour's matching point, predicted
        nby = py + vpy * dt + offs[bj] * ndy
        sxf, syf = self_pred[bi]
        d_pred = nrx * (sxf - nbx) + nry * (syf - nby)      # predicted separation along the normal
        return dist_now, d_pred, _grad_pt(nrx, nry, offs[bi])

    # Each entry: (grad(3,), rhs, slack_weight). push => grad.du + s >= rhs (apart);
    # pull => grad.du - s <= rhs (together). The per-constraint slack weight is what
    # makes a boundary HARD (near-infinite penalty -> inviolable) or SOFT (moderate
    # penalty -> gives way when holding it would cost more motion).
    push: list = []
    pull: list = []
    hard_active = False

    # --- collision: TWO boundaries per near neighbour (velocity-aware / TTC) ---
    # Predict the neighbour forward by its own velocity so a head-on pair sees the real
    # closing speed (static neighbours halve it -> react too late).
    #   HARD floor (min_dist): SYMMETRIC (lambda=0.5) + huge weight -> absolute, EVERY
    #     dog (even top priority) shares it, so under pressure they all slow/shift and
    #     no one is pushed past it or stuck retreating -- the "fluid" behaviour.
    #   SOFT comfort (comfort_dist): PRIORITY-weighted -> the lower-priority dog gives
    #     up the comfort zone first (yields), the higher-priority one flows through.
    reach = influence + 2.0 * capsule_half
    for px, py, p_j, _sg, vpx, vpy, nyaw in neighbors:
        if math.hypot(cx - px, cy - py) - min_dist > reach:
            continue  # cheap centre pre-cull before the full capsule test
        dist_now, d_pred, grad = _capsule(px, py, vpx, vpy, nyaw)
        req_hard = (dist_now - gamma * (dist_now - min_dist)) - d_pred
        if req_hard > 1e-4:
            push.append((grad, 0.5 * req_hard, w_hard))   # symmetric share
            hard_active = True
        req_soft = (dist_now - gamma * (dist_now - comfort_dist)) - d_pred
        if req_soft > 1e-4:
            lam = max(lam_min, p_j / (p_self + p_j))
            push.append((grad, lam * req_soft, w_soft))    # priority share

    # --- cohesion: nearest SAME-GROUP neighbour only (don't break formation) ---
    same_group = [(px, py) for px, py, _p, sg, _vx, _vy, _yaw in neighbors if sg]
    if same_group:
        npx, npy = min(same_group, key=lambda p: math.hypot(cx - p[0], cy - p[1]))
        nd = math.hypot(cx - npx, cy - npy)
        if nd > max_dist:
            target = max_dist + (1.0 - gamma) * (nd - max_dist)
            d_pred = math.hypot(nx0 - npx, ny0 - npy)
            if d_pred > target:
                u = ((nx0 - npx) / d_pred, (ny0 - npy) / d_pred)  # centre -> groupmate dir
                pull.append((_grad_pt(u[0], u[1], 0.0), target - d_pred, w_soft))

    if not push and not pull:
        return True

    # --- assemble the slack QP: vars = [du_vx, du_vy, du_omega, s_0 ... s_{m-1}] ---
    m = len(push) + len(pull)
    n = 3 + m
    rows, lo, hi, slack_w = [], [], [], []

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
    for grad, rhs, w in push:   # grad.du + s >= rhs
        rows.append(_row([(0, grad[0]), (1, grad[1]), (2, grad[2]), (3 + k, 1.0)]))
        lo.append(rhs); hi.append(np.inf); slack_w.append(w); k += 1
    for grad, rhs, w in pull:   # grad.du - s <= rhs
        rows.append(_row([(0, grad[0]), (1, grad[1]), (2, grad[2]), (3 + k, -1.0)]))
        lo.append(-np.inf); hi.append(rhs); slack_w.append(w); k += 1

    P = sparse.csc_matrix(np.diag([1.0, 1.0, 3.0] + slack_w))  # penalise yaw -> sidestep
    q = np.zeros(n)
    A = sparse.csc_matrix(np.array(rows, dtype=float))
    prob = osqp.OSQP()
    prob.setup(P, q, A, np.array(lo), np.array(hi), verbose=False, eps_abs=1e-5, eps_rel=1e-5)
    res = prob.solve()

    if res.x is not None and np.all(np.isfinite(res.x)) and "solved" in str(res.info.status):
        action.target_joint_positions[3] = float(vx + res.x[0])
        action.target_joint_positions[4] = float(vy + res.x[1])
        action.target_joint_positions[5] = float(om + res.x[2])
    elif hard_active:
        # Degenerate solver failure with the HARD floor active must fail safe: stop.
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
            neighbors: iterable of ``(x, y[, priority[, same_group[, vx, vy[, yaw]]]])``
                for the OTHER dogs. priority defaults to 1.0 (symmetric); same_group
                defaults to 1 (pass 0 for other groups); world velocity (vx, vy) defaults
                to 0 (static, makes the guard velocity-aware/TTC); yaw defaults to 0 (the
                neighbour's heading, for the body-capsule distance).
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
            (float(p[0]), float(p[1]),
             float(p[2]) if len(p) >= 3 else 1.0,
             float(p[3]) if len(p) >= 4 else 1.0,
             float(p[4]) if len(p) >= 6 else 0.0,
             float(p[5]) if len(p) >= 6 else 0.0,
             float(p[6]) if len(p) >= 7 else 0.0)
            for p in neighbors
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
