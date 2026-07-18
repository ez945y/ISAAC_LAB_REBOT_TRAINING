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

DAM 0.7.0 (``Guardrail``, dict-in/command-out): the observation is the
``base_pose`` group ``[x, y, yaw]`` and the action is the command ``[vx, vy,
omega]`` directly — no more 6D obs/action packing. Neighbour positions change
every call, so they are injected into the guard through the ``solvers`` mapping
via a small mutable holder — the boundary callback reads them alongside the
rollout solver.

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

from .capsule_geometry import seg_seg_closest
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

    __slots__ = ("points", "self_priority",
                 "last_hard_slack", "last_n_push", "last_n_pull")

    def __init__(self) -> None:
        self.points: list[tuple] = []  # (x, y, priority, same_group, vx, vy, yaw, static)
        self.self_priority: float = 1.0
        # Per-call telemetry written back by the boundary callback (E4.3 /
        # dual-channel monitoring): worst hard-floor slack the QP bought this
        # call (nan = solver failure with the hard floor active), and the
        # number of push/pull constraints assembled.
        self.last_hard_slack: float = 0.0
        self.last_n_push: int = 0
        self.last_n_pull: int = 0


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
        "swirl": "Tangential circulation bias near a neighbour (escapes head-on standoff "
                 "/ local minima -> liveness, all in the QP).",
        "wall_min_dist": "Optional separate hard floor (m) for STATIC neighbours; None -> "
                         "min_dist. Ablation E3.4: a circumscribed-disc body inflates dog-dog "
                         "clearance by 2h but dog-wall by only 1h.",
        "velocity_aware": "Predict neighbours forward by their observed velocity (default). "
                          "False -> treat neighbours as static (ablation E3.5).",
        "grad_mode": "'analytic' (default): overshoot-safe frozen-normal surrogate gradient. "
                     "'autograd': naive true-distance gradient via torch (ablation E3.7).",
        "drive_mode": "'holonomic' (default) or 'differential' (E1.2): a diff-drive base "
                      "has no lateral channel, so the QP's cheap-sidestep cost profile and "
                      "the swirl's lateral bias are remapped to steering (yaw).",
    },
)
def go2_min_max_separation(
    *,
    base_pose,
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
    swirl: float = 0.0,
    wall_min_dist: float | None = None,
    velocity_aware: bool = True,
    grad_mode: str = "analytic",
    drive_mode: str = "holonomic",
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
    neighbors = list(getattr(holder, "points", []) or [])  # (x,y,prio,sg,vx,vy,yaw,static)
    if holder is not None:
        holder.last_hard_slack, holder.last_n_push, holder.last_n_pull = 0.0, 0, 0
    if not neighbors:
        return True
    p_self = float(getattr(holder, "self_priority", 1.0))

    state = [float(v) for v in base_pose]                   # [x, y, yaw]
    command = [float(v) for v in action.target_joint_positions[:3]]  # [vx, vy, omega]
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

    # CAPSULE body model: each dog's elongated body is the sweep of a disc along its
    # spine segment ``centre ± capsule_half * heading`` (h=0 -> point/disc). The
    # inter-dog distance is the EXACT closest segment-segment pair, so head-on a pair
    # reacts on nose-to-nose distance (keeps centres further apart) while side-by-side
    # dogs can pack to body width -- exactly why capsules beat a circle for a long body.

    def _capsule(px, py, vpx, vpy, nyaw, nb_half=capsule_half):
        """(current_dist, predicted_separation, grad) between the two body capsules.

        The gradient is along the CURRENT closest-pair separation direction (not through
        the predicted point): a fast step can overshoot/pass through a very near
        neighbour, and a raw distance gradient would then flip sign (tell the dog to
        speed up *through* it). Projecting on the current normal is overshoot-safe.

        ``nb_half=0`` treats the neighbour as a POINT (static wall samples): giving
        wall points a body half-length fattens them by h on both sides and can make
        narrow gaps spuriously infeasible (E3.3 finding F6).
        """
        ndx, ndy = math.cos(nyaw), math.sin(nyaw)
        s_off, t_off, dist_now, ax0, ay0, bx0, by0 = seg_seg_closest(
            cx, cy, cs, sn, capsule_half, px, py, ndx, ndy, nb_half
        )
        sep = max(dist_now, 1e-6)
        nrx, nry = (ax0 - bx0) / sep, (ay0 - by0) / sep   # unit normal, neighbour -> self
        nbx = px + vpx * dt + t_off * ndx                  # neighbour's closest point, predicted
        nby = py + vpy * dt + t_off * ndy
        sxf = nx0 + s_off * cp                             # own closest point under predicted pose
        syf = ny0 + s_off * sp
        d_pred = nrx * (sxf - nbx) + nry * (syf - nby)      # predicted separation along the normal
        return dist_now, d_pred, _grad_pt(nrx, nry, s_off), (nrx, nry)

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
    nearest_d, nearest_n, nearest_pj = math.inf, None, 1.0
    for nb in neighbors:
        px, py, p_j, _sg, vpx, vpy, nyaw = nb[:7]
        is_static = len(nb) >= 8 and nb[7] >= 0.5
        if not velocity_aware:
            vpx = vpy = 0.0    # ablation E3.5: neighbours treated as static
        if math.hypot(cx - px, cy - py) - min_dist > reach:
            continue  # cheap centre pre-cull before the full capsule test
        nb_half = 0.0 if is_static else capsule_half
        dist_now, d_pred, grad, nrm = _capsule(px, py, vpx, vpy, nyaw, nb_half=nb_half)
        if grad_mode == "autograd":
            # Ablation E3.7: naive true-distance gradient (rotates with the
            # geometry; can flip sign through overlap). dist_now/normal still
            # come from the exact analytic closest pair above.
            from .torchgrad import pred_dist_and_grad
            d_pred, grad = pred_dist_and_grad(
                cx, cy, syaw, (vx, vy, om), dt, capsule_half,
                px, py, nyaw, nb_half, vpx, vpy, solver.max_v, solver.max_omega)
        floor = min_dist if (wall_min_dist is None or not is_static) else wall_min_dist
        req_hard = (dist_now - gamma * (dist_now - floor)) - d_pred
        if req_hard > 1e-4:
            # A static obstacle cannot do its half of the avoiding: this dog
            # carries the full hard requirement (share 1.0, not 0.5).
            push.append((grad, (1.0 if is_static else 0.5) * req_hard, w_hard))
            hard_active = True
        if is_static:
            # Walls exert ONLY the hard floor. Comfort is a SOCIAL distance
            # between agents; applying it to static geometry makes any corridor
            # narrower than 2*comfort_dist permanently impassable (finding F5:
            # dogs froze 1.4 m short of a 1.6 m gap). Swirl likewise never
            # circulates around a wall.
            continue
        if dist_now < nearest_d:
            nearest_d, nearest_n, nearest_pj = dist_now, nrm, p_j
        req_soft = (dist_now - gamma * (dist_now - comfort_dist)) - d_pred
        if req_soft > 1e-4:
            lam = max(lam_min, p_j / (p_self + p_j))
            push.append((grad, lam * req_soft, w_soft))    # priority share

    # --- cohesion: nearest SAME-GROUP neighbour only (don't break formation) ---
    same_group = [(nb[0], nb[1]) for nb in neighbors
                  if nb[3] and not (len(nb) >= 8 and nb[7] >= 0.5)]
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
    # Lateral cap may differ (max_vy=0 -> differential drive: no sidestep,
    # the QP must resolve conflicts with braking/turning only — E1.2).
    _mvy = getattr(solver, "max_vy", None)
    _mvy = solver.max_v if _mvy is None else _mvy
    lo += [-solver.max_v - vx, -_mvy - vy, -solver.max_omega - om]
    hi += [solver.max_v - vx, _mvy - vy, solver.max_omega - om]
    for k in range(m):                                              # slacks >= 0
        rows.append(_row([(3 + k, 1.0)])); lo.append(0.0); hi.append(np.inf)

    k = 0
    for grad, rhs, w in push:   # grad.du + s >= rhs
        rows.append(_row([(0, grad[0]), (1, grad[1]), (2, grad[2]), (3 + k, 1.0)]))
        lo.append(rhs); hi.append(np.inf); slack_w.append(w); k += 1
    for grad, rhs, w in pull:   # grad.du - s <= rhs
        rows.append(_row([(0, grad[0]), (1, grad[1]), (2, grad[2]), (3 + k, -1.0)]))
        lo.append(-np.inf); hi.append(rhs); slack_w.append(w); k += 1

    # Cost weights on [du_vx, du_vy, du_omega]: braking (du_vx) is EXPENSIVE and a
    # lateral sidestep (du_vy) is CHEAP, so the QP slides around a neighbour instead
    # of stopping dead in its path; yaw is dearest (prefer strafe over re-heading).
    # A differential drive has no sidestep — steering IS its lateral mechanism, so
    # yaw becomes the cheap axis instead (E1.2 kinematic adapter).
    diag3 = [2.0, 0.5, 3.0] if drive_mode != "differential" else [2.0, 0.5, 0.5]
    P = sparse.csc_matrix(np.diag(diag3 + slack_w))
    q = np.zeros(n)
    # SWIRL (liveness): a gentle tangential bias around the nearest neighbour makes
    # conflicting dogs circulate the SAME way instead of standing off head-on or
    # settling in a local minimum. It only nudges the objective -- the hard/soft
    # safety constraints are untouched -- so it adds liveness without risking safety.
    if swirl > 0.0 and nearest_n is not None and nearest_d < comfort_dist:
        prox = max(0.0, 1.0 - (nearest_d - min_dist) / max(comfort_dist - min_dist, 1e-6))
        # Priority-weighted: the lower-priority dog circulates MORE (its yield share),
        # the higher-priority less but not zero -> both sidestep, the low one more.
        share = max(lam_min, nearest_pj / (p_self + nearest_pj))
        scale = swirl * prox * share
        # 90deg CCW tangent of the separation normal — a WORLD direction. Rotate it
        # into this dog's BODY frame before biasing [du_vx, du_vy]: without the
        # rotation two head-on dogs (yaws pi apart) get pushed to the SAME world
        # side, stay collinear, and the standoff never breaks (bug found by E3.2).
        tx, ty = -nearest_n[1], nearest_n[0]
        q[0] = -scale * (tx * cs + ty * sn)
        q[1] = -scale * (-tx * sn + ty * cs)
        if drive_mode == "differential":
            # No lateral channel to bias: circulate by STEERING toward the
            # tangent instead. cross(heading, tangent) is the signed heading
            # error to the tangent direction — bias du_omega to close it.
            q[1] = 0.0
            q[2] = -scale * (cs * ty - sn * tx)
    A = sparse.csc_matrix(np.array(rows, dtype=float))
    prob = osqp.OSQP()
    prob.setup(P, q, A, np.array(lo), np.array(hi), verbose=False, eps_abs=1e-5, eps_rel=1e-5)
    res = prob.solve()

    if holder is not None:
        holder.last_n_push, holder.last_n_pull = len(push), len(pull)
    if res.x is not None and np.all(np.isfinite(res.x)) and "solved" in str(res.info.status):
        action.target_joint_positions[0] = float(vx + res.x[0])
        action.target_joint_positions[1] = float(vy + res.x[1])
        action.target_joint_positions[2] = float(om + res.x[2])
        if holder is not None:
            # E4.3 telemetry: worst slack the QP bought on a HARD constraint
            # this call (slacks are ordered push-first in the variable vector).
            holder.last_hard_slack = max(
                (float(res.x[3 + i]) for i, (_g, _r, w) in enumerate(push)
                 if w >= w_hard * 0.999), default=0.0)
    elif hard_active:
        # Degenerate solver failure with the HARD floor active must fail safe: stop.
        action.target_joint_positions[0] = 0.0
        action.target_joint_positions[1] = 0.0
        action.target_joint_positions[2] = 0.0
        if holder is not None:
            holder.last_hard_slack = math.nan
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
        # DAM 0.7.0: Guardrail is dict-in/command-out; on REJECT return an
        # explicit stop twist (a velocity-controlled base must not "hold").
        self._guard = dam.Guardrail(
            self._resolve_stackfile(stackfile),
            task=task,
            degrees_mode=False,  # pose "joints" are metres/radians, not motor degrees
            solvers={_SOLVER_KEY: self.solver, _NEIGHBOR_KEY: self._neighbors},
            safe_action=[0.0, 0.0, 0.0],
            quiet=True,
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
             float(p[6]) if len(p) >= 7 else 0.0,
             float(p[7]) if len(p) >= 8 else 0.0)  # static: hard floor only
            for p in neighbors
        ]
        self._neighbors.self_priority = float(self_priority)

        raw = command[0]       # [vx, vy, omega]
        pose = state[0, :3]    # [x, y, yaw]

        safe = torch.as_tensor(
            self._guard({
                "base_pose": [float(v) for v in pose],
                "action": np.asarray(raw.detach().cpu(), dtype=np.float64),
            }),
            dtype=raw.dtype, device=raw.device,
        ).reshape(-1)

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
    def last_hard_slack(self) -> float:
        """Worst hard-floor slack the QP bought on the last call (E4.3 telemetry;
        nan = solver failure with the hard floor active). Pair with a
        measured-distance monitor: slack only sees infeasibility-driven erosion."""
        return self._neighbors.last_hard_slack

    @property
    def last_n_push(self) -> int:
        return self._neighbors.last_n_push

    @property
    def last_n_pull(self) -> int:
        return self._neighbors.last_n_pull

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
