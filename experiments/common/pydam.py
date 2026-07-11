"""PyDamFilter: dependency-light reference implementation of the DAM guard.

Mirrors ``go2_min_max_separation`` (tools/controll_scripts/safety/go2_dam_wrapper.py)
in pure numpy + scipy (SLSQP instead of OSQP), so every ablation script can be
developed and fake-data-validated on machines WITHOUT torch/osqp/dam.

⚠️ Role: pipeline validation + local ablation previews. Thesis numbers for the
"dam" method come from the real Go2DAMWrapper on the Isaac machine, and
``experiments/tests/test_pydam_vs_dam.py`` (Isaac machine) guards against the
two implementations drifting apart.

Every yaml param is a constructor knob, plus ablation-only switches the real
callback doesn't expose (``velocity_aware``, ``cost_diag``).
Extra telemetry in the info dict: ``hard_slack_max`` (E4.3), ``n_push``, ``n_pull``.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_MOD_PATH = Path(__file__).resolve().parents[2] / "tools/controll_scripts/safety/capsule_geometry.py"
_spec = importlib.util.spec_from_file_location("capsule_geometry", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
seg_seg_closest = _mod.seg_seg_closest


class PyDamFilter:
    name = "pydam"

    def __init__(
        self,
        *,
        min_dist: float = 0.7,
        comfort_dist: float = 1.5,
        max_dist: float = 4.0,
        dt: float = 0.2,
        gamma: float = 0.4,
        influence: float = 2.5,
        lam_min: float = 0.3,
        w_hard: float = 1000.0,
        w_soft: float = 30.0,
        capsule_half: float = 0.25,
        swirl: float = 0.6,
        max_v: float = 1.5,
        max_omega: float = 2.0,
        velocity_aware: bool = True,   # ablation E3.5: False -> neighbours treated as static
        cost_diag: tuple = (2.0, 0.5, 3.0),
    ):
        self.p = dict(
            min_dist=min_dist, comfort_dist=comfort_dist, max_dist=max_dist,
            dt=dt, gamma=gamma, influence=influence, lam_min=lam_min,
            w_hard=w_hard, w_soft=w_soft, capsule_half=capsule_half, swirl=swirl,
            max_v=max_v, max_omega=max_omega,
        )
        self.velocity_aware = velocity_aware
        self.cost_diag = tuple(cost_diag)

    # ------------------------------------------------------------------
    def filter(self, aid, cmd, pose, neighbors, self_priority: float = 1.0):
        p = self.p
        dt, gamma = p["dt"], p["gamma"]
        h = p["capsule_half"]
        vx, vy, om = cmd
        cx, cy, syaw = pose
        cs, sn = math.cos(syaw), math.sin(syaw)

        # rollout (same first-order holonomic model as HolonomicSolver)
        vxc = max(-p["max_v"], min(p["max_v"], vx))
        vyc = max(-p["max_v"], min(p["max_v"], vy))
        omc = max(-p["max_omega"], min(p["max_omega"], om))
        nx0 = cx + (vxc * cs - vyc * sn) * dt
        ny0 = cy + (vxc * sn + vyc * cs) * dt
        nyaw0 = syaw + omc * dt
        cp, sp = math.cos(nyaw0), math.sin(nyaw0)

        def grad_pt(nrx, nry, o):
            return np.array([
                (nrx * cs + nry * sn) * dt,
                (-nrx * sn + nry * cs) * dt,
                o * dt * (-nrx * sp + nry * cp),
            ])

        push, pull = [], []
        hard_active = False
        reach = p["influence"] + 2.0 * h
        nearest_d, nearest_n, nearest_pj = math.inf, None, 1.0

        for nb in neighbors:
            px, py, p_j, sg = nb[0], nb[1], nb[2], nb[3]
            vpx = nb[4] if (self.velocity_aware and len(nb) >= 6) else 0.0
            vpy = nb[5] if (self.velocity_aware and len(nb) >= 6) else 0.0
            nyaw = nb[6] if len(nb) >= 7 else 0.0
            if math.hypot(cx - px, cy - py) - p["min_dist"] > reach:
                continue
            ndx, ndy = math.cos(nyaw), math.sin(nyaw)
            s_off, t_off, dist_now, ax0, ay0, bx0, by0 = seg_seg_closest(
                cx, cy, cs, sn, h, px, py, ndx, ndy, h)
            sep = max(dist_now, 1e-6)
            nrx, nry = (ax0 - bx0) / sep, (ay0 - by0) / sep
            nbx = px + vpx * dt + t_off * ndx
            nby = py + vpy * dt + t_off * ndy
            sxf, syf = nx0 + s_off * cp, ny0 + s_off * sp
            d_pred = nrx * (sxf - nbx) + nry * (syf - nby)
            g = grad_pt(nrx, nry, s_off)
            if dist_now < nearest_d:
                nearest_d, nearest_n, nearest_pj = dist_now, (nrx, nry), p_j
            req_hard = (dist_now - gamma * (dist_now - p["min_dist"])) - d_pred
            if req_hard > 1e-4:
                push.append((g, 0.5 * req_hard, p["w_hard"]))
                hard_active = True
            req_soft = (dist_now - gamma * (dist_now - p["comfort_dist"])) - d_pred
            if req_soft > 1e-4:
                lam = max(p["lam_min"], p_j / (self_priority + p_j))
                push.append((g, lam * req_soft, p["w_soft"]))

        same_group = [(nb[0], nb[1]) for nb in neighbors if nb[3] >= 0.5]
        if same_group:
            npx, npy = min(same_group, key=lambda q: math.hypot(cx - q[0], cy - q[1]))
            nd = math.hypot(cx - npx, cy - npy)
            if nd > p["max_dist"]:
                target = p["max_dist"] + (1.0 - gamma) * (nd - p["max_dist"])
                d_pred_c = math.hypot(nx0 - npx, ny0 - npy)
                if d_pred_c > target:
                    u = ((nx0 - npx) / d_pred_c, (ny0 - npy) / d_pred_c)
                    pull.append((grad_pt(u[0], u[1], 0.0), target - d_pred_c, p["w_soft"]))

        if not push and not pull:
            return cmd, {"decision": "PASS", "hard_slack_max": 0.0,
                         "n_push": 0, "n_pull": 0}

        # ---- slack QP: z = [du(3), s(m)];  min 0.5 z'Pz + q'z ------------
        m = len(push) + len(pull)
        n = 3 + m
        slack_w = [w for _, _, w in push] + [w for _, _, w in pull]
        Pdiag = np.array(list(self.cost_diag) + slack_w)
        q = np.zeros(n)
        if p["swirl"] > 0.0 and nearest_n is not None and nearest_d < p["comfort_dist"]:
            prox = max(0.0, 1.0 - (nearest_d - p["min_dist"]) /
                       max(p["comfort_dist"] - p["min_dist"], 1e-6))
            share = max(p["lam_min"], nearest_pj / (self_priority + nearest_pj))
            scale = p["swirl"] * prox * share
            q[0] = -scale * (-nearest_n[1])
            q[1] = -scale * (nearest_n[0])

        cons = []
        for k, (g, rhs, _w) in enumerate(push):     # g.du + s_k >= rhs
            gi, ki = np.array(g, dtype=float), 3 + k
            cons.append({"type": "ineq",
                         "fun": (lambda z, gi=gi, ki=ki, rhs=rhs: gi @ z[:3] + z[ki] - rhs),
                         "jac": (lambda z, gi=gi, ki=ki: _row(n, gi, ki, 1.0))})
        for k, (g, rhs, _w) in enumerate(pull):     # rhs - (g.du - s_k) >= 0
            gi, ki = np.array(g, dtype=float), 3 + len(push) + k
            cons.append({"type": "ineq",
                         "fun": (lambda z, gi=gi, ki=ki, rhs=rhs: rhs - gi @ z[:3] + z[ki]),
                         "jac": (lambda z, gi=gi, ki=ki: -_row(n, gi, ki, -1.0))})

        bounds = [(-p["max_v"] - vx, p["max_v"] - vx),
                  (-p["max_v"] - vy, p["max_v"] - vy),
                  (-p["max_omega"] - om, p["max_omega"] - om)] + [(0.0, None)] * m

        def cost(z):
            return 0.5 * float(z @ (Pdiag * z)) + float(q @ z)

        def cost_jac(z):
            return Pdiag * z + q

        res = minimize(cost, np.zeros(n), jac=cost_jac, bounds=bounds,
                       constraints=cons, method="SLSQP",
                       options={"maxiter": 200, "ftol": 1e-8})

        if res.success and np.all(np.isfinite(res.x)):
            du = res.x[:3]
            slacks = res.x[3:]
            hard_slack = max((float(slacks[i]) for i, (_g, _r, w) in enumerate(push)
                              if w >= p["w_hard"] * 0.999), default=0.0)
            safe = (vx + float(du[0]), vy + float(du[1]), om + float(du[2]))
            delta = max(abs(d) for d in du)
            return safe, {"decision": "CLAMP" if delta > 1e-5 else "PASS",
                          "hard_slack_max": hard_slack,
                          "n_push": len(push), "n_pull": len(pull)}
        if hard_active:
            return (0.0, 0.0, 0.0), {"decision": "REJECT", "hard_slack_max": math.nan,
                                     "n_push": len(push), "n_pull": len(pull)}
        return cmd, {"decision": "PASS", "hard_slack_max": 0.0,
                     "n_push": len(push), "n_pull": len(pull)}


def _row(n, gi, ki, s_coeff):
    r = np.zeros(n)
    r[:3] = gi
    r[ki] = s_coeff
    return r
