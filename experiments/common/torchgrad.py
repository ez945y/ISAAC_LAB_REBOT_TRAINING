"""Autograd (torch) gradient path for the capsule CBF constraint — E3.7 only.

The production guard uses an ANALYTIC surrogate: it projects the predicted
closest-pair positions onto the CURRENT separation normal, so the gradient
direction is frozen for the horizon ("overshoot-safe": a step that passes
through a near neighbour cannot flip the gradient sign and tell the dog to
speed up through it — see go2_dam_wrapper._capsule).

This module computes the NAIVE alternative: the true predicted capsule
distance as a differentiable torch expression of the command, with the
gradient from autograd. Near/through overlap the true-distance gradient
rotates with the geometry and CAN flip its radial sign — that is the failure
mode E3.7 quantifies. Requires torch (DAM venv); imported lazily.
"""

from __future__ import annotations

import torch


def _dot(ax, ay, bx, by):
    return ax * bx + ay * by


def seg_seg_dist_torch(p1x, p1y, d1x, d1y, h1, p2x, p2y, d2x, d2y, h2):
    """Differentiable segment-segment distance (Ericson §5.1.9, clamped).

    Segments are centre ± h * direction (unit). All args are torch scalars;
    a.e.-differentiable (clamps are kinks, torch picks a subgradient).
    """
    a1x, a1y = p1x - h1 * d1x, p1y - h1 * d1y   # segment 1 start
    b1x, b1y = p1x + h1 * d1x, p1y + h1 * d1y
    a2x, a2y = p2x - h2 * d2x, p2y - h2 * d2y
    b2x, b2y = p2x + h2 * d2x, p2y + h2 * d2y
    ux, uy = b1x - a1x, b1y - a1y               # segment 1 span (2*h1 long)
    vx, vy = b2x - a2x, b2y - a2y
    wx, wy = a1x - a2x, a1y - a2y
    aa = _dot(ux, uy, ux, uy) + 1e-12
    bb = _dot(ux, uy, vx, vy)
    cc = _dot(vx, vy, vx, vy) + 1e-12
    dd = _dot(ux, uy, wx, wy)
    ee = _dot(vx, vy, wx, wy)
    den = aa * cc - bb * bb
    s = torch.where(den.abs() > 1e-9, (bb * ee - cc * dd) / (den + 1e-12),
                    torch.zeros_like(den))
    s = s.clamp(0.0, 1.0)
    t = ((bb * s + ee) / cc).clamp(0.0, 1.0)
    # UNCONDITIONAL Gauss-Seidel re-projection of s given the final t. This is
    # what makes the degenerate cases correct: for a point neighbour (cc→0) or
    # (anti)parallel spines (den→0) the fallback s=0 pins the closest point at
    # the TAIL; the re-projection moves it to the true closest point (found by
    # E3.7: head-on d_pred was overestimated by up to 2h -> under-braking).
    s = ((bb * t - dd) / aa).clamp(0.0, 1.0)
    cx1, cy1 = a1x + s * ux, a1y + s * uy
    cx2, cy2 = a2x + t * vx, a2y + t * vy
    return torch.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + 1e-12)


def pred_dist_and_grad(cx, cy, syaw, cmd, dt, h,
                       px, py, nyaw, nb_h, vpx, vpy, max_v, max_omega):
    """(d_pred, grad[3]) — true predicted capsule distance and its autograd
    gradient w.r.t. the raw command [vx, vy, omega].

    Mirrors pydam's holonomic rollout exactly: translate with the CURRENT
    heading, orient the predicted capsule with yaw + omega*dt.
    """
    cmd_t = torch.tensor([float(cmd[0]), float(cmd[1]), float(cmd[2])],
                         dtype=torch.float64, requires_grad=True)
    vxc = cmd_t[0].clamp(-max_v, max_v)
    vyc = cmd_t[1].clamp(-max_v, max_v)
    omc = cmd_t[2].clamp(-max_omega, max_omega)
    cs = torch.tensor(float(syaw), dtype=torch.float64).cos()
    sn = torch.tensor(float(syaw), dtype=torch.float64).sin()
    nx0 = cx + (vxc * cs - vyc * sn) * dt
    ny0 = cy + (vxc * sn + vyc * cs) * dt
    nyaw0 = syaw + omc * dt
    d1x, d1y = nyaw0.cos(), nyaw0.sin()
    pfx = torch.tensor(px + vpx * dt, dtype=torch.float64)
    pfy = torch.tensor(py + vpy * dt, dtype=torch.float64)
    n2x = torch.tensor(nyaw, dtype=torch.float64).cos()
    n2y = torch.tensor(nyaw, dtype=torch.float64).sin()
    d = seg_seg_dist_torch(nx0, ny0, d1x, d1y, torch.tensor(h, dtype=torch.float64),
                           pfx, pfy, n2x, n2y, torch.tensor(nb_h, dtype=torch.float64))
    g = torch.autograd.grad(d, cmd_t)[0]
    return float(d.detach()), g.detach().numpy()
