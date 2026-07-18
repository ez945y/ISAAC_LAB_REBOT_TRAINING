# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Differentiable planar holonomic rollout for an omnidirectional legged base.

The Go2 quadruped is holonomic: its RL policy accepts a body-frame command
``[vx, vy, omega]`` (forward, lateral, yaw-rate) and walks it out. So unlike the
unicycle :class:`AckermannSolver` (``[v, omega]`` — forward + turn only), a DAM
safety correction here can **sidestep** (lateral ``vy``) instead of only braking
or spinning, which is the natural way to open/close inter-dog distance.

Why write our own: Isaac Sim only ships a holonomic *wheel* controller
(``isaacsim.robot...HolonomicController`` — body velocity -> mecanum wheel speeds,
the inverse direction, non-differentiable, wheel-geometry bound) and Isaac Lab
ships none. A DAM L1 boundary needs the opposite: a tiny, torch-differentiable
forward predictor of the planar pose so the QP can take gradients of an
inter-robot distance through the rollout. No wheels — the leg gait is the RL
policy's job; this only predicts where the base lands.

Drop-in for the same DAM callback pattern as :class:`AckermannSolver`: same
``rollout(state, command, dt)`` shape contract, but ``command`` is 3-D.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .ackermann_solver import _as_tensor, _wrap_angle


@dataclass(frozen=True)
class HolonomicSolver:
    """Roll out a planar omnidirectional base under a body-frame ``[vx, vy, omega]``."""

    default_dt: float = 1.0 / 60.0
    max_v: float = 1.5      # linear cap per body axis (m/s), >= the policy's command range
    max_omega: float = 2.0  # yaw-rate cap (rad/s)
    # Optional separate LATERAL cap (m/s). None -> max_v (holonomic, default —
    # behaviour unchanged). 0.0 models a nonholonomic differential-drive base:
    # the guard's QP then cannot buy safety with a sidestep and must brake/turn
    # (RQ1 cross-embodiment, E1.2).
    max_vy: float | None = None

    def rollout(self, state, command, dt: float | None = None) -> torch.Tensor:
        """Predict the next planar state from ``state`` and ``[vx, vy, omega]``.

        Args:
            state: ``[x, y, yaw]`` (world frame).
            command: ``[vx, vy, omega]`` (body frame; vx forward, vy left, omega yaw-rate).
            dt: integration step in seconds. Defaults to ``default_dt``.

        Returns:
            ``torch.Tensor`` ``[x_next, y_next, yaw_next]`` — differentiable in ``command``.
        """
        dt = self.default_dt if dt is None else float(dt)
        state_t = _as_tensor(state).reshape(-1)
        cmd_t = self.clamp_command(command).reshape(-1)
        if state_t.numel() < 3:
            raise ValueError(f"HolonomicSolver.rollout expected state [x, y, yaw], got {tuple(state_t.shape)}")
        if cmd_t.numel() != 3:
            raise ValueError(f"HolonomicSolver.rollout expected command [vx, vy, omega], got {tuple(cmd_t.shape)}")

        x, y, yaw = state_t[0], state_t[1], state_t[2]
        vx, vy, omega = cmd_t[0], cmd_t[1], cmd_t[2]
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        # First-order body-frame integration (constant velocity over dt). Exact arc
        # integration buys nothing at the short horizons a per-step guard uses, and
        # this keeps the rollout simple and cleanly differentiable.
        next_x = x + (vx * cos_y - vy * sin_y) * dt
        next_y = y + (vx * sin_y + vy * cos_y) * dt
        next_yaw = _wrap_angle(yaw + omega * dt)
        return torch.stack([next_x, next_y, next_yaw])

    def clamp_command(self, command) -> torch.Tensor:
        """Clamp ``[vx, vy, omega]`` to configured actuator limits."""
        cmd = _as_tensor(command).reshape(-1)
        if cmd.numel() != 3:
            raise ValueError(f"HolonomicSolver expected command [vx, vy, omega], got {tuple(cmd.shape)}")
        mvy = self.max_v if self.max_vy is None else self.max_vy
        return torch.stack([
            torch.clamp(cmd[0], -self.max_v, self.max_v),
            torch.clamp(cmd[1], -mvy, mvy),
            torch.clamp(cmd[2], -self.max_omega, self.max_omega),
        ])
