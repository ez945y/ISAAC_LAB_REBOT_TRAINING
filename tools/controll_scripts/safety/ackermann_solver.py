# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Ackermann/unicycle rollout helper for mobile-base DAM guards.

The Jetbot is differential drive, but the safety action space is the usual
mobile-base command ``[v, omega]``:

* ``v``: forward linear velocity in m/s.
* ``omega``: yaw rate in rad/s.

For a differential-drive robot, the final conversion to wheel angular velocities
is a simple geometry transform. No URDF is required for this level of safety
checking; URDF/USD assets become useful only when collision geometry or exact
link placement is part of the guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

import torch


@dataclass(frozen=True)
class AckermannSolver:
    """Roll out a planar mobile base and convert commands to wheel speeds."""

    track_width: float = 0.12
    wheel_radius: float = 1.0
    default_dt: float = 1.0 / 60.0
    max_v: float = 1.4
    max_omega: float = 6.0

    def rollout(self, state, command, dt: float | None = None):
        """Predict the next planar state from ``state`` and ``[v, omega]``.

        Args:
            state: ``[x, y, yaw]`` or a mapping with ``x``, ``y``, ``yaw``.
            command: ``[v, omega]``.
            dt: integration step in seconds. Defaults to ``default_dt``.

        Returns:
            A ``torch.Tensor`` shaped like ``[x_next, y_next, yaw_next]``.
        """
        dt = self.default_dt if dt is None else float(dt)
        state_t = _as_tensor(state).reshape(-1)
        cmd_t = self.clamp_command(command).reshape(-1)
        if state_t.numel() < 3:
            raise ValueError(f"AckermannSolver.rollout expected state [x, y, yaw], got {tuple(state_t.shape)}")
        if cmd_t.numel() != 2:
            raise ValueError(f"AckermannSolver.rollout expected command [v, omega], got {tuple(cmd_t.shape)}")

        x, y, yaw = state_t[0], state_t[1], state_t[2]
        v, omega = cmd_t[0], cmd_t[1]
        next_yaw = _wrap_angle(yaw + omega * dt)

        if torch.abs(omega) < 1e-6:
            next_x = x + v * torch.cos(yaw) * dt
            next_y = y + v * torch.sin(yaw) * dt
        else:
            radius = v / omega
            next_x = x + radius * (torch.sin(next_yaw) - torch.sin(yaw))
            next_y = y - radius * (torch.cos(next_yaw) - torch.cos(yaw))
        return torch.stack([next_x, next_y, next_yaw])

    def clamp_command(self, command) -> torch.Tensor:
        """Clamp ``[v, omega]`` to configured actuator limits."""
        cmd = _as_tensor(command).reshape(-1)
        if cmd.numel() != 2:
            raise ValueError(f"AckermannSolver expected command [v, omega], got {tuple(cmd.shape)}")
        return torch.stack([
            torch.clamp(cmd[0], -self.max_v, self.max_v),
            torch.clamp(cmd[1], -self.max_omega, self.max_omega),
        ])

    def command_to_wheels(self, command) -> torch.Tensor:
        """Convert ``[v, omega]`` to left/right wheel angular velocities."""
        v, omega = self.clamp_command(command)
        left_linear = v - omega * self.track_width / 2.0
        right_linear = v + omega * self.track_width / 2.0
        return torch.stack([left_linear / self.wheel_radius, right_linear / self.wheel_radius])

    def wheels_to_command(self, wheels) -> torch.Tensor:
        """Convert left/right wheel angular velocities to ``[v, omega]``."""
        wheel_t = _as_tensor(wheels).reshape(-1)
        if wheel_t.numel() != 2:
            raise ValueError(f"AckermannSolver expected wheels [left, right], got {tuple(wheel_t.shape)}")
        left_linear = wheel_t[0] * self.wheel_radius
        right_linear = wheel_t[1] * self.wheel_radius
        v = (left_linear + right_linear) * 0.5
        omega = (right_linear - left_linear) / self.track_width
        return torch.stack([v, omega])

    def target_to_command(
        self,
        state,
        target_xy,
        *,
        drive_gain: float = 2.0,
        turn_gain: float = 4.0,
    ) -> torch.Tensor:
        """Small nominal controller from planar state to a ``[v, omega]`` command."""
        state_t = _as_tensor(state).reshape(-1)
        target_t = _as_tensor(target_xy).reshape(-1)
        if state_t.numel() < 3:
            raise ValueError("target_to_command expected state [x, y, yaw]")
        if target_t.numel() < 2:
            raise ValueError("target_to_command expected target [x, y]")
        delta = target_t[:2] - state_t[:2]
        desired_heading = torch.atan2(delta[1], delta[0])
        heading_error = _wrap_angle(desired_heading - state_t[2])
        distance = torch.linalg.norm(delta)
        v = drive_gain * distance * torch.cos(heading_error)
        omega = turn_gain * heading_error
        return self.clamp_command(torch.stack([v, omega]))


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    if isinstance(value, dict):
        return torch.tensor([value["x"], value["y"], value["yaw"]], dtype=torch.float32)
    if all(hasattr(value, name) for name in ("x", "y", "yaw")):
        return torch.tensor([value.x, value.y, value.yaw], dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)
