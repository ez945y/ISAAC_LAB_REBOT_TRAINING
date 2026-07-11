# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""RobotAgent — the ABI between high-level dispatch and low-level locomotion.

The scheduler and missions only ever touch :class:`RobotAgent` (``move_to`` /
``get_state`` / ``set_group`` / ``step``). They never import Isaac or know the
robot is a Go2. Swapping the embodiment means providing a different
:class:`LocomotionBackend` — nothing above this line changes.

``RobotAgent.step`` turns the current ``move_to`` target into a body-frame
velocity command ``[v_x, v_y, w_z]`` and hands it to the backend; the velocity
math (:func:`velocity_command`) is a pure function so it is unit-testable
without a simulator.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class AgentState:
    x: float
    y: float
    yaw: float


@runtime_checkable
class LocomotionBackend(Protocol):
    """What an embodiment must provide for an agent to walk (e.g. a Go2 policy)."""

    def get_pose(self) -> tuple[float, float, float]:
        """Return ``(x, y, yaw)`` in the world frame."""

    def apply(self, dt: float, command: tuple[float, float, float]) -> None:
        """Apply a body-frame velocity command ``(v_x, v_y, w_z)`` for one step."""


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def velocity_command(
    pose: tuple[float, float, float],
    target: tuple[float, float],
    *,
    arrive_tol: float = 0.25,
    max_v: float = 1.5,
    max_w: float = 2.0,
    k_v: float = 2.2,
    k_w: float = 2.5,
    turn_creep: float = 0.25,
) -> tuple[tuple[float, float, float], bool]:
    """Pure unicycle controller: pose + target -> (``[v_x, v_y, w_z]``, arrived).

    Forward speed scales with heading alignment (turn toward the target, then go),
    so the dog does not sprint sideways. But a legged RL policy turns far better
    while walking than spinning in place, so when the dog is pointed away from the
    target we keep a small forward ``turn_creep`` (fraction of the desired speed)
    instead of zero — it arcs around toward alignment rather than stalling.
    """
    x, y, yaw = pose
    dx, dy = target[0] - x, target[1] - y
    dist = math.hypot(dx, dy)
    if dist < arrive_tol:
        return (0.0, 0.0, 0.0), True

    heading_err = _wrap(math.atan2(dy, dx) - yaw)
    w = max(-max_w, min(max_w, k_w * heading_err))
    align = math.cos(heading_err)
    v_des = min(max_v, k_v * dist)
    # Scale by alignment when roughly facing the target; otherwise creep forward so
    # the policy can execute a turning gait (it cannot turn well at v=0).
    v = v_des * max(turn_creep, align)
    return (v, 0.0, w), False


@dataclass
class RobotAgent:
    """Stable handle the scheduler/missions drive (the ABI)."""

    agent_id: str
    backend: LocomotionBackend
    group_id: str | None = None
    target: tuple[float, float] | None = None
    arrived: bool = False
    # tuning forwarded to velocity_command
    params: dict = field(default_factory=dict)
    # Optional safety filter applied to the command each step BEFORE it reaches the
    # backend: ``(cmd, pose) -> cmd`` where cmd is ``(vx, vy, wz)`` and pose is
    # ``(x, y, yaw)``. Stays a plain callable so the ABI carries no Isaac/DAM import
    # (e.g. the Go2 squad injects the inter-dog DAM guard here).
    safety: Callable[[tuple[float, float, float], tuple[float, float, float]],
                     tuple[float, float, float]] | None = None

    # -- commands the scheduler / missions use --------------------------------

    def move_to(self, xy: tuple[float, float]) -> None:
        self.target = (float(xy[0]), float(xy[1]))
        self.arrived = False

    def halt(self) -> None:
        self.target = None
        self.arrived = True

    def set_group(self, group_id: str | None) -> None:
        self.group_id = group_id

    def get_state(self) -> AgentState:
        x, y, yaw = self.backend.get_pose()
        return AgentState(x, y, yaw)

    # -- driven every physics step by the runtime -----------------------------

    def step(self, dt: float) -> tuple[float, float, float]:
        if self.target is None:
            cmd = (0.0, 0.0, 0.0)
            if self.safety is not None:
                cmd = tuple(self.safety(cmd, self.backend.get_pose()))
            self.backend.apply(dt, cmd)
            return cmd
        cmd, reached = velocity_command(self.backend.get_pose(), self.target, **self.params)
        # Latch arrival: a legged robot orbits/jitters around its slot rather than
        # parking exactly on it, so `reached` flickers. Once a slot is reached, stay
        # arrived until a new `move_to` — otherwise group "all arrived" never latches
        # in a single tick. The command still nudges the dog back toward the slot.
        if reached:
            self.arrived = True
        # Safety filter (e.g. inter-dog DAM) gets the LAST word before the backend.
        if self.safety is not None:
            cmd = tuple(self.safety(cmd, self.backend.get_pose()))
        self.backend.apply(dt, cmd)
        return cmd
