# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Missions — the decoupled "action" layer.

A mission is a self-contained task assigned to one group. It only touches the
:class:`RobotAgent` ABI (``move_to`` / ``arrived`` / ``get_state``); it knows
nothing about the scheduler internals, the simulator, or the robot model. New
behaviours (patrol, escort, search) are added by subclassing :class:`Mission`
without touching the scheduler.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from .formation import Formation, slots
from .robot_agent import RobotAgent


class Mission(ABC):
    """A task bound to a group of agents."""

    name: str = "mission"

    @abstractmethod
    def update(self, agents: list[RobotAgent], dt: float) -> bool:
        """Advance the mission one tick. Return True when complete."""

    def on_assigned(self, agents: list[RobotAgent]) -> None:  # optional hook
        """Called once when the mission is bound to a group."""


class FormationMoveToArea(Mission):
    """Move the group, in formation, to a target area and hold.

    Assigns each member a formation slot around ``area_xy`` (heading points from
    the group's current centroid toward the area) and reports done once every
    member has arrived at its slot.
    """

    name = "formation_move_to_area"

    def __init__(
        self,
        area_xy: tuple[float, float],
        *,
        shape: Formation = Formation.WEDGE,
        spacing: float = 1.2,
    ) -> None:
        self.area_xy = (float(area_xy[0]), float(area_xy[1]))
        self.shape = shape
        self.spacing = spacing

    def _heading(self, agents: list[RobotAgent]) -> float:
        cx = sum(a.get_state().x for a in agents) / len(agents)
        cy = sum(a.get_state().y for a in agents) / len(agents)
        return math.atan2(self.area_xy[1] - cy, self.area_xy[0] - cx)

    def on_assigned(self, agents: list[RobotAgent]) -> None:
        self._assign_slots(agents)

    def _assign_slots(self, agents: list[RobotAgent]) -> None:
        heading = self._heading(agents)
        pts = slots(self.area_xy, heading, len(agents),
                    shape=self.shape, spacing=self.spacing)
        # Match agents to slots by lateral order so paths don't cross. Zipping in
        # group order makes a column whose order is opposite the slot order swap
        # ends — the two crossing dogs collide and one never reaches its slot.
        # Projecting both onto the formation's lateral axis and pairing in that
        # order gives a non-crossing (monotone) assignment for any start layout.
        lat = (math.cos(heading + math.pi / 2.0), math.sin(heading + math.pi / 2.0))

        def _lateral(p: tuple[float, float]) -> float:
            return p[0] * lat[0] + p[1] * lat[1]

        agents_sorted = sorted(agents, key=lambda a: _lateral((a.get_state().x, a.get_state().y)))
        pts_sorted = sorted(pts, key=_lateral)
        for agent, slot in zip(agents_sorted, pts_sorted):
            agent.move_to(slot)

    def update(self, agents: list[RobotAgent], dt: float) -> bool:
        if not agents:
            return True
        # Re-issue slots if a member has no target yet (e.g. just regrouped).
        if any(a.target is None for a in agents):
            self._assign_slots(agents)
        return all(a.arrived for a in agents)
