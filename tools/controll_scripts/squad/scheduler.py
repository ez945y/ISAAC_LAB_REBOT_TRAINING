# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Squad model + Dispatcher — the scheduling layer (pure logic, no Isaac).

``Squad`` owns the agents and the current grouping. ``Dispatcher`` binds a
:class:`Mission` to each group, ticks them, and performs **regrouping** (e.g.
2×3 → 3×2). Everything here talks only to the :class:`RobotAgent` ABI, so the
whole scheduler is unit-testable headless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .mission import Mission
from .robot_agent import RobotAgent


def split_evenly(agent_ids: list[str], group_size: int, prefix: str = "G") -> dict[str, list[str]]:
    """Chunk ``agent_ids`` into groups of ``group_size`` -> ``{G0: [...], ...}``."""
    out: dict[str, list[str]] = {}
    for i in range(0, len(agent_ids), group_size):
        out[f"{prefix}{i // group_size}"] = agent_ids[i:i + group_size]
    return out


@dataclass
class Squad:
    agents: dict[str, RobotAgent]
    groups: dict[str, list[str]] = field(default_factory=dict)

    def set_groups(self, groups: dict[str, list[str]]) -> None:
        self.groups = {g: list(ids) for g, ids in groups.items()}
        # stamp each agent with its group (and clear orphans)
        for a in self.agents.values():
            a.set_group(None)
        for gid, ids in self.groups.items():
            for aid in ids:
                self.agents[aid].set_group(gid)

    def members(self, group_id: str) -> list[RobotAgent]:
        return [self.agents[aid] for aid in self.groups.get(group_id, [])]

    @property
    def agent_ids(self) -> list[str]:
        return list(self.agents.keys())


class Dispatcher:
    """Assigns missions to groups, ticks them, and handles regrouping."""

    def __init__(self, squad: Squad) -> None:
        self.squad = squad
        self.missions: dict[str, Mission] = {}

    def assign(self, group_id: str, mission: Mission) -> None:
        self.missions[group_id] = mission
        mission.on_assigned(self.squad.members(group_id))

    def regroup(self, groups: dict[str, list[str]]) -> None:
        """Re-partition the squad. Clears missions + per-agent targets so the
        next assignment starts cleanly."""
        for a in self.squad.agents.values():
            a.halt()
            a.target = None
        self.missions.clear()
        self.squad.set_groups(groups)

    def update(self, dt: float) -> dict[str, bool]:
        """Tick every group's mission. Returns ``{group_id: done}``."""
        return {gid: m.update(self.squad.members(gid), dt) for gid, m in self.missions.items()}

    def all_done(self, dt: float = 0.0) -> bool:
        status = self.update(dt)
        return bool(status) and all(status.values())
