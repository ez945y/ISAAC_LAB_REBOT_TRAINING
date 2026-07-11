"""Safety-filter implementations behind one interface (the RQ2 method column).

    filter(aid, cmd, pose, neighbors, self_priority) -> (safe_cmd, info)

cmd/safe_cmd: body-frame (vx, vy, omega). pose: world (x, y, yaw).
neighbors: [(x, y, priority, same_group, wvx, wvy, yaw), ...] world frame.

B0 RawFilter   pass-through (no safety)
B1 StopFilter  threshold stop with resume hysteresis
B2 OrcaFilter  TODO (needs a vetted ORCA implementation; see EXPERIMENTS.md)
B3 DamFilter   the real Go2DAMWrapper (torch/osqp/dam -> Isaac machine)
"""

from __future__ import annotations

import math


class RawFilter:
    name = "raw"

    def filter(self, aid, cmd, pose, neighbors, self_priority=1.0):
        return cmd, {"decision": "PASS"}


class StopFilter:
    """Full stop while any neighbour is inside stop_dist (centre distance).

    Resumes only once every neighbour is beyond resume_dist — the hysteresis
    prevents rapid stop/go chatter right at the threshold. Static obstacles
    (neighbour tuple index 7 == 1.0) use the smaller wall thresholds: a
    threshold-stop safety layer reacts to agents, while walls are the
    planner's job — it only refuses to actually touch one.
    """

    name = "stop"

    def __init__(self, stop_dist: float = 1.5, resume_dist: float = 1.8,
                 wall_stop: float = 0.45, wall_resume: float = 0.6):
        self.stop_dist = stop_dist
        self.resume_dist = resume_dist
        self.wall_stop = wall_stop
        self.wall_resume = wall_resume
        self._latched: set[str] = set()

    def filter(self, aid, cmd, pose, neighbors, self_priority=1.0):
        x, y, _ = pose
        near_dyn, near_wall = math.inf, math.inf
        for n in neighbors:
            d = math.hypot(x - n[0], y - n[1])
            if len(n) >= 8 and n[7] >= 0.5:
                near_wall = min(near_wall, d)
            else:
                near_dyn = min(near_dyn, d)
        if aid in self._latched:
            if near_dyn > self.resume_dist and near_wall > self.wall_resume:
                self._latched.discard(aid)
        elif near_dyn < self.stop_dist or near_wall < self.wall_stop:
            self._latched.add(aid)
        if aid in self._latched:
            return (0.0, 0.0, 0.0), {"decision": "STOP", "nearest": min(near_dyn, near_wall)}
        return cmd, {"decision": "PASS", "nearest": min(near_dyn, near_wall)}


class OrcaFilter:
    """Placeholder for the ORCA baseline (B2). Use a vetted implementation
    (e.g. the RVO2 python bindings) rather than a hand-rolled one, so baseline
    numbers are unimpeachable."""

    name = "orca"

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "ORCA baseline not wired up yet — install RVO2 bindings and adapt here."
        )


class DamFilter:
    """The real DAM guard (Go2DAMWrapper). Imports torch/dam lazily so machines
    without them can still run raw/stop experiments."""

    name = "dam"

    def __init__(self, stackfile: str = "go2_squad_safety.yaml", device: str = "cpu"):
        import sys
        from pathlib import Path
        tools = Path(__file__).resolve().parents[2] / "tools"
        if str(tools) not in sys.path:
            sys.path.insert(0, str(tools))
        import torch  # noqa: F401
        from controll_scripts.safety import Go2DAMWrapper
        self._torch = torch
        self._wrap = Go2DAMWrapper(stackfile, device=device)

    def filter(self, aid, cmd, pose, neighbors, self_priority=1.0):
        t = self._torch
        c = t.tensor([list(cmd)], dtype=t.float32)
        s = t.tensor([list(pose)], dtype=t.float32)
        safe = self._wrap.filter(c, s, neighbors, self_priority=self_priority)
        return tuple(safe[0].tolist()), {
            "decision": self._wrap.last_decision,
            "delta": self._wrap.last_delta,
        }

    def close(self):
        self._wrap.close()


class FilterRouter:
    """Per-agent filter routing (E4.4 non-cooperative mix: some agents run a
    different filter — e.g. raw — than the rest)."""

    name = "router"

    def __init__(self, default, overrides: dict | None = None):
        self.default = default
        self.overrides = overrides or {}

    def filter(self, aid, cmd, pose, neighbors, self_priority=1.0):
        f = self.overrides.get(aid, self.default)
        return f.filter(aid, cmd, pose, neighbors, self_priority=self_priority)

    def close(self):
        for f in {id(x): x for x in [self.default, *self.overrides.values()]}.values():
            if hasattr(f, "close"):
                f.close()


def make_filter(name: str, **kwargs):
    if name == "pydam":
        from .pydam import PyDamFilter
        return PyDamFilter(**kwargs)
    table = {"raw": RawFilter, "stop": StopFilter, "orca": OrcaFilter, "dam": DamFilter}
    if name not in table:
        raise ValueError(f"unknown filter '{name}' (choose from raw, stop, orca, dam, pydam)")
    return table[name](**kwargs)
