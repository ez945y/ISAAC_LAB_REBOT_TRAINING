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
    """ORCA baseline (B2) on the official RVO2 bindings (sybrenstuvel/
    Python-RVO2 wrapping the vetted RVO2 C++ library) — NOT hand-rolled.

    Adapter: one throwaway PyRVOSimulator per filter call — self agent plus
    every neighbour — stepped once; the ORCA-optimal velocity becomes the
    command. Notes for the thesis table:
      * disc body model, radius = min_dist/2 per agent (centre separation
        0.7 m, same floor as DAM) — expect disc_in-style true-capsule dips
        (E3.4); ORCA has no capsule support.
      * wall points enter as zero-max-speed agents (RVO2 polygon obstacles
        don't fit our point-chain walls); ORCA's reciprocity halves the
        avoidance against them — a known conservative-optimist mismatch.
      * no priority concept; omega passes through untouched.
    Self velocity is approximated by the raw command (world frame): the
    holonomic dog tracks its command within one step.
    """

    name = "orca"

    def __init__(self, radius: float = 0.35, max_speed: float = 1.5,
                 neighbor_dist: float = 2.5, time_horizon: float = 2.0,
                 dt: float = 0.02):
        import rvo2  # vetted bindings; venv-only
        self._rvo2 = rvo2
        self.radius = radius
        self.max_speed = max_speed
        self.neighbor_dist = neighbor_dist
        self.time_horizon = time_horizon
        self.dt = dt

    def filter(self, aid, cmd, pose, neighbors, self_priority=1.0):
        import math
        vx, vy, om = cmd
        cx, cy, yaw = pose
        cs, sn = math.cos(yaw), math.sin(yaw)
        wvx, wvy = vx * cs - vy * sn, vx * sn + vy * cs   # pref velocity, world
        sim = self._rvo2.PyRVOSimulator(
            self.dt, self.neighbor_dist, 10, self.time_horizon,
            self.time_horizon, self.radius, self.max_speed)
        me = sim.addAgent((cx, cy))
        sim.setAgentVelocity(me, (wvx, wvy))
        sim.setAgentPrefVelocity(me, (wvx, wvy))
        for nb in neighbors:
            px, py = nb[0], nb[1]
            nvx = nb[4] if len(nb) >= 6 else 0.0
            nvy = nb[5] if len(nb) >= 6 else 0.0
            static = len(nb) >= 8 and nb[7] >= 0.5
            a = sim.addAgent((px, py))
            if static:
                sim.setAgentMaxSpeed(a, 0.0)
                sim.setAgentVelocity(a, (0.0, 0.0))
                sim.setAgentPrefVelocity(a, (0.0, 0.0))
            else:
                sim.setAgentVelocity(a, (nvx, nvy))
                sim.setAgentPrefVelocity(a, (nvx, nvy))
        sim.doStep()
        nwx, nwy = sim.getAgentVelocity(me)
        bvx = nwx * cs + nwy * sn
        bvy = -nwx * sn + nwy * cs
        changed = abs(bvx - vx) > 1e-5 or abs(bvy - vy) > 1e-5
        return (bvx, bvy, om), {"decision": "CLAMP" if changed else "PASS"}


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
