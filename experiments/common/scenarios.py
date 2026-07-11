"""Seeded scenario library S1-S5 (shared by every RQ experiment).

Each generator returns a list of AgentSpec whose starts/goals are jittered by a
seeded RNG, so every (scenario, seed) pair is a reproducible episode.

S1 crossing      two groups on orthogonal paths          -> priority, coordination
S2 head-on       two dogs swap positions                 -> deadlock, swirl, TTC
S3 bottleneck    N dogs through a gap between wall rows  -> soft/hard layering, congestion
S4 corridor      two dogs pass in a narrow corridor      -> capsule vs disc geometry
S5 random cruise N dogs, random starts/goals in an arena -> long-run statistics
Walls are chains of static point-agents (spacing ~ capsule width) until the
guard supports true segment obstacles — noted in EXPERIMENTS.md.
"""

from __future__ import annotations

import math
import random

from .kinesim import AgentSpec


def _wall(prefix: str, x0: float, y0: float, x1: float, y1: float, spacing: float = 0.35) -> list[AgentSpec]:
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / spacing) + 1)
    out = []
    for i in range(n):
        f = i / (n - 1)
        out.append(AgentSpec(
            aid=f"{prefix}{i}", x=x0 + f * (x1 - x0), y=y0 + f * (y1 - y0), yaw=0.0,
            gx=x0 + f * (x1 - x0), gy=y0 + f * (y1 - y0),
            group="WALL", priority=1e6, static=True,
        ))
    return out


def s1_crossing(seed: int, n_per_group: int = 2, half: float = 5.0,
                pri_g0: float = 1.0, pri_g1: float = 1.0) -> list[AgentSpec]:
    rng = random.Random(seed)
    specs = []
    for i in range(n_per_group):  # G0: west -> east
        y = (i - (n_per_group - 1) / 2) * 1.6 + rng.uniform(-0.3, 0.3)
        specs.append(AgentSpec(f"G0_{i}", -half, y, 0.0, half, y + rng.uniform(-0.3, 0.3),
                               group="G0", priority=pri_g0))
    for i in range(n_per_group):  # G1: south -> north
        x = (i - (n_per_group - 1) / 2) * 1.6 + rng.uniform(-0.3, 0.3)
        specs.append(AgentSpec(f"G1_{i}", x, -half, math.pi / 2, x + rng.uniform(-0.3, 0.3), half,
                               group="G1", priority=pri_g1))
    return specs


def s2_headon(seed: int, gap: float = 8.0, jitter: float = 0.15) -> list[AgentSpec]:
    """Two dogs swap places. ``jitter`` sets the lateral start offset: 0.15 m is
    the mild default; 0.0 is the PERFECT standoff (deadlock stress test — the
    sidestep-cheap QP alone cannot break exact symmetry, only swirl can)."""
    rng = random.Random(seed)
    jy = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
    return [
        AgentSpec("A", -gap / 2, jy, 0.0, gap / 2, -jy, group="G0"),
        AgentSpec("B", gap / 2, -jy, math.pi, -gap / 2, jy, group="G1"),
    ]


def s3_bottleneck(seed: int, n: int = 4, gap_width: float = 1.6, depth: float = 4.0,
                  min_start_sep: float = 1.5) -> list[AgentSpec]:
    rng = random.Random(seed)
    starts: list[tuple] = []
    for _ in range(n):  # all start south of the gap, min separation between starts
        for _ in range(500):
            p = (rng.uniform(-2.5, 2.5), -depth - rng.uniform(0.0, 2.0))
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= min_start_sep for q in starts):
                break
        else:
            raise RuntimeError(f"s3_bottleneck: could not place {n} starts {min_start_sep} m apart")
        starts.append(p)
    specs = [
        AgentSpec(f"D{i}", sx, sy, math.pi / 2,
                  rng.uniform(-1.5, 1.5), depth + rng.uniform(-0.5, 0.5), group="G0")
        for i, (sx, sy) in enumerate(starts)
    ]
    g = gap_width / 2
    specs += _wall("WL", -6.0, 0.0, -g, 0.0)
    specs += _wall("WR", g, 0.0, 6.0, 0.0)
    return specs


def s4_corridor(seed: int, width: float = 2.2, length: float = 8.0) -> list[AgentSpec]:
    rng = random.Random(seed)
    jy = rng.uniform(-0.1, 0.1)
    h = width / 2
    specs = [
        AgentSpec("A", -length / 2, jy, 0.0, length / 2, jy, group="G0"),
        AgentSpec("B", length / 2, -jy, math.pi, -length / 2, -jy, group="G1"),
    ]
    specs += _wall("WT", -length / 2, h, length / 2, h)
    specs += _wall("WB", -length / 2, -h, length / 2, -h)
    return specs


def s5_cruise(seed: int, n: int = 6, half: float = 6.0, min_sep: float = 2.0) -> list[AgentSpec]:
    rng = random.Random(seed)

    def sample(existing):
        for _ in range(200):
            p = (rng.uniform(-half, half), rng.uniform(-half, half))
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= min_sep for q in existing):
                return p
        return p  # dense fallback

    starts, goals = [], []
    for _ in range(n):
        starts.append(sample(starts))
        goals.append(sample(goals))
    return [
        AgentSpec(f"D{i}", sx, sy, rng.uniform(-math.pi, math.pi), gx, gy,
                  group=f"G{i % 2}")
        for i, ((sx, sy), (gx, gy)) in enumerate(zip(starts, goals))
    ]


SCENARIOS = {
    "S1": s1_crossing,
    "S2": s2_headon,
    "S3": s3_bottleneck,
    "S4": s4_corridor,
    "S5": s5_cruise,
}
