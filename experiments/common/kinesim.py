"""Lightweight 2D holonomic multi-agent kinematic simulator (no Isaac).

Purpose: run the safety-filter experiments (RQ2-RQ4) against the SAME
first-order holonomic model the DAM guard assumes, so filter behaviour can be
measured cheaply over hundreds of seeded episodes. Isaac-in-the-loop validation
of headline results comes later; this is the statistics engine.

Each agent: pose [x, y, yaw], goal (gx, gy), group, priority. Per tick:
    nominal P-controller -> safety filter -> first-order holonomic integration.
Static obstacle "agents" never move and are exposed to filters as neighbours.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class AgentSpec:
    aid: str
    x: float
    y: float
    yaw: float
    gx: float
    gy: float
    group: str = "G0"
    priority: float = 1.0
    static: bool = False  # static obstacle: never moves, never filtered
    # Route waypoints visited (in order) before heading to the goal. Part of
    # the TASK definition, shared by every method: the reactive filter is
    # deliberately myopic and will not route around walls (S3 needs the raw
    # command stream to aim through the gap, or every method just piles into
    # the wall's local minimum).
    waypoints: tuple = ()


@dataclass
class AgentState:
    spec: AgentSpec
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    wvx: float = 0.0   # last world-frame velocity (for velocity-aware filters)
    wvy: float = 0.0
    done: bool = False
    done_time: float | None = None
    path_len: float = 0.0
    wp_idx: int = 0    # next waypoint to visit (== len(waypoints) → head to goal)

    @classmethod
    def from_spec(cls, s: AgentSpec) -> "AgentState":
        return cls(spec=s, x=s.x, y=s.y, yaw=s.yaw)


@dataclass
class SimConfig:
    dt: float = 0.02          # 50 Hz, matches the demo11 control rate
    vmax: float = 1.5         # matches HolonomicSolver.max_v
    wmax: float = 2.0         # matches HolonomicSolver.max_omega
    k_lin: float = 1.2
    k_yaw: float = 2.0
    goal_tol: float = 0.3
    waypoint_tol: float = 0.6  # looser than goal_tol: waypoints are route hints
    max_time: float = 60.0
    # -- robustness knobs (RQ4) --------------------------------------------
    exec_noise_std: float = 0.0    # E4.1: gaussian noise on the EXECUTED world velocity (m/s)
    obs_noise_std: float = 0.0     # E4.2: gaussian noise on observed neighbour positions (m)
    obs_delay_steps: int = 0       # E4.2: neighbour observations delayed by k control steps
    cmd_delay_steps: int = 0       # E1.3: ACTUATION latency — the executed command is the
    #                                filtered command from k control steps ago (transport
    #                                delay between compute and actuation; 0 = instantaneous)
    diff_drive: bool = False       # E1.2: differential-drive base — the nominal controller
    #                                emits [v, 0, omega] (heading-first unicycle P-control;
    #                                no strafe). Pair with a max_vy=0 guard so the QP cannot
    #                                buy safety laterally either.
    # -- perception budget ---------------------------------------------------
    # Only the nearest K static (wall) points within this range are fed to the
    # filter: along a straight wall the nearest points geometrically dominate
    # the farther ones, and unbounded wall chains blow up the QP constraint
    # count for zero behavioural difference.
    max_static_neighbors: int = 6
    static_cull_dist: float = 3.5


@dataclass
class StepRecord:
    t: float
    poses: dict            # aid -> (x, y, yaw)
    raw_cmds: dict         # aid -> (vx, vy, om)
    safe_cmds: dict        # aid -> (vx, vy, om)
    filter_s: dict         # aid -> wall seconds spent in the filter
    decisions: dict = field(default_factory=dict)  # aid -> filter info dict


class KineSim:
    """Steps a squad of holonomic agents through a safety filter to their goals."""

    def __init__(self, specs: list[AgentSpec], safety_filter, cfg: SimConfig | None = None,
                 rng_seed: int = 0):
        self.cfg = cfg or SimConfig()
        self.filter = safety_filter
        self.agents = {s.aid: AgentState.from_spec(s) for s in specs}
        self._rng = random.Random(rng_seed)   # noise source (exec/obs), episode-reproducible
        self._hist: deque = deque(maxlen=max(1, self.cfg.obs_delay_steps + 1))
        self._obs: dict = {}                   # aid -> observed (x, y, wvx, wvy, yaw)
        # E1.3 actuation delay: per-agent FIFO of filtered commands; the oldest
        # entry is what actually reaches the base this step (zeros during warmup).
        self._cmd_q: dict = {}                 # aid -> deque of safe cmds

    # -- nominal controller: body-frame P-control toward the goal ------------
    def nominal(self, ag: AgentState) -> tuple[float, float, float]:
        c = self.cfg
        wps = ag.spec.waypoints
        while ag.wp_idx < len(wps):
            wx, wy = wps[ag.wp_idx]
            if math.hypot(wx - ag.x, wy - ag.y) < c.waypoint_tol:
                ag.wp_idx += 1
                continue
            # Also advance once the dog has CROSSED the waypoint's plane
            # (normal = incoming route direction): congestion can shove a dog
            # past a waypoint it never got within tol of, and without this it
            # would forever command BACK into the crowd behind it.
            px_, py_ = wps[ag.wp_idx - 1] if ag.wp_idx > 0 else (ag.spec.x, ag.spec.y)
            dx_, dy_ = wx - px_, wy - py_
            if (ag.x - wx) * dx_ + (ag.y - wy) * dy_ > 0.0:
                ag.wp_idx += 1
                continue
            break
        if ag.wp_idx < len(wps):
            tx, ty = wps[ag.wp_idx]
        else:
            tx, ty = ag.spec.gx, ag.spec.gy
        dx, dy = tx - ag.x, ty - ag.y
        dist = math.hypot(dx, dy)
        if dist < c.goal_tol:
            return (0.0, 0.0, 0.0)
        if c.diff_drive:
            # Unicycle P-control: turn toward the target, drive forward scaled by
            # heading alignment (no reverse, no strafe) — standard diff-drive
            # go-to-goal. The cos gating stops it driving while facing away.
            err = wrap_angle(math.atan2(dy, dx) - ag.yaw)
            om = max(-c.wmax, min(c.wmax, c.k_yaw * err))
            v = min(c.vmax, c.k_lin * dist) * max(0.0, math.cos(err))
            return (v, 0.0, om)
        # Desired world velocity, capped, rotated into the body frame.
        scale = min(1.0, c.vmax / max(c.k_lin * dist, 1e-9)) * c.k_lin
        wvx, wvy = scale * dx, scale * dy
        cs, sn = math.cos(ag.yaw), math.sin(ag.yaw)
        bvx = wvx * cs + wvy * sn
        bvy = -wvx * sn + wvy * cs
        om = max(-c.wmax, min(c.wmax, c.k_yaw * wrap_angle(math.atan2(dy, dx) - ag.yaw)))
        return (bvx, bvy, om)

    def _snapshot_observations(self):
        """Refresh what agents OBSERVE this step: possibly delayed + noisy states.

        Ground truth is snapshotted per step; the observation used is the one
        ``obs_delay_steps`` ticks old, with fresh position noise per step. One
        shared observation per observed agent (all observers see the same value).
        """
        c = self.cfg
        truth = {aid: (ag.x, ag.y, ag.wvx, ag.wvy, ag.yaw) for aid, ag in self.agents.items()}
        self._hist.append(truth)
        # deque maxlen = obs_delay_steps + 1, so the oldest entry IS the k-step-old
        # observation once the buffer fills (and the best available during warmup).
        delayed = self._hist[0]
        self._obs = {}
        for aid, (x, y, wvx, wvy, yaw) in delayed.items():
            if c.obs_noise_std > 0.0 and not self.agents[aid].spec.static:
                x += self._rng.gauss(0.0, c.obs_noise_std)
                y += self._rng.gauss(0.0, c.obs_noise_std)
            self._obs[aid] = (x, y, wvx, wvy, yaw)

    def _neighbors_of(self, aid: str) -> list[tuple]:
        """(x, y, priority, same_group, wvx, wvy, yaw, static) for every OTHER agent.

        Positions/velocities come from the (possibly delayed/noisy) observation
        snapshot; an agent's OWN pose stays ground truth (odometry >> inter-robot
        perception). The trailing ``static`` flag (1.0 = wall point) is extra
        context for baseline filters; Go2DAMWrapper reads indices 0-6 only.
        """
        me = self.agents[aid]
        out = []
        statics = []
        for oid, ag in self.agents.items():
            if oid == aid:
                continue
            ox, oy, owvx, owvy, oyaw = self._obs[oid]
            if ag.spec.static:
                d = math.hypot(me.x - ox, me.y - oy)
                if d <= self.cfg.static_cull_dist:
                    statics.append((d, (ox, oy, ag.spec.priority, 0.0, 0.0, 0.0, oyaw, 1.0)))
                continue
            same = 1.0 if ag.spec.group == me.spec.group else 0.0
            out.append((ox, oy, ag.spec.priority, same, owvx, owvy, oyaw, 0.0))
        statics.sort(key=lambda t: t[0])
        out.extend(t[1] for t in statics[: self.cfg.max_static_neighbors])
        return out

    def run(self, recorder=None) -> list[StepRecord]:
        c = self.cfg
        steps = int(round(c.max_time / c.dt))
        trace: list[StepRecord] = []
        for k in range(steps):
            t = k * c.dt
            self._snapshot_observations()
            rec = StepRecord(t=t, poses={}, raw_cmds={}, safe_cmds={}, filter_s={})
            # 1) decide every command from the SAME snapshot (synchronous update)
            pending = {}
            for aid, ag in self.agents.items():
                if ag.spec.static:
                    continue
                raw = self.nominal(ag)
                t0 = time.perf_counter()
                safe, info = self.filter.filter(
                    aid, raw, (ag.x, ag.y, ag.yaw),
                    self._neighbors_of(aid), self_priority=ag.spec.priority,
                )
                rec.filter_s[aid] = time.perf_counter() - t0
                pending[aid] = (raw, safe)
                rec.raw_cmds[aid] = raw
                rec.safe_cmds[aid] = safe
                rec.decisions[aid] = info
            # 2) integrate (first-order holonomic, same as the guard's model)
            for aid, (raw, safe) in pending.items():
                ag = self.agents[aid]
                if c.cmd_delay_steps > 0:
                    q = self._cmd_q.setdefault(
                        aid, deque([(0.0, 0.0, 0.0)] * c.cmd_delay_steps,
                                   maxlen=c.cmd_delay_steps + 1))
                    q.append(safe)
                    safe = q.popleft()
                vx, vy, om = safe
                vx = max(-c.vmax, min(c.vmax, vx))
                vy = max(-c.vmax, min(c.vmax, vy))
                om = max(-c.wmax, min(c.wmax, om))
                cs, sn = math.cos(ag.yaw), math.sin(ag.yaw)
                wvx = vx * cs - vy * sn
                wvy = vx * sn + vy * cs
                if c.exec_noise_std > 0.0:  # E4.1: policy tracks the command imperfectly
                    wvx += self._rng.gauss(0.0, c.exec_noise_std)
                    wvy += self._rng.gauss(0.0, c.exec_noise_std)
                nx, ny = ag.x + wvx * c.dt, ag.y + wvy * c.dt
                ag.path_len += math.hypot(nx - ag.x, ny - ag.y)
                ag.x, ag.y, ag.yaw = nx, ny, wrap_angle(ag.yaw + om * c.dt)
                ag.wvx, ag.wvy = wvx, wvy
                if not ag.done and math.hypot(ag.spec.gx - ag.x, ag.spec.gy - ag.y) < c.goal_tol:
                    ag.done, ag.done_time = True, t
            for aid, ag in self.agents.items():
                rec.poses[aid] = (ag.x, ag.y, ag.yaw)
            trace.append(rec)
            if recorder is not None:
                recorder.step(self, rec)
            if all(ag.done for ag in self.agents.values() if not ag.spec.static):
                break
        return trace
