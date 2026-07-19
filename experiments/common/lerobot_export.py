"""E5 (RQ5) support: decision-time frame capture + LeRobot export + events.

RecordingFilter wraps any safety filter and captures, at DECISION time (the
same tick, same observation snapshot the guard saw), everything a per-agent
frame needs: the ego-centric observation vector, the raw proposal, the
verified command and the guard telemetry. One kinesim episode therefore
yields one LeRobot episode PER AGENT — the guard is a per-agent filter, so
the reusable unit of data is the per-agent (observation -> action) stream,
not the whole squad state.

export_pool() writes a pool of collected episodes as a LeRobotDataset
(v3 layout: parquet data + jsonl meta, no videos) plus two sidecars in meta/:
    episode_manifest.jsonl   episode_index -> (scenario, seed, method, aid)
    boundary_events.jsonl    RSMF-style boundary-event records with context
                             windows (the RQ5 failure-sample log)

lerobot / numpy are imported lazily: the fake-data smoke path (raw/stop,
--export none) stays stdlib-only, per the ground rules.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import seg_seg_closest

# -- ego observation layout ---------------------------------------------------
EGO_K = 6           # nearest neighbours encoded (dynamic + culled statics)
PAD_DIST = 10.0     # "far away, dead ahead" padding for missing neighbours
OBS_DIM = 6 + 6 * EGO_K

OBS_NAMES = (
    ["tgt_bx", "tgt_by", "goal_bx", "goal_by", "v_bx", "v_by"]
    + [f"n{i}_{f}" for i in range(EGO_K)
       for f in ("dx_b", "dy_b", "dvx_b", "dvy_b", "same_group", "static")]
)

DECISION_CODE = {"PASS": 0, "CLAMP": 1, "STOP": 2, "REJECT": 3, "FAULT": 4}

DOG_FLOOR = 0.7      # guard hard floor (dog-dog capsule), = MetricsConfig.min_dist
WALL_FLOOR = 0.35    # dog-wall clearance floor, = MetricsConfig.wall_dist
CAPSULE_HALF = 0.25  # dog body half-length (walls are points)

INTERVENE_EPS = 1e-5   # |safe - raw| above this counts as an intervention
EVENT_WINDOW_S = 0.5   # context window kept before/after each event


def ego_observation(ag, neighbors) -> list[float]:
    """42-dim body-frame ego view: current target, final goal, own velocity,
    then the EGO_K nearest neighbours as (rel pos, rel vel, same_group, static).

    ``ag`` is the kinesim AgentState AT decision time (wp_idx already advanced
    by nominal() this tick); ``neighbors`` is exactly the observed neighbour
    list the filter received.
    """
    x, y, yaw = ag.x, ag.y, ag.yaw
    cs, sn = math.cos(yaw), math.sin(yaw)

    def to_body(wx: float, wy: float) -> tuple[float, float]:
        return (wx * cs + wy * sn, -wx * sn + wy * cs)

    wps = ag.spec.waypoints
    tx, ty = wps[ag.wp_idx] if ag.wp_idx < len(wps) else (ag.spec.gx, ag.spec.gy)
    obs = list(to_body(tx - x, ty - y))
    obs += to_body(ag.spec.gx - x, ag.spec.gy - y)
    obs += to_body(ag.wvx, ag.wvy)

    ranked = sorted(neighbors, key=lambda n: math.hypot(n[0] - x, n[1] - y))
    for i in range(EGO_K):
        if i < len(ranked):
            n = ranked[i]
            obs += to_body(n[0] - x, n[1] - y)
            obs += to_body(n[4] - ag.wvx, n[5] - ag.wvy)
            obs += [n[3], n[7] if len(n) >= 8 else 0.0]
        else:
            obs += [PAD_DIST, 0.0, 0.0, 0.0, 0.0, 0.0]
    return obs


@dataclass
class Frame:
    obs: list[float]
    raw: tuple
    safe: tuple
    decision: int
    delta: float
    hard_slack: float        # non-finite (REJECT) stored as -1.0
    min_dog_capsule: float   # exact capsule distance to nearest dynamic neighbour
    min_wall_capsule: float  # exact capsule distance to nearest wall point


class RecordingFilter:
    """Wraps a safety filter; captures one Frame per (step, agent) at decision
    time. Set ``.sim`` right after constructing the KineSim (the ego view needs
    the agent's goal/waypoint state and last world velocity)."""

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "?")
        self.sim = None
        self.frames: dict[str, list[Frame]] = {}

    def filter(self, aid, cmd, pose, neighbors, self_priority=1.0):
        safe, info = self.inner.filter(aid, cmd, pose, neighbors,
                                       self_priority=self_priority)
        ag = self.sim.agents[aid]
        min_dog, min_wall = math.inf, math.inf
        for n in neighbors:
            static = len(n) >= 8 and n[7] >= 0.5
            d = seg_seg_closest(
                pose[0], pose[1], math.cos(pose[2]), math.sin(pose[2]), CAPSULE_HALF,
                n[0], n[1], math.cos(n[6]), math.sin(n[6]),
                0.0 if static else CAPSULE_HALF,
            )[2]
            if static:
                min_wall = min(min_wall, d)
            else:
                min_dog = min(min_dog, d)
        delta = info.get("delta") if isinstance(info, dict) else None
        if delta is None:
            delta = max(abs(safe[k] - cmd[k]) for k in range(3))
        hs = info.get("hard_slack_max", 0.0) if isinstance(info, dict) else 0.0
        if not isinstance(hs, (int, float)) or not math.isfinite(hs):
            hs = -1.0
        self.frames.setdefault(aid, []).append(Frame(
            obs=ego_observation(ag, neighbors),
            raw=tuple(cmd), safe=tuple(safe),
            decision=DECISION_CODE.get(
                info.get("decision", "PASS") if isinstance(info, dict) else "PASS", 0),
            delta=float(delta), hard_slack=float(hs),
            min_dog_capsule=min_dog if math.isfinite(min_dog) else PAD_DIST,
            min_wall_capsule=min_wall if math.isfinite(min_wall) else PAD_DIST,
        ))
        return safe, info

    def close(self):
        if hasattr(self.inner, "close"):
            self.inner.close()


@dataclass
class AgentEpisode:
    scenario: str
    seed: int
    method: str
    aid: str
    dt: float
    frames: list[Frame]
    done: bool
    done_step: int | None    # frame index where the goal was reached (None = never)


def collect_agent_episodes(sim, recorder_filter, scenario, seed, method,
                           tail_s: float = 1.0) -> list[AgentEpisode]:
    """Slice the RecordingFilter capture into per-agent episodes, truncated
    ``tail_s`` after goal arrival (the tail teaches 'hold still at the goal';
    beyond it the frames are trivial)."""
    out = []
    dt = sim.cfg.dt
    tail = int(round(tail_s / dt))
    for aid, frames in recorder_filter.frames.items():
        ag = sim.agents[aid]
        done_step = None
        if ag.done and ag.done_time is not None:
            done_step = min(int(round(ag.done_time / dt)), len(frames) - 1)
            frames = frames[: done_step + 1 + tail]
        out.append(AgentEpisode(scenario=scenario, seed=seed, method=method,
                                aid=aid, dt=dt, frames=frames,
                                done=ag.done, done_step=done_step))
    return out


# -- boundary events (the RQ5 failure-sample log) -----------------------------

def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end) inclusive frame indices."""
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def boundary_events(ep: AgentEpisode, episode_index: int) -> list[dict]:
    """RSMF-style event records: one per contiguous run of each boundary
    condition, with a +-EVENT_WINDOW_S context window and a reason string."""
    fr = ep.frames
    win = int(round(EVENT_WINDOW_S / ep.dt))
    n = len(fr)
    conditions = {
        "guard_intervention": [f.delta > INTERVENE_EPS and f.decision != 3 for f in fr],
        "qp_reject": [f.decision in (3, 4) for f in fr],
        "hard_slack": [f.hard_slack > 1e-6 for f in fr],
        "violation_dog": [f.min_dog_capsule < DOG_FLOOR for f in fr],
        "violation_wall": [f.min_wall_capsule < WALL_FLOOR for f in fr],
    }
    reasons = {
        "guard_intervention": "guard modified the proposal (|safe-raw| > eps)",
        "qp_reject": "QP infeasible/fault -> fallback command",
        "hard_slack": "QP bought slack on the hard floor",
        "violation_dog": f"dog-dog capsule distance below {DOG_FLOOR} m floor",
        "violation_wall": f"dog-wall clearance below {WALL_FLOOR} m floor",
    }
    events = []
    for etype, flags in conditions.items():
        for s, e in _runs(flags):
            seg = fr[s:e + 1]
            ws, we = max(0, s - win), min(n - 1, e + win)
            events.append({
                "event_type": etype,
                "episode_index": episode_index,
                "scenario": ep.scenario, "seed": ep.seed,
                "method": ep.method, "aid": ep.aid,
                "frame_start": s, "frame_end": e,
                "t_start_s": round(s * ep.dt, 4), "t_end_s": round(e * ep.dt, 4),
                "n_frames": e - s + 1,
                "peak_delta": max(f.delta for f in seg),
                "peak_hard_slack": max(f.hard_slack for f in seg),
                "min_dog_capsule_m": min(f.min_dog_capsule for f in seg),
                "min_wall_capsule_m": min(f.min_wall_capsule for f in seg),
                "decisions": sorted({f.decision for f in seg}),
                "raw_at_peak": list(max(seg, key=lambda f: f.delta).raw),
                "safe_at_peak": list(max(seg, key=lambda f: f.delta).safe),
                "reason": reasons[etype],
                "window_start": ws, "window_end": we,
                "window_complete": bool(ws == s - win and we == e + win),
            })
    return events


EVENT_REQUIRED_FIELDS = (
    "event_type", "episode_index", "scenario", "seed", "method", "aid",
    "frame_start", "frame_end", "reason", "raw_at_peak", "safe_at_peak",
    "window_start", "window_end",
)


# -- LeRobot export ------------------------------------------------------------

def lerobot_features() -> dict:
    return {
        "observation.state": {"dtype": "float32", "shape": (OBS_DIM,),
                              "names": OBS_NAMES},
        "observation.min_dog_capsule_m": {"dtype": "float32", "shape": (1,), "names": None},
        "observation.min_wall_capsule_m": {"dtype": "float32", "shape": (1,), "names": None},
        "action": {"dtype": "float32", "shape": (3,), "names": ["vx", "vy", "omega"]},
        "action.raw": {"dtype": "float32", "shape": (3,), "names": ["vx", "vy", "omega"]},
        "guard.decision": {"dtype": "int64", "shape": (1,), "names": None},
        "guard.delta": {"dtype": "float32", "shape": (1,), "names": None},
        "guard.hard_slack": {"dtype": "float32", "shape": (1,), "names": None},
        "guard.intervened": {"dtype": "int64", "shape": (1,), "names": None},
    }


def export_pool(episodes: list[AgentEpisode], root: Path, repo_id: str,
                fps: int) -> dict:
    """Write one LeRobotDataset for a pool + the two meta sidecars.
    Returns {n_episodes, n_frames, n_events, root}."""
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, root=root,
                               robot_type="go2_kinesim", use_videos=False,
                               features=lerobot_features())
    manifest, events = [], []
    n_frames = 0
    for idx, ep in enumerate(episodes):
        task = (f"{ep.scenario}: drive to goal, {ep.method} safety filter")
        for f in ep.frames:
            ds.add_frame({
                "observation.state": np.asarray(f.obs, dtype=np.float32),
                "observation.min_dog_capsule_m": np.array([f.min_dog_capsule], dtype=np.float32),
                "observation.min_wall_capsule_m": np.array([f.min_wall_capsule], dtype=np.float32),
                "action": np.asarray(f.safe, dtype=np.float32),
                "action.raw": np.asarray(f.raw, dtype=np.float32),
                "guard.decision": np.array([f.decision], dtype=np.int64),
                "guard.delta": np.array([f.delta], dtype=np.float32),
                "guard.hard_slack": np.array([f.hard_slack], dtype=np.float32),
                "guard.intervened": np.array([int(f.delta > INTERVENE_EPS)], dtype=np.int64),
                "task": task,
            })
        ds.save_episode()
        n_frames += len(ep.frames)
        manifest.append({"episode_index": idx, "scenario": ep.scenario,
                         "seed": ep.seed, "method": ep.method, "aid": ep.aid,
                         "n_frames": len(ep.frames), "done": ep.done,
                         "done_step": ep.done_step})
        events.extend(boundary_events(ep, idx))
    ds.finalize()
    meta = root / "meta"
    with open(meta / "episode_manifest.jsonl", "w") as fh:
        for row in manifest:
            fh.write(json.dumps(row) + "\n")
    with open(meta / "boundary_events.jsonl", "w") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    return {"n_episodes": len(episodes), "n_frames": n_frames,
            "n_events": len(events), "root": str(root)}
