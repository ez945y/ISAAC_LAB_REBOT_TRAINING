"""Per-episode safety/performance metrics + cross-seed aggregation.

The EpisodeRecorder streams alongside KineSim.run() (no giant traces kept) and
produces one flat dict per episode; aggregate() turns many episodes into
mean +/- 95% CI rows ready for CSV / markdown.

Distances are EXACT capsule (segment-segment) distances via seg_seg_closest —
the same geometry the DAM guard uses — with two thresholds:
    dog-dog  min_dist   (body-capsule floor, default 0.7 m, guard's hard floor)
    dog-wall wall_dist  (clearance to static obstacle points, default 0.35 m)
"""

from __future__ import annotations

import importlib.util
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

# capsule_geometry is pure math; load by path to avoid the torch-importing
# safety package __init__ on machines without torch.
_MOD_PATH = Path(__file__).resolve().parents[2] / "tools/controll_scripts/safety/capsule_geometry.py"
_spec = importlib.util.spec_from_file_location("capsule_geometry", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
seg_seg_closest = _mod.seg_seg_closest


@dataclass
class MetricsConfig:
    min_dist: float = 0.7        # dog-dog capsule floor (= guard hard floor)
    wall_dist: float = 0.35      # dog-wall clearance floor
    capsule_half: float = 0.25   # dog body half-length (walls are points)
    progress_window: float = 5.0  # deadlock: window with < progress_eps motion
    progress_eps: float = 0.05


class EpisodeRecorder:
    def __init__(self, cfg: MetricsConfig | None = None):
        self.cfg = cfg or MetricsConfig()
        self.steps = 0
        self.dt = None
        self.min_dogdog = math.inf
        self.min_dogwall = math.inf
        self.viol_steps_dog = 0
        self.viol_steps_wall = 0
        self.max_depth_dog = 0.0
        self.interventions = 0
        self.cmd_deltas: list[float] = []
        self.filter_times: list[float] = []
        self._progress_ref: dict = {}   # aid -> (t, x, y) start of current window
        self._stalled: set = set()
        self.max_hard_slack = 0.0       # E4.3: worst hard-floor slack the QP ever bought

    def _capsule_dist(self, a, b, ha, hb) -> float:
        return seg_seg_closest(
            a[0], a[1], math.cos(a[2]), math.sin(a[2]), ha,
            b[0], b[1], math.cos(b[2]), math.sin(b[2]), hb,
        )[2]

    def step(self, sim, rec):
        cfg = self.cfg
        self.steps += 1
        self.dt = sim.cfg.dt
        agents = sim.agents
        dyn = [(aid, ag) for aid, ag in agents.items() if not ag.spec.static]
        stat = [(aid, ag) for aid, ag in agents.items() if ag.spec.static]

        # -- pairwise capsule distances -----------------------------------
        step_viol_dog = False
        for i in range(len(dyn)):
            _, a = dyn[i]
            for j in range(i + 1, len(dyn)):
                _, b = dyn[j]
                d = self._capsule_dist((a.x, a.y, a.yaw), (b.x, b.y, b.yaw),
                                       cfg.capsule_half, cfg.capsule_half)
                self.min_dogdog = min(self.min_dogdog, d)
                if d < cfg.min_dist:
                    step_viol_dog = True
                    self.max_depth_dog = max(self.max_depth_dog, cfg.min_dist - d)
        if step_viol_dog:
            self.viol_steps_dog += 1
        step_viol_wall = False
        for _, a in dyn:
            for _, w in stat:
                d = self._capsule_dist((a.x, a.y, a.yaw), (w.x, w.y, w.yaw),
                                       cfg.capsule_half, 0.0)
                self.min_dogwall = min(self.min_dogwall, d)
                if d < cfg.wall_dist:
                    step_viol_wall = True
        if step_viol_wall:
            self.viol_steps_wall += 1

        # -- intervention & smoothness ------------------------------------
        for aid, raw in rec.raw_cmds.items():
            safe = rec.safe_cmds[aid]
            delta = max(abs(safe[k] - raw[k]) for k in range(3))
            self.cmd_deltas.append(delta)
            if delta > 1e-5:
                self.interventions += 1
        self.filter_times.extend(rec.filter_s.values())
        for info in rec.decisions.values():
            hs = info.get("hard_slack_max", 0.0) if isinstance(info, dict) else 0.0
            if isinstance(hs, (int, float)) and math.isfinite(hs):
                self.max_hard_slack = max(self.max_hard_slack, hs)

        # -- deadlock: unfinished agent with ~no motion over a window ------
        for aid, ag in dyn:
            if ag.done:
                self._progress_ref.pop(aid, None)
                self._stalled.discard(aid)
                continue
            ref = self._progress_ref.get(aid)
            if ref is None:
                self._progress_ref[aid] = (rec.t, ag.x, ag.y)
            elif rec.t - ref[0] >= cfg.progress_window:
                if math.hypot(ag.x - ref[1], ag.y - ref[2]) < cfg.progress_eps:
                    self._stalled.add(aid)
                self._progress_ref[aid] = (rec.t, ag.x, ag.y)

    def finish(self, sim, scenario: str, seed: int, method: str) -> dict:
        dyn = [ag for ag in sim.agents.values() if not ag.spec.static]
        n_cmd = max(1, len(self.cmd_deltas))
        done_times = [ag.done_time for ag in dyn if ag.done]
        ratios = []
        for ag in dyn:
            straight = math.hypot(ag.spec.gx - ag.spec.x, ag.spec.gy - ag.spec.y)
            if ag.done and straight > 1e-6:
                ratios.append(ag.path_len / straight)
        ft_ms = sorted(t * 1e3 for t in self.filter_times)

        def pct(p):
            return ft_ms[min(len(ft_ms) - 1, int(p * len(ft_ms)))] if ft_ms else 0.0

        per_group: dict = {}
        for ag in dyn:  # per-group metrics (E3.1 emergent yielding)
            g = ag.spec.group
            straight = math.hypot(ag.spec.gx - ag.spec.x, ag.spec.gy - ag.spec.y)
            ratio = (ag.path_len / straight) if (ag.done and straight > 1e-6) else math.nan
            per_group.setdefault(g, []).append(
                (ag.done_time if ag.done else sim.cfg.max_time,
                 ag.done_time if ag.done else math.nan,   # completed dogs only
                 ratio))
        group_fields = {}
        for g, vals in sorted(per_group.items()):
            group_fields[f"completion_{g}_s"] = statistics.mean(v[0] for v in vals)
            done_ts = [v[1] for v in vals if math.isfinite(v[1])]
            group_fields[f"completion_{g}_done_s"] = statistics.mean(done_ts) if done_ts else math.nan
            ratios_g = [v[2] for v in vals if math.isfinite(v[2])]
            group_fields[f"path_ratio_{g}"] = statistics.mean(ratios_g) if ratios_g else math.nan

        return {
            "scenario": scenario, "seed": seed, "method": method,
            "n_dogs": len(dyn),
            **group_fields,
            "max_hard_slack_m": self.max_hard_slack,
            "all_done": all(ag.done for ag in dyn),
            "timeouts": sum(not ag.done for ag in dyn),
            "deadlock": bool(self._stalled and not all(ag.done for ag in dyn)),
            "makespan_s": max(done_times) if len(done_times) == len(dyn) else sim.cfg.max_time,
            "mean_completion_s": statistics.mean(done_times) if done_times else sim.cfg.max_time,
            "path_ratio": statistics.mean(ratios) if ratios else math.nan,
            "min_dogdog_m": self.min_dogdog if math.isfinite(self.min_dogdog) else math.nan,
            "min_dogwall_m": self.min_dogwall if math.isfinite(self.min_dogwall) else math.nan,
            "viol_steps_dog": self.viol_steps_dog,
            "max_viol_depth_m": self.max_depth_dog,
            "viol_steps_wall": self.viol_steps_wall,
            "intervention_rate": self.interventions / n_cmd,
            "mean_dcmd": statistics.mean(self.cmd_deltas) if self.cmd_deltas else 0.0,
            "max_dcmd": max(self.cmd_deltas) if self.cmd_deltas else 0.0,
            "filter_p50_ms": pct(0.50),
            "filter_p99_ms": pct(0.99),
        }


_SKIP_FIELDS = {"seed"}


def aggregate(rows: list[dict], keys: tuple = ("scenario", "method")) -> list[dict]:
    """Group by ``keys``: mean +/- 95% CI for numeric fields, rate for booleans.

    Fields are auto-detected from the rows, so dynamic columns (per-group
    completion, sweep parameters stored as strings in the keys) just work.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    all_fields: list[str] = []
    for r in rows:
        for f in r:
            if f not in all_fields:
                all_fields.append(f)
    out = []
    for gkey, rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        agg = dict(zip(keys, gkey))
        agg["n_seeds"] = len(rs)
        for f in all_fields:
            if f in keys or f in _SKIP_FIELDS:
                continue
            vals = [r[f] for r in rs if f in r]
            if not vals:
                continue
            if all(isinstance(v, bool) for v in vals):
                agg[f + "_rate"] = sum(vals) / len(vals)
            elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                fin = [v for v in vals if math.isfinite(v)]
                if not fin:
                    agg[f], agg[f + "_ci95"] = math.nan, math.nan
                    continue
                m = statistics.mean(fin)
                sd = statistics.stdev(fin) if len(fin) > 1 else 0.0
                agg[f] = m
                agg[f + "_ci95"] = 1.96 * sd / math.sqrt(len(fin))
        out.append(agg)
    return out


def write_csv(path, rows: list[dict]) -> None:
    """CSV with the UNION of keys across rows (dynamic per-scenario fields)."""
    import csv
    fields: list[str] = []
    for r in rows:
        for f in r:
            if f not in fields:
                fields.append(f)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def to_markdown(agg_rows: list[dict], fields: list[str] | None = None) -> str:
    fields = fields or ["makespan_s", "min_dogdog_m", "viol_steps_dog",
                        "deadlock_rate", "all_done_rate", "intervention_rate", "filter_p99_ms"]
    keys = [k for k in agg_rows[0] if not k.endswith("_ci95") and k != "n_seeds"
            and k not in fields and not isinstance(agg_rows[0][k], float)]
    head = keys + ["n_seeds"] + fields
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join("---" for _ in head) + "|"]
    for r in agg_rows:
        cells = [str(r.get(k, "")) for k in keys] + [str(r["n_seeds"])]
        for f in fields:
            v = r.get(f, math.nan)
            ci = r.get(f + "_ci95")
            cells.append(f"{v:.3f}±{ci:.3f}" if isinstance(ci, float) and math.isfinite(ci)
                         else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
