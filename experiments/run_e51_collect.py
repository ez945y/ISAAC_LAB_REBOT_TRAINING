#!/usr/bin/env python3
"""E5.1 (RQ5): runtime data collection -> LeRobot dataset + boundary-event log.

Collects two pools over the same (scenario, seed) grid:
    dam   guard-filtered episodes  -> the reusable "training data" pool
    raw   unfiltered episodes      -> the failure-rich "raw data" contrast pool

Each kinesim episode yields one LeRobot episode PER AGENT (ego-centric
42-dim observation, body-frame action; action.raw + guard telemetry kept as
extra feature channels). Boundary events (interventions, QP rejects, hard
slack, floor violations) are logged RSMF-style with +-0.5 s context windows
to meta/boundary_events.jsonl, and graded for the RQ5 quality table
(completeness / window completeness / semantic diversity).

Fake-data smoke (stdlib only, no export):
    python3 experiments/run_e51_collect.py --methods raw,stop --scenarios S2 --seeds 2 --export none

Full run (local DAM venv):
    "$VENV" experiments/run_e51_collect.py --methods dam,raw --scenarios S2,S3,S5 --seeds 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.filters import make_filter  # noqa: E402
from common.kinesim import KineSim, SimConfig  # noqa: E402
from common.lerobot_export import (EVENT_REQUIRED_FIELDS, AgentEpisode,  # noqa: E402
                                   RecordingFilter, boundary_events,
                                   collect_agent_episodes, export_pool)
from common.metrics import (EpisodeRecorder, MetricsConfig, aggregate,  # noqa: E402
                            to_markdown, write_csv)
from common.scenarios import SCENARIOS  # noqa: E402


def run_episode(scenario: str, seed: int, method: str, sim_cfg: SimConfig,
                met_cfg: MetricsConfig) -> tuple[list[AgentEpisode], dict]:
    specs = SCENARIOS[scenario](seed)
    filt = RecordingFilter(make_filter(method))
    sim = KineSim(specs, filt, sim_cfg, rng_seed=seed)
    filt.sim = sim
    rec = EpisodeRecorder(met_cfg)
    sim.run(recorder=rec)
    row = rec.finish(sim, scenario, seed, method)
    episodes = collect_agent_episodes(sim, filt, scenario, seed, method)
    filt.close()
    return episodes, row


def quality_report(pool: str, episodes: list[AgentEpisode]) -> dict:
    """RQ5 quality metrics over the pool's boundary events."""
    events = []
    for idx, ep in enumerate(episodes):
        events.extend(boundary_events(ep, idx))
    complete = sum(all(ev.get(f) is not None for f in EVENT_REQUIRED_FIELDS)
                   for ev in events)
    win_ok = sum(ev["window_complete"] for ev in events)
    types = Counter(ev["event_type"] for ev in events)
    per_scen = {}
    for ev in events:
        per_scen.setdefault(ev["scenario"], set()).add(ev["event_type"])
    return {
        "pool": pool,
        "episodes": len(episodes),
        "frames": sum(len(ep.frames) for ep in episodes),
        "events": len(events),
        "field_completeness": complete / len(events) if events else 1.0,
        "window_completeness": win_ok / len(events) if events else 1.0,
        "event_types": dict(types),
        "types_per_scenario": {k: sorted(v) for k, v in sorted(per_scen.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="E5.1: LeRobot data collection (RQ5)")
    ap.add_argument("--scenarios", default="S2,S3,S5")
    ap.add_argument("--methods", default="dam,raw")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--max-time", type=float, default=60.0)
    ap.add_argument("--export", choices=["lerobot", "none"], default="lerobot")
    ap.add_argument("--out", default="experiments/results/e51_lerobot")
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    sim_cfg = SimConfig(max_time=args.max_time)
    met_cfg = MetricsConfig()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = int(round(1.0 / sim_cfg.dt))

    rows, reports = [], []
    for method in methods:
        pool: list[AgentEpisode] = []
        for scen in scenarios:
            for i in range(args.seeds):
                seed = args.seed_base + i
                eps, row = run_episode(scen, seed, method, sim_cfg, met_cfg)
                pool.extend(eps)
                rows.append(row)
                print(f"[{method}] {scen} seed={seed}  agents={len(eps)} "
                      f"frames={sum(len(e.frames) for e in eps)} "
                      f"done={row['all_done']} minDD={row['min_dogdog_m']:.2f} "
                      f"viol={row['viol_steps_dog']}")
        rep = quality_report(method, pool)
        if args.export == "lerobot":
            root = out_dir / f"go2squad_{method}"
            if root.exists():
                raise SystemExit(f"refusing to overwrite existing dataset {root}")
            info = export_pool(pool, root, repo_id=f"local/go2squad_{method}", fps=fps)
            # load-back check: the dataset must round-trip through the official reader
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            back = LeRobotDataset(f"local/go2squad_{method}", root=root)
            assert back.num_frames == info["n_frames"], \
                f"load-back mismatch: {back.num_frames} != {info['n_frames']}"
            sample = back[0]
            assert tuple(sample["observation.state"].shape) == (42,)
            rep["lerobot_root"] = str(root)
            rep["loadback_frames"] = back.num_frames
            print(f"[{method}] exported {info['n_episodes']} episodes / "
                  f"{info['n_frames']} frames / {info['n_events']} events -> {root}")
        reports.append(rep)

    write_csv(out_dir / "episodes.csv", rows)
    agg = aggregate(rows)
    write_csv(out_dir / "aggregate.csv", agg)
    with open(out_dir / "quality.json", "w") as fh:
        json.dump(reports, fh, indent=2)

    lines = ["# E5.1 — LeRobot collection & boundary-event quality (RQ5)", ""]
    lines.append(f"Scenarios {scenarios} x seeds {args.seeds} (base {args.seed_base}), "
                 f"fps {fps}, per-agent ego episodes.")
    lines.append("")
    lines.append("| pool | episodes | frames | events | field-complete | window-complete | event types |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in reports:
        types = ", ".join(f"{k}:{v}" for k, v in sorted(r["event_types"].items()))
        lines.append(f"| {r['pool']} | {r['episodes']} | {r['frames']} | {r['events']} "
                     f"| {r['field_completeness']:.3f} | {r['window_completeness']:.3f} "
                     f"| {types or '-'} |")
    lines.append("")
    lines.append("## Pool sanity (same metrics as E2)")
    lines.append("")
    lines.append(to_markdown(agg))
    (out_dir / "summary.md").write_text("\n".join(lines))
    print(f"\nwrote {out_dir}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
