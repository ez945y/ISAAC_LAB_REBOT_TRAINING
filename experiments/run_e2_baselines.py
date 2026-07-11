#!/usr/bin/env python3
"""E2 (RQ2): main safety-vs-performance comparison across filter methods.

Runs every (scenario x method x seed) episode in the lightweight kinematic sim,
writes per-episode rows to CSV and an aggregated mean±95%CI markdown table.

Methods: raw (B0), stop (B1), orca (B2, TODO), dam (B3 — needs torch/osqp/dam).

Fake-data smoke test on any machine (no torch needed):
    python3 experiments/run_e2_baselines.py --methods raw,stop --scenarios S1,S2 --seeds 5

Full run on the Isaac-side machine (still sim-lite, no Isaac needed):
    python3 experiments/run_e2_baselines.py --methods raw,stop,dam --scenarios S1,S2,S3,S4,S5 --seeds 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.filters import make_filter  # noqa: E402
from common.kinesim import KineSim, SimConfig  # noqa: E402
from common.metrics import (EpisodeRecorder, MetricsConfig, aggregate,  # noqa: E402
                            to_markdown, write_csv)
from common.scenarios import SCENARIOS  # noqa: E402


def run_episode(scenario: str, seed: int, method: str, sim_cfg: SimConfig,
                met_cfg: MetricsConfig) -> dict:
    specs = SCENARIOS[scenario](seed)
    filt = make_filter(method)  # fresh filter per episode: no cross-episode latch/guard state
    sim = KineSim(specs, filt, sim_cfg, rng_seed=seed)
    rec = EpisodeRecorder(met_cfg)
    sim.run(recorder=rec)
    row = rec.finish(sim, scenario, seed, method)
    if hasattr(filt, "close"):
        filt.close()
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="E2: baseline comparison (RQ2)")
    ap.add_argument("--scenarios", default="S1,S2,S5",
                    help=f"comma list from {sorted(SCENARIOS)} (default S1,S2,S5)")
    ap.add_argument("--methods", default="raw,stop", help="comma list: raw,stop,orca,dam")
    ap.add_argument("--seeds", type=int, default=5, help="seeds per (scenario, method)")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--max-time", type=float, default=60.0)
    ap.add_argument("--out", default="experiments/results/e2")
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    sim_cfg = SimConfig(max_time=args.max_time)
    met_cfg = MetricsConfig()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(scenarios) * len(methods) * args.seeds
    k = 0
    for scen in scenarios:
        for meth in methods:
            for i in range(args.seeds):
                seed = args.seed_base + i
                row = run_episode(scen, seed, meth, sim_cfg, met_cfg)
                rows.append(row)
                k += 1
                print(f"[{k}/{total}] {scen} {meth} seed={seed}  "
                      f"done={row['all_done']} makespan={row['makespan_s']:.1f}s "
                      f"minDD={row['min_dogdog_m']:.2f}m viol={row['viol_steps_dog']}")

    ep_csv = out_dir / "episodes.csv"
    write_csv(ep_csv, rows)

    agg = aggregate(rows)
    agg_csv = out_dir / "aggregate.csv"
    write_csv(agg_csv, agg)

    md = to_markdown(agg)
    (out_dir / "summary.md").write_text(md + "\n")
    print("\n" + md)
    print(f"\nwrote {ep_csv}, {agg_csv}, {out_dir/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
