#!/usr/bin/env python3
"""E4.2 (RQ4): observation noise + latency — the guard reasons about
neighbours it perceives imperfectly.

Unlike E4.1 (execution noise, corrected by reality every step), observation
error corrupts the CONSTRAINT itself: noise creates phantom requirements
(over-braking) and missed closures (dips); latency shifts every neighbour
back along its track — at 200 ms a 1.5 m/s dog is mis-placed by 0.3 m,
comparable to the whole γ margin. Velocity-aware prediction partially
compensates for latency (it extrapolates the delayed velocity) — E4.2
quantifies how far that carries.

Slices (full cross is 16 conditions — sweep each axis through the origin):
    noise  σ_obs ∈ {0, 0.05, 0.1, 0.2} m at delay 0
    delay  {100, 200, 500} ms (5/10/25 steps @ 50 Hz) at σ_obs 0
    real   the realistic operating point: σ_obs 0.1 m + 100 ms

Scenarios: S2 (clean margin) + S3 (congestion, margins already thin).
Own pose stays ground truth (odometry ≫ inter-robot perception).

Read: minDD / viol vs each axis (which binds first?), deadlock (phantom
braking can freeze), makespan (over-braking tax), hard slack.

    python3 experiments/run_e42_obsnoise.py --seeds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ablation import Condition, run_sweep  # noqa: E402
from common.pydam import PyDamFilter  # noqa: E402

MD_FIELDS = ["min_dogdog_m", "viol_steps_dog", "viol_steps_wall",
             "all_done_rate", "deadlock_rate", "makespan_s", "mean_dcmd",
             "intervention_rate", "max_hard_slack_m"]


def main() -> int:
    ap = argparse.ArgumentParser(description="E4.2 observation noise + latency")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="experiments/results/e42_obsnoise")
    args = ap.parse_args()

    conditions = [
        Condition(label=f"noise{s}",
                  make_filter=lambda: PyDamFilter(),
                  sim_overrides={"obs_noise_std": s})
        for s in (0.0, 0.05, 0.1, 0.2)
    ] + [
        Condition(label=f"delay{k*20}ms",
                  make_filter=lambda: PyDamFilter(),
                  sim_overrides={"obs_delay_steps": k})
        for k in (5, 10, 25)
    ] + [
        Condition(label="real_0.1m_100ms",
                  make_filter=lambda: PyDamFilter(),
                  sim_overrides={"obs_noise_std": 0.1, "obs_delay_steps": 5}),
    ]
    run_sweep("E4.2", ["S2", "S3"], conditions, args.seeds, args.out,
              markdown_fields=MD_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
