#!/usr/bin/env python3
"""E3.6 (RQ3): γ / dt sweep — conservatism, and the linearization blind spot.

The discrete CBF condition allows the separation to shrink by at most
γ·(dist − min_dist) per prediction horizon dt. The textbook read (small γ =
conservative) is NOT what happens in the slacked multi-agent QP — the probe
found finding F9 instead:

  * small γ activates constraints far out, so dogs spend a LONG time
    manoeuvring tangentially at close range. There the normal-projection
    d_pred under-estimates closure (the closest-point pair slides along the
    capsule spines and the normal rotates within the horizon), each QP binds
    its half-share with EQUALITY (zero margin), and the un-modelled
    second-order term (~0.17 m per 0.2 s horizon at a 90° crossing) goes
    straight into penetration — up to full capsule PASS-THROUGH (minDD 0.0,
    S1 γ=0.1 dt=0.2).
  * γ→1 is a strict one-step barrier: floor held with margin, but brake-wall
    deadlocks appear (S1) — liveness cost instead of safety cost.
  * dt is the binding parameter for the blind spot: re-linearizing faster
    (dt 0.2→0.1) removes the pass-through at every γ tested.

Sweep γ ∈ {0.1, 0.2, 0.4, 0.7, 1.0} at dt = 0.2 (stackfile default γ = 0.4),
plus dt ∈ {0.1, 0.4} at γ = 0.4 and dt = 0.1 at γ = 0.1 (interaction).

Scenarios: S1 (two-group crossing — the tangential-blind-spot geometry) and
S3 (bottleneck — sustained congestion, the F7 crowd-crush case).

Read: makespan_s vs min_dogdog_m/viol_steps_dog per (γ, dt) — the thesis
Pareto figure; where the stackfile default sits; the F9 regime boundaries.

    python3 experiments/run_e36_gamma.py --seeds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ablation import Condition, run_sweep  # noqa: E402
from common.pydam import PyDamFilter  # noqa: E402

MD_FIELDS = ["makespan_s", "mean_completion_s", "all_done_rate", "deadlock_rate",
             "min_dogdog_m", "viol_steps_dog", "mean_dcmd", "intervention_rate",
             "max_hard_slack_m"]


def main() -> int:
    ap = argparse.ArgumentParser(description="E3.6 gamma/dt sweep")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--gammas", default="0.1,0.2,0.4,0.7,1.0")
    ap.add_argument("--out", default="experiments/results/e36_gamma")
    args = ap.parse_args()

    conditions = [
        Condition(label=f"g{g}_dt0.2",
                  make_filter=lambda g=float(g): PyDamFilter(gamma=g))
        for g in (x.strip() for x in args.gammas.split(",") if x.strip())
    ] + [
        Condition(label=f"g0.4_dt{dt}",
                  make_filter=lambda dt=dt: PyDamFilter(gamma=0.4, dt=dt))
        for dt in (0.1, 0.4)
    ] + [
        Condition(label="g0.1_dt0.1",   # F9 interaction: short horizon at low γ
                  make_filter=lambda: PyDamFilter(gamma=0.1, dt=0.1)),
    ]
    run_sweep("E3.6", ["S1", "S3"], conditions, args.seeds, args.out,
              markdown_fields=MD_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
