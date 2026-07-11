#!/usr/bin/env python3
"""E3.5 (RQ3): velocity-aware ablation — does predicting the neighbour forward
by its OWN velocity (TTC-style d_pred) buy safety over treating neighbours as
static, in the scenario where it should matter most?

Conditions on S2 (head-on swap, closing speed = 2*vmax):
    va_on    stackfile behaviour: d_pred uses the neighbour's observed velocity
    va_off   neighbour treated as static -> perceived closing rate is HALVED,
             so the CBF reacts a step too late on every approach

Each condition runs at vmax in {1.0, 1.5, 2.0}: the static assumption's error
grows with closing speed, so the va_off floor dip should DEEPEN with vmax
while va_on holds the floor at every speed (thesis figure: dip vs speed).
S2 is deterministic up to the jitter offset (probe showed zero seed variance,
jitter axis has no effect with swirl on), so a handful of seeds suffices.

Read: min_dogdog_m + viol_steps_dog (late reaction shows up as deeper dips),
makespan/deadlock (velocity awareness should not cost liveness), mean_dcmd
(earlier, gentler interventions vs late, harsh ones).

    python3 experiments/run_e35_velocity.py --seeds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ablation import Condition, run_sweep  # noqa: E402
from common.pydam import PyDamFilter  # noqa: E402

MD_FIELDS = ["min_dogdog_m", "viol_steps_dog", "deadlock_rate", "all_done_rate",
             "makespan_s", "mean_dcmd", "intervention_rate", "max_hard_slack_m"]


def main() -> int:
    ap = argparse.ArgumentParser(description="E3.5 velocity-aware ablation")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="experiments/results/e35_velocity")
    args = ap.parse_args()

    conditions = [
        Condition(label=f"va_{'on' if va else 'off'}_v{v}",
                  make_filter=lambda va=va, v=v: PyDamFilter(velocity_aware=va, max_v=v),
                  sim_overrides={"vmax": v},
                  scenario_kwargs={"jitter": 0.15})
        for va in (True, False)
        for v in (1.0, 1.5, 2.0)
    ]
    run_sweep("E3.5", ["S2"], conditions, args.seeds, args.out,
              markdown_fields=MD_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
