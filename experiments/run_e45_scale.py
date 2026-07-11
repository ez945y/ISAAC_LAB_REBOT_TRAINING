#!/usr/bin/env python3
"""E4.5 (RQ4): scale sweep — S5 cruise with N ∈ {2, 4, 8, 12, 16} dogs in a
fixed ±6 m arena (scale AND density rise together).

Read:
  * safety vs N: min_dogdog / viol_steps (more simultaneous conflicts, F7
    pressure grows), deadlock (does liveness survive density?)
  * throughput vs N: makespan (detours multiply)
  * cost vs N: filter p50/p99 per call — the QP constraint count grows with
    the number of neighbours inside the influence radius. NOTE: pydam/SLSQP
    timing is a TREND indicator only; absolute numbers for the robot come
    from the real dam+OSQP path on the Isaac machine (E1.2).

    python3 experiments/run_e45_scale.py --seeds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ablation import Condition, run_sweep  # noqa: E402
from common.pydam import PyDamFilter  # noqa: E402

MD_FIELDS = ["min_dogdog_m", "viol_steps_dog", "all_done_rate", "deadlock_rate",
             "makespan_s", "mean_completion_s", "intervention_rate",
             "filter_p50_ms", "filter_p99_ms", "max_hard_slack_m"]


def main() -> int:
    ap = argparse.ArgumentParser(description="E4.5 scale sweep")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--ns", default="2,4,8,12,16")
    ap.add_argument("--out", default="experiments/results/e45_scale")
    args = ap.parse_args()

    conditions = [
        Condition(label=f"n{n}",
                  make_filter=lambda: PyDamFilter(),
                  scenario_kwargs={"n": int(n)})
        for n in (x.strip() for x in args.ns.split(",") if x.strip())
    ]
    run_sweep("E4.5", ["S5"], conditions, args.seeds, args.out,
              markdown_fields=MD_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
