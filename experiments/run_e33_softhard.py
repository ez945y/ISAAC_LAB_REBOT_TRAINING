#!/usr/bin/env python3
"""E3.3 (RQ3): soft/hard layering ablation — does the two-tier constraint
(hard floor 0.7 + soft comfort 1.5) buy congestion throughput over a single
hard boundary, without giving up the floor?

Conditions on S3 (bottleneck):
    layered      comfort_dist=1.5, w_soft=30 (stackfile default)
    hard_only    comfort_dist=min_dist → the soft tier collapses onto the floor
    comfort_hard comfort_dist=1.5 but w_soft=w_hard → comfort enforced like a
                 wall (the "everything is sacred" anti-pattern: expect clogging)

Read: makespan/throughput + mean|Δu| (smoothness) vs min_dogdog (the floor must
hold comparably in all conditions).

    python3 experiments/run_e33_softhard.py --seeds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ablation import Condition, run_sweep  # noqa: E402
from common.filters import make_guard  # noqa: E402

MD_FIELDS = ["makespan_s", "mean_completion_s", "all_done_rate", "deadlock_rate",
             "min_dogdog_m", "viol_steps_dog", "mean_dcmd", "max_hard_slack_m"]


def main() -> int:
    ap = argparse.ArgumentParser(description="E3.3 soft/hard layering ablation")
    ap.add_argument("--method", default="pydam", choices=["pydam", "dam"],
                    help="guard implementation (dam needs the DAM venv)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n", type=int, default=4, help="dogs through the bottleneck")
    ap.add_argument("--out", default="experiments/results/e33_softhard")
    args = ap.parse_args()
    if args.method != "pydam" and args.out == ap.get_default("out"):
        args.out += f"_{args.method}"

    sk = {"n": args.n}
    conditions = [
        Condition(label="layered", scenario_kwargs=sk,
                  make_filter=lambda: make_guard(args.method)),
        Condition(label="hard_only", scenario_kwargs=sk,
                  make_filter=lambda: make_guard(args.method, comfort_dist=0.7)),
        Condition(label="comfort_hard", scenario_kwargs=sk,
                  make_filter=lambda: make_guard(args.method, w_soft=1000.0)),
    ]
    run_sweep("E3.3", ["S3"], conditions, args.seeds, args.out,
              markdown_fields=MD_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
