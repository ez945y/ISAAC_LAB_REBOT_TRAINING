#!/usr/bin/env python3
"""E3.4 (RQ3): capsule vs disc body model — what the elongated body buys.

Conditions on S4 (head-on pass in a 2.2 m corridor; wall-clear lateral band
0.8 m, so two capsule spines at min_dist 0.7 JUST fit side by side):

    capsule    h=0.25, min_dist 0.7 (stackfile default)
    disc_in    inscribed disc: h=0, min_dist 0.7 — under-approximates the
               body; its own model is satisfied while the TRUE capsules close
               to ~min_dist − 2h nose-to-nose
    disc_circ  circumscribed disc: h=0, dog-dog min_dist 0.7+2h=1.2,
               dog-wall 0.7+h=0.95 — safe cover of the body, but the pass
               needs 1.2 m of lateral room in an 0.8 m band -> infeasible

Ground truth: metrics always measure TRUE capsule distance (h=0.25),
regardless of what body model the filter believes in.

Read: capsule completes with the floor held; disc_in completes but the true
min distance dips far below the floor (unsafe under-approximation); disc_circ
deadlocks / times out (over-approximation kills liveness in tight spaces).

    python3 experiments/run_e34_capsule.py --seeds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ablation import Condition, run_sweep  # noqa: E402
from common.filters import make_guard  # noqa: E402

MD_FIELDS = ["all_done_rate", "deadlock_rate", "makespan_s", "min_dogdog_m",
             "viol_steps_dog", "viol_steps_wall", "mean_dcmd",
             "intervention_rate", "max_hard_slack_m"]


def main() -> int:
    ap = argparse.ArgumentParser(description="E3.4 capsule vs disc body model")
    ap.add_argument("--method", default="pydam", choices=["pydam", "dam"],
                    help="guard implementation (dam needs the DAM venv)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="experiments/results/e34_capsule")
    args = ap.parse_args()
    if args.method != "pydam" and args.out == ap.get_default("out"):
        args.out += f"_{args.method}"

    conditions = [
        Condition(label="capsule",
                  make_filter=lambda: make_guard(args.method)),
        Condition(label="disc_in",
                  make_filter=lambda: make_guard(args.method, capsule_half=0.0)),
        Condition(label="disc_circ",
                  make_filter=lambda: make_guard(args.method, capsule_half=0.0,
                                                  min_dist=1.2,
                                                  wall_min_dist=0.95)),
    ]
    run_sweep("E3.4", ["S4"], conditions, args.seeds, args.out,
              markdown_fields=MD_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
