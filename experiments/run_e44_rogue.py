#!/usr/bin/env python3
"""E4.4 (RQ4): non-cooperative agents — what breaks when a dog bypasses the
filter?

The hard floor is enforced with a SYMMETRIC 0.5 share: each dog covers half
the required avoidance, trusting the neighbour to cover the other half. A
rogue (raw, unfiltered) dog covers nothing, so against it the compliant dogs
systematically under-brake by half the requirement. Velocity-aware prediction
sees the rogue coming (its velocity is real), but the share model does not
know it won't yield. Expected: floor dip against rogues ≈ half the margin;
liveness of the compliant crowd should survive (they route around).

Conditions (FilterRouter overrides; rogues run raw):
    coop     everyone filtered (baseline)
    rogue1   one rogue: S1 G1_0 / S5 D0
    rogue2   two rogues: S1 G1_0+G0_1 / S5 D0+D1

Scenarios: S1 (structured crossing) + S5 (unstructured cruise, 6 dogs).
Global min_dogdog is dominated by rogue encounters, which is the signal.

    python3 experiments/run_e44_rogue.py --seeds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ablation import Condition, run_sweep  # noqa: E402
from common.filters import FilterRouter, RawFilter, make_guard  # noqa: E402


MD_FIELDS = ["min_dogdog_m", "viol_steps_dog", "max_viol_depth_m",
             "all_done_rate", "deadlock_rate", "makespan_s",
             "max_hard_slack_m"]

# per-scenario rogue ids merged into one overrides dict: ids that don't exist
# in the running scenario simply never match
ROGUES_1 = ("G1_0", "D0")
ROGUES_2 = ("G1_0", "G0_1", "D0", "D1")


def _router(method: str, rogues: tuple = ()) -> FilterRouter:
    return FilterRouter(make_guard(method), {aid: RawFilter() for aid in rogues})


def main() -> int:
    ap = argparse.ArgumentParser(description="E4.4 non-cooperative agents")
    ap.add_argument("--method", default="pydam", choices=["pydam", "dam"],
                    help="guard implementation (dam needs the DAM venv)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="experiments/results/e44_rogue")
    args = ap.parse_args()
    if args.method != "pydam" and args.out == ap.get_default("out"):
        args.out += f"_{args.method}"

    conditions = [
        Condition(label="coop", make_filter=lambda: _router(args.method)),
        Condition(label="rogue1", make_filter=lambda: _router(args.method, ROGUES_1)),
        Condition(label="rogue2", make_filter=lambda: _router(args.method, ROGUES_2)),
    ]
    run_sweep("E4.4", ["S1", "S5"], conditions, args.seeds, args.out,
              markdown_fields=MD_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
