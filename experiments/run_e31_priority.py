#!/usr/bin/env python3
"""E3.1 (RQ3): priority ablation — does asymmetric priority yield emergent
right-of-way (high-priority group keeps its schedule) and break the symmetric
crossing deadlock (finding F2)?

Conditions on S1 (orthogonal group crossing):
    priority_on   G0 priority 3.0, G1 priority 1.0 (matches demo11 AUTO_PRIORITY style)
    symmetric     everyone 1.0

Read: completion_G0_s vs completion_G1_s (yielding), deadlock_rate (liveness).

    python3 experiments/run_e31_priority.py --seeds 10           # pydam (any machine)
    "$VENV" experiments/run_e31_priority.py --method dam ...      # real guard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ablation import Condition, run_sweep  # noqa: E402
from common.filters import make_filter  # noqa: E402

MD_FIELDS = ["completion_G0_done_s", "completion_G1_done_s", "path_ratio_G0",
             "path_ratio_G1", "deadlock_rate", "all_done_rate", "min_dogdog_m",
             "viol_steps_dog"]


def main() -> int:
    ap = argparse.ArgumentParser(description="E3.1 priority ablation")
    ap.add_argument("--method", default="pydam", choices=["pydam", "dam"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-per-group", type=int, default=2)
    ap.add_argument("--out", default="experiments/results/e31_priority")
    args = ap.parse_args()

    conditions = [
        Condition(
            label="priority_on",
            make_filter=lambda: make_filter(args.method),
            scenario_kwargs={"n_per_group": args.n_per_group, "pri_g0": 3.0, "pri_g1": 1.0},
        ),
        Condition(
            label="symmetric",
            make_filter=lambda: make_filter(args.method),
            scenario_kwargs={"n_per_group": args.n_per_group, "pri_g0": 1.0, "pri_g1": 1.0},
        ),
    ]
    run_sweep("E3.1", ["S1"], conditions, args.seeds, args.out,
              markdown_fields=MD_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
