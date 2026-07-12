#!/usr/bin/env python3
"""E3.2 (RQ3): swirl ablation — does the tangential circulation bias resolve
head-on standoffs (liveness) without costing safety?

Conditions on S2 (near-symmetric head-on swap):
    swirl_0.6   stackfile default
    swirl_0     bias off — pure push constraints, symmetric standoff expected

Read: deadlock_rate + makespan (liveness), min_dogdog/violations (safety must
not degrade with swirl on).

    python3 experiments/run_e32_swirl.py --seeds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ablation import Condition, run_sweep  # noqa: E402
from common.filters import make_guard  # noqa: E402

MD_FIELDS = ["deadlock_rate", "all_done_rate", "makespan_s", "min_dogdog_m",
             "viol_steps_dog", "path_ratio", "intervention_rate"]


def main() -> int:
    ap = argparse.ArgumentParser(description="E3.2 swirl ablation")
    ap.add_argument("--method", default="pydam", choices=["pydam", "dam"],
                    help="guard implementation (dam needs the DAM venv)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--swirls", default="0.0,0.6",
                    help="comma list of swirl gains (default 0.0,0.6)")
    ap.add_argument("--out", default="experiments/results/e32_swirl")
    args = ap.parse_args()
    if args.method != "pydam" and args.out == ap.get_default("out"):
        args.out += f"_{args.method}"

    conditions = [
        Condition(label=f"swirl_{s}_j{j}",
                  make_filter=lambda s=float(s): make_guard(args.method, swirl=s),
                  scenario_kwargs={"jitter": j})
        for s in (x.strip() for x in args.swirls.split(",") if x.strip())
        for j in (0.0, 0.15)   # perfect standoff + mild asymmetry
    ]
    run_sweep("E3.2", ["S2"], conditions, args.seeds, args.out,
              markdown_fields=MD_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
