#!/usr/bin/env python3
"""E4.1 (RQ4): tracking-error injection — the guard commands a velocity; the
robot executes it imperfectly. How much execution noise does the CBF margin
absorb before the floor gives?

kinesim adds zero-mean gaussian noise (std σ) to the EXECUTED world velocity
each 20 ms step; the filter never sees it (it reasons about the commanded
velocity, exactly like the real stack where the locomotion policy tracks the
command imperfectly). σ sweep: {0, 0.1, 0.2, 0.4} m/s vs vmax 1.5 — the Go2
walk policy tracks to roughly 0.1–0.2 m/s error.

Scenarios: S2 (head-on, clean 0.785 m margin at σ=0) and S3 (bottleneck,
margin already eroded to ~0.45 by crowd crush — noise stacks on top).

Read: minDD / viol_steps vs σ (graceful or cliff?), all_done/deadlock (noise
can also BREAK symmetric deadlocks — dither is a known liveness hack), and
max_hard_slack (does the QP visibly work harder).

    python3 experiments/run_e41_tracking.py --seeds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ablation import Condition, run_sweep  # noqa: E402
from common.filters import make_guard  # noqa: E402

MD_FIELDS = ["min_dogdog_m", "viol_steps_dog", "viol_steps_wall",
             "all_done_rate", "deadlock_rate", "makespan_s", "mean_dcmd",
             "max_hard_slack_m"]


def main() -> int:
    ap = argparse.ArgumentParser(description="E4.1 tracking-error injection")
    ap.add_argument("--method", default="pydam", choices=["pydam", "dam"],
                    help="guard implementation (dam needs the DAM venv)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--sigmas", default="0.0,0.1,0.2,0.4")
    ap.add_argument("--out", default="experiments/results/e41_tracking")
    args = ap.parse_args()
    if args.method != "pydam" and args.out == ap.get_default("out"):
        args.out += f"_{args.method}"

    conditions = [
        Condition(label=f"sig{s}",
                  make_filter=lambda: make_guard(args.method),
                  sim_overrides={"exec_noise_std": float(s)})
        for s in (x.strip() for x in args.sigmas.split(",") if x.strip())
    ]
    run_sweep("E4.1", ["S2", "S3"], conditions, args.seeds, args.out,
              markdown_fields=MD_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
