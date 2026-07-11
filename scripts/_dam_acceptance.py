# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Acceptance / evaluation harness for the Go2 inter-dog DAM guard.

Runs the --auto crossing demo twice (DAM on vs off), records per-step
observations, and scores them:
  - min pairwise distance over the run  (collision = below --threshold)
  - collision-frame count
  - per-group avoidance correction      (yielding: low-priority >> high-priority)
  - completion (did every dog finish near its last target)

Prints a metrics table + PASS/FAIL. Used to iterate the guard "until it's good".

Run (venv + libgomp, like demo11):
    python scripts/_dam_acceptance.py [--max-seconds 26] [--threshold 0.45]
    python scripts/_dam_acceptance.py --eval-only on.jsonl off.jsonl   # score existing logs
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_sim(dam: bool, out: str, max_seconds: float, priority: str) -> None:
    cmd = [sys.executable, os.path.join(REPO, "scripts", "11_go2_squad_dispatch.py"),
           "--auto", "--record-obs", out, "--max-seconds", str(max_seconds)]
    if dam:
        cmd += ["--dam", "--dam-priority", priority]
    print(f"[acc] running {'DAM-ON ' if dam else 'DAM-OFF'} -> {out}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _by_time(path: str) -> dict:
    frames: dict[float, dict] = {}
    for line in open(path):
        r = json.loads(line)
        frames.setdefault(r["t"], {})[r["agent"]] = r
    return frames


def evaluate(path: str, threshold: float) -> dict:
    frames = _by_time(path)
    min_dist = math.inf
    collision_frames = 0
    corr: dict[str, list] = {}
    last_arrived = {}
    for t, dogs in frames.items():
        pts = {a: (r["obs"][0], r["obs"][1]) for a, r in dogs.items()}
        frame_min = math.inf
        for a, b in itertools.combinations(pts, 2):
            d = math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])
            frame_min = min(frame_min, d)
        if frame_min < math.inf:
            min_dist = min(min_dist, frame_min)
            if frame_min < threshold:
                collision_frames += 1
        for a, r in dogs.items():
            dev = max(abs(r["safe"][i] - r["nominal"][i]) for i in range(3))
            if r["nearest"] < 1.3 and r["decision"] not in ("PASS", "OFF"):
                corr.setdefault(r["group"], []).append(dev)
            last_arrived[a] = r["arrived"]
    return {
        "frames": len(frames),
        "min_dist": min_dist,
        "collision_frames": collision_frames,
        "group_corr": {g: (sum(v) / len(v) if v else 0.0) for g, v in sorted(corr.items())},
        "arrived": sum(last_arrived.values()),
        "n_dogs": len(last_arrived),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="DAM acceptance harness")
    ap.add_argument("--max-seconds", type=float, default=26.0)
    ap.add_argument("--threshold", type=float, default=0.45, help="collision distance (m)")
    ap.add_argument("--priority", default="G0:5,G1:1")
    ap.add_argument("--eval-only", nargs=2, metavar=("ON", "OFF"), default=None)
    args = ap.parse_args()

    if args.eval_only:
        on_path, off_path = args.eval_only
    else:
        on_path, off_path = "/tmp/acc_dam_on.jsonl", "/tmp/acc_dam_off.jsonl"
        run_sim(True, on_path, args.max_seconds, args.priority)
        run_sim(False, off_path, args.max_seconds, args.priority)

    on, off = evaluate(on_path, args.threshold), evaluate(off_path, args.threshold)

    print("\n================ DAM ACCEPTANCE ================")
    print(f"{'metric':<22}{'DAM OFF':>12}{'DAM ON':>12}")
    print(f"{'min pairwise dist':<22}{off['min_dist']:>12.3f}{on['min_dist']:>12.3f}")
    print(f"{'collision frames':<22}{off['collision_frames']:>12d}{on['collision_frames']:>12d}")
    print(f"{'arrived':<22}{str(off['arrived'])+'/'+str(off['n_dogs']):>12}"
          f"{str(on['arrived'])+'/'+str(on['n_dogs']):>12}")
    print(f"per-group correction (DAM ON): {on['group_corr']}")

    g = on["group_corr"]
    yields = g.get("G1", 0.0) > 1.5 * g.get("G0", 0.0) and g.get("G0", 0.0) >= 0.0
    no_collision = on["min_dist"] >= args.threshold
    helps = on["min_dist"] >= off["min_dist"]
    completes = on["arrived"] >= on["n_dogs"] - 1  # allow one straggler

    print("\n  checks:")
    print(f"   [{'PASS' if no_collision else 'FAIL'}] no collisions (min >= {args.threshold})")
    print(f"   [{'PASS' if helps else 'FAIL'}] DAM improves min distance vs off")
    print(f"   [{'PASS' if yields else 'FAIL'}] low-priority G1 yields more than G0")
    print(f"   [{'PASS' if completes else 'FAIL'}] squad completes the choreography")
    ok = no_collision and helps and yields and completes
    print("\n  RESULT:", "PASS" if ok else "FAIL", "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
