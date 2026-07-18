#!/usr/bin/env python3
"""Live progress of the Isaac suite — reads the checkpoint files, no log needed.

    python experiments/isaac/status.py            # one-shot
    watch -n 30 python experiments/isaac/status.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common.registry import build_experiments  # noqa: E402


def main() -> int:
    exps = build_experiments()
    now = time.time()
    total_done = total_all = 0
    print(f"{'exp':6s} {'episodes':>12s} {'%':>5s}  {'last episode finished':>22s}  newest row")
    for key, exp in exps.items():
        out = _HERE / "results" / exp.key
        ckpt = out / "episodes.jsonl"
        total = len(exp.scenarios) * len(exp.conditions) * exp.seeds
        total_all += total
        if not ckpt.exists():
            print(f"{key:6s} {'0/' + str(total):>12s} {'0':>5s}")
            continue
        rows = [json.loads(line) for line in ckpt.read_text().splitlines() if line.strip()]
        total_done += len(rows)
        age = (now - ckpt.stat().st_mtime) / 60
        last = rows[-1] if rows else {}
        desc = (f"{last.get('scenario', '')} {last.get('condition', '')} "
                f"seed={last.get('seed', '')} wall={last.get('wall_s', '?')}s"
                if rows else "")
        print(f"{key:6s} {f'{len(rows)}/{total}':>12s} {100 * len(rows) // max(1, total):>4d}%"
              f"  {age:>18.1f} min ago  {desc}")
    print(f"\nTOTAL  {total_done}/{total_all} ({100 * total_done // max(1, total_all)}%)")
    print("regenerate the comparison anytime:  python experiments/isaac/compare.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
