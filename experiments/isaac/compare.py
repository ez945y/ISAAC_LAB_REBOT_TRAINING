#!/usr/bin/env python3
"""Compare the Isaac-in-the-loop results against the kinesim reference.

No Isaac needed. Loads per-episode rows from both pools, restricts each
(scenario, condition) cell to the seed intersection, re-aggregates with the
same mean±95%CI machinery, and writes one side-by-side markdown per
experiment plus a headline COMPARISON.md.

    python experiments/isaac/compare.py
    python experiments/isaac/compare.py --exp e2,e32
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from common.metrics import aggregate  # noqa: E402
from common.registry import KINESIM_DIRS, build_experiments  # noqa: E402

REPO = _HERE.parents[1]


def _parse(v: str):
    if v == "" or v is None:
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)   # handles 'nan'/'inf' too
    except ValueError:
        return v


def load_rows(path: Path) -> list[dict]:
    if (path / "episodes.csv").exists():
        with (path / "episodes.csv").open() as f:
            rows = [{k: _parse(v) for k, v in r.items() if _parse(v) is not None}
                    for r in csv.DictReader(f)]
    elif (path / "episodes.jsonl").exists():
        rows = [json.loads(line) for line in
                (path / "episodes.jsonl").read_text().splitlines() if line.strip()]
    else:
        return []
    for r in rows:                       # e2 rows have method but no condition
        r.setdefault("condition", r.get("method"))
    return rows


def _cell(r: dict, f: str) -> str:
    v = r.get(f, math.nan)
    ci = r.get(f + "_ci95")
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        s = f"{v:.3f}"
        if isinstance(ci, (int, float)) and math.isfinite(ci) and ci > 0:
            s += f"±{ci:.3f}"
        return s
    return str(v)


def _flag(kv, kc, iv, ic) -> str:
    """≈ when the means sit inside each other's combined 95% CI (or within an
    absolute 0.05 when both CIs are degenerate); ≠ otherwise."""
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (kv, iv)):
        return ""
    tol = max((kc or 0.0) + (ic or 0.0), 0.05)
    return "≈" if abs(kv - iv) <= tol else "≠"


def compare_experiment(key: str, exp, kdir: Path, idir: Path) -> str | None:
    krows, irows = load_rows(kdir), load_rows(idir)
    if not krows or not irows:
        return None
    cells: dict[tuple, dict[str, list]] = {}
    for src, rows in (("kinesim", krows), ("isaac", irows)):
        for r in rows:
            cells.setdefault((r["scenario"], r["condition"]), {}).setdefault(src, []).append(r)

    lines = [f"## {exp.name} — kinesim (real dam) vs Isaac (Go2 + real dam)", ""]
    fields = exp.md_fields
    for (scen, cond), by_src in sorted(cells.items()):
        k, i = by_src.get("kinesim", []), by_src.get("isaac", [])
        seeds = sorted({r["seed"] for r in k} & {r["seed"] for r in i})
        if not seeds:
            continue
        k = [r for r in k if r["seed"] in seeds]
        i = [r for r in i if r["seed"] in seeds]
        ka = aggregate(k, keys=("scenario", "condition"))[0]
        ia = aggregate(i, keys=("scenario", "condition"))[0]
        falls = sum(r.get("n_falls", 0) for r in i)
        lines.append(f"### {scen} / {cond}  (n={len(seeds)} matched seeds"
                     + (f", isaac falls={falls}" if falls else "") + ")")
        lines.append("| metric | kinesim | isaac | match |")
        lines.append("|---|---|---|---|")
        for f in fields:
            base = f[:-5] if f.endswith("_rate") else f
            if f not in ka and f not in ia and base not in ka:
                continue
            fk = f if f in ka else base
            lines.append(f"| {f} | {_cell(ka, fk)} | {_cell(ia, fk)} | "
                         f"{_flag(ka.get(fk), ka.get(fk + '_ci95'), ia.get(fk), ia.get(fk + '_ci95'))} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="kinesim vs Isaac comparison")
    ap.add_argument("--exp", default="all")
    ap.add_argument("--kinesim-root", default=str(REPO / "experiments/results"))
    ap.add_argument("--isaac-root", default=str(_HERE / "results"))
    ap.add_argument("--out", default=str(_HERE / "results/COMPARISON.md"))
    args = ap.parse_args()

    exps = build_experiments()
    keys = list(exps) if args.exp == "all" else \
        [k.strip() for k in args.exp.split(",") if k.strip()]
    sections = []
    for k in keys:
        exp = exps[k]
        sec = compare_experiment(k, exp, Path(args.kinesim_root) / KINESIM_DIRS[k],
                                 Path(args.isaac_root) / exp.key)
        if sec is None:
            print(f"[compare] {k}: missing episodes on one side — skipped")
            continue
        sections.append(sec)
        print(f"[compare] {k}: ok")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Isaac-in-the-loop vs kinesim — consistency comparison\n\n"
              "Same scenarios, same seeds (intersection), same metrics pipeline, "
              "same REAL dam 0.7 guard; only the execution differs "
              "(first-order holonomic integration vs Go2 RL policy + PhysX). "
              "`match` = means within combined 95% CI (min tol 0.05).\n\n")
    out.write_text(header + "\n\n".join(sections) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
