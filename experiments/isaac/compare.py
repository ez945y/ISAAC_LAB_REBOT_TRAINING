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


def collect(exp, kdir: Path, idir: Path) -> list[tuple] | None:
    """[(scenario, condition, n_seeds, isaac_falls, k_agg, i_agg)] on matched seeds."""
    krows, irows = load_rows(kdir), load_rows(idir)
    if not krows or not irows:
        return None
    cells: dict[tuple, dict[str, list]] = {}
    for src, rows in (("kinesim", krows), ("isaac", irows)):
        for r in rows:
            cells.setdefault((r["scenario"], r["condition"]), {}).setdefault(src, []).append(r)
    out = []
    for (scen, cond), by_src in sorted(cells.items()):
        k, i = by_src.get("kinesim", []), by_src.get("isaac", [])
        seeds = sorted({r["seed"] for r in k} & {r["seed"] for r in i})
        if not seeds:
            continue
        k = [r for r in k if r["seed"] in seeds]
        i = [r for r in i if r["seed"] in seeds]
        out.append((scen, cond, len(seeds), sum(r.get("n_falls", 0) for r in i),
                    aggregate(k, keys=("scenario", "condition"))[0],
                    aggregate(i, keys=("scenario", "condition"))[0]))
    return out


def render_markdown(exp, cells: list[tuple]) -> str:
    lines = [f"## {exp.name} — kinesim (real dam) vs Isaac (Go2 + real dam)", ""]
    for scen, cond, n, falls, ka, ia in cells:
        lines.append(f"### {scen} / {cond}  (n={n} matched seeds"
                     + (f", isaac falls={falls}" if falls else "") + ")")
        lines.append("| metric | kinesim | isaac | match |")
        lines.append("|---|---|---|---|")
        for f in exp.md_fields:
            base = f[:-5] if f.endswith("_rate") else f
            if f not in ka and f not in ia and base not in ka:
                continue
            fk = f if f in ka else base
            lines.append(f"| {f} | {_cell(ka, fk)} | {_cell(ia, fk)} | "
                         f"{_flag(ka.get(fk), ka.get(fk + '_ci95'), ia.get(fk), ia.get(fk + '_ci95'))} |")
        lines.append("")
    return "\n".join(lines)


# Two fixed categorical slots (validated reference palette; never re-assigned):
_C_KINESIM = "#2a78d6"   # slot 1 (blue)  — kinesim, always
_C_ISAAC = "#1baf7a"     # slot 2 (aqua)  — Isaac, always
_PANELS = [("min_dogdog_m", "min capsule distance (m)"),
           ("makespan_s", "makespan (s)"),
           ("all_done", "completion rate")]


def render_figure(key: str, exp, cells: list[tuple], fig_dir: Path) -> Path:
    """One PNG per experiment: paired bars kinesim-vs-Isaac per condition with
    95% CI whiskers, one panel per headline metric, shared category axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{scen}\n{cond}" for scen, cond, *_ in cells]
    x = range(len(labels))
    fig, axes = plt.subplots(len(_PANELS), 1, sharex=True,
                             figsize=(max(6.4, 1.05 * len(labels) + 1.5),
                                      2.2 * len(_PANELS)))
    for ax, (field, title) in zip(axes, _PANELS):
        rate = field + "_rate"
        for off, src, color, idx in ((-0.19, "kinesim", _C_KINESIM, 4),
                                     (0.19, "isaac", _C_ISAAC, 5)):
            agg = [c[idx] for c in cells]
            fk = [rate if rate in a else field for a in agg]
            vals = [a.get(f, math.nan) for a, f in zip(agg, fk)]
            errs = [a.get(f + "_ci95") or 0.0 for a, f in zip(agg, fk)]
            ax.bar([xi + off for xi in x], vals, width=0.36, color=color,
                   yerr=errs, error_kw={"elinewidth": 1, "ecolor": "#666660"},
                   label=src, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_title(title, fontsize=9, loc="left", color="#40403a")
        ax.grid(axis="y", color="#e8e8e3", linewidth=0.8, zorder=0)
        ax.tick_params(labelsize=8, color="#c3c2b7")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        if field == "min_dogdog_m":
            ax.axhline(0.7, color="#8a8a82", linewidth=1, linestyle=(0, (4, 3)))
            ax.annotate("hard floor 0.7 m", xy=(-0.45, 0.7),
                        fontsize=7, color="#6f6f67", ha="left", va="bottom")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xticks(list(x), labels, fontsize=7.5)
    fig.suptitle(f"{exp.name}: kinesim vs Isaac (matched seeds)", fontsize=11,
                 x=0.02, ha="left", color="#26261f")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"cmp_{key}.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="kinesim vs Isaac comparison")
    ap.add_argument("--exp", default="all")
    ap.add_argument("--kinesim-root", default=str(REPO / "experiments/results"))
    ap.add_argument("--isaac-root", default=str(_HERE / "results"))
    ap.add_argument("--out", default=str(_HERE / "results/COMPARISON.md"))
    ap.add_argument("--no-figures", action="store_true",
                    help="skip PNG generation (figures land in results/figures/)")
    args = ap.parse_args()

    exps = build_experiments()
    keys = list(exps) if args.exp == "all" else \
        [k.strip() for k in args.exp.split(",") if k.strip()]
    fig_dir = Path(args.isaac_root) / "figures"
    sections = []
    for k in keys:
        exp = exps[k]
        cells = collect(exp, Path(args.kinesim_root) / KINESIM_DIRS[k],
                        Path(args.isaac_root) / exp.key)
        if not cells:
            print(f"[compare] {k}: missing episodes on one side — skipped")
            continue
        sec = render_markdown(exp, cells)
        if not args.no_figures:
            fig = render_figure(k, exp, cells, fig_dir)
            sec += f"\n![{exp.name} comparison](figures/{fig.name})\n"
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
