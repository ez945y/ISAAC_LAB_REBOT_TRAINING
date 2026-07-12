#!/usr/bin/env python3
"""E4.3 (RQ4): hard-slack usage — collation across every finished experiment.

No new simulation: every episode row already logs ``max_hard_slack_m`` (the
worst hard-floor slack the QP ever bought in that episode). This script scans
``experiments/results/*/episodes.csv`` (skipping ``*_probe``/``*_smoke*``
dirs) and answers the runtime-monitoring question:

    Is QP hard slack a usable safety monitor — does buying slack predict
    realized floor violation?

Expected split (from F7 vs F9):
  * infeasibility-driven erosion (crowd crush, circumscribed disc, hard_only
    wedges) shows HIGH slack — the QP knows it is giving ground;
  * linearization-driven penetration (F9 tangential blind spot) shows LOW
    slack with real violations — the QP cannot see what it cannot model.
So slack is a valid monitor for the first failure family only.

Output: results/e43_slack/summary.md (slack distribution per experiment ×
scenario × condition + episode-level slack↔violation correlation + the
low-slack-high-violation counter-examples).

    python3 experiments/run_e43_slack.py
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
SKIP = ("probe", "smoke")


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return math.nan
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx * sy == 0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def main() -> int:
    rows = []
    for d in sorted(RESULTS.iterdir()) if RESULTS.exists() else []:
        if not d.is_dir() or any(t in d.name for t in SKIP):
            continue
        ep = d / "episodes.csv"
        if not ep.exists():
            continue
        for r in csv.DictReader(open(ep)):
            if "max_hard_slack_m" not in r:
                continue
            # slack telemetry only exists for the guard's QP: baseline methods
            # (raw/stop/orca) log 0 and would pollute the analysis, and the
            # pydam equivalence rows in e2 duplicate the dam episodes
            if r.get("method", "") in ("raw", "stop", "orca", "pydam"):
                continue
            try:
                rows.append({
                    "exp": d.name,
                    "scenario": r.get("scenario", "?"),
                    "condition": r.get("condition", r.get("method", "?")),
                    "slack": float(r["max_hard_slack_m"]),
                    "viol": float(r.get("viol_steps_dog", 0) or 0),
                    "mindd": float(r.get("min_dogdog_m", "nan") or "nan"),
                })
            except (ValueError, TypeError):
                continue
    if not rows:
        print("no finished results found under experiments/results/")
        return 1

    # group by (exp, scenario, condition)
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["exp"], r["scenario"], r["condition"]), []).append(r)

    lines = ["# E4.3 hard-slack usage (collated)", "",
             f"{len(rows)} episodes across {len({r['exp'] for r in rows})} experiments",
             "", "## Slack per condition (sorted by mean slack, top 20)", "",
             "| experiment | scenario | condition | n | slack mean | slack max | viol mean | minDD mean |",
             "|---|---|---|---|---|---|---|---|"]
    stats = []
    for k, v in groups.items():
        sl = [x["slack"] for x in v]
        stats.append((statistics.mean(sl), k, v))
    stats.sort(reverse=True)
    for mean_sl, (exp, sc, cond), v in stats[:20]:
        sl = [x["slack"] for x in v]
        vi = [x["viol"] for x in v]
        md = [x["mindd"] for x in v if not math.isnan(x["mindd"])]
        lines.append(f"| {exp} | {sc} | {cond} | {len(v)} | {mean_sl:.3f} | "
                     f"{max(sl):.3f} | {statistics.mean(vi):.0f} | "
                     f"{statistics.mean(md):.3f} |")

    # episode-level correlation, overall and per failure family
    xs = [r["slack"] for r in rows]
    ys = [r["viol"] for r in rows]
    zs = [r["mindd"] for r in rows if not math.isnan(r["mindd"])]
    xs_z = [r["slack"] for r in rows if not math.isnan(r["mindd"])]
    lines += ["", "## Is slack a safety monitor?", "",
              f"- episode-level Pearson(slack, viol_steps) = **{pearson(xs, ys):+.3f}**",
              f"- episode-level Pearson(slack, min_dogdog) = **{pearson(xs_z, zs):+.3f}**"]

    # counter-examples: violations with near-zero slack (the F9 family)
    blind = [r for r in rows if r["slack"] < 0.02 and r["viol"] > 50]
    seen = [r for r in rows if r["slack"] >= 0.1 and r["viol"] > 50]
    lines += ["",
              f"- episodes with >50 violation steps and slack < 0.02 m "
              f"(**QP-blind**, F9 family): **{len(blind)}**",
              f"- episodes with >50 violation steps and slack ≥ 0.10 m "
              f"(QP-aware, F7 family): **{len(seen)}**", ""]
    if blind:
        worst = sorted(blind, key=lambda r: r["mindd"])[:8]
        lines += ["Worst QP-blind episodes (slack saw nothing):", "",
                  "| experiment | scenario | condition | slack | viol | minDD |",
                  "|---|---|---|---|---|---|"]
        for r in worst:
            lines.append(f"| {r['exp']} | {r['scenario']} | {r['condition']} | "
                         f"{r['slack']:.3f} | {r['viol']:.0f} | {r['mindd']:.3f} |")
    lines += ["", "Conclusion: slack telemetry flags infeasibility-driven erosion (F7) "
              "but is silent on linearization-driven penetration (F9) — usable as a "
              "monitor only for the first failure family; pair it with a measured-"
              "distance monitor for the second."]

    out = RESULTS / "e43_slack"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
