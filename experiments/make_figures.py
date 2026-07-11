#!/usr/bin/env python3
"""Thesis figures for Chapter 4 — regenerated from experiments/results/*.

Run with the DAM venv (matplotlib):
    "$VENV" experiments/make_figures.py

Design rules follow the data-viz method: one axis per chart (no dual axes),
categorical hues in fixed validated order (blue/aqua/yellow/green — palette
validated for CVD, low-contrast slots relieved with direct labels), thin
marks, recessive grid, direct labels over legends where ≤ 4 series, hard
floor drawn as a labelled neutral reference line. Output: PNG (300 dpi) +
PDF into experiments/results/figures/.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / "results"
OUT = ROOT / "figures"

# validated categorical palette (light mode), fixed order
C = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100",
     "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
FLOOR = 0.7

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
    "savefig.bbox": "tight", "savefig.dpi": 300,
})


def load(exp: str, key: str = "condition") -> dict:
    """aggregate.csv -> {(scenario, condition): {field: float}}"""
    out = {}
    with open(ROOT / exp / "aggregate.csv") as f:
        for r in csv.DictReader(f):
            k = (r["scenario"], r.get(key) or r.get("method"))
            out[k] = {c: (float(v) if v not in ("", "nan") else math.nan)
                      for c, v in r.items() if c not in ("scenario", key, "method")}
    return out


def floor_line(ax, label=True):
    ax.axhline(FLOOR, ls="--", lw=1.0, color=INK2, zorder=1)
    if label:
        ax.annotate("hard floor 0.7 m", xy=(0.02, FLOOR), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=INK2)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- fig 4-1
def fig_e2_pareto():
    agg = load("e2_full", key="method")
    methods = [("raw", C["blue"]), ("stop", C["aqua"]),
               ("orca", C["yellow"]), ("dam", C["green"])]
    scens = ["S1", "S2", "S3", "S4", "S5"]
    titles = {"S1": "S1 crossing", "S2": "S2 head-on", "S3": "S3 bottleneck",
              "S4": "S4 corridor", "S5": "S5 cruise"}
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.7), sharey=True)
    for ax, sc in zip(axes, scens):
        floor_line(ax, label=(sc == "S1"))
        for m, col in methods:
            d = agg[(sc, m)]
            x, y = d["makespan_s"], d["min_dogdog_m"]
            ax.errorbar(x, y, xerr=d["makespan_s_ci95"], yerr=d["min_dogdog_m_ci95"],
                        fmt="o", ms=5, color=col, ecolor=col, elinewidth=0.8,
                        capsize=0, zorder=3)
            dx, dy = (-4, 5) if m == "stop" else (5, 4)
            ax.annotate(m, (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, color=INK, ha="left" if dx > 0 else "right")
        ax.set_title(titles[sc], fontsize=9)
        ax.set_xlim(0, 66)
        ax.set_xticks([0, 20, 40, 60])
    axes[0].set_ylabel("min inter-dog capsule distance (m)")
    axes[0].set_ylim(-0.05, 1.45)
    axes[2].set_xlabel("makespan (s)  —  60 s = timeout")
    fig.suptitle("Safety vs throughput per scenario (50 seeds; up-left is better; "
                 "raw collides, stop times out, DAM alone holds the floor and finishes)",
                 fontsize=9, y=1.06)
    save(fig, "fig4-1_e2_pareto")


# ---------------------------------------------------------------- fig 4-2
def fig_e33_layering():
    agg = load("e33_softhard")
    conds = ["layered", "hard_only", "comfort_hard"]
    labels = ["layered\n(hard+soft)", "hard only", "comfort\nas hard"]
    panels = [("all_done_rate", "completion rate", None, 1.08),
              ("min_dogdog_m", "min distance (m)", FLOOR, 1.1),
              ("viol_steps_dog", "floor-violation steps", None, None)]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.6))
    for ax, (f, ttl, fl, top) in zip(axes, panels):
        vals = [agg[("S3", c)][f] for c in conds]
        errs = [agg[("S3", c)].get(f + "_ci95", 0) or 0 for c in conds]
        bars = ax.bar(labels, vals, width=0.55, color=C["blue"], zorder=3)
        ax.errorbar(labels, vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=0.8, zorder=4)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}" if v < 10 else f"{v:.0f}",
                        (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8)
        if fl:
            floor_line(ax, label=False)
        if top:
            ax.set_ylim(0, max(max(vals) * 1.25, top))
        ax.set_title(ttl, fontsize=9)
    fig.suptitle("E3.3 — soft/hard layering on S3 bottleneck (20 seeds): removing the soft "
                 "tier costs BOTH liveness and safety; hardening comfort clogs the funnel",
                 fontsize=9, y=1.08)
    save(fig, "fig4-2_e33_layering")


# ---------------------------------------------------------------- fig 4-3
def fig_e35_velocity():
    agg = load("e35_velocity")
    vs = [1.0, 1.5, 2.0]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.7))
    series = [("va_on", "velocity-aware", C["blue"]),
              ("va_off", "static assumption", C["red"])]
    for ax, (f, ttl, fl) in zip(axes, [("min_dogdog_m", "min distance (m)", True),
                                       ("viol_steps_dog", "floor-violation steps", False)]):
        for key, lab, col in series:
            ys = [agg[("S2", f"{key}_v{v}")][f] for v in vs]
            ax.plot(vs, ys, "-o", ms=5, lw=1.6, color=col, zorder=3)
            ax.annotate(lab, (vs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=INK, va="center")
        if fl:
            floor_line(ax)
            ax.set_ylim(0.55, 0.95)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("vmax (m/s) — closing speed = 2·vmax")
        ax.set_xticks(vs)
        ax.set_xlim(0.9, 2.45)
    fig.suptitle("E3.5 — velocity-aware prediction on S2 head-on (20 seeds): the static "
                 "assumption halves the perceived closing rate and dips deeper as speed rises",
                 fontsize=9, y=1.06)
    save(fig, "fig4-3_e35_velocity")


# ---------------------------------------------------------------- fig 4-4
def fig_e36_gamma():
    agg = load("e36_gamma")
    gammas = [0.1, 0.2, 0.4, 0.7, 1.0]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 4.6), sharex=True)
    for j, sc in enumerate(["S1", "S3"]):
        for i, (f, ylab, fl) in enumerate([("min_dogdog_m", "min distance (m)", True),
                                           ("all_done_rate", "completion rate", False)]):
            ax = axes[i][j]
            ys = [agg[(sc, f"g{g}_dt0.2")][f] for g in gammas]
            es = [agg[(sc, f"g{g}_dt0.2")].get(f + "_ci95", 0) or 0 for g in gammas]
            ax.errorbar(gammas, ys, yerr=es, fmt="-o", ms=5, lw=1.6,
                        color=C["blue"], ecolor=C["blue"], elinewidth=0.8, zorder=3)
            ax.axvline(0.4, ls=":", lw=1.0, color=INK2, zorder=1)
            if i == 0:
                ax.set_title(f"{sc}", fontsize=9)
                floor_line(ax, label=(j == 0))
                ax.set_ylim(0, 1.0)
            else:
                ax.set_ylim(0, 1.1)
                ax.set_xlabel("γ  (dt = 0.2 s;  dotted line = stackfile default)")
            if j == 0:
                ax.set_ylabel(ylab)
            ax.set_xticks(gammas)
    fig.suptitle("E3.6 — γ sweep (20 seeds): safety margin rises MONOTONICALLY with γ while "
                 "liveness falls — low γ erodes via slack + the tangential blind spot (F9), "
                 "high γ builds a brake-wall", fontsize=9, y=1.0)
    save(fig, "fig4-4_e36_gamma")


# ---------------------------------------------------------------- fig 4-5
def fig_e42_latency():
    agg = load("e42_obsnoise")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.7), sharey=True)
    ax = axes[0]
    xs = [0, 100, 200, 500]
    ys = [agg[("S2", "noise0.0")]["min_dogdog_m"]] + \
         [agg[("S2", f"delay{d}ms")]["min_dogdog_m"] for d in (100, 200, 500)]
    ax.plot(xs, ys, "-o", ms=5, lw=1.6, color=C["blue"], zorder=3)
    floor_line(ax)
    ax.set_title("observation latency (σ_obs = 0)", fontsize=9)
    ax.set_xlabel("delay (ms)")
    ax.set_ylabel("min distance (m), S2")
    ax.set_xticks(xs)
    ax = axes[1]
    sig = [0.0, 0.05, 0.1, 0.2]
    ys = [agg[("S2", f"noise{s}")]["min_dogdog_m"] for s in sig]
    ax.plot(sig, ys, "-o", ms=5, lw=1.6, color=C["blue"], zorder=3)
    floor_line(ax, label=False)
    ax.set_title("observation noise (delay = 0)", fontsize=9)
    ax.set_xlabel("σ_obs (m)")
    ax.set_xticks(sig)
    axes[0].set_ylim(0.45, 0.9)
    fig.suptitle("E4.2 — perception degradation on S2 (20 seeds): latency binds first; "
                 "velocity extrapolation carries cleanly to ~200 ms",
                 fontsize=9, y=1.06)
    save(fig, "fig4-5_e42_perception")


# ---------------------------------------------------------------- fig 4-6
def fig_e44_rogue():
    agg = load("e44_rogue")
    conds = ["coop", "rogue1", "rogue2"]
    xt = ["0 rogue", "1 rogue", "2 rogues"]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    w = 0.34
    for k, (sc, col, lab) in enumerate([("S1", C["blue"], "S1 crossing"),
                                        ("S5", C["aqua"], "S5 cruise")]):
        vals = [agg[(sc, c)]["min_dogdog_m"] for c in conds]
        errs = [agg[(sc, c)]["min_dogdog_m_ci95"] for c in conds]
        xs = [i + (k - 0.5) * (w + 0.03) for i in range(3)]
        bars = ax.bar(xs, vals, width=w, color=col, zorder=3, label=lab)
        ax.errorbar(xs, vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=0.8, zorder=4)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7.5)
    floor_line(ax, label=False)
    ax.annotate("hard floor 0.7 m", xy=(0.99, FLOOR), xycoords=("axes fraction", "data"),
                xytext=(0, -10), textcoords="offset points", fontsize=7.5,
                color=INK2, ha="right")
    ax.set_xticks(range(3), xt)
    ax.set_ylabel("min distance (m)")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("E4.4 — unfiltered agents erode the floor dose-dependently\n"
                 "(compliant dogs cover only their half-share) but nothing collides",
                 fontsize=9)
    save(fig, "fig4-6_e44_rogue")


# ---------------------------------------------------------------- fig 4-7
def fig_e45_scale():
    agg = load("e45_scale")
    ns = [2, 4, 8, 12, 16]
    panels = [("min_dogdog_m", "min distance (m)", True, None),
              ("makespan_s", "makespan (s)", False, (0, 12)),
              ("filter_p99_ms", "filter p99 (ms)", False, (0, 1.0))]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.6))
    for ax, (f, ylab, fl, ylim) in zip(axes, panels):
        ys = [agg[("S5", f"n{n}")][f] for n in ns]
        es = [agg[("S5", f"n{n}")].get(f + "_ci95", 0) or 0 for n in ns]
        ax.errorbar(ns, ys, yerr=es, fmt="-o", ms=5, lw=1.6,
                    color=C["blue"], ecolor=C["blue"], elinewidth=0.8, zorder=3)
        if fl:
            floor_line(ax)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(ylab, fontsize=9)
        ax.set_xticks(ns)
        ax.set_xlabel("number of dogs (fixed ±6 m arena)")
    fig.suptitle("E4.5 — scale sweep (20 seeds): 100% completion at every N; density erodes "
                 "the margin gracefully; p99 latency stays at 3.5% of the 50 Hz budget",
                 fontsize=9, y=1.08)
    save(fig, "fig4-7_e45_scale")


if __name__ == "__main__":
    fig_e2_pareto()
    fig_e33_layering()
    fig_e35_velocity()
    fig_e36_gamma()
    fig_e42_latency()
    fig_e44_rogue()
    fig_e45_scale()
    print("all figures written to", OUT)
