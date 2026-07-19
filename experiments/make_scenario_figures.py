#!/usr/bin/env python3
"""Scenario figures (S1–S5): the concrete layout (from the REAL scenario
generators, seed 0) plus the behaviour each experiment reads.

    python experiments/make_scenario_figures.py
    -> experiments/thesis/figures/scen_S{1..5}.png   (individual)
    -> experiments/thesis/figures/scen_all.{png,pdf} (combined 2x3, for thesis)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.transforms import Affine2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.scenarios import SCENARIOS  # noqa: E402

plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "Noto Sans CJK TC", "Noto Sans CJK JP",
    "Heiti TC", "Hiragino Sans GB", "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

# fixed categorical palette (validated reference instance; groups keep slots)
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
INK, MUT, GRID = "#26261f", "#6f6f67", "#e8e8e3"
CAP_H, CAP_R = 0.25, 0.16          # body capsule half-length / radius (draw)


def group_color(specs, spec) -> str:
    groups = sorted({s.group for s in specs if not s.static})
    return PALETTE[groups.index(spec.group) % len(PALETTE)]


def draw_capsule(ax, x, y, yaw, color):
    tr = Affine2D().rotate_around(x, y, yaw) + ax.transData
    ax.add_patch(Rectangle((x - CAP_H, y - CAP_R), 2 * CAP_H, 2 * CAP_R,
                           facecolor=color, edgecolor="white", linewidth=0.8,
                           transform=tr, zorder=5))
    for s in (-1, 1):
        ax.add_patch(Circle((x + s * CAP_H * math.cos(yaw),
                             y + s * CAP_H * math.sin(yaw)), CAP_R,
                            facecolor=color, edgecolor="white",
                            linewidth=0.8, zorder=5))
    # nose marker: small white dot at the front end
    ax.plot([x + CAP_H * math.cos(yaw)], [y + CAP_H * math.sin(yaw)],
            "o", color="white", markersize=2.5, zorder=6)


def draw_route(ax, spec, color):
    pts = [(spec.x, spec.y), *spec.waypoints, (spec.gx, spec.gy)]
    for a, b in zip(pts[:-1], pts[1:]):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                                     linestyle=(0, (4, 3)), color=color,
                                     linewidth=1.1, alpha=0.75, zorder=3))
    ax.plot([spec.gx], [spec.gy], marker="*", markersize=13, color=color,
            markeredgecolor="white", markeredgewidth=0.6, zorder=6)


def draw_walls(ax, specs):
    chains: dict[str, list] = {}
    for s in specs:
        if s.static:
            chains.setdefault(s.aid.rstrip("0123456789"), []).append(s)
    for pts in chains.values():
        ax.plot([p.x for p in pts], [p.y for p in pts], color=INK,
                linewidth=5, solid_capstyle="round", zorder=4)
        ax.plot([p.x for p in pts], [p.y for p in pts], ".", color="white",
                markersize=2, zorder=4.5)


def _wrap(s: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(s, width))


def setup_ax(ax, title, subtitle, read, xlim, ylim, compact,
             note_xy=(0.02, 0.02)):
    fs_title, fs_note, wrap_w = (15, 11, 34) if compact else (12, 9, 34)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(labelsize=8.5 if compact else 8, color=GRID)
    for side in ax.spines.values():
        side.set_color(GRID)
    ax.set_title(title + "\n" + subtitle, loc="left", fontsize=fs_title,
                 color=INK, pad=8)
    if compact:
        # fill the grid cell (aligned titles); note goes below the panel so it
        # can never cover the drawing
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(_wrap("觀察重點：" + read, 30), fontsize=fs_note,
                      loc="left", color=INK, labelpad=10)
        ax.xaxis.label.set_bbox(dict(boxstyle="round,pad=0.45",
                                     facecolor="#f6f6f2", edgecolor=GRID))
    else:
        ax.set_aspect("equal")
        ax.text(*note_xy, _wrap("觀察重點：" + read, wrap_w),
                transform=ax.transAxes,
                fontsize=fs_note, color=INK, va="bottom",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#f6f6f2",
                          edgecolor=GRID))


def scene_s1(ax, compact=False):
    specs = SCENARIOS["S1"](0)
    setup_ax(ax,
             "S1 交叉（crossing）",
             "兩隊各 2 台，路徑十字交會於中央",
             "優先權讓行（高優先隊直行、低優先隊繞行或等待）、"
             "對稱交會的死鎖風險、交會區最小間距",
             (-6.5, 6.5), (-6.5, 6.5), compact)
    for s in specs:
        c = group_color(specs, s)
        draw_capsule(ax, s.x, s.y, s.yaw, c)
        draw_route(ax, s, c)
    ax.add_patch(Circle((0, 0), 2.0, facecolor="none", edgecolor="#e34948",
                        linewidth=1.2, linestyle=(0, (3, 3)), zorder=2))
    fs = 12 if compact else 9
    ax.annotate("交會衝突區", (0.1, 2.15), fontsize=fs, color="#e34948")
    ax.annotate("G0 隊（西→東）", (-6.2, 1.9), fontsize=fs, color=PALETTE[0])
    ax.annotate("G1 隊（南→北）", (1.7, -4.5), fontsize=fs, color=PALETTE[1])


def scene_s2(ax, compact=False):
    specs = SCENARIOS["S2"](0)
    setup_ax(ax,
             "S2 對頭（head-on swap）",
             "兩台正面互換位置，航線正對；起點僅 ±0.15 m 側向擾動",
             "切向繞行能否打破對稱僵局、速度感知預測的提前反應、"
             "0.7 m 硬性底線是否守住",
             (-5.6, 5.6), (-4.2, 4.2), compact)
    for s in specs:
        c = group_color(specs, s)
        draw_capsule(ax, s.x, s.y, s.yaw, c)
        draw_route(ax, s, c)
    a = specs[0]
    ax.add_patch(Circle((a.x, a.y), 0.7, facecolor="none", edgecolor="#e34948",
                        linewidth=1.1, linestyle=(0, (3, 3)), zorder=2))
    fs = 12 if compact else 9
    ax.annotate("硬性底線 0.7 m", (a.x - 0.6, a.y + 0.85), fontsize=fs - 0.5,
                color="#e34948")
    # expected mutual circulation: two curved arrows around the midpoint
    for sgn in (+1, -1):
        ax.add_patch(FancyArrowPatch((sgn * 1.6, sgn * 0.45), (-sgn * 1.6, sgn * 0.45),
                                     connectionstyle=f"arc3,rad={0.35 * sgn}",
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=MUT, linewidth=1.2, zorder=3))
    ax.annotate("期望：同向切向繞行錯車", (-2.5, 1.9), fontsize=fs, color=MUT)


def scene_s3(ax, compact=False):
    specs = SCENARIOS["S3"](0)
    setup_ax(ax,
             "S3 瓶頸（bottleneck）",
             "4 台從南側經 1.6 m 縫隙到北側；路徑統一經縫隙中心",
             "瓶頸前的排隊與僵持、軟硬分層的提早讓行、"
             "擁擠對硬性底線的侵蝕",
             (-4.5, 4.5), (-7.5, 7.0), compact)
    draw_walls(ax, specs)
    for s in specs:
        if s.static:
            continue
        draw_capsule(ax, s.x, s.y, s.yaw, PALETTE[0])
        draw_route(ax, s, PALETTE[0])
    fs = 12 if compact else 9
    ax.annotate("牆（點鏈，硬約束）", (-3.4, 0.45), fontsize=fs, color=INK)
    ax.annotate("1.6 m 縫隙（一次僅容一台通過）", (0.0, -1.05),
                fontsize=fs - 0.5, color="#e34948", ha="center")


def scene_s4(ax, compact=False):
    specs = SCENARIOS["S4"](0)
    setup_ax(ax,
             "S4 走廊（corridor pass）",
             "2.2 m 走廊內兩台對向會車，必須錯車、無法繞出",
             "長條身體的幾何處理：膠囊模型允許貼身錯車；"
             "內接圓盤誤判安全、外接圓盤判定無法通過",
             (-5.2, 5.2), (-3.4, 3.4), compact)
    draw_walls(ax, specs)
    for s in specs:
        if s.static:
            continue
        c = group_color(specs, s)
        draw_capsule(ax, s.x, s.y, s.yaw, c)
        draw_route(ax, s, c)
    fs = 12 if compact else 9
    ax.annotate("走廊寬 2.2 m", (-1.3, 1.35), fontsize=fs, color=INK)
    ax.annotate("期望：靠邊側身錯車", (-1.85, -1.5), fontsize=fs, color=MUT)


def scene_s5(ax, compact=False):
    specs = SCENARIOS["S5"](0)
    setup_ax(ax,
             "S5 巡航（random cruise）",
             "N 台（預設 6，可掃至 16）隨機起訖、各自獨立任務",
             "間距維持與完成時間的長期平衡、規模與密度壓力、"
             "混入非合作代理的效應",
             (-7.0, 7.0), (-7.0, 7.0), compact)
    for i, s in enumerate(specs):
        c = PALETTE[i % len(PALETTE)]
        draw_capsule(ax, s.x, s.y, s.yaw, c)
        draw_route(ax, s, c)
    ax.annotate("±6 m 固定場地（N 增加 = 密度上升）", (-6.7, 6.35),
                fontsize=12 if compact else 9, color=MUT)


SCENES = [("S1", scene_s1), ("S2", scene_s2), ("S3", scene_s3),
          ("S4", scene_s4), ("S5", scene_s5)]


def main() -> int:
    out = Path(__file__).resolve().parent / "thesis/figures"
    out.mkdir(parents=True, exist_ok=True)

    # individual figures (working copies)
    for key, draw in SCENES:
        fig, ax = plt.subplots(figsize=(7.0, 6.2))
        draw(ax, compact=False)
        fig.tight_layout()
        path = out / f"scen_{key}.png"
        fig.savefig(path, dpi=150, facecolor="white")
        plt.close(fig)
        print("wrote", path)

    # combined landscape 2x3 grid (thesis figure; last cell blank)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.0),
                             constrained_layout=True)
    flat = axes.ravel()
    for (key, draw), ax in zip(SCENES, flat):
        draw(ax, compact=True)
    flat[-1].axis("off")
    for ext in ("png", "pdf"):
        path = out / f"scen_all.{ext}"
        fig.savefig(path, dpi=150, facecolor="white")
        print("wrote", path)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
