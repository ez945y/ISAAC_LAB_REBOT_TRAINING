#!/usr/bin/env python3
"""One annotated figure per scenario (S1–S5): the concrete layout (from the
REAL scenario generators, seed 0) plus what behaviour the experiment reads.

    python experiments/make_scenario_figures.py
    -> experiments/thesis/figures/scen_S{1..5}.png
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

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK TC", "sans-serif"]
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


def _wrap(s: str, width: int = 34) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(s, width))


def base_fig(title, subtitle, read, xlim, ylim, note_xy=(0.02, 0.02)):
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(labelsize=8, color=GRID)
    for side in ax.spines.values():
        side.set_color(GRID)
    ax.set_title(title + "\n" + subtitle, loc="left", fontsize=12, color=INK,
                 pad=8)
    ax.text(*note_xy, _wrap("要讀的行為：" + read), transform=ax.transAxes, fontsize=9,
            color=INK, va="bottom",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f6f6f2",
                      edgecolor=GRID))
    return fig, ax


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def main() -> int:
    out = Path(__file__).resolve().parent / "thesis/figures"
    out.mkdir(parents=True, exist_ok=True)

    # -- S1 crossing --------------------------------------------------------
    specs = SCENARIOS["S1"](0)
    fig, ax = base_fig(
        "S1 十字交叉（crossing）",
        "兩隊各 2 隻，路線互相垂直，在中央同時交會",
        "優先權讓行（高優先隊直行、低優先隊繞/等）、對稱時的交會死鎖（F2）、"
        "交會區最小間距",
        (-6.5, 6.5), (-6.5, 6.5))
    for s in specs:
        c = group_color(specs, s)
        draw_capsule(ax, s.x, s.y, s.yaw, c)
        draw_route(ax, s, c)
    ax.add_patch(Circle((0, 0), 2.0, facecolor="none", edgecolor="#e34948",
                        linewidth=1.2, linestyle=(0, (3, 3)), zorder=2))
    ax.annotate("交會衝突區", (0.1, 2.15), fontsize=9, color="#e34948")
    ax.annotate("G0 隊（西→東）", (-6.2, 1.9), fontsize=9, color=PALETTE[0])
    ax.annotate("G1 隊（南→北）", (1.7, -5.8), fontsize=9, color=PALETTE[1])
    finish(fig, out / "scen_S1.png")

    # -- S2 head-on ---------------------------------------------------------
    specs = SCENARIOS["S2"](0)
    fig, ax = base_fig(
        "S2 頭對頭互換（head-on swap）",
        "兩隻互換位置，航線正對；起點僅 ±0.15 m 側向抖動",
        "swirl 同向繞行是否打破僵局、速度感知（TTC）的提前反應、"
        "0.7 m 硬底線是否守住",
        (-5.6, 5.6), (-4.2, 4.2))
    for s in specs:
        c = group_color(specs, s)
        draw_capsule(ax, s.x, s.y, s.yaw, c)
        draw_route(ax, s, c)
    a = specs[0]
    ax.add_patch(Circle((a.x, a.y), 0.7, facecolor="none", edgecolor="#e34948",
                        linewidth=1.1, linestyle=(0, (3, 3)), zorder=2))
    ax.annotate("硬底線 0.7 m", (a.x - 0.6, a.y + 0.85), fontsize=8.5,
                color="#e34948")
    # expected mutual circulation: two curved arrows around the midpoint
    for sgn in (+1, -1):
        ax.add_patch(FancyArrowPatch((sgn * 1.6, sgn * 0.45), (-sgn * 1.6, sgn * 0.45),
                                     connectionstyle=f"arc3,rad={0.35 * sgn}",
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=MUT, linewidth=1.2, zorder=3))
    ax.annotate("期望：同向繞行錯身（swirl）", (-2.5, 1.9), fontsize=9, color=MUT)
    finish(fig, out / "scen_S2.png")

    # -- S3 bottleneck ------------------------------------------------------
    specs = SCENARIOS["S3"](0)
    fig, ax = base_fig(
        "S3 瓶頸漏斗（bottleneck）",
        "4 隻從南側經 1.6 m 缺口到北側；路線統一經缺口中心",
        "漏斗內的排隊與僵持、軟硬分層的讓行（拿掉軟層會怎樣）、"
        "群擠對硬底線的侵蝕（F7）",
        (-4.5, 4.5), (-7.5, 7.0))
    draw_walls(ax, specs)
    for s in specs:
        if s.static:
            continue
        draw_capsule(ax, s.x, s.y, s.yaw, PALETTE[0])
        draw_route(ax, s, PALETTE[0])
    ax.annotate("牆（點鏈，硬約束）", (-3.4, 0.45), fontsize=9, color=INK)
    ax.annotate("1.6 m 缺口（單車道：一次只過一隻）", (0.0, -1.05), fontsize=8.5,
                color="#e34948", ha="center")
    finish(fig, out / "scen_S3.png")

    # -- S4 corridor --------------------------------------------------------
    specs = SCENARIOS["S4"](0)
    fig, ax = base_fig(
        "S4 窄走廊會車（corridor pass）",
        "2.2 m 走廊內兩隻對向通過——必須錯車，繞不出去",
        "長條身體的幾何處理：膠囊模型允許貼身錯車；圓盤模型不是誤判安全"
        "（內接）就是判定過不去（外接）→ E3.4 / ORCA 的機制",
        (-5.2, 5.2), (-3.4, 3.4))
    draw_walls(ax, specs)
    for s in specs:
        if s.static:
            continue
        c = group_color(specs, s)
        draw_capsule(ax, s.x, s.y, s.yaw, c)
        draw_route(ax, s, c)
    ax.annotate("走廊寬 2.2 m", (-1.3, 1.35), fontsize=9, color=INK)
    ax.annotate("期望：靠邊側身錯車", (-1.85, -1.5), fontsize=9, color=MUT)
    finish(fig, out / "scen_S4.png")

    # -- S5 cruise ----------------------------------------------------------
    specs = SCENARIOS["S5"](0)
    fig, ax = base_fig(
        "S5 開放巡航（random cruise）",
        "N 隻（預設 6，可掃到 16）隨機起訖、各自獨立任務",
        "長期統計（間距維持 vs 完成時間的整體平衡）、規模/密度壓力（E4.5）、"
        "混入非合作者的效應（E4.4）",
        (-7.0, 7.0), (-7.0, 7.0))
    for i, s in enumerate(specs):
        c = PALETTE[i % len(PALETTE)]
        draw_capsule(ax, s.x, s.y, s.yaw, c)
        draw_route(ax, s, c)
    ax.annotate("±6 m 固定場地（N 增加 = 密度上升）", (-6.7, 6.35), fontsize=9,
                color=MUT)
    finish(fig, out / "scen_S5.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
