#!/usr/bin/env python3
"""RQ5（失敗樣本收集與再利用）論文圖 — 從 isaac/results/e5_loop 重生。

    source ~/IsaacLab/env_isaaclab/bin/activate
    python experiments/make_e5_figures.py

配色與樣式沿用 make_figures.py 的驗證過調色盤（色盲友善、固定順序、細筆、
淡格線、少圖例多直標）。輸出 PNG(300dpi)+PDF 到 isaac/results/e5_loop/figures/。
每張圖的說明見 experiments/E5_FIGURES.md。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / "isaac" / "results" / "e5_loop"
OUT = ROOT / "figures"

C = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100",
     "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
    "savefig.bbox": "tight", "savefig.dpi": 300,
})

# 中文字型（Isaac 機器上有 Noto CJK；找不到就退回預設，英文標籤仍可讀）
for _f in ("Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP",
           "WenQuanYi Zen Hei", "Droid Sans Fallback"):
    try:
        import matplotlib.font_manager as fm
        if any(_f in f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [_f]
            plt.rcParams["axes.unicode_minus"] = False
            break
    except Exception:
        pass


def load_bench() -> dict:
    """{(scenario, condition): {field: float}} from bench/aggregate.csv."""
    out = {}
    with open(ROOT / "bench" / "aggregate.csv") as f:
        for r in csv.DictReader(f):
            k = (r["scenario"], r["condition"])
            out[k] = {c: (float(v) if v not in ("", "nan") else math.nan)
                      for c, v in r.items() if c not in ("scenario", "condition")}
    return out


def load_quality() -> dict:
    """{pool: quality dict} from datasets/*.quality.json."""
    q = {}
    for name in ("base", "extra", "damfix"):
        p = ROOT / "datasets" / f"go2squad_{name}.quality.json"
        if p.exists():
            q[name] = json.loads(p.read_text())
    return q


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)
    print(f"wrote {OUT / stem}.png / .pdf")


# 中文標籤（用論文正文的詞：策略/過濾器/原始·對照·修正資料/控制器；
# 程式代號只在資料鍵）
POOL_ZH = {"base": "原始資料", "extra": "對照資料", "damfix": "修正資料"}
POOL_SEEDS = {"base": 150, "extra": 60, "damfix": 60}
COND_ZH = {"nominal+dam": "控制器＋過濾器", "bc_v0+dam": "初版策略＋過濾器",
           "bc_v1c+dam": "對照策略＋過濾器", "bc_v1+dam": "改良策略＋過濾器"}


def fig_completeness(q: dict) -> None:
    """圖1：三個資料池的每局邊界事件數（完整性＋護欄修正池最密）。"""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    pools = ["base", "extra", "damfix"]
    colors = [C["blue"], C["yellow"], C["green"]]
    ev_per_seed = [q[p]["events"] / POOL_SEEDS[p] for p in pools]
    xs = range(len(pools))
    ax.bar(xs, ev_per_seed, width=0.6, color=colors)
    for x, p, v in zip(xs, pools, ev_per_seed):
        ax.annotate(f"{v:.1f}", xy=(x, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.annotate(f"{q[p]['events']} 事件\n欄位完整率 {q[p]['field_completeness']:.3f}",
                    xy=(x, 0), xytext=(0, 12), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color="white")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([POOL_ZH[p] for p in pools])
    ax.set_ylabel("每個收集局的邊界事件數")
    ax.set_title("資料池品質：修正資料的事件最密（策略犯錯多之處，修正紀錄就密）",
                 fontsize=9.5, loc="left")
    save(fig, "e5_1_completeness")


def _grouped_intervention(ax, b: dict, scen: str, conds: list) -> None:
    colors = {"nominal+dam": INK2, "bc_v0+dam": C["blue"],
              "bc_v1c+dam": C["yellow"], "bc_v1+dam": C["green"]}
    xs = range(len(conds))
    vals = [b[(scen, c)]["intervention_rate"] for c in conds]
    err = [b[(scen, c)].get("intervention_rate_ci95", 0.0) for c in conds]
    ax.bar(xs, vals, yerr=err, width=0.62, capsize=3,
           color=[colors[c] for c in conds])
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:.3f}", xy=(x, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([COND_ZH[c] for c in conds], fontsize=7.5, rotation=12)
    ax.set_ylabel("過濾器介入率（越低＝越少靠過濾器）")
    ax.set_title(scen, fontsize=9.5, loc="left")


def fig_reuse(b: dict) -> None:
    """圖2：護欄介入率 by 條件（核心再利用結果＋對照）。"""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.6))
    _grouped_intervention(a1, b, "S2", ["nominal+dam", "bc_v0+dam", "bc_v1+dam"])
    _grouped_intervention(a2, b, "S5",
                          ["nominal+dam", "bc_v0+dam", "bc_v1c+dam", "bc_v1+dam"])
    top = max(a1.get_ylim()[1], a2.get_ylim()[1])
    a1.set_ylim(0, top); a2.set_ylim(0, top)
    fig.suptitle("再利用價值：重訓後的改良策略最少依賴過濾器，且勝過吃等量資料的對照策略",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "e5_2_reuse_intervention")


def fig_tradeoff(b: dict) -> None:
    """圖3：S5 完成率 vs 違規步（誠實的雙面代價）。"""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    conds = ["nominal+dam", "bc_v0+dam", "bc_v1c+dam", "bc_v1+dam"]
    colors = {"nominal+dam": INK2, "bc_v0+dam": C["blue"],
              "bc_v1c+dam": C["yellow"], "bc_v1+dam": C["green"]}
    for c in conds:
        d = b[("S5", c)]
        ax.scatter(d["all_done_rate"], d["viol_steps_dog"], s=90,
                   color=colors[c], zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate(COND_ZH[c], xy=(d["all_done_rate"], d["viol_steps_dog"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("完成率（越右＝越能走到終點）")
    ax.set_ylabel("違規步數（越低＝越安全）")
    ax.set_xlim(0, 1.08)
    ax.annotate("理想在右下：又安全又完成", xy=(1.08, ax.get_ylim()[0]),
                xytext=(-4, 4), textcoords="offset points", ha="right",
                fontsize=7.5, color=INK2)
    ax.set_title("S5 的雙面代價：改良策略最安全，卻也最走不到終點（完成率 0.15）",
                 fontsize=9.5, loc="left")
    save(fig, "e5_3_safety_liveness_tradeoff")


def main() -> int:
    b = load_bench()
    q = load_quality()
    if q:
        fig_completeness(q)
    fig_reuse(b)
    fig_tradeoff(b)
    print(f"\n圖輸出於 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
