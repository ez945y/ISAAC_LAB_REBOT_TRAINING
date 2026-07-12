#!/usr/bin/env python3
"""E3.7 (RQ3): analytic surrogate vs naive autograd gradient — why the guard
freezes the constraint normal.

The production guard differentiates the PROJECTION of the predicted closest
pair onto the CURRENT separation normal (analytic, overshoot-safe). The naive
alternative — autograd through the true predicted capsule distance — is exact
for benign approaches (census: matches to ~3 decimals) but degenerates when
the predicted poses reach OVERLAP: distance clamps at 0 and its gradient
vanishes, so the QP holds a huge requirement with no usable direction. The
observable symptom is a spurious full stop (REJECT / zero command) or a
slack-absorbed non-reaction, exactly the failure that motivated the analytic
rewrite (see go2_dam_wrapper._capsule docstring; the historical retain_graph
incident is a separate implementation accident, not reproduced here).

Parts:
  1. census  — N random close-range capsule configs: gradient agreement vs
     distance bucket + degenerate-gradient fraction (writes census.md).
  2. sweep   — S2+S3 × {analytic, autograd} × vmax {1.5, 2.0}, 20 seeds:
     stop_steps / reject_steps / min_dogdog / filter p99 (autograd's torch
     cost). S2 exercises the benign-orbit regime (probe: no stops, but floor
     erosion + 2-3x latency); S3's crowd crush is where predicted overlap —
     and hence the degenerate-gradient stop mode — actually occurs.

Needs the DAM venv (torch) for the autograd conditions:
    "$VENV" experiments/run_e37_gradient.py --seeds 20
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ablation import Condition, run_sweep  # noqa: E402
from common.filters import make_guard  # noqa: E402
from common.pydam import seg_seg_closest  # noqa: E402

MD_FIELDS = ["stop_steps", "reject_steps", "min_dogdog_m", "viol_steps_dog",
             "all_done_rate", "deadlock_rate", "makespan_s",
             "intervention_rate", "filter_p99_ms"]


def census(n: int, out_path: Path) -> None:
    from common import torchgrad  # torch
    rng = random.Random(0)
    h, dt = 0.25, 0.2
    buckets: dict[str, list] = {"far (1.0-1.5)": [], "mid (0.5-1.0)": [],
                                "near (0.2-0.5)": []}
    degen = 0
    overlap_cases = 0
    for _ in range(n):
        cx, cy, syaw = 0.0, 0.0, rng.uniform(-math.pi, math.pi)
        ang = rng.uniform(-math.pi, math.pi)
        d_ctr = rng.uniform(0.2, 2.0)
        px, py = cx + d_ctr * math.cos(ang), cy + d_ctr * math.sin(ang)
        nyaw = rng.uniform(-math.pi, math.pi)
        cmd = (rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(-2, 2))
        vpx, vpy = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        cs, sn = math.cos(syaw), math.sin(syaw)
        s_off, t_off, dist_now, ax0, ay0, bx0, by0 = seg_seg_closest(
            cx, cy, cs, sn, h, px, py, math.cos(nyaw), math.sin(nyaw), h)
        if not (0.2 <= dist_now <= 1.5):
            continue
        sep = max(dist_now, 1e-6)
        nrx, nry = (ax0 - bx0) / sep, (ay0 - by0) / sep
        g_a = [(nrx * cs + nry * sn) * dt, (-nrx * sn + nry * cs) * dt,
               s_off * dt * (-nrx * math.sin(syaw) + nry * math.cos(syaw))]
        d_t, g_t = torchgrad.pred_dist_and_grad(
            cx, cy, syaw, cmd, dt, h, px, py, nyaw, h, vpx, vpy, 1.5, 2.0)
        na = math.hypot(g_a[0], g_a[1])
        nt = math.hypot(g_t[0], g_t[1])
        if d_t < 1e-3:                      # predicted overlap
            overlap_cases += 1
            if nt < 0.02:                   # gradient died at contact
                degen += 1
            continue
        cos = ((g_a[0] * g_t[0] + g_a[1] * g_t[1]) / (na * nt)) if na * nt > 1e-9 else 1.0
        key = ("far (1.0-1.5)" if dist_now >= 1.0 else
               "mid (0.5-1.0)" if dist_now >= 0.5 else "near (0.2-0.5)")
        buckets[key].append(cos)
    lines = ["# E3.7 gradient census", "",
             f"{n} random configs, dist_now in [0.2, 1.5], both capsules h=0.25", "",
             "| bucket | n | mean cos(analytic, autograd) | min cos |",
             "|---|---|---|---|"]
    for k, v in buckets.items():
        if v:
            lines.append(f"| {k} | {len(v)} | {sum(v)/len(v):.4f} | {min(v):.4f} |")
    lines += ["", f"predicted-overlap cases: {overlap_cases}; "
                  f"degenerate gradient (|g_xy| < 0.02): {degen} "
                  f"({100.0*degen/max(1,overlap_cases):.0f}% of overlaps)"]
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="E3.7 analytic vs autograd gradient")
    ap.add_argument("--method", default="pydam", choices=["pydam", "dam"],
                    help="guard implementation (dam needs the DAM venv)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--census", type=int, default=2000)
    ap.add_argument("--out", default="experiments/results/e37_gradient")
    args = ap.parse_args()
    if args.method != "pydam" and args.out == ap.get_default("out"):
        args.out += f"_{args.method}"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    census(args.census, out / "census.md")

    conditions = [
        Condition(label=f"{mode}_v{v}",
                  make_filter=lambda mode=mode, v=v: make_guard(args.method, grad_mode=mode, max_v=v),
                  sim_overrides={"vmax": v})
        for mode in ("analytic", "autograd")
        for v in (1.5, 2.0)
    ]
    run_sweep("E3.7", ["S2", "S3"], conditions, args.seeds, args.out,
              markdown_fields=MD_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
