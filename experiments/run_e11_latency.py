#!/usr/bin/env python3
"""E1.1 (RQ1, real-time): full-distribution latency characterisation of the
REAL dam guard under replayed real workloads.

Defence-grade design decisions:
  * Inputs are NOT synthetic: filter inputs (command, pose, neighbour set,
    priority) are captured from actual scenario episodes (S5 at N = 2..16
    dogs; S3 with wall constraints), then replayed against a fresh guard.
    The latency distribution is therefore over the same constraint-count
    distribution the deployed filter sees.
  * The measured call is ``DamFilter.filter`` — the FULL wrapper the robot
    pays (torch conversion + Guardrail + OSQP), not a cherry-picked inner QP.
  * Tail-complete reporting: p50/p90/p99/p99.9/max + deadline-overrun counts
    (2 ms soft, 20 ms = one 50 Hz period hard), cold-start first call
    separately, THREE independent replay runs for run-to-run stability.
  * Machine + version fingerprint recorded next to the numbers.
  * Cross-check column: the p50/p99 measured INSIDE the Isaac control loop
    (experiments/isaac/results/e45_scale) for the same N.

    source ~/IsaacLab/env_isaaclab/bin/activate
    python experiments/run_e11_latency.py            # ~2 min capture + bench
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.filters import make_filter  # noqa: E402
from common.kinesim import KineSim, SimConfig  # noqa: E402
from common.scenarios import SCENARIOS  # noqa: E402


class CaptureFilter:
    """Wraps the real filter; records every call's inputs for later replay."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[tuple] = []

    def filter(self, aid, cmd, pose, neighbors, self_priority=1.0):
        self.calls.append((aid, cmd, pose, list(neighbors), self_priority))
        return self.inner.filter(aid, cmd, pose, neighbors, self_priority=self_priority)

    def close(self):
        if hasattr(self.inner, "close"):
            self.inner.close()


def capture_workload(config: str, seeds: int) -> list[tuple]:
    """Run real episodes with the real guard, capturing the filter inputs."""
    calls: list[tuple] = []
    for seed in range(seeds):
        if config.startswith("S5xN"):
            n = int(config.split("N")[1])
            specs = SCENARIOS["S5"](seed, n=n)
        elif config == "S3walls":
            specs = SCENARIOS["S3"](seed)
        else:
            raise ValueError(config)
        cap = CaptureFilter(make_filter("dam"))
        KineSim(specs, cap, SimConfig(max_time=30.0), rng_seed=seed).run()
        calls.extend(cap.calls)
        cap.close()
    return calls


def bench(calls: list[tuple], runs: int) -> dict:
    """Replay ``calls`` against a fresh guard ``runs`` times; full stats."""
    out: dict = {"n_calls": len(calls), "runs": []}
    for run_idx in range(runs):
        filt = make_filter("dam")
        t0 = time.perf_counter()
        filt.filter(*calls[0][:4], self_priority=calls[0][4])
        cold_ms = (time.perf_counter() - t0) * 1e3     # first-ever call: imports/JIT/alloc
        for c in calls[:200]:                          # warmup
            filt.filter(*c[:4], self_priority=c[4])
        lat = []
        for c in calls:
            t0 = time.perf_counter()
            filt.filter(*c[:4], self_priority=c[4])
            lat.append((time.perf_counter() - t0) * 1e3)
        filt.close()
        lat.sort()

        def pct(p):
            return lat[min(len(lat) - 1, int(p * len(lat)))]

        out["runs"].append({
            "cold_first_call_ms": round(cold_ms, 3),
            "mean_ms": round(statistics.mean(lat), 4),
            "p50_ms": round(pct(0.50), 4), "p90_ms": round(pct(0.90), 4),
            "p99_ms": round(pct(0.99), 4), "p999_ms": round(pct(0.999), 4),
            "max_ms": round(lat[-1], 4),
            "over_2ms": sum(1 for v in lat if v > 2.0),
            "over_20ms": sum(1 for v in lat if v > 20.0),
        })
    return out


def fingerprint() -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return "?"
    import numpy, osqp, scipy, torch  # noqa: F401  (versions only)
    import dam
    return {
        "machine": platform.machine(), "python": platform.python_version(),
        "cpu_model": sh("lscpu | grep -m1 'Model name' | cut -d: -f2").strip(),
        "cpus": sh("nproc"),
        "governor": sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null"),
        "dam": dam.__version__, "torch": torch.__version__,
        "osqp": osqp.__version__, "scipy": scipy.__version__,
        "numpy": numpy.__version__,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def isaac_crosscheck(isaac_dir: Path) -> dict:
    """p50/p99 measured inside the Isaac control loop (e45), for the table."""
    out = {}
    f = isaac_dir / "e45_scale/episodes.jsonl"
    if not f.exists():
        return out
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        out[cond] = (statistics.mean(r["filter_p50_ms"] for r in rs),
                     statistics.mean(r["filter_p99_ms"] for r in rs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="E1.1 guard latency benchmark")
    ap.add_argument("--seeds", type=int, default=5, help="episodes per config to capture")
    ap.add_argument("--runs", type=int, default=3, help="independent replay runs")
    ap.add_argument("--out", default="experiments/results/e11_latency")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [f"S5xN{n}" for n in (2, 4, 8, 12, 16)] + ["S3walls"]

    results = {}
    for cfg in configs:
        print(f"[e11] capturing {cfg} ...", flush=True)
        calls = capture_workload(cfg, args.seeds)
        print(f"[e11] {cfg}: {len(calls)} calls captured; benchmarking "
              f"x{args.runs} runs ...", flush=True)
        results[cfg] = bench(calls, args.runs)

    meta = fingerprint()
    xc = isaac_crosscheck(Path(__file__).resolve().parent / "isaac/results")
    (out_dir / "raw.json").write_text(json.dumps(
        {"meta": meta, "results": results, "isaac_in_loop": xc}, indent=1))

    lines = ["# E1.1 — dam guard latency (full distribution, replayed real workloads)", "",
             "Machine: " + ", ".join(f"{k}={v}" for k, v in meta.items()
                                     if k in ("cpu_model", "cpus", "machine", "governor")),
             "Versions: " + ", ".join(f"{k} {meta[k]}" for k in
                                      ("dam", "osqp", "torch", "scipy", "numpy")), "",
             "Budget: one 50 Hz control period = **20 ms** (hard); 2 ms soft threshold = 10 % of budget.",
             "Worst run of "
             f"{args.runs} independent replays shown; per-run details in raw.json.", "",
             "| workload | calls | p50 | p90 | p99 | p99.9 | max | >2 ms | >20 ms | cold 1st | isaac in-loop p50/p99 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    n_to_cond = {"S5xN2": "n2", "S5xN4": "n4", "S5xN8": "n8",
                 "S5xN12": "n12", "S5xN16": "n16"}
    for cfg, res in results.items():
        worst = max(res["runs"], key=lambda r: r["p99_ms"])
        cold = max(r["cold_first_call_ms"] for r in res["runs"])
        x = xc.get(n_to_cond.get(cfg, ""), None)
        xs = f"{x[0]:.2f} / {x[1]:.2f}" if x else "—"
        lines.append(
            f"| {cfg} | {res['n_calls']} | {worst['p50_ms']:.3f} | {worst['p90_ms']:.3f} | "
            f"{worst['p99_ms']:.3f} | {worst['p999_ms']:.3f} | {worst['max_ms']:.3f} | "
            f"{worst['over_2ms']} | {worst['over_20ms']} | {cold:.1f} | {xs} |")
    md = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(md)
    print("\n" + md)
    print(f"wrote {out_dir}/summary.md, raw.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
