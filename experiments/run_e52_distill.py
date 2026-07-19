#!/usr/bin/env python3
"""E5.2 (RQ5): reuse-value closed loop — BC distillation from the E5.1 pools.

Trains one behaviour-cloning MLP per pool (loaded THROUGH the LeRobot reader,
so the round trip itself is part of the claim), then evaluates each student
UNGUARDED in kinesim on held-out seeds:

    nominal+raw   what the plain controller does (failure baseline)
    nominal+dam   the teacher (upper reference)
    bc_dam+raw    student of guard-corrected actions — did safety transfer?
    bc_raw+raw    student of the raw pool — controls for "BC just smooths"

If bc_dam beats bc_raw / nominal+raw on violations, the collected data
demonstrably carries the guard's safety behaviour, i.e. it has reuse value
beyond logging (RQ5's 再利用價值 claim).

    "$VENV" experiments/run_e52_distill.py --data experiments/results/e51_lerobot
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.filters import make_filter  # noqa: E402
from common.kinesim import KineSim, SimConfig  # noqa: E402
from common.lerobot_export import ego_observation  # noqa: E402
from common.metrics import (EpisodeRecorder, MetricsConfig, aggregate,  # noqa: E402
                            to_markdown, write_csv)
from common.scenarios import SCENARIOS  # noqa: E402


def load_pool(root: Path, repo_id: str):
    """(obs, action) float32 tensors via the official LeRobot reader."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=root)
    hf = ds.hf_dataset.with_format("torch")
    X = torch.stack(list(hf["observation.state"])).float()
    Y = torch.stack(list(hf["action"])).float()
    return X, Y


def train_bc(X, Y, epochs: int, seed: int = 0):
    import torch
    torch.manual_seed(seed)
    mean, std = X.mean(0), X.std(0).clamp_min(1e-3)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 3),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            pred = net((X[idx] - mean) / std)
            loss = torch.nn.functional.mse_loss(pred, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"    epoch {ep + 1:3d}  mse {tot / n:.5f}")
    net.eval()
    return net, mean, std


class BCSim(KineSim):
    """KineSim whose NOMINAL controller is the distilled policy: same ego view
    the dataset frames were built from, unguarded execution."""

    def __init__(self, specs, safety_filter, cfg, rng_seed, policy):
        super().__init__(specs, safety_filter, cfg, rng_seed=rng_seed)
        self._policy = policy

    def nominal(self, ag):
        super().nominal(ag)  # advances waypoint state exactly like the teacher
        import torch
        obs = ego_observation(ag, self._neighbors_of(ag.spec.aid))
        net, mean, std = self._policy
        with torch.no_grad():
            v = net((torch.tensor([obs]) - mean) / std)[0].tolist()
        c = self.cfg
        return (max(-c.vmax, min(c.vmax, v[0])),
                max(-c.vmax, min(c.vmax, v[1])),
                max(-c.wmax, min(c.wmax, v[2])))


def run_episode(scenario, seed, cond, sim_cfg, met_cfg, policies) -> dict:
    specs = SCENARIOS[scenario](seed)
    if cond.startswith("bc_"):
        # "bc_<pool>" = student unguarded; "bc_<pool>+dam" = student behind the
        # guard (does retraining lower the guard's intervention rate?)
        parts = cond.split("+")
        filt = make_filter(parts[1] if len(parts) > 1 else "raw")
        sim = BCSim(specs, filt, sim_cfg, rng_seed=seed, policy=policies[parts[0][3:]])
    else:  # "nominal+<method>"
        filt = make_filter(cond.split("+")[1])
        sim = KineSim(specs, filt, sim_cfg, rng_seed=seed)
    rec = EpisodeRecorder(met_cfg)
    sim.run(recorder=rec)
    row = rec.finish(sim, scenario, seed, cond)
    if hasattr(filt, "close"):
        filt.close()
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="E5.2: BC distillation reuse loop (RQ5)")
    ap.add_argument("--data", default="experiments/results/e51_lerobot")
    ap.add_argument("--pools", default="dam,raw")
    ap.add_argument("--scenarios", default="S2,S3,S5")
    ap.add_argument("--eval-seeds", type=int, default=10)
    ap.add_argument("--eval-seed-base", type=int, default=100,
                    help="disjoint from the E5.1 collection seeds")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-time", type=float, default=60.0)
    ap.add_argument("--out", default="experiments/results/e52_distill")
    args = ap.parse_args()

    data = Path(args.data)
    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = {}
    for pool in pools:
        root = data / f"go2squad_{pool}"
        X, Y = load_pool(root, f"local/go2squad_{pool}")
        print(f"[train] bc_{pool}: {X.shape[0]} frames from {root}")
        policies[pool] = train_bc(X, Y, epochs=args.epochs)

    conds = (["nominal+raw", "nominal+dam"]
             + [f"bc_{p}" for p in pools]
             + [f"bc_{p}+dam" for p in pools])
    sim_cfg = SimConfig(max_time=args.max_time)
    met_cfg = MetricsConfig()
    rows = []
    for cond in conds:
        for scen in scenarios:
            for i in range(args.eval_seeds):
                seed = args.eval_seed_base + i
                row = run_episode(scen, seed, cond, sim_cfg, met_cfg, policies)
                rows.append(row)
                print(f"[{cond}] {scen} seed={seed} done={row['all_done']} "
                      f"minDD={row['min_dogdog_m']:.2f} viol={row['viol_steps_dog']} "
                      f"makespan={row['makespan_s']:.1f}")

    write_csv(out_dir / "episodes.csv", rows)
    agg = aggregate(rows)
    write_csv(out_dir / "aggregate.csv", agg)
    md = ["# E5.2 — BC distillation from E5.1 LeRobot pools (RQ5 reuse value)",
          "",
          f"Held-out seeds {args.eval_seed_base}..{args.eval_seed_base + args.eval_seeds - 1}. "
          f"bc_<pool> = student unguarded (does safety transfer?); bc_<pool>+dam = student "
          f"behind the guard (does retraining cut the intervention rate vs nominal+dam?).",
          "",
          to_markdown(agg)]
    (out_dir / "summary.md").write_text("\n".join(md))
    print(f"\nwrote {out_dir}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
