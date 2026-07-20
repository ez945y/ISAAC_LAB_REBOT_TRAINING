#!/usr/bin/env python3
# ruff: noqa: E402
"""E5 v2 (RQ5): deployment-grade failure-sample reuse loop, Isaac-in-the-loop.

Redesign of the first E5.1/E5.2 pass after review (2026-07-19): that pass ran
entirely in kinesim, distilled the SCRIPTED nominal controller both times, and
used only 30 sim episodes per pool — so "intervention dropped 65%" measured BC
smoothing of a hand-coded teacher, not the value of runtime-collected failure
samples. This version closes the loop the way RQ5 states it:

    A collect   nominal+raw  S5 x seeds 0..(base+extra-1)    failure-rich base
    B train     v0  = BC student on the base pool alone
                ("learned to drive from unguarded demos, never saw a guard")
    C collect   v0+dam       S5 x seeds 500..  — the guard corrects the
                STUDENT'S OWN mistakes at deployment = the actual runtime
                failure samples RQ5 talks about (DAgger-style corrections)
    D train     v1  = BC on base + damfix
                v1c = BC on base + extra-raw  (same data volume, no guard
                corrections — the "just more data" control)
    E bench     S5 held-out seeds 1000.. (all conds) + S2 zero-shot probe:
                nominal+raw / nominal+dam / bc_v0(+dam) / bc_v1(+dam) /
                bc_v1c(+dam)

Every pool is spooled per-episode (compressed npz, resume-safe), then exported
as a LeRobot v3 dataset with the E5.1 sidecars (episode manifest + RSMF-style
boundary events); students are trained THROUGH the official LeRobot reader —
the round trip is part of the RQ5 claim. S5 is the training scene because it
is the only generator whose seeds change the episode (random starts+goals,
6 dogs); S2's seed jitter is +-0.15 m, so it serves as the zero-shot probe.

    # kinesim smoke (minutes):
    python experiments/run_e5_loop.py --backend kinesim --smoke
    # the real thing (Isaac machine, long-running; resume = rerun same cmd):
    source ~/IsaacLab/env_isaaclab/bin/activate
    export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
    python experiments/run_e5_loop.py --backend isaac
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter, deque
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from common.filters import make_filter
from common.kinesim import KineSim, SimConfig
from common.lerobot_export import (EVENT_REQUIRED_FIELDS, INTERVENE_EPS,  # noqa: F401
                                   AgentEpisode, Frame, RecordingFilter,
                                   boundary_events, collect_agent_episodes,
                                   ego_observation, lerobot_features)
from common.metrics import EpisodeRecorder, MetricsConfig
from common.scenarios import SCENARIOS
from common.sweep_io import Checkpointer

TRAIN_SCEN = "S5"        # only generator with real per-seed diversity
PROBE_SCEN = "S2"        # zero-shot probe (head-on; near-identical across seeds)
FIX_SEED_BASE = 500      # stage-C collection seeds (disjoint from base pool)
BENCH_SEED_BASE = 1000   # held-out evaluation seeds
MAX_DOGS = 6             # S5 squad size; S2 uses 2 of the slots

BENCH_FIELDS = ["makespan_s", "min_dogdog_m", "viol_steps_dog",
                "all_done_rate", "intervention_rate", "n_falls"]


# -- BC student ---------------------------------------------------------------

def build_net(obs_dim: int = 42, hidden: int = 256):
    import torch
    return torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 3),
    )


def load_policy(path: Path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(ck["obs_dim"], ck["hidden"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return (net, ck["mean"], ck["std"])


def bc_cmd(sim, ag, policy):
    """One inference of the distilled policy on the SAME ego view the dataset
    frames were built from; clamped to the sim's command limits."""
    import torch
    net, mean, std = policy
    obs = ego_observation(ag, sim._neighbors_of(ag.spec.aid))
    with torch.no_grad():
        v = net((torch.tensor([obs], dtype=torch.float32) - mean) / std)[0].tolist()
    c = sim.cfg
    return (max(-c.vmax, min(c.vmax, v[0])),
            max(-c.vmax, min(c.vmax, v[1])),
            max(-c.wmax, min(c.wmax, v[2])))


class BCKineSim(KineSim):
    """KineSim whose NOMINAL controller is the distilled student."""

    def __init__(self, specs, safety_filter, cfg, rng_seed, policy):
        super().__init__(specs, safety_filter, cfg, rng_seed=rng_seed)
        self._bc = policy

    def nominal(self, ag):
        super().nominal(ag)  # advances waypoint state exactly like the teacher
        return bc_cmd(self, ag, self._bc)


# -- per-episode spool (resume-safe raw capture) ------------------------------

def spool_write(path: Path, episodes: list[AgentEpisode]) -> None:
    import numpy as np
    arrays, agents = {}, []
    ref = episodes[0]
    for ep in episodes:
        fr = ep.frames
        a = ep.aid
        arrays[f"{a}__obs"] = np.asarray([f.obs for f in fr], np.float32)
        arrays[f"{a}__safe"] = np.asarray([f.safe for f in fr], np.float32)
        arrays[f"{a}__raw"] = np.asarray([f.raw for f in fr], np.float32)
        arrays[f"{a}__dec"] = np.asarray([f.decision for f in fr], np.int64)
        arrays[f"{a}__delta"] = np.asarray([f.delta for f in fr], np.float32)
        arrays[f"{a}__hs"] = np.asarray([f.hard_slack for f in fr], np.float32)
        arrays[f"{a}__mind"] = np.asarray([f.min_dog_capsule for f in fr], np.float32)
        arrays[f"{a}__minw"] = np.asarray([f.min_wall_capsule for f in fr], np.float32)
        agents.append({"aid": a, "done": ep.done, "done_step": ep.done_step,
                       "n": len(fr)})
    meta = {"scenario": ref.scenario, "seed": ref.seed, "method": ref.method,
            "dt": ref.dt, "agents": agents}
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, meta=np.array(json.dumps(meta)), **arrays)
    tmp.rename(path)


def spool_read(path: Path) -> list[AgentEpisode]:
    import numpy as np
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    out = []
    for a in meta["agents"]:
        aid = a["aid"]
        obs, safe, raw = z[f"{aid}__obs"], z[f"{aid}__safe"], z[f"{aid}__raw"]
        dec, delta, hs = z[f"{aid}__dec"], z[f"{aid}__delta"], z[f"{aid}__hs"]
        mind, minw = z[f"{aid}__mind"], z[f"{aid}__minw"]
        frames = [Frame(obs=obs[i].tolist(),
                        raw=tuple(float(v) for v in raw[i]),
                        safe=tuple(float(v) for v in safe[i]),
                        decision=int(dec[i]), delta=float(delta[i]),
                        hard_slack=float(hs[i]),
                        min_dog_capsule=float(mind[i]),
                        min_wall_capsule=float(minw[i]))
                  for i in range(a["n"])]
        out.append(AgentEpisode(scenario=meta["scenario"], seed=meta["seed"],
                                method=meta["method"], aid=aid, dt=meta["dt"],
                                frames=frames, done=a["done"],
                                done_step=a["done_step"]))
    return out


# -- LeRobot export (streamed from the spool) ---------------------------------

def _squeeze_scalar_buffer(ds) -> None:
    """lerobot 0.4.4 buffers shape-(1,) features as np.array([v]) per frame,
    which numpy>=2 refuses to scalar-convert inside datasets.Dataset.from_dict
    (works on older numpy — the Mac run predates this). Squeeze them to python
    scalars right before save_episode()."""
    buf = ds.episode_buffer
    for key, ft in ds.features.items():
        if ft.get("shape") == (1,) and isinstance(buf.get(key), list):
            buf[key] = [v.item() if hasattr(v, "item") else v for v in buf[key]]


def export_dataset(npz_paths: list[Path], root: Path, repo_id: str, fps: int,
                   robot_type: str) -> dict:
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, root=root,
                               robot_type=robot_type, use_videos=False,
                               features=lerobot_features())
    manifest, events = [], []
    idx = n_frames = 0
    for p in npz_paths:
        for ep in spool_read(p):
            if not ep.frames:
                continue
            task = f"{ep.scenario}: drive to goal, {ep.method}"
            for f in ep.frames:
                ds.add_frame({
                    "observation.state": np.asarray(f.obs, dtype=np.float32),
                    "observation.min_dog_capsule_m": np.array([f.min_dog_capsule], dtype=np.float32),
                    "observation.min_wall_capsule_m": np.array([f.min_wall_capsule], dtype=np.float32),
                    "action": np.asarray(f.safe, dtype=np.float32),
                    "action.raw": np.asarray(f.raw, dtype=np.float32),
                    "guard.decision": np.array([f.decision], dtype=np.int64),
                    "guard.delta": np.array([f.delta], dtype=np.float32),
                    "guard.hard_slack": np.array([f.hard_slack], dtype=np.float32),
                    "guard.intervened": np.array([int(f.delta > INTERVENE_EPS)], dtype=np.int64),
                    "task": task,
                })
            _squeeze_scalar_buffer(ds)
            ds.save_episode()
            manifest.append({"episode_index": idx, "scenario": ep.scenario,
                             "seed": ep.seed, "method": ep.method, "aid": ep.aid,
                             "n_frames": len(ep.frames), "done": ep.done,
                             "done_step": ep.done_step})
            events.extend(boundary_events(ep, idx))
            n_frames += len(ep.frames)
            idx += 1
    ds.finalize()
    meta = root / "meta"
    with open(meta / "episode_manifest.jsonl", "w") as fh:
        for row in manifest:
            fh.write(json.dumps(row) + "\n")
    with open(meta / "boundary_events.jsonl", "w") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")

    # load-back check through the official reader (part of the RQ5 claim)
    back = LeRobotDataset(repo_id, root=root)
    assert back.num_frames == n_frames, \
        f"load-back mismatch: {back.num_frames} != {n_frames}"

    complete = sum(all(ev.get(f) is not None for f in EVENT_REQUIRED_FIELDS)
                   for ev in events)
    types = Counter(ev["event_type"] for ev in events)
    per_scen: dict[str, set] = {}
    for ev in events:
        per_scen.setdefault(ev["scenario"], set()).add(ev["event_type"])
    return {
        "root": str(root), "episodes": idx, "frames": n_frames,
        "events": len(events),
        "field_completeness": complete / len(events) if events else 1.0,
        "window_completeness": (sum(ev["window_complete"] for ev in events)
                                / len(events)) if events else 1.0,
        "event_types": dict(types),
        "types_per_scenario": {k: sorted(v) for k, v in sorted(per_scen.items())},
    }


def export_if_needed(name: str, npz_paths: list[Path], out: Path, fps: int,
                     robot_type: str) -> dict:
    root = out / "datasets" / f"go2squad_{name}"
    marker = out / "datasets" / f"go2squad_{name}.quality.json"
    if marker.exists() and root.exists():
        q = json.loads(marker.read_text())
        print(f"[export] {name}: reusing existing dataset "
              f"({q['episodes']} eps / {q['frames']} frames)", flush=True)
        return q
    print(f"[export] {name}: {len(npz_paths)} spool episodes -> {root}", flush=True)
    q = export_dataset(npz_paths, root, f"local/go2squad_{name}", fps, robot_type)
    marker.write_text(json.dumps(q, indent=2))
    print(f"[export] {name}: {q['episodes']} eps / {q['frames']} frames / "
          f"{q['events']} events, load-back OK", flush=True)
    return q


# -- training -----------------------------------------------------------------

def _dataset_tensors(root: Path, repo_id: str):
    """(X, Y, seed_per_frame) loaded THROUGH the official LeRobot reader."""
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root)
    hf = ds.hf_dataset

    def col(name, width=None):
        try:  # arrow fast path (2M+ frames would crawl through per-row stacking)
            import pyarrow as pa
            arr = hf.data.column(name).combine_chunks()
            if pa.types.is_fixed_size_list(arr.type) or pa.types.is_list(arr.type):
                flat = arr.flatten().to_numpy(zero_copy_only=False)
                return np.asarray(flat, dtype=np.float32).reshape(len(arr), -1)
            return arr.to_numpy(zero_copy_only=False)
        except Exception:
            t = hf.with_format("torch")
            v = torch.stack(list(t[name]))
            return v.numpy()

    X = torch.from_numpy(np.ascontiguousarray(col("observation.state"))).float()
    Y = torch.from_numpy(np.ascontiguousarray(col("action"))).float()
    epi = np.asarray(col("episode_index")).reshape(-1).astype(np.int64)
    ep_seed = {}
    with open(root / "meta" / "episode_manifest.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            ep_seed[row["episode_index"]] = row["seed"]
    seeds = np.array([ep_seed[int(e)] for e in epi], dtype=np.int64)
    return X, Y, torch.from_numpy(seeds)


def train_policy(specs: list[tuple[Path, str]], out_path: Path, *, epochs: int,
                 batch: int = 8192, lr: float = 1e-3, val_mod: int = 10,
                 patience: int = 10, seed: int = 0) -> None:
    """BC on the given LeRobot datasets; episode-level val split (sim seed
    ending in ``val_mod - 1``), early stop on val MSE, best weights kept."""
    import torch
    torch.manual_seed(seed)
    Xs, Ys, Ss = [], [], []
    for root, rid in specs:
        X, Y, S = _dataset_tensors(root, rid)
        print(f"    loaded {rid}: {X.shape[0]} frames", flush=True)
        Xs.append(X), Ys.append(Y), Ss.append(S)
    X, Y, S = torch.cat(Xs), torch.cat(Ys), torch.cat(Ss)
    val = (S % val_mod) == (val_mod - 1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, Ytr = X[~val].to(dev), Y[~val].to(dev)
    Xva, Yva = X[val].to(dev), Y[val].to(dev)
    mean, std = Xtr.mean(0), Xtr.std(0).clamp_min(1e-3)
    net = build_net(X.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = Xtr.shape[0]
    best, best_state, best_ep, log = math.inf, None, -1, []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred = net((Xtr[idx] - mean) / std)
            loss = torch.nn.functional.mse_loss(pred, Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            va = torch.nn.functional.mse_loss(
                net((Xva - mean) / std), Yva).item() if Xva.shape[0] else tot / n
        log.append({"epoch": ep + 1, "train_mse": tot / n, "val_mse": va})
        if va < best - 1e-7:
            best, best_ep = va, ep
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"    epoch {ep + 1:3d}  train {tot / n:.5f}  val {va:.5f}"
                  f"{'  *' if best_ep == ep else ''}", flush=True)
        if ep - best_ep >= patience:
            print(f"    early stop at epoch {ep + 1} (best val {best:.5f} "
                  f"@ epoch {best_ep + 1})", flush=True)
            break
    torch.save({"state_dict": best_state, "mean": mean.cpu(), "std": std.cpu(),
                "obs_dim": X.shape[1], "hidden": 256,
                "train_frames": int(n), "val_frames": int(Xva.shape[0]),
                "best_val_mse": best, "log": log}, out_path)
    print(f"    saved {out_path} (train {n} / val {Xva.shape[0]} frames, "
          f"best val {best:.5f})", flush=True)


def train_if_needed(name: str, specs: list[tuple[Path, str]], out: Path,
                    epochs: int):
    path = out / "policies" / f"{name}.pt"
    if path.exists():
        print(f"[train] {name}: reusing {path}", flush=True)
    else:
        print(f"[train] {name}: BC on {[rid for _, rid in specs]}", flush=True)
        train_policy(specs, path, epochs=epochs)
    return load_policy(path)


# -- condition parsing --------------------------------------------------------

def parse_cond(cond: str) -> tuple[str | None, str]:
    """'nominal+raw' -> (None, 'raw'); 'bc_v0+dam' -> ('v0', 'dam');
    'bc_v1' -> ('v1', 'raw')  (unguarded = passthrough filter)."""
    head, _, fname = cond.partition("+")
    pol = head[3:] if head.startswith("bc_") else None
    return pol, (fname or "raw")


# -- job execution ------------------------------------------------------------

class Ctx:
    """Backend context: kinesim is stateless; isaac carries the arena pool."""

    def __init__(self, backend: str, args):
        self.backend = backend
        self.pool = None
        self.K = 1
        if backend == "isaac":
            sys.path.insert(0, str(_HERE / "isaac"))
            from sim_backend import make_pool
            self.pool = make_pool("go2")
            self.K = args.arenas or max(1, min(4, args.slot_cap // MAX_DOGS))
            self.pool.configure(self.K, MAX_DOGS, [])   # S5/S2 have no walls
            print(f"[isaac] pool ready: {self.K} arenas x {MAX_DOGS} dogs",
                  flush=True)


def run_jobs(ctx: Ctx, jobs: list[tuple], ck: Checkpointer, policies: dict,
             spool_dir: Path | None, max_time: float, tag: str) -> None:
    """jobs = [(scenario, seed, cond, record)]; already-checkpointed episodes
    must be filtered out by the caller."""
    sim_cfg = SimConfig(max_time=max_time)
    met_cfg = MetricsConfig()
    total, k = len(jobs), 0

    def finish(sim, filt, rec, scen, seed, cond, record, extras):
        nonlocal k
        row = rec.finish(sim, scen, seed, method=cond)
        row["condition"] = cond
        row.update(extras)
        if record and not extras.get("diverged"):
            eps = collect_agent_episodes(sim, filt, scen, seed, cond)
            spool_write(spool_dir / f"{scen}_{seed}.npz", eps)
        if hasattr(filt, "close"):
            filt.close()
        ck.add(row)
        k += 1
        print(f"[{tag} {k}/{total}] {scen} {cond} seed={seed}  "
              f"done={row['all_done']} makespan={row['makespan_s']:.1f}s "
              f"minDD={row['min_dogdog_m']:.2f} viol={row['viol_steps_dog']} "
              f"falls={extras.get('n_falls', 0)}", flush=True)

    if ctx.backend == "kinesim":
        for scen, seed, cond, record in jobs:
            specs = SCENARIOS[scen](seed)
            pol, fname = parse_cond(cond)
            inner = make_filter(fname)
            filt = RecordingFilter(inner) if record else inner
            sim = (BCKineSim(specs, filt, sim_cfg, seed, policies[pol]) if pol
                   else KineSim(specs, filt, sim_cfg, rng_seed=seed))
            if record:
                filt.sim = sim
            rec = EpisodeRecorder(met_cfg)
            sim.run(recorder=rec)
            finish(sim, filt, rec, scen, seed, cond, record,
                   {"n_falls": 0, "diverged": False})
        return

    # -- isaac: multi-arena interleaved episodes (mirrors isaac/runner.py) ----
    from sim_backend import PHYSICS_DT, SETTLE_S, IsaacArenaSim

    class BCIsaacSim(IsaacArenaSim):
        def __init__(self, specs, filt, cfg, rng_seed, pool, arena, policy):
            super().__init__(specs, filt, cfg, rng_seed=rng_seed,
                             pool=pool, arena=arena)
            self._bc = policy

        def nominal(self, ag):
            super().nominal(ag)
            return bc_cmd(self, ag, self._bc)

    settle_ticks = int(round(SETTLE_S / sim_cfg.dt))
    substeps = int(round(sim_cfg.dt / PHYSICS_DT))
    queue = deque(jobs)
    active: dict[int, dict] = {}
    while queue or active:
        for a in range(ctx.K):
            if a in active or not queue:
                continue
            scen, seed, cond, record = queue.popleft()
            specs = SCENARIOS[scen](seed)
            pol, fname = parse_cond(cond)
            inner = make_filter(fname)
            filt = RecordingFilter(inner) if record else inner
            if pol:
                sim = BCIsaacSim(specs, filt, sim_cfg, seed, ctx.pool, a,
                                 policies[pol])
            else:
                sim = IsaacArenaSim(specs, filt, sim_cfg, rng_seed=seed,
                                    pool=ctx.pool, arena=a)
            if record:
                filt.sim = sim
            active[a] = {"sim": sim, "filt": filt, "rec": EpisodeRecorder(met_cfg),
                         "settle": settle_ticks, "meta": (scen, seed, cond, record),
                         "t0": time.perf_counter()}
        cmds = ctx.pool.zero_cmds()
        for st in active.values():
            if st["settle"] == 0:
                st["sim"].control_tick(cmds)
        for _ in range(substeps):
            ctx.pool.step_physics(cmds)
        for a in list(active):
            st = active[a]
            if st["settle"] > 0:
                st["settle"] -= 1
                if st["settle"] == 0:
                    st["sim"].begin()
                continue
            if not st["sim"].post_step(st["rec"]):
                continue
            scen, seed, cond, record = st["meta"]
            sim = st["sim"]
            finish(sim, st["filt"], st["rec"], scen, seed, cond, record,
                   {"n_falls": len(sim.fallen), "diverged": sim.diverged,
                    "arena": a, "wall_s": round(time.perf_counter() - st["t0"], 1)})
            del active[a]


def collect_stage(ctx: Ctx, out: Path, label: str, jobs: list[tuple],
                  policies: dict, max_time: float) -> Path:
    """Run a collection sweep (record=True) into spool/<label>; resume-safe."""
    ck = Checkpointer(out / f"collect_{label}")
    spool = out / "spool" / label
    spool.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not ck.skip(j[0], j[2], j[1])]
    print(f"[collect:{label}] {len(todo)}/{len(jobs)} episodes to run", flush=True)
    if todo:
        run_jobs(ctx, todo, ck, policies, spool, max_time, f"collect:{label}")
    ck.finalize(BENCH_FIELDS)
    ck.close()
    return spool


# -- main ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="E5 v2: failure-sample reuse loop")
    ap.add_argument("--backend", choices=["kinesim", "isaac"], default="kinesim")
    ap.add_argument("--out", default=None,
                    help="default: experiments/results/e5_loop (kinesim) or "
                         "experiments/isaac/results/e5_loop (isaac)")
    ap.add_argument("--base-seeds", type=int, default=300)
    ap.add_argument("--extra-seeds", type=int, default=100)
    ap.add_argument("--fix-seeds", type=int, default=100)
    ap.add_argument("--bench-seeds", type=int, default=20)
    ap.add_argument("--probe-seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--max-time", type=float, default=60.0)
    ap.add_argument("--arenas", type=int, default=None)
    ap.add_argument("--slot-cap", type=int, default=24)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end pass (kinesim, minutes)")
    args = ap.parse_args()

    if args.smoke:
        args.base_seeds, args.extra_seeds = 16, 6
        args.fix_seeds, args.bench_seeds, args.probe_seeds = 6, 4, 3
        args.epochs, args.max_time = 12, 40.0

    out = Path(args.out) if args.out else (
        _HERE / ("isaac/results/e5_loop" if args.backend == "isaac"
                 else "results/e5_loop"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "datasets").mkdir(exist_ok=True)
    (out / "policies").mkdir(exist_ok=True)
    fps = int(round(1.0 / SimConfig().dt))
    robot_type = f"go2_{args.backend}"
    t0 = time.perf_counter()

    if args.backend == "isaac":
        # boot headless Isaac BEFORE Ctx imports sim_backend
        from isaacsim import SimulationApp
        global simulation_app
        simulation_app = SimulationApp({"headless": True})
        import os

        import omni.kit.app

        def _enable(ext_id):
            omni.kit.app.get_app().get_extension_manager() \
                .set_extension_enabled_immediate(ext_id, True)

        import isaacsim
        from omni.ext import ExtensionPathType
        dep = os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated")
        omni.kit.app.get_app().get_extension_manager() \
            .add_path(dep, ExtensionPathType.COLLECTION)
        _enable("isaacsim.core.api")
        _enable("isaacsim.robot.policy.examples")
        simulation_app.update()

    ctx = Ctx(args.backend, args)
    policies: dict[str, tuple] = {}

    # A: base + extra-raw collection (one sweep, split at export time)
    n_raw = args.base_seeds + args.extra_seeds
    jobs = [(TRAIN_SCEN, s, "nominal+raw", True) for s in range(n_raw)]
    spool_raw = collect_stage(ctx, out, "raw", jobs, policies, args.max_time)

    def existing(spool: Path, scen: str, seeds) -> list[Path]:
        return [p for p in (spool / f"{scen}_{s}.npz" for s in seeds)
                if p.exists()]

    q_base = export_if_needed(
        "base", existing(spool_raw, TRAIN_SCEN, range(args.base_seeds)),
        out, fps, robot_type)
    q_extra = export_if_needed(
        "extra", existing(spool_raw, TRAIN_SCEN,
                          range(args.base_seeds, n_raw)),
        out, fps, robot_type)

    # B: v0 = BC on the base pool
    ds_base = (out / "datasets" / "go2squad_base", "local/go2squad_base")
    ds_extra = (out / "datasets" / "go2squad_extra", "local/go2squad_extra")
    policies["v0"] = train_if_needed("v0", [ds_base], out, args.epochs)

    # C: guard corrects the deployed student -> damfix pool
    jobs = [(TRAIN_SCEN, FIX_SEED_BASE + i, "bc_v0+dam", True)
            for i in range(args.fix_seeds)]
    spool_fix = collect_stage(ctx, out, "damfix", jobs, policies, args.max_time)
    q_fix = export_if_needed(
        "damfix", existing(spool_fix, TRAIN_SCEN,
                           range(FIX_SEED_BASE, FIX_SEED_BASE + args.fix_seeds)),
        out, fps, robot_type)

    # D: v1 (base + guard corrections) vs v1c (base + more raw, the control)
    ds_fix = (out / "datasets" / "go2squad_damfix", "local/go2squad_damfix")
    policies["v1"] = train_if_needed("v1", [ds_base, ds_fix], out, args.epochs)
    policies["v1c"] = train_if_needed("v1c", [ds_base, ds_extra], out, args.epochs)

    # E: held-out benchmark + zero-shot probe
    conds = ["nominal+raw", "nominal+dam", "bc_v0", "bc_v0+dam",
             "bc_v1", "bc_v1+dam", "bc_v1c", "bc_v1c+dam"]
    jobs = [(TRAIN_SCEN, BENCH_SEED_BASE + i, c, False)
            for c in conds for i in range(args.bench_seeds)]
    jobs += [(PROBE_SCEN, BENCH_SEED_BASE + i, c, False)
             for c in conds[:6] for i in range(args.probe_seeds)]
    ck = Checkpointer(out / "bench")
    todo = [j for j in jobs if not ck.skip(j[0], j[2], j[1])]
    print(f"[bench] {len(todo)}/{len(jobs)} episodes to run", flush=True)
    if todo:
        run_jobs(ctx, todo, ck, policies, None, args.max_time, "bench")
    md = ck.finalize(BENCH_FIELDS)
    ck.close()

    # report
    lines = ["# E5 v2 — failure-sample reuse loop "
             f"({args.backend}, {time.strftime('%Y-%m-%d')})", "",
             f"Train scene {TRAIN_SCEN} (base {args.base_seeds} + extra "
             f"{args.extra_seeds} raw seeds; damfix {args.fix_seeds} seeds from "
             f"{FIX_SEED_BASE}); bench {TRAIN_SCEN} x {args.bench_seeds} + "
             f"{PROBE_SCEN} x {args.probe_seeds} held-out seeds from "
             f"{BENCH_SEED_BASE}.", "",
             "## Dataset quality (E5.1-style)", "",
             "| pool | episodes | frames | events | field-complete | window-complete | event types |",
             "|---|---|---|---|---|---|---|"]
    for name, q in (("base", q_base), ("extra", q_extra), ("damfix", q_fix)):
        types = ", ".join(f"{k}:{v}" for k, v in sorted(q["event_types"].items()))
        lines.append(f"| {name} | {q['episodes']} | {q['frames']} | {q['events']} "
                     f"| {q['field_completeness']:.3f} "
                     f"| {q['window_completeness']:.3f} | {types or '-'} |")
    lines += ["", "## Students", ""]
    for name in ("v0", "v1", "v1c"):
        import torch
        ck_ = torch.load(out / "policies" / f"{name}.pt", map_location="cpu",
                         weights_only=False)
        lines.append(f"- **{name}**: train {ck_['train_frames']} / val "
                     f"{ck_['val_frames']} frames, best val MSE "
                     f"{ck_['best_val_mse']:.5f} ({len(ck_['log'])} epochs)")
    lines += ["", "## Benchmark (held-out seeds)", "", md, ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"\nwrote {out}/REPORT.md  ({(time.perf_counter() - t0) / 60:.1f} min)",
          flush=True)
    return 0


if __name__ == "__main__":
    simulation_app = None
    try:
        code = main()
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        if simulation_app is not None:
            simulation_app.close()
    raise SystemExit(code)
