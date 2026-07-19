# Thesis Experiments — Progress & Handoff

Branch: `exp/thesis-experiments` (forked from `feat/go2-squad-dispatch` after its PR).
Maps to the RQ design (RQ1–RQ4; RQ5 reinstated 2026-07-19 as E5.x — failure-sample
collection + reuse, see below). One experiment script at a time:
finish + fake-data-validate the current one before starting the next.

## Isaac machine environment (rst_spark, DGX Spark / aarch64) — read this first here

Everything below in this section was verified 2026-07-18 on the Isaac machine.

```bash
# ALL python runs (kinesim AND isaac) use the Isaac Lab venv:
source ~/IsaacLab/env_isaaclab/bin/activate
# anything that imports isaacsim ALSO needs (aarch64 quirk, same as demo10/11 .sh):
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
```

- **dam 0.7.0 is live here**: `~/DAM` was upgraded 0.6.0 → v0.7.0 on 2026-07-18
  (`git pull --ff-only` to 2385331; pre-existing local edits to assets/presets.yaml
  + examples/stackfiles/jetbot_lane_safety.yaml are in `git stash`
  ("pre-0.7-local-mods")). The venv imports it editable-style; `Guardrail` works.
  Sanity: kinesim S2+dam reproduces the thesis numbers exactly
  (makespan 5.96 s, minDD 0.784, 0 viol).
- venv also has: torch 2.10 (cu130), osqp 1.0.5, scipy, yaml. **No pyrvo2** →
  ORCA (B2) cannot run here; Isaac-side E2 skips orca.
- **Isaac-in-the-loop rerun of E2/E3.x/E4.x lives in `experiments/isaac/`**
  (headless SimulationApp + Go2FlatTerrainPolicy + real dam Guardrail; the old
  kinesim scripts/results are untouched):

```bash
# Isaac rerun (one long-lived headless app; ≤4 parallel arenas, batched policy;
# per-episode checkpoint, resumable):
python experiments/isaac/run_isaac_suite.py --exp all            # or --exp e2,e32,...
# kinesim dam-reference, same manifest/checkpoint format (done 2026-07-18 via the
# original run_e*.py scripts → experiments/results/<exp>_dam + e2_full; for future
# regens use the registry twin):
python experiments/run_kinesim_suite.py --exp all
# compare the two pools (no Isaac needed):
python experiments/isaac/compare.py                              # → isaac/results/COMPARISON.md
```

  **Status 2026-07-18 (evening)**: kinesim reference pool COMPLETE on this
  machine (`experiments/results/`, thesis E2 table reproduced EXACTLY: dam S1
  0.527 / S2 0.786 / S3 0.456 / S4 0.612 / S5 0.914). Isaac suite RUNNING
  (order e32,e31,e34,e35,e2,e33,e41,e42,e36,e37,e44,e45; resume-safe; log
  in the session scratchpad, results under `experiments/isaac/results/`).
  Interim per-experiment verdicts (see `isaac/results/COMPARISON.md`,
  regenerate any time with `python experiments/isaac/compare.py`):
  - **e32 swirl** ✅ story holds (0.6 → minDD 0.70/0 viol; off → 0.55 w/ viols;
    kinesim's perfect-symmetry 0.088 m standoff doesn't materialise — real
    gait asymmetry breaks the tie). makespan +18–20 % (under-tracking).
  - **e31 priority** ✅ yielding + floor ≈; residual 10 % non-completion on
    both sides (Isaac labels it deadlock, kinesim plain timeout).
  - **e34 capsule vs disc** ✅ headline claim reproduces exactly (capsule
    0.636 vs disc_in 0.457 vs disc_circ 100 % dlk + floor collapse); 0 wall
    violations, 0 falls beside REAL walls.
  - **e35 velocity-aware** ✅ dose-response ordering identical; Isaac floors
    uniformly 0.06–0.17 m lower, gap grows with speed (tracking-error margin
    erosion — the embodiment effect the experiment predicts). No falls at
    vmax 2.0.
  - Systematic embodiment deltas so far (consistent, explainable):
    makespan +18–20 %, floors ~0.05–0.15 m lower, small viol-step counts
    where kinesim had zero.
  Backend probe findings baked in: (1) go2 asset root-prim z reads ~0.22
  standing / ~0.10 mid-gait → fall detection is tilt-only; (2) policy
  under-tracks ~20 % (cmd 1.0 → 0.83 m/s, 1.5 → 1.2). Speed: batched policy
  runtime (one (N,48) inference for all dogs, 8.6× on the policy path) + ≤4
  parallel arenas ⇒ ~2.2× suite throughput; remaining bottleneck is PhysX
  itself (~7 ms/step @ 8 dogs).

  **Architecture (refactored 2026-07-18, second pass)** — one source of truth,
  two backends, shared plumbing (design details live ONLY in
  `experiments/isaac/README.md`; this is the map):

  | piece | file | role |
  |---|---|---|
  | sweep manifest | `common/registry.py` | ALL E2/E3.x/E4.x conditions, defined once (`build_experiments()` + `KINESIM_DIRS`) |
  | checkpoint IO | `common/sweep_io.py` | per-episode jsonl append + resume + csv/agg/summary finalize (both backends) |
  | Isaac backend | `isaac/sim_backend.py` | `Go2Pool` (K arenas × S dogs, one world) + `IsaacArenaSim(KineSim)` tick API |
  | Isaac runner | `isaac/runner.py` | multi-arena scheduler (≤4 episodes in parallel, barrier on wall change) |
  | entry points | `isaac/run_isaac_suite.py`, `run_kinesim_suite.py` | same manifest, Isaac vs kinesim execution |
  | comparison | `isaac/compare.py` | seed-matched side-by-side → `isaac/results/COMPARISON.md` |

  The 12 historical `run_e*.py` stay as the pydam/raw/orca + E3.7-census entry
  points — do NOT add new conditions there, add them to the registry.

## Local environment (IMPORTANT)

Use the DAM dev venv for anything needing torch/osqp/dam — **on this Mac**:

```bash
VENV="/Users/chenyizhong/Documents/Claude/Projects/Security Guard.nosync/.venv/bin/python3"
"$VENV" experiments/run_e2_baselines.py --methods raw,stop,pydam,dam ...
```

Every ablation/robustness sweep (e31–e45) also takes `--method dam` to run
on the REAL guard instead of pydam — knob mapping, new callback switches
(`wall_min_dist` / `velocity_aware` / `grad_mode`) and the hard-slack
telemetry interface are documented in **`experiments/DAM_ABLATION.md`**
(added 2026-07-12; equivalence gate PASS + e35 dam smoke = thesis numbers).

- venv has: numpy, scipy, torch 2.10, dam **0.7.0** (editable install of the
  local DAM repo at `.../Security Guard.nosync`), osqp 1.1.3 (installed via
  `uv pip install osqp --python "$VENV"` on 2026-07-11), and **pyrvo2**
  (official sybrenstuvel/Python-RVO2 bindings, installed 2026-07-12 for the
  ORCA baseline). Rebuild recipe if ever needed: clone the repo, then
  `uv pip install cython setuptools wheel --python "$VENV"` and
  `CMAKE_POLICY_VERSION_MINIMUM=3.5 uv pip install --no-build-isolation . --python "$VENV"`
  (two gotchas: setup.py imports Cython at build time → no-build-isolation;
  CMake 4 rejects the repo's old cmake_minimum_required → policy env var).
- Plain `python3` (no venv) runs raw/stop/pydam experiments (stdlib+numpy/scipy).

## Ground rules

- **Fake-data first**: every script must run and produce sane numbers on the
  lightweight kinematic sim (`common/kinesim.py`) with `raw`/`stop` filters on
  any machine (stdlib + nothing else). DAM/ORCA run locally in the venv;
  only Isaac-in-the-loop validation (E1.x) needs the Isaac machine.
- Results go to `experiments/results/` (gitignored, regenerable).
- Every episode is `(scenario, seed)`-reproducible; report mean ± 95% CI.

## RQ1 experiments (E1.1–E1.3) — ✅ COMPLETE (2026-07-19)

All three finished on both backends; full verdict in `FINDINGS.md` § RQ1,
numbers in `results/e11_latency/`, `results/e12_crossembodiment_dam/` +
`isaac/results/e12_crossembodiment/`, `results/e13_actlatency_dam/` +
`isaac/results/e13_actlatency/`; seed-matched tables/figures in
`isaac/results/COMPARISON.md`. Headlines: E1.1 p99.9 ≤ 0.53 ms, zero >2 ms in
105k calls, latency monotone+bounded in constraint count; E1.2 architecture
transfers via `drive_mode=differential` adapter, rotational-sweep erosion (F13)
maps a real diff-drive safety-liveness trade-off (b120 safe everywhere but S4
30 % completion), Isaac↔kinesim floors ±0.03; E1.3 empirical latency margin
≥75× (actuation-side, distinct from E4.2's perception-side).

## RQ5 experiments (E5.1–E5.2) — ✅ COMPLETE (kinesim, 2026-07-19)

Verdict in `FINDINGS.md` § RQ5 (+F14); numbers in `results/e51_lerobot/summary.md`
and `results/e52_distill/summary.md`. Headlines: E5.1 two LeRobot v3 datasets
(120+120 ego episodes, 82.6k frames, 420 boundary events, field completeness
1.000, load-back verified, seed-identical to the e2_full reference); E5.2
guard-corrected data cuts the guard's intervention rate 35–65 % after naive BC
retraining (S2 0.291→0.102, control bc_raw+dam 0.259) with 0 violations, while
the UNGUARDED student collapses in congested scenes — reusable data, non-
distillable guard. Isaac-in-the-loop recollection = future work (registry twin).

RQ5 (paper77 wording): 運行時所收集之失敗樣本是否具備完整性與再利用價值，
能支援後續模型優化？ Thesis-v2 D3 downgraded this to future work; E5.x is the
"下一版實驗" that closes it, re-grounded in the multi-robot setting. Two claims,
one experiment each:

- **E5.1 完整性 — collection + LeRobot export** (`run_e51_collect.py`,
  helper `common/lerobot_export.py`): runs the SAME (scenario, seed) grid
  under two filters — `dam` (guard-corrected **training-data pool**) and `raw`
  (failure-rich **raw-data pool**) — S2/S3/S5 × seeds 0–9. A `RecordingFilter`
  captures, at decision time, a per-agent ego view (42-dim body frame: current
  target + final goal + own vel + 6 nearest neighbours × (rel pos, rel vel,
  same_group, static)), the raw proposal, the verified action and guard
  telemetry (decision/delta/hard-slack + exact capsule distances). Export =
  one **LeRobot v3 dataset per pool** (`results/e51_lerobot/go2squad_{dam,raw}`,
  50 fps, parquet, no video; extra channels `action.raw`, `guard.*`,
  `observation.min_{dog,wall}_capsule_m`) + sidecars
  `meta/episode_manifest.jsonl` and `meta/boundary_events.jsonl` (RSMF-style
  events: intervention / qp_reject / hard_slack / violation_{dog,wall} runs
  with ±0.5 s context windows). Quality table = field completeness, window
  completeness, semantic diversity (paper77 表5.5 analogue) + load-back check
  through the official `LeRobotDataset` reader.
- **E5.2 再利用價值 — BC distillation closed loop** (`run_e52_distill.py`):
  train an MLP student per pool (loaded THROUGH the LeRobot reader — the
  round trip is part of the claim), then run students **unguarded** on
  held-out seeds 100–109 against `nominal+raw` (failure baseline) and
  `nominal+dam` (teacher reference). bc_dam ≪ bc_raw on violations ⇒ the
  guard-corrected pool demonstrably carries transferable safety behaviour —
  a data-driven reuse proof, not just log grading. bc_raw is the control for
  "BC merely smooths".

Fake-data smoke (stdlib): `python3 experiments/run_e51_collect.py --methods
raw,stop --scenarios S2 --seeds 2 --export none`. Full runs need the local DAM
venv (lerobot 0.4.4 is installed there).

## Status (compact)

Detailed result blurbs + findings F1–F11 + decision log: **`experiments/FINDINGS.md`**.
Isaac-vs-kinesim consistency: **`experiments/isaac/results/COMPARISON.md`**;
live progress: `python experiments/isaac/status.py`.

| Item | kinesim (thesis pool) | Isaac rerun |
|---|---|---|
| E2 baselines | ✅ (Pareto story; pydam≡dam) | ✅ story holds; raw physically crashes (4 falls); S3 dam 3/10 funnel wedge |
| E3.1 priority | ✅ | ✅ consistent |
| E3.2 swirl | ✅ (F4 fix) | ✅ consistent |
| E3.3 layering | ✅ | ✅ ordering holds; hard_only shows 3rd failure face (0.6 done + 6 falls); see F12 |
| E3.4 capsule | ✅ | ✅ consistent (headline claim reproduces) |
| E3.5 velocity-aware | ✅ | ✅ dose-response identical; floors 0.06–0.17 m lower with speed |
| E3.6 γ/dt | ✅ | ✅ monotone γ curve + γ0.7 S3 wedge anomaly reproduce |
| E3.7 gradient | ✅ | ✅ analytic>autograd floor + latency ratio reproduce |
| E4.1 exec noise | ✅ | ✅ graceful-to-a-fault reproduces (dither even helps) |
| E4.2 obs noise/latency | ✅ | ✅ latency-binds-first ladder reproduces (floors −0.06–0.09) |
| E4.3 slack collation | ✅ | ✅ done — Isaac Pearson(slack,viol) +0.17 vs kinesim +0.50: dual-channel conclusion STRONGER on real embodiment |
| E4.4 rogue | ✅ | ✅ dose-dependent dip, no catastrophe |
| E4.5 scale | ✅ | ✅ 100% liveness at every N, 0 falls @16 dogs, p99 1.35 ms; density erosion deeper (N≥12) |
| E5.1 LeRobot collection | ✅ (240 eps, 82.6k frames, 420 events, load-back OK) | ⏳ future (pool recollect via registry twin) |
| E5.2 BC reuse loop | ✅ (intervention −35…−65%, guard non-distillable) | ⏳ future |

Systematic embodiment deltas so far: makespan +18–20 % (policy under-tracks
~20 %), hard floors ~0.05–0.17 m lower, small viol counts where kinesim had 0.

## How to run

```bash
# fake-data smoke (any machine, stdlib only):
python3 experiments/tests/test_capsule_geometry.py
python3 experiments/run_e2_baselines.py --methods raw,stop --scenarios S1,S2,S3,S4,S5 --seeds 5

# full E2 with the real guard (local DAM venv works; Isaac itself not needed):
"$VENV" experiments/run_e2_baselines.py --methods raw,stop,orca,pydam,dam --scenarios S1,S2,S3,S4,S5 --seeds 50 --out experiments/results/e2_full

# ablations (pydam by default; add --method dam where supported):
python3 experiments/run_e31_priority.py --seeds 20      # E3.1 priority
python3 experiments/run_e32_swirl.py    --seeds 20      # E3.2 swirl x jitter
python3 experiments/run_e33_softhard.py --seeds 20      # E3.3 layering
python3 experiments/run_e35_velocity.py --seeds 20      # E3.5 velocity-aware x vmax
python3 experiments/run_e36_gamma.py    --seeds 20      # E3.6 gamma/dt sweep (slow: 320 eps)
python3 experiments/run_e34_capsule.py  --seeds 20      # E3.4 capsule vs disc
"$VENV" experiments/run_e37_gradient.py --seeds 20      # E3.7 analytic vs autograd (torch)
python3 experiments/run_e41_tracking.py --seeds 20      # E4.1 exec-noise sweep
python3 experiments/run_e42_obsnoise.py --seeds 20      # E4.2 obs noise + latency
python3 experiments/run_e43_slack.py                    # E4.3 slack collation (analysis only)
python3 experiments/run_e44_rogue.py    --seeds 20      # E4.4 non-cooperative agents
python3 experiments/run_e45_scale.py    --seeds 20      # E4.5 scale sweep

# E5 (RQ5) — LeRobot collection + reuse loop (local DAM venv; lerobot 0.4.4):
"$VENV" experiments/run_e51_collect.py --methods dam,raw --scenarios S2,S3,S5 --seeds 10
"$VENV" experiments/run_e52_distill.py --data experiments/results/e51_lerobot

# implementation-consistency gate (run after ANY guard/pydam change):
"$VENV" experiments/tests/test_pydam_vs_dam.py
python3 experiments/tests/test_capsule_geometry.py
```

