# Thesis Experiments — Progress & Handoff

Branch: `exp/thesis-experiments` (forked from `feat/go2-squad-dispatch` after its PR).
Maps to the RQ design (RQ1–RQ4; RQ5 dropped). One experiment script at a time:
finish + fake-data-validate the current one before starting the next.

## Ground rules

- **Fake-data first**: every script must run and produce sane numbers on the
  lightweight kinematic sim (`common/kinesim.py`) with `raw`/`stop` filters on
  any machine (stdlib + nothing else). DAM/ORCA runs and Isaac-in-the-loop
  validation happen later on the Isaac machine.
- Results go to `experiments/results/` (gitignored, regenerable).
- Every episode is `(scenario, seed)`-reproducible; report mean ± 95% CI.

## Status board

| Item | What | Status |
|---|---|---|
| Capsule upgrade | 3-sphere sampling → exact segment-segment capsule in `go2_dam_wrapper.py` (+ new `capsule_geometry.py`) | ✅ code done; geometry unit-tested (500 random cases vs brute force). ⚠️ wrapper import NOT tested (needs torch/dam → Isaac machine) |
| Shared infra | `common/`: kinesim, scenarios S1–S5, filters (raw/stop/orca-TODO/dam), metrics | ✅ done, smoke-tested |
| **E2 baselines (RQ2)** | `run_e2_baselines.py`: scenario × method × seed → episodes.csv + aggregate.csv + summary.md | ✅ script done, fake-data validated (raw/stop, S1–S5). ⏳ pending: `dam` run on Isaac machine; ORCA baseline |
| E3.1 priority ablation | S1, priority on/off, per-group completion | ⬜ next up |
| E3.2 swirl ablation | S2, swirl on/off, deadlock rate | ⬜ |
| E3.3 soft/hard ablation | S3, layered vs hard-only, throughput | ⬜ |
| E3.5 velocity-aware ablation | S2, TTC vs static neighbours | ⬜ |
| E3.6 γ / dt sweep | S1+S3, conservatism-performance curve | ⬜ |
| E3.4 capsule vs disc | S4, equal-radius disc comparison | ⬜ |
| E3.7 analytic vs autograd | S2, spurious-stop rate | ⬜ |
| E4.1 tracking-error injection | noise on executed vs commanded velocity | ⬜ |
| E4.2 state noise + latency | noisy/delayed neighbour observations | ⬜ |
| E4.3 slack usage | log max hard-slack across all runs (piggyback) | ⬜ |
| E4.4 non-cooperative agent | 1–2 dogs bypass the filter | ⬜ |
| E4.5 scale sweep | S5 with N ∈ {2,4,8,12,16} (+ E1.2 timing) | ⬜ |
| E1.x cross-embodiment / real latency | on Isaac machine | ⬜ |

## How to run

```bash
# fake-data smoke (any machine, stdlib only):
python3 experiments/tests/test_capsule_geometry.py
python3 experiments/run_e2_baselines.py --methods raw,stop --scenarios S1,S2,S3,S4,S5 --seeds 5

# full E2 on the Isaac-side machine (needs torch/osqp/dam in the venv; Isaac itself not needed):
python3 experiments/run_e2_baselines.py --methods raw,stop,dam --scenarios S1,S2,S3,S4,S5 --seeds 50 --out experiments/results/e2_full
```

## Handoff notes (read before continuing)

1. **Verify the capsule change on the Isaac machine** before trusting any DAM
   numbers: `python -c "import sys; sys.path.insert(0,'tools'); from controll_scripts.safety import Go2DAMWrapper"`,
   then a short demo11 run. The old 3-sphere code was replaced by
   `seg_seg_closest()`; the QP gradient now uses the continuous spine offset
   `s_off` instead of a sampled offset.
2. **Neighbour tuples grew an 8th element** (`static` flag, 1.0 = wall point),
   produced by `kinesim._neighbors_of`. `Go2DAMWrapper.filter` reads indices
   0–6 only, so it is unaffected; `StopFilter` uses it to apply smaller wall
   thresholds. Keep index compatibility if you extend the tuple.
3. **ORCA baseline (B2) is a stub** — wire up vetted RVO2 python bindings, do
   not hand-roll ORCA (baseline numbers must be unimpeachable).
4. **Walls are chains of static point-agents** (spacing 0.35 m). The guard has
   a single `min_dist` for all neighbours, so dog-wall clearance is also 0.7 m
   for DAM: the S3 gap (1.6 m) is only barely feasible. Per-type min-dist (or
   true segment obstacles) is a queued guard improvement; metrics already
   report dog-wall distance separately (`wall_dist` = 0.35 m floor).
5. **Stop baseline deadlocks ~100% in symmetric scenarios** (mutual latch below
   resume distance). That is the expected narrative (the weakness of threshold
   stopping), not a bug.
6. **kinesim uses the same first-order holonomic model as the guard** — by
   design there is zero model mismatch here. Model-mismatch robustness is
   exactly E4.1/E4.2's job (inject execution/observation error); Isaac-loop
   runs close the rest of the gap.
7. Fresh filter instance per episode (no state bleed); `DamFilter` lazily
   imports torch/dam so raw/stop runs work on machines without them.

## Decision log

- 2026-07-11 Branch created; true-capsule replaces 3-sphere sampling
  (inter-sample gaps under-estimated the body on oblique crossings).
- 2026-07-11 E2 fake-data findings that drove fixes: S3 starts could overlap
  (now min-sep sampled in a deeper spawn band, hard error if unplaceable);
  StopFilter froze 1.5 m from wall chains (now separate wall thresholds
  0.45/0.6 via the static flag).
