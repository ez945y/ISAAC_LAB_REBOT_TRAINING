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

## RQ5 實驗 — 失敗樣本的收集與再利用（Isaac 全流程，2026-07-19 重新設計）

RQ5 問的是：**系統運行時收集到的失敗樣本，是不是「完整」、而且能拿來讓策略
變得更好？** 第一版（舊腳本 `run_e51_collect.py` / `run_e52_distill.py`，已提交
於 03b0535，全在輕量模擬器上跑）有三個要害：(1) 完全沒用 Isaac；(2) 兩次都是拿
「手寫控制器」去做模仿，不是糾正策略本身；(3) 每個資料池只有 30 局，太少。這一版
（`run_e5_loop.py`）改成**完全在 Isaac（真 Go2 四足 + 真 dam 過濾器）上跑的閉環**，
並嚴格對照原本規劃的三步流程：

| 原本規劃的步驟 | 這一版怎麼做 |
|---|---|
| ① 收集訓練資料 | **收集甲**：控制器直接開、不加過濾器，跑 S5 → 得到「原始資料池」 |
| ② 測試（不加過濾器） | **測試階段**：把只用原始資料訓出的「初版策略」關掉過濾器跑，看它多不安全 |
| ③ 收集修正資料、再訓練、再測 | **收集乙**：初版策略上線、過濾器即時糾正它的錯、把糾正紀錄回收成「修正資料池」→ 訓「改良策略」→ 再測 |

**名詞對照**（正文一律用中文說法，程式代號放這裡供重現，不在敘述中出現）：

| 本文說法 | 意思 | 程式代號 |
|---|---|---|
| 原始資料池 | 控制器直接跑、失敗多、違規多的原始紀錄 | `base` |
| 對照資料池 | 同樣多的額外原始紀錄，一樣沒有過濾器修正 | `extra` |
| 修正資料池 | 初版策略上線後、被過濾器糾正的那些時刻 | `damfix` |
| 初版策略 | 只用原始資料池訓練出來的策略 | `v0` |
| 改良策略 | 用「原始資料池 + 修正資料池」訓練 | `v1` |
| 對照策略 | 用「原始資料池 + 對照資料池」訓練（資料量相同，但沒有過濾器修正） | `v1c` |
| 測試 | 用訓練時沒見過的場景種子來評估 | `bench` |

**為什麼要多一組「對照資料池／對照策略」**：改良策略若比初版好，可能是「過濾器修正
有價值」，也可能只是「多餵了資料」。對照策略吃的是**同樣多、但沒有過濾器修正**的資料。
唯有改良策略贏過對照策略，才能證明是**過濾器修正本身**帶來安全，而不是資料變多——
少了這組控制，口試委員一句「你只是資料比較多」就破功。

**場景選擇**：訓練用 **S5**（隨機起點/終點、6 隻狗巡航），因為它是唯一「換一個種子
就換一整局」的場景，資料才有多樣性；**S2**（兩狗對衝）每個種子只差 ±0.15 m、幾乎
同一條軌跡，所以留著當「訓練沒見過的場景型態」的零樣本泛化探針。

**資料格式**：每個資料池匯出成一份 LeRobot v3 資料集（parquet + jsonl，50 fps），
每一幀含 42 維機體座標觀測（目前航點 + 終點 + 自身速度 + 最近 6 鄰居的相對位置/速度/
同組旗標/牆旗標）、過濾器核可後的動作、控制器的原始提案、以及過濾器遙測（有沒有介入、
修正量、硬約束鬆弛、精確膠囊距離）；另附兩個側車檔：episode 清單、邊界事件紀錄
（介入／拒解／硬鬆弛／違規，每筆帶前後各 ±0.5 秒的上下文視窗）。三個策略全部
**透過官方 LeRobot 讀取器載入訓練**——這個「寫出去、再讀回來訓練」的來回本身，
就是「格式可再利用」的證明。

**本次規模（精簡版，全部每局存檔、可斷點續跑）**：原始資料 150 種子、對照資料
60 種子、過濾器修正 60 種子；測試用訓練沒見過的種子（S5 每條件 20 個 + S2 探針每
條件 10 個）。結果出爐後寫入 `FINDINGS.md` § RQ5，報告在
`isaac/results/e5_loop/REPORT.md`。

```bash
source ~/IsaacLab/env_isaaclab/bin/activate
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
python experiments/run_e5_loop.py --backend isaac \
    --base-seeds 150 --extra-seeds 60 --fix-seeds 60 --bench-seeds 20 --probe-seeds 10
# 產出：experiments/isaac/results/e5_loop/{datasets/, policies/, collect_*/, bench/, REPORT.md}
# 斷了就重跑同一行，已完成的局會自動略過續跑。
# kinesim 端到端小規模冒煙（分鐘級，驗管線用）：加 --backend kinesim --smoke
```

**結果（2026-07-20 跑完，6.3 小時）**：完整結論與數字在 `FINDINGS.md` § RQ5（+F15），
圖規格在 `E5_FIGURES.md`，原始表格在 `isaac/results/e5_loop/REPORT.md`。三句話：
(1) 修正資料可再利用——改良策略需要過濾器的程度低於控制器、也低於吃等量資料的
對照策略（S5 介入率 0.050 vs 0.109 vs 0.232），且在訓練沒見過的 S2 場景零樣本成立
（0.179 vs 0.281）；(2) 過濾器不可拿掉——任何策略關掉過濾器在擁擠場景一律崩（違規步
1000+）；(3) 誠實代價——天真的單幀重訓把安全學成過度保守，S5 完成率崩到 0.15
（控制器 0.95），論文須與介入率下降並列。

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
| RQ5 舊版（kinesim） | ✅ 已提交 03b0535，但被下方 Isaac 閉環取代（手寫控制器蒸餾、30 局，說服力不足） | — |
| RQ5 收集（原始/對照/修正資料池） | — | ✅ 完成（900+360+360 局，欄位完整率 1.000，修正資料池事件最密 534） |
| RQ5 再利用（初版→改良/對照策略→測試） | — | ✅ 完成，雙面結論（介入率 S5 −78%/S2 −36%、贏對照組；但完成率換安全）見 F15 |

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

# RQ5 — 失敗樣本收集與再利用閉環（Isaac 全流程，見上方 RQ5 段落）:
source ~/IsaacLab/env_isaaclab/bin/activate
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
python experiments/run_e5_loop.py --backend isaac \
    --base-seeds 150 --extra-seeds 60 --fix-seeds 60 --bench-seeds 20 --probe-seeds 10
# 舊版（kinesim，已被上式取代，保留供對照）:
# "$VENV" experiments/run_e51_collect.py --methods dam,raw --scenarios S2,S3,S5 --seeds 10
# "$VENV" experiments/run_e52_distill.py --data experiments/results/e51_lerobot

# implementation-consistency gate (run after ANY guard/pydam change):
"$VENV" experiments/tests/test_pydam_vs_dam.py
python3 experiments/tests/test_capsule_geometry.py
```

