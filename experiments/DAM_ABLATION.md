# 正式版 DAM 消融介面（--method dam）

讓論文的每一個消融／魯棒性實驗都能直接在**正式執行路徑**
（`Go2DAMWrapper` → dam 0.7 Guardrail → OSQP）上產出數據，而不只是
pydam 參考實作。2026-07-12 建立；等價閘門與 smoke 驗證紀錄見文末。

**架構釐清**：dam 套件（Security Guard repo）只是中介層——提供
Guardrail 執行框架、裁決聚合與 stackfile 載入；guard 的實際邏輯是
掛進去的**外部 callback**（`go2_min_max_separation`，住在本 repo
`tools/controll_scripts/safety/`）。本次所有開關都加在 callback 層，
**dam 套件本身零改動**。「正式版」在本文件中一律指「經 dam Guardrail
＋OSQP 執行的 callback 路徑」，與 pydam（numpy/SLSQP 直跑同一數學）
相對。

## 用法

所有掃描腳本（e31–e45，e43 除外——它只做彙整）都接受 `--method`：

```bash
VENV="/Users/chenyizhong/Documents/Claude/Projects/Security Guard.nosync/.venv/bin/python3"

# 參考實作（預設，免 venv）
python3 experiments/run_e35_velocity.py --seeds 20

# 正式版 dam（需 venv：torch / dam 0.7 / osqp）
"$VENV" experiments/run_e35_velocity.py --method dam --seeds 20
```

`--method dam` 且未指定 `--out` 時，輸出目錄自動加 `_dam` 後綴
（如 `results/e35_velocity_dam/`），**永不覆蓋論文既有數據**。

## 旋鈕 → 消融軸對照

兩實作旋鈕**同名**（`make_guard(method, **旋鈕)`，見
`common/filters.py`）。dam 端：callback 參數合併進暫存 stackfile；
`max_v`/`max_omega` 建構 `HolonomicSolver`。

| 消融軸 | 旋鈕 | callback 支援 |
|---|---|---|
| E3.1 優先權 | `self_priority` / 鄰居 priority（呼叫時傳入） | 既有 |
| E3.2 swirl | `swirl` | 既有 |
| E3.3 軟硬分層 | 僅硬層 = `comfort_dist=0.7`；舒適硬化 = `w_soft=1000` | 既有 |
| E3.4 身體模型 | 內接圓盤 = `capsule_half=0`；外接 = `capsule_half=0, min_dist=1.2, wall_min_dist=0.95` | `wall_min_dist` **本次新增** |
| E3.5 速度感知 | `velocity_aware=False` | **本次新增** |
| E3.6 γ/Δt | `gamma`、`dt` | 既有 |
| E3.7 梯度路徑 | `grad_mode="autograd"` | **本次新增** |
| E4.1/E4.2/E4.5 | 模擬層變因（雜訊/延遲/N），guard 不變 | 不需 |
| E4.3 鬆弛遙測 | info dict `hard_slack_max`（見下） | **本次新增** |
| E4.4 非合作 | `FilterRouter` 混編 raw（實驗管線層，與 callback 無關） | 不需 |

`cost_diag` 為 pydam 專屬（正式版 QP 硬編碼 `[2.0, 0.5, 3.0]`），
傳給 dam 會直接 raise。

## 本次在 callback 層新增的東西

全部在 `tools/controll_scripts/safety/`（dam 套件零改動）：

1. **`go2_min_max_separation` 三個新參數**（預設值 = 原行為，等價
   閘門保證零漂移）：
   - `wall_min_dist`（None→`min_dist`）：靜態鄰居的獨立硬底線。
   - `velocity_aware`（True）：False 時鄰居視為靜態。
   - `grad_mode`（"analytic"）："autograd" 走真距離自動微分梯度。
2. **鬆弛遙測**：callback 每次呼叫把「硬層鬆弛最大值」（求解失敗且
   硬層激活時為 nan）、push/pull 約束數寫回 `_NeighborHolder`；
   `Go2DAMWrapper` 以 `last_hard_slack` / `last_n_push` /
   `last_n_pull` 屬性曝露；`DamFilter.filter()` 放進 info dict，
   metrics 管線（`max_hard_slack_m` 欄）自動收。**部署含義**：這是
   論文「鬆弛遙測＋實測距離雙通道監視」建議裡鬆弛通道的正式版接口。
3. **`torchgrad.py` 搬家**：自 `experiments/common/` 移至
   `tools/controll_scripts/safety/`（正式版與 pydam 共用同一份，
   `experiments/common/torchgrad.py` 只剩載入 shim）——
   `grad_mode="autograd"` 在兩實作跑的是同一段程式碼。

## 鐵律

- **改了 guard（callback / wrapper / stackfile）之後，必跑等價閘門**：
  `"$VENV" experiments/tests/test_pydam_vs_dam.py`（300 隨機情境，
  最壞分量誤差須為 0.000）。
- pydam 的建構子預設值鏡射 `go2_squad_safety.yaml`——改 yaml 要同步
  改 `common/pydam.py` 預設值，反之亦然。

## 驗證紀錄（2026-07-12）

- 等價閘門：300 案例、189 clamped、worst err 0.000 → **PASS**
  （新參數預設下零行為漂移）。
- dam smoke（`run_e35_velocity.py --method dam --seeds 2`）：六條件
  底線 0.824/0.688/0.619（va_off）、0.877/0.785/0.747（va_on）、
  侵蝕 0/5/10——與論文表值及 pydam **至小數第三位一致**；
  `max_hard_slack_m` 遙測欄有值。
