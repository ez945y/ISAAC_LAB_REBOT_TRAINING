# Isaac Sim 5.1 / Kit 110 遷移筆記

環境升級到 **Isaac Sim 5.1 / Kit 110.1.1** 後，原本能跑的 demo 連續踩了好幾個
破壞性變動。這份文件記錄每個雷的「症狀 → 根因 → 解法」，方便之後在遠端
（GB10 / NoMachine SSH）重現或排查。

執行環境：
- 機器：`rst_spark@edgexpert-19ce`，GPU = **NVIDIA GB10**（Grace-Blackwell, ARM/aarch64）
- 連線：NoMachine 遠端桌面，`DISPLAY=:1`，x11
- Kit：`110.1.1+production`，kernel `210.1.5`

---

## 1. PhysX view 回傳 warp array，不支援進階索引

**症狀**
```
RuntimeError: Item indexing is not supported on wp.array objects
```
出現在 `root_physx_view.get_dof_limits()[0, ids, 0]` 之類的索引。

**根因**
Kit 110 把 `root_physx_view.get_*()`（`get_dof_limits`、`get_jacobians`、
`get_generalized_mass_matrices`、`get_gravity_compensation_forces`）的回傳值
從 `torch.Tensor` 改成 `warp.array`，不支援 PyTorch 風格的 fancy indexing。

**解法**
新增 `controll_scripts/utils/warp_compat.py` 的 `physx_to_torch()`，索引前先轉
torch：
```python
from controll_scripts.utils import physx_to_torch
jl = physx_to_torch(robot.root_physx_view.get_dof_limits())
lower = jl[0, arm_ids, 0]
```

---

## 2. `robot.data.*` 也變成 warp ProxyArray

**症狀**
```
RuntimeError: quat_inv() Expected a value of type 'Tensor' ... found type 'ProxyArray'
Cast error: Unable to cast ProxyArray(... dtype=quatf ...) to Tensor
```
出現在把 `robot.data.root_quat_w` 等丟進 `math_utils.subtract_frame_transforms`
/ `quat_inv` 時。

**根因**
Kit 110 的 warp 後端讓 `robot.data.*` 的姿態/速度/關節資料回傳 `ProxyArray`
（dtype 可能是 `vec3` / `quatf` 等 warp 結構型別），無法直接餵 torch 數學函式。

**解法**
同一個 `physx_to_torch()`（已支援 `ProxyArray`，`vec3→3`、`quatf→4` 展平）。
所有「讀 `robot.data.X` → 餵 torch 運算」的地方都包一層：
```python
ee_pos_w  = physx_to_torch(robot.data.body_pos_w)[:, ee_idx]
root_quat = physx_to_torch(robot.data.root_quat_w)
```
已套用：`controllers/base.py`、`ik_controller.py`、`osc_controller.py`、
`safety/isaac_resolver.py`、`scripts/11_dam_safety_demo.py`。

> ⚠️ 還沒走到的程式路徑（OSC 模式、其他腳本主迴圈）可能還有沒包到的
> `robot.data.*`。再撞到 `ProxyArray` / `wp.array` 的 cast error，作法一樣：
> 找到那個讀取，用 `physx_to_torch(...)` 包起來。

---

## 3. `Se3Keyboard` → `omni.appwindow` 未載入

**症狀**
```
AttributeError: module 'omni' has no attribute 'appwindow'
# 或
ModuleNotFoundError: No module named 'omni.appwindow'
```

**根因**
`isaaclab.devices.Se3Keyboard` 內部呼叫 `omni.appwindow.get_default_app_window()`。
`omni.appwindow` 是 Kit extension，**只有在 app 以 GUI / 視窗模式啟動時才會啟用**。
本環境預設 `headless=True`（見下節），所以該 extension `enabled=False`，鍵盤掛掉。

**注意：不要在 module 頂層 `import omni.appwindow`**
`controll_scripts` 在 AppLauncher 之後*立刻*被 import，那時 extension 還沒就緒，
module 層級 import 會 `ModuleNotFoundError` 把整個套件匯入打掛。已改成在
`KeyboardInputDevice.__init__` 建構 `Se3Keyboard` *之前*做延遲 import，並在失敗時
丟出清楚的中文錯誤（`input_devices/keyboard.py`）。

---

## 4. 預設 headless / 遠端看畫面要用 livestream

**診斷**（`tools/livestream/diag_appwindow.py`，只啟動 app 查狀態）
```
[DIAG] headless 設定 : True            ← 沒下 --headless 也是 True
[DIAG] omni.appwindow enabled= False   ← 因 headless 被關
```

**根因**
此 Isaac Lab build 在這台 GB10 + NoMachine 上**預設 headless=True**（沒有
`HEADLESS` env var，就是內建預設）。GB10(ARM) + NoMachine 虛擬顯示要開原生
Vulkan 視窗本來就難成立。

**解法：WebRTC livestream**（NVIDIA 對遠端/headless 的官方做法）
```bash
python scripts/11_dam_safety_demo.py --controller ik --livestream 2
```
- livestream 模式仍是 headless，但會啟用串流用的 app window → `omni.appwindow`
  變 `enabled=True` → 鍵盤可用（事件透過 WebRTC client 轉發）。

**怎麼連上去看**
1. 在 GB10 上 `hostname -I` 取得 IP。
2. 本機（Mac）下載 **Isaac Sim WebRTC Streaming Client**（NVIDIA 獨立 app）。
3. Client 填 IP、埠 **8211**、Connect。demo 要保持在跑。
4. 需要時放行防火牆：`sudo ufw allow 8211`。

> 不需要鍵盤、純驗證 DAM 行為時，用 scripted demo（完全不碰視窗）：
> `python scripts/10_dam_scripted_comparison_demo.py --mode compare`

---

## 5. `set_joint_position_target` 改名

**症狀**
```
DeprecationWarning: 'set_joint_position_target' will be deprecated.
Please use 'set_joint_position_target_index' instead.
```

**解法**
全域 1:1 改名（**只動 position**，velocity / effort 沒被 deprecate 不要碰），
參數不變：
```bash
grep -rl "set_joint_position_target(" scripts/ tools/ --include="*.py" | \
  while IFS= read -r f; do
    perl -pi -e 's/set_joint_position_target\(/set_joint_position_target_index(/g' "$f"
  done
```
> 待確認：新方法簽名假設與舊版相同 `(target, joint_ids=...)`。若重跑報 `TypeError`，
> 代表簽名有變（可能要全 DOF tensor），需逐一調整。

---

## 6. 待處理：GB10 PyTorch 算力不支援（sm_121）

**警告**（每次啟動都出現）
```
Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

**意義**
GB10 是 sm_121，目前裝的 PyTorch 只支援到 sm_120。目前只是 warning，但 GPU 上
的算子可能在後續步驟才真正失敗。鍵盤/串流這關通過後若 GPU 運算報錯，要升級到
支援 GB10 / sm_121 的 PyTorch build（對應 CUDA 13.x / cu130）。尚未處理。

---

## 通用排查心法

- **改名類 deprecation** → 可全域無腦替換（先 grep 確認範圍，改完 grep 驗殘留 +
  `py_compile`）。
- **型別 / 語意變動**（warp array、ProxyArray）→ 不能無腦換，要在「讀取 → 運算」
  邊界做轉換（`physx_to_torch`），一處一處包。
- **extension / 啟動模式問題**（omni.appwindow、headless）→ 先用 `tools/livestream/diag_appwindow.py`
  之類的最小腳本確認狀態，再決定解法，不要猜。
