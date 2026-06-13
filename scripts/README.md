# Demo Scripts

Numbered demo scripts for learning Isaac Lab fundamentals — from basic scene setup
to robot teleoperation, dataset replay, and DAM safety demos.

## Scripts

| # | Script | Description |
|---|--------|-------------|
| 01 | `01_basic_auto_drive.py` | Basic scene with Jetbot auto-driving |
| 02 | `02_keyboard_control.py` | WASD keyboard control with command smoothing |
| 03 | `03_domino_fpv.py` | Domino physics + first-person camera following |
| 04 | `04_trajectory_record.py` | Trajectory recording & loop playback |
| 05 | `05_robot_demo.py` | SO-ARM-101 robot demo with IK/OSC controller |
| 06 | `06_teleoperate_demo.py` | Leader arm teleoperation via socket |
| 07 | `07_moving_from_dataset.py` | Load & replay LeRobot dataset episodes in sim with video export |
| 08 | `08_augmented_replay.py` | Augmented replay with multiple episodes, cube configs & cameras |
| 09 | `09_dam_car_scripted_comparison_demo.py` | Twin-lane Jetbot RAW vs DAM SafetyGuard comparison |
| 10 | `10_dam_scripted_comparison_demo.py` | Scripted RAW vs DAM safety comparison for recording |
| 11 | `11_dam_safety_demo.py` | Keyboard EE control with DAM filtering |
| 12 | `12_dam_teleoperate_demo.py` | Leader-arm teleoperation with DAM filtering |
## Progression

```
01  Basic scene + auto drive
 ↓
02  + Keyboard control
 ↓
03  + Domino physics + FPV camera
 ↓
04  + Trajectory recording / playback
 ↓
05  + SO-ARM-101 robot (IK/OSC)
 ↓
06  + Leader arm teleoperation
 ↓
07  + LeRobot dataset replay in sim
 ↓
08  + Augmented replay (multi-episode, multi-camera)
 ↓
09  + Scripted Jetbot safety comparison
 ↓
10  + Scripted SO-ARM DAM comparison
 ↓
11  + Keyboard DAM safety filtering
 ↓
12  + Leader-arm DAM teleoperation
```

## Usage

All scripts are standalone. Run from the repo root:

```bash
python scripts/07_moving_from_dataset.py --dataset MikeChenYZ/so101_isaac_mimic_test --episode 0 --video --video_dir ./videos --enable_cameras
python scripts/08_augmented_replay.py --enable_cameras
python scripts/09_dam_car_scripted_comparison_demo.py
python scripts/10_dam_scripted_comparison_demo.py --mode compare
```

Dataset conversion and streaming utilities live under `tools/`.

## WebRTC Livestreaming (remote viewing)

Any script can be streamed to the **Isaac Sim WebRTC Streaming Client** (e.g. on a
Mac) by adding `--livestream 2`:

```bash
source ~/IsaacLab/env_isaaclab/bin/activate
python scripts/09_dam_car_scripted_comparison_demo.py --livestream 2
```

On the client: enter the server IP, ports **49100** (signal) / **47998** (stream), Connect.

### Why `livestream_support.py` exists

Isaac Lab's `--livestream 2` only enables the livestream extension on its headless
experience — it does **not** configure what a remote client actually needs. The
official `isaacsim.exp.full.streaming` app bakes those in; `tools/livestream/livestream_support.py`
reproduces them so you don't hand-type a long `--kit_args="..."`. Every script here
already calls it (one line after `parse_args()`, before `AppLauncher`):

```python
args_cli = parser.parse_args()
from tools.livestream.livestream_support import apply_livestream_defaults
apply_livestream_defaults(args_cli)          # no-op unless --livestream is set
app_launcher = AppLauncher(args_cli)
```

It injects, only when streaming:

| Setting | Fixes |
|---|---|
| `primaryStream/publicIp` (auto-detected LAN IP) | black screen over NAT/VPN (mode 2 defaults to 127.0.0.1) |
| `--no-window` | "Cannot stream video frame" resolution mismatch |
| `app/livestream/allowResize` + `primaryStream/allowDynamicResize` | client gets one frame then drops |
| `visualizer=kit` | stream stalls after a few frames (Kit visualizer pumps `app.update()`) |
| `runLoops/main/rateLimitEnabled=true` (60 Hz) | steps dumped all at once / no stream cadence |
| `primaryStream/enableEventTracing=false` | stops `NvStreamer-*.etli` log spam |

### Notes

- **Advertised IP** is auto-detected (the host's private LAN IP). Override with
  `export LIVESTREAM_PUBLIC_IP=<ip>` before running.
- **Finished scripted demos**: if a demo should remain viewable after its scripted
  replay finishes, keep the official visualizer loop alive instead of checking
  livestream flags:
  ```python
  while sim.is_headless_or_exist_active_visualizer():
      sim.step()
  ```
- **Network**: the client must be able to reach the server on TCP 49100 + UDP 47998
  (open them in the server firewall; over a VPN the routing must reach the server's
  LAN). `tools/livestream/livestream_keepalive.py` is a minimal blue-cube scene for
  testing the stream in isolation.
