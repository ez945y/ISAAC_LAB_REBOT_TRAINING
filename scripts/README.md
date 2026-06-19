# Demo Scripts

Numbered demo scripts for learning Isaac Lab fundamentals — from basic scene setup
to robot teleoperation, dataset replay, and DAM safety demos.

> **Heads-up — this machine runs headless (no local window).** You *watch* a demo by
> connecting the **Isaac Sim WebRTC client** and adding `--livestream 2` — otherwise
> you'll only see console logs and think nothing happened. The `demoNN_*.sh` launchers
> turn streaming on (and source the env/ROS) for you. First: `source ~/IsaacLab/env_isaaclab/bin/activate`.
> See **WebRTC Livestreaming** below for the client IP/ports.

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
| 09 | `09_dam_car_scripted_comparison_demo.py` | Twin-lane Jetbot RAW vs DAM, in-process (single command) |
| 10 | `10_dam_car_ros_comparison_demo.py` | Twin-lane Jetbot RAW vs DAM, **DAM driven over ROS 2** — needs the guard node (use `demo10_ros_webrtc.sh`) |
| 11 | `11_go2_squad_dispatch.py` | Go2 6-dog squad dispatch — drive it live over ROS/keyboard (use `demo11_go2_squad_ros.sh`) |
| 12 | `12_dam_teleoperate_demo.py` | Leader-arm teleoperation with DAM filtering |
| 13 | `13_dam_safety_demo.py` | Keyboard end-effector control with DAM filtering |
| 14 | `14_dam_scripted_comparison_demo.py` | Twin-arm SO-ARM-101 scripted RAW vs DAM (recording); `demo14_arm_comparison.sh` |

10 and 11 boot the **native** isaacsim app (their ROS-bridge / RL-policy pieces don't
run under Isaac Lab's `AppLauncher`); the rest are Isaac Lab scripts. The `demoNN_*.sh`
launchers in this folder handle the env + ROS sourcing + multi-process wiring for you.

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
09  + Jetbot RAW vs DAM safety (in-process)
 ↓
10  + Jetbot RAW vs DAM over ROS 2 bridge
 ↓
11  + Go2 squad dispatch (ROS / keyboard control)
 ↓
12  + Leader-arm DAM teleoperation
 ↓
13  + Keyboard DAM safety filtering
 ↓
14  + Twin-arm scripted DAM comparison
```

## Usage

Scripts 01–08 are standalone — run from the repo root:

```bash
python scripts/07_moving_from_dataset.py --dataset MikeChenYZ/so101_isaac_mimic_test --episode 0 --video --video_dir ./videos --enable_cameras
python scripts/08_augmented_replay.py --enable_cameras
```

Dataset conversion and streaming utilities live under `tools/`.

## DAM / robotics demos (09–14)

All accept `--livestream 2` to stream to the WebRTC client (see below). The
`demoNN_*.sh` launchers source the env + ROS and wire up the extra processes — use
them for 10/11/14; the rest run with a single `python` command.

**09 — Jetbot RAW vs DAM, one process:**
```bash
python scripts/09_dam_car_scripted_comparison_demo.py --livestream 2
```

**10 — Jetbot RAW vs DAM, DAM driven over ROS 2** (sim + guard node, two processes):
```bash
scripts/demo10_ros_webrtc.sh           # launches guard node + sim + stream
# (camera: scripts/demo10_ros_webrtc.sh --cam-eye "-3,0,2.5" --cam-target "0.7,0,0.1")
```
<sub>Manual: T1 `python scripts/10_dam_car_ros_comparison_demo.py --livestream 2`,
T2 `python tools/ros/dam_jetbot_guard_node.py` — both with ROS 2 sourced.
The DAM car only moves once the guard connects (a few seconds).</sub>

**11 — Go2 squad dispatch, drive it live:**
```bash
scripts/demo11_go2_squad_ros.sh        # sim + interactive dispatch client
```
<sub>Manual: T1 `python scripts/11_go2_squad_dispatch.py --ros-control --livestream 2`,
T2 `python tools/ros/go2_squad_dispatch_client.py --interactive`.
Also `--auto` for the hands-off 2×3→3×2 run, or no `--ros-control` for keyboard drive
in the sim terminal.</sub>

**12 — leader-arm teleop + DAM:**
```bash
python scripts/12_dam_teleoperate_demo.py --livestream 2
```

**13 — keyboard end-effector control + DAM:**
```bash
python scripts/13_dam_safety_demo.py --livestream 2
```

**14 — twin-arm scripted RAW vs DAM:**
```bash
scripts/demo14_arm_comparison.sh --mode compare      # or --mode dam / --mode raw
```

## WebRTC Livestreaming (remote viewing)

Any script can be streamed to the **Isaac Sim WebRTC Streaming Client** (e.g. on a
Mac) by adding `--livestream 2`:

```bash
source ~/IsaacLab/env_isaaclab/bin/activate
python scripts/09_dam_car_scripted_comparison_demo.py --livestream 2
```

On the client: enter the server IP, ports **49100** (signal) / **47998** (stream), Connect.

> **Native demos (10, 11):** these boot the native isaacsim app, not `AppLauncher`,
> so `apply_livestream_defaults` doesn't apply. They inject the same proven settings
> via `native_livestream_argv()` (incl. `streamType=webrtc`) before `SimulationApp` —
> their `.sh` launchers handle it, and also kill any prior stream first (a hard-killed
> run leaks the NVENC encoder → `NVST_R_BUSY` on the next start).

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
