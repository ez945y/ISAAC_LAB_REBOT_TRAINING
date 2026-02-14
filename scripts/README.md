# Standalone Scripts

Incremental experiment scripts for learning Isaac Lab fundamentals — from basic scene setup to robot teleoperation and dataset replay.

## Scripts

| # | Script | Description |
|---|--------|-------------|
| 01 | `01_basic_auto_drive.py` | Basic scene with Jetbot auto-driving |
| 02 | `02_keyboard_control.py` | WASD keyboard control with command smoothing |
| 03 | `03_domino_fpv.py` | Domino physics + first-person camera following |
| 04 | `04_trajectory_record.py` | Trajectory recording & loop playback |
| 05 | `05_robot_demo.py` | SO-ARM-101 robot demo with IK/OSC controller |
| 06 | `06_teleoperate_demo.py` | Leader arm teleoperation via socket |
| 07 | `07_moving_from_dataset.py` | Load & inspect LeRobot dataset episodes |
| 08 | `08_replay_from_dataset.py` | Replay dataset actions in simulation (direct joint position, 30 FPS) |

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
07  + LeRobot dataset loading
 ↓
08  + Dataset replay in sim
```

## Usage

All scripts are standalone. Run from the repo root:

```bash
python scripts/01_basic_auto_drive.py
python scripts/07_moving_from_dataset.py --episode 0 --fps 30
```