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
| 07 | `07_moving_from_dataset.py` | Load & replay LeRobot dataset episodes in sim with video export |
| 08 | `08_augmented_replay.py` | Augmented replay with multiple episodes, cube configs & cameras |
| 09 | `09_moving_to_hdf5.py` | FK→IK round-trip conversion: LeRobot joints → EE-space HDF5 |

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
09  + FK→IK conversion to EE-space HDF5
```

## Usage

All scripts are standalone. Run from the repo root:

```bash
python scripts/07_moving_from_dataset.py --episode 0 --video --video_dir ./videos --enable_cameras
python scripts/08_augmented_replay.py --enable_cameras
python scripts/09_moving_to_hdf5.py --dataset MikeChenYZ/soarm-fmb-v2 --output ./datasets/move_demo.hdf5
```