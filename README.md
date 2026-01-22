# Isaac Lab Practice

This is a personal practice project for learning [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

## Overview

This repository serves as a workspace to explore Isaac Lab features, create custom environments, and experiment with robot simulation and imitation learning.

## Requirements

- **OS**: Linux (Ubuntu 24.04)
- **Python**: 3.11
- **NVIDIA Isaac Sim**: 5.1+
- **Isaac Lab**: 2.3.0+

## Projects

### SO-ARM Mimic

- **[so_arm_mimic](./projects/so_arm_mimic)**: Imitation learning environment for SO-ARM-101 robot arm. Features cube stacking task with Leader Arm teleoperation, Isaac Lab Mimic integration, and camera observations.

### Control Scripts

- **[controll_scripts](./tools/controll_scripts)**: Control utilities for SO-ARM-101 including Leader Arm socket device, Pinocchio FK, and input device implementations.

### Standalone Scripts

- **[scripts](./scripts)**: Independent Isaac Lab experiment scripts for quick prototyping.
  - `01_basic_auto_drive.py` - Basic scene with Jetbot auto-driving
  - `02_keyboard_control.py` - WASD keyboard control with command smoothing
  - `03_domino_fpv.py` - Domino physics + first-person camera following
  - `04_trajectory_record.py` - Trajectory recording/playback, dominoes, deformable objects
  - `06_teleoperate_demo.py` - Teleoperation demonstration

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ez945y/ISAAC_LAB_REBOT_TRAINING.git
cd ISAAC_LAB_REBOT_TRAINING
```

### 2. Install Project Dependencies
```bash

# Install
pip install -e .


### 3. Quick Start - Record Demos
```bash
cd projects/so_arm_mimic

python scripts/tools/record_demos.py \
    --task Isaac-PickPlace-SOArm-Mimic-v0 \
    --teleop_device leader_arm \
    --num_demos 10 \
    --enable_cameras
```

## License

BSD-3-Clause
