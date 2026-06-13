# Robot Tools

This directory contains utility scripts and assets for working with robot data, models, and teleoperation.

## Directory Structure

```
tools/
├── controll_scripts/           # Robot control modules and assets
│   ├── input_devices/          # Teleoperation device implementations
│   │   ├── se3_leader_arm.py   # Isaac Lab compatible leader arm device
│   │   └── leader_arm.py       # Base leader arm implementation
│   ├── so_arm_101/             # SO-ARM-101 robot assets
│   │   └── SO-ARM101.usd       # Robot USD model
│   ├── controllers/            # Robot controllers
│   └── configs/                # Configuration files
├── contoller_client/   
│   ├── teleoperate_port.py     # Leader arm sender (Mac side)
│   ├── teleop_processors.py    # Teleoperation support module
├── datasets/                   # Dataset conversion utilities
│   ├── moving_to_hdf5.py       # LeRobot joints → EE-space HDF5
│   └── convert_to_lerobot.py   # IsaacLab HDF5 → LeRobot Dataset
├── livestream/                 # WebRTC livestream helpers and diagnostics
│   ├── livestream_support.py   # Shared AppLauncher livestream defaults
│   ├── livestream_keepalive.py # Minimal streaming smoke-test scene
│   └── diag_appwindow.py       # App window / extension diagnostic
├── streaming/                  # TCP camera streaming utilities
│   ├── stream_top_sender.py    # Isaac top-camera TCP sender
│   └── stream_top_receiver.py  # OpenCV TCP receiver
└── README.md
```

## controll_scripts

The `controll_scripts` directory contains shared robot control modules and assets. 
This directory is symlinked to `isaaclab_mimic/isaaclab_mimic/controll_scripts`, allowing imports like:

```python
from isaaclab_mimic.controll_scripts.input_devices.se3_leader_arm import Se3LeaderArm
```

This ensures these assets are accessible to all projects.

## Teleoperation Client (For Mac/LeRobot Side)

Scripts in `contoller_client/` are for the workstation connected to the physical leader arm.

### teleoperate_port.py

Reads joint positions from a physical leader arm using LeRobot and sends them over network.

## Dataset Tools

```bash
python tools/datasets/moving_to_hdf5.py --dataset MikeChenYZ/soarm-fmb-v2 --output ./datasets/move_demo.hdf5
python tools/datasets/convert_to_lerobot.py --repo_id MikeChenYZ/so101_isaac_mimic_test --push_to_hub
```

## Streaming Tools

```bash
python tools/streaming/stream_top_sender.py --enable_cameras
python tools/streaming/stream_top_receiver.py
```

## Livestream Tools

`livestream/` contains support code and diagnostics used by the demos, but it is
not itself part of the numbered demo progression:

```bash
python tools/livestream/livestream_keepalive.py --livestream 2
python tools/livestream/diag_appwindow.py --livestream 2
```
