# SO-ARM Mimic Extension

This extension provides environment configurations and tools for the SO-ARM 101 robot within the Isaac Lab Mimic framework.

## Installation

Follow the [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

```bash
cd projects/so_arm_mimic
```

## Directory Structure

```
so_arm_mimic/
├── datasets/                   # Demonstration datasets
├── scripts/                    # Utility scripts
│   ├── tools/                      # Utility tools
│   │   ├── record_demos.py         # Data recording
│   │   ├── replay_demos.py         # Data replay
│   │   ├── regenerate_demos.py     # Data regeneration (visual)
│   │   ├── convert_hdf5_to_lerobot.py # Format conversion
│   │   └── view_port.py            # Camera utility
│   ├── robomimic/              # RoboMimic tools
│   ├── isaaclab_mimic/         # Isaac Lab Mimic tools
├── source/                     # Source package
│   ├── envs/                   # Environment definitions
└── setup.py
```

## Usage

### 1. Recording Demonstrations

Record new demonstrations using a teleop device (e.g. Leader Arm).

```bash
python scripts/tools/record_demos.py \
    --task Isaac-PickPlace-SOArm-Camera-Mimic-v0 \
    --teleop_device leader_arm \
    --num_demos 10 \
    --enable_cameras
```

```bash
python scripts/tools/record_demos.py \
    --task Isaac-PickPlace-SOArm-Rel-Mimic-v0 \
    --teleop_device keyboard \
    --num_demos 20 \
    --enable_cameras
```

python scripts/tools/record_demos.py \
    --task PickPlace-SOArm-Camera-Mimic-v0 \
    --teleop_device leader_arm \
    --num_demos 10 \
    --enable_cameras
    
### 2. Replaying Demonstrations

Replay recorded HDF5 demonstrations to verify correctness.

```bash
python scripts/tools/replay_demos.py \
    --task Isaac-PickPlace-SOArm-Camera-Mimic-v0 \
    --dataset_file ./datasets/so_arm_demos.hdf5 \
    --enable_cameras
```

### 3. Merging Demonstrations

Merge multiple HDF5 demonstration files into a single file.

```bash
python scripts/tools/merge_hdf5_datasets.py \
--input_files datasets/dataset1.hdf5 datasets/dataset2.hdf5 datasets/dataset3.hdf5\
--output_file datasets/dataset_merged.hdf5
```

### 4. Regenerating for Visual Training

Replays actions from an existing HDF5 dataset in a new environment configuration and records new observations (e.g. images).

```bash
python scripts/tools/regenerate_demos.py \
    --task Isaac-PickPlace-SOArm-Camera-Mimic-v0 \
    --input_file ./datasets/dataset_merged.hdf5 \
    --output_file ./datasets/dataset_merged_camera.hdf5 \
    --enable_cameras
```

### 5. Deleting Episodes

Deletes episodes from an existing HDF5 dataset.

```bash
python scripts/tools/delete_episodes.py \
    --input_file ./datasets/dataset3.hdf5 \
    --output_file ./datasets/dataset_merged.hdf5 \
    -d 0
```


### 6. Annotating Demonstrations

Annotate demonstrations to add rewards and other information.

```bash
python scripts/isaaclab_mimic/annotate_demos.py \
    --device cpu --task Isaac-PickPlace-SOArm-Camera-Mimic-v0  \
    --auto --enable_cameras \
    --input_file ./datasets/dataset_merged.hdf5 \
    --output_file ./datasets/annotated_dataset.hdf5
```

### 7. Generate datasets
```bash
python scripts/isaaclab_mimic/generate_dataset.py \
    --task Isaac-PickPlace-SOArm-Camera-Mimic-v0 \
    --device cpu --enable_cameras --num_envs 10 --generation_num_trials 10 \
    --input_file ./datasets/annotated_dataset.hdf5 \
    --output_file ./datasets/generated_dataset.hdf5
```

### 8. Data Conversion (to LeRobot)

Converts Isaac Lab HDF5 demonstration files to LeRobot dataset format for training imitation learning models.

**Requirements:**
```bash
pip install h5py numpy
pip install lerobot  # Optional
```

**Usage:**
```bash
python scripts/tools/convert_hdf5_to_lerobot.py \
    --input ./datasets/so_arm_demos.hdf5 \
    --output ./lerobot_datasets/so_arm_stack \
    --robot-type so_arm \
    --fps 30
```

| Isaac Lab Key | LeRobot Key | Description |
|---------------|-------------|-------------|
| `observations/policy/joint_pos` | `observation.state` | Robot joint positions |
| `actions` | `action` | Control commands (joint positions) |
