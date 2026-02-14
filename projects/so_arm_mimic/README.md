# Imitation Learning — SO-ARM Mimic

SO-ARM-101 cube pick-and-place via **imitation learning** using Isaac Lab Mimic framework.

## Task

| Task ID | Robot | Description |
|---------|-------|-------------|
| `Isaac-PickPlace-SOArm-Abs-Mimic-v0` | SO-ARM-101 | Absolute joint control |
| `Isaac-PickPlace-SOArm-Rel-Mimic-v0` | SO-ARM-101 | Relative joint control |

## Pipeline

```
1. Record demos  →  2. Merge  →  3. Annotate  →  4. Generate augmented data  →  5. Train
        ↓                                                                          ↓
   (Leader Arm                                                              (RoboMimic /
    or Keyboard)                                                             LeRobot)
```

## Usage

```bash
cd projects/so_arm_mimic
```

### 1. Record Demonstrations

```bash
python scripts/tools/record_demos.py \
    --task Isaac-PickPlace-SOArm-Abs-Mimic-v0 \
    --teleop_device leader_arm \
    --num_demos 10 --enable_cameras
```

### 2. Replay Demonstrations

```bash
python scripts/tools/replay_demos.py \
    --task Isaac-PickPlace-SOArm-Abs-Mimic-v0 \
    --dataset_file ./datasets/so_arm_demos.hdf5 --enable_cameras
```

### 3. Merge Datasets

```bash
python scripts/tools/merge_hdf5_datasets.py \
    --input_files datasets/dataset1.hdf5 datasets/dataset2.hdf5 \
    --output_file datasets/dataset_merged.hdf5
```

### 4. Regenerate with Camera Observations

```bash
python scripts/tools/regenerate_demos.py \
    --task Isaac-PickPlace-SOArm-Abs-Mimic-v0 \
    --input_file ./datasets/dataset_merged.hdf5 \
    --output_file ./datasets/dataset_merged_camera.hdf5
```

### 5. Annotate

```bash
python scripts/isaaclab_mimic/annotate_demos.py \
    --device cpu --task Isaac-PickPlace-SOArm-Abs-Mimic-v0 \
    --input_file ./datasets/dataset_merged.hdf5 \
    --output_file ./datasets/annotated_dataset.hdf5
```

### 6. Generate Augmented Dataset

```bash
python scripts/isaaclab_mimic/generate_dataset.py \
    --task Isaac-PickPlace-SOArm-Abs-Mimic-v0 \
    --device cpu --num_envs 10 --generation_num_trials 10 \
    --input_file ./datasets/annotated_dataset.hdf5 \
    --output_file ./datasets/generated_dataset.hdf5
```

### 7. Convert to LeRobot Format

```bash
python scripts/tools/convert_hdf5_to_lerobot.py \
    --input ./datasets/so_arm_demos.hdf5 \
    --output ./lerobot_datasets/so_arm_stack \
    --robot-type so_arm --fps 30
```

| Isaac Lab Key | LeRobot Key | Description |
|---------------|-------------|-------------|
| `observations/policy/joint_pos` | `observation.state` | Joint positions |
| `actions` | `action` | Control commands |

### Utility: Delete Episodes

```bash
python scripts/tools/delete_episodes.py \
    --input_file ./datasets/dataset.hdf5 \
    --output_file ./datasets/dataset_clean.hdf5 -d 0
```

## Structure

```
so_arm_mimic/
├── datasets/                       # Demonstration datasets
├── scripts/
│   ├── tools/                      # Record, replay, merge, convert
│   ├── robomimic/                  # RoboMimic training tools
│   └── isaaclab_mimic/             # Annotate & generate
└── source/                         # Environment definitions
```
