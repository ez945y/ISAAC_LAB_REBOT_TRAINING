# Imitation Learning — SO-ARM Mimic

SO-ARM-101 cube pick-and-place via **imitation learning** using Isaac Lab Mimic framework.

## Tasks

| Task ID | Description |
|---------|-------------|
| `Isaac-PickPlace-SOArm-Abs-Mimic-v0` | Stack 3 cubes — absolute IK control |
| `Isaac-PickPlace-SOArm-Rel-Mimic-v0` | Stack 3 cubes — relative IK control |
| `Isaac-Move-SOArm-Abs-Mimic-v0` | Move 1 cube right→left — absolute IK control |

## Pipeline

```
1. Record demos  →  2. Merge  →  3. Annotate  →  4. Generate augmented data  →  5. Train
        ↓                                                                          ↓
   (Leader Arm                                                              (RoboMimic /
    or Keyboard)                                                             LeRobot)
```

### Move Cube Pipeline (from LeRobot dataset)

```
LeRobot joints → 09_moving_to_hdf5.py (FK→IK) → HDF5 → Replay / Annotate / Generate
```

## Usage

```bash
cd projects/so_arm_mimic
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
```

### 1. Record Demonstrations

```bash
python scripts/tools/record_demos.py \
    --task Isaac-PickPlace-SOArm-Abs-Mimic-v0 \
    --teleop_device leader_arm \
    --num_demos 10 --enable_cameras
```

### 2. Convert LeRobot Dataset to EE-Space HDF5

```bash
# Single episode
python scripts/09_moving_to_hdf5.py \
    --dataset MikeChenYZ/soarm-fmb-v2 --episode 0 \
    --output ./datasets/move_demo.hdf5

# All episodes
python scripts/09_moving_to_hdf5.py \
    --dataset MikeChenYZ/soarm-fmb-v2 --all_episodes \
    --output ./datasets/move_demo.hdf5
```

### 3. Replay Demonstrations

```bash
# Move task
python projects/so_arm_mimic/scripts/tools/replay_demos.py \
    --task Isaac-Move-SOArm-Abs-Mimic-v0 \
    --dataset_file ./datasets/move_demo.hdf5 --enable_cameras
```

```bash
# Evaluation on generated data
python projects/so_arm_mimic/scripts/tools/replay_demos.py \
    --task Isaac-Move-SOArm-Abs-Mimic-v0 \
    --dataset_file ./datasets/move_generated.hdf5 --enable_cameras
```

### 4. Merge Datasets

```bash
python scripts/tools/merge_hdf5_datasets.py \
    --input_files datasets/dataset1.hdf5 datasets/dataset2.hdf5 \
    --output_file datasets/dataset_merged.hdf5
```


### 5. Annotate

```bash
python projects/so_arm_mimic/scripts/isaaclab_mimic/annotate_demos.py \
    --device cpu --task Isaac-Move-SOArm-Abs-Mimic-v0 \
    --input_file ./datasets/move_demo.hdf5 \
    --output_file ./datasets/move_annotated.hdf5
```

### 6. Generate Augmented Dataset

```bash
nice -n 10 python projects/so_arm_mimic/scripts/isaaclab_mimic/generate_dataset.py \
    --task Isaac-Move-SOArm-Abs-Mimic-v0 \
    --device cpu --num_envs 5 --generation_num_trials 1 \
    --input_file ./datasets/move_annotated.hdf5 \
    --output_file ./datasets/move_generated.hdf5 --enable_cameras --headless
```

In case starting from "--num_envs 5 --generation_num_trials 1" if its ok then try bigger


### Utility: Delete Episodes

```bash
python scripts/tools/delete_episodes.py \
    --input_file ./datasets/dataset.hdf5 \
    --output_file ./datasets/dataset_clean.hdf5 -d 0
```
### Utility: Regenerate cover Observations（optional for experiments）

```bash
python scripts/tools/regenerate_demos.py \
    --task Isaac-PickPlace-SOArm-Abs-Mimic-v0 \
    --input_file ./datasets/dataset_merged.hdf5 \
    --output_file ./datasets/dataset_merged_camera.hdf5
```

## Structure

```
so_arm_mimic/
├── datasets/                       # Demonstration datasets
├── scripts/
│   ├── tools/                      # Record, replay, merge, convert
│   ├── robomimic/                  # RoboMimic training tools
│   └── isaaclab_mimic/             # Annotate & generate
└── source/
    ├── envs/                       # Environment definitions
    └── mdp/                        # MDP: observations, terminations
```
