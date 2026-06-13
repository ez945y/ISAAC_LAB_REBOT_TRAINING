#!/usr/bin/env python
"""
將 LeRobot 資料集的關節位置動作，透過 FK → IK round-trip 轉換為 EE 座標，
然後儲存為 Isaac Lab Mimic 格式的 HDF5 檔案。

Pipeline per frame:
  1. Load LeRobot dataset → denormalize joints (5 arm + 1 gripper) to radians
  2. Pinocchio FK: arm joints(5) → EE pos(3) + quat(4)
  3. DifferentialIK (absolute, dls_5dof): EE pose → solved joints(5)
  4. Pinocchio FK: solved joints → verified EE pos/quat
  5. Log round-trip error
  6. Save verified EE poses as HDF5 actions: [pos(3), quat(4), gripper(1)]

Usage:
    python tools/datasets/moving_to_hdf5.py \
        --dataset MikeChenYZ/soarm-fmb-v2 \
        --episode 0 \
        --output ./datasets/move_demo.hdf5
"""

import argparse
import os
import sys
import math

import torch
import numpy as np

# ── Path setup ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SIM2REAL_DIR = os.path.join(ROOT_DIR, "tools", "sim2real")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SIM2REAL_DIR)

# ── Imports ─────────────────────────────────────────────────────
import pinocchio as pin
from tools.sim2real.action import (
    ARM_JOINT_NAMES, GRIPPER_JOINT_NAME,
    JOINT_LOWER, JOINT_UPPER,
    denormalize_joints,
)
from differential_ik import DifferentialIKController
from differential_ik_cfg import DifferentialIKControllerCfg


# ── LeRobot dataset loading ────────────────────────────────────
class EpisodeLoader:
    """Loads a LeRobot dataset once and provides fast per-episode access via cached index."""

    def __init__(self, dataset_id: str, needed_episodes: set = None, cache_dir: str = None):
        """
        Args:
            dataset_id: LeRobot dataset ID
            needed_episodes: optional set of episode indices we care about;
                             only missing episodes will be scanned.
            cache_dir: directory to store the episode_index_cache.json
        """
        import json as _json
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        print(f"[INFO] 載入 LeRobot dataset: {dataset_id} ...")
        self.dataset = LeRobotDataset(dataset_id)
        self._json = _json

        # ── 1. Determine cache path ──
        if cache_dir is None:
            cache_dir = SCRIPT_DIR
        self._cache_path = os.path.join(cache_dir, "episode_index_cache.json")

        # ── 2. Try loading cached index ──
        self._index = {}
        cache_hit = False
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, "r") as f:
                    cache = _json.load(f)
                if cache.get("dataset_id") == dataset_id:
                    # Restore index (JSON keys are strings, convert back to int)
                    self._index = {int(k): tuple(v) for k, v in cache.get("episodes", {}).items()}
                    cached_eps = set(self._index.keys())
                    print(f"[INFO] 從快取載入索引: {self._cache_path} (已有 {len(cached_eps)} 個 episodes)")
                    cache_hit = True
                else:
                    print(f"[INFO] 快取 dataset_id 不匹配，重建索引")
            except Exception as e:
                print(f"[WARNING] 無法讀取快取: {e}")

        # ── 3. Determine which episodes still need scanning ──
        if needed_episodes is not None:
            missing = needed_episodes - set(self._index.keys())
        else:
            missing = None  # scan all

        if cache_hit and needed_episodes is not None and not missing:
            print(f"[INFO] 所有需要的 episodes 都已在快取中，跳過掃描")
        else:
            # Need to scan (either no cache, or missing episodes)
            self._scan_episodes(missing)
            # Save updated cache
            self._save_cache(dataset_id)

        print(f"[INFO] 索引就緒，共 {len(self._index)} 個 episodes")

    def _scan_episodes(self, needed_set: set = None):
        """Scan dataset to build index. Stores ALL discovered episodes into self._index."""
        # Try fast path via episode_data_index first
        if hasattr(self.dataset, 'episode_data_index'):
            ep_data_idx = self.dataset.episode_data_index
            from_indices = ep_data_idx["from"].tolist()
            to_indices = ep_data_idx["to"].tolist()
            for ep_i, (start, end) in enumerate(zip(from_indices, to_indices)):
                self._index[ep_i] = (start, end)
            return

        # Fallback: sequential scan
        total = len(self.dataset)
        max_needed = max(needed_set) if needed_set else None
        print(f"[INFO] 掃描建立索引 (共 {total} 幀)...")

        current_ep = None
        start = 0
        for i in range(total):
            ep_idx = self.dataset[i]["episode_index"].item()
            if ep_idx != current_ep:
                if current_ep is not None:
                    # Store EVERY discovered episode
                    self._index[current_ep] = (start, i)
                    # Early exit: stop scanning if we've passed the last needed episode
                    if max_needed is not None and current_ep >= max_needed:
                        print(f"\r[INFO] 掃描進度: {i}/{total} — 已找齊所需要的 episodes，提前結束")
                        break
                current_ep = ep_idx
                start = i
            if i % 1000 == 0:
                print(f"\r[INFO] 掃描進度: {i}/{total} 幀, 已找到 {len(self._index)} 個 episodes", end="", flush=True)
        else:
            # Loop finished without break — record the last episode
            if current_ep is not None:
                self._index[current_ep] = (start, total)
        print()

    def _save_cache(self, dataset_id: str):
        """Save episode index to JSON cache file."""
        cache = {
            "dataset_id": dataset_id,
            "episodes": {str(k): list(v) for k, v in sorted(self._index.items())},
        }
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        with open(self._cache_path, "w") as f:
            self._json.dump(cache, f, indent=2)
        print(f"[INFO] 索引已更新並存入快取: {self._cache_path}")

    @property
    def episode_indices(self) -> list:
        return sorted(self._index.keys())

    def load(self, episode_idx: int):
        """快速讀取指定 episode 的 action、observation.state、timestamp"""
        if episode_idx not in self._index:
            raise ValueError(f"Episode {episode_idx} 不存在於資料集中 (可用: {self.episode_indices})")

        start, end = self._index[episode_idx]
        actions = []
        obs_states = []
        timestamps = []
        for i in range(start, end):
            sample = self.dataset[i]
            actions.append(sample["action"])
            obs_states.append(sample["observation.state"])
            timestamps.append(sample["timestamp"].item())

        actions_tensor = torch.stack(actions)       # [N, 6]
        obs_states_tensor = torch.stack(obs_states) # [N, 6]
        print(f"[INFO] 載入 episode {episode_idx}，共 {len(actions_tensor)} 幀 (index {start}:{end})")
        return actions_tensor, obs_states_tensor, timestamps


# ── Pinocchio FK helper ─────────────────────────────────────────
class PinocchioFK:
    """Pinocchio-based forward kinematics for SO-ARM-101 (5-DOF reduced model)."""

    def __init__(self, urdf_path: str):
        full_model = pin.buildModelFromUrdf(urdf_path)
        controlled_joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']

        all_joint_names = [full_model.names[i] for i in range(1, full_model.njoints)]
        joints_to_lock = [
            full_model.getJointId(name)
            for name in all_joint_names
            if name not in controlled_joints
        ]

        q_reference = pin.neutral(full_model)
        self.model = pin.buildReducedModel(full_model, joints_to_lock, q_reference)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId("gripper_link")

    def compute(self, q_rad_np: np.ndarray):
        """
        Args:
            q_rad_np: (5,) numpy array, arm joint positions in radians

        Returns:
            ee_pos: (3,) numpy array
            ee_quat_wxyz: (4,) numpy array [w, x, y, z]
            jacobian: (6, 5) numpy array
        """
        pin.forwardKinematics(self.model, self.data, q_rad_np)
        pin.updateFramePlacements(self.model, self.data)

        oMf = self.data.oMf[self.ee_frame_id]
        ee_pos = oMf.translation.copy()
        q_pin = pin.Quaternion(oMf.rotation)
        ee_quat_wxyz = np.array([q_pin.w, q_pin.x, q_pin.y, q_pin.z])

        jac = pin.computeFrameJacobian(
            self.model, self.data, q_rad_np,
            self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        return ee_pos, ee_quat_wxyz, jac


# ── FK → IK round-trip conversion ──────────────────────────────
def convert_episode(
    actions_norm: torch.Tensor,
    obs_states_norm: torch.Tensor,
    fk: PinocchioFK,
    device: str = "cpu",
):
    """
    Convert a full episode from joint-space to EE-space with FK→IK round-trip verification.

    Args:
        actions_norm: [N, 6] normalized actions from LeRobot dataset
        obs_states_norm: [N, 6] normalized observations from LeRobot dataset
        fk: PinocchioFK instance
        device: torch device string

    Returns:
        ee_actions: [N, 8] tensor of [pos(3), quat(4), gripper(1)]
        errors: [N] tensor of position errors (meters) from round-trip
    """
    # Denormalize to radians
    actions_rad = denormalize_joints(actions_norm)   # [N, 6]
    obs_rad = denormalize_joints(obs_states_norm)     # [N, 6]

    num_frames = actions_rad.shape[0]

    # Setup IK controller (absolute mode, dls_5dof)
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls_5dof",
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=device)

    ee_actions = torch.zeros(num_frames, 8, device=device)  # [pos(3), quat(4), gripper(1)]
    errors = torch.zeros(num_frames, device=device)

    for i in range(num_frames):
        arm_joints_rad = actions_rad[i, :5].numpy().astype(np.float64)
        gripper_rad = actions_rad[i, 5].item()

        # ── Step 1: FK → EE pose ──
        ee_pos_np, ee_quat_np, jac_np = fk.compute(arm_joints_rad)

        ee_pos = torch.from_numpy(ee_pos_np).to(device).float().unsqueeze(0)       # [1, 3]
        ee_quat = torch.from_numpy(ee_quat_np).to(device).float().unsqueeze(0)     # [1, 4] wxyz
        jacobian = torch.from_numpy(jac_np).to(device).float().unsqueeze(0)         # [1, 6, 5]
        q_curr = torch.from_numpy(arm_joints_rad).to(device).float().unsqueeze(0)  # [1, 5]

        # ── Step 2: IK → solved joints ──
        # Set absolute EE pose as command [pos(3), quat(4)]
        command = torch.cat([ee_pos, ee_quat], dim=1)  # [1, 7]
        ik_controller.set_command(command, ee_pos, ee_quat)
        solved_joints = ik_controller.compute(ee_pos, ee_quat, jacobian, q_curr)  # [1, 5]

        # ── Step 3: FK on solved joints → verified EE ──
        solved_np = solved_joints.squeeze(0).detach().cpu().numpy().astype(np.float64)
        ee_pos_v, ee_quat_v, _ = fk.compute(solved_np)

        # ── Step 4: Compute round-trip error ──
        pos_error = np.linalg.norm(ee_pos_np - ee_pos_v)
        errors[i] = pos_error

        # ── Step 5: Store verified EE pose + gripper ──
        # Use the IK-verified EE pose (closest reachable)
        ee_actions[i, :3] = torch.from_numpy(ee_pos_v).float()
        ee_actions[i, 3:7] = torch.from_numpy(
            np.array([
                pin.Quaternion(fk.data.oMf[fk.ee_frame_id].rotation).w,
                pin.Quaternion(fk.data.oMf[fk.ee_frame_id].rotation).x,
                pin.Quaternion(fk.data.oMf[fk.ee_frame_id].rotation).y,
                pin.Quaternion(fk.data.oMf[fk.ee_frame_id].rotation).z,
            ])
        ).float()
        ee_actions[i, 7] = gripper_rad

    return ee_actions, errors


# ── Build initial_state for HDF5 ────────────────────────────────
def build_initial_state(
    first_action_rad: torch.Tensor,
    cube_pos: list = None,
    cube_quat: list = None,
    platform_pos: list = None,
    platform_quat: list = None,
):
    """
    Build the initial_state dict matching Isaac Lab's scene state format.

    Args:
        first_action_rad: [6] tensor (5 arm + 1 gripper) in radians
        cube_pos: [3] list, cube initial position
        cube_quat: [4] list, cube initial quaternion [w, x, y, z]
        platform_pos: [3] list, platform block initial position
        platform_quat: [4] list, platform block initial quaternion [w, x, y, z]

    Returns:
        dict: initial state with articulation + rigid_object entries
    """
    if cube_pos is None:
        # cube_pos = [0.36, -0.02, 0.0]
        cube_pos = [0.38, -0.04, 0.0]
    if cube_quat is None:
        cube_quat = [1.0, 0.0, 0.0, 0.0]
    if platform_pos is None:
        platform_pos = [0.38, 0.23, 0.0]
    if platform_quat is None:
        platform_quat = [1.0, 0.0, 0.0, 0.0]

    # Robot joint positions (all 6 joints)
    joint_pos = first_action_rad.clone()  # [6]
    joint_vel = torch.zeros_like(joint_pos)  # [6]

    # Build the state dict matching Isaac Lab format
    # Keys must match interactive_scene.get_state() / reset_to():
    #   articulation: root_pose, root_velocity, joint_position, joint_velocity
    #   rigid_object: root_pose, root_velocity
    initial_state = {
        "articulation": {
            "robot": {
                "root_pose": torch.tensor([[0.0, -0.01, 0.05, 1.0, 0.0, 0.0, 0.0]]),  # [1, 7] pos+quat
                "root_velocity": torch.zeros(1, 6),           # [1, 6]
                "joint_position": joint_pos.unsqueeze(0),     # [1, 6]
                "joint_velocity": joint_vel.unsqueeze(0),     # [1, 6]
            }
        },
        "rigid_object": {
            "cube_1": {
                "root_pose": torch.tensor([cube_pos + cube_quat]).float(),  # [1, 7]
                "root_velocity": torch.zeros(1, 6),
            },
            "platform_block": {
                "root_pose": torch.tensor([platform_pos + platform_quat]).float(),  # [1, 7]
                "root_velocity": torch.zeros(1, 6),
            },
        },
    }

    return initial_state


# ── Write HDF5 ──────────────────────────────────────────────────
def write_hdf5(
    output_path: str,
    ee_actions: torch.Tensor,
    initial_state: dict,
    env_name: str = "Isaac-Move-SOArm-Abs-Mimic-v0",
    demo_id: int = 0,
    append: bool = False,
):
    """
    Write episode to HDF5 in Isaac Lab Mimic format using pure h5py (no SimulationApp needed).

    Args:
        output_path: path to output .hdf5 file
        ee_actions: [N, 8] tensor of [pos(3), quat(4), gripper(1)]
        initial_state: dict with initial scene state
        env_name: gym environment ID
        demo_id: demo index (for multi-episode files)
        append: if True, append to existing file
    """
    import h5py
    import json

    mode = "a" if append else "w"
    f = h5py.File(output_path, mode)

    # ── Top-level metadata (matching HDF5DatasetFileHandler.create format) ──
    if "data" not in f:
        data_grp = f.create_group("data")
        # env_args must be a JSON string with "env_name" and "type" keys
        # type=2 is gym environment type (robomimic compatible)
        data_grp.attrs["env_args"] = json.dumps({"env_name": env_name, "type": 2})
        data_grp.attrs["total"] = 0
    else:
        data_grp = f["data"]

    # ── Demo group ──
    demo_name = f"demo_{demo_id}"
    if demo_name in data_grp:
        del data_grp[demo_name]

    demo_grp = data_grp.create_group(demo_name)

    # actions: [N, 8]
    actions_np = ee_actions.cpu().numpy()
    demo_grp.create_dataset("actions", data=actions_np, compression="gzip")

    # success flag + metadata
    demo_grp.attrs["success"] = True
    demo_grp.attrs["num_samples"] = actions_np.shape[0]

    # ── initial_state (recursive dict → HDF5 groups) ──
    def write_dict_to_group(grp, d):
        for key, value in d.items():
            if isinstance(value, dict):
                sub_grp = grp.create_group(key)
                write_dict_to_group(sub_grp, value)
            elif isinstance(value, torch.Tensor):
                grp.create_dataset(key, data=value.cpu().numpy(), compression="gzip")
            elif isinstance(value, np.ndarray):
                grp.create_dataset(key, data=value, compression="gzip")

    init_grp = demo_grp.create_group("initial_state")
    write_dict_to_group(init_grp, initial_state)

    # Update total count
    data_grp.attrs["total"] = len([k for k in data_grp.keys() if k.startswith("demo_")])

    f.flush()
    f.close()

    print(f"[INFO] HDF5 已儲存: {output_path}")
    print(f"  - demo: {demo_name}")
    print(f"  - actions shape: {actions_np.shape}")
    print(f"  - env_name: {env_name}")


# ── Per-episode configurations ──────────────────────────────────
# Map episode index → cube initial state.
# Fill in entries below, then run with --use_configs to generate
# a multi-demo HDF5 from these configurations.
#
# Keys:   LeRobot dataset episode index (int)
# Values: dict with:
#   cube_pos:   [x, y, z]     initial cube position
#   cube_quat:  [w, x, y, z]  initial cube quaternion
#
# EPISODE_CONFIGS = {
#     2: {"cube_pos": [0.38, -0.04, 0.0], "cube_quat": [1.0, 0.0, 0.0, 0.0]},
#     3: {"cube_pos": [0.38, -0.04, 0.0], "cube_quat": [1.0, 0.0, 0.0, 0.0]},
#     8: {"cube_pos": [0.38, -0.04, 0.0], "cube_quat": [1.0, 0.0, 0.0, 0.0]},
#     9: {"cube_pos": [0.38, -0.04, 0.0], "cube_quat": [1.0, 0.0, 0.0, 0.0]},
#     12: {"cube_pos": [0.28, -0.04, 0.0], "cube_quat": [1.0, 0.0, 0.0, 0.0]},
#     13: {"cube_pos": [0.28, -0.04, 0.0], "cube_quat": [1.0, 0.0, 0.0, 0.0]},
#     18: {"cube_pos": [0.28, -0.04, 0.0], "cube_quat": [1.0, 0.0, 0.0, 0.0]},
#     19: {"cube_pos": [0.28, -0.04, 0.0], "cube_quat": [1.0, 0.0, 0.0, 0.0]},
# }

EPISODE_CONFIGS = {
    2: {"cube_pos": [0.38, -0.028, 0.0], "cube_quat": [1.0, 0.0, 0.0, 0.0]},
    3: {"cube_pos": [0.38, -0.028, 0.0], "cube_quat": [1.0, 0.0, 0.0, 0.0]},
    12: {"cube_pos": [0.28, -0.035, 0.0], "cube_quat": [1.0, 0.0, 0.0, 0.0]},
    13: {"cube_pos": [0.28, -0.035, 0.0], "cube_quat": [1.0, 0.0, 0.0, 0.0]},
    22: {"cube_pos": [0.18, -0.05, 0.0], "cube_quat": [1.0, 0.0, 0.0, 0.0]},
    23: {"cube_pos": [0.18, -0.05, 0.0], "cube_quat": [1.0, 0.0, 0.0, 0.0]},
    32: {"cube_pos": [0.36, -0.07, 0.0], "cube_quat": [0.7071, 0.0, 0.0, -0.7071]},
    33: {"cube_pos": [0.36, -0.07, 0.0], "cube_quat": [0.7071, 0.0, 0.0, -0.7071]},
    42: {"cube_pos": [0.35, -0.15, 0.0], "cube_quat": [0.7071, 0.0, 0.0, -0.7071]},
    43: {"cube_pos": [0.35, -0.15, 0.0], "cube_quat": [0.7071, 0.0, 0.0, -0.7071]},
    52: {"cube_pos": [0.35, -0.22, 0.0], "cube_quat": [0.7071, 0.0, 0.0, -0.7071]},
}

# ── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot joint-space dataset to EE-space HDF5 via FK→IK round-trip"
    )
    parser.add_argument("--dataset", type=str, default="MikeChenYZ/soarm-fmb-v2",
                        help="LeRobot dataset ID")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to convert (ignored when --use_configs)")
    parser.add_argument("--output", type=str, default="./datasets/move_demo.hdf5",
                        help="Output HDF5 file path")
    parser.add_argument("--urdf", type=str,
                        default=os.path.join(SIM2REAL_DIR, "so101_new_calib.urdf"),
                        help="URDF path for Pinocchio FK")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")
    parser.add_argument("--cube_pos", type=float, nargs=3,
                        default=[0.38, -0.04, 0.0],
                        help="Initial cube position (ignored when --use_configs)")
    parser.add_argument("--cube_quat", type=float, nargs=4,
                        default=[1.0, 0.0, 0.0, 0.0],
                        help="Initial cube quaternion (ignored when --use_configs)")
    parser.add_argument("--all_episodes", action="store_true", default=False,
                        help="Convert all episodes (ignored when --use_configs)")
    parser.add_argument("--use_configs", action="store_true", default=True,
                        help="Use EPISODE_CONFIGS dict instead of CLI args")
    args = parser.parse_args()

    # Load URDF and build FK model
    print(f"[INFO] Loading URDF: {args.urdf}")
    fk = PinocchioFK(args.urdf)
    print(f"[INFO] Pinocchio model: {fk.model.nq} DOF, EE frame: {fk.ee_frame_id}")

    # ── Load LeRobot dataset once ──
    needed = set(EPISODE_CONFIGS.keys()) if args.use_configs else None
    cache_dir = os.path.dirname(os.path.abspath(args.output))
    loader = EpisodeLoader(args.dataset, needed_episodes=needed, cache_dir=cache_dir)

    # ── Determine which episodes to process and their cube configs ──
    if args.use_configs:
        # Use the EPISODE_CONFIGS dict
        if not EPISODE_CONFIGS:
            raise ValueError("EPISODE_CONFIGS is empty. Please fill in episode entries.")
        episode_configs = EPISODE_CONFIGS
        print(f"[INFO] 使用 EPISODE_CONFIGS，共 {len(episode_configs)} 筆:")
        for ep_idx, cfg in episode_configs.items():
            print(f"  Episode {ep_idx}: pos={cfg['cube_pos']}, quat={cfg['cube_quat']}")
    else:
        # Legacy mode: single episode or all episodes with shared cube config
        if args.all_episodes:
            episode_indices = loader.episode_indices
            print(f"[INFO] 找到 {len(episode_indices)} 個 episodes: {episode_indices}")
        else:
            episode_indices = [args.episode]

        # Build configs with shared cube_pos/cube_quat
        episode_configs = {
            ep_idx: {
                "cube_pos": args.cube_pos,
                "cube_quat": args.cube_quat,
            }
            for ep_idx in episode_indices
        }

    # ── Process each episode ──
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    first_written = True

    for demo_id, (ep_idx, cfg) in enumerate(episode_configs.items()):
        print(f"\n{'='*60}")
        print(f"[INFO] 處理 Episode {ep_idx} (demo_{demo_id})")
        print(f"{'='*60}")

        # Load episode (fast — uses pre-built index)
        actions, obs_states, timestamps = loader.load(ep_idx)

        # FK → IK round-trip conversion
        ee_actions, errors = convert_episode(actions, obs_states, fk, device=args.device)

        # Report errors
        mean_err = errors.mean().item()
        max_err = errors.max().item()
        max_err_frame = errors.argmax().item()
        print(f"\n[REPORT] Episode {ep_idx} FK→IK round-trip errors:")
        print(f"  Mean: {mean_err:.6f} m")
        print(f"  Max:  {max_err:.6f} m (frame {max_err_frame})")

        if max_err > 0.01:
            print(f"  [WARNING] Max error > 1cm — some frames may be near singularity")

        # Build initial state with per-episode cube config
        first_action_rad = denormalize_joints(actions[0:1]).squeeze(0)  # [6]
        initial_state = build_initial_state(
            first_action_rad,
            cube_pos=cfg["cube_pos"],
            cube_quat=cfg["cube_quat"],
        )

        # Write HDF5 (append after first demo)
        write_hdf5(
            output_path, ee_actions, initial_state,
            demo_id=demo_id,
            append=(not first_written),
        )
        first_written = False

    print(f"\n[DONE] 轉換完成！共 {len(episode_configs)} 筆 demo 寫入 {output_path}")


if __name__ == "__main__":
    main()
