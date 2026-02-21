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
    python scripts/09_moving_to_hdf5.py \
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
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
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
def load_episode_data(dataset_id: str, episode_idx: int):
    """讀取指定 episode 的 action、observation.state、timestamp"""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(dataset_id)
    actions = []
    obs_states = []
    timestamps = []
    for i in range(len(dataset)):
        sample = dataset[i]
        ep_idx = sample["episode_index"].item()
        if ep_idx < episode_idx:
            continue
        if ep_idx > episode_idx:
            break
        actions.append(sample["action"])
        obs_states.append(sample["observation.state"])
        timestamps.append(sample["timestamp"].item())

    if not actions:
        raise ValueError(f"Episode {episode_idx} 不存在於資料集中")

    actions_tensor = torch.stack(actions)       # [N, 6]
    obs_states_tensor = torch.stack(obs_states) # [N, 6]
    print(f"[INFO] 載入 episode {episode_idx}，共 {len(actions_tensor)} 幀")
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

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Frame {i+1:>4d}/{num_frames}: "
                  f"pos=[{ee_pos_v[0]:.4f}, {ee_pos_v[1]:.4f}, {ee_pos_v[2]:.4f}] "
                  f"error={pos_error:.6f}m")

    return ee_actions, errors


# ── Build initial_state for HDF5 ────────────────────────────────
def build_initial_state(
    first_action_rad: torch.Tensor,
    cube_pos: list = None,
    cube_quat: list = None,
):
    """
    Build the initial_state dict matching Isaac Lab's scene state format.

    Args:
        first_action_rad: [6] tensor (5 arm + 1 gripper) in radians
        cube_pos: [3] list, cube initial position
        cube_quat: [4] list, cube initial quaternion [w, x, y, z]

    Returns:
        dict: initial state with articulation + rigid_object entries
    """
    if cube_pos is None:
        # cube_pos = [0.36, -0.02, 0.0]
        cube_pos = [0.38, -0.04, 0.0]
    if cube_quat is None:
        # cube_quat = [0.707, 0.0, 0.0, 0.707]
        cube_quat = [1.0, 0.0, 0.0, 0.0]

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
                "root_pose": torch.tensor([[0.01, -0.005, 0.05, 1.0, 0.0, 0.0, 0.0]]),  # [1, 7] pos+quat
                "root_velocity": torch.zeros(1, 6),           # [1, 6]
                "joint_position": joint_pos.unsqueeze(0),     # [1, 6]
                "joint_velocity": joint_vel.unsqueeze(0),     # [1, 6]
            }
        },
        "rigid_object": {
            "cube_1": {
                "root_pose": torch.tensor([cube_pos + cube_quat]).float(),  # [1, 7]
                "root_velocity": torch.zeros(1, 6),
            }
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

    # success flag
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


# ── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot joint-space dataset to EE-space HDF5 via FK→IK round-trip"
    )
    parser.add_argument("--dataset", type=str, default="MikeChenYZ/soarm-fmb-v2",
                        help="LeRobot dataset ID")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to convert")
    parser.add_argument("--output", type=str, default="./datasets/move_demo.hdf5",
                        help="Output HDF5 file path")
    parser.add_argument("--urdf", type=str,
                        default=os.path.join(SIM2REAL_DIR, "so101_new_calib.urdf"),
                        help="URDF path for Pinocchio FK")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")
    parser.add_argument("--cube_pos", type=float, nargs=3,
                        default=[0.36, -0.02, 0.0],
                        help="Initial cube position [x, y, z]")
    parser.add_argument("--cube_quat", type=float, nargs=4,
                        default=[0.707, 0.0, 0.0, 0.707],
                        help="Initial cube quaternion [w, x, y, z]")
    parser.add_argument("--all_episodes", action="store_true", default=False,
                        help="Convert all episodes in the dataset")
    args = parser.parse_args()

    # Load URDF and build FK model
    print(f"[INFO] Loading URDF: {args.urdf}")
    fk = PinocchioFK(args.urdf)
    print(f"[INFO] Pinocchio model: {fk.model.nq} DOF, EE frame: {fk.ee_frame_id}")

    if args.all_episodes:
        # Discover all episodes
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(args.dataset)
        episode_indices = sorted(set(
            dataset[i]["episode_index"].item() for i in range(len(dataset))
        ))
        print(f"[INFO] 找到 {len(episode_indices)} 個 episodes: {episode_indices}")
    else:
        episode_indices = [args.episode]

    for ep_idx in episode_indices:
        print(f"\n{'='*60}")
        print(f"[INFO] 處理 Episode {ep_idx}")
        print(f"{'='*60}")

        # Load episode
        actions, obs_states, timestamps = load_episode_data(args.dataset, ep_idx)

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

        # Build initial state
        first_action_rad = denormalize_joints(actions[0:1]).squeeze(0)  # [6]
        initial_state = build_initial_state(
            first_action_rad,
            cube_pos=args.cube_pos,
            cube_quat=args.cube_quat,
        )

        # Determine output path — all episodes go to single file in --all_episodes mode
        output_path = args.output

        # Write HDF5
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        is_first = (ep_idx == episode_indices[0])
        write_hdf5(
            output_path, ee_actions, initial_state,
            demo_id=ep_idx,
            append=(not is_first),
        )

    print(f"\n[DONE] 轉換完成！")


if __name__ == "__main__":
    main()
