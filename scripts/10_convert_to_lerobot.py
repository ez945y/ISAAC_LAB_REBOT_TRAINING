"""
# leisaac/scripts/convert/isaaclab2lerobotv3.py
This script converts IsaacLab HDF5 datasets into LeRobot Dataset v3 format.

Since LeRobot is evolving rapidly, compatibility with the latest LeRobot versions is not guaranteed.
Please install the following specific versions of the dependencies:

pip install lerobot==0.4.2
pip install numpy==1.26.0

"""

import argparse
import os

from isaaclab.app import AppLauncher
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert IsaacLab dataset to LeRobot Dataset v3.")
parser.add_argument("--task_name", type=str, default="Isaac-Move-SOArm-Abs-Mimic-v0", help="Name of the task.")
parser.add_argument(
    "--task_type",
    type=str,
    default=None,
    help=(
        "Specify task type. If your dataset is recorded with keyboard/gamepad, you should set it to"
        " 'keyboard'/'gamepad', otherwise not to set it and keep default value None."
    ),
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="MikeChenYZ/so101_isaac_mimic",
    help="Repository ID",
)
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="Frames per second",
)
parser.add_argument(
    "--hdf5_root",
    type=str,
    default="./datasets",
    help="HDF5 root directory",
)
parser.add_argument(
    "--hdf5_files",
    type=str,
    default="move_generated.hdf5",
    help="HDF5 files (comma-separated). If not provided, uses dataset.hdf5 in hdf5_root",
)
parser.add_argument(
    "--task_description",
    type=str,
    default="grasp the block to the platform",
    help="Task description. If not provided, will use the description defined in the task.",
)
parser.add_argument(
    "--push_to_hub",
    action="store_true",
    help="Push to hub",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# default arguments
default_args = {
    "headless": True,
    "enable_cameras": True,
}
app_launcher_args = vars(args_cli)
app_launcher_args.update(default_args)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import so_arm_mimic
import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg
# from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
# from leisaac.utils.env_utils import get_task_type
# from leisaac.utils.robot_utils import build_feature_from_env


# leisaac/source/leisaac/leisaac/enhance/datasets/lerobot_dataset_handler.py
from isaaclab.utils import configclass
from lerobot.datasets.lerobot_dataset import LeRobotDataset


@configclass
class LeRobotDatasetCfg:
    """Configuration for the LeRobotDataset."""

    repo_id: str = None
    """Lerobot Dataset repository ID."""
    fps: int = 30
    """Lerobot Dataset frames per second."""
    robot_type: str = "so101_follower"
    """Robot type: so101_follower or bi_so101_follower, etc."""
    features: dict = None
    """Features for the LeRobotDataset."""
    action_align: bool = False
    """Whether the action shape equals to the joint number. If action align, we will convert action to lerobot limit range."""


# leisaac/source/leisaac/leisaac/utils/robot_utils.py
from dataclasses import asdict, dataclass, field
from isaaclab.sensors import Camera
from isaaclab.envs import ManagerBasedEnv

@dataclass
class StateFeatureItem:
    dtype: str = "float32"
    shape: tuple = (6,)
    names: list[str] = field(
        default_factory=lambda: ["joint1.pos", "joint2.pos", "joint3.pos", "joint4.pos", "joint5.pos", "joint6.pos"]
    )

@dataclass
class VideoFeatureItem:
    dtype: str = "video"
    shape: list = field(default_factory=lambda: [480, 640, 3])  # [h, w, c]
    names: list[str] = field(default_factory=lambda: ["height", "width", "channels"])
    video_info: dict = field(
        default_factory=lambda: {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        }
    )

def build_feature_from_env(env: ManagerBasedEnv | DirectRLEnv, dataset_cfg: LeRobotDatasetCfg) -> dict:
    """
    Build the feature from the environment.
    """
    features = {}

    default_feature_joint_names = env.cfg.default_feature_joint_names
    if isinstance(env, ManagerBasedEnv):
        action_dim = env.action_manager.total_action_dim
    else:
        action_dim = env.actions.shape[-1]

    if action_dim != len(default_feature_joint_names):
        # [A bit tricky, currently works because the action dimension matches the joints only when we use leader control]
        action_joint_names = [f"dim_{index}" for index in range(action_dim)]
        dataset_cfg.action_align = False
    else:
        action_joint_names = default_feature_joint_names
        dataset_cfg.action_align = True
    features["action"] = asdict(StateFeatureItem(dtype="float32", shape=(action_dim,), names=action_joint_names))
    features["observation.state"] = asdict(
        StateFeatureItem(dtype="float32", shape=(len(default_feature_joint_names),), names=default_feature_joint_names)
    )

    for camera_key, camera_sensor in env.scene.sensors.items():
        if isinstance(camera_sensor, Camera):
            height, width = camera_sensor.image_shape
            video_feature_item = VideoFeatureItem(
                dtype="video", shape=[height, width, 3], names=["height", "width", "channels"]
            )
            video_feature_item.video_info["video.height"] = height
            video_feature_item.video_info["video.width"] = width
            video_feature_item.video_info["video.fps"] = dataset_cfg.fps
            features[f"observation.images.{camera_key}"] = asdict(video_feature_item)

    return features

# from tools.sim2real.action import normalize_joints
import numpy as np
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT_NAME = "gripper"
JOINT_LIMITS = {
    #                   (lower,  upper)
    "shoulder_pan":    (-1.8243,   1.8243),
    "shoulder_lift":   (-1.7691,   1.7691),
    "elbow_flex":      (-1.6026,   1.6026),
    "wrist_flex":      (-1.8067,   1.8067),
    "wrist_roll":      (-3.0741,   3.0741),
    "gripper":         (0.0,   1.7453),
}

_JOINT_ORDER = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
JOINT_LOWER = torch.tensor([JOINT_LIMITS[j][0] for j in _JOINT_ORDER])  # [6]
JOINT_UPPER = torch.tensor([JOINT_LIMITS[j][1] for j in _JOINT_ORDER])  # [6]

def normalize_joints(values):
    """
    Isaac Sim 弧度 → LeRobot 正規化格式
    支援 torch.Tensor 或 np.ndarray
    輸入: [..., 6] radians
    輸出: [..., 6] LeRobot 格式 (-100~100 for arm, 0~100 for gripper)
    """
    # 自動判斷輸入類型，並轉成 torch 或 numpy
    is_torch = isinstance(values, torch.Tensor)
    device = values.device if is_torch else None

    # 轉成 numpy 方便統一處理（如果原本是 torch，先 cpu().numpy()）
    if is_torch:
        values_np = values.cpu().numpy()
    else:
        values_np = values  # 已經是 numpy

    # 轉成 float32 避免精度問題
    values_np = values_np.astype(np.float32)

    # lower / upper 轉 numpy
    lower_np = JOINT_LOWER.cpu().numpy().astype(np.float32)
    upper_np = JOINT_UPPER.cpu().numpy().astype(np.float32)

    result = np.zeros_like(values_np)

    # Arm joints: radians → -100 ~ 100
    arm_range = upper_np[:5] - lower_np[:5]
    result[..., :5] = ((values_np[..., :5] - lower_np[:5]) / arm_range) * 200.0 - 100.0

    # Gripper: radians → 0 ~ 100
    gripper_range = upper_np[5:6] - lower_np[5:6]
    result[..., 5:6] = ((values_np[..., 5:6] - lower_np[5:6]) / gripper_range) * 100.0

    if is_torch:
        return torch.from_numpy(result).to(device=device, dtype=torch.float32)
    else:
        return result

# leisaac/source/leisaac/leisaac/tasks/template/single_arm_env_cfg.py
def build_lerobot_frame(episode_data: EpisodeData, dataset_cfg: LeRobotDatasetCfg) -> dict:
    obs_data = episode_data._data["obs"]
    action = obs_data["actions"][-1]
    if dataset_cfg.action_align:
        processed_action = normalize_joints(action.unsqueeze(0)).squeeze(0)
    else:
        processed_action = action.cpu().numpy()
    frame = {
        "action": processed_action,
        "observation.state": normalize_joints(obs_data["joint_pos"][-1].unsqueeze(0)).squeeze(0),
        "task": args_cli.task_description,
    }
    for frame_key in dataset_cfg.features.keys():
        if not frame_key.startswith("observation.images"):
            continue
        camera_key = frame_key.split(".")[-1]
        frame[frame_key] = obs_data[camera_key][-1].cpu().numpy()

    return frame

def split_episode(episode: EpisodeData, num_frames: int) -> list[EpisodeData]:
    def slice_at_index(data, idx: int):
        """Take the idx-th frame from the nested data structure."""
        if isinstance(data, dict):
            return {k: slice_at_index(v, idx) for k, v in data.items()}
        if isinstance(data, torch.Tensor):
            safe_idx = idx if idx < data.shape[0] else 0
            return [data[safe_idx]]
        return data

    full_data = episode.data
    sub_episodes: list[EpisodeData] = []
    for idx in range(num_frames):
        sub_episode = EpisodeData()
        sub_episode.data = slice_at_index(full_data, idx)
        sub_episodes.append(sub_episode)

    return sub_episodes


def add_episode(
    dataset: LeRobotDataset,
    episode: EpisodeData,
    env: ManagerBasedRLEnv | DirectRLEnv,
    dataset_cfg: LeRobotDatasetCfg,
    task: str,
):
    all_data = episode.data
    num_frames = all_data["actions"].shape[0]
    if num_frames < 10:
        print(f"Episode {episode.env_id} has less than 10 frames, skip it")
        return False

    episode_list = split_episode(episode, num_frames)
    # skip the first 5 frames
    for frame_index in tqdm(range(5, num_frames), desc="Processing each frame"):
        frame = build_lerobot_frame(episode_list[frame_index], dataset_cfg)
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].detach().cpu()
        if task is not None:
            frame["task"] = task
        dataset.add_frame(frame=frame)
    return True


def convert_isaaclab_to_lerobot():
    """automatically build features and dataset"""
    env_cfg = parse_env_cfg(args_cli.task_name, device=args_cli.device, num_envs=1)
    # task_type = get_task_type(args_cli.task_name, args_cli.task_type)
    task_type = "so101leader"
    # env_cfg.use_teleop_device(task_type)

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(args_cli.task_name, cfg=env_cfg).unwrapped

    dataset_cfg = LeRobotDatasetCfg(
        repo_id=args_cli.repo_id,
        fps=args_cli.fps,
        robot_type=env_cfg.robot_name,
    )
    dataset_cfg.features = build_feature_from_env(env, dataset_cfg)

    dataset = LeRobotDataset.create(
        repo_id=dataset_cfg.repo_id,
        fps=dataset_cfg.fps,
        robot_type=dataset_cfg.robot_type,
        features=dataset_cfg.features,
    )

    if args_cli.hdf5_files is None:
        hdf5_files_list = [os.path.join(args_cli.hdf5_root, "dataset.hdf5")]
    else:
        hdf5_files_list = [
            os.path.join(args_cli.hdf5_root, f.strip()) if not os.path.isabs(f.strip()) else f.strip()
            for f in args_cli.hdf5_files.split(",")
        ]

    now_episode_index = 0
    for hdf5_id, hdf5_file in enumerate(hdf5_files_list):
        print(f"[{hdf5_id+1}/{len(hdf5_files_list)}] Processing hdf5 file: {hdf5_file}")

        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(hdf5_file)

        episode_names = dataset_file_handler.get_episode_names()
        print(f"Found {len(episode_names)} episodes: {episode_names}")
        for episode_name in tqdm(episode_names, desc="Processing each episode"):
            episode = dataset_file_handler.load_episode(episode_name, device=args_cli.device)
            if not episode.success:
                print(f"Episode {episode_name} is not successful, skip it")
                continue
            valid = add_episode(dataset, episode, env, dataset_cfg, args_cli.task_description)
            if valid:
                now_episode_index += 1
                dataset.save_episode()
                print(f"Saving episode {now_episode_index} successfully")
            else:
                dataset.clear_episode_buffer()

        dataset_file_handler.close()

    dataset.finalize()

    if args_cli.push_to_hub:
        dataset.push_to_hub()

    print("Finished converting IsaacLab dataset to LeRobot dataset")
    env.close()


if __name__ == "__main__":
    convert_isaaclab_to_lerobot()