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
import sys

from isaaclab.app import AppLauncher
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import numpy as np

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
SIM2REAL_DIR = os.path.join(ROOT_DIR, "tools", "sim2real")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SIM2REAL_DIR)

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
from tools.sim2real.action import normalize_joints
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
    action_from_ee: bool = True 
    """New added: Whether the action is from EE pose."""



# leisaac/source/leisaac/leisaac/assets/robots/lerobot.py
SO101_FOLLOWER_USD_JOINT_LIMLITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10, 100.0),
}

SO101_FOLLOWER_MOTOR_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}


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


def convert_leisaac_action_to_lerobot(action: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert the action from LeIsaac to Lerobot. Just convert value, not include the format.
    """
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    processed_action = np.zeros_like(action)
    joint_limits = SO101_FOLLOWER_USD_JOINT_LIMLITS
    motor_limits = SO101_FOLLOWER_MOTOR_LIMITS
    action = action / torch.pi * 180.0  # convert to degree

    for idx, joint_name in enumerate(joint_limits):
        motor_limit_range = motor_limits[joint_name]
        joint_limit_range = joint_limits[joint_name]
        joint_range = joint_limit_range[1] - joint_limit_range[0]
        motor_range = motor_limit_range[1] - motor_limit_range[0]
        joint_degree = action[:, idx] - joint_limit_range[0]
        processed_action[:, idx] = joint_degree / joint_range * motor_range + motor_limit_range[0]

    return processed_action


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

    joint_dim = len(default_feature_joint_names)  # 6

    if action_dim == joint_dim:
        dataset_cfg.action_align = True
        dataset_cfg.action_from_ee = False
    else:
        # EE pose (8維) → 需要 IK 轉換，但 feature 仍輸出 6 維
        dataset_cfg.action_align = False

    features["action"] = asdict(StateFeatureItem(dtype="float32", shape=(joint_dim,), names=default_feature_joint_names))
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


# leisaac/source/leisaac/leisaac/tasks/template/single_arm_env_cfg.py
def build_lerobot_frame(
    episode_data: EpisodeData,
    dataset_cfg: LeRobotDatasetCfg,
) -> dict:
    obs_data = episode_data._data["obs"]
    action = obs_data["actions"][-1]

    if dataset_cfg.action_from_ee and (not dataset_cfg.action_align):
        processed_action = normalize_joints(obs_data["ik_joint_target"][-1].unsqueeze(0)).squeeze(0)
        state = normalize_joints(episode_data._data["states"]["articulation"]["robot"]["joint_position"][-1].unsqueeze(0)).squeeze(0)
        # state = normalize_joints(obs_data["joint_pos"][-1].unsqueeze(0)).squeeze(0)
        print("action:", obs_data["ik_joint_target"][-1])
        print("joint_pos:", obs_data["joint_pos"][-1])
        print("stats:", episode_data._data["states"]["articulation"]["robot"]["joint_position"])

    else:
        if dataset_cfg.action_align:
            processed_action = convert_leisaac_action_to_lerobot(action.unsqueeze(0)).squeeze(0)
        else:
            processed_action = action.cpu().numpy()
        state = convert_leisaac_action_to_lerobot(obs_data["joint_pos"][-1].unsqueeze(0)).squeeze(0)
    frame = {
        "action": processed_action,
        "observation.state": state,
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
    print(list(all_data.keys()))
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