from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(object_ee_distance / std)

def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def grasped_and_approaching(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    grasped_object_cfg: SceneEntityCfg,      # 被抓的物體，例如 cube_2
    target_object_cfg: SceneEntityCfg,       # 目標放置位置，例如 cube_1
    grasp_diff_threshold: float = 0.02,      # 抓取判斷的距離閾值
    approach_std: float = 0.05,              # 接近 reward 的 std（越小越嚴格）
) -> torch.Tensor:
    """
    密集獎勵：當成功抓住物體後，該物體越靠近目標物體的位置就給越高分。
    - 如果還沒抓住：reward = 0
    - 抓住後：reward = 1 - tanh(distance / std)，距離越近越接近 1
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    grasped_obj: RigidObject = env.scene[grasped_object_cfg.name]
    target_obj: RigidObject = env.scene[target_object_cfg.name]

    # Step 1: 先判斷是否成功抓住
    object_pos = grasped_obj.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(
        object_pos - end_effector_pos + torch.tensor([0.0, 0.0, 0.1], device=object_pos.device),
        dim=1
    )

    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        is_grasped = torch.logical_and(
            pose_diff < grasp_diff_threshold,
            torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[0]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            ) > env.cfg.gripper_threshold,
        )
    else:
        raise ValueError("No gripper_joint_names found in environment config")

    # Step 2: 如果抓住，計算被抓物體到目標物體的距離
    dist = torch.norm(grasped_obj.data.root_pos_w - target_obj.data.root_pos_w, dim=1)

    # 獎勵：抓住時才給分，距離越近越好
    reward = torch.where(
        is_grasped,
        1.0 - torch.tanh(dist / approach_std),
        0.0
    )

    return reward

def ee_floor_penalty(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    floor_height_threshold: float = 0.02,   # EE z 低於這個高度就算碰到地板
    penalty_strength: float = 1.0           # 懲罰強度
) -> torch.Tensor:
    """
    懲罰 end-effector 太靠近地板（z 高度過低）。
    當 EE z < threshold 時給負分，防止手臂壓地板或做出不自然低姿態。
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_z = ee_frame.data.target_pos_w[:, 0, 2]  # EE 的 z 座標 (world frame)

    # 當高度低於閾值時給懲罰（越低越嚴重）
    penalty = torch.where(
        ee_pos_z < floor_height_threshold,
        penalty_strength * (floor_height_threshold - ee_pos_z),  # 越低懲罰越大
        0.0
    )

    return -penalty  # 負值作為懲罰