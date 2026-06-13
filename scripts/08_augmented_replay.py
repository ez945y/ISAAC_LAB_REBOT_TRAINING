"""
從 LeRobot 資料集讀取動作，直接以關節位置控制方式回放到 Isaac Sim
- 支援參數化的 POSE_TABLE（每筆記錄自動從五個固定相機錄影）
- episode + cube + pos + rot 全部從表讀取
- 預載所有 4 個 cube，根據表只顯示/移動指定的那個
- 使用五個獨立相機（top_center / top_up / top_down / top_left / top_right）錄影
  → 不再動態偏移，直接使用 USD 中已存在的五個相機 prim
"""

import argparse
import os
import sys
import time
import math
import cv2

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="Replay dataset actions in Isaac Sim (direct joint position)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--fps", type=float, default=30.0, help="回放幀率")
parser.add_argument("--video", action="store_true", default=True, help="是否錄製影片")
parser.add_argument("--video_dir", type=str, default="./videos", help="影片儲存目錄")
parser.add_argument("--dataset", type=str, default="MikeChenYZ/soarm-fmb-v2", help="LeRobot dataset ID")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Auto-configure WebRTC livestream (publicIp + dynamic resize) when --livestream is set.
from livestream_support import apply_livestream_defaults
apply_livestream_defaults(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from tools.sim2real.action import normalize_joints, denormalize_joints

# ── 常數 ────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD_PATH = os.path.join(_BASE_DIR, "tools", "controll_scripts", "so_arm_101", "SO-ARM101v2.usd")

CUBE_USD_PATHS = {
    2: os.path.join(_BASE_DIR, "tools", "exp", "test", "bodies", "3_2.usd"),
    3: os.path.join(_BASE_DIR, "tools", "exp", "test", "bodies", "3_3.usd"),
    4: os.path.join(_BASE_DIR, "tools", "exp", "test", "bodies", "3_4.usd"),
    5: os.path.join(_BASE_DIR, "tools", "exp", "test", "bodies", "3_5.usd"),
}

# ── 自訂表 ──────────────────────────────────────────────────────────
POSE_TABLE = [
    {"episode": 0, "cube": 2, "pos": (0.38, -0.04, 0.0), "rot": (1.0, 0.0, 0.0, 0.0)},
    {"episode": 1, "cube": 3, "pos": (0.38, -0.04, 0.0), "rot": (1.0, 0.0, 0.0, 0.0)},
    {"episode": 2, "cube": 4, "pos": (0.36, -0.02, 0.0), "rot": (0.707, 0.0, 0.0, 0.707)},
    {"episode": 3, "cube": 5, "pos": (0.36, -0.02, 0.0), "rot": (0.707, 0.0, 0.0, 0.707)},
]

CAM_NAMES = ["center", "up", "down", "left", "right"]  # 五個相機名稱

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT_NAME = "gripper"

JOINT_LIMITS = {
    "shoulder_pan":    (-1.8243,   1.8243),
    "shoulder_lift":   (-1.7691,   1.7691),
    "elbow_flex":      (-1.6026,   1.6026),
    "wrist_flex":      (-1.8067,   1.8067),
    "wrist_roll":      (-3.0741,   3.0741),
    "gripper":         (0.0,   1.7453),
}

_JOINT_ORDER = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
JOINT_LOWER = torch.tensor([JOINT_LIMITS[j][0] for j in _JOINT_ORDER])
JOINT_UPPER = torch.tensor([JOINT_LIMITS[j][1] for j in _JOINT_ORDER])


# ── 場景配置 ──────────────────────────────────────────────────────
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))

    # 預載 4 個 cube
    cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_2",
        spawn=sim_utils.UsdFileCfg(usd_path=CUBE_USD_PATHS[2], scale=(0.001,0.001,0.001), mass_props=sim_utils.MassPropertiesCfg(mass=0.066)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0), rot=(1.0,0.0,0.0,0.0)),
    )
    cube_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_3",
        spawn=sim_utils.UsdFileCfg(usd_path=CUBE_USD_PATHS[3], scale=(0.001,0.001,0.001), mass_props=sim_utils.MassPropertiesCfg(mass=0.066)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0), rot=(1.0,0.0,0.0,0.0)),
    )
    cube_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_4",
        spawn=sim_utils.UsdFileCfg(usd_path=CUBE_USD_PATHS[4], scale=(0.001,0.001,0.001), mass_props=sim_utils.MassPropertiesCfg(mass=0.066)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0), rot=(1.0,0.0,0.0,0.0)),
    )
    cube_5 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_5",
        spawn=sim_utils.UsdFileCfg(usd_path=CUBE_USD_PATHS[5], scale=(0.001,0.001,0.001), mass_props=sim_utils.MassPropertiesCfg(mass=0.066)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0), rot=(1.0,0.0,0.0,0.0)),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True, enabled_self_collisions=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.01, -0.005, 0.05)),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=ARM_JOINT_NAMES, effort_limit=40, stiffness=17.8, damping=0.6),
            "gripper": ImplicitActuatorCfg(joint_names_expr=[GRIPPER_JOINT_NAME], effort_limit=40, stiffness=17.8, damping=0.6),
        },
    )

    wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/gripper_link/wrist_cam",
        update_period=1 / 30,
        height=480,
        width=640,
        data_types=["rgb", "semantic_segmentation"],
        spawn=None
    )

    # 五個固定 top 相機（假設 USD 已存在這些 prim）
    top_center = TiledCameraCfg(
        prim_path="/World/envs/env_0/robot/top_cam",
        update_period=1 / 30,
        height=480,
        width=640,
        data_types=["rgb", "semantic_segmentation"],
        spawn=None
    )
    top_up = TiledCameraCfg(
        prim_path="/World/envs/env_0/robot/top_cam_up",
        update_period=1 / 30,
        height=480,
        width=640,
        data_types=["rgb", "semantic_segmentation"],
        spawn=None
    )
    top_down = TiledCameraCfg(
        prim_path="/World/envs/env_0/robot/top_cam_down",
        update_period=1 / 30,
        height=480,
        width=640,
        data_types=["rgb", "semantic_segmentation"],
        spawn=None
    )
    top_left = TiledCameraCfg(
        prim_path="/World/envs/env_0/robot/top_cam_left",
        update_period=1 / 30,
        height=480,
        width=640,
        data_types=["rgb", "semantic_segmentation"],
        spawn=None
    )
    top_right = TiledCameraCfg(
        prim_path="/World/envs/env_0/robot/top_cam_right",
        update_period=1 / 30,
        height=480,
        width=640,
        data_types=["rgb", "semantic_segmentation"],
        spawn=None
    )


# ── 設定 cube ────────────────────────────────────────────────────────
def configure_cube(scene, active_cube_id, pos, rot, device):
    # active 的位置（正常顯示）
    pose_active = torch.tensor([[pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]]], device=device)
    
    # 非 active 的位置：放超遠（z = -1000 或更遠）
    FAR_Z = -1000.0
    pose_hidden = torch.tensor([[0.0, 0.0, FAR_Z, 1.0, 0.0, 0.0, 0.0]], device=device)

    for cid in [2, 3, 4, 5]:
        cube_obj = scene[f"cube_{cid}"]
        
        if cid == active_cube_id:
            cube_obj.write_root_pose_to_sim(pose_active)
        else:
            cube_obj.write_root_pose_to_sim(pose_hidden)
        
        cube_obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))


def load_episode_data(dataset_id: str, episode_idx: int):
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

    actions_tensor = torch.stack(actions)
    obs_states_tensor = torch.stack(obs_states)
    print(f"[INFO] 載入 episode {episode_idx}，共 {len(actions_tensor)} 幀")
    return actions_tensor, obs_states_tensor


# ── 主程式 ──────────────────────────────────────────────────────────
def main():
    print(f"\n[INFO] 從 POSE_TABLE 讀取，共 {len(POSE_TABLE)} 筆記錄，每筆從五個固定相機錄影\n")
    print("[INFO] 將使用單次模擬 + 多相機同時 update 方式，加速錄製\n")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=(0.0, 0.0, -9.81))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_dt = sim.get_physics_dt()

    scene_cfg = ReplaySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    robot.update(dt=sim_dt)

    arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    joint_limits = robot.data.joint_limits.clone()
    for jid in arm_joint_ids:
        joint_limits[:, jid, 0] = -2 * math.pi
        joint_limits[:, jid, 1] = 2 * math.pi
    robot.write_joint_position_limit_to_sim(joint_limits)
    robot.update(dt=sim_dt)

    all_joint_ids = arm_joint_ids + robot.find_joints([GRIPPER_JOINT_NAME])[0]

    dt_target = 1.0 / args_cli.fps
    num_physics_steps = max(1, int(dt_target / sim_dt))
    device = sim.device

    wrist_cam = scene["wrist"]
    top_cams = {
        "center": scene["top_center"],
        "up":     scene["top_up"],
        "down":   scene["top_down"],
        "left":   scene["top_left"],
        "right":  scene["top_right"],
    }

    all_top_cams = list(top_cams.values())  # 用來同時 update

    # 暖機
    for _ in range(5):
        sim.step()
    robot.update(sim_dt * 5)

    # 用來收集所有 pose 的影片路徑，之後合併用
    video_paths_per_cam = {cam: [] for cam in CAM_NAMES}
    seg_paths_per_cam = {cam: [] for cam in CAM_NAMES}

    # 逐筆記錄
    for record_idx, record in enumerate(POSE_TABLE):
        episode = record["episode"]
        cube_id = record["cube"]
        pos = record["pos"]
        rot = record["rot"]

        print(f"\n[Record {record_idx+1}/{len(POSE_TABLE)}] ep={episode} cube={cube_id} pos={pos} rot={rot}")

        actions, obs_states = load_episode_data(args_cli.dataset, episode)
        actions_rad = denormalize_joints(actions).to(device)

        configure_cube(scene, cube_id, pos, rot, device)

        init_pos = actions_rad[0].unsqueeze(0)
        robot.set_joint_position_target_index(init_pos, joint_ids=all_joint_ids)
        robot.write_data_to_sim()

        for _ in range(10):
            sim.step()
        robot.update(sim_dt * 10)

        # 準備收集這個 record 的所有相機畫面
        wrist_frames_all = []   # wrist 共用
        top_frames_per_cam = {cam: [] for cam in CAM_NAMES}
        wrist_segs_all = []
        top_segs_per_cam = {cam: [] for cam in CAM_NAMES}

        prev_pos = actions_rad[0]
        total_frames = len(actions_rad)

        print("  [INFO] 開始單次模擬 + 多相機同時錄製...")

        for frame_idx in range(total_frames):
            if not simulation_app.is_running():
                break

            cur_action = actions_rad[frame_idx]
            next_pos = cur_action

            for step_i in range(num_physics_steps):
                alpha = (step_i + 1) / num_physics_steps
                interp_pos = prev_pos + alpha * (next_pos - prev_pos)
                robot.set_joint_position_target_index(interp_pos.unsqueeze(0), joint_ids=all_joint_ids)
                robot.write_data_to_sim()
                sim.step()

            robot.update(sim_dt * num_physics_steps)
            prev_pos = next_pos

            if args_cli.video:
                wrist_cam.update(dt=sim_dt * num_physics_steps)
                for cam in all_top_cams:
                    cam.update(dt=sim_dt * num_physics_steps)

                # wrist 畫面共用
                wrist_img = wrist_cam.data.output["rgb"][0].cpu().numpy()[:, :, :3]
                wrist_seg = wrist_cam.data.output["semantic_segmentation"][0].cpu().numpy()
                wrist_frames_all.append(wrist_img)
                wrist_segs_all.append(wrist_seg)

                # 每個 top cam 獨立收集
                for cam_name, cam in top_cams.items():
                    top_img = cam.data.output["rgb"][0].cpu().numpy()[:, :, :3]
                    top_seg = cam.data.output["semantic_segmentation"][0].cpu().numpy()
                    top_frames_per_cam[cam_name].append(top_img)
                    top_segs_per_cam[cam_name].append(top_seg)

            if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
                print(f"      frame {frame_idx}/{total_frames}")

            time.sleep(max(0, dt_target - sim_dt * num_physics_steps))

        # 儲存這個 record 的所有相機影片
        if args_cli.video:
            for cam_name in CAM_NAMES:
                suffix = f"pose{record_idx}_{cam_name}"
                wrist_rgb_path, top_rgb_path = save_videos(
                    wrist_frames_all, top_frames_per_cam[cam_name],
                    args_cli.video_dir, episode, cube_id, suffix, args_cli.fps
                )
                wrist_seg_path, top_seg_path = save_semantic_videos(
                    wrist_segs_all, top_segs_per_cam[cam_name],
                    args_cli.video_dir, episode, cube_id, suffix, args_cli.fps
                )

                # 收集 RGB top
                if top_rgb_path:
                    video_paths_per_cam[cam_name].append(top_rgb_path)  # 原有 RGB 收集
                
                # 新增：收集 seg top
                if top_seg_path:
                    seg_paths_per_cam[cam_name].append(top_seg_path)    # 新增 seg 收集

            print(f"    ✓ Cam {cam_name} 完成 (單次模擬)")

        print(f"  ✓ Record {record_idx+1} (5 cams) 完成")

    # ── 所有 record 完成後，合併每個相機的影片 ──────────────────────────────
    print("\n[INFO] 開始合併所有 pose 的影片成單一檔案...")
    merge_videos_per_camera(video_paths_per_cam, seg_paths_per_cam, args_cli.video_dir, args_cli.fps)


# ── 合併函式 ──────────────────────────────────────────────────────────
def merge_videos_per_camera(rgb_paths_per_cam, seg_paths_per_cam, video_dir, fps):
    import cv2
    import os
    
    # 1. 合併 segmentation（每個相機一部）
    for cam_name, paths in seg_paths_per_cam.items():
        if not paths:
            print(f"[SKIP] {cam_name} 沒有 semantic 影片可合併")
            continue
        
        output_path = os.path.join(video_dir, f"combined_all_poses_{cam_name}_semantic.mp4")
        print(f"  合併 Semantic {cam_name} → {output_path} ({len(paths)} 段)")
        
        merge_multiple_videos(paths, output_path, fps)
    
    # 2. RGB 可以選擇不合併，或合併成一部全視角大檔
    all_rgb_paths = []
    for cam_name, paths in rgb_paths_per_cam.items():
        all_rgb_paths.extend(paths)
    
    if all_rgb_paths:
        output_all_rgb = os.path.join(video_dir, "combined_all_rgb_top.mp4")
        print(f"  合併所有 RGB top 成單一部 → {output_all_rgb} ({len(all_rgb_paths)} 段)")
        merge_multiple_videos(all_rgb_paths, output_all_rgb, fps)
    else:
        print("[INFO] RGB 維持單獨小檔，沒有合併成一部大檔")


def merge_multiple_videos(input_paths, output_path, fps):
    if not input_paths:
        return
    
    cap = cv2.VideoCapture(input_paths[0])
    if not cap.isOpened():
        print(f"無法開啟 {input_paths[0]}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for path in input_paths:
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
    
    writer.release()
    print(f"  ✓ 合併完成: {output_path}")


def save_videos(wrist_frames, top_frames, video_dir, ep, cube_id, suffix, fps):
    import cv2
    os.makedirs(video_dir, exist_ok=True)
    
    wrist_path = None
    top_path = None
    
    if wrist_frames:
        h, w = wrist_frames[0].shape[:2]
        wrist_path = os.path.join(video_dir, f"ep{ep}_cube{cube_id}_{suffix}_wrist.mp4")
        writer = cv2.VideoWriter(wrist_path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h))
        for f in wrist_frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[VIDEO RGB] {wrist_path}")
    
    if top_frames:
        h, w = top_frames[0].shape[:2]
        top_path = os.path.join(video_dir, f"ep{ep}_cube{cube_id}_{suffix}_top.mp4")
        writer = cv2.VideoWriter(top_path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h))
        for f in top_frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[VIDEO RGB] {top_path}")
    
    return wrist_path, top_path


def save_semantic_videos(wrist_segs, top_segs, video_dir, ep, cube_id, suffix, fps):
    import cv2
    os.makedirs(video_dir, exist_ok=True)
    
    wrist_seg_path = None
    top_seg_path = None
    
    if wrist_segs:
        h, w = wrist_segs[0].shape[:2]
        wrist_seg_path = os.path.join(video_dir, f"ep{ep}_cube{cube_id}_{suffix}_wrist_semantic.mp4")
        writer = cv2.VideoWriter(wrist_seg_path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h))
        for seg in wrist_segs:
            writer.write(cv2.cvtColor(seg, cv2.COLOR_RGBA2BGR))
        writer.release()
        print(f"[VIDEO SEG] {wrist_seg_path}")
    
    if top_segs:
        h, w = top_segs[0].shape[:2]
        top_seg_path = os.path.join(video_dir, f"ep{ep}_cube{cube_id}_{suffix}_top_semantic.mp4")
        writer = cv2.VideoWriter(top_seg_path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h))
        for seg in top_segs:
            writer.write(cv2.cvtColor(seg, cv2.COLOR_RGBA2BGR))
        writer.release()
        print(f"[VIDEO SEG] {top_seg_path}")
    
    return wrist_seg_path, top_seg_path


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()