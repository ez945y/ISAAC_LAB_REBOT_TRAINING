"""
從 LeRobot 資料集讀取動作，在 Isaac Sim 中回放並將 top 鏡頭即時串流至 TCP。

基於 07_moving_from_dataset.py 的場景配置與回放邏輯，
將 top camera 渲染結果以 JPEG 壓縮後透過 TCP 送出，
供 stream_top_receiver.py 在外部接收顯示。

Protocol:
  每幀: [4 bytes: frame_size (big-endian uint32)] + [frame_size bytes: JPEG data]
  結束: [4 bytes: 0x00000000]

Usage:
    # 終端 1 (Isaac Lab 環境):
    python scripts/11_stream_top_sender.py --port 9999 --enable_cameras

    # 終端 2 (任意有 cv2 的環境):
    python scripts/12_stream_top_receiver.py --port 9999
"""

import argparse
import os
import sys
import time
import math
import socket
import struct

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="Replay episode in Isaac Sim & stream top camera via TCP")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episode", type=int, default=0, help="要回放的 episode 編號")
parser.add_argument("--dataset", type=str, default="MikeChenYZ/soarm-fmb-v2", help="LeRobot dataset ID")
parser.add_argument("--fps", type=float, default=30.0, help="回放幀率")
parser.add_argument("--port", type=int, default=9999, help="TCP streaming port")
parser.add_argument("--host", type=str, default="0.0.0.0", help="TCP bind address")
parser.add_argument("--quality", type=int, default=90, help="JPEG quality (1-100)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 以下為 sim 啟動後可用的 import ──────────────────────────────
import cv2
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from tools.sim2real.action import *


# ── 載入資料集 ─────────────────────────────────────────────────
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

    actions_tensor = torch.stack(actions)
    obs_states_tensor = torch.stack(obs_states)
    print(f"[INFO] 載入 episode {episode_idx}，共 {len(actions_tensor)} 幀")
    return actions_tensor, obs_states_tensor, timestamps


# ── 場景配置 (同 07_moving_from_dataset.py) ─────────────────────
USD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "controll_scripts", "so_arm_101", "SO-ARM101v3.usd",
)
OBJECT_USD_PATH_1 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "exp", "test", "bodies", "3_1.usd",
)
OBJECT_USD_PATH_2 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "exp", "test", "bodies", "3_4.usd",
)


def denormalize_joints(norm_values: torch.Tensor) -> torch.Tensor:
    lower = JOINT_LOWER.to(norm_values.device)
    upper = JOINT_UPPER.to(norm_values.device)
    result = torch.zeros_like(norm_values)
    arm_norm_01 = (norm_values[..., :5] + 100.0) / 200.0
    result[..., :5] = lower[:5] + arm_norm_01 * (upper[:5] - lower[:5])
    gripper_norm_01 = norm_values[..., 5:6] / 100.0
    result[..., 5:6] = lower[5:6] + gripper_norm_01 * (upper[5:6] - lower[5:6])
    return result


class StreamSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )
    test_object_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/test_object_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=OBJECT_USD_PATH_1,
            scale=(0.001, 0.001, 0.001),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.36, 0.215, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    test_object_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/test_object_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=OBJECT_USD_PATH_2,
            scale=(0.001, 0.001, 0.001),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.066),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.0005,
            ),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=8,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.36, -0.02, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, -0.005, 0.05)),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=ARM_JOINT_NAMES,
                effort_limit=40,
                stiffness=17.8,
                damping=0.6,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[GRIPPER_JOINT_NAME],
                effort_limit=40,
                stiffness=17.8,
                damping=0.6,
            ),
        },
    )
    # 俯瞰攝影機 (只需要 top)
    top = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/top_cam",
        update_period=1 / 30,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=None,
    )


# ── TCP 串流輔助 ────────────────────────────────────────────────
def encode_and_send(conn: socket.socket, img_rgb: np.ndarray, quality: int):
    """將 RGB 影像壓縮為 JPEG 並透過 TCP 送出"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ret, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ret:
        return False
    jpeg_data = buf.tobytes()
    header = struct.pack('>I', len(jpeg_data))
    try:
        conn.sendall(header + jpeg_data)
        return True
    except (BrokenPipeError, ConnectionResetError):
        return False


def send_end_signal(conn: socket.socket):
    """送出結束信號 (frame_size = 0)"""
    try:
        conn.sendall(struct.pack('>I', 0))
    except Exception:
        pass


# ── 主程式 ──────────────────────────────────────────────────────
def main():
    # 載入資料集
    actions, obs_states, ds_timestamps = load_episode_data(args_cli.dataset, args_cli.episode)

    # 初始化模擬
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=64,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
        ),
    )

    sim = sim_utils.SimulationContext(sim_cfg)
    sim_dt = sim.get_physics_dt()

    # 建立場景
    scene_cfg = StreamSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # 重置
    sim.reset()
    robot = scene["robot"]
    robot.update(dt=sim_dt)

    # 將 viewport 設為 top_cam
    try:
        import omni.kit.viewport.utility as vp_utils
        viewport = vp_utils.get_active_viewport()
        viewport.set_active_camera("/World/envs/env_0/robot/top_cam")
        print("[INFO] Viewport 已切換至 top_cam")
    except Exception as e:
        print(f"[WARN] 無法設定 viewport camera: {e}")
        sim.set_camera_view([0.6, 0.0, 0.9], [0.0, 0.0, 0.08])

    # 放寬手臂關節限制
    arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    joint_limits = robot.data.joint_limits.clone()
    for jid in arm_joint_ids:
        joint_limits[:, jid, 0] = -2 * math.pi
        joint_limits[:, jid, 1] = 2 * math.pi
    robot.write_joint_limits_to_sim(joint_limits)
    robot.update(dt=sim_dt)
    print(f"[INFO] 已放寬手臂關節限制至 ±2π")

    # 取得關節 ID
    arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_joint_ids, _ = robot.find_joints([GRIPPER_JOINT_NAME])
    all_joint_ids = arm_joint_ids + gripper_joint_ids

    dt_target = 1.0 / args_cli.fps
    num_physics_steps = max(1, int(dt_target / sim_dt))

    device = sim.device
    actions_rad = denormalize_joints(actions).to(device)
    obs_states_rad = denormalize_joints(obs_states).to(device)

    total_frames = len(actions_rad)
    top_cam = scene["top"]

    print(f"[INFO] Arm joint IDs: {arm_joint_ids}")
    print(f"[INFO] Gripper joint IDs: {gripper_joint_ids}")
    print(f"[INFO] Physics dt: {sim_dt:.4f}s, 每幀 {num_physics_steps} 步")
    print(f"[INFO] 回放 FPS: {args_cli.fps}, 共 {total_frames} 幀")

    # ── TCP server ──
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args_cli.host, args_cli.port))
    server.listen(1)
    print(f"\n[INFO] TCP server 啟動，等待連線: {args_cli.host}:{args_cli.port}")
    print(f"[INFO] 在另一個終端執行: python scripts/stream_top_receiver.py --port {args_cli.port}")

    conn, addr = server.accept()
    print(f"[INFO] 已連線: {addr}")

    # ── 取得物件初始狀態 (用於重置) ──
    obj2 = scene["test_object_2"]
    obj2_init_pos = torch.tensor([[0.36, -0.02, 0.0]], device=device)
    obj2_init_rot = torch.tensor([[0.707, 0.0, 0.0, 0.707]], device=device)
    obj2_init_vel = torch.zeros(1, 6, device=device)

    # ── 循環回放 ──
    loop_count = 0
    streaming = True

    print(f"\n[INFO] 開始循環回放並串流 (Ctrl+C 或斷線結束)...")

    while simulation_app.is_running() and streaming:
        loop_count += 1
        print(f"\n{'='*50}")
        print(f"[INFO] 回放第 {loop_count} 輪")
        print(f"{'='*50}")

        # 重置 robot 到初始位姿
        init_pos = actions_rad[0].unsqueeze(0)  # [1, 6]
        robot.set_joint_position_target(init_pos, joint_ids=all_joint_ids)
        robot.write_data_to_sim()

        # 重置物件位置
        obj2_pose = torch.cat([obj2_init_pos, obj2_init_rot], dim=1)  # [1, 7]
        obj2.write_root_pose_to_sim(obj2_pose)
        obj2.write_root_velocity_to_sim(obj2_init_vel)

        # 跑幾步讓重置生效
        for _ in range(10):
            sim.step()
        robot.update(sim_dt * 10)
        obj2.update(sim_dt * 10)

        prev_pos = actions_rad[0]

        for frame_idx in range(total_frames):
            if not simulation_app.is_running():
                streaming = False
                break

            start_time = time.time()

            cur_action = actions_rad[frame_idx]
            next_pos = cur_action

            # Physics sub-steps with interpolation
            for step_i in range(num_physics_steps):
                alpha = (step_i + 1) / num_physics_steps
                interp_pos = prev_pos + alpha * (next_pos - prev_pos)
                all_pos = interp_pos.unsqueeze(0)
                robot.set_joint_position_target(all_pos, joint_ids=all_joint_ids)
                robot.write_data_to_sim()
                sim.step()

            robot.update(sim_dt * num_physics_steps)
            prev_pos = next_pos

            # 擷取 top camera
            top_cam.update(dt=sim_dt * num_physics_steps)
            top_img = top_cam.data.output["rgb"][0].cpu().numpy()[:, :, :3]  # RGBA → RGB

            # 串流送出
            if not encode_and_send(conn, top_img, args_cli.quality):
                print(f"[INFO] 連線中斷 (frame {frame_idx})")
                streaming = False
                break

            if frame_idx % 30 == 0:
                print(f"[INFO] Loop {loop_count} | frame {frame_idx}/{total_frames}")

            # 控制幀率
            elapsed = time.time() - start_time
            sleep_time = dt_target - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if streaming:
            print(f"[INFO] 第 {loop_count} 輪完畢，重新開始...")

    conn.close()
    server.close()
    print("[INFO] TCP server 已關閉")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
