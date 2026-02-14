"""
從 LeRobot 資料集讀取動作，直接以關節位置控制方式回放到 Isaac Sim

- 不使用 IK / OSC 控制器
- 直接透過 write_joint_state_to_sim 將關節位置寫入模擬
- 30 FPS 回放
- 支援雙機位攝影錄影 (top_view + front_view, 640x480)
"""

import argparse
import os
import sys
import time
import math

from isaaclab.app import AppLauncher

# 將上層目錄加入 path 以便匯入 controll_scripts（僅取 USD 路徑）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="Replay dataset actions in Isaac Sim (direct joint position)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episode", type=int, default=0, help="要回放的 episode 編號")
parser.add_argument("--dataset", type=str, default="MikeChenYZ/soarm-fmb-v2", help="LeRobot dataset ID")
parser.add_argument("--fps", type=float, default=30.0, help="回放幀率")
parser.add_argument("--video", action="store_true", default=False, help="是否錄製影片")
parser.add_argument("--video_dir", type=str, default="./videos", help="影片儲存目錄")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 以下為 sim 啟動後可用的 import ──────────────────────────────
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg


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
        actions.append(sample["action"])             # [6] (5 arm + 1 gripper), degree
        obs_states.append(sample["observation.state"])  # [6], degree
        timestamps.append(sample["timestamp"].item())   # float, seconds

    if not actions:
        raise ValueError(f"Episode {episode_idx} 不存在於資料集中")

    actions_tensor = torch.stack(actions)       # [num_frames, 6]
    obs_states_tensor = torch.stack(obs_states) # [num_frames, 6]
    timestamps_list = timestamps
    print(f"[INFO] 載入 episode {episode_idx}，共 {len(actions_tensor)} 幀")
    return actions_tensor, obs_states_tensor, timestamps_list


# ── 場景配置 ────────────────────────────────────────────────────
USD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "controll_scripts", "so_arm_101", "SO-ARM101v2.usd",
)

# 5 個手臂關節 + 1 個夾爪 (與 Isaac Sim USD 中的關節名稱對應)
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT_NAME = "gripper"


class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
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
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.08)),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=ARM_JOINT_NAMES,
                effort_limit=0.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[GRIPPER_JOINT_NAME],
                effort_limit=0.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    # ── 攝影機 (640x480, 30Hz) ── spawn 新攝影機（避開 USD 中已有的 prim 名稱）
    # 手腕攝影機 — 掛在 gripper_link 下，跟隨手臂末端移動
    wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/gripper_link/wrist_cam",
        update_period=1 / 30,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=None
    )
    # 俯瞰攝影機 — 前方 60cm、高 90cm，從上往下微微上揚 5°
    top = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/top_cam",
        update_period=1 / 30,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=None
    )


# ── 主程式 ──────────────────────────────────────────────────────
def main():
    # 載入資料集
    actions, obs_states, ds_timestamps = load_episode_data(args_cli.dataset, args_cli.episode)

    # 初始化模擬
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.4, 0.4, 0.4], [0.0, 0.0, 0.15])
    sim_dt = sim.get_physics_dt()

    # 建立場景
    scene_cfg = ReplaySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # 重置
    sim.reset()
    robot = scene["robot"]
    robot.update(dt=sim_dt)

    # 放寬手臂關節限制，避免資料集動作被 clamp
    arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    joint_limits = robot.data.joint_limits.clone()  # [num_envs, num_joints, 2]
    for jid in arm_joint_ids:
        joint_limits[:, jid, 0] = -2 * math.pi  # lower
        joint_limits[:, jid, 1] = 2 * math.pi   # upper
    robot.write_joint_limits_to_sim(joint_limits)
    robot.update(dt=sim_dt)
    print(f"[INFO] 已放寬手臂關節限制至 ±2π")

    # 取得關節 ID
    arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_joint_ids, _ = robot.find_joints([GRIPPER_JOINT_NAME])
    all_joint_ids = arm_joint_ids + gripper_joint_ids

    print(f"[INFO] Arm joint IDs: {arm_joint_ids}")
    print(f"[INFO] Gripper joint IDs: {gripper_joint_ids}")
    print(f"[INFO] Physics dt: {sim_dt:.4f}s")
    print(f"[INFO] 回放 FPS: {args_cli.fps}")

    dt_target = 1.0 / args_cli.fps  # 每幀目標時間
    num_physics_steps = max(1, int(dt_target / sim_dt))  # 每幀需要幾步 physics
    print(f"[INFO] 每幀執行 {num_physics_steps} 步 physics step")

    # 將 actions / obs_states 搬到目標 device，並預先轉為 radian
    device = sim.device
    actions_rad = (actions * (math.pi / 180.0)).to(device)   # [num_frames, 6]
    obs_states_rad = (obs_states * (math.pi / 180.0)).to(device)

    # wrist_roll (index 4) 乘以 2 倍
    actions_rad[:, 4] *= 1.9
    obs_states_rad[:, 4] *= 1.9
    print(f"[INFO] wrist_roll (index 4) 已乘以 1.9 倍")

    # ── 影片錄製準備 ──
    wrist_frames = []
    top_frames = []
    if args_cli.video:
        wrist_cam = scene["wrist"]
        top_cam = scene["top"]
        print(f"[INFO] 錄影模式已啟用 (640x480, wrist + top)")

    # 回放迴圈
    frame_idx = 0
    total_frames = len(actions_rad)
    sim_time = 0.0  # 模擬時間累計

    # 用第一幀的 action 作為初始位置
    prev_pos = actions_rad[0]  # [6]

    def _fmt(t):
        return "[" + ", ".join(f"{v:>7.4f}" for v in t.tolist()) + "]"

    print(f"\n[INFO] 開始回放，共 {total_frames} 幀 ...")
    print(f"{'Frame':>6} | {'DS_t':>7} | {'Sim_t':>7} | {'DS obs_state (rad)':^52} | {'DS action (rad)':^52} | {'Sim joints (rad)':^52}")
    print("-" * 230)

    while simulation_app.is_running():
        start_time = time.time()

        if frame_idx < total_frames:
            cur_action = actions_rad[frame_idx]      # [6] radian
            obs_state_rad = obs_states_rad[frame_idx]  # [6] radian
            ds_t = ds_timestamps[frame_idx]             # dataset timestamp (s)

            # 下一幀目標（用於插值），最後一幀就保持不變
            next_pos = cur_action  # 本幀的目標位置

            # ── 在 physics sub-steps 之間做線性插值 ──
            for step_i in range(num_physics_steps):
                alpha = (step_i + 1) / num_physics_steps  # 0→1
                interp_pos = prev_pos + alpha * (next_pos - prev_pos)  # 線性插值
                all_pos = interp_pos.unsqueeze(0)  # [1, 6]
                all_vel = torch.zeros_like(all_pos)
                robot.write_joint_state_to_sim(all_pos, all_vel, joint_ids=all_joint_ids)
                sim.step()

            robot.update(sim_dt * num_physics_steps)

            # 更新 prev_pos 供下一幀插值用
            prev_pos = next_pos

            # ── 讀取 sim 觀測到的實際關節位置 ──
            sim_joint_pos = robot.data.joint_pos[0, all_joint_ids]  # [6]

            # ── 擷取攝影機畫面 ──
            if args_cli.video:
                wrist_cam.update(dt=sim_dt * num_physics_steps)
                top_cam.update(dt=sim_dt * num_physics_steps)
                # RGB data: [num_envs, H, W, 3]
                wrist_img = wrist_cam.data.output["rgb"][0].cpu().numpy()
                top_img = top_cam.data.output["rgb"][0].cpu().numpy()
                wrist_frames.append(wrist_img[:, :, :3])   # 取 RGB (RGBA → RGB)
                top_frames.append(top_img[:, :, :3])

            # ── Log ──
            print(
                f"{frame_idx:>4d}/{total_frames:>4d} | "
                f"{ds_t:>7.3f} | {sim_time:>7.3f} | "
                f"obs: {_fmt(obs_state_rad)} | "
                f"act: {_fmt(cur_action)} | "
                f"sim: {_fmt(sim_joint_pos)}"
            )

            frame_idx += 1
        else:
            # 回放結束
            if frame_idx == total_frames:
                print("\n[INFO] 回放完成！")
                if args_cli.video:
                    _save_videos(wrist_frames, top_frames, args_cli)
                    break  # 錄完影就結束
                else:
                    print("[INFO] 保持最後姿態，按 Ctrl+C 結束。")
                frame_idx += 1  # 只印一次
            sim.step()
            robot.update(sim_dt)

        sim_time += dt_target

        # 控制幀率
        elapsed = time.time() - start_time
        sleep_time = dt_target - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def _save_videos(wrist_frames, top_frames, args):
    """將兩機位的 RGB 幀列表存成 MP4 影片"""
    import cv2

    os.makedirs(args.video_dir, exist_ok=True)
    fps = int(args.fps)
    ep = args.episode

    for name, frames in [("wrist", wrist_frames), ("top", top_frames)]:
        if not frames:
            continue
        h, w = frames[0].shape[:2]
        path = os.path.join(args.video_dir, f"ep{ep}_{name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[INFO] 影片已儲存: {path}  ({len(frames)} frames, {w}x{h}, {fps} FPS)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()