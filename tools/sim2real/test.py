import logging
import time
import torch
from dataclasses import dataclass, field
from pathlib import Path

# --- 導入 LeRobot 官方核心 ---
from lerobot.configs import parser
from lerobot.robots import (
    RobotConfig,
    make_robot_from_config,
    so_follower,  # 確保註冊
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
import math

# 視覺提取器
# from mobilenet_extractor import MobileNetFeatureExtractor, MobileNetFeatureExtractorCfg
from extractor import FeatureExtractor, FeatureExtractorConfig
from ik_solver import SO101OfficialIKSolver

@dataclass
class SimpleIKConfig:
    robot: RobotConfig
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    move_speed: float = 0.02  # 每秒移動 2cm
    fps: int = 30

@parser.wrap()
def verify_relative_ik(cfg: SimpleIKConfig):
    register_third_party_plugins()
    cfg.robot.use_degrees = True
    robot = make_robot_from_config(cfg.robot)
    
    ik_bridge = SO101OfficialIKSolver("so101_new_calib.urdf", device=cfg.device)
    
    try:
        robot.connect()
        arm_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
        
        # 1. 讀取物理馬達當下的真實角度 (Degree)
        obs = robot.get_observation()
        initial_angles = torch.tensor([obs[f"{name}.pos"] for name in arm_names], device=cfg.device)
        
        # 2. 透過 FK 算出目前的末端座標 (僅供 Debug 顯示)
        init_pos_xyz = ik_bridge.forward_kinematics(initial_angles)
        print(f"\n[FK] 當前物理末端位置: X={init_pos_xyz[0]:.4f}, Y={init_pos_xyz[1]:.4f}, Z={init_pos_xyz[2]:.4f}")

        # 3. 關鍵修正：給予「零位移」的 Delta Pose
        # 因為你的 solve 內部邏輯是：取當前角度 -> 做 FK -> 加 Delta -> 解出新角度
        # 如果 Delta 是 0，它算出來的就是「維持現狀」的角度（包含抵消數值誤差後的精確解）
        zero_delta = torch.zeros(6, device=cfg.device) 
        target_angles = ik_bridge.solve(initial_angles, zero_delta)

        print(f"[IK] 鎖定角度已計算完成: {target_angles.tolist()}")
        print("機器人開始鎖定，嘗試對抗重力...")

        while True:
            start_t = time.perf_counter()
            
            # --- 持續發送鎖定的目標角度 ---
            action = {f"{name}.pos": target_angles[i].item() for i, name in enumerate(arm_names)}
            action["gripper.pos"] = 0.0 # 夾爪保持原狀
            
            robot.send_action(action)
            
            # 觀察誤差
            current_obs = robot.get_observation()
            real_angles = [current_obs[f"{n}.pos"] for n in arm_names]
            
            # 觀察主要的兩個重力受災區：Lift 和 Elbow
            diff_lift = target_angles[1].item() - real_angles[1]
            diff_elbow = target_angles[2].item() - real_angles[2]
            
            print(f"\r[LOCK] L_Diff: {diff_lift:+.2f} | E_Diff: {diff_elbow:+.2f} | FPS: {1/(time.perf_counter()-start_t):.1f}", end="")
            
            precise_sleep(max(1/cfg.fps - (time.perf_counter() - start_t), 0))

    except KeyboardInterrupt:
        print("\n停止鎖定並釋放馬達。")
    finally:
        robot.disconnect()

@parser.wrap()
def test_trajectory(cfg: SimpleIKConfig):
    register_third_party_plugins()
    cfg.robot.use_degrees = True
    robot = make_robot_from_config(cfg.robot)
    ik_bridge = SO101OfficialIKSolver("so101_new_calib.urdf", device=cfg.device)
    
    try:
        robot.connect()
        arm_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
        
        # 1. 鎖定啟動那一刻的「初始角度」作為絕對基準
        obs = robot.get_observation()
        start_angles = torch.tensor([obs[f"{n}.pos"] for n in arm_names], device=cfg.device)
        
        print("\n[READY] 執行垂直上升測試：目標 +5cm")
        
        target_z = 0.2  # 目標上升 5 公分
        duration = 5.0   # 花 5 秒時間完成
        start_time = time.time()

        while True:
            loop_start_t = time.perf_counter()
            elapsed = time.time() - start_time
            
            # 計算當前應該到達的高度 (0 ~ 0.05)
            # 使用 min 確保到 5 公分後就停住不再增加
            current_dz = min(target_z * (elapsed / duration), target_z)
            
            # --- 構造 Delta Pose ---
            # 根據你之前的回饋，如果 X 或 Z 有一個是上下，我們先試試 Z
            # 如果發現它不是往「正上」走，我們再來微調座標軸
            delta_pose = torch.tensor([current_dz, 0.0, current_dz, 0.0, 0.0, 0.0], device=cfg.device).float()
            
            # 關鍵：每次都從 start_angles 出發解算，保證軌跡絕對平滑
            target_angles = ik_bridge.solve(start_angles, delta_pose)
            
            # 執行
            action = {f"{name}.pos": target_angles[i].item() for i, name in enumerate(arm_names)}
            action["gripper.pos"] = 0.0
            robot.send_action(action)
            
            print(f"\r[UP] Elapsed: {elapsed:.1f}s | Delta Z: {current_dz:.3f} | Lift: {target_angles[1]:.2f}", end="")
            
            if elapsed > duration + 1.0:
                print("\n[INFO] 已到達目標高度並維持鎖定。")
                # 這裡不 break，讓它持續發送 action 鎖定在 5cm 處
            
            precise_sleep(max(1/cfg.fps - (time.perf_counter() - loop_start_t), 0))

    except KeyboardInterrupt:
        print("\n[STOP] 停止。")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    # verify_relative_ik()
    test_trajectory()