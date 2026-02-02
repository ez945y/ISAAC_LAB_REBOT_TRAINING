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

# 視覺提取器
# from mobilenet_extractor import MobileNetFeatureExtractor, MobileNetFeatureExtractorCfg
from extractor import FeatureExtractor, FeatureExtractorConfig
from ik_solver import SO101OfficialIKSolver

@dataclass
class IsaacDeployConfig:
    robot: RobotConfig
    pretrained: str = "exported/policy.pt"
    fps: int = 30 # 頻率同步：物理 60Hz / decimation 2 = 15Hz
    action_scale: float = 0.5 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_vision: bool = False # 如果相機連不上的話，設為 True 使用零矩陣測試邏輯
    camera_warmup: float = 2.0 # 增加相機初使化超時時間

@parser.wrap()
def deploy(cfg: IsaacDeployConfig):
    init_logging()
    logger = logging.getLogger("SO101-Deploy")
    device = cfg.device
    ik_bridge = SO101OfficialIKSolver("so101_new_calib.urdf", device=device)
    
    # 初始化機器人
    cfg.robot.use_degrees = True # 強制使用角度模式，方便偵錯與轉換
    
    # 增加相機初使化時間
    for cam_name in cfg.robot.cameras:
        cfg.robot.cameras[cam_name].warmup_s = cfg.camera_warmup

    robot = make_robot_from_config(cfg.robot)
    
    try:
        robot.connect()
    except Exception as e:
        if cfg.dummy_vision:
            logger.warning(f"相機連接失敗，但 dummy_vision=True，將使用零矩陣進行測試。錯誤: {e}")
        else:
            raise e

    policy = torch.jit.load(cfg.pretrained, map_location=device)
    policy.eval()

    # img_cfg = MobileNetFeatureExtractorCfg(embedding_dim=128, use_fp16=(device == "cuda"))
    # vision_extractor = MobileNetFeatureExtractor(img_cfg, device=device)

    image_cfg = FeatureExtractorConfig(
        model_name="theia-tiny-patch16-224-cddsv",
        device="cpu",
    )
    vision_extractor = FeatureExtractor(image_cfg)

    last_action = torch.zeros((1, 7), device=device) # 6+1維
    drawer_target = torch.ones((1, 1), device=device) # 1維
    arm_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    joint_names = arm_names + ['gripper']

    logger.info(f"Deployment Ready. Degree Mode: {cfg.robot.use_degrees}. Target: 276 dims.")

    try:
        while True:
            start_t = time.perf_counter()
            robot_obs = robot.get_observation()

            # Isaac Lab 訓練的模型預期輸入是弧度。observations 276 維。
            current_pos_deg = torch.tensor([robot_obs[f"{n}.pos"] for n in joint_names], device=device).unsqueeze(0)
            current_vel_deg = torch.tensor([robot_obs[f"{n}.vel"] for n in joint_names], device=device).unsqueeze(0)
            
            current_pos_rad = torch.deg2rad(current_pos_deg)
            current_vel_rad = torch.deg2rad(current_vel_deg)
            
            # 數值部分拼接 (12 + 1 + 7 = 20維)
            numeric_obs = torch.cat([
                current_pos_rad, 
                current_vel_rad, 
                drawer_target,
                last_action
            ], dim=-1)

            # --- 影像讀取與處理 ---
            if not cfg.dummy_vision:
                top_frame = robot.cameras["top"].async_read() 
                front_frame = robot.cameras["front"].async_read() 
                top_input = torch.from_numpy(top_frame).unsqueeze(0).to(device)
                front_input = torch.from_numpy(front_frame).unsqueeze(0).to(device)
            else:
                top_input = torch.zeros((1, 480, 640, 3), device=device)
                front_input = torch.zeros((1, 480, 640, 3), device=device)

            front_emb = vision_extractor.step(front_input)  # (1, emb_dim)
            top_emb = vision_extractor.step(top_input)  # (1, emb_dim)

            obs_vector = torch.cat([
                current_pos_rad, 
                current_vel_rad, 
                drawer_target,
                last_action,
                front_emb,
                top_emb
            ], dim=-1)

            # 推理
            with torch.no_grad():
                action_raw = policy(obs_vector)
            
            action_ee = action_raw * cfg.action_scale
            
            # --- IK 解算 (傳入角度，回傳角度) ---
            target_arm_pos = ik_bridge.solve(
                current_pos_deg[0, :5], 
                action_ee[0, :6]
            )
            target_gripper = torch.rad2deg(action_ee[0, 6]).item()

            # --- 安全限幅與平滑化 ---
            # 1. 增量限制 (Delta Clamp): 每步最大允許移動 6.0 度 1秒最多180度
            max_delta_deg = 6.0
            joint_delta = target_arm_pos - current_pos_deg[0, :5]
            joint_delta = torch.clamp(joint_delta, min=-max_delta_deg, max=max_delta_deg)
            target_arm_pos = current_pos_deg[0, :5] + joint_delta

            # 2. 物理限幅 (Joint Limits): 根據 URDF 設定
            lower_limits = torch.tensor([-110.0, -100.0, -97.0, -95.0, -160.0], device=device)
            upper_limits = torch.tensor([110.0, 100.0, 97.0, 95.0, 160.0], device=device)
            target_arm_pos = torch.clamp(target_arm_pos, min=lower_limits, max=upper_limits)

            target_list = [round(x.item(), 3) for x in target_arm_pos]
            current_list = [round(x.item(), 3) for x in current_pos_deg[0, :5]]
            diff_list = [round((target_arm_pos[i] - current_pos_deg[0, i]).item(), 3) for i in range(5)]
            
            if True: # DRY RUN
                print(f"\r[{time.strftime('%H:%M:%S')}] [DRY RUN] Current: {current_list} | Target: {target_list} | Delta: {diff_list} | Grip: {target_gripper:.2f} deg", end="", flush=True)
            else:
                print(f"\r[{time.strftime('%H:%M:%S')}] Current: {current_list} | Target: {target_list} | Delta: {diff_list} | Grip: {target_gripper:.2f} deg", end="", flush=True)
                action_dict = {f"{n}.pos": target_arm_pos[i].item() for i, n in enumerate(arm_names)}
                action_dict["gripper.pos"] = target_gripper
                robot.send_action(action_dict)
        
            last_action = action_raw.clone()

            # 頻率控制
            dt_s = time.perf_counter() - start_t
            sleep_time = max(1 / cfg.fps - dt_s, 0.0)
            # logger.debug(f"Loop dt: {dt_s:.3f}s, sleeping: {sleep_time:.3f}s")
            precise_sleep(sleep_time)

    finally:
        robot.disconnect()

def main():
    register_third_party_plugins()
    deploy()

if __name__ == "__main__":
    main()