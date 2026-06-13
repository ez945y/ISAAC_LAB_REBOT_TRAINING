# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import torch
import numpy as np # 需要 numpy 來進行輪間距計算

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot, a deformable object, and a static stand to an Isaac Lab environment with keyboard control."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Auto-configure WebRTC livestream (publicIp + dynamic resize) when --livestream is set.
from livestream_support import apply_livestream_defaults
apply_livestream_defaults(args_cli)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# **********************************************************************
# 核心修改區塊：引入 Se2Keyboard
# **********************************************************************
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# 引入 Isaac Lab 專用的移動底盤鍵盤控制器
from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg 

# --- Configuration for Assets (保持不變) ---s

# 假設 Jetbot 的輪間距 (Track Width) 為 0.12 米 (實際值請查閱模型文件)
JETBOT_TRACK_WIDTH = 0.12 
# 假設 Jetbot 的最大速度 (Linear Velocity) 為 4.0 m/s (增加速度)
JETBOT_MAX_LINEAR_VEL = 4.0 
# 假設 Jetbot 的最大角速度 (Angular Velocity) 為 12.0 rad/s (進一步增加轉向速度)
JETBOT_MAX_ANGULAR_VEL = 12.0 

JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

DEFORMABLE_CUBE_CONFIG = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/DeformableCube",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.2, 0.2, 0.5),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 1.0, 0.5)),
    debug_vis=True,
)

STAND_CUBOID_CONFIG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/CollisionStand",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.3, 0.3, 0.3),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.0)),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -1.0, 0.25)),
)

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene with Jetbot, a deformable cube, and a static stand."""
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    DeformableCube = DEFORMABLE_CUBE_CONFIG.replace(prim_path="{ENV_REGEX_NS}/DeformableCube")
    CollisionStand = STAND_CUBOID_CONFIG.replace(prim_path="{ENV_REGEX_NS}/CollisionStand")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """
    Runs the synchronous simulation loop, controlling the Jetbot with keyboard input
    using Isaac Lab's Se2Keyboard device.
    """
    sim_dt = sim.get_physics_dt()
    count = 0

    # **********************************************************************
    # 鍵盤輸入初始化 (使用 Se2Keyboard 並自定義鍵位映射)
    # **********************************************************************
    # 配置鍵盤靈敏度 (SE(2) 控制)
    keyboard_cfg = Se2KeyboardCfg(
        v_x_sensitivity=JETBOT_MAX_LINEAR_VEL,    # x 軸靈敏度 (W/S)
        v_y_sensitivity=JETBOT_MAX_ANGULAR_VEL,   # y 軸靈敏度 (A/D 轉向)
        omega_z_sensitivity=0.0                   # 不使用 z 軸旋轉
    )
    keyboard_device = Se2Keyboard(cfg=keyboard_cfg)
    
    # 自定義鍵位映射，添加 W/S/A/D 鍵支持
    # 使用正確的鍵名格式（大寫）
    import numpy as np
    keyboard_device._INPUT_KEY_MAPPING.update({
        # 添加 W/S/A/D 鍵支持
        "W": np.array([1.0, 0.0, 0.0]) * JETBOT_MAX_LINEAR_VEL,      # W: 前進
        "S": np.array([-1.0, 0.0, 0.0]) * JETBOT_MAX_LINEAR_VEL,     # S: 後退
        "A": np.array([0.0, -1.0, 0.0]) * JETBOT_MAX_ANGULAR_VEL,    # A: 左轉 (修正方向)
        "D": np.array([0.0, 1.0, 0.0]) * JETBOT_MAX_ANGULAR_VEL,     # D: 右轉 (修正方向)
    })
    
    print(f"[INFO]: 鍵盤配置 - v_x_sensitivity: {keyboard_cfg.v_x_sensitivity}")
    print(f"[INFO]: 鍵盤配置 - v_y_sensitivity: {keyboard_cfg.v_y_sensitivity}")
    print("[INFO]: 請確保 Isaac Sim 視窗有焦點，然後按 W/S/A/D 鍵或 Arrow 鍵")
    print("[INFO]: W/Arrow Up/Numpad8: 前進, S/Arrow Down/Numpad2: 後退")
    print("[INFO]: A/Arrow Left/Numpad4: 左轉, D/Arrow Right/Numpad6: 右轉")

    def setup():
        nonlocal count
        count = 0
        
        # 重設 Jetbot
        root_jetbot_state = scene["Jetbot"].data.default_root_state.clone()
        root_jetbot_state[:, :3] += scene.env_origins
        scene["Jetbot"].write_root_pose_to_sim(root_jetbot_state[:, :7])
        scene["Jetbot"].write_root_velocity_to_sim(root_jetbot_state[:, 7:])
        joint_pos, joint_vel = (
            scene["Jetbot"].data.default_joint_pos.clone(),
            scene["Jetbot"].data.default_joint_vel.clone(),
        )
        scene["Jetbot"].write_joint_state_to_sim(joint_pos, joint_vel)
        
        # 重設鍵盤裝置
        keyboard_device.reset()
        scene.reset()
        print("[INFO]: Resetting Jetbot state...")
        print("[INFO]: 注意：重置會清除所有按鍵狀態，請重新按下按鍵")

    # 註冊 'R' 鍵的回調函式
    # Se2Keyboard 的 add_callback 已經處理了鍵盤事件訂閱
    keyboard_device.add_callback("R", setup)


    def update_loop():
        nonlocal count
        
        # **********************************************************************
        # 鍵盤控制邏輯 (使用 Se2Keyboard + 命令平滑化)
        # **********************************************************************
        # 1. 獲取鍵盤命令：[v_x, v_y, omega_z]
        # Se2Keyboard 輸出的是 3 元素的 Tensor
        raw_commands = keyboard_device.advance()
        
        # 2. 命令平滑化：解決間歇性問題
        if count == 0:
            # 初始化平滑化緩衝區
            update_loop.smoothed_commands = torch.zeros_like(raw_commands)
            update_loop.command_history = []
        
        # 記錄命令歷史（最近10個命令）
        update_loop.command_history.append(raw_commands.clone())
        if len(update_loop.command_history) > 10:
            update_loop.command_history.pop(0)
        
        # 計算平滑化命令：使用簡單的延遲機制
        if len(update_loop.command_history) >= 2:
            # 使用最近2個命令的邏輯
            prev_cmd = update_loop.command_history[-2]
            curr_cmd = update_loop.command_history[-1]
            
            # 如果當前命令為零但前一個命令非零，保持前一個命令
            # 這可以防止短暫的零值中斷
            if torch.allclose(curr_cmd, torch.zeros_like(curr_cmd), atol=0.1) and \
               not torch.allclose(prev_cmd, torch.zeros_like(prev_cmd), atol=0.1):
                commands = prev_cmd  # 保持前一個非零命令
            else:
                commands = curr_cmd  # 使用當前命令
        else:
            commands = raw_commands
        
        # 記錄前一個狀態用於比較
        if count == 0:
            prev_commands = torch.zeros_like(commands)
        else:
            prev_commands = getattr(update_loop, 'prev_commands', torch.zeros_like(commands))
        
        # 檢查命令是否突然變化（用於監控平滑化效果）
        if count > 0:
            command_change = torch.abs(commands - prev_commands).sum()
            if command_change > 0.1:  # 如果命令變化很大
                if count % 100 == 0:  # 每100步打印一次，減少輸出
                    print(f"[SMOOTH]: 平滑化生效 - 變化量: {command_change}")
                    print(f"[SMOOTH]: 前一個: {prev_commands}")
                    print(f"[SMOOTH]: 當前: {commands}")
        
        # 保存當前狀態
        update_loop.prev_commands = commands.clone()
        
        # 調試信息：檢查 commands 的形狀和值
        if count % 100 == 0:  # 每100步打印一次，避免過多輸出
            print(f"[DEBUG]: commands shape: {commands.shape}")
            print(f"[DEBUG]: commands values: {commands}")
            print(f"[DEBUG]: commands[0] (v_x): {commands[0]}")
            print(f"[DEBUG]: commands[1] (v_y): {commands[1]}")
            print(f"[DEBUG]: commands[2] (omega_z): {commands[2]}")
        
        # 2. 轉換為 Jetbot 輪子速度 (差速模型)
        # 對於 Jetbot：
        # - Arrow Up/Numpad8: v_x > 0 (前進)
        # - Arrow Down/Numpad2: v_x < 0 (後退)
        # - Arrow Left/Numpad4: v_y > 0 (左轉)
        # - Arrow Right/Numpad6: v_y < 0 (右轉)
        
        # 提取速度
        v_x = commands[0]  # x 軸移動 (前進/後退)
        v_y = commands[1]  # y 軸移動 (左/右轉向)
        
        # 將 y 軸移動轉換為旋轉 (A/D 轉向)
        # 當左箭頭被按下時，v_y > 0，我們想要左轉 (負的 omega_z)
        # 當右箭頭被按下時，v_y < 0，我們想要右轉 (正的 omega_z)
        omega_z = -v_y  # 負號是因為 y 軸和旋轉方向的關係
        
        # 計算輪子速度 (Differential Drive Model)
        # v_L = v_x - omega_z * L / 2
        # v_R = v_x + omega_z * L / 2
        half_track = JETBOT_TRACK_WIDTH / 2.0
        
        # 創建動作 Tensor [v_L, v_R]
        left_vel = v_x - omega_z * half_track
        right_vel = v_x + omega_z * half_track
        
        # 檢測到非零輸入時立即打印（減少頻率）
        if abs(v_x) > 0.01 or abs(omega_z) > 0.01:
            if count % 50 == 0:  # 每50步打印一次，減少輸出
                print(f"[KEYBOARD]: 檢測到輸入! v_x: {v_x}, v_y: {v_y}, omega_z: {omega_z}")
                print(f"[KEYBOARD]: 平滑命令: {commands}")
                print(f"[KEYBOARD]: 輪子速度 - left_vel: {left_vel}, right_vel: {right_vel}")
        
        # 監控鍵盤狀態變化（用於診斷間歇性問題）
        if count > 0:
            # 檢查是否從非零突然變為零（這可能是問題）
            if abs(v_x) < 0.01 and abs(omega_z) < 0.01:
                if np.allclose(keyboard_device._base_command, 0.0):
                    if count % 200 == 0:  # 每200步打印一次狀態
                        print(f"[STATUS]: 當前無按鍵 - _base_command: {keyboard_device._base_command}")
            else:
                # 有按鍵時，檢查 _base_command 是否與 commands 一致
                if count % 50 == 0:  # 每50步檢查一次
                    base_vx = keyboard_device._base_command[0]
                    base_vy = keyboard_device._base_command[1]
                    if abs(base_vx - v_x) > 0.1 or abs(base_vy - v_y) > 0.1:
                        print(f"[WARNING]: 命令不一致! _base_command: {keyboard_device._base_command}")
                        print(f"[WARNING]: commands: {commands}")
        
        # 合併為動作 Tensor
        # 對於單個環境，創建 [1, 2] 形狀的 tensor 以匹配 set_joint_velocity_target 的期望
        action = torch.stack((left_vel, right_vel), dim=0).unsqueeze(0)  # [2] -> [1, 2]
        
        # 調試信息：檢查 action 的形狀
        if count % 100 == 0:  # 每100步打印一次，避免過多輸出
            print(f"[DEBUG]: action shape: {action.shape}")
            print(f"[DEBUG]: v_x: {v_x}, omega_z: {omega_z}")
            print(f"[DEBUG]: left_vel: {left_vel}, right_vel: {right_vel}")


        # 3. 寫入動作、步進模擬
        scene["Jetbot"].set_joint_velocity_target(action)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1
        return True

    # 運行主循環
    setup()
    while simulation_app.is_running():
        update_loop()

def main():
    """Main function to setup and run the simulation."""
    # 物理模擬配置
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([-2.0, 0.0, 1.5], [0.0, 0.0, 0.5])
    
    # 場景配置
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    # 確認 Jetbot 關節數，確保 set_joint_velocity_target 的維度正確 (2 個輪子)
    if scene["Jetbot"].num_joints != 2:
        print("[WARNING]: Jetbot 關節數不是 2，請檢查 USD 模型或 Actuator 配置。")

    print("[INFO]: Setup complete. Press W/S/A/D to move the Jetbot, R to reset.")
    
    # 運行同步模擬循環
    run_simulator(sim, scene)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        simulation_app.close()