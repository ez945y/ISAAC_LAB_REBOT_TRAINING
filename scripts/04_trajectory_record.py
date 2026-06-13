# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import sys
import torch
import numpy as np # 需要 numpy 來進行輪間距計算

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from tools.livestream.livestream_support import apply_livestream_defaults
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
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# 引入 Isaac Lab 專用的移動底盤鍵盤控制器
from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg 

# --- Configuration for Assets (保持不變) ---s

# 假設 Jetbot 的輪間距 (Track Width) 為 0.12 米 (實際值請查閱模型文件)
JETBOT_TRACK_WIDTH = 0.12 
# 假設 Jetbot 的最大速度 (Linear Velocity) 為 6.0 m/s (進一步增加速度)
JETBOT_MAX_LINEAR_VEL = 20.0 
# 假設 Jetbot 的最大角速度 (Angular Velocity) 為 18.0 rad/s (進一步增加轉向速度)
JETBOT_MAX_ANGULAR_VEL = 30.0 

JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

DEFORMABLE_CUBE_CONFIG = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/DeformableCube",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.3, 0.3, 0.3),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(1.6, 0.1, 0.85)),  # 放在第一個骨牌前面
    debug_vis=True,
)

# 骨牌配置 - 基礎尺寸
DOMINO_BASE_CONFIG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Domino",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.2, 0.05, 0.1),  # 骨牌形狀：長、薄、高 (x=長度, y=寬度, z=高度)
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # 增加質量讓骨牌更容易倒下
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0)),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,    # 靜摩擦係數
            dynamic_friction=0.6,   # 動摩擦係數
            restitution=0.2         # 增加彈性係數讓骨牌更容易傳遞動量
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.05)),
)

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene with Jetbot, a deformable cube, and a triangle tower of green blocks."""
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    
    
    DeformableCube = DEFORMABLE_CUBE_CONFIG.replace(prim_path="{ENV_REGEX_NS}/DeformableCube")
    
    # 骨牌設置：4個越來越高的綠色骨牌 (順時針旋轉90度，沿x軸排列)
    Domino1 = DOMINO_BASE_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Domino1",
        spawn=DOMINO_BASE_CONFIG.spawn.replace(size=(0.05, 0.2, 0.3)),  # 長、薄、高 (x=長度, y=寬度, z=高度)
        init_state=DOMINO_BASE_CONFIG.InitialStateCfg(pos=(0.5, 0.0, 0.15))  
    )
    Domino2 = DOMINO_BASE_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Domino2", 
        spawn=DOMINO_BASE_CONFIG.spawn.replace(size=(0.05, 0.2, 0.45)),  
        init_state=DOMINO_BASE_CONFIG.InitialStateCfg(pos=(0.8, 0.0, 0.225)) 
    )
    Domino3 = DOMINO_BASE_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Domino3",
        spawn=DOMINO_BASE_CONFIG.spawn.replace(size=(0.05, 0.2, 0.6)),  
        init_state=DOMINO_BASE_CONFIG.InitialStateCfg(pos=(1.1, 0.0, 0.3))  
    )
    Domino4 = DOMINO_BASE_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Domino4",
        spawn=DOMINO_BASE_CONFIG.spawn.replace(size=(0.05, 0.2, 0.75)),  
        init_state=DOMINO_BASE_CONFIG.InitialStateCfg(pos=(1.4, 0.0, 0.375))  
    )

def setup_deformable_constraint(scene: InteractiveScene):
    """
    設置橘色立方體的固定約束，讓它的一個節點固定在空中
    """
    try:
        # 獲取橘色立方體
        deformable_cube = scene["DeformableCube"]
        
        # 獲取默認的節點狀態
        nodal_state = deformable_cube.data.default_nodal_state_w.clone()
        
        # 使用正確的nodal_kinematic_target格式
        nodal_kinematic_target = deformable_cube.data.nodal_kinematic_target.clone()
        
        # 將節點狀態複製到運動學目標的前3個元素（位置）
        nodal_kinematic_target[..., :3] = nodal_state[..., :3]
        
        # 將所有節點設置為自由移動（1.0 = 自由，0.0 = 固定）
        nodal_kinematic_target[..., 3] = 1.0
        
        # 找到頂部節點（z座標最高的節點）
        nodal_positions = nodal_state[0, :, :3]  # 第一個實例的所有節點位置
        top_node_idx = torch.argmax(nodal_positions[:, 2])  # 找到z座標最高的節點索引
        
        # 固定頂部節點
        # 0.0 表示該節點被約束（固定），1.0 表示自由移動
        nodal_kinematic_target[0, top_node_idx, 3] = 0.0  # 固定頂部節點
        
        print(f"[INFO]: 固定節點索引: {top_node_idx}, 位置: {nodal_positions[top_node_idx]}")
        
        # 寫入節點運動學目標到模擬
        deformable_cube.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
        
        print("[INFO]: 橘色立方體固定約束設置完成 - 頂部節點已固定")
        
    except Exception as e:
        print(f"[WARNING]: 無法設置固定約束: {e}")
        print("[INFO]: 橘色立方體將正常掉落")

def update_camera_follow_jetbot(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """
    更新鏡頭跟隨jetbot，設置為第一人稱視角，固定在jetbot屁股上方
    """
    # 獲取jetbot的位置和朝向
    jetbot_pos = scene["Jetbot"].data.root_pos_w[0]  # [x, y, z]
    jetbot_quat = scene["Jetbot"].data.root_quat_w[0]  # [w, x, y, z]
    
    # 使用torch的內建函數將四元數轉換為旋轉矩陣
    import torch
    
    # 將四元數轉換為旋轉矩陣
    # jetbot_quat格式: [w, x, y, z]
    w, x, y, z = jetbot_quat[0], jetbot_quat[1], jetbot_quat[2], jetbot_quat[3]
    
    # 計算旋轉矩陣的元素
    rotation_matrix = torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=torch.float32)
    
    # 獲取前向向量 (jetbot的x軸是前方)
    forward_vector = rotation_matrix[:, 0].cpu().numpy()  # 第一列是x軸方向，先移到CPU
    
    # 計算屁股位置 (jetbot後方)
    # 假設jetbot長度約0.2米，屁股在後方0.1米處
    rear_offset = -0.7  # 負值表示後方
    rear_pos = jetbot_pos.cpu().numpy() + forward_vector * rear_offset
    
    # 設置鏡頭位置：在屁股上方更後方，避免只看到天線
    camera_height = 0.6  # 增加鏡頭高度，抬高視角
    camera_back_offset = 0.4  # 大幅增加鏡頭向後偏移，提供更好的視野
    
    # 計算鏡頭位置
    camera_pos = rear_pos + np.array([0, 0, camera_height]) + forward_vector * (-camera_back_offset)
    
    # 計算鏡頭朝向：看向jetbot前方更遠的地方
    look_at_pos = jetbot_pos.cpu().numpy() + forward_vector * 1.0  # 看向jetbot前方1.0米處，提供更好的視野
    
    # 設置鏡頭視角
    sim.set_camera_view(camera_pos, look_at_pos)

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """
    Runs the synchronous simulation loop, controlling the Jetbot with keyboard input
    using Isaac Lab's Se2Keyboard device.
    """
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # 骨牌名稱列表
    domino_names = ["Domino1", "Domino2", "Domino3", "Domino4"]
    
    # 軌跡記錄系統
    trajectory_recording = False
    trajectory_playing = False
    trajectory_data = {}  # 字典存儲每個物件的軌跡
    trajectory_playback_index = 0
    trajectory_length = 0

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
    
    # 自定義鍵位映射，添加 W/S/A/D/I/O 鍵支持
    # 使用正確的鍵名格式（大寫）
    import numpy as np
    keyboard_device._INPUT_KEY_MAPPING.update({
        # 添加 W/S/A/D 鍵支持
        "W": np.array([1.0, 0.0, 0.0]) * JETBOT_MAX_LINEAR_VEL,      # W: 前進
        "S": np.array([-1.0, 0.0, 0.0]) * JETBOT_MAX_LINEAR_VEL,     # S: 後退
        "A": np.array([0.0, -1.0, 0.0]) * JETBOT_MAX_ANGULAR_VEL,    # A: 左轉 (修正方向)
        "D": np.array([0.0, 1.0, 0.0]) * JETBOT_MAX_ANGULAR_VEL,     # D: 右轉 (修正方向)
        # 軌跡記錄控制
        "I": np.array([0.0, 0.0, 0.0]),  # I: 開始記錄軌跡
        "O": np.array([0.0, 0.0, 0.0]),  # O: 停止記錄軌跡
    })
    
    print(f"[INFO]: 鍵盤配置 - v_x_sensitivity: {keyboard_cfg.v_x_sensitivity}")
    print(f"[INFO]: 鍵盤配置 - v_y_sensitivity: {keyboard_cfg.v_y_sensitivity}")
    print("[INFO]: 請確保 Isaac Sim 視窗有焦點，然後按 W/S/A/D 鍵或 Arrow 鍵")
    print("[INFO]: W/Arrow Up/Numpad8: 前進, S/Arrow Down/Numpad2: 後退")
    print("[INFO]: A/Arrow Left/Numpad4: 左轉, D/Arrow Right/Numpad6: 右轉")
    print("[INFO]: I: 開始記錄軌跡, O: 停止記錄軌跡, R: 重置並清除軌跡")

    def start_trajectory_recording():
        """開始記錄軌跡"""
        nonlocal trajectory_recording, trajectory_data, trajectory_length
        trajectory_recording = True
        trajectory_data = {}
        trajectory_length = 0
        print("[INFO]: 開始記錄軌跡...")
    
    def stop_trajectory_recording():
        """停止記錄軌跡並開始播放"""
        nonlocal trajectory_recording, trajectory_playing, trajectory_playback_index
        trajectory_recording = False
        trajectory_playing = True
        trajectory_playback_index = 0
        print(f"[INFO]: 停止記錄軌跡，開始播放... (軌跡長度: {trajectory_length})")
    
    def record_trajectory_frame():
        """記錄當前幀的物件位置"""
        nonlocal trajectory_data, trajectory_length
        
        # 記錄 Jetbot 位置和朝向
        jetbot_pos = scene["Jetbot"].data.root_pos_w[0].cpu().numpy()
        jetbot_quat = scene["Jetbot"].data.root_quat_w[0].cpu().numpy()
        
        if "Jetbot" not in trajectory_data:
            trajectory_data["Jetbot"] = {"positions": [], "orientations": []}
        
        trajectory_data["Jetbot"]["positions"].append(jetbot_pos.copy())
        trajectory_data["Jetbot"]["orientations"].append(jetbot_quat.copy())
        
        # 記錄骨牌位置
        for domino_name in domino_names:
            if domino_name in scene.rigid_objects:
                domino_pos = scene[domino_name].data.root_pos_w[0].cpu().numpy()
                domino_quat = scene[domino_name].data.root_quat_w[0].cpu().numpy()
                
                if domino_name not in trajectory_data:
                    trajectory_data[domino_name] = {"positions": [], "orientations": []}
                
                trajectory_data[domino_name]["positions"].append(domino_pos.copy())
                trajectory_data[domino_name]["orientations"].append(domino_quat.copy())
        
        trajectory_length += 1
    
    def play_trajectory_frame():
        """播放軌跡的當前幀"""
        nonlocal trajectory_playback_index, trajectory_playing
        
        if trajectory_playback_index >= trajectory_length:
            # 循環播放
            trajectory_playback_index = 0
            print("[INFO]: 軌跡播放完成，重新開始循環播放...")
        
        # 設置 Jetbot 位置和朝向
        if "Jetbot" in trajectory_data and trajectory_playback_index < len(trajectory_data["Jetbot"]["positions"]):
            jetbot_pos = torch.tensor(trajectory_data["Jetbot"]["positions"][trajectory_playback_index])
            jetbot_quat = torch.tensor(trajectory_data["Jetbot"]["orientations"][trajectory_playback_index])
            
            # 創建完整的狀態向量 [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
            root_state = torch.zeros(13)
            root_state[:3] = jetbot_pos
            root_state[3:7] = jetbot_quat
            root_state = root_state.unsqueeze(0)  # [13] -> [1, 13]
            
            scene["Jetbot"].write_root_pose_to_sim(root_state[:, :7])
            scene["Jetbot"].write_root_velocity_to_sim(root_state[:, 7:])
        
        # 設置骨牌位置和朝向
        for domino_name in domino_names:
            if domino_name in trajectory_data and trajectory_playback_index < len(trajectory_data[domino_name]["positions"]):
                domino_pos = torch.tensor(trajectory_data[domino_name]["positions"][trajectory_playback_index])
                domino_quat = torch.tensor(trajectory_data[domino_name]["orientations"][trajectory_playback_index])
                
                root_state = torch.zeros(13)
                root_state[:3] = domino_pos
                root_state[3:7] = domino_quat
                root_state = root_state.unsqueeze(0)
                
                scene[domino_name].write_root_pose_to_sim(root_state[:, :7])
                scene[domino_name].write_root_velocity_to_sim(root_state[:, 7:])
        
        trajectory_playback_index += 1

    def setup():
        nonlocal count, trajectory_recording, trajectory_playing, trajectory_data, trajectory_playback_index, trajectory_length
        count = 0
        
        # 清除軌跡數據
        trajectory_recording = False
        trajectory_playing = False
        trajectory_data = {}
        trajectory_playback_index = 0
        trajectory_length = 0
        
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
        
        # 重設所有骨牌到原始位置和狀態
        for domino_name in domino_names:
            if domino_name in scene.rigid_objects:
                # 使用 RigidObject 的重置方法
                root_state = scene[domino_name].data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                scene[domino_name].write_root_pose_to_sim(root_state[:, :7])
                scene[domino_name].write_root_velocity_to_sim(root_state[:, 7:])
                print(f"[INFO]: 重設 {domino_name} 到原始位置: {root_state[0, :3]}")
            else:
                print(f"[WARNING]: 無法找到 {domino_name} 在 scene.rigid_objects 中")

        # 重設鍵盤裝置
        keyboard_device.reset()
        # 清除內部緩衝區
        scene.reset()
        
        # 設置橘色立方體的固定約束
        setup_deformable_constraint(scene)
        
        print("[INFO]: Resetting Jetbot state...")
        print("[INFO]: 重設所有骨牌到原始位置")
        print("[INFO]: 注意：重置會清除所有按鍵狀態，請重新按下按鍵")

    # 註冊鍵盤回調函式
    # Se2Keyboard 的 add_callback 已經處理了鍵盤事件訂閱
    keyboard_device.add_callback("R", setup)
    keyboard_device.add_callback("I", start_trajectory_recording)
    keyboard_device.add_callback("O", stop_trajectory_recording)


    def update_loop():
        nonlocal count, trajectory_recording, trajectory_playing
        
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
        
        
        # 保存當前狀態
        update_loop.prev_commands = commands.clone()
        
        # 調試信息：檢查 commands 的形狀和值
        if count % 100 == 0:  # 每100步打印一次，避免過多輸出
            print(f"[DEBUG]: commands values: {commands}")
        
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
        
        # 合併為動作 Tensor
        # 對於單個環境，創建 [1, 2] 形狀的 tensor 以匹配 set_joint_velocity_target 的期望
        action = torch.stack((left_vel, right_vel), dim=0).unsqueeze(0)  # [2] -> [1, 2]

        # 3. 軌跡記錄和播放邏輯
        if trajectory_recording:
            # 記錄當前幀
            record_trajectory_frame()
        elif trajectory_playing:
            # 播放軌跡
            play_trajectory_frame()
            # 在播放模式下，不執行鍵盤控制
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            update_camera_follow_jetbot(sim, scene)
            count += 1
            return True
        
        # 4. 寫入動作、步進模擬
        scene["Jetbot"].set_joint_velocity_target(action)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        
        # 5. 更新第一人稱鏡頭跟隨jetbot
        update_camera_follow_jetbot(sim, scene)
        
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
    # 移除固定鏡頭設置，改為動態跟隨
    # sim.set_camera_view([-2.0, 0.0, 1.5], [0.0, 0.0, 0.5])
    
    # 場景配置
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    # 確認 Jetbot 關節數，確保 set_joint_velocity_target 的維度正確 (2 個輪子)
    if scene["Jetbot"].num_joints != 2:
        print("[WARNING]: Jetbot 關節數不是 2，請檢查 USD 模型或 Actuator 配置。")

    print("[INFO]: Setup complete. Press W/S/A/D to move the Jetbot, R to reset.")
    print("[INFO]: 可變形方塊塔已設置完成 - 7個可變形方塊（6個綠色三角形塔 + 1個橘色懸掛立方體）")
    print("[INFO]: 橘色立方體的頂部節點已固定，其他部分可以自由變形和擺動")
    
    # 運行同步模擬循環
    run_simulator(sim, scene)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        simulation_app.close()
