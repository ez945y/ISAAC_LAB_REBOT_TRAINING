# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
SO-ARM-101 機器人控制範例腳本

使用 robot_control 庫進行模組化控制。
支持 IK 和 OSC 兩種控制器。
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Add parent directory to path for controll_scripts module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="SO-ARM-101 control demo")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--controller",
    type=str,
    default="ik",
    choices=["ik", "osc"],
    help="controller type: ik (Differential IK) or osc (Operational Space)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Auto-configure WebRTC livestream (publicIp + dynamic resize) when --livestream is set.
from tools.livestream.livestream_support import apply_livestream_defaults
apply_livestream_defaults(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# Import controll_scripts library
from controll_scripts import (
    ControllerFactory,
    ControllerType,
    KeyboardInputDevice,
    SOArm101Config,
)


# ========================================
# Scene configuration
# ========================================

def create_scene_cfg(robot_config: SOArm101Config, for_osc: bool) -> type:
    """Create scene configuration dynamically"""
    
    class DynamicSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0),
        )
        robot = robot_config.get_articulation_cfg(for_osc=for_osc).replace(
            prim_path="{ENV_REGEX_NS}/robot"
        )
        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.03, 0.03, 0.03),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.015)),
        )
    
    return DynamicSceneCfg


def main():
    # Select controller type
    if args_cli.controller == "ik":
        controller_type = ControllerType.IK
        for_osc = False
    else:
        controller_type = ControllerType.OSC if args_cli.controller == "osc" else ControllerType.IK
    
    print(f"\n使用控制器: {args_cli.controller.upper()}")
    
    # Create robot configuration
    robot_config = SOArm101Config()
    
    # Check USD file
    if not os.path.exists(robot_config.usd_path):
        print(f"[ERROR]: USD 不存在: {robot_config.usd_path}")
        simulation_app.close()
        return
    
    # Initialize simulation
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.4, 0.4, 0.4], [0.0, 0.0, 0.15])
    sim_dt = sim.get_physics_dt()
    
    # Create scene
    SceneCfg = create_scene_cfg(robot_config, for_osc)
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Create target marker (use coordinate frame to display direction)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.prim_path = "/World/Visuals/target_frame"
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # 縮小座標框架
    target_marker = VisualizationMarkers(frame_marker_cfg)
    
    # Reset simulation
    sim.reset()
    
    # Get robot
    robot = scene["robot"]
    robot.update(dt=sim_dt)
    
    # Create controller using factory
    controller = ControllerFactory.create(
        controller_type=controller_type,
        robot=robot,
        robot_config=robot_config,
        device=sim.device,
        num_envs=args_cli.num_envs,
    )
    
    # 創建輸入設備
    input_device = KeyboardInputDevice(
        initial_pose=controller.current_ee_pose,
        device=sim.device,
    )
    
    # 保存初始狀態用於重置
    cube: RigidObject = scene["cube"]
    cube_initial_pose = torch.tensor([[0.3, 0.0, 0.015, 1.0, 0.0, 0.0, 0.0]], device=sim.device)
    
    print(f"\n初始 EE 位置: {controller.current_ee_pose[0, :3].cpu().numpy()}")
    print("開始控制迴圈...\n")
    
    step_count = 0
    
    while simulation_app.is_running():
        # 更新輸入
        target_pose, gripper_pos, reset_requested = input_device.update()
        
        # 處理重置
        if reset_requested:
            controller.reset()
            input_device.reset_target(controller.current_ee_pose)
            
            # 重置立方體
            cube.write_root_pose_to_sim(cube_initial_pose)
            cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=sim.device))
            
            print("[INFO]: 場景已重置")
            continue
        
        # 更新目標標記（位置 + 方向）
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        # 將目標姿態轉換到世界座標系
        import isaaclab.utils.math as math_utils
        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
            root_pos_w, root_quat_w, target_pose[:, 0:3], target_pose[:, 3:7]
        )
        target_marker.visualize(target_pos_w, target_quat_w)
        
        # 計算並應用控制
        controller.compute(target_pose, gripper_pos)
        
        # 檢查誤差，當目標不可達時同步到實際位置
        current_pose = controller.current_ee_pose
        error = torch.norm(target_pose[0, 0:3] - current_pose[0, 0:3]).item()
        if error > 0.15:  # 誤差超過 15cm，視為不可達
            input_device.sync_to_actual(current_pose)
        
        # 更新目標標記（使用同步後的位置和方向）
        synced_pose = input_device.target_pose
        synced_pos_w, synced_quat_w = math_utils.combine_frame_transforms(
            root_pos_w, root_quat_w, synced_pose[:, 0:3], synced_pose[:, 3:7]
        )
        target_marker.visualize(synced_pos_w, synced_quat_w)
        
        # 步進模擬
        sim.step()
        robot.update(sim_dt)
        step_count += 1
        
        # 調試輸出
        if step_count % 200 == 0:
            print(
                f"目標: [{target_pose[0, 0]:.3f}, {target_pose[0, 1]:.3f}, {target_pose[0, 2]:.3f}] | "
                f"實際: [{current_pose[0, 0]:.3f}, {current_pose[0, 1]:.3f}, {current_pose[0, 2]:.3f}] | "
                f"誤差: {error:.4f} | 夾爪: {gripper_pos:.2f}"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
