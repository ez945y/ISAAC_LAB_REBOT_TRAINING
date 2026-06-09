# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""
SO-ARM-101 + DAM Safety Guard Demo (Isaac Sim)

Demonstrates the local DAM SafetyGuard package as a real-time safety filter
inside an Isaac Sim control loop.

Every joint target computed by the IK controller passes through DAM's guard
pipeline before reaching the physics engine. Actions that violate position,
velocity, or acceleration limits are clamped in real-time.

Install:
    Install the local DAM package that provides dam.SafetyGuard.

Usage:
    # Keyboard IK control with DAM safety
    python scripts/dam_safety_demo.py

    # Show clamping in action — high-speed keyboard input will be smoothed
    python scripts/dam_safety_demo.py --controller ik
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="SO-ARM-101 + DAM safety demo")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--controller", type=str, default="ik", choices=["ik"],
    help="Controller type. DAM joint-target interception currently supports IK.",
)
parser.add_argument(
    "--stackfile", type=str, default=None,
    help="Path to DAM stackfile (default: bundled soarm_isaac_safety.yaml)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs != 1:
    parser.error("DAM safety demo currently supports --num_envs 1 only.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports (Isaac Sim must be initialized first) ────────────

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from controll_scripts import (
    ControllerFactory,
    ControllerType,
    KeyboardInputDevice,
    SOArm101Config,
)
from controll_scripts.safety import DAMSafetyWrapper


# ── Scene ────────────────────────────────────────────────────────────────

def create_scene_cfg(robot_config: SOArm101Config, for_osc: bool) -> type:

    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0),
        )
        robot = robot_config.get_articulation_cfg(for_osc=for_osc).replace(
            prim_path="{ENV_REGEX_NS}/robot",
        )
        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.03, 0.03, 0.03),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.8, 0.2),
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.015)),
        )

    return SceneCfg


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    controller_type = ControllerType.IK
    for_osc = False

    robot_config = SOArm101Config()

    if not os.path.exists(robot_config.usd_path):
        print(f"[ERROR] USD not found: {robot_config.usd_path}")
        simulation_app.close()
        return

    # ── Simulation setup ────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.4, 0.4, 0.4], [0.0, 0.0, 0.15])
    sim_dt = sim.get_physics_dt()

    SceneCfg = create_scene_cfg(robot_config, for_osc)
    scene = InteractiveScene(SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.prim_path = "/World/Visuals/target_frame"
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    target_marker = VisualizationMarkers(frame_marker_cfg)

    sim.reset()

    robot = scene["robot"]
    robot.update(dt=sim_dt)

    controller = ControllerFactory.create(
        controller_type=controller_type,
        robot=robot,
        robot_config=robot_config,
        device=sim.device,
        num_envs=args_cli.num_envs,
    )

    input_device = KeyboardInputDevice(
        initial_pose=controller.current_ee_pose,
        device=sim.device,
    )

    # ── DAM Safety Guard ────────────────────────────────────────────────
    stackfile = args_cli.stackfile or "soarm_isaac_safety.yaml"
    dam = DAMSafetyWrapper(
        stackfile=stackfile,
        robot_config=robot_config,
        device=sim.device,
        task="default",
    )
    dam.attach_isaac_controller(robot, controller, robot_config)

    print("\n" + "=" * 60)
    print("  SO-ARM-101 + DAM Safety Guard")
    print(f"  Controller: {args_cli.controller.upper()}")
    print(f"  Stackfile:  {stackfile}")
    print("  DAM filters every joint target in real-time.")
    print("  Try fast keyboard inputs — DAM will clamp them.")
    print("=" * 60 + "\n")

    # ── Control loop ────────────────────────────────────────────────────
    cube: RigidObject = scene["cube"]
    cube_initial = torch.tensor(
        [[0.3, 0.0, 0.015, 1.0, 0.0, 0.0, 0.0]], device=sim.device,
    )
    arm_joint_ids = controller._arm_joint_ids
    gripper_joint_ids = controller._gripper_joint_ids
    step_count = 0

    while simulation_app.is_running():
        target_pose, gripper_pos, reset_requested = input_device.update()

        if reset_requested:
            controller.reset()
            input_device.reset_target(controller.current_ee_pose)
            cube.write_root_pose_to_sim(cube_initial)
            cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=sim.device))
            print("[INFO] Scene reset")
            continue

        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w

        # Visualize target
        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
            root_pos_w, root_quat_w, target_pose[:, 0:3], target_pose[:, 3:7],
        )
        target_marker.visualize(target_pos_w, target_quat_w)

        # ── DAM resolver-backed EE safety filter ───────────────────
        current_pos = robot.data.joint_pos[:, arm_joint_ids]
        current_gripper = robot.data.joint_pos[:, gripper_joint_ids[0]].item()
        gripper_target = (
            controller._gripper_lower
            + gripper_pos * (controller._gripper_upper - controller._gripper_lower)
        )
        safe_targets = dam.filter_ee(
            target_pose, current_pos,
            gripper_action=gripper_target,
            gripper_obs=current_gripper,
        )

        # ── Apply safe targets to simulation ───────────────────────
        robot.set_joint_position_target(safe_targets, arm_joint_ids)
        safe_gripper = torch.tensor([[dam.last_safe_gripper]], device=sim.device)
        robot.set_joint_position_target(safe_gripper, gripper_joint_ids)
        robot.write_data_to_sim()

        sim.step()
        robot.update(sim_dt)
        step_count += 1

        # Sync unreachable targets
        current_ee = controller.current_ee_pose
        error = torch.norm(target_pose[0, :3] - current_ee[0, :3]).item()
        if error > 0.15:
            input_device.sync_to_actual(current_ee)

        # Status output
        if step_count % 120 == 0:
            tag = f"\033[93mCLAMP\033[0m" if dam.last_clamped else f"\033[92mPASS \033[0m"
            print(
                f"[DAM {tag}] "
                f"step={step_count:5d}  "
                f"clamp_rate={dam.clamp_rate:.1%}  "
                f"ee_err={error:.4f}  "
                f"grip={gripper_pos:.2f}"
            )

    # ── Cleanup ─────────────────────────────────────────────────────────
    dam.close()
    print(f"\n[DAM] Session complete: {dam.step_count} steps, {dam.clamp_rate:.1%} clamped")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
