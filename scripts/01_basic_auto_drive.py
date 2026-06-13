# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot, a deformable object, and a static stand to an Isaac Lab environment."
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

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# --- Configuration for Assets ---

# Configuration for the Jetbot mobile robot
JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    # Set up implicit actuation for the wheels
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

# Configuration for the Deformable Cube (floats in the air)
DEFORMABLE_CUBE_CONFIG = DeformableObjectCfg(
    # Use ENV_REGEX_NS to ensure the asset is spawned in all environments
    prim_path="{ENV_REGEX_NS}/DeformableCube",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.2, 0.2, 0.5),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
    ),
    # Initial state is fixed at a height of 1.0m
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 1.0, 0.5)),
    debug_vis=True,
)

# Configuration for the Static Collision Stand Cuboid (fixed to the world)
STAND_CUBOID_CONFIG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/CollisionStand",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.3, 0.3, 0.3), # A large, thin base
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.0)),
    ),
    # Place the stand slightly above the ground
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -1.0, 0.25)),
)


class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene with Jetbot, a deformable cube, and a static stand."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Assets
    Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    DeformableCube = DEFORMABLE_CUBE_CONFIG.replace(prim_path="{ENV_REGEX_NS}/DeformableCube")
    CollisionStand = STAND_CUBOID_CONFIG.replace(prim_path="{ENV_REGEX_NS}/CollisionStand")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """
    Runs the simulation loop, controlling the Jetbot.
    """
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            # reset the Jetbot's root state to its initial position offset by the environment origins
            root_jetbot_state = scene["Jetbot"].data.default_root_state.clone()
            root_jetbot_state[:, :3] += scene.env_origins

            # copy the default root state to the sim for the jetbot's pose and velocity
            scene["Jetbot"].write_root_pose_to_sim(root_jetbot_state[:, :7])
            scene["Jetbot"].write_root_velocity_to_sim(root_jetbot_state[:, 7:])

            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["Jetbot"].data.default_joint_pos.clone(),
                scene["Jetbot"].data.default_joint_vel.clone(),
            )
            scene["Jetbot"].write_joint_state_to_sim(joint_pos, joint_vel)

            # The deformable cube and stand are automatically reset by the scene
            # since they are defined with initial states.

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting Jetbot state...")

        # drive around logic for Jetbot
        if count % 100 < 75:
            # Drive straight by setting equal wheel velocities
            action = torch.Tensor([[10.0, 10.0]])
        else:
            # Turn by applying different velocities
            action = torch.Tensor([[5.0, -5.0]])

        scene["Jetbot"].set_joint_velocity_target(action)

        # Apply commands and step the simulation
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function to setup and run the simulation."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 0.0, 1.5], [0.0, 0.0, 0.5])

    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
