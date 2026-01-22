"""
一個獨立的、用於機器狗航點巡邏任務的環境設定檔。
直接繼承自 ManagerBasedRLEnvCfg，並整合了所有客製化設定。
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm, TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.utils import configclass
from .assets.navigation import UNITREE_A1_CFG, WALL_CFG
from . import mdp

# -- 航點位置的集中設定 --
WAYPOINT_POSITIONS = [(2.8, 0.0), (2.8, -2.3), (0.6, -4.3), (-1.6, 0.0)]

@configclass
class WaypointSceneCfg(InteractiveSceneCfg):
    """巡邏任務的場景設定。"""
    # 地面設定為平面
    terrain = AssetBaseCfg(
            prim_path="/World/ground",  # prim_path 屬於 AssetBaseCfg
            spawn=sim_utils.GroundPlaneCfg(
                size=(100.0, 100.0),      
                color=(0.5, 0.5, 0.5),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            )
        )

    # 機器人: Unitree A1
    robot: ArticulationCfg = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 360度旋轉光達 (用於障礙物偵測)
    spinning_lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        mesh_prim_paths=["/World/Wall"],
        ray_alignment="base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),
        pattern_cfg=patterns.LidarPatternCfg(
            channels=12, vertical_fov_range=(-15.0, 15.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=0.2
        ),
        debug_vis=True,
        max_distance=100,
    )

    # 接觸力感測器
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # 牆壁障礙物
    wall = WALL_CFG
    
    # 燈光
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.75, 0.75, 0.75)),
    )

    # 動態生成航點視覺化標記
    def __post_init__(self):
        super().__post_init__()
        for i, (x, y) in enumerate(WAYPOINT_POSITIONS, 1):
            setattr(self, f"waypoint_{i}", AssetBaseCfg(
                prim_path=f"/World/Waypoint_{i}",
                spawn=sim_utils.SphereCfg(
                    radius=0.15,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, -0.1)),
            ))


@configclass
class CommandsCfg:
    """指令設定: 使用固定的姿態目標指令，包含航點管理功能。"""

    base_velocity = mdp.GoalPoseCommandCfg(
        asset_name="robot",
        default_goal=[2.8, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0],  # 預設目標 [x, y, z, qw, qx, qy, qz]
        # 提示：resampling_time_range 已棄用，建議改用 EventManager 來觸發重設
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        # 航點管理設定：合併 (x, y, yaw_deg)
        waypoints=[(2.8, 0.0, 0.0), (2.8, -4.3, -90.0), (-1.6, -4.3, -180.0), (-1.6, 0.0, -270.0)],
        tolerance=0.3,
        command_name="base_velocity",
    )

@configclass
class ActionsCfg:
    """動作設定: 關節位置控制。"""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )

@configclass
class ObservationsCfg:
    """觀察設定: 包含本體感覺和航點資訊。"""
    @configclass
    class PolicyCfg(ObsGroup):
        """主要的策略觀察群組 (向量型)"""
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "base_velocity"})
        goal_heading_error_sin_cos = ObsTerm(func=mdp.goal_heading_error_sin_cos, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True # 將所有觀察項拼接成單一向量

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """獎勵設定: 包含運動獎勵和航點獎勵。"""
    staying_alive = RewTerm(func=mdp.alive_reward, weight=0.5)
    track_velocity_exp = RewTerm(
        func=mdp.track_desired_velocity_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "desired_speed": 0.5, "std": 0.5}
    )
    track_position = RewTerm(
        func=mdp.position_progress_reward, 
        weight=3.0, 
        params={
            "command_name": "base_velocity"
        }
    )
    track_heading = RewTerm(
        func=mdp.heading_alignment_reward,
        weight=2.0,  
        params={"command_name": "base_velocity", "heading_coefficient": 0.5}
    )
    
    waypoint_reached = RewTerm(
        func=mdp.waypoint_reached_reward, 
        weight=50.0, 
        params={
            "command_name": "base_velocity"
        }
    )

    height_l2_penalty = RewTerm(func=mdp.height_l2_penalty, weight=-2.0, params={"target_height": 0.38})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-6)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.5)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), # << 確保有 body_names 過濾器
            "threshold": 0.25,
        }
    )

@configclass
class TerminationsCfg:
    """終止條件設定。"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), 
            "threshold": 1.0
        },
    )

@configclass
class EventCfg:
    """事件設定: 控制重置時的隨機化。"""
    # 重置機器人根部狀態
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-0.2, 0.2)},
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0)},
        },
    )
    # 重置關節狀態
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

    # 增加基座質量的隨機化
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={ "asset_cfg": SceneEntityCfg("robot", body_names="trunk"), "mass_distribution_params": (-1.0, 3.0), "operation": "add"},
    )
    reset_waypoint_nav = EventTerm(
        func=mdp.reset_waypoint_navigation,
        mode="reset",
        params={"command_name": "base_velocity"}
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    ##TODO
    pass



@configclass
class WaypointNavigationEnvCfg(ManagerBasedRLEnvCfg):
    """用於航點巡邏任務的獨立環境設定。"""
    
    # 嵌套上面定義的所有設定類別
    scene: WaypointSceneCfg = WaypointSceneCfg(num_envs=4096, env_spacing=0.0)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    
    # 相機視角設定
    viewer: ViewerCfg = ViewerCfg(
        eye=(4.5, 2.0, 5.0),  # 相機位置 (x, y, z)
        lookat=(1.0, -2.0, 0.0),  # 相機目標位置
        origin_type="world"  # 使用世界座標系
    )

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 40.0
        # 模擬器設定
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # 更新感測器更新頻率
        self.scene.spinning_lidar.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt

class WaypointNavigationEnvCfg_PLAY(WaypointNavigationEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 1
        # disable randomization for play
        self.observations.policy.enable_corruption = False
