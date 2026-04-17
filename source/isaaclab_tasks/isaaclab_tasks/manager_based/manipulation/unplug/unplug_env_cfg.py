# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import EventTermCfg as EventTerm
#from isaaclab_tasks.manager_based.manipulation.unplug.mdp import franka_unplug_events
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from . import mdp

"""This file defines the foundational configuration for the unplug environment.
It includes the scene configuration, MDP settings, and event configurations."""


##
# Scene definition -> defines the scene with a table, robot "Franka Emika", and objects
##
@configclass
class ObjectTableSceneCfgRGB(InteractiveSceneCfg):
    """Configuration for the scene with a robot and RGB camera."""

    # robots: will be populated by the derived env cfg
    robot: ArticulationCfg = MISSING # type: ignore
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING # type: ignore
    # USB connectors: will be populated by the derived env cfg
    plug: RigidObjectCfg = MISSING # type: ignore
    socket: RigidObjectCfg = MISSING # type: ignore

    # SeattleTable
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.0], rot=[0.707, 0, 0, 0.707]), #previously z_table = 0.011
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    #         scale=(1, 1, 0.6762),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.0, 0.0, 0.0)  # real table color
    #         ),
    #     ),
    # )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.25, -0.05, 0.0], rot=[0.707, 0, 0, 0.707]), #previously z_table = 0.011
        spawn=UsdFileCfg(
            usd_path=f"/home/alessio/Downloads/table.usd",
            scale = (1.0, 1.0, 0.5),            
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.68]), # type: ignore
        spawn=GroundPlaneCfg(), # type: ignore
    )
    # Camera attached to robot link
    camera: TiledCameraCfg =  TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/zed_mini_cam1",
        update_period=0.0025,
        height=1*376, #originally 240 
        width=1*672, #originally 320
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.06,
            focus_distance=0.5,
            horizontal_aperture=4.8,
            vertical_aperture=3.6,
            clipping_range=(0.01, 1.0),
        ),
        # spawn=sim_utils.PinholeCameraCfg( #Match the real ZED Mini camera intrinsics
        #     focal_length=2.8132,
        #     focus_distance=0.5,
        #     horizontal_aperture=4.8,
        #     vertical_aperture=3.6,
        #     clipping_range=(0.01, 1.0),
        # ),
        offset=TiledCameraCfg.OffsetCfg(
                        #pos=(-0.10, 0.0, 0.1034-0.08),rot=[0.653, 0.271, 0.271, 0.653] #x -0.1, y = 0
                        pos=(-0.1115, 0.0481, 0.1034-0.0883),
                        rot=[0.6435702, -0.2768679, 0.2724621, -0.6594891] # w, x, y, z quaternion format
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg -> Set to Abs IK control for our stacking task
    arm_action: mdp.JointPositionActionCfg = MISSING # type: ignore
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING # type: ignore

# Define three observation groups: Policy, RGBCameraPolicy, and SubtaskCfg
@configclass
class ObservationsCfgRGB:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""
        
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        object = ObsTerm(func=mdp.object_obs)
        plug_pos = ObsTerm(func=mdp.plug_position_in_world_frame)
        socket_pos = ObsTerm(func=mdp.socket_position_in_world_frame)
        plug_quat = ObsTerm(func=mdp.plug_orientation_in_world_frame)
        socket_quat = ObsTerm(func=mdp.socket_orientation_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        

        image = ObsTerm(
            func=mdp.image,
          params={"sensor_cfg": SceneEntityCfg("camera"), "normalize": False},
       )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False


    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""
        
        # Subtask 1: Grasp the plug
        grasp_plug = ObsTerm(
            func=mdp.object_grasped, # TODO
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("plug"),
            },
        )
        
        # Subtask 2: Pull plug away from socket (unplugged)
        unplug_complete = ObsTerm(
            func=mdp.plug_unplugged,  # TODO
            params={
                "plug_cfg": SceneEntityCfg("plug"),
                "socket_cfg": SceneEntityCfg("socket"),
                "unplug_distance_threshold": 0.10,  # 10cm away = unplugged
            },
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # Create observation groups "key value pairs"
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

# The domain randomization events are applied during environment resets
@configclass 
class DomainRandomizationCfg:
    """Domain randomization settings for the unplug environment."""

    # randomize_actuator_gains = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "stiffness_distribution_params": (0.7, 1.1),
    #         "damping_distribution_params": None,
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_robot_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "mass_distribution_params": (0.98, 1.02),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_control_latency = EventTerm(
    #     func=franka_unplug_events.randomize_control_latency,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "latency_steps_range": (0, 1),
    #     },
    # )

##
# Main environment configuration
##
@configclass
class UnplugEnvCfgRGB(ManagerBasedRLEnvCfg):
    """Configuration for the unplug environment."""

    # Scene settings
    scene: ObjectTableSceneCfgRGB = ObjectTableSceneCfgRGB(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfgRGB = ObservationsCfgRGB()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    # Domain randomization settings
    #domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()

    # Unused managers -> These are not used in imitation learning pipelines
    commands = None
    rewards = None
    events = None
    curriculum = None

    # Configuration for viewing and interacting with the environment through an XR device.
    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        # Control frequency is 20Hz (50ms), decimation is 5, so simulation runs at 100Hz (10ms)
        self.decimation = 5
        self.episode_length_s = 15.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        # Enable domain randomization by default (can be disabled in child configs)
        self.enable_domain_randomization = False

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # viewer settings
        self.viewer.eye = (1.5, 0.0, 1.2)
        self.viewer.lookat = (0.6, 0.0, 0.1)
