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
from isaaclab.managers import EventTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from . import mdp

"""This file defines the foundational configuration for the unplug environment.
It includes the scene (table, socket & plug, ground, light) configuration, MDP settings, and event configurations."""

SOCKET_X = 0.300 #0.60
SOCKET_Y = 0.000 #0.15
SOCKET_Z = 0.361 #0.01

# ─────────────────────────────────────────────────────────────────────────────
# SCENE DEFINITION (TABLE, GROUND, CAMERA, ROBOT (inherited), PLUG & SOCKET (inherited))
# ─────────────────────────────────────────────────────────────────────────────
@configclass
class ObjectTableSceneCfgRGB(InteractiveSceneCfg):
    """Configuration for the scene with a robot and RGB camera."""
    
    # ───────────────────────────── ROBOT ─────────────────────────────────────────
    #robot: will already populated by robot-specific child class
    robot: ArticulationCfg = MISSING # type: ignore

    # end-effector sensor:  will be populated by robot-specific child class
    ee_frame: FrameTransformerCfg = MISSING # type: ignore


    # ───────────────────────────── TABLE ─────────────────────────────────────────
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.25, -0.05, 0.0], rot=[0.707, 0, 0, 0.707]), #previously z_table = 0.011
        spawn=UsdFileCfg(
            usd_path=f"/home/alessio/Downloads/table.usd",
            scale = (1.0, 1.0, 0.5),
            semantic_tags=[("class", "table")],         
        ),
    )


    # ───────────────────────────── GROUND ─────────────────────────────────────────
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.68]), # type: ignore
        spawn=GroundPlaneCfg(
            semantic_tags=[("class", "ground")],
        ), # type: ignore
    )
    

    # ───────────────────────────── SOCKET ─────────────────────────────────────────

    # Rigid body properties for connectors 
    # Not used since custom USD models have rigid body properties defined ***
    # socket_properties = RigidBodyPropertiesCfg(
    #     solver_position_iteration_count=32, #TODO: adjust according need
    #     solver_velocity_iteration_count=1,
    #     max_angular_velocity=1000.0,
    #     max_linear_velocity=1000.0,
    #     max_depenetration_velocity=0.1,
    #     disable_gravity=True, #TODO: Disable gravity to keep it fixed in place, as we are not simulating the full socket but just a rigid body representing it
    # )

    socket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Socket",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(SOCKET_X, SOCKET_Y, SOCKET_Z), rot=(0.0, 0.0, 1.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"/home/alessio/Downloads/workstation.usd",
            scale=(1.0, 1.0, 1.0),
            #rigid_props=socket_properties, #***
            semantic_tags=[("class", "socket")],
        ),
    )

    # ───────────────────────────── PLUG ─────────────────────────────────────────

    # Rigid body properties for connectors
    # Not used since custom USD models have rigid body properties defined **
    # plug_properties = RigidBodyPropertiesCfg(
    #     solver_position_iteration_count=32, #TODO: adjust according need
    #     solver_velocity_iteration_count=1,
    #     max_angular_velocity=1000.0,
    #     max_linear_velocity=1000.0,
    #     max_depenetration_velocity=0.1,
    #     disable_gravity=False,
    # )

    plug = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plug",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(SOCKET_X - 0.017, SOCKET_Y, SOCKET_Z - 0.12125), rot=(0.0, 0.0, 1.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"/home/alessio/Downloads/usbc_plug_new.usd",
            scale=(1.0, 1.0, 1.0),
            #rigid_props=plug_properties, #**
            semantic_tags=[("class", "plug")],
        ),
    )


    # ───────────────────────────── CAMERA ────────────────────────────────────────
    camera: TiledCameraCfg =  TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/zed_mini_cam1",
        update_period=0.0025,
        height=1*376, #originally 240 
        width=1*672, #originally 320
        data_types=["rgb"],
        # spawn=sim_utils.PinholeCameraCfg(
        #     focal_length=3.06,
        #     focus_distance=0.5,
        #     horizontal_aperture=4.8,
        #     vertical_aperture=3.6,
        #     clipping_range=(0.01, 1.0),
        # ),
        spawn=sim_utils.PinholeCameraCfg( #Match the real ZED Mini camera intrinsics
            focal_length=2.8132,
            focus_distance=400.0,
            horizontal_aperture=6.0,#5.376,
            vertical_aperture=None,
            clipping_range=(0.01, 10.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
                        #pos=(-0.10, 0.0, 0.1034-0.08),rot=[0.653, 0.271, 0.271, 0.653] #x -0.1, y = 0
                        pos=(-0.1115, 0.0481, 0.1034-0.0883),
                        rot=[0.6435702, -0.2768679, 0.2724621, -0.6594891] # w, x, y, z quaternion format
        ),
    )


    # ───────────────────────────── LIGHTS ────────────────────────────────────────
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# BASIC SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg -> Set to Abs IK control for our stacking task
    arm_action: mdp.JointPositionActionCfg = MISSING # type: ignore
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# MDP SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class ObservationsCfgRGB:
    """Observation specifications for the MDP."""
    # Define three observation groups: Policy, RGBCameraPolicy, and SubtaskCfg

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""
        
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        
        object = ObsTerm(func=mdp.object_obs)
        plug_pos = ObsTerm(func=mdp.plug_position_in_world_frame)
        socket_pos = ObsTerm(func=mdp.socket_position_in_world_frame)
        plug_quat = ObsTerm(func=mdp.plug_orientation_in_world_frame)
        socket_quat = ObsTerm(func=mdp.socket_orientation_in_world_frame)
        
        image = ObsTerm(func=mdp.image, 
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


@configclass
class EventCfg:
    """Event terms for the MDP."""

    randomize_plug_and_socket = EventTerm(
        func=mdp.randomize_plug_and_socket_unified,
        mode="reset",
        params={
            "socket_cfg": SceneEntityCfg("socket"),
            "plug_cfg": SceneEntityCfg("plug"),
            # "relative_offset": (0.017, 0.0, 0.12125),
            "relative_offset": (-0.017, 0.0, -0.12125),
            "socket_initial_pos": (SOCKET_X, SOCKET_Y, SOCKET_Z),
            "socket_pos_range": {
                # "pos_min": [-0.0, -0.0, 0.0], # Min X, Y offset in meters
                # "pos_max": [0.0, 0.0, 0.0],   # Max X, Y offset in meters
                "pos_min": [0.0, 0.0, 0.0], # Min X, Y offset in meters
                "pos_max": [0.4, 0.2, 0.0],   # Max X, Y offset in meters
            },
        },
    )

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {},      # Empty dict = reset to the exact default USD position
    #         "velocity_range": {},  # Empty dict = reset to zero velocity
    #         "asset_cfg": SceneEntityCfg("plug"),
    #     },
    # )

    # randomize_clight_brightness = EventTerm(
    #     func=mdp.randomize_scene_lighting_domelight,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("light"),
    #         "intensity_range": (1500, 5000),
    #     },
    # )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENVIRONMENT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
@configclass
class UnplugEnvCfgRGB(ManagerBasedRLEnvCfg):
    """Configuration for the unplug environment."""

    # ───────────────────────────── SCENE ─────────────────────────────────────────
    scene: ObjectTableSceneCfgRGB = ObjectTableSceneCfgRGB(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    
    # ───────────────────────────── BASIC SETTINGS ─────────────────────────────────
    actions: ActionsCfg = ActionsCfg()

    # ───────────────────────────── MDP SETTINGS ───────────────────────────────────
    observations: ObservationsCfgRGB = ObservationsCfgRGB()
    terminations: TerminationsCfg = TerminationsCfg()
    events : EventCfg = EventCfg() # type: ignore
    #domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None


    # ─────────────────────────── SIMULATION SETTINGS ─────────────────────────────────
    def __post_init__(self):
        """Post initialization."""

        # Control frequency is 20Hz (50ms), decimation is 5, so simulation runs at 100Hz (10ms)
        self.decimation = 5
        self.episode_length_s = 15.0

        # Simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        # Physics settings
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        #self.sim.physx.friction_correlation_distance = 0.00625

        # Viewer settings
        self.viewer.eye = (1.5, 0.0, 1.2)
        self.viewer.lookat = (0.6, 0.0, 0.1)
