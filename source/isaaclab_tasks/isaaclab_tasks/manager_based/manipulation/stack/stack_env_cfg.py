# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from . import mdp

"""This file defines the foundational configuration for the stack environment.
It includes the scene configuration, MDP settings, and event configurations."""


##
# Scene definition -> defines the scene with a table, robot "Franka Emika", and objects
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot-configuration and end-effector frames
    """

    # robots: will be populated by the derived env cfg
    robot: ArticulationCfg = MISSING # type: ignore
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING # type: ignore

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        # Add table offset in z direction to match the real robots configuration -> Offset coordinate frame by 0.011m
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.011], rot=[0.707, 0, 0, 0.707]), # type: ignore
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1, 1, 0.6762),  # (x_scale, y_scale, z_scale)
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
        height=1*240,
        width=1*320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.06,
            focus_distance=50.0,
            horizontal_aperture=4.8,
            vertical_aperture=3.6,
            clipping_range=(0.01, 1.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
                        pos=(-0.10, 0.0, 0.1034-0.08),rot=[0.653, 0.271, 0.271, 0.653]
        ),
    )
    camera2: TiledCameraCfg =  TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0/zed_mini_cam2",
        update_period=0.0025,
        height=1*240,
        width=1*320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,             # 4 mm lens
            focus_distance=10.0,          # meters, adjust based on scene
            horizontal_aperture=4.8,      # sensor width in mm
            vertical_aperture=3.6,        # sensor height in mm
            clipping_range=(0.01, 1.5)  # near/far plane
        ),
        offset=TiledCameraCfg.OffsetCfg(
                        pos=[0.7, -0.4, 0.40],  # adjust height as neededpos=[1.0, 0.0, 0.25],#
                      rot=[0.462, -0.800, -0.331, 0.191] # rot= [0.5, -0.5, -0.5, 0.5]#

        ),
    )

    camera3: TiledCameraCfg =  TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0/zed_mini_cam3",
        update_period=0.0025,
        height=1*240,
        width=1*320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,             # 4 mm lens
            focus_distance=10.0,          # meters, adjust based on scene
            horizontal_aperture=4.8,      # sensor width in mm
            vertical_aperture=3.6,        # sensor height in mm
            clipping_range=(0.01, 1.5)  # near/far plane
        ),
        offset=TiledCameraCfg.OffsetCfg(
                        pos=[0.5, -0.75, 0.25],#[1.0, -0.5, 0.60],  # adjust height as needed
                        rot=[0.7071, -0.7071, 0.0, 0.0]#0.524, -0.679, -0.500, 0.277]

        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
@configclass
class ObjectTableSceneCfgRGB(InteractiveSceneCfg):
    """Configuration for the scene with a robot and RGB camera."""

    # robots: will be populated by the derived env cfg
    robot: ArticulationCfg = MISSING # type: ignore
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING # type: ignore

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.011], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1, 1, 0.6762),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0)  # real table color
            ),
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
        height=1*240,
        width=1*320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.06,
            focus_distance=50.0,
            horizontal_aperture=4.8,
            vertical_aperture=3.6,
            clipping_range=(0.01, 1.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
                        pos=(-0.10, 0.0, 0.1034-0.08),rot=[0.653, 0.271, 0.271, 0.653]
        ),
    )
    camera2: TiledCameraCfg =  TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0/zed_mini_cam2",
        update_period=0.0025,
        height=1*240,
        width=1*320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,             # 4 mm lens
            focus_distance=10.0,          # meters, adjust based on scene
            horizontal_aperture=4.8,      # sensor width in mm
            vertical_aperture=3.6,        # sensor height in mm
            clipping_range=(0.01, 1.5)  # near/far plane
        ),
        offset=TiledCameraCfg.OffsetCfg(
                        pos=[0.7, -0.4, 0.40],  # adjust height as neededpos=[1.0, 0.0, 0.25],#
                      rot=[0.462, -0.800, -0.331, 0.191] # rot= [0.5, -0.5, -0.5, 0.5]#

        ),
    )

    camera3: TiledCameraCfg =  TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0/zed_mini_cam3",
        update_period=0.0025,
        height=1*240,
        width=1*320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,             # 4 mm lens
            focus_distance=10.0,          # meters, adjust based on scene
            horizontal_aperture=4.8,      # sensor width in mm
            vertical_aperture=3.6,        # sensor height in mm
            clipping_range=(0.01, 1.5)  # near/far plane
        ),
        offset=TiledCameraCfg.OffsetCfg(
                        pos=[0.5, -0.75, 0.25],#[1.0, -0.5, 0.60],  # adjust height as needed
                        rot=[0.7071, -0.7071, 0.0, 0.0]#0.524, -0.679, -0.500, 0.277]

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
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        # Implement observation noise -> Simulate state estimation errors
        object = ObsTerm(func=mdp.object_obs_with_noise,
                        params={"position_noise_std": 0.0075 , "orientation_noise_std": 0.00})
        eef_pos = ObsTerm(func=mdp.ee_frame_pos_with_noise,
                        params={"noise_std": 0.005})
        eef_quat = ObsTerm(func=mdp.ee_frame_quat_with_noise,
                        params={"noise_std": 0.001})
        gripper_pos = ObsTerm(func=mdp.gripper_pos_with_noise,
                        params={"noise_std": 0.02})

        # Keep other observations as they are
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
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

        # cube_1: blue, cube_2: red, cube_3: green
        grasp_1 = ObsTerm(
            func=mdp.object_grasped, # Checks if cube_1 is grasped by the robot
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        stack_1 = ObsTerm(
            func=mdp.object_stacked, # Checks if cube_2 is stacked on cube_1
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped, # Checks if cube_3 is grasped by the robot
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
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
class ObservationsCfgRGB:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        # Implement observation noise -> Simulate state estimation errors
        #object = ObsTerm(func=mdp.object_obs_with_noise,
        #               params={"position_noise_std": 0.0075 , "orientation_noise_std": 0.01})
        #eef_pos = ObsTerm(func=mdp.ee_frame_pos_with_noise,
        #                params={"noise_std": 0.000})
        #eef_quat = ObsTerm(func=mdp.ee_frame_quat_with_noise,
        #                params={"noise_std": 0.000})
        #gripper_pos = ObsTerm(func=mdp.gripper_pos_with_noise,
        #                params={"noise_std": 0.0000})
        
        
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        object = ObsTerm(func=mdp.object_obs)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        

        image = ObsTerm(
            func=mdp.image,
          params={"sensor_cfg": SceneEntityCfg("camera"), "normalize": False},
       )
        image2 = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera2"), "normalize": False},
        )
        image3 = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera3"), "normalize": False},
        )
        """ 
        img_features = ObsTerm(
            func=mdp.image_features_concat,
            params={
                "sensor_cfg1": SceneEntityCfg("camera"),
                "sensor_cfg2": SceneEntityCfg("camera2"),
                "model_name": "resnet18",
            }
        ) 
        img_features = ObsTerm(
            func=mdp.image_concat,
            params={
                "sensor_cfg1": SceneEntityCfg("camera"),
                "sensor_cfg2": SceneEntityCfg("camera2")
            )"""
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""
        #image = ObsTerm(
        #    func=mdp.image_features,
        #    params={"sensor_cfg": SceneEntityCfg("camera"),"data_type": "rgb", "model_name": "resnet18"},
        #)
        #image2 = ObsTerm(
        #    func=mdp.image_features,
        #    params={"sensor_cfg": SceneEntityCfg("camera2"),"data_type": "rgb", "model_name": "resnet18"},
        #)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        # cube_1: blue, cube_2: red, cube_3: green
        grasp_1 = ObsTerm(
            func=mdp.object_grasped, # Checks if cube_1 is grasped by the robot
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        stack_1 = ObsTerm(
            func=mdp.object_stacked, # Checks if cube_2 is stacked on cube_1
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped, # Checks if cube_3 is grasped by the robot
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )
        stack_2 = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_3"),
                "lower_object_cfg": SceneEntityCfg("cube_2"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # Create observation groups "key value pairs"
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()
# The domain randomization events are applied during environment resets
@configclass 
class DomainRandomizationCfg:
    """Domain randomization settings for the stacking environment."""

    # Use IsaacLab's built-in actuator gains randomization
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,  # Built-in function
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.7,1.1),  # ±70% variation
            "damping_distribution_params": None,
            "operation": "scale",  # Scale the base values
            "distribution": "uniform",
        },
    )

    # Use IsaacLab's built-in mass randomization
    randomize_robot_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # Built-in function
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.98,1.02),  # ±5% mass variation
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Randomize cube masses using built-in function
    randomize_cube_1_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_1"),
            "mass_distribution_params": (0.2, 0.8),  # ±20% mass variation
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_cube_2_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_2"),
            "mass_distribution_params": (0.2, 0.8),  # ±20% mass variation
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_cube_3_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_3"),
            "mass_distribution_params": (0.2, 0.8),  # ±20% mass variation
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Randomize control latency using a custom function
    randomize_control_latency = EventTerm(
        func=franka_stack_events.randomize_control_latency,  # Custom function for control latency
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "latency_steps_range": (0, 1),  # 0-3 timesteps delay (0-150ms at 20Hz)
        },
    )
    """randomize_visual_color= EventTerm(
                func=mdp.randomize_visual_color,         # your class
                # parameters passed to __init__ of your class
                params={
                    "asset_cfg": SceneEntityCfg("cube_1"),
                    "colors": {                      # random RGB ranges
                        "r": (0.0, 1.0),
                        "g": (0.0, 1.0),
                        "b": (0.0, 1.0),
                    },
                    "event_name": "COLOR_EVENT",
                    "mesh_name": "body_0/mesh",
                },
                mode="reset",                        # trigger at reset
            )
    randomize_visual_color= EventTerm(
                func=mdp.randomize_visual_color,         # your class
                # parameters passed to __init__ of your class
                params={
                    "asset_cfg": SceneEntityCfg("cube_2"),
                    "colors": {                      # random RGB ranges
                        "r": (0.0, 1.0),
                        "g": (0.0, 1.0),
                        "b": (0.0, 1.0),
                    },
                    "event_name": "COLOR_EVENT",
                    "mesh_name": "body_0/mesh",
                },
                mode="reset",                        # trigger at reset
            )
    randomize_visual_color= EventTerm(
                func=mdp.randomize_visual_color,         # your class
                # parameters passed to __init__ of your class
                params={
                    "asset_cfg": SceneEntityCfg("cube_3"),
                    "colors": {                      # random RGB ranges
                        "r": (0.0, 1.0),
                        "g": (0.0, 1.0),
                        "b": (0.0, 1.0),
                    },
                    "event_name": "COLOR_EVENT",
                    "mesh_name": "body_0/mesh",
                },
                mode="reset",                        # trigger at reset
            )"""
 

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")}
    )

    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")}
    )

    cube_3_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_3")}
    )
    # Success condition: all cubes stacked -> Very important for annotate_demos.py
    success = DoneTerm(func=mdp.cubes_stacked)

# Define the main stacking environment configuration -> Inherits from ManagerBasedRLEnvCfg "Manager based workflow"
@configclass
class StackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    # Domain randomization settings
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()

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
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2

        # Enable domain randomization by default (can be disabled in child configs)
        self.enable_domain_randomization = True

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
@configclass
class StackEnvCfgRGB(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: ObjectTableSceneCfgRGB = ObjectTableSceneCfgRGB(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfgRGB = ObservationsCfgRGB()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    # Domain randomization settings
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()

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
        self.episode_length_s = 20.0
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
