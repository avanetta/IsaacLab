# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch
from isaaclab.assets.asset_base import AssetBase
from isaaclab.envs.manager_based_env import ManagerBasedEnv
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
# stack_joint_pos_env_cfg inherits from stack_env_cfg
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg_original import StackEnvCfg, StackEnvCfgRGB
from isaaclab.envs.mdp import events as mdp1

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
# Import the real robot configuration
from isaaclab_assets.robots.franka import FRANKA_PANDA_REAL_ROBOT_CFG, FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_assets.robots.fr3 import FR3_WITH_HAND_CFG

"""Inherits from the StackEnvCfg "base env for stacking task" and specializes the config for joint position control."""

# EventCfg is used to define randomization and reset behaviour of the environment.
@configclass
class EventCfg:
    """Configuration for events."""

    # Standard reset events
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.005,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.1, 0.1), "z": (0.0203, 0.0203), "yaw": (3.1415, 3.1415, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )
    # Cube color randomization - based on RGB base colors
    # Cube 1: Blue -> Orange/brown (90, 30, 0) with variation
    # Cube color randomization - based on RGB base colors
    # Cube color randomization - based on RGB base colors



        
    randomize_clight_brightness = EventTerm(
        func=franka_stack_events.randomize_scene_lighting_domelight,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("light"),
            "intensity_range": (1500, 5000),
        },
    )



    """randomize_cube_2_color = EventTerm(
        func=mdp1.randomize_visual_color,
        mode="reset",
        params={
            "event_name": "randomize_cube_2_color",
            "asset_cfg": SceneEntityCfg("cube_2"),
            "colors": {
                "r": (0.30, 0.40),
                "g": (0.08, 0.15),
                "b": (0.00, 0.05),
            },
            "mesh_name": "",
        },
    )   

    randomize_cube_3_color = EventTerm(
        func=mdp1.randomize_visual_color,
        mode="reset",
        params={
            "event_name": "randomize_cube_3_color",
            "asset_cfg": SceneEntityCfg("cube_3"),
            "colors": {
                "r": (0.00, 0.05),
                "g": (0.25, 0.35),
                "b": (0.08, 0.15),
            },
            "mesh_name": "",
        },
    )"""

@configclass
class FrankaCubeStackEnvCfg(StackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set standard events
        self.events = EventCfg()
        
        # Add domain randomization events if enabled, enter if self.enable_domain_randomization is True
        if self.enable_domain_randomization:
            # Loop through every attribute (event) in the domain randomization config class
            for attr_name in dir(self.domain_randomization):
                if not attr_name.startswith('_') and hasattr(getattr(self.domain_randomization, attr_name), 'func'):
                    # Add each domain randomization event to self.events -> This makes all domain randomization events available in the environment event system
                    setattr(self.events, f"domain_rand_{attr_name}", getattr(self.domain_randomization, attr_name))

        # Fill in the missing components of the base config: Set real franka configuration as robot
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]  # type: ignore
        
        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]  # type: ignore
    
        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]  # type: ignore

        # Set actions for the specific robot type (franka) -> Action space is joint position
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["panda_joint.*"], 
            scale=1, 
            use_default_offset=False # Absolute joint position control, no offset needed
        )

        # Gripper actions
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Rigid body properties of each cube
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Set each stacking cube deterministically
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[1, 0, 0, 0]),    # type: ignore
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_1")],
            ),
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0203], rot=[1, 0, 0, 0]),  # type: ignore
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_2")],
            ),
        )
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 0.0203], rot=[1, 0, 0, 0]),  # type: ignore
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_3")],
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # Defines end-effector frame transformer
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.00, 0.0, 0.1034],  # type: ignore  pos=[0.0, 0.0, 0.1034]
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
                    name="base",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),  # type: ignore
                    ),
                ),
            ],
        )
class FrankaCubeStackEnvCfgRGB(StackEnvCfgRGB):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set standard events
        self.events = EventCfg()
        # Add domain randomization events if enabled, enter if self.enable_domain_randomization is True
        if self.enable_domain_randomization:
            # Loop through every attribute (event) in the domain randomization config class
            for attr_name in dir(self.domain_randomization):
                if not attr_name.startswith('_') and hasattr(getattr(self.domain_randomization, attr_name), 'func'):
                    # Add each domain randomization event to self.events -> This makes all domain randomization events available in the environment event system
                    setattr(self.events, f"domain_rand_{attr_name}", getattr(self.domain_randomization, attr_name))
        # Fill in the missing components of the base config: Set real franka configuration as robot
        self.scene.robot = FRANKA_PANDA_REAL_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]  # type: ignore
        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]  # type: ignore
    
        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]  # type: ignore
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7"
            ],
            scale=1,
            use_default_offset=False
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_joint1", "panda_finger_joint2"],
            open_command_expr={
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
            close_command_expr={
                "panda_finger_joint1": 0.0,
                "panda_finger_joint2": 0.0,
            },
        )


        # Set actions for the specific robot type (franka) -> Action space is joint position
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["panda_joint.*"], 
            scale=1, 
            use_default_offset=False # Absolute joint position control, no offset needed
        )
        """

        # Gripper actions
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["fr3_finger.*"],
            open_command_expr={"fr3_finger_.*": 0.04},
            close_command_expr={"fr3_finger_.*": 0.0},
        )"""

        # Rigid body properties of each cube
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Set each stacking cube deterministically
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[1, 0, 0, 0]),    # type: ignore
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1,1,1),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_1")],
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.00/255.0, 30.0/255.0, 90.0/255.0)  # RGB color
                )
            ),
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0203], rot=[1, 0, 0, 0]),  # type: ignore
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_2")],
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(131.0/255.0, 12.0/255.0, 0.00/255.0)  # RGB color
                ),
            ),
        )
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 0.0203], rot=[1, 0, 0, 0]),  # type: ignore
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_3")],
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=( 30.0/255.0,80.0/255.0, 0.00/255.0)  # RGB color
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # Defines end-effector frame transformer

        """self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fr3_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fr3_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.00, 0.0, 0.1034]),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fr3_link0",fr3_lin
                    name="base",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fr3_hand",
                    name="camera",
                    offset=OffsetCfg(
                        pos=(0.04, 0.0, 0.1034-0.02)
                    ),
                ),
            ],
        )"""
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        # In FrankaCubeStackEnvCfgRGB.__post_init__(), replace the ee_frame configuration:
    
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",  # Base link as reference
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",  # Use panda_link7 instead of fr3_hand
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),  # link7 to TCP: 0.107 + hand offset 0.1034
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_link0", 
                    name="base",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",  # Use panda  _link7 for camera too
                    name="camera",
                    offset=OffsetCfg(pos=[0.04, 0.0, 0.1034 - 0.02]),  # Camera offset from TCP
                ),
            ],
        )


