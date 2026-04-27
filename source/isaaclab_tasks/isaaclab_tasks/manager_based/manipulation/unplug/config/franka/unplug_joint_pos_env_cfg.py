# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.manipulation.unplug import mdp
from isaaclab_tasks.manager_based.manipulation.unplug.unplug_env_cfg import UnplugEnvCfgRGB

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG




@configclass
class EventCfg:
    """Configuration for events."""

    

    
    
# ─────────────────────────────────────────────────────────────────────────────
# TERMINATIONS
# ─────────────────────────────────────────────────────────────────────────────
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(
        func=mdp.plug_successfully_unplugged,
        params={
            "unplug_distance": 0.15,
            "robot_cfg": SceneEntityCfg("robot"),
            "plug_cfg": SceneEntityCfg("plug"),
        }
    )

    plug_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={
            "minimum_height": 0.0, 
            "asset_cfg": SceneEntityCfg("plug")
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCENE, ACTION & TRANSFORMATIONS SPECIFIC FOR FRANKA ARM
# ─────────────────────────────────────────────────────────────────────────────
@configclass
class FrankaUnplugJointPosEnvCfgRGB(UnplugEnvCfgRGB):
    """Configuration for the Franka Unplug Environment."""

    #events: EventCfg = EventCfg()
    terminations:TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()


        # ─────────────────────────────────────────────────────────────────────────────
        # ROBOT DEFINITION
        # ─────────────────────────────────────────────────────────────────────────────
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") 
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # ─────────────────────────────────────────────────────────────────────────────
        # EVENTS (ADDED TO INHERITED EVENTS)
        # ─────────────────────────────────────────────────────────────────────────────
        self.events.init_franka_arm_pose = EventTerm(
            func=mdp.set_default_joint_pose,
            mode="reset",
            params={
                "default_pose": [-0.7196815, 0.51913553, -0.1370281, -1.9037826, 0.80270785, 1.04565, 1.1946661, 0.04, 0.04], #15x with light with position change
                #"default_pose": [-0.28446823, 1.0074301, -0.2614324, -1.3803188, 1.3432448, 1.1092516, 1.4370606, 0.04, 0.04], #30x with light
            },
        )
        self.events.randomize_franka_joint_state = EventTerm(
            func=mdp.randomize_joint_by_gaussian_offset,
            mode="reset",
            params={
                "mean": 0.0,
                "std": 0.05,
                "asset_cfg": SceneEntityCfg("robot"),
                },
        )

        # self.events.randomize_robot_task_space = EventTerm(
        #         func=mdp.randomize_joint_by_task_space_gaussian,
        #         mode="reset",
        #         params={
        #             "mean": (0.0, -0.0, 0.0),
        #             "std": (0.05, 0.05, 0.05), # X, Y, Z std for task space gaussian noise in meters
        #             "asset_cfg": SceneEntityCfg("robot"),
        #         },
        #     )

        
        # self.events.randomize_robot_base = EventTerm(
        #     func=mdp.randomize_robot_base_position,
        #     mode="reset", # Triggers every time the environment resets
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot"),
        #         "x_range": (-0.1, 0.1),
        #         "y_range": (-0.1, 0.1),
        #         "z_range": (0.0, 0.0),
        #     },
        # )
        
        
        # ─────────────────────────────────────────────────────────────────────────────
        # ROBOT ACTION DEFINITION
        # ─────────────────────────────────────────────────────────────────────────────

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=1, use_default_offset=False # Absolute joint position control, no offset needed
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_joint1", "panda_finger_joint2"],
            open_command_expr={
                    "panda_finger_joint1": 0.04, 
                    "panda_finger_joint2": 0.04
            },
            close_command_expr={
                    "panda_finger_joint1": 0.00, 
                    "panda_finger_joint2": 0.00
            },
        )

        # utilities for gripper status check
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005


        # ─────────────────────────────────────────────────────────────────────────────
        # TRANSFORMATIONS
        # ─────────────────────────────────────────────────────────────────────────────

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"


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
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/panda_hand",  # Use panda  _link7 for camera too
                #     name="camera",
                #     offset=OffsetCfg(pos=[0.04, 0.0, 0.1034 - 0.02]),  # Camera offset from TCP
                # ),
            ],
        )

