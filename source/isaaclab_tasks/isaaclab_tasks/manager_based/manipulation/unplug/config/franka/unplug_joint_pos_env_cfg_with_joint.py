# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.unplug import mdp
#from isaaclab_tasks.manager_based.manipulation.unplug.mdp import franka_unplug_events as events
from isaaclab_tasks.manager_based.manipulation.unplug.unplug_env_cfg import UnplugEnvCfgRGB

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

# socket_x = 0.60
# socket_y = 0.15
# socket_z = 0.01
SOCKET_X = 0.300
SOCKET_Y = 0.000
SOCKET_Z = 0.361

@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=mdp.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [-0.7196815, 0.51913553, -0.1370281, -1.9037826, 0.80270785, 1.04565, 1.1946661, 0.04, 0.04], #15x with light with position change
            #"default_pose": [-0.28446823, 1.0074301, -0.2614324, -1.3803188, 1.3432448, 1.1092516, 1.4370606, 0.04, 0.04], #30x with light
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=mdp.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # Randomly position the socket with offset in range & place plug relative to socket
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
                # "pos_min": [-0.05, -0.05, 0.0], # Min X, Y offset in meters
                # "pos_max": [0.0, 0.05, 0.0],   # Max X, Y offset in meters
                # "pos_min": [-0.05, -0.05, 0.0], # Min X, Y offset in meters
                # "pos_max": [0.05, 0.05, 0.0],   # Max X, Y offset in meters
                # "pos_min": [-0.0, -0.0, 0.0], # Min X, Y offset in meters
                # "pos_max": [0.0, 0.0, 0.0],   # Max X, Y offset in meters
                "pos_min": [0.0, 0.0, 0.0], # Min X, Y offset in meters
                "pos_max": [0.4, 0.2, 0.0],   # Max X, Y offset in meters
            },
        },
    )

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
        params={"minimum_height": 0.0, "asset_cfg": SceneEntityCfg("plug")}
    )


##
# Main environment configuration
##
@configclass
class FrankaUnplugJointPosEnvCfgRGB(UnplugEnvCfgRGB):
    """Configuration for the Franka Unplug Environment."""

    events: EventCfg = EventCfg()
    terminations:TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events



        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") 
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        
        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]
        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

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

        

        # Rigid body properties for connectors
        socket_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=32, #TODO: adjust according need
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=0.1,
            disable_gravity=True, #TODO: Disable gravity to keep it fixed in place, as we are not simulating the full socket but just a rigid body representing it
        )

        # Socket
        self.scene.socket = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Socket",
            # init_state=RigidObjectCfg.InitialStateCfg(pos=(socket_x, socket_y, socket_z), rot=(1.0, 0.0, 0.0, 0.0)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(SOCKET_X, SOCKET_Y, SOCKET_Z), rot=(0.0, 0.0, 1.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"/home/alessio/Downloads/workstation.usd",
                scale=(1.0, 1.0, 1.0),
                #rigid_props=socket_properties,
                semantic_tags=[("class", "socket")],
            ),
        )

        # Rigid body properties for connectors
        plug_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=32, #TODO: adjust according need
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=0.1,
            disable_gravity=False,
        )

        # Plug #TODO: adjust 
        self.scene.plug = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plug",
            # init_state=RigidObjectCfg.InitialStateCfg(pos=(socket_x + 0.017, socket_y, socket_z + 0.12125), rot=(1.0, 0.0, 0.0, 0.0)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(SOCKET_X - 0.017, SOCKET_Y, SOCKET_Z - 0.12125), rot=(0.0, 0.0, 1.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"/home/alessio/Downloads/usbc_plug_new.usd",
                scale=(1.0, 1.0, 1.0),
                #rigid_props=plug_properties,
                semantic_tags=[("class", "plug")],
            ),
        )


        #TODO: see if marker thing is needed
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"


        #TODO: check if needed and if yes, check OFFSETS
        # Listens to the required transforms
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

