# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.unplug import mdp

from . import unplug_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG, FRANKA_PANDA_REAL_ROBOT_CFG, FRANKA_PANDA_CFG  # isort: skip

@configclass
class ActionsCfg:
    """Action specifications for IK control."""
    
    arm_action: DifferentialInverseKinematicsActionCfg = MISSING  # type: ignore
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING  # type: ignore


##
# Main environment configuration
##
@configclass
class FrankaUnplugIKAbsEnvCfgRGB(unplug_joint_pos_env_cfg.FrankaUnplugJointPosEnvCfgRGB):
    """Configuration for Franka USB Unplug with absolute IK control."""
    
    actions: ActionsCfg = ActionsCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            # Define custom offset from the flange frame
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.1034), # Introduce a small offset to match the real robot's hand position
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            # Use Differential Inverse Kinematics Controller 
            controller=DifferentialIKControllerCfg(
                command_type="pose", 
                use_relative_mode=False, 
                ik_method="dls"),
            
        )

