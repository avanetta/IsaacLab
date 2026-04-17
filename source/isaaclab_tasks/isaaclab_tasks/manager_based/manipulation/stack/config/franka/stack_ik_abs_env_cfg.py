# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass

# This config inhertits from the stack_joint_pos_env_cfg module
from . import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.fr3 import FR3_WITH_HAND_CFG 
from isaaclab_assets.robots.franka import FRANKA_PANDA_REAL_ROBOT_CFG
'''This class inherits from the stack_joint_pos_env_cfg and specializes the config for inverse kinematics control.
We keep most of the configurations from the parent class, but set the controller to use differential inverse kinematics 
instead of joint position control.'''

@configclass
class FrankaCubeStackEnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__() 

        # Set robot with stiffness and damping values of the real robot
        #self.scene.robot = FR3_WITH_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Set actions for the specific robot type (franka), inverse kinematics control
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            # Define custom offset from the flange frame
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.1034),# Introduce a small offset to match the real robot's hand position
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            # Use Differential Inverse Kinematics Controller 
            controller=DifferentialIKControllerCfg(
                command_type="pose", 
                use_relative_mode=False, # we're working with absolute poses
                ik_method="dls",
            ),
        ) # type: ignore

class FrankaCubeStackEnvCfgRGB(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfgRGB):
    def __post_init__(self):
        # post init of parent
        super().__post_init__() 
        #self.scene.robot = FR3_WITH_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Set actions for the specific robot type (franka), inverse kinematics control
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            # Define custom offset from the flange frame
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.1034),  # Introduce a small offset to match the real robot's hand position
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            # Use Differential Inverse Kinematics Controller 
            controller=DifferentialIKControllerCfg(
                command_type="pose", 
                use_relative_mode=False, # we're working with absolute poses
                ik_method="dls",
            ),
        ) # type: ignore
        
        
