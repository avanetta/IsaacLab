# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control
* :obj:`FRANKA_ROBOTIQ_GRIPPER_CFG`: Franka robot with Robotiq_2f_85 gripper

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import PhysicsMaterialCfg, RigidBodyMaterialCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import numpy as np

##
# Configuration
##

FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        #usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        usd_path = "/home/alessio/Downloads/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.05,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=64,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            # effort_limit_sim=200.0,
            # stiffness=2e3,
            # damping=1e2,
            effort_limit_sim=50.0,
            stiffness=500,
            damping=50,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""


FRANKA_ROBOTIQ_GRIPPER_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.variants = {"Gripper": "Robotiq_2F_85"}
FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "finger_joint": 0.0,
    ".*_inner_finger_joint": 0.0,
    ".*_inner_finger_knuckle_joint": 0.0,
    ".*_outer_.*_joint": 0.0,
}
FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.pos = (-0.85, 0, 0.76)
FRANKA_ROBOTIQ_GRIPPER_CFG.actuators = {
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=5200.0,
        velocity_limit_sim=2.175,
        stiffness=1100.0,
        damping=80.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit_sim=720.0,
        velocity_limit_sim=2.61,
        stiffness=1000.0,
        damping=80.0,
    ),
    "gripper_drive": ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],  # "right_outer_knuckle_joint" is its mimic joint
        effort_limit_sim=1650,
        velocity_limit_sim=10.0,
        stiffness=17,
        damping=0.02,
    ),
    # enable the gripper to grasp in a parallel manner
    "gripper_finger": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_joint"],
        effort_limit_sim=50,
        velocity_limit_sim=10.0,
        stiffness=0.2,
        damping=0.001,
    ),
    # set PD to zero for passive joints in close-loop gripper
    "gripper_passive": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
        effort_limit_sim=1.0,
        velocity_limit_sim=10.0,
        stiffness=0.0,
        damping=0.0,
    ),
}


"""Configuration of Franka Emika Panda robot with Robotiq_2f_85 gripper."""

"""Configuration of Franka Emika Panda robot matching real robot joint stiffnesses.
This configuration uses the exact joint-specific stiffness values used on the real robot:
We calculated the joint stiffnesses from the cartesian stiffness
"""
FRANKA_PANDA_REAL_ROBOT_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_REAL_ROBOT_CFG.spawn.rigid_props.disable_gravity = True  # Match typical robot control
FRANKA_PANDA_REAL_ROBOT_CFG.actuators = {
    # The gains in joint space we're calculated using the following mapping that maps the operational space parameters
    # defined in cartesian_...hpp to joint space parameters. K_J = J^T * K_C * J
    # where J is the Jacobian of the end-effector in joint space.
    # Joints 1-4: Higher stiffness (shoulder/elbow region)
    "panda_joint1": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint1"],
        effort_limit=87.0,
        velocity_limit=2.175,
        stiffness=343.07,  # Match your kp[0]
        damping=2.5*np.sqrt(343.07),     # Reasonable damping (10% of stiffness)
    ),
    "panda_joint2": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint2"],
        effort_limit=87.0,
        velocity_limit=2.175,
        stiffness=307.50,  # Match your kp[1]
        damping=2.5*np.sqrt(307.50),
    ),
    "panda_joint3": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint3"],
        effort_limit=87.0,
        velocity_limit=2.175,
        stiffness=328.39,  # Match your kp[2]
        damping=2.5*np.sqrt(328.39),
    ),
    "panda_joint4": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint4"],
        effort_limit=87.0,
        velocity_limit=2.175,
        stiffness=487.13,  # Match your kp[3]
        damping=2.5*np.sqrt(487.13),
    ),
    # Joints 5-7: Lower stiffness (wrist region)
    "panda_joint5": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint5"],
        effort_limit=12.0,
        velocity_limit=2.61,
        stiffness=95.65,   # Match your kp[4]
        damping=2.5*np.sqrt(95.65),
    ),
    "panda_joint6": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint6"],
        effort_limit=12.0,
        velocity_limit=2.61,
        stiffness=142.41,   # Match your kp[5]
        damping=2.5*np.sqrt(142.41),
    ),
    "panda_joint7": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint7"],
        effort_limit=12.0,
        velocity_limit=2.61,
        stiffness=80.0,   # Match your kp[6]
        damping=2.5*np.sqrt(80.0),
    ),
    # Gripper
    "panda_hand": ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint.*"],
        effort_limit=100.0,
        velocity_limit=0.1, #adjusted from 0.2
        stiffness=1200,
        damping=70,
    ),
    # Gripper
    # "panda_hand": ImplicitActuatorCfg(
    #         joint_names_expr=["panda_finger_joint.*"],
    #         effort_limit=200.0,
    #         velocity_limit=0.5,
    #         stiffness=2e3,
    #         damping=1e2,
    # ),
}
