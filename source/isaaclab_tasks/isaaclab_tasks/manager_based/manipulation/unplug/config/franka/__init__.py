# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import agents


##
# Register Gym environments.
##



# Inverse Kinematics - Absolute Pose Control
gym.register(
    id="Isaac-Unplug-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unplug_ik_abs_env_cfg:FrankaUnplugIKAbsEnvCfgRGB",
    },
    disable_env_checker=True,
)



