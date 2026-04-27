# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_REAL_ROBOT_CFG
from isaaclab_assets.robots.fr3 import FR3_WITH_HAND_CFG

from isaaclab_tasks.manager_based.manipulation.unplug.config.franka.unplug_ik_abs_env_cfg import FrankaUnplugIKAbsEnvCfgRGB
    
"""Inherits from the FrankaCubeStackEnvCfg (absolute IK control) and specializes the config for absolute pose control in a mimic environment."""

@configclass
class FrankaUnplugIKAbsMimicEnvCfgRGB(FrankaUnplugIKAbsEnvCfgRGB, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Franka Unplug IK Abs env.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "demo_src_unplug_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = True
        self.datagen_config.generation_interpolate_from_last_target_pose = True

        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1


        # Subtask configurations for the unplug task:
        #   1. Grasp the plug while it is in the socket
        #   2. Unplug it (pull it out of the socket)
        #   3. Hold the plug steady
        subtask_configs = []

        # Subtask 1: Reach and grasp the plug in the socket
        subtask_configs.append(
            SubTaskConfig(
                object_ref="plug",
                subtask_term_signal="grasp_plug",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )

        # Subtask 2: Pull the plug out of the socket
        subtask_configs.append(
            SubTaskConfig(
                object_ref="plug",
                subtask_term_signal="unplug_complete",
                # subtask_term_offset_range=(10, 20),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,#0.015
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )

        # Comment this for data generation
        # Subtask 3: Hold the plug
        # subtask_configs.append(
        #     SubTaskConfig(
        #         object_ref="plug",
        #         subtask_term_signal=None, #None for final subtask
        #         subtask_term_offset_range=(0, 0),
        #         selection_strategy="nearest_neighbor_object",
        #         selection_strategy_kwargs={"nn_k": 3},
        #         action_noise=0.0, #0.015
        #         num_interpolation_steps=5,
        #         num_fixed_steps=0,
        #         apply_noise_during_interpolation=False,
        #     )
        # )
        
        
        self.subtask_configs["franka"] = subtask_configs
