# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_REAL_ROBOT_CFG
from isaaclab_assets.robots.fr3 import FR3_WITH_HAND_CFG

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_abs_env_cfg_original import FrankaCubeStackEnvCfgRGB

"""Inherits from the FrankaCubeStackEnvCfg (absolute IK control) and specializes the config for absolute pose control in a mimic environment."""

@configclass
class FrankaCubeStackIKAbsMimicEnvCfgRGB(FrankaCubeStackEnvCfgRGB, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Franka Cube Stack IK Abs env.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Explicitly ensure same robot config
        #self.scene.robot = FRANKA_PANDA_REAL_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        

        # Override the existing values
        self.datagen_config.name = "demo_src_stack_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = True
        self.datagen_config.generation_interpolate_from_last_target_pose = True

        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # The following are the subtask configurations for the stack task.
        # SubtaskConfig is defined in source/isaaclab_mimic/isaaclab_mimic/envs/mimic_env_cfg.py
        # Tune these parameters for maximizing the success rate of the task.
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_2",
                subtask_term_signal="grasp_1",
                subtask_term_offset_range=(10,20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.000,#0.015
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal="stack_1",
                subtask_term_offset_range=(10,20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0, #0.015
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_3",
                subtask_term_signal="grasp_2",
                subtask_term_offset_range=(10,20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,#0.015
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_2",
                subtask_term_signal="stack_2",  # Fixed: added comma and lowercase
                subtask_term_offset_range=(0,0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["franka"] = subtask_configs
