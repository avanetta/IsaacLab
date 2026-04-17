# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class FrankaUnplugIKAbsMimicEnvRGB(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for Franka Unplug IK Abs env.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end effector pose."""
        if env_ids is None:
            env_ids = slice(None)

        # Retrieve end effector pose from the observation buffer
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self, 
        target_eef_pose_dict: dict, 
        gripper_action_dict: dict, 
        action_noise_dict: dict | None = None, 
        env_id: int = 0
    ) -> torch.Tensor:
        """Convert target pose to action.

        This method transforms a dictionary of target end-effector poses and gripper actions
        into a single action tensor that can be used by the environment.

        The function:
        1. Extracts target position and rotation from the pose dictionary
        2. Extracts gripper action for the end effector
        3. Concatenates position and quaternion rotation into a pose action
        4. Optionally adds noise to the pose action for exploration
        5. Combines pose action with gripper action into a final action tensor

        Args:
            target_eef_pose_dict: Dictionary containing target end-effector pose(s),
                with keys as eef names and values as pose tensors.
            gripper_action_dict: Dictionary containing gripper action(s),
                with keys as eef names and values as action tensors.
            action_noise_dict: Dictionary containing noise magnitudes for each end-effector.
                If provided, random noise is generated and added to the pose action.
            env_id: Environment ID for multi-environment setups, defaults to 0.

        Returns:
            torch.Tensor: A single action tensor combining pose and gripper commands.
        """
        # target position and rotation
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # get gripper action for single eef
        (gripper_action,) = gripper_action_dict.values()

        # convert rotation matrix to quaternion and ensure it's normalized
        target_quat = PoseUtils.quat_from_matrix(target_rot)
        target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True)  # normalize quaternion
        
        # add noise to action if specified
        pose_action = torch.cat([target_pos, target_quat], dim=0)
        if action_noise_dict is not None:
            eef_name = list(target_eef_pose_dict.keys())[0]
            if eef_name in action_noise_dict:
                noise_magnitude = action_noise_dict[eef_name]
                # Add noise to position
                pos_noise = noise_magnitude * torch.randn_like(target_pos)
                # Add smaller noise to quaternion to avoid large rotational changes
                quat_noise = (noise_magnitude * 0.1) * torch.randn_like(target_quat)
                
                pose_action[:3] += pos_noise
                pose_action[3:7] += quat_noise
                # Re-normalize quaternion after adding noise
                pose_action[3:7] = pose_action[3:7] / torch.norm(pose_action[3:7])

        return torch.cat([pose_action, gripper_action], dim=0).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert action to target pose."""
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        target_pos = action[:, :3]
        target_quat = action[:, 3:7]
        target_rot = PoseUtils.matrix_from_quat(target_quat)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper actions."""
        # last dimension is gripper action
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get subtask termination signals."""
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["grasp_plug"] = subtask_terms["grasp_plug"][env_ids]
        signals["unplug_complete"] = subtask_terms["unplug_complete"][env_ids]
        return signals
    
    def set_object_pose(self, root_pose: torch.Tensor, obj_name: str, env_ids: Sequence[int] | None = None) -> None:
        """Set the pose of an object in the scene.
        
        Args:
            root_pose: Target pose tensor. Shape is (batch_size, 7) for [x, y, z, qw, qx, qy, qz]
                       or (7,) for single pose.
            obj_name: Name of the object ("plug" or "socket")
            env_ids: Environment indices to set the pose for. If None, all envs are set.
            
        """
        # Handle env_ids conversion
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        # Handle single pose (7,) vs batch
        if root_pose.dim() == 1:
            root_pose = root_pose.unsqueeze(0)
        
        # Extract position and quaternion
        pos = root_pose[..., :3].clone()
        quat = root_pose[..., 3:7]
        
        if obj_name == "plug" and "plug" in self.scene.rigid_objects:
            plug = self.scene.rigid_objects["plug"]
            # Get the default root state and modify it
            root_state = plug.data.default_root_state[env_ids].clone()
            root_state[:, :3] = pos
            root_state[:, 3:7] = quat
            print(f"Setting plug pose for env_ids {env_ids} to pos {pos} and quat {quat}")
            plug.write_root_state_to_sim(root_state, env_ids)
        elif obj_name == "socket" and "socket" in self.scene.rigid_objects:
            socket = self.scene.rigid_objects["socket"]
            # Get the default root state and modify it
            root_state = socket.data.default_root_state[env_ids].clone()
            root_state[:, :3] = pos
            root_state[:, 3:7] = quat
            print(f"Setting socket pose for env_ids {env_ids} to pos {pos} and quat {quat}")
            socket.write_root_state_to_sim(root_state, env_ids)
        else:
            raise ValueError(f"Unknown object name: {obj_name}. Must be 'plug' or 'socket'.")

    def get_expected_attached_object(self, eef_name: str, subtask_index: int, env_cfg) -> str | None:
        """
        (SkillGen) Return the expected attached object for the given EEF/subtask.

        Assumes 'stack' subtasks place the object grasped in the preceding 'grasp' subtask.
        Returns None for 'grasp' (or others) at subtask start.
        """
        if eef_name not in env_cfg.subtask_configs:
            return None

        subtask_configs = env_cfg.subtask_configs[eef_name]
        if not (0 <= subtask_index < len(subtask_configs)):
            return None

        current_cfg = subtask_configs[subtask_index]
        # If stacking, expect we are holding the object grasped in the prior subtask
        if "stack" in str(current_cfg.subtask_term_signal).lower():
            if subtask_index > 0:
                prev_cfg = subtask_configs[subtask_index - 1]
                if "grasp" in str(prev_cfg.subtask_term_signal).lower():
                    return prev_cfg.object_ref
        return None

 #Copyright (c) 2024-2025, The Isaac Lab Project Developers.
 #All rights reserved.

# SPDX-License-Identifier: Apache-2.0
#import torch
#from collections.abc import Sequence
#import isaaclab.utils.math as PoseUtils
#from isaaclab.envs import ManagerBasedRLMimicEnv
#class FrankaCubeStackIKAbsMimicEnvRGB(ManagerBasedRLMimicEnv):
#    """
#    Isaac Lab Mimic environment wrapper class for Franka Cube Stack IK Rel env.
#    """
#
#    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
#        """
#        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.
#
#        Args:
#            eef_name: Name of the end effector.
#            env_ids: Environment indices to get the pose for. If None, all envs are considered.
#
#        Returns:
#            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
#        """
#        if env_ids is None:
#            env_ids = slice(None)
#
#        # Retrieve end effector pose from the observation buffer
#        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
#        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
#        # Quaternion format is w,x,y,z
#        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))
#
#    def target_eef_pose_to_action(
#        self,
#        target_eef_pose_dict: dict,
#        gripper_action_dict: dict,
#        action_noise_dict: dict | None = None,
#        env_id: int = 0,
#    ) -> torch.Tensor:
#        """
#        Takes a target pose and gripper action for the end effector controller and returns an action
#        (usually a normalized delta pose action) to try and achieve that target pose.
#        Noise is added to the target pose action if specified.
#
#        Args:
#            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
#            gripper_action_dict: Dictionary of gripper actions for each end-effector.
#            noise: Noise to add to the action. If None, no noise is added.
#            env_id: Environment index to get the action for.
#
#        Returns:
#            An action torch.Tensor that's compatible with env.step().
#        """
#        eef_name = list(self.cfg.subtask_configs.keys())[0]
#
#        # target position and rotation
#        (target_eef_pose,) = target_eef_pose_dict.values()
#        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)
#
#        # current position and rotation
#        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
#        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)
#
#        # normalized delta position action
#        delta_position = target_pos - curr_pos
#
#        # normalized delta rotation action
#        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
#        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
#        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)
#
#        # get gripper action for single eef
#        (gripper_action,) = gripper_action_dict.values()
#
#        # add noise to action
#        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
#        if action_noise_dict is not None:
#            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
#            pose_action += noise
#            pose_action = torch.clamp(pose_action, -1.0, 1.0)
#
#        return torch.cat([pose_action, gripper_action], dim=0)
#
#    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
#        """
#        Converts action (compatible with env.step) to a target pose for the end effector controller.
#        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
#        from a demonstration trajectory using the recorded actions.
#
#        Args:
#            action: Environment action. Shape is (num_envs, action_dim)
#
#        Returns:
#            A dictionary of eef pose torch.Tensor that @action corresponds to
#        """
#        eef_name = list(self.cfg.subtask_configs.keys())[0]
#
#        delta_position = action[:, :3]
#        delta_rotation = action[:, 3:6]
#
#        # current position and rotation
#        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
#        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)
#
#        # get pose target
#        target_pos = curr_pos + delta_position
#
#        # Convert delta_rotation to axis angle form
#        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
#        delta_rotation_axis = delta_rotation / delta_rotation_angle
#
#        # Handle invalid division for the case when delta_rotation_angle is close to zero
#        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
#        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]
#
#        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
#        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
#        target_rot = torch.matmul(delta_rot_mat, curr_rot)
#
#        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()
#
#        return {eef_name: target_poses}
#
#    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
#        """
#        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).
#
#        Args:
#            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).
#
#        Returns:
#            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
#        """
#        # last dimension is gripper action
#        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}
#
#    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
#        """
#        Gets a dictionary of termination signal flags for each subtask in a task. The flag is 1
#        when the subtask has been completed and 0 otherwise. The implementation of this method is
#        required if intending to enable automatic subtask term signal annotation when running the
#        dataset annotation tool. This method can be kept unimplemented if intending to use manual
#        subtask term signal annotation.
#
#        Args:
#            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.
#
#        Returns:
#            A dictionary termination signal flags (False or True) for each subtask.
#        """
#        if env_ids is None:
#            env_ids = slice(None)
#
#        signals = dict()
#        subtask_terms = self.obs_buf["subtask_terms"]
#        signals["grasp_1"] = subtask_terms["grasp_1"][env_ids]
#        signals["grasp_2"] = subtask_terms["grasp_2"][env_ids]
#        signals["stack_1"] = subtask_terms["stack_1"][env_ids]
#        # final subtask is placing cubeC on cubeA (motion relative to cubeA) - but final subtask signal is not needed
#        return signals
#
#    def get_expected_attached_object(self, eef_name: str, subtask_index: int, env_cfg) -> str | None:
#        """
#        (SkillGen) Return the expected attached object for the given EEF/subtask.
#
#        Assumes 'stack' subtasks place the object grasped in the preceding 'grasp' subtask.
#        Returns None for 'grasp' (or others) at subtask start.
#        """
#        if eef_name not in env_cfg.subtask_configs:
#            return None
#
#        subtask_configs = env_cfg.subtask_configs[eef_name]
#        if not (0 <= subtask_index < len(subtask_configs)):
#            return None
#
#        current_cfg = subtask_configs[subtask_index]
#        # If stacking, expect we are holding the object grasped in the prior subtask
#        if "stack" in str(current_cfg.subtask_term_signal).lower():
#            if subtask_index > 0:
#                prev_cfg = subtask_configs[subtask_index - 1]
#                if "grasp" in str(prev_cfg.subtask_term_signal).lower():
#                    return prev_cfg.object_ref
#        return None