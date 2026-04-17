# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import random

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import sample_uniform
from isaaclab.assets import Articulation, AssetBase, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set the default pose for robots in specified environments."""
    asset: Articulation = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)

def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Add Gaussian noise to joint states."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Add gaussian noise to joint states
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper poses (last 2 joints)
    joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    # Set into the physics simulation
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

def randomize_scene_lighting_domelight(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    # Sample new light intensity
    new_intensity = random.uniform(intensity_range[0], intensity_range[1])

    # Set light intensity to light prim
    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    intensity_attr.Set(new_intensity)

def randomize_joint_by_task_space_gaussian(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: tuple[float,float,float],
    std: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the robot joint states by applying a Gaussian offset in task space."""
    # 1. Access assets
    robot: Articulation = env.scene[asset_cfg.name]
    
    # 2. Sample Cartesian noise (per-axis)
    std_tensor = torch.tensor(std, device=env.device)
    mean_tensor = torch.tensor(mean, device=env.device)
    # Shape: (num_reset_envs, 3)
    task_noise = torch.randn(len(env_ids), 3, device=env.device) * std_tensor + mean_tensor
    
    # 3. Get the Jacobian for the End-Effector
    # We use the Jacobian to map Cartesian delta -> Joint delta: Δq ≈ J⁻¹ * Δx
    # 'ee_frame' must be defined in your SceneCfg
    ee_jacobian = robot.root_physx_view.get_jacobians()[:, 0, :3, :7] # Position part for first 7 joints
    
    # 4. Compute Joint Delta using Pseudo-Inverse
    # Δq = J_pinv * Δx
    ee_jac_pinv = torch.linalg.pinv(ee_jacobian[env_ids])
    joint_delta = torch.bmm(ee_jac_pinv, task_noise.unsqueeze(-1)).squeeze(-1)
    
    # 5. Apply to default pose
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, :7] += joint_delta
    
    # 6. Safety: Clamp to limits
    joint_limits = robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])
    
    # 7. Keep gripper fingers at default
    joint_pos[:, -2:] = robot.data.default_joint_pos[env_ids, -2:]
    
    # 8. Write to simulation
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

def randomize_plug_and_socket_unified(
    env, 
    env_ids, 
    socket_cfg, 
    plug_cfg, 
    relative_offset, 
    socket_pos_range,
    socket_initial_pos: tuple = (0.3, 0.0, 0.361)
):
    """Randomize socket and plug positions around their initial position in each environment's local frame."""
    # Get assets
    socket = env.scene[socket_cfg.name]
    plug = env.scene[plug_cfg.name]

    # Get the environment origins (world position of each environment's local frame origin)
    env_origins = env.scene.env_origins[env_ids]  # Shape: (num_reset_envs, 3)
    
    # 1. Randomize Socket Position in environment's local frame, then transform to world
    socket_state = socket.data.default_root_state[env_ids].clone()
    
    pos_min = torch.tensor(socket_pos_range["pos_min"], device=socket.device)
    pos_max = torch.tensor(socket_pos_range["pos_max"], device=socket.device)
    
    # Generate random offset for each environment (in environment frame)
    random_shift = sample_uniform(pos_min, pos_max, (len(env_ids), 3), device=socket.device)
    
    # Socket position = env_origin + local_initial_pos + random_offset (all in environment frame)
    initial_pos = torch.tensor(socket_initial_pos, device=socket.device, dtype=torch.float32)
    socket_state[:, :3] = env_origins + initial_pos + random_shift
    
    # 2. Set Plug Position relative to the NEW socket position
    plug_state = plug.data.default_root_state[env_ids].clone()
    offset = torch.tensor(relative_offset, device=socket.device)
    plug_state[:, :3] = socket_state[:, :3] + offset

    # 3. WRITE TO SIM
    socket.write_root_state_to_sim(socket_state, env_ids)
    plug.write_root_state_to_sim(plug_state, env_ids)

    # print(f"Randomized plug state to {plug_state[0, :3].cpu().numpy()}")
    # print(f"Current plug state is {plug.data.root_pos_w[env_ids][0].cpu().numpy()}")


def randomize_robot_base_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_range: tuple[float, float] = (0.0, 0.0),
    y_range: tuple[float, float] = (-0.1, 0.1),
    z_range: tuple[float, float] = (0.0, 0.0),
):
    """Randomizes the robot's base (root) position within a specified range."""
    # Handle startup/None case
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Access the robot asset (Articulation)
    robot: Articulation = env.scene[asset_cfg.name]

    # 1. Get the default root state (usually the position set in the USD/Cfg)
    # Shape: (num_envs, 13)
    root_state = robot.data.default_root_state[env_ids].clone()

    # 2. Sample random offsets for each axis
    num_resets = len(env_ids)
    offsets = torch.zeros((num_resets, 3), device=env.device)
    
    offsets[:, 0] = math_utils.sample_uniform(x_range[0], x_range[1], (num_resets,), device=env.device)
    offsets[:, 1] = math_utils.sample_uniform(y_range[0], y_range[1], (num_resets,), device=env.device)
    offsets[:, 2] = math_utils.sample_uniform(z_range[0], z_range[1], (num_resets,), device=env.device)

    # 3. Apply the offset to the base position
    root_state[:, :3] += offsets

    # 4. Write the new base pose to the simulation
    robot.write_root_state_to_sim(root_state, env_ids)

    # 5. IMPORTANT: Re-apply joint positions
    # Teleporting the base can sometimes cause joint "jitter" if the 
    # physics state isn't fully refreshed. We force the joints to default.
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
# TODO: Add more events according needs & copy from unplug task