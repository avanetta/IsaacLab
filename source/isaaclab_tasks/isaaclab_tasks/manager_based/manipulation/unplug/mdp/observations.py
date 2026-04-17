# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def plug_position_in_world_frame(
    env: ManagerBasedRLEnv,
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
) -> torch.Tensor:
    """The position of the plug in the world frame."""
    plug: RigidObject = env.scene[plug_cfg.name]
    return plug.data.root_pos_w

def socket_position_in_world_frame(
    env: ManagerBasedRLEnv,
    socket_cfg: SceneEntityCfg = SceneEntityCfg("socket"),
) -> torch.Tensor:
    """The position of the socket in the world frame."""
    socket: RigidObject = env.scene[socket_cfg.name]
    return socket.data.root_pos_w

def plug_orientation_in_world_frame(
    env: ManagerBasedRLEnv,
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
) -> torch.Tensor:
    """The orientation of the plug in the world frame."""
    plug: RigidObject = env.scene[plug_cfg.name]
    return plug.data.root_quat_w

def socket_orientation_in_world_frame(
    env: ManagerBasedRLEnv,
    socket_cfg: SceneEntityCfg = SceneEntityCfg("socket"),
) -> torch.Tensor:
    """The orientation of the socket in the world frame."""
    socket: RigidObject = env.scene[socket_cfg.name]
    return socket.data.root_quat_w

def object_obs(
    env: ManagerBasedRLEnv,
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
    socket_cfg: SceneEntityCfg = SceneEntityCfg("socket"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """Object observations for unplug task."""
    plug: RigidObject = env.scene[plug_cfg.name]
    socket: RigidObject = env.scene[socket_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    plug_pos_w = plug.data.root_pos_w
    plug_quat_w = plug.data.root_quat_w
    socket_pos_w = socket.data.root_pos_w
    socket_quat_w = socket.data.root_quat_w
    
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_plug = plug_pos_w - ee_pos_w
    plug_to_socket = socket_pos_w - plug_pos_w

    return torch.cat(
        (
            plug_pos_w - env.scene.env_origins,
            plug_quat_w,
            socket_pos_w - env.scene.env_origins,
            socket_quat_w,
            gripper_to_plug,
            plug_to_socket,
        ),
        dim=1,
    )

def object_obs_with_noise(
    env: ManagerBasedRLEnv,
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
    socket_cfg: SceneEntityCfg = SceneEntityCfg("socket"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    position_noise_std: float = 0.0, #TODO
    orientation_noise_std: float = 0.0, #TODO
):
    """Object observations with noise for sim-to-real transfer."""
    obs = object_obs(env, plug_cfg, socket_cfg, ee_frame_cfg)
    
    if position_noise_std > 0.0:
        # Add noise to positions (indices 0-2 for plug pos, 7-9 for socket pos, etc.)
        position_noise = torch.randn_like(obs) * position_noise_std
        obs = obs + position_noise
    
    if orientation_noise_std > 0.0:
        # Add noise to orientations (quaternions)
        orientation_noise = torch.randn_like(obs) * orientation_noise_std
        obs = obs + orientation_noise
        
    return obs

def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos

def ee_frame_pos_with_noise(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    noise_std: float = 0.0,
) -> torch.Tensor:
    """End effector position with noise."""
    pos = ee_frame_pos(env, ee_frame_cfg)
    if noise_std > 0.0:
        noise = torch.randn_like(pos) * noise_std
        pos = pos + noise
    return pos

def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat

def ee_frame_quat_with_noise(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    noise_std: float = 0.0,
) -> torch.Tensor:
    """End effector quaternion with noise."""
    quat = ee_frame_quat(env, ee_frame_cfg)
    if noise_std > 0.0:
        noise = torch.randn_like(quat) * noise_std
        quat = quat + noise
    return quat

def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Obtain the versatile gripper position of both Gripper and Suction Cup.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        # Handle multiple surface grippers by concatenating their states
        gripper_states = []
        for gripper_name, surface_gripper in env.scene.surface_grippers.items():
            gripper_states.append(surface_gripper.state.view(-1, 1))

        if len(gripper_states) == 1:
            return gripper_states[0]
        else:
            return torch.cat(gripper_states, dim=1)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
            finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)
            return torch.cat((finger_joint_1, finger_joint_2), dim=1)
        else:
            raise NotImplementedError("[Error] Cannot find gripper_joint_names in the environment config")

def gripper_pos_with_noise(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    noise_std: float = 0.0,
) -> torch.Tensor:
    """Gripper position with noise."""
    pos = gripper_pos(env, robot_cfg)
    if noise_std > 0.0:
        noise = torch.randn_like(pos) * noise_std
        pos = pos + noise
    return pos

def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    # #Implementation Copilot
    # robot: Articulation = env.scene[robot_cfg.name]
    # ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # object: RigidObject = env.scene[object_cfg.name]

    # object_pos = object.data.root_pos_w
    # end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    # pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    # if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
    #     surface_gripper = env.scene.surface_grippers["surface_gripper"]
    #     suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
    #     suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
    #     grasped = torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold)

    # else:
    #     if hasattr(env.cfg, "gripper_joint_names"):
    #         gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    #         assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"

    #         
    #         grasped = torch.logical_and(
    #             pose_diff < diff_threshold,
    #             torch.abs(
    #                 robot.data.joint_pos[:, gripper_joint_ids[0]]
    #                 - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
    #             )
    #             > env.cfg.gripper_threshold,
    #         )
    #         grasped = torch.logical_and(
    #             grasped,
    #             torch.abs(
    #                 robot.data.joint_pos[:, gripper_joint_ids[1]]
    #                 - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
    #             )
    #             > env.cfg.gripper_threshold,
    #         )

    # Implementation according stack task
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)


    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)) > gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped, torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)) > gripper_threshold
    )

    return grasped

def image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    normalize: bool = True,
) -> torch.Tensor:
    """Camera RGB image observation."""
    from isaaclab.sensors import TiledCamera
    
    camera: TiledCamera = env.scene[sensor_cfg.name]
    
    # Get RGB data - shape is (num_envs, height, width, channels)
    image_data = camera.data.output["rgb"]
    
    if normalize:
        # Normalize to [0, 1] if not already
        if image_data.dtype == torch.uint8:
            image_data = image_data.float() / 255.0
    
    return image_data

def ee_frame_pose_in_base_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """
    The end effector pose in the robot base frame.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_frame_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_in_base, ee_quat_in_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_frame_pos_w, ee_frame_quat_w
    )

    if return_key == "pos":
        return ee_pos_in_base
    elif return_key == "quat":
        return ee_quat_in_base
    else:
        return torch.cat((ee_pos_in_base, ee_quat_in_base), dim=1)

def plug_unplugged(
    env: ManagerBasedRLEnv,
    plug_cfg: SceneEntityCfg,
    socket_cfg: SceneEntityCfg,
    unplug_distance_threshold: float = 0.15,
) -> torch.Tensor:
    """Check if plug is unplugged (far enough from socket along the y-axis only).

    Only the y-axis distance is considered to avoid false positives when the plug
    simply falls down (z-axis) rather than being pulled out.
    """
    plug: RigidObject = env.scene[plug_cfg.name]
    socket: RigidObject = env.scene[socket_cfg.name]

    # y-axis distance only (index 1)
    y_distance = torch.abs(plug.data.root_pos_w[:, 1] - socket.data.root_pos_w[:, 1])
    return y_distance > unplug_distance_threshold