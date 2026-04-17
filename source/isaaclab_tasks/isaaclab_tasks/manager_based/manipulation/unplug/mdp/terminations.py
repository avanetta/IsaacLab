from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def plug_successfully_unplugged(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
    socket_cfg: SceneEntityCfg = SceneEntityCfg("socket"),
    unplug_distance: float = 0.25,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """Check if plug is unplugged (y-axis) AND gripper is holding the plug.

    Only the y-axis distance is considered to avoid false positives when the plug
    simply falls down (z-axis) rather than being pulled out.
    The gripper must be closed (holding the plug) for a successful unplug.
    """
    plug: RigidObject = env.scene[plug_cfg.name]
    socket: RigidObject = env.scene[socket_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Check y-axis distance only
    y_distance = torch.abs(plug.data.root_pos_w[:, 1] - socket.data.root_pos_w[:, 1])
    is_unplugged = y_distance > unplug_distance

    # Check gripper is closed (holding the plug)
    if hasattr(env.cfg, "gripper_joint_names"):
        
        gripper_open_val = gripper_open_val.detach().clone().to(env.device)

        # Gripper is holding when finger positions are far from the open value
        is_holding = torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val) > gripper_threshold
        is_holding = torch.logical_and(
            is_holding,
            torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val) > gripper_threshold,
        )
        return torch.logical_and(is_unplugged, is_holding)
    else:
        # If no gripper check possible, just return unplugged status
        return is_unplugged

def is_plug_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_open_val: float = 0.04,
    gripper_threshold: float = 0.005,
    dist_threshold: float = 0.05
):
    robot: Articulation = env.scene[robot_cfg.name]
    plug: RigidObject = env.scene[plug_cfg.name]
    
    # Only one gripper index (index 7 for 8-dim qpos)
    gripper_pos = robot.data.joint_pos[:, 7] 
    
    # If open is 0.04, then 'holding' means the value is significantly smaller (closer to 0.0)
    is_holding = torch.abs(gripper_pos - gripper_open_val) > gripper_threshold

    # 2. Check distance between EEF and Plug
    # Note: Use the actual end-effector frame name from your robot config
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, 0, :] 
    plug_pos = plug.data.root_pos_w[:, :3]

    dist = torch.norm(ee_pos - plug_pos, dim=-1)
    is_near = dist < dist_threshold

    return torch.logical_and(is_holding, is_near)