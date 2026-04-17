import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObjectCfg, ArticulationCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def position_command_error(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    """Calculate the L2 distance between the robot end-effector and the plug."""
    # Get the EEF position (source)
    source_pos = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids[0]]
    # Get the Plug position (target)
    target_pos = env.scene[target_cfg.name].data.root_pos_w
    
    return torch.norm(target_pos - source_pos, dim=-1)

def orientation_command_error(
    env: "ManagerBasedRLEnv", 
    asset_cfg: SceneEntityCfg, 
    target_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Calculate the orientation error between the end-effector and the target object.
    Returns a value between 0.0 (perfect alignment) and 1.0 (opposite alignment).
    """
    # 1. Get current EEF orientation (World Frame)
    # asset_cfg.body_ids[0] should point to "panda_hand"
    ee_quat = env.scene[asset_cfg.name].data.body_quat_w[:, asset_cfg.body_ids[0]]
    
    # 2. Get Target (Plug) orientation (World Frame)
    target_quat = env.scene[target_cfg.name].data.root_quat_w
    
    # 3. Calculate the absolute dot product
    # Quaternions q and -q represent the same rotation, so we use the absolute value
    dot_product = torch.sum(ee_quat * target_quat, dim=-1).abs()
    
    # 4. Return the error (1 - dot_product)
    # 0.0 means orientations are identical; 1.0 means they are orthogonal/opposite
    return 1.0 - dot_product