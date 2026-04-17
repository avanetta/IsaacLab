


# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic.

This script loads a robomimic policy and plays it in an Isaac Lab environment.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: If provided, overeride the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
    save_observations: If provided, save successful observations to file.
    obs_output_file: Output file path for saved observations.
"""

"""Launch Isaac Sim Simulator first."""


import argparse
from collections import deque
import pickle
import json 
import numpy as np
import os
import sys
from pathlib import Path
import pandas as pd
from traitlets import default  # Add pandas import for CSV handling
from isaaclab.app import AppLauncher
import cv2




# Ensure repo root is on PYTHONPATH so top-level modules (e.g., `act_copy`) are importable
# even when this script is executed from within `scripts/...`.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--data_path", type=str, default=None, help="Dataapath")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument("--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor.")
parser.add_argument("--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor.")
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")

# Arguments for observation logging
parser.add_argument("--save_observations", action="store_true", default=False, help="Save successful observations to file.")
parser.add_argument("--obs_output_file", type=str, default="successful_observations.csv", help="Output file path for saved observations.")
parser.add_argument("--csv_output_file", type=str, default="successful_observations.csv", help="Output CSV file path for detailed observations.")
parser.add_argument('--velocity_control', action='store_true')
parser.add_argument('--context_length', type=int, default=1, help='context_length')
parser.add_argument("--custom_cube_poses", type=str, default=None, required=False, 
                   help="JSON file path with custom cube configurations")
parser.add_argument('--joint_pos_dim', type=int, default=7, help='Dimension of joint positions')


AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



"""Rest everything follows."""

import copy
import gymnasium as gym
import torch
import isaaclab_mimic.envs  # noqa: F401 - registers mimic Gym environments
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg

from diffusion.diffusion_policy_v6 import DiffusionPolicyV6


class ImageProcessor:
    """9-channel temporal image stack: [current, previous, difference]."""

    def __init__(self, image_history):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        self.image_history = image_history
        self.image_history = deque(maxlen=2)

    def preprocess_single_image(self, rgb_image):
        img = cv2.resize(rgb_image, (224, 224)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = (img - self.mean) / self.std
        return img

    def update(self, rgb_image):
        processed = self.preprocess_single_image(rgb_image)
        self.image_history.append(processed)
        if len(self.image_history) < 2:
            self.image_history.append(processed)
        current = self.image_history[-1]
        previous = self.image_history[-2]
        diff = current - previous
        return np.concatenate([current, previous, diff], axis=0)

def rollout(
    policy,
    env,
    success_term,
    plug_dropping_term,
    horizon,
    device,
    stats_torch,
    save_observations=False,
    trial_id=0,
    velocity_control=False,
    history_len=5,
    joint_pos_dim=7,
):
    """
    Rollout for Diffusion Policy v6 in Isaac Lab.
    """
    import pandas as pd

    # 1. Buffers for Temporal Context (Required by v6)
    image_history = deque(maxlen=history_len) 
    qpos_history = deque(maxlen=history_len)
    qvel_history = deque(maxlen=history_len)

    image_processor = ImageProcessor(image_history)


    plan_queue = deque()

    # 2. Reset env
    obs_dict, _ = env.reset()
    
    trial_result = (0, "fail")  # Default if horizon reached without termination
    chunk_data_log = []

    # 3. Rollout loop
    for t in range(horizon):

        # CREATE OBSERVATION
        isaac_qpos = env.scene["robot"].data.joint_pos.squeeze(0)  # [8]: 7 joints + 1 gripper
        isaac_qvel = env.scene["robot"].data.joint_vel.squeeze(0)  # [8]: 7 joints + 1 gripper vel
        
        if joint_pos_dim == 9:
            # 7 joints + 2 finger values (both equal to gripper)
            curr_qpos = torch.cat([isaac_qpos[:7], isaac_qpos[7:8], isaac_qpos[7:8]])
            curr_qvel = torch.cat([isaac_qvel[:7], isaac_qvel[7:8], isaac_qvel[7:8]])
        else:
            # Default: just 7 joints
            curr_qpos = isaac_qpos[:7]
            curr_qvel = isaac_qvel[:7]
        
        # Ablation for checking impact of joint history: zero out the joint positions to test if the policy can still succeed based on image input alone
        #curr_qpos = torch.zeros(joint_pos_dim)
        
        qpos_history.append(curr_qpos)
        
        while len(qpos_history) < history_len:
            qpos_history.append(curr_qpos)
        while len(qvel_history) < history_len:
            qvel_history.append(curr_qvel)

        policy_obs = obs_dict["policy"]
        curr_raw_img = policy_obs["image"].squeeze(0) # [H, W, 3]
        if isinstance(curr_raw_img, torch.Tensor):
            curr_raw_img = curr_raw_img.cpu().numpy()
        
        # Ablation for checking impact of image input: zero out the image to test if the policy can still succeed based on joint history alone
        #curr_raw_img = np.zeros(curr_raw_img.shape, dtype=np.uint8)

        img_stack = image_processor.update(curr_raw_img)


        qpos_stack = torch.stack(list(qpos_history), dim=0) # [5, joint_pos_dim]  # MODIFIED
        qvel_stack = torch.stack(list(qvel_history), dim=0) # [5, joint_pos_dim]  # MODIFIED



        # INFERENCE
        with torch.no_grad():
            if velocity_control:
                obs = {
                    'img_stack': torch.from_numpy(img_stack).float().unsqueeze(0).to(device),
                    'joint_pos_history': qpos_stack.unsqueeze(0).to(device),
                    'joint_vel_history': qvel_stack.unsqueeze(0).to(device)
                }
            else:
                obs = {
                    'img_stack': torch.from_numpy(img_stack).float().unsqueeze(0).to(device),
                    'joint_pos_history': qpos_stack.unsqueeze(0).to(device)
                }
            
            output = policy.forward_inference(obs, use_ddim=True)
            
            action_chunk = output['actions'].squeeze(0) # [64, 7]
            target_pose = action_chunk[0] # [7]
            gripper_val = output['gripper_cmd'].flatten()[0].item()
            gripper_val = 1.0 if gripper_val > 0.75 else -1.0
        


        # APPLY STEP
        pos = target_pose[:3]
        quat_raw = target_pose[3:7]
        quat_corrected = quat_raw[[3, 0, 1, 2]] # Convert from [qx, qy, qz, qw] to [qw, qx, qy, qz]
        final_action = torch.zeros((1, 8), device=device)
        final_action[0, :3] = pos
        final_action[0, 3:7] = quat_corrected
        final_action[0, 7] = gripper_val 

        print(f"Trial {trial_id} Step {t}: Executing action {final_action.cpu().numpy()}")  # Debug print
        obs_dict, _, terminated, truncated, _ = env.step(final_action)



        # SUCCESS / FAIL
        is_success = bool(success_term.func(env, **success_term.params)[0])
        if is_success:
            print(f"✅ SUCCESS: Trial {trial_id} @ step {t}")
            trial_result = (1, "success")
            break
        
        if terminated or truncated:
            reason = "TERMINATED (DROP)" if terminated else "TIMEOUT"
            print(f"❌ {reason}: Trial {trial_id} @ step {t}")
            trial_result = (0, reason)
            break


    return trial_result, None

def update_environment_events(env):
    """Add a reset event that places the plug at a fixed position every reset.

    The plug is always reset to its default spawn pose so the starting
    conditions are deterministic across rollouts.
    """
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.envs.mdp.events import reset_root_state_uniform

    # Fixed plug pose — zero ranges → no randomisation
    env.cfg.events.reset_plug_fixed = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("plug"),
            "pose_range": {},          # all zeros → default root state
            "velocity_range": {},      # all zeros → zero velocity
        },
    )

    # Re-build the event manager so the new term is active
    from isaaclab.managers import EventManager
    env.event_manager = EventManager(env.cfg.events, env)

    print("[INFO] Plug reset event installed")

def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.observations.policy.concatenate_terms = False
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    success_term = env_cfg.terminations.success
    plug_dropping_term = env_cfg.terminations.plug_dropping
    env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # 2. Initialize Diffusion Policy
    # Match the params from your colleague's "Main model" block
    checkpoint = torch.load(args_cli.checkpoint, map_location='cpu')
    trained_use_joint_vel = checkpoint.get("use_joint_vel", False)

    print(f"Using use_joint_vel={trained_use_joint_vel} based on checkpoint data")
    policy = DiffusionPolicyV6(
        action_horizon=16,
        action_dim=8,
        diffusion_steps=100,
        ddim_steps=10, # Faster inference
        history_len=5,
        #use_joint_vel = trained_use_joint_vel,  # Ensure consistency with training
        joint_pos_dim=args_cli.joint_pos_dim,  # MODIFIED: Pass joint_pos_dim to model
    )
    
    # 3. Load weights
    print(f"Loading weights from {args_cli.checkpoint}")
        # 1. Load the full checkpoint dictionary
    checkpoint = torch.load(args_cli.checkpoint, map_location=device)

        # 2. Extract the actual weights from the 'model_state_dict' key
    if "model_state_dict" in checkpoint:
        raw_state_dict = checkpoint["model_state_dict"]
    else:
        raw_state_dict = checkpoint

        # 3. Fix the 'gripper_pos_weight' float vs Tensor mismatch
        # The error says it expected a Tensor but got a float.
    if "gripper_pos_weight" in raw_state_dict:
        weight_val = raw_state_dict["gripper_pos_weight"]
        if isinstance(weight_val, (float, int)):
            raw_state_dict["gripper_pos_weight"] = torch.tensor(weight_val, dtype=torch.float32)

        # 4. Filter out keys that aren't actually part of the model layers
        # (Like 'use_joint_vel' or 'epoch' which were saved in the dict)
    model_dict = policy.state_dict()
    filtered_dict = {k: v for k, v in raw_state_dict.items() if k in model_dict}

        # 5. Load it into the policy
    policy.load_state_dict(filtered_dict, strict=False)
    print(f"Successfully loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    policy.to(device)
    policy.eval()

    # 4. Run Rollouts
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        
        trial_result, _ = rollout(
            policy, env, success_term, plug_dropping_term,
            args_cli.horizon, device, None, trial_id=trial,
            velocity_control=args_cli.velocity_control,
            joint_pos_dim=args_cli.joint_pos_dim, 
        )
        results.append(trial_result)
        
        # MODIFIED: Print intermediate results with running success rate
        success_flag, reason = trial_result
        print(f"[INFO] Trial {trial}: ({success_flag}, {reason})")
        
        # Calculate and print running success rate
        running_success_count = sum(1 for success_flag, _ in results if success_flag == 1)
        running_success_rate = (running_success_count / len(results)) * 100
        print(f"Running Success Rate: {running_success_count}/{len(results)} = {running_success_rate:.1f}%\n")
    
    # MODIFIED: Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    # Count outcomes by reason
    outcome_counts = {}
    success_count = 0
    for success_flag, reason in results:
        if success_flag == 1:
            success_count += 1
            outcome_counts["success"] = outcome_counts.get("success", 0) + 1
        else:
            outcome_counts[reason] = outcome_counts.get(reason, 0) + 1
    
    # Print counts
    for outcome, count in sorted(outcome_counts.items()):
        percentage = (count / len(results)) * 100
        print(f"{outcome:15s}: {count:3d} ({percentage:5.1f}%)")
    
    # Print success rate
    print(f"\nSuccess Rate: {success_count}/{len(results)} = {(success_count/len(results))*100:.1f}%")
    print("="*50 + "\n")
    
    env.close()


if __name__ == '__main__':

    
    main()
    # close sim app
    simulation_app.close()