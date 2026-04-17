


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
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
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


def rollout(
    policy,
    env,
    success_term,
    horizon,
    device,
    stats_torch,
    save_observations=False,
    trial_id=0,
    velocity_control=False,
    context_length=4,
    trial = 0
):
    """
    Memory-safe rollout for ACT policy in Isaac Lab with conditional CSV naming.
    """
    import pandas as pd
    import os

    # === Unpack stats (already on GPU) ===
    qpos_mean = stats_torch["qpos_mean"]
    qpos_std = stats_torch["qpos_std"]
    action_mean = stats_torch["action_mean"]
    action_std = stats_torch["action_std"]

    pre_process_qpos = lambda q: (q - qpos_mean) / qpos_std
    post_process = lambda a: a * action_std + action_mean
    if velocity_control:
        qvel_mean = stats_torch["qvel_mean"]
        qvel_std = stats_torch["qvel_std"]
        pre_process_qvel = lambda q: (q - qvel_mean) / qvel_std

    

    # === Cache temporal ensemble buffers ===
    num_queries = 32 
    action_dim = env.action_space.shape[1]
    all_time_actions = torch.zeros(horizon, horizon + num_queries, action_dim, device=device)

    chunk_data_log = [] # Store data for CSV
    buffer_pos = []
    buffer_vel = []
    buffer_images = []

    # === Reset env ===
    obs_dict, _ = env.reset()
    observation_log = [] if save_observations else None

    # Print initial plug position
    plug_pos = env.scene["plug"].data.root_pos_w[0]
    print(f"[INFO] Trial {trial_id}: Initial plug position: {plug_pos.cpu().numpy()}")

    # Final status flags
    outcome_str = "fail"
    trial_result = False

    # === Rollout loop ===
    for t in range(horizon):
        policy_obs = obs_dict["policy"]
        qpos_new = env.scene["robot"].data.joint_pos.squeeze(0)[:-1]
        qvel_new = env.scene["robot"].data.joint_vel.squeeze(0)[:-1]

        # Ablation for checking impact of joint history: zero out the joint positions to test if the policy can still succeed based on image input alone
        #qpos_new = torch.zeros(qpos_new.shape, device=device)

        def process_image(img: torch.Tensor, device):
            if img.dim() == 4: img = img.squeeze(0)
            img = img.permute(2, 0, 1)
            return img.to(device)
        
        img1_new = process_image(policy_obs["image"], device)
        
        # Ablation for checking impact of image input: zero out the image to test if the policy can still succeed based on joint history alone
        #img1_new = torch.zeros_like(img1_new)  # Comment out to use actual image input

        images_new = (img1_new.unsqueeze(0).float() / 255.0)

        # Context Buffer Management
        buffer_pos.append(qpos_new)
        buffer_vel.append(qvel_new)
        buffer_images.append(images_new)
        if len(buffer_pos) > context_length:
            buffer_pos.pop(0); buffer_vel.pop(0); buffer_images.pop(0)
        else:
            while len(buffer_pos) < context_length:
                buffer_pos.append(qpos_new); buffer_vel.append(qvel_new); buffer_images.append(images_new)

        qpos = torch.stack(buffer_pos, dim=0)
        qvel = torch.stack(buffer_vel, dim=0) if velocity_control else None
        images = torch.stack(buffer_images, dim=0).permute(1, 0, 2, 3, 4)

        # --- ACT Inference ---
        with torch.inference_mode():
            policy.eval()
            actions_raw = policy(
                pre_process_qpos(qpos).unsqueeze(0),
                images.unsqueeze(0), 
                qvel=pre_process_qvel(qvel).unsqueeze(0) if velocity_control else None,
            ) 

            post_processed_chunk = post_process(actions_raw[0])
            all_time_actions[[t], t : t + actions_raw.shape[1]] = actions_raw[0, :, :]

            # Temporal Ensembling
            actions_t = all_time_actions[:, t]
            valid = torch.any(actions_t != 0, dim=1)
            actions_t = actions_t[valid]

            N = actions_t.shape[0]
            k = 0.01 # Added a small k for actual ensembling, or 0 for mean
            weights = torch.exp(-k * torch.arange(N, device=device))
            weights = (weights / weights.sum()).unsqueeze(1)
            
            raw_action = (actions_t * weights).sum(dim=0, keepdim=True)
            final_action = post_process(raw_action)

        # Check if y-action exceeds y-plug position
        plug_pos = env.scene["plug"].data.root_pos_w[0]
        action_y = final_action[0, 1].item()
        plug_y = plug_pos[1].item()
        y_diff = action_y - plug_y
        if action_y > plug_y:
            print(f"[INFO] Trial {trial_id} Step {t}: Y-action: {action_y:.4f}, Y-plug: {plug_y:.4f}, Diff: {y_diff:.4f}, Action: {final_action[0].cpu().numpy()}")

        # Log data for later saving
        chunk_data_log.append({
            "step": t,
            "action_chunk": post_processed_chunk.cpu().numpy().tolist(), 
            "final_action": final_action[0].cpu().numpy().tolist(),
            "plug_x": plug_pos[0].item(),
            "plug_y": plug_pos[1].item(),
            "plug_z": plug_pos[2].item()
        })

        # --- Map gripper action to environment format ---
        # Convert gripper command: > -0.5 → 0.04 (open), <= -0.5 → 0.0 (closed)
        final_action_isaac = final_action.clone()
        gripper_cmd = final_action[0, -1].item()
        final_action_isaac[0, -1] = 1.0 if gripper_cmd > -0.5 else -1.0

        # --- Step env ---
        #print(f"Trial {trial_id} Step {t}: Executing action {final_action_isaac.cpu().numpy()}")  # Debug print
        obs_dict, _, terminated, truncated, _ = env.step(final_action_isaac.view(1, action_dim).clone())

        # --- Check Termination ---
        is_success = bool(success_term.func(env, **success_term.params)[0])

        if is_success:
            print(f"✅ SUCCESS: Trial {trial_id} @ step {t}")
            outcome_str = "success"
            trial_result = True
            break
        if terminated or truncated:
            reason = "TERMINATED" if terminated else "TIMEOUT"
            print(f"❌ {reason}: Trial {trial_id} @ step {t}")
            outcome_str = "fail"
            trial_result = False
            break

        if t % 100 == 0: torch.cuda.empty_cache()

    # === SAVE CSV AFTER ROLLOUT FINISHES ===
    if chunk_data_log:
        df = pd.DataFrame(chunk_data_log)
        # Construct the new filename
        csv_path = f"action_debug_trial_{trial_id}_{outcome_str}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {outcome_str} debug info to {csv_path}")

    return trial_result, observation_log

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

def load_custom_cube_configurations(file_path):
    """Load custom cube configurations from JSON file.
    
    Args:
        file_path: Path to JSON file containing cube configurations
        
    Returns:
        List of cube configurations, each containing poses for 3 cubes
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Support both new multi-configuration format and legacy single configuration
        if isinstance(data, dict) and "configurations" in data:
            # New format: {"configurations": [{"name": ..., "poses": [...]}, ...]}
            configurations = data["configurations"]
            print(f"[INFO] Loaded {len(configurations)} cube configurations:")
            for i, config in enumerate(configurations):
                print(f"  {i}: {config['name']} - {config.get('description', 'No description')}")
            return configurations
        elif isinstance(data, list):
            # Legacy format: [{"pos": ..., "quat": ...}, ...]
            print(f"[INFO] Converting legacy format to new configuration format")
            legacy_config = {
                "name": "legacy_config",
                "description": "Converted from legacy format",
                "poses": data
            }
            return [legacy_config]
        else:
            print(f"[ERROR] Invalid JSON format in {file_path}")
            return None
            
    except FileNotFoundError:
        print(f"[ERROR] Configuration file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {file_path}: {e}")
        return None


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    # command line parameters
    
    # Load the Isaac Lab environment for the specified task
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False

    # Set terminations
    env_cfg.terminations.time_out = None

    # Disable recorder
    env_cfg.recorders = None

    # Extract success checking function
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Set seed
    torch.manual_seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Acquire device -> Running neural network inference on approapriate device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Load robomimic policy (BC,LSTMGMM, etc.) from checkpoint
    #policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)

    lr_backbone = 1e-5
    backbone = 'resnet18'

    enc_layers = 4 #TODO 4
    dec_layers = 6 #TODO 6
    nheads = 8
    policy_config = {   'ckpt_dir': args_cli.checkpoint,
                        'num_queries': 32, #TODO 64
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'dim_feedforward': 2048, #TODO delete
                        'hidden_dim': 256, #TODO delete
                        'camera_names': ["image"],
                        'velocity_control': args_cli.velocity_control,
                        'context_length': args_cli.context_length
                        
                        }
    from act_copy.policy_runner import ACTPolicy
    policy = ACTPolicy(policy_config)
    loading_status = policy.load_state_dict(torch.load(args_cli.checkpoint))
    policy.cuda()
    policy.eval()
    
    # Read sequence_length from the policy configuration
    
    # Run policy
    results = []
    all_successful_observations = []
    all_csv_data = []
    stats_path = os.path.join(args_cli.data_path, "dataset_act_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    stats_torch = {
        "qpos_mean": torch.from_numpy(stats["qpos_mean"]).float().to(device),
        "qpos_std": torch.from_numpy(stats["qpos_std"]).float().to(device),
        "qvel_mean": torch.from_numpy(stats["qvel_mean"]).float().to(device) if args_cli.velocity_control else None,
        "qvel_std": torch.from_numpy(stats["qvel_std"]).float().to(device) if args_cli.velocity_control else None,
        "action_mean": torch.from_numpy(stats["action_mean"]).float().to(device),
        "action_std": torch.from_numpy(stats["action_std"]).float().to(device),
    }
    
    # Run multiple independent rollouts for statistical evaluation
    best_start = 0
    best_end = 0
    best_success = 0
    #for  end in range(64):
    #    for start in range(10):
    # Install a fixed-plug-pose reset event so the plug always starts at the same place
    #update_environment_events(env)

    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")

        success, obs_log = rollout(
                policy,
                env,
                success_term,
                args_cli.horizon,
                device,
                stats_torch,
                save_observations=args_cli.save_observations,
                trial_id=trial,
                velocity_control=args_cli.velocity_control,
                context_length=args_cli.context_length,
                trial =trial    
        )       
        results.append(success)
        
        # Collect successful observations if --save_observations is enabled
        # Only save data from successful trials
        if args_cli.save_observations and obs_log is not None:
            all_successful_observations.append({
                "trial": trial,
                "observations": obs_log,
                "success": True,
                "metadata": {
                    "task": args_cli.task,
                    "checkpoint": args_cli.checkpoint,
                    "horizon": args_cli.horizon,
                    "frequency_hz": 20,
                    "dt": 0.05,
                    "total_steps": len(obs_log),
                    "norm_factor_min": args_cli.norm_factor_min,
                    "norm_factor_max": args_cli.norm_factor_max
                }
            })
            
            # Collect CSV data from successful trials
            
        
        print(f"[INFO] Trial {trial}: {success}\n")

        print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
        print(f"Success rate: {results.count(True) / len(results)}")
        print(f"Trial Results: {results}\n")
        """if results.count(True) > best_success:
            best_success = results.count(True)
            best_start = start
            best_end = end"""
    print(f"Best start: {best_start}, Best end: {best_end}, with success: {best_success} out of {len(results)}")

    # Save all successful observations (if enabled and collected)
    if args_cli.save_observations and all_successful_observations:
        # Get the directory from the CSV output file path
        csv_dir = os.path.dirname(args_cli.csv_output_file)
        csv_basename = os.path.splitext(os.path.basename(args_cli.csv_output_file))[0]
        
        # Construct PKL and JSON file paths in the same directory as CSV
        pkl_file = os.path.join(csv_dir, f"{csv_basename}.pkl")
        json_file = os.path.join(csv_dir, f"{csv_basename}.json")
        
        # Save as pickle
        with open(pkl_file, "wb") as f:
            pickle.dump(all_successful_observations, f)
        print(f"\nSaved {len(all_successful_observations)} successful trajectories to {pkl_file}")
        
        # Save CSV data
        if all_csv_data:
            df = pd.DataFrame(all_csv_data)
            df.to_csv(args_cli.csv_output_file, index=False)
            print(f"Saved detailed observations to CSV: {args_cli.csv_output_file}")
            print(f"CSV contains {len(df)} rows with columns: {list(df.columns)}")
        
        # Also save as JSON for easier inspection
        json_data = []
        for traj_data in all_successful_observations:
            json_traj = {
                "trial": traj_data["trial"], 
                "metadata": traj_data["metadata"],
                "observations": []
            }
            for obs in traj_data["observations"]:
                json_obs = {}
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        json_obs[key] = value.tolist()
                    else:
                        json_obs[key] = value
                json_traj["observations"].append(json_obs)
            json_data.append(json_traj)
        
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Also saved as JSON: {json_file}")
        
        # Print summary statistics
        lengths = [len(traj["observations"]) for traj in all_successful_observations]
        print(f"Episode lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
        
        # Print sample observation shapes for verification
        if all_successful_observations:
            sample_obs = all_successful_observations[0]["observations"][0]
            print(f"\nSample observation shapes:")
            for key, value in sample_obs.items():
                if key not in ["step", "timestamp"]:
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
    elif args_cli.save_observations and not all_successful_observations:
        print(f"\n[WARNING] No successful trajectories to save. Success rate: {results.count(True) / len(results)}")

    env.close()

if __name__ == '__main__':

    
    main()
    # close sim app
    simulation_app.close()