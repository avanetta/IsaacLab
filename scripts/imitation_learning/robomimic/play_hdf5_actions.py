import argparse
import os
import h5py
import torch
import gymnasium as gym
import numpy as np
import random
from isaaclab.app import AppLauncher

# 1. Setup Parser
parser = argparse.ArgumentParser(description="Replay HDF5 actions in Isaac Lab.")
parser.add_argument("--task", type=str, required=True, help="Name of the task (e.g., sim_usbc).")
parser.add_argument("--hdf5_path", type=str, required=True, help="Path to your demonstrations.hdf5")
parser.add_argument("--num_demos", type=int, default=None, help="Number of demos to replay.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3. Imports that require simulation_app to be running
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler
import isaaclab_mimic.envs 

def replay_demo(env, episode, demo_name):
    """Feeds recorded actions and set up initial state in the simulation."""
    print(f"\n[>>>] Replaying {demo_name} | Steps: {len(episode.data['actions'])}")
    
    obs, _ = env.reset()
    
    # Set object poses from the recorded initial state if available
    initial_state = episode.data.get("initial_state", {})
    if "rigid_object" in initial_state:
        for obj_name, obj_data in initial_state["rigid_object"].items():
            if "root_pose" in obj_data:
                pose_data = obj_data["root_pose"]
                # Handle both numpy arrays and tensors
                if isinstance(pose_data, torch.Tensor):
                    root_pose = pose_data.float().to(env.device)
                else:
                    root_pose = torch.from_numpy(pose_data).float().to(env.device)
                try:
                    env.set_object_pose(root_pose, obj_name=obj_name)
                    print(f"  Set initial pose for '{obj_name}'")
                except (AttributeError, RuntimeError, ValueError) as e:
                    print(f"  Warning: Could not set object pose for {obj_name}: {e}")
    
    actions = episode.data["actions"]
    for t in range(len(actions)):
        # Convert action to torch tensor on the simulation device
        if isinstance(actions[t], torch.Tensor):
            action_tensor = actions[t].to(env.device).float().unsqueeze(0)
        else:
            action_tensor = torch.from_numpy(actions[t]).to(env.device).float().unsqueeze(0)
        
        # Step the physics
        obs, reward, terminated, truncated, info = env.step(action_tensor)
        
        if terminated or truncated:
            print(f"[!] Env signaled end at step {t}")
            break
            
    print(f"[SUCCESS] Finished {demo_name}")

def main():
    # Setup Env
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.terminations.time_out = None 
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    if not os.path.exists(args_cli.hdf5_path):
        print(f"[ERROR] File not found: {args_cli.hdf5_path}")
        return

    # Load dataset using HDF5DatasetFileHandler
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.hdf5_path)
    
    episode_names = sorted(list(dataset_file_handler.get_episode_names()))
    
    # Randomly select N demos if specified
    if args_cli.num_demos:
        if args_cli.num_demos > len(episode_names):
            print(f"[WARNING] Requested {args_cli.num_demos} demos but only {len(episode_names)} available. Using all.")
            episode_names = episode_names
        else:
            episode_names = random.sample(episode_names, args_cli.num_demos)
            print(f"[INFO] Randomly selected {args_cli.num_demos} demos from {len(dataset_file_handler.get_episode_names())} total")
    
    for episode_name in episode_names:
        # Load full episode data (including initial state and object poses)
        episode = dataset_file_handler.load_episode(episode_name, env.device)
        replay_demo(env, episode, episode_name)

    print("\n[INFO] All replays completed.")
    env.close()
    dataset_file_handler.close()

if __name__ == '__main__':
    main()
    simulation_app.close()