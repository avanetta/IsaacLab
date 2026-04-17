import h5py
import os
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import json
import argparse

def find_csv_for_demo(csv_folder, demo_num):
    """
    Find the CSV file matching pattern demo_x_recording_* for a given demo number.
    """
    pattern = os.path.join(csv_folder, f"demo_{demo_num}_recording_*.csv")
    matching_files = glob.glob(pattern)
    if not matching_files:
        raise FileNotFoundError(f"No CSV file found for demo_{demo_num} in {csv_folder}")
    return matching_files[0]  # Return first match

def add_scene_state_from_poses_csv(target_path, poses_csv_path, socket_offset=None):
    """
    Adds per-demo plug and socket poses to the HDF5 file from a CSV file.
    Loads socket poses from CSV and calculates plug pose from socket.
    
    Args:
        target_path: Path to the HDF5 file to update
        poses_csv_path: Path to CSV file with columns: demo_id,x,y,z,qw,qx,qy,qz (socket poses)
        socket_offset: Offset from socket pose to plug pose. Default: [-0.017, 0, -0.12125]
    """
    if socket_offset is None:
        socket_offset = np.array([-0.017, 0, -0.12125], dtype=np.float32)
    else:
        socket_offset = np.array(socket_offset, dtype=np.float32)
    
    if not os.path.exists(target_path):
        print(f"Error: Target file {target_path} not found.")
        return
    
    if not os.path.exists(poses_csv_path):
        print(f"Error: Poses CSV file {poses_csv_path} not found.")
        return
    
    # Load poses CSV (socket poses)
    poses_df = pd.read_csv(poses_csv_path)
    print(f"Loaded socket poses for {len(poses_df)} demos from {poses_csv_path}")
    
    root_velocity = np.zeros((1, 6), dtype=np.float32)  # [vx, vy, vz, wx, wy, wz]
    
    with h5py.File(target_path, 'a') as f:
        if 'data' not in f:
            print("Error: 'data' group not found in HDF5.")
            return
        
        data_group = f['data']
        
        for demo_key in tqdm(data_group.keys(), desc="Adding scene states from CSV"):
            demo = data_group[demo_key]
            
            # Extract demo number from demo_X
            try:
                demo_num = int(demo_key.split('_')[1])
            except (ValueError, IndexError):
                print(f"Warning: Could not extract demo number from {demo_key}, skipping...")
                continue
            
            # Look up pose in CSV (socket pose)
            demo_poses = poses_df[poses_df['demo_id'] == demo_num]
            if demo_poses.empty:
                print(f"Warning: No pose found for demo_{demo_num} in CSV, skipping...")
                continue
            
            # Extract socket pose from CSV (no transformations)
            pose_row = demo_poses.iloc[0]
            socket_pose_flat = np.array([
                pose_row['x'], pose_row['y'], pose_row['z'],
                pose_row['qw'], pose_row['qx'], pose_row['qy'], pose_row['qz']
            ], dtype=np.float32)
            
            socket_pose = socket_pose_flat.reshape(1, 7)
            
            # Calculate plug pose from socket
            plug_pose_flat = socket_pose_flat.copy()
            plug_pose_flat[:3] += socket_offset  # Add offset to position
            plug_pose = plug_pose_flat.reshape(1, 7)
            
            # Ensure high-level groups exist
            for group in ["states/rigid_object", "initial_state/rigid_object"]:
                if group not in demo:
                    demo.create_group(group)
            
            # Add plug and socket poses
            for asset_name, pose_data in [("plug", plug_pose), ("socket", socket_pose)]:
                # Path 1: states (Temporal/Trajectory data)
                state_path = f"states/rigid_object/{asset_name}/root_pose"
                if state_path in demo:
                    del demo[state_path]
                demo.create_dataset(state_path, data=pose_data)
                
                velocity_path = f"states/rigid_object/{asset_name}/root_velocity"
                if velocity_path in demo:
                    del demo[velocity_path]
                demo.create_dataset(velocity_path, data=root_velocity)
                
                # Path 2: initial_state (The reset snapshot)
                init_path = f"initial_state/rigid_object/{asset_name}/root_pose"
                if init_path in demo:
                    del demo[init_path]
                demo.create_dataset(init_path, data=pose_data)
                
                init_vel_path = f"initial_state/rigid_object/{asset_name}/root_velocity"
                if init_vel_path in demo:
                    del demo[init_vel_path]
                demo.create_dataset(init_vel_path, data=root_velocity)
    
    print(f"\nSuccess! Added scene states from CSV to all demos.")

def preprocess_dataset(source_path, target_path, task_name, csv_folder, poses_csv_path=None):
    """
    Combines gripper width loading and 'prepare_metadata' into one pass.
    1. Loads gripper width from CSV files and converts to finger positions (qpos).
    2. Nests demos inside a 'data' group.
    3. Adds IsaacLab metadata and scene states (plug/socket from poses CSV).
    """
    # If poses_csv_path not provided, look for object_poses.csv in csv_folder
    if poses_csv_path is None:
        poses_csv_path = os.path.join(csv_folder, "object_poses.csv")
    
    if not os.path.exists(source_path):
        print(f"Error: Source file {source_path} not found.")
        return

    env_args = {
        "task_name": task_name,
        "env_name": task_name,
        "type": "isaac_lab",
        "control_type": "ee_pose", 
    }

    with h5py.File(source_path, 'r') as s, h5py.File(target_path, 'w') as t:
        # Create mandatory 'data' group
        dest_data_group = t.create_group('data')
        
        # Identify demos (handle files that already have 'data' or don't)
        source_root = s['data'] if 'data' in s else s
        demo_keys = [k for k in source_root.keys() if k.startswith('demo_')]
        
        print(f"Processing {len(demo_keys)} demos for task: {task_name}")

        for demo_key in tqdm(demo_keys, desc="Step 1/2: Processing Gripper & Nesting"):
            source_demo = source_root[demo_key]
            dest_demo = dest_data_group.create_group(demo_key)
            
            # --- 1. Load Gripper Width from CSV ---
            demo_num = int(demo_key.split('_')[1])  # Extract number from demo_X
            csv_path = find_csv_for_demo(csv_folder, demo_num)
            df = pd.read_csv(csv_path)
            
            if 'actual_gripper_width' not in df.columns:
                raise ValueError(f"Column 'actual_gripper_width' not found in {csv_path}")
            
            gripper_width = df['actual_gripper_width'].values.astype(np.float32)
            finger_pos = gripper_width / 2.0
            gripper_column = np.stack([finger_pos], axis=1) # [T, 1]

            # --- 2. Copy Structure and Apply Transformation ---
            def process_item(name, obj):
                if isinstance(obj, h5py.Group):
                    if obj != source_demo:
                        dest_demo.create_group(name)
                elif isinstance(obj, h5py.Dataset):
                    data = obj[()]
                    # Merge qpos with the new smooth gripper column
                    if name == 'states/articulation/robot/joint_position':
                        new_data = np.concatenate([data, gripper_column], axis=1)
                    else:
                        new_data = data
                    
                    dset = dest_demo.create_dataset(name, data=new_data, compression=obj.compression)
                    for k, v in obj.attrs.items(): dset.attrs[k] = v

            source_demo.visititems(process_item)

        # --- Step 2: Metadata Attachment (from prepare_metadata) ---
        print("\nStep 2/2: Attaching Metadata...")
        dest_data_group.attrs["env_args"] = json.dumps(env_args)
        dest_data_group.attrs["env_name"] = task_name
        dest_data_group.attrs["num_demos"] = len(demo_keys)
        dest_data_group.attrs["teleop_device"] = "keyboard"
        
        for attr_name, attr_val in s.attrs.items():
            t.attrs[attr_name] = attr_val

    # Add scene states (plug/socket poses) from CSV
    if os.path.exists(poses_csv_path):
        add_scene_state_from_poses_csv(target_path, poses_csv_path)
    else:
        print(f"Warning: Poses CSV not found at {poses_csv_path}, skipping scene state setup")
    print(f"\nDone! Preprocessed dataset saved to: {target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess HDF5 for Isaac Lab Mimic")
    parser.add_argument("--input", type=str, required=True, help="Path to raw recorded HDF5")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed HDF5")
    parser.add_argument("--task", type=str, default="Isaac-Unplug-Franka-IK-Abs-Mimic-RGB-v0")
    parser.add_argument("--csv_folder", type=str, required=True, help="Path to folder containing demo_x_recording_*.csv and object_poses.csv files")
    
    args = parser.parse_args()
    
    preprocess_dataset(args.input, args.output, args.task, args.csv_folder)