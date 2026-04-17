import h5py
import os
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as T
from PIL import Image
import json

class ImageAugmenter:
    """Fast image augmentation pipeline using PyTorch."""
    
    def __init__(self, apply_crop=True):
        self.apply_crop = apply_crop
        
        # Color jitter
        self.color_jitter = T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        
        # Random crop (only for non-wrist cameras)
        self.random_crop = T.RandomResizedCrop(
            size=(224, 224),  # Adjust based on your image size
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        )
        
        # Gaussian noise will be added separately
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def augment(self, image_array, is_wrist_cam=False):
        """
        Augment a single image.
        
        Args:
            image_array: numpy array of shape (H, W, C) with values in [0, 255]
            is_wrist_cam: if True, skip cropping
        
        Returns:
            Augmented image as numpy array
        """
        # Convert to PIL Image for torchvision transforms
        if image_array.dtype == np.uint8:
            img = Image.fromarray(image_array)
        else:
            img = Image.fromarray((image_array * 255).astype(np.uint8))
        
        # Apply color jitter
        img = self.color_jitter(img)
        
        # Apply random crop (skip for wrist camera)
        if self.apply_crop and not is_wrist_cam:
            img = self.random_crop(img)
        
        # Convert to tensor and add Gaussian noise
        img_tensor = T.ToTensor()(img).to(self.device)
        
        # Add Gaussian noise (std=0.02 in normalized space)
        noise = torch.randn_like(img_tensor) * 0.02
        img_tensor = torch.clamp(img_tensor + noise, 0, 1)
        
        # Convert back to numpy
        img_array = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        return img_array
    
    def augment_batch(self, images, is_wrist_cam=False):
        """
        Augment a batch of images efficiently.
        
        Args:
            images: numpy array of shape (T, H, W, C)
            is_wrist_cam: if True, skip cropping
        
        Returns:
            Augmented images as numpy array
        """
        augmented = []
        for img in images:
            augmented.append(self.augment(img, is_wrist_cam))
        return np.stack(augmented, axis=0)

def create_shortened_dataset_copy(source_path, target_path, skip_steps=50):
    """
    Creates a new HDF5 file where the first 50 steps of every demo are removed.
    The original file at source_path is not modified.
    """
    if not os.path.exists(source_path):
        print(f"Error: Source file {source_path} not found.")
        return

    with h5py.File(source_path, 'r') as s, h5py.File(target_path, 'w') as t:
        # Check if the root has a 'data' group as per your class logic
        if 'data' not in s:
            print("Error: Could not find 'data' group in source HDF5.")
            return
            
        dest_data_group = t.create_group('data')
        demo_keys = list(s['data'].keys())
        
        for demo_key in tqdm(demo_keys, desc="Processing episodes"):
            source_demo = s['data'][demo_key]
            dest_demo = dest_data_group.create_group(demo_key)
            
            # Recursively copy and slice datasets within each demo
            def process_demo_contents(name, obj):
                # Calculate the relative path within the demo group
                # (e.g., 'obs/cam_high' or 'actions')
                if isinstance(obj, h5py.Dataset):
                    data = obj[()]
                    
                    # Only slice if the dataset has a time dimension and is long enough
                    if len(data.shape) > 0 and data.shape[0] > skip_steps:
                        new_data = data[skip_steps:]
                    else:
                        # For metadata or very short demos, keep as is or empty
                        new_data = data
                    
                    # Create dataset in the new file with original compression if available
                    dset = dest_demo.create_dataset(name, data=new_data, compression=obj.compression)
                    
                    # Copy attributes (metadata like 'sim' or 'stats')
                    for attr_name, attr_val in obj.attrs.items():
                        dset.attrs[attr_name] = attr_val
                
                elif isinstance(obj, h5py.Group) and obj != source_demo:
                    # Create internal subgroups (like 'obs' or 'states')
                    dest_demo.create_group(name)

            source_demo.visititems(process_demo_contents)

    print(f"\nSuccess! Copy created at: {target_path}")

def merge_datasets(source_path1, source_path2, target_path):
    """
    Merges two HDF5 files with the same layout into a single file.
    Demos from both files are combined and renamed sequentially as demo_0, demo_1, ..., demo_n.
    
    Args:
        source_path1: Path to first source HDF5 file
        source_path2: Path to second source HDF5 file
        target_path: Path to merged output HDF5 file
    """
    if not os.path.exists(source_path1):
        print(f"Error: Source file {source_path1} not found.")
        return
    
    if not os.path.exists(source_path2):
        print(f"Error: Source file {source_path2} not found.")
        return
    
    with h5py.File(source_path1, 'r') as s1, h5py.File(source_path2, 'r') as s2, \
         h5py.File(target_path, 'w') as t:
        
        # Check both files have 'data' group
        if 'data' not in s1:
            print("Error: Could not find 'data' group in first source HDF5.")
            return
        if 'data' not in s2:
            print("Error: Could not find 'data' group in second source HDF5.")
            return
        
        dest_data_group = t.create_group('data')
        
        # Get demo keys from both files
        demo_keys1 = list(s1['data'].keys())
        demo_keys2 = list(s2['data'].keys())
        
        total_demos = len(demo_keys1) + len(demo_keys2)
        
        print(f"First dataset: {len(demo_keys1)} demos")
        print(f"Second dataset: {len(demo_keys2)} demos")
        print(f"Total demos to merge: {total_demos}")
        print(f"Output will contain: demo_0 to demo_{total_demos - 1}")
        
        def copy_demo(source_demo, dest_group, demo_key):
            """Helper function to recursively copy a demo and all its contents."""
            dest_demo = dest_group.create_group(demo_key)
            
            # Store group structure
            groups_to_create = {}
            
            # First pass: collect all groups
            def collect_groups(name, obj):
                if isinstance(obj, h5py.Group) and obj != source_demo:
                    groups_to_create[name] = obj
            
            source_demo.visititems(collect_groups)
            
            # Create all groups first
            for group_name in sorted(groups_to_create.keys()):
                dest_demo.create_group(group_name)
            
            # Second pass: copy all datasets
            def copy_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data = obj[()]
                    dset = dest_demo.create_dataset(name, data=data, compression=obj.compression)
                    
                    # Copy attributes
                    for attr_name, attr_val in obj.attrs.items():
                        dset.attrs[attr_name] = attr_val
            
            source_demo.visititems(copy_dataset)
        
        # Counter for sequential naming
        demo_counter = 0
        
        # Copy demos from first file
        print("\nCopying demos from first dataset...")
        for original_key in tqdm(demo_keys1, desc="First dataset"):
            new_key = f"demo_{demo_counter}"
            copy_demo(s1['data'][original_key], dest_data_group, new_key)
            demo_counter += 1
        
        # Copy demos from second file
        print("\nCopying demos from second dataset...")
        for original_key in tqdm(demo_keys2, desc="Second dataset"):
            new_key = f"demo_{demo_counter}"
            copy_demo(s2['data'][original_key], dest_data_group, new_key)
            demo_counter += 1
        
    print(f"\nSuccess! Merged dataset created at: {target_path}")
    print(f"Total demos in merged file: {demo_counter} (demo_0 to demo_{demo_counter - 1})")

def delete_demo_and_reindex(hdf5_path, demo_indices):
    """
    Deletes specific demos from an HDF5 file and renumbers all remaining demos sequentially.
    
    Example:
        - Before: demo_0, demo_1, demo_2, demo_3, demo_4
        - Delete: [1, 3]
        - After: demo_0, demo_1 (was demo_2), demo_2 (was demo_4)
    
    Args:
        hdf5_path: Path to the HDF5 file to modify (modified in-place)
        demo_indices: Integer or list of integers specifying demos to delete
    """
    # Convert single integer to list for backward compatibility
    if isinstance(demo_indices, int):
        demo_indices = [demo_indices]
    else:
        demo_indices = list(demo_indices)
    
    # Convert to set for faster lookup
    indices_to_delete = set(demo_indices)
    
    if not os.path.exists(hdf5_path):
        print(f"Error: File {hdf5_path} not found.")
        return
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'data' not in f:
            print("Error: Could not find 'data' group in HDF5.")
            return
        
        # Sort by numeric index, not string (so demo_2 comes before demo_10)
        all_demos = sorted(list(f['data'].keys()), key=lambda x: int(x.split('_')[1]) if x.startswith('demo_') else float('inf'))
        
        # Validate demos exist
        missing = []
        for idx in indices_to_delete:
            if f"demo_{idx}" not in all_demos:
                missing.append(idx)
        
        if missing:
            print(f"Error: The following demo indices not found: {missing}")
            print(f"Available demos: {all_demos}")
            return
        
        print(f"Deleting {len(indices_to_delete)} demo(s): {sorted(indices_to_delete)}")
        remaining = len(all_demos) - len(indices_to_delete)
        print(f"Keeping {remaining} demo(s), will be reindexed as demo_0 to demo_{remaining - 1}")
    
    # Use temporary file for safe rewriting
    temp_path = hdf5_path + ".tmp"
    
    with h5py.File(hdf5_path, 'r') as source:
        with h5py.File(temp_path, 'w') as target:
            dest_data_group = target.create_group('data')
            
            new_idx = 0
            for old_demo_key in all_demos:
                try:
                    old_idx = int(old_demo_key.split('_')[1])
                except (ValueError, IndexError):
                    continue
                
                # Skip demos marked for deletion
                if old_idx in indices_to_delete:
                    print(f"  Skipping {old_demo_key}...")
                    continue
                
                # Rename to sequential numbering
                new_key = f"demo_{new_idx}"
                if old_demo_key != new_key:
                    print(f"  Renaming {old_demo_key} → {new_key}")
                
                # Copy demo with new name
                source_demo = source['data'][old_demo_key]
                dest_demo = dest_data_group.create_group(new_key)
                
                # Recursively copy all data and groups
                def copy_item(name, obj):
                    if isinstance(obj, h5py.Group):
                        if obj != source_demo:  # Skip root group
                            dest_demo.create_group(name)
                    elif isinstance(obj, h5py.Dataset):
                        data = obj[()]
                        dset = dest_demo.create_dataset(
                            name, 
                            data=data, 
                            compression=obj.compression
                        )
                        # Copy attributes
                        for attr_name, attr_val in obj.attrs.items():
                            dset.attrs[attr_name] = attr_val
                
                source_demo.visititems(copy_item)
                new_idx += 1
    
    # Save corrected version with _corrected suffix
    base_path, ext = os.path.splitext(hdf5_path)
    corrected_path = f"{base_path}_corrected{ext}"
    os.replace(temp_path, corrected_path)
    print(f"\nSuccess! Deleted demos {sorted(indices_to_delete)} and reindexed all remaining demos.")
    print(f"Saved corrected file: {corrected_path}")

def add_gripper_to_qpos(source_path, target_path, pos=False, no_shift=False):
    """
    Copies the raw gripper column (last column of 'actions') and appends it
    to 'states/articulation/robot/joint_position'.
    
    If pos=False (default) and no_shift=False:
        The column is shifted down by one step (first element duplicated at the
        top, last element dropped) so that the state at time t reflects the
        gripper command issued at time t-1.  Values are kept as-is (1.0 / -1.0).
    
    If pos=False and no_shift=True:
        Values are kept as-is (1.0 / -1.0) with no shifting applied.
    
    If pos=True:
        The gripper command is converted to finger joint positions:
            1.0  (open)  -> 0.04
           -1.0  (close) -> 0.0
        No shifting is applied (state at time t = command at time t).
    
    Before: qpos shape [T, 7], action shape [T, 8]
    After: qpos shape [T, 8] with gripper state appended
    
    Args:
        source_path: Path to source HDF5 file
        target_path: Path to target HDF5 file
        pos: If True, map gripper to joint positions (0.04/0.0) without shifting
        no_shift: If True, skip the one-step shift when pos=False
    """
    if not os.path.exists(source_path):
        print(f"Error: Source file {source_path} not found.")
        return
    
    with h5py.File(source_path, 'r') as s, h5py.File(target_path, 'w') as t:
        if 'data' not in s:
            print("Error: Could not find 'data' group in source HDF5.")
            return
        
        dest_data_group = t.create_group('data')
        demo_keys = list(s['data'].keys())
        
        print(f"Processing {len(demo_keys)} demonstrations...")
        print("Appending gripper column from actions to joint_position...\n")
        
        for demo_key in tqdm(demo_keys, desc="Processing episodes"):
            source_demo = s['data'][demo_key]
            dest_demo = dest_data_group.create_group(demo_key)
            
            # Derive gripper state from actions
            gripper_column = None
            if 'actions' in source_demo:
                actions = source_demo['actions'][()]
                raw_gripper = actions[:, -1].astype(np.float32)  # Shape [T]
                
                if pos:
                    # Map command values to finger joint positions:
                    #   1.0 (open)  -> 0.04
                    #  -1.0 (close) -> 0.0
                    gripper_pos = np.where(raw_gripper > 0, 0.04, 0.0).astype(np.float32)
                    gripper_column = gripper_pos.reshape(-1, 1)
                    print(f"  {demo_key}: gripper column (pos mode) shape = {gripper_column.shape}, "
                          f"open={np.sum(gripper_pos == 0.04)}, closed={np.sum(gripper_pos == 0.0)}")
                elif no_shift:
                    # No shifting: use raw gripper values directly
                    gripper_column = raw_gripper.reshape(-1, 1)
                    print(f"  {demo_key}: gripper column (raw, no shift) shape = {gripper_column.shape}")
                else:
                    # Shift down by 1: duplicate first element at top, drop last element
                    shifted = np.empty_like(raw_gripper)
                    shifted[0] = raw_gripper[0]
                    shifted[1:] = raw_gripper[:-1]
                    gripper_column = shifted.reshape(-1, 1)
                    print(f"  {demo_key}: gripper column (raw mode) shape = {gripper_column.shape}")
            else:
                print(f"  Warning: No 'actions' dataset in {demo_key}, skipping gripper append")
            
            # Store group structure
            groups_to_create = {}
            
            # First pass: collect all groups
            def collect_groups(name, obj):
                if isinstance(obj, h5py.Group) and obj != source_demo:
                    groups_to_create[name] = obj
            
            source_demo.visititems(collect_groups)
            
            # Create all groups first
            for group_name in sorted(groups_to_create.keys()):
                dest_demo.create_group(group_name)
            
            # Second pass: copy datasets with modification for qpos
            def copy_dataset(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                
                data = obj[()]
                
                # Check if this is specifically the joint_position dataset we want to modify
                if name == 'states/articulation/robot/joint_position' and gripper_column is not None:
                    # Verify shapes match
                    if data.shape[0] == gripper_column.shape[0]:
                        new_data = np.concatenate([data, gripper_column], axis=1)
                        print(f"    Modified {name}: {data.shape} → {new_data.shape}")
                    else:
                        print(f"    Warning: Shape mismatch for {name}. Keeping original.")
                        new_data = data
                else:
                    new_data = data
                
                # Create dataset
                dset = dest_demo.create_dataset(name, data=new_data, compression=obj.compression)
                
                # Copy attributes
                for attr_name, attr_val in obj.attrs.items():
                    dset.attrs[attr_name] = attr_val
            
            source_demo.visititems(copy_dataset)
    
    print(f"\nSuccess! Dataset with gripper appended to qpos created at: {target_path}")

def add_gripper_smooth(source_path, target_path, closing_duration=13):
    """
    Appends smooth gripper finger position to both 
    'states/articulation/robot/joint_position' and 'obs/joint_pos'.
    The result has 8D qpos: 7 arm joints + 1 gripper finger.
    The closing movement starts exactly when the action sequence switches from 1 to -1.
    """
    if not os.path.exists(source_path):
        print(f"Error: Source file {source_path} not found.")
        return
    
    with h5py.File(source_path, 'r') as s, h5py.File(target_path, 'w') as t:
        dest_data_group = t.create_group('data')
        
        for demo_key in tqdm(list(s['data'].keys()), desc="Smoothing gripper (8-dim)"):
            source_demo = s['data'][demo_key]
            dest_demo = dest_data_group.create_group(demo_key)
            
            # 1. Get Actions and compute smooth gripper trajectory
            actions = source_demo['actions'][()]
            raw_gripper_cmd = actions[:, -1] # 1.0 or -1.0
            T = len(raw_gripper_cmd)
            
            # Initialize finger at 'open' (0.04)
            finger_pos = np.ones(T, dtype=np.float32) * 0.04
            
            # Detect switch 1 -> -1 and interpolate
            # We look for index i where cmd[i-1] == 1 and cmd[i] == -1
            closing_starts = np.where((raw_gripper_cmd[:-1] == 1) & (raw_gripper_cmd[1:] == -1))[0] + 1
            
            for start_idx in closing_starts:
                # Create a ramp over closing_duration
                end_idx = min(start_idx + closing_duration, T)
                duration = end_idx - start_idx
                
                # Linear ramp from 0.04 down to 0.0
                ramp = np.linspace(0.04, 0.0, duration)
                finger_pos[start_idx:end_idx] = ramp
                
                # Keep fingers closed after ramp ends
                finger_pos[end_idx:] = 0.0
            
            # Ensure 1s (open) stay at 0.04
            finger_pos[raw_gripper_cmd == 1] = 0.04

            # Single gripper column
            gripper_columns = finger_pos.reshape(-1, 1) # [T, 1]
            
            # 2. Copy/Modify structure
            def copy_item(name, obj):
                if isinstance(obj, h5py.Group) and obj != source_demo:
                    dest_demo.create_group(name)
                elif isinstance(obj, h5py.Dataset):
                    data = obj[()]
                    # Append gripper column to both joint_position datasets
                    if name in ('states/articulation/robot/joint_position', 'obs/joint_pos'):
                        new_data = np.concatenate([data, gripper_columns], axis=1)
                    else:
                        new_data = data
                    
                    dset = dest_demo.create_dataset(name, data=new_data, compression=obj.compression)
                    for k, v in obj.attrs.items(): dset.attrs[k] = v
            
            source_demo.visititems(copy_item)
            
    print(f"\nSuccess! Smooth gripper (8-dim qpos with 1 gripper finger) saved to: {target_path}")

def create_augmented_dataset_copy(source_path, target_path, noise_steps=50, 
                                   high_noise_std=0.2, low_noise_std=0.05):
    """
    Creates a new HDF5 file with noise augmentation applied to the entire dataset.
    
    Args:
        source_path: Path to source HDF5 file
        target_path: Path to target HDF5 file
        noise_steps: Number of initial steps to apply high noise to joint positions
        high_noise_std: Standard deviation of noise for first noise_steps
        low_noise_std: Standard deviation of noise for remaining steps
    """
    if not os.path.exists(source_path):
        print(f"Error: Source file {source_path} not found.")
        return

    # Initialize image augmenter
    img_augmenter = ImageAugmenter()
    
    # Keywords to identify different data types
    joint_keywords = ['joint', 'qpos', 'position', 'eef_pos', 'eef_quat']
    image_keywords = ['image', 'cam', 'rgb', 'img']
    
    with h5py.File(source_path, 'r') as s, h5py.File(target_path, 'w') as t:
        if 'data' not in s:
            print("Error: Could not find 'data' group in source HDF5.")
            return
            
        dest_data_group = t.create_group('data')
        demo_keys = list(s['data'].keys())
        
        print(f"Processing {len(demo_keys)} demonstrations with augmentation...")
        
        for demo_key in tqdm(demo_keys, desc="Augmenting episodes"):
            source_demo = s['data'][demo_key]
            dest_demo = dest_data_group.create_group(demo_key)
            
            # Store group structure
            groups_to_create = {}
            
            # First pass: collect all groups
            def collect_groups(name, obj):
                if isinstance(obj, h5py.Group) and obj != source_demo:
                    groups_to_create[name] = obj
            
            source_demo.visititems(collect_groups)
            
            # Create all groups first
            for group_name in sorted(groups_to_create.keys()):
                dest_demo.create_group(group_name)
            
            # Second pass: process datasets
            def process_dataset(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                
                data = obj[()]
                
                # Check if this is time-series data
                is_temporal = len(data.shape) > 0 and data.shape[0] > 1
                
                if not is_temporal:
                    # Non-temporal data: copy as-is
                    new_data = data
                else:
                    # Determine data type by name
                    name_lower = name.lower()
                    is_joint_data = any(kw in name_lower for kw in joint_keywords)
                    is_image_data = any(kw in name_lower for kw in image_keywords)
                    
                    # Check if this is image 0 (wrist camera)
                    is_wrist_cam = 'image_0' in name_lower or 'cam_0' in name_lower
                    
                    if is_joint_data:
                        # Apply noise to joint positions
                        new_data = data.copy()
                        seq_len = data.shape[0]
                        
                        # High noise for first noise_steps
                        if seq_len > 0:
                            high_noise_len = min(noise_steps, seq_len)
                            high_noise = np.random.normal(0, high_noise_std, 
                                                         new_data[:high_noise_len].shape)
                            new_data[:high_noise_len] += high_noise
                            
                            # Low noise for remaining steps
                            if seq_len > noise_steps:
                                low_noise = np.random.normal(0, low_noise_std,
                                                            new_data[noise_steps:].shape)
                                new_data[noise_steps:] += low_noise
                    
                    elif is_image_data:
                        # Apply image augmentation
                        print(f"\n  Augmenting images in {name}...")
                        new_data = img_augmenter.augment_batch(data, is_wrist_cam=is_wrist_cam)
                    
                    else:
                        # Other temporal data: copy as-is
                        new_data = data
                
                # Create dataset with same compression
                dset = dest_demo.create_dataset(name, data=new_data, compression=obj.compression)
                
                # Copy attributes
                for attr_name, attr_val in obj.attrs.items():
                    dset.attrs[attr_name] = attr_val
            
            source_demo.visititems(process_dataset)

    print(f"\nSuccess! Augmented dataset created at: {target_path}")

def add_isaaclab_scene_state(target_path, asset_names=["plug", "socket"]):
    """
    Adds required rigid object poses to both 'states' and 'initial_state'
    groups to ensure Isaac Lab can replay and reset the environment.
    """
    with h5py.File(target_path, 'a') as f:
        if 'data' not in f:
            print("Error: 'data' group not found in HDF5.")
            return
            
        data_group = f['data']
        
        # Define the dummy poses based on your specific task coordinates
        # Plug pose
        plug_pose = np.array([[0.617, 0.15, 0.13125, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        # Socket pose
        socket_pose = np.array([[0.6, 0.15, 0.01, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        root_velocity = np.zeros((1, 6), dtype=np.float32)  # [vx, vy, vz, wx, wy, wz]
        
        poses = [plug_pose, socket_pose]

        for demo_name in tqdm(data_group.keys(), desc="Updating object states"):
            demo = data_group[demo_name]
            
            # 1. Ensure high-level groups exist
            for group in ["states/rigid_object", "initial_state/rigid_object"]:
                if group not in demo:
                    demo.create_group(group)
            
            # 2. Add the specific assets to both locations
            for asset_name, pose_data in zip(asset_names, poses):
                # Path 1: states (Temporal/Trajectory data)
                state_path = f"states/rigid_object/{asset_name}/root_pose"
                if state_path not in demo:
                    demo.create_dataset(state_path, data=pose_data)
                velocity_path = f"states/rigid_object/{asset_name}/root_velocity"
                if velocity_path not in demo:
                    demo.create_dataset(velocity_path, data=root_velocity)

                # Path 2: initial_state (The reset snapshot)
                init_path = f"initial_state/rigid_object/{asset_name}/root_pose"
                if init_path not in demo:
                    demo.create_dataset(init_path, data=pose_data)
                init_vel_path = f"initial_state/rigid_object/{asset_name}/root_velocity"
                if init_vel_path not in demo:
                    demo.create_dataset(init_vel_path, data=root_velocity)

        print(f"\nSuccess! Added {asset_names} to both 'states' and 'initial_state' groups.")

def fix_metadata_and_nest_in_data(source_path, target_path, task_name="Isaac-Unplug-Franka-IK-Abs-Mimic-RGB-v0"):
    """
    Creates a NEW HDF5 file where all demos are nested inside a 'data' group,
    and adds the required IsaacLab metadata to that 'data' group.
    """
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
        print(f"Creating nested dataset at: {target_path}")
        
        # 1. Create the mandatory 'data' group
        dest_data_group = t.create_group('data')
        
        # 2. Identify the source of the demos
        # If source already has 'data', use its children. Otherwise use root.
        source_root = s['data'] if 'data' in s else s
        demo_keys = [k for k in source_root.keys() if k.startswith('demo_')]
        
        # 3. Copy demos INTO the new 'data' group
        for key in tqdm(demo_keys, desc="Moving demos into 'data'"):
            source_root.copy(key, dest_data_group)
            
        # 4. Attach metadata to the 'data' group (NOT the file root)
        # IsaacLab's handler accesses self._hdf5_data_group.attrs
        dest_data_group.attrs["env_args"] = json.dumps(env_args)
        dest_data_group.attrs["env_name"] = task_name
        dest_data_group.attrs["num_demos"] = len(demo_keys)
        dest_data_group.attrs["teleop_device"] = "keyboard"
        
        # Also copy global attributes to the root just in case
        for attr_name, attr_val in s.attrs.items():
            t.attrs[attr_name] = attr_val

    print(f"Success! Dataset structured for IsaacLab at: {target_path}")
    add_isaaclab_scene_state(target_path, asset_names=["plug", "socket"])

def cleanup_dataset_for_8dim_and_camera(source_path, target_path):
    """
    1. Reduces joint_position from 9 to 8 (7 arm + 1 gripper).
    2. Reduces joint_velocity from 9 to 7 (7 arm only).
    3. Renames obs/image to obs/camera.
    """
    if not os.path.exists(source_path):
        print(f"Error: Source file {source_path} not found.")
        return

    with h5py.File(source_path, 'r') as s, h5py.File(target_path, 'w') as t:
        source_root = s['data'] if 'data' in s else s
        dest_data_group = t.create_group('data')
        
        # Copy global metadata
        for k, v in s.attrs.items(): t.attrs[k] = v
        if 'data' in s:
            for k, v in s['data'].attrs.items(): dest_data_group.attrs[k] = v

        demo_keys = [k for k in source_root.keys() if k.startswith('demo_')]

        for demo_key in tqdm(demo_keys, desc="Cleaning (Pos:8, Vel:7, Cam Rename)"):
            source_demo = source_root[demo_key]
            dest_demo = dest_data_group.create_group(demo_key)

            def process_and_copy(name, obj):
                if isinstance(obj, h5py.Group):
                    # Rename 'image' group to 'camera'
                    new_group_name = name.replace('obs/image', 'obs/camera')
                    if new_group_name not in dest_demo:
                        dest_demo.create_group(new_group_name)
                    return

                # --- Handle Datasets ---
                data = obj[()]
                new_name = name
                
                # 1. Rename obs/image dataset to obs/camera
                if 'obs/image' in name:
                    new_name = name.replace('obs/image', 'obs/camera')

                # 2. Specific Slicing for Position vs Velocity
                if name.endswith('joint_position'):
                    if data.shape[-1] == 9:
                        data = data[:, :8]  # Keep 7 arm + 1 gripper
                        print(f"  {demo_key}: Sliced position to {data.shape}")
                
                elif name.endswith('joint_velocity'):
                    if data.shape[-1] == 9:
                        data = data[:, :7]  # Keep 7 arm only
                        print(f"  {demo_key}: Sliced velocity to {data.shape}")
                
                # Create the dataset in target
                dset = dest_demo.create_dataset(new_name, data=data, compression=obj.compression)
                
                # Copy attributes
                for attr_name, attr_val in obj.attrs.items():
                    dset.attrs[attr_name] = attr_val

            source_demo.visititems(process_and_copy)

    print(f"\nSuccess! Cleaned dataset (8 pos, 7 vel) saved to: {target_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HDF5 dataset utilities")
    parser.add_argument('--no_shift', action='store_true',
                        help='Skip the one-step gripper shift in add_gripper mode')
    args = parser.parse_args()
    
    dataset_frequency = 20  # Hz
    time_to_cut = 2  # sec
    skip_steps = int(time_to_cut * dataset_frequency)
    
    # --- Configuration ---
    MODE = "add_gripper_smooth"  # Choose: "cut", "augment", "merge", "delete_demo", or "add_gripper", "add_gripper_smooth", "prepare_metadata"
    pos = True  # If True, map gripper commands (1/-1) to joint positions (0.04/0.0)
    
    SOURCE = "act_copy/data/dataset.hdf5"  # Specify your source dataset here
    TASK = "Isaac-Unplug-Franka-IK-Abs-Mimic-RGB-v0"

    if MODE == "cut":
        TARGET = "shortened_data.hdf5"
        create_shortened_dataset_copy(SOURCE, TARGET, skip_steps=skip_steps)
    
    elif MODE == "augment":
        TARGET = "augmented_data.hdf5"
        create_augmented_dataset_copy(
            SOURCE, 
            TARGET, 
            noise_steps=skip_steps,
            high_noise_std=0.2,
            low_noise_std=0.05
        )
    
    elif MODE == "merge":
        SOURCE2 = "datasets/generated_dataset_small_50demos_cleaned.hdf5"  # Specify your second dataset here
        TARGET = "act_copy/data/merged_data.hdf5"
        merge_datasets(SOURCE, SOURCE2, TARGET)
    
    elif MODE == "delete_demo":
        DEMO_INDICES = [1, 5, 8, 9, 14, 15, 17, 18]  # List of demo indices to delete (e.g., [1, 3, 5] or single int 2)
        delete_demo_and_reindex(SOURCE, DEMO_INDICES)
    
    elif MODE == "add_gripper":
        if pos: 
            if args.no_shift:
                TARGET = "act_copy/data/dataset_real_dynamics_with_gripper_pos_no_shift.hdf5"
            else:
                TARGET = "act_copy/data/dataset_real_dynamics_with_gripper_pos.hdf5"

        else:
            if args.no_shift:
                TARGET = "act_copy/data/dataset_real_dynamics_with_gripper_no_shift.hdf5"
            else:
                TARGET = "act_copy/data/dataset_real_dynamics_with_gripper.hdf5"
        add_gripper_to_qpos(SOURCE, TARGET, pos=pos, no_shift=args.no_shift)

    elif MODE == "add_gripper_smooth":
        TARGET = "act_copy/data/dataset_real_dynamics_with_2_gripper_smooth.hdf5"
        add_gripper_smooth(SOURCE, TARGET, closing_duration=13)
    
    elif MODE == "prepare_metadata":
        TARGET = "act_copy/data/dataset_real_dynamics_with_gripper_smooth_metadata.hdf5"
        fix_metadata_and_nest_in_data(SOURCE, TARGET, task_name=TASK)
    
    elif MODE == "clean_for_sim":
        SOURCE = "datasets/generated_dataset_100demos_005std_005range.hdf5"
        TARGET = "datasets/generated_dataset_100demos_005std_005range_cleaned.hdf5"
        cleanup_dataset_for_8dim_and_camera(SOURCE, TARGET)

    else:
        print(f"Error: Unknown mode '{MODE}'. Choose 'cut', 'augment', 'merge', 'delete_demo', 'add_gripper', or 'prepare_metadata'.")