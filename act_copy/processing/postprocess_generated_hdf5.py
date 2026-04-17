import h5py
import os
import numpy as np
from tqdm import tqdm
import argparse

def process_and_merge(source_generated, source_rec, target_path):
    """
    1. Cleans the generated dataset (Pos:8, Vel:7, Rename: Camera).
    2. Merges it with the main dataset into a new final file.
    """
    if not os.path.exists(source_generated):
        print(f"Error: Generated file {source_generated} not found.")
        return
    
    # We open the main dataset if it exists to get the starting demo index
    existing_demos = 0
    if os.path.exists(source_rec):
        with h5py.File(source_rec, 'r') as m:
            existing_demos = len(m['data'].keys())
            print(f"Found {existing_demos} existing demos in main dataset.")

    with h5py.File(source_generated, 'r') as s_gen, \
         h5py.File(source_rec, 'r' if os.path.exists(source_rec) else 'w') as s_main, \
         h5py.File(target_path, 'w') as t:

        dest_data_group = t.create_group('data')
        
        # --- Step 1: Copy Main Dataset Demos As-Is ---
        if os.path.exists(source_rec):
            print("Copying main dataset to target...")
            for key in tqdm(s_main['data'].keys(), desc="Main Dataset"):
                s_main.copy(f"data/{key}", dest_data_group)
        
        # --- Step 2: Clean and Copy Generated Demos ---
        print(f"\nCleaning and merging {len(s_gen['data'].keys())} generated demos...")
        
        # We start naming from demo_N where N is the current count
        gen_keys = sorted(list(s_gen['data'].keys()))
        
        for i, gen_key in enumerate(tqdm(gen_keys, desc="Generated Dataset")):
            new_demo_name = f"demo_{existing_demos + i}"
            source_demo = s_gen['data'][gen_key]
            dest_demo = dest_data_group.create_group(new_demo_name)

            def clean_and_copy(name, obj):
                if isinstance(obj, h5py.Group):
                    # Rename 'image' to 'camera'
                    new_group_name = name.replace('obs/image', 'obs/camera')
                    if new_group_name not in dest_demo:
                        dest_demo.create_group(new_group_name)
                    return

                # --- Handle Datasets ---
                data = obj[()]
                new_name = name
                
                # Rename 'image' dataset to 'camera'
                if 'obs/image' in name:
                    new_name = name.replace('obs/image', 'obs/camera')

                # Slice Position to 8 (Arm + Gripper)
                if name.endswith('joint_position') and data.shape[-1] == 9:
                    data = data[:, :8]
                
                # Slice Velocity to 7 (Arm only)
                elif name.endswith('joint_velocity') and data.shape[-1] == 9:
                    data = data[:, :7]
                
                # Create dataset in the merged file
                dset = dest_demo.create_dataset(new_name, data=data, compression=obj.compression)
                for attr_name, attr_val in obj.attrs.items():
                    dset.attrs[attr_name] = attr_val

            source_demo.visititems(clean_and_copy)

        # Copy top-level attributes from the main dataset (metadata)
        if os.path.exists(source_rec):
            for k, v in s_main['data'].attrs.items(): dest_data_group.attrs[k] = v
            dest_data_group.attrs["num_demos"] = existing_demos + len(gen_keys)

    print(f"\nSuccess! Final merged and cleaned dataset saved to: {target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and Merge Generated HDF5")
    parser.add_argument("--gen", type=str, required=True, help="Path to newly generated HDF5")
    parser.add_argument("--rec", type=str, required=True, help="Path to your existing recorded dataset")
    parser.add_argument("--output", type=str, required=True, help="Path for the final merged result")
    
    args = parser.parse_args()
    process_and_merge(args.gen, args.rec, args.output)