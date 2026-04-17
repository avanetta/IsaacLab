"""
Preprocess HDF5 demonstration dataset into training chunks for Diffusion Policy v6.

Input: HDF5 file with structure:
  data/demo_N/obs/camera     [T, H, W, 3] uint8
  data/demo_N/obs/eef_pos    [T, 3]
  data/demo_N/obs/eef_quat   [T, 4]       xyzw convention
  data/demo_N/obs/joint_pos  [T, 7]
  data/demo_N/actions        [T, 8]       [pos(3), quat_wxyz(4), gripper{-1,+1}]

Output chunks with sample keys:
  image_stack:       [9, 224, 224]   (current + previous + diff, ImageNet normalized)
  joint_pos_history: [5, 7]
  action:            [H, 8]          [pos(3), quat_xyzw(4), gripper{0,1}]

Usage:
    python scripts/data_processing/preprocess_hdf5_v6.py \
        --input_file dataset.hdf5 \
        --output_dir data_v6
"""

import argparse
import h5py
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

HISTORY_LEN = 5
ACTION_HORIZON = 32
TARGET_IMG_SIZE = (224, 224)
CHUNK_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def preprocess_image(img):
    img = Image.fromarray(img).resize((TARGET_IMG_SIZE[1], TARGET_IMG_SIZE[0]))
    img = np.array(img, dtype=np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)


def create_image_stack(frames, idx):
    img_curr = preprocess_image(frames[idx])
    img_prev = preprocess_image(frames[idx - 1]) if idx > 0 else img_curr.copy()
    return np.concatenate([img_curr, img_prev, img_curr - img_prev], axis=0)


def enforce_quat_continuity(quats):
    for t in range(1, len(quats)):
        if np.dot(quats[t], quats[t - 1]) < 0:
            quats[t] = -quats[t]
    return quats


def enforce_quat_continuity_7d(poses):
    for t in range(1, len(poses)):
        if np.dot(poses[t, 3:], poses[t - 1, 3:]) < 0:
            poses[t, 3:] = -poses[t, 3:]
    return poses


def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def process_demo(demo):
    camera = demo['obs']['camera'][()]
    joint_pos = demo['obs']['joint_pos'][()]
    actions_raw = demo['actions'][()]
    T = len(camera)

    # Convert action quaternions wxyz -> xyzw
    action_poses = np.zeros((T, 7), dtype=np.float32)
    action_poses[:, :3] = actions_raw[:, :3]
    for t in range(T):
        action_poses[t, 3:7] = wxyz_to_xyzw(actions_raw[t, 3:7])
    action_poses = enforce_quat_continuity_7d(action_poses)

    # Gripper: {-1, +1} -> {0, 1}
    gripper = ((actions_raw[:, 7] + 1.0) / 2.0).astype(np.float32)

    # Build 8D action targets: [pose(7), gripper(1)] for each horizon step
    action_targets = np.zeros((T, ACTION_HORIZON, 8), dtype=np.float32)
    for t in range(T):
        for h in range(ACTION_HORIZON):
            idx = min(t + h + 1, T - 1)
            action_targets[t, h, :7] = action_poses[idx]
            action_targets[t, h, 7] = gripper[idx]

    # Create samples
    samples = []
    for t in range(HISTORY_LEN - 1, T - ACTION_HORIZON - 1):
        samples.append({
            'image_stack':       create_image_stack(camera, t).astype(np.float32),
            'joint_pos_history': joint_pos[t - HISTORY_LEN + 1:t + 1].astype(np.float32),
            'action':            action_targets[t].astype(np.float32),
        })
    return samples


def save_chunks(samples, output_dir, name):
    split_dir = output_dir / name
    split_dir.mkdir(exist_ok=True)
    chunk, chunk_idx = [], 0
    for s in samples:
        chunk.append(s)
        if len(chunk) >= CHUNK_SIZE:
            torch.save(chunk, split_dir / f"chunk_{chunk_idx:04d}.pt")
            chunk, chunk_idx = [], chunk_idx + 1
    if chunk:
        torch.save(chunk, split_dir / f"chunk_{chunk_idx:04d}.pt")
        chunk_idx += 1
    return chunk_idx


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {args.input_file}")
    print(f"Output: {output_dir}")
    print(f"History: {HISTORY_LEN} | Horizon: {ACTION_HORIZON}")

    f = h5py.File(args.input_file, 'r')
    demo_names = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))
    print(f"Demos: {len(demo_names)}")

    all_samples = []
    for name in tqdm(demo_names, desc="Processing"):
        samples = process_demo(f['data'][name])
        all_samples.extend(samples)
    f.close()

    print(f"Total samples: {len(all_samples)}")

    # Print sample shapes
    for k, v in all_samples[0].items():
        shape = v.shape if isinstance(v, np.ndarray) else type(v).__name__
        print(f"  {k}: {shape}")

    # Gripper distribution
    g = np.array([s['action'][0, 7] for s in all_samples])
    print(f"Gripper: {(g > 0.5).sum()} closed, {(g < 0.5).sum()} open")

    # Train/val split
    np.random.seed(args.seed)
    idx = np.random.permutation(len(all_samples))
    split = int(len(all_samples) * args.train_split)
    train_samples = [all_samples[i] for i in idx[:split]]
    val_samples = [all_samples[i] for i in idx[split:]]

    n_train = save_chunks(train_samples, output_dir, 'train_chunks')
    n_val = save_chunks(val_samples, output_dir, 'val_chunks')
    print(f"Train: {len(train_samples)} samples ({n_train} chunks)")
    print(f"Val:   {len(val_samples)} samples ({n_val} chunks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess HDF5 dataset for v6")
    parser.add_argument('--input_file', type=str, default='dataset_real_dynamics_with_gripper_smooth.hdf5')
    parser.add_argument('--output_dir', type=str, default='data_v6')
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
