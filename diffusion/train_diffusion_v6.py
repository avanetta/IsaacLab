"""
Training script for Diffusion Policy v6.

8D action space: [x, y, z, qx, qy, qz, qw, gripper]
Observations: RGB image stack (9ch) + joint position history (5x7)
Gripper is part of the diffusion action space (no separate classifier head).

Usage:
    python scripts/training/train_diffusion_v6.py \
        --dataset_dir data_v6 \
        --save_dir checkpoints/diffusion_v6 \
        --pretrained --epochs 2000
"""

import sys
import gc
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime
import csv

sys.path.append(str(Path(__file__).parent.parent.parent))
from diffusion_policy_v6 import DiffusionPolicyV6


# EMA

class EMAModel:
    """Exponential Moving Average of model weights for stable inference."""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        del self.backup


# Image Augmentation

IMAGE_AUG_CONFIG = {
    'brightness': 0.5, 'contrast': 0.4, 'saturation': 0.4, 'hue': 0.2,
    'gaussian_noise_std': 0.03,
    'erasing_p': 0.4, 'erasing_scale': (0.02, 0.2),
    'blur_kernel': 7, 'blur_sigma': (0.1, 3.0), 'blur_p': 0.5,
}


def augment_image_stack(img_stack, config=IMAGE_AUG_CONFIG):
    """Apply augmentation to a 9-channel image stack [current, previous, diff]."""
    img_curr = img_stack[:3]
    img_prev = img_stack[3:6]

    # Color jitter (same transform for both frames)
    if np.random.rand() < 0.5:
        jitter = T.ColorJitter(
            brightness=config['brightness'], contrast=config['contrast'],
            saturation=config['saturation'], hue=config['hue'])
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        img_curr = jitter(img_curr)
        torch.manual_seed(seed)
        img_prev = jitter(img_prev)

    # Gaussian noise
    if np.random.rand() < 0.3:
        noise = torch.randn_like(img_curr) * config['gaussian_noise_std']
        img_curr = img_curr + noise
        img_prev = img_prev + noise

    # Gaussian blur
    if np.random.rand() < config['blur_p']:
        blur = T.GaussianBlur(kernel_size=config['blur_kernel'], sigma=config['blur_sigma'])
        img_curr = blur(img_curr)
        img_prev = blur(img_prev)

    # Random erasing (current frame only)
    if np.random.rand() < config['erasing_p']:
        img_curr = T.RandomErasing(p=1.0, scale=config['erasing_scale'])(img_curr)

    # Recompute diff after augmentation
    img_diff = img_curr - img_prev
    return torch.cat([img_curr, img_prev, img_diff], dim=0)


# Dataset

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.from_numpy(x).float()


class V6Dataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = to_tensor(s['image_stack'])
        if self.augment:
            img = augment_image_stack(img)
        return {
            'img_stack':         img,
            'joint_pos_history': to_tensor(s['joint_pos_history']),
            'action_gt':         to_tensor(s['action']),
        }


def load_chunks(chunks_dir):
    samples = []
    for cf in sorted(Path(chunks_dir).glob("chunk_*.pt")):
        samples.extend(torch.load(cf, weights_only=False))
    return samples


# Training Loop

def train_epoch(model, loader, optimizer, ema, grad_clip, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in tqdm(loader, desc="Train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _ = model.forward_train(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        ema.update()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in tqdm(loader, desc="Val", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _ = model.forward_train(batch)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


# Main

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load data
    dataset_dir = Path(args.dataset_dir)
    train_samples = load_chunks(dataset_dir / "train_chunks")
    val_samples = load_chunks(dataset_dir / "val_chunks")
    print(f"Train: {len(train_samples)} samples | Val: {len(val_samples)} samples")

    train_dataset = V6Dataset(train_samples, augment=(args.augmentation != 'none'))
    val_dataset = V6Dataset(val_samples, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = DiffusionPolicyV6(
        action_horizon=args.action_horizon, action_dim=8,
        diffusion_steps=args.diffusion_steps, ddim_steps=args.ddim_steps,
        pretrained=args.pretrained, freeze_vision_backbone=args.freeze_vision_backbone,
        joint_pos_dim=args.joint_pos_dim,  # MODIFIED: Pass joint_pos_dim to model
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Normalization stats from training data
    all_jp = torch.stack([to_tensor(s['joint_pos_history']) for s in train_samples])
    all_act = torch.stack([to_tensor(s['action']) for s in train_samples])
    # MODIFIED: joint_pos_dim changed from hardcoded 7 to args.joint_pos_dim
    model.normalizer.joint_pos_mean = all_jp.reshape(-1, args.joint_pos_dim).mean(0)
    model.normalizer.joint_pos_std = all_jp.reshape(-1, args.joint_pos_dim).std(0).clamp(min=1e-6)
    model.normalizer.action_mean = all_act.reshape(-1, 8).mean(0)
    model.normalizer.action_std = all_act.reshape(-1, 8).std(0).clamp(min=1e-6)
    print(f"Action std: {model.normalizer.action_std.numpy()}")
    del all_jp, all_act; gc.collect()

    model = model.to(device)
    ema = EMAModel(model, decay=args.ema_decay)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader),
                                  eta_min=args.lr * 0.01)

    # Logging
    log_path = save_dir / f"training_log.csv"
    log_csv = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_csv)
    log_writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, ema, args.grad_clip, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        log_writer.writerow([epoch, train_loss, val_loss, lr])
        log_csv.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch, 'val_loss': val_loss,
                'model_state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_dir / 'best_model.pt')
            print(f"  -> Saved best model (val_loss: {val_loss:.4f})")

        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch, 'val_loss': val_loss,
                'model_state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
            }, save_dir / f'checkpoint_epoch_{epoch}.pt')

    log_csv.close()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion Policy v6")
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--action_horizon', type=int, default=16)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--ddim_steps', type=int, default=10)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--freeze_vision_backbone', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--augmentation', type=str, default='moderate', choices=['none', 'moderate'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_every', type=int, default=50)
    # MODIFIED: Added joint_pos_dim for compatibility with variable joint position dimensions
    parser.add_argument('--joint_pos_dim', type=int, default=7)
    args = parser.parse_args()
    main(args)
