"""
Visualize what the ACT policy's ResNet18 backbone is attending to.

Three visualization modes:
  1. gradcam   – Grad-CAM heatmaps showing which image regions drive actions
  2. features  – Raw feature map activations from ResNet18 layer4
  3. ablation  – Compares policy output with real vs. zeroed images

Usage:
    python act_copy/visualize_features.py \
        --checkpoint act_copy/ckpt/policy_best.ckpt \
        --dataset act_copy/data/dataset_real_dynamics_with_gripper_pos.hdf5 \
        --stats act_copy/ckpt/dataset_act_stats.pkl \
        --mode gradcam \
        --output_dir act_copy/vis
"""

import argparse
import os
import pickle
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib.colors import Normalize
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from act_copy.detr.models.detr_vae import build
from act_copy.policy_runner import ACTPolicy


def load_policy(ckpt_path, device="cuda"):
    """Load the trained ACT policy."""

    policy_config = {
        "ckpt_dir": ckpt_path,
        "num_queries": 32,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 6,
        "nheads": 8,
        "hidden_dim": 256,
        "dim_feedforward": 2048,
        "camera_names": ["image"],
        "velocity_control": False,
        "context_length": 4,
    }
    policy = ACTPolicy(policy_config)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.to(device)
    policy.eval()
    return policy


def load_sample(dataset_path, stats_path, demo_idx=0, step_idx=50, device="cuda", context_length=4):
    """Load a context window of observations."""
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    qpos_mean = torch.from_numpy(stats["qpos_mean"]).float().to(device)
    qpos_std = torch.from_numpy(stats["qpos_std"]).float().to(device)

    with h5py.File(dataset_path, "r") as f:
        demo = f[f"data/demo_{demo_idx}"]
        
        # 1. Handle qpos context
        # Extract indices for the context window (clipping at 0)
        start_idx = max(0, step_idx - context_length + 1)
        indices = range(start_idx, step_idx + 1)
        
        qpos_list = []
        img_list = []
        
        for idx in indices:
            # Joint positions
            q_np = demo["states/articulation/robot/joint_position"][idx]
            qpos_list.append(torch.from_numpy(q_np).float().to(device))
            
            # Images
            i_np = demo["obs/camera"][idx] # [H, W, 3]
            i_tensor = torch.from_numpy(i_np).permute(2, 0, 1).float() / 255.0 # [3, H, W]
            img_list.append(i_tensor.to(device))

        # Pad if we are at the beginning of the demo
        while len(qpos_list) < context_length:
            qpos_list.insert(0, qpos_list[0])
            img_list.insert(0, img_list[0])

        # Stack into [context_length, dim]
        qpos_norm = (torch.stack(qpos_list) - qpos_mean) / qpos_std
        qpos_norm = qpos_norm.unsqueeze(0) # [1, context, 8]
        
        # Stack into [1, num_cams, context, 3, H, W]
        img_stack = torch.stack(img_list, dim=0) # [context, 3, H, W]
        img_batch = img_stack.unsqueeze(0).unsqueeze(0) # [1, 1, context, 3, H, W]
        
        # Keep the current image for display
        img_np_current = demo["obs/camera"][step_idx]

    return qpos_norm, img_batch, img_np_current


# ──────────────────────────────────────────────────────────────────────
# 1. Grad-CAM
# ──────────────────────────────────────────────────────────────────────
def gradcam(policy, qpos, image, device="cuda"):
    """
    Compute Grad-CAM for the ResNet18 backbone inside the ACT policy.
    Returns a heatmap [H, W] in [0, 1].
    """
    # Access the backbone (first camera)
    model = policy.model  # DETRVAE
    backbone = model.backbones[0][0]  # Joiner → Backbone (BackboneBase)

    # Hook into the last conv layer (layer4)
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["layer4"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["layer4"] = grad_out[0]

    # The body is IntermediateLayerGetter wrapping the ResNet
    target_layer = backbone.body.layer4
    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    # Enable gradients for this pass
    policy.zero_grad()
    for p in policy.parameters():
        p.requires_grad_(True)

    # Normalize image (same as policy.__call__)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    current_img = image[:, 0, -1]
    img_norm = normalize(current_img.squeeze(0))  # [3, H, W]
    img_norm = img_norm.unsqueeze(0)  # [1, 3, H, W]

    # Forward through backbone only
    features_dict = backbone.body(img_norm)
    feature_map = features_dict["0"]  # layer4 output: [1, 512, h, w]

    # We need a scalar to backprop from; use the mean activation
    # (Alternatively, we could run the full policy and backprop from a specific action dim)
    scalar = feature_map.mean()
    scalar.backward()

    # Grad-CAM: weight each channel by its gradient, then ReLU
    grads = gradients["layer4"]  # [1, 512, h, w]
    weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, 512, 1, 1]
    cam = (weights * activations["layer4"]).sum(dim=1, keepdim=True)  # [1, 1, h, w]
    cam = F.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()

    # Normalize to [0, 1]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    h_fwd.remove()
    h_bwd.remove()

    # Restore eval mode
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    return cam


def gradcam_action_driven(policy, qpos, image, action_idx=2, device="cuda"):
    """
    Grad-CAM driven by a specific action dimension (e.g., action_idx=2 for Z).
    Shows which image regions most influence that particular action output.
    """
    model = policy.model
    backbone = model.backbones[0][0]

    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["layer4"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["layer4"] = grad_out[0]

    target_layer = backbone.body.layer4
    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    # Enable gradients
    for p in policy.parameters():
        p.requires_grad_(True)
    policy.zero_grad()

    # Normalize image
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_for_policy = normalize(image.squeeze(0).squeeze(0).squeeze(0)).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # Full forward pass through policy
    actions = policy(qpos, img_for_policy)  # [1, 64, 8]

    # Backprop from a specific action dimension (first timestep)
    target = actions[0, 0, action_idx]
    target.backward(retain_graph=True)

    grads = gradients["layer4"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations["layer4"]).sum(dim=1, keepdim=True)
    cam = F.relu(cam).squeeze().detach().cpu().numpy()

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    h_fwd.remove()
    h_bwd.remove()
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    return cam


# ──────────────────────────────────────────────────────────────────────
# 2. Feature map visualization
# ──────────────────────────────────────────────────────────────────────
def visualize_feature_maps(policy, image, device="cuda", top_k=16):
    """Extract and visualize the top-K most activated feature channels."""
    model = policy.model
    backbone = model.backbones[0][0]

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_norm = normalize(image.squeeze(0).squeeze(0).squeeze(0)).unsqueeze(0)

    with torch.no_grad():
        features_dict = backbone.body(img_norm)
        feature_map = features_dict["0"]  # [1, 512, h, w]

    fm = feature_map.squeeze(0).cpu().numpy()  # [512, h, w]

    # Rank channels by mean activation
    channel_means = fm.mean(axis=(1, 2))
    top_indices = np.argsort(channel_means)[-top_k:][::-1]

    return fm, top_indices


# ──────────────────────────────────────────────────────────────────────
# 3. Image ablation
# ──────────────────────────────────────────────────────────────────────
def ablation_test(policy, qpos, image, device="cuda"):
    """Compare policy output with real image vs. zeroed/random image."""
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        # Real image
        actions_real = policy(qpos, image).cpu().numpy()[0, 0]

        # Zero image (ImageNet-mean gray after normalization)
        zero_img = torch.zeros_like(image)
        actions_zero = policy(qpos, zero_img).cpu().numpy()[0, 0]

        # Random image
        rand_img = torch.rand_like(image)
        actions_rand = policy(qpos, rand_img).cpu().numpy()[0, 0]

    return actions_real, actions_zero, actions_rand


# ──────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────
def overlay_heatmap(img_np, heatmap, alpha=0.5):
    """Overlay a heatmap on an RGB image."""
    import cv2
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap_color + (1 - alpha) * img_np).astype(np.uint8)
    return blended


def main():
    parser = argparse.ArgumentParser(description="Visualize ACT policy image features")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--stats", type=str, required=True)
    parser.add_argument("--mode", type=str, default="gradcam", choices=["gradcam", "features", "ablation", "all"])
    parser.add_argument("--demo_idx", type=int, default=0)
    parser.add_argument("--step_idx", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="act_copy/vis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading policy...")
    policy = load_policy(args.checkpoint, device)

    print(f"Loading sample from demo_{args.demo_idx}, step {args.step_idx}...")
    qpos, image, img_np = load_sample(args.dataset, args.stats, args.demo_idx, args.step_idx, device)

    run_all = args.mode == "all"

    # ── Grad-CAM ──
    if args.mode == "gradcam" or run_all:
        print("Computing Grad-CAM (mean activation)...")
        cam = gradcam(policy, qpos, image, device)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        blended = overlay_heatmap(img_np, cam)
        axes[1].imshow(blended)
        axes[1].set_title("Grad-CAM (most important regions)")
        axes[1].axis("off")

        plt.tight_layout()
        path = os.path.join(args.output_dir, f"gradcam_demo{args.demo_idx}_step{args.step_idx}.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
        plt.close()

    # ── Feature maps ──
    if args.mode == "features" or run_all:
        print("Extracting feature maps...")
        fm, top_indices = visualize_feature_maps(policy, image, device, top_k=16)

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        for i, idx in enumerate(top_indices):
            ax = axes[i // 4, i % 4]
            ax.imshow(fm[idx], cmap="viridis")
            ax.set_title(f"Ch {idx} (mean={fm[idx].mean():.2f})")
            ax.axis("off")
        plt.suptitle("Top-16 ResNet18 Layer4 Feature Channels", fontsize=14)
        plt.tight_layout()
        path = os.path.join(args.output_dir, f"features_demo{args.demo_idx}_step{args.step_idx}.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
        plt.close()

    # ── Ablation ──
    if args.mode == "ablation" or run_all:
        print("Running ablation test...")
        a_real, a_zero, a_rand = ablation_test(policy, qpos, image, device)

        labels = ["x", "y", "z", "qw", "qx", "qy", "qz", "grip"]
        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - width, a_real, width, label="Real image", color="steelblue")
        ax.bar(x, a_zero, width, label="Zero image", color="salmon")
        ax.bar(x + width, a_rand, width, label="Random image", color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Action value (normalized)")
        ax.set_title("Ablation: Real vs Zero vs Random Image")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(args.output_dir, f"ablation_demo{args.demo_idx}_step{args.step_idx}.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
        plt.close()

        # Print numeric comparison
        diff_zero = np.abs(a_real - a_zero)
        diff_rand = np.abs(a_real - a_rand)
        print(f"\n  Action difference (real vs zero image):")
        for i, l in enumerate(labels):
            print(f"    {l}: {diff_zero[i]:.4f}")
        print(f"  Mean difference: {diff_zero.mean():.4f}")
        print(f"\n  Action difference (real vs random image):")
        for i, l in enumerate(labels):
            print(f"    {l}: {diff_rand[i]:.4f}")
        print(f"  Mean difference: {diff_rand.mean():.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
