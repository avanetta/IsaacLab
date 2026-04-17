"""
Diffusion Policy v6 — Joint-space observations, Cartesian actions.

Self-contained model file. No external dependencies beyond PyTorch/torchvision.

Observations:
  img_stack:         [B, 9, 224, 224]  RGB (current + previous + diff)
  joint_pos_history: [B, T, 7]         joint angles over T history steps

Actions:
  8D: [x, y, z, qx, qy, qz, qw, gripper]  (xyzw quaternion convention)
  Gripper predicted by diffusion as a continuous value, thresholded at 0.5.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# Vision Encoder

class VisionEncoder(nn.Module):
    """ResNet-18 backbone for 9-channel image input -> [B, 128]."""

    def __init__(self, output_dim=128, pretrained=False):
        super().__init__()
        resnet = resnet18(pretrained=pretrained)
        self.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                w = resnet.conv1.weight
                self.conv1.weight[:, 0:3] = w
                self.conv1.weight[:, 3:6] = w
                self.conv1.weight[:, 6:9] = w
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


# Time Embedding

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        return self.mlp(torch.cat([emb.sin(), emb.cos()], dim=-1))


# 1D U-Net with FiLM conditioning

class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.film = nn.Linear(cond_dim, out_ch * 2)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.Mish()

    def forward(self, x, cond):
        h = self.act(self.norm1(self.conv1(x)))
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None]) + shift[:, :, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class ConditionalUNet1D(nn.Module):
    def __init__(self, action_dim=8, cond_dim=256, hidden_dims=(128, 256, 512)):
        super().__init__()
        self.input_proj = nn.Conv1d(action_dim, hidden_dims[0], 1)
        self.down_blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dims[i], hidden_dims[i + 1], cond_dim)
            for i in range(len(hidden_dims) - 1)])
        self.bottleneck = ResidualBlock1D(hidden_dims[-1], hidden_dims[-1], cond_dim)
        self.up_blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dims[i] * 2, hidden_dims[i - 1], cond_dim)
            for i in range(len(hidden_dims) - 1, 0, -1)])
        self.output_proj = nn.Conv1d(hidden_dims[0], action_dim, 1)

    def forward(self, x_t, cond):
        x = self.input_proj(x_t.transpose(1, 2))
        skips = []
        for down in self.down_blocks:
            x = down(x, cond)
            skips.append(x)
            x = F.avg_pool1d(x, 2)
        x = self.bottleneck(x, cond)
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = torch.cat([F.interpolate(x, scale_factor=2, mode='nearest'), skip], dim=1)
            x = up(x, cond)
        return self.output_proj(x).transpose(1, 2)


# Normalizer

class Normalizer(nn.Module):
    def __init__(self, joint_pos_dim=7, action_dim=8):
        super().__init__()
        for name, dim in [('joint_pos', joint_pos_dim), ('action', action_dim)]:
            self.register_buffer(f'{name}_mean', torch.zeros(dim))
            self.register_buffer(f'{name}_std', torch.ones(dim))

    def normalize_joint_pos(self, x):
        return (x - self.joint_pos_mean) / self.joint_pos_std

    def normalize_action(self, x):
        return (x - self.action_mean) / self.action_std

    def denormalize_action(self, x):
        return x * self.action_std + self.action_mean


# Proprioceptive Encoder

class ProprioEncoder(nn.Module):
    """Flattened joint_pos history [B, T, 7] -> [B, 64]."""

    def __init__(self, joint_pos_dim=7, history_len=5, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_pos_dim * history_len, 128), nn.ReLU(),
            nn.Linear(128, output_dim), nn.ReLU())

    def forward(self, joint_pos):
        return self.net(joint_pos.reshape(joint_pos.shape[0], -1))


# Main Model

class DiffusionPolicyV6(nn.Module):
    def __init__(self, action_horizon=16, action_dim=8, diffusion_steps=100,
                 ddim_steps=10, history_len=5, pretrained=False,
                 freeze_vision_backbone=False, joint_pos_dim=7): 
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.diffusion_steps = diffusion_steps

        self.vision_encoder = VisionEncoder(output_dim=128, pretrained=pretrained)
        if freeze_vision_backbone:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        self.proprio_encoder = ProprioEncoder(joint_pos_dim=joint_pos_dim, history_len=history_len, output_dim=64)  # MODIFIED: Pass joint_pos_dim
        self.obs_dim = 192  # 128 vision + 64 proprio

        self.time_embed = SinusoidalTimeEmbedding(dim=64)
        self.unet = ConditionalUNet1D(action_dim=action_dim,
                                      cond_dim=self.obs_dim + 64, hidden_dims=[128, 256, 512])
        self.normalizer = Normalizer(joint_pos_dim=joint_pos_dim, action_dim=action_dim)  # MODIFIED: Pass joint_pos_dim

        # Noise schedule
        betas = self._cosine_beta_schedule(diffusion_steps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        ddim_timesteps = torch.arange(0, diffusion_steps, diffusion_steps // ddim_steps)
        self.register_buffer('ddim_timesteps', ddim_timesteps)

    @staticmethod
    def _cosine_beta_schedule(T, s=0.008):
        t = torch.linspace(0, T, T + 1)
        ac = torch.cos((t / T + s) / (1 + s) * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        return torch.clip(1 - ac[1:] / ac[:-1], 0.0001, 0.9999)

    def encode_observation(self, obs):
        jp = self.normalizer.normalize_joint_pos(obs['joint_pos_history'])
        v = self.vision_encoder(obs['img_stack'])
        p = self.proprio_encoder(jp)
        return torch.cat([v, p], dim=-1)

    def forward_train(self, batch):
        B, device = batch['img_stack'].shape[0], batch['img_stack'].device
        s = self.encode_observation(batch)

        a0 = self.normalizer.normalize_action(batch['action_gt'])
        t = torch.randint(0, self.diffusion_steps, (B,), device=device)
        eps = torch.randn_like(a0)
        ab = self.alphas_cumprod[t].view(B, 1, 1)
        x_t = ab.sqrt() * a0 + (1 - ab).sqrt() * eps

        cond = torch.cat([s, self.time_embed(t.float())], dim=-1)
        loss = F.mse_loss(self.unet(x_t, cond), eps)
        return loss, {'diffusion_loss': loss.item()}

    @torch.no_grad()
    def forward_inference(self, obs, use_ddim=True):
        B, device = obs['img_stack'].shape[0], obs['img_stack'].device
        s = self.encode_observation(obs)

        x = torch.randn(B, self.action_horizon, self.action_dim, device=device)
        steps = list(reversed(self.ddim_timesteps.tolist() if use_ddim
                              else range(self.diffusion_steps)))

        for i, t in enumerate(steps):
            cond = torch.cat([s, self.time_embed(
                torch.full((B,), t, device=device).float())], dim=-1)
            eps = self.unet(x, cond)
            ab = self.alphas_cumprod[t]
            ab_prev = (self.alphas_cumprod[steps[i + 1]]
                       if i < len(steps) - 1 else torch.tensor(1.0, device=device))
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps if t > 0 else x0

        out = self.normalizer.denormalize_action(x)
        q = out[..., 3:7]
        out[..., 3:7] = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        return {
            'actions': out[..., :7],
            #'gripper_cmd': (out[..., 7] > 0.5).float(),
            'gripper_cmd': out[..., 7],
        }
