"""
Diffusion Policy v6 — Joint-space observations, Cartesian actions

Self-contained: all architecture components (VisionEncoder, UNet, etc.) are
defined in this file. No external model imports needed.

Observations:
  - RGB image stack: [B, 9, 224, 224] (current + previous + diff)
  - joint_pos_history: [B, T, 7] joint angles
  - joint_vel_history: [B, T, 7] joint velocities (optional, controlled by use_joint_vel)

Actions:
  - 7D pose [x, y, z, qx, qy, qz, qw] in xyzw convention
  - Gripper: separate binary classifier head from proprio embedding

Two variants controlled by use_joint_vel flag:
  - v6a (use_joint_vel=False): joint_pos(7) only  -> 35D proprio input
  - v6b (use_joint_vel=True):  joint_pos(7) + joint_vel(7) -> 70D proprio input
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18


# ── VisionEncoder ─────────────────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    """
    ResNet-18 encoder for 9-channel visual input.
    Input: [B, 9, 224, 224] (current + previous + difference)
    Output: [B, 128] visual embedding
    """
    def __init__(self, output_dim=128, pretrained=False):
        super().__init__()
        resnet = resnet18(pretrained=pretrained)
        self.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                pretrained_weight = resnet.conv1.weight
                self.conv1.weight[:, :3, :, :] = pretrained_weight
                self.conv1.weight[:, 3:6, :, :] = pretrained_weight
                self.conv1.weight[:, 6:9, :, :] = pretrained_weight
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ── SinusoidalTimeEmbedding ──────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


# ── ResidualBlock1D ───────────────────────────────────────────────────────────

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.film = nn.Linear(cond_dim, out_channels * 2)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.Mish()

    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h)
        film_params = self.film(cond)
        scale, shift = film_params.chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None]) + shift[:, :, None]
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.residual(x)


# ── ConditionalUNet1D ─────────────────────────────────────────────────────────

class ConditionalUNet1D(nn.Module):
    def __init__(self, action_dim=7, action_horizon=64,
                 cond_dim=256, hidden_dims=[128, 256, 512]):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.input_proj = nn.Conv1d(action_dim, hidden_dims[0], 1)

        self.down_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.down_blocks.append(ResidualBlock1D(hidden_dims[i], hidden_dims[i+1], cond_dim))

        self.bottleneck = ResidualBlock1D(hidden_dims[-1], hidden_dims[-1], cond_dim)

        self.up_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.up_blocks.append(ResidualBlock1D(hidden_dims[i] * 2, hidden_dims[i-1], cond_dim))

        self.output_proj = nn.Conv1d(hidden_dims[0], action_dim, 1)

    def forward(self, x_t, cond):
        x = x_t.transpose(1, 2)
        x = self.input_proj(x)
        skips = []
        for down in self.down_blocks:
            x = down(x, cond)
            skips.append(x)
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
        x = self.bottleneck(x, cond)
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = up(x, cond)
        x = self.output_proj(x)
        return x.transpose(1, 2)


# ── Normalizer v6 ─────────────────────────────────────────────────────────────

class Normalizer(nn.Module):
    def __init__(self, joint_pos_dim=7, joint_vel_dim=7, action_dim=7):
        super().__init__()
        for name, dim in [
            ('joint_pos', joint_pos_dim),
            ('joint_vel', joint_vel_dim),
            ('action',    action_dim),
        ]:
            self.register_buffer(f'{name}_mean', torch.zeros(dim))
            self.register_buffer(f'{name}_std',  torch.ones(dim))

    def normalize_joint_pos(self, x):
        return (x - self.joint_pos_mean) / self.joint_pos_std

    def normalize_joint_vel(self, x):
        return (x - self.joint_vel_mean) / self.joint_vel_std

    def normalize_action(self, x):
        return (x - self.action_mean) / self.action_std

    def denormalize_action(self, x):
        return x * self.action_std + self.action_mean


# ── ProprioEncoder v6 ─────────────────────────────────────────────────────────

class ProprioEncoder(nn.Module):
    """
    Encodes joint_pos [+ joint_vel] history into a fixed-size embedding.

    v6a (use_joint_vel=False): [B, T, 7]  → input_dim = 7 * T = 35
    v6b (use_joint_vel=True):  [B, T, 14] → input_dim = 14 * T = 70
    """
    def __init__(self, joint_pos_dim=7, joint_vel_dim=7,
                 use_joint_vel=False, history_len=5, output_dim=64):
        super().__init__()
        self.use_joint_vel = use_joint_vel

        per_step = joint_pos_dim + (joint_vel_dim if use_joint_vel else 0)
        input_dim = per_step * history_len

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )

    def forward(self, joint_pos, joint_vel=None):
        """
        joint_pos: [B, T, 7]
        joint_vel: [B, T, 7] or None
        Returns:   [B, 64]
        """
        B = joint_pos.shape[0]
        if self.use_joint_vel and joint_vel is not None:
            x = torch.cat([joint_pos, joint_vel], dim=-1)  # [B, T, 14]
        else:
            x = joint_pos                                   # [B, T, 7]
        x = x.reshape(B, -1)
        return self.net(x)


# ── Main model ────────────────────────────────────────────────────────────────

class DiffusionPolicyV6(nn.Module):
    def __init__(
        self,
        action_horizon=64,
        action_dim=7,
        diffusion_steps=100,
        ddim_steps=10,
        history_len=5,
        use_joint_vel=False,
        pretrained=False,
        freeze_vision_backbone=False,
        gripper_pos_weight=1.0,
    ):
        super().__init__()

        self.action_horizon  = action_horizon
        self.action_dim      = action_dim
        self.diffusion_steps = diffusion_steps
        self.ddim_steps      = ddim_steps
        self.history_len     = history_len
        self.use_joint_vel   = use_joint_vel

        # Vision encoder
        self.vision_encoder = VisionEncoder(output_dim=128, pretrained=pretrained)
        if freeze_vision_backbone:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # Proprio encoder
        self.proprio_encoder = ProprioEncoder(
            joint_pos_dim=7, joint_vel_dim=7,
            use_joint_vel=use_joint_vel,
            history_len=history_len, output_dim=64
        )

        self.obs_dim = 128 + 64  # 192

        # Time embedding + U-Net
        self.time_embed = SinusoidalTimeEmbedding(dim=64)
        self.unet = ConditionalUNet1D(
            action_dim=action_dim,
            action_horizon=action_horizon,
            cond_dim=self.obs_dim + 64,  # 256
            hidden_dims=[128, 256, 512]
        )

        # Gripper classifier (proprio-only)
        self.gripper_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Normalizer
        self.normalizer = Normalizer(joint_pos_dim=7, joint_vel_dim=7, action_dim=7)

        # Gripper pos_weight
        self.register_buffer(
            'gripper_pos_weight',
            torch.tensor(gripper_pos_weight, dtype=torch.float32)
        )

        # DDPM noise schedule
        betas = self._cosine_beta_schedule(diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        # DDIM timesteps
        ddim_stride = diffusion_steps // ddim_steps
        ddim_timesteps = torch.arange(0, diffusion_steps, ddim_stride)
        self.register_buffer('ddim_timesteps', ddim_timesteps)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def normalize_quaternion(self, q):
        return q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)

    def encode_observation(self, obs):
        """
        Args:
            obs: dict with keys:
                - img_stack:           [B, 9, 224, 224]
                - joint_pos_history:   [B, T, 7]
                - joint_vel_history:   [B, T, 7]  (ignored if use_joint_vel=False)
        Returns:
            s: [B, 192] full observation embedding
            p: [B, 64]  proprio-only embedding (for gripper head)
        """
        jp = self.normalizer.normalize_joint_pos(obs['joint_pos_history'])
        jv = None
        if self.use_joint_vel and 'joint_vel_history' in obs:
            jv = self.normalizer.normalize_joint_vel(obs['joint_vel_history'])

        v = self.vision_encoder(obs['img_stack'])
        p = self.proprio_encoder(jp, jv)
        s = torch.cat([v, p], dim=-1)
        return s, p

    def forward_train(self, batch):
        """
        Args:
            batch: dict with keys:
                - img_stack:           [B, 9, 224, 224]
                - joint_pos_history:   [B, T, 7]
                - joint_vel_history:   [B, T, 7]
                - action_gt:           [B, H, 7]
                - gripper_gt:          [B] float {0, 1}
        """
        B      = batch['img_stack'].shape[0]
        device = batch['img_stack'].device

        s, p = self.encode_observation(batch)

        # Diffusion loss (epsilon prediction)
        a0 = self.normalizer.normalize_action(batch['action_gt'])
        t  = torch.randint(0, self.diffusion_steps, (B,), device=device)
        eps = torch.randn_like(a0)
        alpha_bar_t = self.alphas_cumprod[t].view(B, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * a0 + torch.sqrt(1 - alpha_bar_t) * eps

        t_emb    = self.time_embed(t.float())
        cond     = torch.cat([s, t_emb], dim=-1)
        eps_pred = self.unet(x_t, cond)
        diffusion_loss = F.mse_loss(eps_pred, eps)

        # Gripper loss
        gripper_logits = self.gripper_classifier(p).squeeze(-1)
        gripper_loss = F.binary_cross_entropy_with_logits(
            gripper_logits,
            batch['gripper_gt'],
            pos_weight=self.gripper_pos_weight
        )

        total_loss = diffusion_loss + 0.1 * gripper_loss

        metrics = {
            'diffusion_loss': diffusion_loss.item(),
            'gripper_loss':   gripper_loss.item(),
            'gripper_acc':    ((gripper_logits > 0).float() == batch['gripper_gt']).float().mean().item(),
        }
        return total_loss, metrics

    @torch.no_grad()
    def forward_inference(self, obs, use_ddim=True):
        """
        Args:
            obs: dict with keys:
                - img_stack:           [B, 9, 224, 224]
                - joint_pos_history:   [B, T, 7]
                - joint_vel_history:   [B, T, 7]
        Returns:
            dict with actions [B, H, 7], gripper_cmd [B]
        """
        B      = obs['img_stack'].shape[0]
        device = obs['img_stack'].device

        s, p = self.encode_observation(obs)

        x_t = torch.randn(B, self.action_horizon, self.action_dim, device=device)
        timesteps = list(reversed(
            self.ddim_timesteps.tolist() if use_ddim else range(self.diffusion_steps)
        ))

        for i, t in enumerate(timesteps):
            t_batch  = torch.full((B,), t, device=device, dtype=torch.long)
            t_emb    = self.time_embed(t_batch.float())
            cond     = torch.cat([s, t_emb], dim=-1)
            eps_pred = self.unet(x_t, cond)

            alpha_bar_t = self.alphas_cumprod[t]
            if use_ddim:
                alpha_bar_prev = self.alphas_cumprod[timesteps[i + 1]] \
                    if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
            else:
                alpha_bar_prev = self.alphas_cumprod[t - 1] if t > 0 \
                    else torch.tensor(1.0, device=device)

            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            x_t = (torch.sqrt(alpha_bar_prev) * x0_pred +
                   torch.sqrt(1 - alpha_bar_prev) * eps_pred) if t > 0 else x0_pred

        actions = self.normalizer.denormalize_action(x_t)
        actions[..., 3:7] = self.normalize_quaternion(actions[..., 3:7])

        gripper_logits = self.gripper_classifier(p).squeeze(-1)
        gripper_cmd    = (gripper_logits > 0).float()

        return {
            'actions':     actions,
            'gripper_cmd': gripper_cmd,
        }


def compute_gripper_pos_weight(samples):
    """
    Compute pos_weight for gripper BCE from dataset statistics.
    gripper_gt: 1.0=closed, 0.0=open
    pos_weight = n_open / n_closed
    """
    gripper_vals = np.array([s['gripper_cmd'] for s in samples])
    n_closed = (gripper_vals > 0.5).sum()
    n_open   = (gripper_vals < 0.5).sum()

    if n_open == 0:
        print("Warning: no open gripper samples — pos_weight=1.0")
        return 1.0

    pos_weight = float(np.clip(n_open / n_closed, 0.05, 1.0))

    print(f"Gripper: {n_closed} closed ({100*n_closed/len(gripper_vals):.1f}%), "
          f"{n_open} open ({100*n_open/len(gripper_vals):.1f}%)")
    print(f"  pos_weight = {pos_weight:.3f}")

    return pos_weight
