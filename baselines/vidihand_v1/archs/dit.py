"""Wan2.1-1.3B control-conditioned transformer and Wan2.1 VAE, adapted from Wan2.1 (Apache-2.0, Alibaba)."""

import math

import torch
import torch.nn as nn

from .wan import (
    ROPE_MAX_LEN, AttentionBlock, CausalConv3d, Head, RMS_norm, Resample, ResidualBlock, WanAttentionBlock, causal,
    rope_params, sinusoidal_embedding_1d,
)


def flash_attention(q, k, v):
    from flash_attn import flash_attn_func  # required: the model was trained and scored with FlashAttention-2

    return flash_attn_func(q, k, v)


class ControlBlock(WanAttentionBlock):
    def __init__(self, dim, ffn_dim, num_heads, eps, first):
        super().__init__(dim, ffn_dim, num_heads, eps, flash_attention)
        if first:
            self.before_proj = nn.Linear(dim, dim)
        self.after_proj = nn.Linear(dim, dim)


class ControlDiT(nn.Module):
    def __init__(self, in_dim=16, control_dim=96, dim=1536, ffn_dim=8960, out_dim=16, num_heads=12, num_layers=30,
                 text_len=512, freq_dim=256, text_dim=4096, eps=1e-6, patch_size=(1, 2, 2)):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.text_len, self.dim, self.freq_dim, self.out_dim = text_len, dim, freq_dim, out_dim
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.control_patch_embedding = nn.Conv3d(control_dim, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(WanAttentionBlock(dim, ffn_dim, num_heads, eps, flash_attention) for _ in range(num_layers))
        self.control_blocks = nn.ModuleList(ControlBlock(dim, ffn_dim, num_heads, eps, i == 0) for i in range(0, num_layers, 2))
        self.head = Head(dim, out_dim, self.patch_size, eps)
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(ROPE_MAX_LEN, d - 4 * (d // 6)),
            rope_params(ROPE_MAX_LEN, 2 * (d // 6)),
            rope_params(ROPE_MAX_LEN, 2 * (d // 6)),
        ], dim=1)

    def forward(self, x, t, context, seq_len, control_context, tap=None):
        """Velocity [B, 16, F, H, W], or the output tokens of block ``tap``; control hints follow every even block."""
        dtype = x.dtype
        if self.freqs.device != self.patch_embedding.weight.device:
            self.freqs = self.freqs.to(self.patch_embedding.weight.device)
        pad = lambda u: torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = torch.cat([pad(u.flatten(2).transpose(1, 2)) for u in x])
        with torch.autocast("cuda", dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        context = self.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context
        ]))
        args = (e0, grid_sizes, self.freqs, context, dtype)
        c = torch.cat([pad(self.control_patch_embedding(u.unsqueeze(0)).flatten(2).transpose(1, 2)) for u in control_context])
        hints = []
        for i, block in enumerate(self.control_blocks):
            if i == 0:
                c = block.before_proj(c.to(block.before_proj.weight.dtype)) + x
            c = block(c, *args)
            hints.append(block.after_proj(c.to(block.after_proj.weight.dtype)))
        for i, block in enumerate(self.blocks):
            x = block(x, *args)
            if i % 2 == 0:
                x = x + hints[i // 2]
            if i == tap:
                return x
        x = self.head(x, e)
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = torch.einsum("fhwpqrc->cfphqwr", u[: math.prod(v)].view(*v, *self.patch_size, c))
            out.append(u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)]))
        return torch.stack(out)


LATENT_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
               0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
LATENT_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
              3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
Z_DIM = 16


class UpResample(Resample):
    def __init__(self, dim, mode):
        super().__init__(dim, mode)
        self.resample[1] = nn.Conv2d(dim, dim // 2, 3, padding=1)


class Encoder3d(nn.Module):
    def __init__(self, dim=96, z_dim=2 * Z_DIM, dim_mult=(1, 2, 4, 4), temperal_downsample=(False, True, True)):
        super().__init__()
        dims = [dim * u for u in (1, *dim_mult)]
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)
        layers = []
        for i, (i_dim, o_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers += [ResidualBlock(i_dim, o_dim), ResidualBlock(o_dim, o_dim)]
            if i != len(dim_mult) - 1:
                layers.append(Resample(o_dim, "downsample3d" if temperal_downsample[i] else "downsample2d"))
        self.downsamples = nn.Sequential(*layers)
        self.middle = nn.Sequential(ResidualBlock(dims[-1], dims[-1]), AttentionBlock(dims[-1]), ResidualBlock(dims[-1], dims[-1]))
        self.head = nn.Sequential(RMS_norm(dims[-1], images=False), nn.SiLU(), CausalConv3d(dims[-1], z_dim, 3, padding=1))

    def forward(self, x, state, state_idx):
        x = causal(self.conv1, x, state, state_idx)
        for layer in self.downsamples:
            x = layer(x, state, state_idx)
        for layer in self.middle:
            x = layer(x, state, state_idx) if isinstance(layer, ResidualBlock) else layer(x)
        for layer in self.head:
            x = causal(layer, x, state, state_idx) if isinstance(layer, CausalConv3d) else layer(x)
        return x


class Decoder3d(nn.Module):
    def __init__(self, dim=96, z_dim=Z_DIM, dim_mult=(1, 2, 4, 4), temperal_upsample=(True, True, False)):
        super().__init__()
        dims = [dim * u for u in (dim_mult[-1], *dim_mult[::-1])]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0]), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0]))
        layers = []
        for i, (i_dim, o_dim) in enumerate(zip(dims[:-1], dims[1:])):
            i_dim = i_dim // 2 if i > 0 else i_dim
            layers += [ResidualBlock(i_dim, o_dim), ResidualBlock(o_dim, o_dim), ResidualBlock(o_dim, o_dim)]
            if i != len(dim_mult) - 1:
                layers.append(UpResample(o_dim, "upsample3d" if temperal_upsample[i] else "upsample2d"))
        self.upsamples = nn.Sequential(*layers)
        self.head = nn.Sequential(RMS_norm(dims[-1], images=False), nn.SiLU(), CausalConv3d(dims[-1], 3, 3, padding=1))

    def forward(self, x, state, state_idx):
        x = causal(self.conv1, x, state, state_idx)
        for layer in self.middle:
            x = layer(x, state, state_idx) if isinstance(layer, ResidualBlock) else layer(x)
        for layer in self.upsamples:
            x = layer(x, state, state_idx)
        for layer in self.head:
            x = causal(layer, x, state, state_idx) if isinstance(layer, CausalConv3d) else layer(x)
        return x


class WanVAE21(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder3d()
        self.conv1 = CausalConv3d(2 * Z_DIM, 2 * Z_DIM, 1)
        self.conv2 = CausalConv3d(Z_DIM, Z_DIM, 1)
        self.decoder = Decoder3d()
        self.register_buffer("mean", torch.tensor(LATENT_MEAN), persistent=False)
        self.register_buffer("inv_std", 1.0 / torch.tensor(LATENT_STD), persistent=False)

    @torch.no_grad()
    def encode(self, video):
        """[1, 3, T, H, W] in [-1, 1] -> normalised posterior mean [1, 16, 1 + (T - 1) / 4, H / 8, W / 8]."""
        state = [None] * sum(isinstance(m, CausalConv3d) for m in self.encoder.modules())
        f = video.shape[2]
        chunks = [video[:, :, :1]] + [video[:, :, 1 + 4 * (i - 1):1 + 4 * i] for i in range(1, 1 + (f - 1) // 4)]
        out = torch.cat([self.encoder(chunk, state, [0]) for chunk in chunks], 2)
        mu = self.conv1(out).chunk(2, dim=1)[0]
        mean, inv_std = self.mean.to(video.dtype), self.inv_std.to(video.dtype)
        return (mu - mean.view(1, Z_DIM, 1, 1, 1)) * inv_std.view(1, Z_DIM, 1, 1, 1)

    @torch.no_grad()
    def decode(self, z):
        """Normalised latent [1, 16, F, H, W] -> video [1, 3, 1 + 4 (F - 1), 8 H, 8 W] in [-1, 1]."""
        mean, inv_std = self.mean.to(z.dtype), self.inv_std.to(z.dtype)
        x = self.conv2(z / inv_std.view(1, Z_DIM, 1, 1, 1) + mean.view(1, Z_DIM, 1, 1, 1))
        state = [None] * sum(isinstance(m, CausalConv3d) for m in self.decoder.modules())
        return torch.cat([self.decoder(x[:, :, i:i + 1], state, [0]) for i in range(z.shape[2])], 2).clamp_(-1, 1)
