"""Wan transformer and causal-VAE building blocks of the Wan2.1 control model in dit.py, adapted from Wan2.1 (Apache-2.0, the Alibaba Wan Team)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ROPE_MAX_LEN = 1024


def sinusoidal_embedding_1d(dim, position):
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


@torch.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    freqs = torch.outer(
        torch.arange(max_seq_len, device="cpu"),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2, device="cpu").to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


@torch.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        output.append(torch.cat([x_i, x[i, seq_len:]]))
    return torch.stack(output).to(x.dtype)


def attention(q, k, v):
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous()


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        if self.weight is not None and x.dtype != self.weight.dtype:
            return super().forward(x.to(self.weight.dtype)).to(x.dtype)
        return super().forward(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6, attend=attention):
        super().__init__()
        self.attend = attend
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def forward(self, x, grid_sizes, freqs, dtype):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
        k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
        v = self.v(x.to(dtype)).view(b, s, n, d)
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        x = self.attend(q.to(dtype), k.to(dtype), v.to(dtype)).to(dtype)
        return self.o(x.flatten(2))


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, dtype):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        x = self.attend(q.to(dtype), k.to(dtype), v.to(dtype))
        return self.o(x.flatten(2).to(dtype))


class WanAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, eps=1e-6, attend=attention):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, eps, attend)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.cross_attn = WanCrossAttention(dim, num_heads, eps, attend)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.empty(1, 6, dim))

    def forward(self, x, e, grid_sizes, freqs, context, dtype):
        e = (self.modulation + e).chunk(6, dim=1)
        y = self.self_attn((self.norm1(x) * (1 + e[1]) + e[0]).to(dtype), grid_sizes, freqs, dtype)
        x = x + y * e[2]
        x = x + self.cross_attn(self.norm3(x), context, dtype)
        y = self.ffn((self.norm2(x) * (1 + e[4]) + e[3]).to(dtype))
        return x + y * e[5]


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.empty(1, 2, dim))

    def forward(self, x, e):
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.norm(x) * (1 + e[1]) + e[0]
        if self.head.weight.dtype != x.dtype:
            x = x.to(self.head.weight.dtype)
        return self.head(x)


STATE_T = 2


def flat_time(x):
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)


def unflat_time(x, t):
    bt, c, h, w = x.shape
    return x.reshape(bt // t, t, c, h, w).permute(0, 2, 1, 3, 4)


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, prev=None):
        padding = list(self._padding)
        if prev is not None and self._padding[4] > 0:
            x = torch.cat([prev.to(x.device), x], dim=2)
            padding[4] -= prev.shape[2]
        return super().forward(F.pad(x, padding))


def causal(layer, x, state, state_idx):
    idx = state_idx[0]
    prev = x[:, :, -STATE_T:].clone()
    if prev.shape[2] < 2 and state[idx] is not None:
        prev = torch.cat([state[idx][:, :, -1].unsqueeze(2).to(prev.device), prev], dim=2)
    x = layer(x, state[idx])
    state[idx] = prev
    state_idx[0] += 1
    return x


class RMS_norm(nn.Module):
    def __init__(self, dim, images=True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones((dim, 1, 1) if images else (dim, 1, 1, 1)))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma + 0.0


class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.mode = mode
        if mode.startswith("up"):
            self.resample = nn.Sequential(Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim, 3, padding=1))
        else:
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        if mode == "upsample3d":
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample3d":
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    def forward(self, x, state, state_idx):
        b, c, t, h, w = x.shape
        if self.mode == "upsample3d":
            idx = state_idx[0]
            if state[idx] is None:
                state[idx] = "first"  # the first chunk skips the temporal upsampling
            else:
                first = isinstance(state[idx], str)
                prev = x[:, :, -STATE_T:].clone()
                if prev.shape[2] < 2:
                    pad = torch.zeros_like(prev) if first else state[idx][:, :, -1].unsqueeze(2).to(prev.device)
                    prev = torch.cat([pad, prev], dim=2)
                x = self.time_conv(x) if first else self.time_conv(x, state[idx])
                state[idx] = prev
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), 3).reshape(b, c, t * 2, h, w)
            state_idx[0] += 1
        t = x.shape[2]
        x = unflat_time(self.resample(flat_time(x)), t)
        if self.mode == "downsample3d":
            idx = state_idx[0]
            if state[idx] is None:
                state[idx] = x.clone()
            else:
                prev = x[:, :, -1:].clone()
                x = self.time_conv(torch.cat([state[idx][:, :, -1:], x], 2))
                state[idx] = prev
            state_idx[0] += 1
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(), CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(0.0), CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, state, state_idx):
        h = self.shortcut(x)
        for layer in self.residual:
            x = causal(layer, x, state, state_idx) if isinstance(layer, CausalConv3d) else layer(x)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = self.norm(flat_time(x))
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v).squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        return unflat_time(self.proj(x), t) + identity
