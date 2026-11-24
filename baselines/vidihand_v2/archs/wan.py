"""Wan2.2 transformer and causal-VAE building blocks, adapted from Wan2.2 (Apache-2.0)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


STATE_T = 2


def sinusoidal_embedding_1d(dim, position):
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def rope_params(max_seq_len, dim):
    freqs = torch.outer(torch.arange(max_seq_len, device="cpu"),
                        1.0 / torch.pow(10000, torch.arange(0, dim, 2, device="cpu").to(torch.float64).div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


@torch.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    f, h, w = grid_sizes[0].tolist()
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    xc = torch.view_as_complex(x[0].to(torch.float32).reshape(f * h * w, n, -1, 2))
    rot = torch.cat([freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1), freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                     freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1)
    return torch.view_as_real(xc * rot).flatten(2)[None].to(x.dtype)


def attention(q, k, v):
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous()


class WanRMSNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(x.dtype) * self.weight


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=1e-6)

    def forward(self, x):
        if self.weight is not None and x.dtype != self.weight.dtype:
            return super().forward(x.to(self.weight.dtype)).to(x.dtype)
        return super().forward(x)


class WanAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q, self.k, self.v, self.o = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.norm_q, self.norm_k = WanRMSNorm(dim), WanRMSNorm(dim)

    def cross(self, x, context, dtype, attend=attention):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        return self.o(attend(q.to(dtype), k.to(dtype), v.to(dtype)).flatten(2).to(dtype))


class WanAttentionBlock(nn.Module):

    def __init__(self, dim, ffn_dim, num_heads):
        super().__init__()
        self.norm1, self.norm2 = WanLayerNorm(dim), WanLayerNorm(dim)
        self.norm3 = WanLayerNorm(dim, elementwise_affine=True)
        self.self_attn, self.cross_attn = WanAttention(dim, num_heads), WanAttention(dim, num_heads)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.empty(1, 6, dim))


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


def causal(layer, x, state, idx):
    i = idx[0]
    prev = x[:, :, -STATE_T:].clone()
    if prev.shape[2] < STATE_T and state[i] is not None:
        prev = torch.cat([state[i][:, :, -1].unsqueeze(2).to(prev.device), prev], dim=2)
    x = layer(x, state[i])
    state[i] = prev
    idx[0] += 1
    return x


class RMS_norm(nn.Module):

    def __init__(self, dim, images=True):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones((dim, 1, 1) if images else (dim, 1, 1, 1)))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma + 0.0


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.residual = nn.Sequential(RMS_norm(in_dim, images=False), nn.SiLU(), CausalConv3d(in_dim, out_dim, 3, padding=1),
                                      RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(0.0), CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, state, idx):
        h = self.shortcut(x)
        for layer in self.residual:
            x = causal(layer, x, state, idx) if isinstance(layer, CausalConv3d) else layer(x)
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


class Resample(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        if mode not in ("downsample2d", "downsample3d"):
            raise ValueError(f"Resample mode is downsample2d or downsample3d, got {mode!r}")
        self.mode = mode
        self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        if mode == "downsample3d":
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    def forward(self, x, state, idx):
        t = x.shape[2]
        x = unflat_time(self.resample(flat_time(x)), t)
        if self.mode == "downsample3d":
            i = idx[0]
            if state[i] is None:
                state[i] = x.clone()
            else:
                prev = x[:, :, -1:].clone()
                x = self.time_conv(torch.cat([state[i][:, :, -1:], x], 2))
                state[i] = prev
            idx[0] += 1
        return x


class AvgDown3D(nn.Module):
    def __init__(self, c_in, c_out, factor_t, factor_s):
        super().__init__()
        self.c_out, self.factor_t, self.factor_s = c_out, factor_t, factor_s
        self.group = c_in * factor_t * factor_s * factor_s // c_out

    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t, 0))
        b, c, t, h, w = x.shape
        ft, fs = self.factor_t, self.factor_s
        x = x.view(b, c, t // ft, ft, h // fs, fs, w // fs, fs).permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        return x.view(b, self.c_out, self.group, t // ft, h // fs, w // fs).mean(dim=2)


class DownBlock(nn.Module):

    def __init__(self, c_in, c_out, blocks, down_t, down_s):
        super().__init__()
        self.avg_shortcut = AvgDown3D(c_in, c_out, 2 if down_t else 1, 2 if down_s else 1)
        layers = [ResidualBlock(c_in if i == 0 else c_out, c_out) for i in range(blocks)]
        if down_s:
            layers.append(Resample(c_out, "downsample3d" if down_t else "downsample2d"))
        self.downsamples = nn.Sequential(*layers)

    def forward(self, x, state, state_idx):
        y = x
        for layer in self.downsamples:
            y = layer(y, state, state_idx)
        return y + self.avg_shortcut(x)


class Encoder3d(nn.Module):
    def __init__(self, dim, z_dim, dim_mult, blocks, down_t):
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)
        self.downsamples = nn.Sequential(*(DownBlock(a, b, blocks, i < len(down_t) and down_t[i], i != len(dim_mult) - 1)
                                           for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))))
        self.middle = nn.Sequential(ResidualBlock(dims[-1], dims[-1]), AttentionBlock(dims[-1]), ResidualBlock(dims[-1], dims[-1]))
        self.head = nn.Sequential(RMS_norm(dims[-1], images=False), nn.SiLU(), CausalConv3d(dims[-1], z_dim, 3, padding=1))

    def forward(self, x, state):
        idx = [0]
        x = causal(self.conv1, x, state, idx)
        for layer in self.downsamples:
            x = layer(x, state, idx)
        x = self.middle[2](self.middle[1](self.middle[0](x, state, idx)), state, idx)
        return causal(self.head[2], self.head[1](self.head[0](x)), state, idx)


class VAEEncoder(nn.Module):
    """uint8 video [T, H, W, 3] (T = 4k + 1) -> normalised latent means [z_dim, k + 1, H / 16, W / 16]."""

    def __init__(self, z_dim, dim, dim_mult, blocks, down_t, latent_mean, latent_std, first, cont):
        super().__init__()
        self.z_dim, self.first, self.cont = z_dim, first, cont
        self.encoder = Encoder3d(dim, 2 * z_dim, dim_mult, blocks, down_t)
        self.conv1 = CausalConv3d(2 * z_dim, 2 * z_dim, 1)
        self.register_buffer("mean", torch.tensor(latent_mean), persistent=False)
        self.register_buffer("inv_std", 1.0 / torch.tensor(latent_std), persistent=False)

    def forward(self, video, bar=None):
        device = self.conv1.weight.device
        state = [None] * sum(isinstance(m, CausalConv3d) for m in self.encoder.modules())
        mean, inv_std = (v.to(torch.bfloat16).view(1, self.z_dim, 1, 1, 1) for v in (self.mean, self.inv_std))
        latents, a = [], 0
        while a < len(video):
            b = min(a + (self.cont if a else self.first), len(video))
            x = torch.from_numpy(video[a:b]).to(device).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0 * 2.0 - 1.0
            _, c, t, h, w = x.shape
            x = x.to(torch.bfloat16).view(1, c, t, h // 2, 2, w // 2, 2).permute(0, 1, 6, 4, 2, 3, 5).reshape(1, c * 4, t, h // 2, w // 2)
            bounds = ([(0, 1)] if a == 0 else []) + [(i, i + 4) for i in range(1 if a == 0 else 0, t, 4)]
            out = []
            for i, j in bounds:
                out.append(self.encoder(x[:, :, i:j], state))
                if bar is not None:
                    bar.update(j - i)
            latents.append(((self.conv1(torch.cat(out, 2)).chunk(2, dim=1)[0] - mean) * inv_std)[0].cpu())
            a = b
        return torch.cat(latents, 1)
