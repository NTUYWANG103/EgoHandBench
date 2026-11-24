"""The ViDiHand2 network: Wan2.2-Fun-5B-Control blocks with hand query tokens, and the decoding heads."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.vidihand_v2.archs.arch_utils import ray_features, rot6d_to_rotmat
from baselines.vidihand_v2.archs.wan import RMS_norm, WanAttentionBlock, attention, rope_apply, rope_params, sinusoidal_embedding_1d
from baselines.vidihand_v2.mano import FINGERS, PALM


CACHE_BIAS_PAD = 8
ROPE_MAX_LEN = 1024
DTYPE = torch.bfloat16


def time_dims(head_dim):
    return head_dim - 4 * (head_dim // 6)


def token_freqs(head_dim, t_pos):
    dt, ds = time_dims(head_dim), 2 * (head_dim // 6)
    zero = torch.zeros_like(t_pos)
    parts = []
    for pos, dim in ((t_pos, dt), (zero, ds), (zero, ds)):
        angle = pos.double()[..., None] * (1.0 / torch.pow(10000.0, torch.arange(0, dim, 2, device=pos.device, dtype=torch.float64).div(dim)))
        parts.append(torch.complex(torch.cos(angle), torch.sin(angle)))
    return torch.cat(parts, -1)


@torch.autocast("cuda", enabled=False)
def rope_tokens(x, freqs):
    b, n, h, d = x.shape
    xc = torch.view_as_complex(x.float().reshape(b, n, h, d // 2, 2))
    return torch.view_as_real(xc * freqs[None][:, :, None, :]).flatten(3).to(x.dtype)


def rope_shift_time(x, shift):
    b, n, h, d = x.shape
    dt = time_dims(d)
    inv = 1.0 / torch.pow(10000.0, torch.arange(0, dt, 2, device=x.device, dtype=torch.float64).div(dt))
    rot = torch.polar(torch.ones_like(inv), float(shift) * inv)
    xc = torch.view_as_complex(x.float().reshape(b, n, h, d // 2, 2))
    xc = torch.cat([xc[..., :dt // 2] * rot.to(torch.complex64), xc[..., dt // 2:]], -1)
    return torch.view_as_real(xc).flatten(3).to(x.dtype)


def attention_horizon(q, latents, per_q, segments, horizon, bias):
    keeps = [0 if not per else k.shape[1] // per for k, _, per in segments]
    vis = [tuple(min(n, horizon - t) for n in keeps) for t in range(latents)]
    d = q.shape[-1]
    lead = lambda x, c: x.new_zeros(*x.shape[:-1], c)

    def groups(aug):
        if aug:
            e = (bias.float() * d ** 0.5).view(1, 1, -1, 1)
        g = []
        for (k, v, per), n in zip(segments, keeps):
            if aug:
                ch = e.to(k.dtype).expand(k.shape[0], k.shape[1], -1, 1) if per else lead(k, 1)
                k, v = torch.cat([k, ch, lead(k, CACHE_BIAS_PAD - 1)], -1), torch.cat([v, lead(v, CACHE_BIAS_PAD)], -1)
            if per:
                g.append([[k], [v], per, n])
            else:
                g[-1][0].append(k)
                g[-1][1].append(v)
        return [(torch.cat(ks, 1), torch.cat(vs, 1), per, n) for ks, vs, per, n in g]

    def group_vis(vis_t):
        return [vis_t[i] for i, (_, _, per) in enumerate(segments) if per]

    def keys(gs, n_vis):
        parts = [(k[:, (n - m) * per:], v[:, (n - m) * per:]) for (k, v, per, n), m in zip(gs, n_vis)]
        return parts[0] if len(parts) == 1 else (torch.cat([a for a, _ in parts], 1), torch.cat([b for _, b in parts], 1))

    plain_g = gated_g = q_aug = None
    out, t = [], 0
    while t < latents:
        t2 = t + 1
        while t2 < latents and vis[t2] == vis[t]:
            t2 += 1
        if any(vis[t]):
            if gated_g is None:
                gated_g, q_aug = groups(True), torch.cat([q, lead(q, 1) + 1, lead(q, CACHE_BIAS_PAD - 1)], -1)
            kk, vv = keys(gated_g, group_vis(vis[t]))
            o = F.scaled_dot_product_attention(q_aug[:, t * per_q:t2 * per_q].transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2), scale=1.0 / d ** 0.5)
            out.append(o.transpose(1, 2)[..., :d].contiguous())
        else:
            if plain_g is None:
                plain_g = groups(False)
            kk, vv = keys(plain_g, group_vis(vis[t]))
            out.append(attention(q[:, t * per_q:t2 * per_q], kk, vv))
        t = t2
    return torch.cat(out, 1)


class LoRALinear(nn.Linear):

    def __init__(self, in_features, out_features, rank):
        super().__init__(in_features, out_features)
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(out_features, rank))

    def forward(self, x):
        out = super().forward(x)
        return out + F.linear(F.linear(x, self.lora_a.to(x)), self.lora_b.to(x))


def lora_block(dim, ffn, heads, rank, targets):
    block = WanAttentionBlock(dim, ffn, heads)
    for target in targets:
        parent, name = target.split(".")
        owner = getattr(block, parent)
        old = owner[int(name)] if name.isdigit() else getattr(owner, name)
        new = LoRALinear(old.in_features, old.out_features, rank)
        if name.isdigit():
            owner[int(name)] = new
        else:
            setattr(owner, name, new)
    return block


class Tower(nn.Module):
    def __init__(self, dim, ffn, heads, text_dim, text_len, freq_dim, tap, rank, lora_targets, z_dim, cond_dim, hidden, n_tokens, ar_from,
                 ar_horizon):
        super().__init__()
        self.dim, self.heads, self.text_dim, self.text_len, self.freq_dim = dim, heads, text_dim, text_len, freq_dim
        self.n_tokens, self.ar_from, self.ar_horizon, self.cond_dim = n_tokens, ar_from, ar_horizon, cond_dim
        self.patch_embedding = nn.Conv3d(z_dim + cond_dim, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.patch_lora = nn.Conv3d(z_dim, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(lora_block(dim, ffn, heads, rank, lora_targets) for _ in range(tap + 1))
        self.task_up = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim))
        self.task_down = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden))
        self.gate_video = nn.Parameter(torch.empty(tap + 1, heads))
        self.gate_token = nn.Parameter(torch.empty(tap + 1, heads))

    def prepare(self, prompt):
        device = self.patch_embedding.weight.device
        d = self.dim // self.heads
        self.freqs = torch.cat([rope_params(ROPE_MAX_LEN, time_dims(d)), rope_params(ROPE_MAX_LEN, 2 * (d // 6)),
                                rope_params(ROPE_MAX_LEN, 2 * (d // 6))], 1).to(device)
        c = self.freqs.shape[1]
        self.key_freqs = self.freqs.clone()
        self.key_freqs[:, c - 2 * (c // 3):] = 1.0
        with torch.autocast("cuda", dtype=DTYPE):
            with torch.autocast("cuda", dtype=torch.float32):
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, torch.zeros(1, device=device)).float())
                self.e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            prompt = prompt.to(device, DTYPE)
            self.context = self.text_embedding(torch.stack([torch.cat([prompt, prompt.new_zeros(self.text_len - len(prompt), self.text_dim)])]))

    @staticmethod
    def attn_input(block, x, e):
        return (block.norm1(x) * (1 + e[1]) + e[0]).to(DTYPE)

    def attn_output(self, block, x, att, e):
        x = x + block.self_attn.o(att.to(DTYPE).flatten(2)) * e[2]
        x = x + block.cross_attn.cross(block.norm3(x), self.context, DTYPE)
        return x + block.ffn((block.norm2(x) * (1 + e[4]) + e[3]).to(DTYPE)) * e[5]

    def video_block(self, block, e, x, grid_sizes, ar, bias, keep, out):
        sa = block.self_attn
        y = self.attn_input(block, x, e)
        b, s, n, d = *y.shape[:2], sa.num_heads, sa.head_dim
        kn = sa.norm_k(sa.k(y))
        v = sa.v(y)
        v4 = v.view(b, s, n, d)
        k = rope_apply(kn.view(b, s, n, d), grid_sizes, self.freqs)
        kv = (k, v4) if keep else None
        if not out:
            return None, kn, v, kv
        q = rope_apply(sa.norm_q(sa.q(y)).view(b, s, n, d), grid_sizes, self.freqs)
        if ar is not None:
            f_, h_, w_ = (int(g) for g in grid_sizes[0].tolist())
            att = attention_horizon(q.to(DTYPE), f_, h_ * w_, [(ar[0].to(DTYPE), ar[1].to(DTYPE), h_ * w_), (k.to(DTYPE), v4.to(DTYPE), 0)],
                                    self.ar_horizon, bias)
        else:
            att = attention(q.to(DTYPE), k.to(DTYPE), v4.to(DTYPE))
        return self.attn_output(block, x, att, e), kn, v, kv

    def token_block(self, block, e, ht, kn, v, grid_sizes, tok_freqs, ar, bias):
        sa = block.self_attn
        b, lv, n, d = v.shape[0], v.shape[1], sa.num_heads, sa.head_dim
        kv_k = rope_apply(kn.view(b, lv, n, d), grid_sizes, self.key_freqs)
        kv_v = v.view(b, lv, n, d)
        nt = ht.shape[1]
        tmp = self.attn_input(block, ht, e)
        q = rope_tokens(sa.norm_q(sa.q(tmp)).view(b, nt, n, d), tok_freqs)
        k = rope_tokens(sa.norm_k(sa.k(tmp)).view(b, nt, n, d), tok_freqs)
        vt = sa.v(tmp).view(b, nt, n, d)
        if ar is not None:
            f_ = int(grid_sizes[0][0])
            y = attention_horizon(q.to(DTYPE), f_, nt // f_, [(ar[2].to(DTYPE), ar[3].to(DTYPE), lv // f_), (kv_k.to(DTYPE), kv_v.to(DTYPE), 0),
                                                              (ar[4].to(DTYPE), ar[5].to(DTYPE), nt // f_), (k.to(DTYPE), vt.to(DTYPE), 0)], self.ar_horizon, bias)
        else:
            y = attention(q.to(DTYPE), torch.cat([kv_k.to(DTYPE), k.to(DTYPE)], 1), torch.cat([kv_v.to(DTYPE), vt.to(DTYPE)], 1))
        return self.attn_output(block, ht, y, e), (kv_k, kv_v, k, vt)

    def run_block(self, i, hv, ht, grid_sizes, tok_freqs, ar, keep, out=True):
        block = self.blocks[i]
        e = (block.modulation + self.e0).chunk(6, dim=1)
        hv_out, kn, v, vkv = self.video_block(block, e, hv, grid_sizes, None if ar is None else ar[:2], None if ar is None else self.gate_video[i],
                                              keep, out)
        ht, tkv = self.token_block(block, e, ht, kn, v, grid_sizes, tok_freqs, ar, None if ar is None else self.gate_token[i])
        return hv_out, ht, ((vkv + tkv) if keep else None)

    def forward(self, x, seeds, cache, capture, fork=True):
        """x [1, z, f, H, W] -> (video features [1, dim, f, H / 2, W / 2], token states [f, tokens, hidden]), the cache-free read or None,
        the per-block keys / values to cache."""
        f, h, w = x.shape[2], x.shape[3] // 2, x.shape[4] // 2
        tokens = self.task_up(seeds[None])
        t_pos = torch.arange(f, device=x.device, dtype=torch.float64).repeat_interleave(self.n_tokens)
        tok_freqs = token_freqs(self.dim // self.heads, t_pos)
        with torch.autocast("cuda", dtype=DTYPE):
            xb = x.to(DTYPE)
            xy = torch.cat([xb[0], xb.new_zeros(self.cond_dim, *xb.shape[2:])])[None]
            hv = self.patch_embedding(xy) + F.conv3d(xy[:, :x.shape[1]], self.patch_lora.weight.to(xy), self.patch_lora.bias.to(xy), stride=(1, 2, 2))
            grid_sizes = torch.tensor([hv.shape[2:]], dtype=torch.long)
            hv = hv.flatten(2).transpose(1, 2).contiguous()
            ht = tokens.to(hv.dtype)
            hv_f = ht_f = None
            caps = []
            for i in range(len(self.blocks)):
                ar = cache[i] if cache is not None else None
                if ar is not None and fork:
                    if hv_f is None:
                        hv_f, ht_f = hv, ht
                    hv_f, ht_f, _ = self.run_block(i, hv_f, ht_f, grid_sizes, tok_freqs, None, False)
                last = i == len(self.blocks) - 1 and hv_f is not None
                hv, ht, kv = self.run_block(i, hv, ht, grid_sizes, tok_freqs, ar, capture and i >= self.ar_from, not last)
                caps.append(kv)
        fold = lambda t: t[:, :f * h * w].reshape(1, f, h, w, self.dim).permute(0, 4, 1, 2, 3).contiguous().float()
        read = lambda v, t: (fold(v), self.task_down(t.float()).reshape(f, self.n_tokens, -1))
        free = read(hv_f, ht_f) if hv_f is not None else None
        return (free[0] if free is not None else fold(hv), self.task_down(ht.float()).reshape(f, self.n_tokens, -1)), free, caps

    def next_cache(self, caps, f):
        keep = f - 1
        out = []
        for c in caps:
            if c is None:
                out.append(None)
                continue
            out.append(tuple(v[:, :keep * (v.shape[1] // f)] if j % 2 else rope_shift_time(v[:, :keep * (v.shape[1] // f)], -keep)
                             for j, v in enumerate(c)))
        return out


STRIDE = 4
NULL_BIAS = -2.0


def mlp(dim, hidden, out, norm=False):
    return nn.Sequential(*([nn.LayerNorm(dim)] if norm else []), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, out))


class BiConv3d(nn.Conv3d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, (3, 1, 1))

    def forward(self, x):
        return super().forward(F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate"))


class ResBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.residual = nn.Sequential(RMS_norm(dim, images=False), nn.SiLU(), BiConv3d(dim, dim),
                                      RMS_norm(dim, images=False), nn.SiLU(), BiConv3d(dim, dim))

    def forward(self, x):
        return x + self.residual(x)


class TimeUp(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.blocks = nn.Sequential(ResBlock(dim))
        self.time_conv = BiConv3d(dim, 2 * dim)

    def forward(self, x):
        y = self.blocks(x)
        b, c, t, h, w = y.shape
        y = self.time_conv(y).reshape(b, 2, c, t, h, w)
        return torch.stack((y[:, 0], y[:, 1]), 3).reshape(b, c, 2 * t, h, w) + x.repeat_interleave(2, dim=2)


class LinearPath(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.c = nn.Parameter(torch.empty(dim, 3))
        self.u1 = nn.Parameter(torch.empty(dim, 2, 3))
        self.u2 = nn.Parameter(torch.empty(dim, 2, 3))

    @staticmethod
    def _conv(x, w):
        xp = F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")
        w = w.to(x.dtype)[None, :, :, None, None]
        return w[:, :, 0:1] * xp[:, :, :-2] + w[:, :, 1:2] * xp[:, :, 1:-1] + w[:, :, 2:3] * xp[:, :, 2:]

    def _up(self, x, u):
        y0, y1 = x + self._conv(x, u[:, 0]), x + self._conv(x, u[:, 1])
        n, c, t, h, w = x.shape
        return torch.stack((y0, y1), 3).reshape(n, c, 2 * t, h, w)

    def forward(self, x):
        return self._up(self._up(self._conv(x, self.c), self.u1), self.u2)


class TimeDecoder(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.base = LinearPath(dim)
        self.in_proj = nn.Conv3d(dim, hidden, 1)
        self.conv_in = BiConv3d(hidden, hidden)
        self.middle = nn.Sequential(ResBlock(hidden), ResBlock(hidden))
        self.up = nn.Sequential(TimeUp(hidden), TimeUp(hidden))
        self.head_norm = RMS_norm(hidden, images=False)
        self.head = BiConv3d(hidden, hidden)
        self.out_proj = nn.Conv3d(hidden, dim, 1)

    def forward(self, x):
        y = self.up(self.middle(self.conv_in(self.in_proj(x))))
        return self.base(x) + self.out_proj(self.head(F.silu(self.head_norm(y))))

    def frames(self, x, n, lead):
        return self(x)[:, :, 3 - lead:3 - lead + n]


def slot_frames(dec, feat, n, lead):
    return dec.frames(feat.permute(0, 3, 1, 2).unsqueeze(-1), n, lead).squeeze(-1).permute(0, 2, 3, 1)


def grid_frames(dec, grid, n, h, w, lead):
    b, f, _, d = grid.shape
    return dec.frames(grid.permute(0, 3, 1, 2).reshape(b, d, f, h, w), n, lead).reshape(b, d, -1, h * w).permute(0, 2, 3, 1)


def coord_frames(dec, j2d, j3d, m2, n, lead):
    b, f, s, j, _ = j2d.shape
    var = (m2.float() - j2d.float() ** 2).clamp_min(0)
    x = torch.cat([j2d.float(), j3d.float(), var], -1).reshape(b, f, s * j, 7)
    y = slot_frames(dec, x, n, lead).reshape(b, n, s, j, 7)
    return y[..., :2].to(j2d.dtype).contiguous(), (y[..., :2] ** 2 + y[..., 5:7].clamp_min(0)).to(m2.dtype)


class HandGridAttn(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()
        self.qnorm, self.kvnorm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=heads, batch_first=True)
        self.out = nn.Linear(dim, dim)

    def forward(self, hand, grid):
        b, t, s, d = hand.shape
        kv = self.kvnorm(grid).reshape(b * t, grid.shape[2], d)
        y, _ = self.attn(self.qnorm(hand).reshape(b * t, s, d), kv, kv, need_weights=False)
        return self.out(y).reshape(b, t, s, d)


class Projector(nn.Module):
    def __init__(self, tower_dim, hidden, heads, head_hidden, ray_freqs, n_hand, n_joint, n_register, time_hidden):
        super().__init__()
        self.hidden, self.n_hand, self.n_joint = hidden, n_hand, n_joint
        self.input_proj = nn.Sequential(nn.Linear(tower_dim, hidden), nn.LayerNorm(hidden))
        self.spatial_pe_grid = nn.Parameter(torch.empty(1, hidden, 16, 16))
        self.ray_proj = nn.Sequential(nn.Linear(4 * ray_freqs, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.register_buffer("freq_bands", 2.0 ** torch.arange(ray_freqs).float(), persistent=False)
        self.slot_embed = nn.Parameter(torch.empty(2, hidden))
        self.hand_token = nn.Parameter(torch.empty(1, n_hand, hidden))
        self.joint_token = nn.Parameter(torch.empty(1, n_joint, hidden))
        self.register_token = nn.Parameter(torch.empty(1, n_register, hidden))
        self.final_norm = nn.LayerNorm(hidden)
        self.readout_qnorm, self.readout_kvnorm = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.readout_attn = nn.MultiheadAttention(hidden, num_heads=heads, batch_first=True)
        self.head_joint_3d = mlp(hidden, head_hidden, 3, norm=True)
        self.hand_fuse = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.hand_xattn = HandGridAttn(hidden, heads)
        self.head_global_orient, self.head_hand_pose = mlp(hidden, head_hidden, 6), mlp(hidden, head_hidden, 15 * 6)
        self.head_exists_2d, self.head_exists_3d = mlp(hidden, head_hidden, 1), mlp(hidden, head_hidden, 1)
        self.head_betas_hand = mlp(hidden, head_hidden, 10, norm=True)
        self.time = nn.ModuleDict(dict(hand=TimeDecoder(hidden, time_hidden), joint=TimeDecoder(hidden, time_hidden),
                                       coord=TimeDecoder(7, time_hidden), xgrid=TimeDecoder(hidden, time_hidden)))

    def seeds(self, f):
        hand = (self.hand_token + self.slot_embed.view(1, self.n_hand, self.hidden)).expand(f, -1, -1)
        return torch.cat([hand, self.joint_token.expand(f, -1, -1), self.register_token.expand(f, -1, -1)], 1).reshape(-1, self.hidden)

    def grid(self, feat, intr, canvas_wh):
        _, c, f, h, w = feat.shape
        x = self.input_proj(feat.permute(0, 2, 3, 4, 1).reshape(f, h * w, c))
        pe = F.interpolate(self.spatial_pe_grid.float(), size=(h, w), mode="bilinear", align_corners=True)
        x = x + pe.reshape(1, self.hidden, h * w).transpose(1, 2).to(x.dtype)
        rpe = {k: self.ray_proj(ray_features(h, w, k, canvas_wh, self.freq_bands)).to(x.dtype) for k in set(intr)}
        return x + torch.stack([rpe[k] for k in intr])

    def hands(self, special, x, n, lead, h, w):
        f = x.shape[0]
        s = self.final_norm(special).view(1, f, -1, self.hidden)
        hand = s[:, :, :self.n_hand]
        hand = slot_frames(self.time["hand"], hand, n, lead)
        hand = hand + self.hand_fuse(hand)
        hand = hand + self.hand_xattn(hand, grid_frames(self.time["xgrid"], x.view(1, f, h * w, self.hidden), n, h, w, lead))
        return s, hand.reshape(n, 2, self.hidden)

    @staticmethod
    def soft_moments(attn, h, w):
        n, j, _ = attn.shape
        hm = attn.reshape(n, j, h, w)
        us = torch.linspace(0.0, 1.0, w, device=attn.device, dtype=attn.dtype)
        vs = torch.linspace(0.0, 1.0, h, device=attn.device, dtype=attn.dtype)
        wsum = hm.sum(dim=(-1, -2)).clamp(min=1e-06)
        mu = torch.stack([(hm.sum(dim=-2) * us).sum(dim=-1) / wsum, (hm.sum(dim=-1) * vs).sum(dim=-1) / wsum], dim=-1)
        hm = attn.float().reshape(n, j, h, w)
        us, vs = us.float(), vs.float()
        wsum = hm.sum(dim=(-1, -2)).clamp(min=1e-06)
        m2 = torch.stack([(hm.sum(dim=-2) * us.square()).sum(dim=-1) / wsum, (hm.sum(dim=-1) * vs.square()).sum(dim=-1) / wsum], dim=-1)
        return mu, m2

    def hand_heads(self, hand):
        return dict(shape_tokens=hand, frame_betas=self.head_betas_hand(hand), exists_2d=torch.sigmoid(self.head_exists_2d(hand).squeeze(-1)),
                    exists_3d=torch.sigmoid(self.head_exists_3d(hand).squeeze(-1)))

    def hand_states(self, feat, special, n, lead, intr, canvas_wh):
        h, w = feat.shape[3:]
        return self.hand_heads(self.hands(special, self.grid(feat, intr, canvas_wh), n, lead, h, w)[1])

    def forward(self, feat, special, n, lead, intr, canvas_wh):
        f, h, w = feat.shape[2:]
        x = self.grid(feat, intr, canvas_wh)
        s, hand = self.hands(special, x, n, lead, h, w)
        jt = s[:, :, self.n_hand:self.n_hand + self.n_joint].reshape(f, self.n_joint, self.hidden)
        kv = self.readout_kvnorm(x)
        jf, attn = self.readout_attn(self.readout_qnorm(jt), kv, kv, need_weights=True, average_attn_weights=True)
        mu, m2 = self.soft_moments(attn, h, w)
        rootrel = self.head_joint_3d(jf).reshape(1, f, 2, 21, 3)
        rootrel = rootrel - rootrel[:, :, :, 0:1, :]
        j2d, j2d_m2 = coord_frames(self.time["coord"], mu.reshape(1, f, 2, 21, 2), rootrel, m2.reshape(1, f, 2, 21, 2), n, lead)
        joint_feat = slot_frames(self.time["joint"], jf.reshape(1, f, self.n_joint, self.hidden), n, lead).reshape(n, 2, 21, self.hidden)
        return dict(global_orient=rot6d_to_rotmat(self.head_global_orient(hand)),
                    hand_pose=rot6d_to_rotmat(self.head_hand_pose(hand).reshape(-1, 6)).reshape(n, 2, 15, 3, 3),
                    hand_feat=hand, joint_feat=joint_feat, joints2d=j2d[0], joints2d_m2=j2d_m2[0], **self.hand_heads(hand))


def catmull_weights(f, n, lead, device):
    pos = ((torch.arange(n, device=device, dtype=torch.float64) - lead) / 4).clamp(0, f - 1)
    x = torch.eye(f, device=device, dtype=torch.float32)[None]
    i1 = pos.floor().clamp(0, f - 2).long()
    u = (pos - i1.double()).float()
    i0, i2, i3 = (i1 - 1).clamp(0, f - 1), i1 + 1, (i1 + 2).clamp(0, f - 1)
    p0, p1, p2, p3 = x[..., i0], x[..., i1], x[..., i2], x[..., i3]
    u2, u3 = u * u, u * u * u
    y = 0.5 * (2 * p1 + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (3 * p1 - p0 - 3 * p2 + p3) * u3)
    knot = (pos - pos.round()).abs() < 1e-9
    y[..., knot] = x[..., pos[knot].round().long()]
    return y[0].T


def read_maps(logits, null_logit, lse, hw4, window):
    h4, w4 = hw4
    arg = logits.argmax(-1)
    cy, cx = arg // w4, arg % w4
    k = torch.arange(-window, window + 1, device=logits.device)
    dy, dx = torch.meshgrid(k, k, indexing="ij")
    iy, ix = cy[..., None] + dy.reshape(-1), cx[..., None] + dx.reshape(-1)
    inside = (iy >= 0) & (iy < h4) & (ix >= 0) & (ix < w4)
    win = logits.gather(-1, (iy.clamp(0, h4 - 1) * w4 + ix.clamp(0, w4 - 1)).long()).masked_fill(~inside, float("-inf"))
    p = torch.softmax(win, -1)
    u, v = (ix.float() + 0.5) * STRIDE, (iy.float() + 0.5) * STRIDE
    mu, mv = (p * u).sum(-1), (p * v).sum(-1)
    var = 0.5 * ((p * (u - mu[..., None]).square()).sum(-1) + (p * (v - mv[..., None]).square()).sum(-1))
    return (torch.stack((mu, mv), -1), var.clamp_min(1.0), torch.sigmoid(null_logit.float() - lse),
            torch.exp(torch.logsumexp(win, -1) - torch.logaddexp(lse, null_logit.float())))


def conv_block(c_in, c_out, groups, dilation=1):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=dilation, dilation=dilation), nn.GroupNorm(groups, c_out), nn.GELU(),
                         nn.Conv2d(c_out, c_out, 3, padding=dilation, dilation=dilation), nn.GroupNorm(groups, c_out), nn.GELU())


class Temporal(nn.Module):
    def __init__(self, c, kernel):
        super().__init__()
        self.dw = nn.Conv1d(c, c, kernel, padding=kernel // 2, groups=c)
        self.mean = nn.Conv2d(c, c, 1)

    def forward(self, x):
        t, c, h, w = x.shape
        y = self.dw(x.permute(2, 3, 1, 0).reshape(h * w, c, t)).reshape(h, w, c, t).permute(3, 2, 0, 1)
        return x + (y + self.mean(x.sum(0, keepdim=True) / x.new_tensor(t)))


class JointCouple(nn.Module):
    def __init__(self, width, widths, groups, kernel):
        super().__init__()
        c0, c1, c2, c3 = widths
        self.stem = nn.Conv2d(3 * 42 + width, c0, 1)
        self.enc0, self.enc1, self.enc2, self.enc3 = (conv_block(c, c, groups) for c in widths)
        self.down1, self.down2, self.down3 = nn.Conv2d(c0, c1, 3, 2, 1), nn.Conv2d(c1, c2, 3, 2, 1), nn.Conv2d(c2, c3, 3, 2, 1)
        self.t2, self.t3, self.tb = Temporal(c2, kernel), Temporal(c3, kernel), Temporal(c3, kernel)
        self.bott = conv_block(c3, c3, groups, dilation=2)
        self.glob = nn.Conv2d(c3, c3, 1)
        self.dec2, self.dec1, self.dec0 = conv_block(c3 + c2, c2, groups), conv_block(c2 + c1, c1, groups), conv_block(c1 + c0, c0, groups)
        self.head = nn.Conv2d(c0, 42, 1)

    def forward(self, logits, null_logit, tower, hw4, anchor):
        t, j, n = logits.shape
        h4, w4 = hw4
        up = lambda x, ref: F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            L = logits.float()
            lse_in = torch.logsumexp(L, -1)
            logp = (L - lse_in[..., None]).reshape(t, j, h4, w4)
            pnull = torch.sigmoid(null_logit.float() - lse_in)
            tw = F.interpolate(tower.float(), size=(h4, w4), mode="bilinear", align_corners=False)
            a, v = anchor[0].float().reshape(t, j, 2), anchor[1].float().reshape(t, j)
            cu = (torch.arange(w4, device=L.device, dtype=torch.float32) + 0.5) * STRIDE
            cv = (torch.arange(h4, device=L.device, dtype=torch.float32) + 0.5) * STRIDE
            ok = torch.isfinite(a).all(-1) & torch.isfinite(v)
            a0, v0 = torch.where(ok[..., None], a, torch.zeros_like(a)), torch.where(ok, v, torch.zeros_like(v)).clamp_min(0.0)
            d2 = (cu[None, None, None, :] - a0[..., 0, None, None]) ** 2 + (cv[None, None, :, None] - a0[..., 1, None, None]) ** 2
            gauss = torch.exp(-d2 / (2.0 * (v0[..., None, None] + STRIDE ** 2))) * ok[..., None, None].float()
            x = torch.cat([logp, pnull[..., None, None].expand(t, j, h4, w4), tw, gauss], 1)
            e0 = self.enc0(self.stem(x))
            e1 = self.enc1(self.down1(e0))
            e2 = self.t2(self.enc2(self.down2(e1)))
            e3 = self.t3(self.enc3(self.down3(e2)))
            b = self.bott(e3)
            b = b + self.glob(b.mean((-2, -1), keepdim=True))
            b = self.tb(b)
            d2 = self.dec2(torch.cat((up(b, e2), e2), 1))
            d1 = self.dec1(torch.cat((up(d2, e1), e1), 1))
            d0 = self.dec0(torch.cat((up(d1, e0), e0), 1))
            delta = self.head(d0).reshape(t, j, n)
            shift = torch.logsumexp(L + delta, -1) - lse_in
            return L + (delta - shift[..., None])


class ReadMix(nn.Module):
    def __init__(self, radius, hidden):
        super().__init__()
        self.offsets = tuple(k for k in range(-radius, radius + 1) if k)
        self.mlp = nn.Sequential(nn.Linear(7 + len(self.offsets), hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.joint_bias = nn.Parameter(torch.empty(42))

    def weights(self, read, var, p_null, frame_weights):
        t, j = var.shape
        dev = var.device
        read, var, p_null = read.float(), var.float().clamp_min(1e-6), p_null.float().clamp(0.0, 1.0)
        pos = (frame_weights.float() * torch.arange(frame_weights.shape[1], device=dev, dtype=torch.float32)).sum(-1)
        ph = 2.0 * math.pi * (pos - pos.floor())
        ar = torch.arange(t, device=dev)
        n = len(self.offsets)
        eye = torch.eye(n, device=dev, dtype=torch.float32)
        lv = (var.log() - 2.0) / 2.0
        cs, ms = [], []
        for i, k in enumerate(self.offsets):
            idx = (ar + k).clamp(0, t - 1)
            inside = (ar + k >= 0) & (ar + k < t)
            ms.append(inside.float()[:, None] * (1.0 - p_null[idx]))
            d2 = (read[idx] - read).square().sum(-1) / (var + var[idx])
            c = torch.stack((ph.sin()[:, None].expand(t, j), ph.cos()[:, None].expand(t, j), lv, lv[idx], torch.log1p(d2), p_null, p_null[idx]), -1)
            cs.append(torch.cat((c, eye[i].expand(t, j, n)), -1))
        cue, mask = torch.stack(cs, 2), torch.stack(ms, 2)
        if not (torch.isfinite(cue).all() and torch.isfinite(mask).all()):
            raise ValueError("ViDiHand2: a non-finite heat-map read reached the read mixer")
        u = self.mlp(cue.to(self.mlp[0].weight.dtype))[..., 0].float() + self.joint_bias.float()[None, :, None]
        return torch.where(mask > 0, u.clamp_min(0.0) * mask, torch.zeros_like(u))

    def pool(self, logits, lse, a):
        ell = logits - lse[..., None]
        w = a / (1.0 + a.sum(-1, keepdim=True))
        z = logits
        for i, k in enumerate(self.offsets):
            z = z + w[..., i, None] * (torch.roll(ell, -int(k), 0) - ell)
        return z - (torch.logsumexp(z, -1) - lse)[..., None]


class HeatmapReader(nn.Module):
    def __init__(self, tower_dim, hidden, width, window, batch, imagenet_mean, imagenet_std, couple, tmix, time_hidden):
        super().__init__()
        self.width, self.window, self.batch = width, window, batch
        self.register_buffer("mean", torch.tensor(imagenet_mean).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(imagenet_std).reshape(1, 3, 1, 1), persistent=False)
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(32, width, 3, stride=2, padding=1), nn.GELU())
        self.tower_proj = nn.Conv2d(tower_dim, width, 1)
        self.fuse = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, width, 3, padding=1))
        self.q_norm = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, width)
        self.null_head = nn.Linear(hidden, 1)
        self.jc = JointCouple(width, **couple)
        self.tmix = ReadMix(**tmix)
        self.time = TimeDecoder(width, time_hidden)

    def forward(self, rgb, feat, joint_feat, anchor, lead):
        t, h, w, _ = rgb.shape
        hw4 = (h // STRIDE, w // STRIDE)
        lat = self.tower_proj(feat[0].transpose(0, 1))
        tower = self.time.frames(lat.transpose(0, 1).unsqueeze(0), t, lead)[0].transpose(0, 1)
        jn = self.q_norm(joint_feat)
        q = self.q_proj(jn)
        null = self.null_head(jn)[..., 0].float() + (math.log(hw4[0] * hw4[1]) + NULL_BIAS)
        logits = []
        for i in range(0, t, self.batch):
            x = (rgb[i:i + self.batch].permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
            s = self.stem(x)
            s = s + F.interpolate(tower[i:i + self.batch].to(s.dtype), size=s.shape[-2:], mode="bilinear", align_corners=False)
            e = self.fuse(s)
            with torch.autocast(device_type=e.device.type, enabled=False):
                logits.append(torch.bmm(q[i:i + self.batch].float(), e.float().flatten(2)) / math.sqrt(self.width))
        logits = torch.cat(logits)
        logits = self.jc(logits, null, tower, hw4, anchor)
        lse = torch.logsumexp(logits, -1)
        read_px, var_px, p_null, win_mass = read_maps(logits, null, lse, hw4, self.window)
        a = self.tmix.weights(read_px, var_px, p_null, catmull_weights(feat.shape[2], t, lead, rgb.device))
        pooled = self.tmix.pool(logits, lse, a)
        return dict(read_px=read_px, var_px=var_px, p_null=p_null, win_mass=win_mass, pooled_px=read_maps(pooled, null, lse, hw4, self.window)[0])


class ManoHeat(nn.Module):
    def __init__(self, hidden, head_hidden, bins, rot_span, beta_span):
        super().__init__()
        self.bins, self.rot_span, self.beta_span = bins, rot_span, beta_span
        self.rot = mlp(hidden, head_hidden, 16 * 3 * bins, norm=True)
        self.beta = mlp(hidden, head_hidden, 10 * bins, norm=True)

    def expect(self, p, span):
        mid = (self.bins - 1) // 2
        c = ((torch.arange(self.bins, device=p.device, dtype=torch.float64) - mid).to(p.dtype) * (span / mid))[mid + 1:]
        return ((p[..., mid + 1:] - p[..., :mid].flip(-1)) * c).sum(-1)

    @staticmethod
    def exp_minus_eye(v):
        th2 = (v * v).sum(-1)[..., None, None]
        small = th2 < 1e-6
        th = torch.where(small, torch.ones_like(th2), th2).sqrt()
        a = torch.where(small, 1.0 - th2 / 6.0, torch.sin(th) / th)
        b = torch.where(small, 0.5 - th2 / 24.0, (1.0 - torch.cos(th)) / torch.where(small, torch.ones_like(th2), th2))
        x, y, z = v.unbind(-1)
        o = torch.zeros_like(x)
        k = torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(*v.shape[:-1], 3, 3)
        return a * k + b * (v[..., :, None] * v[..., None, :] - th2 * torch.eye(3, device=v.device, dtype=v.dtype))

    def forward(self, hand_feat, shape_feat, shape_presence, R, L, betas):
        t = hand_feat.shape[0]
        e = self.exp_minus_eye(self.expect(self.rot(hand_feat.float()).reshape(t, 2, 16, 3, self.bins).softmax(-1), self.rot_span))
        logp = self.beta(shape_feat.float()).reshape(t, 2, 10, self.bins).float().log_softmax(-1)
        w = shape_presence.float()
        post = (logp * w[..., None, None]).sum(0) / w.sum(0).clamp_min(1e-6)[..., None, None]
        return R + R @ e[:, :, 0], L + L @ e[:, :, 1:], betas + self.expect(post.softmax(-1), self.beta_span)


class AmodalHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, joint_feat, base):
        return base.float() + self.mlp(self.norm(joint_feat.float()))


class ReadEvidence(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.norm, self.lin = nn.LayerNorm(hidden), nn.Linear(hidden, 1)
        self.a, self.b = nn.Parameter(torch.empty(4)), nn.Parameter(torch.empty(()))

    def forward(self, joint_feat, stats):
        return (stats.float() * self.a).sum(-1) + self.b + self.lin(self.norm(joint_feat.float()))[..., 0]


POOLS = FINGERS + (PALM,)
PARTS = dict(t=((5,), 1.0), qt=((4,), 2.0), st=((4,), 1.0), q=((0, 1, 2, 3), 2.0), s=((0, 1, 2, 3), 1.0))


class Parts(nn.Module):

    def __init__(self, hidden, cues=16, width=64):
        super().__init__()
        self.hidden = hidden
        self.head = nn.ModuleDict({n: nn.Sequential(nn.Linear(hidden * (1 if n == "t" else 2) + cues, width), nn.GELU(), nn.Linear(width, 1)) for n in PARTS})
        self.a = nn.ParameterDict({n: nn.Parameter(torch.empty(())) for n in PARTS})

    def forward(self, part_in, geo):
        pp = part_in[:, :6 * self.hidden].reshape(len(part_in), 6, self.hidden).float()
        lp, pres = part_in[:, 6 * self.hidden:6 * self.hidden + 6].float(), part_in[:, -1].float()
        sc = torch.cat([geo.float(), pres[:, None], lp], -1)
        out = {}
        for n, (groups, cap) in PARTS.items():
            x = lambda g: torch.cat([pp[:, g]] + ([] if n == "t" else [pp[:, 5]]) + [sc], -1)
            v = cap * torch.sigmoid(torch.stack([self.head[n](x(g))[..., 0] + self.a[n] * lp[:, g] for g in groups], -1))
            out[n] = v if len(groups) > 1 else v[..., 0]
        return out


class Gate(nn.Module):
    def __init__(self, hidden, clamp):
        super().__init__()
        self.clamp = clamp
        self.norm, self.lin = nn.LayerNorm(hidden), nn.Linear(hidden, 1)
        self.a, self.b = nn.Parameter(torch.empty(())), nn.Parameter(torch.empty(()))
        self.parts = Parts(hidden)

    def forward(self, joint_feat, e, pres):
        e = e.float().clamp(self.clamp, 1 - self.clamp)
        nf, le = self.norm(joint_feat.float()), torch.log(e / (1 - e))
        pp = torch.stack([nf[..., list(j), :].mean(-2) for j in POOLS], -2)
        lp = torch.stack([le[..., list(j)].mean(-1) for j in POOLS], -1)
        return torch.sigmoid(self.a * le + self.b + self.lin(nf)[..., 0]), torch.cat([pp.flatten(-2), lp, pres.float()[..., None]], -1)


class MotionPrecision(nn.Module):
    def __init__(self, cues, hidden, acceleration):
        super().__init__()
        self.inp = nn.Linear(cues, hidden)
        self.out = nn.Linear(hidden, 2, bias=False)
        self.register_buffer("b", torch.tensor([math.log(a) for a in acceleration]), persistent=False)

    def forward(self, phi):
        f = lambda x: self.out(F.gelu(self.inp(x.to(self.inp.weight.dtype))))
        still = phi.new_zeros(phi.shape[-1])
        still[4] = still[5] = still[6] = 1.0
        h = f(phi)[..., 0] - f(still)[..., 0]
        return torch.stack([self.b[0] + h, self.b[1].expand_as(h)], -1)


class ReadPrecision(nn.Module):

    def __init__(self, cues, hidden, envelope):
        super().__init__()
        self.envelope = envelope
        self.inp = nn.Linear(cues, hidden)
        self.out = nn.Linear(hidden, 1)
        self.kappa = nn.Parameter(torch.empty(20))

    def forward(self, psi):
        raw = self.out(F.gelu(self.inp(psi.to(self.inp.weight.dtype))))[..., 0]
        return self.envelope * torch.tanh(raw / self.envelope)

    def sigma(self):
        k = self.kappa.double()
        return torch.cat([torch.zeros(1, dtype=k.dtype, device=k.device), k.exp()])
