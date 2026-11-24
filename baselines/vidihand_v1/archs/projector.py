"""Decoder from DiT tokens of one window to per-frame MANO of a left (slot 0) and a right (slot 1) hand."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SPATIAL_TOKENS = 2048
N_J = 21
N_SLOTS = 2


def _decoder_layer(hidden, heads=8):
    return nn.ModuleDict({
        "self_attn": nn.MultiheadAttention(hidden, heads, batch_first=True), "self_attn_norm": nn.LayerNorm(hidden),
        "cross_attn": nn.MultiheadAttention(hidden, heads, batch_first=True), "cross_attn_norm": nn.LayerNorm(hidden),
        "ffn": nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden)),
        "ffn_norm": nn.LayerNorm(hidden),
    })


def _mlp(hidden, out, width=256):
    return nn.Sequential(nn.Linear(hidden, width), nn.GELU(), nn.Linear(width, out))


OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]  # MANO16 + tips -> OpenPose


def _run_layer_weighted(layer, q, kv):
    n = layer["self_attn_norm"](q)
    q = q + layer["self_attn"](n, n, n)[0]
    n = layer["cross_attn_norm"](q)
    q = q + layer["cross_attn"](n, kv, kv)[0]
    return q + layer["ffn"](layer["ffn_norm"](q))


def _rot6d(x):
    x = x.float().reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    b1 = F.normalize(x[:, :, 0])
    b2 = F.normalize(x[:, :, 1] - torch.einsum("bi,bi->b", b1, x[:, :, 1]).unsqueeze(-1) * b1)
    return torch.stack((b1, b2, torch.linalg.cross(b1, b2)), dim=-1)


def _interp(x, n):
    """[F, *shape] -> [n, *shape], linear in time."""
    shape = x.shape[1:]
    t = x.reshape(1, x.shape[0], -1).permute(0, 2, 1).contiguous()
    return F.interpolate(t, size=n, mode="linear", align_corners=True).permute(0, 2, 1).reshape(n, *shape)


class ControlProjector(nn.Module):
    def __init__(self, mano, in_dim=1536, hidden=512, shared_layers=4, cls_layers=1, reg_layers=2, heads=8, margin=0.15):
        super().__init__()
        self.mano = mano  # utils.mano.ManoModel, kept outside the state dict
        self.margin = margin
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden))
        self.spatial_pe = nn.Parameter(torch.zeros(1, MAX_SPATIAL_TOKENS, hidden))
        self.temporal_pe = nn.Parameter(torch.zeros(128, 1, hidden))
        self.ray_pe = nn.Sequential(nn.Linear(32, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.register_buffer("freq", 2.0 ** torch.arange(8).float(), persistent=False)
        self.hand_queries = nn.Parameter(torch.zeros(N_SLOTS, hidden))
        self.slot_embed = nn.Parameter(torch.zeros(N_SLOTS, hidden))
        self.shared = nn.ModuleList(_decoder_layer(hidden, heads) for _ in range(shared_layers))
        self.cls = nn.ModuleList(_decoder_layer(hidden, heads) for _ in range(cls_layers))
        self.reg = nn.ModuleList(_decoder_layer(hidden, heads) for _ in range(reg_layers))
        self.heatmap = nn.Conv2d(hidden, N_SLOTS * N_J, 1)
        self.xattn_h2j = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.xattn_j2h = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm_h, self.norm_j = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.joint_offset = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 2))
        self.temporal_refine = nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(), nn.Conv1d(hidden, hidden, 5, padding=2))
        self.temporal_mixer = nn.Sequential(nn.Conv1d(hidden, hidden, 7, padding=3, groups=hidden), nn.GELU(), nn.Conv1d(hidden, hidden, 1))
        self.head_go, self.head_hp, self.head_betas = _mlp(hidden, 6), _mlp(hidden, 15 * 6), _mlp(hidden, 10)
        self.head_logz = _mlp(hidden, 1)
        self.head_exists = _mlp(hidden, 1, width=128)

    def ray(self, h, w, k, img_w, img_h):
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        i, j = torch.meshgrid(torch.arange(h, device=self.freq.device, dtype=torch.float32),
                              torch.arange(w, device=self.freq.device, dtype=torch.float32), indexing="ij")
        az = torch.atan(((j + 0.5) * img_w / w - cx) / fx).reshape(-1, 1)
        el = torch.atan(((i + 0.5) * img_h / h - cy) / fy).reshape(-1, 1)
        f = self.freq.unsqueeze(0)
        return self.ray_pe(torch.cat([torch.sin(az * f), torch.cos(az * f), torch.sin(el * f), torch.cos(el * f)], -1))

    def soft_argmax(self, logits):
        *lead, h, w = logits.shape
        p = F.softmax(logits.reshape(-1, h * w), dim=-1)
        scale, m = 1.0 + 2.0 * self.margin, self.margin
        u = ((torch.arange(w, device=p.device, dtype=p.dtype) + 0.5) / w) * scale - m
        v = ((torch.arange(h, device=p.device, dtype=p.dtype) + 0.5) / h) * scale - m
        pu = (p * u.unsqueeze(0).expand(h, -1).reshape(-1)).sum(dim=-1)
        pv = (p * v.unsqueeze(1).expand(-1, w).reshape(-1)).sum(dim=-1)
        return torch.stack([pu, pv], dim=-1).view(*lead, 2)

    def pnp(self, uv, can, log_z, k, img_w, img_h, valid):
        """tz = exp(log z); (tx, ty) = pixel least squares of root-frame joints ``can`` onto ``uv`` (one frame)."""
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        tz = log_z.exp()
        z = (can[..., 2] + tz.unsqueeze(-1)).clamp(min=0.05)
        k_u, k_v = fx / z, fy / z
        a_u = (uv[..., 0] * img_w - cx) - fx * can[..., 0] / z
        a_v = (uv[..., 1] * img_h - cy) - fy * can[..., 1] / z
        m = valid.to(can.dtype) * ((can[..., 2] + tz.unsqueeze(-1)) > 0.05).to(can.dtype)
        tx = (k_u * a_u * m).sum(dim=-1) / (k_u * k_u * m).sum(dim=-1).clamp_min(1e-8)
        ty = (k_v * a_v * m).sum(dim=-1) / (k_v * k_v * m).sum(dim=-1).clamp_min(1e-8)
        empty = m.sum(dim=-1) < 1.0
        return torch.stack([torch.where(empty, torch.zeros_like(tx), tx), torch.where(empty, torch.zeros_like(ty), ty), tz], dim=-1)

    def forward(self, feat, K, canvas_wh):
        """feat [F_lat, H, W, C], K [n, 3, 3] per video frame -> go [n, 2, 3, 3], hp [n, 2, 15, 3, 3],
        betas [n, 2, 10], cam_trans [n, 2, 3] camera m, presence [n, 2]."""
        f_lat, h, w, c = feat.shape
        n, hw = len(K), h * w
        img_w, img_h = float(canvas_wh[0]), float(canvas_wh[1])
        if hw > MAX_SPATIAL_TOKENS or f_lat > self.temporal_pe.shape[0]:
            raise ValueError(f"vidihand_v1 token grid {f_lat}x{h}x{w} exceeds the decoder's position tables")
        x = self.input_proj(feat.reshape(f_lat, hw, c).float())
        x = x + self.spatial_pe[:, :hw]
        for f in range(f_lat):
            x[f] += self.ray(h, w, K[min(f * n // f_lat, n - 1)], img_w, img_h)
        x = x + self.temporal_pe[:f_lat]
        logits = self.heatmap(x.transpose(1, 2).reshape(f_lat, -1, h, w)).view(f_lat, N_SLOTS, N_J, h, w)
        uv = self.soft_argmax(logits)
        q = (self.hand_queries + self.slot_embed).unsqueeze(0).expand(f_lat, N_SLOTS, -1).contiguous()
        for layer in self.shared:
            q = _run_layer_weighted(layer, q, x)
        qj = torch.einsum("nkw,nwc->nkc", F.softmax(logits.reshape(f_lat, N_SLOTS * N_J, hw).to(q.dtype), dim=-1), x)
        qh = self.norm_h(q + self.xattn_h2j(q, qj, qj)[0])
        qj = self.norm_j(qj + self.xattn_j2h(qj, q, q)[0])
        cls, reg = qh, qh
        for layer in self.cls:
            cls = _run_layer_weighted(layer, cls, x)
        for layer in self.reg:
            reg = _run_layer_weighted(layer, reg, x)
        r = F.interpolate(reg.permute(1, 2, 0).contiguous(), size=n, mode="linear", align_corners=True)
        r = r + self.temporal_refine(r)
        r = r + self.temporal_mixer(r)
        reg = r.permute(2, 0, 1).contiguous()
        cls = F.interpolate(cls.permute(1, 2, 0).contiguous(), size=n, mode="linear", align_corners=True).permute(2, 0, 1).contiguous()
        go = _rot6d(self.head_go(reg).reshape(-1, 6)).reshape(n, N_SLOTS, 3, 3)
        hp = _rot6d(self.head_hp(reg).reshape(-1, 6)).reshape(n, N_SLOTS, 15, 3, 3)
        betas = self.head_betas(reg)
        log_z = torch.exp(self.head_logz(reg))[..., 0].clamp(min=1e-6).log()
        presence = torch.sigmoid(self.head_exists(cls).squeeze(-1))
        uv = _interp(uv, n) + _interp(self.joint_offset(qj).view(f_lat, N_SLOTS, N_J, 2), n)
        rotations = torch.cat((go[:, :, None], hp), 2).cpu().numpy()
        zero = np.zeros((n, 3), np.float32)
        can = torch.as_tensor(np.stack([self.mano.forward(rotations[:, s], betas[:, s].cpu().numpy(), zero, s)[1][:, OPENPOSE]
                                        for s in range(N_SLOTS)], 1), device=feat.device).contiguous()
        m = self.margin
        inside = (uv[..., 0] >= -m) & (uv[..., 0] <= 1.0 + m) & (uv[..., 1] >= -m) & (uv[..., 1] <= 1.0 + m)
        cam_trans = torch.stack([self.pnp(uv[i], can[i], log_z[i], K[i], img_w, img_h, inside[i]) for i in range(n)])
        return dict(go=go, hp=hp, betas=betas, cam_trans=cam_trans, presence=presence)
