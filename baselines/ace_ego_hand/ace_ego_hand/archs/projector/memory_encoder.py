"""VGGT-style register-token alternating spatial-temporal encoder.

The default projector pools the H*W spatial grid into 1-2 tokens **before**
the temporal encoder, so every 2D/3D head regresses 21 joints from a single
pooled vector and the bidirectional temporal attention only sees that summary.

This module ports VGGT-Omega's register/camera-token + alternating-attention
idea (see the aggregator module of the public VGGT-Omega release): a small
set of per-frame learnable tokens iteratively

  1. CROSS-ATTEND the full pre-pool patch grid x (spatial read, within frame), and
  2. SELF-ATTEND across all frames (temporal exchange, bidirectional),

interleaved for ``num_layers`` rounds. Spatial detail therefore stays accessible
*inside* the temporal pathway (the tokens re-read the grid after every temporal
refinement) instead of being pooled away once up front. The patch grid is used as
a fixed read-only memory (Perceiver/DETR-decoder style) so cost is
O(n_special * H*W) per frame, not O((H*W)^2) — cheap enough to keep batch 16.

Two query granularities (``query_mode``), both config-gated at the call site
(``temporal_arch='alternating'``); the default ``'pooled'`` model never
instantiates this:

* ``'hand'``: one hand token per slot + register tokens. The hand-pose
  heads read the hand token (like VGGT's CameraHead reads the camera token).
* ``'joint'``: additionally one query token per (slot, joint). After the
  alternating stack the **temporally-refined** joint tokens do a final cross-attn
  to the grid; the attention map IS a per-joint heatmap -> soft-argmax 2D, and the
  attended feature -> per-joint 3D rootrel. This is a spatially-grounded
  readout whose queries are refined by the temporal stack instead of being
  static and bypassing it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ── RoPE on the temporal (frame) axis (GENMO-style) ──
# Relative temporal position via rotary embedding instead of a learned absolute PE.
# Benefits: no length cap (encoding regenerated for any F_lat), relative encoding
# (generalizes across windows; smooths boundaries when >1 window is in one forward,
# which sliding-window inference exploits). Ported from GENMO's rotary_embedding.py.
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)


def _rope_encoding(head_dim: int, n_pos: int, device, dtype) -> torch.Tensor:
    """(n_pos, head_dim) interleaved sin/cos frequencies (base 10000)."""
    t = torch.arange(n_pos, device=device, dtype=torch.float32)
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    f = torch.einsum("i,j->ij", t, freqs)                 # (n_pos, head_dim/2)
    return torch.repeat_interleave(f, 2, dim=-1).to(dtype)  # (n_pos, head_dim)


def _apply_rope(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Rotate t=(B,H,L,hd) by per-position freqs=(L,hd)."""
    cos, sin = freqs.cos(), freqs.sin()
    return t * cos + _rotate_half(t) * sin


class _RoPETemporalLayer(nn.Module):
    """Pre-LN transformer layer with RoPE on the FRAME axis. Drop-in for the
    temporal block: sequence is (B, F_lat*n_special, D) interleaved [frame, token];
    RoPE positions = frame index (so attention is relative in frame distance;
    within-frame siblings get relative-0). Full bidirectional (no mask)."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        self.nh = num_heads
        self.hd = hidden_dim // num_heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        h = int(hidden_dim * ffn_mult)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, h), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(h, hidden_dim))

    def forward(self, sp: torch.Tensor, F_lat: int, n_special: int, mask=None) -> torch.Tensor:
        B, L, D = sp.shape  # L == F_lat * n_special
        h = self.norm1(sp)
        q = self.q(h).view(B, L, self.nh, self.hd).transpose(1, 2)
        k = self.k(h).view(B, L, self.nh, self.hd).transpose(1, 2)
        v = self.v(h).view(B, L, self.nh, self.hd).transpose(1, 2)
        frame_pos = torch.arange(F_lat, device=sp.device).repeat_interleave(n_special)  # (L,)
        freqs = _rope_encoding(self.hd, F_lat, sp.device, sp.dtype)[frame_pos]  # (L, hd)
        q = _apply_rope(freqs, q)
        k = _apply_rope(freqs, k)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
        if mask is not None:
            # temporal-scope hierarchy: (L,L) bool, True=blocked, broadcast over B,nh.
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        sp = sp + self.drop(self.proj(out))
        sp = sp + self.mlp(self.norm2(sp))
        return sp


class _AltLayer(nn.Module):
    """One alternating round: cross-attn(special -> patches) then temporal self-attn."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_mult: int, dropout: float,
                 temporal_rope: bool = False) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_rope = bool(temporal_rope)
        # Temporal (and within-frame sibling) exchange over the special tokens.
        if self.temporal_rope:
            # RoPE-based temporal attention (relative frame position, no cap).
            self.temporal = _RoPETemporalLayer(hidden_dim, num_heads, ffn_mult, dropout)
        else:
            # A standard pre-LN encoder layer brings its own residual + FFN + norms.
            self.temporal = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * int(ffn_mult),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

    def forward(
        self,
        special: torch.Tensor,  # (B*F_lat, n_special, D)
        patches: torch.Tensor,  # (B*F_lat, H*W, D), fixed read-only memory
        *,
        B: int,
        F_lat: int,
        temporal_mask=None,
    ) -> torch.Tensor:
        n_special = special.shape[1]
        D = special.shape[-1]
        # 1) spatial read: each frame's special tokens cross-attend its patch grid.
        q = self.norm_q(special)
        kv = self.norm_kv(patches)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        special = special + attn_out
        # 2) temporal + same-frame-sibling exchange across the whole clip
        #    (bidirectional: every special token sees every other, all frames).
        sp = special.view(B, F_lat, n_special, D).reshape(B, F_lat * n_special, D)
        if self.temporal_rope:
            sp = self.temporal(sp, F_lat, n_special, mask=temporal_mask)
        else:
            sp = self.temporal(sp)  # non-RoPE path has no windowed-mask support
        special = sp.reshape(B, F_lat, n_special, D).reshape(B * F_lat, n_special, D)
        return special


# MANO 16-joint kinematic tree (0 = wrist/global, then 5 fingers x 3 joints each,
# grouped as consecutive 3-chains rooted at the wrist). The exact finger identity
# doesn't matter for the mask/lever-weights — only the chain grouping does.
MANO_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)


def mano_kinematic_mask(num_bones: int = 16) -> torch.Tensor:
    """Bool attention mask (num_bones, num_bones), True = BLOCKED. A bone may attend
    itself, its parent, its children, and its siblings (same parent) — a subject-
    invariant skeletal prior so a confident wrist/MCP constrains ambiguous fingers."""
    p = MANO_PARENTS[:num_bones]
    allowed = torch.zeros(num_bones, num_bones, dtype=torch.bool)
    for i in range(num_bones):
        allowed[i, i] = True
        if p[i] >= 0:
            allowed[i, p[i]] = True
            allowed[p[i], i] = True
    for i in range(num_bones):
        for j in range(num_bones):
            if p[i] != -1 and p[i] == p[j]:
                allowed[i, j] = True
    return ~allowed


class _PoseBoneDecoder(nn.Module):
    """Per-bone MANO rotation decoder (DETR-style per-joint pose
    head): ``num_bones`` learnable bone tokens per hand (bone 0 = global_orient,
    1..15 = hand_pose joints), conditioned on the hand context, refined by
    kinematic-masked self-attention among bones + cross-attention to that hand's
    spatially-grounded per-joint tokens. Each bone then reads ITS OWN rotation —
    replacing the single-pooled-vector pose MLP that capped PA-MPJPE / g_orient."""

    def __init__(
        self,
        hidden_dim: int,
        num_bones: int,
        num_layers: int,
        num_heads: int,
        ffn_mult: int,
        dropout: float,
        use_kin_mask: bool = True,
    ) -> None:
        super().__init__()
        self.num_bones = int(num_bones)
        self.bone_embed = nn.Parameter(torch.randn(1, num_bones, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "sa_norm": nn.LayerNorm(hidden_dim),
                    "sa": nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                    "ca_qnorm": nn.LayerNorm(hidden_dim),
                    "ca_kvnorm": nn.LayerNorm(hidden_dim),
                    "ca": nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                    "ffn_norm": nn.LayerNorm(hidden_dim),
                    "ffn": nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * int(ffn_mult)),
                        nn.GELU(),
                        nn.Linear(hidden_dim * int(ffn_mult), hidden_dim),
                    ),
                }
            )
            for _ in range(int(num_layers))
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        if use_kin_mask:
            self.register_buffer("kin_mask", mano_kinematic_mask(num_bones), persistent=False)
        else:
            self.kin_mask = None

    def forward(self, hand_ctx: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # hand_ctx: (N, 1, D) per-slot hand context; kv: (N, n_joint, D) per-slot joint tokens
        bone = self.bone_embed.expand(hand_ctx.shape[0], -1, -1) + hand_ctx
        for L in self.layers:
            q = L["sa_norm"](bone)
            sa, _ = L["sa"](q, q, q, attn_mask=self.kin_mask, need_weights=False)
            bone = bone + sa
            qc = L["ca_qnorm"](bone)
            kc = L["ca_kvnorm"](kv)
            ca, _ = L["ca"](qc, kc, kc, need_weights=False)
            bone = bone + ca
            bone = bone + L["ffn"](L["ffn_norm"](bone))
        return self.final_norm(bone)  # (N, num_bones, D)


class _MemGRU(nn.Module):
    """GRU-style gated update for the cross-window memory bank.

    ``m_out = (1-z)*n + z*m_in`` with the update gate ``z`` biased to KEEP the
    previous state, so a confidently-tracked memory persists across windows and
    only changes when the current window provides strong new evidence. Standard
    content-based gate.
    """

    def __init__(self, hidden_dim: int, keep_bias: float = 1.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_rz = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.to_n = nn.Linear(2 * hidden_dim, hidden_dim)
        self.keep_bias = float(keep_bias)

    def forward(self, m_post: torch.Tensor, m_in: torch.Tensor) -> torch.Tensor:
        h = self.norm(m_post)
        r, z = self.to_rz(torch.cat([h, m_in], dim=-1)).chunk(2, dim=-1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z + self.keep_bias)  # bias toward keeping m_in
        n = torch.tanh(self.to_n(torch.cat([h, r * m_in], dim=-1)))
        return (1.0 - z) * n + z * m_in


class _HeatmapManoHead(nn.Module):
    """Heatmap-grounded MANO hand feature. A learned heatmap
    (Linear == 1x1 conv) pools per-(slot,joint) features from the RAW spatial grid;
    the hand token then cross-attends those per-joint features. This gives the MANO
    pose heads raw-grid evidence instead of a lossy joint-token/pooled summary,
    which measurably improves global orientation. Zero-init out_proj => step-0
    no-op residual."""

    def __init__(self, hidden_dim, num_slots, num_joints, num_heads, dropout):
        super().__init__()
        self.num_slots = int(num_slots)
        self.num_joints = int(num_joints)
        self.heatmap = nn.Linear(hidden_dim, self.num_slots * self.num_joints)
        self.xattn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        nn.init.zeros_(self.xattn.out_proj.weight)
        nn.init.zeros_(self.xattn.out_proj.bias)  # step-0 no-op
        self.norm_h = nn.LayerNorm(hidden_dim)

    def forward(self, x, hand, B, F_lat):
        # x: (BF, HW, D) raw grid; hand: (BF, K, D) hand tokens
        BF, _, D = x.shape
        K, J = self.num_slots, self.num_joints
        logits = self.heatmap(x).transpose(1, 2)           # (BF, K*J, HW)
        attn_w = torch.softmax(logits, dim=-1)
        q_joint = torch.einsum("nkw,nwc->nkc", attn_w, x)  # (BF, K*J, D) pooled from RAW grid
        q_joint = q_joint.reshape(BF, K, J, D).reshape(BF * K, J, D)
        h = hand.reshape(BF * K, 1, D)
        attn_out, _ = self.xattn(h, q_joint, q_joint, need_weights=False)
        fused = self.norm_h(h + attn_out).reshape(BF, K, D)
        return fused.reshape(B, F_lat, K, D)


class MemoryAlternatingEncoder(nn.Module):
    """Register-token alternating spatial-temporal encoder + cross-window memory.

    An alternating spatial/temporal attention encoder PLUS a persistent
    memory bank of ``mem_tokens`` special tokens that is carried ACROSS 81-frame
    windows (``mem_in`` -> ``mem_out``). The memory tokens are appended AFTER the
    register block, get a frame-shared type embed + per-hand slot identity (NO
    latent temporal PE — they are window-global), join every layer's spatial
    cross-attn + bidirectional temporal self-attn, and are pooled over frames +
    GRU-gated into ``mem_out`` after the stack. ``mem_tokens=0`` => identical to
    the base encoder (state_dict + numerics) so memory-free checkpoints strict-load.

    Args:
        hidden_dim: token dimension (matches the projector's ``hidden_dim``).
        num_slots: number of hand tokens read out by the pose heads (= MAX_HANDS).
        num_register: extra learnable scratch tokens (VGGT registers).
        num_layers: alternating rounds.
        num_heads / ffn_mult / dropout: attention hyper-params.
        query_mode: ``'hand'`` or ``'joint'`` (adds per-joint queries
            + soft-argmax 2D + per-joint 3D readout).
        num_joints: joints per hand (only used when ``query_mode='joint'``).

    Returns from ``forward``: ``(hand_lat, shape_token, joint_2d, joint_rootrel)``.
        ``hand_lat``      : ``(B, F_lat, num_slots, D)`` per-hand features.
        ``shape_token``   : ``(B, D)`` clip-level shape summary (register mean).
        ``joint_2d``      : ``(B, F_lat, num_slots, num_joints, 2)`` in [0,1], or None.
        ``joint_rootrel`` : ``(B, F_lat, num_slots, num_joints, 3)``, or None.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_slots: int,
        num_register: int = 4,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        query_mode: str = "hand",
        num_joints: int = 21,
        produce_mano_feat: bool = False,
        produce_bone_feat: bool = False,
        produce_heatmap_mano: bool = False,
        num_bones: int = 16,
        bone_num_layers: int = 2,
        bone_use_kin_mask: bool = True,
        mem_tokens: int = 0,
        mem_states_per_hand: int = 4,
        mem_update_kind: str = "gru",
        temporal_rope: bool = False,
        camera_tokens: int = 0,
        temporal_window: int = 0,
        hand_mult: int = 1,
    ) -> None:
        super().__init__()
        if query_mode not in ("hand", "joint"):
            raise ValueError(f"query_mode must be 'hand' or 'joint', got {query_mode!r}")
        # RoPE temporal axis (relative, no length cap) — replaces the learned
        # latent_temporal_pe add. Config-gated; off => identical to base.
        self.temporal_rope = bool(temporal_rope)
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.num_register = int(num_register)
        self.query_mode = query_mode
        self.num_joints = int(num_joints)
        # Cross-window memory. mem_tokens=0 => no params, identical base.
        self.mem_tokens = int(mem_tokens)
        self.mem_states_per_hand = int(mem_states_per_hand)
        # MANO-grounding modules only exist when explicitly requested, so the
        # default joint-mode state_dict does not contain them.
        self.produce_mano_feat = bool(produce_mano_feat) and query_mode == "joint"
        self.produce_bone_feat = bool(produce_bone_feat) and query_mode == "joint"
        self.produce_heatmap_mano = bool(produce_heatmap_mano) and query_mode == "joint"
        self.num_bones = int(num_bones)
        self.n_joint = self.num_slots * self.num_joints if query_mode == "joint" else 0
        # hand_mult: K query tokens per hand slot (capacity to decode go/hp/trans;
        # pooled to ONE per-hand feature for the heads — NOT per-bone/1-per-output).
        # n_hand = num_slots*K. hand_mult=1 => byte-identical to the base encoder.
        self.hand_mult = int(hand_mult)
        self.n_hand = self.num_slots * self.hand_mult
        # Camera-decoupled orientation: per-frame camera token(s), slot-agnostic
        # (one shared camera). n_camera=0 => no params, byte-identical to the base.
        self.n_camera = int(camera_tokens)
        # core (non-mem) special tokens; memory rows are appended after these.
        # Order: [hand(n_hand) | joint | camera | register] — camera BEFORE register
        # so the bounded register slice is unchanged when n_camera=0 (ckpt-compat).
        self.n_core = self.n_hand + self.n_joint + self.n_camera + self.num_register
        self.n_special = self.n_core + self.mem_tokens
        # Temporal-scope hierarchy: FORM tokens (hand+joint) are
        # per-frame → attend only a local ±window in time; the tokens AFTER them
        # (camera/register/mem = "global carriers") attend the whole clip. Global
        # context reaches form tokens via same-frame attention to those carriers.
        # temporal_window=0 => no mask (byte-identical full-clip attention).
        self.temporal_window = int(temporal_window)
        self.n_form = self.n_hand + self.n_joint

        self.hand_token = nn.Parameter(torch.randn(1, self.n_hand, hidden_dim) * 0.02)
        self.register_token = nn.Parameter(torch.randn(1, self.num_register, hidden_dim) * 0.02)
        if self.n_camera > 0:
            self.camera_token = nn.Parameter(torch.randn(1, self.n_camera, hidden_dim) * 0.02)
        if self.mem_tokens > 0:
            if mem_update_kind != "gru":
                raise ValueError(f"mem_update_kind must be 'gru' (got {mem_update_kind!r})")
            self.mem_init = nn.Parameter(torch.randn(1, self.mem_tokens, hidden_dim) * 0.02)
            self.mem_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.mem_update = _MemGRU(hidden_dim)
        if query_mode == "joint":
            self.joint_token = nn.Parameter(torch.randn(1, self.n_joint, hidden_dim) * 0.02)
            # Final spatially-grounded readout: the temporally-refined joint tokens
            # cross-attend the grid -> heatmap (soft-argmax 2D) + feature (3D).
            self.readout_qnorm = nn.LayerNorm(hidden_dim)
            self.readout_kvnorm = nn.LayerNorm(hidden_dim)
            self.readout_attn = nn.MultiheadAttention(
                hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            head_hidden = max(128, hidden_dim // 2)
            self.head_joint_3d = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 3),
            )
            # Per-hand MANO feature: a learned query pools that hand's per-joint
            # tokens, giving the MANO global_orient/hand_pose heads access to the
            # same spatially-grounded per-joint detail (instead of the single
            # pooled hand token). Fixes the MANO-head pooling bottleneck (the
            # reason PA-MPJPE / g_orient lag behind).
            # Gated so default joint-mode checkpoints are unchanged.
            if self.produce_mano_feat:
                self.mano_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
                self.mano_pool_attn = nn.MultiheadAttention(
                    hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
                )
                self.mano_qnorm = nn.LayerNorm(hidden_dim)
                self.mano_kvnorm = nn.LayerNorm(hidden_dim)
                self.mano_norm = nn.LayerNorm(hidden_dim)
            # Per-bone MANO rotation decoder (gated; absent in default joint
            # mode so existing checkpoints are unchanged).
            if self.produce_bone_feat:
                self.bone_decoder = _PoseBoneDecoder(
                    hidden_dim,
                    num_bones=self.num_bones,
                    num_layers=int(bone_num_layers),
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    use_kin_mask=bool(bone_use_kin_mask),
                )
            # Heatmap-grounded MANO head (pools the RAW grid, not joint tokens).
            if self.produce_heatmap_mano:
                self.heatmap_mano = _HeatmapManoHead(
                    hidden_dim,
                    num_slots=self.num_slots,
                    num_joints=self.num_joints,
                    num_heads=num_heads,
                    dropout=dropout,
                )

        self.layers = nn.ModuleList(
            _AltLayer(hidden_dim, num_heads, ffn_mult, dropout, temporal_rope=self.temporal_rope)
            for _ in range(int(num_layers))
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.shape_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,  # (B*F_lat, H*W, D), patch tokens (input_proj + spatial_pe applied)
        *,
        B: int,
        F_lat: int,
        H: int,
        W: int,
        slot_embed: torch.Tensor,  # (num_slots, D) hand identity, shared with the projector
        latent_temporal_pe: torch.Tensor,  # (1, F_lat, D) latent-frame positional encoding
        mem_in: torch.Tensor | None = None,  # (B, mem_tokens, D) carried memory, or None
    ):
        D = self.hidden_dim
        BF = B * F_lat

        # Build per-frame CORE special tokens: [hand(n_hand=K*slots), (joint), (camera), register].
        # slot_embed applies per hand slot, repeated over that slot's K tokens.
        hand = self.hand_token + slot_embed.repeat_interleave(self.hand_mult, dim=0).view(1, self.n_hand, D)
        hand = hand.expand(BF, -1, -1)
        parts = [hand]
        if self.query_mode == "joint":
            parts.append(self.joint_token.expand(BF, -1, -1))
        if self.n_camera > 0:
            parts.append(self.camera_token.expand(BF, -1, -1))
        parts.append(self.register_token.expand(BF, -1, -1))
        core = torch.cat(parts, dim=1)  # (BF, n_core, D)

        core = core.view(B, F_lat, self.n_core, D)
        if not self.temporal_rope:
            # learned absolute latent temporal PE so the temporal self-attn knows
            # frame order (CORE tokens only). Skipped under RoPE — it encodes frame
            # position relatively inside the temporal attention instead.
            core = core + latent_temporal_pe[:, :F_lat, :].reshape(1, F_lat, 1, D)

        mem_in_eff = None
        if self.mem_tokens > 0:
            # carried memory (or learned init) -> per-frame-shared memory rows:
            # type embed + per-hand slot identity, NO temporal PE (window-global).
            mem_in_eff = self.mem_init.expand(B, -1, -1) if mem_in is None else mem_in
            mem_tok = mem_in_eff + self.mem_type_embed  # (B, mem_tokens, D)
            sph = self.mem_states_per_hand
            ids = []
            for t in range(self.mem_tokens):
                if sph > 0 and t < sph:
                    ids.append(slot_embed[0])
                elif sph > 0 and t < 2 * sph:
                    ids.append(slot_embed[1])
                else:
                    ids.append(slot_embed.mean(0))
            mem_tok = mem_tok + torch.stack(ids, dim=0).view(1, self.mem_tokens, D)
            mem_block = mem_tok.unsqueeze(1).expand(B, F_lat, self.mem_tokens, D)
            special = torch.cat([core, mem_block], dim=2)  # (B, F_lat, n_special, D)
        else:
            special = core
        special = special.reshape(BF, self.n_special, D)

        # temporal-scope hierarchy mask (built once, reused across the 4 rounds).
        temporal_mask = self._build_temporal_mask(F_lat, x.device)

        for layer in self.layers:
            special = layer(special, x, B=B, F_lat=F_lat, temporal_mask=temporal_mask)

        special = self.final_norm(special)
        special = special.view(B, F_lat, self.n_special, D)
        # hand block = n_hand = num_slots*K tokens → pool the K per slot to one
        # per-hand feature for the heads (the capacity boost, pooled not per-output).
        hand_lat = special[:, :, : self.n_hand, :].view(
            B, F_lat, self.num_slots, self.hand_mult, D
        ).mean(dim=3)  # (B, F_lat, num_slots, D)
        # Camera block sits between joint and register (cam_lat=None when off).
        cam_start = self.n_hand + self.n_joint
        cam_lat = (special[:, :, cam_start : cam_start + self.n_camera, :]
                   if self.n_camera > 0 else None)
        # register block ONLY (bounded so camera + appended memory rows are excluded;
        # numerically identical to the old slice when n_camera=mem_tokens=0).
        reg_lat = special[:, :, cam_start + self.n_camera : self.n_core, :]
        shape_token = self.shape_norm(reg_lat.mean(dim=(1, 2)))  # (B, D)

        joint_2d = joint_rootrel = mano_feat = bone_feat = None
        if self.query_mode == "joint":
            jt = special[:, :, self.n_hand : self.n_hand + self.n_joint, :]
            jt = jt.reshape(BF, self.n_joint, D)  # temporally-refined joint queries
            q = self.readout_qnorm(jt)
            kv = self.readout_kvnorm(x)
            feat, attn_w = self.readout_attn(
                q, kv, kv, need_weights=True, average_attn_weights=True
            )  # attn_w: (BF, n_joint, H*W) softmax over grid -> per-joint heatmap
            joint_2d = self._soft_argmax(attn_w, H, W)  # (BF, n_joint, 2) in [0,1]
            rootrel = self.head_joint_3d(feat)  # (BF, n_joint, 3)
            joint_2d = joint_2d.reshape(B, F_lat, self.num_slots, self.num_joints, 2)
            rootrel = rootrel.reshape(B, F_lat, self.num_slots, self.num_joints, 3)
            rootrel = rootrel - rootrel[:, :, :, 0:1, :]  # root-relative (joint 0)
            joint_rootrel = rootrel

            # Per-hand MANO feature: learned query pools each hand's joint tokens.
            if self.produce_mano_feat:
                jt_h = jt.reshape(BF * self.num_slots, self.num_joints, D)
                mq = self.mano_query.expand(BF * self.num_slots, 1, D)
                mf, _ = self.mano_pool_attn(
                    self.mano_qnorm(mq), self.mano_kvnorm(jt_h), self.mano_kvnorm(jt_h),
                    need_weights=False,
                )
                mano_feat = self.mano_norm(mf).reshape(B, F_lat, self.num_slots, D)

            # Heatmap-grounded MANO feature from the RAW grid (reads x, not
            # joint tokens). hand_lat = the special hand tokens.
            if self.produce_heatmap_mano:
                hand_bf = hand_lat.reshape(BF, self.num_slots, D)
                mano_feat = self.heatmap_mano(x, hand_bf, B, F_lat)

            # Per-bone rotation features. Each hand's bone tokens are
            # conditioned on the hand token and cross-attend that hand's per-joint
            # tokens (kv), refined by kinematic-masked self-attn among bones.
            if self.produce_bone_feat:
                hand_ctx = hand_lat.reshape(BF * self.num_slots, 1, D)
                kv = jt.reshape(BF, self.num_slots, self.num_joints, D).reshape(
                    BF * self.num_slots, self.num_joints, D
                )
                bone = self.bone_decoder(hand_ctx, kv)  # (BF*slots, num_bones, D)
                bone_feat = bone.reshape(B, F_lat, self.num_slots, self.num_bones, D)

        # Memory write/gate: pool the post-stack memory rows over frames, then a
        # GRU-gated update fuses them with the incoming memory -> mem_out.
        mem_out = None
        if self.mem_tokens > 0:
            mem_post = special[:, :, self.n_core :, :].mean(dim=1)  # (B, mem_tokens, D)
            mem_out = self.mem_update(mem_post, mem_in_eff)

        return hand_lat, shape_token, joint_2d, joint_rootrel, mano_feat, bone_feat, mem_out, cam_lat

    def _build_temporal_mask(self, F_lat: int, device):
        """(L,L) bool temporal-attention mask, True=BLOCKED. FORM tokens (hand+joint,
        index < n_form within each frame) attend only keys within ±temporal_window
        frames; "global carrier" tokens (camera/register/mem, index >= n_form) both
        attend and are attended across the WHOLE clip — so global context reaches the
        per-frame form tokens via the carriers, without smearing form across time.
        Returns None when disabled (window<=0), not RoPE, or F_lat<=1."""
        W = self.temporal_window
        if not self.temporal_rope or W <= 0 or F_lat <= 1:
            return None
        L = F_lat * self.n_special
        pos = torch.arange(L, device=device)
        frame = pos // self.n_special           # (L,) frame index of each position
        tok = pos % self.n_special              # (L,) token index within its frame
        is_global = tok >= self.n_form          # (L,) camera/register/mem carriers
        fq = frame.unsqueeze(1); fk = frame.unsqueeze(0)
        gq = is_global.unsqueeze(1); gk = is_global.unsqueeze(0)
        allow = gk | gq | ((fq - fk).abs() <= W)   # (L,L)
        return ~allow

    @staticmethod
    def _soft_argmax(attn_w: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """attn_w (BF, n_q, H*W) softmax over the grid -> (BF, n_q, 2) (u,v) in [0,1]."""
        BF, n_q, _ = attn_w.shape
        hm = attn_w.reshape(BF, n_q, H, W)
        us = torch.linspace(0.0, 1.0, W, device=attn_w.device, dtype=attn_w.dtype)
        vs = torch.linspace(0.0, 1.0, H, device=attn_w.device, dtype=attn_w.dtype)
        wsum = hm.sum(dim=(-1, -2)).clamp(min=1e-6)
        u = (hm.sum(dim=-2) * us).sum(dim=-1) / wsum
        v = (hm.sum(dim=-1) * vs).sum(dim=-1) / wsum
        return torch.stack([u, v], dim=-1)
