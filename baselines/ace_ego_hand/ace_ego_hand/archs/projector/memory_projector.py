"""Bidirectional temporal projector with segment-level MANO betas.

Consumes DiT features for a whole clip and returns per-frame bimanual MANO.
Two properties matter:

* a full bidirectional TransformerEncoder runs over the entire output segment;
* a CLS token predicts one shape vector (`betas`) for the whole 81-frame segment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_HANDS = 2


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrices."""
    orig_shape = x.shape[:-1]
    orig_dtype = x.dtype
    x = x.float().reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(
        a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1,
        dim=-1,
    )
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    rot = torch.stack((b1, b2, b3), dim=-1).to(orig_dtype)
    return rot.reshape(*orig_shape, 3, 3)


class MANOIKHead(nn.Module):
    """Small learnable inverse-kinematics head: predicted root-relative 3D joints
    -> MANO rot6d (global_orient + 15 finger-joint rotations).

    "Predict-joints-then-IK-to-MANO": instead of regressing MANO
    directly from pooled features, derive it FROM the predicted joints via a small
    cross-joint-attention module. This lets MANO-less (joints-only) datasets still
    supervise MANO through the shared joints->IK path.

    Input:  J of shape (M, 21, 3), root-relative (wrist-anchored) joints.
    Output: go_6d (M, 6), hp_6d (M, 15*6).

    The final linear of each output head is zero-init'd with an identity-rot6d bias,
    so at init the head emits ~identity rotations (a sane starting pose).
    """

    def __init__(
        self,
        num_joints: int = 21,
        hidden: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_joints = int(num_joints)
        self.hidden = int(hidden)
        self.joint_embed = nn.Linear(3, self.hidden)
        self.joint_pe = nn.Parameter(torch.zeros(self.num_joints, self.hidden))
        nn.init.normal_(self.joint_pe, std=0.02)
        # Learned CLS/pool token prepended to the 21 joint tokens.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden,
            nhead=int(num_heads),
            dim_feedforward=4 * self.hidden,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.head_go = nn.Linear(self.hidden, 6)
        self.head_hp = nn.Linear(self.hidden, 15 * 6)
        # Identity-rot6d init so the head starts from a sane identity pose.
        nn.init.zeros_(self.head_go.weight)
        self.head_go.bias.data = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        nn.init.zeros_(self.head_hp.weight)
        self.head_hp.bias.data = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ).repeat(15)

    def forward(self, joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # joints: (M, 21, 3) root-relative.
        M = joints.shape[0]
        tok = self.joint_embed(joints) + self.joint_pe.unsqueeze(0)  # (M, 21, H)
        cls = self.cls_token.expand(M, -1, -1)  # (M, 1, H)
        seq = torch.cat([cls, tok], dim=1)  # (M, 22, H)
        enc = self.encoder(seq)  # (M, 22, H)
        pooled = enc[:, 0, :]  # CLS -> (M, H)
        go_6d = self.head_go(pooled)  # (M, 6)
        hp_6d = self.head_hp(pooled)  # (M, 90)
        return go_6d, hp_6d


class RayPositionalEncoding(nn.Module):
    """Per-token ray-direction PE from camera intrinsics.

    Makes the spatial tokens intrinsics-aware so the orientation read-out is
    camera-conditioned. Plücker-style azimuth/elevation multi-freq sin/cos,
    projected to hidden_dim, zero-init last layer -> starts as a no-op residual."""

    def __init__(self, hidden_dim: int, n_freq_bands: int = 8) -> None:
        super().__init__()
        self.n_freq_bands = int(n_freq_bands)
        pe_dim = 4 * self.n_freq_bands
        self.ray_proj = nn.Sequential(
            nn.Linear(pe_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.ray_proj[2].weight)
        nn.init.zeros_(self.ray_proj[2].bias)
        self.register_buffer("freq_bands", 2.0 ** torch.arange(self.n_freq_bands).float())

    def forward(self, H_pat: int, W_pat: int, intr: dict, device, dtype,
                rays: torch.Tensor | None = None) -> torch.Tensor:
        if rays is not None:
            # Self-ray-PE: az/el straight from the PREDICTED ray field (3,H,W)
            # — no K, no camera model. This is the K-free ray-PE.
            dx, dy, dz = rays[0], rays[1], rays[2]
            azimuth = torch.atan2(dx, dz.abs().clamp_min(1e-6)).reshape(-1, 1)
            elevation = torch.atan2(dy, dz.abs().clamp_min(1e-6)).reshape(-1, 1)
            freq = self.freq_bands.unsqueeze(0)
            pe = torch.cat([torch.sin(azimuth * freq), torch.cos(azimuth * freq),
                            torch.sin(elevation * freq), torch.cos(elevation * freq)], dim=-1)
            return self.ray_proj(pe).unsqueeze(0).to(dtype)
        fx = float(intr["fx"]); fy = float(intr["fy"]); cx = float(intr["cx"]); cy = float(intr["cy"])
        img_w = float(intr["image_width"]); img_h = float(intr["image_height"])
        j = torch.arange(W_pat, device=device, dtype=torch.float32)
        i = torch.arange(H_pat, device=device, dtype=torch.float32)
        grid_i, grid_j = torch.meshgrid(i, j, indexing="ij")
        u_pixel = (grid_j + 0.5) * img_w / W_pat
        v_pixel = (grid_i + 0.5) * img_h / H_pat
        from baselines.ace_ego_hand.ace_ego_hand.archs.geodit_arch import _reject_non_pinhole
        _reject_non_pinhole(intr)
        azimuth = torch.atan((u_pixel - cx) / fx).reshape(-1, 1)
        elevation = torch.atan((v_pixel - cy) / fy).reshape(-1, 1)
        freq = self.freq_bands.unsqueeze(0)
        pe = torch.cat(
            [torch.sin(azimuth * freq), torch.cos(azimuth * freq),
             torch.sin(elevation * freq), torch.cos(elevation * freq)], dim=-1
        )
        return self.ray_proj(pe).unsqueeze(0).to(dtype)  # [1, HW, hidden]


def _ray_pe_intrinsics(intr: dict, frame_index: int = 0) -> dict:
    """Select the camera at a latent frame's source pixel-frame index."""
    per_frame = intr.get("per_frame")
    base = per_frame[frame_index] if per_frame is not None else intr
    out = {
        "fx": base["fx"], "fy": base["fy"], "cx": base["cx"], "cy": base["cy"],
        "image_width": intr["image_width"], "image_height": intr["image_height"],
    }
    return out


class MemorySegmentBetasTransformer(nn.Module):
    """Offline projector with full-segment temporal attention.

    Args:
        latent_channels: channel count of cached VDM features.
        hidden_dim: transformer token dimension.
        num_temporal_layers: number of bidirectional encoder layers.
        num_heads: attention heads for spatial pooling and temporal encoder.
        ffn_mult: temporal encoder feed-forward expansion.
        dropout: dropout used by transformer blocks.
        max_spatial_tokens: maximum H*W token count supported by learned PE.
        max_video_frames: maximum output frame count supported by learned PE.
        cam_trans_decode: `hamer`, `inv_proj`, or `mixed_pnp` (in-model
            differentiable decode: keep tz from the log-z head,
            re-solve (tx, ty) by per-joint pixel LSQ anchoring canonical MANO
            joints to the model's own direct_joints2d; gated rows fall back to
            the inv_proj decode; requires set_mano_models()).
        spatial_pe_mode: `flat` (original learned flat token table, raster-index
            sliced), `interp2d` (learned 16x16 grid bilinearly resized to the
            (H, W) token grid — resolution-agnostic), or `none`.
        betas_scope: `segment` shares one beta vector across slots and frames;
            `per_hand` predicts one beta vector per slot, still constant over time.
        per_hand_betas_source: for `betas_scope='per_hand'`, `cls_slot`
            predicts from `shape_cls + slot_embed`; `slot_mean` predicts from
            each hand slot's mean temporal token.
        detach_betas_source: stop beta-head gradients from updating the shared
            temporal tokens. This keeps clip-shape supervision from distorting
            pose/camera features while still training the beta head.
        spatial_pool_mode: `frame` pools one token per latent frame before hand
            slots are added; `slot` pools one token per hand slot and latent
            frame before bidirectional temporal attention.
    """

    def __init__(
        self,
        latent_channels: int = 1536,
        hidden_dim: int = 512,
        num_temporal_layers: int = 4,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        max_spatial_tokens: int = 2048,
        max_video_frames: int = 256,
        cam_trans_decode: str = "inv_proj",
        spatial_pe_mode: str = "flat",
        betas_scope: str = "segment",
        per_hand_betas_source: str = "cls_slot",
        detach_betas_source: bool = False,
        spatial_pool_mode: str = "frame",
        num_direct_joints: int = 21,
        num_slots: int = MAX_HANDS,
        joints2d_readout: str = "pooled",
        temporal_arch: str = "pooled",
        alt_num_register: int = 4,
        alt_num_layers: int | None = None,
        alt_query_mode: str = "hand",
        mano_readout: str = "pooled",
        alt_mem_tokens: int = 0,
        alt_mem_states_per_hand: int = 4,
        alt_mem_update_kind: str = "gru",
        alt_mem_bptt: str = "truncated",
        alt_shape_memory: bool = False,
        alt_temporal_rope: bool = False,
        occ_sim_prob: float = 0.0,
        occ_sim_max_span: float = 0.4,
        bone_num_layers: int = 2,
        bone_use_kin_mask: bool = True,
        camera_tokens: int = 0,
        use_ray_pe: bool = False,
        ray_pe_n_freqs: int = 8,
        temporal_window: int = 0,
        hand_mult: int = 1,
        input_norm: bool = False,
        mano_from_joints_ik: bool = False,
        ik_num_layers: int = 4,
        ik_num_heads: int = 4,
    ) -> None:
        super().__init__()
        if cam_trans_decode not in ("hamer", "inv_proj", "mixed_pnp"):
            raise ValueError(
                "cam_trans_decode must be 'hamer', 'inv_proj' or 'mixed_pnp', "
                f"got {cam_trans_decode!r}"
            )
        if cam_trans_decode == "mixed_pnp" and int(num_direct_joints) != 21:
            raise ValueError(
                "cam_trans_decode='mixed_pnp' requires num_direct_joints=21 "
                f"(canonical MANO OpenPose joints), got {num_direct_joints}"
            )
        if spatial_pe_mode not in ("flat", "interp2d", "none"):
            raise ValueError(
                f"spatial_pe_mode must be 'flat', 'interp2d' or 'none', got {spatial_pe_mode!r}"
            )
        if betas_scope not in ("segment", "per_hand"):
            raise ValueError(
                f"betas_scope must be 'segment' or 'per_hand', got {betas_scope!r}"
            )
        if per_hand_betas_source not in ("cls_slot", "slot_mean"):
            raise ValueError(
                "per_hand_betas_source must be 'cls_slot' or 'slot_mean', "
                f"got {per_hand_betas_source!r}"
            )
        if spatial_pool_mode not in ("frame", "slot"):
            raise ValueError(
                f"spatial_pool_mode must be 'frame' or 'slot', got {spatial_pool_mode!r}"
            )
        if num_slots != MAX_HANDS:
            raise ValueError(f"Only {MAX_HANDS} hand slots are supported, got {num_slots}")

        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.max_spatial_tokens = int(max_spatial_tokens)
        self.max_video_frames = int(max_video_frames)
        self.cam_trans_decode = cam_trans_decode
        self.spatial_pe_mode = spatial_pe_mode
        # mixed_pnp: frozen smplx MANO models {False: left, True: right},
        # attached AFTER construction via set_mano_models(). Kept in a plain dict
        # (NOT nn.ModuleDict) so the projector's state_dict / optimizer / DDP
        # reducer never see them — pure frozen geometry functions.
        self._mano_models: dict | None = None
        self.betas_scope = betas_scope
        self.per_hand_betas_source = per_hand_betas_source
        self.detach_betas_source = bool(detach_betas_source)
        self.spatial_pool_mode = spatial_pool_mode
        self.num_direct_joints = int(num_direct_joints)
        if joints2d_readout not in ("pooled", "spatial"):
            raise ValueError(
                f"joints2d_readout must be 'pooled' or 'spatial', got {joints2d_readout!r}"
            )
        self.joints2d_readout = joints2d_readout
        if temporal_arch not in ("pooled", "alternating"):
            raise ValueError(
                f"temporal_arch must be 'pooled' or 'alternating', got {temporal_arch!r}"
            )
        self.temporal_arch = temporal_arch
        if alt_query_mode not in ("hand", "joint"):
            raise ValueError(
                f"alt_query_mode must be 'hand' or 'joint', got {alt_query_mode!r}"
            )
        self.alt_query_mode = alt_query_mode
        if mano_readout not in ("pooled", "joint_grounded", "per_bone", "heatmap_grid"):
            raise ValueError(
                "mano_readout must be 'pooled', 'joint_grounded', 'per_bone' or "
                f"'heatmap_grid', got {mano_readout!r}"
            )
        if mano_readout in ("joint_grounded", "per_bone", "heatmap_grid") and not (
            temporal_arch == "alternating" and alt_query_mode == "joint"
        ):
            raise ValueError(
                f"mano_readout={mano_readout!r} requires temporal_arch='alternating' "
                "and alt_query_mode='joint'"
            )
        self.mano_readout = mano_readout
        # Derive MANO from predicted joints via a learnable IK head. Default
        # False => byte-identical to the pooled-regression base (no ik_head params).
        self.mano_from_joints_ik = bool(mano_from_joints_ik)
        # Cross-window memory. 0 => no mem params; identical base.
        if alt_mem_bptt not in ("truncated", "full"):
            raise ValueError(f"alt_mem_bptt must be 'truncated' or 'full', got {alt_mem_bptt!r}")
        if int(alt_mem_tokens) > 0 and temporal_arch != "alternating":
            raise ValueError("alt_mem_tokens>0 requires temporal_arch='alternating'")
        self.mem_tokens = int(alt_mem_tokens)
        self.mem_states_per_hand = int(alt_mem_states_per_hand)
        self.mem_update_kind = alt_mem_update_kind
        self.mem_bptt = alt_mem_bptt
        self._mem_in = None   # functional carry set by forward_batched
        self._mem_out = None
        # Optionally expose the internal patch grid x (post
        # input_proj + spatial_pe + ray_pe = the KV the hand/joint queries
        # cross-attend) so an auxiliary geometry head can decode it. Off by
        # default => the view is never taken (byte-identical to the base).
        self._capture_grid = False
        self._grid_out = None
        # Shape-carry: carry a CLIP-CONSTANT betas source (hand scale) across windows
        # via EMA, so all windows of a source share one robustly-aggregated shape
        # (anchors monocular depth and keeps scale consistent across seams). Side-channel
        # (_shape_in/_shape_out) so the (mem_in, return_mem) contract is untouched.
        self.shape_memory = bool(alt_shape_memory)
        self.temporal_rope = bool(alt_temporal_rope)  # RoPE temporal axis
        # Camera-decoupled orientation: per-frame camera token(s) predict the frame-0-relative
        # world->cam rotation C(t); go = C @ G factors the head motion out of the
        # camera-space root orientation. 0 => off (identical to the base).
        self.camera_tokens = int(camera_tokens)
        if self.camera_tokens > 0 and temporal_arch != "alternating":
            raise ValueError("camera_tokens>0 requires temporal_arch='alternating'")
        # Intrinsics-aware ray PE. Zero-init last layer => no-op
        # residual at start, so use_ray_pe=false is byte-identical to the base.
        self.use_ray_pe = bool(use_ray_pe)
        if self.use_ray_pe:
            self.ray_pe = RayPositionalEncoding(self.hidden_dim, n_freq_bands=int(ray_pe_n_freqs))
        # Temporal-scope hierarchy: form tokens (hand+joint) attend only ±window
        # frames; register/camera/mem attend the whole clip. 0 => full attention.
        self.temporal_window = int(temporal_window)
        # hand_mult: K query tokens per hand slot (decode-capacity boost, pooled).
        self.hand_mult = int(hand_mult)
        # Occlusion simulation: during training, randomly blank the
        # joint-signal channels (0:3, joints-banana) for a contiguous frame span
        # so the model learns to inpaint the hand from temporal context (a learned
        # motion prior for occlusion-fill). 0.0 => off (identical to before).
        self.occ_sim_prob = float(occ_sim_prob)
        self.occ_sim_max_span = float(occ_sim_max_span)
        if self.shape_memory:
            self.shape_ema_logit = nn.Parameter(torch.tensor(0.5))  # sig(0.5)≈0.62 keep-prior
        self._shape_in = None
        self._shape_out = None
        # Per-bone MANO rotation decoder. num_bones = 1 global_orient + 15 hand_pose.
        self.num_mano_bones = 1 + 15
        self.bone_num_layers = int(bone_num_layers)
        self.bone_use_kin_mask = bool(bone_use_kin_mask)
        self.alt_num_register = int(alt_num_register)
        self.alt_num_layers = int(alt_num_layers) if alt_num_layers is not None else int(num_temporal_layers)

        # Optional per-channel input standardization (for mixed-dataset training).
        self.input_norm = (nn.InstanceNorm2d(self.latent_channels, affine=True)
                           if bool(input_norm) else None)
        self.input_proj = nn.Sequential(
            nn.Linear(self.latent_channels, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        # Spatial PE. 'flat' = the ORIGINAL learned flat table sliced by
        # raster index (same param name/shape => old ckpts load byte-identically,
        # and in flat mode this randn call keeps the ctor RNG stream unchanged).
        # Defect: raster slicing is stride-scrambled across grids (15x21 vs 15x15
        # vs 15x26 vs 32x32) and rows beyond the largest trained H*W stay
        # untrained. 'interp2d' = a learned 16x16 grid bilinearly resized to
        # (H, W) at forward — resolution-agnostic, spatially smooth, consistent
        # row/col identity across grids. 'none' = no learned spatial PE (ray-PE,
        # if enabled, is the only spatial signal). Only the mode's own param is
        # instantiated so each mode's state_dict is minimal.
        if self.spatial_pe_mode == "flat":
            self.spatial_pe = nn.Parameter(
                torch.randn(1, self.max_spatial_tokens, self.hidden_dim) * 0.02
            )
        elif self.spatial_pe_mode == "interp2d":
            self.spatial_pe_grid = nn.Parameter(
                torch.randn(1, self.hidden_dim, 16, 16) * 0.02
            )
        self.latent_temporal_pe = nn.Parameter(
            torch.randn(1, 64, self.hidden_dim) * 0.02
        )
        self.video_temporal_pe = nn.Parameter(
            torch.randn(1, self.max_video_frames, self.hidden_dim) * 0.02
        )

        self.frame_pool_query = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        if self.spatial_pool_mode == "slot":
            self.slot_pool_query = nn.Parameter(
                torch.randn(1, self.num_slots, self.hidden_dim) * 0.02
            )
        self.spatial_pool = nn.MultiheadAttention(
            self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_pool_norm = nn.LayerNorm(self.hidden_dim)

        self.shape_cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * int(ffn_mult),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=int(num_temporal_layers),
            norm=nn.LayerNorm(self.hidden_dim),
        )

        self.slot_embed = nn.Parameter(torch.randn(self.num_slots, self.hidden_dim) * 0.02)
        self.hand_fuse = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )

        head_hidden = max(128, self.hidden_dim // 2)
        self.head_global_orient = nn.Sequential(
            nn.Linear(self.hidden_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 6),
        )
        self.head_hand_pose = nn.Sequential(
            nn.Linear(self.hidden_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 15 * 6),
        )
        self.head_cam_trans = nn.Sequential(
            nn.Linear(self.hidden_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 3),
        )
        self.head_exists_2d = nn.Sequential(
            nn.Linear(self.hidden_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )
        self.head_exists_3d = nn.Sequential(
            nn.Linear(self.hidden_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )
        self.head_direct_joints_rootrel = nn.Sequential(
            nn.Linear(self.hidden_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, self.num_direct_joints * 3),
        )
        self.head_direct_wrist_cam = nn.Sequential(
            nn.Linear(self.hidden_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 3),
        )
        self.head_direct_joints2d = nn.Sequential(
            nn.Linear(self.hidden_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, self.num_direct_joints * 2),
        )
        # Optional learnable IK head. When ON, MANO (go/hp) is derived FROM the
        # predicted root-relative joints via cross-joint attention,
        # OVERRIDING the pooled-feature regression. The
        # direct-joint heads stay (they feed IK + their own joint losses). Only
        # instantiated when ON, so the default (OFF) state_dict does not contain it.
        if self.mano_from_joints_ik:
            self.ik_head = MANOIKHead(
                num_joints=self.num_direct_joints,
                num_layers=int(ik_num_layers),
                num_heads=int(ik_num_heads),
            )
        # Optional spatial-aware joint readout (cross-attn to pre-pool
        # features + soft-argmax). Only instantiated in 'spatial' mode, so the
        # default 'pooled' state_dict does not contain it.
        if self.joints2d_readout == "spatial":
            from baselines.ace_ego_hand.ace_ego_hand.archs.projector.spatial_readout import SpatialJointReadout
            self.spatial_readout = SpatialJointReadout(
                hidden_dim=self.hidden_dim,
                num_slots=self.num_slots,
                num_joints=self.num_direct_joints,
                num_heads=num_heads,
                dropout=dropout,
            )
        # VGGT-style register-token alternating encoder. Only instantiated in
        # 'alternating' mode, so the default 'pooled' state_dict does not contain
        # it. It replaces the spatial-pool + temporal-encoder
        # path (those modules stay built but are bypassed in alternating mode).
        if self.temporal_arch == "alternating":
            from baselines.ace_ego_hand.ace_ego_hand.archs.projector.memory_encoder import (
                MemoryAlternatingEncoder,
            )
            self.alternating_encoder = MemoryAlternatingEncoder(
                self.hidden_dim,
                num_slots=self.num_slots,
                num_register=self.alt_num_register,
                num_layers=self.alt_num_layers,
                num_heads=num_heads,
                ffn_mult=ffn_mult,
                dropout=dropout,
                query_mode=self.alt_query_mode,
                num_joints=self.num_direct_joints,
                produce_mano_feat=(self.mano_readout == "joint_grounded"),
                produce_bone_feat=(self.mano_readout == "per_bone"),
                produce_heatmap_mano=(self.mano_readout == "heatmap_grid"),
                num_bones=self.num_mano_bones,
                bone_num_layers=self.bone_num_layers,
                bone_use_kin_mask=self.bone_use_kin_mask,
                mem_tokens=self.mem_tokens,
                mem_states_per_hand=self.mem_states_per_hand,
                mem_update_kind=self.mem_update_kind,
                temporal_rope=self.temporal_rope,
                camera_tokens=self.camera_tokens,
                temporal_window=self.temporal_window,
                hand_mult=self.hand_mult,
            )
            if self.camera_tokens > 0:
                # CameraHead: pool the per-frame camera tokens -> 6D -> C(t) rotmat.
                # Zero-init weight + identity-6D bias => C(t)=I at start, so
                # go = C@G = G reproduces the base exactly; camera_rot_geo then pulls
                # C toward the true (frame-0-relative) camera trajectory.
                cam_head_hidden = max(128, self.hidden_dim // 2)
                self.head_camera_rot = nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, cam_head_hidden),
                    nn.GELU(),
                    nn.Linear(cam_head_hidden, 6),
                )
                nn.init.zeros_(self.head_camera_rot[-1].weight)
                self.head_camera_rot[-1].bias.data = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                )
            if self.mano_readout == "per_bone":
                # Shared per-bone rotation head: reads each bone token's own 6D.
                bone_head_hidden = max(128, self.hidden_dim // 2)
                self.head_bone_rot = nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, bone_head_hidden),
                    nn.GELU(),
                    nn.Linear(bone_head_hidden, 6),
                )
        self.head_betas_segment = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 10),
        )
        self.head_betas_hand = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 10),
        )

        nn.init.zeros_(self.head_cam_trans[-1].weight)
        self.head_cam_trans[-1].bias.data = torch.tensor([0.0, 0.0, -0.693])
        nn.init.zeros_(self.head_betas_segment[-1].weight)
        nn.init.zeros_(self.head_betas_segment[-1].bias)
        nn.init.zeros_(self.head_betas_hand[-1].weight)
        nn.init.zeros_(self.head_betas_hand[-1].bias)
        nn.init.zeros_(self.head_exists_2d[-1].weight)
        self.head_exists_2d[-1].bias.data.fill_(-1.0)
        nn.init.zeros_(self.head_exists_3d[-1].weight)
        self.head_exists_3d[-1].bias.data.fill_(-1.0)
        nn.init.zeros_(self.head_direct_wrist_cam[-1].weight)
        self.head_direct_wrist_cam[-1].bias.data = torch.tensor([0.0, 0.0, 0.5])

    def forward(
        self,
        latents: torch.Tensor,
        n_video_frames: int,
        intrinsics: dict | None = None,
        intrinsics_list: list[dict] | None = None,
    ):
        """Dispatch single-segment or batched input."""
        single = latents.dim() == 4
        if single:
            latents = latents.unsqueeze(0)
        if intrinsics_list is None and intrinsics is not None:
            intrinsics_list = [intrinsics] * latents.shape[0]
        out = self.forward_batched(
            latents,
            n_video_frames=n_video_frames,
            intrinsics_list=intrinsics_list,
        )
        return out[0] if single else out

    def _shape_carry(self, source: torch.Tensor) -> torch.Tensor:
        """Shape-carry: EMA-carry a clip-constant betas source across windows.

        ``state = a*shape_in + (1-a)*source`` with a learnable keep-rate ``a``; the
        carried ``state`` (in ``self._shape_out``) is what the caller passes back as
        ``shape_in`` for the next window. No-op (returns ``source``) when off.
        """
        if not self.shape_memory:
            return source
        if self._shape_in is None:
            state = source
        else:
            a = torch.sigmoid(self.shape_ema_logit)
            state = a * self._shape_in + (1.0 - a) * source
        self._shape_out = state
        return state

    def forward_batched(
        self,
        latents: torch.Tensor | list[torch.Tensor],
        n_video_frames: int,
        intrinsics_list: list[dict] | None = None,
        pred_rays: torch.Tensor | None = None,   # self-ray-PE: (B,3,H,W) predicted ray field
        mem_in: torch.Tensor | None = None,
        return_mem: bool = False,
        shape_in: torch.Tensor | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Run a batch of same-shaped segments.

        Args:
            latents: `[B, C, F_lat, H, W]`, a single `[C, F_lat, H, W]`, or a
                list of single-segment tensors.
            n_video_frames: output temporal length, normally 81.
            intrinsics_list: per-segment camera intrinsics. Required for
                `cam_trans_decode='inv_proj'`.
            mem_in: `(B, mem_tokens, D)` carried cross-window memory, or None to
                start a fresh chunk (uses the learned ``mem_init``). Only used
                when ``alt_mem_tokens>0``.
            return_mem: when True, return ``(results, mem_out)`` so the caller can
                carry memory to the next window. Default False => list-of-dicts,
                identical to the base.
        """
        self._mem_in = mem_in
        self._mem_out = None
        self._shape_in = shape_in
        self._shape_out = None
        if isinstance(latents, list):
            latents = torch.stack(latents, dim=0)
        if latents.dim() == 4:
            latents = latents.unsqueeze(0)
        if latents.dim() != 5:
            raise ValueError(f"latents must be 4D or 5D, got {tuple(latents.shape)}")
        if self.cam_trans_decode in ("inv_proj", "mixed_pnp") and intrinsics_list is None:
            raise ValueError(
                f"intrinsics_list is required for {self.cam_trans_decode} cam_trans decode"
            )

        B, C, F_lat, H, W = latents.shape
        if C != self.latent_channels:
            raise ValueError(
                f"Expected latent_channels={self.latent_channels}, got {C}"
            )
        if H * W > self.max_spatial_tokens:
            raise ValueError(
                f"Spatial tokens {H}*{W}={H * W} exceed {self.max_spatial_tokens}"
            )
        if not self.temporal_rope and F_lat > self.latent_temporal_pe.shape[1]:
            # With RoPE the learned latent PE is unused, so no F_lat cap
            # (relative encoding regenerates for any length — enables sliding-window
            # inference).
            raise ValueError(
                f"Latent frames {F_lat} exceed {self.latent_temporal_pe.shape[1]}"
            )
        if not self.temporal_rope and n_video_frames > self.max_video_frames:
            # No video-frame cap under RoPE (learned video PE unused).
            raise ValueError(
                f"Video frames {n_video_frames} exceed {self.max_video_frames}"
            )

        if self.training and self.occ_sim_prob > 0.0 and F_lat >= 2 and C >= 6:
            # Occlusion simulation: per item, with prob occ_sim_prob, zero
            # the joint-signal channels (0:3) over a random contiguous frame span
            # (<= occ_sim_max_span * F_lat, never the whole clip). Keeps scene RGB
            # (3:6). GT/loss unchanged -> model learns to predict masked frames
            # from temporal context = a learned inpainting prior for occlusion.
            latents = latents.clone()
            max_span = max(1, min(F_lat - 1, int(round(self.occ_sim_max_span * F_lat))))
            for b in range(B):
                if float(torch.rand(())) < self.occ_sim_prob:
                    span = int(torch.randint(1, max_span + 1, ()))
                    s = int(torch.randint(0, F_lat - span + 1, ()))
                    latents[b, 0:3, s:s + span] = 0.0

        if self.input_norm is not None:
            # Per-channel, per-frame spatial standardization (InstanceNorm) → removes
            # cross-dataset input-scale differences (ARCTIC renders ~2x HOT3D means) so
            # the shared input_proj sees a consistent distribution when training on
            # mixed cameras. No-op-ish for single-dataset (affine can undo it).
            xn = latents.permute(0, 2, 1, 3, 4).reshape(B * F_lat, C, H, W)
            xn = self.input_norm(xn)
            latents = xn.reshape(B, F_lat, C, H, W).permute(0, 2, 1, 3, 4)
        x = latents.permute(0, 2, 3, 4, 1).reshape(B * F_lat, H * W, C)
        x = self.input_proj(x)
        if self.spatial_pe_mode == "flat":
            x = x + self.spatial_pe[:, : H * W, :]
        elif self.spatial_pe_mode == "interp2d":
            # Bilinear-resize the learned 16x16 PE grid to the token grid
            # so every (row, col) cell gets a smooth, grid-shape-agnostic PE.
            pe = F.interpolate(
                self.spatial_pe_grid.float(),
                size=(H, W),
                mode="bilinear",
                align_corners=True,
            )  # (1, hidden, H, W)
            x = x + pe.reshape(1, self.hidden_dim, H * W).transpose(1, 2).to(x.dtype)
        # 'none': no learned spatial PE (ray-PE below is the only spatial signal).
        # VAE latent t is anchored at pixel frame 4*t; do not broadcast frame 0 K.
        if self.use_ray_pe and (intrinsics_list is not None or pred_rays is not None):
            rpes = torch.stack(
                [
                    self.ray_pe(H, W, None if pred_rays is not None else _ray_pe_intrinsics(intrinsics_list[b], 4 * t),
                                x.device, x.dtype,
                                rays=(pred_rays[b] if pred_rays is not None else None))[0]
                    for b in range(B)
                    for t in range(F_lat)
                ],
                dim=0,
            ).view(B, F_lat, H * W, self.hidden_dim)
            x = (x.view(B, F_lat, H * W, self.hidden_dim) + rpes).reshape(
                B * F_lat, H * W, self.hidden_dim
            )

        if self._capture_grid:
            # (B, F_lat, H*W, hidden) — the exact grid the queries cross-attend.
            self._grid_out = x.view(B, F_lat, H * W, self.hidden_dim)

        alt_j2d = alt_j3d = alt_mano_feat = alt_bone_feat = alt_cam_feat = None
        if self.temporal_arch == "alternating":
            (
                hand_feat,
                shape_token,
                alt_j2d,
                alt_j3d,
                alt_mano_feat,
                alt_bone_feat,
                alt_cam_feat,
            ) = self._encode_alternating(
                x,
                B=B,
                F_lat=F_lat,
                H=H,
                W=W,
                n_video_frames=n_video_frames,
            )
        elif self.spatial_pool_mode == "slot":
            hand_feat, shape_token = self._encode_slot_tokens(
                x,
                B=B,
                F_lat=F_lat,
                n_video_frames=n_video_frames,
            )
        else:
            hand_feat, shape_token = self._encode_frame_tokens(
                x,
                B=B,
                F_lat=F_lat,
                n_video_frames=n_video_frames,
            )

        hand_feat = hand_feat + self.hand_fuse(hand_feat)
        flat_feat = hand_feat.reshape(B * n_video_frames, self.num_slots, self.hidden_dim)

        # MANO pose readout. Default 'pooled' keeps go/hp on the slot token so old
        # checkpoints reproduce. 'joint_grounded' reads a per-joint-pooled
        # feature. 'per_bone' decodes each bone's rotation from its OWN token.
        if self.mano_readout == "per_bone" and alt_bone_feat is not None:
            # alt_bone_feat: (B, n_video, slots, num_bones, D); bone 0 = global_orient.
            bone_rot = self.head_bone_rot(alt_bone_feat).reshape(
                B * n_video_frames, self.num_slots, self.num_mano_bones, 6
            )
            go_6d = bone_rot[:, :, 0, :]
            hp_6d = bone_rot[:, :, 1:, :].reshape(B * n_video_frames, self.num_slots, 15 * 6)
        else:
            if self.mano_readout in ("joint_grounded", "heatmap_grid") and alt_mano_feat is not None:
                mano_flat = alt_mano_feat.reshape(B * n_video_frames, self.num_slots, self.hidden_dim)
            else:
                mano_flat = flat_feat
            go_6d = self.head_global_orient(mano_flat)
            hp_6d = self.head_hand_pose(mano_flat)
        cam_raw = self.head_cam_trans(flat_feat)
        exists_2d = torch.sigmoid(self.head_exists_2d(flat_feat).squeeze(-1))
        exists_3d = torch.sigmoid(self.head_exists_3d(flat_feat).squeeze(-1))
        direct_rootrel = self.head_direct_joints_rootrel(flat_feat).reshape(
            B,
            n_video_frames,
            self.num_slots,
            self.num_direct_joints,
            3,
        )
        direct_rootrel = direct_rootrel - direct_rootrel[:, :, :, 0:1, :]
        direct_wrist_cam = self.head_direct_wrist_cam(flat_feat).reshape(
            B,
            n_video_frames,
            self.num_slots,
            3,
        )
        direct_joints_cam = direct_rootrel + direct_wrist_cam.unsqueeze(-2)
        direct_joints2d = torch.sigmoid(self.head_direct_joints2d(flat_feat)).reshape(
            B,
            n_video_frames,
            self.num_slots,
            self.num_direct_joints,
            2,
        )
        if self.joints2d_readout == "spatial":
            # Override the pooled 2D + rootrel with spatially-grounded reads
            # (cross-attn to pre-pool features `x`). Wrist (root) stays from the
            # pooled head; cam joints recomposed from the new rootrel.
            sp_2d, sp_rootrel = self.spatial_readout(
                x, B=B, F_lat=F_lat, H=H, W=W, n_video_frames=n_video_frames
            )
            direct_joints2d = sp_2d
            direct_rootrel = sp_rootrel
            direct_joints_cam = direct_rootrel + direct_wrist_cam.unsqueeze(-2)
        if alt_j2d is not None:
            # Alternating encoder's per-joint readout (soft-argmax 2D + 3D)
            # overrides the pooled hand-token regression. Wrist stays from the
            # pooled head; cam joints recomposed from the new rootrel.
            direct_joints2d = alt_j2d
            direct_rootrel = alt_j3d
            direct_joints_cam = direct_rootrel + direct_wrist_cam.unsqueeze(-2)

        if self.mano_from_joints_ik:
            # IK head: derive MANO (go/hp) FROM the (final) predicted root-relative
            # joints via the learnable IK head, OVERRIDING the pooled-feature
            # regression above. direct_rootrel is (B, n_video, slots, J, 3) and
            # already wrist-anchored; flatten to (M, J, 3) with M = B*n_video*slots.
            ik_in = direct_rootrel.reshape(
                B * n_video_frames * self.num_slots, self.num_direct_joints, 3
            )
            ik_go, ik_hp = self.ik_head(ik_in)  # (M, 6), (M, 90)
            go_6d = ik_go.reshape(B * n_video_frames, self.num_slots, 6).to(go_6d.dtype)
            hp_6d = ik_hp.reshape(B * n_video_frames, self.num_slots, 15 * 6).to(hp_6d.dtype)

        # The cam_trans decode runs after the go/hp/betas blocks below
        # (mixed_pnp needs the final MANO params + direct_joints2d). For
        # hamer/inv_proj it is independent of them (a function of cam_raw +
        # intrinsics only).
        # Camera-decoupled orientation: hand head now outputs G = hand orient in the frame-0
        # camera frame; the camera token gives C(t) = frame-0-relative world->cam
        # rotation. go = C @ G reconstructs the camera-frame root orientation. C is
        # DETACHED here so go-derived losses train only G; camera_rot_geo (on the
        # returned `cam_rot`) is the sole gradient into C. camera_tokens=0 => go = G.
        G = rot6d_to_rotmat(go_6d).reshape(B, n_video_frames, self.num_slots, 3, 3)
        cam_rot = None
        if self.camera_tokens > 0 and alt_cam_feat is not None:
            cam_pooled = alt_cam_feat.mean(dim=2)  # (B, n_video, D) mean over cam tokens
            cam_6d = self.head_camera_rot(cam_pooled)  # (B, n_video, 6)
            cam_rot = rot6d_to_rotmat(cam_6d.reshape(-1, 6)).reshape(
                B, n_video_frames, 3, 3
            )
            go_mat = torch.matmul(cam_rot.detach().unsqueeze(2), G)
        else:
            go_mat = G
        go = go_mat.reshape(B, n_video_frames, self.num_slots, 1, 3, 3)
        hp = rot6d_to_rotmat(hp_6d.reshape(-1, 6)).reshape(
            B,
            n_video_frames,
            self.num_slots,
            15,
            3,
            3,
        )

        if self.betas_scope == "segment":
            beta_source = shape_token.detach() if self.detach_betas_source else shape_token
            beta_source = self._shape_carry(beta_source)  # shape EMA (no-op if off)
            segment_betas = self.head_betas_segment(beta_source)
            betas = segment_betas.view(B, 1, 1, 10).expand(
                B,
                n_video_frames,
                self.num_slots,
                10,
            )
        else:
            if self.per_hand_betas_source == "slot_mean":
                hand_shape_tokens = hand_feat.mean(dim=1)
            else:
                hand_shape_tokens = shape_token.unsqueeze(1) + self.slot_embed.unsqueeze(0)
            if self.detach_betas_source:
                hand_shape_tokens = hand_shape_tokens.detach()
            hand_shape_tokens = self._shape_carry(hand_shape_tokens)  # shape EMA
            segment_betas = self.head_betas_hand(hand_shape_tokens)
            betas = segment_betas.unsqueeze(1).expand(
                B,
                n_video_frames,
                self.num_slots,
                10,
            )

        if self.cam_trans_decode == "mixed_pnp":
            # Differentiable in-model mixed_pnp decode —
            # anchors (tx, ty) to the model's own 2D joints via per-joint LSQ,
            # so the cam_trans loss trains the joints_2d + MANO heads + log-z
            # instead of a pooled per-camera (u, v) regression.
            cam_trans = self._decode_cam_trans_mixed_pnp(
                cam_raw,
                go_mat=go[:, :, :, 0],
                hp_mat=hp,
                betas=betas,
                joints2d=direct_joints2d,
                B=B,
                n_video_frames=n_video_frames,
                intrinsics_list=intrinsics_list,
                pred_rays=(pred_rays if getattr(self, "self_ray_decode", False) else None),
            )
        else:
            cam_trans = self._decode_cam_trans(
                cam_raw,
                B=B,
                n_video_frames=n_video_frames,
                intrinsics_list=intrinsics_list,
                pred_rays=(pred_rays if getattr(self, "self_ray_decode", False) else None),
            )
        cam_trans = cam_trans.reshape(B, n_video_frames, self.num_slots, 3)
        exists_2d = exists_2d.reshape(B, n_video_frames, self.num_slots)
        exists_3d = exists_3d.reshape(B, n_video_frames, self.num_slots)
        slot_is_right = torch.tensor(
            [0.0, 1.0],
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, 1, self.num_slots)
        is_right = slot_is_right.expand(B, n_video_frames, self.num_slots)

        results = []
        for b_idx in range(B):
            result = {
                "go": go[b_idx],
                "hp": hp[b_idx],
                "global_orient": go[b_idx, :, :, 0],
                "hand_pose": hp[b_idx],
                "betas": betas[b_idx],
                "cam_trans": cam_trans[b_idx],
                "exists_2d": exists_2d[b_idx],
                "exists_3d": exists_3d[b_idx],
                "presence": exists_3d[b_idx],
                "direct_joints_rootrel": direct_rootrel[b_idx],
                "direct_wrist_cam": direct_wrist_cam[b_idx],
                "direct_joints_cam": direct_joints_cam[b_idx],
                "direct_joints2d": direct_joints2d[b_idx],
                "is_right": is_right[b_idx],
                "shape_token": shape_token[b_idx],
                "segment_betas": segment_betas[b_idx],
            }
            if cam_rot is not None:
                result["cam_rot"] = cam_rot[b_idx]  # (n_video, 3, 3) frame-0-rel C(t)
            results.append(result)
        if return_mem:
            return results, self._mem_out
        return results

    def _encode_frame_tokens(
        self,
        x: torch.Tensor,
        *,
        B: int,
        F_lat: int,
        n_video_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.frame_pool_query.expand(B * F_lat, -1, -1)
        pooled, _ = self.spatial_pool(q, x, x)
        frame_lat = self.spatial_pool_norm(q + pooled).squeeze(1)
        frame_lat = frame_lat.view(B, F_lat, self.hidden_dim)
        frame_lat = frame_lat + self.latent_temporal_pe[:, :F_lat, :]

        if n_video_frames != F_lat:
            t = frame_lat.transpose(1, 2)
            frame_tokens = F.interpolate(
                t,
                size=n_video_frames,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
        else:
            frame_tokens = frame_lat
        frame_tokens = frame_tokens + self.video_temporal_pe[:, :n_video_frames, :]

        cls = self.shape_cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, frame_tokens], dim=1)
        seq = self.temporal_encoder(seq)
        shape_token = seq[:, 0]
        frame_tokens = seq[:, 1:]
        hand_feat = frame_tokens.unsqueeze(2) + self.slot_embed.view(
            1,
            1,
            self.num_slots,
            -1,
        )
        return hand_feat, shape_token

    def _encode_slot_tokens(
        self,
        x: torch.Tensor,
        *,
        B: int,
        F_lat: int,
        n_video_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.slot_pool_query.expand(B * F_lat, -1, -1)
        pooled, _ = self.spatial_pool(q, x, x)
        hand_lat = self.spatial_pool_norm(q + pooled)
        hand_lat = hand_lat.view(B, F_lat, self.num_slots, self.hidden_dim)
        hand_lat = hand_lat + self.latent_temporal_pe[:, :F_lat, :].view(
            1,
            F_lat,
            1,
            self.hidden_dim,
        )
        hand_lat = hand_lat + self.slot_embed.view(1, 1, self.num_slots, self.hidden_dim)

        if n_video_frames != F_lat:
            t = hand_lat.permute(0, 2, 3, 1).reshape(
                B * self.num_slots,
                self.hidden_dim,
                F_lat,
            )
            hand_tokens = F.interpolate(
                t,
                size=n_video_frames,
                mode="linear",
                align_corners=True,
            )
            hand_tokens = hand_tokens.reshape(
                B,
                self.num_slots,
                self.hidden_dim,
                n_video_frames,
            ).permute(0, 3, 1, 2)
        else:
            hand_tokens = hand_lat
        hand_tokens = hand_tokens + self.video_temporal_pe[:, :n_video_frames, :].view(
            1,
            n_video_frames,
            1,
            self.hidden_dim,
        )
        hand_tokens = hand_tokens + self.slot_embed.view(1, 1, self.num_slots, self.hidden_dim)

        seq_tokens = hand_tokens.reshape(B, n_video_frames * self.num_slots, self.hidden_dim)
        cls = self.shape_cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, seq_tokens], dim=1)
        seq = self.temporal_encoder(seq)
        shape_token = seq[:, 0]
        hand_feat = seq[:, 1:].reshape(B, n_video_frames, self.num_slots, self.hidden_dim)
        return hand_feat, shape_token

    def _encode_alternating(
        self,
        x: torch.Tensor,
        *,
        B: int,
        F_lat: int,
        H: int,
        W: int,
        n_video_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """VGGT-style register-token path: keep spatial detail inside the
        temporal pathway instead of pooling to 1-2 tokens before it.

        ``x`` is the pre-pool patch grid ``(B*F_lat, H*W, hidden)``. The
        alternating encoder returns per-latent-frame hand features (and, in
        ``query_mode='joint'``, per-joint 2D + 3D from a spatially-grounded
        readout on temporally-refined queries). Hand features are temporally
        upsampled by feature interpolation (+ video PE); joint coords are
        upsampled by linear COORDINATE interpolation (cleaner than features).
        Returns ``(hand_tokens, shape_token, joint_2d, joint_rootrel)`` with the
        last two ``None`` in ``query_mode='hand'``.
        """
        hand_lat, shape_token, j2d_lat, j3d_lat, mano_lat, bone_lat, mem_out, cam_lat = self.alternating_encoder(
            x,
            B=B,
            F_lat=F_lat,
            H=H,
            W=W,
            slot_embed=self.slot_embed,
            latent_temporal_pe=(None if self.temporal_rope
                                else self.latent_temporal_pe[:, :F_lat, :]),
            mem_in=self._mem_in,
        )
        self._mem_out = mem_out  # captured by forward_batched(return_mem=True)
        # hand_lat / mano_lat: (B, F_lat, num_slots, hidden) -> feature-interp to video
        hand_tokens = self._interp_feat_t(hand_lat, n_video_frames)
        if not self.temporal_rope:
            # Skip the learned absolute video PE under RoPE (no length cap;
            # the heads are per-frame so this is just a learned per-frame bias).
            hand_tokens = hand_tokens + self.video_temporal_pe[:, :n_video_frames, :].view(
                1,
                n_video_frames,
                1,
                self.hidden_dim,
            )
        mano_feat = None
        if mano_lat is not None:
            mano_feat = self._interp_feat_t(mano_lat, n_video_frames)
            if not self.temporal_rope:
                # Skip the learned video PE under RoPE (no length cap) — same
                # guard as hand_tokens above. Without it, n_video_frames >
                # max_video_frames would break the view.
                mano_feat = mano_feat + self.video_temporal_pe[:, :n_video_frames, :].view(
                    1, n_video_frames, 1, self.hidden_dim
                )
        bone_feat = None
        if bone_lat is not None:
            # (B,F_lat,slots,num_bones,D) -> interp time -> (B,n_video,slots,num_bones,D)
            B_, F_, S_, NB_, D_ = bone_lat.shape
            bf = self._interp_feat_t(bone_lat.reshape(B_, F_, S_ * NB_, D_), n_video_frames)
            bone_feat = bf.reshape(B_, n_video_frames, S_, NB_, D_)
        cam_feat = None
        if cam_lat is not None:
            # Camera tokens: (B, F_lat, n_camera, D) -> feature-interp over time -> video res.
            cam_feat = self._interp_feat_t(cam_lat, n_video_frames)
        j2d = self._interp_coords_t(j2d_lat, n_video_frames)
        j3d = self._interp_coords_t(j3d_lat, n_video_frames)
        return hand_tokens, shape_token, j2d, j3d, mano_feat, bone_feat, cam_feat

    def _interp_feat_t(self, feat_lat: torch.Tensor, n_video_frames: int) -> torch.Tensor:
        """Linear feature interpolation of (B, F_lat, S, D) over time -> (B, n_video, S, D)."""
        B, F_lat, S, D = feat_lat.shape
        if n_video_frames == F_lat:
            return feat_lat
        t = feat_lat.permute(0, 2, 3, 1).reshape(B * S, D, F_lat)
        t = F.interpolate(t, size=n_video_frames, mode="linear", align_corners=True)
        return t.reshape(B, S, D, n_video_frames).permute(0, 3, 1, 2)

    @staticmethod
    def _interp_coords_t(coords: torch.Tensor | None, n_video_frames: int) -> torch.Tensor | None:
        """Linear interpolation of per-joint coords (B, F_lat, S, J, C) over the
        temporal axis -> (B, n_video_frames, S, J, C). Coords interpolate cleanly
        (heatmaps would not). No-op if already at video resolution or ``None``."""
        if coords is None:
            return None
        B, F_lat, S, J, C = coords.shape
        if F_lat == n_video_frames:
            return coords
        t = coords.permute(0, 2, 3, 4, 1).reshape(B * S * J * C, 1, F_lat)
        t = F.interpolate(t, size=n_video_frames, mode="linear", align_corners=True)
        return t.reshape(B, S, J, C, n_video_frames).permute(0, 4, 1, 2, 3).contiguous()

    def _decode_cam_trans(
        self,
        cam_raw: torch.Tensor,
        *,
        B: int,
        n_video_frames: int,
        intrinsics_list: list[dict] | None,
        pred_rays: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tz = torch.exp(cam_raw[..., 2:3])
        if self.cam_trans_decode == "hamer":
            return torch.cat([cam_raw[..., 0:1] * tz, cam_raw[..., 1:2] * tz, tz], dim=-1)
        if pred_rays is not None:
            # Full K-free decode: sample the PREDICTED ray field at the wrist's
            # (u,v) -> ray (dx,dy,dz); tx=(dx/dz)*tz, ty=(dy/dz)*tz. No K anywhere.
            # cam_raw[...,0:1] is u_norm in [-1,1] (grid_sample convention directly).
            uv = torch.stack([cam_raw[..., 0], cam_raw[..., 1]], dim=-1)     # (B,slots,2) in [-1,1]
            grid = uv.reshape(B, -1, 1, 2)                                   # (B, S, 1, 2)
            rays = torch.nn.functional.grid_sample(
                pred_rays, grid, mode="bilinear", align_corners=True)        # (B,3,S,1)
            rays = rays[..., 0].permute(0, 2, 1).reshape(*cam_raw.shape[:-1], 3)  # (...,3)
            dz = rays[..., 2:3].abs().clamp_min(1e-6)
            tx = rays[..., 0:1] / dz * tz
            ty = rays[..., 1:2] / dz * tz
            return torch.cat([tx, ty, tz], dim=-1)

        params = self._build_intrinsics_params(
            intrinsics_list,
            B=B,
            n_video_frames=n_video_frames,
            device=cam_raw.device,
            dtype=cam_raw.dtype,
        ).unsqueeze(1)
        fx = params[..., 0:1]
        fy = params[..., 1:2]
        cx = params[..., 2:3]
        cy = params[..., 3:4]
        img_w = params[..., 4:5]
        img_h = params[..., 5:6]
        u_px = (cam_raw[..., 0:1] + 1.0) * 0.5 * img_w
        v_px = (cam_raw[..., 1:2] + 1.0) * 0.5 * img_h
        tx = (u_px - cx) * tz / fx
        ty = (v_px - cy) * tz / fy
        return torch.cat([tx, ty, tz], dim=-1)

    # ------------------------------------------------------ mixed_pnp decode
    def set_mano_models(self, models: dict) -> None:
        """Attach frozen smplx MANO models for the mixed_pnp cam_trans decode.

        ``models`` is ``{False: mano_left, True: mano_right}`` (frozen
        full-rotation RIGHT/mirror adapters — the exact models the loss stack used), already on
        the projector's device. Stored in a PLAIN dict so they never enter the
        projector's state_dict / optimizer / DDP reducer.
        """
        self._mano_models = models

    def _mixed_pnp_canonical_joints(
        self,
        go_mat: torch.Tensor,     # (N, slots, 3, 3) FINAL global-orient rotmats
        hp_mat: torch.Tensor,     # (N, slots, 15, 3, 3)
        betas: torch.Tensor,      # (N, slots, 10)
        chunk: int = 8192,
    ) -> torch.Tensor:
        """Canonical (pre-cam_trans) 21 OpenPose joints per slot, differentiable.

        Mirrors the loss path (_mano_geometry_losses) EXACTLY: rotmat inputs ->
        mano_forward_batch_full (full-rotation RIGHT/mirror forward,
        16 joints + 5 tip vertices -> OpenPose-21 remap),
        with slot 0 -> mano_models[False] (left), slot 1 -> mano_models[True]
        (right) — the projector's fixed slot->side convention (slot_is_right).
        Chunked for memory. Returns (N, slots, 21, 3) in the input dtype.
        """
        from baselines.ace_ego_hand.ace_ego_hand.mano_utils import mano_forward_batch_full  # noqa: WPS433

        N = go_mat.shape[0]
        out_dtype = betas.dtype
        per_slot = []
        for s in range(self.num_slots):
            mano = self._mano_models[bool(s == 1)]
            go_s = go_mat[:, s].reshape(N, 3, 3).float()
            hp_s = hp_mat[:, s].reshape(N, 15, 3, 3).float()
            be_s = betas[:, s].reshape(N, 10).float()
            chunks = []
            for a in range(0, N, chunk):
                j, _ = mano_forward_batch_full(
                    go_s[a:a + chunk], hp_s[a:a + chunk], be_s[a:a + chunk], mano
                )
                chunks.append(j)
            per_slot.append(torch.cat(chunks, dim=0))
        return torch.stack(per_slot, dim=1).to(out_dtype)  # (N, slots, 21, 3)

    # mixed_pnp acceptance-gate constants.
    _PNP_Z_NEAR = 0.05
    _PNP_INTERIOR_MARGIN = 0.02
    _PNP_MIN_INTERIOR = 6
    _PNP_RESID_FRAC = 0.25
    _PNP_RESID_MIN_PX = 15.0

    def _decode_cam_trans_mixed_pnp(
        self,
        cam_raw: torch.Tensor,     # (B*n_video, slots, 3) raw head output
        *,
        go_mat: torch.Tensor,      # (B, n_video, slots, 3, 3) FINAL rotmats
        hp_mat: torch.Tensor,      # (B, n_video, slots, 15, 3, 3)
        betas: torch.Tensor,       # (B, n_video, slots, 10)
        joints2d: torch.Tensor,    # (B, n_video, slots, 21, 2) in [0, 1] (FINAL)
        B: int,
        n_video_frames: int,
        intrinsics_list: list[dict] | None,
        pred_rays: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Differentiable in-model mixed_pnp decode.

        Full K-free: when `pred_rays` (B,3,H,W predicted ray field) is
        given, the per-joint 2D anchor uses the PREDICTED-ray bearing sampled at
        each joint pixel instead of the pinhole `(u-cx)/fx`. Same LSQ (k_u=1/z,
        a_u=bx-c_x/z), no K anywhere, so mixed_pnp keeps its 2D anchoring under
        full K-free.

        Keep tz = exp(log-z head); re-solve (tx, ty) by a closed-form per-row
        pixel LSQ that aligns the projected canonical MANO joints to the model's
        own direct_joints2d. Per-joint gate m (z front, interior 2D) and a row
        acceptance gate (>=6 interior joints, post-solve RMS residual within
        max(0.25*hand_diag_px, 15px)) are computed under torch.no_grad(); rows
        that fail fall back to the inv_proj decode via torch.where, so gradients
        flow only through each row's selected branch. Training with this decode
        routes the cam_trans loss into the joints_2d head + MANO heads + log-z.
        """
        if self._mano_models is None:
            raise RuntimeError(
                "cam_trans_decode='mixed_pnp' requires frozen MANO models: call "
                "projector.set_mano_models({False: mano_left, True: mano_right}) "
                "after construction (GeoDitModel does this when the projector "
                "config sets cam_trans_decode: mixed_pnp)."
            )
        # Fallback branch: the ORIGINAL inv_proj decode. _decode_cam_trans only
        # special-cases 'hamer', so calling it with mode 'mixed_pnp' executes
        # exactly the inv_proj math (same _build_intrinsics_params plumbing).
        inv_ct = self._decode_cam_trans(
            cam_raw, B=B, n_video_frames=n_video_frames, intrinsics_list=intrinsics_list,
            pred_rays=pred_rays,   # ray-mode: fallback is ray-based inv_proj too (no K)
        )  # (N, slots, 3)

        N = B * n_video_frames
        J = self.num_direct_joints
        tz = torch.exp(cam_raw[..., 2:3])                            # (N, slots, 1)
        params = self._build_intrinsics_params(
            intrinsics_list,
            B=B,
            n_video_frames=n_video_frames,
            device=cam_raw.device,
            dtype=cam_raw.dtype,
        ).unsqueeze(1)                                               # (N, 1, 6)
        fx = params[..., 0:1]
        fy = params[..., 1:2]
        cx = params[..., 2:3]
        cy = params[..., 3:4]
        img_w = params[..., 4:5]
        img_h = params[..., 5:6]

        # Optionally detach go/hp/betas/joints2d so the
        # cam_trans loss trains ONLY the log-z/cam head + raymap, not the MANO
        # articulation heads (mirrors the joints-only cam_trans loss's intended
        # detach). Off => byte-identical.
        _dt = getattr(self, "mixed_pnp_detach_mano", False)
        _go = go_mat.detach() if _dt else go_mat
        _hp = hp_mat.detach() if _dt else hp_mat
        _be = betas.detach() if _dt else betas
        canon = self._mixed_pnp_canonical_joints(
            _go.reshape(N, self.num_slots, 3, 3),
            _hp.reshape(N, self.num_slots, 15, 3, 3),
            _be.reshape(N, self.num_slots, 10),
        )                                                            # (N, slots, 21, 3)
        c_x, c_y, c_z = canon[..., 0], canon[..., 1], canon[..., 2]  # (N, slots, J)
        j2d = joints2d.reshape(N, self.num_slots, J, 2).to(cam_raw.dtype)
        if _dt:
            j2d = j2d.detach()
        u_px = j2d[..., 0] * img_w                                   # (N, slots, J)
        v_px = j2d[..., 1] * img_h

        ray_mode = pred_rays is not None
        if ray_mode:
            # sample the predicted ray field at each joint's 2D pixel -> bearing
            # (bx,by)=(rx/rz, ry/rz). camera is constant per clip, so broadcast
            # pred_rays (B,3,H,W) across the n_video frames.
            Hh, Ww = pred_rays.shape[-2], pred_rays.shape[-1]
            pr = (pred_rays.unsqueeze(1).expand(B, n_video_frames, 3, Hh, Ww)
                  .reshape(N, 3, Hh, Ww))
            grid = (j2d * 2.0 - 1.0).reshape(N, self.num_slots * J, 1, 2)   # [-1,1]
            rs = torch.nn.functional.grid_sample(
                pr, grid, mode="bilinear", align_corners=True)             # (N,3,S*J,1)
            rs = rs[..., 0].permute(0, 2, 1).reshape(N, self.num_slots, J, 3)
            rz = rs[..., 2].abs().clamp_min(1e-6)
            bx = rs[..., 0] / rz                                          # (N,slots,J)
            by = rs[..., 1] / rz

        z_raw = c_z + tz                                             # (N, slots, J)
        z_j = z_raw.clamp(min=self._PNP_Z_NEAR)
        with torch.no_grad():
            # Per-joint vote gate: point in front of the camera AND strictly
            # interior in 2D (soft-argmax saturates at the border for
            # out-of-FOV fingers and would poison the LSQ).
            mrg = self._PNP_INTERIOR_MARGIN
            interior = (
                (j2d[..., 0] > mrg) & (j2d[..., 0] < 1.0 - mrg)
                & (j2d[..., 1] > mrg) & (j2d[..., 1] < 1.0 - mrg)
            )
            m = ((z_raw > self._PNP_Z_NEAR) & interior).to(cam_raw.dtype)

        eps = 1e-8
        if ray_mode:
            # ray bearings replace (u-cx)/fx; k_u=1/z, a_u=bx - c_x/z (no K).
            inv_z = 1.0 / z_j
            k_u = inv_z
            a_u = bx - c_x / z_j
            k_v = inv_z
            a_v = by - c_y / z_j
        else:
            k_u = fx / z_j
            a_u = (u_px - cx) - fx * c_x / z_j
            k_v = fy / z_j
            a_v = (v_px - cy) - fy * c_y / z_j
        tx = (m * k_u * a_u).sum(-1) / (m * k_u * k_u).sum(-1).clamp(min=eps)
        ty = (m * k_v * a_v).sum(-1) / (m * k_v * k_v).sum(-1).clamp(min=eps)
        pnp_ct = torch.cat([tx.unsqueeze(-1), ty.unsqueeze(-1), tz], dim=-1)

        with torch.no_grad():
            # Row acceptance: enough interior votes AND the projected canonical
            # joints actually fit joints_2d after the shift (scale-aware RMS).
            n_int = m.sum(-1)                                        # (N, slots)
            if ray_mode:
                # accept gate in bearing space, scaled to pseudo-pixels by a
                # nominal focal (img_w) so the px thresholds keep their meaning.
                f_ref = img_w
                u_obs = bx * f_ref
                v_obs = by * f_ref
                u_fit = (c_x + tx.unsqueeze(-1)) / z_j * f_ref
                v_fit = (c_y + ty.unsqueeze(-1)) / z_j * f_ref
            else:
                u_obs, v_obs = u_px, v_px
                u_fit = fx * (c_x + tx.unsqueeze(-1)) / z_j + cx
                v_fit = fy * (c_y + ty.unsqueeze(-1)) / z_j + cy
            r2 = (u_fit - u_obs) ** 2 + (v_fit - v_obs) ** 2
            rms = ((r2 * m).sum(-1) / n_int.clamp(min=1.0)).sqrt()
            big = torch.tensor(1e9, device=m.device, dtype=m.dtype)
            u_max = torch.where(m > 0, u_obs, -big).amax(-1)
            u_min = torch.where(m > 0, u_obs, big).amin(-1)
            v_max = torch.where(m > 0, v_obs, -big).amax(-1)
            v_min = torch.where(m > 0, v_obs, big).amin(-1)
            diag = ((u_max - u_min) ** 2 + (v_max - v_min) ** 2).clamp(min=0.0).sqrt()
            thresh = torch.maximum(
                self._PNP_RESID_FRAC * diag,
                torch.tensor(self._PNP_RESID_MIN_PX, device=m.device, dtype=m.dtype),
            )
            accept = (n_int >= self._PNP_MIN_INTERIOR) & (rms <= thresh)  # (N, slots)

        # Selected-branch gradients only: where() multiplies each branch's grad
        # by its (constant) selection mask.
        return torch.where(accept.unsqueeze(-1), pnp_ct, inv_ct)

    @staticmethod
    def _build_intrinsics_params(
        intrinsics_list: list[dict] | None,
        *,
        B: int,
        n_video_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if intrinsics_list is None:
            raise ValueError("intrinsics_list is required")
        values = []
        for b_idx in range(B):
            intr = intrinsics_list[b_idx]
            per_frame = intr.get("per_frame")
            img_w = float(intr["image_width"])
            img_h = float(intr["image_height"])
            if per_frame is not None:
                if len(per_frame) != n_video_frames:
                    raise ValueError("ACE camera and output frame timelines differ")
                for f_idx in range(n_video_frames):
                    pf = per_frame[f_idx]
                    values.append([pf["fx"], pf["fy"], pf["cx"], pf["cy"], img_w, img_h])
            else:
                row = [intr["fx"], intr["fy"], intr["cx"], intr["cy"], img_w, img_h]
                values.extend(row for _ in range(n_video_frames))
        return torch.tensor(values, device=device, dtype=dtype)
