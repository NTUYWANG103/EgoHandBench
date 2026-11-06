"""GeoDiT — the composed end-to-end architecture.

Wan2.2-5B backbone (multi-head surgery) -> feature taps -> one projector per
tap layer -> bimanual MANO. With a frozen backbone several taps can be trained
simultaneously (one independent projector each) to rank layers; the released
configuration keeps a single tap and lets LoRA unfreeze the backbone.

The projector is ``MemorySegmentBetasTransformer`` with
``latent_channels = dit.dim (3072)`` — a ``[B, C, F_lat, H, W]`` contract, so
no bridge module is needed (the projector's own ``input_proj`` Linear is the
bridge).
"""

from __future__ import annotations

import inspect
import os

import torch
import torch.nn as nn

from baselines.ace_ego_hand.ace_ego_hand.archs.projector.memory_projector import MemorySegmentBetasTransformer
from baselines.ace_ego_hand.ace_ego_hand.archs.wan_backbone import (
    _VIDEOX_ROOT,
    WanBackbone,
    fold_tokens,
    fold_tokens_pixelshuffle,
)


def _build_flow_scheduler(videox_config: str | None):
    """The exact FlowMatchEulerDiscreteScheduler the backbone was trained with
    (config scheduler_kwargs: 1000 steps, shift=5.0). D-mode needs the same
    σ↔timestep mapping so the DiT's adaLN conditioning stays in-distribution."""
    from omegaconf import OmegaConf
    from diffusers import FlowMatchEulerDiscreteScheduler

    if videox_config is None:
        videox_config = os.path.join(_VIDEOX_ROOT, "config/wan2.2/wan_civitai_5b.yaml")
    sk = OmegaConf.to_container(OmegaConf.load(videox_config)["scheduler_kwargs"])
    keep = set(inspect.signature(FlowMatchEulerDiscreteScheduler.__init__).parameters)
    return FlowMatchEulerDiscreteScheduler(**{k: v for k, v in sk.items() if k in keep})


def load_projector_warmstart(projector: nn.Module, ckpt_path: str) -> tuple[int, int]:
    """Warm-start a projector from a prior checkpoint, skipping shape mismatches.

    A projector trained on render pixels (latent_channels=9) transfers to the
    3072-d DiT feature input except for ``input_proj.*`` (and anything else
    whose shape changed), which stays fresh-init. Returns (n_loaded, n_skipped).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    own = projector.state_dict()
    keep, skipped = {}, []
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            keep[k] = v
        else:
            skipped.append(k)
    missing = [k for k in own if k not in keep]
    projector.load_state_dict(keep, strict=False)
    print(f"[projector-warmstart] loaded {len(keep)}/{len(own)} tensors; "
          f"skipped(shape/absent)={len(skipped)} fresh={len(missing)} "
          f"(fresh incl. {sorted(missing)[:4]}...)")
    return len(keep), len(skipped)



def _reject_non_pinhole(intr: dict) -> None:
    """Fail loudly on a camera this release cannot model.

    The released checkpoints were trained on pinhole datasets only, and the
    fisheye ray-field paths (OpenCV-KB and MEI unified-omnidirectional) are not
    part of this release. Silently falling through to the pinhole formula is the
    dangerous outcome, not the error: on a wide-FOV MEI capture the pinhole
    unprojection collapses the horizontal bearing by roughly 7x and the
    predictions come out plausible-looking and wrong. So refuse instead.
    """
    if intr.get("fisheye") or str(intr.get("model", "")).lower() in ("mei", "kb", "fisheye"):
        raise NotImplementedError(
            "These intrinsics describe a fisheye camera (model="
            f"{intr.get('model')!r}, fisheye={bool(intr.get('fisheye'))}). This "
            "release ships the pinhole ray field only, and the published "
            "checkpoints do not support native fisheye input. Undistort to a "
            "pinhole view first and pass the pinhole intrinsics."
        )

class GeoDiT(nn.Module):
    def __init__(
        self,
        model_root: str,
        tap_layers: list[int],
        projector_kwargs: dict,
        caption_embed_path: str,
        videox_config: str | None = None,
        mh_expand: bool = True,
        train_backbone: bool = False,
        gradient_checkpointing: bool = False,
        projector_warmstart: str | None = None,
        lora: dict | None = None,
        emode_t: float = 1000.0,
        tap_grid: str = "patch",
        aux_geo: bool = False,
        feature_mode: str = "emode",
        nv_sigma: float = 0.5,
        geo_control_mods: tuple = (),
        geo_control_source: str = "render3",
        raymap: bool = False,
        self_ray_pe: bool = False,
        self_ray_decode: bool = False,
        mixed_pnp_detach_mano: bool = False,
        gen_render: bool = False,
        gen_render_detach: bool = False,
        gen_concat_gt: bool = False,
        gen_overlay: bool = False,
    ) -> None:
        super().__init__()
        self._videox_config = videox_config
        self.noise_scheduler = None  # lazily built on the first D-mode step
        # Feature-extraction regime. "emode" = x=0 render-generation (multi-head
        # model). "noised_video" = the released regime: the DiT
        # denoises the REAL video (video-in-x), single forward, tap mid-block —
        # requires mh_expand=False.
        self.feature_mode = str(feature_mode)
        self.nv_sigma = float(nv_sigma)
        # Optional GT geometry control, disabled in the released configuration:
        # inject GT geometry-render latents (render3 = depth[0:48]⊕joints[48:96]⊕
        # normal[96:144]) into the otherwise all-zero noised_video control tensor.
        # These channels are free and do not compete with the RGB input (in x).
        # () = off.
        self.geo_control_mods = tuple(geo_control_mods)
        # GT-overlay control: "render3" (default) injects the geometry-render
        # slices above; "overlay" injects the 48ch VAE(normal-overlay) latent (the
        # representation the generator produces) into the primary [0:48] control
        # slot instead.
        self.geo_control_source = str(geo_control_source)
        if self.feature_mode == "noised_video" and mh_expand:
            raise ValueError("feature_mode=noised_video needs mh_expand=False (x=48ch video)")
        self.backbone = WanBackbone(
            model_root,
            config_path=videox_config,
            mh_expand=mh_expand,
            gradient_checkpointing=gradient_checkpointing,
            emode_t=emode_t,
        )
        self.tap_layers = [int(x) for x in tap_layers]
        self.train_backbone = bool(train_backbone)
        if self.train_backbone:
            self.backbone.enable_training(**(lora or {}))
        if tap_grid not in ("patch", "pixelshuffle"):
            raise ValueError(f"tap_grid must be patch|pixelshuffle, got {tap_grid!r}")
        self.tap_grid = tap_grid
        pk = dict(projector_kwargs)
        pk.pop("self_ray_decode", None)  # net-level flag; not a projector ctor arg
        base_lc = (self.backbone.dim // 4 if tap_grid == "pixelshuffle"
                   else self.backbone.dim)
        # Generative render head: a fresh Conv3d(base->144) head DECODES the tapped RGB
        # feature into the render super-latent (depth|joints|normal), supervised by
        # render3 (plain latent-L2, NO GT inject), then CONCATENATES its 144ch onto
        # the feature so the projector reads [feat||gen] = base+144. The geometry
        # is generated from RGB, never taken from GT. Off => latent_channels=base.
        self.gen_render = bool(gen_render)
        # Variant: detach the generated render BEFORE the concat, so the projector
        # reads the ACTUAL generated geometry but MANO gradient does NOT flow back
        # into the gen head (which then trains ONLY on the render loss = a faithful
        # render generator). False = end-to-end.
        self.gen_render_detach = bool(gen_render_detach)
        # Optional GT render concat, disabled in the released configuration:
        # bypass gen_render_heads and concat the batch's detached GT render3
        # latents (aligned to the tap grid exactly as the gen output would be)
        # so the projector reads [feat||GT-render3]. False = off.
        self.gen_concat_gt = bool(gen_concat_gt)
        # Generator-only mode: clean latent + t=0, FULL 30-block forward + the pretrained 48ch head,
        # prediction = -head_out (negate-v), supervised by the RGB+normal-overlay
        # latent (coverage-weighted). No projector/MANO in the train step (the
        # projector modules still exist so ckpt schema/eval plumbing is unchanged).
        self.gen_overlay = bool(gen_overlay)
        if self.gen_overlay:
            if self.feature_mode != "noised_video" or self.nv_sigma != 0.0:
                raise ValueError("gen_overlay needs feature_mode=noised_video + "
                                 "nv_sigma=0.0 (clean latent, t=0)")
            if mh_expand:
                raise ValueError("gen_overlay needs mh_expand=false "
                                 "(the PRETRAINED 48ch video head)")
        self._gen_in = base_lc
        pk["latent_channels"] = base_lc + (
            144 if (self.gen_render or self.gen_concat_gt) else 0)
        self.projectors = nn.ModuleDict(
            {f"l{i}": MemorySegmentBetasTransformer(**pk) for i in self.tap_layers}
        )
        # Full K-free: also decode cam_trans from the PREDICTED ray field
        # (sample rays at the wrist uv -> tx=(dx/dz)*tz) instead of back-projecting
        # with K. Requires self_ray_pe (pred_rays are already threaded to the
        # projector). Off => the K-based inv_proj/mixed_pnp decode is unchanged.
        self.self_ray_decode = bool(self_ray_decode)
        for _p in self.projectors.values():
            _p.self_ray_decode = self.self_ray_decode
        # Under self_ray_decode the mixed_pnp cam_trans decode
        # is a differentiable fn of UNDETACHED go/hp/betas/joints2d, so the
        # cam_trans loss backprops into the MANO articulation heads (the joints-only
        # cam_trans loss detaches them by design; the full-MANO in-decode path did
        # not). Where the predicted-ray bearings are inaccurate this biases the
        # gradient into hp/betas and root-relative PA degrades.
        # This flag detaches go/hp/betas/joints2d inside the mixed_pnp solve so
        # cam_trans trains only the log-z/cam head + raymap. Off => byte-identical.
        self.mixed_pnp_detach_mano = bool(mixed_pnp_detach_mano)
        for _p in self.projectors.values():
            _p.mixed_pnp_detach_mano = self.mixed_pnp_detach_mano
        # Auxiliary geometry head: a per-tap 1x1 conv that decodes the tapped feature back to the
        # render super-latent (144ch = depth⊕joints⊕normal). Forcing L* to be
        # decodable to the geometry makes the learned feature carry the same
        # orientation/depth signal the geometry renders give explicitly.
        self.aux_geo = bool(aux_geo)
        if self.aux_geo:
            self.aux_geo_heads = nn.ModuleDict(
                {f"l{i}": nn.Conv3d(base_lc,
                                    self.backbone.noisy_channels, kernel_size=1)
                 for i in self.tap_layers})
        # Wan-side dense CAMERA raymap head (GenCeption-style): decode the tapped DiT
        # feature grid -> per-token 3D ray DIRECTION. The DiT never sees K (ray-PE is
        # downstream in the projector), so this is a camera-from-image auxiliary
        # head that regularizes the features toward camera awareness.
        # The last conv is zero-initialized, so the head starts as a no-op.
        self.raymap = bool(raymap)
        # Input-side K-free: feed the PREDICTED ray field (raymap head) into the projector
        # ray-PE instead of GT-K rays -> the projector never sees intrinsics on the
        # input side (deployable K-free). Requires raymap=True. Off => byte-identical.
        self.self_ray_pe = bool(self_ray_pe)
        if self.raymap:
            self.raymap_heads = nn.ModuleDict(
                {f"l{i}": nn.Conv3d(base_lc, 3, kernel_size=1)
                 for i in self.tap_layers})
            for h in self.raymap_heads.values():
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)
        if self.gen_render:
            # fresh render-generation head: base_lc -> 144ch super-latent (depth|joints|
            # normal); its output is CONCATENATED onto feat before the projector reads it.
            self.gen_render_heads = nn.ModuleDict(
                {f"l{i}": nn.Conv3d(self._gen_in, 144, kernel_size=1)
                 for i in self.tap_layers})
        if projector_warmstart:
            for name, proj in self.projectors.items():
                load_projector_warmstart(proj, projector_warmstart)
        emb = torch.load(caption_embed_path, map_location="cpu", weights_only=False)
        if isinstance(emb, dict):
            emb = emb["embed"]
        self.register_buffer("caption_embed", emb.to(torch.float32), persistent=False)

    # ------------------------------------------------------------------ #
    _RENDER3_CHAN = {"depth": (0, 48), "joints": (48, 96), "normal": (96, 144)}

    def _build_geo_control(self, geo_lat, B, F, H, W, dev, cdtype):
        """Assemble the 100-ch control from GT render3 modality slices.
        1st requested mod → [0:48] (primary latent slot), 2nd → [52:100] (masked-
        latent slot); the 4 mask channels [48:52] stay zero."""
        y = geo_lat.new_zeros(B, self.backbone.control_channels, F, H, W,
                              dtype=cdtype, device=dev)
        if self.geo_control_source == "overlay":
            # geo_lat is the 48ch VAE(normal-overlay) latent -> primary [0:48] slot
            # (the same slot RGB VAE latents occupy in build_control, so the
            # pretrained control adapter reads it natively). normal-only: [52:100]=0.
            y[:, 0:48] = geo_lat[:, 0:48].to(dev, cdtype)
            return y
        slots = [(0, 48), (52, 100)]
        for i, mod in enumerate(self.geo_control_mods[:2]):
            lo, hi = self._RENDER3_CHAN[mod]
            s0, s1 = slots[i]
            y[:, s0:s1] = geo_lat[:, lo:hi].to(dev, cdtype)
        return y


    def extract_features(
        self,
        ctrl_lat: torch.Tensor,
        visible_mask_lat: torch.Tensor | None = None,
        run_head: bool = False,
        geo_lat: torch.Tensor | None = None,
        span_mask: torch.Tensor | None = None,
        span_mask_channel: bool = True,
    ):
        """Single-forward feature tap. `emode`: x=zeros(render), control=RGB.
        `noised_video`: x=(1-σ)·VAE(video)+σ·eps, control=0.

        Span-dropout: ``span_mask`` (B, F_lat) bool marks DROPPED latent
        frames — the input video latent is zeroed there and (if
        ``span_mask_channel``) the free 4-ch mask slot [48:52] of the control is
        set to 1 (the RGB-visibility interface: visible→0, dropped→1).
        None => span-dropout off. Loss supervision on dropped
        frames is unchanged (GT exists) => the model must inpaint from context."""
        bb = self.backbone
        B, _, F, H, W = ctrl_lat.shape
        dev = bb.dit.patch_embedding.weight.device
        cdtype = bb.compute_dtype
        if self.feature_mode == "noised_video":
            # DiT denoises the real video (video is x), tapped mid-block. Fresh eps
            # per forward (regularizer, matching a live feature extractor). σ ties t.
            x0 = ctrl_lat.to(dev, cdtype)
            if span_mask is not None:
                keep = (~span_mask.to(dev)).to(cdtype).view(B, 1, F, 1, 1)
                x0 = x0 * keep                       # dropped spans: video latent → 0
            sig = float(self.nv_sigma)
            x = (1.0 - sig) * x0 + sig * torch.randn_like(x0)
            if self.geo_control_mods and geo_lat is not None:
                y = self._build_geo_control(geo_lat, B, F, H, W, dev, cdtype)
            else:
                y = ctrl_lat.new_zeros(B, bb.control_channels, F, H, W, dtype=cdtype, device=dev)
            if span_mask is not None and span_mask_channel:
                m = span_mask.to(dev).to(cdtype).view(B, 1, F, 1, 1)
                y[:, 48:52] = y[:, 48:52] * 0 + m    # mask slot: dropped → 1
            t = torch.full((B,), sig * 1000.0, device=dev, dtype=torch.float32)
        else:  # "emode": x=0 render slots, RGB in the control channels
            x = ctrl_lat.new_zeros(B, bb.noisy_channels, F, H, W, dtype=cdtype, device=dev)
            y = bb.build_control(ctrl_lat.to(dev, cdtype), visible_mask_lat)
            t = torch.full((B,), bb.t_emode, device=dev, dtype=torch.float32)
        ctx = [self.caption_embed.to(dev, cdtype)] * B
        with torch.amp.autocast(device_type="cuda", dtype=cdtype):
            if self.train_backbone:
                taps, head_out, grid = bb.forward_features(
                    x, y, ctx, t, tap_layers=tuple(self.tap_layers), run_head=run_head)
            else:
                with torch.no_grad():
                    taps, head_out, grid = bb.forward_features(
                        x, y, ctx, t, tap_layers=tuple(self.tap_layers), run_head=run_head)
        return taps, head_out, grid

    def forward(self, mode: str = "emode", **kwargs) -> dict:
        """Dispatch entry so DDP can wrap the trainable net: DDP only forwards
        ``__call__`` -> ``forward``, but the trainer needs emode/dmode. Routing
        the loss-producing forward through here lets DDP's reducer see which
        trainable params participated each step (find_unused_parameters) and fire
        the gradient all-reduce. Single-GPU code still calls forward_emode/_dmode
        directly (byte-identical); only the DDP path goes through this dispatch."""
        if mode == "emode":
            return self.forward_emode(**kwargs)
        if mode == "dmode":
            return self.forward_dmode(**kwargs)
        if mode == "gen_overlay":
            return self.forward_gen_overlay(**kwargs)
        if mode == "e2e":
            return self.forward_e2e(**kwargs)
        raise ValueError(f"unknown forward mode {mode!r}")

    def forward_e2e(self, ctrl_lat: torch.Tensor, n_video_frames: int,
                    intrinsics_list: list[dict], e2e_detach: bool = True) -> dict:
        """e2e generate-then-inject in ONE forward (so DDP wraps a single
        forward->backward; two separate ddp_net forwards break the reducer).
        Pass1: generate the normal-overlay (all 30 blocks + head, negate-v).
        Pass2: inject the generated overlay -> control[0:48] -> projector -> MANO.
        Returns forward_emode's dict + ``head_out`` (Pass1, for the gen loss)."""
        _, gen_head, _ = self.extract_features(ctrl_lat, run_head=True)   # Pass 1
        geo_lat = (-gen_head).float()
        if e2e_detach:
            geo_lat = geo_lat.detach()
        out = self.forward_emode(                                         # Pass 2
            ctrl_lat, n_video_frames, intrinsics_list, run_head=False, geo_lat=geo_lat)
        out["gen_head"] = gen_head
        return out

    def forward_gen_overlay(self, ctrl_lat: torch.Tensor) -> dict:
        """Generator-only forward: x = clean VAE(video)
        latent, t=0 (enforced by the gen_overlay __init__ guard), FULL forward
        through all 30 blocks + the PRETRAINED 48ch head. The caller negates the
        head output (``pred = -head_out``, negate-v) and scores it against the
        overlay latent — no projector, no MANO."""
        _, head_out, _ = self.extract_features(ctrl_lat, run_head=True)
        return {"head_out": head_out}

    def forward_emode(
        self,
        ctrl_lat: torch.Tensor,
        n_video_frames: int,
        intrinsics_list: list[dict],
        visible_mask_lat: torch.Tensor | None = None,
        run_head: bool = False,
        geo_lat: torch.Tensor | None = None,
        span_mask: torch.Tensor | None = None,
        span_mask_channel: bool = True,
    ) -> dict:
        """Extraction forward -> per-tap projector predictions.

        Returns ``{"preds": {layer: list-of-per-sample dicts}, "head_out": ...}``.
        """
        taps, head_out, grid = self.extract_features(
            ctrl_lat, visible_mask_lat=visible_mask_lat, run_head=run_head, geo_lat=geo_lat,
            span_mask=span_mask, span_mask_channel=span_mask_channel)
        preds = {}
        aux_geo: dict = {}
        raymap: dict = {}
        gen_render: dict = {}
        fold = fold_tokens_pixelshuffle if self.tap_grid == "pixelshuffle" else fold_tokens
        for layer in self.tap_layers:
            feat = fold(taps[layer], grid).float()
            if not self.train_backbone:
                feat = feat.detach()
            if self.aux_geo:
                aux_geo[layer] = self.aux_geo_heads[f"l{layer}"](feat)
            if self.raymap:
                raymap[layer] = self.raymap_heads[f"l{layer}"](feat)   # (B,3,F,H,W)
            if self.gen_concat_gt:
                # Concat GT render3 (DETACHED) in place of the generated
                # 144ch — aligned to the tap grid exactly as the gen output's
                # loss target was (nearest interpolate). gen_render_heads (if
                # built) are BYPASSED: no forward, no gen_render loss entry.
                if geo_lat is None:
                    raise ValueError("gen_concat_gt=true needs render3 in the batch "
                                     "(set data.with_render3 / load render3 at eval)")
                gt3 = geo_lat.detach().float().to(feat.device)
                if gt3.shape[2:] != feat.shape[2:]:
                    gt3 = torch.nn.functional.interpolate(
                        gt3, size=feat.shape[2:], mode="nearest")
                feat = torch.cat([feat, gt3.to(feat.dtype)], dim=1)    # (B, base+144, F,H,W)
            elif self.gen_render:
                pr = self.gen_render_heads[f"l{layer}"](feat)          # (B,144,F,H,W)
                gen_render[layer] = pr                                 # undetached: render loss trains the head
                pr_read = pr.detach() if self.gen_render_detach else pr
                feat = torch.cat([feat, pr_read], dim=1)               # (B, base+144, F,H,W)
            # Mean the predicted ray field over frames (camera ~constant per clip)
            # -> (B,3,H,W); feed to the projector ray-PE in place of GT-K rays. This
            # drops K on the INPUT side; dropping it in the cam_trans decode too is
            # self_ray_decode. The projector prefers pred_rays for the ray-PE.
            pr_rays = raymap[layer].mean(dim=2) if (self.self_ray_pe and layer in raymap) else None
            preds[layer] = self.projectors[f"l{layer}"].forward_batched(
                feat, n_video_frames=n_video_frames,
                intrinsics_list=intrinsics_list, pred_rays=pr_rays)
        return {"preds": preds, "head_out": head_out, "aux_geo": aux_geo,
                "raymap": raymap, "gen_render": gen_render}

    # ---------------------------------------------------------------- D-mode
    def _sample_flow_noise(self, x0: torch.Tensor, weighting_scheme: str = "none"):
        """Flow-matching sampling: draw timesteps by ``weighting_scheme`` density, map to the
        scheduler's shifted σ, build ``x_σ = (1-σ)x0 + σε`` and target ``v* = ε-x0``."""
        if self.noise_scheduler is None:
            self.noise_scheduler = _build_flow_scheduler(self._videox_config)
        sched = self.noise_scheduler
        B = x0.shape[0]
        n_ts = int(sched.config.num_train_timesteps)
        try:
            from diffusers.training_utils import compute_density_for_timestep_sampling
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme, batch_size=B,
                logit_mean=0.0, logit_std=1.0, mode_scale=1.29)
        except Exception:
            u = torch.rand(B)
        idx = (u * n_ts).long().clamp(0, n_ts - 1)
        timesteps = sched.timesteps[idx].to(x0.device)                    # (B,) 0..1000
        sigmas = sched.sigmas[idx].to(x0.device, x0.dtype).view(B, *([1] * (x0.ndim - 1)))
        noise = torch.randn_like(x0)
        noisy = (1.0 - sigmas) * x0 + sigmas * noise
        target = noise - x0
        try:
            from diffusers.training_utils import compute_loss_weighting_for_sd3
            weighting = compute_loss_weighting_for_sd3(weighting_scheme, sigmas)
        except Exception:
            weighting = torch.ones_like(sigmas)
        return noisy, timesteps, target, weighting

    def forward_dmode(self, ctrl_lat: torch.Tensor, render3: torch.Tensor,
                      weighting_scheme: str = "none") -> dict:
        """Diffusion mode: standard flow-matching over the render super-latent,
        conditioned on the RGB control (render loss only — no projector/MANO).
        Predicts velocity v; caller compares to ``target`` with ``weighting``."""
        bb = self.backbone
        dev = bb.dit.patch_embedding.weight.device
        cdtype = bb.compute_dtype
        x0 = render3.to(dev, torch.float32)
        noisy, timesteps, target, weighting = self._sample_flow_noise(x0, weighting_scheme)
        y = bb.build_control(ctrl_lat.to(dev, cdtype))
        ctx = [self.caption_embed.to(dev, cdtype)] * x0.shape[0]
        with torch.amp.autocast(device_type="cuda", dtype=cdtype):
            _, head_out, _ = bb.forward_features(
                noisy.to(cdtype), y, ctx, timesteps, tap_layers=(), run_head=True)
        return {"head_out": head_out, "target": target, "weighting": weighting}

    def _fresh_head_params(self):
        """Fresh-init modules that train at the projector LR (projectors + aux-geo)."""
        params = list(self.projectors.parameters())
        if self.aux_geo:
            params += list(self.aux_geo_heads.parameters())
        if self.raymap:
            params += list(self.raymap_heads.parameters())
        if self.gen_render:
            params += list(self.gen_render_heads.parameters())
        return params

    def trainable_parameters(self):
        params = self._fresh_head_params()
        if self.train_backbone:
            params += [p for p in self.backbone.parameters() if p.requires_grad]
        return params

