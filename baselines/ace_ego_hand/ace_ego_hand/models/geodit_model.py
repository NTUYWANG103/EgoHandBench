"""GeoDitModel — the inference wrapper around GeoDiT.

Builds the network from an option dict (backbone + per-tap projectors), attaches
the frozen MANO models the mixed_pnp translation decode needs, and loads a
released checkpoint via ``load_inference``. Prediction itself runs through
``ace_ego_hand.inference.predict_video``.
"""

from __future__ import annotations

import torch

from baselines.ace_ego_hand.ace_ego_hand.archs.geodit_arch import GeoDiT
from baselines.ace_ego_hand.ace_ego_hand.mano_utils import setup_mano_models


class GeoDitModel:
    def __init__(self, opt: dict, device: torch.device) -> None:
        self.opt = opt
        self.device = device
        p = opt["paths"]
        bb = opt.get("backbone", {})
        self.net = GeoDiT(
            model_root=p["model_root"],
            tap_layers=bb.get("tap_layers", [16]),
            projector_kwargs=opt["projector"],
            caption_embed_path=p["caption_embed"],
            videox_config=p.get("videox_config"),
            mh_expand=bb.get("mh_expand", True),
            train_backbone=bb.get("train_backbone", False),
            gradient_checkpointing=bb.get("gradient_checkpointing", False),
            projector_warmstart=p.get("projector_warmstart"),
            lora=bb.get("lora"),
            emode_t=float(bb.get("emode_t", 1000.0)),
            tap_grid=bb.get("tap_grid", "patch"),
            aux_geo=bb.get("aux_geo", False),
            feature_mode=bb.get("feature_mode", "emode"),
            nv_sigma=float(bb.get("nv_sigma", 0.5)),
            geo_control_mods=tuple(bb.get("geo_control_mods", ())),
            geo_control_source=str(bb.get("geo_control_source", "render3")),
            raymap=bool(bb.get("raymap", False)),
            self_ray_pe=bool(bb.get("self_ray_pe", False)),
            self_ray_decode=bool(bb.get("self_ray_decode", False)),
            mixed_pnp_detach_mano=bool(bb.get("mixed_pnp_detach_mano", False)),
            gen_render=bool(bb.get("gen_render", False)),
            gen_render_detach=bool(bb.get("gen_render_detach", False)),
            gen_concat_gt=bool(bb.get("gen_concat_gt", False)),
            gen_overlay=bool(bb.get("gen_overlay", False)),
        ).to(device)
        self.net.backbone.eval()

        self.mano_models = setup_mano_models(opt, device)
        # The mixed_pnp cam_trans decode runs a differentiable MANO forward
        # INSIDE the projector — hand it the frozen models ({False: left,
        # True: right}, already on `device`). Plain-dict attach, so they never
        # enter the projector's state_dict.
        if str(opt["projector"].get("cam_trans_decode", "inv_proj")) == "mixed_pnp":
            for proj in self.net.projectors.values():
                proj.set_mano_models(self.mano_models)
        if p.get("probe_warmstart"):
            self._load_probe_projectors(p["probe_warmstart"])
        self.params = self.net.trainable_parameters()
        # full_warmstart: preload weights (projectors + backbone deltas + aux
        # heads) from a checkpoint named in the config, same format as --ckpt.
        if p.get("full_warmstart"):
            self.load_inference(p["full_warmstart"])   # paths already absolutized
            print(f"[model] full_warmstart (weights only) from {p['full_warmstart']}")
        n_train = sum(x.numel() for x in self.params)
        print(f"[model] taps={self.net.tap_layers} trainable={n_train/1e6:.2f}M "
              f"backbone(frozen)={sum(x.numel() for x in self.net.backbone.parameters())/1e9:.2f}B")

    def _load_probe_projectors(self, path: str) -> None:
        """Warm-start projectors from a projector-only checkpoint (per-tap key match)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["projectors"]
        own = self.net.projectors.state_dict()
        keep = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        self.net.projectors.load_state_dict(keep, strict=False)
        print(f"[probe-warmstart] loaded {len(keep)}/{len(own)} projector tensors from {path}")


    # ---------------------------------------------------------------- ckpt

    def load_inference(self, path: str) -> None:
        """Load projectors (+ backbone trainable deltas, if present) for eval."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # warmstart_strict (default True = byte-identical): set False in a config
        # to warm-start a projector whose architecture differs from the ckpt (e.g.
        # flat vs interp2d PE, hand vs joint queries). Shared tensors transfer and
        # the parameters of the differing components are freshly initialized.
        strict = bool(self.opt.get("warmstart_strict", True))
        res = self.net.projectors.load_state_dict(ckpt["projectors"], strict=strict)
        if not strict:
            print(f"[model] non-strict projector warmstart: "
                  f"{len(res.missing_keys)} missing (fresh-init) / "
                  f"{len(res.unexpected_keys)} unexpected (ignored)")
        if self.net.raymap and "raymap_heads" in ckpt:
            self.net.raymap_heads.load_state_dict(ckpt["raymap_heads"])
        if self.net.gen_render and "gen_render_heads" in ckpt:
            self.net.gen_render_heads.load_state_dict(ckpt["gen_render_heads"])
        bt = ckpt.get("backbone_trainable")
        if bt:
            own = dict(self.net.backbone.dit.named_parameters())
            n = 0
            for k, v in bt.items():
                if k in own and own[k].shape == v.shape:
                    own[k].data.copy_(v.to(own[k].dtype))
                    n += 1
            print(f"[model] loaded {n}/{len(bt)} backbone trainable tensors")
        self.net.projectors.eval()

