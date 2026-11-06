"""Wan2.2-Fun-5B-Control backbone wrapper.

Loads the VideoX-Fun framework from ``third_party`` (see the top-level README), applies the
multi-head surgery (patch_embedding 148->244, head 48->144) that turns the
single-latent DiT into a 3-modality geometry model, and exposes
:meth:`WanBackbone.forward_features` — a faithful minimal re-implementation of
``Wan2_2Transformer3DModel.forward`` that

* returns intermediate block hidden states (the projector taps), and
* composes with per-block gradient checkpointing: taps are captured *between*
  checkpoint segments, so gradients flow from the MANO loss through the taps
  into the (LoRA) blocks with bounded activation memory.

Dropped relative to the stock forward (deliberate): full_ref/subject_ref
tokens, ti2v first-frame anchoring (per-token timesteps), y_camera, TeaCache,
sequence parallelism.

E-mode (extraction) convention: ``x = zeros(n_mod*48 ch)``, scalar
``t = t_max`` (sigma = 1), control ``y = [VAE(RGB) 48 | mask 4 | masked 48]``
with the mask/masked channels zero when RGB is fully visible (the trainer's
``t2v_flag=0`` regime, i.e. the in-distribution default). With zeros-x the
surgery is output-identical to the unexpanded base model
(``W_noisy @ 0 = 0``), so features of the frozen backbone equal the base model's.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VIDEOX_ROOT = os.path.join(_ROOT, "third_party")

from omegaconf import OmegaConf  # noqa: E402
from baselines.ace_ego_hand.third_party.videox_fun.models.wan_transformer3d import (  # noqa: E402
    Wan2_2Transformer3DModel,
    sinusoidal_embedding_1d,
)

#: Modality order of the 144-ch geometry super-latent: color-coded depth,
#: L/R-colored joint skeleton, surface normals.
MODALITIES = ("depth_banana", "joints_clean", "normal_banana")
LATENT_C = 48  # Wan2.2 VAE latent channels (one modality)


def mh_expand_dit(transformer3d: nn.Module, n_mod: int = 3) -> None:
    """Multi-head surgery.

    Expands patch_embedding in-ch (148 -> n_mod*48 + 100) and head out-ch
    (48 -> n_mod*48); init tiled from the pretrained weights.
    """
    pe = transformer3d.patch_embedding
    dim, k, s = pe.out_channels, pe.kernel_size, pe.stride
    with torch.no_grad():
        cm = transformer3d.head.out_dim  # 48
        old_in = pe.in_channels          # 148 = 48 noisy + 100 control
        new_in = n_mod * cm + (old_in - cm)
        new_pe = nn.Conv3d(new_in, dim, kernel_size=k, stride=s,
                           dtype=pe.weight.dtype, device=pe.weight.device)
        for m in range(n_mod):
            new_pe.weight[:, m * cm:(m + 1) * cm] = pe.weight[:, 0:cm]
        new_pe.weight[:, n_mod * cm:] = pe.weight[:, cm:old_in]
        new_pe.bias.copy_(pe.bias)
        transformer3d.patch_embedding = new_pe

        head = transformer3d.head
        ow, ob = head.head.weight, head.head.bias
        pk = ow.shape[0] // cm  # prod(patch_size) = 4
        new_head = nn.Linear(ow.shape[1], pk * n_mod * cm,
                             dtype=ow.dtype, device=ow.device)
        new_head.weight.copy_(
            ow.view(pk, cm, -1).repeat(1, n_mod, 1).reshape(pk * n_mod * cm, -1))
        new_head.bias.copy_(ob.view(pk, cm).repeat(1, n_mod).reshape(-1))
        head.head = new_head
        head.out_dim = n_mod * cm
        transformer3d.out_dim = n_mod * cm


class WanBackbone(nn.Module):
    """Frozen-by-default Wan2.2-5B DiT with feature taps.

    Args:
        model_root: checkpoint dir (``Wan2.2-Fun-5B-Control``).
        config_path: VideoX-Fun yaml (``config/wan2.2/wan_civitai_5b.yaml``).
        mh_expand: apply the multi-head surgery (144-ch super-latent I/O).
        dtype: parameter/compute dtype (bf16).
        gradient_checkpointing: per-block ckpt for backbone training.
    """

    def __init__(
        self,
        model_root: str,
        config_path: str | None = None,
        mh_expand: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = False,
        emode_t: float = 1000.0,
    ) -> None:
        super().__init__()
        if config_path is None:
            config_path = os.path.join(_VIDEOX_ROOT, "config/wan2.2/wan_civitai_5b.yaml")
        config = OmegaConf.load(config_path)
        sub_path = config["transformer_additional_kwargs"].get(
            "transformer_low_noise_model_subpath", "./")
        self.dit = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(model_root, sub_path),
            transformer_additional_kwargs=OmegaConf.to_container(
                config["transformer_additional_kwargs"]),
        ).to(dtype)
        self.n_mod = len(MODALITIES) if mh_expand else 1
        if mh_expand:
            mh_expand_dit(self.dit, n_mod=self.n_mod)
        self.dit.requires_grad_(False)
        self.dit.eval()
        self._grad_ckpt = bool(gradient_checkpointing)
        self.compute_dtype = dtype
        self.noisy_channels = self.n_mod * LATENT_C          # 144 (or 48)
        self.control_channels = self.dit.patch_embedding.in_channels - self.noisy_channels  # 100
        self.dim = self.dit.dim                              # 3072
        self.num_layers = len(self.dit.blocks)               # 30
        # E-mode timestep conditioning (0..1000 scale). Default 1000 = the
        # max-sigma end (sigma=1, "global layout" adaLN regime); lower t (e.g.
        # 100) is the "fine detail" regime — with x=0 always, t is purely an
        # adaLN conditioning choice.
        self.t_emode = float(emode_t)

    # --------------------------------------------------------------- LoRA
    #: every Linear inside a WanAttentionBlock (verified by introspection)
    LORA_TARGETS = ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
                    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
                    "ffn.0", "ffn.2")

    def enable_training(self, lora_rank: int = 64, lora_alpha: int = 64,
                        lora_dropout: float = 0.0,
                        train_patch_embed: bool = True, train_head: bool = True) -> None:
        """Unlock the backbone: inject LoRA into all 30 blocks; fully train the (expanded)
        patch_embedding + head. LoRA/patch/head params are kept fp32 (autocast
        downcasts per-op); the frozen base stays bf16."""
        from peft import LoraConfig, inject_adapter_in_model
        cfg = LoraConfig(r=int(lora_rank), lora_alpha=int(lora_alpha),
                         lora_dropout=float(lora_dropout),
                         target_modules=list(self.LORA_TARGETS))
        self.dit = inject_adapter_in_model(cfg, self.dit)
        n_lora = 0
        for name, p in self.dit.named_parameters():
            if "lora_" in name:
                p.requires_grad_(True)
                p.data = p.data.float()
                n_lora += p.numel()
        if train_patch_embed:
            self.dit.patch_embedding.float().requires_grad_(True)
        if train_head:
            self.dit.head.float().requires_grad_(True)
        n_pe = sum(p.numel() for p in self.dit.patch_embedding.parameters())
        n_hd = sum(p.numel() for p in self.dit.head.parameters())
        print(f"[wan-backbone] LoRA r={lora_rank} a={lora_alpha}: {n_lora/1e6:.1f}M lora"
              f" + patch_embed {n_pe/1e6:.2f}M + head {n_hd/1e6:.2f}M trainable")


    # ------------------------------------------------------------------ utils
    def build_control(
        self,
        ctrl_lat: torch.Tensor,
        visible_mask_lat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble the 100-ch control tensor.

        ``[VAE(RGB) 48 | mask 4 | masked-latent 48]``. Default (all visible):
        mask/masked = zeros — the trainer's dominant ``t2v_flag = 0`` regime.
        Span-dropout: pass ``visible_mask_lat (B, F_lat)`` with 0 on dropped
        latent frames — RGB channels are zeroed there and the 4 mask channels
        flag the drop (the GENMO cond-exists bit, native-channel edition).
        """
        B, C, F, H, W = ctrl_lat.shape
        assert C == LATENT_C, f"control latent must be {LATENT_C}ch, got {C}"
        pad = self.control_channels - LATENT_C  # 52 = 4 mask + 48 masked
        if visible_mask_lat is None:
            extra = ctrl_lat.new_zeros(B, pad, F, H, W)
            return torch.cat([ctrl_lat, extra], dim=1)
        vis = visible_mask_lat.to(ctrl_lat.dtype).view(B, 1, F, 1, 1)
        ctrl = ctrl_lat * vis
        mask4 = (1.0 - vis).expand(B, 4, F, H, W)
        masked48 = ctrl_lat.new_zeros(B, LATENT_C, F, H, W)
        return torch.cat([ctrl, mask4, masked48], dim=1)

    # --------------------------------------------------------------- forward
    def forward_features(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: list[torch.Tensor],
        t: torch.Tensor,
        tap_layers: tuple[int, ...] = (16,),
        run_head: bool = True,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor | None, torch.Tensor]:
        """Minimal faithful re-implementation of the stock forward.

        Args:
            x: noisy/extraction latent ``(B, noisy_channels, F, H, W)``.
            y: control latent ``(B, control_channels, F, H, W)``.
            context: list of B text-embedding tensors ``(L_i, 4096)``.
            t: scalar timesteps ``(B,)`` (0..1000 scale).
            tap_layers: block indices whose OUTPUT hidden state is returned.
            run_head: also run head+unpatchify (needed for the render loss;
                skip it when only taps are needed to save ~1 block worth of time).

        Returns:
            ``(taps, head_out, grid_sizes)`` — taps[i] is ``(B, L, dim)``;
            head_out is ``(B, out_dim, F, H, W)`` or None; grid_sizes ``(B,3)``.
        """
        model = self.dit
        device = model.patch_embedding.weight.device
        if model.freqs.device != device and torch.device(type="meta") != device:
            model.freqs = model.freqs.to(device)

        B, _, F, H, W = x.shape
        xs = torch.cat([x, y], dim=1)                          # channel concat
        xs = model.patch_embedding(xs)                          # (B, dim, F, H/2, W/2)
        grid = torch.tensor(xs.shape[2:], dtype=torch.long)
        grid_sizes = grid.unsqueeze(0).repeat(B, 1)
        seq_len = int(grid.prod().item())
        xs = xs.flatten(2).transpose(1, 2)                      # (B, L, dim)
        seq_lens = torch.tensor([seq_len] * B, dtype=torch.long)
        dtype = xs.dtype

        # time embeddings (fp32 island, mirrors the stock forward)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            e = model.time_embedding(
                sinusoidal_embedding_1d(model.freq_dim, t).float())
            e0 = model.time_projection(e).unflatten(1, (6, model.dim))

        # text context: pad each to text_len, embed
        ctx = model.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        context_lens = None

        taps: dict[int, torch.Tensor] = {}
        want = set(int(i) for i in tap_layers)
        max_tap = max(want) if want else -1
        for i, block in enumerate(model.blocks):
            if torch.is_grad_enabled() and self._grad_ckpt:
                xs = torch.utils.checkpoint.checkpoint(
                    block, xs, e0, seq_lens, grid_sizes, model.freqs,
                    ctx, context_lens, dtype, t, use_reentrant=False)
            else:
                xs = block(xs, e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                           freqs=model.freqs, context=ctx,
                           context_lens=context_lens, dtype=dtype, t=t)
            if i in want:
                taps[i] = xs
            if not run_head and i >= max_tap:
                return taps, None, grid_sizes

        if torch.is_grad_enabled() and self._grad_ckpt:
            out = torch.utils.checkpoint.checkpoint(model.head, xs, e, use_reentrant=False)
        else:
            out = model.head(xs, e)
        out = model.unpatchify(out, grid_sizes)
        out = torch.stack(out)                                  # (B, out_dim, F, H, W)
        return taps, out, grid_sizes


def fold_tokens(tokens: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
    """(B, L, D) block hidden states -> (B, D, F, Hp, Wp) grid for the projector.

    Token order is f-major, then h, then w (see ``unpatchify``'s
    ``view(*grid, ...)``), so a plain reshape is exact.
    """
    B, L, D = tokens.shape
    f, h, w = (int(v) for v in grid_sizes[0].tolist())
    assert L == f * h * w, f"token count {L} != grid {f}x{h}x{w}"
    return tokens.view(B, f, h, w, D).permute(0, 4, 1, 2, 3).contiguous()


def fold_tokens_pixelshuffle(tokens: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
    """(B, L, D) -> (B, D/4, F, 2*Hp, 2*Wp) — spatial resolution unlock.

    Undoes the DiT's (1,2,2) patching SPATIALLY: each token's D channels are
    split into 4 groups of D/4, laid out on a fixed 2x2 sub-grid. The channel
    group <-> quadrant assignment is arbitrary (patch_embedding is a learned
    Conv, there is no canonical inverse) but CONSISTENT, so the projector's
    input_proj learns with it; what matters is 4x more spatial tokens =
    finer spatial-PE / ray-PE / soft-argmax granularity.
    """
    B, L, D = tokens.shape
    f, h, w = (int(v) for v in grid_sizes[0].tolist())
    assert L == f * h * w and D % 4 == 0
    x = tokens.view(B, f, h, w, 2, 2, D // 4)              # (B,f,h,w,ph,pw,C)
    x = x.permute(0, 6, 1, 2, 4, 3, 5)                     # (B,C,f,h,ph,w,pw)
    return x.reshape(B, D // 4, f, 2 * h, 2 * w).contiguous()
