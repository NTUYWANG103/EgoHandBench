"""GCTStream with a 7D extrinsics head and a parallel 2D FoV head."""

from typing import Dict, Optional

import torch
from baselines.mint.model_train._vendor.lingbot_map.models.gct_stream import GCTStream

from baselines.mint.model_train.models.split_fov_head.extrinsics_head import ExtrinsicsCameraCausalHead
from baselines.mint.model_train.models.split_fov_head.fov_head import FovHead


def combine_pose_enc(extrinsics: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
    """Combine ``trans3 + quat4`` and ``fov_h + fov_w`` into the canonical pose."""
    if extrinsics.shape[:-1] != fov.shape[:-1]:
        raise ValueError(
            f"extrinsics/fov leading shapes differ: {extrinsics.shape} vs {fov.shape}"
        )
    if extrinsics.shape[-1] != 7 or fov.shape[-1] != 2:
        raise ValueError(
            f"expected extrinsics[...,7] and fov[...,2], got {extrinsics.shape} and {fov.shape}"
        )
    return torch.cat((extrinsics, fov), dim=-1)


class SplitFovNetwork(GCTStream):
    """Predict camera extrinsics and field of view with independent heads."""

    def __init__(self, fov_head: Optional[dict] = None, **backbone_cfg):
        super().__init__(**backbone_cfg)
        fov_cfg = dict(fov_head or {})
        fov_cfg.setdefault("in_dim", 2 * self.embed_dim)
        self.fov_head = FovHead(**fov_cfg)

    def _build_camera_head(self):
        return ExtrinsicsCameraCausalHead(
            dim_in=2 * self.embed_dim,
            sliding_window_size=self.sliding_window_size,
            attend_to_scale_frames=self.attend_to_scale_frames,
            num_iterations=self.camera_num_iterations,
            kv_cache_sliding_window=self.kv_cache_sliding_window,
            kv_cache_scale_frames=self.kv_cache_scale_frames,
            kv_cache_cross_frame_special=self.kv_cache_cross_frame_special,
            kv_cache_include_scale_frames=self.kv_cache_include_scale_frames,
            kv_cache_camera_only=self.kv_cache_camera_only,
            enable_3d_rope=self.enable_camera_3d_rope,
            max_frame_num=self.max_frame_num,
            rope_theta=self.camera_rope_theta,
        )

    def _predict_camera(
        self,
        aggregated_tokens_list: list,
        mask: Optional[torch.Tensor] = None,
        causal_inference: bool = False,
        num_frame_for_scale: Optional[int] = None,
        sliding_window_size: Optional[int] = None,
        num_frame_per_block: int = 1,
        gather_outputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        prediction = super()._predict_camera(
            aggregated_tokens_list,
            mask=mask,
            causal_inference=causal_inference,
            num_frame_for_scale=num_frame_for_scale,
            sliding_window_size=sliding_window_size,
            num_frame_per_block=num_frame_per_block,
            gather_outputs=gather_outputs,
        )
        fov = self.fov_head(aggregated_tokens_list[-1])
        extrinsics_list = prediction["pose_enc_list"]
        prediction["pose_enc_list"] = [
            combine_pose_enc(extrinsics, fov) for extrinsics in extrinsics_list
        ]
        prediction["pose_enc"] = prediction["pose_enc_list"][-1]
        return prediction
