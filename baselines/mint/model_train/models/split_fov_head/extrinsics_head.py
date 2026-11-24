"""Seven-dimensional camera head for translation and quaternion only."""

import torch
import torch.nn as nn


class ExtrinsicsCameraCausalHead:
    """Construct a camera head whose iterative state excludes field of view."""

    def __new__(cls, dim_in: int = 2048, **kwargs):
        from baselines.mint.model_train._vendor.lingbot_map.heads.camera_head import CameraCausalHead
        from baselines.mint.model_train._vendor.lingbot_map.layers import Mlp

        head = CameraCausalHead(dim_in=dim_in, fl_act="linear", **kwargs)
        head.target_dim = 7
        head.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, 7))
        head.embed_pose = nn.Linear(7, dim_in)
        head.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=7,
            drop=0,
        )
        return head
