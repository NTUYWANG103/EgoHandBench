"""Independent field-of-view head over the final aggregated camera token."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class FovHead(nn.Module):
    """Predict vertical and horizontal field of view independently of extrinsics."""

    def __init__(
        self,
        in_dim: int = 2048,
        dim: int = 512,
        num_heads: int = 8,
        trunk_depth: int = 2,
        initial_fov_deg=(60.0, 75.0),
        min_fov_rad: float = 1.0e-4,
    ):
        super().__init__()
        from baselines.mint.model_train._vendor.lingbot_map.layers.block import Block

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if trunk_depth < 0:
            raise ValueError(f"trunk_depth must be non-negative, got {trunk_depth}")
        if len(initial_fov_deg) != 2 or any(float(value) <= 0 for value in initial_fov_deg):
            raise ValueError("initial_fov_deg must contain two positive values")
        if min_fov_rad < 0:
            raise ValueError(f"min_fov_rad must be non-negative, got {min_fov_rad}")

        self.input_norm = nn.LayerNorm(in_dim)
        self.input_proj = nn.Linear(in_dim, dim)
        self.trunk = nn.ModuleList(
            [Block(dim=dim, num_heads=num_heads, mlp_ratio=4.0) for _ in range(trunk_depth)]
        )
        self.output_norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, 2)
        self.min_fov_rad = float(min_fov_rad)

        initial_rad = [math.radians(float(value)) for value in initial_fov_deg]
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.output.bias.copy_(
                torch.tensor([_inverse_softplus(value) for value in initial_rad])
            )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, sequence, 2]`` FoV values in radians."""
        if tokens.ndim != 4:
            raise ValueError(f"tokens must have shape [B,S,P,C], got {tuple(tokens.shape)}")
        if tokens.shape[2] < 1:
            raise ValueError("tokens must contain the camera token at index 0")

        hidden = self.input_proj(self.input_norm(tokens[:, :, 0]))
        for block in self.trunk:
            hidden = block(hidden)
        raw_fov = self.output(self.output_norm(hidden))
        return F.softplus(raw_fov) + self.min_fov_rad
