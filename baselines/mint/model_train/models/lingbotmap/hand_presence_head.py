#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent frame-level hand-presence head over backbone patch tokens."""

import torch
import torch.nn as nn


class HandPresenceHead(nn.Module):
    """Use one learned query per side to classify left/right hand presence."""

    def __init__(self, in_dim: int = 2048, dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.token_proj = nn.Linear(in_dim, dim)
        self.token_norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(1, 2, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.out_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor, patch_start_idx: int) -> torch.Tensor:
        """Return left/right logits from ``tokens[B,S,P,C]`` as ``[B,S,2]``."""
        if tokens.ndim != 4:
            raise ValueError(f"tokens must have shape [B,S,P,C], got {tuple(tokens.shape)}")
        B, S, P, C = tokens.shape
        if not 0 <= patch_start_idx < P:
            raise ValueError(
                f"patch_start_idx must be in [0, {P}), got {patch_start_idx}"
            )

        patches = tokens[:, :, patch_start_idx:].reshape(B * S, P - patch_start_idx, C)
        kv = self.token_norm(self.token_proj(patches))
        query = self.query.expand(B * S, -1, -1)
        query = query + self.cross_attn(query, kv, kv, need_weights=False)[0]
        logits = self.classifier(self.out_norm(query)).squeeze(-1)
        return logits.reshape(B, S, 2)
