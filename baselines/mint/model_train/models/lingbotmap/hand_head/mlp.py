#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

PER_HAND = 3 + 6 + 90 + 10  # 109
OUT_DIM = 2 * PER_HAND       # 218


class MlpHandHead(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden: int = 1024,
                 out_dim: int = OUT_DIM, num_layers: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.mlp = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, tokens: torch.Tensor, patch_start_idx: int = 0) -> torch.Tensor:
        """Internal helper."""
        feat = tokens.mean(dim=2)     # [B, S, C]  (C = 2*embed_dim)
        return self.mlp(feat)
