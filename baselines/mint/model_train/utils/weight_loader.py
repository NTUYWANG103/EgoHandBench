#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import fnmatch
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from baselines.mint.model_train.utils.logging import rank0_print

_TORCH_EXTS = (".pt", ".pth", ".bin", ".ckpt")


def load_state_dict_file(path: str, device: str = "cpu") -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path, device=device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def load_pretrained(
    module: nn.Module,
    path: str,
    prefix_map: Optional[List[Tuple[str, str]]] = None,
    exclude: Optional[List[str]] = None,
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[list, list]:
    sd = load_state_dict_file(path)
    if prefix_map:
        remapped = {}
        for k, v in sd.items():
            for old, new in prefix_map:
                if k.startswith(old):
                    k = new + k[len(old):]
                    break
            remapped[k] = v
        sd = remapped
    if exclude:
        n_before = len(sd)
        sd = {k: v for k, v in sd.items()
              if not any(fnmatch.fnmatch(k, pat) for pat in exclude)}
        if verbose:
            rank0_print(f"[train]  {exclude}; {n_before - len(sd)}.")
    missing, unexpected = module.load_state_dict(sd, strict=strict)
    if verbose:
        rank0_print(f"[weight_loader] {os.path.basename(path)}: "
                    f"loaded {len(sd)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
    return missing, unexpected
