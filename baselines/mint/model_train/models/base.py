#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Internal helper."""
from typing import Dict

import torch
import torch.nn as nn


class BaseStudent(nn.Module):
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
