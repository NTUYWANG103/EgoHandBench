#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public loss construction API."""
from baselines.mint.model_train.losses.criterion import Criterion, build_criterion

__all__ = ["Criterion", "build_criterion"]
