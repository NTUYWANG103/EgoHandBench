#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset


class BaseClipDataset(Dataset):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.clip_len = int(cfg.get("clip_len", 4))
        self.size_hw = tuple(cfg.get("size_hw", (378, 518)))

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
