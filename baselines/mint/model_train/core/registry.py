#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Dict


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._table: Dict[str, Any] = {}

    def register(self, key: str):
        def deco(cls):
            if key in self._table:
                raise KeyError(f"[train]  {self.name}; {key}.")
            self._table[key] = cls
            return cls
        return deco

    def get(self, key: str):
        if key not in self._table:
            raise KeyError(f"[train]  {self.name}; {key!r}; {sorted(self._table)}.")
        return self._table[key]

    def build_from_cfg(self, cfg: Dict[str, Any], **extra):
        """Internal helper."""
        if "name" not in cfg:
            raise KeyError(f"[train]  {self.name}; {cfg}.")
        return self.get(cfg["name"])(cfg, **extra)

    def keys(self):
        return sorted(self._table)


MODELS = Registry("models")
DATASETS = Registry("datasets")
LOSSES = Registry("losses")
