#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Internal helper."""
import logging
import os
import sys


def is_main_process() -> bool:
    """Internal helper."""
    return int(os.environ.get("RANK", "0")) == 0


def get_logger(name: str = "model_train") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO if is_main_process() else logging.WARNING)
        logger.propagate = False
    return logger


def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)
