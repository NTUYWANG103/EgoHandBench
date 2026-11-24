"""Native WiLoR imports used by the in-process PAD initializer."""

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
from baselines.pad_hand.WiLoR.wilor.models import load_wilor
from baselines.pad_hand.WiLoR.wilor.utils import recursive_to
from baselines.pad_hand.WiLoR.wilor.datasets.vitdet_dataset import ViTDetDataset
from baselines.pad_hand.WiLoR.wilor.utils.renderer import cam_crop_to_full
from ultralytics import YOLO

__all__ = ["load_wilor", "recursive_to", "ViTDetDataset", "cam_crop_to_full", "YOLO"]  # used as wi.<name> by pipeline.py
