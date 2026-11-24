"""Model registration and construction."""

from baselines.mint.model_train.core.registry import MODELS
from baselines.mint.model_train.models.lingbotmap import student as _lingbotmap_student  # noqa: F401
from baselines.mint.model_train.models.split_fov_head import student as _split_fov_head_student  # noqa: F401


def build_model(cfg: dict):
    """Build a registered model from its configuration mapping."""
    return MODELS.build_from_cfg(cfg)
