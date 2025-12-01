"""Core modules for Physica pipeline."""

from core.config import PipelineConfig, ShapeOptConfig, PoseOptConfig, RetargetConfig
from core.smpl_model import SMPLModelWrapper

__all__ = [
    "PipelineConfig",
    "ShapeOptConfig",
    "PoseOptConfig",
    "RetargetConfig",
    "SMPLModelWrapper",
]
