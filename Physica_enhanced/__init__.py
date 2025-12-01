"""
Physica Enhanced - Modular pipeline for AddBiomechanics → SMPL conversion.

High-performance, vectorized implementation with:
- Shape-first optimization strategy
- Keyframe sampling + interpolation for fast pose optimization
- Mixed precision and torch.compile support
- Comprehensive retargeting with joint synthesis
"""

__version__ = "1.0.0"

from physica_pipeline import PhysicaPipeline, PipelineResult
from core import PipelineConfig, SMPLModelWrapper

__all__ = [
    "PhysicaPipeline",
    "PipelineResult",
    "PipelineConfig",
    "SMPLModelWrapper",
]
