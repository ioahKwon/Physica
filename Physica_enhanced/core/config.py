#!/usr/bin/env python3
"""Configuration dataclasses for the Physica pipeline."""

from dataclasses import dataclass, field
from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@dataclass
class ShapeOptConfig:
    """Configuration for shape optimization (global β over all frames)."""
    lr: float = 1e-2
    max_iters: int = 50
    sample_frames: int = 100
    batch_size: int = 32
    tolerance: float = 1e-4
    weight_position: float = 1.0
    weight_bone_length: float = 0.1
    use_mixed_precision: bool = True


@dataclass
class PoseOptConfig:
    """Configuration for pose optimization (θ with keyframe sampling + interpolation)."""
    lr: float = 5e-3
    max_iters: int = 10  # Fast refinement (≤10 iters)
    keyframe_ratio: float = 0.2  # Sample 20% of frames as keyframes
    min_keyframes: int = 10
    max_keyframes: int = 100
    interpolation_method: Literal["linear", "slerp", "cubic"] = "slerp"
    weight_position: float = 1.0
    weight_bone_direction: float = 0.5
    weight_smoothness: float = 0.05
    use_mixed_precision: bool = True


@dataclass
class RetargetConfig:
    """Configuration for OpenSim→SMPL retargeting."""
    coordinate_system: Literal["z_up", "y_up"] = "y_up"
    apply_pre_alignment: bool = True
    apply_post_alignment: bool = True
    synthesize_missing_joints: bool = True
    # Synthesis rules: spine→proportional, clavicle→shoulder offset, toe→foot forward, neck→head down
    synthesis_method: Literal["proportional", "kinematic"] = "proportional"
    spine_ratio: float = 0.5  # spine2 at 50% between spine1 and spine3
    clavicle_offset: float = 0.1  # 10cm lateral from spine to shoulder
    toe_offset: float = 0.1  # 10cm forward from foot
    neck_ratio: float = 0.5  # neck at 50% between head and spine3


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    shape_opt: ShapeOptConfig = field(default_factory=ShapeOptConfig)
    pose_opt: PoseOptConfig = field(default_factory=PoseOptConfig)
    retarget: RetargetConfig = field(default_factory=RetargetConfig)
    device: str = "cuda"
    use_torch_compile: bool = True
    num_betas: int = 10
    verbose: bool = True

    @classmethod
    def default(cls) -> "PipelineConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def fast(cls) -> "PipelineConfig":
        """Fast configuration for quick iterations."""
        return cls(
            shape_opt=ShapeOptConfig(max_iters=30, sample_frames=50, batch_size=64),
            pose_opt=PoseOptConfig(max_iters=5, keyframe_ratio=0.15, max_keyframes=50),
            retarget=RetargetConfig(),
            use_torch_compile=True
        )
