"""
Configuration and constants for the addb2skel pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import torch


# =============================================================================
# Path Configuration
# =============================================================================

SKEL_MODEL_PATH = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'
ADDB_DATA_PATH = '/egr/research-zijunlab/kwonjoon/02_Dataset/AddB'
OUTPUT_PATH = '/egr/research-zijunlab/kwonjoon/03_Output/addb2skel'


# =============================================================================
# SKEL Model Constants
# =============================================================================

SKEL_NUM_JOINTS = 24
SKEL_NUM_BETAS = 10
SKEL_NUM_POSE_DOF = 46  # Euler angles for all DOFs

# Scapula DOF indices in SKEL pose vector
SCAPULA_DOF_INDICES = {
    'right': {
        'abduction': 26,
        'elevation': 27,
        'upward_rot': 28,
    },
    'left': {
        'abduction': 36,
        'elevation': 37,
        'upward_rot': 38,
    },
}

# Humerus (shoulder) DOF indices
HUMERUS_DOF_INDICES = {
    'right': [29, 30, 31],  # shoulder_x, y, z
    'left': [39, 40, 41],
}

# Spine DOF indices
SPINE_DOF_INDICES = [6, 7, 8, 9, 10, 11]  # lumbar + thorax

# Scapula DOF bounds (radians)
SCAPULA_DOF_BOUNDS = (-0.5, 0.5)


# =============================================================================
# AddB Constants
# =============================================================================

ADDB_NUM_JOINTS = 20


# =============================================================================
# Optimization Configuration
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for the optimization pipeline."""

    # Device
    device: str = 'cuda'

    # Scale estimation (Stage 1)
    scale_lr: float = 0.01
    scale_iters: int = 200

    # Pose optimization (Stage 2)
    pose_lr: float = 0.01
    pose_iters: int = 300

    # Loss weights
    weight_joint: float = 1.0
    weight_bone_dir: float = 0.5
    weight_bone_len: float = 0.3
    weight_shoulder: float = 1.0
    weight_width: float = 0.5
    weight_pose_reg: float = 0.01
    weight_spine_reg: float = 0.05
    weight_scapula_reg: float = 0.1
    weight_temporal: float = 0.1

    # Virtual acromial vertex indices (computed at runtime)
    acromial_vertex_indices: Optional[Dict[str, List[int]]] = None

    def get_device(self) -> torch.device:
        """Get torch device."""
        if self.device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')


@dataclass
class ConversionResult:
    """Result of AddB to SKEL conversion."""

    # Core outputs
    skel_joints: 'torch.Tensor'  # [T, 24, 3]
    skel_poses: 'torch.Tensor'   # [T, 46]
    skel_betas: 'torch.Tensor'   # [10] or [T, 10]
    skel_trans: 'torch.Tensor'   # [T, 3]

    # Optionally mesh
    skel_vertices: Optional['torch.Tensor'] = None  # [T, V, 3]

    # Diagnostics
    mpjpe_mm: float = 0.0
    per_joint_error: Optional[Dict[str, float]] = None
    scapula_dofs: Optional[Dict[str, float]] = None

    # Metadata
    num_frames: int = 0
    gender: str = 'male'


# =============================================================================
# Default configuration instance
# =============================================================================

DEFAULT_CONFIG = OptimizationConfig()
