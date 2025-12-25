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
SKEL_NUM_VERTICES = 6890  # Same as SMPL

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

# Spine DOF indices (from working compare_smpl_skel.py)
# DOF 17-19: lumbar, DOF 20-22: thorax, DOF 23-25: head
SPINE_DOF_INDICES = list(range(17, 26))  # lumbar + thorax + head

# Scapula DOF bounds (radians)
SCAPULA_DOF_BOUNDS = (-0.5, 0.5)

# Per-joint weights for optimization (critical for good results!)
# Based on compare_smpl_skel.py: shoulders 10x, spine 5x, pelvis/femurs 2x
# Tuned to reduce high-error joints (acromial, walker_knee)
#
# NOTE: acromial (AddB surface landmark) → scapula (SKEL lateral shoulder)
# Scapula is more lateral than humerus, providing better shoulder width match.
# Humerus (glenohumeral) is not directly used in joint loss.
SKEL_JOINT_WEIGHTS = {
    'pelvis': 20.0,      # ROOT - highest weight! Must anchor the skeleton
    'femur_r': 5.0,      # hip
    'femur_l': 5.0,      # hip
    'tibia_r': 3.0,      # knee
    'tibia_l': 3.0,      # knee
    'talus_r': 2.0,      # ankle
    'talus_l': 2.0,      # ankle
    'calcn_r': 1.0,
    'calcn_l': 1.0,
    'toes_r': 1.0,
    'toes_l': 1.0,
    'lumbar': 15.0,      # spine - very high weight
    'thorax': 5.0,       # spine
    'head': 1.0,
    'scapula_r': 10.0,   # acromial→scapula: HIGH weight for shoulder width
    'scapula_l': 10.0,   # acromial→scapula: HIGH weight for shoulder width
    'humerus_r': 1.0,    # humerus: not directly in loss, low weight
    'humerus_l': 1.0,    # humerus: not directly in loss, low weight
    'ulna_r': 10.0,      # elbow - HIGH weight
    'ulna_l': 10.0,      # elbow - HIGH weight
    'radius_r': 8.0,     # wrist - HIGH weight
    'radius_l': 8.0,     # wrist - HIGH weight
    'hand_r': 15.0,      # hand - HIGHEST weight for arm chain
    'hand_l': 15.0,      # hand - HIGHEST weight for arm chain
}

# SKEL joint name to index mapping
SKEL_JOINT_TO_IDX = {
    'pelvis': 0,
    'femur_r': 1, 'tibia_r': 2, 'talus_r': 3, 'calcn_r': 4, 'toes_r': 5,
    'femur_l': 6, 'tibia_l': 7, 'talus_l': 8, 'calcn_l': 9, 'toes_l': 10,
    'lumbar': 11, 'thorax': 12, 'head': 13,
    'scapula_r': 14, 'humerus_r': 15, 'ulna_r': 16, 'radius_r': 17, 'hand_r': 18,
    'scapula_l': 19, 'humerus_l': 20, 'ulna_l': 21, 'radius_l': 22, 'hand_l': 23,
}


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

    # Loss weights (tuned based on compare_smpl_skel.py)
    weight_joint: float = 1.0
    weight_bone_dir: float = 0.3    # 0.5 → 0.3
    weight_bone_len: float = 1.0    # 0.3 → 1.0 (critical!)
    weight_shoulder: float = 1.0
    weight_width: float = 10.0      # 0.5 → 10.0 (critical! matches working code)
    weight_pose_reg: float = 0.01
    weight_spine_reg: float = 0.1   # 0.05 → 0.1
    weight_scapula_reg: float = 0.05  # 0.1 → 0.05
    weight_temporal: float = 0.01   # 0.1 → 0.01 (was too high)

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
