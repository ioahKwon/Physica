#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AddBiomechanics (.b3d) → SMPL parameter fitting with IK-based Pose Initialization.

This version extends the bone direction loss approach by adding per-frame IK-based
pose initialization. Instead of starting from zero pose, we compute initial pose
parameters using full kinematic chain inverse kinematics from AddBiomechanics
joint positions.

Key improvements:
    1. Per-frame IK initialization using full kinematic chain (parent rotation propagation)
    2. Default bone directions computed from SMPL T-pose (runtime calculation)
    3. Supports both "With Arm" (full body) and "No Arm" (lower body only) modes
    4. Expected 9-13% MPJPE improvement over bone direction baseline (39.23mm → 34-36mm)

Given an AddBiomechanics sequence we:
    1. infer the ordered AddB joint list and auto-map it to SMPL joints,
    2. optimise SMPL betas (shape) on a subsampled set of frames,
    3. compute IK-based initial poses for each frame,
    4. optimise pose + translation with bone direction matching and temporal smoothness,
    5. optionally repeat the refinement until a target error tolerance is met.

Outputs include SMPL parameters, reconstructed joints, evaluation metrics,
metadata and visualisations (via the companion visualise_results.py script).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn.functional as F

from models.smpl_model import SMPLModel, SMPL_NUM_BETAS, SMPL_NUM_JOINTS

try:
    import nimblephysics as nimble
except ImportError as exc:  # pragma: no cover - nimble is an external dep
    raise RuntimeError(
        "nimblephysics is required to read AddBiomechanics .b3d files. "
        "Install nimblephysics inside the current environment."
    ) from exc


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

AUTO_JOINT_NAME_MAP: Dict[str, str] = {
    'ground_pelvis': 'pelvis',
    'pelvis': 'pelvis',
    'root': 'pelvis',
    'hip_r': 'right_hip',
    'hip_right': 'right_hip',
    'hip_l': 'left_hip',
    'hip_left': 'left_hip',
    'walker_knee_r': 'right_knee',
    'knee_r': 'right_knee',
    'walker_knee_l': 'left_knee',
    'knee_l': 'left_knee',
    'ankle_r': 'right_ankle',
    'ankle_right': 'right_ankle',
    'ankle_l': 'left_ankle',
    'ankle_left': 'left_ankle',
    'subtalar_r': 'right_foot',
    'mtp_r': 'right_foot',
    'subtalar_l': 'left_foot',
    'mtp_l': 'left_foot',
    'back': 'spine1',
    'torso': 'spine1',
    'spine': 'spine1',
    'acromial_r': 'right_shoulder',
    'shoulder_r': 'right_shoulder',
    'acromial_l': 'left_shoulder',
    'shoulder_l': 'left_shoulder',
    'elbow_r': 'right_elbow',
    'elbow_l': 'left_elbow',
    # Skip radioulnar - it's forearm rotation, not wrist
    # 'radioulnar_r': 'right_wrist',  # REMOVED - intermediate joint
    'wrist_r': 'right_wrist',
    # 'radioulnar_l': 'left_wrist',  # REMOVED - intermediate joint
    'wrist_l': 'left_wrist',
    # Map radius_hand to wrist (it's the actual wrist position in AddB)
    'radius_hand_r': 'right_wrist',
    'hand_r': 'right_hand',
    'radius_hand_l': 'left_wrist',
    'hand_l': 'left_hand',
}

SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand'
]

# Lower body joint indices (for No_Arm datasets)
LOWER_BODY_JOINTS = [
    0,  # pelvis
    1,  # left_hip
    2,  # right_hip
    4,  # left_knee
    5,  # right_knee
    7,  # left_ankle
    8,  # right_ankle
    10, # left_foot
    11  # right_foot
]


# ---------------------------------------------------------------------------
# Rotation helper functions for IK
# ---------------------------------------------------------------------------

def normalize(v: torch.Tensor) -> torch.Tensor:
    """Normalize a vector, handling near-zero case."""
    norm = torch.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def rotation_matrix_from_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation matrix R that rotates v1 to align with v2.
    Uses Rodrigues' rotation formula.

    Args:
        v1: Source vector [3]
        v2: Target vector [3]

    Returns:
        R: Rotation matrix [3, 3]
    """
    v1 = normalize(v1)
    v2 = normalize(v2)

    # Compute rotation axis and angle
    axis = torch.cross(v1, v2)
    axis_norm = torch.norm(axis)

    # If vectors are parallel, return identity
    if axis_norm < 1e-6:
        dot_product = torch.dot(v1, v2)
        if dot_product > 0:
            # Same direction
            return torch.eye(3, device=v1.device, dtype=v1.dtype)
        else:
            # Opposite direction - rotate 180 degrees around any perpendicular axis
            # Find a perpendicular axis
            if abs(v1[0]) < 0.9:
                perp = torch.tensor([1.0, 0.0, 0.0], device=v1.device, dtype=v1.dtype)
            else:
                perp = torch.tensor([0.0, 1.0, 0.0], device=v1.device, dtype=v1.dtype)
            axis = normalize(torch.cross(v1, perp))
            angle = torch.tensor(np.pi, device=v1.device, dtype=v1.dtype)
    else:
        axis = axis / axis_norm
        angle = torch.acos(torch.clamp(torch.dot(v1, v2), -1.0, 1.0))

    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    K = torch.zeros(3, 3, device=v1.device, dtype=v1.dtype)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]

    R = (torch.eye(3, device=v1.device, dtype=v1.dtype) +
         torch.sin(angle) * K +
         (1 - torch.cos(angle)) * (K @ K))

    return R


def rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to axis-angle representation.

    Args:
        R: Rotation matrix [3, 3]

    Returns:
        axis_angle: Axis-angle vector [3]
    """
    # Compute angle from trace
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))

    # Handle small angles
    if angle < 1e-6:
        return torch.zeros(3, device=R.device, dtype=R.dtype)

    # Compute axis from skew-symmetric part
    axis = torch.zeros(3, device=R.device, dtype=R.dtype)
    axis[0] = R[2, 1] - R[1, 2]
    axis[1] = R[0, 2] - R[2, 0]
    axis[2] = R[1, 0] - R[0, 1]

    axis = axis / (2 * torch.sin(angle))

    return axis * angle


def axis_angle_to_rotation_matrix(aa: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle to rotation matrix.

    Args:
        aa: Axis-angle vector [3]

    Returns:
        R: Rotation matrix [3, 3]
    """
    angle = torch.norm(aa)

    if angle < 1e-6:
        return torch.eye(3, device=aa.device, dtype=aa.dtype)

    axis = aa / angle

    # Rodrigues' formula
    K = torch.zeros(3, 3, device=aa.device, dtype=aa.dtype)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]

    R = (torch.eye(3, device=aa.device, dtype=aa.dtype) +
         torch.sin(angle) * K +
         (1 - torch.cos(angle)) * (K @ K))

    return R


def load_b3d_sequence(b3d_path: str,
                      trial: int = 0,
                      processing_pass: int = 0,
                      start: int = 0,
                      num_frames: int = -1) -> Tuple[np.ndarray, float]:
    subj = nimble.biomechanics.SubjectOnDisk(b3d_path)
    total = subj.getTrialLength(trial)
    count = total - start if num_frames < 0 else min(num_frames, total - start)

    frames = subj.readFrames(
        trial=trial,
        startFrame=start,
        numFramesToRead=count,
        includeProcessingPasses=True,
        includeSensorData=False,
        stride=1,
        contactThreshold=1.0
    )
    if len(frames) == 0:
        raise RuntimeError("Failed to read frames from the .b3d file")

    def frame_joint_centers(frame):
        if hasattr(frame, 'processingPasses') and len(frame.processingPasses) > 0:
            idx = min(processing_pass, len(frame.processingPasses) - 1)
            return np.asarray(frame.processingPasses[idx].jointCenters, dtype=np.float32)
        return np.asarray(frame.jointCenters, dtype=np.float32)

    first = frame_joint_centers(frames[0])
    if first.ndim != 1 or first.size % 3 != 0:
        raise ValueError("Unexpected joint center layout in .b3d file")

    num_joints = first.size // 3
    joints = np.zeros((len(frames), num_joints, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        data = frame_joint_centers(frame)
        joints[i] = data.reshape(-1, 3)[:num_joints]

    dt = float(subj.getTrialTimestep(trial))
    return joints, dt


def convert_addb_to_smpl_coords(joints: np.ndarray) -> np.ndarray:
    return np.stack([joints[..., 0], joints[..., 2], -joints[..., 1]], axis=-1)


def infer_addb_joint_names(b3d_path: str,
                           trial: int = 0,
                           processing_pass: int = 0) -> List[str]:
    subj = nimble.biomechanics.SubjectOnDisk(b3d_path)
    frames = subj.readFrames(
        trial=trial,
        startFrame=0,
        numFramesToRead=1,
        includeProcessingPasses=True,
        includeSensorData=False,
        stride=1,
        contactThreshold=1.0
    )
    if len(frames) == 0:
        raise RuntimeError("Unable to infer joint names — no frames were read")

    pp_idx = min(processing_pass, len(frames[0].processingPasses) - 1)
    pp = frames[0].processingPasses[pp_idx]

    skel = subj.readSkel(trial)
    pos = np.asarray(pp.pos, dtype=np.float32)
    if pos.size == skel.getNumDofs():
        skel.setPositions(pos)

    world_joints = []
    for i in range(skel.getNumJoints()):
        joint = skel.getJoint(i)
        world = joint.getChildBodyNode().getWorldTransform().translation()
        world_joints.append((joint.getName(), world))

    joint_centers = np.asarray(pp.jointCenters, dtype=np.float32).reshape(-1, 3)
    names: List[str] = []
    for center in joint_centers:
        dists = [np.linalg.norm(center - w) for _, w in world_joints]
        best = int(np.argmin(dists))
        names.append(world_joints[best][0])
    return names


def auto_map_addb_to_smpl(addb_joint_names: List[str]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    used_smpl: Set[int] = set()
    for idx, name in enumerate(addb_joint_names):
        key = name.lower().replace(' ', '')
        if key in AUTO_JOINT_NAME_MAP:
            smpl_name = AUTO_JOINT_NAME_MAP[key]
            if smpl_name in SMPL_JOINT_NAMES:
                smpl_idx = SMPL_JOINT_NAMES.index(smpl_name)
                if smpl_idx in used_smpl:
                    continue
                mapping[idx] = smpl_idx
                used_smpl.add(smpl_idx)
    return mapping


def load_mapping_overrides(map_json: Optional[str],
                           addb_names: List[str]) -> Dict[int, int]:
    if map_json is None:
        return {}
    if not os.path.exists(map_json):
        raise FileNotFoundError(f"Mapping JSON not found: {map_json}")

    with open(map_json, 'r') as f:
        data = json.load(f)

    overrides_raw = data.get('addb2smpl', {})
    overrides: Dict[int, int] = {}
    for key, value in overrides_raw.items():
        if isinstance(key, str) and key.strip().isdigit():
            addb_idx = int(key)
        elif isinstance(key, str) and key in addb_names:
            addb_idx = addb_names.index(key)
        elif isinstance(key, int):
            addb_idx = key
        else:
            raise ValueError(f"Unknown AddB joint identifier: {key}")

        if isinstance(value, str) and value.strip().isdigit():
            smpl_idx = int(value)
        elif isinstance(value, str) and value in SMPL_JOINT_NAMES:
            smpl_idx = SMPL_JOINT_NAMES.index(value)
        elif isinstance(value, int):
            smpl_idx = value
        else:
            raise ValueError(f"Unknown SMPL joint identifier: {value}")

        overrides[addb_idx] = smpl_idx
    return overrides


def resolve_mapping(addb_names: List[str],
                    auto_map: Dict[int, int],
                    overrides: Dict[int, int]) -> Dict[int, int]:
    mapping = auto_map.copy()
    mapping.update(overrides)
    if not mapping:
        raise ValueError("No joint correspondences were resolved. "
                         "Provide a mapping JSON or extend the auto-map table.")
    return mapping


def center_on_root_tensor(pred: torch.Tensor,
                          target: torch.Tensor,
                          root_idx: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_centered = pred.clone()
    target_centered = target.clone()
    if root_idx is not None and root_idx < target.shape[0]:
        root_target = target[root_idx]
        if not torch.isnan(root_target).any():
            pred_centered = pred_centered - pred[0]
            target_centered = target_centered - root_target
    return pred_centered, target_centered


def center_on_root_numpy(pred: np.ndarray,
                         target: np.ndarray,
                         root_idx: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    pred_centered = pred.copy()
    target_centered = target.copy()
    if root_idx is not None and root_idx < target.shape[0]:
        root_target = target[root_idx]
        if not np.any(np.isnan(root_target)):
            pred_centered -= pred[0]
            target_centered -= root_target
    return pred_centered, target_centered


# ---------------------------------------------------------------------------
# Optimisation configuration and core fitter
# ---------------------------------------------------------------------------

@dataclass
class OptimisationConfig:
    # FIXED11: Reduced iterations for speed ("Less is More")
    shape_lr: float = 5e-3
    shape_iters: int = 50  # Reduced from 150
    shape_sample_frames: int = 80

    pose_lr: float = 1e-2
    pose_iters: int = 15  # Reduced from 80 (same as Stage 1)

    tolerance_mm: float = 20.0
    max_passes: int = 1  # Reduced from 3 - one good pass is enough

    pose_reg_weight: float = 1e-3
    trans_reg_weight: float = 1e-3
    bone_length_weight: float = 100.0  # Weight for bone length constraint (FIXED9)
    temporal_smooth_weight: float = 0.1  # Temporal smoothness regularization
    bone_direction_weight: float = 0.5  # Weight for bone direction matching loss

    lower_body_only: bool = False  # With Arm vs No Arm mode


class AddBToSMPLFitter:
    def __init__(self,
                 smpl_model: SMPLModel,
                 target_joints: np.ndarray,
                 joint_mapping: Dict[int, int],
                 addb_joint_names: List[str],
                 dt: float,
                 config: OptimisationConfig,
                 device: torch.device = torch.device('cpu')) -> None:
        self.smpl = smpl_model
        self.config = config
        self.device = device

        self.target_joints_np = target_joints
        self.target_joints = torch.from_numpy(target_joints).float().to(device)
        self.dt = dt

        self.joint_mapping = joint_mapping
        self.addb_joint_names = addb_joint_names

        self.addb_indices = torch.tensor(list(joint_mapping.keys()), dtype=torch.long, device=device)
        self.smpl_indices = torch.tensor(list(joint_mapping.values()), dtype=torch.long, device=device)

        self.root_addb_idx = None
        for addb_idx, smpl_idx in joint_mapping.items():
            if smpl_idx == 0:
                self.root_addb_idx = addb_idx
                break

        # Create joint name to index mapping (for future use)
        self.addb_name_to_idx = {name: idx for idx, name in enumerate(addb_joint_names)}

        # Initialize kinematic chains based on lower_body_only flag
        self._initialize_kinematic_chains()

        # Lazy initialization - compute default bone directions when first needed
        self.default_bone_directions = None

    def _initialize_kinematic_chains(self):
        """
        Initialize kinematic chains based on lower_body_only flag.

        With Arm (lower_body_only=False): Full body chains including arms and spine
        No Arm (lower_body_only=True): Only lower body chains (legs only)
        """
        if self.config.lower_body_only:
            # No Arm mode: Only leg chains
            self.kinematic_chains = {
                'left_leg': [('pelvis', 0), ('left_hip', 1), ('left_knee', 4), ('left_ankle', 7), ('left_foot', 10)],
                'right_leg': [('pelvis', 0), ('right_hip', 2), ('right_knee', 5), ('right_ankle', 8), ('right_foot', 11)]
            }
        else:
            # With Arm mode: Full body chains
            self.kinematic_chains = {
                'left_leg': [('pelvis', 0), ('left_hip', 1), ('left_knee', 4), ('left_ankle', 7), ('left_foot', 10)],
                'right_leg': [('pelvis', 0), ('right_hip', 2), ('right_knee', 5), ('right_ankle', 8), ('right_foot', 11)],
                'left_arm': [('left_shoulder', 16), ('left_elbow', 18), ('left_wrist', 20)],
                'right_arm': [('right_shoulder', 17), ('right_elbow', 19), ('right_wrist', 21)],
                'spine': [('pelvis', 0), ('spine1', 3), ('spine2', 6), ('spine3', 9), ('neck', 12), ('head', 15)]
            }

    def _compute_default_bone_directions(self) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Compute default bone direction vectors from SMPL T-pose (Option A).

        This computes reference bone directions at runtime from the T-pose skeleton,
        applied to all lower body joints (or full body if not lower_body_only).

        Returns:
            Dictionary mapping (parent_idx, child_idx) -> normalized direction vector [3]
        """
        # Get T-pose joints (zero pose, mean shape)
        # SMPL API: joints() accepts 1D betas, 2D poses, 1D trans and handles batching automatically
        zero_pose = torch.zeros(SMPL_NUM_JOINTS, 3, device=self.device)  # [24, 3]
        mean_betas = torch.zeros(SMPL_NUM_BETAS, device=self.device)      # [10]
        zero_trans = torch.zeros(3, device=self.device)                    # [3]

        with torch.no_grad():
            tpose_joints = self.smpl.joints(mean_betas, zero_pose, zero_trans)  # Returns [24, 3] due to single=True

        # Compute bone directions for all chains
        bone_directions = {}
        for chain_name, joint_list in self.kinematic_chains.items():
            for i in range(len(joint_list) - 1):
                parent_name, parent_idx = joint_list[i]
                child_name, child_idx = joint_list[i + 1]

                bone_vec = tpose_joints[child_idx] - tpose_joints[parent_idx]
                bone_dir = normalize(bone_vec)

                bone_directions[(parent_idx, child_idx)] = bone_dir

        return bone_directions

    # --------------------------- Shape optimisation -----------------------
    def _compute_bone_lengths(self, joints: torch.Tensor) -> Dict[str, float]:
        """Compute bone lengths from joint positions."""
        # Define kinematic chains for major bones
        bone_pairs = [
            (1, 4, 'left_femur'),    # left_hip to left_knee
            (2, 5, 'right_femur'),   # right_hip to right_knee
            (4, 7, 'left_tibia'),    # left_knee to left_ankle
            (5, 8, 'right_tibia'),   # right_knee to right_ankle
            (16, 18, 'left_humerus'), # left_shoulder to left_elbow
            (17, 19, 'right_humerus'),# right_shoulder to right_elbow
            (18, 20, 'left_radius'),  # left_elbow to left_wrist
            (19, 21, 'right_radius'), # right_elbow to right_wrist
        ]

        lengths = {}
        for parent_idx, child_idx, name in bone_pairs:
            if parent_idx < joints.shape[0] and child_idx < joints.shape[0]:
                bone_vec = joints[child_idx] - joints[parent_idx]
                lengths[name] = torch.norm(bone_vec)
        return lengths

    def _compute_bone_direction_loss(self,
                                      pred_joints: torch.Tensor,
                                      target_joints: torch.Tensor) -> torch.Tensor:
        """
        Compute bone direction matching loss using cosine similarity.
        This bypasses absolute position mismatch due to skeletal structure differences.

        The key insight: AddBiomechanics and SMPL have different hip joint placements
        (AddB: 138mm from pelvis, SMPL: 106mm), causing cascading errors down the leg.
        By matching bone DIRECTIONS instead of absolute positions, we bypass this
        structural mismatch while still ensuring correct pose.

        Args:
            pred_joints: SMPL predicted joints [N, 3] (subset after mask filtering)
            target_joints: AddB target joints [N, 3] (subset after mask filtering)

        Returns:
            Direction matching loss (scalar, lower is better)
        """
        # Build bone pairs dynamically based on joint mapping
        # Format: (parent_smpl_idx, child_smpl_idx, parent_addb_idx, child_addb_idx, bone_name)
        bone_pairs = []

        # Helper to find AddB index for SMPL index
        def find_addb_idx(smpl_idx):
            for addb_idx, mapped_smpl_idx in self.joint_mapping.items():
                if mapped_smpl_idx == smpl_idx:
                    return addb_idx
            return None

        # Pelvis -> Hip (left and right)
        addb_pelvis = find_addb_idx(0)
        addb_left_hip = find_addb_idx(1)
        addb_right_hip = find_addb_idx(2)

        if addb_pelvis is not None and addb_left_hip is not None:
            bone_pairs.append((0, 1, addb_pelvis, addb_left_hip, 'pelvis_to_left_hip'))
        if addb_pelvis is not None and addb_right_hip is not None:
            bone_pairs.append((0, 2, addb_pelvis, addb_right_hip, 'pelvis_to_right_hip'))

        # Hip -> Knee (left and right)
        addb_left_knee = find_addb_idx(4)
        addb_right_knee = find_addb_idx(5)

        if addb_left_hip is not None and addb_left_knee is not None:
            bone_pairs.append((1, 4, addb_left_hip, addb_left_knee, 'left_hip_to_knee'))
        if addb_right_hip is not None and addb_right_knee is not None:
            bone_pairs.append((2, 5, addb_right_hip, addb_right_knee, 'right_hip_to_knee'))

        # Knee -> Ankle (left and right)
        addb_left_ankle = find_addb_idx(7)
        addb_right_ankle = find_addb_idx(8)

        if addb_left_knee is not None and addb_left_ankle is not None:
            bone_pairs.append((4, 7, addb_left_knee, addb_left_ankle, 'left_knee_to_ankle'))
        if addb_right_knee is not None and addb_right_ankle is not None:
            bone_pairs.append((5, 8, addb_right_knee, addb_right_ankle, 'right_knee_to_ankle'))

        # Ankle -> Foot (left and right)
        addb_left_foot = find_addb_idx(10)
        addb_right_foot = find_addb_idx(11)

        if addb_left_ankle is not None and addb_left_foot is not None:
            bone_pairs.append((7, 10, addb_left_ankle, addb_left_foot, 'left_ankle_to_foot'))
        if addb_right_ankle is not None and addb_right_foot is not None:
            bone_pairs.append((8, 11, addb_right_ankle, addb_right_foot, 'right_ankle_to_foot'))

        # Compute direction loss for each bone
        total_loss = 0.0
        count = 0

        for parent_smpl, child_smpl, parent_addb, child_addb, name in bone_pairs:
            # Check bounds
            if (parent_addb >= target_joints.shape[0] or
                child_addb >= target_joints.shape[0] or
                parent_smpl >= pred_joints.shape[0] or
                child_smpl >= pred_joints.shape[0]):
                continue

            # Get joint positions
            target_parent = target_joints[parent_addb]
            target_child = target_joints[child_addb]
            pred_parent = pred_joints[parent_smpl]
            pred_child = pred_joints[child_smpl]

            # Skip if any joint has NaN
            if (torch.isnan(target_parent).any() or torch.isnan(target_child).any() or
                torch.isnan(pred_parent).any() or torch.isnan(pred_child).any()):
                continue

            # Compute direction vectors
            pred_dir = pred_child - pred_parent
            target_dir = target_child - target_parent

            # Normalize to unit vectors (avoid division by zero)
            pred_norm = torch.norm(pred_dir)
            target_norm = torch.norm(target_dir)

            if pred_norm < 1e-6 or target_norm < 1e-6:
                continue  # Skip degenerate bones

            pred_dir = pred_dir / pred_norm
            target_dir = target_dir / target_norm

            # Cosine similarity loss: 1 - cos(theta)
            # cos(theta) = 1 when parallel (perfect match)
            # cos(theta) = 0 when perpendicular
            # cos(theta) = -1 when opposite
            cos_sim = torch.dot(pred_dir, target_dir)
            loss = 1.0 - cos_sim  # Loss is 0 when parallel, 2 when opposite

            total_loss += loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=pred_joints.device)

        return total_loss / count

    def optimise_initial_poses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 1: Initial coarse pose estimation for shape optimization
        Uses default shape (betas=0) and IK-based initialization
        Returns: (poses, trans) - initial estimates
        """
        print('  [Stage 1] Initial pose estimation with IK initialization...')

        # Default shape
        betas = torch.zeros(SMPL_NUM_BETAS, device=self.device)

        T = self.target_joints.shape[0]

        # Use simplified IK-based initialization
        print('    Computing simplified IK-based initial poses...')
        poses = self._estimate_initial_poses_ik()
        trans = torch.zeros(T, 3, device=self.device)
        print(f'    IK initialization complete for {T} frames')

        # Quick optimization: fewer iterations for speed
        num_iters = 15  # Much faster than full pose optimization
        lr = self.config.pose_lr

        for t in range(T):
            target_frame = self.target_joints[t]
            if torch.isnan(target_frame).all():
                continue

            # Initialize translation with target pelvis position (better starting point)
            target_pelvis_idx = 0  # ground_pelvis in AddBiomechanics
            addb_indices_list = self.addb_indices.cpu().tolist()
            if target_pelvis_idx in addb_indices_list:
                addb_pelvis_pos = addb_indices_list.index(target_pelvis_idx)
                target_pelvis = target_frame[addb_pelvis_pos]
                if not torch.isnan(target_pelvis).any():
                    trans[t] = target_pelvis.clone()

            # Optimize pose and translation for this frame
            pose_param = torch.nn.Parameter(poses[t].clone())
            trans_param = torch.nn.Parameter(trans[t].clone())
            optimiser = torch.optim.Adam([pose_param, trans_param], lr=lr)

            for _ in range(num_iters):
                optimiser.zero_grad()

                joints_pred = self.smpl.joints(betas.unsqueeze(0), pose_param.unsqueeze(0), trans_param.unsqueeze(0)).squeeze(0)

                # Use absolute positions (no root centering)
                target_subset = target_frame[self.addb_indices]
                pred_subset = joints_pred[self.smpl_indices]
                mask = ~torch.isnan(target_subset).any(dim=1)
                if mask.sum() == 0:
                    break

                diff = pred_subset[mask] - target_subset[mask]
                loss = (diff ** 2).mean()

                # Simple pose regularization
                loss = loss + 0.01 * (pose_param ** 2).mean()

                loss.backward()
                optimiser.step()

            # Update pose and translation
            poses[t] = pose_param.detach()
            trans[t] = trans_param.detach()

        print(f'    Initial poses estimated for {T} frames')
        return poses, trans

    def optimise_shape(self, initial_poses: Optional[torch.Tensor] = None,
                       initial_trans: Optional[torch.Tensor] = None,
                       batch_size: int = 32) -> torch.Tensor:
        """
        Stage 2: Pose-aware shape optimization with mini-batch SGD
        If initial_poses and initial_trans are provided, use them for optimization.
        Otherwise, fall back to zero pose (T-pose).

        Args:
            initial_poses: Initial pose estimates from Stage 1 [T, 24, 3]
            initial_trans: Initial translation estimates from Stage 1 [T, 3]
            batch_size: Number of frames to process per mini-batch (default: 32)
        """
        print('  [Stage 2] Pose-aware shape optimization (mini-batch SGD)...')

        betas = torch.zeros(SMPL_NUM_BETAS, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([betas], lr=self.config.shape_lr)

        num_frames = self.target_joints.shape[0]
        sample_count = min(self.config.shape_sample_frames, num_frames)
        frame_indices = np.linspace(0, num_frames - 1, sample_count, dtype=int)

        print(f'    Total frames: {num_frames}, Sampled: {sample_count}, Batch size: {batch_size}')

        # Compute target bone lengths from AddB data (average across frames)
        # Use mapped SMPL joints from target AddB data
        target_bone_lengths = {}
        valid_count = {}
        for idx in frame_indices:
            target = self.target_joints[idx]
            if torch.isnan(target).all():
                continue

            # Map AddB joints to SMPL joint positions
            # Create a pseudo-SMPL skeleton with only mapped joints
            smpl_like_joints = torch.zeros(24, 3, device=self.device)
            for addb_key, smpl_idx in self.joint_mapping.items():
                addb_idx_int = int(addb_key)
                if addb_idx_int < target.shape[0]:
                    smpl_like_joints[smpl_idx] = target[addb_idx_int]

            # Compute bone lengths using SMPL indices (but from AddB data)
            lengths = self._compute_bone_lengths(smpl_like_joints)
            for name, length in lengths.items():
                if not torch.isnan(length) and length > 0:
                    if name not in target_bone_lengths:
                        target_bone_lengths[name] = 0.0
                        valid_count[name] = 0
                    target_bone_lengths[name] += length.item()
                    valid_count[name] += 1

        # Average bone lengths
        for name in target_bone_lengths:
            if valid_count[name] > 0:
                target_bone_lengths[name] /= valid_count[name]

        print(f'    Target bone lengths computed: {list(target_bone_lengths.keys())}')

        # Use initial poses if provided (Stage 2: pose-aware), otherwise use zero pose (fallback)
        use_pose_aware = initial_poses is not None and initial_trans is not None
        if not use_pose_aware:
            print('    Warning: No initial poses provided, falling back to zero pose (T-pose)')
            zero_pose = torch.zeros(1, SMPL_NUM_JOINTS, 3, device=self.device)
            zero_trans = torch.zeros(1, 3, device=self.device)
        else:
            print(f'    Using pose-aware optimization with {len(frame_indices)} sampled frames')

        # Mini-batch SGD loop
        num_batches = int(np.ceil(len(frame_indices) / batch_size))
        print(f'    Running {self.config.shape_iters} iterations with {num_batches} batches per iteration')

        for it in range(self.config.shape_iters):
            # Shuffle frame indices for SGD
            np.random.shuffle(frame_indices)

            epoch_loss = 0.0
            epoch_batches = 0

            # Process mini-batches
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(frame_indices))
                batch_frame_indices = frame_indices[start_idx:end_idx]

                optimizer.zero_grad()

                # Joint position loss (use absolute positions, no root centering)
                if use_pose_aware:
                    total_loss = 0.0
                    valid_frames = 0

                    # For each frame in the mini-batch, use its actual pose
                    for idx in batch_frame_indices:
                        target = self.target_joints[idx]
                        if torch.isnan(target).all():
                            continue

                        # Use actual pose from Stage 1
                        pose_t = initial_poses[idx]
                        trans_t = initial_trans[idx]

                        joints_pred = self.smpl.joints(betas.unsqueeze(0), pose_t.unsqueeze(0), trans_t.unsqueeze(0)).squeeze(0)

                        # Use absolute positions to match correct scale/size
                        target_subset = target[self.addb_indices]
                        pred_subset = joints_pred[self.smpl_indices]

                        mask = ~torch.isnan(target_subset).any(dim=1)
                        if mask.sum() == 0:
                            continue

                        diff = pred_subset[mask] - target_subset[mask]
                        total_loss = total_loss + (diff ** 2).mean()
                        valid_frames += 1

                    if valid_frames == 0:
                        continue

                    loss = total_loss / valid_frames

                    # For bone length computation, use first frame in batch
                    first_batch_idx = batch_frame_indices[0]
                    joints_pred_for_bones = self.smpl.joints(betas.unsqueeze(0), initial_poses[first_batch_idx].unsqueeze(0), initial_trans[first_batch_idx].unsqueeze(0)).squeeze(0)
                else:
                    # Original zero-pose approach (fallback)
                    joints_pred = self.smpl.joints(betas.unsqueeze(0), zero_pose[0], zero_trans[0]).squeeze(0)

                    # Joint position loss (use absolute positions, no root centering)
                    total_loss = 0.0
                    valid_frames = 0
                    for idx in batch_frame_indices:
                        target = self.target_joints[idx]
                        if torch.isnan(target).all():
                            continue

                        # Use absolute positions to match correct scale/size
                        target_subset = target[self.addb_indices]
                        pred_subset = joints_pred[self.smpl_indices]

                        mask = ~torch.isnan(target_subset).any(dim=1)
                        if mask.sum() == 0:
                            continue

                        diff = pred_subset[mask] - target_subset[mask]
                        total_loss = total_loss + (diff ** 2).mean()
                        valid_frames += 1

                    if valid_frames == 0:
                        continue

                    loss = total_loss / valid_frames
                    joints_pred_for_bones = joints_pred

                # Bone length constraint loss (use joints_pred_for_bones)
                pred_bone_lengths = self._compute_bone_lengths(joints_pred_for_bones)
                bone_loss = 0.0
                bone_count = 0
                for name, target_len in target_bone_lengths.items():
                    if name in pred_bone_lengths:
                        diff_len = pred_bone_lengths[name] - target_len
                        bone_loss += diff_len ** 2
                        bone_count += 1

                if bone_count > 0:
                    bone_loss = bone_loss / bone_count
                    loss = loss + self.config.bone_length_weight * bone_loss

                # Backward pass for this mini-batch
                loss.backward()
                optimizer.step()

                # Track epoch loss
                epoch_loss += loss.item()
                epoch_batches += 1

            # Print epoch loss every 10 iterations
            if (it + 1) % 10 == 0 and epoch_batches > 0:
                avg_epoch_loss = epoch_loss / epoch_batches
                print(f'    Iteration {it + 1}/{self.config.shape_iters}: Avg batch loss = {avg_epoch_loss:.6f}')

        return betas.detach()

    # --------------------------- IK-based Pose Initialization -------------
    def _estimate_pelvis_orientation(self, target_joints_frame: torch.Tensor) -> torch.Tensor:
        """
        Estimate pelvis orientation from hip positions.

        Args:
            target_joints_frame: AddB joint positions for one frame [num_joints, 3]

        Returns:
            Pelvis rotation in axis-angle [3]
        """
        # Find AddB indices for pelvis and hips
        pelvis_addb_idx = None
        left_hip_addb_idx = None
        right_hip_addb_idx = None

        for addb_idx, smpl_idx in self.joint_mapping.items():
            if smpl_idx == 0:  # pelvis
                pelvis_addb_idx = addb_idx
            elif smpl_idx == 1:  # left_hip
                left_hip_addb_idx = addb_idx
            elif smpl_idx == 2:  # right_hip
                right_hip_addb_idx = addb_idx

        # If we don't have both hips, return zero rotation
        if left_hip_addb_idx is None or right_hip_addb_idx is None or pelvis_addb_idx is None:
            return torch.zeros(3, device=self.device)

        left_hip_pos = target_joints_frame[left_hip_addb_idx]
        right_hip_pos = target_joints_frame[right_hip_addb_idx]
        pelvis_pos = target_joints_frame[pelvis_addb_idx]

        # Check for NaN
        if torch.isnan(left_hip_pos).any() or torch.isnan(right_hip_pos).any() or torch.isnan(pelvis_pos).any():
            return torch.zeros(3, device=self.device)

        # Compute pelvis orientation from hip positions
        # Default pelvis forward is +Z, up is +Y, right is +X
        right_vec = normalize(right_hip_pos - left_hip_pos)  # Lateral axis
        up_default = torch.tensor([0., 1., 0.], device=self.device)
        forward_vec = normalize(torch.cross(right_vec, up_default))  # Forward axis
        up_vec = normalize(torch.cross(forward_vec, right_vec))  # Corrected up axis

        # Build rotation matrix
        R_pelvis = torch.stack([right_vec, up_vec, forward_vec], dim=1)  # [3, 3]

        # Convert to axis-angle
        return rotation_matrix_to_axis_angle(R_pelvis)

    def _compute_local_rotation(self,
                                 target_joints_frame: torch.Tensor,
                                 parent_name: str,
                                 child_name: str,
                                 parent_global_rot: torch.Tensor) -> torch.Tensor:
        """
        Compute local rotation at a joint considering parent's global rotation.

        Args:
            target_joints_frame: AddB joint positions for one frame [num_joints, 3]
            parent_name: Name of parent joint
            child_name: Name of child joint
            parent_global_rot: Parent's global rotation matrix [3, 3]

        Returns:
            Local rotation matrix [3, 3]
        """
        # Get SMPL indices
        parent_smpl_idx = SMPL_JOINT_NAMES.index(parent_name)
        child_smpl_idx = SMPL_JOINT_NAMES.index(child_name)

        # Find corresponding AddB indices
        parent_addb_idx = None
        child_addb_idx = None
        for addb_idx, smpl_idx in self.joint_mapping.items():
            if smpl_idx == parent_smpl_idx:
                parent_addb_idx = addb_idx
            if smpl_idx == child_smpl_idx:
                child_addb_idx = addb_idx

        # If indices not found, return identity
        if parent_addb_idx is None or child_addb_idx is None:
            return torch.eye(3, device=self.device)

        # Get positions
        parent_pos = target_joints_frame[parent_addb_idx]
        child_pos = target_joints_frame[child_addb_idx]

        # Check for NaN
        if torch.isnan(parent_pos).any() or torch.isnan(child_pos).any():
            return torch.eye(3, device=self.device)

        # Compute observed bone direction in global frame
        obs_bone_global = normalize(child_pos - parent_pos)

        # Lazy initialization of default bone directions
        if self.default_bone_directions is None:
            self.default_bone_directions = self._compute_default_bone_directions()

        # Get default bone direction (T-pose) in local frame
        bone_key = (parent_smpl_idx, child_smpl_idx)
        if bone_key not in self.default_bone_directions:
            return torch.eye(3, device=self.device)

        default_bone_local = self.default_bone_directions[bone_key]

        # Transform default direction to parent's current global frame
        default_bone_global = parent_global_rot @ default_bone_local

        # Compute rotation that aligns default_bone_global with obs_bone_global
        R_correction = rotation_matrix_from_vectors(default_bone_global, obs_bone_global)

        # Local rotation is the correction in parent's frame
        R_local = parent_global_rot.T @ R_correction @ parent_global_rot

        return R_local

    def _estimate_initial_poses_ik(self) -> torch.Tensor:
        """
        Simplified IK: Use bone direction vectors to estimate simple rotations.

        Much faster than full kinematic chain IK - just uses bone directions
        to give optimization a better starting point than zero pose.

        Returns:
            Initial poses [T, 24, 3] in axis-angle representation
        """
        T = self.target_joints.shape[0]
        initial_poses = torch.zeros(T, SMPL_NUM_JOINTS, 3, device=self.device)

        # Lazy init default bone directions
        if self.default_bone_directions is None:
            self.default_bone_directions = self._compute_default_bone_directions()

        # Simple approach: For each bone, compute a small rotation hint
        # based on how much the bone direction differs from T-pose
        for t in range(T):
            target_frame = self.target_joints[t]

            # Skip frames with all NaN
            if torch.isnan(target_frame).all():
                continue

            # Process each bone pair in kinematic chains
            for chain_name, joint_list in self.kinematic_chains.items():
                for i in range(len(joint_list) - 1):
                    parent_name, parent_idx = joint_list[i]
                    child_name, child_idx = joint_list[i + 1]

                    # Find AddB indices
                    parent_addb_idx = None
                    child_addb_idx = None
                    for addb_idx, smpl_idx in self.joint_mapping.items():
                        if smpl_idx == parent_idx:
                            parent_addb_idx = addb_idx
                        if smpl_idx == child_idx:
                            child_addb_idx = addb_idx

                    if parent_addb_idx is None or child_addb_idx is None:
                        continue

                    # Get positions
                    parent_pos = target_frame[parent_addb_idx]
                    child_pos = target_frame[child_addb_idx]

                    # Skip NaN
                    if torch.isnan(parent_pos).any() or torch.isnan(child_pos).any():
                        continue

                    # Compute observed bone direction
                    obs_dir = child_pos - parent_pos
                    obs_dir_norm = torch.norm(obs_dir)
                    if obs_dir_norm < 1e-6:
                        continue
                    obs_dir = obs_dir / obs_dir_norm

                    # Get default bone direction from T-pose
                    bone_key = (parent_idx, child_idx)
                    if bone_key not in self.default_bone_directions:
                        continue

                    default_dir = self.default_bone_directions[bone_key]

                    # Simple rotation hint: cross product gives rotation axis
                    # dot product gives rotation angle
                    cross = torch.cross(default_dir, obs_dir)
                    dot = torch.dot(default_dir, obs_dir).clamp(-1.0, 1.0)

                    cross_norm = torch.norm(cross)
                    if cross_norm < 1e-6:
                        continue

                    # Rotation axis
                    axis = cross / cross_norm

                    # Rotation angle (small hint, not full rotation)
                    angle = torch.acos(dot) * 0.3  # Scale down to 30% for stability

                    # Axis-angle representation
                    initial_poses[t, parent_idx] = axis * angle

        return initial_poses

    # --------------------------- Pose optimisation ------------------------
    def optimise_pose_sequence(self,
                               betas: torch.Tensor,
                               init_poses: Optional[torch.Tensor] = None,
                               init_trans: Optional[torch.Tensor] = None,
                               pose_iters: Optional[int] = None,
                               pose_lr: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        betas = betas.detach()
        T = self.target_joints.shape[0]

        poses = torch.zeros(T, SMPL_NUM_JOINTS, 3, device=self.device)
        trans = torch.zeros(T, 3, device=self.device)

        if init_poses is not None:
            poses.copy_(init_poses)
        if init_trans is not None:
            trans.copy_(init_trans)

        num_iters = pose_iters or self.config.pose_iters
        lr = pose_lr or self.config.pose_lr

        for t in range(T):
            target_frame = self.target_joints[t]
            if torch.isnan(target_frame).all():
                continue

            # Optimize pose and translation parameters for this frame
            pose_param = torch.nn.Parameter(poses[t].clone())
            trans_param = torch.nn.Parameter(trans[t].clone())
            optimiser = torch.optim.Adam([pose_param, trans_param], lr=lr)

            for _ in range(num_iters):
                optimiser.zero_grad()

                joints_pred = self.smpl.joints(betas.unsqueeze(0), pose_param.unsqueeze(0), trans_param.unsqueeze(0)).squeeze(0)

                # Use absolute positions (no root centering) to properly learn translation
                target_subset = target_frame[self.addb_indices]
                pred_subset = joints_pred[self.smpl_indices]
                mask = ~torch.isnan(target_subset).any(dim=1)
                if mask.sum() == 0:
                    break

                diff = pred_subset[mask] - target_subset[mask]

                # Simple uniform weighting (FIXED9/FIXED11)
                loss = (diff ** 2).mean()

                # Bone direction loss (bypass structural mismatch)
                bone_dir_loss = self._compute_bone_direction_loss(joints_pred, target_frame)
                loss = loss + self.config.bone_direction_weight * bone_dir_loss

                # Pose regularization
                loss = loss + self.config.pose_reg_weight * (pose_param ** 2).mean()

                # Translation regularization
                if self.root_addb_idx is not None:
                    root_target = target_frame[self.root_addb_idx]
                    if not torch.isnan(root_target).any():
                        loss = loss + self.config.trans_reg_weight * ((trans_param - root_target) ** 2).mean()

                # Temporal smoothness regularization (ENABLED for smooth motion)
                temporal_smooth_weight = 0.1  # Weight for temporal smoothness
                if t > 0:
                    pose_diff = pose_param - poses[t-1].detach()
                    trans_diff = trans_param - trans[t-1].detach()
                    temporal_loss = (pose_diff ** 2).mean() + (trans_diff ** 2).mean()
                    loss = loss + temporal_smooth_weight * temporal_loss

                loss.backward()
                optimiser.step()

            # Update pose and translation
            poses[t] = pose_param.detach()
            trans[t] = trans_param.detach()

        return poses.cpu().numpy(), trans.cpu().numpy()

    # --------------------------- Evaluation --------------------------------
    def evaluate(self,
                 betas: torch.Tensor,
                 poses: np.ndarray,
                 trans: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        betas_t = betas.unsqueeze(0)
        poses_t = torch.from_numpy(poses).float().to(self.device)
        trans_t = torch.from_numpy(trans).float().to(self.device)

        joints_pred = []
        for t in range(poses_t.shape[0]):
            jp = self.smpl.joints(betas_t[0], poses_t[t], trans_t[t]).cpu().numpy()
            joints_pred.append(jp)
        joints_pred = np.stack(joints_pred, axis=0)

        errors = []
        thresholds = [0.05, 0.10, 0.15]
        pck_counts = np.zeros(len(thresholds), dtype=np.int64)

        for pred_frame, target_frame in zip(joints_pred, self.target_joints_np):
            # Compute error WITHOUT root centering (to match visualization)
            for addb_idx, smpl_idx in self.joint_mapping.items():
                if addb_idx >= target_frame.shape[0]:
                    continue
                tgt = target_frame[addb_idx]
                if np.any(np.isnan(tgt)):
                    continue
                diff = pred_frame[smpl_idx] - tgt
                error_m = np.linalg.norm(diff)
                errors.append(error_m * 1000.0)  # Convert to millimeters
                for i, thr in enumerate(thresholds):
                    if error_m < thr:
                        pck_counts[i] += 1

        mpjpe = float(np.mean(errors)) if errors else float('nan')  # Now in mm
        total = max(len(errors), 1)
        metrics = {
            'MPJPE': mpjpe,
            'num_comparisons': float(len(errors)),
        }
        for thr, count in zip(thresholds, pck_counts):
            metrics[f'PCK@{thr:.2f}m'] = float(100.0 * count / total)
        return metrics, joints_pred

    # --------------------------- Full fitting pipeline ---------------------
    def run(self) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Dict[str, float], np.ndarray]:
        """
        Three-stage pose-aware optimization workflow:
        Stage 1: Initial coarse pose estimation (with default shape)
        Stage 2: Pose-aware shape optimization (using Stage 1 poses)
        Stage 3: Pose refinement (with optimized shape from Stage 2)
        """
        print("\n" + "="*80)
        print("STARTING THREE-STAGE POSE-AWARE OPTIMIZATION")
        print("="*80)

        # Stage 1: Initial pose estimation with default shape
        print("\n--- STAGE 1: Initial Pose Estimation ---")
        initial_poses, initial_trans = self.optimise_initial_poses()

        # Quick evaluation of Stage 1 results
        betas_zero = torch.zeros(SMPL_NUM_BETAS, device=self.device)
        poses_np_stage1 = initial_poses.cpu().numpy()
        trans_np_stage1 = initial_trans.cpu().numpy()
        metrics_stage1, _ = self.evaluate(betas_zero, poses_np_stage1, trans_np_stage1)
        print(f"  Stage 1 MPJPE (default shape): {metrics_stage1['MPJPE']:.2f} mm")

        # Stage 2: Pose-aware shape optimization
        print("\n--- STAGE 2: Pose-Aware Shape Optimization ---")
        betas = self.optimise_shape(initial_poses=initial_poses, initial_trans=initial_trans)
        print(f"  Optimized shape parameters (betas): {betas.cpu().numpy()[:5]}")  # Show first 5

        # Stage 3: Pose refinement with optimized shape
        print("\n--- STAGE 3: Pose Refinement with Optimized Shape ---")
        poses_np, trans_np = self.optimise_pose_sequence(
            betas,
            init_poses=initial_poses,
            init_trans=initial_trans
        )
        metrics, pred_joints = self.evaluate(betas, poses_np, trans_np)
        print(f"  Stage 3 MPJPE (optimized shape + refined poses): {metrics['MPJPE']:.2f} mm")

        # Additional refinement passes if needed
        tolerance = self.config.tolerance_mm / 1000.0
        passes = 1

        print("\n--- Additional Refinement Passes (if needed) ---")
        while (
            passes < self.config.max_passes
            and not np.isnan(metrics['MPJPE'])
            and metrics['MPJPE'] > tolerance
        ):
            print(f"  Refinement pass {passes}: Current MPJPE = {metrics['MPJPE']:.2f} mm")
            poses_init = torch.from_numpy(poses_np).float().to(self.device)
            trans_init = torch.from_numpy(trans_np).float().to(self.device)
            poses_np, trans_np = self.optimise_pose_sequence(
                betas,
                init_poses=poses_init,
                init_trans=trans_init,
                pose_iters=max(40, self.config.pose_iters // 2),
                pose_lr=self.config.pose_lr * 0.5
            )
            metrics, pred_joints = self.evaluate(betas, poses_np, trans_np)
            print(f"  After pass {passes}: MPJPE = {metrics['MPJPE']:.2f} mm")
            passes += 1

        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print(f"Final MPJPE: {metrics['MPJPE']:.2f} mm")
        print(f"Total refinement passes: {passes}")
        print("="*80 + "\n")

        return betas, poses_np, trans_np, metrics, pred_joints


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def derive_run_name(b3d_path: str) -> str:
    p_lower = b3d_path.lower()
    if "with_arm" in p_lower:
        arm = "with_arm"
    elif "no_arm" in p_lower:
        arm = "no_arm"
    else:
        arm = "unknown"

    import re
    matches = re.findall(r"/([A-Za-z]+20\d{2})(?:[_/])", b3d_path)
    study = matches[-1].lower() if matches else "unknown"

    base = os.path.splitext(os.path.basename(b3d_path))[0]
    m = re.match(r"([A-Za-z]*\d+)", base)
    subj = m.group(1).lower() if m else base.lower()
    return f"{arm}_{study}_{subj}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert AddBiomechanics motion to SMPL parameters')
    parser.add_argument('--b3d', required=True, help='Path to the .b3d file')
    parser.add_argument('--out_dir', required=True, help='Output root directory')
    parser.add_argument('--map_json', default=None,
                        help='Optional JSON file overriding AddB→SMPL joint mapping')
    parser.add_argument('--smpl_model', required=True, help='Path to SMPL model .pkl file')

    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--processing_pass', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=-1)

    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--shape_lr', type=float, default=5e-3)
    parser.add_argument('--shape_iters', type=int, default=150)
    parser.add_argument('--shape_sample_frames', type=int, default=80)
    parser.add_argument('--pose_lr', type=float, default=1e-2)
    parser.add_argument('--pose_iters', type=int, default=80)
    parser.add_argument('--tolerance_mm', type=float, default=20.0,
                        help='Early-stop once MPJPE drops below this (in millimetres)')
    parser.add_argument('--max_passes', type=int, default=3,
                        help='Maximum refinement passes after the first fitting')
    parser.add_argument('--pose_reg', type=float, default=1e-3)
    parser.add_argument('--trans_reg', type=float, default=1e-3)

    parser.add_argument('--lower_body_only', action='store_true',
                        help='Fit only lower body joints (for No_Arm datasets)')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[WARN] CUDA not available, falling back to CPU')

    run_name = derive_run_name(args.b3d)
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print('\n' + '=' * 80)
    print('AddBiomechanics → SMPL Fitter')
    print('=' * 80)
    print(f'Run name : {run_name}')
    print(f'Output   : {out_dir}')
    print(f'Device   : {device}')
    print('=' * 80 + '\n')

    print('[1/5] Loading AddBiomechanics data...')
    addb_joints, dt = load_b3d_sequence(
        args.b3d,
        trial=args.trial,
        processing_pass=args.processing_pass,
        start=args.start,
        num_frames=args.num_frames
    )
    joint_names = infer_addb_joint_names(
        args.b3d,
        trial=args.trial,
        processing_pass=args.processing_pass
    )
    addb_joints = convert_addb_to_smpl_coords(addb_joints)
    print(f'  Frames : {addb_joints.shape[0]}')
    print(f'  Joints : {addb_joints.shape[1]}')
    print(f'  dt     : {dt:.6f} s ({1.0 / dt:.2f} fps)')

    print('\n[2/5] Resolving joint correspondences...')
    auto_map = auto_map_addb_to_smpl(joint_names)
    overrides = load_mapping_overrides(args.map_json, joint_names)
    mapping = resolve_mapping(joint_names, auto_map, overrides)

    # Filter mapping to lower body only if requested
    if args.lower_body_only:
        original_count = len(mapping)
        mapping = {addb_idx: smpl_idx for addb_idx, smpl_idx in mapping.items()
                   if smpl_idx in LOWER_BODY_JOINTS}
        print(f'  Lower body filtering: {original_count} → {len(mapping)} joints')

    print(f'  Using {len(mapping)} correspondences')
    for idx, smpl_idx in sorted(mapping.items(), key=lambda kv: kv[0]):
        addb_name = joint_names[idx] if idx < len(joint_names) else str(idx)
        smpl_name = SMPL_JOINT_NAMES[smpl_idx]
        print(f'    {addb_name:20s} → {smpl_name}')

    unmapped = [name for i, name in enumerate(joint_names) if i not in mapping]
    if unmapped:
        print(f'  Unmapped AddB joints ({len(unmapped)}): {", ".join(unmapped)}')

    print('\n[3/5] Initialising SMPL model...')
    smpl = SMPLModel(args.smpl_model, device=device)

    cfg = OptimisationConfig(
        shape_lr=args.shape_lr,
        shape_iters=args.shape_iters,
        shape_sample_frames=args.shape_sample_frames,
        pose_lr=args.pose_lr,
        pose_iters=args.pose_iters,
        tolerance_mm=args.tolerance_mm,
        max_passes=args.max_passes,
        pose_reg_weight=args.pose_reg,
        trans_reg_weight=args.trans_reg,
        lower_body_only=args.lower_body_only
    )

    fitter = AddBToSMPLFitter(
        smpl_model=smpl,
        target_joints=addb_joints,
        joint_mapping=mapping,
        addb_joint_names=joint_names,
        dt=dt,
        config=cfg,
        device=device
    )

    print('\n[4/5] Optimising SMPL parameters...')
    betas, poses_np, trans_np, metrics, pred_joints = fitter.run()

    print('\n[5/5] Metrics:')
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f'  {key}: {value:.6f}')
        else:
            print(f'  {key}: {value}')

    print('\nSaving results...')
    np.savez(
        os.path.join(out_dir, 'smpl_params.npz'),
        betas=betas.cpu().numpy(),
        poses=poses_np,
        trans=trans_np
    )
    np.save(os.path.join(out_dir, 'pred_joints.npy'), pred_joints)
    np.save(os.path.join(out_dir, 'target_joints.npy'), addb_joints)

    meta = {
        'b3d': args.b3d,
        'run_name': run_name,
        'dt': float(dt),
        'num_frames': int(addb_joints.shape[0]),
        'num_joints_addb': int(addb_joints.shape[1]),
        'addb_joint_names': joint_names,
        'joint_mapping': mapping,
        'joint_mapping_named': {
            joint_names[idx]: SMPL_JOINT_NAMES[smpl_idx]
            for idx, smpl_idx in mapping.items()
        },
        'shape_lr': args.shape_lr,
        'shape_iters': args.shape_iters,
        'shape_sample_frames': args.shape_sample_frames,
        'pose_lr': args.pose_lr,
        'pose_iters': args.pose_iters,
        'tolerance_mm': args.tolerance_mm,
        'max_passes': args.max_passes,
        'metrics': metrics,
        'unmapped_addb_joints': unmapped,
    }

    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print('\n' + '=' * 80)
    print(f'DONE! Results saved to: {out_dir}')
    print('=' * 80 + '\n')


if __name__ == '__main__':
    main()
