#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AddBiomechanics (.b3d) → SMPL parameter fitting - v11_with_physics.

This version (v11_with_physics) extends v10_foot_orient_late with physics data extraction:

NEW in v11: **PHYSICS DATA EXTRACTION**
    - Extracts ground reaction forces, joint torques, velocities, accelerations
    - Extracts center of mass position, velocity, acceleration
    - Extracts residual forces at root
    - Physics data is extracted for the EXACT SAME sampled frames used for optimization
    - Saves physics GT data for future training:
        * ground_reaction_forces.npy: [T, 6] (3 force + 3 torque)
        * joint_torques.npy: [T, N_dofs]
        * joint_velocities.npy: [T, N_dofs]
        * joint_accelerations.npy: [T, N_dofs]
        * com_position.npy: [T, 3]
        * com_velocity.npy: [T, 3]
        * com_acceleration.npy: [T, 3]
        * residual_forces.npy: [T, 6]
        * sampled_frame_indices.npy: [T]

This version (v10_foot_orient_late) implements FIX 10 to apply foot orientation loss
ONLY in later stages (Stage 3 and Stage 4), not in early stages (Stage 1-2):

v3_enhanced features (baseline):
    1. Per-Joint Learning Rate
    2. Bone Length Soft Constraint
    3. Multi-Stage Bone Direction Weights
    4. Contact-Aware Optimization
    5. Per-joint position weighting
    → Achieved: ~75mm MPJPE on P020_split2

FIX 10: **FOOT ORIENTATION LOSS - LATE STAGE APPLICATION** (NEW!)
    - Problem: SMPL foot joints (indices 10, 11) pointing sideways (77-85°)
      instead of forward like AddBiomechanics GT toe joints (MTP, indices 5, 10 in AddB data)
    - Solution: Add direct foot orientation supervision loss
        - SMPL foot vectors: left_foot_vec = joints[:,10] - joints[:,7] (left_ankle)
                            right_foot_vec = joints[:,11] - joints[:,8] (right_ankle)
        - AddB toe vectors: left_toe_vec = target_joints[:,10] - target_joints[:,8] (left ankle in AddB)
                           right_toe_vec = target_joints[:,5] - target_joints[:,3] (right ankle in AddB)
        - Normalize both vectors
        - Compute cosine similarity loss: 1 - dot_product(normalized vectors)
        - Returns mean loss for both feet
    - Applied ONLY to Stage 3 (per-frame) and Stage 4 (sequence) with weight 0.2
    - Rationale: Stage 1-2 need to establish overall pose first; foot orientation
      fine-tuning should come later to avoid interfering with initial convergence

Expected improvements:
    - Foot orientation: 77-85° → closer to forward direction
    - MPJPE: maintained or improved
    - Visual quality: correct foot orientation
    - Better convergence by applying constraint only after initial pose is established

Given an AddBiomechanics sequence we:
    1. infer the ordered AddB joint list and auto-map it to SMPL joints,
    2. optimise SMPL betas (shape) on a subsampled set of frames,
    3. hierarchically optimise pose + translation with all enhancements + FOOT ORIENTATION (late stage),
    4. apply sequence-level refinement with velocity/ground constraints + FOOT ORIENTATION.

Outputs include SMPL parameters, reconstructed joints, evaluation metrics,
metadata and visualisations.
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
    import yaml
except ImportError:
    yaml = None  # YAML config optional

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
    # FIX 9: Include MTP joints for foot orientation loss
    # These joints are needed to compute the target foot direction vectors
    'mtp_r': 'right_foot',
    'mtp_l': 'left_foot',
    'toe_r': 'right_foot',
    'toe_l': 'left_foot',
    'right_toe': 'right_foot',
    'left_toe': 'left_foot',
    # 'subtalar_r': SKIPPED - no SMPL equivalent
    # 'subtalar_l': SKIPPED - no SMPL equivalent
    # FIX 12: DISABLE back->spine2 mapping
    # Problem: AddB 'back' is only 8.6cm above pelvis, but SMPL 'spine2' is 23.8cm above pelvis
    # This mismatch causes the optimizer to bend the spine forward ~60+ degrees to match positions
    # Solution: Remove mapping - let shoulder joints guide upper body orientation instead
    # 'back': 'spine2',  # DISABLED - causes severe forward lean
    # 'torso': 'spine2',  # DISABLED
    # 'spine': 'spine2',  # DISABLED
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
    # ENHANCEMENT: Head/Neck variants (for --include_head_neck)
    'neck': 'neck',
    'cervical': 'neck',
    'cervical_spine': 'neck',
    'c_spine': 'neck',
    'c7': 'neck',
    'neck_joint': 'neck',
    'head': 'head',
    'skull': 'head',
    'cranium': 'head',
    'head_joint': 'head',
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

# ENHANCEMENT: Head/Neck joint indices (for --include_head_neck)
HEAD_NECK_JOINTS = [
    12,  # neck
    15,  # head
]

# Full body with head/neck (when --include_head_neck is enabled)
FULL_BODY_WITH_HEAD_JOINTS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,  # Lower body + spine
    12,  # neck
    13, 14,  # collars
    15,  # head
    16, 17, 18, 19, 20, 21, 22, 23,  # Arms + hands
]


# ---------------------------------------------------------------------------
# Adaptive Frame Sampling (for long sequences)
# ---------------------------------------------------------------------------

def adaptive_frame_sampling(
    joints: np.ndarray,
    max_frames: int = 500
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Uniformly sample frames to reduce long sequences while preserving temporal information.

    Strategy:
    - If T <= max_frames: Use all frames (no sampling)
    - If T > max_frames: Sample every N-th frame where N = ceil(T / max_frames)
    - Always include first and last frame

    Args:
        joints: [T, N, 3] joint positions array
        max_frames: Maximum number of frames to keep (default: 500)

    Returns:
        sampled_joints: [T', N, 3] where T' <= max_frames
        selected_indices: Original frame indices [T']
        stride: Sampling stride used

    Example:
        2,000 frames → stride 4 → 500 frames (4× faster optimization)
    """
    T = joints.shape[0]

    if T <= max_frames:
        # No sampling needed
        return joints, np.arange(T), 1

    # Compute stride to achieve ~max_frames
    stride = int(np.ceil(T / max_frames))
    selected_indices = np.arange(0, T, stride)

    # Ensure last frame is always included
    if selected_indices[-1] != T - 1:
        selected_indices = np.append(selected_indices[:-1], T - 1)

    sampled_joints = joints[selected_indices]

    print(f'  [Adaptive Sampling] {T} frames → {len(selected_indices)} frames (stride={stride})')
    print(f'    Speed improvement: ~{stride:.1f}× faster optimization')

    return sampled_joints, selected_indices, stride


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


def extract_physics_for_sampled_frames(
    b3d_path: str,
    sampled_frame_indices: np.ndarray,
    trial: int = 0,
    processing_pass: int = 0,
    start: int = 0
) -> dict:
    """
    Extract physics data from .b3d file for the EXACT sampled frames.

    Args:
        b3d_path: Path to .b3d file
        sampled_frame_indices: Array of frame indices that were sampled (relative to start frame)
        trial: Trial index
        processing_pass: Processing pass (0=KINEMATICS, 1=FILTER, 2=DYNAMICS)
        start: Starting frame offset

    Returns:
        dict with keys:
            - grf: (T, 6) ground reaction forces and torques [fx, fy, fz, tx, ty, tz]
            - joint_torques: (T, N_dof) joint torques
            - joint_velocities: (T, N_dof) joint velocities
            - joint_accelerations: (T, N_dof) joint accelerations
            - com_pos: (T, 3) center of mass position
            - com_vel: (T, 3) center of mass velocity
            - com_acc: (T, 3) center of mass acceleration
            - residual_forces: (T, 6) residual forces/moments at root
            - sampled_frame_indices: (T,) the frame indices that were sampled
            - n_dof: number of DOFs
    """
    subj = nimble.biomechanics.SubjectOnDisk(b3d_path)

    # Get skeleton to know DOF count
    skel = subj.readSkel(processingPass=processing_pass)
    n_dof = skel.getNumDofs()

    # Number of sampled frames
    num_sampled = len(sampled_frame_indices)

    # Initialize arrays for sampled frames only
    grf = np.zeros((num_sampled, 6), dtype=np.float32)  # 3 force + 3 torque
    joint_torques = np.zeros((num_sampled, n_dof), dtype=np.float32)
    joint_velocities = np.zeros((num_sampled, n_dof), dtype=np.float32)
    joint_accelerations = np.zeros((num_sampled, n_dof), dtype=np.float32)
    com_pos = np.zeros((num_sampled, 3), dtype=np.float32)
    com_vel = np.zeros((num_sampled, 3), dtype=np.float32)
    com_acc = np.zeros((num_sampled, 3), dtype=np.float32)
    residual_forces = np.zeros((num_sampled, 6), dtype=np.float32)

    # Read all frames first (we need to read sequentially from the .b3d file)
    total_frames = subj.getTrialLength(trial)
    max_frame_idx = int(sampled_frame_indices[-1])
    frames_to_read = min(max_frame_idx + 1, total_frames - start)

    frames = subj.readFrames(
        trial=trial,
        startFrame=start,
        numFramesToRead=frames_to_read,
        includeProcessingPasses=True,
        includeSensorData=False,
        stride=1,
        contactThreshold=1.0
    )

    # Extract physics data only for sampled frames
    for out_idx, frame_idx in enumerate(sampled_frame_indices):
        frame_idx = int(frame_idx)
        if frame_idx >= len(frames):
            continue

        frame = frames[frame_idx]

        if hasattr(frame, 'processingPasses') and len(frame.processingPasses) > processing_pass:
            pp = frame.processingPasses[processing_pass]

            # Ground reaction forces - sum all contact forces
            if hasattr(pp, 'groundContactForce'):
                grf_raw = np.array(pp.groundContactForce, dtype=np.float32)
                if len(grf_raw) > 0:
                    grf_reshaped = grf_raw.reshape(-1, 3)
                    grf[out_idx, :3] = grf_reshaped.sum(axis=0)  # Sum forces

            # Ground reaction torques
            if hasattr(pp, 'groundContactTorque'):
                torque_raw = np.array(pp.groundContactTorque, dtype=np.float32)
                if len(torque_raw) > 0:
                    torque_reshaped = torque_raw.reshape(-1, 3)
                    grf[out_idx, 3:] = torque_reshaped.sum(axis=0)  # Sum torques

            # Joint torques
            if hasattr(pp, 'tau'):
                tau_raw = np.array(pp.tau, dtype=np.float32)
                if len(tau_raw) >= n_dof:
                    joint_torques[out_idx] = tau_raw[:n_dof]
                else:
                    joint_torques[out_idx, :len(tau_raw)] = tau_raw

            # Joint velocities
            if hasattr(pp, 'vel'):
                vel_raw = np.array(pp.vel, dtype=np.float32)
                if len(vel_raw) >= n_dof:
                    joint_velocities[out_idx] = vel_raw[:n_dof]
                else:
                    joint_velocities[out_idx, :len(vel_raw)] = vel_raw

            # Joint accelerations
            if hasattr(pp, 'acc'):
                acc_raw = np.array(pp.acc, dtype=np.float32)
                if len(acc_raw) >= n_dof:
                    joint_accelerations[out_idx] = acc_raw[:n_dof]
                else:
                    joint_accelerations[out_idx, :len(acc_raw)] = acc_raw

            # Center of mass
            if hasattr(pp, 'comPos'):
                com_pos[out_idx] = np.array(pp.comPos, dtype=np.float32)
            if hasattr(pp, 'comVel'):
                com_vel[out_idx] = np.array(pp.comVel, dtype=np.float32)
            if hasattr(pp, 'comAcc'):
                com_acc[out_idx] = np.array(pp.comAcc, dtype=np.float32)

            # Residual forces/moments
            if hasattr(pp, 'residualWrenchInRootFrame'):
                res_raw = np.array(pp.residualWrenchInRootFrame, dtype=np.float32)
                if len(res_raw) >= 6:
                    residual_forces[out_idx] = res_raw[:6]

    return {
        'grf': grf,
        'joint_torques': joint_torques,
        'joint_velocities': joint_velocities,
        'joint_accelerations': joint_accelerations,
        'com_pos': com_pos,
        'com_vel': com_vel,
        'com_acc': com_acc,
        'residual_forces': residual_forces,
        'sampled_frame_indices': sampled_frame_indices,
        'n_dof': n_dof
    }


def convert_addb_to_smpl_coords(joints: np.ndarray) -> np.ndarray:
    """
    Convert AddBiomechanics (OpenSim) coordinates to SMPL coordinates

    AddBiomechanics (OpenSim):   SMPL:
    X = forward                  X = right
    Y = up              →        Y = up
    Z = right                    Z = forward

    Transformation: (X, Y, Z) → (Z, Y, X)
    """
    converted = joints.copy()
    converted[..., 0] = joints[..., 2]  # X_smpl = Z_addb (right)
    converted[..., 1] = joints[..., 1]  # Y_smpl = Y_addb (up)
    converted[..., 2] = joints[..., 0]  # Z_smpl = X_addb (forward)
    return converted


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
    shape_iters: int = 100  # Reduced from 150 for 24h deadline (-33%)
    shape_sample_frames: int = 80

    pose_lr: float = 1e-2
    pose_iters: int = 40  # Reduced from 80 for 24h deadline (-50%, biggest impact)

    tolerance_mm: float = 20.0
    max_passes: int = 1  # Reduced from 3 for 24h deadline (-67%, second biggest impact)

    pose_reg_weight: float = 1e-3
    trans_reg_weight: float = 1e-3
    bone_length_weight: float = 100.0  # Weight for bone length constraint (FIXED9)
    temporal_smooth_weight: float = 0.5  # Temporal smoothness regularization (increased from 0.1 to reduce jitter)
    bone_direction_weight: float = 0.5  # Weight for bone direction matching loss

    # Enhancement 1: Per-Joint Learning Rates
    foot_lr: float = 5e-3  # Lower LR for feet (more stable)
    knee_lr: float = 1e-2  # Medium LR for knees
    hip_lr: float = 1.5e-2  # Higher LR for hips (more DOF)

    # Enhancement 2: Bone Length Soft Constraint
    bone_length_soft_weight: float = 0.1  # Weight for soft bone length consistency
    bone_length_ratio_weight: float = 0.5  # Weight for bone length ratio matching

    # Enhancement 3: Multi-Stage Bone Direction Weights (will be computed dynamically)
    # Stage-specific weights defined in _get_stage_bone_weights()

    # Enhancement 4: Contact-Aware Optimization
    contact_weight_multiplier: float = 2.0  # Multiplier when foot contacts ground

    # Enhancement 5: Velocity Smoothness Loss (논문 Eq. 8, λ=0.05)
    velocity_smooth_weight: float = 0.5  # Weight for velocity smoothness (increased to reduce jitter)

    # Enhancement 6: Joint Angle Limits
    angle_limit_weight: float = 1.0  # Weight for joint angle limit violations

    # Enhancement 7: Ground Penetration Penalty
    ground_penetration_weight: float = 10.0  # Strong penalty for feet below ground

    # Enhancement 8: Hierarchical Optimization
    use_hierarchical_opt: bool = False  # Disable for now - needs tuning

    # FIX 4: Foot position computed analytically - no foot orientation loss needed
    # Removed foot_orientation_weight - using analytical foot calculation instead

    # FIX 4: Absolute bone length matching weight
    absolute_bone_length_weight: float = 1.0  # Weight for absolute bone length matching

    # FIX 6 (v6): Relative pelvis constraint (allows natural running lean)
    pelvis_upright_weight: float = 2.0  # Weaker constraint - maintain natural orientation

    # FIX 5 (v5): Ankle rotation limits
    ankle_limit_weight: float = 5.0  # Limit ankle rotation to anatomical range


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

        # FIX 4: Store AddB foot joint indices for analytical foot position calculation
        self._find_addb_foot_indices()

        # FIX 6 (v6): Store AddB GT pelvis orientation for relative constraint
        # Will be computed from initial pose estimation
        self.addb_pelvis_reference = None

    # --------------------------- Shape optimisation -----------------------
    def _detect_keyframes(self, num_samples: int = 50) -> np.ndarray:
        """
        ENHANCEMENT: Smart keyframe detection based on joint movement.

        Instead of uniform sampling, select frames with high movement/velocity.
        This gives better shape estimation with fewer samples.

        Args:
            num_samples: Number of keyframes to select

        Returns:
            Array of keyframe indices
        """
        T = self.target_joints_np.shape[0]

        if T <= num_samples:
            return np.arange(T)

        # Compute frame-to-frame joint velocities
        velocities = np.diff(self.target_joints_np, axis=0)  # [T-1, N, 3]

        # Compute movement magnitude per frame (ignoring NaN)
        movement = np.zeros(T - 1)
        for t in range(T - 1):
            valid_mask = ~np.isnan(velocities[t]).any(axis=1)
            if valid_mask.sum() > 0:
                movement[t] = np.linalg.norm(velocities[t][valid_mask], axis=1).mean()

        # Always include first and last frame
        keyframe_indices = [0, T - 1]
        remaining_samples = num_samples - 2

        if remaining_samples > 0:
            # Sort frames by movement (descending)
            sorted_indices = np.argsort(movement)[::-1]

            # Select top N frames with highest movement
            selected = sorted_indices[:remaining_samples]
            keyframe_indices.extend(selected.tolist())

        # Sort and return
        keyframe_indices = sorted(set(keyframe_indices))
        return np.array(keyframe_indices, dtype=int)

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

    # --------------------------- FIX 4: Analytical Foot Position -----------------------
    def _find_addb_foot_indices(self) -> None:
        """
        Find AddB foot joint indices for analytical foot position calculation.

        Need: subtalar (heel), mtp (toe) for foot direction vector
        """
        def find_idx(names_list):
            for name in names_list:
                if name in self.addb_name_to_idx:
                    return self.addb_name_to_idx[name]
            return None

        # AddB foot joints: subtalar (heel), mtp (toe)
        self.addb_left_subtalar_idx = find_idx(['subtalar_l', 'calcn_l', 'left_heel'])
        self.addb_right_subtalar_idx = find_idx(['subtalar_r', 'calcn_r', 'right_heel'])
        self.addb_left_mtp_idx = find_idx(['mtp_l', 'toe_l', 'left_toe'])
        self.addb_right_mtp_idx = find_idx(['mtp_r', 'toe_r', 'right_toe'])

        # SMPL foot joint indices
        self.smpl_left_ankle_idx = 7
        self.smpl_right_ankle_idx = 8
        self.smpl_left_foot_idx = 10
        self.smpl_right_foot_idx = 11

        # Check if analytical foot calculation is possible
        self.has_analytical_foot = all([
            self.addb_left_subtalar_idx is not None,
            self.addb_right_subtalar_idx is not None,
            self.addb_left_mtp_idx is not None,
            self.addb_right_mtp_idx is not None
        ])

        if self.has_analytical_foot:
            print(f"  [Analytical Foot] AddB foot joints found:")
            print(f"    Left:  {self.addb_joint_names[self.addb_left_subtalar_idx]} → {self.addb_joint_names[self.addb_left_mtp_idx]}")
            print(f"    Right: {self.addb_joint_names[self.addb_right_subtalar_idx]} → {self.addb_joint_names[self.addb_right_mtp_idx]}")
            print(f"    SMPL foot will be computed as: ankle + 0.75 × (subtalar→mtp direction) × foot_length")
        else:
            print("  [Analytical Foot] WARNING: Could not find foot joints - using default SMPL foot position")

    def _apply_analytical_foot_position(self, smpl_joints: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        FIX 4 + FIX 5: Apply analytical foot position calculation with temporal smoothing.

        Replaces SMPL foot joint positions with analytical calculation:
        foot_position = ankle + 0.75 × (subtalar→mtp direction) × foot_length

        FIX 5 adds:
        - Temporal smoothing to prevent jitter
        - Outlier detection for sudden direction changes
        - Validation of foot angle

        Args:
            smpl_joints: SMPL joints [24, 3]
            frame_idx: Frame index for accessing GT data

        Returns:
            Modified SMPL joints with analytical foot positions [24, 3]
        """
        if not self.has_analytical_foot:
            return smpl_joints

        # Get GT foot joints for this frame
        target_frame = self.target_joints_np[frame_idx]

        # Left foot
        left_subtalar = target_frame[self.addb_left_subtalar_idx]
        left_mtp = target_frame[self.addb_left_mtp_idx]

        if not (np.isnan(left_subtalar).any() or np.isnan(left_mtp).any()):
            # Compute foot direction and length from GT
            foot_direction_left = left_mtp - left_subtalar
            foot_length_left = np.linalg.norm(foot_direction_left)

            # FIX 5: Data quality check
            if foot_length_left < 0.15 or foot_length_left > 0.35:
                print(f"[WARNING] Frame {frame_idx}: Abnormal left foot length {foot_length_left*1000:.1f}mm")
                # Skip analytical foot, keep SMPL default
            elif foot_length_left > 1e-6:
                foot_direction_left = foot_direction_left / foot_length_left

                # FIX 5: Outlier detection (sudden rotation check)
                if hasattr(self, 'prev_foot_dir_left'):
                    angle_change = np.arccos(np.clip(
                        np.dot(foot_direction_left, self.prev_foot_dir_left), -1, 1
                    ))
                    if angle_change > np.radians(30):  # 30° threshold
                        print(f"[WARNING] Frame {frame_idx}: Sudden left foot rotation {np.degrees(angle_change):.1f}° - using previous direction")
                        foot_direction_left = self.prev_foot_dir_left

                # FIX 5: Temporal smoothing (exponential moving average)
                if hasattr(self, 'prev_foot_dir_left'):
                    alpha = 0.7  # 70% current, 30% previous
                    foot_direction_left = (alpha * foot_direction_left +
                                          (1-alpha) * self.prev_foot_dir_left)
                    foot_direction_left = foot_direction_left / np.linalg.norm(foot_direction_left)

                # FIX 5: Validate foot angle (should be roughly horizontal)
                vertical_component = abs(foot_direction_left[1])  # Y component
                if vertical_component > 0.3:  # >17° up/down
                    print(f"[WARNING] Frame {frame_idx}: Abnormal left foot angle (vertical={vertical_component:.2f})")
                    # Project to horizontal plane
                    foot_direction_left[1] = 0
                    foot_direction_left = foot_direction_left / np.linalg.norm(foot_direction_left)

                # Store for next frame
                self.prev_foot_dir_left = foot_direction_left.copy()

                # Get SMPL ankle position
                smpl_left_ankle = smpl_joints[self.smpl_left_ankle_idx]

                # Compute analytical foot position: ankle + 0.75 × direction × length
                smpl_joints[self.smpl_left_foot_idx] = (
                    smpl_left_ankle + 0.75 * foot_direction_left * foot_length_left
                )

        # Right foot
        right_subtalar = target_frame[self.addb_right_subtalar_idx]
        right_mtp = target_frame[self.addb_right_mtp_idx]

        if not (np.isnan(right_subtalar).any() or np.isnan(right_mtp).any()):
            # Compute foot direction and length from GT
            foot_direction_right = right_mtp - right_subtalar
            foot_length_right = np.linalg.norm(foot_direction_right)

            # FIX 5: Data quality check
            if foot_length_right < 0.15 or foot_length_right > 0.35:
                print(f"[WARNING] Frame {frame_idx}: Abnormal right foot length {foot_length_right*1000:.1f}mm")
                # Skip analytical foot, keep SMPL default
            elif foot_length_right > 1e-6:
                foot_direction_right = foot_direction_right / foot_length_right

                # FIX 5: Outlier detection (sudden rotation check)
                if hasattr(self, 'prev_foot_dir_right'):
                    angle_change = np.arccos(np.clip(
                        np.dot(foot_direction_right, self.prev_foot_dir_right), -1, 1
                    ))
                    if angle_change > np.radians(30):  # 30° threshold
                        print(f"[WARNING] Frame {frame_idx}: Sudden right foot rotation {np.degrees(angle_change):.1f}° - using previous direction")
                        foot_direction_right = self.prev_foot_dir_right

                # FIX 5: Temporal smoothing (exponential moving average)
                if hasattr(self, 'prev_foot_dir_right'):
                    alpha = 0.7  # 70% current, 30% previous
                    foot_direction_right = (alpha * foot_direction_right +
                                           (1-alpha) * self.prev_foot_dir_right)
                    foot_direction_right = foot_direction_right / np.linalg.norm(foot_direction_right)

                # FIX 5: Validate foot angle (should be roughly horizontal)
                vertical_component = abs(foot_direction_right[1])  # Y component
                if vertical_component > 0.3:  # >17° up/down
                    print(f"[WARNING] Frame {frame_idx}: Abnormal right foot angle (vertical={vertical_component:.2f})")
                    # Project to horizontal plane
                    foot_direction_right[1] = 0
                    foot_direction_right = foot_direction_right / np.linalg.norm(foot_direction_right)

                # Store for next frame
                self.prev_foot_dir_right = foot_direction_right.copy()

                # Get SMPL ankle position
                smpl_right_ankle = smpl_joints[self.smpl_right_ankle_idx]

                # Compute analytical foot position: ankle + 0.75 × direction × length
                smpl_joints[self.smpl_right_foot_idx] = (
                    smpl_right_ankle + 0.75 * foot_direction_right * foot_length_right
                )

        return smpl_joints

    # --------------------------- Enhancement Methods -----------------------
    def _compute_bone_length_ratio_loss(self,
                                         pred_joints: torch.Tensor,
                                         target_joints: torch.Tensor) -> torch.Tensor:
        """
        Compute bone length RATIO loss instead of absolute length.

        Key insight: AddBiomechanics and SMPL have different skeletal proportions
        (e.g., AddB hip offset: 138mm, SMPL: 106mm). Matching absolute bone lengths
        leads to structural conflicts. Instead, we match RATIOS which are invariant
        to scale and skeleton structure.

        Args:
            pred_joints: SMPL predicted joints [24, 3] or [T, 24, 3]
            target_joints: AddB target joints [N, 3] or [T, N, 3] (with mapping)

        Returns:
            Bone length ratio loss (scalar)
        """
        is_sequence = pred_joints.ndim == 3

        if not is_sequence:
            # Single frame: [24, 3] and [N, 3]
            pred_joints = pred_joints.unsqueeze(0)  # [1, 24, 3]
            # Create mapped target
            target_mapped = torch.zeros(1, 24, 3, device=self.device)
            for addb_idx, smpl_idx in self.joint_mapping.items():
                if addb_idx < target_joints.shape[0]:
                    target_mapped[0, smpl_idx] = target_joints[addb_idx]
            target_joints = target_mapped
        else:
            # Sequence: [T, 24, 3] and [T, N, 3]
            T = pred_joints.shape[0]
            target_mapped = torch.zeros(T, 24, 3, device=self.device)
            for t in range(T):
                for addb_idx, smpl_idx in self.joint_mapping.items():
                    if addb_idx < target_joints.shape[1]:
                        target_mapped[t, smpl_idx] = target_joints[t, addb_idx]
            target_joints = target_mapped

        # Define bone pairs for ratio computation
        # Format: (parent_idx, child_idx, sibling_parent, sibling_child, name)
        ratio_pairs = [
            # Femur/Tibia ratio (left leg)
            (1, 4, 4, 7, 'left_femur_tibia'),  # left_hip→knee / left_knee→ankle
            # Femur/Tibia ratio (right leg)
            (2, 5, 5, 8, 'right_femur_tibia'),  # right_hip→knee / right_knee→ankle
            # Humerus/Radius ratio (left arm)
            (16, 18, 18, 20, 'left_humerus_radius'),  # left_shoulder→elbow / left_elbow→wrist
            # Humerus/Radius ratio (right arm)
            (17, 19, 19, 21, 'right_humerus_radius'),  # right_shoulder→elbow / right_elbow→wrist
        ]

        total_loss = 0.0
        count = 0

        for p1, c1, p2, c2, name in ratio_pairs:
            # Compute bone lengths for target
            target_bone1 = target_joints[:, c1] - target_joints[:, p1]  # [T, 3]
            target_bone2 = target_joints[:, c2] - target_joints[:, p2]  # [T, 3]
            target_len1 = torch.norm(target_bone1, dim=-1)  # [T]
            target_len2 = torch.norm(target_bone2, dim=-1)  # [T]

            # Compute bone lengths for prediction
            pred_bone1 = pred_joints[:, c1] - pred_joints[:, p1]  # [T, 3]
            pred_bone2 = pred_joints[:, c2] - pred_joints[:, p2]  # [T, 3]
            pred_len1 = torch.norm(pred_bone1, dim=-1)  # [T]
            pred_len2 = torch.norm(pred_bone2, dim=-1)  # [T]

            # Skip if any bone is degenerate or has NaN
            valid_mask = (
                (target_len1 > 1e-6) & (target_len2 > 1e-6) &
                (pred_len1 > 1e-6) & (pred_len2 > 1e-6) &
                ~torch.isnan(target_len1) & ~torch.isnan(target_len2) &
                ~torch.isnan(pred_len1) & ~torch.isnan(pred_len2)
            )

            if valid_mask.sum() == 0:
                continue

            # Compute ratios
            target_ratio = target_len1[valid_mask] / target_len2[valid_mask]  # [V]
            pred_ratio = pred_len1[valid_mask] / pred_len2[valid_mask]  # [V]

            # Ratio loss: (target_ratio - pred_ratio)^2
            ratio_diff = target_ratio - pred_ratio
            total_loss += (ratio_diff ** 2).mean()
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.device)

        return total_loss / count

    def _compute_absolute_bone_length_loss(self,
                                            pred_joints: torch.Tensor,
                                            target_joints: torch.Tensor) -> torch.Tensor:
        """
        FIX 4: Compute absolute bone length matching loss.

        Match absolute bone lengths between AddB and SMPL to ensure proper scaling.
        This addresses the issue where AddB and SMPL have different skeletal proportions.

        Args:
            pred_joints: SMPL predicted joints [24, 3] or [T, 24, 3]
            target_joints: AddB target joints [N, 3] or [T, N, 3]

        Returns:
            Absolute bone length loss (scalar)
        """
        is_sequence = pred_joints.ndim == 3

        if not is_sequence:
            # Single frame: [24, 3] and [N, 3]
            pred_joints = pred_joints.unsqueeze(0)  # [1, 24, 3]
            # Create mapped target
            target_mapped = torch.zeros(1, 24, 3, device=self.device)
            for addb_idx, smpl_idx in self.joint_mapping.items():
                if addb_idx < target_joints.shape[0]:
                    target_mapped[0, smpl_idx] = target_joints[addb_idx]
            target_joints = target_mapped
        else:
            # Sequence: [T, 24, 3] and [T, N, 3]
            T = pred_joints.shape[0]
            target_mapped = torch.zeros(T, 24, 3, device=self.device)
            for t in range(T):
                for addb_idx, smpl_idx in self.joint_mapping.items():
                    if addb_idx < target_joints.shape[1]:
                        target_mapped[t, smpl_idx] = target_joints[t, addb_idx]
            target_joints = target_mapped

        # Define bone pairs for absolute length matching
        # Format: (parent_idx, child_idx, name)
        bone_pairs = [
            # Left leg
            (1, 4, 'left_femur'),      # left_hip → left_knee
            (4, 7, 'left_tibia'),      # left_knee → left_ankle
            # Right leg
            (2, 5, 'right_femur'),     # right_hip → right_knee
            (5, 8, 'right_tibia'),     # right_knee → right_ankle
            # Left arm
            (16, 18, 'left_humerus'),  # left_shoulder → left_elbow
            (18, 20, 'left_radius'),   # left_elbow → left_wrist
            # Right arm
            (17, 19, 'right_humerus'), # right_shoulder → right_elbow
            (19, 21, 'right_radius'),  # right_elbow → right_wrist
        ]

        total_loss = 0.0
        count = 0

        for parent_idx, child_idx, name in bone_pairs:
            # Compute bone lengths for target
            target_bone = target_joints[:, child_idx] - target_joints[:, parent_idx]  # [T, 3]
            target_len = torch.norm(target_bone, dim=-1)  # [T]

            # Compute bone lengths for prediction
            pred_bone = pred_joints[:, child_idx] - pred_joints[:, parent_idx]  # [T, 3]
            pred_len = torch.norm(pred_bone, dim=-1)  # [T]

            # Skip if any bone is degenerate or has NaN
            valid_mask = (
                (target_len > 1e-6) & (pred_len > 1e-6) &
                ~torch.isnan(target_len) & ~torch.isnan(pred_len)
            )

            if valid_mask.sum() == 0:
                continue

            # Absolute length difference: |length_SMPL - length_AddB|²
            length_diff = pred_len[valid_mask] - target_len[valid_mask]
            total_loss += (length_diff ** 2).mean()
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.device)

        return total_loss / count

    def _compute_bone_length_soft_loss(self, pred_joints: torch.Tensor) -> torch.Tensor:
        """
        Enhancement 2: Bone Length Soft Constraint
        Compute soft constraint loss to maintain consistent bone lengths across frames.

        Args:
            pred_joints: SMPL joints [T, N, 3] where T=num_frames, N=24

        Returns:
            Bone length variance loss (scalar)
        """
        T = pred_joints.shape[0]
        if T < 2:
            return torch.tensor(0.0, device=self.device)

        # Define bone pairs for lower body
        bone_pairs = [
            (1, 4, 'left_femur'),    # left_hip to left_knee
            (2, 5, 'right_femur'),   # right_hip to right_knee
            (4, 7, 'left_tibia'),    # left_knee to left_ankle
            (5, 8, 'right_tibia'),   # right_knee to right_ankle
        ]

        total_loss = 0.0
        count = 0

        for parent_idx, child_idx, name in bone_pairs:
            # Compute bone lengths across all frames [T]
            bone_vecs = pred_joints[:, child_idx] - pred_joints[:, parent_idx]  # [T, 3]
            bone_lengths = torch.norm(bone_vecs, dim=1)  # [T]

            # Penalize variance in bone length
            mean_length = bone_lengths.mean()
            variance = ((bone_lengths - mean_length) ** 2).mean()
            total_loss += variance
            count += 1

        return total_loss / max(count, 1)

    def _get_stage_bone_weights(self, iteration: int, total_iterations: int) -> Dict[str, float]:
        """
        Enhancement 3: Multi-Stage Bone Direction Weights
        Return bone-specific weights based on optimization stage.

        Args:
            iteration: Current iteration number
            total_iterations: Total number of iterations

        Returns:
            Dictionary mapping bone names to weights
        """
        progress = iteration / max(total_iterations, 1)

        # Define 3 stages: Early (0-33%), Mid (33-66%), Late (66-100%)
        if progress < 0.33:
            # Early stage: Higher weight on stable bones (femur, tibia)
            return {
                'pelvis_to_left_hip': 1.0,
                'pelvis_to_right_hip': 1.0,
                'left_hip_to_knee': 1.5,      # Femur - more stable
                'right_hip_to_knee': 1.5,
                'left_knee_to_ankle': 1.5,    # Tibia - more stable
                'right_knee_to_ankle': 1.5,
                'left_ankle_to_foot': 0.8,    # Feet - less weight early
                'right_ankle_to_foot': 0.8,
            }
        elif progress < 0.66:
            # Mid stage: Balanced weights
            return {
                'pelvis_to_left_hip': 1.0,
                'pelvis_to_right_hip': 1.0,
                'left_hip_to_knee': 1.0,
                'right_hip_to_knee': 1.0,
                'left_knee_to_ankle': 1.0,
                'right_knee_to_ankle': 1.0,
                'left_ankle_to_foot': 1.0,
                'right_ankle_to_foot': 1.0,
            }
        else:
            # Late stage: Higher weight on feet for ground contact
            return {
                'pelvis_to_left_hip': 1.0,
                'pelvis_to_right_hip': 1.0,
                'left_hip_to_knee': 1.0,
                'right_hip_to_knee': 1.0,
                'left_knee_to_ankle': 1.0,
                'right_knee_to_ankle': 1.0,
                'left_ankle_to_foot': 1.5,    # Feet - higher weight for ground contact
                'right_ankle_to_foot': 1.5,
            }

    def _detect_foot_contact(self, pred_joints: torch.Tensor) -> torch.Tensor:
        """
        Enhancement 4: Contact-Aware Optimization
        Detect when feet are in contact with ground based on vertical position.

        Args:
            pred_joints: SMPL joints [T, N, 3] where T=num_frames, N=24

        Returns:
            Contact mask [T, 2] where [:, 0] is left foot, [:, 1] is right foot
        """
        T = pred_joints.shape[0]

        # Get foot joint positions (left_foot=10, right_foot=11)
        left_foot_y = pred_joints[:, 10, 1]   # [T] - y is vertical axis
        right_foot_y = pred_joints[:, 11, 1]  # [T]

        # Compute ground level as 5th percentile of foot heights
        all_foot_y = torch.cat([left_foot_y, right_foot_y])
        ground_level = torch.quantile(all_foot_y, 0.05)

        # Contact threshold: within 2cm of ground
        contact_threshold = 0.02  # 2cm in meters

        left_contact = (left_foot_y - ground_level).abs() < contact_threshold
        right_contact = (right_foot_y - ground_level).abs() < contact_threshold

        contact_mask = torch.stack([left_contact, right_contact], dim=1)  # [T, 2]
        return contact_mask

    def _compute_velocity_smoothness_loss(self, poses: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        """
        Enhancement 5: Velocity Smoothness Loss
        Penalize sudden changes in velocity (jerk minimization).

        Args:
            poses: Pose parameters [T, 24, 3]
            trans: Translation parameters [T, 3]

        Returns:
            Velocity smoothness loss (scalar)
        """
        T = poses.shape[0]
        if T < 3:
            return torch.tensor(0.0, device=self.device)

        # Compute velocities (first derivative)
        pose_vel = poses[1:] - poses[:-1]  # [T-1, 24, 3]
        trans_vel = trans[1:] - trans[:-1]  # [T-1, 3]

        # Compute accelerations (second derivative / velocity change)
        pose_acc = pose_vel[1:] - pose_vel[:-1]  # [T-2, 24, 3]
        trans_acc = trans_vel[1:] - trans_vel[:-1]  # [T-2, 3]

        # Loss is the magnitude of acceleration (jerk)
        pose_jerk = (pose_acc ** 2).mean()
        trans_jerk = (trans_acc ** 2).mean()

        return pose_jerk + trans_jerk

    def _compute_joint_angle_limits_loss(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Enhancement 6: Joint Angle Limits
        Penalize biomechanically impossible joint angles.

        Args:
            poses: Pose parameters [T, 24, 3] (axis-angle representation)

        Returns:
            Angle limit violation loss (scalar)
        """
        # Define angle limits for lower body joints (in radians)
        # Format: (joint_idx, axis, min_angle, max_angle)
        limits = {
            # Knee joints (4, 5) - knees can't bend backwards
            4: {'x_min': -2.5, 'x_max': 0.1},  # left_knee: flexion only
            5: {'x_min': -2.5, 'x_max': 0.1},  # right_knee: flexion only

            # Hip joints (1, 2) - limited range
            1: {'x_min': -1.5, 'x_max': 1.5, 'y_min': -0.5, 'y_max': 0.8},  # left_hip
            2: {'x_min': -1.5, 'x_max': 1.5, 'y_min': -0.8, 'y_max': 0.5},  # right_hip
        }

        total_loss = 0.0
        count = 0

        for joint_idx, joint_limits in limits.items():
            joint_poses = poses[:, joint_idx, :]  # [T, 3]

            # Check x-axis limits (flexion/extension)
            if 'x_min' in joint_limits:
                violation_min = torch.relu(joint_limits['x_min'] - joint_poses[:, 0])
                violation_max = torch.relu(joint_poses[:, 0] - joint_limits['x_max'])
                total_loss += (violation_min ** 2).mean() + (violation_max ** 2).mean()
                count += 2

            # Check y-axis limits (abduction/adduction)
            if 'y_min' in joint_limits:
                violation_min = torch.relu(joint_limits['y_min'] - joint_poses[:, 1])
                violation_max = torch.relu(joint_poses[:, 1] - joint_limits['y_max'])
                total_loss += (violation_min ** 2).mean() + (violation_max ** 2).mean()
                count += 2

        return total_loss / max(count, 1)

    def _compute_pelvis_upright_loss(self, poses: torch.Tensor) -> torch.Tensor:
        """
        FIX 6 (v6): Relative Pelvis Constraint
        Penalize deviation from AddBiomechanics GT pelvis orientation.
        Instead of forcing pelvis to be upright, maintain the natural
        forward lean that exists in running motion.

        Args:
            poses: Pose parameters [T, 24, 3] (axis-angle)

        Returns:
            Relative pelvis loss (scalar)
        """
        pelvis_pose = poses[:, 0, :]  # [T, 3] - pelvis rotation

        # If reference not set, compute it from initial poses (mean orientation)
        if self.addb_pelvis_reference is None:
            # Use mean pelvis orientation as reference (natural running posture)
            self.addb_pelvis_reference = pelvis_pose.detach().mean(dim=0)

        # Penalize deviation from reference orientation
        # This allows natural forward lean but prevents excessive changes
        deviation = pelvis_pose - self.addb_pelvis_reference
        deviation_loss = (deviation ** 2).mean()

        return deviation_loss

    def _compute_ankle_limit_loss(self, poses: torch.Tensor) -> torch.Tensor:
        """
        FIX 5 (v5): Ankle Rotation Limits
        Limit ankle rotation to anatomically plausible range.
        Prevents extreme dorsiflexion/plantarflexion.

        Args:
            poses: Pose parameters [T, 24, 3] (axis-angle)

        Returns:
            Ankle limit loss (scalar)
        """
        ankle_left = poses[:, 7, :]   # [T, 3] - left ankle
        ankle_right = poses[:, 8, :]  # [T, 3] - right ankle

        # Compute rotation magnitudes
        ankle_rot_left = torch.norm(ankle_left, dim=1)   # [T]
        ankle_rot_right = torch.norm(ankle_right, dim=1)  # [T]

        # Limit to ~60° (1.05 radians)
        # Typical ankle ROM: dorsiflexion 20°, plantarflexion 50°
        max_ankle_rotation = 1.05

        limit_loss = (
            torch.relu(ankle_rot_left - max_ankle_rotation).mean() +
            torch.relu(ankle_rot_right - max_ankle_rotation).mean()
        )

        return limit_loss

    def _compute_ground_penetration_loss(self, pred_joints: torch.Tensor, ground_level: Optional[float] = None) -> torch.Tensor:
        """
        Enhancement 7: Ground Penetration Penalty
        Strong penalty when feet go below ground level.

        Args:
            pred_joints: SMPL joints [T, 24, 3] or [24, 3]
            ground_level: Ground level (if None, auto-detect from 5th percentile)

        Returns:
            Ground penetration loss (scalar)
        """
        is_sequence = pred_joints.ndim == 3

        if is_sequence:
            # [T, 24, 3]
            left_foot_y = pred_joints[:, 10, 1]   # [T]
            right_foot_y = pred_joints[:, 11, 1]  # [T]

            if ground_level is None:
                all_foot_y = torch.cat([left_foot_y, right_foot_y])
                ground_level = torch.quantile(all_foot_y, 0.05)

            # Penalize when feet go below ground
            left_penetration = torch.relu(ground_level - left_foot_y)
            right_penetration = torch.relu(ground_level - right_foot_y)

            return (left_penetration ** 2).mean() + (right_penetration ** 2).mean()
        else:
            # Single frame [24, 3]
            left_foot_y = pred_joints[10, 1]
            right_foot_y = pred_joints[11, 1]

            if ground_level is None:
                ground_level = min(left_foot_y, right_foot_y)

            left_penetration = torch.relu(ground_level - left_foot_y)
            right_penetration = torch.relu(ground_level - right_foot_y)

            return left_penetration ** 2 + right_penetration ** 2

    def _compute_foot_orientation_loss(self,
                                       pred_joints: torch.Tensor,
                                       target_joints: torch.Tensor) -> torch.Tensor:
        """
        FIX 9: Compute foot orientation loss to align SMPL foot direction with AddB toe direction.

        Problem: SMPL foot joints (indices 10, 11) point sideways (77-85°) instead of forward
        like AddBiomechanics GT toe joints (MTP).

        Solution: Force SMPL foot direction (ankle→foot) to match AddB toe direction (ankle→MTP).

        Args:
            pred_joints: SMPL predicted joints [24, 3] or [T, 24, 3]
            target_joints: AddB target joints [N, 3] or [T, N, 3]

        Returns:
            Foot orientation loss (scalar, lower is better)
        """
        # Handle both single frame [24, 3] and batch [T, 24, 3]
        if pred_joints.dim() == 2:
            pred_joints = pred_joints.unsqueeze(0)  # [1, 24, 3]
            target_joints = target_joints.unsqueeze(0)  # [1, N, 3]
            squeeze_output = True
        else:
            squeeze_output = False

        total_loss = 0.0
        valid_count = 0

        # Find AddB indices for ankles and MTP joints
        addb_left_ankle_idx = None
        addb_right_ankle_idx = None
        addb_left_mtp_idx = None
        addb_right_mtp_idx = None

        for addb_idx, smpl_idx in self.joint_mapping.items():
            if smpl_idx == 7:  # left_ankle
                addb_left_ankle_idx = addb_idx
            elif smpl_idx == 8:  # right_ankle
                addb_right_ankle_idx = addb_idx
            elif smpl_idx == 10:  # left_foot (mapped from mtp_l)
                addb_left_mtp_idx = addb_idx
            elif smpl_idx == 11:  # right_foot (mapped from mtp_r)
                addb_right_mtp_idx = addb_idx

        # Check if we have all required joints
        if (addb_left_ankle_idx is None or addb_right_ankle_idx is None or
            addb_left_mtp_idx is None or addb_right_mtp_idx is None):
            # Cannot compute foot orientation loss without MTP joints
            return torch.tensor(0.0, device=pred_joints.device)

        # Check bounds
        max_addb_idx = target_joints.shape[1]
        if (addb_left_ankle_idx >= max_addb_idx or addb_right_ankle_idx >= max_addb_idx or
            addb_left_mtp_idx >= max_addb_idx or addb_right_mtp_idx >= max_addb_idx):
            return torch.tensor(0.0, device=pred_joints.device)

        for t in range(pred_joints.shape[0]):
            # SMPL foot vectors (ankle → foot)
            smpl_left_ankle = pred_joints[t, 7]  # left_ankle
            smpl_right_ankle = pred_joints[t, 8]  # right_ankle
            smpl_left_foot = pred_joints[t, 10]  # left_foot
            smpl_right_foot = pred_joints[t, 11]  # right_foot

            smpl_left_vec = smpl_left_foot - smpl_left_ankle
            smpl_right_vec = smpl_right_foot - smpl_right_ankle

            # AddB toe vectors (ankle → MTP)
            addb_left_ankle = target_joints[t, addb_left_ankle_idx]
            addb_right_ankle = target_joints[t, addb_right_ankle_idx]
            addb_left_mtp = target_joints[t, addb_left_mtp_idx]
            addb_right_mtp = target_joints[t, addb_right_mtp_idx]

            # Skip if any target joints are NaN
            if (torch.isnan(addb_left_ankle).any() or torch.isnan(addb_left_mtp).any() or
                torch.isnan(addb_right_ankle).any() or torch.isnan(addb_right_mtp).any()):
                continue

            addb_left_vec = addb_left_mtp - addb_left_ankle
            addb_right_vec = addb_right_mtp - addb_right_ankle

            # Normalize vectors
            smpl_left_vec_norm = F.normalize(smpl_left_vec, dim=-1, eps=1e-8)
            smpl_right_vec_norm = F.normalize(smpl_right_vec, dim=-1, eps=1e-8)
            addb_left_vec_norm = F.normalize(addb_left_vec, dim=-1, eps=1e-8)
            addb_right_vec_norm = F.normalize(addb_right_vec, dim=-1, eps=1e-8)

            # Cosine similarity loss: 1 - dot_product
            left_cos_sim = torch.sum(smpl_left_vec_norm * addb_left_vec_norm)
            right_cos_sim = torch.sum(smpl_right_vec_norm * addb_right_vec_norm)

            left_loss = 1.0 - left_cos_sim
            right_loss = 1.0 - right_cos_sim

            total_loss += (left_loss + right_loss)
            valid_count += 2

        if valid_count > 0:
            loss = total_loss / valid_count
        else:
            loss = torch.tensor(0.0, device=pred_joints.device)

        if squeeze_output:
            loss = loss.squeeze()

        return loss

    def _compute_bone_direction_loss(self,
                                      pred_joints: torch.Tensor,
                                      target_joints: torch.Tensor,
                                      bone_weights: Optional[Dict[str, float]] = None,
                                      contact_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute bone direction matching loss using cosine similarity.
        This bypasses absolute position mismatch due to skeletal structure differences.

        The key insight: AddBiomechanics and SMPL have different hip joint placements
        (AddB: 138mm from pelvis, SMPL: 106mm), causing cascading errors down the leg.
        By matching bone DIRECTIONS instead of absolute positions, we bypass this
        structural mismatch while still ensuring correct pose.

        Enhancement 3 & 4: Supports multi-stage bone weights and contact-aware weighting.

        Args:
            pred_joints: SMPL predicted joints [N, 3] (subset after mask filtering)
            target_joints: AddB target joints [N, 3] (subset after mask filtering)
            bone_weights: Optional dict mapping bone names to weights (Enhancement 3)
            contact_mask: Optional [2] tensor with left/right foot contact (Enhancement 4)

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

            # Enhancement 3: Apply bone-specific weight
            weight = 1.0
            if bone_weights is not None and name in bone_weights:
                weight = bone_weights[name]

            # Enhancement 4: Apply contact-aware weight multiplier
            if contact_mask is not None:
                # For foot bones, apply contact multiplier when foot is in contact
                if 'left_ankle_to_foot' in name and contact_mask[0]:
                    weight *= self.config.contact_weight_multiplier
                elif 'right_ankle_to_foot' in name and contact_mask[1]:
                    weight *= self.config.contact_weight_multiplier

            total_loss += loss * weight
            count += weight  # Weighted count for proper averaging

        if count == 0:
            return torch.tensor(0.0, device=pred_joints.device)

        return total_loss / count

    def _initialize_foot_rotations(self, poses: torch.Tensor) -> torch.Tensor:
        """
        CRITICAL FIX: Initialize foot rotations from AddBiomechanics ankle→foot directions.

        Without this, foot rotations start at [0,0,0] and never move due to weak gradients.
        With this, feet start with realistic orientations that optimizer can refine.

        Args:
            poses: [T, 24, 3] pose parameters (will modify in-place for foot joints)

        Returns:
            Modified poses with initialized foot rotations
        """
        print('    [FOOT INIT] Initializing foot rotations from AddB data...')

        T = poses.shape[0]
        LEFT_ANKLE_SMPL = 7
        RIGHT_ANKLE_SMPL = 8
        LEFT_FOOT_SMPL = 10
        RIGHT_FOOT_SMPL = 11

        # Find AddB indices for ankle and foot
        def find_addb_idx(smpl_idx):
            if smpl_idx in self.smpl_indices.cpu().tolist():
                pos = self.smpl_indices.cpu().tolist().index(smpl_idx)
                return self.addb_indices[pos].item()
            return None

        addb_left_ankle = find_addb_idx(LEFT_ANKLE_SMPL)
        addb_right_ankle = find_addb_idx(RIGHT_ANKLE_SMPL)
        addb_left_foot = find_addb_idx(LEFT_FOOT_SMPL)
        addb_right_foot = find_addb_idx(RIGHT_FOOT_SMPL)

        init_count = 0

        for t in range(T):
            target_frame = self.target_joints[t]

            # Initialize left foot
            if addb_left_ankle is not None and addb_left_foot is not None:
                ankle_pos = target_frame[addb_left_ankle]
                foot_pos = target_frame[addb_left_foot]

                if not torch.isnan(ankle_pos).any() and not torch.isnan(foot_pos).any():
                    # Set plantarflexion angle (~17 degrees)
                    angle = 0.3  # radians
                    poses[t, LEFT_FOOT_SMPL, 0] = angle
                    init_count += 1

            # Initialize right foot
            if addb_right_ankle is not None and addb_right_foot is not None:
                ankle_pos = target_frame[addb_right_ankle]
                foot_pos = target_frame[addb_right_foot]

                if not torch.isnan(ankle_pos).any() and not torch.isnan(foot_pos).any():
                    # Set plantarflexion angle (~17 degrees)
                    angle = 0.3  # radians
                    poses[t, RIGHT_FOOT_SMPL, 0] = angle
                    init_count += 1

        print(f'    [FOOT INIT] Initialized {init_count}/{T*2} foot rotations (non-zero starting points)')

        return poses

    def optimise_initial_poses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 1: Initial coarse pose estimation for shape optimization
        Uses default shape (betas=0) and quickly estimates poses
        Returns: (poses, trans) - initial estimates
        """
        print('  [Stage 1] Initial pose estimation (coarse)...')

        # Default shape
        betas = torch.zeros(SMPL_NUM_BETAS, device=self.device)

        T = self.target_joints.shape[0]
        poses = torch.zeros(T, SMPL_NUM_JOINTS, 3, device=self.device)
        trans = torch.zeros(T, 3, device=self.device)

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

        # CRITICAL FIX: Initialize foot rotations from AddB data
        poses = self._initialize_foot_rotations(poses)

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

        # ENHANCEMENT: Smart keyframe sampling instead of uniform sampling
        frame_indices = self._detect_keyframes(num_samples=sample_count)

        print(f'    Total frames: {num_frames}, Keyframes sampled: {len(frame_indices)}, Batch size: {batch_size}')
        print(f'    Keyframe strategy: High-movement frames prioritized for better shape estimation')

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

        # ENHANCEMENT: Early stopping for shape optimization
        prev_epoch_loss = float('inf')
        convergence_threshold = 1e-6
        min_iters_before_stop = 20

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

                        # ENHANCEMENT: Mixed precision for GPU
                        if self.device.type == 'cuda':
                            with torch.cuda.amp.autocast():
                                joints_pred = self.smpl.joints(betas.unsqueeze(0), pose_t.unsqueeze(0), trans_t.unsqueeze(0)).squeeze(0)
                        else:
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

                # ENHANCEMENT: Bone length ratio loss (scale-invariant)
                if use_pose_aware:
                    # For pose-aware, use the batch frames
                    for idx in batch_frame_indices:
                        target = self.target_joints[idx]
                        if torch.isnan(target).all():
                            continue
                        pose_t = initial_poses[idx]
                        trans_t = initial_trans[idx]
                        joints_pred_single = self.smpl.joints(betas.unsqueeze(0), pose_t.unsqueeze(0), trans_t.unsqueeze(0)).squeeze(0)
                        ratio_loss = self._compute_bone_length_ratio_loss(joints_pred_single, target)
                        loss = loss + self.config.bone_length_ratio_weight * ratio_loss
                        break  # Only compute for first frame in batch (efficiency)
                else:
                    # For zero-pose, use the computed joints
                    if len(batch_frame_indices) > 0:
                        target = self.target_joints[batch_frame_indices[0]]
                        ratio_loss = self._compute_bone_length_ratio_loss(joints_pred_for_bones, target)
                        loss = loss + self.config.bone_length_ratio_weight * ratio_loss

                # Backward pass for this mini-batch
                loss.backward()
                optimizer.step()

                # ENHANCEMENT: Beta constraint (논문 Eq. 5: |βi| < 5)
                with torch.no_grad():
                    betas.clamp_(-5.0, 5.0)

                # Track epoch loss
                epoch_loss += loss.item()
                epoch_batches += 1

            # Print epoch loss every 10 iterations
            if (it + 1) % 10 == 0 and epoch_batches > 0:
                avg_epoch_loss = epoch_loss / epoch_batches
                print(f'    Iteration {it + 1}/{self.config.shape_iters}: Avg batch loss = {avg_epoch_loss:.6f}')

            # ENHANCEMENT: Early stopping check
            if it >= min_iters_before_stop and epoch_batches > 0:
                avg_epoch_loss = epoch_loss / epoch_batches
                loss_change = abs(avg_epoch_loss - prev_epoch_loss)
                if loss_change < convergence_threshold:
                    print(f'    Early stopping at iteration {it + 1}: converged (loss change = {loss_change:.8f})')
                    break
                prev_epoch_loss = avg_epoch_loss

        return betas.detach()

    # --------------------------- Pose optimisation ------------------------
    def optimise_hierarchical(self,
                              betas: torch.Tensor,
                              init_poses: torch.Tensor,
                              init_trans: torch.Tensor,
                              pose_iters: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhancement 8: Hierarchical Optimization
        Optimize joints in hierarchical order: Pelvis → Hip → Knee → Ankle → Foot

        Args:
            betas: Shape parameters
            init_poses: Initial poses
            init_trans: Initial translations
            pose_iters: Iterations per hierarchy level

        Returns:
            (poses, trans) as numpy arrays
        """
        print('  [Hierarchical Optimization] Optimizing in sequence: Pelvis→Hip→Knee→Ankle→Foot...')

        T = init_poses.shape[0]
        poses = init_poses.clone()
        trans = init_trans.clone()

        # Define hierarchy: (joint_indices, name)
        hierarchy_levels = [
            ([0], 'Pelvis + Translation'),  # Pelvis + global translation
            ([1, 2], 'Hips'),               # Left & right hips
            ([4, 5], 'Knees'),              # Left & right knees
            ([7, 8], 'Ankles'),             # Left & right ankles
            ([10, 11], 'Feet'),             # Left & right feet
        ]

        for level_idx, (joint_indices, level_name) in enumerate(hierarchy_levels):
            print(f'    Level {level_idx+1}/5: Optimizing {level_name}...')

            # Create parameters for this level
            level_poses = torch.nn.Parameter(poses.clone())
            level_trans = torch.nn.Parameter(trans.clone())

            optimizer = torch.optim.Adam([level_poses, level_trans], lr=self.config.pose_lr)

            for _ in range(pose_iters):
                optimizer.zero_grad()

                # Compute loss averaged over all frames
                total_loss = 0.0
                frame_count = 0
                all_joints_pred = []  # For foot orientation loss

                for t in range(T):
                    target_frame = self.target_joints[t]
                    if torch.isnan(target_frame).all():
                        continue

                    joints_pred = self.smpl.joints(betas.unsqueeze(0), level_poses[t].unsqueeze(0), level_trans[t].unsqueeze(0)).squeeze(0)
                    all_joints_pred.append(joints_pred)

                    # Position loss
                    target_subset = target_frame[self.addb_indices]
                    pred_subset = joints_pred[self.smpl_indices]
                    mask = ~torch.isnan(target_subset).any(dim=1)
                    if mask.sum() == 0:
                        continue

                    diff = pred_subset[mask] - target_subset[mask]
                    loss = (diff ** 2).mean()

                    total_loss += loss
                    frame_count += 1

                if frame_count > 0:
                    total_loss = total_loss / frame_count

                    # Add regularization only for the joints being optimized
                    for joint_idx in joint_indices:
                        total_loss += self.config.pose_reg_weight * (level_poses[:, joint_idx] ** 2).mean()

                    # FIX 4: Foot orientation loss removed - using analytical foot position

                    total_loss.backward()

                    # Only update gradients for the current level joints
                    with torch.no_grad():
                        # Zero out gradients for other joints
                        mask = torch.ones(SMPL_NUM_JOINTS, dtype=torch.bool, device=self.device)
                        mask[joint_indices] = False
                        level_poses.grad[:, mask, :] = 0

                    optimizer.step()

            # Update poses and trans
            poses = level_poses.detach()
            trans = level_trans.detach()

        print('  [Hierarchical Optimization] Complete!')
        return poses.cpu().numpy(), trans.cpu().numpy()

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

            # Enhancement 1: Per-Joint Learning Rates
            # We'll use a custom gradient scaling approach instead of parameter groups
            # to avoid non-leaf tensor issues
            optimiser = torch.optim.Adam([pose_param, trans_param], lr=lr)

            # Define joint-specific learning rate multipliers
            # Lower body
            foot_joints = [10, 11]  # left_foot, right_foot
            knee_joints = [4, 5]     # left_knee, right_knee
            hip_joints = [1, 2]      # left_hip, right_hip

            # ENHANCEMENT: Upper body (arms) - lower LR for stability
            shoulder_joints = [16, 17]  # left_shoulder, right_shoulder
            elbow_joints = [18, 19]     # left_elbow, right_elbow
            wrist_joints = [20, 21]     # left_wrist, right_wrist

            lr_multipliers = torch.ones(SMPL_NUM_JOINTS, device=self.device)
            # Lower body
            lr_multipliers[foot_joints] = self.config.foot_lr / lr  # 0.5x
            lr_multipliers[knee_joints] = self.config.knee_lr / lr  # 1.0x
            lr_multipliers[hip_joints] = self.config.hip_lr / lr    # 1.5x
            # Upper body (arms) - lower LR for stability
            lr_multipliers[shoulder_joints] = 0.8  # Medium-low LR
            lr_multipliers[elbow_joints] = 0.7     # Lower LR (more prone to noise)
            lr_multipliers[wrist_joints] = 0.5     # Lowest LR (most unstable)

            # ENHANCEMENT: Early stopping with convergence check
            prev_loss = float('inf')
            convergence_threshold = 1e-5
            min_iters_before_stop = 10

            for iter_idx in range(num_iters):
                optimiser.zero_grad()

                # ENHANCEMENT: GPU memory optimization with mixed precision
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        joints_pred = self.smpl.joints(betas.unsqueeze(0), pose_param.unsqueeze(0), trans_param.unsqueeze(0)).squeeze(0)
                else:
                    joints_pred = self.smpl.joints(betas.unsqueeze(0), pose_param.unsqueeze(0), trans_param.unsqueeze(0)).squeeze(0)

                # Use absolute positions (no root centering) to properly learn translation
                target_subset = target_frame[self.addb_indices]
                pred_subset = joints_pred[self.smpl_indices]
                mask = ~torch.isnan(target_subset).any(dim=1)
                if mask.sum() == 0:
                    break

                diff = pred_subset[mask] - target_subset[mask]

                # Per-joint weighting: Give higher weight to foot joints (SMPL indices 10, 11)
                # This ensures feet are well-optimized
                joint_weights = torch.ones(len(self.smpl_indices), device=self.device)
                for i, smpl_idx in enumerate(self.smpl_indices):
                    if smpl_idx in [10, 11]:  # left_foot, right_foot
                        joint_weights[i] = 3.0  # 3x weight for feet
                    elif smpl_idx in [7, 8]:  # left_ankle, right_ankle
                        joint_weights[i] = 2.0  # 2x weight for ankles

                # Apply per-joint weights to the masked differences
                weighted_diff = diff * torch.sqrt(joint_weights[mask].unsqueeze(1))
                loss = (weighted_diff ** 2).mean()

                # Enhancement 3: Get stage-specific bone weights
                bone_weights = self._get_stage_bone_weights(iter_idx, num_iters)

                # Enhancement 4: Detect foot contact (need full sequence for detection)
                # For single-frame optimization, we'll skip contact detection
                # Contact detection will be added in refinement phase
                contact_mask = None

                # Bone direction loss with enhancements (bypass structural mismatch)
                bone_dir_loss = self._compute_bone_direction_loss(
                    joints_pred, target_frame, bone_weights, contact_mask
                )
                loss = loss + self.config.bone_direction_weight * bone_dir_loss

                # FIX 10: Foot orientation loss (Stage 3 - reduced weight)
                foot_orient_loss = self._compute_foot_orientation_loss(joints_pred, target_frame)
                foot_orient_weight = 0.2
                loss = loss + foot_orient_weight * foot_orient_loss

                # Pose regularization
                loss = loss + self.config.pose_reg_weight * (pose_param ** 2).mean()

                # Translation regularization
                if self.root_addb_idx is not None:
                    root_target = target_frame[self.root_addb_idx]
                    if not torch.isnan(root_target).any():
                        loss = loss + self.config.trans_reg_weight * ((trans_param - root_target) ** 2).mean()

                # Temporal smoothness regularization (ENABLED for smooth motion)
                temporal_smooth_weight = 0.5  # Weight for temporal smoothness (increased to reduce jitter)
                if t > 0:
                    pose_diff = pose_param - poses[t-1].detach()
                    trans_diff = trans_param - trans[t-1].detach()
                    temporal_loss = (pose_diff ** 2).mean() + (trans_diff ** 2).mean()
                    loss = loss + temporal_smooth_weight * temporal_loss

                loss.backward()

                # Enhancement 1: Apply per-joint learning rate multipliers
                if pose_param.grad is not None:
                    # Scale gradients by joint-specific multipliers
                    pose_param.grad *= lr_multipliers.unsqueeze(1)  # [24, 1] to broadcast over [24, 3]

                optimiser.step()

                # ENHANCEMENT: Early stopping check
                if iter_idx >= min_iters_before_stop:
                    loss_change = abs(loss.item() - prev_loss)
                    if loss_change < convergence_threshold:
                        # Converged, stop early
                        break
                prev_loss = loss.item()

            # Update pose and translation
            poses[t] = pose_param.detach()
            trans[t] = trans_param.detach()

        return poses.cpu().numpy(), trans.cpu().numpy()

    def optimise_with_sequence_enhancements(self,
                                             betas: torch.Tensor,
                                             init_poses: torch.Tensor,
                                             init_trans: torch.Tensor,
                                             num_iters: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhancement 2 & 4: Sequence-level refinement with bone length soft constraint
        and contact-aware optimization.

        This method optimizes all frames jointly to enforce bone length consistency
        and apply contact-aware weights.

        Args:
            betas: Shape parameters [10]
            init_poses: Initial poses [T, 24, 3]
            init_trans: Initial translations [T, 3]
            num_iters: Number of refinement iterations

        Returns:
            (refined_poses, refined_trans) as numpy arrays
        """
        print(f'  [Sequence Enhancement] Refining with bone length + contact-aware (iters={num_iters})...')

        betas = betas.detach()
        T = self.target_joints.shape[0]

        # Initialize from provided values
        poses = torch.nn.Parameter(init_poses.clone())
        trans = torch.nn.Parameter(init_trans.clone())

        # Use default pose learning rate
        optimizer = torch.optim.Adam([poses, trans], lr=self.config.pose_lr * 0.5)  # Lower LR for refinement

        # Detect foot contacts once (based on initial poses)
        with torch.no_grad():
            pred_joints_all = []
            for t in range(T):
                jp = self.smpl.joints(betas.unsqueeze(0), poses[t].unsqueeze(0), trans[t].unsqueeze(0)).squeeze(0)
                pred_joints_all.append(jp)
            pred_joints_tensor = torch.stack(pred_joints_all, dim=0)  # [T, 24, 3]
            contact_masks = self._detect_foot_contact(pred_joints_tensor)  # [T, 2]

        # ENHANCEMENT: Early stopping for sequence refinement
        prev_loss = float('inf')
        convergence_threshold = 1e-5
        min_iters_before_stop = 10

        for iter_idx in range(num_iters):
            optimizer.zero_grad()

            # Compute joints for all frames
            # ENHANCEMENT: Mixed precision for GPU memory efficiency
            pred_joints_all = []
            for t in range(T):
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        jp = self.smpl.joints(betas.unsqueeze(0), poses[t].unsqueeze(0), trans[t].unsqueeze(0)).squeeze(0)
                else:
                    jp = self.smpl.joints(betas.unsqueeze(0), poses[t].unsqueeze(0), trans[t].unsqueeze(0)).squeeze(0)
                pred_joints_all.append(jp)
            pred_joints_tensor = torch.stack(pred_joints_all, dim=0)  # [T, 24, 3]

            # Enhancement 2: Bone length soft constraint
            bone_length_loss = self._compute_bone_length_soft_loss(pred_joints_tensor)

            # FIX 4: Absolute bone length matching
            absolute_bone_loss = self._compute_absolute_bone_length_loss(pred_joints_tensor, self.target_joints)

            # Per-frame losses with enhancements
            total_loss = 0.0
            frame_count = 0

            # Get stage-specific bone weights
            bone_weights = self._get_stage_bone_weights(iter_idx, num_iters)

            for t in range(T):
                target_frame = self.target_joints[t]
                if torch.isnan(target_frame).all():
                    continue

                joints_pred = pred_joints_tensor[t]

                # Position loss
                target_subset = target_frame[self.addb_indices]
                pred_subset = joints_pred[self.smpl_indices]
                mask = ~torch.isnan(target_subset).any(dim=1)
                if mask.sum() == 0:
                    continue

                diff = pred_subset[mask] - target_subset[mask]

                # Per-joint weighting: Give higher weight to foot joints (SMPL indices 10, 11)
                joint_weights = torch.ones(len(self.smpl_indices), device=self.device)
                for i, smpl_idx in enumerate(self.smpl_indices):
                    if smpl_idx in [10, 11]:  # left_foot, right_foot
                        joint_weights[i] = 3.0  # 3x weight for feet
                    elif smpl_idx in [7, 8]:  # left_ankle, right_ankle
                        joint_weights[i] = 2.0  # 2x weight for ankles

                weighted_diff = diff * torch.sqrt(joint_weights[mask].unsqueeze(1))
                pos_loss = (weighted_diff ** 2).mean()

                # Enhancement 3 & 4: Bone direction loss with stage weights + contact
                contact_mask = contact_masks[t]  # [2]
                bone_dir_loss = self._compute_bone_direction_loss(
                    joints_pred, target_frame, bone_weights, contact_mask
                )

                frame_loss = pos_loss + self.config.bone_direction_weight * bone_dir_loss

                # Pose regularization
                frame_loss = frame_loss + self.config.pose_reg_weight * (poses[t] ** 2).mean()

                # Translation regularization
                if self.root_addb_idx is not None:
                    root_target = target_frame[self.root_addb_idx]
                    if not torch.isnan(root_target).any():
                        frame_loss = frame_loss + self.config.trans_reg_weight * ((trans[t] - root_target) ** 2).mean()

                # Temporal smoothness
                if t > 0:
                    pose_diff = poses[t] - poses[t-1]
                    trans_diff = trans[t] - trans[t-1]
                    temporal_loss = (pose_diff ** 2).mean() + (trans_diff ** 2).mean()
                    frame_loss = frame_loss + self.config.temporal_smooth_weight * temporal_loss

                total_loss += frame_loss
                frame_count += 1

            if frame_count > 0:
                total_loss = total_loss / frame_count

            # Add bone length soft constraint
            total_loss = total_loss + self.config.bone_length_soft_weight * bone_length_loss

            # FIX 4: Add absolute bone length matching
            total_loss = total_loss + self.config.absolute_bone_length_weight * absolute_bone_loss

            # Enhancement 5: Velocity Smoothness Loss
            velocity_loss = self._compute_velocity_smoothness_loss(poses, trans)
            total_loss = total_loss + self.config.velocity_smooth_weight * velocity_loss

            # Enhancement 6: Joint Angle Limits
            angle_loss = self._compute_joint_angle_limits_loss(poses)
            total_loss = total_loss + self.config.angle_limit_weight * angle_loss

            # Enhancement 7: Ground Penetration Penalty
            ground_loss = self._compute_ground_penetration_loss(pred_joints_tensor)
            total_loss = total_loss + self.config.ground_penetration_weight * ground_loss

            # FIX 10: Foot orientation loss (Stage 4 - reduced weight)
            foot_orient_loss = self._compute_foot_orientation_loss(pred_joints_tensor, self.target_joints)
            foot_orient_weight = 0.2
            total_loss = total_loss + foot_orient_weight * foot_orient_loss

            # FIX 5 (v5 Option 1): Pelvis Upright Constraint
            pelvis_upright_loss = self._compute_pelvis_upright_loss(poses)
            total_loss = total_loss + self.config.pelvis_upright_weight * pelvis_upright_loss

            # FIX 5 (v5): Ankle Rotation Limits
            ankle_limit_loss = self._compute_ankle_limit_loss(poses)
            total_loss = total_loss + self.config.ankle_limit_weight * ankle_limit_loss

            total_loss.backward()
            optimizer.step()

            if (iter_idx + 1) % 10 == 0:
                print(f'    Iter {iter_idx+1}/{num_iters}: Loss={total_loss.item():.6f}, '
                      f'BoneLen={bone_length_loss.item():.6f}, AbsBone={absolute_bone_loss.item():.6f}, '
                      f'Vel={velocity_loss.item():.6f}, Angle={angle_loss.item():.6f}, Ground={ground_loss.item():.6f}, '
                      f'FootOrient={foot_orient_loss.item():.6f}, '
                      f'Pelvis={pelvis_upright_loss.item():.6f}, Ankle={ankle_limit_loss.item():.6f}')

            # ENHANCEMENT: Early stopping check
            if iter_idx >= min_iters_before_stop:
                loss_change = abs(total_loss.item() - prev_loss)
                if loss_change < convergence_threshold:
                    print(f'    Early stopping at iteration {iter_idx + 1}: converged (loss change = {loss_change:.8f})')
                    break
            prev_loss = total_loss.item()

        return poses.detach().cpu().numpy(), trans.detach().cpu().numpy()

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
            # FIX 4: Apply analytical foot position
            jp = self._apply_analytical_foot_position(jp, t)
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
        Four-stage enhanced optimization workflow:
        Stage 1: Initial coarse pose estimation (with default shape)
        Stage 2: Pose-aware shape optimization (using Stage 1 poses)
        Stage 3: Pose refinement with per-joint LR + multi-stage bone weights
        Stage 4: Sequence-level enhancements (bone length + contact-aware)
        """
        print("\n" + "="*80)
        print("STARTING FOUR-STAGE ENHANCED OPTIMIZATION")
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

        # Enhancement 8: Use hierarchical optimization if enabled
        if self.config.use_hierarchical_opt:
            poses_np, trans_np = self.optimise_hierarchical(
                betas,
                init_poses=initial_poses,
                init_trans=initial_trans,
                pose_iters=10
            )
        else:
            poses_np, trans_np = self.optimise_pose_sequence(
                betas,
                init_poses=initial_poses,
                init_trans=initial_trans
            )

        metrics_stage3, _ = self.evaluate(betas, poses_np, trans_np)
        print(f"  Stage 3 MPJPE (optimized shape + refined poses): {metrics_stage3['MPJPE']:.2f} mm")

        # Stage 4: Sequence-level enhancements (bone length + contact-aware)
        print("\n--- STAGE 4: Sequence-Level Enhancements ---")
        poses_tensor = torch.from_numpy(poses_np).float().to(self.device)
        trans_tensor = torch.from_numpy(trans_np).float().to(self.device)
        poses_np, trans_np = self.optimise_with_sequence_enhancements(
            betas,
            init_poses=poses_tensor,
            init_trans=trans_tensor,
            num_iters=30  # Reduced from 50 for 24h deadline (-40%)
        )
        metrics, pred_joints = self.evaluate(betas, poses_np, trans_np)
        print(f"  Stage 4 MPJPE (with bone length + contact enhancements): {metrics['MPJPE']:.2f} mm")
        print(f"  Improvement from Stage 3: {metrics_stage3['MPJPE'] - metrics['MPJPE']:.2f} mm")

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
# Configuration loading
# ---------------------------------------------------------------------------

def load_config_from_yaml(yaml_path: str) -> Dict:
    """
    Load optimization configuration from YAML file.

    Args:
        yaml_path: Path to YAML config file

    Returns:
        Dictionary with configuration parameters
    """
    if yaml is None:
        raise ImportError("pyyaml is required to load YAML configs. Install with: pip install pyyaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def config_to_optimization_config(config_dict: Dict) -> OptimisationConfig:
    """
    Convert YAML config dictionary to OptimisationConfig dataclass.

    Args:
        config_dict: Dictionary from YAML

    Returns:
        OptimisationConfig instance
    """
    return OptimisationConfig(
        shape_lr=config_dict['shape']['lr'],
        shape_iters=config_dict['shape']['iters'],
        shape_sample_frames=config_dict['shape']['sample_frames'],
        pose_lr=config_dict['pose']['lr'],
        pose_iters=config_dict['pose']['iters'],
        tolerance_mm=config_dict['convergence']['tolerance_mm'],
        max_passes=config_dict['convergence']['max_passes'],
        pose_reg_weight=config_dict['pose']['reg_weight'],
        trans_reg_weight=config_dict['translation']['reg_weight'],
        bone_length_weight=config_dict['enhancements']['bone_length_weight'],
        temporal_smooth_weight=config_dict['sequence']['temporal_smooth_weight'],
        bone_direction_weight=config_dict['enhancements']['bone_direction_weight'],
        foot_lr=config_dict['joint_learning_rates']['foot_lr'],
        knee_lr=config_dict['joint_learning_rates']['knee_lr'],
        hip_lr=config_dict['joint_learning_rates']['hip_lr'],
        bone_length_soft_weight=config_dict['enhancements']['bone_length_soft_weight'],
        bone_length_ratio_weight=config_dict['enhancements']['bone_length_ratio_weight'],
        contact_weight_multiplier=config_dict['enhancements']['contact_weight_multiplier'],
        velocity_smooth_weight=config_dict['enhancements']['velocity_smooth_weight'],
        angle_limit_weight=config_dict['enhancements']['angle_limit_weight'],
        ground_penetration_weight=config_dict['enhancements']['ground_penetration_weight'],
        use_hierarchical_opt=False,  # Not in config, always False
    )


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

    # ENHANCEMENT: YAML config file support
    parser.add_argument('--config', default=None,
                        help='Path to YAML configuration file (overrides default hyperparameters)')
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

    # ENHANCEMENT: Head/Neck inclusion (교수님 요청)
    parser.add_argument('--include_head_neck', action='store_true',
                        help='Include head and neck joints in optimization (for with_arm datasets)')

    # ENHANCEMENT: Adaptive frame sampling (교수님 요청 - 긴 시퀀스 처리)
    parser.add_argument('--max_frames_optimize', type=int, default=500,
                        help='Maximum frames to optimize. Longer sequences sampled uniformly (default: 500)')
    parser.add_argument('--disable_frame_sampling', action='store_true',
                        help='Disable adaptive frame sampling (use all frames, may be slow)')

    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[WARN] CUDA not available, falling back to CPU')

    run_name = derive_run_name(args.b3d)
    # Add prefix to distinguish from baseline v10 and include physics extraction
    run_name = "v11_with_physics_" + run_name
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
    addb_joints_full, dt = load_b3d_sequence(
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
    addb_joints_full = convert_addb_to_smpl_coords(addb_joints_full)
    print(f'  Frames (original): {addb_joints_full.shape[0]}')
    print(f'  Joints : {addb_joints_full.shape[1]}')
    print(f'  dt     : {dt:.6f} s ({1.0 / dt:.2f} fps)')

    # ENHANCEMENT: Adaptive frame sampling (교수님 요청 - 긴 시퀀스 처리)
    if not args.disable_frame_sampling and addb_joints_full.shape[0] > args.max_frames_optimize:
        print(f'\n  Adaptive frame sampling enabled (max_frames={args.max_frames_optimize}):')
        addb_joints, selected_frame_indices, sampling_stride = adaptive_frame_sampling(
            addb_joints_full,
            max_frames=args.max_frames_optimize
        )
        frame_sampling_enabled = True
    else:
        addb_joints = addb_joints_full
        selected_frame_indices = np.arange(len(addb_joints))
        sampling_stride = 1
        frame_sampling_enabled = False
        if args.disable_frame_sampling:
            print(f'  Frame sampling disabled (--disable_frame_sampling)')
        else:
            print(f'  No frame sampling needed ({addb_joints.shape[0]} <= {args.max_frames_optimize})')

    print(f'  Frames (optimized): {addb_joints.shape[0]}')

    print('\n[2/5] Resolving joint correspondences...')
    auto_map = auto_map_addb_to_smpl(joint_names)
    overrides = load_mapping_overrides(args.map_json, joint_names)
    mapping = resolve_mapping(joint_names, auto_map, overrides)

    # ENHANCEMENT: Filter mapping based on joint inclusion flags (교수님 요청)
    original_count = len(mapping)
    if args.lower_body_only:
        # No_Arm datasets: Lower body only
        mapping = {addb_idx: smpl_idx for addb_idx, smpl_idx in mapping.items()
                   if smpl_idx in LOWER_BODY_JOINTS}
        print(f'  Lower body filtering: {original_count} → {len(mapping)} joints')
    elif args.include_head_neck:
        # With_Arm datasets: Full body INCLUDING head/neck
        mapping = {addb_idx: smpl_idx for addb_idx, smpl_idx in mapping.items()
                   if smpl_idx in FULL_BODY_WITH_HEAD_JOINTS}
        print(f'  Full body + head/neck filtering: {original_count} → {len(mapping)} joints')
    else:
        # Default: Full body EXCLUDING head/neck
        mapping = {addb_idx: smpl_idx for addb_idx, smpl_idx in mapping.items()
                   if smpl_idx not in HEAD_NECK_JOINTS}
        print(f'  Full body (no head/neck): {original_count} → {len(mapping)} joints')

    print(f'  Using {len(mapping)} correspondences')
    for idx, smpl_idx in sorted(mapping.items(), key=lambda kv: kv[0]):
        addb_name = joint_names[idx] if idx < len(joint_names) else str(idx)
        smpl_name = SMPL_JOINT_NAMES[smpl_idx]
        # Highlight head/neck joints if included
        if smpl_idx in HEAD_NECK_JOINTS and args.include_head_neck:
            print(f'    {addb_name:20s} → {smpl_name} [HEAD/NECK]')
        else:
            print(f'    {addb_name:20s} → {smpl_name}')

    unmapped = [name for i, name in enumerate(joint_names) if i not in mapping]
    if unmapped:
        print(f'  Unmapped AddB joints ({len(unmapped)}): {", ".join(unmapped)}')

    print('\n[3/5] Initialising SMPL model...')
    smpl = SMPLModel(args.smpl_model, device=device)

    # ENHANCEMENT: Load config from YAML if provided
    if args.config:
        print(f'  Loading configuration from: {args.config}')
        config_dict = load_config_from_yaml(args.config)
        cfg = config_to_optimization_config(config_dict)
        print('  Configuration loaded from YAML file')
    else:
        print('  Using command-line arguments for configuration')
        cfg = OptimisationConfig(
            shape_lr=args.shape_lr,
            shape_iters=args.shape_iters,
            shape_sample_frames=args.shape_sample_frames,
            pose_lr=args.pose_lr,
            pose_iters=args.pose_iters,
            tolerance_mm=args.tolerance_mm,
            max_passes=args.max_passes,
            pose_reg_weight=args.pose_reg,
            trans_reg_weight=args.trans_reg
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

    print('\n[4/6] Optimising SMPL parameters...')
    betas, poses_np, trans_np, metrics, pred_joints = fitter.run()

    print('\n[5/6] Extracting physics data for sampled frames...')
    physics_data = extract_physics_for_sampled_frames(
        b3d_path=args.b3d,
        sampled_frame_indices=selected_frame_indices,
        trial=args.trial,
        processing_pass=args.processing_pass,
        start=args.start
    )
    print(f'  Extracted physics data for {len(selected_frame_indices)} sampled frames')
    print(f'  DOFs: {physics_data["n_dof"]}')
    print(f'  GRF shape: {physics_data["grf"].shape}')
    print(f'  Joint torques shape: {physics_data["joint_torques"].shape}')

    print('\n[6/6] Metrics:')
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

    # Save physics data
    print('Saving physics data...')
    np.save(os.path.join(out_dir, 'ground_reaction_forces.npy'), physics_data['grf'])
    np.save(os.path.join(out_dir, 'joint_torques.npy'), physics_data['joint_torques'])
    np.save(os.path.join(out_dir, 'joint_velocities.npy'), physics_data['joint_velocities'])
    np.save(os.path.join(out_dir, 'joint_accelerations.npy'), physics_data['joint_accelerations'])
    np.save(os.path.join(out_dir, 'com_position.npy'), physics_data['com_pos'])
    np.save(os.path.join(out_dir, 'com_velocity.npy'), physics_data['com_vel'])
    np.save(os.path.join(out_dir, 'com_acceleration.npy'), physics_data['com_acc'])
    np.save(os.path.join(out_dir, 'residual_forces.npy'), physics_data['residual_forces'])
    np.save(os.path.join(out_dir, 'sampled_frame_indices.npy'), physics_data['sampled_frame_indices'])
    print(f'  Saved physics data to {out_dir}/')

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
        # ENHANCEMENT: Frame sampling info (교수님 요청)
        'frame_sampling': {
            'enabled': frame_sampling_enabled,
            'original_num_frames': int(addb_joints_full.shape[0]),
            'sampled_num_frames': int(addb_joints.shape[0]),
            'selected_frame_indices': selected_frame_indices.tolist(),
            'sampling_stride': int(sampling_stride),
            'max_frames_optimize': args.max_frames_optimize,
        },
        # ENHANCEMENT: Head/neck inclusion info (교수님 요청)
        'joint_inclusion': {
            'lower_body_only': args.lower_body_only,
            'include_head_neck': args.include_head_neck,
        },
        # NEW in v11: Physics data info
        'physics_data': {
            'n_dof': int(physics_data['n_dof']),
            'grf_shape': list(physics_data['grf'].shape),
            'joint_torques_shape': list(physics_data['joint_torques'].shape),
            'includes': [
                'ground_reaction_forces.npy',
                'joint_torques.npy',
                'joint_velocities.npy',
                'joint_accelerations.npy',
                'com_position.npy',
                'com_velocity.npy',
                'com_acceleration.npy',
                'residual_forces.npy',
                'sampled_frame_indices.npy'
            ],
            'description': {
                'grf': '[T, 6] ground reaction forces (fx, fy, fz) and torques (tx, ty, tz)',
                'joint_torques': '[T, N_dofs] joint torques',
                'joint_velocities': '[T, N_dofs] joint velocities',
                'joint_accelerations': '[T, N_dofs] joint accelerations',
                'com_position': '[T, 3] center of mass position',
                'com_velocity': '[T, 3] center of mass velocity',
                'com_acceleration': '[T, 3] center of mass acceleration',
                'residual_forces': '[T, 6] residual forces/moments at root',
                'sampled_frame_indices': '[T] original frame indices that were sampled'
            }
        },
    }

    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print('\n' + '=' * 80)
    print(f'DONE! Results saved to: {out_dir}')
    print('=' * 80 + '\n')


if __name__ == '__main__':
    main()
