#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMPL vs SKEL 비교 최적화 스크립트

동일한 AddBiomechanics (.b3d) 데이터를 SMPL과 SKEL 두 모델로 최적화하고,
결과를 OBJ 파일로 저장하여 비교할 수 있게 합니다.

Usage:
    python compare_smpl_skel.py --b3d <path> --out_dir <path> [--num_frames N]

Output:
    {out_dir}/
    ├── smpl/
    │   ├── frame_0000.obj
    │   ├── smpl_params.npz
    │   └── joints.npy
    ├── skel/
    │   ├── frame_0000.obj
    │   ├── skel_params.npz
    │   └── joints.npy
    └── comparison_metrics.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

# SMPL/SKEL 모델 import
from models.smpl_model import SMPLModel, SMPL_NUM_BETAS, SMPL_NUM_JOINTS
from models.skel_model import (
    SKELModelWrapper,
    SKEL_NUM_BETAS,
    SKEL_NUM_JOINTS,
    SKEL_NUM_POSE_DOF,
    AUTO_JOINT_NAME_MAP_SKEL,
    SKEL_JOINT_NAMES,
    SKEL_JOINT_NAMES_WITH_ACROMIAL,
)

# nimblephysics for .b3d loading
try:
    import nimblephysics as nimble
except ImportError as exc:
    raise RuntimeError(
        "nimblephysics is required to read AddBiomechanics .b3d files. "
        "Install: pip install nimblephysics"
    ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SMPL_MODEL_PATH = '/egr/research-zijunlab/kwonjoon/01_Code/Physica/models/SMPL_NEUTRAL.pkl'
SKEL_MODEL_PATH = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'

# SMPL Joint Names (for reference)
SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand'
]

# AddB → SMPL joint mapping
# NOTE: AddB와 SMPL은 동일한 좌표계 사용
# AddB _r → SMPL right, AddB _l → SMPL left (직접 매핑)
AUTO_JOINT_NAME_MAP_SMPL: Dict[str, str] = {
    'ground_pelvis': 'pelvis',
    'pelvis': 'pelvis',
    'root': 'pelvis',
    # AddB _r → SMPL right (직접 매핑)
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
    'mtp_r': 'right_foot',
    'mtp_l': 'left_foot',
    'toe_r': 'right_foot',
    'toe_l': 'left_foot',
    'acromial_r': 'right_shoulder',
    'shoulder_r': 'right_shoulder',
    'acromial_l': 'left_shoulder',
    'shoulder_l': 'left_shoulder',
    'elbow_r': 'right_elbow',
    'elbow_l': 'left_elbow',
    'wrist_r': 'right_wrist',
    'wrist_l': 'left_wrist',
    'radius_hand_r': 'right_wrist',
    'radius_hand_l': 'left_wrist',
    'hand_r': 'right_hand',
    'hand_l': 'left_hand',
    'neck': 'neck',
    'head': 'head',
}


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def load_b3d_data(b3d_path: str, num_frames: int = -1, processing_pass: int = 0) -> Tuple[np.ndarray, List[str], List[int], Dict]:
    """
    Load .b3d file and extract joint positions

    Returns:
        joints: [T, N, 3] joint positions
        joint_names: list of joint names
        joint_parents: list of parent indices (-1 for root)
        subject_info: dict with height_m and mass_kg from AddB subject metadata
    """
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    # Get skeleton
    trial_idx = 0
    skel = subject.readSkel(trial_idx)

    # Get frames
    num_trials = subject.getNumTrials()
    if num_trials == 0:
        raise ValueError(f"No trials found in {b3d_path}")

    trial_len = subject.getTrialLength(trial_idx)

    if num_frames > 0:
        trial_len = min(trial_len, num_frames)

    # Read all frames at once
    frames = subject.readFrames(
        trial=trial_idx,
        startFrame=0,
        numFramesToRead=trial_len,
        contactThreshold=20
    )

    if len(frames) == 0:
        raise ValueError(f"No frames read from {b3d_path}")

    # Get processing pass index
    pp_idx = min(processing_pass, len(frames[0].processingPasses) - 1)

    # Infer joint names from first frame
    pp = frames[0].processingPasses[pp_idx]
    pos = np.asarray(pp.pos, dtype=np.float32)
    if pos.size == skel.getNumDofs():
        skel.setPositions(pos)

    # Get world joint positions, names, and parent info
    world_joints = []
    joint_parent_map = {}  # joint_name -> parent_name
    for i in range(skel.getNumJoints()):
        joint = skel.getJoint(i)
        world = joint.getChildBodyNode().getWorldTransform().translation()
        joint_name = joint.getName()
        world_joints.append((joint_name, world))

        # Get parent body node
        parent_body = joint.getParentBodyNode()
        if parent_body is not None:
            # Find parent joint name
            parent_joint_name = None
            for j in range(skel.getNumJoints()):
                if skel.getJoint(j).getChildBodyNode() == parent_body:
                    parent_joint_name = skel.getJoint(j).getName()
                    break
            joint_parent_map[joint_name] = parent_joint_name
        else:
            joint_parent_map[joint_name] = None

    # Use jointCenters from processing pass
    joint_centers_first = np.asarray(pp.jointCenters, dtype=np.float32).reshape(-1, 3)
    joint_names = []
    for center in joint_centers_first:
        dists = [np.linalg.norm(center - w) for _, w in world_joints]
        best = int(np.argmin(dists))
        joint_names.append(world_joints[best][0])

    # Build parent index list for the joint_names order
    joint_parents = []
    for name in joint_names:
        parent_name = joint_parent_map.get(name)
        if parent_name is not None and parent_name in joint_names:
            parent_idx = joint_names.index(parent_name)
        else:
            parent_idx = -1
        joint_parents.append(parent_idx)

    # Extract joint positions for all frames
    all_joints = []
    for frame in frames:
        pp = frame.processingPasses[pp_idx]
        joint_centers = np.asarray(pp.jointCenters, dtype=np.float32).reshape(-1, 3)
        all_joints.append(joint_centers)

    joints = np.array(all_joints)  # [T, N, 3]

    # Extract subject body info (height, mass, sex, age)
    subject_info = {
        'height_m': subject.getHeightM(),
        'mass_kg': subject.getMassKg(),
        'sex': subject.getBiologicalSex(),  # 'male' or 'female'
        'age': subject.getAgeYears(),
    }

    print(f"Loaded {len(all_joints)} frames from {b3d_path}")
    print(f"  Joint names: {joint_names}")
    print(f"  Joint positions shape: {joints.shape}")
    print(f"  Subject: Height={subject_info['height_m']:.2f}m, Mass={subject_info['mass_kg']:.1f}kg, Sex={subject_info['sex']}")

    return joints, joint_names, joint_parents, subject_info


# ---------------------------------------------------------------------------
# Beta Sensitivity Constants (measured from experiments)
# ---------------------------------------------------------------------------

# SKEL Baselines (betas=0)
SKEL_BASELINE = {
    'male': {'height_mm': 1581, 'shoulder_mm': 351, 'pelvis_mm': 167},
    'female': {'height_mm': 1459, 'shoulder_mm': 303, 'pelvis_mm': 164},
}

# SKEL Beta Sensitivity (mm per unit beta)
# Note: Male and Female have OPPOSITE directions for beta[0]
SKEL_SENSITIVITY = {
    'male': {
        'beta0_height': -70.9,   # beta[0] +1 → height -70.9mm
        'beta0_shoulder': -11.8,
        'beta1_height': +13.2,
        'beta1_shoulder': -13.6,  # beta[1] +1 → shoulder -13.6mm
    },
    'female': {
        'beta0_height': +65.5,   # beta[0] +1 → height +65.5mm (opposite!)
        'beta0_shoulder': +11.3,
        'beta1_height': +11.9,
        'beta1_shoulder': -13.8,  # beta[1] +1 → shoulder -13.8mm
    },
}

# SMPL Baselines (betas=0)
SMPL_BASELINE = {
    'male': {'height_mm': 1575, 'shoulder_mm': 391, 'pelvis_mm': 119},
    'female': {'height_mm': 1442, 'shoulder_mm': 327, 'pelvis_mm': 140},
    'neutral': {'height_mm': 1509, 'shoulder_mm': 359, 'pelvis_mm': 130},  # approximate
}

# SMPL Beta Sensitivity (mm per unit beta)
SMPL_SENSITIVITY = {
    'male': {
        'beta0_height': -69.2,
        'beta0_shoulder': -12.1,
        'beta1_height': +15.7,
        'beta1_shoulder': -14.8,
    },
    'female': {
        'beta0_height': +64.5,
        'beta0_shoulder': +11.3,
        'beta1_height': +14.9,
        'beta1_shoulder': -13.1,
    },
    'neutral': {
        'beta0_height': -70.0,  # approximate
        'beta0_shoulder': -12.0,
        'beta1_height': +15.0,
        'beta1_shoulder': -14.0,
    },
}


def extract_body_proportions(joints: np.ndarray, joint_names: List[str]) -> Dict:
    """
    Extract body proportions from AddB joint positions.

    Args:
        joints: [T, N, 3] joint positions
        joint_names: list of joint names

    Returns:
        dict with shoulder_width, pelvis_width, femur_length, etc. (all in meters)
    """
    # Use first frame for proportions
    j = joints[0]

    def get_idx(name):
        return joint_names.index(name) if name in joint_names else None

    # Get indices
    acr_r = get_idx('acromial_r')
    acr_l = get_idx('acromial_l')
    hip_r = get_idx('hip_r')
    hip_l = get_idx('hip_l')
    knee_r = get_idx('walker_knee_r')
    knee_l = get_idx('walker_knee_l')
    ankle_r = get_idx('ankle_r')
    ankle_l = get_idx('ankle_l')
    elbow_r = get_idx('elbow_r')
    elbow_l = get_idx('elbow_l')
    wrist_r = get_idx('radius_hand_r')
    wrist_l = get_idx('radius_hand_l')

    proportions = {}

    # Shoulder width
    if acr_r is not None and acr_l is not None:
        proportions['shoulder_width'] = np.linalg.norm(j[acr_r] - j[acr_l])

    # Pelvis width
    if hip_r is not None and hip_l is not None:
        proportions['pelvis_width'] = np.linalg.norm(j[hip_r] - j[hip_l])

    # Femur length (average of both legs)
    if knee_r is not None and hip_r is not None:
        femur_r = np.linalg.norm(j[knee_r] - j[hip_r])
        if knee_l is not None and hip_l is not None:
            femur_l = np.linalg.norm(j[knee_l] - j[hip_l])
            proportions['femur_length'] = (femur_r + femur_l) / 2
        else:
            proportions['femur_length'] = femur_r

    # Tibia length
    if ankle_r is not None and knee_r is not None:
        tibia_r = np.linalg.norm(j[ankle_r] - j[knee_r])
        if ankle_l is not None and knee_l is not None:
            tibia_l = np.linalg.norm(j[ankle_l] - j[knee_l])
            proportions['tibia_length'] = (tibia_r + tibia_l) / 2
        else:
            proportions['tibia_length'] = tibia_r

    # Humerus length
    if elbow_r is not None and acr_r is not None:
        humerus_r = np.linalg.norm(j[elbow_r] - j[acr_r])
        if elbow_l is not None and acr_l is not None:
            humerus_l = np.linalg.norm(j[elbow_l] - j[acr_l])
            proportions['humerus_length'] = (humerus_r + humerus_l) / 2
        else:
            proportions['humerus_length'] = humerus_r

    return proportions


def estimate_skel_beta(subject_info: Dict, body_proportions: Dict, gender: str,
                       device: torch.device) -> torch.Tensor:
    """
    Estimate SKEL beta parameters from AddB subject info and body proportions.

    Uses measured sensitivity values to compute initial beta values.

    Args:
        subject_info: dict with height_m, mass_kg, sex, age
        body_proportions: dict with shoulder_width, pelvis_width, etc. (in meters)
        gender: 'male' or 'female'
        device: torch device

    Returns:
        betas: [10] tensor of estimated beta values
    """
    betas = torch.zeros(SKEL_NUM_BETAS, device=device)

    baseline = SKEL_BASELINE[gender]
    sens = SKEL_SENSITIVITY[gender]

    # Get target values from AddB
    target_height_mm = subject_info.get('height_m', 1.7) * 1000
    target_shoulder_mm = body_proportions.get('shoulder_width', 0.35) * 1000

    # Solve for beta[0] and beta[1] using linear system:
    # target_height = baseline_height + beta0 * sens_height_0 + beta1 * sens_height_1
    # target_shoulder = baseline_shoulder + beta0 * sens_shoulder_0 + beta1 * sens_shoulder_1

    # Set up linear system: A @ [beta0, beta1]^T = b
    A = np.array([
        [sens['beta0_height'], sens['beta1_height']],
        [sens['beta0_shoulder'], sens['beta1_shoulder']]
    ])
    b = np.array([
        target_height_mm - baseline['height_mm'],
        target_shoulder_mm - baseline['shoulder_mm']
    ])

    # Solve least squares
    try:
        beta_01, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        betas[0] = beta_01[0]
        betas[1] = beta_01[1]
    except np.linalg.LinAlgError:
        # Fallback: solve height only
        betas[0] = (target_height_mm - baseline['height_mm']) / sens['beta0_height']

    # Clamp to reasonable range
    betas = torch.clamp(betas, -5.0, 5.0)

    print(f"  Estimated SKEL betas from AddB:")
    print(f"    Target: height={target_height_mm:.0f}mm, shoulder={target_shoulder_mm:.0f}mm")
    print(f"    Baseline ({gender}): height={baseline['height_mm']}mm, shoulder={baseline['shoulder_mm']}mm")
    print(f"    Estimated beta[0]={betas[0].item():.3f}, beta[1]={betas[1].item():.3f}")

    return betas


def estimate_smpl_beta(subject_info: Dict, body_proportions: Dict, gender: str,
                       device: torch.device) -> torch.Tensor:
    """
    Estimate SMPL beta parameters from AddB subject info and body proportions.

    Uses measured sensitivity values to compute initial beta values.

    Args:
        subject_info: dict with height_m, mass_kg, sex, age
        body_proportions: dict with shoulder_width, pelvis_width, etc. (in meters)
        gender: 'male', 'female', or 'neutral'
        device: torch device

    Returns:
        betas: [10] tensor of estimated beta values
    """
    betas = torch.zeros(SMPL_NUM_BETAS, device=device)

    baseline = SMPL_BASELINE[gender]
    sens = SMPL_SENSITIVITY[gender]

    # Get target values from AddB
    target_height_mm = subject_info.get('height_m', 1.7) * 1000
    target_shoulder_mm = body_proportions.get('shoulder_width', 0.35) * 1000

    # Solve for beta[0] and beta[1] using linear system
    A = np.array([
        [sens['beta0_height'], sens['beta1_height']],
        [sens['beta0_shoulder'], sens['beta1_shoulder']]
    ])
    b = np.array([
        target_height_mm - baseline['height_mm'],
        target_shoulder_mm - baseline['shoulder_mm']
    ])

    # Solve least squares
    try:
        beta_01, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        betas[0] = beta_01[0]
        betas[1] = beta_01[1]
    except np.linalg.LinAlgError:
        # Fallback: solve height only
        betas[0] = (target_height_mm - baseline['height_mm']) / sens['beta0_height']

    # Clamp to reasonable range
    betas = torch.clamp(betas, -5.0, 5.0)

    print(f"  Estimated SMPL betas from AddB:")
    print(f"    Target: height={target_height_mm:.0f}mm, shoulder={target_shoulder_mm:.0f}mm")
    print(f"    Baseline ({gender}): height={baseline['height_mm']}mm, shoulder={baseline['shoulder_mm']}mm")
    print(f"    Estimated beta[0]={betas[0].item():.3f}, beta[1]={betas[1].item():.3f}")

    return betas


def build_joint_mapping(addb_joint_names: List[str], target_joint_names: List[str],
                        mapping_dict: Dict[str, str]) -> Tuple[List[int], List[int]]:
    """Build index mapping from AddB joints to target model joints"""
    addb_indices = []
    target_indices = []

    for i, name in enumerate(addb_joint_names):
        name_lower = name.lower()
        if name_lower in mapping_dict:
            target_name = mapping_dict[name_lower]
            if target_name in target_joint_names:
                target_idx = target_joint_names.index(target_name)
                addb_indices.append(i)
                target_indices.append(target_idx)

    return addb_indices, target_indices


def save_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    """Save mesh as OBJ file"""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def create_sphere(center: np.ndarray, radius: float = 0.02, n_segments: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a UV sphere mesh at the given center

    Args:
        center: [3] center position
        radius: sphere radius (default 2cm)
        n_segments: number of latitude/longitude segments

    Returns:
        vertices: [V, 3]
        faces: [F, 3]
    """
    vertices = []
    faces = []

    # Generate vertices
    for i in range(n_segments + 1):
        lat = np.pi * i / n_segments - np.pi / 2
        for j in range(n_segments):
            lon = 2 * np.pi * j / n_segments
            x = radius * np.cos(lat) * np.cos(lon) + center[0]
            y = radius * np.cos(lat) * np.sin(lon) + center[1]
            z = radius * np.sin(lat) + center[2]
            vertices.append([x, y, z])

    # Generate faces
    for i in range(n_segments):
        for j in range(n_segments):
            p1 = i * n_segments + j
            p2 = i * n_segments + (j + 1) % n_segments
            p3 = (i + 1) * n_segments + (j + 1) % n_segments
            p4 = (i + 1) * n_segments + j
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

    return np.array(vertices), np.array(faces)


def create_cylinder(start: np.ndarray, end: np.ndarray, radius: float = 0.008, n_segments: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a cylinder mesh between two points

    Args:
        start: [3] start position
        end: [3] end position
        radius: cylinder radius (default 8mm)
        n_segments: number of circular segments

    Returns:
        vertices: [V, 3]
        faces: [F, 3]
    """
    # Direction vector
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    direction = direction / length

    # Find perpendicular vectors
    if abs(direction[0]) < 0.9:
        perp1 = np.cross(direction, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(direction, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    vertices = []
    faces = []

    # Generate circle vertices at start and end
    for t, center in enumerate([start, end]):
        for i in range(n_segments):
            angle = 2 * np.pi * i / n_segments
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(center + offset)

    # Generate side faces
    for i in range(n_segments):
        p1 = i
        p2 = (i + 1) % n_segments
        p3 = n_segments + (i + 1) % n_segments
        p4 = n_segments + i
        faces.append([p1, p2, p3])
        faces.append([p1, p3, p4])

    # Cap faces (optional - makes it look better)
    # Start cap
    start_center_idx = len(vertices)
    vertices.append(start)
    for i in range(n_segments):
        faces.append([start_center_idx, (i + 1) % n_segments, i])

    # End cap
    end_center_idx = len(vertices)
    vertices.append(end)
    for i in range(n_segments):
        faces.append([end_center_idx, n_segments + i, n_segments + (i + 1) % n_segments])

    return np.array(vertices), np.array(faces)


def create_joint_spheres(joints: np.ndarray, radius: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create spheres at all joint positions

    Args:
        joints: [N, 3] joint positions
        radius: sphere radius

    Returns:
        vertices: combined vertices
        faces: combined faces (with adjusted indices)
    """
    all_verts = []
    all_faces = []
    vert_offset = 0

    for joint in joints:
        verts, faces = create_sphere(joint, radius)
        all_verts.append(verts)
        all_faces.append(faces + vert_offset)
        vert_offset += len(verts)

    if len(all_verts) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    return np.vstack(all_verts), np.vstack(all_faces)


def create_skeleton_bones(joints: np.ndarray, parents: List[int], radius: float = 0.008) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create cylinder bones connecting joints based on parent hierarchy

    Args:
        joints: [N, 3] joint positions
        parents: parent index for each joint (-1 for root)
        radius: cylinder radius

    Returns:
        vertices: combined vertices
        faces: combined faces
    """
    all_verts = []
    all_faces = []
    vert_offset = 0

    for i, parent_idx in enumerate(parents):
        if parent_idx >= 0 and parent_idx < len(joints):
            verts, faces = create_cylinder(joints[parent_idx], joints[i], radius)
            if len(verts) > 0:
                all_verts.append(verts)
                all_faces.append(faces + vert_offset)
                vert_offset += len(verts)

    if len(all_verts) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    return np.vstack(all_verts), np.vstack(all_faces)


# Joint parent hierarchies
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# SKEL_PARENTS는 이제 모델에서 동적으로 가져옴 (SKELModelWrapper.parents property)


def compute_mpjpe(pred_joints: np.ndarray, target_joints: np.ndarray,
                  pred_indices: List[int], target_indices: List[int],
                  exclude_target_indices: List[int] = None) -> float:
    """Compute Mean Per-Joint Position Error (mm)

    Args:
        pred_joints: Predicted joint positions [T, J_pred, 3]
        target_joints: Target joint positions [T, J_target, 3]
        pred_indices: Indices into pred_joints for mapped joints
        target_indices: Indices into target_joints (AddB) for mapped joints
        exclude_target_indices: AddB joint indices to exclude from MPJPE calculation
    """
    # Filter out excluded joints
    if exclude_target_indices is not None and len(exclude_target_indices) > 0:
        valid_pairs = [(pi, ti) for pi, ti in zip(pred_indices, target_indices)
                       if ti not in exclude_target_indices]
        if len(valid_pairs) == 0:
            return float('nan')
        pred_indices, target_indices = zip(*valid_pairs)
        pred_indices, target_indices = list(pred_indices), list(target_indices)

    pred = pred_joints[:, pred_indices]
    target = target_joints[:, target_indices]

    # Handle NaN values
    valid_mask = ~np.isnan(target).any(axis=-1)
    if valid_mask.sum() == 0:
        return float('nan')

    errors = np.linalg.norm(pred[valid_mask] - target[valid_mask], axis=-1)
    return float(errors.mean() * 1000)  # Convert to mm


# ---------------------------------------------------------------------------
# Bone Direction Loss (from v11)
# ---------------------------------------------------------------------------

# SMPL bone pairs: (parent_idx, child_idx) - defines limb directions
# Using SMPL joint indices:
# 0=pelvis, 1=left_hip, 2=right_hip, 3=spine1, 4=left_knee, 5=right_knee
# 6=spine2, 7=left_ankle, 8=right_ankle, 9=spine3, 10=left_foot, 11=right_foot
# 12=neck, 13=left_collar, 14=right_collar, 15=head, 16=left_shoulder, 17=right_shoulder
# 18=left_elbow, 19=right_elbow, 20=left_wrist, 21=right_wrist, 22=left_hand, 23=right_hand
SMPL_BONE_PAIRS = [
    (0, 1), (0, 2),      # pelvis → hips
    (1, 4), (4, 7),      # left leg: hip → knee → ankle
    (2, 5), (5, 8),      # right leg: hip → knee → ankle
    (7, 10), (8, 11),    # ankles → feet
    (16, 18), (18, 20),  # left arm: shoulder → elbow → wrist
    (17, 19), (19, 21),  # right arm: shoulder → elbow → wrist
]

# SKEL bone pairs (SKEL joint indices)
# 0=pelvis, 1=femur_r, 2=tibia_r, 3=talus_r, 4=calcn_r, 5=toes_r
# 6=femur_l, 7=tibia_l, 8=talus_l, 9=calcn_l, 10=toes_l
# 11=lumbar, 12=thorax, 13=head, 14=scapula_r, 15=humerus_r
# 16=ulna_r, 17=radius_r, 18=hand_r, 19=scapula_l, 20=humerus_l
# 21=ulna_l, 22=radius_l, 23=hand_l
SKEL_BONE_PAIRS = [
    (0, 1), (0, 6),      # pelvis → femurs
    (1, 2), (2, 3),      # right leg: femur → tibia → talus
    (6, 7), (7, 8),      # left leg: femur → tibia → talus
    (3, 4), (8, 9),      # talus → calcn (feet)
    (0, 11), (11, 12),   # spine: pelvis → lumbar → thorax
    (12, 13),            # thorax → head
    (15, 16), (16, 17),  # right arm: humerus → ulna → radius
    (20, 21), (21, 22),  # left arm: humerus → ulna → radius
]


def build_bone_pair_mapping(addb_joint_names: List[str], model_joint_names: List[str],
                            joint_mapping: Dict[str, str], model_bone_pairs: List[Tuple[int, int]]
                            ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Build mapping between AddB bone pairs and model bone pairs.

    Returns list of ((addb_parent_idx, addb_child_idx), (model_parent_idx, model_child_idx))
    """
    # Build reverse mapping: model_joint_name -> addb_joint_name
    model_to_addb = {}
    for addb_name, model_name in joint_mapping.items():
        if model_name not in model_to_addb:
            model_to_addb[model_name] = addb_name

    # Find addb indices for model joint names
    model_idx_to_addb_idx = {}
    for model_idx, model_name in enumerate(model_joint_names):
        if model_name in model_to_addb:
            addb_name = model_to_addb[model_name]
            # Find in addb_joint_names (case-insensitive)
            for i, name in enumerate(addb_joint_names):
                if name.lower() == addb_name.lower():
                    model_idx_to_addb_idx[model_idx] = i
                    break

    # Build bone pair mapping
    bone_pairs = []
    for model_parent, model_child in model_bone_pairs:
        if model_parent in model_idx_to_addb_idx and model_child in model_idx_to_addb_idx:
            addb_parent = model_idx_to_addb_idx[model_parent]
            addb_child = model_idx_to_addb_idx[model_child]
            bone_pairs.append(((addb_parent, addb_child), (model_parent, model_child)))

    return bone_pairs


def compute_bone_direction_loss(pred_joints: torch.Tensor, target_joints: torch.Tensor,
                                 bone_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> torch.Tensor:
    """
    Compute bone direction loss using cosine similarity.

    Args:
        pred_joints: [T, J_pred, 3] predicted joints
        target_joints: [T, J_target, 3] target joints
        bone_pairs: list of ((target_parent, target_child), (pred_parent, pred_child))

    Returns:
        loss: scalar tensor
    """
    if len(bone_pairs) == 0:
        return torch.tensor(0.0, device=pred_joints.device)

    loss = torch.tensor(0.0, device=pred_joints.device)
    valid_count = 0

    for (t_parent, t_child), (p_parent, p_child) in bone_pairs:
        # Target bone direction
        target_dir = target_joints[:, t_child] - target_joints[:, t_parent]
        target_length = torch.norm(target_dir, dim=-1, keepdim=True)

        # Skip if bone has zero length
        valid = (target_length.squeeze(-1) > 1e-6)
        if valid.sum() == 0:
            continue

        # Predicted bone direction
        pred_dir = pred_joints[:, p_child] - pred_joints[:, p_parent]

        # Normalize
        target_dir_norm = F.normalize(target_dir, dim=-1)
        pred_dir_norm = F.normalize(pred_dir, dim=-1)

        # Cosine similarity loss: 1 - cos(angle) = 0 when aligned
        cosine_sim = (target_dir_norm * pred_dir_norm).sum(dim=-1)
        bone_loss = (1 - cosine_sim)[valid].mean()

        loss = loss + bone_loss
        valid_count += 1

    if valid_count > 0:
        loss = loss / valid_count

    return loss


def compute_joint_angle_limits_loss(poses: torch.Tensor) -> torch.Tensor:
    """
    Compute loss for joint angle limits (SMPL poses).

    Enforces biomechanically plausible joint angles:
    - Knees: flexion only (x-axis), no hyperextension
    - Elbows: flexion only

    Args:
        poses: [T, 24, 3] SMPL pose parameters (axis-angle)

    Returns:
        loss: scalar tensor
    """
    loss = torch.tensor(0.0, device=poses.device)

    # Knee limits (joints 4=left_knee, 5=right_knee)
    # Knee x-axis: flexion [-2.5, 0.1] radians
    for knee_idx in [4, 5]:
        knee_x = poses[:, knee_idx, 0]
        # Penalize hyperextension (x > 0.1)
        loss = loss + torch.relu(knee_x - 0.1).mean()
        # Penalize over-flexion (x < -2.5)
        loss = loss + torch.relu(-2.5 - knee_x).mean()

    # Elbow limits (joints 18=left_elbow, 19=right_elbow)
    # Elbow y-axis: flexion [0, 2.5] for left, [-2.5, 0] for right
    # Left elbow
    elbow_l_y = poses[:, 18, 1]
    loss = loss + torch.relu(-elbow_l_y).mean()  # y should be >= 0
    loss = loss + torch.relu(elbow_l_y - 2.5).mean()

    # Right elbow
    elbow_r_y = poses[:, 19, 1]
    loss = loss + torch.relu(elbow_r_y).mean()  # y should be <= 0
    loss = loss + torch.relu(-2.5 - elbow_r_y).mean()

    return loss


def compute_skel_joint_angle_limits_loss(poses: torch.Tensor) -> torch.Tensor:
    """
    Compute loss for joint angle limits (SKEL poses).

    SKEL has 46 DOFs with specific joint ordering from SKEL_POSE_PARAM_NAMES:
    - DOF 0-2: pelvis_tilt, pelvis_list, pelvis_rotation
    - DOF 3-5: hip_flexion_r, hip_adduction_r, hip_rotation_r
    - DOF 6: knee_angle_r (RIGHT KNEE)
    - DOF 7-9: ankle_angle_r, subtalar_angle_r, mtp_angle_r
    - DOF 10-12: hip_flexion_l, hip_adduction_l, hip_rotation_l
    - DOF 13: knee_angle_l (LEFT KNEE)
    - DOF 14-16: ankle_angle_l, subtalar_angle_l, mtp_angle_l
    - DOF 17-25: lumbar/thorax/head rotations
    - DOF 26-35: right arm (scapula, shoulder, elbow, radioulnar, wrist)
    - DOF 36-45: left arm

    Args:
        poses: [T, 46] SKEL pose parameters

    Returns:
        loss: scalar tensor
    """
    loss = torch.tensor(0.0, device=poses.device)

    # === KNEE 과신전 방지 (DOF 6, 13) - 가장 중요 ===
    for idx in [6, 13]:
        knee = poses[:, idx]
        loss = loss + torch.relu(-knee - 0.1).mean() * 10.0   # 과신전 방지
        loss = loss + torch.relu(knee - 2.6).mean() * 1.0     # 과굴곡 방지

    # === 극단적인 pelvis tilt만 방지 ===
    pelvis_list = poses[:, 1]  # 좌우 기울기
    loss = loss + torch.relu(torch.abs(pelvis_list) - 0.5).mean() * 2.0

    # === 극단적인 hip adduction만 방지 (다리 심하게 꼬이는 것만) ===
    hip_add_r = poses[:, 4]
    hip_add_l = poses[:, 11]
    loss = loss + torch.relu(torch.abs(hip_add_r) - 0.6).mean() * 3.0
    loss = loss + torch.relu(torch.abs(hip_add_l) - 0.6).mean() * 3.0

    return loss


# ---------------------------------------------------------------------------
# Virtual Shoulder Position (for acromial matching in SKEL)
# ---------------------------------------------------------------------------

def compute_virtual_shoulder_positions(
    skel_joints: torch.Tensor,
    alpha: float = 0.65
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute virtual shoulder positions as weighted average of scapula and humerus.

    AddBiomechanics의 acromial joint (견봉 - 어깨 끝)은 SKEL의 어떤 단일 joint와도
    직접 대응하지 않습니다. scapula는 acromial보다 안쪽(medial), humerus는 더 안쪽+아래.
    따라서 두 joint의 가중 평균으로 acromial에 근접한 "가상 어깨 joint"를 생성합니다.

    Args:
        skel_joints: [T, 24, 3] SKEL joint positions
        alpha: Weight for scapula (1.0 = pure scapula, 0.0 = pure humerus)
               Default 0.65 since scapula is more lateral (closer to acromial)

    Returns:
        virtual_shoulder_r: [T, 3] right virtual shoulder position
        virtual_shoulder_l: [T, 3] left virtual shoulder position

    SKEL joint indices:
        14: scapula_r, 15: humerus_r
        19: scapula_l, 20: humerus_l
    """
    # Right shoulder: weighted average of scapula_r (14) and humerus_r (15)
    virtual_shoulder_r = alpha * skel_joints[:, 14] + (1 - alpha) * skel_joints[:, 15]

    # Left shoulder: weighted average of scapula_l (19) and humerus_l (20)
    virtual_shoulder_l = alpha * skel_joints[:, 19] + (1 - alpha) * skel_joints[:, 20]

    return virtual_shoulder_r, virtual_shoulder_l


# ---------------------------------------------------------------------------
# Dynamic Virtual Acromial (data-driven vertex selection)
# ---------------------------------------------------------------------------

def compute_acromial_offset_from_addb(
    addb_joints: np.ndarray,
    addb_joint_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average offset from humerus to acromial using AddB data.

    Formula: d_R = mean_t(J_acromial_R(t) - J_humerus_R(t))

    Args:
        addb_joints: [T, N, 3] AddB joint positions across T frames
        addb_joint_names: List of joint names in AddB

    Returns:
        offset_r: [3] average offset vector for right side (acromial - humerus)
        offset_l: [3] average offset vector for left side
    """
    # Find acromial indices
    acr_r_names = ['acromial_r', 'shoulder_r']
    acr_l_names = ['acromial_l', 'shoulder_l']

    acr_r_idx = None
    acr_l_idx = None
    for name in acr_r_names:
        if name in addb_joint_names:
            acr_r_idx = addb_joint_names.index(name)
            break
    for name in acr_l_names:
        if name in addb_joint_names:
            acr_l_idx = addb_joint_names.index(name)
            break

    if acr_r_idx is None or acr_l_idx is None:
        raise ValueError(f"AddB data missing acromial joints. Available: {addb_joint_names}")

    # Find humerus/elbow indices to estimate GH joint position
    # AddB typically has acromial but not explicit GH/humerus joint
    # We estimate GH from acromial + arm direction
    elbow_r_idx = None
    elbow_l_idx = None
    for name in ['elbow_r', 'elbow_right']:
        if name in addb_joint_names:
            elbow_r_idx = addb_joint_names.index(name)
            break
    for name in ['elbow_l', 'elbow_left']:
        if name in addb_joint_names:
            elbow_l_idx = addb_joint_names.index(name)
            break

    acr_r = addb_joints[:, acr_r_idx, :]  # [T, 3]
    acr_l = addb_joints[:, acr_l_idx, :]  # [T, 3]

    if elbow_r_idx is not None and elbow_l_idx is not None:
        elbow_r = addb_joints[:, elbow_r_idx, :]
        elbow_l = addb_joints[:, elbow_l_idx, :]

        # Arm direction: acromial -> elbow (normalized)
        arm_dir_r = elbow_r - acr_r
        arm_dir_r = arm_dir_r / (np.linalg.norm(arm_dir_r, axis=-1, keepdims=True) + 1e-8)

        arm_dir_l = elbow_l - acr_l
        arm_dir_l = arm_dir_l / (np.linalg.norm(arm_dir_l, axis=-1, keepdims=True) + 1e-8)

        # GH joint is typically ~4-5cm along arm axis from acromial (toward elbow)
        GH_OFFSET_MAGNITUDE = 0.045  # 4.5cm - anatomical estimate

        # Estimated GH (humerus) position
        gh_r_est = acr_r + GH_OFFSET_MAGNITUDE * arm_dir_r
        gh_l_est = acr_l + GH_OFFSET_MAGNITUDE * arm_dir_l

        # Offset from estimated GH to acromial (this is what we add to SKEL humerus)
        offset_r = np.mean(acr_r - gh_r_est, axis=0)  # [3]
        offset_l = np.mean(acr_l - gh_l_est, axis=0)  # [3]
    else:
        # Fallback: use anatomically reasonable lateral offset
        # Acromial is lateral to GH by ~4-5cm
        # In world coords, this depends on pose - use average acromial position as reference
        offset_r = np.array([0.0, 0.045, 0.0])  # Adjust based on coordinate system
        offset_l = np.array([0.0, -0.045, 0.0])

    print(f"  Acromial offset from AddB:")
    print(f"    Right: [{offset_r[0]*1000:.1f}, {offset_r[1]*1000:.1f}, {offset_r[2]*1000:.1f}] mm")
    print(f"    Left:  [{offset_l[0]*1000:.1f}, {offset_l[1]*1000:.1f}, {offset_l[2]*1000:.1f}] mm")

    return offset_r, offset_l


def find_virtual_acromial_vertices(
    skel_model: 'SKELModelWrapper',
    betas: torch.Tensor,
    offset_r: np.ndarray,
    offset_l: np.ndarray,
    k: int = 4,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, List[int]]:
    """
    Find K nearest SKEL mesh vertices to target acromial positions.

    Uses neutral pose (T-pose) SKEL mesh.

    Args:
        skel_model: SKELModelWrapper instance
        betas: [10] shape parameters
        offset_r: [3] offset from humerus to acromial (right)
        offset_l: [3] offset from humerus to acromial (left)
        k: number of nearest vertices to find
        device: torch device

    Returns:
        Dict with 'right' and 'left' keys, each containing list of K vertex indices
    """
    from scipy.spatial import KDTree

    betas_t = betas.detach().to(device)
    if betas_t.dim() == 1:
        betas_t = betas_t.unsqueeze(0)  # [1, 10]

    # Generate SKEL mesh in neutral pose
    poses = torch.zeros(1, SKEL_NUM_POSE_DOF, device=device)
    trans = torch.zeros(1, 3, device=device)

    with torch.no_grad():
        vertices, joints = skel_model.forward(betas_t, poses, trans)

    # Ensure [6890, 3] and [24, 3]
    if vertices.dim() == 3:
        vertices = vertices[0]
        joints = joints[0]

    vertices_np = vertices.cpu().numpy()
    joints_np = joints.cpu().numpy()

    # SKEL joint indices
    HUMERUS_R_IDX = 15
    HUMERUS_L_IDX = 20

    # Compute target acromial positions
    humerus_r = joints_np[HUMERUS_R_IDX]  # [3]
    humerus_l = joints_np[HUMERUS_L_IDX]  # [3]

    target_acr_r = humerus_r + offset_r  # [3]
    target_acr_l = humerus_l + offset_l  # [3]

    # Build KDTree for efficient nearest neighbor search
    tree = KDTree(vertices_np)

    # Find K nearest neighbors
    dist_r, idx_r = tree.query(target_acr_r, k=k)
    dist_l, idx_l = tree.query(target_acr_l, k=k)

    # Convert to list
    idx_r = idx_r.tolist() if hasattr(idx_r, 'tolist') else [idx_r]
    idx_l = idx_l.tolist() if hasattr(idx_l, 'tolist') else [idx_l]

    print(f"  Virtual acromial vertices (K={k}):")
    print(f"    Right target: [{target_acr_r[0]:.3f}, {target_acr_r[1]:.3f}, {target_acr_r[2]:.3f}]")
    print(f"    Right vertices: {idx_r}, dist: {dist_r}")
    print(f"    Left target: [{target_acr_l[0]:.3f}, {target_acr_l[1]:.3f}, {target_acr_l[2]:.3f}]")
    print(f"    Left vertices: {idx_l}, dist: {dist_l}")

    return {
        'right': idx_r,
        'left': idx_l
    }


def compute_dynamic_virtual_acromial(
    vertices: torch.Tensor,
    vertex_idx: Dict[str, List[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute virtual acromial positions from mesh vertices using dynamic indices.

    Args:
        vertices: [T, 6890, 3] or [6890, 3] SKEL mesh vertices
        vertex_idx: Dict with 'right' and 'left' vertex index lists

    Returns:
        virtual_r: [T, 3] or [3] right virtual acromial position
        virtual_l: [T, 3] or [3] left virtual acromial position
    """
    right_idx = vertex_idx['right']
    left_idx = vertex_idx['left']

    if vertices.dim() == 2:
        # Single frame: [6890, 3]
        virtual_r = vertices[right_idx, :].mean(dim=0)  # [3]
        virtual_l = vertices[left_idx, :].mean(dim=0)   # [3]
    else:
        # Multiple frames: [T, 6890, 3]
        virtual_r = vertices[:, right_idx, :].mean(dim=1)  # [T, 3]
        virtual_l = vertices[:, left_idx, :].mean(dim=1)   # [T, 3]

    return virtual_r, virtual_l


# ---------------------------------------------------------------------------
# SMPL Optimization
# ---------------------------------------------------------------------------

def optimize_smpl(target_joints: np.ndarray, addb_joint_names: List[str],
                  device: torch.device, num_iters: int = 100,
                  gender: str = 'neutral',
                  subject_info: Optional[Dict] = None,
                  body_proportions: Optional[Dict] = None) -> Dict:
    """
    Optimize SMPL parameters to fit target joints

    Args:
        target_joints: [T, N, 3] target joint positions from AddB
        addb_joint_names: list of AddB joint names
        device: torch device
        num_iters: number of optimization iterations
        gender: 'male', 'female', or 'neutral'
        subject_info: dict with height_m, mass_kg, sex (for beta initialization)
        body_proportions: dict with shoulder_width, pelvis_width, etc. (for beta initialization)

    Returns dict with:
        - betas: shape parameters
        - poses: pose parameters [T, 24, 3]
        - trans: translations [T, 3]
        - vertices: mesh vertices [T, 6890, 3]
        - joints: joint positions [T, 24, 3]
    """
    print(f"\n=== SMPL Optimization (gender={gender}) ===")

    # Load model with gender
    smpl = SMPLModel(gender=gender, device=device)

    # Build joint mapping
    addb_indices, smpl_indices = build_joint_mapping(
        addb_joint_names, SMPL_JOINT_NAMES, AUTO_JOINT_NAME_MAP_SMPL
    )
    print(f"  Mapped {len(addb_indices)} joints")

    # Build bone pair mapping for bone direction loss
    bone_pairs = build_bone_pair_mapping(
        addb_joint_names, SMPL_JOINT_NAMES, AUTO_JOINT_NAME_MAP_SMPL, SMPL_BONE_PAIRS
    )
    print(f"  Mapped {len(bone_pairs)} bone pairs for direction loss")

    T = target_joints.shape[0]
    target = torch.tensor(target_joints, dtype=torch.float32, device=device)

    # Initialize parameters - use beta estimation if subject_info provided
    if subject_info is not None and body_proportions is not None:
        betas = estimate_smpl_beta(subject_info, body_proportions, gender, device)
        betas.requires_grad = True
    else:
        betas = torch.zeros(SMPL_NUM_BETAS, device=device, requires_grad=True)
    poses = torch.zeros(T, SMPL_NUM_JOINTS, 3, device=device, requires_grad=True)
    trans = torch.zeros(T, 3, device=device, requires_grad=True)

    # Initialize translation from pelvis
    pelvis_idx = addb_joint_names.index('ground_pelvis') if 'ground_pelvis' in addb_joint_names else 0
    with torch.no_grad():
        trans[:] = target[:, pelvis_idx]
    trans.requires_grad = True

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': [betas], 'lr': 0.01},
        {'params': [poses], 'lr': 0.02},
        {'params': [trans], 'lr': 0.01}
    ])

    # Loss weights (tuned for balance between joint position accuracy and mesh quality)
    # NOTE: bone direction loss는 좌우 반전 매핑 문제로 인해 일시 비활성화
    bone_dir_weight = 0.0  # Bone direction loss weight (disabled due to L/R mapping issue)
    angle_limit_weight = 0.5  # Joint angle limits weight

    # Optimization loop
    for it in range(num_iters):
        optimizer.zero_grad()

        # Forward
        all_verts = []
        all_joints = []
        for t in range(T):
            verts, joints = smpl.forward(betas.unsqueeze(0), poses[t].unsqueeze(0), trans[t].unsqueeze(0))
            all_verts.append(verts)
            all_joints.append(joints)

        pred_joints = torch.stack(all_joints, dim=0).squeeze(1)  # [T, 24, 3]

        # Joint position loss
        pred_subset = pred_joints[:, smpl_indices]
        target_subset = target[:, addb_indices]

        valid_mask = ~torch.isnan(target_subset).any(dim=-1)
        if valid_mask.sum() > 0:
            loss = F.mse_loss(pred_subset[valid_mask], target_subset[valid_mask])
        else:
            loss = torch.tensor(0.0, device=device)

        # Bone direction loss (key for preventing mesh twisting)
        bone_dir_loss = compute_bone_direction_loss(pred_joints, target, bone_pairs)
        loss = loss + bone_dir_weight * bone_dir_loss

        # Joint angle limits (enforce biomechanical constraints)
        angle_limit_loss = compute_joint_angle_limits_loss(poses)
        loss = loss + angle_limit_weight * angle_limit_loss

        # Regularization
        # General pose regularization (increased from 0.001 to 0.01)
        loss = loss + 0.01 * (poses ** 2).mean()
        loss = loss + 0.01 * (betas ** 2).mean()

        # Stronger regularization on spine joints (3=spine1, 6=spine2, 9=spine3)
        spine_indices = [3, 6, 9]
        spine_poses = poses[:, spine_indices, :]
        loss = loss + 0.05 * (spine_poses ** 2).mean()

        # Also regularize collar joints (13=left_collar, 14=right_collar)
        collar_indices = [13, 14]
        collar_poses = poses[:, collar_indices, :]
        loss = loss + 0.02 * (collar_poses ** 2).mean()

        loss.backward()
        optimizer.step()

        if (it + 1) % 20 == 0:
            mpjpe = compute_mpjpe(pred_joints.detach().cpu().numpy(),
                                  target_joints, smpl_indices, addb_indices)
            print(f"  Iter {it+1}/{num_iters}: Loss={loss.item():.4f}, MPJPE={mpjpe:.1f}mm, BoneDir={bone_dir_loss.item():.4f}")

    # Final forward pass
    all_verts = []
    all_joints = []
    with torch.no_grad():
        for t in range(T):
            verts, joints = smpl.forward(betas.unsqueeze(0), poses[t].unsqueeze(0), trans[t].unsqueeze(0))
            all_verts.append(verts.cpu().numpy())
            all_joints.append(joints.cpu().numpy())

    return {
        'betas': betas.detach().cpu().numpy(),
        'poses': poses.detach().cpu().numpy(),
        'trans': trans.detach().cpu().numpy(),
        'vertices': np.concatenate(all_verts, axis=0),
        'joints': np.concatenate(all_joints, axis=0),
        'faces': smpl.faces.cpu().numpy() if smpl.faces is not None else None,
        'addb_indices': addb_indices,
        'model_indices': smpl_indices,
    }


# ---------------------------------------------------------------------------
# Bone Length Loss (for body shape/proportion matching)
# ---------------------------------------------------------------------------

# SKEL ↔ AddB bone pair mapping for bone length loss
# Format: (skel_parent_name, skel_child_name, addb_parent_name, addb_child_name)
SKEL_ADDB_BONE_LENGTH_PAIRS = [
    # Legs
    ('pelvis', 'femur_r', 'ground_pelvis', 'hip_r'),       # pelvis → hip_r
    ('pelvis', 'femur_l', 'ground_pelvis', 'hip_l'),       # pelvis → hip_l
    ('femur_r', 'tibia_r', 'hip_r', 'walker_knee_r'),      # femur R
    ('femur_l', 'tibia_l', 'hip_l', 'walker_knee_l'),      # femur L
    ('tibia_r', 'talus_r', 'walker_knee_r', 'ankle_r'),    # tibia R
    ('tibia_l', 'talus_l', 'walker_knee_l', 'ankle_l'),    # tibia L
    ('talus_r', 'calcn_r', 'ankle_r', 'subtalar_r'),       # ankle → subtalar R
    ('talus_l', 'calcn_l', 'ankle_l', 'subtalar_l'),       # ankle → subtalar L
    # Shoulder width (thorax → humerus: acromial이 humerus에 매핑됨)
    ('thorax', 'humerus_r', 'back', 'acromial_r'),         # 등 → 어깨-팔 연결점 R
    ('thorax', 'humerus_l', 'back', 'acromial_l'),         # 등 → 어깨-팔 연결점 L
    # Upper arms (humerus → ulna)
    ('humerus_r', 'ulna_r', 'acromial_r', 'elbow_r'),      # 어깨 → 팔꿈치 (상완) R
    ('humerus_l', 'ulna_l', 'acromial_l', 'elbow_l'),      # 어깨 → 팔꿈치 (상완) L
    # Forearms
    ('ulna_r', 'hand_r', 'elbow_r', 'radius_hand_r'),      # forearm R (ulna → hand)
    ('ulna_l', 'hand_l', 'elbow_l', 'radius_hand_l'),      # forearm L
]


def build_bone_length_pairs(addb_joint_names: List[str], skel_joint_names: List[str],
                            bone_pair_defs: List[Tuple[str, str, str, str]]
                            ) -> List[Tuple[int, int, int, int]]:
    """
    Build index-based bone length pairs from name-based definitions.

    Args:
        addb_joint_names: AddB joint names
        skel_joint_names: SKEL joint names
        bone_pair_defs: list of (skel_parent_name, skel_child_name, addb_parent_name, addb_child_name)

    Returns:
        List of (skel_parent_idx, skel_child_idx, addb_parent_idx, addb_child_idx)
    """
    pairs = []

    # Build name-to-index maps (case-insensitive)
    addb_name_to_idx = {name.lower(): i for i, name in enumerate(addb_joint_names)}
    skel_name_to_idx = {name.lower(): i for i, name in enumerate(skel_joint_names)}

    for skel_parent, skel_child, addb_parent, addb_child in bone_pair_defs:
        skel_p_idx = skel_name_to_idx.get(skel_parent.lower())
        skel_c_idx = skel_name_to_idx.get(skel_child.lower())
        addb_p_idx = addb_name_to_idx.get(addb_parent.lower())
        addb_c_idx = addb_name_to_idx.get(addb_child.lower())

        if all(idx is not None for idx in [skel_p_idx, skel_c_idx, addb_p_idx, addb_c_idx]):
            pairs.append((skel_p_idx, skel_c_idx, addb_p_idx, addb_c_idx))

    return pairs


def compute_bone_length_loss(pred_joints: torch.Tensor, target_joints: torch.Tensor,
                              bone_pairs: List[Tuple[int, int, int, int]]) -> torch.Tensor:
    """
    Compute bone length matching loss.

    Args:
        pred_joints: [T, J_pred, 3] predicted joints (SKEL)
        target_joints: [T, J_target, 3] target joints (AddB)
        bone_pairs: list of (pred_parent_idx, pred_child_idx, target_parent_idx, target_child_idx)

    Returns:
        loss: scalar tensor (MSE of bone lengths)
    """
    if len(bone_pairs) == 0:
        return torch.tensor(0.0, device=pred_joints.device)

    loss = torch.tensor(0.0, device=pred_joints.device)
    valid_count = 0

    for p_parent, p_child, t_parent, t_child in bone_pairs:
        # Predicted bone vector and length
        pred_bone = pred_joints[:, p_child] - pred_joints[:, p_parent]
        pred_len = torch.norm(pred_bone, dim=-1)

        # Target bone vector and length
        target_bone = target_joints[:, t_child] - target_joints[:, t_parent]
        target_len = torch.norm(target_bone, dim=-1)

        # Skip if target bone has zero length (invalid data)
        valid_mask = target_len > 1e-6
        if valid_mask.sum() == 0:
            continue

        # MSE loss on bone lengths
        bone_loss = ((pred_len[valid_mask] - target_len[valid_mask]) ** 2).mean()
        loss = loss + bone_loss
        valid_count += 1

    if valid_count > 0:
        loss = loss / valid_count

    return loss


# ---------------------------------------------------------------------------
# Virtual Acromial (for skin-based shoulder fitting)
# ---------------------------------------------------------------------------

# SKEL skin vertex indices for acromial (shoulder outer surface)
# These vertices represent the outer shoulder surface of SKEL skin mesh
SKEL_ACROMIAL_VERTEX_IDX = {
    'right': [4125, 4124, 5293, 5290],
    'left': [635, 636, 1830, 1829]
}


def compute_virtual_acromial(vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute virtual acromial position from SKEL skin mesh vertices.

    This gives us the actual shoulder surface position, which should match
    AddB's acromial joint position (which is also on the skin surface).

    Args:
        vertices: [T, V, 3] SKEL mesh vertices

    Returns:
        virtual_r: [T, 3] right shoulder surface position
        virtual_l: [T, 3] left shoulder surface position
    """
    right_idx = SKEL_ACROMIAL_VERTEX_IDX['right']
    left_idx = SKEL_ACROMIAL_VERTEX_IDX['left']

    virtual_r = vertices[:, right_idx, :].mean(dim=1)  # [T, 3]
    virtual_l = vertices[:, left_idx, :].mean(dim=1)   # [T, 3]

    return virtual_r, virtual_l


# ---------------------------------------------------------------------------
# Beta Initialization from AddB Proportions
# ---------------------------------------------------------------------------

# SKEL default values (betas=0)
SKEL_DEFAULT_HUMERUS_WIDTH = 0.351  # meters (351mm)
SKEL_DEFAULT_HEIGHT = 1.565  # meters

# Height-based beta estimation constants (measured empirically)
SKEL_BASELINE_HEIGHT = 1.565  # meters (height when betas=0)
BETA0_HEIGHT_SENSITIVITY = -0.071  # meters per beta[0] unit (negative = taller)

# Beta sensitivity (measured empirically)
# How much each beta unit changes the metric
BETA_SHOULDER_SENSITIVITY = {
    0: 0.01175,   # +23.5mm per 2 units = 0.01175m per unit
    1: 0.01355,   # +27.1mm per 2 units = 0.01355m per unit (most effective)
}


def estimate_beta_from_height(target_height: float, device: torch.device) -> torch.Tensor:
    """
    Estimate SKEL beta parameters from subject height.

    Uses the empirically measured relationship between beta[0] and body height.
    Beta[0] is the primary shape component that controls overall body size.

    Relationship (measured empirically):
        target_height = baseline_height + sensitivity * beta[0]
        1.85m = 1.565m + (-0.071) * (-4.0)  →  beta[0] = -4.0 for 1.85m

    Args:
        target_height: Subject height in meters (from AddB subject.getHeightM())
        device: torch device

    Returns:
        betas: [10] beta values with beta[0] set based on height
    """
    betas = torch.zeros(SKEL_NUM_BETAS, device=device)

    # Calculate beta[0] from height difference
    # Formula: target_height = baseline_height + sensitivity * beta0
    # Solve:   beta0 = (target_height - baseline_height) / sensitivity
    height_diff = target_height - SKEL_BASELINE_HEIGHT
    beta0 = height_diff / BETA0_HEIGHT_SENSITIVITY

    # Clamp to safe range [-5, 5]
    beta0 = max(-5.0, min(5.0, beta0))
    betas[0] = beta0

    return betas


def estimate_initial_beta_from_addb(
    target_joints: np.ndarray,
    addb_joint_names: List[str],
    device: torch.device
) -> torch.Tensor:
    """
    Estimate initial beta values from AddB body proportions.

    Uses shoulder width (acromial_r to acromial_l distance) to estimate
    beta[1] which has the most impact on shoulder width.

    Args:
        target_joints: [T, J, 3] AddB joint positions
        addb_joint_names: List of joint names
        device: torch device

    Returns:
        betas: [10] initial beta values
    """
    betas = torch.zeros(SKEL_NUM_BETAS, device=device)

    # Get acromial positions
    if 'acromial_r' in addb_joint_names and 'acromial_l' in addb_joint_names:
        acr_r_idx = addb_joint_names.index('acromial_r')
        acr_l_idx = addb_joint_names.index('acromial_l')

        acr_r = target_joints[:, acr_r_idx, :]  # [T, 3]
        acr_l = target_joints[:, acr_l_idx, :]  # [T, 3]

        # Compute target shoulder width (mean across frames)
        target_width = np.linalg.norm(acr_r - acr_l, axis=-1).mean()  # meters

        # How much wider than default?
        width_diff = target_width - SKEL_DEFAULT_HUMERUS_WIDTH

        # Use beta[1] (most effective for shoulder width)
        # beta[1] = width_diff / sensitivity
        beta_1_init = width_diff / BETA_SHOULDER_SENSITIVITY[1]

        # Clamp to reasonable range [-3, 3]
        beta_1_init = np.clip(beta_1_init, -3.0, 3.0)

        betas[1] = beta_1_init

        print(f"  Beta initialization from AddB:")
        print(f"    Target shoulder width: {target_width * 1000:.1f} mm")
        print(f"    Default width: {SKEL_DEFAULT_HUMERUS_WIDTH * 1000:.1f} mm")
        print(f"    Difference: {width_diff * 1000:.1f} mm")
        print(f"    Initial beta[1]: {beta_1_init:.3f}")

    return betas


# ---------------------------------------------------------------------------
# Shoulder Width Loss
# ---------------------------------------------------------------------------

def compute_shoulder_width_loss(
    pred_joints: torch.Tensor,
    target_width: torch.Tensor
) -> torch.Tensor:
    """
    Loss to match SKEL shoulder (humerus) width to target width.

    This directly penalizes difference between SKEL humerus-to-humerus width
    and the target AddB acromial-to-acromial width.

    Args:
        pred_joints: [T, 24, 3] SKEL joint positions
        target_width: [T] target shoulder width per frame (meters)

    Returns:
        loss: scalar loss value
    """
    # Use scapula for shoulder width (acromial → scapula 매핑에 맞춤)
    scapula_r_idx = SKEL_JOINT_NAMES.index('scapula_r')
    scapula_l_idx = SKEL_JOINT_NAMES.index('scapula_l')

    scapula_r = pred_joints[:, scapula_r_idx, :]  # [T, 3]
    scapula_l = pred_joints[:, scapula_l_idx, :]  # [T, 3]

    # Current shoulder width (scapula 간 거리)
    pred_width = torch.norm(scapula_r - scapula_l, dim=-1)  # [T]

    # MSE loss on width
    loss = F.mse_loss(pred_width, target_width)

    return loss


# ---------------------------------------------------------------------------
# SKEL IK-based Pose Initialization
# ---------------------------------------------------------------------------

# SKEL bone pairs for IK initialization (parent_joint_idx → child_joint_idx)
# Maps to SKEL DOF indices for rotation
SKEL_IK_BONE_MAP = {
    # Legs
    (0, 1): [3, 4, 5],      # pelvis → femur_r: hip_r DOFs
    (1, 2): [6],            # femur_r → tibia_r: knee_r DOF
    (0, 6): [10, 11, 12],   # pelvis → femur_l: hip_l DOFs
    (6, 7): [13],           # femur_l → tibia_l: knee_l DOF
    # Spine
    (0, 11): [17, 18, 19],  # pelvis → lumbar: lumbar DOFs
    (11, 12): [20, 21, 22], # lumbar → thorax: thorax DOFs
    # Arms (simplified - just shoulder)
    (12, 15): [28, 29, 30], # thorax → humerus_r: shoulder_r DOFs
    (12, 20): [38, 39, 40], # thorax → humerus_l: shoulder_l DOFs
}


def estimate_initial_poses_ik_skel(
    target_joints: torch.Tensor,
    addb_indices: List[int],
    skel_indices: List[int],
    skel: 'SKELModelWrapper',
    device: torch.device
) -> torch.Tensor:
    """
    Simple IK-based pose initialization for SKEL.

    Uses bone direction vectors to estimate initial rotations,
    giving optimization a better starting point than zero pose.

    Args:
        target_joints: [T, N_addb, 3] target joint positions
        addb_indices: AddB joint indices for mapped joints
        skel_indices: SKEL joint indices for mapped joints
        skel: SKEL model wrapper
        device: torch device

    Returns:
        Initial poses [T, 46] in Euler angle representation
    """
    T = target_joints.shape[0]
    initial_poses = torch.zeros(T, SKEL_NUM_POSE_DOF, device=device)

    # Build mapping: skel_joint_idx → addb_joint_idx
    skel_to_addb = {}
    for addb_idx, skel_idx in zip(addb_indices, skel_indices):
        skel_to_addb[skel_idx] = addb_idx

    # Compute default bone directions from T-pose
    with torch.no_grad():
        betas_zero = torch.zeros(SKEL_NUM_BETAS, device=device)
        poses_zero = torch.zeros(1, SKEL_NUM_POSE_DOF, device=device)
        trans_zero = torch.zeros(1, 3, device=device)
        _, tpose_joints = skel.forward(betas_zero.unsqueeze(0), poses_zero, trans_zero)
        tpose_joints = tpose_joints[0]  # [24, 3]

    # Process each frame
    for t in range(T):
        target_frame = target_joints[t]

        # Skip frames with all NaN
        if torch.isnan(target_frame).all():
            continue

        # Process each bone in IK map
        for (parent_skel_idx, child_skel_idx), dof_indices in SKEL_IK_BONE_MAP.items():
            # Check if both joints are mapped
            if parent_skel_idx not in skel_to_addb or child_skel_idx not in skel_to_addb:
                continue

            parent_addb_idx = skel_to_addb[parent_skel_idx]
            child_addb_idx = skel_to_addb[child_skel_idx]

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
            default_dir = tpose_joints[child_skel_idx] - tpose_joints[parent_skel_idx]
            default_dir_norm = torch.norm(default_dir)
            if default_dir_norm < 1e-6:
                continue
            default_dir = default_dir / default_dir_norm

            # Compute rotation from default to observed
            cross = torch.cross(default_dir, obs_dir)
            dot = torch.dot(default_dir, obs_dir).clamp(-1.0, 1.0)

            cross_norm = torch.norm(cross)
            if cross_norm < 1e-6:
                continue

            # Rotation axis and angle
            axis = cross / cross_norm
            angle = torch.acos(dot) * 0.3  # Scale down for stability

            # Distribute rotation to DOFs (simplified: just use first DOF)
            if len(dof_indices) > 0:
                # Simple heuristic: spread rotation across DOFs
                for i, dof_idx in enumerate(dof_indices):
                    if i < 3:  # x, y, z components
                        initial_poses[t, dof_idx] = axis[i] * angle

    return initial_poses


# ---------------------------------------------------------------------------
# SKEL Optimization
# ---------------------------------------------------------------------------

def optimize_skel(target_joints: np.ndarray, addb_joint_names: List[str],
                  device: torch.device, num_iters: int = 100,
                  virtual_acromial_weight: float = 0.0,
                  shoulder_width_weight: float = 0.0,
                  use_beta_init: bool = False,
                  subject_height: float = None,
                  gender: str = 'male',
                  subject_info: Optional[Dict] = None,
                  body_proportions: Optional[Dict] = None,
                  use_dynamic_virtual_acromial: bool = False,
                  dynamic_acromial_k: int = 4) -> Dict:
    """
    Optimize SKEL parameters to fit target joints

    Args:
        target_joints: [T, N, 3] target joint positions from AddB
        addb_joint_names: list of AddB joint names
        device: torch device
        num_iters: number of optimization iterations
        virtual_acromial_weight: weight for virtual acromial loss (uses mesh vertices)
        shoulder_width_weight: weight for shoulder width loss
        use_beta_init: legacy beta initialization from shoulder width
        subject_height: height for beta initialization (fallback)
        gender: 'male' or 'female' (default: 'male')
        subject_info: dict with height_m, mass_kg, sex (for beta initialization)
        body_proportions: dict with shoulder_width, pelvis_width, etc. (for beta initialization)
        use_dynamic_virtual_acromial: if True, compute acromial vertex indices from AddB offset
        dynamic_acromial_k: number of nearest vertices for dynamic virtual acromial

    Returns dict with:
        - betas: shape parameters
        - poses: pose parameters [T, 46]
        - trans: translations [T, 3]
        - vertices: mesh vertices [T, 6890, 3]
        - joints: joint positions [T, 24, 3]
        - virtual_acromial_vertex_idx: (if dynamic) computed vertex indices
    """
    print(f"\n=== SKEL Optimization (gender={gender}) ===")

    # Load model with acromial joints enabled (26 joints instead of 24)
    # Use humerus mapping (acromial_r → humerus_r) instead of acromial regressor
    skel = SKELModelWrapper(
        model_path=SKEL_MODEL_PATH,
        gender=gender,
        device=device,
        add_acromial_joints=False  # Use humerus mapping, not acromial regressor
    )

    # Build joint mapping (full) - use 24-joint SKEL names (acromial → humerus mapping)
    addb_indices_full, skel_indices_full = build_joint_mapping(
        addb_joint_names, SKEL_JOINT_NAMES, AUTO_JOINT_NAME_MAP_SKEL
    )
    print(f"  Mapped {len(addb_indices_full)} joints (acromial → humerus mapping)")

    # Get acromial indices from AddB
    acr_r_idx = addb_joint_names.index('acromial_r') if 'acromial_r' in addb_joint_names else None
    acr_l_idx = addb_joint_names.index('acromial_l') if 'acromial_l' in addb_joint_names else None
    has_acromial = acr_r_idx is not None and acr_l_idx is not None

    # With add_acromial_joints=True, acromial is now a regular joint (index 24, 25)
    # No need to use virtual_acromial_loss - just include in regular joint loss
    # Set virtual_acromial_weight to 0 to disable the old virtual acromial loss
    virtual_acromial_weight = 0.0  # Disable old virtual acromial (now using regressor-based joints)

    # Remove acromial from joint loss (will use virtual acromial loss instead)
    if has_acromial and virtual_acromial_weight > 0:
        addb_indices = []
        skel_indices = []
        for ai, si in zip(addb_indices_full, skel_indices_full):
            if ai == acr_r_idx or ai == acr_l_idx:
                continue  # Skip acromial joints
            addb_indices.append(ai)
            skel_indices.append(si)
        print(f"  Joint loss uses {len(addb_indices)} joints (excluding acromial)")
        print(f"  Virtual acromial loss enabled with weight={virtual_acromial_weight}")
    else:
        addb_indices = addb_indices_full
        skel_indices = skel_indices_full

    # Build bone pair mapping for bone direction loss
    bone_pairs = build_bone_pair_mapping(
        addb_joint_names, SKEL_JOINT_NAMES, AUTO_JOINT_NAME_MAP_SKEL, SKEL_BONE_PAIRS
    )
    print(f"  Mapped {len(bone_pairs)} bone pairs for direction loss")

    # Build bone length pairs for body shape matching
    bone_length_pairs = build_bone_length_pairs(
        addb_joint_names, SKEL_JOINT_NAMES, SKEL_ADDB_BONE_LENGTH_PAIRS
    )
    print(f"  Mapped {len(bone_length_pairs)} bone pairs for length loss")

    T = target_joints.shape[0]
    target = torch.tensor(target_joints, dtype=torch.float32, device=device)

    # Get AddB acromial positions for virtual acromial loss and shoulder width
    if has_acromial:
        addb_acr_r = target[:, acr_r_idx, :]  # [T, 3]
        addb_acr_l = target[:, acr_l_idx, :]  # [T, 3]
        # Compute target shoulder width per frame (for shoulder width loss)
        target_shoulder_width = torch.norm(addb_acr_r - addb_acr_l, dim=-1)  # [T]

    # Initialize parameters
    # Priority: 1) subject_info + body_proportions, 2) height-only, 3) legacy, 4) zeros
    if subject_info is not None and body_proportions is not None:
        # Best: use full body info for beta estimation
        betas = estimate_skel_beta(subject_info, body_proportions, gender, device)
        betas.requires_grad = True
    elif subject_height is not None:
        # Fallback: height-only initialization
        betas = estimate_beta_from_height(subject_height, device)
        betas.requires_grad = True
        print(f"  Beta initialized from height {subject_height:.2f}m → beta[0]={betas[0].item():.2f} (will optimize)")
    elif use_beta_init and has_acromial:
        # Legacy: shoulder-width based initialization
        betas = estimate_initial_beta_from_addb(target_joints, addb_joint_names, device)
        betas.requires_grad = True
    else:
        betas = torch.zeros(SKEL_NUM_BETAS, device=device, requires_grad=True)
    poses = torch.zeros(T, SKEL_NUM_POSE_DOF, device=device, requires_grad=False)
    trans = torch.zeros(T, 3, device=device, requires_grad=False)

    # Initialize translation from pelvis
    pelvis_idx = addb_joint_names.index('ground_pelvis') if 'ground_pelvis' in addb_joint_names else 0
    with torch.no_grad():
        trans[:] = target[:, pelvis_idx]

    # IK-based pose initialization
    print("  Computing IK-based pose initialization...")
    with torch.no_grad():
        initial_poses = estimate_initial_poses_ik_skel(
            target, addb_indices, skel_indices, skel, device
        )
        poses[:] = initial_poses
    print(f"  IK init: pose norm = {torch.norm(poses).item():.4f}")

    # 3-Stage Optimization Setup
    # Stage 1: Beta already initialized from subject_info (done above)
    # Stage 2: Pose + Trans optimization (beta fixed)
    # Stage 3: Joint optimization (all parameters)

    # Calculate iterations per stage
    stage2_iters = num_iters // 2  # First half: pose only (beta fixed)
    stage3_iters = num_iters - stage2_iters  # Second half: joint optimization

    print(f"  3-Stage Optimization: Stage2={stage2_iters} iters (pose), Stage3={stage3_iters} iters (joint)")

    # Dynamic virtual acromial: compute offset and vertex indices
    virtual_acromial_vertex_idx = None
    if use_dynamic_virtual_acromial and has_acromial and virtual_acromial_weight > 0:
        print(f"  Computing dynamic virtual acromial (K={dynamic_acromial_k})...")
        # Step 1: Compute offset from AddB data
        offset_r, offset_l = compute_acromial_offset_from_addb(target_joints, addb_joint_names)
        # Step 2: Find K nearest vertices on SKEL mesh (using initial betas)
        virtual_acromial_vertex_idx = find_virtual_acromial_vertices(
            skel, betas.detach(), offset_r, offset_l, k=dynamic_acromial_k, device=device
        )

    # Loss weights (simplified for better MPJPE - reduced extra constraints)
    bone_dir_weight = 0.1   # Reduced bone direction loss
    bone_length_weight = 0.0  # DISABLED - was causing narrower shoulders

    # Per-joint weights for important joints (pelvis, hips, spine, shoulders)
    joint_weights = torch.ones(len(skel_indices), device=device)
    important_skel_joints = ['pelvis', 'femur_r', 'femur_l']  # 2.0x weight
    spine_joints = ['lumbar_body', 'thorax']  # 5.0x weight for spine alignment
    shoulder_joints = ['scapula_r', 'scapula_l']  # Scapula mapping for wider shoulders
    for i, skel_idx in enumerate(skel_indices):
        # Use 24-joint names (no acromial extension)
        joint_name = SKEL_JOINT_NAMES[skel_idx].lower()
        if joint_name in important_skel_joints:
            joint_weights[i] = 2.0  # 2배 가중치
        elif joint_name in spine_joints:
            joint_weights[i] = 5.0  # spine 정렬을 위해 높은 가중치
        elif joint_name in shoulder_joints:
            # Moderate weight for scapula (mapped from AddB acromial)
            joint_weights[i] = 2.0

    # ==========================================================================
    # STAGE 2: Pose + Trans optimization (beta fixed)
    # ==========================================================================
    print(f"\n  === Stage 2: Pose Optimization (beta fixed) ===")

    # Enable gradients for pose and trans only
    betas.requires_grad = False
    poses.requires_grad = True
    trans.requires_grad = True

    optimizer_pose = torch.optim.Adam([
        {'params': [poses], 'lr': 0.02},
        {'params': [trans], 'lr': 0.01}
    ])

    for it in range(stage2_iters):
        optimizer_pose.zero_grad()

        # Forward - SKEL can handle batched input
        verts, joints = skel.forward(
            betas.unsqueeze(0).expand(T, -1),
            poses,
            trans
        )

        pred_joints = joints  # [T, 24, 3]

        # Joint position loss with per-joint weights
        pred_subset = pred_joints[:, skel_indices]
        target_subset = target[:, addb_indices]

        valid_mask = ~torch.isnan(target_subset).any(dim=-1)
        if valid_mask.sum() > 0:
            # Weighted MSE loss
            diff = pred_subset - target_subset  # [T, J, 3]
            sq_diff = (diff ** 2).sum(dim=-1)  # [T, J]
            weighted_sq_diff = sq_diff * joint_weights.unsqueeze(0)  # [T, J]
            loss = weighted_sq_diff[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        # Bone direction loss (key for preventing mesh twisting)
        bone_dir_loss = compute_bone_direction_loss(pred_joints, target, bone_pairs)
        loss = loss + bone_dir_weight * bone_dir_loss

        # Minimal regularization (simplified for better MPJPE)
        loss = loss + 0.001 * (poses ** 2).mean()

        loss.backward()
        optimizer_pose.step()

        if (it + 1) % 20 == 0:
            mpjpe = compute_mpjpe(pred_joints.detach().cpu().numpy(),
                                  target_joints, skel_indices, addb_indices)
            print(f"  Stage2 Iter {it+1}/{stage2_iters}: Loss={loss.item():.4f}, MPJPE={mpjpe:.1f}mm, "
                  f"BoneDir={bone_dir_loss.item():.4f}")

    # ==========================================================================
    # STAGE 3: Joint optimization (all parameters)
    # ==========================================================================
    print(f"\n  === Stage 3: Joint Optimization (all params) ===")

    # Enable gradients for all parameters
    betas.requires_grad = True
    poses.requires_grad = True
    trans.requires_grad = True

    optimizer_all = torch.optim.Adam([
        {'params': [betas], 'lr': 0.02},  # Lower LR for fine-tuning
        {'params': [poses], 'lr': 0.01},
        {'params': [trans], 'lr': 0.005}
    ])

    for it in range(stage3_iters):
        optimizer_all.zero_grad()

        # Forward - SKEL can handle batched input
        verts, joints = skel.forward(
            betas.unsqueeze(0).expand(T, -1),
            poses,
            trans
        )

        pred_joints = joints  # [T, 24, 3]

        # Joint position loss with per-joint weights
        pred_subset = pred_joints[:, skel_indices]
        target_subset = target[:, addb_indices]

        valid_mask = ~torch.isnan(target_subset).any(dim=-1)
        if valid_mask.sum() > 0:
            # Weighted MSE loss
            diff = pred_subset - target_subset  # [T, J, 3]
            sq_diff = (diff ** 2).sum(dim=-1)  # [T, J]
            weighted_sq_diff = sq_diff * joint_weights.unsqueeze(0)  # [T, J]
            loss = weighted_sq_diff[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        # Bone direction loss (key for preventing mesh twisting)
        bone_dir_loss = compute_bone_direction_loss(pred_joints, target, bone_pairs)
        loss = loss + bone_dir_weight * bone_dir_loss

        # Bone length loss (key for body proportion/shape matching)
        bone_len_loss = compute_bone_length_loss(pred_joints, target, bone_length_pairs)
        loss = loss + bone_length_weight * bone_len_loss

        # Virtual acromial loss (skin surface → AddB acromial)
        if has_acromial and virtual_acromial_weight > 0:
            if virtual_acromial_vertex_idx is not None:
                virtual_r, virtual_l = compute_dynamic_virtual_acromial(verts, virtual_acromial_vertex_idx)
            else:
                virtual_r, virtual_l = compute_virtual_acromial(verts)
            virtual_acr_loss = (
                F.mse_loss(virtual_r, addb_acr_r) +
                F.mse_loss(virtual_l, addb_acr_l)
            )
            loss = loss + virtual_acromial_weight * virtual_acr_loss

        # Shoulder width loss (match humerus width to AddB acromial width)
        if has_acromial and shoulder_width_weight > 0:
            shoulder_w_loss = compute_shoulder_width_loss(pred_joints, target_shoulder_width)
            loss = loss + shoulder_width_weight * shoulder_w_loss

        # Minimal regularization (simplified for better MPJPE)
        loss = loss + 0.001 * (poses ** 2).mean()
        loss = loss + 0.001 * (betas ** 2).mean()

        loss.backward()
        optimizer_all.step()

        if (it + 1) % 20 == 0:
            mpjpe = compute_mpjpe(pred_joints.detach().cpu().numpy(),
                                  target_joints, skel_indices, addb_indices)
            betas_norm = torch.norm(betas).item()
            va_loss_str = f", VirtualAcr={virtual_acr_loss.item():.4f}" if has_acromial and virtual_acromial_weight > 0 else ""
            sw_loss_str = f", ShoulderW={shoulder_w_loss.item():.4f}" if has_acromial and shoulder_width_weight > 0 else ""
            # Also compute current shoulder width for monitoring (scapula 간 거리)
            scapula_r_idx = SKEL_JOINT_NAMES.index('scapula_r')
            scapula_l_idx = SKEL_JOINT_NAMES.index('scapula_l')
            curr_width = torch.norm(pred_joints[:, scapula_r_idx] - pred_joints[:, scapula_l_idx], dim=-1).mean().item() * 1000
            tgt_width = target_shoulder_width.mean().item() * 1000 if has_acromial else 0
            print(f"  Stage3 Iter {it+1}/{stage3_iters}: Loss={loss.item():.4f}, MPJPE={mpjpe:.1f}mm, "
                  f"BoneLen={bone_len_loss.item():.4f}, Betas||={betas_norm:.3f}, "
                  f"Width={curr_width:.0f}mm (tgt={tgt_width:.0f}mm){va_loss_str}{sw_loss_str}")

    # Final forward pass with skeleton mesh
    with torch.no_grad():
        verts, joints, skel_verts = skel.forward(
            betas.unsqueeze(0).expand(T, -1),
            poses,
            trans,
            return_skeleton=True
        )

    # Get actual parent hierarchy from SKEL model
    skel_parents = skel.parents.tolist()
    print(f"  SKEL parents from model: {skel_parents}")

    return {
        'betas': betas.detach().cpu().numpy(),
        'poses': poses.detach().cpu().numpy(),
        'trans': trans.detach().cpu().numpy(),
        'vertices': verts.cpu().numpy(),
        'joints': joints.cpu().numpy(),
        'skel_vertices': skel_verts.cpu().numpy(),  # skeleton mesh vertices
        'faces': skel.faces,
        'skel_faces': skel.skel_faces,  # skeleton mesh faces
        'parents': skel_parents,  # joint parent hierarchy from model
        'addb_indices': addb_indices,
        'model_indices': skel_indices,
        'virtual_acromial_vertex_idx': virtual_acromial_vertex_idx,  # dynamic vertex indices
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Compare SMPL vs SKEL optimization')
    parser.add_argument('--b3d', type=str, required=True, help='Path to .b3d file')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to process')
    parser.add_argument('--num_iters', type=int, default=100, help='Optimization iterations')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save_every', type=int, default=1, help='Save OBJ every N frames')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    addb_dir = os.path.join(args.out_dir, 'addb')
    smpl_dir = os.path.join(args.out_dir, 'smpl')
    skel_dir = os.path.join(args.out_dir, 'skel')
    os.makedirs(addb_dir, exist_ok=True)
    os.makedirs(smpl_dir, exist_ok=True)
    os.makedirs(skel_dir, exist_ok=True)

    # Load data (now returns joint_parents and subject_info)
    target_joints, joint_names, addb_parents, subject_info = load_b3d_data(args.b3d, args.num_frames)

    # Extract body proportions from AddB joints for beta estimation
    body_proportions = extract_body_proportions(target_joints, joint_names)
    print(f"\nBody proportions from AddB:")
    for k, v in body_proportions.items():
        print(f"  {k}: {v*1000:.1f} mm")

    # Determine gender for model selection
    sex = subject_info.get('sex', 'male')  # Default to male if not available
    # Map AddB sex to model gender (AddB uses 'male'/'female', models use same)
    gender = sex if sex in ['male', 'female'] else 'male'
    print(f"\nSubject info: height={subject_info.get('height_m', 'N/A')}m, "
          f"mass={subject_info.get('mass_kg', 'N/A')}kg, "
          f"sex={sex}, age={subject_info.get('age', 'N/A')}")

    # SMPL optimization (with gender-specific model and beta initialization)
    smpl_result = optimize_smpl(
        target_joints, joint_names, device, args.num_iters,
        gender=gender,
        subject_info=subject_info,
        body_proportions=body_proportions
    )

    # SKEL optimization (with gender-specific model and beta initialization)
    # Enable acromial loss weights to match shoulder width from AddB
    skel_result = optimize_skel(
        target_joints, joint_names, device, args.num_iters,
        virtual_acromial_weight=0.0,       # 0 = acromial→humerus 직접 매핑 (20 joints)
        shoulder_width_weight=1.0,         # Mild shoulder width constraint
        use_beta_init=True,                # Initialize beta from AddB proportions
        use_dynamic_virtual_acromial=False, # Disabled when virtual_acromial_weight=0
        gender=gender,
        subject_info=subject_info,
        body_proportions=body_proportions
    )

    # Save results
    print("\n=== Saving Results ===")

    # =========================================================================
    # Save AddB (original data)
    # =========================================================================
    np.save(os.path.join(addb_dir, 'joints.npy'), target_joints)
    np.save(os.path.join(addb_dir, 'joint_names.npy'), np.array(joint_names))
    np.save(os.path.join(addb_dir, 'joint_parents.npy'), np.array(addb_parents))

    addb_count = 0
    for t in range(0, len(target_joints), args.save_every):
        joints_t = target_joints[t]

        # Joint spheres
        verts, faces = create_joint_spheres(joints_t, radius=0.02)
        if len(verts) > 0:
            save_obj(verts, faces, os.path.join(addb_dir, f'joints_frame_{t:04d}.obj'))

        # Skeleton bones
        verts, faces = create_skeleton_bones(joints_t, addb_parents, radius=0.01)
        if len(verts) > 0:
            save_obj(verts, faces, os.path.join(addb_dir, f'skeleton_frame_{t:04d}.obj'))

        addb_count += 1
    print(f"  Saved {addb_count} AddB joint/skeleton OBJ files")

    # =========================================================================
    # Save SMPL
    # =========================================================================
    np.savez(os.path.join(smpl_dir, 'smpl_params.npz'),
             betas=smpl_result['betas'],
             poses=smpl_result['poses'],
             trans=smpl_result['trans'])
    np.save(os.path.join(smpl_dir, 'joints.npy'), smpl_result['joints'])

    smpl_count = 0
    for t in range(0, len(smpl_result['vertices']), args.save_every):
        # Skin mesh
        if smpl_result['faces'] is not None:
            save_obj(smpl_result['vertices'][t], smpl_result['faces'],
                     os.path.join(smpl_dir, f'mesh_frame_{t:04d}.obj'))

        # Joint spheres
        joints_t = smpl_result['joints'][t]
        verts, faces = create_joint_spheres(joints_t, radius=0.02)
        if len(verts) > 0:
            save_obj(verts, faces, os.path.join(smpl_dir, f'joints_frame_{t:04d}.obj'))

        # Skeleton bones
        verts, faces = create_skeleton_bones(joints_t, SMPL_PARENTS, radius=0.01)
        if len(verts) > 0:
            save_obj(verts, faces, os.path.join(smpl_dir, f'skeleton_frame_{t:04d}.obj'))

        smpl_count += 1
    print(f"  Saved {smpl_count} SMPL mesh/joint/skeleton OBJ files")

    # =========================================================================
    # Save SKEL
    # =========================================================================
    np.savez(os.path.join(skel_dir, 'skel_params.npz'),
             betas=skel_result['betas'],
             poses=skel_result['poses'],
             trans=skel_result['trans'])
    np.save(os.path.join(skel_dir, 'joints.npy'), skel_result['joints'])
    np.save(os.path.join(skel_dir, 'joint_parents.npy'), np.array(skel_result['parents']))

    # Use actual parent hierarchy from model (not hardcoded)
    skel_parents = skel_result['parents']
    print(f"  Using SKEL parents: {skel_parents}")

    skel_count = 0
    for t in range(0, len(skel_result['vertices']), args.save_every):
        # Skin mesh
        if skel_result['faces'] is not None:
            save_obj(skel_result['vertices'][t], skel_result['faces'],
                     os.path.join(skel_dir, f'mesh_frame_{t:04d}.obj'))

        # Skeleton mesh (SKEL's internal skeleton)
        if skel_result.get('skel_vertices') is not None and skel_result.get('skel_faces') is not None:
            save_obj(skel_result['skel_vertices'][t], skel_result['skel_faces'],
                     os.path.join(skel_dir, f'skeleton_frame_{t:04d}.obj'))

        # Joint spheres
        joints_t = skel_result['joints'][t]
        verts, faces = create_joint_spheres(joints_t, radius=0.02)
        if len(verts) > 0:
            save_obj(verts, faces, os.path.join(skel_dir, f'joints_frame_{t:04d}.obj'))

        # Joint bones (line connections) - using actual parent hierarchy from model
        verts, faces = create_skeleton_bones(joints_t, skel_parents, radius=0.008)
        if len(verts) > 0:
            save_obj(verts, faces, os.path.join(skel_dir, f'bones_frame_{t:04d}.obj'))

        skel_count += 1
    print(f"  Saved {skel_count} SKEL mesh/skeleton/joint/bones OBJ files")

    # =========================================================================
    # Compute and save comparison metrics
    # =========================================================================
    # For SKEL, exclude acromial_r, acromial_l from MPJPE (scapula mapping is kept but
    # position differs significantly from AddB acromial)
    skel_mpjpe_exclude_joints = []
    for exclude_name in ['acromial_r', 'acromial_l']:
        if exclude_name in joint_names:
            skel_mpjpe_exclude_joints.append(joint_names.index(exclude_name))

    smpl_mpjpe = compute_mpjpe(smpl_result['joints'], target_joints,
                               smpl_result['model_indices'], smpl_result['addb_indices'])
    skel_mpjpe = compute_mpjpe(skel_result['joints'], target_joints,
                               skel_result['model_indices'], skel_result['addb_indices'],
                               exclude_target_indices=skel_mpjpe_exclude_joints)

    metrics = {
        'input_file': args.b3d,
        'num_frames': len(target_joints),
        'num_iters': args.num_iters,
        'addb_joint_names': joint_names,
        'addb_joint_parents': addb_parents,
        'smpl': {
            'mpjpe_mm': smpl_mpjpe,
            'num_mapped_joints': len(smpl_result['addb_indices']),
            'joint_parents': SMPL_PARENTS,
        },
        'skel': {
            'mpjpe_mm': skel_mpjpe,
            'num_mapped_joints': len(skel_result['addb_indices']),
            'joint_parents': skel_result['parents'],
        },
    }

    with open(os.path.join(args.out_dir, 'comparison_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== Comparison Results ===")
    print(f"  SMPL MPJPE: {smpl_mpjpe:.1f} mm ({len(smpl_result['addb_indices'])} joints)")
    print(f"  SKEL MPJPE: {skel_mpjpe:.1f} mm ({len(skel_result['addb_indices'])} joints)")
    print(f"\nOutput saved to: {args.out_dir}")
    print(f"\nOutput structure:")
    print(f"  {args.out_dir}/")
    print(f"  ├── addb/           # Original AddBiomechanics data")
    print(f"  │   ├── frame_XXXX_joints.obj    # joint positions (spheres)")
    print(f"  │   └── frame_XXXX_skeleton.obj  # joint connections (bones)")
    print(f"  ├── smpl/           # SMPL optimization results")
    print(f"  │   ├── frame_XXXX_mesh.obj      # skin mesh")
    print(f"  │   ├── frame_XXXX_joints.obj    # joint positions (spheres)")
    print(f"  │   └── frame_XXXX_skeleton.obj  # joint connections (bones)")
    print(f"  ├── skel/           # SKEL optimization results")
    print(f"  │   ├── frame_XXXX_mesh.obj      # skin mesh")
    print(f"  │   ├── frame_XXXX_skeleton.obj  # SKEL skeleton mesh (bones)")
    print(f"  │   ├── frame_XXXX_joints.obj    # joint positions (spheres)")
    print(f"  │   └── frame_XXXX_bones.obj     # joint connections (lines)")
    print(f"  └── comparison_metrics.json")


if __name__ == '__main__':
    main()
