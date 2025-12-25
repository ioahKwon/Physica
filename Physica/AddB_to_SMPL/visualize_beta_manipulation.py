#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beta Manipulation Visualization: Vary each beta parameter to observe shape changes

This script generates meshes with different beta values to verify:
1. If beta parameters behave as expected (taller, wider, etc.)
2. If visualization code is reliable

Usage:
    python visualize_beta_manipulation.py --model smpl --beta_idx 0 --values -3,-2,-1,0,1,2,3
    python visualize_beta_manipulation.py --model skel --beta_idx 0 --values -3,-2,-1,0,1,2,3
    python visualize_beta_manipulation.py --all  # Generate all beta variations for both models
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.smpl_model import SMPLModel, SMPL_PARENTS
from models.skel_model import SKELModelWrapper


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_BASE_DIR = '/egr/research-zijunlab/kwonjoon/03_Output/output_12_15/beta_manipulation'

SMPL_MODEL_DIR = '/egr/research-zijunlab/kwonjoon/01_Code/Physica/models'
SKEL_MODEL_DIR = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'

# SKEL parent indices (from SKEL kinematic tree)
SKEL_PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 12, 14, 15, 16, 17, 12, 19, 20, 21, 22]

# Beta meanings (approximate, based on PCA components)
BETA_MEANINGS = {
    0: "Height/Overall size",
    1: "Weight/Body mass",
    2: "Shoulder width / Hip ratio",
    3: "Arm length",
    4: "Leg length",
    5: "Torso length",
    6: "Chest size",
    7: "Waist size",
    8: "Hip width",
    9: "Head size",
}

# Default beta values to test
DEFAULT_VALUES = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# OBJ Utilities
# ---------------------------------------------------------------------------

def save_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    """Save mesh as OBJ file"""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def create_sphere(center: np.ndarray, radius: float = 0.02, n_segments: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Create a UV sphere mesh at the given center"""
    vertices = []
    faces = []

    for i in range(n_segments + 1):
        lat = np.pi * i / n_segments - np.pi / 2
        for j in range(n_segments):
            lon = 2 * np.pi * j / n_segments
            x = radius * np.cos(lat) * np.cos(lon) + center[0]
            y = radius * np.cos(lat) * np.sin(lon) + center[1]
            z = radius * np.sin(lat) + center[2]
            vertices.append([x, y, z])

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
    """Create a cylinder mesh between two points"""
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    direction = direction / length

    if abs(direction[0]) < 0.9:
        perp1 = np.cross(direction, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(direction, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    vertices = []
    faces = []

    for t, center in enumerate([start, end]):
        for i in range(n_segments):
            angle = 2 * np.pi * i / n_segments
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(center + offset)

    for i in range(n_segments):
        p1 = i
        p2 = (i + 1) % n_segments
        p3 = n_segments + (i + 1) % n_segments
        p4 = n_segments + i
        faces.append([p1, p2, p3])
        faces.append([p1, p3, p4])

    start_center_idx = len(vertices)
    vertices.append(start)
    for i in range(n_segments):
        faces.append([start_center_idx, (i + 1) % n_segments, i])

    end_center_idx = len(vertices)
    vertices.append(end)
    for i in range(n_segments):
        faces.append([end_center_idx, n_segments + i, n_segments + (i + 1) % n_segments])

    return np.array(vertices), np.array(faces)


def create_joint_spheres(joints: np.ndarray, radius: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """Create spheres at all joint positions"""
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
    """Create cylinder bones connecting joints based on parent hierarchy"""
    all_verts = []
    all_faces = []
    vert_offset = 0

    for i, parent in enumerate(parents):
        if parent >= 0:
            verts, faces = create_cylinder(joints[parent], joints[i], radius)
            if len(verts) > 0:
                all_verts.append(verts)
                all_faces.append(faces + vert_offset)
                vert_offset += len(verts)

    if len(all_verts) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    return np.vstack(all_verts), np.vstack(all_faces)


# ---------------------------------------------------------------------------
# Measurement Functions
# ---------------------------------------------------------------------------

def measure_smpl(joints: np.ndarray) -> dict:
    """Measure key body dimensions from SMPL joints"""
    # SMPL joint indices
    # 0: pelvis, 1: left_hip, 2: right_hip, 3: spine1
    # 12: neck, 15: head, 16: left_shoulder, 17: right_shoulder
    # 18: left_elbow, 19: right_elbow, 20: left_wrist, 21: right_wrist
    # 4: left_knee, 5: right_knee, 7: left_ankle, 8: right_ankle

    return {
        'height': (joints[15, 1] - min(joints[7, 1], joints[8, 1])) * 1000,  # head to ankle
        'shoulder_width': np.linalg.norm(joints[16] - joints[17]) * 1000,
        'hip_width': np.linalg.norm(joints[1] - joints[2]) * 1000,
        'arm_length': (np.linalg.norm(joints[16] - joints[18]) + np.linalg.norm(joints[18] - joints[20])) * 1000,
        'leg_length': (np.linalg.norm(joints[1] - joints[4]) + np.linalg.norm(joints[4] - joints[7])) * 1000,
    }


def measure_skel(joints: np.ndarray) -> dict:
    """Measure key body dimensions from SKEL joints"""
    # SKEL joint indices (24 joints)
    # 0: pelvis, 1: femur_r, 6: femur_l
    # 15: humerus_r, 20: humerus_l (shoulder joints)
    # 2: tibia_r, 7: tibia_l (knee)
    # 3: talus_r, 8: talus_l (ankle)
    # 12: head, 13: skull

    return {
        'height': (joints[13, 1] - min(joints[3, 1], joints[8, 1])) * 1000,  # skull to ankle
        'shoulder_width': np.linalg.norm(joints[15] - joints[20]) * 1000,
        'hip_width': np.linalg.norm(joints[1] - joints[6]) * 1000,
        'arm_length': (np.linalg.norm(joints[15] - joints[16]) + np.linalg.norm(joints[16] - joints[17])) * 1000,
        'leg_length': (np.linalg.norm(joints[1] - joints[2]) + np.linalg.norm(joints[2] - joints[3])) * 1000,
    }


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------

def visualize_smpl_beta(beta_idx: int, values: List[float], device: torch.device):
    """
    Generate SMPL meshes varying a single beta parameter
    """
    print(f"\n{'='*60}")
    print(f"SMPL Beta[{beta_idx}] Manipulation: {BETA_MEANINGS.get(beta_idx, 'Unknown')}")
    print(f"{'='*60}")

    # Output directory
    out_dir = os.path.join(OUTPUT_BASE_DIR, 'smpl', f'beta_{beta_idx}')
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    smpl = SMPLModel(gender='male', device=device)

    # Zero pose (T-pose)
    pose = torch.zeros(24, 3, dtype=torch.float32, device=device)
    trans = torch.zeros(3, dtype=torch.float32, device=device)

    measurements = []

    for val in values:
        # Create beta with single non-zero value
        betas = torch.zeros(10, dtype=torch.float32, device=device)
        betas[beta_idx] = val

        # Forward pass
        verts, joints = smpl.forward(betas, pose, trans)
        verts = verts.cpu().numpy()
        joints = joints.cpu().numpy()

        # Save mesh
        val_str = f"{val:+.1f}".replace('.', 'p').replace('-', 'n').replace('+', 'p')
        save_obj(verts, smpl.faces.cpu().numpy(),
                 os.path.join(out_dir, f'mesh_beta{beta_idx}_{val_str}.obj'))

        # Save joints
        joint_verts, joint_faces = create_joint_spheres(joints, radius=0.02)
        save_obj(joint_verts, joint_faces,
                 os.path.join(out_dir, f'joints_beta{beta_idx}_{val_str}.obj'))

        # Save skeleton
        skel_verts, skel_faces = create_skeleton_bones(joints, list(SMPL_PARENTS.numpy()), radius=0.01)
        if len(skel_verts) > 0:
            save_obj(skel_verts, skel_faces,
                     os.path.join(out_dir, f'skeleton_beta{beta_idx}_{val_str}.obj'))

        # Measure
        meas = measure_smpl(joints)
        meas['beta_value'] = val
        measurements.append(meas)

        print(f"  beta[{beta_idx}]={val:+.1f}: height={meas['height']:.1f}mm, "
              f"shoulder={meas['shoulder_width']:.1f}mm, hip={meas['hip_width']:.1f}mm")

    # Save measurements
    np.save(os.path.join(out_dir, 'measurements.npy'), measurements)
    print(f"\n  Output saved to: {out_dir}")

    return measurements


def visualize_skel_beta(beta_idx: int, values: List[float], device: torch.device):
    """
    Generate SKEL meshes varying a single beta parameter
    """
    print(f"\n{'='*60}")
    print(f"SKEL Beta[{beta_idx}] Manipulation: {BETA_MEANINGS.get(beta_idx, 'Unknown')}")
    print(f"{'='*60}")

    # Output directory
    out_dir = os.path.join(OUTPUT_BASE_DIR, 'skel', f'beta_{beta_idx}')
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    skel = SKELModelWrapper(SKEL_MODEL_DIR, gender='male', device=device)

    # Zero pose
    pose = torch.zeros(1, 46, dtype=torch.float32, device=device)
    trans = torch.zeros(1, 3, dtype=torch.float32, device=device)

    measurements = []

    for val in values:
        # Create beta with single non-zero value
        betas = torch.zeros(1, 10, dtype=torch.float32, device=device)
        betas[0, beta_idx] = val

        # Forward pass
        verts, joints = skel.forward(betas, pose, trans)
        verts = verts[0].cpu().numpy()
        joints = joints[0].cpu().numpy()

        # Save mesh
        val_str = f"{val:+.1f}".replace('.', 'p').replace('-', 'n').replace('+', 'p')
        faces = skel.model.skin_f.cpu().numpy()
        save_obj(verts, faces,
                 os.path.join(out_dir, f'mesh_beta{beta_idx}_{val_str}.obj'))

        # Save joints
        joint_verts, joint_faces = create_joint_spheres(joints, radius=0.02)
        save_obj(joint_verts, joint_faces,
                 os.path.join(out_dir, f'joints_beta{beta_idx}_{val_str}.obj'))

        # Save skeleton
        skel_verts, skel_faces = create_skeleton_bones(joints, SKEL_PARENTS, radius=0.01)
        if len(skel_verts) > 0:
            save_obj(skel_verts, skel_faces,
                     os.path.join(out_dir, f'skeleton_beta{beta_idx}_{val_str}.obj'))

        # Measure
        meas = measure_skel(joints)
        meas['beta_value'] = val
        measurements.append(meas)

        print(f"  beta[{beta_idx}]={val:+.1f}: height={meas['height']:.1f}mm, "
              f"shoulder={meas['shoulder_width']:.1f}mm, hip={meas['hip_width']:.1f}mm")

    # Save measurements
    np.save(os.path.join(out_dir, 'measurements.npy'), measurements)
    print(f"\n  Output saved to: {out_dir}")

    return measurements


def generate_all_variations(device: torch.device):
    """Generate beta variations for all 10 betas, for both SMPL and SKEL"""
    print("\n" + "="*70)
    print("Generating ALL beta variations for SMPL and SKEL")
    print("="*70)

    values = DEFAULT_VALUES

    # Generate for all betas (0-9)
    for beta_idx in range(10):
        visualize_smpl_beta(beta_idx, values, device)
        visualize_skel_beta(beta_idx, values, device)

    print("\n" + "="*70)
    print(f"All done! Output at: {OUTPUT_BASE_DIR}")
    print("="*70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Beta Manipulation Visualization')
    parser.add_argument('--model', type=str, choices=['smpl', 'skel'], help='Model type')
    parser.add_argument('--beta_idx', type=int, help='Beta index to vary (0-9)')
    parser.add_argument('--values', type=str, default='-3,-2,-1,0,1,2,3',
                        help='Comma-separated beta values to test')
    parser.add_argument('--all', action='store_true', help='Generate all beta variations')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    if args.all:
        generate_all_variations(device)
    elif args.model and args.beta_idx is not None:
        values = [float(v) for v in args.values.split(',')]

        if args.model == 'smpl':
            visualize_smpl_beta(args.beta_idx, values, device)
        else:
            visualize_skel_beta(args.beta_idx, values, device)
    else:
        print("Please specify --all or (--model and --beta_idx)")
        print("Examples:")
        print("  python visualize_beta_manipulation.py --all")
        print("  python visualize_beta_manipulation.py --model smpl --beta_idx 0")
        print("  python visualize_beta_manipulation.py --model skel --beta_idx 2 --values -5,-2,0,2,5")


if __name__ == '__main__':
    main()
