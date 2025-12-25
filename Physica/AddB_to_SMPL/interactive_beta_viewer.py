#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Beta Viewer: Manually adjust beta parameters and generate mesh

This script allows you to manually specify beta values and generate mesh output.

Usage:
    # Single beta adjustment
    python interactive_beta_viewer.py --model smpl --betas 2.0,0,0,0,0,0,0,0,0,0

    # Multiple betas
    python interactive_beta_viewer.py --model skel --betas 1.5,-0.5,0.3,0,0,0,0,0,0,0

    # With custom output name
    python interactive_beta_viewer.py --model smpl --betas 2,1,0,0,0,0,0,0,0,0 --name tall_wide

    # With pose from saved subject
    python interactive_beta_viewer.py --model smpl --betas 0,0,0,0,0,0,0,0,0,0 --subject Subject1 --frame 0
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

OUTPUT_BASE_DIR = '/egr/research-zijunlab/kwonjoon/03_Output/output_12_15/interactive_beta'
INPUT_BASE_DIR = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare'

SMPL_MODEL_DIR = '/egr/research-zijunlab/kwonjoon/01_Code/Physica/models'
SKEL_MODEL_DIR = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'

# SKEL parent indices
SKEL_PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 12, 14, 15, 16, 17, 12, 19, 20, 21, 22]


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
    return {
        'height': (joints[15, 1] - min(joints[7, 1], joints[8, 1])) * 1000,
        'shoulder_width': np.linalg.norm(joints[16] - joints[17]) * 1000,
        'hip_width': np.linalg.norm(joints[1] - joints[2]) * 1000,
    }


def measure_skel(joints: np.ndarray) -> dict:
    """Measure key body dimensions from SKEL joints"""
    return {
        'height': (joints[13, 1] - min(joints[3, 1], joints[8, 1])) * 1000,
        'shoulder_width': np.linalg.norm(joints[15] - joints[20]) * 1000,
        'hip_width': np.linalg.norm(joints[1] - joints[6]) * 1000,
    }


# ---------------------------------------------------------------------------
# Main Functions
# ---------------------------------------------------------------------------

def generate_smpl(betas: List[float], pose: torch.Tensor, trans: torch.Tensor,
                  output_name: str, device: torch.device):
    """Generate SMPL mesh with specified betas"""

    out_dir = os.path.join(OUTPUT_BASE_DIR, 'smpl')
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    smpl = SMPLModel(gender='male', device=device)

    # Prepare betas
    betas_tensor = torch.tensor(betas, dtype=torch.float32, device=device)

    # Forward pass
    verts, joints = smpl.forward(betas_tensor, pose, trans)
    verts = verts.cpu().numpy()
    joints = joints.cpu().numpy()

    # Save mesh
    save_obj(verts, smpl.faces.cpu().numpy(),
             os.path.join(out_dir, f'{output_name}_mesh.obj'))

    # Save joints
    joint_verts, joint_faces = create_joint_spheres(joints, radius=0.02)
    save_obj(joint_verts, joint_faces,
             os.path.join(out_dir, f'{output_name}_joints.obj'))

    # Save skeleton
    skel_verts, skel_faces = create_skeleton_bones(joints, list(SMPL_PARENTS.numpy()), radius=0.01)
    if len(skel_verts) > 0:
        save_obj(skel_verts, skel_faces,
                 os.path.join(out_dir, f'{output_name}_skeleton.obj'))

    # Measure
    meas = measure_smpl(joints)

    print(f"\nSMPL Output: {output_name}")
    print(f"  Betas: {betas}")
    print(f"  Height: {meas['height']:.1f} mm")
    print(f"  Shoulder width: {meas['shoulder_width']:.1f} mm")
    print(f"  Hip width: {meas['hip_width']:.1f} mm")
    print(f"  Output: {out_dir}/{output_name}_*.obj")


def generate_skel(betas: List[float], pose: torch.Tensor, trans: torch.Tensor,
                  output_name: str, device: torch.device):
    """Generate SKEL mesh with specified betas"""

    out_dir = os.path.join(OUTPUT_BASE_DIR, 'skel')
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    skel = SKELModelWrapper(SKEL_MODEL_DIR, gender='male', device=device)

    # Prepare betas
    betas_tensor = torch.tensor([betas], dtype=torch.float32, device=device)

    # Forward pass
    verts, joints = skel.forward(betas_tensor, pose, trans)
    verts = verts[0].cpu().numpy()
    joints = joints[0].cpu().numpy()

    # Save mesh
    faces = skel.model.skin_f.cpu().numpy()
    save_obj(verts, faces,
             os.path.join(out_dir, f'{output_name}_mesh.obj'))

    # Save joints
    joint_verts, joint_faces = create_joint_spheres(joints, radius=0.02)
    save_obj(joint_verts, joint_faces,
             os.path.join(out_dir, f'{output_name}_joints.obj'))

    # Save skeleton
    skel_verts, skel_faces = create_skeleton_bones(joints, SKEL_PARENTS, radius=0.01)
    if len(skel_verts) > 0:
        save_obj(skel_verts, skel_faces,
                 os.path.join(out_dir, f'{output_name}_skeleton.obj'))

    # Measure
    meas = measure_skel(joints)

    print(f"\nSKEL Output: {output_name}")
    print(f"  Betas: {betas}")
    print(f"  Height: {meas['height']:.1f} mm")
    print(f"  Shoulder width: {meas['shoulder_width']:.1f} mm")
    print(f"  Hip width: {meas['hip_width']:.1f} mm")
    print(f"  Output: {out_dir}/{output_name}_*.obj")


def main():
    parser = argparse.ArgumentParser(description='Interactive Beta Viewer')
    parser.add_argument('--model', type=str, required=True, choices=['smpl', 'skel'],
                        help='Model type (smpl or skel)')
    parser.add_argument('--betas', type=str, required=True,
                        help='Comma-separated 10 beta values (e.g., 2.0,0,0,0,0,0,0,0,0,0)')
    parser.add_argument('--name', type=str, default=None,
                        help='Output name (default: auto-generated from betas)')
    parser.add_argument('--subject', type=str, default=None,
                        help='Load pose from saved subject (e.g., Subject1)')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index when loading from subject')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parse betas
    betas = [float(v) for v in args.betas.split(',')]
    if len(betas) != 10:
        print(f"Error: Need exactly 10 beta values, got {len(betas)}")
        return

    # Generate output name
    if args.name:
        output_name = args.name
    else:
        # Auto-generate name from non-zero betas
        non_zero = [(i, v) for i, v in enumerate(betas) if abs(v) > 0.01]
        if non_zero:
            parts = [f"b{i}_{v:.1f}".replace('.', 'p').replace('-', 'n') for i, v in non_zero]
            output_name = "_".join(parts)
        else:
            output_name = "zero_betas"

    # Create output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Load pose from subject or use T-pose
    if args.subject:
        subject_dir = os.path.join(INPUT_BASE_DIR, f'final_{args.subject}')
        if args.model == 'smpl':
            params_path = os.path.join(subject_dir, 'smpl', 'smpl_params.npz')
            params = np.load(params_path)
            pose = torch.tensor(params['poses'][args.frame], dtype=torch.float32, device=device)
            trans = torch.tensor(params['trans'][args.frame], dtype=torch.float32, device=device)
            print(f"Loaded pose from {args.subject} frame {args.frame} (SMPL)")
        else:
            params_path = os.path.join(subject_dir, 'skel', 'skel_params.npz')
            params = np.load(params_path)
            pose = torch.tensor(params['poses'][args.frame:args.frame+1], dtype=torch.float32, device=device)
            trans = torch.tensor(params['trans'][args.frame:args.frame+1], dtype=torch.float32, device=device)
            print(f"Loaded pose from {args.subject} frame {args.frame} (SKEL)")
    else:
        # T-pose
        if args.model == 'smpl':
            pose = torch.zeros(24, 3, dtype=torch.float32, device=device)
            trans = torch.zeros(3, dtype=torch.float32, device=device)
        else:
            pose = torch.zeros(1, 46, dtype=torch.float32, device=device)
            trans = torch.zeros(1, 3, dtype=torch.float32, device=device)
        print("Using T-pose")

    # Generate
    if args.model == 'smpl':
        generate_smpl(betas, pose, trans, output_name, device)
    else:
        generate_skel(betas, pose, trans, output_name, device)

    print(f"\nDone!")


if __name__ == '__main__':
    main()
