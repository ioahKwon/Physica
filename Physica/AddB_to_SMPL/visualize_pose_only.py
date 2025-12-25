#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pose-only Visualization: Standard betas (zeros) + estimated pose

This script loads saved pose parameters and generates mesh with zero betas
to isolate pose effects from shape effects.

Purpose:
- If shoulder looks wrong with zero betas -> pose estimation problem OR visualization code problem
- If shoulder looks correct with zero betas -> betas/shape estimation problem

Usage:
    python visualize_pose_only.py --subject Subject1
    python visualize_pose_only.py --all  # Run for all 4 subjects
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

SUBJECTS = ['Subject1', 'Subject11', 'Subject13', 'Subject17']

INPUT_BASE_DIR = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare'
OUTPUT_BASE_DIR = '/egr/research-zijunlab/kwonjoon/03_Output/output_12_15'

SMPL_MODEL_DIR = '/egr/research-zijunlab/kwonjoon/01_Code/Physica/models'
SKEL_MODEL_DIR = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'

# SKEL parent indices (from SKEL kinematic tree)
SKEL_PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 12, 14, 15, 16, 17, 12, 19, 20, 21, 22]

# AddB parent indices (20 joints)
ADDB_PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 11, 16, 17, 18]


# ---------------------------------------------------------------------------
# OBJ Utilities (copied from compare_smpl_skel.py for standalone use)
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
# AddB GT Visualization
# ---------------------------------------------------------------------------

def visualize_addb_gt(subject: str, output_dir: str):
    """
    Generate AddB GT visualization with skeleton lines

    Args:
        subject: Subject name (e.g., 'Subject1')
        output_dir: Output directory for this subject
    """
    print(f"\n  === AddB GT ===" )

    # Input path
    input_dir = os.path.join(INPUT_BASE_DIR, f'final_{subject}', 'addb')
    joints_path = os.path.join(input_dir, 'joints.npy')

    if not os.path.exists(joints_path):
        print(f"    [WARNING] AddB joints not found: {joints_path}")
        return

    # Output directory
    addb_out_dir = os.path.join(output_dir, 'addb')
    os.makedirs(addb_out_dir, exist_ok=True)

    # Load AddB joints
    joints = np.load(joints_path)  # [T, 20, 3]
    T = joints.shape[0]
    print(f"    Processing {T} frames...")

    # Generate OBJ files for each frame
    for t in range(T):
        joints_t = joints[t]  # [20, 3]

        # Save joint spheres
        joint_verts, joint_faces = create_joint_spheres(joints_t, radius=0.02)
        save_obj(joint_verts, joint_faces,
                 os.path.join(addb_out_dir, f'joints_frame_{t:04d}.obj'))

        # Save skeleton bones
        skel_verts, skel_faces = create_skeleton_bones(joints_t, ADDB_PARENTS, radius=0.01)
        if len(skel_verts) > 0:
            save_obj(skel_verts, skel_faces,
                     os.path.join(addb_out_dir, f'skeleton_frame_{t:04d}.obj'))

    # Compute shoulder width (acromial_r = 12, acromial_l = 16)
    shoulder_width = np.linalg.norm(joints[0, 12] - joints[0, 16]) * 1000
    print(f"    AddB GT shoulder width (acromial): {shoulder_width:.1f} mm")
    print(f"    Saved {T} AddB GT frames")


# ---------------------------------------------------------------------------
# Main Visualization Functions
# ---------------------------------------------------------------------------

def visualize_subject(subject: str, device: torch.device):
    """
    Generate pose-only visualization for a single subject

    Args:
        subject: Subject name (e.g., 'Subject1')
        device: torch device
    """
    print(f"\n{'='*60}")
    print(f"Processing {subject}")
    print(f"{'='*60}")

    # Input paths
    input_dir = os.path.join(INPUT_BASE_DIR, f'final_{subject}')
    smpl_params_path = os.path.join(input_dir, 'smpl', 'smpl_params.npz')
    skel_params_path = os.path.join(input_dir, 'skel', 'skel_params.npz')

    # Check if input files exist
    if not os.path.exists(smpl_params_path):
        print(f"  [ERROR] SMPL params not found: {smpl_params_path}")
        return
    if not os.path.exists(skel_params_path):
        print(f"  [ERROR] SKEL params not found: {skel_params_path}")
        return

    # Output directory
    output_dir = os.path.join(OUTPUT_BASE_DIR, 'pose_only_visualization', subject)
    smpl_out_dir = os.path.join(output_dir, 'smpl')
    skel_out_dir = os.path.join(output_dir, 'skel')
    os.makedirs(smpl_out_dir, exist_ok=True)
    os.makedirs(skel_out_dir, exist_ok=True)

    # Load saved parameters
    print(f"\n  Loading saved parameters...")
    smpl_params = np.load(smpl_params_path)
    skel_params = np.load(skel_params_path)

    print(f"    SMPL: betas={smpl_params['betas'].shape}, poses={smpl_params['poses'].shape}, trans={smpl_params['trans'].shape}")
    print(f"    SKEL: betas={skel_params['betas'].shape}, poses={skel_params['poses'].shape}, trans={skel_params['trans'].shape}")

    # Load models
    print(f"\n  Loading models...")
    smpl = SMPLModel(gender='male', device=device)
    skel = SKELModelWrapper(SKEL_MODEL_DIR, gender='male', device=device)

    # =========================================================================
    # SMPL: Zero betas + estimated pose
    # =========================================================================
    print(f"\n  === SMPL: Zero betas + estimated pose ===")

    smpl_poses = torch.tensor(smpl_params['poses'], dtype=torch.float32, device=device)  # [T, 24, 3]
    smpl_trans = torch.tensor(smpl_params['trans'], dtype=torch.float32, device=device)  # [T, 3]
    smpl_zero_betas = torch.zeros(10, dtype=torch.float32, device=device)

    # Also save with original betas for comparison
    smpl_orig_betas = torch.tensor(smpl_params['betas'], dtype=torch.float32, device=device)

    T = smpl_poses.shape[0]
    print(f"    Processing {T} frames...")

    for t in range(T):
        # Forward pass with zero betas
        pose_t = smpl_poses[t]  # [24, 3]
        trans_t = smpl_trans[t]  # [3]

        verts_zero, joints_zero = smpl.forward(smpl_zero_betas, pose_t, trans_t)
        verts_zero = verts_zero.cpu().numpy()
        joints_zero = joints_zero.cpu().numpy()

        # Save mesh
        save_obj(verts_zero, smpl.faces.cpu().numpy(),
                 os.path.join(smpl_out_dir, f'mesh_frame_{t:04d}.obj'))

        # Save joint spheres
        joint_verts, joint_faces = create_joint_spheres(joints_zero, radius=0.02)
        save_obj(joint_verts, joint_faces,
                 os.path.join(smpl_out_dir, f'joints_frame_{t:04d}.obj'))

        # Save skeleton bones
        skel_verts, skel_faces = create_skeleton_bones(joints_zero, list(SMPL_PARENTS.numpy()), radius=0.01)
        if len(skel_verts) > 0:
            save_obj(skel_verts, skel_faces,
                     os.path.join(smpl_out_dir, f'skeleton_frame_{t:04d}.obj'))

    print(f"    Saved {T} SMPL frames (zero betas)")

    # Compute and print shoulder width
    verts_zero, joints_zero = smpl.forward(smpl_zero_betas, smpl_poses[0], smpl_trans[0])
    verts_orig, joints_orig = smpl.forward(smpl_orig_betas, smpl_poses[0], smpl_trans[0])

    def get_shoulder_width(joints):
        # SMPL: left_shoulder=16, right_shoulder=17
        return np.linalg.norm(joints[16] - joints[17]) * 1000

    print(f"    SMPL joint shoulder width:")
    print(f"      Zero betas: {get_shoulder_width(joints_zero.cpu().numpy()):.1f} mm")
    print(f"      Orig betas: {get_shoulder_width(joints_orig.cpu().numpy()):.1f} mm")

    # =========================================================================
    # SKEL: Zero betas + estimated pose
    # =========================================================================
    print(f"\n  === SKEL: Zero betas + estimated pose ===")

    skel_poses = torch.tensor(skel_params['poses'], dtype=torch.float32, device=device)  # [T, 46]
    skel_trans = torch.tensor(skel_params['trans'], dtype=torch.float32, device=device)  # [T, 3]
    skel_zero_betas = torch.zeros(10, dtype=torch.float32, device=device)

    skel_orig_betas = torch.tensor(skel_params['betas'], dtype=torch.float32, device=device)

    T = skel_poses.shape[0]
    print(f"    Processing {T} frames...")

    for t in range(T):
        pose_t = skel_poses[t]  # [46]
        trans_t = skel_trans[t]  # [3]

        verts_zero, joints_zero = skel.forward(skel_zero_betas.unsqueeze(0), pose_t.unsqueeze(0), trans_t.unsqueeze(0))
        verts_zero = verts_zero[0].cpu().numpy()
        joints_zero = joints_zero[0].cpu().numpy()

        # Save mesh
        faces = skel.model.skin_f.cpu().numpy()
        save_obj(verts_zero, faces,
                 os.path.join(skel_out_dir, f'mesh_frame_{t:04d}.obj'))

        # Save joint spheres
        joint_verts, joint_faces = create_joint_spheres(joints_zero, radius=0.02)
        save_obj(joint_verts, joint_faces,
                 os.path.join(skel_out_dir, f'joints_frame_{t:04d}.obj'))

        # Save skeleton bones
        skel_bone_verts, skel_bone_faces = create_skeleton_bones(joints_zero, SKEL_PARENTS, radius=0.01)
        if len(skel_bone_verts) > 0:
            save_obj(skel_bone_verts, skel_bone_faces,
                     os.path.join(skel_out_dir, f'skeleton_frame_{t:04d}.obj'))

    print(f"    Saved {T} SKEL frames (zero betas)")

    # Compute and print shoulder width
    verts_zero, joints_zero = skel.forward(skel_zero_betas.unsqueeze(0), skel_poses[0:1], skel_trans[0:1])
    verts_orig, joints_orig = skel.forward(skel_orig_betas.unsqueeze(0), skel_poses[0:1], skel_trans[0:1])

    def get_skel_shoulder_width(joints):
        # SKEL: humerus_l=20, humerus_r=15
        return np.linalg.norm(joints[0, 20] - joints[0, 15]) * 1000

    print(f"    SKEL joint shoulder width (humerus):")
    print(f"      Zero betas: {get_skel_shoulder_width(joints_zero.cpu().numpy()):.1f} mm")
    print(f"      Orig betas: {get_skel_shoulder_width(joints_orig.cpu().numpy()):.1f} mm")

    # =========================================================================
    # AddB GT: Ground truth joints with skeleton lines
    # =========================================================================
    visualize_addb_gt(subject, output_dir)

    print(f"\n  Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Pose-only Visualization')
    parser.add_argument('--subject', type=str, help='Subject name (e.g., Subject1)')
    parser.add_argument('--all', action='store_true', help='Run for all 4 subjects')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    if args.all:
        for subject in SUBJECTS:
            visualize_subject(subject, device)
    elif args.subject:
        if args.subject not in SUBJECTS:
            print(f"Warning: {args.subject} not in standard subjects list {SUBJECTS}")
        visualize_subject(args.subject, device)
    else:
        print("Please specify --subject <name> or --all")
        print(f"Available subjects: {SUBJECTS}")
        return

    print(f"\n{'='*60}")
    print(f"All done! Output at: {OUTPUT_BASE_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
