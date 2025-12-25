#!/usr/bin/env python3
"""
Visualize SKEL with Optimized Beta Parameters

Purpose: Generate OBJ files from optimized beta for comparison
- Uses optimized beta, pose, and translation from existing results
- Generates OBJ files for visual comparison with default beta

Usage:
    python visualize_optimized_beta.py \
        --skel_params /path/to/skel_params.npz \
        --out_dir /path/to/output \
        --gender male \
        --num_frames 5
"""

import os
import sys
import argparse
from typing import Tuple

import numpy as np
import torch

# Add SKEL to path
sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/SKEL')
sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

from models.skel_model import SKELModelWrapper, SKEL_NUM_BETAS


# ---------------------------------------------------------------------------
# Visualization Helper Functions
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

    for j_idx in range(len(joints)):
        verts, faces = create_sphere(joints[j_idx], radius=radius)
        all_verts.append(verts)
        all_faces.append(faces + vert_offset)
        vert_offset += len(verts)

    if len(all_verts) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    return np.concatenate(all_verts, axis=0), np.concatenate(all_faces, axis=0)


def create_skeleton_bones(joints: np.ndarray, parents: np.ndarray, radius: float = 0.008) -> Tuple[np.ndarray, np.ndarray]:
    """Create cylinder bones connecting joints via parent hierarchy"""
    all_verts = []
    all_faces = []
    vert_offset = 0

    for j_idx in range(len(joints)):
        parent_idx = parents[j_idx]
        if parent_idx < 0:
            continue

        start = joints[parent_idx]
        end = joints[j_idx]
        verts, faces = create_cylinder(start, end, radius=radius)

        if len(verts) > 0:
            all_verts.append(verts)
            all_faces.append(faces + vert_offset)
            vert_offset += len(verts)

    if len(all_verts) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    return np.concatenate(all_verts, axis=0), np.concatenate(all_faces, axis=0)


# ---------------------------------------------------------------------------
# Main Visualization Function
# ---------------------------------------------------------------------------

def visualize_with_optimized_beta(
    skel_params_path: str,
    out_dir: str,
    gender: str = 'male',
    num_frames: int = None,
    device: torch.device = torch.device('cpu')
):
    """
    Generate visualization with optimized beta

    Args:
        skel_params_path: Path to skel_params.npz
        out_dir: Output directory for OBJ files
        gender: 'male' or 'female'
        num_frames: Number of frames to process (None = all)
        device: torch device
    """

    print("=" * 70)
    print("SKEL Visualization with Optimized Beta")
    print("=" * 70)

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Load optimized parameters
    print(f"\nLoading optimized parameters from: {skel_params_path}")
    data = np.load(skel_params_path)

    optimized_betas = data['betas']  # [10]
    optimized_poses = data['poses']  # [T, 46]
    optimized_trans = data['trans']  # [T, 3]

    T = len(optimized_poses)
    if num_frames is not None:
        T = min(T, num_frames)

    print(f"  Optimized betas: {optimized_betas.shape} = {optimized_betas}")
    print(f"  Optimized poses: {optimized_poses.shape}")
    print(f"  Optimized trans: {optimized_trans.shape}")
    print(f"  Processing {T} frames")

    # Initialize SKEL model
    print(f"\nInitializing SKEL model ({gender})...")
    skel_model_path = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'
    skel = SKELModelWrapper(
        model_path=skel_model_path,
        gender=gender,
        device=device,
        add_acromial_joints=False  # Use standard 24 joints
    )

    # Use optimized beta
    optimized_betas_tensor = torch.from_numpy(optimized_betas).float().to(device)
    print(f"\nUsing optimized beta: {optimized_betas_tensor.cpu().numpy()}")

    # Convert to torch tensors
    poses_tensor = torch.from_numpy(optimized_poses[:T]).float().to(device)
    trans_tensor = torch.from_numpy(optimized_trans[:T]).float().to(device)

    # Forward pass
    print(f"\nGenerating meshes with optimized beta...")
    vertices, joints = skel.forward(
        betas=optimized_betas_tensor.unsqueeze(0).expand(T, -1),  # [T, 10]
        poses=poses_tensor,  # [T, 46]
        trans=trans_tensor   # [T, 3]
    )

    # Convert to numpy
    vertices_np = vertices.detach().cpu().numpy()  # [T, 6890, 3]
    joints_np = joints.detach().cpu().numpy()      # [T, 24, 3]
    faces = skel.faces

    # Get joint parents for skeleton visualization
    joint_parents = np.array([
        -1, 0, 1, 2, 3, 4,  # pelvis, femur_r, tibia_r, talus_r, calcn_r, toes_r
        0, 6, 7, 8, 9,      # femur_l, tibia_l, talus_l, calcn_l, toes_l
        0, 11, 12,          # lumbar_body, thorax, head
        12, 14, 15, 16, 17, # scapula_r, humerus_r, ulna_r, radius_r, hand_r
        12, 19, 20, 21, 22  # scapula_l, humerus_l, ulna_l, radius_l, hand_l
    ])

    print(f"\nSaving OBJ files to: {out_dir}")

    # Save results for each frame
    for t in range(T):
        # Skin mesh
        if faces is not None:
            obj_path = os.path.join(out_dir, f'optimized_beta_mesh_frame_{t:04d}.obj')
            save_obj(vertices_np[t], faces, obj_path)

        # Joint spheres
        joints_verts, joints_faces = create_joint_spheres(joints_np[t], radius=0.02)
        if len(joints_verts) > 0:
            obj_path = os.path.join(out_dir, f'optimized_beta_joints_frame_{t:04d}.obj')
            save_obj(joints_verts, joints_faces, obj_path)

        # Skeleton bones
        bones_verts, bones_faces = create_skeleton_bones(joints_np[t], joint_parents, radius=0.008)
        if len(bones_verts) > 0:
            obj_path = os.path.join(out_dir, f'optimized_beta_bones_frame_{t:04d}.obj')
            save_obj(bones_verts, bones_faces, obj_path)

        if (t + 1) % 10 == 0 or t == T - 1:
            print(f"  Processed frame {t+1}/{T}")

    # Save summary
    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("SKEL Visualization with Optimized Beta\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Input parameters: {skel_params_path}\n")
        f.write(f"Gender: {gender}\n")
        f.write(f"Num frames: {T}\n\n")
        f.write(f"Optimized beta: {optimized_betas}\n\n")
        f.write("Purpose:\n")
        f.write("  Reference visualization with optimized beta.\n")
        f.write("  Compare with default_beta version to isolate beta effects.\n")

    print(f"\n✓ Done! Summary saved to: {summary_path}")
    print(f"✓ Generated {T} frames with optimized beta")
    print(f"\nComparison directories:")
    print(f"  Optimized beta: {out_dir}")
    print(f"  Default beta:   debug_default_beta_Subject1/")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Visualize SKEL with optimized beta')
    parser.add_argument('--skel_params', type=str, required=True,
                        help='Path to skel_params.npz')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for OBJ files')
    parser.add_argument('--gender', type=str, default='male', choices=['male', 'female'],
                        help='Gender (male or female)')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Number of frames to process (default: all)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use')

    args = parser.parse_args()

    device = torch.device(args.device)

    visualize_with_optimized_beta(
        skel_params_path=args.skel_params,
        out_dir=args.out_dir,
        gender=args.gender,
        num_frames=args.num_frames,
        device=device
    )


if __name__ == '__main__':
    main()
