#!/usr/bin/env python3
"""
Beta Parameter Sweep Visualization

Purpose: Test each beta parameter individually to see shoulder effects
- Sweep each beta[i] through values: -2, -1, 0, +1, +2
- Keep all other betas at 0
- Use fixed pose (optimized or T-pose)
- Generate OBJ files for visual inspection

If shoulders look wrong in ALL cases → visualization code problem
If shoulders look wrong only for specific beta → that beta is the issue

Usage:
    python beta_sweep_visualization.py \
        --model skel \
        --pose_source /path/to/skel_params.npz \
        --out_dir /path/to/output \
        --gender male \
        --frame 0
"""

import os
import sys
import argparse
from typing import Tuple

import numpy as np
import torch
import smplx

# Add paths
sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/SKEL')
sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

from models.skel_model import SKELModelWrapper, SKEL_NUM_BETAS
from models.smpl_model import SMPLModel, SMPL_NUM_BETAS


# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------

def save_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    """Save mesh as OBJ file"""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def create_sphere(center: np.ndarray, radius: float = 0.02, n_segments: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Create sphere"""
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


def create_joint_spheres(joints: np.ndarray, radius: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """Create joint spheres"""
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


# ---------------------------------------------------------------------------
# Beta Sweep
# ---------------------------------------------------------------------------

def beta_sweep_skel(
    pose_source: str,
    out_dir: str,
    gender: str,
    frame_idx: int,
    use_tpose: bool,
    device: torch.device
):
    """
    Sweep SKEL beta parameters

    Args:
        pose_source: Path to skel_params.npz (for pose)
        out_dir: Output directory
        gender: 'male' or 'female'
        frame_idx: Which frame to use
        use_tpose: If True, use T-pose instead of optimized pose
        device: torch device
    """

    print("=" * 70)
    print("SKEL Beta Parameter Sweep")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)

    # Load pose
    if use_tpose:
        print("\nUsing T-pose (pose = zeros)")
        pose = torch.zeros(46, device=device)
        trans = torch.zeros(3, device=device)
    else:
        print(f"\nLoading optimized pose from: {pose_source}")
        data = np.load(pose_source)
        pose = torch.from_numpy(data['poses'][frame_idx]).float().to(device)
        trans = torch.from_numpy(data['trans'][frame_idx]).float().to(device)

    # Initialize SKEL
    skel_path = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'
    skel = SKELModelWrapper(
        model_path=skel_path,
        gender=gender,
        device=device,
        add_acromial_joints=False
    )

    # Beta sweep values
    sweep_values = [-2.0, -1.0, 0.0, 1.0, 2.0]

    print(f"\nSweeping each beta through: {sweep_values}")
    print(f"Total visualizations: {10 * len(sweep_values) + 1} (10 betas + 1 baseline)")

    # Baseline: all zeros
    print("\n[Baseline] Beta = zeros(10)")
    beta_zeros = torch.zeros(10, device=device)
    verts, joints = skel.forward(beta_zeros.unsqueeze(0), pose.unsqueeze(0), trans.unsqueeze(0))

    obj_path = os.path.join(out_dir, 'skel_beta_baseline_mesh.obj')
    save_obj(verts[0].detach().cpu().numpy(), skel.faces, obj_path)

    joints_verts, joints_faces = create_joint_spheres(joints[0].detach().cpu().numpy(), radius=0.02)
    if len(joints_verts) > 0:
        obj_path = os.path.join(out_dir, 'skel_beta_baseline_joints.obj')
        save_obj(joints_verts, joints_faces, obj_path)

    # Sweep each beta
    for beta_idx in range(10):
        print(f"\n[Beta {beta_idx}]")

        for val in sweep_values:
            beta = torch.zeros(10, device=device)
            beta[beta_idx] = val

            verts, joints = skel.forward(beta.unsqueeze(0), pose.unsqueeze(0), trans.unsqueeze(0))

            # Save mesh
            obj_path = os.path.join(out_dir, f'skel_beta{beta_idx:02d}_val{val:+.1f}_mesh.obj')
            save_obj(verts[0].detach().cpu().numpy(), skel.faces, obj_path)

            # Save joints
            joints_verts, joints_faces = create_joint_spheres(joints[0].detach().cpu().numpy(), radius=0.02)
            if len(joints_verts) > 0:
                obj_path = os.path.join(out_dir, f'skel_beta{beta_idx:02d}_val{val:+.1f}_joints.obj')
                save_obj(joints_verts, joints_faces, obj_path)

            print(f"  beta[{beta_idx}] = {val:+.1f} → saved")

    # Save summary
    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("SKEL Beta Parameter Sweep\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Gender: {gender}\n")
        f.write(f"Frame: {frame_idx}\n")
        f.write(f"Pose: {'T-pose (zeros)' if use_tpose else 'Optimized'}\n")
        f.write(f"Sweep values: {sweep_values}\n\n")
        f.write("Files generated:\n")
        f.write("  - skel_beta_baseline_*.obj (all betas = 0)\n")
        f.write("  - skel_beta{i:02d}_val{v}_*.obj (beta[i] = v, others = 0)\n\n")
        f.write("Purpose:\n")
        f.write("  Test if any specific beta parameter causes shoulder issues.\n")
        f.write("  If ALL look wrong → visualization problem\n")
        f.write("  If specific beta looks wrong → that beta is the problem\n")

    print(f"\n✓ SKEL beta sweep complete!")
    print(f"✓ Summary: {summary_path}")


def beta_sweep_smpl(
    pose_source: str,
    out_dir: str,
    gender: str,
    frame_idx: int,
    use_tpose: bool,
    device: torch.device
):
    """
    Sweep SMPL beta parameters

    Args:
        pose_source: Path to smpl_params.npz (for pose)
        out_dir: Output directory
        gender: 'male' or 'female'
        frame_idx: Which frame to use
        use_tpose: If True, use T-pose instead of optimized pose
        device: torch device
    """

    print("=" * 70)
    print("SMPL Beta Parameter Sweep")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)

    # Load pose
    if use_tpose:
        print("\nUsing T-pose (pose = zeros)")
        pose = torch.zeros(72, device=device)  # SMPL has 72 pose params
        trans = torch.zeros(3, device=device)
    else:
        print(f"\nLoading optimized pose from: {pose_source}")
        data = np.load(pose_source)
        pose = torch.from_numpy(data['poses'][frame_idx]).float().to(device)
        trans = torch.from_numpy(data['trans'][frame_idx]).float().to(device)

    # Initialize SMPL
    smpl = SMPLModel(gender=gender, device=device)

    # Beta sweep values
    sweep_values = [-2.0, -1.0, 0.0, 1.0, 2.0]

    print(f"\nSweeping each beta through: {sweep_values}")
    print(f"Total visualizations: {10 * len(sweep_values) + 1}")

    # Baseline
    print("\n[Baseline] Beta = zeros(10)")
    beta_zeros = torch.zeros(10, device=device)
    verts, joints_temp = smpl.forward(betas=beta_zeros, poses=pose, trans=trans)

    verts_np = verts.detach().cpu().numpy()
    faces = smpl.faces.cpu().numpy() if smpl.faces is not None else None

    obj_path = os.path.join(out_dir, 'smpl_beta_baseline_mesh.obj')
    save_obj(verts_np, faces, obj_path)

    joints_np = joints_temp.detach().cpu().numpy()
    joints_verts, joints_faces = create_joint_spheres(joints_np, radius=0.02)
    if len(joints_verts) > 0:
        obj_path = os.path.join(out_dir, 'smpl_beta_baseline_joints.obj')
        save_obj(joints_verts, joints_faces, obj_path)

    # Sweep each beta
    for beta_idx in range(10):
        print(f"\n[Beta {beta_idx}]")

        for val in sweep_values:
            beta = torch.zeros(10, device=device)
            beta[beta_idx] = val

            verts, joints_temp = smpl.forward(betas=beta, poses=pose, trans=trans)

            verts_np = verts.detach().cpu().numpy()

            # Save mesh
            obj_path = os.path.join(out_dir, f'smpl_beta{beta_idx:02d}_val{val:+.1f}_mesh.obj')
            save_obj(verts_np, faces, obj_path)

            # Save joints
            joints_np = joints_temp.detach().cpu().numpy()
            joints_verts, joints_faces = create_joint_spheres(joints_np, radius=0.02)
            if len(joints_verts) > 0:
                obj_path = os.path.join(out_dir, f'smpl_beta{beta_idx:02d}_val{val:+.1f}_joints.obj')
                save_obj(joints_verts, joints_faces, obj_path)

            print(f"  beta[{beta_idx}] = {val:+.1f} → saved")

    # Save summary
    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("SMPL Beta Parameter Sweep\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Gender: {gender}\n")
        f.write(f"Frame: {frame_idx}\n")
        f.write(f"Pose: {'T-pose (zeros)' if use_tpose else 'Optimized'}\n")
        f.write(f"Sweep values: {sweep_values}\n\n")
        f.write("Files generated:\n")
        f.write("  - smpl_beta_baseline_*.obj (all betas = 0)\n")
        f.write("  - smpl_beta{i:02d}_val{v}_*.obj (beta[i] = v, others = 0)\n\n")
        f.write("Purpose:\n")
        f.write("  Test if any specific beta parameter causes shoulder issues.\n")
        f.write("  Compare with SKEL to see if it's model-specific.\n")

    print(f"\n✓ SMPL beta sweep complete!")
    print(f"✓ Summary: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Beta parameter sweep visualization')
    parser.add_argument('--model', type=str, required=True, choices=['skel', 'smpl'],
                        help='Model type (skel or smpl)')
    parser.add_argument('--pose_source', type=str, default=None,
                        help='Path to params.npz (for pose). If None, uses T-pose')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--gender', type=str, default='male', choices=['male', 'female'],
                        help='Gender')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index to use (default: 0)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device')

    args = parser.parse_args()

    device = torch.device(args.device)
    use_tpose = (args.pose_source is None)

    if args.model == 'skel':
        beta_sweep_skel(
            pose_source=args.pose_source,
            out_dir=args.out_dir,
            gender=args.gender,
            frame_idx=args.frame,
            use_tpose=use_tpose,
            device=device
        )
    else:  # smpl
        beta_sweep_smpl(
            pose_source=args.pose_source,
            out_dir=args.out_dir,
            gender=args.gender,
            frame_idx=args.frame,
            use_tpose=use_tpose,
            device=device
        )


if __name__ == '__main__':
    main()
