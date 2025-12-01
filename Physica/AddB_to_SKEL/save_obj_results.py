#!/usr/bin/env python3
"""
Save OBJ files from shoulder correction optimization results.
"""

import os
import sys
import json
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SKEL')

from optimize_with_shoulder import optimize_skel_with_shoulder


def save_obj(vertices, faces, filepath):
    """Save mesh as OBJ file."""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def save_joints_obj(joints, filepath, radius=0.01):
    """Save joints as sphere markers in OBJ file."""
    with open(filepath, 'w') as f:
        for i, j in enumerate(joints):
            # Simple point marker
            f.write(f"v {j[0]:.6f} {j[1]:.6f} {j[2]:.6f}\n")


def save_skeleton_obj(joints, parents, filepath):
    """Save skeleton as lines in OBJ file."""
    with open(filepath, 'w') as f:
        # Write vertices
        for j in joints:
            f.write(f"v {j[0]:.6f} {j[1]:.6f} {j[2]:.6f}\n")
        # Write lines (edges)
        for i, parent in enumerate(parents):
            if parent >= 0:
                f.write(f"l {parent+1} {i+1}\n")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load saved AddB joints
    input_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_baseline_check'
    addb_joints = np.load(os.path.join(input_dir, 'addb', 'joints.npy'))

    with open(os.path.join(input_dir, 'comparison_metrics.json'), 'r') as f:
        metrics = json.load(f)
    joint_names = metrics['addb_joint_names']

    print(f"Loaded AddB joints: {addb_joints.shape}")

    # Run optimization with best config (shoulder_weight=0.2)
    print("\n" + "=" * 60)
    print("Running optimization with shoulder_weight=0.2")
    print("=" * 60)
    result = optimize_skel_with_shoulder(
        addb_joints, joint_names, device,
        num_iters=300,
        shoulder_weight=0.2,
        width_weight=0.0,
    )

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f'/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/shoulder_obj_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'skel'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'addb'), exist_ok=True)

    # Save OBJ files for each frame
    T = addb_joints.shape[0]
    faces = result['faces']
    skel_faces = result['skel_faces']
    parents = result['parents']

    print(f"\nSaving {T} frames to {out_dir}")

    for t in range(T):
        # SKEL mesh
        mesh_path = os.path.join(out_dir, 'skel', f'mesh_frame_{t:04d}.obj')
        save_obj(result['vertices'][t], faces, mesh_path)

        # SKEL skeleton mesh
        skel_mesh_path = os.path.join(out_dir, 'skel', f'skeleton_frame_{t:04d}.obj')
        save_obj(result['skel_vertices'][t], skel_faces, skel_mesh_path)

        # SKEL joints
        joints_path = os.path.join(out_dir, 'skel', f'joints_frame_{t:04d}.obj')
        save_skeleton_obj(result['joints'][t], parents, joints_path)

        # AddB joints
        addb_path = os.path.join(out_dir, 'addb', f'joints_frame_{t:04d}.obj')
        addb_parents = metrics['addb_joint_parents']
        save_skeleton_obj(addb_joints[t], addb_parents, addb_path)

        print(f"  Frame {t}: saved mesh, skeleton, joints")

    # Save parameters
    np.savez(
        os.path.join(out_dir, 'skel', 'skel_params.npz'),
        betas=result['betas'],
        poses=result['poses'],
        trans=result['trans'],
    )
    np.save(os.path.join(out_dir, 'skel', 'joints.npy'), result['joints'])
    np.save(os.path.join(out_dir, 'addb', 'joints.npy'), addb_joints)

    # Save metrics
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(result['metrics'], f, indent=2)

    print(f"\n✓ All files saved to: {out_dir}")
    print(f"\nMetrics:")
    print(f"  MPJPE: {result['metrics']['mpjpe_mm']:.1f}mm")
    print(f"  Virtual acromial error: {result['metrics']['virtual_avg_error_mm']:.1f}mm")


if __name__ == '__main__':
    main()
