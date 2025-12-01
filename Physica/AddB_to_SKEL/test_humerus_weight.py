#!/usr/bin/env python3
"""
Test Beta Optimization for shoulder width matching.
Uses the updated compare_smpl_skel.py with:
- Beta initialization from AddB body proportions
- Shoulder width loss
"""

import os
import sys
import json
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SKEL')

from compare_smpl_skel import optimize_skel, SKEL_JOINT_NAMES, SKEL_ACROMIAL_VERTEX_IDX
from models.skel_model import SKELModelWrapper


def save_obj(vertices, faces, filepath):
    """Save mesh as OBJ file."""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def save_skeleton_obj(joints, parents, filepath):
    """Save skeleton as lines in OBJ file."""
    with open(filepath, 'w') as f:
        for j in joints:
            f.write(f"v {j[0]:.6f} {j[1]:.6f} {j[2]:.6f}\n")
        for i, parent in enumerate(parents):
            if parent >= 0:
                f.write(f"l {parent+1} {i+1}\n")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load saved AddB joints from baseline check
    input_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_baseline_check'
    addb_joints = np.load(os.path.join(input_dir, 'addb', 'joints.npy'))

    with open(os.path.join(input_dir, 'comparison_metrics.json'), 'r') as f:
        metrics = json.load(f)
    joint_names = metrics['addb_joint_names']
    addb_parents = metrics['addb_joint_parents']

    print(f"Loaded AddB joints: {addb_joints.shape}")

    # Run optimization (beta starts from 0, optimized freely)
    print("\n" + "=" * 60)
    print("Running optimization with free beta optimization")
    print("=" * 60)

    result = optimize_skel(
        addb_joints, joint_names, device,
        num_iters=300
        # No subject_height → beta starts from 0 and optimizes freely
    )

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f'/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/humerus_weight_test_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'skel'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'addb'), exist_ok=True)

    # Get faces from result
    faces = result['faces']
    skel_faces = result['skel_faces']
    parents = result['parents']

    # Compute metrics
    skel_joints = result['joints']
    T = skel_joints.shape[0]

    # Get joint indices
    humerus_r_idx = SKEL_JOINT_NAMES.index('humerus_r')
    humerus_l_idx = SKEL_JOINT_NAMES.index('humerus_l')
    acr_r_idx = joint_names.index('acromial_r')
    acr_l_idx = joint_names.index('acromial_l')

    # Compute widths
    humerus_r = skel_joints[:, humerus_r_idx, :]
    humerus_l = skel_joints[:, humerus_l_idx, :]
    addb_acr_r = addb_joints[:, acr_r_idx, :]
    addb_acr_l = addb_joints[:, acr_l_idx, :]

    humerus_width = np.linalg.norm(humerus_r - humerus_l, axis=-1).mean() * 1000
    addb_width = np.linalg.norm(addb_acr_r - addb_acr_l, axis=-1).mean() * 1000

    # Compute virtual acromial width (skin surface)
    skel_verts = result['vertices']
    right_idx = SKEL_ACROMIAL_VERTEX_IDX['right']
    left_idx = SKEL_ACROMIAL_VERTEX_IDX['left']
    virtual_r = skel_verts[:, right_idx, :].mean(axis=1)  # [T, 3]
    virtual_l = skel_verts[:, left_idx, :].mean(axis=1)   # [T, 3]
    virtual_width = np.linalg.norm(virtual_r - virtual_l, axis=-1).mean() * 1000

    # Position errors
    humerus_err_r = np.linalg.norm(humerus_r - addb_acr_r, axis=-1).mean() * 1000
    humerus_err_l = np.linalg.norm(humerus_l - addb_acr_l, axis=-1).mean() * 1000
    virtual_err_r = np.linalg.norm(virtual_r - addb_acr_r, axis=-1).mean() * 1000
    virtual_err_l = np.linalg.norm(virtual_l - addb_acr_l, axis=-1).mean() * 1000

    # MPJPE: compute from mapped joints
    addb_indices = result['addb_indices']
    model_indices = result['model_indices']
    pred_subset = skel_joints[:, model_indices, :]
    target_subset = addb_joints[:, addb_indices, :]
    mpjpe = np.linalg.norm(pred_subset - target_subset, axis=-1).mean() * 1000

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nAddB shoulder width:     {addb_width:.1f} mm")
    print(f"SKEL humerus width:      {humerus_width:.1f} mm (skeleton)")
    print(f"SKEL virtual width:      {virtual_width:.1f} mm (skin surface)")
    print(f"\nWidth difference:")
    print(f"  Humerus vs AddB:       {abs(humerus_width - addb_width):.1f} mm")
    print(f"  Virtual vs AddB:       {abs(virtual_width - addb_width):.1f} mm")
    print(f"\nHumerus → Acromial error (skeleton):")
    print(f"  Right: {humerus_err_r:.1f} mm")
    print(f"  Left:  {humerus_err_l:.1f} mm")
    print(f"  Avg:   {(humerus_err_r + humerus_err_l) / 2:.1f} mm")
    print(f"\nVirtual → Acromial error (skin surface):")
    print(f"  Right: {virtual_err_r:.1f} mm")
    print(f"  Left:  {virtual_err_l:.1f} mm")
    print(f"  Avg:   {(virtual_err_r + virtual_err_l) / 2:.1f} mm")
    print(f"\nMPJPE: {mpjpe:.1f} mm")

    # Save OBJ files
    print(f"\nSaving {T} frames to {out_dir}")
    for t in range(T):
        # SKEL mesh
        mesh_path = os.path.join(out_dir, 'skel', f'mesh_frame_{t:04d}.obj')
        save_obj(result['vertices'][t], faces, mesh_path)

        # SKEL skeleton
        skel_mesh_path = os.path.join(out_dir, 'skel', f'skeleton_frame_{t:04d}.obj')
        save_obj(result['skel_vertices'][t], skel_faces, skel_mesh_path)

        # SKEL joints
        joints_path = os.path.join(out_dir, 'skel', f'joints_frame_{t:04d}.obj')
        save_skeleton_obj(skel_joints[t], parents, joints_path)

        # AddB joints
        addb_path = os.path.join(out_dir, 'addb', f'joints_frame_{t:04d}.obj')
        save_skeleton_obj(addb_joints[t], addb_parents, addb_path)

        print(f"  Frame {t}: saved")

    # Save parameters
    np.savez(
        os.path.join(out_dir, 'skel', 'skel_params.npz'),
        betas=result['betas'],
        poses=result['poses'],
        trans=result['trans'],
    )
    np.save(os.path.join(out_dir, 'skel', 'joints.npy'), skel_joints)
    np.save(os.path.join(out_dir, 'addb', 'joints.npy'), addb_joints)

    # Save metrics
    output_metrics = {
        'mpjpe_mm': mpjpe,
        'addb_width_mm': addb_width,
        'humerus_width_mm': humerus_width,
        'virtual_width_mm': virtual_width,
        'humerus_width_diff_mm': abs(humerus_width - addb_width),
        'virtual_width_diff_mm': abs(virtual_width - addb_width),
        'humerus_error_r_mm': humerus_err_r,
        'humerus_error_l_mm': humerus_err_l,
        'humerus_error_avg_mm': (humerus_err_r + humerus_err_l) / 2,
        'virtual_error_r_mm': virtual_err_r,
        'virtual_error_l_mm': virtual_err_l,
        'virtual_error_avg_mm': (virtual_err_r + virtual_err_l) / 2,
        'beta_frozen': False,
        'final_betas': result['betas'].tolist(),
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(output_metrics, f, indent=2)

    print(f"\n✓ All files saved to: {out_dir}")

    # Comparison with previous results
    print("\n" + "=" * 60)
    print("COMPARISON WITH PREVIOUS (humerus_weight=30 only)")
    print("=" * 60)
    print(f"                     Before (hw=30 only)    After (beta init + sw loss)")
    print(f"SKEL humerus width:      360.0 mm              {humerus_width:.1f} mm")
    print(f"SKEL virtual width:        N/A                 {virtual_width:.1f} mm")
    print(f"Humerus diff from AddB:   10.4 mm              {abs(humerus_width - addb_width):.1f} mm")
    print(f"Virtual diff from AddB:    N/A                 {abs(virtual_width - addb_width):.1f} mm")
    print(f"MPJPE:                    23.4 mm              {mpjpe:.1f} mm")
    print(f"\nFinal betas: {result['betas']}")


if __name__ == '__main__':
    main()
