#!/usr/bin/env python3
"""
Test Acromial Widening - The CORRECT Shoulder Tip Widening

Use acromial vertex indices (outer shoulder surface) to widen shoulders.
These vertices are on the actual shoulder tips, not neck/collar area.

ACROMIAL_R_VERTICES = [4816, 4819, 4873, 4917, 4916, 4912, 4888, 5009, 4259, 4817]
ACROMIAL_L_VERTICES = [1340, 1341, 1399, 1444, 1445, 1441, 770, 1438, 1415, 1343]
"""

import os
import sys
import pickle
import numpy as np
import torch
from scipy.spatial.distance import cdist

sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/SKEL')
sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

from models.skel_model import SKELModelWrapper
from models.smpl_model import SMPLModel

# Acromial vertex indices from skel_model.py
ACROMIAL_R_VERTICES = [4816, 4819, 4873, 4917, 4916, 4912, 4888, 5009, 4259, 4817]
ACROMIAL_L_VERTICES = [1340, 1341, 1399, 1444, 1445, 1441, 770, 1438, 1415, 1343]

def save_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    """Save mesh as OBJ file"""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def main():
    device = torch.device('cpu')
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_acromial_widen'
    os.makedirs(out_dir, exist_ok=True)

    # Load params
    skel_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/skel/skel_params.npz')
    smpl_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/smpl/smpl_params.npz')

    skel_beta = torch.from_numpy(skel_params['betas']).float().to(device)
    skel_poses = torch.from_numpy(skel_params['poses']).float().to(device)
    skel_trans = torch.from_numpy(skel_params['trans']).float().to(device)

    smpl_beta = torch.from_numpy(smpl_params['betas']).float().to(device)
    smpl_poses = torch.from_numpy(smpl_params['poses']).float().to(device)
    smpl_trans = torch.from_numpy(smpl_params['trans']).float().to(device)

    frame = 0

    print("=" * 70)
    print("Acromial Widening - Actual Shoulder Tip Widening")
    print("=" * 70)

    # Initialize SKEL
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device=device,
        add_acromial_joints=False
    )

    # Get original mesh
    print("\n[1] Original SKEL...")
    skel_verts_orig, skel_joints_orig = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_poses[frame:frame+1],
        trans=skel_trans[frame:frame+1]
    )

    verts_np = skel_verts_orig[0].detach().cpu().numpy().copy()

    save_obj(verts_np, skel.faces, os.path.join(out_dir, '1_original.obj'))

    print(f"  Mesh shape: {verts_np.shape}")

    # Check acromial vertex positions
    print(f"\nAcromial vertex positions:")
    left_acromial_pos = verts_np[ACROMIAL_L_VERTICES].mean(axis=0)
    right_acromial_pos = verts_np[ACROMIAL_R_VERTICES].mean(axis=0)
    print(f"  Left acromial center: {left_acromial_pos}")
    print(f"  Right acromial center: {right_acromial_pos}")
    print(f"  Current shoulder width: {np.linalg.norm(left_acromial_pos - right_acromial_pos):.4f}m")

    # Find all vertices near acromial points to create smooth widening
    # We'll use distance-based weighting from acromial centers

    # Compute distance from each vertex to nearest acromial center
    all_acromial = ACROMIAL_L_VERTICES + ACROMIAL_R_VERTICES
    acromial_positions = verts_np[all_acromial]

    # For each vertex, find distance to nearest acromial vertex
    distances_to_left = cdist(verts_np, verts_np[ACROMIAL_L_VERTICES]).min(axis=1)
    distances_to_right = cdist(verts_np, verts_np[ACROMIAL_R_VERTICES]).min(axis=1)

    # Create smooth falloff weights based on distance
    # Vertices close to acromial get higher weights
    falloff_radius = 0.15  # 15cm radius of influence

    weights_left = np.exp(-distances_to_left**2 / (2 * (falloff_radius/3)**2))
    weights_left[distances_to_left > falloff_radius] = 0

    weights_right = np.exp(-distances_to_right**2 / (2 * (falloff_radius/3)**2))
    weights_right[distances_to_right > falloff_radius] = 0

    print(f"\nVertices with non-zero weight:")
    print(f"  Left shoulder: {np.sum(weights_left > 0.01)}")
    print(f"  Right shoulder: {np.sum(weights_right > 0.01)}")

    # Test different offsets
    offsets = [0.02, 0.05, 0.08, 0.10]

    for offset in offsets:
        print(f"\n[Offset {offset:.2f}m = {int(offset*100)}cm]")

        verts_modified = np.array(verts_np, copy=True)

        # Move left shoulder vertices outward (positive X)
        for v_idx in range(len(verts_modified)):
            if weights_left[v_idx] > 0.001:
                verts_modified[v_idx, 0] += offset * weights_left[v_idx]

            # Move right shoulder vertices outward (negative X)
            if weights_right[v_idx] > 0.001:
                verts_modified[v_idx, 0] -= offset * weights_right[v_idx]

        save_obj(verts_modified, skel.faces,
                 os.path.join(out_dir, f'2_acromial_offset_{int(offset*100)}cm.obj'))

        # Verify displacement
        new_left_pos = verts_modified[ACROMIAL_L_VERTICES].mean(axis=0)
        new_right_pos = verts_modified[ACROMIAL_R_VERTICES].mean(axis=0)
        new_width = np.linalg.norm(new_left_pos - new_right_pos)

        print(f"  New shoulder width: {new_width:.4f}m")
        print(f"  Width increase: {(new_width - np.linalg.norm(left_acromial_pos - right_acromial_pos))*100:.2f}cm")

    # SMPL reference
    print("\n[Reference] SMPL...")
    smpl = SMPLModel(gender='male', device=device)
    smpl_verts, _ = smpl.forward(
        betas=smpl_beta.unsqueeze(0),
        poses=smpl_poses[frame:frame+1],
        trans=smpl_trans[frame:frame+1]
    )
    save_obj(smpl_verts[0].detach().cpu().numpy(), smpl.faces.cpu().numpy(),
             os.path.join(out_dir, '3_smpl.obj'))

    print("\n" + "=" * 70)
    print("Files saved to:")
    print(f"  {out_dir}")
    print("\nFiles:")
    print("  1_original.obj")
    for o in offsets:
        print(f"  2_acromial_offset_{int(o*100)}cm.obj")
    print("  3_smpl.obj")
    print("\nNow using acromial vertices (actual shoulder tips)!")
    print("This should widen the outer shoulder area, not the neck.")

if __name__ == '__main__':
    main()
