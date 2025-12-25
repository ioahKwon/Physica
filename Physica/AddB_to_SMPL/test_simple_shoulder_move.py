#!/usr/bin/env python3
"""
Simple Shoulder Movement - Direct Vertex Manipulation

Just move the damn vertices based on skinning weights!
"""

import os
import sys
import pickle
import numpy as np
import torch

sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/SKEL')
sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

from models.skel_model import SKELModelWrapper
from models.smpl_model import SMPLModel

def save_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    """Save mesh as OBJ file"""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def main():
    device = torch.device('cpu')
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_simple_move'
    os.makedirs(out_dir, exist_ok=True)

    # Load pkl
    skel_pkl = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1/skel_male.pkl'
    with open(skel_pkl, 'rb') as f:
        skel_data = pickle.load(f, encoding='latin1')

    weights = skel_data['skin_weights'].toarray()

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
    print("Simple Shoulder Movement - Direct Vertex Manipulation")
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

    verts_np = skel_verts_orig[0].detach().cpu().numpy().copy()  # Explicit copy!
    joints_np = skel_joints_orig[0].detach().cpu().numpy().copy()

    save_obj(verts_np, skel.faces, os.path.join(out_dir, '1_original.obj'))

    print(f"  Mesh shape: {verts_np.shape}")
    print(f"  First 3 vertices:")
    for i in range(3):
        print(f"    v{i}: {verts_np[i]}")

    # Get shoulder weights from pkl (indices 16=left, 17=right)
    weights_left = weights[:, 16]
    weights_right = weights[:, 17]

    print(f"\nShoulder weights info:")
    print(f"  Left shoulder - vertices with weight > 0.01: {np.sum(weights_left > 0.01)}")
    print(f"  Right shoulder - vertices with weight > 0.01: {np.sum(weights_right > 0.01)}")

    # Test different offsets
    offsets = [0.05, 0.10, 0.20]

    for offset in offsets:
        print(f"\n[Offset {offset:.2f}m = {int(offset*100)}cm]")

        # Force deep copy
        verts_modified = np.array(verts_np, copy=True)

        # Verify it's a different array
        print(f"  verts_np id: {id(verts_np)}, verts_modified id: {id(verts_modified)}")
        print(f"  Same object? {verts_np is verts_modified}")

        # Count how many vertices we actually modify
        count_left = 0
        count_right = 0

        # Since left and right shoulder weights overlap completely,
        # we need to distinguish by X coordinate
        # Negative X = right side, Positive X = left side (in body coordinates)

        # Move vertices based on shoulder weights AND position
        for v_idx in range(len(verts_modified)):
            w_shoulder = max(weights_left[v_idx], weights_right[v_idx])  # They're the same anyway

            if w_shoulder > 0.001:
                x_pos = verts_modified[v_idx, 0]

                # If on right side (negative X), move more negative (outward right)
                if x_pos < 0:
                    verts_modified[v_idx, 0] -= offset * w_shoulder
                    count_right += 1
                # If on left side (positive X), move more positive (outward left)
                else:
                    verts_modified[v_idx, 0] += offset * w_shoulder
                    count_left += 1

        print(f"  Modified {count_left} left shoulder vertices")
        print(f"  Modified {count_right} right shoulder vertices")

        save_obj(verts_modified, skel.faces,
                 os.path.join(out_dir, f'2_offset_{int(offset*100)}cm.obj'))

        # Check if actually different
        diff = np.abs(verts_modified - verts_np).max()
        print(f"  Max vertex displacement: {diff:.6f}m")

        # Check specific shoulder vertices
        left_verts_idx = np.where(weights_left > 0.01)[0]
        if len(left_verts_idx) > 0:
            v_idx = left_verts_idx[0]
            print(f"  Sample left vertex {v_idx}:")
            print(f"    Original: {verts_np[v_idx]}")
            print(f"    Modified: {verts_modified[v_idx]}")
            print(f"    Weight: {weights_left[v_idx]:.4f}")

        # Print first few modified vertices
        print(f"  First 3 vertices after modification:")
        for i in range(3):
            print(f"    v{i}: {verts_modified[i]} (diff: {verts_modified[i] - verts_np[i]})")

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

if __name__ == '__main__':
    main()
