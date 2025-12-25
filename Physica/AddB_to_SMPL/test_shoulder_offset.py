#!/usr/bin/env python3
"""
Test Direct Shoulder Vertex Offset

Purpose: Directly modify shoulder vertices by adding lateral offset
- Load SKEL model
- Find shoulder vertices from skinning weights
- Add X-direction offset to widen shoulders
- Maintain smooth blending

Usage:
    python test_shoulder_offset.py
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

def get_shoulder_vertices(weights: np.ndarray, joint_names: list, weight_threshold: float = 0.05):
    """
    Find vertices affected by shoulder joints

    Returns:
        right_verts, left_verts, weights_r, weights_l
    """
    # Find humerus joint indices
    humerus_r_idx = None
    humerus_l_idx = None
    scapula_r_idx = None
    scapula_l_idx = None

    for i, name in enumerate(joint_names):
        if name == 'humerus_r':
            humerus_r_idx = i
        elif name == 'humerus_l':
            humerus_l_idx = i
        elif name == 'scapula_r':
            scapula_r_idx = i
        elif name == 'scapula_l':
            scapula_l_idx = i

    # Combine humerus + scapula weights
    weights_r = weights[:, humerus_r_idx]
    weights_l = weights[:, humerus_l_idx]

    if scapula_r_idx is not None:
        weights_r = np.maximum(weights_r, weights[:, scapula_r_idx])
    if scapula_l_idx is not None:
        weights_l = np.maximum(weights_l, weights[:, scapula_l_idx])

    # Find affected vertices
    verts_r = np.where(weights_r > weight_threshold)[0]
    verts_l = np.where(weights_l > weight_threshold)[0]

    return verts_r, verts_l, weights_r, weights_l

def main():
    device = torch.device('cpu')
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_shoulder_offset'
    os.makedirs(out_dir, exist_ok=True)

    # Load SKEL pkl to get skinning weights
    skel_pkl_path = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1/skel_male.pkl'
    with open(skel_pkl_path, 'rb') as f:
        skel_data = pickle.load(f, encoding='latin1')

    weights = skel_data['skin_weights'].toarray()  # Convert sparse to dense
    joint_names = skel_data['joints_name']

    # Load parameters
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
    print("Direct Shoulder Vertex Offset Test")
    print("=" * 70)

    # Get shoulder vertices
    print("\nFinding shoulder vertices from skinning weights...")
    verts_r, verts_l, weights_r, weights_l = get_shoulder_vertices(weights, joint_names, weight_threshold=0.05)

    print(f"  Right shoulder vertices: {len(verts_r)}")
    print(f"  Left shoulder vertices: {len(verts_l)}")

    # Initialize models
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device=device,
        add_acromial_joints=False
    )

    smpl = SMPLModel(gender='male', device=device)

    # Generate original SKEL
    print("\n[1] Original SKEL...")
    skel_verts_orig, skel_joints_orig = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_poses[frame:frame+1],
        trans=skel_trans[frame:frame+1]
    )

    verts_np = skel_verts_orig[0].detach().cpu().numpy()
    save_obj(verts_np, skel.faces, os.path.join(out_dir, '1_skel_original.obj'))

    right_shoulder = skel_joints_orig[0, 15].cpu().numpy()
    left_shoulder = skel_joints_orig[0, 20].cpu().numpy()
    orig_width = np.linalg.norm(right_shoulder - left_shoulder)
    print(f"  Shoulder width: {orig_width:.4f}m")

    # Test different offset scales
    offset_scales = [0.01, 0.02, 0.03, 0.05]  # meters

    for offset_m in offset_scales:
        print(f"\n[Offset {offset_m:.2f}m]")

        verts_modified = verts_np.copy()

        # Apply weighted offset to right shoulder vertices (negative X = outward)
        for v_idx in verts_r:
            weight = np.asscalar(weights_r[v_idx]) if hasattr(np, 'asscalar') else weights_r[v_idx].item()
            verts_modified[v_idx, 0] -= offset_m * weight  # Move left (negative X)

        # Apply weighted offset to left shoulder vertices (positive X = outward)
        for v_idx in verts_l:
            weight = np.asscalar(weights_l[v_idx]) if hasattr(np, 'asscalar') else weights_l[v_idx].item()
            verts_modified[v_idx, 0] += offset_m * weight  # Move right (positive X)

        save_obj(verts_modified, skel.faces,
                 os.path.join(out_dir, f'2_skel_offset_{offset_m:.2f}m.obj'))

        # Estimate shoulder width from modified vertices
        # Use approximate joint positions (won't be exact since we only moved vertices)
        print(f"  Applied lateral offset: ±{offset_m:.2f}m (weighted by skinning)")

    # SMPL reference
    print("\n[Reference] SMPL...")
    smpl_verts, smpl_joints = smpl.forward(
        betas=smpl_beta.unsqueeze(0),
        poses=smpl_poses[frame:frame+1],
        trans=smpl_trans[frame:frame+1]
    )
    save_obj(smpl_verts[0].detach().cpu().numpy(), smpl.faces.cpu().numpy(),
             os.path.join(out_dir, '3_smpl_reference.obj'))

    smpl_right = smpl_joints[0, 17].cpu().numpy()
    smpl_left = smpl_joints[0, 16].cpu().numpy()
    smpl_width = np.linalg.norm(smpl_right - smpl_left)
    print(f"  SMPL shoulder width: {smpl_width:.4f}m")
    print(f"  Difference from SKEL: {(smpl_width - orig_width):.4f}m")

    print("\n" + "=" * 70)
    print("Complete! Files saved to:")
    print(f"  {out_dir}")
    print("\nFiles generated:")
    print("  1_skel_original.obj       ← Original SKEL")
    print("  2_skel_offset_0.01m.obj   ← ±1cm offset")
    print("  2_skel_offset_0.02m.obj   ← ±2cm offset")
    print("  2_skel_offset_0.03m.obj   ← ±3cm offset")
    print("  2_skel_offset_0.05m.obj   ← ±5cm offset")
    print("  3_smpl_reference.obj      ← SMPL reference")

if __name__ == '__main__':
    main()
