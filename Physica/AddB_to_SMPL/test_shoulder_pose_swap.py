#!/usr/bin/env python3
"""
Test Shoulder Pose Swap

Purpose: Replace SKEL shoulder pose parameters with SMPL shoulder pose
- Keep SKEL body/legs/arms pose
- Only change shoulder rotation parameters
- See if shoulder visualization improves

Usage:
    python test_shoulder_pose_swap.py
"""

import os
import sys
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

def axis_angle_to_euler(axis_angle):
    """Convert SMPL axis-angle to approximate Euler angles"""
    # Simple approximation - may not be perfect but good enough for testing
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-6:
        return np.zeros(3)
    axis = axis_angle / angle
    # Return scaled axis as Euler approximation
    return axis_angle

def main():
    device = torch.device('cpu')
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_shoulder_swap'
    os.makedirs(out_dir, exist_ok=True)

    # Load optimized parameters
    skel_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/skel/skel_params.npz')
    smpl_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/smpl/smpl_params.npz')

    skel_beta = torch.from_numpy(skel_params['betas']).float().to(device)
    skel_poses = torch.from_numpy(skel_params['poses']).float().to(device)  # [T, 46]
    skel_trans = torch.from_numpy(skel_params['trans']).float().to(device)

    smpl_beta = torch.from_numpy(smpl_params['betas']).float().to(device)
    smpl_poses = torch.from_numpy(smpl_params['poses']).float().to(device)  # [T, 72]
    smpl_trans = torch.from_numpy(smpl_params['trans']).float().to(device)

    frame = 0  # Test on first frame

    print("=" * 70)
    print("Shoulder Pose Swap Test")
    print("=" * 70)

    # SKEL pose parameter indices (from skel_model.py)
    # 26-28: scapula_abduction_r, scapula_elevation_r, scapula_upward_rot_r
    # 29-31: shoulder_r_x, shoulder_r_y, shoulder_r_z
    # 36-38: scapula_abduction_l, scapula_elevation_l, scapula_upward_rot_l
    # 39-41: shoulder_l_x, shoulder_l_y, shoulder_l_z

    # SMPL pose parameter indices
    # SMPL poses: [T, 24, 3] = [frames, joints, axis-angle]
    # Joint 16: left_shoulder, Joint 17: right_shoulder

    print(f"\nSKEL pose shape: {skel_poses.shape}")
    print(f"SMPL pose shape: {smpl_poses.shape}")

    # Original SKEL
    print(f"\nSKEL shoulder params (frame {frame}):")
    print(f"  Right scapula [26-28]: {skel_poses[frame, 26:29].numpy()}")
    print(f"  Right shoulder [29-31]: {skel_poses[frame, 29:32].numpy()}")
    print(f"  Left scapula [36-38]: {skel_poses[frame, 36:39].numpy()}")
    print(f"  Left shoulder [39-41]: {skel_poses[frame, 39:42].numpy()}")

    print(f"\nSMPL shoulder params (frame {frame}):")
    print(f"  Right shoulder (joint 17): {smpl_poses[frame, 17].numpy()}")
    print(f"  Left shoulder (joint 16): {smpl_poses[frame, 16].numpy()}")

    # Initialize models
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device=device,
        add_acromial_joints=False
    )

    # Test 1: Original SKEL
    print("\n[1] Original SKEL pose...")
    skel_verts_orig, _ = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_poses[frame:frame+1],
        trans=skel_trans[frame:frame+1]
    )
    save_obj(skel_verts_orig[0].detach().cpu().numpy(), skel.faces,
             os.path.join(out_dir, 'skel_original_shoulder.obj'))

    # Test 2: SKEL with SMPL shoulder rotation
    print("\n[2] SKEL with SMPL shoulder rotation...")
    skel_pose_modified = skel_poses[frame].clone()

    # Copy SMPL shoulder rotations to SKEL
    # This is an approximation - SMPL uses axis-angle, SKEL uses Euler
    # For testing purposes, we'll just copy the values directly

    # Right shoulder: SMPL joint 17 → SKEL[29:32]
    skel_pose_modified[29:32] = smpl_poses[frame, 17]

    # Left shoulder: SMPL joint 16 → SKEL[39:42]
    skel_pose_modified[39:42] = smpl_poses[frame, 16]

    print(f"\nModified SKEL shoulder params:")
    print(f"  Right shoulder [29-31]: {skel_pose_modified[29:32].numpy()}")
    print(f"  Left shoulder [39-41]: {skel_pose_modified[39:42].numpy()}")

    skel_verts_swap, _ = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_pose_modified.unsqueeze(0),
        trans=skel_trans[frame:frame+1]
    )
    save_obj(skel_verts_swap[0].detach().cpu().numpy(), skel.faces,
             os.path.join(out_dir, 'skel_smpl_shoulder.obj'))

    # Test 3: Try zeroing shoulder rotation
    print("\n[3] SKEL with zero shoulder rotation...")
    skel_pose_zero_shoulder = skel_poses[frame].clone()
    skel_pose_zero_shoulder[29:32] = 0  # Right shoulder
    skel_pose_zero_shoulder[39:42] = 0  # Left shoulder

    skel_verts_zero, _ = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_pose_zero_shoulder.unsqueeze(0),
        trans=skel_trans[frame:frame+1]
    )
    save_obj(skel_verts_zero[0].detach().cpu().numpy(), skel.faces,
             os.path.join(out_dir, 'skel_zero_shoulder.obj'))

    # Also try scapula parameters
    print("\n[4] SKEL with zero scapula rotation...")
    skel_pose_zero_scapula = skel_poses[frame].clone()
    skel_pose_zero_scapula[26:29] = 0  # Right scapula
    skel_pose_zero_scapula[36:39] = 0  # Left scapula

    skel_verts_zero_scap, _ = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_pose_zero_scapula.unsqueeze(0),
        trans=skel_trans[frame:frame+1]
    )
    save_obj(skel_verts_zero_scap[0].detach().cpu().numpy(), skel.faces,
             os.path.join(out_dir, 'skel_zero_scapula.obj'))

    # Test 5: SMPL for reference
    print("\n[5] SMPL original (for reference)...")
    smpl = SMPLModel(gender='male', device=device)
    smpl_verts, _ = smpl.forward(
        betas=smpl_beta.unsqueeze(0),
        poses=smpl_poses[frame:frame+1],
        trans=smpl_trans[frame:frame+1]
    )
    save_obj(smpl_verts[0].detach().cpu().numpy(), smpl.faces.cpu().numpy(),
             os.path.join(out_dir, 'smpl_reference.obj'))

    print("\n" + "=" * 70)
    print("Complete! Files saved to:")
    print(f"  {out_dir}")
    print("\nCompare:")
    print("  1. skel_original_shoulder.obj     ← Original SKEL (bad shoulders)")
    print("  2. skel_smpl_shoulder.obj         ← SKEL with SMPL shoulder rotation")
    print("  3. skel_zero_shoulder.obj         ← SKEL with zero shoulder rotation")
    print("  4. skel_zero_scapula.obj          ← SKEL with zero scapula rotation")
    print("  5. smpl_reference.obj             ← SMPL (good shoulders)")
    print("\nIf #2 or #3 or #4 look good → Shoulder POSE was the problem!")
    print("If all SKEL look bad → Problem is in beta or model itself")

if __name__ == '__main__':
    main()
