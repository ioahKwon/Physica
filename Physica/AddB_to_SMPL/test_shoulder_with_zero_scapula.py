#!/usr/bin/env python3
"""
Test Shoulder Pose with Zero Scapula

Purpose: Convert SMPL shoulder to SKEL shoulder while keeping scapula=0
- SKEL has scapula joints but SMPL doesn't
- Maybe scapula pose parameters are causing the issue
- Test: scapula=0, shoulder=converted from SMPL

Usage:
    python test_shoulder_with_zero_scapula.py
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

def axis_angle_to_rotation_matrix(axis_angle):
    """Convert axis-angle to rotation matrix using Rodrigues formula"""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.eye(3)

    axis = axis_angle / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

def rotation_matrix_to_euler_xyz(R):
    """Convert rotation matrix to XYZ Euler angles"""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def main():
    device = torch.device('cpu')
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_zero_scapula'
    os.makedirs(out_dir, exist_ok=True)

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
    print("Shoulder Pose Test with Zero Scapula")
    print("=" * 70)

    # Initialize SKEL
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device=device,
        add_acromial_joints=False
    )

    # Test 1: Original SKEL
    print("\n[1] Original SKEL (with optimized scapula + shoulder)...")
    print(f"  Right scapula [26-28]: {skel_poses[frame, 26:29].numpy()}")
    print(f"  Right shoulder [29-31]: {skel_poses[frame, 29:32].numpy()}")
    print(f"  Left scapula [36-38]: {skel_poses[frame, 36:39].numpy()}")
    print(f"  Left shoulder [39-41]: {skel_poses[frame, 39:42].numpy()}")

    skel_verts_orig, _ = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_poses[frame:frame+1],
        trans=skel_trans[frame:frame+1]
    )
    save_obj(skel_verts_orig[0].detach().cpu().numpy(), skel.faces,
             os.path.join(out_dir, '1_skel_original.obj'))

    # Test 2: Zero scapula, keep original shoulder
    print("\n[2] SKEL with zero scapula, original shoulder...")
    skel_pose_zero_scap = skel_poses[frame].clone()
    skel_pose_zero_scap[26:29] = 0  # Right scapula = 0
    skel_pose_zero_scap[36:39] = 0  # Left scapula = 0

    print(f"  Right scapula [26-28]: {skel_pose_zero_scap[26:29].numpy()}")
    print(f"  Right shoulder [29-31]: {skel_pose_zero_scap[29:32].numpy()}")

    skel_verts_zero_scap, _ = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_pose_zero_scap.unsqueeze(0),
        trans=skel_trans[frame:frame+1]
    )
    save_obj(skel_verts_zero_scap[0].detach().cpu().numpy(), skel.faces,
             os.path.join(out_dir, '2_skel_zero_scapula.obj'))

    # Test 3: Zero scapula, zero shoulder
    print("\n[3] SKEL with zero scapula + zero shoulder (partial T-pose)...")
    skel_pose_zero_both = skel_poses[frame].clone()
    skel_pose_zero_both[26:29] = 0  # Right scapula = 0
    skel_pose_zero_both[29:32] = 0  # Right shoulder = 0
    skel_pose_zero_both[36:39] = 0  # Left scapula = 0
    skel_pose_zero_both[39:42] = 0  # Left shoulder = 0

    skel_verts_zero_both, _ = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_pose_zero_both.unsqueeze(0),
        trans=skel_trans[frame:frame+1]
    )
    save_obj(skel_verts_zero_both[0].detach().cpu().numpy(), skel.faces,
             os.path.join(out_dir, '3_skel_zero_both.obj'))

    # Test 4: Zero scapula, converted SMPL shoulder
    print("\n[4] SKEL with zero scapula + converted SMPL shoulder...")

    smpl_right_shoulder_aa = smpl_poses[frame, 17].cpu().numpy()
    smpl_left_shoulder_aa = smpl_poses[frame, 16].cpu().numpy()

    print(f"\nSMPL shoulders (axis-angle):")
    print(f"  Right: {smpl_right_shoulder_aa}")
    print(f"  Left:  {smpl_left_shoulder_aa}")

    # Convert to Euler
    R_right = axis_angle_to_rotation_matrix(smpl_right_shoulder_aa)
    R_left = axis_angle_to_rotation_matrix(smpl_left_shoulder_aa)
    euler_right = rotation_matrix_to_euler_xyz(R_right)
    euler_left = rotation_matrix_to_euler_xyz(R_left)

    print(f"\nConverted to Euler XYZ:")
    print(f"  Right: {euler_right}")
    print(f"  Left:  {euler_left}")

    skel_pose_converted = skel_poses[frame].clone()
    skel_pose_converted[26:29] = 0  # Right scapula = 0
    skel_pose_converted[29:32] = torch.from_numpy(euler_right).float()
    skel_pose_converted[36:39] = 0  # Left scapula = 0
    skel_pose_converted[39:42] = torch.from_numpy(euler_left).float()

    skel_verts_converted, _ = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_pose_converted.unsqueeze(0),
        trans=skel_trans[frame:frame+1]
    )
    save_obj(skel_verts_converted[0].detach().cpu().numpy(), skel.faces,
             os.path.join(out_dir, '4_skel_zero_scapula_converted_shoulder.obj'))

    # Test 5: SMPL reference
    print("\n[5] SMPL reference...")
    smpl = SMPLModel(gender='male', device=device)
    smpl_verts, _ = smpl.forward(
        betas=smpl_beta.unsqueeze(0),
        poses=smpl_poses[frame:frame+1],
        trans=smpl_trans[frame:frame+1]
    )
    save_obj(smpl_verts[0].detach().cpu().numpy(), smpl.faces.cpu().numpy(),
             os.path.join(out_dir, '5_smpl_reference.obj'))

    print("\n" + "=" * 70)
    print("Complete! Files saved to:")
    print(f"  {out_dir}")
    print("\nCompare:")
    print("  1. skel_original.obj                       ← Original (bad shoulders)")
    print("  2. skel_zero_scapula.obj                   ← Zero scapula, keep shoulder")
    print("  3. skel_zero_both.obj                      ← Zero scapula + shoulder")
    print("  4. skel_zero_scapula_converted_shoulder.obj ← Zero scapula + SMPL shoulder")
    print("  5. smpl_reference.obj                      ← SMPL reference")
    print("\nDiagnosis:")
    print("  If #2 looks good → scapula was the problem!")
    print("  If #3 looks good → both scapula and shoulder pose were problems")
    print("  If #4 looks good → need SMPL shoulder rotation with zero scapula")

if __name__ == '__main__':
    main()
