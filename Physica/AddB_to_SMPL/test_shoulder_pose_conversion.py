#!/usr/bin/env python3
"""
Test Shoulder Pose with Proper Rotation Conversion

Convert SMPL axis-angle to SKEL Euler angles properly
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
    # Following the convention: R = Rz(gamma) * Ry(beta) * Rx(alpha)
    # But we want ZYX order which is more common

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
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_shoulder_converted'
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
    print("Shoulder Pose Conversion Test (Proper Rotation Conversion)")
    print("=" * 70)

    # Initialize SKEL
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device=device,
        add_acromial_joints=False
    )

    # Test 1: Original SKEL
    print("\n[1] Original SKEL...")
    skel_verts_orig, _ = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_poses[frame:frame+1],
        trans=skel_trans[frame:frame+1]
    )
    save_obj(skel_verts_orig[0].detach().cpu().numpy(), skel.faces,
             os.path.join(out_dir, 'skel_original.obj'))

    # Test 2: Convert SMPL shoulder rotation to Euler
    print("\n[2] SKEL with converted SMPL shoulder rotation...")

    smpl_right_shoulder_aa = smpl_poses[frame, 17].cpu().numpy()  # axis-angle
    smpl_left_shoulder_aa = smpl_poses[frame, 16].cpu().numpy()

    print(f"\nSMPL right shoulder (axis-angle): {smpl_right_shoulder_aa}")
    print(f"SMPL left shoulder (axis-angle): {smpl_left_shoulder_aa}")

    # Convert to rotation matrices
    R_right = axis_angle_to_rotation_matrix(smpl_right_shoulder_aa)
    R_left = axis_angle_to_rotation_matrix(smpl_left_shoulder_aa)

    # Convert to Euler angles
    euler_right = rotation_matrix_to_euler_xyz(R_right)
    euler_left = rotation_matrix_to_euler_xyz(R_left)

    print(f"\nConverted right shoulder (Euler XYZ): {euler_right}")
    print(f"Converted left shoulder (Euler XYZ): {euler_left}")

    # Apply to SKEL pose
    skel_pose_converted = skel_poses[frame].clone()
    skel_pose_converted[29:32] = torch.from_numpy(euler_right).float()
    skel_pose_converted[39:42] = torch.from_numpy(euler_left).float()

    skel_verts_converted, _ = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_pose_converted.unsqueeze(0),
        trans=skel_trans[frame:frame+1]
    )
    save_obj(skel_verts_converted[0].detach().cpu().numpy(), skel.faces,
             os.path.join(out_dir, 'skel_converted_shoulder.obj'))

    # Test 3: SMPL reference
    print("\n[3] SMPL reference...")
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
    print("  1. skel_original.obj           ← SKEL original (bad shoulders)")
    print("  2. skel_converted_shoulder.obj ← SKEL with converted SMPL rotation")
    print("  3. smpl_reference.obj          ← SMPL reference")
    print("\nNote: Conversion is axis-angle → rotation matrix → Euler XYZ")

if __name__ == '__main__':
    main()
