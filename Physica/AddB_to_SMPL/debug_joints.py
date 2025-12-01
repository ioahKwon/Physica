#!/usr/bin/env python3
"""
Debug SMPL joints positions
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel

# Load SMPL model
device = torch.device('cpu')
smpl_model = SMPLModel('models/smpl_model.pkl', device=device)

# Load SMPL parameters
data = np.load('/tmp/baseline_rotated_all_joints.npz')
betas = data['betas']
poses = data['poses']
trans = data['trans']

print(f"Betas shape: {betas.shape}")
print(f"Poses shape: {poses.shape}")
print(f"Trans shape: {trans.shape}")

# Frame 0
frame_idx = 0
betas_frame = np.zeros((1, 10), dtype=np.float32)

if len(poses.shape) == 3:
    poses_frame = poses[frame_idx:frame_idx+1].reshape(1, -1)
else:
    poses_frame = poses[frame_idx:frame_idx+1]

trans_frame = trans[frame_idx:frame_idx+1]

print(f"\nFrame {frame_idx}:")
print(f"Betas: {betas_frame.shape}")
print(f"Poses: {poses_frame.shape}")
print(f"Trans: {trans_frame.shape}")
print(f"Trans value: {trans_frame[0]}")

# Convert to tensors
betas_t = torch.from_numpy(betas_frame).float().to(device)
poses_t = torch.from_numpy(poses_frame).float().to(device)
trans_t = torch.from_numpy(trans_frame).float().to(device)

# Get joints
with torch.no_grad():
    vertices, joints = smpl_model.forward(betas=betas_t, poses=poses_t, trans=trans_t)

joints = joints.cpu().numpy()[0]  # (24, 3)

print(f"\nJoints shape: {joints.shape}")
print(f"\nJoint positions:")
joint_names = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]

for i, (name, joint) in enumerate(zip(joint_names, joints)):
    print(f"{i:2d} {name:12s}: X={joint[0]:7.3f}, Y={joint[1]:7.3f}, Z={joint[2]:7.3f}")

print(f"\nPose parameters (first 10):")
poses_reshaped = poses_frame.reshape(24, 3)
for i in range(min(10, 24)):
    rot = poses_reshaped[i]
    print(f"{i:2d} {joint_names[i]:12s}: [{rot[0]:7.3f}, {rot[1]:7.3f}, {rot[2]:7.3f}]")
