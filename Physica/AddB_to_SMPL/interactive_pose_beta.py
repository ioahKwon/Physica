#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Pose & Beta Viewer

Usage Examples:
    # 1. Beta만 조절 (T-pose)
    python interactive_pose_beta.py --model smpl --beta 0:2.0          # beta[0]=2.0
    python interactive_pose_beta.py --model skel --beta 0:-1.5,1:0.5   # beta[0]=-1.5, beta[1]=0.5
    
    # 2. Pose만 조절 (zero beta)
    python interactive_pose_beta.py --model smpl --pose 1:0.5,0,0      # L_Hip rotation
    python interactive_pose_beta.py --model skel --pose 6:1.0          # knee_angle_r = 1.0 rad
    
    # 3. Beta + Pose 동시 조절
    python interactive_pose_beta.py --model smpl --beta 0:1.5 --pose 4:0.8,0,0
    
    # 4. Subject의 pose 불러와서 beta만 조절
    python interactive_pose_beta.py --model smpl --subject Subject11 --beta 0:2.0
    
    # 5. Subject의 beta 불러와서 pose만 조절
    python interactive_pose_beta.py --model skel --subject Subject11 --use_subject_beta --pose 6:1.0
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.smpl_model import SMPLModel, SMPL_PARENTS
from models.skel_model import SKELModelWrapper

# Constants
OUTPUT_DIR = '/egr/research-zijunlab/kwonjoon/03_Output/output_12_15/interactive'
INPUT_BASE_DIR = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare'
SKEL_MODEL_DIR = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'

# Joint names for reference
SMPL_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
    'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck',
    'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
]

SKEL_POSE_NAMES = [
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
    'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
    'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l',
    'lumbar_bending', 'lumbar_extension', 'lumbar_twist',
    'thorax_bending', 'thorax_extension', 'thorax_twist',
    'head_bending', 'head_extension', 'head_twist',
    'scapula_abduction_r', 'scapula_elevation_r', 'scapula_upward_rot_r',
    'shoulder_r_x', 'shoulder_r_y', 'shoulder_r_z',
    'elbow_flexion_r', 'pro_sup_r', 'wrist_flexion_r', 'wrist_deviation_r',
    'scapula_abduction_l', 'scapula_elevation_l', 'scapula_upward_rot_l',
    'shoulder_l_x', 'shoulder_l_y', 'shoulder_l_z',
    'elbow_flexion_l', 'pro_sup_l', 'wrist_flexion_l', 'wrist_deviation_l'
]

SKEL_PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 12, 14, 15, 16, 17, 12, 19, 20, 21, 22]


def save_obj(vertices, faces, filepath):
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def create_sphere(center, radius=0.02, n_segments=8):
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


def create_cylinder(start, end, radius=0.008, n_segments=8):
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    direction = direction / length
    if abs(direction[0]) < 0.9:
        perp1 = np.cross(direction, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(direction, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    vertices = []
    faces = []
    for t, center in enumerate([start, end]):
        for i in range(n_segments):
            angle = 2 * np.pi * i / n_segments
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(center + offset)
    for i in range(n_segments):
        p1 = i
        p2 = (i + 1) % n_segments
        p3 = n_segments + (i + 1) % n_segments
        p4 = n_segments + i
        faces.append([p1, p2, p3])
        faces.append([p1, p3, p4])
    return np.array(vertices), np.array(faces)


def create_skeleton(joints, parents):
    all_verts = []
    all_faces = []
    vert_offset = 0
    # Joint spheres
    for joint in joints:
        verts, faces = create_sphere(joint, radius=0.02)
        all_verts.append(verts)
        all_faces.append(faces + vert_offset)
        vert_offset += len(verts)
    # Bones
    for i, parent in enumerate(parents):
        if parent >= 0:
            verts, faces = create_cylinder(joints[parent], joints[i], radius=0.01)
            if len(verts) > 0:
                all_verts.append(verts)
                all_faces.append(faces + vert_offset)
                vert_offset += len(verts)
    return np.vstack(all_verts), np.vstack(all_faces)


def parse_beta(beta_str):
    """Parse beta string like '0:2.0,1:-1.5' into dict"""
    betas = {}
    if not beta_str:
        return betas
    for item in beta_str.split(','):
        if ':' in item:
            idx, val = item.split(':')
            betas[int(idx)] = float(val)
    return betas


def parse_smpl_pose(pose_str):
    """Parse SMPL pose string like '1:0.5,0.3,0.1' into dict
    Format: joint_idx:rx,ry,rz
    """
    poses = {}
    if not pose_str:
        return poses
    for item in pose_str.split(';'):  # Use ; to separate joints
        if ':' in item:
            parts = item.split(':')
            idx = int(parts[0])
            vals = [float(v) for v in parts[1].split(',')]
            poses[idx] = vals
    return poses


def parse_skel_pose(pose_str):
    """Parse SKEL pose string like '6:1.0,32:0.5' into dict
    Format: param_idx:value
    """
    poses = {}
    if not pose_str:
        return poses
    for item in pose_str.split(','):
        if ':' in item:
            idx, val = item.split(':')
            poses[int(idx)] = float(val)
    return poses


def print_help():
    print("\n" + "="*70)
    print("SMPL Joint Indices:")
    print("="*70)
    for i, name in enumerate(SMPL_JOINT_NAMES):
        print(f"  {i:2d}: {name}")
    
    print("\n" + "="*70)
    print("SKEL Pose Parameter Indices:")
    print("="*70)
    for i, name in enumerate(SKEL_POSE_NAMES):
        print(f"  {i:2d}: {name}")
    
    print("\n" + "="*70)
    print("Beta Parameters (both SMPL and SKEL):")
    print("="*70)
    print("  0: Height (larger = shorter)")
    print("  1: Weight (larger = thinner)")
    print("  2-9: Additional shape variations")
    print("  Range: -2 to +2 recommended")


def main():
    parser = argparse.ArgumentParser(description='Interactive Pose & Beta Viewer')
    parser.add_argument('--model', type=str, required=True, choices=['smpl', 'skel'])
    parser.add_argument('--beta', type=str, default='', help='Beta: "0:2.0,1:-1.5"')
    parser.add_argument('--pose', type=str, default='', help='Pose: SMPL="1:0.5,0.3,0.1;4:1.0,0,0" SKEL="6:1.0,32:0.5"')
    parser.add_argument('--subject', type=str, default=None, help='Load pose from subject')
    parser.add_argument('--frame', type=int, default=0, help='Frame index')
    parser.add_argument('--use_subject_beta', action='store_true', help='Also use subject beta')
    parser.add_argument('--name', type=str, default='output', help='Output name')
    parser.add_argument('--show_help', action='store_true', help='Show joint/param indices')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    if args.show_help:
        print_help()
        return
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Parse beta
    beta_dict = parse_beta(args.beta)
    
    if args.model == 'smpl':
        # Load model
        smpl = SMPLModel(gender='male', device=device)
        
        # Initialize pose and beta
        if args.subject:
            params = np.load(os.path.join(INPUT_BASE_DIR, f'final_{args.subject}', 'smpl', 'smpl_params.npz'))
            pose = torch.tensor(params['poses'][args.frame], dtype=torch.float32, device=device)
            trans = torch.tensor(params['trans'][args.frame], dtype=torch.float32, device=device)
            if args.use_subject_beta:
                betas = torch.tensor(params['betas'], dtype=torch.float32, device=device)
            else:
                betas = torch.zeros(10, dtype=torch.float32, device=device)
            print(f"Loaded pose from {args.subject} frame {args.frame}")
        else:
            pose = torch.zeros(24, 3, dtype=torch.float32, device=device)
            trans = torch.zeros(3, dtype=torch.float32, device=device)
            betas = torch.zeros(10, dtype=torch.float32, device=device)
        
        # Apply beta modifications
        for idx, val in beta_dict.items():
            betas[idx] = val
            print(f"  Beta[{idx}] = {val}")
        
        # Parse and apply pose modifications
        pose_dict = parse_smpl_pose(args.pose)
        for idx, vals in pose_dict.items():
            for i, v in enumerate(vals[:3]):
                pose[idx, i] = v
            print(f"  Pose[{idx}] ({SMPL_JOINT_NAMES[idx]}) = {vals}")
        
        # Forward pass
        verts, joints = smpl.forward(betas, pose, trans)
        verts = verts.cpu().numpy()
        joints = joints.cpu().numpy()
        
        # Save
        out_path = os.path.join(OUTPUT_DIR, f'smpl_{args.name}')
        save_obj(verts, smpl.faces.cpu().numpy(), f'{out_path}_mesh.obj')
        skel_verts, skel_faces = create_skeleton(joints, list(SMPL_PARENTS.numpy()))
        save_obj(skel_verts, skel_faces, f'{out_path}_skeleton.obj')
        
        print(f"\nSaved: {out_path}_mesh.obj, {out_path}_skeleton.obj")
        
    else:  # skel
        skel = SKELModelWrapper(SKEL_MODEL_DIR, gender='male', device=device)
        
        if args.subject:
            params = np.load(os.path.join(INPUT_BASE_DIR, f'final_{args.subject}', 'skel', 'skel_params.npz'))
            pose = torch.tensor(params['poses'][args.frame:args.frame+1], dtype=torch.float32, device=device)
            trans = torch.tensor(params['trans'][args.frame:args.frame+1], dtype=torch.float32, device=device)
            if args.use_subject_beta:
                betas = torch.tensor(params['betas'], dtype=torch.float32, device=device).unsqueeze(0)
            else:
                betas = torch.zeros(1, 10, dtype=torch.float32, device=device)
            print(f"Loaded pose from {args.subject} frame {args.frame}")
        else:
            pose = torch.zeros(1, 46, dtype=torch.float32, device=device)
            trans = torch.zeros(1, 3, dtype=torch.float32, device=device)
            betas = torch.zeros(1, 10, dtype=torch.float32, device=device)
        
        # Apply beta modifications
        for idx, val in beta_dict.items():
            betas[0, idx] = val
            print(f"  Beta[{idx}] = {val}")
        
        # Parse and apply pose modifications
        pose_dict = parse_skel_pose(args.pose)
        for idx, val in pose_dict.items():
            pose[0, idx] = val
            print(f"  Pose[{idx}] ({SKEL_POSE_NAMES[idx]}) = {val}")
        
        # Forward pass
        verts, joints = skel.forward(betas, pose, trans)
        verts = verts[0].cpu().numpy()
        joints = joints[0].cpu().numpy()
        
        # Save
        out_path = os.path.join(OUTPUT_DIR, f'skel_{args.name}')
        save_obj(verts, skel.model.skin_f.cpu().numpy(), f'{out_path}_mesh.obj')
        skel_verts, skel_faces = create_skeleton(joints, SKEL_PARENTS)
        save_obj(skel_verts, skel_faces, f'{out_path}_skeleton.obj')
        
        print(f"\nSaved: {out_path}_mesh.obj, {out_path}_skeleton.obj")


if __name__ == '__main__':
    main()
