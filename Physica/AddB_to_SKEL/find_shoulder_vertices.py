#!/usr/bin/env python3
"""
Find shoulder/acromial vertices in SKEL mesh

Since SMPL/SMPLify-X vertex indices don't match SKEL,
we need to find the correct vertex indices near the shoulder.
"""

import sys
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

import numpy as np
import torch

from models.skel_model import SKELModelWrapper, SKEL_JOINT_NAMES

SKEL_MODEL_PATH = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'


def find_shoulder_vertices():
    """
    Find vertices near the shoulder joints in SKEL mesh.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load SKEL model
    skel = SKELModelWrapper(model_path=SKEL_MODEL_PATH, gender='male', device=device)

    # Get T-pose mesh (zero pose)
    betas = torch.zeros(1, 10, device=device)
    poses = torch.zeros(1, 46, device=device)
    trans = torch.zeros(1, 3, device=device)

    with torch.no_grad():
        verts, joints = skel.forward(betas, poses, trans)

    verts = verts[0].cpu().numpy()  # [6890, 3]
    joints = joints[0].cpu().numpy()  # [24, 3]

    print(f"Vertices shape: {verts.shape}")
    print(f"Joints shape: {joints.shape}")

    # Find shoulder-related joints
    print("\n=== Shoulder-related joint positions (T-pose) ===")
    shoulder_joints = ['thorax', 'scapula_r', 'humerus_r', 'scapula_l', 'humerus_l']
    for name in shoulder_joints:
        idx = SKEL_JOINT_NAMES.index(name)
        pos = joints[idx]
        print(f"  {name} (idx {idx}): {pos}")

    # Get humerus positions (these are the shoulder-arm connection points)
    humerus_r_idx = SKEL_JOINT_NAMES.index('humerus_r')
    humerus_l_idx = SKEL_JOINT_NAMES.index('humerus_l')
    humerus_r = joints[humerus_r_idx]
    humerus_l = joints[humerus_l_idx]

    print(f"\n=== Finding vertices near shoulder ===")

    # For acromial (shoulder tip), we want vertices that are:
    # 1. Near the humerus joint (but slightly lateral/superior)
    # 2. On the outer edge of the shoulder

    # Find vertices closest to humerus_r
    dists_r = np.linalg.norm(verts - humerus_r, axis=1)
    closest_r = np.argsort(dists_r)[:20]

    print(f"\nClosest 20 vertices to humerus_r ({humerus_r}):")
    for i, idx in enumerate(closest_r):
        print(f"  {i+1}. Vertex {idx}: {verts[idx]} (dist: {dists_r[idx]*1000:.1f}mm)")

    # Find vertices closest to humerus_l
    dists_l = np.linalg.norm(verts - humerus_l, axis=1)
    closest_l = np.argsort(dists_l)[:20]

    print(f"\nClosest 20 vertices to humerus_l ({humerus_l}):")
    for i, idx in enumerate(closest_l):
        print(f"  {i+1}. Vertex {idx}: {verts[idx]} (dist: {dists_l[idx]*1000:.1f}mm)")

    # For acromial, we want vertices that are more lateral (larger |x|) and superior (larger y)
    # Let's look at vertices near but slightly above/lateral to humerus

    # Right side: x should be more negative (lateral), y should be similar or higher
    # Create a target point that is slightly lateral/superior to humerus
    acromial_target_r = humerus_r.copy()
    acromial_target_r[0] -= 0.03  # more lateral (negative x)
    acromial_target_r[1] += 0.02  # slightly higher

    dists_acr_r = np.linalg.norm(verts - acromial_target_r, axis=1)
    closest_acr_r = np.argsort(dists_acr_r)[:20]

    print(f"\n=== Candidate acromial vertices (right) ===")
    print(f"Target position (lateral/superior to humerus): {acromial_target_r}")
    print("Closest vertices:")
    for i, idx in enumerate(closest_acr_r):
        v = verts[idx]
        print(f"  {i+1}. Vertex {idx}: [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}] (dist: {dists_acr_r[idx]*1000:.1f}mm)")

    # Left side: x should be more positive (lateral)
    acromial_target_l = humerus_l.copy()
    acromial_target_l[0] += 0.03  # more lateral (positive x)
    acromial_target_l[1] += 0.02  # slightly higher

    dists_acr_l = np.linalg.norm(verts - acromial_target_l, axis=1)
    closest_acr_l = np.argsort(dists_acr_l)[:20]

    print(f"\n=== Candidate acromial vertices (left) ===")
    print(f"Target position (lateral/superior to humerus): {acromial_target_l}")
    print("Closest vertices:")
    for i, idx in enumerate(closest_acr_l):
        v = verts[idx]
        print(f"  {i+1}. Vertex {idx}: [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}] (dist: {dists_acr_l[idx]*1000:.1f}mm)")

    # Also check the SMPLify-X indices to see where they actually are
    print(f"\n=== Checking SMPLify-X vertex indices in SKEL mesh ===")
    smplify_left = [3321, 3325, 3290, 3340]
    smplify_right = [5624, 5630, 5690, 5700]

    print("SMPLify-X left acromion vertices (SHOULD be at shoulder height ~1.4m):")
    for idx in smplify_left:
        print(f"  Vertex {idx}: {verts[idx]}")

    print("\nSMPLify-X right acromion vertices:")
    for idx in smplify_right:
        print(f"  Vertex {idx}: {verts[idx]}")

    # Recommend new indices
    print("\n" + "="*60)
    print("RECOMMENDED ACROMIAL VERTEX INDICES FOR SKEL:")
    print("="*60)

    # Pick the 4 closest to our acromial target
    recommended_r = closest_acr_r[:4].tolist()
    recommended_l = closest_acr_l[:4].tolist()

    print(f"Right: {recommended_r}")
    print(f"Left:  {recommended_l}")

    # Verify positions
    print("\nVerification:")
    avg_r = verts[recommended_r].mean(axis=0)
    avg_l = verts[recommended_l].mean(axis=0)
    print(f"  Right avg position: {avg_r}")
    print(f"  Left avg position:  {avg_l}")
    print(f"  Shoulder width (vertex avg): {np.linalg.norm(avg_r - avg_l)*1000:.1f}mm")
    print(f"  Shoulder width (humerus):    {np.linalg.norm(humerus_r - humerus_l)*1000:.1f}mm")

    return {
        'right': recommended_r,
        'left': recommended_l,
    }


if __name__ == '__main__':
    find_shoulder_vertices()
