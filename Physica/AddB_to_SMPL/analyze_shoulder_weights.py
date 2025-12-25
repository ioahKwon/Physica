#!/usr/bin/env python3
"""
Analyze SKEL Shoulder Skinning Weights

Purpose: Find which vertices are affected by shoulder joints
- Load SKEL pkl file
- Examine skinning weights for shoulder/humerus joints
- Identify shoulder vertices for targeted manipulation

Usage:
    python analyze_shoulder_weights.py
"""

import pickle
import numpy as np
import os

def main():
    # Load SKEL male model
    skel_pkl_path = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1/skel_male.pkl'

    print("=" * 70)
    print("SKEL Shoulder Skinning Weight Analysis")
    print("=" * 70)

    with open(skel_pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Check available keys
    print("\nAvailable keys in pkl:")
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            print(f"  {key}: {data[key].shape}")
        else:
            print(f"  {key}: {type(data[key])}")

    # Get skinning weights
    weights = data['weights']  # [N_vertices, N_joints]
    print(f"\nSkinning weights shape: {weights.shape}")

    # Get joint names
    joint_names = data.get('joint_names', None)
    if joint_names is not None:
        print(f"\nJoint names ({len(joint_names)}):")
        for i, name in enumerate(joint_names):
            print(f"  {i:2d}: {name}")

    # Find shoulder-related joints
    shoulder_joint_indices = []
    shoulder_names = []

    if joint_names is not None:
        for i, name in enumerate(joint_names):
            if 'shoulder' in name.lower() or 'humerus' in name.lower() or 'scapula' in name.lower():
                shoulder_joint_indices.append(i)
                shoulder_names.append(name)

    print(f"\n{'=' * 70}")
    print(f"Shoulder-related joints:")
    for idx, name in zip(shoulder_joint_indices, shoulder_names):
        print(f"  Joint {idx}: {name}")

    # Analyze weights for each shoulder joint
    for idx, name in zip(shoulder_joint_indices, shoulder_names):
        joint_weights = weights[:, idx]

        # Find vertices with non-zero weights
        affected_verts = np.where(joint_weights > 0.01)[0]  # threshold 1%

        print(f"\n{name} (joint {idx}):")
        print(f"  Vertices affected (weight > 0.01): {len(affected_verts)}")
        print(f"  Max weight: {joint_weights.max():.4f}")
        print(f"  Mean weight (non-zero): {joint_weights[joint_weights > 0].mean():.4f}")

        # Top 10 vertices by weight
        top_indices = np.argsort(joint_weights)[-10:][::-1]
        print(f"  Top 10 vertices by weight:")
        for i, v_idx in enumerate(top_indices):
            print(f"    {i+1}. v{v_idx}: {joint_weights[v_idx]:.4f}")

    # Specifically look for humerus_r and humerus_l (shoulder joints)
    print(f"\n{'=' * 70}")
    print("Focusing on humerus (shoulder) joints...")

    humerus_r_idx = None
    humerus_l_idx = None

    if joint_names is not None:
        for i, name in enumerate(joint_names):
            if name == 'humerus_r':
                humerus_r_idx = i
            elif name == 'humerus_l':
                humerus_l_idx = i

    if humerus_r_idx is not None:
        print(f"\nRight shoulder (humerus_r, joint {humerus_r_idx}):")
        weights_r = weights[:, humerus_r_idx]
        verts_r = np.where(weights_r > 0.1)[0]  # 10% threshold
        print(f"  Vertices with weight > 0.1: {len(verts_r)}")
        print(f"  Vertex indices: {verts_r[:20]}...")  # first 20

        # Save to file
        out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/shoulder_analysis'
        os.makedirs(out_dir, exist_ok=True)

        np.savetxt(os.path.join(out_dir, 'shoulder_r_vertices.txt'),
                   verts_r, fmt='%d')

    if humerus_l_idx is not None:
        print(f"\nLeft shoulder (humerus_l, joint {humerus_l_idx}):")
        weights_l = weights[:, humerus_l_idx]
        verts_l = np.where(weights_l > 0.1)[0]
        print(f"  Vertices with weight > 0.1: {len(verts_l)}")
        print(f"  Vertex indices: {verts_l[:20]}...")

        out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/shoulder_analysis'
        os.makedirs(out_dir, exist_ok=True)

        np.savetxt(os.path.join(out_dir, 'shoulder_l_vertices.txt'),
                   verts_l, fmt='%d')

    print(f"\n{'=' * 70}")
    print("Analysis complete!")
    print(f"Output directory: /egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/shoulder_analysis")

if __name__ == '__main__':
    main()
