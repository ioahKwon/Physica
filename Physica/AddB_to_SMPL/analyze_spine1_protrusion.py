#!/usr/bin/env python3
"""
Analyze spine1 protrusion in FIX 8 results
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel


def read_smpl_joints(smpl_params_path, smpl_model_path):
    """Read SMPL joints from parameters"""
    device = torch.device('cpu')
    smpl_model = SMPLModel(smpl_model_path, device=device)

    data = np.load(smpl_params_path)
    poses = data['poses']
    trans = data['trans']
    betas = data['betas']

    num_frames = len(poses)
    all_joints = []

    print(f"Reading {num_frames} frames...")

    for i in range(num_frames):
        if i % 50 == 0:
            print(f"  Frame {i}/{num_frames}")

        betas_t = torch.from_numpy(betas).float().to(device).unsqueeze(0)

        if len(poses.shape) == 3:
            poses_frame = poses[i].reshape(-1)
        else:
            poses_frame = poses[i]

        poses_t = torch.from_numpy(poses_frame).float().to(device)
        trans_t = torch.from_numpy(trans[i]).float().to(device)

        with torch.no_grad():
            _, joints = smpl_model.forward(betas_t[0], poses_t, trans_t)

        if joints.dim() == 3 and joints.shape[0] == 1:
            joints = joints.squeeze(0)

        all_joints.append(joints.cpu().numpy())

    return np.stack(all_joints, axis=0)


def analyze_spine_geometry(joints):
    """
    Analyze spine geometry

    SMPL spine hierarchy:
    pelvis (0) → spine1 (3) → spine2 (6) → spine3 (9) → neck (12) → head (15)
    """
    pelvis = joints[:, 0]      # [T, 3]
    spine1 = joints[:, 3]      # [T, 3]
    spine2 = joints[:, 6]      # [T, 3]
    spine3 = joints[:, 9]      # [T, 3]
    neck = joints[:, 12]       # [T, 3]

    # Calculate distances
    pelvis_to_spine1 = np.linalg.norm(spine1 - pelvis, axis=1)  # [T]
    spine1_to_spine2 = np.linalg.norm(spine2 - spine1, axis=1)  # [T]
    spine2_to_spine3 = np.linalg.norm(spine3 - spine2, axis=1)  # [T]

    # Calculate relative positions (in pelvis coordinate frame)
    # Y-axis is up, X is right, Z is forward
    spine1_rel = spine1 - pelvis
    spine2_rel = spine2 - pelvis
    spine3_rel = spine3 - pelvis

    # Forward protrusion (X-axis)
    spine1_forward = spine1_rel[:, 0]
    spine2_forward = spine2_rel[:, 0]
    spine3_forward = spine3_rel[:, 0]

    # Lateral deviation (Z-axis)
    spine1_lateral = spine1_rel[:, 2]
    spine2_lateral = spine2_rel[:, 2]

    # Height (Y-axis)
    spine1_height = spine1_rel[:, 1]
    spine2_height = spine2_rel[:, 1]
    spine3_height = spine3_rel[:, 1]

    # Check if spine1 protrudes forward MORE than spine2
    # (This would be unnatural - spine1 should be behind or aligned with spine2)
    protrusion_excess = spine1_forward - spine2_forward  # positive = spine1 sticks out more

    return {
        'pelvis_to_spine1_dist': pelvis_to_spine1,
        'spine1_to_spine2_dist': spine1_to_spine2,
        'spine2_to_spine3_dist': spine2_to_spine3,
        'spine1_forward': spine1_forward,
        'spine2_forward': spine2_forward,
        'spine3_forward': spine3_forward,
        'spine1_lateral': spine1_lateral,
        'spine2_lateral': spine2_lateral,
        'spine1_height': spine1_height,
        'spine2_height': spine2_height,
        'spine3_height': spine3_height,
        'protrusion_excess': protrusion_excess,
    }


def print_analysis(name, stats):
    """Print analysis results"""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    print(f"\n[Spine Segment Lengths]")
    print(f"  pelvis → spine1: {stats['pelvis_to_spine1_dist'].mean()*100:.2f} cm "
          f"(std: {stats['pelvis_to_spine1_dist'].std()*100:.2f} cm)")
    print(f"  spine1 → spine2: {stats['spine1_to_spine2_dist'].mean()*100:.2f} cm "
          f"(std: {stats['spine1_to_spine2_dist'].std()*100:.2f} cm)")
    print(f"  spine2 → spine3: {stats['spine2_to_spine3_dist'].mean()*100:.2f} cm "
          f"(std: {stats['spine2_to_spine3_dist'].std()*100:.2f} cm)")

    print(f"\n[Heights Relative to Pelvis (Y-axis)]")
    print(f"  spine1: {stats['spine1_height'].mean()*100:+.2f} cm "
          f"(std: {stats['spine1_height'].std()*100:.2f} cm)")
    print(f"  spine2: {stats['spine2_height'].mean()*100:+.2f} cm "
          f"(std: {stats['spine2_height'].std()*100:.2f} cm)")
    print(f"  spine3: {stats['spine3_height'].mean()*100:+.2f} cm "
          f"(std: {stats['spine3_height'].std()*100:.2f} cm)")

    print(f"\n[Forward Position Relative to Pelvis (X-axis, + = forward)]")
    print(f"  spine1: {stats['spine1_forward'].mean()*100:+.2f} cm "
          f"(std: {stats['spine1_forward'].std()*100:.2f} cm)")
    print(f"  spine2: {stats['spine2_forward'].mean()*100:+.2f} cm "
          f"(std: {stats['spine2_forward'].std()*100:.2f} cm)")
    print(f"  spine3: {stats['spine3_forward'].mean()*100:+.2f} cm "
          f"(std: {stats['spine3_forward'].std()*100:.2f} cm)")

    print(f"\n[Lateral Deviation (Z-axis, + = right)]")
    print(f"  spine1: {stats['spine1_lateral'].mean()*100:+.2f} cm "
          f"(std: {stats['spine1_lateral'].std()*100:.2f} cm)")
    print(f"  spine2: {stats['spine2_lateral'].mean()*100:+.2f} cm "
          f"(std: {stats['spine2_lateral'].std()*100:.2f} cm)")

    print(f"\n[Protrusion Analysis]")
    excess = stats['protrusion_excess'].mean() * 100
    print(f"  spine1 protrusion excess: {excess:+.2f} cm")
    if excess > 1.0:
        print(f"  ⚠️  WARNING: spine1 protrudes {excess:.2f} cm MORE than spine2!")
        print(f"  This is anatomically incorrect - spine1 should be behind spine2")
    elif excess > 0:
        print(f"  ⚠️  spine1 slightly ahead of spine2 by {excess:.2f} cm")
    else:
        print(f"  ✓ spine1 is behind spine2 by {-excess:.2f} cm (correct)")

    # Check for frames with excessive protrusion
    bad_frames = np.where(stats['protrusion_excess'] > 0.02)[0]  # > 2cm
    if len(bad_frames) > 0:
        print(f"  Frames with >2cm protrusion: {len(bad_frames)}/{len(stats['protrusion_excess'])} "
              f"({len(bad_frames)/len(stats['protrusion_excess'])*100:.1f}%)")


def main():
    smpl_model_path = '/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl'

    versions = {
        'FIX 8 spine2 (back → spine2)': '/tmp/v8_spine2_p020_split5/foot_orient_loss_with_arm_carter2023_p020/smpl_params.npz',
        'FIX 8 spine3 (back → spine3)': '/tmp/v8_spine3_p020_split5/foot_orient_loss_with_arm_carter2023_p020/smpl_params.npz',
    }

    print("="*80)
    print("Spine1 Protrusion Analysis - FIX 8")
    print("="*80)

    all_stats = {}

    for name, path in versions.items():
        print(f"\n\n[Reading {name}]")
        joints = read_smpl_joints(path, smpl_model_path)
        print(f"Loaded {len(joints)} frames with {joints.shape[1]} joints")

        stats = analyze_spine_geometry(joints)
        all_stats[name] = stats
        print_analysis(name, stats)

    # Comparison
    print(f"\n\n{'='*80}")
    print("Comparison Summary")
    print(f"{'='*80}")

    print(f"\n{'Version':<35} {'spine1 protrusion':>20} {'Status':>20}")
    print("-"*80)

    for name, stats in all_stats.items():
        excess = stats['protrusion_excess'].mean() * 100
        status = "❌ BAD" if excess > 1.0 else ("⚠️  OK-ish" if excess > 0 else "✓ GOOD")
        print(f"{name:<35} {excess:+18.2f} cm {status:>20}")


if __name__ == '__main__':
    main()
