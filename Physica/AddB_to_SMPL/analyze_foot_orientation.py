#!/usr/bin/env python3
"""
Analyze foot orientation in FIX 8 results
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


def analyze_foot_orientation(joints):
    """
    Analyze foot orientation

    SMPL leg joints:
    - Left:  hip(1) → knee(4) → ankle(7) → foot(10)
    - Right: hip(2) → knee(5) → ankle(8) → foot(11)

    Coordinate system:
    - X: right(+) / left(-)
    - Y: up(+) / down(-)
    - Z: back(+) / front(-)
    """
    # Left leg
    left_ankle = joints[:, 7]   # [T, 3]
    left_foot = joints[:, 10]   # [T, 3]

    # Right leg
    right_ankle = joints[:, 8]  # [T, 3]
    right_foot = joints[:, 11]  # [T, 3]

    # Foot vectors (ankle → foot)
    left_foot_vec = left_foot - left_ankle      # [T, 3]
    right_foot_vec = right_foot - right_ankle   # [T, 3]

    # Decompose foot vectors
    # X: lateral (negative = inward/medial, positive = outward/lateral)
    # Y: vertical (negative = down, positive = up - shouldn't happen)
    # Z: forward/back (negative = forward, positive = back)

    left_foot_forward = -left_foot_vec[:, 2]    # negative Z = forward
    left_foot_down = -left_foot_vec[:, 1]       # negative Y = down
    left_foot_lateral = left_foot_vec[:, 0]     # positive X = lateral (outward)

    right_foot_forward = -right_foot_vec[:, 2]
    right_foot_down = -right_foot_vec[:, 1]
    right_foot_lateral = right_foot_vec[:, 0]   # positive X = lateral (outward for left, inward for right!)

    # Calculate foot angle relative to forward direction
    # Angle in XZ plane (lateral vs forward)
    left_foot_angle = np.arctan2(left_foot_lateral, left_foot_forward) * 180 / np.pi
    right_foot_angle = np.arctan2(-right_foot_lateral, right_foot_forward) * 180 / np.pi

    # Calculate foot tilt (should point down)
    left_foot_len = np.linalg.norm(left_foot_vec, axis=1)
    right_foot_len = np.linalg.norm(right_foot_vec, axis=1)

    left_down_ratio = left_foot_down / left_foot_len
    right_down_ratio = right_foot_down / right_foot_len

    # Pitch angle (0° = horizontal, 90° = pointing down)
    left_pitch = np.arcsin(np.clip(left_down_ratio, -1, 1)) * 180 / np.pi
    right_pitch = np.arcsin(np.clip(right_down_ratio, -1, 1)) * 180 / np.pi

    return {
        'left_foot_angle': left_foot_angle,
        'right_foot_angle': right_foot_angle,
        'left_pitch': left_pitch,
        'right_pitch': right_pitch,
        'left_foot_forward': left_foot_forward * 100,  # cm
        'left_foot_down': left_foot_down * 100,        # cm
        'left_foot_lateral': left_foot_lateral * 100,  # cm
        'right_foot_forward': right_foot_forward * 100,
        'right_foot_down': right_foot_down * 100,
        'right_foot_lateral': right_foot_lateral * 100,
    }


def print_analysis(name, stats):
    """Print analysis results"""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    print(f"\n[Foot Vector Components (ankle → foot)]")
    print(f"  Left foot:")
    print(f"    Forward: {stats['left_foot_forward'].mean():+.2f} cm (std: {stats['left_foot_forward'].std():.2f})")
    print(f"    Down:    {stats['left_foot_down'].mean():+.2f} cm (std: {stats['left_foot_down'].std():.2f})")
    print(f"    Lateral: {stats['left_foot_lateral'].mean():+.2f} cm (std: {stats['left_foot_lateral'].std():.2f})")
    print(f"  Right foot:")
    print(f"    Forward: {stats['right_foot_forward'].mean():+.2f} cm (std: {stats['right_foot_forward'].std():.2f})")
    print(f"    Down:    {stats['right_foot_down'].mean():+.2f} cm (std: {stats['right_foot_down'].std():.2f})")
    print(f"    Lateral: {stats['right_foot_lateral'].mean():+.2f} cm (std: {stats['right_foot_lateral'].std():.2f})")

    print(f"\n[Foot Orientation Angles]")
    print(f"  Left foot angle:  {stats['left_foot_angle'].mean():+.1f}° (std: {stats['left_foot_angle'].std():.1f}°)")
    print(f"  Right foot angle: {stats['right_foot_angle'].mean():+.1f}° (std: {stats['right_foot_angle'].std():.1f}°)")
    print(f"    (Positive = toe-out, Negative = toe-in)")

    print(f"\n[Foot Pitch (pointing down)]")
    print(f"  Left pitch:  {stats['left_pitch'].mean():+.1f}° (std: {stats['left_pitch'].std():.1f}°)")
    print(f"  Right pitch: {stats['right_pitch'].mean():+.1f}° (std: {stats['right_pitch'].std():.1f}°)")
    print(f"    (Positive = pointing down, Negative = pointing up)")

    # Analysis
    print(f"\n[Analysis]")

    # Check lateral component (should be small)
    avg_left_lat = stats['left_foot_lateral'].mean()
    avg_right_lat = stats['right_foot_lateral'].mean()

    if abs(avg_left_lat) > 2.0:
        print(f"  ⚠️  Left foot has large lateral component: {avg_left_lat:+.2f} cm")
    else:
        print(f"  ✓ Left foot lateral component OK: {avg_left_lat:+.2f} cm")

    if abs(avg_right_lat) > 2.0:
        print(f"  ⚠️  Right foot has large lateral component: {avg_right_lat:+.2f} cm")
    else:
        print(f"  ✓ Right foot lateral component OK: {avg_right_lat:+.2f} cm")

    # Check angle (should be small, around 0-15° toe-out is normal)
    avg_left_angle = stats['left_foot_angle'].mean()
    avg_right_angle = stats['right_foot_angle'].mean()

    if abs(avg_left_angle) > 30:
        print(f"  ⚠️  Left foot angle excessive: {avg_left_angle:+.1f}°")
    elif abs(avg_left_angle) > 20:
        print(f"  ⚠️  Left foot angle large: {avg_left_angle:+.1f}° (20-30° is borderline)")
    else:
        print(f"  ✓ Left foot angle OK: {avg_left_angle:+.1f}°")

    if abs(avg_right_angle) > 30:
        print(f"  ⚠️  Right foot angle excessive: {avg_right_angle:+.1f}°")
    elif abs(avg_right_angle) > 20:
        print(f"  ⚠️  Right foot angle large: {avg_right_angle:+.1f}° (20-30° is borderline)")
    else:
        print(f"  ✓ Right foot angle OK: {avg_right_angle:+.1f}°")

    # Check pitch (should be positive, pointing down)
    avg_left_pitch = stats['left_pitch'].mean()
    avg_right_pitch = stats['right_pitch'].mean()

    if avg_left_pitch < 0:
        print(f"  ❌ Left foot pointing UP: {avg_left_pitch:.1f}° (should be positive)")
    elif avg_left_pitch < 20:
        print(f"  ⚠️  Left foot pitch low: {avg_left_pitch:.1f}° (expected 30-60°)")
    else:
        print(f"  ✓ Left foot pitch OK: {avg_left_pitch:.1f}°")

    if avg_right_pitch < 0:
        print(f"  ❌ Right foot pointing UP: {avg_right_pitch:.1f}° (should be positive)")
    elif avg_right_pitch < 20:
        print(f"  ⚠️  Right foot pitch low: {avg_right_pitch:.1f}° (expected 30-60°)")
    else:
        print(f"  ✓ Right foot pitch OK: {avg_right_pitch:.1f}°")


def main():
    smpl_model_path = '/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl'

    versions = {
        'FIX 8 spine2 (back → spine2)': '/tmp/v8_spine2_p020_split5/foot_orient_loss_with_arm_carter2023_p020/smpl_params.npz',
        'FIX 9 (foot orient loss all stages)': '/tmp/v9_foot_orient_p020_split5/v9_foot_orient_with_arm_carter2023_p020/smpl_params.npz',
        'FIX 10 (foot orient loss late-stage)': '/tmp/v10_foot_orient_late_p020_split5/v10_foot_orient_late_with_arm_carter2023_p020/smpl_params.npz',
    }

    print("="*80)
    print("Foot Orientation Analysis - FIX 8 vs FIX 9")
    print("="*80)

    all_stats = {}

    for name, path in versions.items():
        print(f"\n\n[Reading {name}]")
        joints = read_smpl_joints(path, smpl_model_path)
        print(f"Loaded {len(joints)} frames with {joints.shape[1]} joints")

        stats = analyze_foot_orientation(joints)
        all_stats[name] = stats
        print_analysis(name, stats)

    # Comparison
    print(f"\n\n{'='*80}")
    print("Comparison Summary")
    print(f"{'='*80}")

    print(f"\n{'Version':<35} {'L angle':>10} {'R angle':>10} {'L pitch':>10} {'R pitch':>10}")
    print("-"*80)

    for name, stats in all_stats.items():
        l_angle = stats['left_foot_angle'].mean()
        r_angle = stats['right_foot_angle'].mean()
        l_pitch = stats['left_pitch'].mean()
        r_pitch = stats['right_pitch'].mean()
        print(f"{name:<35} {l_angle:+9.1f}° {r_angle:+9.1f}° {l_pitch:+9.1f}° {r_pitch:+9.1f}°")


if __name__ == '__main__':
    main()
