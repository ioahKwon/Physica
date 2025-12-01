#!/usr/bin/env python3
"""
Analyze AddBiomechanics GT foot orientation
"""

import numpy as np


def analyze_addb_foot_orientation(joints):
    """
    Analyze AddBiomechanics foot orientation

    AddBiomechanics joint structure (20 joints):
    - Right leg: hip_r(1) → walker_knee_r(2) → ankle_r(3) → subtalar_r(4) → mtp_r(5)
    - Left leg:  hip_l(6) → walker_knee_l(7) → ankle_l(8) → subtalar_l(9) → mtp_l(10)

    Coordinate system (same as SMPL):
    - X: right(+) / left(-)
    - Y: up(+) / down(-)
    - Z: back(+) / front(-)
    """
    # Right leg
    right_ankle = joints[:, 3]   # [T, 3]
    right_mtp = joints[:, 5]     # [T, 3] (toe)

    # Left leg
    left_ankle = joints[:, 8]    # [T, 3]
    left_mtp = joints[:, 10]     # [T, 3] (toe)

    # Foot vectors (ankle → toe)
    left_foot_vec = left_mtp - left_ankle       # [T, 3]
    right_foot_vec = right_mtp - right_ankle    # [T, 3]

    # Decompose foot vectors
    left_foot_forward = -left_foot_vec[:, 2]    # negative Z = forward
    left_foot_down = -left_foot_vec[:, 1]       # negative Y = down
    left_foot_lateral = left_foot_vec[:, 0]     # positive X = lateral

    right_foot_forward = -right_foot_vec[:, 2]
    right_foot_down = -right_foot_vec[:, 1]
    right_foot_lateral = right_foot_vec[:, 0]

    # Calculate foot angle in XZ plane
    left_foot_angle = np.arctan2(left_foot_lateral, left_foot_forward) * 180 / np.pi
    right_foot_angle = np.arctan2(-right_foot_lateral, right_foot_forward) * 180 / np.pi

    # Calculate foot pitch
    left_foot_len = np.linalg.norm(left_foot_vec, axis=1)
    right_foot_len = np.linalg.norm(right_foot_vec, axis=1)

    left_down_ratio = left_foot_down / left_foot_len
    right_down_ratio = right_foot_down / right_foot_len

    left_pitch = np.arcsin(np.clip(left_down_ratio, -1, 1)) * 180 / np.pi
    right_pitch = np.arcsin(np.clip(right_down_ratio, -1, 1)) * 180 / np.pi

    # Also check subtalar joints
    right_subtalar = joints[:, 4]
    left_subtalar = joints[:, 9]

    # Ankle → subtalar vectors
    left_subtalar_vec = left_subtalar - left_ankle
    right_subtalar_vec = right_subtalar - right_ankle

    left_subtalar_forward = -left_subtalar_vec[:, 2]
    left_subtalar_lateral = left_subtalar_vec[:, 0]
    right_subtalar_forward = -right_subtalar_vec[:, 2]
    right_subtalar_lateral = right_subtalar_vec[:, 0]

    left_subtalar_angle = np.arctan2(left_subtalar_lateral, left_subtalar_forward) * 180 / np.pi
    right_subtalar_angle = np.arctan2(-right_subtalar_lateral, right_subtalar_forward) * 180 / np.pi

    return {
        # Ankle → MTP (full foot)
        'left_foot_angle': left_foot_angle,
        'right_foot_angle': right_foot_angle,
        'left_pitch': left_pitch,
        'right_pitch': right_pitch,
        'left_foot_forward': left_foot_forward * 100,   # cm
        'left_foot_down': left_foot_down * 100,         # cm
        'left_foot_lateral': left_foot_lateral * 100,   # cm
        'right_foot_forward': right_foot_forward * 100,
        'right_foot_down': right_foot_down * 100,
        'right_foot_lateral': right_foot_lateral * 100,
        # Ankle → subtalar
        'left_subtalar_angle': left_subtalar_angle,
        'right_subtalar_angle': right_subtalar_angle,
    }


def print_analysis(name, stats):
    """Print analysis results"""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    print(f"\n[Foot Vector Components (ankle → toe/MTP)]")
    print(f"  Left foot:")
    print(f"    Forward: {stats['left_foot_forward'].mean():+.2f} cm (std: {stats['left_foot_forward'].std():.2f})")
    print(f"    Down:    {stats['left_foot_down'].mean():+.2f} cm (std: {stats['left_foot_down'].std():.2f})")
    print(f"    Lateral: {stats['left_foot_lateral'].mean():+.2f} cm (std: {stats['left_foot_lateral'].std():.2f})")
    print(f"  Right foot:")
    print(f"    Forward: {stats['right_foot_forward'].mean():+.2f} cm (std: {stats['right_foot_forward'].std():.2f})")
    print(f"    Down:    {stats['right_foot_down'].mean():+.2f} cm (std: {stats['right_foot_down'].std():.2f})")
    print(f"    Lateral: {stats['right_foot_lateral'].mean():+.2f} cm (std: {stats['right_foot_lateral'].std():.2f})")

    print(f"\n[Foot Orientation Angles (ankle → toe)]")
    print(f"  Left foot angle:  {stats['left_foot_angle'].mean():+.1f}° (std: {stats['left_foot_angle'].std():.1f}°)")
    print(f"  Right foot angle: {stats['right_foot_angle'].mean():+.1f}° (std: {stats['right_foot_angle'].std():.1f}°)")
    print(f"    (Positive = toe-out, Negative = toe-in)")

    print(f"\n[Subtalar Orientation (ankle → subtalar)]")
    print(f"  Left subtalar angle:  {stats['left_subtalar_angle'].mean():+.1f}° (std: {stats['left_subtalar_angle'].std():.1f}°)")
    print(f"  Right subtalar angle: {stats['right_subtalar_angle'].mean():+.1f}° (std: {stats['right_subtalar_angle'].std():.1f}°)")

    print(f"\n[Foot Pitch (pointing down)]")
    print(f"  Left pitch:  {stats['left_pitch'].mean():+.1f}° (std: {stats['left_pitch'].std():.1f}°)")
    print(f"  Right pitch: {stats['right_pitch'].mean():+.1f}° (std: {stats['right_pitch'].std():.1f}°)")
    print(f"    (Positive = pointing down, Negative = pointing up)")

    # Analysis
    print(f"\n[Analysis]")

    avg_left_angle = stats['left_foot_angle'].mean()
    avg_right_angle = stats['right_foot_angle'].mean()

    if abs(avg_left_angle) > 30:
        print(f"  ⚠️  Left foot angle excessive: {avg_left_angle:+.1f}°")
    elif abs(avg_left_angle) > 20:
        print(f"  ⚠️  Left foot angle large: {avg_left_angle:+.1f}° (borderline)")
    else:
        print(f"  ✓ Left foot angle OK: {avg_left_angle:+.1f}°")

    if abs(avg_right_angle) > 30:
        print(f"  ⚠️  Right foot angle excessive: {avg_right_angle:+.1f}°")
    elif abs(avg_right_angle) > 20:
        print(f"  ⚠️  Right foot angle large: {avg_right_angle:+.1f}° (borderline)")
    else:
        print(f"  ✓ Right foot angle OK: {avg_right_angle:+.1f}°")


def main():
    # Load AddBiomechanics GT data
    target_joints_path = '/tmp/v8_spine2_p020_split5/foot_orient_loss_with_arm_carter2023_p020/target_joints.npy'

    print("="*80)
    print("AddBiomechanics GT Foot Orientation Analysis")
    print("="*80)
    print(f"Input: {target_joints_path}")
    print()

    joints = np.load(target_joints_path)  # [T, N, 3]
    print(f"Loaded GT joints: shape={joints.shape}")
    print(f"  Frames: {len(joints)}")
    print(f"  Joints: {joints.shape[1]}")

    # Check for NaN
    nan_count = np.isnan(joints).sum()
    if nan_count > 0:
        print(f"  Warning: {nan_count} NaN values found")

    stats = analyze_addb_foot_orientation(joints)
    print_analysis("AddBiomechanics GT (P020_split5)", stats)

    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"  Left foot angle:  {stats['left_foot_angle'].mean():+.1f}° ± {stats['left_foot_angle'].std():.1f}°")
    print(f"  Right foot angle: {stats['right_foot_angle'].mean():+.1f}° ± {stats['right_foot_angle'].std():.1f}°")
    print(f"  Left pitch:       {stats['left_pitch'].mean():+.1f}° ± {stats['left_pitch'].std():.1f}°")
    print(f"  Right pitch:      {stats['right_pitch'].mean():+.1f}° ± {stats['right_pitch'].std():.1f}°")


if __name__ == '__main__':
    main()
