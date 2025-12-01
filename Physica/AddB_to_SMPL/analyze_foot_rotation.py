#!/usr/bin/env python3
"""
Analyze foot rotation angles from SMPL fitting results.

This script extracts and analyzes the rotation angles for ankle and foot joints
to diagnose the "foot pointing backward" problem reported by the professor.

Usage:
    python analyze_foot_rotation.py --result_dir <path_to_results>
"""

import numpy as np
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def axis_angle_to_euler(axis_angle):
    """
    Convert axis-angle representation to Euler angles (degrees).

    Args:
        axis_angle: (3,) array in axis-angle format

    Returns:
        Euler angles (rx, ry, rz) in degrees
    """
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-6:
        return np.zeros(3)

    axis = axis_angle / angle
    rot = R.from_rotvec(axis_angle)
    euler = rot.as_euler('xyz', degrees=True)
    return euler


def analyze_foot_rotations(smpl_params_path, frames_to_check=None):
    """
    Analyze foot rotation angles from SMPL parameters.

    Args:
        smpl_params_path: Path to smpl_params.npz
        frames_to_check: List of frame indices to analyze (default: [0, 50, 100, 150])

    Returns:
        Dictionary with rotation analysis results
    """
    # Load SMPL parameters
    data = np.load(smpl_params_path)
    poses = data['poses']  # Shape: (T, 24, 3) - axis-angle for each joint

    T = poses.shape[0]
    print(f"Total frames: {T}")

    if frames_to_check is None:
        # Sample frames evenly
        frames_to_check = [0, T//4, T//2, 3*T//4, T-1]

    # SMPL joint indices
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    LEFT_FOOT = 10
    RIGHT_FOOT = 11

    results = {
        'frames': frames_to_check,
        'left_ankle': [],
        'right_ankle': [],
        'left_foot': [],
        'right_foot': []
    }

    print("\n" + "="*80)
    print("FOOT ROTATION ANALYSIS")
    print("="*80)

    for frame_idx in frames_to_check:
        if frame_idx >= T:
            continue

        print(f"\n--- Frame {frame_idx} ---")

        # Extract rotation for each foot joint
        left_ankle_aa = poses[frame_idx, LEFT_ANKLE]
        right_ankle_aa = poses[frame_idx, RIGHT_ANKLE]
        left_foot_aa = poses[frame_idx, LEFT_FOOT]
        right_foot_aa = poses[frame_idx, RIGHT_FOOT]

        # Convert to Euler angles
        left_ankle_euler = axis_angle_to_euler(left_ankle_aa)
        right_ankle_euler = axis_angle_to_euler(right_ankle_aa)
        left_foot_euler = axis_angle_to_euler(left_foot_aa)
        right_foot_euler = axis_angle_to_euler(right_foot_aa)

        # Store results
        results['left_ankle'].append(left_ankle_euler)
        results['right_ankle'].append(right_ankle_euler)
        results['left_foot'].append(left_foot_euler)
        results['right_foot'].append(right_foot_euler)

        # Print analysis
        print(f"\nLeft Ankle (Joint {LEFT_ANKLE}):")
        print(f"  Axis-angle: [{left_ankle_aa[0]:+.4f}, {left_ankle_aa[1]:+.4f}, {left_ankle_aa[2]:+.4f}]")
        print(f"  Magnitude:  {np.linalg.norm(left_ankle_aa):.4f} rad ({np.linalg.norm(left_ankle_aa)*180/np.pi:.1f}°)")
        print(f"  Euler XYZ:  [{left_ankle_euler[0]:+6.1f}°, {left_ankle_euler[1]:+6.1f}°, {left_ankle_euler[2]:+6.1f}°]")

        print(f"\nRight Ankle (Joint {RIGHT_ANKLE}):")
        print(f"  Axis-angle: [{right_ankle_aa[0]:+.4f}, {right_ankle_aa[1]:+.4f}, {right_ankle_aa[2]:+.4f}]")
        print(f"  Magnitude:  {np.linalg.norm(right_ankle_aa):.4f} rad ({np.linalg.norm(right_ankle_aa)*180/np.pi:.1f}°)")
        print(f"  Euler XYZ:  [{right_ankle_euler[0]:+6.1f}°, {right_ankle_euler[1]:+6.1f}°, {right_ankle_euler[2]:+6.1f}°]")

        print(f"\nLeft Foot (Joint {LEFT_FOOT}):")
        print(f"  Axis-angle: [{left_foot_aa[0]:+.4f}, {left_foot_aa[1]:+.4f}, {left_foot_aa[2]:+.4f}]")
        print(f"  Magnitude:  {np.linalg.norm(left_foot_aa):.4f} rad ({np.linalg.norm(left_foot_aa)*180/np.pi:.1f}°)")
        print(f"  Euler XYZ:  [{left_foot_euler[0]:+6.1f}°, {left_foot_euler[1]:+6.1f}°, {left_foot_euler[2]:+6.1f}°]")

        print(f"\nRight Foot (Joint {RIGHT_FOOT}):")
        print(f"  Axis-angle: [{right_foot_aa[0]:+.4f}, {right_foot_aa[1]:+.4f}, {right_foot_aa[2]:+.4f}]")
        print(f"  Magnitude:  {np.linalg.norm(right_foot_aa):.4f} rad ({np.linalg.norm(right_foot_aa)*180/np.pi:.1f}°)")
        print(f"  Euler XYZ:  [{right_foot_euler[0]:+6.1f}°, {right_foot_euler[1]:+6.1f}°, {right_foot_euler[2]:+6.1f}°]")

        # Flag abnormal angles
        abnormal = []
        for name, euler in [('Left Ankle', left_ankle_euler), ('Right Ankle', right_ankle_euler),
                            ('Left Foot', left_foot_euler), ('Right Foot', right_foot_euler)]:
            if np.any(np.abs(euler) > 90):
                abnormal.append(f"{name}: {euler}")

        if abnormal:
            print("\n⚠️  ABNORMAL ANGLES DETECTED (>90°):")
            for item in abnormal:
                print(f"    {item}")

    return results


def plot_rotation_over_time(smpl_params_path, output_path):
    """
    Plot foot rotation angles over all frames.
    """
    data = np.load(smpl_params_path)
    poses = data['poses']
    T = poses.shape[0]

    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    LEFT_FOOT = 10
    RIGHT_FOOT = 11

    # Extract all rotations
    left_ankle_angles = []
    right_ankle_angles = []
    left_foot_angles = []
    right_foot_angles = []

    for t in range(T):
        left_ankle_angles.append(axis_angle_to_euler(poses[t, LEFT_ANKLE]))
        right_ankle_angles.append(axis_angle_to_euler(poses[t, RIGHT_ANKLE]))
        left_foot_angles.append(axis_angle_to_euler(poses[t, LEFT_FOOT]))
        right_foot_angles.append(axis_angle_to_euler(poses[t, RIGHT_FOOT]))

    left_ankle_angles = np.array(left_ankle_angles)
    right_ankle_angles = np.array(right_ankle_angles)
    left_foot_angles = np.array(left_foot_angles)
    right_foot_angles = np.array(right_foot_angles)

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Foot Rotation Angles Over Time', fontsize=16, fontweight='bold')

    frames = np.arange(T)

    # Left Ankle
    ax = axes[0, 0]
    ax.plot(frames, left_ankle_angles[:, 0], 'r-', label='X (Flexion/Extension)', linewidth=2)
    ax.plot(frames, left_ankle_angles[:, 1], 'g-', label='Y (Internal/External)', linewidth=2)
    ax.plot(frames, left_ankle_angles[:, 2], 'b-', label='Z (Inversion/Eversion)', linewidth=2)
    ax.axhline(y=90, color='k', linestyle='--', alpha=0.3, label='±90° threshold')
    ax.axhline(y=-90, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Left Ankle (Joint 7)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right Ankle
    ax = axes[0, 1]
    ax.plot(frames, right_ankle_angles[:, 0], 'r-', label='X (Flexion/Extension)', linewidth=2)
    ax.plot(frames, right_ankle_angles[:, 1], 'g-', label='Y (Internal/External)', linewidth=2)
    ax.plot(frames, right_ankle_angles[:, 2], 'b-', label='Z (Inversion/Eversion)', linewidth=2)
    ax.axhline(y=90, color='k', linestyle='--', alpha=0.3, label='±90° threshold')
    ax.axhline(y=-90, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Right Ankle (Joint 8)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Left Foot
    ax = axes[1, 0]
    ax.plot(frames, left_foot_angles[:, 0], 'r-', label='X (Toe Up/Down)', linewidth=2)
    ax.plot(frames, left_foot_angles[:, 1], 'g-', label='Y (Foot In/Out)', linewidth=2)
    ax.plot(frames, left_foot_angles[:, 2], 'b-', label='Z (Foot Roll)', linewidth=2)
    ax.axhline(y=90, color='k', linestyle='--', alpha=0.3, label='±90° threshold')
    ax.axhline(y=-90, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Left Foot (Joint 10)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right Foot
    ax = axes[1, 1]
    ax.plot(frames, right_foot_angles[:, 0], 'r-', label='X (Toe Up/Down)', linewidth=2)
    ax.plot(frames, right_foot_angles[:, 1], 'g-', label='Y (Foot In/Out)', linewidth=2)
    ax.plot(frames, right_foot_angles[:, 2], 'b-', label='Z (Foot Roll)', linewidth=2)
    ax.axhline(y=90, color='k', linestyle='--', alpha=0.3, label='±90° threshold')
    ax.axhline(y=-90, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Right Foot (Joint 11)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved rotation plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze foot rotation angles from SMPL results')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Path to result directory containing smpl_params.npz')
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                       help='Specific frames to analyze (default: evenly sampled)')
    parser.add_argument('--plot', type=str, default=None,
                       help='Output path for rotation plot (default: result_dir/foot_rotations.png)')
    args = parser.parse_args()

    result_path = Path(args.result_dir)
    smpl_params_file = result_path / 'smpl_params.npz'

    if not smpl_params_file.exists():
        print(f"ERROR: {smpl_params_file} does not exist")
        sys.exit(1)

    print(f"Analyzing: {smpl_params_file}")

    # Analyze rotations
    results = analyze_foot_rotations(smpl_params_file, frames_to_check=args.frames)

    # Generate plot
    if args.plot:
        plot_path = args.plot
    else:
        plot_path = result_path / 'foot_rotations.png'

    plot_rotation_over_time(smpl_params_file, plot_path)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
