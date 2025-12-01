#!/usr/bin/env python3
"""
Visualize foot orientation axes to show the "foot pointing backward" problem.

This script renders the local coordinate frames (XYZ axes) for foot joints
to clearly show their orientation in 3D space.
"""

import numpy as np
import torch
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel


def rodrigues_to_matrix(axis_angle):
    """Convert axis-angle to 3x3 rotation matrix."""
    rot = R.from_rotvec(axis_angle)
    return rot.as_matrix()


def visualize_foot_orientation(smpl_model, betas, poses, trans, frame_idx, output_path):
    """
    Visualize foot orientation by drawing local coordinate axes.

    Args:
        smpl_model: SMPL model instance
        betas: Shape parameters (10,)
        poses: Pose parameters (T, 24, 3)
        trans: Translation (T, 3)
        frame_idx: Which frame to visualize
        output_path: Where to save the visualization
    """
    device = betas.device

    # Get SMPL output for this frame
    with torch.no_grad():
        vertices, joints = smpl_model.forward(
            betas=betas,
            poses=poses[frame_idx],
            trans=trans[frame_idx]
        )

    # Convert to numpy
    joints_np = joints.cpu().numpy()  # (24, 3)
    poses_np = poses[frame_idx].cpu().numpy()  # (24, 3)

    # Compute global rotation matrices for each joint
    # This is simplified - in reality SMPL uses kinematic chain
    # But for visualization purposes, we'll show local rotations

    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    LEFT_FOOT = 10
    RIGHT_FOOT = 11

    # Get joint positions
    left_ankle_pos = joints_np[LEFT_ANKLE]
    right_ankle_pos = joints_np[RIGHT_ANKLE]
    left_foot_pos = joints_np[LEFT_FOOT]
    right_foot_pos = joints_np[RIGHT_FOOT]

    # Get rotation matrices
    left_ankle_rot = rodrigues_to_matrix(poses_np[LEFT_ANKLE])
    right_ankle_rot = rodrigues_to_matrix(poses_np[RIGHT_ANKLE])
    left_foot_rot = rodrigues_to_matrix(poses_np[LEFT_FOOT])
    right_foot_rot = rodrigues_to_matrix(poses_np[RIGHT_FOOT])

    # Create figure
    fig = plt.figure(figsize=(20, 10))

    # Left side view
    ax1 = fig.add_subplot(121, projection='3d')
    plot_foot_axes(ax1, left_ankle_pos, left_ankle_rot, 'Left Ankle', scale=0.1)
    plot_foot_axes(ax1, left_foot_pos, left_foot_rot, 'Left Foot', scale=0.1)

    # Draw connection
    ax1.plot([left_ankle_pos[0], left_foot_pos[0]],
             [left_ankle_pos[1], left_foot_pos[1]],
             [left_ankle_pos[2], left_foot_pos[2]], 'k-', linewidth=3, alpha=0.5)

    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_zlabel('Z (meters)')
    ax1.set_title(f'Left Foot Orientation - Frame {frame_idx}', fontsize=14, fontweight='bold')
    ax1.legend()
    set_axes_equal(ax1)
    ax1.view_init(elev=20, azim=45)

    # Right side view
    ax2 = fig.add_subplot(122, projection='3d')
    plot_foot_axes(ax2, right_ankle_pos, right_ankle_rot, 'Right Ankle', scale=0.1)
    plot_foot_axes(ax2, right_foot_pos, right_foot_rot, 'Right Foot', scale=0.1)

    # Draw connection
    ax2.plot([right_ankle_pos[0], right_foot_pos[0]],
             [right_ankle_pos[1], right_foot_pos[1]],
             [right_ankle_pos[2], right_foot_pos[2]], 'k-', linewidth=3, alpha=0.5)

    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_zlabel('Z (meters)')
    ax2.set_title(f'Right Foot Orientation - Frame {frame_idx}', fontsize=14, fontweight='bold')
    ax2.legend()
    set_axes_equal(ax2)
    ax2.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved foot orientation visualization to {output_path}")

    # Print rotation analysis
    print("\n" + "="*80)
    print(f"FOOT ORIENTATION ANALYSIS - Frame {frame_idx}")
    print("="*80)

    print("\nLeft Ankle:")
    print(f"  Position: [{left_ankle_pos[0]:.3f}, {left_ankle_pos[1]:.3f}, {left_ankle_pos[2]:.3f}]")
    print(f"  Rotation (axis-angle): [{poses_np[LEFT_ANKLE][0]:.4f}, {poses_np[LEFT_ANKLE][1]:.4f}, {poses_np[LEFT_ANKLE][2]:.4f}]")
    print(f"  X-axis (red):   [{left_ankle_rot[0,0]:+.3f}, {left_ankle_rot[1,0]:+.3f}, {left_ankle_rot[2,0]:+.3f}]")
    print(f"  Y-axis (green): [{left_ankle_rot[0,1]:+.3f}, {left_ankle_rot[1,1]:+.3f}, {left_ankle_rot[2,1]:+.3f}]")
    print(f"  Z-axis (blue):  [{left_ankle_rot[0,2]:+.3f}, {left_ankle_rot[1,2]:+.3f}, {left_ankle_rot[2,2]:+.3f}]")

    print("\nLeft Foot:")
    print(f"  Position: [{left_foot_pos[0]:.3f}, {left_foot_pos[1]:.3f}, {left_foot_pos[2]:.3f}]")
    print(f"  Rotation (axis-angle): [{poses_np[LEFT_FOOT][0]:.4f}, {poses_np[LEFT_FOOT][1]:.4f}, {poses_np[LEFT_FOOT][2]:.4f}]")
    print(f"  X-axis (red):   [{left_foot_rot[0,0]:+.3f}, {left_foot_rot[1,0]:+.3f}, {left_foot_rot[2,0]:+.3f}]")
    print(f"  Y-axis (green): [{left_foot_rot[0,1]:+.3f}, {left_foot_rot[1,1]:+.3f}, {left_foot_rot[2,1]:+.3f}]")
    print(f"  Z-axis (blue):  [{left_foot_rot[0,2]:+.3f}, {left_foot_rot[1,2]:+.3f}, {left_foot_rot[2,2]:+.3f}]")

    if np.allclose(poses_np[LEFT_FOOT], 0, atol=1e-6):
        print("  ⚠️  WARNING: Foot rotation is IDENTITY (not optimized!)")

    print("\nRight Ankle:")
    print(f"  Position: [{right_ankle_pos[0]:.3f}, {right_ankle_pos[1]:.3f}, {right_ankle_pos[2]:.3f}]")
    print(f"  Rotation (axis-angle): [{poses_np[RIGHT_ANKLE][0]:.4f}, {poses_np[RIGHT_ANKLE][1]:.4f}, {poses_np[RIGHT_ANKLE][2]:.4f}]")

    print("\nRight Foot:")
    print(f"  Position: [{right_foot_pos[0]:.3f}, {right_foot_pos[1]:.3f}, {right_foot_pos[2]:.3f}]")
    print(f"  Rotation (axis-angle): [{poses_np[RIGHT_FOOT][0]:.4f}, {poses_np[RIGHT_FOOT][1]:.4f}, {poses_np[RIGHT_FOOT][2]:.4f}]")

    if np.allclose(poses_np[RIGHT_FOOT], 0, atol=1e-6):
        print("  ⚠️  WARNING: Foot rotation is IDENTITY (not optimized!)")


def plot_foot_axes(ax, position, rotation_matrix, label, scale=0.1):
    """
    Plot XYZ axes at a given position with given rotation.

    Args:
        ax: Matplotlib 3D axis
        position: (3,) position vector
        rotation_matrix: (3, 3) rotation matrix
        label: Label for this coordinate frame
        scale: Length of axes arrows
    """
    origin = position

    # Compute axis endpoints
    x_axis = origin + scale * rotation_matrix[:, 0]
    y_axis = origin + scale * rotation_matrix[:, 1]
    z_axis = origin + scale * rotation_matrix[:, 2]

    # Plot axes
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]],
            'r-', linewidth=4, label=f'{label} X')
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]],
            'g-', linewidth=4, label=f'{label} Y')
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]],
            'b-', linewidth=4, label=f'{label} Z')

    # Plot joint position
    ax.scatter(*origin, c='black', s=200, marker='o', edgecolors='white', linewidths=2)


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def main():
    parser = argparse.ArgumentParser(description='Visualize foot orientation axes')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Path to result directory containing smpl_params.npz')
    parser.add_argument('--smpl_model', type=str, default='models/smpl_model.pkl',
                       help='Path to SMPL model')
    parser.add_argument('--frame', type=int, default=0,
                       help='Frame to visualize')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for visualization')
    args = parser.parse_args()

    result_path = Path(args.result_dir)
    smpl_params_file = result_path / 'smpl_params.npz'

    if not smpl_params_file.exists():
        print(f"ERROR: {smpl_params_file} does not exist")
        sys.exit(1)

    device = torch.device('cpu')

    print("Loading SMPL model...")
    smpl = SMPLModel(args.smpl_model, device=device)

    print("Loading SMPL parameters...")
    data = np.load(smpl_params_file)
    betas = torch.from_numpy(data['betas']).float().to(device)
    poses = torch.from_numpy(data['poses']).float().to(device)
    trans = torch.from_numpy(data['trans']).float().to(device)

    T = poses.shape[0]
    print(f"Loaded {T} frames")

    if args.frame >= T:
        print(f"ERROR: Frame {args.frame} is out of range (max: {T-1})")
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        output_path = result_path / f'foot_orientation_frame_{args.frame:03d}.png'

    print(f"\nVisualizing frame {args.frame}...")
    visualize_foot_orientation(smpl, betas, poses, trans, args.frame, output_path)


if __name__ == '__main__':
    main()
