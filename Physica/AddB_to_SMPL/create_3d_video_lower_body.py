#!/usr/bin/env python3
"""
Create 3D rotating visualization of SMPL fitting results - LOWER BODY ONLY
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import json
import argparse
from pathlib import Path


# SMPL skeleton connections for LOWER BODY ONLY
SMPL_SKELETON_LOWER = [
    (0, 1), (0, 2),  # pelvis to hips
    (1, 4), (2, 5),  # hips to knees
    (4, 7), (5, 8),  # knees to ankles
    (7, 10), (8, 11),  # ankles to feet
]


def load_results(result_dir):
    """Load prediction and target results"""
    result_dir = Path(result_dir)

    pred_joints = np.load(result_dir / 'pred_joints.npy')  # [T, 24, 3]
    target_joints = np.load(result_dir / 'target_joints.npy')  # [T, N, 3]

    with open(result_dir / 'meta.json', 'r') as f:
        meta = json.load(f)

    return pred_joints, target_joints, meta


def create_3d_animation_lower_body(result_dir, output_path=None, fps=30):
    """
    Create 3D rotating animation - LOWER BODY ONLY

    Args:
        result_dir: Path to result directory
        output_path: Output MP4 path
        fps: Frames per second
    """
    pred_joints, target_joints, meta = load_results(result_dir)

    # Handle joint mapping
    joint_mapping = meta.get('joint_mapping', {})
    if pred_joints.shape[1] != target_joints.shape[1] and len(joint_mapping) > 0:
        addb_to_smpl = {int(k): int(v) for k, v in joint_mapping.items()}
        addb_indices = sorted(addb_to_smpl.keys())
        smpl_indices = [addb_to_smpl[i] for i in addb_indices]

        pred_joints_vis = pred_joints[:, smpl_indices, :]
        target_joints_vis = target_joints[:, addb_indices, :]
    else:
        pred_joints_vis = pred_joints
        target_joints_vis = target_joints

    num_frames = pred_joints.shape[0]

    # Extract LOWER BODY joints only (indices 0-11: pelvis, hips, knees, ankles, feet)
    lower_body_indices = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    pred_lower = pred_joints[:, lower_body_indices, :]

    # Compute global bounds for lower body only
    all_joints = np.concatenate([pred_lower.reshape(-1, 3),
                                   target_joints.reshape(-1, 3)], axis=0)
    all_joints = all_joints[~np.isnan(all_joints).any(axis=1)]

    x_min, y_min, z_min = all_joints.min(axis=0)
    x_max, y_max, z_max = all_joints.max(axis=0)

    # Add padding
    padding = 0.2
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range
    z_min -= padding * z_range
    z_max += padding * z_range

    # Setup figure
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_data):
        frame_idx, angle = frame_data
        ax.clear()

        # Get current frame (LOWER BODY ONLY)
        pred = pred_joints[frame_idx]
        target_vis = target_joints_vis[frame_idx]

        # Plot predicted skeleton (LOWER BODY ONLY)
        for i, j in SMPL_SKELETON_LOWER:
            if i < pred.shape[0] and j < pred.shape[0]:
                if not (np.isnan(pred[i]).any() or np.isnan(pred[j]).any()):
                    ax.plot([pred[i, 0], pred[j, 0]],
                           [pred[i, 1], pred[j, 1]],
                           [pred[i, 2], pred[j, 2]],
                           'b-', linewidth=3, alpha=0.8)

        # Plot predicted joints (LOWER BODY ONLY)
        pred_lower_valid = pred[lower_body_indices]
        pred_lower_valid = pred_lower_valid[~np.isnan(pred_lower_valid).any(axis=1)]
        if len(pred_lower_valid) > 0:
            ax.scatter(pred_lower_valid[:, 0], pred_lower_valid[:, 1], pred_lower_valid[:, 2],
                      c='blue', s=80, marker='o', label='Predicted (SMPL)', alpha=0.9, edgecolors='darkblue', linewidths=1.5)

        # Plot target joints
        valid_target = target_vis[~np.isnan(target_vis).any(axis=1)]
        if len(valid_target) > 0:
            ax.scatter(valid_target[:, 0], valid_target[:, 1], valid_target[:, 2],
                      c='red', s=80, marker='^', label='Target (AddB)', alpha=0.9, edgecolors='darkred', linewidths=1.5)

        # Draw connections between corresponding joints
        for k in range(min(pred_joints_vis.shape[1], target_vis.shape[0])):
            if k < pred_joints_vis.shape[1]:
                pred_pt = pred_joints_vis[frame_idx, k]
                target_pt = target_vis[k]
                if not (np.isnan(pred_pt).any() or np.isnan(target_pt).any()):
                    ax.plot([pred_pt[0], target_pt[0]],
                           [pred_pt[1], target_pt[1]],
                           [pred_pt[2], target_pt[2]],
                           'gray', linestyle='--', linewidth=1, alpha=0.4)

        # Compute frame MPJPE
        valid_mask = ~(np.isnan(pred_joints_vis[frame_idx]).any(axis=1) |
                       np.isnan(target_vis).any(axis=1))
        if valid_mask.sum() > 0:
            mpjpe_frame = np.linalg.norm(
                pred_joints_vis[frame_idx][valid_mask] - target_vis[valid_mask],
                axis=1
            ).mean() * 1000
        else:
            mpjpe_frame = 0.0

        # Set fixed bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # Set viewing angle (rotate around Z axis)
        ax.view_init(elev=20, azim=angle)

        # Labels and title
        ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=11, fontweight='bold')

        subject_name = meta.get('run_name', 'Unknown')
        overall_mpjpe = meta.get('MPJPE', 0.0)

        ax.set_title(
            f'{subject_name} - LOWER BODY ONLY\n'
            f'Frame {frame_idx + 1}/{num_frames} | '
            f'Frame MPJPE: {mpjpe_frame:.2f}mm | '
            f'Overall MPJPE: {overall_mpjpe:.2f}mm',
            fontsize=13, fontweight='bold', pad=20
        )

        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Add ground plane
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
                              np.linspace(y_min, y_max, 10))
        zz = np.ones_like(xx) * z_min
        ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

        return ax,

    # Generate frame data: (frame_idx, rotation_angle)
    angles = np.linspace(0, 360, num_frames)
    frame_data = [(i, angle) for i, angle in enumerate(angles)]

    print(f'Creating 3D LOWER BODY animation with {num_frames} frames...')
    anim = FuncAnimation(fig, update, frames=frame_data,
                        interval=1000/fps, blit=False)

    # Save
    if output_path is None:
        output_path = Path(result_dir) / 'animation_3d_lower_body.mp4'
    else:
        output_path = Path(output_path)

    print(f'Saving 3D MP4 to: {output_path}')
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(str(output_path), writer=writer)

    plt.close()
    print(f'✓ 3D Lower Body Video saved: {output_path}')

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create 3D rotating visualization - LOWER BODY ONLY')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Path to result directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output MP4 path')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')

    args = parser.parse_args()
    create_3d_animation_lower_body(args.result_dir, args.output, args.fps)


if __name__ == '__main__':
    main()
