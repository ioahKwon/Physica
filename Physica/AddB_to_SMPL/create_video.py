#!/usr/bin/env python3
"""
Create MP4 video visualization of SMPL fitting results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import json
import argparse
from pathlib import Path


def load_results(result_dir):
    """Load prediction and target joints from result directory."""
    result_dir = Path(result_dir)

    pred_joints = np.load(result_dir / 'pred_joints.npy')  # Shape: (T, J, 3)
    target_joints = np.load(result_dir / 'target_joints.npy')  # Shape: (T, J, 3)

    with open(result_dir / 'meta.json', 'r') as f:
        meta = json.load(f)

    return pred_joints, target_joints, meta


def create_animation(result_dir, output_path=None, fps=30):
    """Create MP4 animation of joint positions over time."""

    pred_joints, target_joints, meta = load_results(result_dir)

    # Handle joint mapping
    joint_mapping = meta.get('joint_mapping', {})
    if pred_joints.shape[1] != target_joints.shape[1] and len(joint_mapping) > 0:
        addb_to_smpl = {int(k): int(v) for k, v in joint_mapping.items()}
        # Get mapped indices
        addb_indices = sorted(addb_to_smpl.keys())
        smpl_indices = [addb_to_smpl[i] for i in addb_indices]

        # Filter both pred and target to only mapped joints
        pred_joints_vis = pred_joints[:, smpl_indices, :]
        target_joints_vis = target_joints[:, addb_indices, :]
    else:
        pred_joints_vis = pred_joints
        target_joints_vis = target_joints

    num_frames = pred_joints_vis.shape[0]

    # Calculate overall bounds for consistent axis scaling
    all_joints = np.concatenate([pred_joints_vis.reshape(-1, 3),
                                  target_joints_vis.reshape(-1, 3)], axis=0)
    x_min, x_max = all_joints[:, 0].min() - 0.2, all_joints[:, 0].max() + 0.2
    y_min, y_max = all_joints[:, 1].min() - 0.2, all_joints[:, 1].max() + 0.2
    z_min, z_max = all_joints[:, 2].min() - 0.2, all_joints[:, 2].max() + 0.2

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))

    def init():
        """Initialize animation."""
        return []

    def update(frame_idx):
        """Update animation for each frame."""
        fig.clear()

        # Calculate MPJPE for this frame
        mpjpe_frame = np.linalg.norm(pred_joints_vis[frame_idx] - target_joints_vis[frame_idx], axis=1).mean() * 1000

        # Top view (XY)
        ax1 = fig.add_subplot(131)
        ax1.scatter(target_joints_vis[frame_idx, :, 0], target_joints_vis[frame_idx, :, 1],
                    c='blue', s=150, alpha=0.7, label='Target (AddB)', edgecolors='darkblue', linewidths=2)
        ax1.scatter(pred_joints_vis[frame_idx, :, 0], pred_joints_vis[frame_idx, :, 1],
                    c='red', s=150, alpha=0.7, label='Predicted (SMPL)', edgecolors='darkred', linewidths=2)

        # Draw lines connecting target and predicted
        for i in range(min(len(target_joints_vis[frame_idx]), len(pred_joints_vis[frame_idx]))):
            ax1.plot([target_joints_vis[frame_idx, i, 0], pred_joints_vis[frame_idx, i, 0]],
                    [target_joints_vis[frame_idx, i, 1], pred_joints_vis[frame_idx, i, 1]],
                    'gray', alpha=0.3, linewidth=1)

        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_xlabel('X (m)', fontsize=11)
        ax1.set_ylabel('Y (m)', fontsize=11)
        ax1.set_title('Top View (XY plane)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Front view (XZ)
        ax2 = fig.add_subplot(132)
        ax2.scatter(target_joints_vis[frame_idx, :, 0], target_joints_vis[frame_idx, :, 2],
                    c='blue', s=150, alpha=0.7, label='Target (AddB)', edgecolors='darkblue', linewidths=2)
        ax2.scatter(pred_joints_vis[frame_idx, :, 0], pred_joints_vis[frame_idx, :, 2],
                    c='red', s=150, alpha=0.7, label='Predicted (SMPL)', edgecolors='darkred', linewidths=2)

        for i in range(min(len(target_joints_vis[frame_idx]), len(pred_joints_vis[frame_idx]))):
            ax2.plot([target_joints_vis[frame_idx, i, 0], pred_joints_vis[frame_idx, i, 0]],
                    [target_joints_vis[frame_idx, i, 2], pred_joints_vis[frame_idx, i, 2]],
                    'gray', alpha=0.3, linewidth=1)

        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(z_min, z_max)
        ax2.set_xlabel('X (m)', fontsize=11)
        ax2.set_ylabel('Z (m)', fontsize=11)
        ax2.set_title('Front View (XZ plane)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        # Side view (YZ)
        ax3 = fig.add_subplot(133)
        ax3.scatter(target_joints_vis[frame_idx, :, 1], target_joints_vis[frame_idx, :, 2],
                    c='blue', s=150, alpha=0.7, label='Target (AddB)', edgecolors='darkblue', linewidths=2)
        ax3.scatter(pred_joints_vis[frame_idx, :, 1], pred_joints_vis[frame_idx, :, 2],
                    c='red', s=150, alpha=0.7, label='Predicted (SMPL)', edgecolors='darkred', linewidths=2)

        for i in range(min(len(target_joints_vis[frame_idx]), len(pred_joints_vis[frame_idx]))):
            ax3.plot([target_joints_vis[frame_idx, i, 1], pred_joints_vis[frame_idx, i, 1]],
                    [target_joints_vis[frame_idx, i, 2], pred_joints_vis[frame_idx, i, 2]],
                    'gray', alpha=0.3, linewidth=1)

        ax3.set_xlim(y_min, y_max)
        ax3.set_ylim(z_min, z_max)
        ax3.set_xlabel('Y (m)', fontsize=11)
        ax3.set_ylabel('Z (m)', fontsize=11)
        ax3.set_title('Side View (YZ plane)', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')

        # Overall title
        mpjpe_overall = meta.get('MPJPE', -1)
        fig.suptitle(f"{meta['run_name']} - Frame {frame_idx+1}/{num_frames}\n"
                     f"Frame MPJPE: {mpjpe_frame:.2f}mm | Overall MPJPE: {mpjpe_overall:.2f}mm",
                     fontsize=15, fontweight='bold')

        plt.tight_layout()
        return []

    # Create animation
    print(f"Creating animation with {num_frames} frames...")
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                        blit=False, repeat=True, interval=1000/fps)

    # Save as MP4
    if output_path is None:
        result_dir = Path(result_dir)
        output_path = result_dir / f"{result_dir.name}_animation.mp4"

    print(f"Saving MP4 to: {output_path}")
    writer = FFMpegWriter(fps=fps, bitrate=5000, codec='libx264')
    anim.save(output_path, writer=writer, dpi=100)

    print(f"✓ Video saved: {output_path}")
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create MP4 video of SMPL fitting results')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Path to result directory containing pred_joints.npy and target_joints.npy')
    parser.add_argument('--output', type=str, help='Output MP4 file path')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')

    args = parser.parse_args()

    create_animation(args.result_dir, args.output, args.fps)


if __name__ == '__main__':
    main()
