#!/usr/bin/env python3
"""
Visualize SMPL fitting results: comparison of predicted vs target joints
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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


def plot_frame_comparison(pred_joints, target_joints, frame_idx, meta, save_path=None):
    """Plot a single frame comparison between predicted and target joints."""
    fig = plt.figure(figsize=(16, 6))

    # Handle joint mapping: pred_joints has 24 joints (SMPL), target_joints has fewer (AddB)
    joint_mapping = meta.get('joint_mapping', {})

    if pred_joints.shape[1] != target_joints.shape[1] and len(joint_mapping) > 0:
        # Map AddB indices to SMPL indices
        addb_to_smpl = {int(k): int(v) for k, v in joint_mapping.items()}
        smpl_indices = [addb_to_smpl[i] for i in sorted(addb_to_smpl.keys())]
        pred_joints_vis = pred_joints[:, smpl_indices, :]
    else:
        pred_joints_vis = pred_joints

    # Get MPJPE for this frame
    mpjpe_frame = np.linalg.norm(pred_joints_vis[frame_idx] - target_joints[frame_idx], axis=1).mean() * 1000

    # Plot 1: Top view
    ax1 = fig.add_subplot(131)
    ax1.scatter(target_joints[frame_idx, :, 0], target_joints[frame_idx, :, 1],
                c='blue', s=100, alpha=0.6, label='Target (AddB)')
    ax1.scatter(pred_joints_vis[frame_idx, :, 0], pred_joints_vis[frame_idx, :, 1],
                c='red', s=100, alpha=0.6, label='Predicted (SMPL)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Top View (XY plane)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Plot 2: Front view
    ax2 = fig.add_subplot(132)
    ax2.scatter(target_joints[frame_idx, :, 0], target_joints[frame_idx, :, 2],
                c='blue', s=100, alpha=0.6, label='Target (AddB)')
    ax2.scatter(pred_joints_vis[frame_idx, :, 0], pred_joints_vis[frame_idx, :, 2],
                c='red', s=100, alpha=0.6, label='Predicted (SMPL)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Front View (XZ plane)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # Plot 3: Side view
    ax3 = fig.add_subplot(133)
    ax3.scatter(target_joints[frame_idx, :, 1], target_joints[frame_idx, :, 2],
                c='blue', s=100, alpha=0.6, label='Target (AddB)')
    ax3.scatter(pred_joints_vis[frame_idx, :, 1], pred_joints_vis[frame_idx, :, 2],
                c='red', s=100, alpha=0.6, label='Predicted (SMPL)')
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (YZ plane)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    # Overall title
    mpjpe_overall = meta.get('MPJPE', -1)
    fig.suptitle(f"{meta['run_name']} - Frame {frame_idx}\n"
                 f"Frame MPJPE: {mpjpe_frame:.2f}mm | Overall MPJPE: {mpjpe_overall:.2f}mm",
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def create_mpjpe_plot(result_dirs, save_path=None):
    """Create bar plot comparing MPJPE across multiple subjects."""
    subjects = []
    mpjpes = []

    for result_dir in result_dirs:
        result_dir = Path(result_dir)
        with open(result_dir / 'meta.json', 'r') as f:
            meta = json.load(f)

        subject_name = result_dir.name.replace('train_', '').replace('test_', '')
        subjects.append(subject_name)
        mpjpes.append(meta.get('MPJPE', -1))

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(subjects)), mpjpes, color='steelblue', alpha=0.7)

    # Add value labels on bars
    for i, (bar, mpjpe) in enumerate(zip(bars, mpjpes)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mpjpe:.2f}mm',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add average line
    avg_mpjpe = np.mean(mpjpes)
    ax.axhline(y=avg_mpjpe, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_mpjpe:.2f}mm')

    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('MPJPE (mm)', fontsize=12)
    ax.set_title('SMPL Fitting Performance (True Minibatch)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved MPJPE comparison to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize SMPL fitting results')
    parser.add_argument('--result_dir', type=str, help='Path to single result directory')
    parser.add_argument('--batch_dir', type=str, help='Path to batch results directory (contains multiple subjects)')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to visualize')
    parser.add_argument('--output', type=str, help='Output path for saving visualization')

    args = parser.parse_args()

    if args.batch_dir:
        # Batch visualization: compare MPJPE across subjects
        batch_dir = Path(args.batch_dir)
        result_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir()])

        if not result_dirs:
            print(f"No result directories found in {batch_dir}")
            return

        # Create MPJPE comparison plot
        output_path = args.output or str(batch_dir / 'mpjpe_comparison.png')
        create_mpjpe_plot(result_dirs, save_path=output_path)

        # Also create individual frame visualizations for each subject
        for result_dir in result_dirs:
            try:
                pred_joints, target_joints, meta = load_results(result_dir)
                frame_output = result_dir / f'frame_{args.frame}_comparison.png'
                plot_frame_comparison(pred_joints, target_joints, args.frame, meta,
                                     save_path=str(frame_output))
            except Exception as e:
                print(f"Error visualizing {result_dir.name}: {e}")

    elif args.result_dir:
        # Single subject visualization
        pred_joints, target_joints, meta = load_results(args.result_dir)
        plot_frame_comparison(pred_joints, target_joints, args.frame, meta,
                             save_path=args.output)
    else:
        print("Please provide either --result_dir or --batch_dir")
        parser.print_help()


if __name__ == '__main__':
    main()
