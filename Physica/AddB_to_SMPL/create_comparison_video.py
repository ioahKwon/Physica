#!/usr/bin/env python3
"""
Create side-by-side 3D animation comparing V2 baseline and V2 + offset
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import json
import os
import argparse

# SMPL skeleton connections (parent joint for each joint)
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
    18, 19, 20, 21
]

def load_results(baseline_dir, offset_dir):
    """Load V2 baseline and offset results"""
    # Baseline
    pred_baseline = np.load(os.path.join(baseline_dir, 'pred_joints.npy'))
    target = np.load(os.path.join(baseline_dir, 'target_joints.npy'))
    smpl_params_baseline = np.load(os.path.join(baseline_dir, 'smpl_params.npz'))

    with open(os.path.join(baseline_dir, 'meta.json'), 'r') as f:
        meta_baseline = json.load(f)

    # Offset
    pred_offset = np.load(os.path.join(offset_dir, 'pred_joints_with_offset.npy'))
    smpl_params_offset = np.load(os.path.join(offset_dir, 'smpl_params_with_offset.npz'))

    with open(os.path.join(offset_dir, 'meta_with_offset.json'), 'r') as f:
        meta_offset = json.load(f)

    return pred_baseline, pred_offset, target, smpl_params_baseline, smpl_params_offset, meta_baseline, meta_offset

def compute_frame_error(pred, target, joint_mapping):
    """Compute MPJPE for a single frame"""
    errors = []
    for addb_idx_str, smpl_idx in joint_mapping.items():
        addb_idx = int(addb_idx_str)
        pred_joint = pred[smpl_idx, :]
        target_joint = target[addb_idx, :]

        if not np.isnan(target_joint).any():
            error = np.linalg.norm(pred_joint - target_joint)
            errors.append(error)

    return np.mean(errors) * 1000 if errors else 0.0

def create_animation(pred_baseline, pred_offset, target, joint_mapping, meta_baseline, meta_offset, output_path, fps=30, subsample=1):
    """Create side-by-side 3D animation"""
    n_frames = pred_baseline.shape[0]

    # Subsample frames if needed
    frame_indices = list(range(0, n_frames, subsample))

    # Extract mapped joints
    addb_indices = [int(k) for k in sorted(joint_mapping.keys(), key=int)]
    smpl_indices = [joint_mapping[str(k)] for k in addb_indices]

    # Set up figure with 3 subplots
    fig = plt.figure(figsize=(20, 6))

    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    # Compute global bounds for consistent view
    all_coords = np.concatenate([
        pred_baseline[:, smpl_indices, :].reshape(-1, 3),
        pred_offset[:, smpl_indices, :].reshape(-1, 3),
        target[:, addb_indices, :].reshape(-1, 3)
    ], axis=0)

    x_min, x_max = np.nanmin(all_coords[:, 0]), np.nanmax(all_coords[:, 0])
    y_min, y_max = np.nanmin(all_coords[:, 1]), np.nanmax(all_coords[:, 1])
    z_min, z_max = np.nanmin(all_coords[:, 2]), np.nanmax(all_coords[:, 2])

    # Add padding
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range
    z_min -= padding * z_range
    z_max += padding * z_range

    def set_axes_equal(ax, title):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y (Up)')
        ax.set_zlabel('Z (Forward)')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        # Set view angle - looking from side, Y-up
        ax.view_init(elev=5, azim=-90)

    # Initialize plots
    scatter1 = ax1.scatter([], [], [], c='blue', marker='o', s=100, alpha=0.6, label='Baseline')
    scatter2 = ax2.scatter([], [], [], c='green', marker='s', s=100, alpha=0.6, label='With Offset')
    scatter3 = ax3.scatter([], [], [], c='red', marker='^', s=100, alpha=0.6, label='Target')

    # Add target as gray points in first two subplots for reference
    scatter1_target = ax1.scatter([], [], [], c='gray', marker='^', s=50, alpha=0.3, label='Target')
    scatter2_target = ax2.scatter([], [], [], c='gray', marker='^', s=50, alpha=0.3, label='Target')

    set_axes_equal(ax1, 'V2 Baseline')
    set_axes_equal(ax2, 'V2 + Offset')
    set_axes_equal(ax3, 'Target (AddBiomechanics)')

    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper left', fontsize=10)
    ax3.legend(loc='upper left', fontsize=10)

    # Title text objects
    title1 = ax1.text2D(0.05, 0.95, '', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    title2 = ax2.text2D(0.05, 0.95, '', transform=ax2.transAxes, fontsize=12, verticalalignment='top')
    title3 = ax3.text2D(0.05, 0.95, '', transform=ax3.transAxes, fontsize=12, verticalalignment='top')

    fig.suptitle(f'V2 Baseline ({meta_baseline["metrics"]["MPJPE"]:.2f} mm) vs V2 + Offset ({meta_offset["metrics_with_offset"]["MPJPE"]:.2f} mm)\nImprovement: {meta_offset["metrics_with_offset"]["improvement_mm"]:.2f} mm ({meta_offset["metrics_with_offset"]["improvement_percent"]:.1f}%)',
                 fontsize=14, fontweight='bold')

    # Track line artists
    skeleton_lines1 = []
    skeleton_lines2 = []
    skeleton_lines3 = []

    def update(frame_idx_in_list):
        nonlocal skeleton_lines1, skeleton_lines2, skeleton_lines3

        frame_idx = frame_indices[frame_idx_in_list]

        # Get full skeleton joints for this frame
        baseline_joints_full = pred_baseline[frame_idx, :, :]  # All 24 joints
        offset_joints_full = pred_offset[frame_idx, :, :]      # All 24 joints

        # Get mapped joints for target
        baseline_joints = pred_baseline[frame_idx, smpl_indices, :]
        offset_joints = pred_offset[frame_idx, smpl_indices, :]
        target_joints = target[frame_idx, addb_indices, :]

        # Compute per-frame errors
        error_baseline = compute_frame_error(pred_baseline[frame_idx], target[frame_idx], joint_mapping)
        error_offset = compute_frame_error(pred_offset[frame_idx], target[frame_idx], joint_mapping)

        # Remove old skeleton lines
        for line in skeleton_lines1:
            line.remove()
        for line in skeleton_lines2:
            line.remove()
        for line in skeleton_lines3:
            line.remove()
        skeleton_lines1 = []
        skeleton_lines2 = []
        skeleton_lines3 = []

        # Draw skeleton bones for baseline (thicker lines)
        for i, parent_idx in enumerate(SMPL_PARENTS):
            if parent_idx >= 0:
                line = ax1.plot([baseline_joints_full[i, 0], baseline_joints_full[parent_idx, 0]],
                               [baseline_joints_full[i, 1], baseline_joints_full[parent_idx, 1]],
                               [baseline_joints_full[i, 2], baseline_joints_full[parent_idx, 2]],
                               'b-', linewidth=4, alpha=0.7)[0]
                skeleton_lines1.append(line)

        # Draw skeleton bones for offset (thicker lines)
        for i, parent_idx in enumerate(SMPL_PARENTS):
            if parent_idx >= 0:
                line = ax2.plot([offset_joints_full[i, 0], offset_joints_full[parent_idx, 0]],
                               [offset_joints_full[i, 1], offset_joints_full[parent_idx, 1]],
                               [offset_joints_full[i, 2], offset_joints_full[parent_idx, 2]],
                               'g-', linewidth=4, alpha=0.7)[0]
                skeleton_lines2.append(line)

        # Draw simple connections for target joints (connect adjacent mapped joints)
        for i in range(len(target_joints) - 1):
            if not (np.isnan(target_joints[i]).any() or np.isnan(target_joints[i+1]).any()):
                line = ax3.plot([target_joints[i, 0], target_joints[i+1, 0]],
                               [target_joints[i, 1], target_joints[i+1, 1]],
                               [target_joints[i, 2], target_joints[i+1, 2]],
                               'r-', linewidth=2, alpha=0.5)[0]
                skeleton_lines3.append(line)

        # Update scatter plots (larger joint markers)
        scatter1._offsets3d = (baseline_joints_full[:, 0], baseline_joints_full[:, 1], baseline_joints_full[:, 2])
        scatter1._sizes = np.array([150] * len(baseline_joints_full))  # Larger joints

        scatter2._offsets3d = (offset_joints_full[:, 0], offset_joints_full[:, 1], offset_joints_full[:, 2])
        scatter2._sizes = np.array([150] * len(offset_joints_full))

        scatter3._offsets3d = (target_joints[:, 0], target_joints[:, 1], target_joints[:, 2])
        scatter3._sizes = np.array([150] * len(target_joints))

        scatter1_target._offsets3d = (target_joints[:, 0], target_joints[:, 1], target_joints[:, 2])
        scatter1_target._sizes = np.array([80] * len(target_joints))

        scatter2_target._offsets3d = (target_joints[:, 0], target_joints[:, 1], target_joints[:, 2])
        scatter2_target._sizes = np.array([80] * len(target_joints))

        # Update titles with frame number and error
        title1.set_text(f'V2 Baseline\nFrame {frame_idx}/{n_frames-1}\nError: {error_baseline:.1f} mm')
        title2.set_text(f'V2 + Offset\nFrame {frame_idx}/{n_frames-1}\nError: {error_offset:.1f} mm')
        title3.set_text(f'Target (AddBiomechanics)\nFrame {frame_idx}/{n_frames-1}')

        return [scatter1, scatter2, scatter3, scatter1_target, scatter2_target, title1, title2, title3] + skeleton_lines1 + skeleton_lines2 + skeleton_lines3

    # Create animation
    print(f"Creating animation with {len(frame_indices)} frames...")
    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices),
                                   interval=1000/fps, blit=False, repeat=True)

    # Save animation
    print(f"Saving to {output_path}...")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='SMPL2AddBiomechanics'), bitrate=5000)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"Video saved: {output_path}")
    print(f"Total frames: {len(frame_indices)}")
    print(f"Duration: {len(frame_indices)/fps:.1f} seconds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dir', type=str, required=True, help='Directory containing V2 baseline results')
    parser.add_argument('--offset_dir', type=str, required=True, help='Directory containing V2 + offset results')
    parser.add_argument('--output', type=str, required=True, help='Output mp4 file path')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--subsample', type=int, default=1, help='Subsample factor (1=all frames, 2=every other frame)')
    args = parser.parse_args()

    print(f"Loading baseline from {args.baseline_dir}...")
    print(f"Loading offset from {args.offset_dir}...")
    pred_baseline, pred_offset, target, smpl_params_baseline, smpl_params_offset, meta_baseline, meta_offset = load_results(args.baseline_dir, args.offset_dir)

    joint_mapping = meta_baseline['joint_mapping']

    print(f"\nDataset info:")
    print(f"  Total frames: {pred_baseline.shape[0]}")
    print(f"  V2 Baseline MPJPE: {meta_baseline['metrics']['MPJPE']:.2f} mm")
    print(f"  V2 + Offset MPJPE: {meta_offset['metrics_with_offset']['MPJPE']:.2f} mm")
    print(f"  Improvement: {meta_offset['metrics_with_offset']['improvement_mm']:.2f} mm ({meta_offset['metrics_with_offset']['improvement_percent']:.1f}%)")

    create_animation(pred_baseline, pred_offset, target, joint_mapping,
                    meta_baseline, meta_offset, args.output,
                    fps=args.fps, subsample=args.subsample)

if __name__ == '__main__':
    main()
