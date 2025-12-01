#!/usr/bin/env python3
"""
Unified 2D multi-view visualization for SMPL fitting results
Auto-detects With_Arm vs No_Arm and shows appropriate body parts
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import json
import argparse
from pathlib import Path


# Lower body joint indices for filtering
LOWER_BODY_INDICES = [0, 1, 2, 4, 5, 7, 8, 10, 11]


def detect_body_type(meta):
    """
    Detect if upper body was optimized based on joint_mapping

    Returns:
        "with_arm" if upper body SMPL joints are mapped, else "no_arm"
    """
    # Check if SMPL upper body joints are in the mapping
    joint_mapping = meta.get('joint_mapping', {})
    if not joint_mapping:
        return "no_arm"

    # Get mapped SMPL indices
    mapped_smpl_indices = set(int(v) for v in joint_mapping.values())

    # Upper body SMPL indices: spine(3), chest(6), neck(9), shoulders(12,13,14), arms(15-23)
    upper_body_indices = {3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}

    # If ANY upper body joint is mapped, we optimized upper body
    has_upper_body_optimized = bool(mapped_smpl_indices & upper_body_indices)

    return "with_arm" if has_upper_body_optimized else "no_arm"


def load_results(result_dir):
    """Load prediction and target results"""
    result_dir = Path(result_dir)

    pred_joints = np.load(result_dir / 'pred_joints.npy')  # [T, 24, 3]
    target_joints = np.load(result_dir / 'target_joints.npy')  # [T, N, 3]

    with open(result_dir / 'meta.json', 'r') as f:
        meta = json.load(f)

    return pred_joints, target_joints, meta


def create_2d_animation(result_dir, output_path=None, fps=30):
    """
    Create 2D multi-view animation with auto-detection of body type

    Args:
        result_dir: Path to result directory
        output_path: Output MP4 path
        fps: Frames per second
    """
    pred_joints, target_joints, meta = load_results(result_dir)

    # Auto-detect body type
    body_type = detect_body_type(meta)
    print(f'Detected body type: {body_type.upper()}')

    use_lower_only = (body_type == "no_arm")
    title_suffix = "LOWER BODY" if use_lower_only else "FULL BODY"

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

    # Filter to lower body if needed
    if use_lower_only:
        pred_display = pred_joints[:, LOWER_BODY_INDICES, :]
    else:
        pred_display = pred_joints

    # Compute global bounds for all frames
    all_joints = np.concatenate([pred_display.reshape(-1, 3),
                                   target_joints.reshape(-1, 3)], axis=0)
    all_joints = all_joints[~np.isnan(all_joints).any(axis=1)]

    x_min, y_min, z_min = all_joints.min(axis=0)
    x_max, y_max, z_max = all_joints.max(axis=0)

    # Add padding
    padding = 0.2
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    z_min -= padding
    z_max += padding

    # Setup figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('', fontsize=14, fontweight='bold')

    def init():
        return []

    def update(frame_idx):
        # Clear all axes
        for ax in axes:
            ax.clear()

        # Get current frame
        pred = pred_joints[frame_idx]
        target_vis = target_joints_vis[frame_idx]

        # Filter predicted joints if lower body only
        if use_lower_only:
            pred_valid = pred[LOWER_BODY_INDICES]
        else:
            pred_valid = pred
        pred_valid = pred_valid[~np.isnan(pred_valid).any(axis=1)]

        valid_target = target_vis[~np.isnan(target_vis).any(axis=1)]

        # --- View 1: Top (XY plane) ---
        ax = axes[0]
        if len(pred_valid) > 0:
            ax.scatter(pred_valid[:, 0], pred_valid[:, 1],
                      c='blue', s=60, marker='o', label='Predicted',
                      alpha=0.8, edgecolors='darkblue', linewidths=2)
        if len(valid_target) > 0:
            ax.scatter(valid_target[:, 0], valid_target[:, 1],
                      c='red', s=60, marker='^', label='Target',
                      alpha=0.8, edgecolors='darkred', linewidths=2)

        # Draw correspondence lines
        for k in range(min(pred_joints_vis.shape[1], target_vis.shape[0])):
            if k < pred_joints_vis.shape[1]:
                pred_pt = pred_joints_vis[frame_idx, k]
                target_pt = target_vis[k]
                if not (np.isnan(pred_pt).any() or np.isnan(target_pt).any()):
                    ax.plot([pred_pt[0], target_pt[0]],
                           [pred_pt[1], target_pt[1]],
                           'gray', linestyle='-', linewidth=1, alpha=0.3)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X (m)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
        ax.set_title('Top View (XY)', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # --- View 2: Front (XZ plane) ---
        ax = axes[1]
        if len(pred_valid) > 0:
            ax.scatter(pred_valid[:, 0], pred_valid[:, 2],
                      c='blue', s=60, marker='o', label='Predicted',
                      alpha=0.8, edgecolors='darkblue', linewidths=2)
        if len(valid_target) > 0:
            ax.scatter(valid_target[:, 0], valid_target[:, 2],
                      c='red', s=60, marker='^', label='Target',
                      alpha=0.8, edgecolors='darkred', linewidths=2)

        # Draw correspondence lines
        for k in range(min(pred_joints_vis.shape[1], target_vis.shape[0])):
            if k < pred_joints_vis.shape[1]:
                pred_pt = pred_joints_vis[frame_idx, k]
                target_pt = target_vis[k]
                if not (np.isnan(pred_pt).any() or np.isnan(target_pt).any()):
                    ax.plot([pred_pt[0], target_pt[0]],
                           [pred_pt[2], target_pt[2]],
                           'gray', linestyle='-', linewidth=1, alpha=0.3)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_xlabel('X (m)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Z (m)', fontsize=10, fontweight='bold')
        ax.set_title('Front View (XZ)', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # --- View 3: Side (YZ plane) ---
        ax = axes[2]
        if len(pred_valid) > 0:
            ax.scatter(pred_valid[:, 1], pred_valid[:, 2],
                      c='blue', s=60, marker='o', label='Predicted',
                      alpha=0.8, edgecolors='darkblue', linewidths=2)
        if len(valid_target) > 0:
            ax.scatter(valid_target[:, 1], valid_target[:, 2],
                      c='red', s=60, marker='^', label='Target',
                      alpha=0.8, edgecolors='darkred', linewidths=2)

        # Draw correspondence lines
        for k in range(min(pred_joints_vis.shape[1], target_vis.shape[0])):
            if k < pred_joints_vis.shape[1]:
                pred_pt = pred_joints_vis[frame_idx, k]
                target_pt = target_vis[k]
                if not (np.isnan(pred_pt).any() or np.isnan(target_pt).any()):
                    ax.plot([pred_pt[1], target_pt[1]],
                           [pred_pt[2], target_pt[2]],
                           'gray', linestyle='-', linewidth=1, alpha=0.3)

        ax.set_xlim(y_min, y_max)
        ax.set_ylim(z_min, z_max)
        ax.set_xlabel('Y (m)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Z (m)', fontsize=10, fontweight='bold')
        ax.set_title('Side View (YZ)', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

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

        # Update main title
        subject_name = meta.get('run_name', 'Unknown')
        overall_mpjpe = meta.get('MPJPE', 0.0)

        fig.suptitle(
            f'{subject_name} - {title_suffix} | '
            f'Frame {frame_idx + 1}/{num_frames} | '
            f'Frame MPJPE: {mpjpe_frame:.2f}mm | '
            f'Overall MPJPE: {overall_mpjpe:.2f}mm',
            fontsize=14, fontweight='bold'
        )

        return axes

    print(f'Creating 2D {title_suffix} multi-view animation with {num_frames} frames...')

    anim = FuncAnimation(fig, update, frames=range(num_frames),
                        init_func=init, interval=1000/fps, blit=False, repeat=True)

    # Save
    if output_path is None:
        subject_name = meta.get('run_name', 'result')
        output_path = Path(result_dir).parent / f'{subject_name}_2d_{body_type}.mp4'
    else:
        output_path = Path(output_path)

    print(f'Saving 2D MP4 to: {output_path}')
    writer = FFMpegWriter(fps=fps, bitrate=2000, codec='libx264')
    anim.save(str(output_path), writer=writer, dpi=100)

    plt.close()
    print(f'✓ 2D Video saved: {output_path}')
    print(f'  Body type: {body_type.upper()}')
    print(f'  File size: {output_path.stat().st_size / 1024:.0f}KB')

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Create 2D multi-view visualization (auto-detects With_Arm vs No_Arm)'
    )
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Path to result directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output MP4 path (default: auto-generated)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')

    args = parser.parse_args()
    create_2d_animation(args.result_dir, args.output, args.fps)


if __name__ == '__main__':
    main()
