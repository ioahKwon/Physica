#!/usr/bin/env python3
"""
3D visualization with thick skeleton to show body volume
Much simpler and more reliable than mesh rendering
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import json
import argparse
from pathlib import Path


def load_results(result_dir):
    """Load prediction and target results"""
    result_dir = Path(result_dir)

    pred_joints = np.load(result_dir / 'pred_joints.npy')
    target_joints = np.load(result_dir / 'target_joints.npy')

    with open(result_dir / 'meta.json', 'r') as f:
        meta = json.load(f)

    return pred_joints, target_joints, meta


def detect_body_type(meta):
    """Detect if upper body was optimized"""
    joint_mapping = meta.get('joint_mapping', {})
    if not joint_mapping:
        return "no_arm"

    mapped_smpl_indices = set(int(v) for v in joint_mapping.values())
    upper_body_indices = {3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
    has_upper_body_optimized = bool(mapped_smpl_indices & upper_body_indices)

    return "with_arm" if has_upper_body_optimized else "no_arm"


def get_skeleton_edges():
    """SMPL skeleton connections"""
    return [
        # Lower body
        (0, 1), (0, 2),  # pelvis to hips
        (1, 4), (2, 5),  # hips to knees
        (4, 7), (5, 8),  # knees to ankles
        (7, 10), (8, 11),  # ankles to feet
        # Upper body
        (0, 3),  # pelvis to spine
        (3, 6), (6, 9),  # spine chain
        (9, 12), (9, 13), (9, 14),  # neck/shoulders
        (13, 16), (14, 17),  # shoulders to elbows
        (16, 18), (17, 19),  # elbows to wrists
        (18, 20), (19, 21),  # wrists to hands
        (20, 22), (21, 23),  # hands
    ]


def create_animation(result_dir, output_path=None, fps=30, rotation_degrees=180):
    """
    Create 3D animation with THICK skeleton to show body volume

    Args:
        result_dir: Path to result directory
        output_path: Output MP4 path
        fps: Frames per second
        rotation_degrees: Total rotation in degrees
    """
    pred_joints, target_joints, meta = load_results(result_dir)

    body_type = detect_body_type(meta)
    print(f'Detected body type: {body_type.upper()}')

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
    skeleton_edges = get_skeleton_edges()

    # Compute separate bounds for target and pred
    target_points = target_joints.reshape(-1, 3)
    target_points = target_points[~np.isnan(target_points).any(axis=1)]
    pred_points = pred_joints.reshape(-1, 3)
    pred_points = pred_points[~np.isnan(pred_points).any(axis=1)]

    target_min = target_points.min(axis=0)
    target_max = target_points.max(axis=0)
    pred_min = pred_points.min(axis=0)
    pred_max = pred_points.max(axis=0)

    target_range = target_max - target_min
    pred_range = pred_max - pred_min

    # Detect if pred has much larger range than target (>3x difference)
    range_ratio = pred_range / (target_range + 1e-6)  # avoid division by zero
    use_target_centered = (range_ratio > 3.0).any()

    if use_target_centered:
        # Use target-centered bounds to make target movement visible
        print(f'Target-centered view (pred {range_ratio.max():.1f}x larger than target)')

        # Center on target's center
        target_center = (target_min + target_max) / 2

        # Use target's range + 50% padding to ensure movement is visible
        padding = 0.5
        x_range = max(target_range[0], 0.5)  # minimum 0.5m range
        y_range = max(target_range[1], 0.5)
        z_range = max(target_range[2], 0.5)

        x_min = target_center[0] - x_range * (1 + padding)
        x_max = target_center[0] + x_range * (1 + padding)
        y_min = target_center[1] - y_range * (1 + padding)
        y_max = target_center[1] + y_range * (1 + padding)
        z_min = target_center[2] - z_range * (1 + padding)
        z_max = target_center[2] + z_range * (1 + padding)
    else:
        # Use combined bounds when ranges are similar
        print(f'Combined bounds (ranges similar)')
        all_points = np.concatenate([pred_points, target_points], axis=0)

        x_min, y_min, z_min = all_points.min(axis=0)
        x_max, y_max, z_max = all_points.max(axis=0)

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

        pred = pred_joints[frame_idx]
        target_vis = target_joints_vis[frame_idx]

        # Draw THICK skeleton to show body volume (VERY thick lines)
        for i, j in skeleton_edges:
            if not (np.isnan(pred[i]).any() or np.isnan(pred[j]).any()):
                ax.plot([pred[i, 0], pred[j, 0]],
                       [pred[i, 1], pred[j, 1]],
                       [pred[i, 2], pred[j, 2]],
                       'dodgerblue', linewidth=8, alpha=0.7, solid_capstyle='round')

        # Draw thinner skeleton for clarity
        for i, j in skeleton_edges:
            if not (np.isnan(pred[i]).any() or np.isnan(pred[j]).any()):
                ax.plot([pred[i, 0], pred[j, 0]],
                       [pred[i, 1], pred[j, 1]],
                       [pred[i, 2], pred[j, 2]],
                       'blue', linewidth=3, alpha=0.9)

        # Plot SMPL joints (larger spheres)
        pred_valid = pred[~np.isnan(pred).any(axis=1)]
        if len(pred_valid) > 0:
            ax.scatter(pred_valid[:, 0], pred_valid[:, 1], pred_valid[:, 2],
                      c='blue', s=120, marker='o', label='SMPL (Predicted)',
                      alpha=1.0, edgecolors='darkblue', linewidths=2, depthshade=False)

        # Plot target joints (AddB)
        valid_target = target_vis[~np.isnan(target_vis).any(axis=1)]
        if len(valid_target) > 0:
            ax.scatter(valid_target[:, 0], valid_target[:, 1], valid_target[:, 2],
                      c='red', s=100, marker='o', label='AddB (Target)',
                      alpha=1.0, edgecolors='darkred', linewidths=2.5,
                      depthshade=False, zorder=10)

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

        # Set viewing angle
        ax.view_init(elev=20, azim=angle)

        # Labels and title
        ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=11, fontweight='bold')

        subject_name = meta.get('run_name', 'Unknown')
        overall_mpjpe = meta.get('MPJPE', 0.0)

        title_suffix = "FULL BODY (Thick Skeleton)" if body_type == "with_arm" else "LOWER BODY (Thick Skeleton)"

        ax.set_title(
            f'{subject_name} - {title_suffix}\n'
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

    # Generate frame data with rotation
    angles = np.linspace(0, rotation_degrees, num_frames)
    frame_data = [(i, angle) for i, angle in enumerate(angles)]

    print(f'Creating 3D THICK SKELETON animation with {num_frames} frames...')
    print(f'Rotation: {rotation_degrees}° total ({rotation_degrees/num_frames:.2f}° per frame)')

    anim = FuncAnimation(fig, update, frames=frame_data,
                        interval=1000/fps, blit=False)

    # Save
    if output_path is None:
        subject_name = meta.get('run_name', 'result')
        output_path = Path(result_dir) / f'{subject_name}_3d_thick_skeleton.mp4'
    else:
        output_path = Path(output_path)

    print(f'Saving 3D MP4 to: {output_path}')
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    anim.save(str(output_path), writer=writer)

    plt.close()
    print(f'✓ 3D Video saved: {output_path}')
    print(f'  Body type: {body_type.upper()}')
    print(f'  File size: {output_path.stat().st_size / 1024:.0f}KB')

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Create 3D visualization with thick skeleton to show body volume'
    )
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Path to result directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output MP4 path (default: auto-generated)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--rotation_degrees', type=int, default=180,
                       help='Total rotation in degrees (default: 180)')

    args = parser.parse_args()
    create_animation(args.result_dir, args.output, args.fps, args.rotation_degrees)


if __name__ == '__main__':
    main()
