#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Script for AddBiomechanics to SMPL Conversion Results
====================================================================

This script provides various visualizations to evaluate the quality of
SMPL parameter fitting to AddBiomechanics data.

Visualizations:
1. Error over time plot
2. Per-joint error distribution
3. 3D skeleton comparison (selected frames)
4. Temporal smoothness analysis
5. Metric summary table

Usage:
    python visualize_results.py --result_dir <path_to_results> [options]
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional

import torch


# ============================================================================
# SMPL Constants
# ============================================================================

SMPL_PARENTS = np.array([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
], dtype=np.int32)

SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
    'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
    'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
]


# ============================================================================
# Data Loading
# ============================================================================

def load_results(result_dir: str) -> Dict:
    """
    Load results from directory.

    Args:
        result_dir: Path to results directory

    Returns:
        results: Dictionary containing all results
    """
    # Load SMPL parameters
    smpl_params = np.load(os.path.join(result_dir, 'smpl_params.npz'))
    betas = smpl_params['betas']
    poses = smpl_params['poses']
    trans = smpl_params['trans']

    # Load joints
    pred_joints = np.load(os.path.join(result_dir, 'pred_joints.npy'))
    target_joints = np.load(os.path.join(result_dir, 'target_joints.npy'))

    # Load metadata
    with open(os.path.join(result_dir, 'meta.json'), 'r') as f:
        meta = json.load(f)

    return {
        'betas': betas,
        'poses': poses,
        'trans': trans,
        'pred_joints': pred_joints,
        'target_joints': target_joints,
        'meta': meta
    }


# ============================================================================
# Metric Computation
# ============================================================================

def compute_per_frame_errors(pred_joints: np.ndarray,
                            target_joints: np.ndarray,
                            joint_mapping: Dict[int, int],
                            root_addb_idx: Optional[int]) -> np.ndarray:
    """
    Compute per-frame average error.

    Args:
        pred_joints: Predicted joints (T, 24, 3)
        target_joints: Target joints (T, J_addb, 3)
        joint_mapping: Joint mapping
        root_addb_idx: AddB index that corresponds to the SMPL pelvis (if available)

    Returns:
        errors: Per-frame errors (T,)
    """
    T = pred_joints.shape[0]
    errors = np.zeros(T)

    for t in range(T):
        pred_centered, target_centered = center_on_root(
            pred_joints[t], target_joints[t], root_addb_idx
        )
        frame_errors = []
        for addb_idx, smpl_idx in joint_mapping.items():
            if addb_idx < target_centered.shape[0]:
                target = target_centered[addb_idx]
                if not np.any(np.isnan(target)):
                    pred = pred_centered[smpl_idx]
                    error = np.linalg.norm(pred - target)
                    frame_errors.append(error)

        errors[t] = np.mean(frame_errors) if frame_errors else np.nan

    return errors


def compute_per_joint_errors(pred_joints: np.ndarray,
                            target_joints: np.ndarray,
                            joint_mapping: Dict[int, int],
                            root_addb_idx: Optional[int]) -> Dict[int, List[float]]:
    """
    Compute per-joint errors across all frames.

    Args:
        pred_joints: Predicted joints (T, 24, 3)
        target_joints: Target joints (T, J_addb, 3)
        joint_mapping: Joint mapping
        root_addb_idx: AddB index that corresponds to the SMPL pelvis (if available)

    Returns:
        errors: Dictionary mapping SMPL joint index to list of errors
    """
    T = pred_joints.shape[0]
    errors = {smpl_idx: [] for smpl_idx in joint_mapping.values()}

    for t in range(T):
        pred_centered, target_centered = center_on_root(
            pred_joints[t], target_joints[t], root_addb_idx
        )
        for addb_idx, smpl_idx in joint_mapping.items():
            if addb_idx < target_centered.shape[0]:
                target = target_centered[addb_idx]
                if not np.any(np.isnan(target)):
                    pred = pred_centered[smpl_idx]
                    error = np.linalg.norm(pred - target)
                    errors[smpl_idx].append(error)

    return errors


def compute_temporal_metrics(joints: np.ndarray) -> Dict[str, float]:
    """
    Compute temporal consistency metrics.

    Args:
        joints: Joint positions (T, J, 3)

    Returns:
        metrics: Dictionary of temporal metrics
    """
    if joints.shape[0] < 3:
        return {'velocity_norm': np.nan, 'acceleration_norm': np.nan, 'jerk_norm': np.nan}

    # Velocity
    velocities = joints[1:] - joints[:-1]
    velocity_norm = np.linalg.norm(velocities, axis=-1).mean()

    # Acceleration
    accelerations = velocities[1:] - velocities[:-1]
    acceleration_norm = np.linalg.norm(accelerations, axis=-1).mean()

    # Jerk
    if joints.shape[0] >= 4:
        jerks = accelerations[1:] - accelerations[:-1]
        jerk_norm = np.linalg.norm(jerks, axis=-1).mean()
    else:
        jerk_norm = np.nan

    return {
        'velocity_norm': velocity_norm,
        'acceleration_norm': acceleration_norm,
        'jerk_norm': jerk_norm
    }


def center_on_root(pred_frame: np.ndarray,
                   target_frame: np.ndarray,
                   root_addb_idx: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Center SMPL and AddB joints around pelvis/root for comparison."""
    pred_centered = pred_frame.copy()
    target_centered = target_frame.copy()

    if root_addb_idx is not None and root_addb_idx < target_frame.shape[0]:
        root_target = target_frame[root_addb_idx]
        if not np.any(np.isnan(root_target)):
            pred_centered -= pred_frame[0]
            target_centered -= root_target

    return pred_centered, target_centered


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_error_over_time(errors: np.ndarray, dt: float, out_path: str):
    """
    Plot error over time.

    Args:
        errors: Per-frame errors (T,)
        dt: Timestep in seconds
        out_path: Output file path
    """
    T = len(errors)
    time = np.arange(T) * dt

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(time, errors * 1000, linewidth=1.5)  # Convert to mm
    ax.axhline(np.nanmean(errors) * 1000, color='r', linestyle='--',
              label=f'Mean: {np.nanmean(errors)*1000:.2f} mm')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Error (mm)', fontsize=12)
    ax.set_title('Per-Frame Average Joint Position Error', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {out_path}")


def plot_per_joint_errors(joint_errors: Dict[int, List[float]], out_path: str):
    """
    Plot per-joint error distribution.

    Args:
        joint_errors: Dictionary mapping joint index to error list
        out_path: Output file path
    """
    # Sort joints by index
    joint_indices = sorted(joint_errors.keys())
    joint_names = [SMPL_JOINT_NAMES[i] if i < len(SMPL_JOINT_NAMES) else f'joint_{i}'
                  for i in joint_indices]

    # Compute statistics
    means = [np.mean(joint_errors[i]) * 1000 for i in joint_indices]  # Convert to mm
    stds = [np.std(joint_errors[i]) * 1000 for i in joint_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(joint_indices))
    ax.bar(x, means, yerr=stds, capsize=3, alpha=0.7, edgecolor='black')

    ax.set_xlabel('Joint', fontsize=12)
    ax.set_ylabel('Error (mm)', fontsize=12)
    ax.set_title('Per-Joint Error Distribution (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(joint_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {out_path}")


def plot_skeleton_3d(pred_joints: np.ndarray, target_joints: np.ndarray,
                    joint_mapping: Dict[int, int], frame_idx: int,
                    root_addb_idx: Optional[int], out_path: str):
    """
    Plot 3D skeleton comparison.

    Args:
        pred_joints: Predicted joints (T, 24, 3)
        target_joints: Target joints (T, J_addb, 3)
        joint_mapping: Joint mapping
        frame_idx: Frame index to visualize
        root_addb_idx: AddB pelvis index (if available) for centering
        out_path: Output file path
    """
    fig = plt.figure(figsize=(15, 5))

    # Extract frame data
    pred_frame = pred_joints[frame_idx]
    target_frame = target_joints[frame_idx]
    pred, target_raw = center_on_root(pred_frame, target_frame, root_addb_idx)

    # Create target joints in SMPL format
    target = np.full((24, 3), np.nan)
    for addb_idx, smpl_idx in joint_mapping.items():
        if addb_idx < target_raw.shape[0]:
            target[smpl_idx] = target_raw[addb_idx]

    # Three views: front, side, top
    views = [
        (90, -90, 'Front View'),   # Front
        (90, 0, 'Side View'),      # Side
        (0, -90, 'Top View')       # Top
    ]

    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')

        # Plot predicted skeleton
        for j in range(24):
            parent = SMPL_PARENTS[j]
            if parent >= 0:
                points = np.array([pred[parent], pred[j]])
                ax.plot(points[:, 0], points[:, 1], points[:, 2],
                       'b-', linewidth=2, label='Predicted' if j == 1 else '')

        # Plot predicted joints
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2],
                  c='blue', s=30, alpha=0.8)

        # Plot target joints
        valid_target = target[~np.isnan(target[:, 0])]
        if len(valid_target) > 0:
            ax.scatter(valid_target[:, 0], valid_target[:, 1], valid_target[:, 2],
                      c='red', s=50, alpha=0.6, marker='x', linewidths=2,
                      label='Target')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)

        # Set equal aspect ratio
        max_range = np.array([pred[:, 0].max() - pred[:, 0].min(),
                             pred[:, 1].max() - pred[:, 1].min(),
                             pred[:, 2].max() - pred[:, 2].min()]).max() / 2.0
        mid_x = (pred[:, 0].max() + pred[:, 0].min()) * 0.5
        mid_y = (pred[:, 1].max() + pred[:, 1].min()) * 0.5
        mid_z = (pred[:, 2].max() + pred[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        if i == 0:
            ax.legend()

    plt.suptitle(f'3D Skeleton Comparison - Frame {frame_idx}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {out_path}")


def compute_global_bounds(joints: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute global center and radius for consistent 3D views."""
    finite_mask = np.isfinite(joints)
    if not np.any(finite_mask):
        center = np.zeros(3, dtype=np.float32)
        radius = 1.0
    else:
        min_vals = np.nanmin(joints, axis=(0, 1))
        max_vals = np.nanmax(joints, axis=(0, 1))
        center = (min_vals + max_vals) / 2.0
        radius = np.max(max_vals - min_vals) / 2.0
        if radius < 1e-3:
            radius = 1.0
    return center.astype(np.float32), float(radius)


def render_skeleton_frame(pred_joints: np.ndarray, target_joints: np.ndarray,
                          joint_mapping: Dict[int, int], frame_idx: int,
                          root_addb_idx: Optional[int], center: np.ndarray,
                          radius: float, dpi: int = 150) -> np.ndarray:
    """Render multi-view skeleton comparison for a single frame and return RGB image."""
    fig = plt.figure(figsize=(12, 4), dpi=dpi)

    pred_frame = pred_joints[frame_idx]
    target_frame = target_joints[frame_idx]
    pred, target_raw = center_on_root(pred_frame, target_frame, root_addb_idx)

    target = np.full((24, 3), np.nan, dtype=np.float32)
    for addb_idx, smpl_idx in joint_mapping.items():
        if addb_idx < target_raw.shape[0]:
            target[smpl_idx] = target_raw[addb_idx]

    views = [
        (90, -90, 'Front View'),
        (90, 0, 'Side View'),
        (0, -90, 'Top View')
    ]

    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')

        for j in range(24):
            parent = SMPL_PARENTS[j]
            if parent >= 0:
                pts = np.array([pred[parent], pred[j]])
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=2,
                        label='Predicted' if j == 1 else '')

        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='blue', s=30, alpha=0.8)

        valid_target = target[~np.isnan(target[:, 0])]
        if len(valid_target) > 0:
            ax.scatter(valid_target[:, 0], valid_target[:, 1], valid_target[:, 2],
                       c='red', s=50, alpha=0.6, marker='x', linewidths=2,
                       label='Target')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

        if i == 0:
            ax.legend(loc='upper right')

    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape((height, width, 3))
    plt.close(fig)
    return image


def create_skeleton_video(pred_joints: np.ndarray, target_joints: np.ndarray,
                          joint_mapping: Dict[int, int], root_addb_idx: Optional[int],
                          out_path: str, fps: int = 24, dpi: int = 150):
    """Create an MP4 video for the full sequence."""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "imageio is required for video export. Install it via `pip install imageio imageio-ffmpeg`."
        ) from exc

    pred_centered_all = pred_joints - pred_joints[:, :1, :]
    center, radius = compute_global_bounds(pred_centered_all)
    frame_range = range(pred_joints.shape[0])

    with imageio.get_writer(out_path, fps=fps, codec='libx264') as writer:
        for idx in frame_range:
            frame_img = render_skeleton_frame(
                pred_joints, target_joints, joint_mapping, idx, root_addb_idx,
                center, radius, dpi=dpi
            )
            writer.append_data(frame_img)
            if (idx + 1) % 50 == 0 or idx == pred_joints.shape[0] - 1:
                print(f"    Added frame {idx + 1}/{pred_joints.shape[0]}")

    print(f"  Saved video: {out_path}")


def plot_temporal_analysis(pred_joints: np.ndarray, target_joints: np.ndarray,
                          dt: float, out_path: str):
    """
    Plot temporal smoothness analysis.

    Args:
        pred_joints: Predicted joints (T, 24, 3)
        target_joints: Target joints (T, J_addb, 3)
        dt: Timestep
        out_path: Output file path
    """
    T = pred_joints.shape[0]
    if T < 2:
        raise ValueError("Need at least 2 frames for temporal analysis.")

    # Compute velocity vectors and speed per joint
    pred_vel_vec = pred_joints[1:] - pred_joints[:-1]  # (T-1, 24, 3)
    pred_speed = np.linalg.norm(pred_vel_vec, axis=-1)  # (T-1, 24)

    # Compute acceleration vectors and magnitude per joint
    if T < 3:
        pred_acc_mag = np.zeros((0, pred_joints.shape[1]), dtype=np.float32)
    else:
        pred_acc_vec = pred_vel_vec[1:] - pred_vel_vec[:-1]  # (T-2, 24, 3)
        pred_acc_mag = np.linalg.norm(pred_acc_vec, axis=-1)  # (T-2, 24)

    time_vel = np.arange(pred_speed.shape[0]) * dt
    time_accel = np.arange(pred_acc_mag.shape[0]) * dt

    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Velocity
    ax = axes[0]
    for j in range(pred_speed.shape[1]):
        ax.plot(time_vel, pred_speed[:, j], alpha=0.3, linewidth=0.5)
    ax.plot(time_vel, pred_speed.mean(axis=1), 'k-', linewidth=2, label='Mean')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Velocity (m/s)', fontsize=12)
    ax.set_title('Joint Velocities', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Acceleration
    ax = axes[1]
    if pred_acc_mag.shape[0] > 0:
        for j in range(pred_acc_mag.shape[1]):
            ax.plot(time_accel, pred_acc_mag[:, j], alpha=0.3, linewidth=0.5)
        ax.plot(time_accel, pred_acc_mag.mean(axis=1), 'k-', linewidth=2, label='Mean')
        ax.legend()
    ax.set_xlim(time_accel[0] if time_accel.size else 0, time_accel[-1] if time_accel.size else max(time_vel[-1], 1))
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Acceleration (m/s²)', fontsize=12)
    ax.set_title('Joint Accelerations', fontsize=12, fontweight='bold')
    if pred_acc_mag.shape[0] == 0:
        ax.text(0.5, 0.5, 'Not enough frames for acceleration', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Temporal Smoothness Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {out_path}")


def create_summary_table(metrics: Dict, temporal_metrics: Dict, out_path: str):
    """
    Create and save metric summary table.

    Args:
        metrics: Main metrics
        temporal_metrics: Temporal metrics
        out_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    table_data.append(['Metric', 'Value'])
    table_data.append(['', ''])  # Separator

    # Main metrics
    table_data.append(['Position Metrics', ''])
    if 'MPJPE' in metrics:
        table_data.append(['  MPJPE', f"{metrics['MPJPE']*1000:.2f} mm"])

    for key in sorted(metrics.keys()):
        if key.startswith('PCK@'):
            table_data.append([f"  {key}", f"{metrics[key]:.2f}%"])

    if 'num_comparisons' in metrics:
        table_data.append(['  # Comparisons', f"{metrics['num_comparisons']:,}"])

    # Temporal metrics
    table_data.append(['', ''])  # Separator
    table_data.append(['Temporal Metrics', ''])
    table_data.append(['  Avg Velocity', f"{temporal_metrics['velocity_norm']:.4f} m/s"])
    table_data.append(['  Avg Acceleration', f"{temporal_metrics['acceleration_norm']:.4f} m/s²"])
    if not np.isnan(temporal_metrics['jerk_norm']):
        table_data.append(['  Avg Jerk', f"{temporal_metrics['jerk_norm']:.4f} m/s³"])

    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.5, 0.5])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    # Style section headers
    section_rows = [2, 2 + 1 + len([k for k in metrics.keys() if not k.startswith('PCK')]) +
                   len([k for k in metrics.keys() if k.startswith('PCK')]) + 1]
    for row in section_rows:
        if row < len(table_data):
            cell = table[(row, 0)]
            cell.set_facecolor('#E0E0E0')
            cell.set_text_props(weight='bold')

    plt.title('Evaluation Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize SMPL fitting results')

    parser.add_argument('--result_dir', required=True,
                       help='Path to results directory')
    parser.add_argument('--out_dir', default=None,
                       help='Output directory for plots (default: result_dir/visualizations)')
    parser.add_argument('--frame_indices', default='0,100,200,-1',
                       help='Frame indices to visualize (comma-separated, -1 for last)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for output images')
    parser.add_argument('--make_video', action='store_true',
                       help='Export an MP4 skeleton video for the full sequence')
    parser.add_argument('--video_fps', type=int, default=24,
                       help='Frames per second for the exported video')

    args = parser.parse_args()

    # Set output directory
    if args.out_dir is None:
        args.out_dir = os.path.join(args.result_dir, 'visualizations')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"SMPL Fitting Visualization")
    print(f"{'='*80}")
    print(f"Results directory: {args.result_dir}")
    print(f"Output directory: {args.out_dir}")
    print(f"{'='*80}\n")

    # Load results
    print("[1/6] Loading results...")
    results = load_results(args.result_dir)

    pred_joints = results['pred_joints']
    target_joints = results['target_joints']
    meta = results['meta']

    T = pred_joints.shape[0]
    dt = meta['dt']
    joint_mapping = {int(k): int(v) for k, v in meta['joint_mapping'].items()}
    root_addb_idx = next((idx for idx, smpl_idx in joint_mapping.items() if smpl_idx == 0), None)

    print(f"  Loaded {T} frames")

    # Compute errors
    print("\n[2/6] Computing errors...")
    frame_errors = compute_per_frame_errors(pred_joints, target_joints, joint_mapping, root_addb_idx)
    joint_errors = compute_per_joint_errors(pred_joints, target_joints, joint_mapping, root_addb_idx)

    print(f"  Mean error: {np.nanmean(frame_errors)*1000:.2f} mm")
    print(f"  Std error: {np.nanstd(frame_errors)*1000:.2f} mm")

    # Plot error over time
    print("\n[3/6] Plotting error over time...")
    plot_error_over_time(frame_errors, dt,
                        os.path.join(args.out_dir, 'error_over_time.png'))

    # Plot per-joint errors
    print("\n[4/6] Plotting per-joint errors...")
    plot_per_joint_errors(joint_errors,
                         os.path.join(args.out_dir, 'per_joint_errors.png'))

    # Plot 3D skeletons
    print("\n[5/6] Plotting 3D skeletons...")
    frame_indices = [int(x) for x in args.frame_indices.split(',')]
    frame_indices = [T - 1 if x == -1 else x for x in frame_indices]
    frame_indices = [x for x in frame_indices if 0 <= x < T]

    for frame_idx in frame_indices:
        plot_skeleton_3d(pred_joints, target_joints, joint_mapping, frame_idx,
                        root_addb_idx,
                        os.path.join(args.out_dir, f'skeleton_3d_frame_{frame_idx:04d}.png'))

    # Plot temporal analysis
    print("\n[6/6] Plotting temporal analysis...")
    plot_temporal_analysis(pred_joints, target_joints, dt,
                          os.path.join(args.out_dir, 'temporal_analysis.png'))

    # Create summary table
    temporal_metrics = compute_temporal_metrics(pred_joints)
    create_summary_table(meta['metrics'], temporal_metrics,
                        os.path.join(args.out_dir, 'metrics_summary.png'))

    if args.make_video:
        print("\n[7/6] Rendering skeleton video...")
        video_path = os.path.join(args.out_dir, 'skeleton_sequence.mp4')
        create_skeleton_video(
            pred_joints, target_joints, joint_mapping, root_addb_idx,
            video_path, fps=args.video_fps, dpi=args.dpi
        )

    print(f"\n{'='*80}")
    print(f"DONE! Visualizations saved to: {args.out_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
