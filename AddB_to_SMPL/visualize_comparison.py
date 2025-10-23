#!/usr/bin/env python3
"""
Visualize comparison between V2 baseline and V2 + offset results
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os

def load_v2_baseline(result_dir):
    """Load V2 baseline results"""
    pred_joints = np.load(os.path.join(result_dir, 'pred_joints.npy'))
    target_joints = np.load(os.path.join(result_dir, 'target_joints.npy'))

    with open(os.path.join(result_dir, 'meta.json'), 'r') as f:
        meta = json.load(f)

    return pred_joints, target_joints, meta

def load_v2_with_offset(result_dir):
    """Load V2 + offset results"""
    pred_joints = np.load(os.path.join(result_dir, 'pred_joints_with_offset.npy'))
    target_joints = np.load(os.path.join(result_dir, 'target_joints.npy'))

    with open(os.path.join(result_dir, 'meta_with_offset.json'), 'r') as f:
        meta = json.load(f)

    return pred_joints, target_joints, meta

def compute_per_joint_errors(pred_joints, target_joints, joint_mapping):
    """Compute per-joint errors using mapping"""
    errors = []
    for addb_idx_str, smpl_idx in joint_mapping.items():
        addb_idx = int(addb_idx_str)
        pred = pred_joints[:, smpl_idx, :]
        target = target_joints[:, addb_idx, :]

        # Compute error
        diff = pred - target
        error = np.linalg.norm(diff, axis=1)
        errors.append(error)

    errors = np.array(errors)  # (num_joints, num_frames)
    mean_errors = np.nanmean(errors, axis=1)  # (num_joints,)

    return errors, mean_errors

def visualize_frame_comparison(pred_baseline, pred_offset, target, frame_idx, joint_mapping, save_path):
    """Visualize a single frame with baseline, offset, and target"""
    fig = plt.figure(figsize=(20, 5))

    # Extract mapped joints
    addb_indices = [int(k) for k in sorted(joint_mapping.keys(), key=int)]
    smpl_indices = [joint_mapping[str(k)] for k in addb_indices]

    pred_baseline_mapped = pred_baseline[frame_idx, smpl_indices, :]
    pred_offset_mapped = pred_offset[frame_idx, smpl_indices, :]
    target_mapped = target[frame_idx, addb_indices, :]

    # V2 Baseline
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.scatter(pred_baseline_mapped[:, 0], pred_baseline_mapped[:, 1], pred_baseline_mapped[:, 2],
                c='blue', marker='o', s=100, label='Baseline')
    ax1.set_title(f'V2 Baseline\n(Frame {frame_idx})')
    ax1.legend()

    # V2 + Offset
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.scatter(pred_offset_mapped[:, 0], pred_offset_mapped[:, 1], pred_offset_mapped[:, 2],
                c='green', marker='s', s=100, label='With Offset')
    ax2.set_title(f'V2 + Offset\n(Frame {frame_idx})')
    ax2.legend()

    # Target
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(target_mapped[:, 0], target_mapped[:, 1], target_mapped[:, 2],
                c='red', marker='^', s=100, label='Target')
    ax3.set_title(f'Target\n(Frame {frame_idx})')
    ax3.legend()

    # Overlay all
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.scatter(pred_baseline_mapped[:, 0], pred_baseline_mapped[:, 1], pred_baseline_mapped[:, 2],
                c='blue', marker='o', s=80, alpha=0.5, label='Baseline')
    ax4.scatter(pred_offset_mapped[:, 0], pred_offset_mapped[:, 1], pred_offset_mapped[:, 2],
                c='green', marker='s', s=80, alpha=0.5, label='With Offset')
    ax4.scatter(target_mapped[:, 0], target_mapped[:, 1], target_mapped[:, 2],
                c='red', marker='^', s=80, alpha=0.5, label='Target')

    # Draw lines from target to predictions
    for i in range(len(target_mapped)):
        ax4.plot([target_mapped[i, 0], pred_baseline_mapped[i, 0]],
                [target_mapped[i, 1], pred_baseline_mapped[i, 1]],
                [target_mapped[i, 2], pred_baseline_mapped[i, 2]],
                'b--', alpha=0.3, linewidth=1)
        ax4.plot([target_mapped[i, 0], pred_offset_mapped[i, 0]],
                [target_mapped[i, 1], pred_offset_mapped[i, 1]],
                [target_mapped[i, 2], pred_offset_mapped[i, 2]],
                'g--', alpha=0.3, linewidth=1)

    ax4.set_title(f'Overlay\n(Frame {frame_idx})')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def main():
    baseline_dir = 'results_v2_baseline/no_arm_camargo2021_ab10'
    offset_dir = 'results_v2_with_offset/no_arm_camargo2021_ab10'
    output_dir = 'visualizations_comparison'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading V2 baseline results...")
    pred_baseline, target, meta_baseline = load_v2_baseline(baseline_dir)

    print("Loading V2 + offset results...")
    pred_offset, _, meta_offset = load_v2_with_offset(offset_dir)

    joint_mapping = meta_baseline['joint_mapping']

    print(f"\nV2 Baseline MPJPE:  {meta_baseline['metrics']['MPJPE']:.2f} mm")
    print(f"V2 + Offset MPJPE:  {meta_offset['metrics_with_offset']['MPJPE']:.2f} mm")
    print(f"Improvement:        {meta_offset['metrics_with_offset']['improvement_mm']:.2f} mm ({meta_offset['metrics_with_offset']['improvement_percent']:.1f}%)")

    # Compute errors over time
    print("\nComputing per-frame errors...")
    errors_baseline, mean_errors_baseline = compute_per_joint_errors(pred_baseline, target, joint_mapping)
    errors_offset, mean_errors_offset = compute_per_joint_errors(pred_offset, target, joint_mapping)

    # Error over time plot
    print("\nPlotting error over time...")
    plt.figure(figsize=(14, 6))

    mean_error_per_frame_baseline = np.nanmean(errors_baseline, axis=0)
    mean_error_per_frame_offset = np.nanmean(errors_offset, axis=0)

    plt.plot(mean_error_per_frame_baseline * 1000, label=f'V2 Baseline ({meta_baseline["metrics"]["MPJPE"]:.2f} mm)', linewidth=2, alpha=0.7)
    plt.plot(mean_error_per_frame_offset * 1000, label=f'V2 + Offset ({meta_offset["metrics_with_offset"]["MPJPE"]:.2f} mm)', linewidth=2, alpha=0.7)

    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Mean Error (mm)', fontsize=12)
    plt.title(f'MPJPE Over Time\nImprovement: {meta_offset["metrics_with_offset"]["improvement_mm"]:.2f} mm ({meta_offset["metrics_with_offset"]["improvement_percent"]:.1f}%)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_over_time_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/error_over_time_comparison.png")

    # Per-joint error comparison
    print("\nPlotting per-joint errors...")
    plt.figure(figsize=(12, 6))

    joint_names = list(meta_baseline['joint_mapping_named'].values())
    x = np.arange(len(joint_names))
    width = 0.35

    plt.bar(x - width/2, mean_errors_baseline * 1000, width, label='V2 Baseline', alpha=0.7)
    plt.bar(x + width/2, mean_errors_offset * 1000, width, label='V2 + Offset', alpha=0.7)

    plt.xlabel('Joint', fontsize=12)
    plt.ylabel('Mean Error (mm)', fontsize=12)
    plt.title('Per-Joint MPJPE Comparison', fontsize=14)
    plt.xticks(x, joint_names, rotation=45, ha='right')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_joint_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/per_joint_comparison.png")

    # Visualize sample frames
    print("\nVisualizing sample frames...")
    n_frames = pred_baseline.shape[0]
    sample_frames = [0, n_frames//4, n_frames//2, 3*n_frames//4, n_frames-1]

    for frame_idx in sample_frames:
        save_path = os.path.join(output_dir, f'frame_{frame_idx:04d}_comparison.png')
        visualize_frame_comparison(pred_baseline, pred_offset, target, frame_idx, joint_mapping, save_path)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Generated {len(sample_frames)} frame comparisons")
    print(f"Generated 2 summary plots")
    print("="*80)

if __name__ == '__main__':
    main()
