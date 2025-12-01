#!/usr/bin/env python3
"""
Create side-by-side MP4 comparison video: Baseline vs Fixed (foot rotation)
Can be used for Option 3 or Option 4 comparisons.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel


def create_comparison_video(baseline_npz, fixed_npz, smpl_model_path, output_mp4):
    """
    Create side-by-side comparison video

    Args:
        baseline_npz: Path to baseline smpl_params.npz
        fixed_npz: Path to Option 3 fixed smpl_params.npz
        smpl_model_path: Path to SMPL model
        output_mp4: Output MP4 file path
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load SMPL model
    print(f"\nLoading SMPL model from {smpl_model_path}...")
    smpl_model = SMPLModel(smpl_model_path, device=device)

    # Load baseline results
    print(f"\nLoading baseline from {baseline_npz}...")
    baseline_data = np.load(baseline_npz)
    betas_baseline = baseline_data['betas']
    poses_baseline = baseline_data['poses']
    trans_baseline = baseline_data['trans']
    T_baseline = poses_baseline.shape[0]
    print(f"  Loaded {T_baseline} frames")

    # Load fixed results
    print(f"\nLoading Option 3 fixed from {fixed_npz}...")
    fixed_data = np.load(fixed_npz)
    betas_fixed = fixed_data['betas']
    poses_fixed = fixed_data['poses']
    trans_fixed = fixed_data['trans']
    T_fixed = poses_fixed.shape[0]
    print(f"  Loaded {T_fixed} frames")

    T = min(T_baseline, T_fixed)
    print(f"\nUsing {T} frames for comparison")

    # Generate SMPL meshes for both
    print("\nGenerating SMPL joints for baseline...")
    joints_baseline = generate_smpl_joints(smpl_model, betas_baseline, poses_baseline, trans_baseline, device, T)

    print("Generating SMPL joints for Option 3...")
    joints_fixed = generate_smpl_joints(smpl_model, betas_fixed, poses_fixed, trans_fixed, device, T)

    # Compute foot rotation magnitudes for display
    baseline_l_rot = np.linalg.norm(poses_baseline[:T, 10, :], axis=1)
    baseline_r_rot = np.linalg.norm(poses_baseline[:T, 11, :], axis=1)
    fixed_l_rot = np.linalg.norm(poses_fixed[:T, 10, :], axis=1)
    fixed_r_rot = np.linalg.norm(poses_fixed[:T, 11, :], axis=1)

    print(f"\nFoot rotation statistics:")
    print(f"  Baseline - Left:  mean={np.degrees(baseline_l_rot.mean()):.2f}°, max={np.degrees(baseline_l_rot.max()):.2f}°")
    print(f"  Baseline - Right: mean={np.degrees(baseline_r_rot.mean()):.2f}°, max={np.degrees(baseline_r_rot.max()):.2f}°")
    print(f"  Option 3 - Left:  mean={np.degrees(fixed_l_rot.mean()):.2f}°, max={np.degrees(fixed_l_rot.max()):.2f}°")
    print(f"  Option 3 - Right: mean={np.degrees(fixed_r_rot.mean()):.2f}°, max={np.degrees(fixed_r_rot.max()):.2f}°")

    # Create visualization
    print(f"\nCreating MP4 video...")
    create_animation(
        joints_baseline, joints_fixed,
        baseline_l_rot, baseline_r_rot,
        fixed_l_rot, fixed_r_rot,
        output_mp4, T
    )

    print(f"\n✓ Saved video to {output_mp4}")


def generate_smpl_joints(smpl_model, betas, poses, trans, device, T):
    """Generate SMPL joint positions"""
    # Expand betas if needed
    if len(betas.shape) == 1:
        betas_expanded = np.tile(betas, (T, 1))
    else:
        betas_expanded = betas[:T]

    # Reshape poses if needed
    if len(poses.shape) == 3:
        poses_flat = poses[:T].reshape(T, -1)
    else:
        poses_flat = poses[:T]

    betas_t = torch.from_numpy(betas_expanded).float().to(device)
    poses_t = torch.from_numpy(poses_flat).float().to(device)
    trans_t = torch.from_numpy(trans[:T]).float().to(device)

    with torch.no_grad():
        vertices, joints = smpl_model.forward(betas=betas_t, poses=poses_t, trans=trans_t)

    return joints.cpu().numpy()  # (T, 24, 3)


def create_animation(joints_baseline, joints_fixed, baseline_l, baseline_r, fixed_l, fixed_r, output_mp4, T):
    """Create side-by-side animation"""
    # SMPL skeleton connections
    smpl_skeleton = [
        (0, 1), (0, 2), (0, 3),  # pelvis to hips/spine
        (1, 4), (2, 5), (3, 6),  # hips to knees, spine to spine
        (4, 7), (5, 8), (6, 9),  # knees to ankles, spine to chest
        (7, 10), (8, 11),  # ankles to feet
        (9, 12), (9, 13), (9, 14),  # chest to neck/collars
        (12, 15),  # neck to head
        (13, 16), (14, 17),  # collars to shoulders
        (16, 18), (17, 19),  # shoulders to elbows
        (18, 20), (19, 21),  # elbows to wrists
        (20, 22), (21, 23),  # wrists to hands
    ]

    # Create figure
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Compute global axis limits (use all frames)
    all_baseline = joints_baseline.reshape(-1, 3)
    all_fixed = joints_fixed.reshape(-1, 3)
    all_points = np.vstack([all_baseline, all_fixed])

    margin = 0.3
    xlim = [all_points[:, 0].min() - margin, all_points[:, 0].max() + margin]
    ylim = [all_points[:, 1].min() - margin, all_points[:, 1].max() + margin]
    zlim = [all_points[:, 2].min() - margin, all_points[:, 2].max() + margin]

    def update(frame):
        ax1.clear()
        ax2.clear()

        # Baseline (left panel)
        baseline = joints_baseline[frame]
        ax1.scatter(baseline[:, 0], baseline[:, 1], baseline[:, 2],
                   c='blue', s=30, alpha=0.6)

        # Highlight foot joints
        ax1.scatter(baseline[10:12, 0], baseline[10:12, 1], baseline[10:12, 2],
                   c='red', s=80, alpha=0.8, marker='o')

        for connection in smpl_skeleton:
            ax1.plot([baseline[connection[0], 0], baseline[connection[1], 0]],
                    [baseline[connection[0], 1], baseline[connection[1], 1]],
                    [baseline[connection[0], 2], baseline[connection[1], 2]],
                    'b-', alpha=0.4, linewidth=2)

        # Option 3 Fixed (right panel)
        fixed = joints_fixed[frame]
        ax2.scatter(fixed[:, 0], fixed[:, 1], fixed[:, 2],
                   c='green', s=30, alpha=0.6)

        # Highlight foot joints
        ax2.scatter(fixed[10:12, 0], fixed[10:12, 1], fixed[10:12, 2],
                   c='red', s=80, alpha=0.8, marker='o')

        for connection in smpl_skeleton:
            ax2.plot([fixed[connection[0], 0], fixed[connection[1], 0]],
                    [fixed[connection[0], 1], fixed[connection[1], 1]],
                    [fixed[connection[0], 2], fixed[connection[1], 2]],
                    'g-', alpha=0.4, linewidth=2)

        # Set limits and labels
        for ax in [ax1, ax2]:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=10, azim=45)

        ax1.set_title(f'Baseline (Frame {frame}/{T})\n'
                     f'L foot: {np.degrees(baseline_l[frame]):.1f}° | '
                     f'R foot: {np.degrees(baseline_r[frame]):.1f}°',
                     fontsize=12, color='blue')
        ax2.set_title(f'Fixed: Foot Rotation Corrected (Frame {frame}/{T})\n'
                     f'L foot: {np.degrees(fixed_l[frame]):.1f}° | '
                     f'R foot: {np.degrees(fixed_r[frame]):.1f}°',
                     fontsize=12, color='green')

        fig.suptitle('Foot Orientation Fix Comparison',
                    fontsize=14, fontweight='bold')

    # Create animation
    anim = FuncAnimation(fig, update, frames=T, interval=50)

    # Save as MP4
    writer = FFMpegWriter(fps=20, bitrate=2000)
    anim.save(output_mp4, writer=writer)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create Option 3 comparison video')
    parser.add_argument('--baseline', type=str, required=True,
                       help='Path to baseline smpl_params.npz')
    parser.add_argument('--fixed', type=str, required=True,
                       help='Path to Option 3 fixed smpl_params.npz')
    parser.add_argument('--smpl_model', type=str,
                       default='models/smpl_model.pkl',
                       help='Path to SMPL model')
    parser.add_argument('--output', type=str, required=True,
                       help='Output MP4 file path')
    args = parser.parse_args()

    create_comparison_video(args.baseline, args.fixed, args.smpl_model, args.output)


if __name__ == '__main__':
    main()
