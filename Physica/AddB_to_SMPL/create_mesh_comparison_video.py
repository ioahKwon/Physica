#!/usr/bin/env python3
"""
Create side-by-side MP4 comparison video with SMPL mesh rendering
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel


def create_mesh_comparison_video(baseline_npz, fixed_npz, smpl_model_path, output_mp4):
    """
    Create side-by-side comparison video with full SMPL mesh rendering

    Args:
        baseline_npz: Path to baseline smpl_params.npz
        fixed_npz: Path to fixed smpl_params.npz
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
    print(f"\nLoading fixed from {fixed_npz}...")
    fixed_data = np.load(fixed_npz)
    betas_fixed = fixed_data['betas']
    poses_fixed = fixed_data['poses']
    trans_fixed = fixed_data['trans']
    T_fixed = poses_fixed.shape[0]
    print(f"  Loaded {T_fixed} frames")

    T = min(T_baseline, T_fixed)
    print(f"\nUsing {T} frames for comparison")

    # Generate SMPL meshes for both
    print("\nGenerating SMPL meshes for baseline...")
    vertices_baseline, joints_baseline, faces = generate_smpl_meshes(
        smpl_model, betas_baseline, poses_baseline, trans_baseline, device, T
    )

    print("Generating SMPL meshes for fixed...")
    vertices_fixed, joints_fixed, _ = generate_smpl_meshes(
        smpl_model, betas_fixed, poses_fixed, trans_fixed, device, T
    )

    # Compute foot rotation magnitudes for display
    baseline_l_rot = np.linalg.norm(poses_baseline[:T, 10, :], axis=1)
    baseline_r_rot = np.linalg.norm(poses_baseline[:T, 11, :], axis=1)
    fixed_l_rot = np.linalg.norm(poses_fixed[:T, 10, :], axis=1)
    fixed_r_rot = np.linalg.norm(poses_fixed[:T, 11, :], axis=1)

    print(f"\nFoot rotation statistics:")
    print(f"  Baseline - Left:  mean={np.degrees(baseline_l_rot.mean()):.2f}°, max={np.degrees(baseline_l_rot.max()):.2f}°")
    print(f"  Baseline - Right: mean={np.degrees(baseline_r_rot.mean()):.2f}°, max={np.degrees(baseline_r_rot.max()):.2f}°")
    print(f"  Fixed - Left:  mean={np.degrees(fixed_l_rot.mean()):.2f}°, max={np.degrees(fixed_l_rot.max()):.2f}°")
    print(f"  Fixed - Right: mean={np.degrees(fixed_r_rot.mean()):.2f}°, max={np.degrees(fixed_r_rot.max()):.2f}°")

    # Create visualization
    print(f"\nCreating MP4 video with mesh rendering...")
    create_animation(
        vertices_baseline, vertices_fixed,
        joints_baseline, joints_fixed,
        faces,
        baseline_l_rot, baseline_r_rot,
        fixed_l_rot, fixed_r_rot,
        output_mp4, T
    )

    print(f"\n✓ Saved video to {output_mp4}")


def generate_smpl_meshes(smpl_model, betas, poses, trans, device, T):
    """Generate SMPL mesh vertices and joints"""
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

    vertices_np = vertices.cpu().numpy()  # (T, 6890, 3)
    joints_np = joints.cpu().numpy()  # (T, 24, 3)
    faces = smpl_model.faces  # (13776, 3)

    return vertices_np, joints_np, faces


def create_animation(vertices_baseline, vertices_fixed, joints_baseline, joints_fixed,
                     faces, baseline_l, baseline_r, fixed_l, fixed_r, output_mp4, T):
    """Create side-by-side animation with mesh rendering"""

    # Compute global axis limits (use all frames)
    all_verts = np.vstack([vertices_baseline.reshape(-1, 3), vertices_fixed.reshape(-1, 3)])

    margin = 0.2
    xlim = [all_verts[:, 0].min() - margin, all_verts[:, 0].max() + margin]
    ylim = [all_verts[:, 1].min() - margin, all_verts[:, 1].max() + margin]
    zlim = [all_verts[:, 2].min() - margin, all_verts[:, 2].max() + margin]

    # Create figure
    fig = plt.figure(figsize=(20, 9))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    def update(frame):
        ax1.clear()
        ax2.clear()

        # Baseline (left panel)
        verts_baseline = vertices_baseline[frame]
        joints_b = joints_baseline[frame]

        # Create mesh
        mesh_baseline = Poly3DCollection(verts_baseline[faces], alpha=0.3, facecolor='lightblue', edgecolor='none')
        ax1.add_collection3d(mesh_baseline)

        # Highlight foot vertices (approximate region)
        # Just highlight the foot joints
        ax1.scatter(joints_b[10:12, 0], joints_b[10:12, 1], joints_b[10:12, 2],
                   c='red', s=100, alpha=1.0, marker='o', label='Foot joints')

        # Fixed (right panel)
        verts_fixed = vertices_fixed[frame]
        joints_f = joints_fixed[frame]

        # Create mesh
        mesh_fixed = Poly3DCollection(verts_fixed[faces], alpha=0.3, facecolor='lightgreen', edgecolor='none')
        ax2.add_collection3d(mesh_fixed)

        # Highlight foot joints
        ax2.scatter(joints_f[10:12, 0], joints_f[10:12, 1], joints_f[10:12, 2],
                   c='red', s=100, alpha=1.0, marker='o', label='Foot joints')

        # Set limits and labels
        for ax in [ax1, ax2]:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=10, azim=45)
            ax.set_box_aspect([1, 1, 1])

        ax1.set_title(f'Baseline (Frame {frame}/{T})\n'
                     f'L foot: {np.degrees(baseline_l[frame]):.1f}° | '
                     f'R foot: {np.degrees(baseline_r[frame]):.1f}°',
                     fontsize=12, color='blue')
        ax2.set_title(f'Fixed: Foot Rotation Corrected (Frame {frame}/{T})\n'
                     f'L foot: {np.degrees(fixed_l[frame]):.1f}° | '
                     f'R foot: {np.degrees(fixed_r[frame]):.1f}°',
                     fontsize=12, color='green')

        fig.suptitle('SMPL Mesh Comparison: Baseline vs Fixed',
                    fontsize=14, fontweight='bold')

    # Create animation
    anim = FuncAnimation(fig, update, frames=T, interval=50)

    # Save as MP4
    writer = FFMpegWriter(fps=20, bitrate=3000)
    anim.save(output_mp4, writer=writer)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create mesh comparison video')
    parser.add_argument('--baseline', type=str, required=True,
                       help='Path to baseline smpl_params.npz')
    parser.add_argument('--fixed', type=str, required=True,
                       help='Path to fixed smpl_params.npz')
    parser.add_argument('--smpl_model', type=str,
                       default='models/smpl_model.pkl',
                       help='Path to SMPL model')
    parser.add_argument('--output', type=str, required=True,
                       help='Output MP4 file path')
    args = parser.parse_args()

    create_mesh_comparison_video(args.baseline, args.fixed, args.smpl_model, args.output)


if __name__ == '__main__':
    main()
