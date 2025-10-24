#!/usr/bin/env python3
"""
Create side-by-side 3D animation with SMPL mesh rendering
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import json
import os
import argparse
import torch
import sys
sys.path.append('/egr/research-zijunlab/kwonjoon/PhysPT/AddB_to_SMPL')
from models.smpl_model import SMPLModel

def load_results(baseline_dir, offset_dir):
    """Load V2 baseline and offset results"""
    # Baseline
    smpl_params_baseline = np.load(os.path.join(baseline_dir, 'smpl_params.npz'))
    target = np.load(os.path.join(baseline_dir, 'target_joints.npy'))

    with open(os.path.join(baseline_dir, 'meta.json'), 'r') as f:
        meta_baseline = json.load(f)

    # Offset
    smpl_params_offset = np.load(os.path.join(offset_dir, 'smpl_params_with_offset.npz'))

    with open(os.path.join(offset_dir, 'meta_with_offset.json'), 'r') as f:
        meta_offset = json.load(f)

    return smpl_params_baseline, smpl_params_offset, target, meta_baseline, meta_offset

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

def create_animation(smpl_model, smpl_params_baseline, smpl_params_offset, target, joint_mapping,
                    meta_baseline, meta_offset, output_path, fps=30, subsample=1):
    """Create side-by-side 3D animation with mesh"""

    # Extract params
    betas_baseline = torch.from_numpy(smpl_params_baseline['betas']).float()
    poses_baseline = torch.from_numpy(smpl_params_baseline['poses']).float()
    trans_baseline = torch.from_numpy(smpl_params_baseline['trans']).float()

    betas_offset = torch.from_numpy(smpl_params_offset['betas']).float()
    poses_offset = torch.from_numpy(smpl_params_offset['poses']).float()
    trans_offset = torch.from_numpy(smpl_params_offset['trans']).float()

    n_frames = poses_baseline.shape[0]
    frame_indices = list(range(0, n_frames, subsample))

    # Extract mapped joints for target
    addb_indices = [int(k) for k in sorted(joint_mapping.keys(), key=int)]
    smpl_indices = [joint_mapping[str(k)] for k in addb_indices]

    # Set up figure with 3 subplots
    fig = plt.figure(figsize=(20, 6))

    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    # Compute bounds from baseline mesh vertices
    print("Computing bounds...")
    verts_baseline = smpl_model(betas_baseline.unsqueeze(0), poses_baseline, trans_baseline).cpu().numpy()
    verts_offset = smpl_model(betas_offset.unsqueeze(0), poses_offset, trans_offset).cpu().numpy()

    all_coords = np.concatenate([
        verts_baseline.reshape(-1, 3),
        verts_offset.reshape(-1, 3),
        target[:, addb_indices, :].reshape(-1, 3)
    ], axis=0)

    x_min, x_max = np.nanmin(all_coords[:, 0]), np.nanmax(all_coords[:, 0])
    y_min, y_max = np.nanmin(all_coords[:, 1]), np.nanmax(all_coords[:, 1])
    z_min, z_max = np.nanmin(all_coords[:, 2]), np.nanmax(all_coords[:, 2])

    # Add padding
    padding = 0.15
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range
    z_min -= padding * z_range
    z_max += padding * z_range

    def set_axes_equal(ax):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=10, azim=90)

    set_axes_equal(ax1)
    set_axes_equal(ax2)
    set_axes_equal(ax3)

    ax1.set_title('V2 Baseline', fontsize=14, fontweight='bold')
    ax2.set_title('V2 + Offset', fontsize=14, fontweight='bold')
    ax3.set_title('Target (AddBiomechanics)', fontsize=14, fontweight='bold')

    # Title text objects
    title1 = ax1.text2D(0.05, 0.05, '', transform=ax1.transAxes, fontsize=10, verticalalignment='bottom')
    title2 = ax2.text2D(0.05, 0.05, '', transform=ax2.transAxes, fontsize=10, verticalalignment='bottom')
    title3 = ax3.text2D(0.05, 0.05, '', transform=ax3.transAxes, fontsize=10, verticalalignment='bottom')

    fig.suptitle(f'V2 Baseline ({meta_baseline["metrics"]["MPJPE"]:.2f} mm) vs V2 + Offset ({meta_offset["metrics_with_offset"]["MPJPE"]:.2f} mm) | Improvement: {meta_offset["metrics_with_offset"]["improvement_mm"]:.2f} mm ({meta_offset["metrics_with_offset"]["improvement_percent"]:.1f}%)',
                 fontsize=14, fontweight='bold')

    # Get SMPL faces for mesh rendering
    faces = smpl_model.faces

    # Initialize mesh collections
    mesh1_collection = None
    mesh2_collection = None
    scatter3 = None

    def update(frame_idx_in_list):
        nonlocal mesh1_collection, mesh2_collection, scatter3

        frame_idx = frame_indices[frame_idx_in_list]

        # Generate meshes for this frame
        verts_baseline_frame = smpl_model(
            betas_baseline.unsqueeze(0),
            poses_baseline[frame_idx].unsqueeze(0),
            trans_baseline[frame_idx].unsqueeze(0)
        ).squeeze(0).cpu().numpy()

        verts_offset_frame = smpl_model(
            betas_offset.unsqueeze(0),
            poses_offset[frame_idx].unsqueeze(0),
            trans_offset[frame_idx].unsqueeze(0)
        ).squeeze(0).cpu().numpy()

        target_joints = target[frame_idx, addb_indices, :]

        # Get joints for error computation
        joints_baseline = smpl_model.joints(
            betas_baseline.unsqueeze(0),
            poses_baseline[frame_idx].unsqueeze(0),
            trans_baseline[frame_idx].unsqueeze(0)
        ).squeeze(0).cpu().numpy()

        joints_offset = smpl_model.joints(
            betas_offset.unsqueeze(0),
            poses_offset[frame_idx].unsqueeze(0),
            trans_offset[frame_idx].unsqueeze(0)
        ).squeeze(0).cpu().numpy()

        # Compute errors
        error_baseline = compute_frame_error(joints_baseline, target[frame_idx], joint_mapping)
        error_offset = compute_frame_error(joints_offset, target[frame_idx], joint_mapping)

        # Remove old collections
        if mesh1_collection is not None:
            mesh1_collection.remove()
        if mesh2_collection is not None:
            mesh2_collection.remove()
        if scatter3 is not None:
            scatter3.remove()

        # Create mesh collections
        mesh1 = []
        for face in faces:
            tri = [verts_baseline_frame[face[0]], verts_baseline_frame[face[1]], verts_baseline_frame[face[2]]]
            mesh1.append(tri)

        mesh2 = []
        for face in faces:
            tri = [verts_offset_frame[face[0]], verts_offset_frame[face[1]], verts_offset_frame[face[2]]]
            mesh2.append(tri)

        mesh1_collection = Poly3DCollection(mesh1, alpha=0.3, facecolor='cyan', edgecolor='blue', linewidth=0.1)
        mesh2_collection = Poly3DCollection(mesh2, alpha=0.3, facecolor='lightgreen', edgecolor='green', linewidth=0.1)

        ax1.add_collection3d(mesh1_collection)
        ax2.add_collection3d(mesh2_collection)

        # Plot target joints
        scatter3 = ax3.scatter(target_joints[:, 0], target_joints[:, 1], target_joints[:, 2],
                               c='red', marker='o', s=100, alpha=0.8)

        # Update titles
        title1.set_text(f'Frame {frame_idx}/{n_frames-1} | Error: {error_baseline:.1f} mm')
        title2.set_text(f'Frame {frame_idx}/{n_frames-1} | Error: {error_offset:.1f} mm')
        title3.set_text(f'Frame {frame_idx}/{n_frames-1}')

        return mesh1_collection, mesh2_collection, scatter3, title1, title2, title3

    # Create animation
    print(f"Creating animation with {len(frame_indices)} frames...")
    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices),
                                   interval=1000/fps, blit=False, repeat=True)

    # Save animation
    print(f"Saving to {output_path}...")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='SMPL2AddBiomechanics'), bitrate=8000)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"Video saved: {output_path}")
    print(f"Total frames: {len(frame_indices)}")
    print(f"Duration: {len(frame_indices)/fps:.1f} seconds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dir', type=str, required=True)
    parser.add_argument('--offset_dir', type=str, required=True)
    parser.add_argument('--smpl_model', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print(f"Loading SMPL model from {args.smpl_model}...")
    smpl_model = SMPLModel(args.smpl_model, device=args.device)

    print(f"Loading baseline from {args.baseline_dir}...")
    print(f"Loading offset from {args.offset_dir}...")
    smpl_params_baseline, smpl_params_offset, target, meta_baseline, meta_offset = load_results(
        args.baseline_dir, args.offset_dir)

    joint_mapping = meta_baseline['joint_mapping']

    print(f"\nDataset info:")
    print(f"  Total frames: {smpl_params_baseline['poses'].shape[0]}")
    print(f"  V2 Baseline MPJPE: {meta_baseline['metrics']['MPJPE']:.2f} mm")
    print(f"  V2 + Offset MPJPE: {meta_offset['metrics_with_offset']['MPJPE']:.2f} mm")
    print(f"  Improvement: {meta_offset['metrics_with_offset']['improvement_mm']:.2f} mm ({meta_offset['metrics_with_offset']['improvement_percent']:.1f}%)")

    create_animation(smpl_model, smpl_params_baseline, smpl_params_offset, target, joint_mapping,
                    meta_baseline, meta_offset, args.output,
                    fps=args.fps, subsample=args.subsample)

if __name__ == '__main__':
    main()
