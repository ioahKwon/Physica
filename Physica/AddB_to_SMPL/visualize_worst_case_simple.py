#!/usr/bin/env python3
"""
Visualize worst case fitting result (Tiziana2019 Subject12, 180mm MPJPE)
Simple matplotlib-based visualization showing SMPL joints vs B3D markers
"""

import torch
import numpy as np
import nimblephysics as nimble
from pathlib import Path
import argparse
import json
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

# Import SMPL model
sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b3d', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject12/Subject12.b3d')
    parser.add_argument('--result_dir', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames/Tiziana2019_Formatted_With_Arm_Subject12')
    parser.add_argument('--smpl_model', type=str, default='models/smpl_model.pkl')
    parser.add_argument('--output_video', type=str, default='/tmp/worst_case_subject12.mp4')
    parser.add_argument('--max_frames', type=int, default=200, help='Max frames to render')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    print("\n" + "="*80)
    print("WORST CASE VISUALIZATION - Tiziana2019 Subject12 (180mm MPJPE)")
    print("="*80)
    print(f"B3D: {args.b3d}")
    print(f"Result: {args.result_dir}")
    print(f"Output: {args.output_video}")
    print("="*80 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load SMPL model
    print("[1/5] Loading SMPL model...")
    smpl = SMPLModel(args.smpl_model, device=device)

    # 2. Load SMPL parameters
    print("[2/5] Loading SMPL fit results...")
    result_path = Path(args.result_dir)
    inner_dirs = list(result_path.glob("with_arm_*"))
    if not inner_dirs:
        print(f"ERROR: No with_arm_* directory found in {result_path}")
        sys.exit(1)

    npz_file = inner_dirs[0] / "smpl_params.npz"
    if not npz_file.exists():
        print(f"ERROR: {npz_file} does not exist")
        sys.exit(1)

    data = np.load(npz_file)
    betas = torch.from_numpy(data['betas']).float().to(device)
    poses = torch.from_numpy(data['poses']).float().to(device)  # (T, 24, 3)
    trans = torch.from_numpy(data['trans']).float().to(device)  # (T, 3)

    T = poses.shape[0]
    print(f"  Loaded {T} frames")

    # Load meta to confirm MPJPE
    meta_file = inner_dirs[0] / "meta.json"
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    mpjpe = meta.get('metrics', {}).get('MPJPE', 0)
    print(f"  MPJPE: {mpjpe:.2f}mm")

    # 3. Load B3D markers
    print("[3/5] Loading B3D markers...")
    subject = nimble.biomechanics.SubjectOnDisk(args.b3d)
    num_frames_b3d = subject.getTrialLength(trial=0)
    print(f"  B3D has {num_frames_b3d} frames")

    # Sample frames to match SMPL
    if num_frames_b3d > args.max_frames:
        indices = np.linspace(0, num_frames_b3d-1, args.max_frames, dtype=int)
    else:
        indices = np.arange(num_frames_b3d)

    # Load all marker data
    markers_list = []
    for idx in indices:
        frames = subject.readFrames(
            trial=0,
            startFrame=int(idx),
            numFramesToRead=1,
            includeProcessingPasses=True,
            includeSensorData=False,
            stride=1
        )
        if frames:
            marker_obs = frames[0].markerObservations
            # markerObservations is a list of 3D positions
            if isinstance(marker_obs, list):
                markers = np.array(marker_obs)
                # Ensure it's (N, 3) shape
                if markers.ndim == 1:
                    markers = markers.reshape(-1, 3)
            else:
                # If it's a dict, extract values
                markers = np.array([pos for pos in marker_obs.values()])
            markers_list.append(markers)

    print(f"  Loaded {len(markers_list)} frames of markers ({markers_list[0].shape[0]} markers per frame)")

    # Interpolate SMPL params if needed
    if T != len(indices):
        print(f"  Interpolating SMPL params from {T} to {len(indices)} frames...")
        from scipy.interpolate import interp1d

        t_old = np.linspace(0, 1, T)
        t_new = np.linspace(0, 1, len(indices))

        poses_np = poses.cpu().numpy().reshape(T, -1)
        poses_interp = interp1d(t_old, poses_np, axis=0, kind='linear')(t_new)
        poses = torch.from_numpy(poses_interp.reshape(len(indices), 24, 3)).float().to(device)

        trans_np = trans.cpu().numpy()
        trans_interp = interp1d(t_old, trans_np, axis=0, kind='linear')(t_new)
        trans = torch.from_numpy(trans_interp).float().to(device)

        T = len(indices)

    # 4. Generate SMPL joints
    print("[4/5] Generating SMPL joints...")
    smpl_joints_list = []

    with torch.no_grad():
        for i in range(T):
            vertices, joints = smpl.forward(
                betas=betas,
                poses=poses[i],
                trans=trans[i]
            )
            joints_np = joints[:24].cpu().numpy()  # 24 joints
            smpl_joints_list.append(joints_np)

    print(f"  Generated {len(smpl_joints_list)} frames of SMPL joints")

    # 5. Render video
    print(f"[5/5] Rendering video ({args.output_video})...")

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()

        # Plot B3D markers (red)
        markers = markers_list[frame]
        ax.scatter(markers[:, 0], markers[:, 1], markers[:, 2],
                  c='red', marker='o', s=50, alpha=0.6, label='B3D Markers')

        # Plot SMPL joints (blue)
        smpl_joints = smpl_joints_list[frame]
        ax.scatter(smpl_joints[:, 0], smpl_joints[:, 1], smpl_joints[:, 2],
                  c='blue', marker='^', s=100, alpha=0.8, label='SMPL Joints')

        # Connect SMPL joints with skeleton
        # SMPL skeleton connections (simplified)
        connections = [
            (0, 1), (0, 2), (0, 3),  # Pelvis to legs
            (1, 4), (2, 5), (3, 6),  # Upper legs
            (4, 7), (5, 8), (6, 9),  # Lower legs
            (9, 12), (9, 13), (9, 14),  # Spine
            (12, 15),  # Neck to head
            (13, 16), (14, 17),  # Shoulders
            (16, 18), (17, 19),  # Upper arms
            (18, 20), (19, 21),  # Lower arms
            (20, 22), (21, 23),  # Hands
        ]

        for i, j in connections:
            if i < smpl_joints.shape[0] and j < smpl_joints.shape[0]:
                ax.plot([smpl_joints[i, 0], smpl_joints[j, 0]],
                       [smpl_joints[i, 1], smpl_joints[j, 1]],
                       [smpl_joints[i, 2], smpl_joints[j, 2]],
                       'b-', linewidth=2, alpha=0.5)

        # Set axis limits (fixed across all frames)
        all_points = np.vstack([markers, smpl_joints])
        margin = 0.5
        ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Subject12 - MPJPE: {mpjpe:.2f}mm | Frame: {frame+1}/{T}', fontsize=16)
        ax.legend(loc='upper right')

        # Set viewing angle
        ax.view_init(elev=10, azim=frame * 360 / T)  # Rotate view

        return ax,

    # Create animation
    anim = FuncAnimation(fig, update, frames=T, interval=1000/args.fps, blit=False)

    # Save as MP4
    writer = FFMpegWriter(fps=args.fps, metadata=dict(artist='SMPL Fit Visualization'), bitrate=5000)
    anim.save(args.output_video, writer=writer)

    plt.close(fig)

    print(f"\n✓ Video saved to: {args.output_video}")
    print(f"  Duration: {T/args.fps:.1f}s")
    print(f"  FPS: {args.fps}")
    print(f"  Frames: {T}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
