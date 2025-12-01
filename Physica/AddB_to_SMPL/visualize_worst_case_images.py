#!/usr/bin/env python3
"""
Visualize worst case fitting result (Tiziana2019 Subject12, 180mm MPJPE)
Save static PNG images showing SMPL joints vs B3D markers
"""

import torch
import numpy as np
import nimblephysics as nimble
from pathlib import Path
import argparse
import json
import sys
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
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
    parser.add_argument('--output_dir', type=str, default='/tmp/worst_case_viz')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to visualize')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("WORST CASE VISUALIZATION - Tiziana2019 Subject12 (180mm MPJPE)")
    print("="*80)
    print(f"B3D: {args.b3d}")
    print(f"Result: {args.result_dir}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")

    device = torch.device('cpu')  # Use CPU for visualization
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Select evenly spaced frames
    if T > args.num_frames:
        frame_indices = np.linspace(0, T-1, args.num_frames, dtype=int)
    else:
        frame_indices = np.arange(T)
        args.num_frames = T

    # Map SMPL frame indices to B3D frame indices
    b3d_indices = np.linspace(0, num_frames_b3d-1, T, dtype=int)[frame_indices]

    # Load markers for selected frames
    markers_list = []
    for idx in b3d_indices:
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

    print(f"  Loaded {len(markers_list)} frames of markers")

    # 4. Generate SMPL joints
    print("[4/5] Generating SMPL joints...")
    smpl_joints_list = []

    with torch.no_grad():
        for frame_idx in frame_indices:
            vertices, joints = smpl.forward(
                betas=betas,
                poses=poses[frame_idx],
                trans=trans[frame_idx]
            )
            smpl_joints_list.append(joints.cpu().numpy())

    print(f"  Generated {len(smpl_joints_list)} frames of SMPL joints")

    # 5. Create visualizations
    print(f"[5/5] Creating {args.num_frames} visualization images...")

    # SMPL skeleton connections
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

    for i, frame_idx in enumerate(frame_indices):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot B3D markers (red)
        markers = markers_list[i]
        ax.scatter(markers[:, 0], markers[:, 1], markers[:, 2],
                  c='red', marker='o', s=30, alpha=0.6, label='B3D Markers')

        # Plot SMPL joints (blue)
        smpl_joints = smpl_joints_list[i]
        ax.scatter(smpl_joints[:, 0], smpl_joints[:, 1], smpl_joints[:, 2],
                  c='blue', marker='^', s=80, alpha=0.8, label='SMPL Joints')

        # Connect SMPL joints with skeleton
        for j, k in connections:
            if j < smpl_joints.shape[0] and k < smpl_joints.shape[0]:
                ax.plot([smpl_joints[j, 0], smpl_joints[k, 0]],
                       [smpl_joints[j, 1], smpl_joints[k, 1]],
                       [smpl_joints[j, 2], smpl_joints[k, 2]],
                       'b-', linewidth=2, alpha=0.5)

        # Set axis limits
        all_points = np.vstack([markers, smpl_joints])
        margin = 0.3
        ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(f'Subject12 - MPJPE: {mpjpe:.2f}mm | Frame: {frame_idx+1}/{T}', fontsize=14)
        ax.legend(loc='upper right', fontsize=11)

        # Set viewing angle
        ax.view_init(elev=15, azim=frame_idx * 360 / args.num_frames)

        # Save figure
        output_file = output_dir / f"frame_{i:03d}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved: {output_file}")

    print(f"\n✓ All {args.num_frames} images saved to: {output_dir}")
    print(f"  MPJPE: {mpjpe:.2f}mm")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
