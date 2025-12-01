#!/usr/bin/env python3
"""
3-way visualization for worst case (Tiziana2019 Subject12)
1. Static PNG images (already done - visualize_worst_case_images.py)
2. MP4 with proper B3D joint centers vs SMPL joints
3. MP4 with skeleton overlay
"""

import torch
import numpy as np
import nimblephysics as nimble
from pathlib import Path
import argparse
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel

def load_b3d_joints(b3d_path, num_frames=-1, processing_pass=0):
    """Load B3D joint centers properly using processingPasses"""
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
    total_frames = subject.getTrialLength(trial=0)

    if num_frames < 0:
        num_frames = total_frames
    else:
        num_frames = min(num_frames, total_frames)

    frames = subject.readFrames(
        trial=0,
        startFrame=0,
        numFramesToRead=num_frames,
        includeProcessingPasses=True,
        includeSensorData=False,
        stride=1,
        contactThreshold=1.0
    )

    def frame_joint_centers(frame):
        if hasattr(frame, 'processingPasses') and len(frame.processingPasses) > 0:
            idx = min(processing_pass, len(frame.processingPasses) - 1)
            return np.asarray(frame.processingPasses[idx].jointCenters, dtype=np.float32)
        return np.asarray(frame.jointCenters, dtype=np.float32)

    first = frame_joint_centers(frames[0])
    if first.ndim != 1 or first.size % 3 != 0:
        raise ValueError("Unexpected joint center layout in .b3d file")

    num_joints = first.size // 3
    joints = np.zeros((len(frames), num_joints, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        data = frame_joint_centers(frame)
        joints[i] = data.reshape(-1, 3)[:num_joints]

    return joints

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b3d', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject12/Subject12.b3d')
    parser.add_argument('--result_dir', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames/Tiziana2019_Formatted_With_Arm_Subject12')
    parser.add_argument('--smpl_model', type=str, default='models/smpl_model.pkl')
    parser.add_argument('--output_dir', type=str, default='/egr/research-zijunlab/kwonjoon')
    parser.add_argument('--max_frames', type=int, default=100)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    print("\n" + "="*80)
    print("3-WAY WORST CASE VISUALIZATION - Tiziana2019 Subject12")
    print("="*80)
    print(f"B3D: {args.b3d}")
    print(f"Result: {args.result_dir}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")

    device = torch.device('cpu')  # Use CPU for visualization

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
    poses = torch.from_numpy(data['poses']).float().to(device)
    trans = torch.from_numpy(data['trans']).float().to(device)

    T = poses.shape[0]
    print(f"  Loaded {T} frames")

    # Load meta to confirm MPJPE
    meta_file = inner_dirs[0] / "meta.json"
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    mpjpe = meta.get('metrics', {}).get('MPJPE', 0)
    print(f"  MPJPE: {mpjpe:.2f}mm")

    # 3. Load B3D joint centers
    print("[3/5] Loading B3D joint centers...")
    b3d_joints = load_b3d_joints(args.b3d, num_frames=args.max_frames)
    print(f"  Loaded {b3d_joints.shape[0]} frames with {b3d_joints.shape[1]} joints")

    # Interpolate SMPL if needed
    if T != b3d_joints.shape[0]:
        print(f"  Interpolating SMPL params from {T} to {b3d_joints.shape[0]} frames...")
        from scipy.interpolate import interp1d

        t_old = np.linspace(0, 1, T)
        t_new = np.linspace(0, 1, b3d_joints.shape[0])

        poses_np = poses.cpu().numpy().reshape(T, -1)
        poses_interp = interp1d(t_old, poses_np, axis=0, kind='linear')(t_new)
        poses = torch.from_numpy(poses_interp.reshape(b3d_joints.shape[0], 24, 3)).float().to(device)

        trans_np = trans.cpu().numpy()
        trans_interp = interp1d(t_old, trans_np, axis=0, kind='linear')(t_new)
        trans = torch.from_numpy(trans_interp).float().to(device)

        T = b3d_joints.shape[0]

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
            smpl_joints_list.append(joints[:24].cpu().numpy())

    print(f"  Generated {len(smpl_joints_list)} frames of SMPL joints")

    # 5. Render video
    print("[5/5] Rendering MP4 video...")
    output_video = Path(args.output_dir) / f"worst_case_subject12_proper.mp4"

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

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

    def update(frame):
        ax.clear()

        # Plot B3D joints (red circles)
        b3d = b3d_joints[frame]
        ax.scatter(b3d[:, 0], b3d[:, 1], b3d[:, 2],
                  c='red', marker='o', s=50, alpha=0.6, label='B3D Ground Truth')

        # Plot SMPL joints (blue triangles)
        smpl_j = smpl_joints_list[frame]
        ax.scatter(smpl_j[:, 0], smpl_j[:, 1], smpl_j[:, 2],
                  c='blue', marker='^', s=100, alpha=0.8, label='SMPL Prediction')

        # Connect SMPL joints with skeleton
        for i, j in connections:
            if i < smpl_j.shape[0] and j < smpl_j.shape[0]:
                ax.plot([smpl_j[i, 0], smpl_j[j, 0]],
                       [smpl_j[i, 1], smpl_j[j, 1]],
                       [smpl_j[i, 2], smpl_j[j, 2]],
                       'b-', linewidth=2, alpha=0.5)

        # Set axis limits (fixed across all frames)
        all_points = np.vstack([b3d, smpl_j])
        margin = 0.5
        ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Subject12 - MPJPE: {mpjpe:.2f}mm | Frame: {frame+1}/{T}', fontsize=16)
        ax.legend(loc='upper right')

        # Rotate view
        ax.view_init(elev=10, azim=frame * 360 / T)

        return ax,

    # Create animation
    anim = FuncAnimation(fig, update, frames=T, interval=1000/args.fps, blit=False)

    # Save as MP4
    writer = FFMpegWriter(fps=args.fps, metadata=dict(artist='SMPL Fit Visualization'), bitrate=5000)
    anim.save(str(output_video), writer=writer)

    plt.close(fig)

    print(f"\n✓ Video saved to: {output_video}")
    print(f"  Duration: {T/args.fps:.1f}s")
    print(f"  FPS: {args.fps}")
    print(f"  Frames: {T}")
    print(f"  B3D joints: {b3d_joints.shape[1]}")
    print(f"  SMPL joints: 24")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
