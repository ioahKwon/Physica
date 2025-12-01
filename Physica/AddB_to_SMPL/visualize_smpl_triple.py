#!/usr/bin/env python3
"""
Create single MP4 with 3 views side-by-side:
[Target] [Predicted] [Overlay]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import json
from pathlib import Path
import sys

def visualize_subject_triple(result_dir, output_mp4):
    """
    Visualize a single subject's SMPL results in 3 views
    """
    result_path = Path(result_dir)

    # Find the inner directory (with_arm_*)
    inner_dirs = list(result_path.glob("with_arm_*"))
    if not inner_dirs:
        print(f"❌ No with_arm_* directory found in {result_dir}")
        return False

    data_dir = inner_dirs[0]

    # Load data
    print(f"📂 Loading from: {data_dir}")

    pred_joints_file = data_dir / "pred_joints.npy"
    target_joints_file = data_dir / "target_joints.npy"
    meta_file = data_dir / "meta.json"

    if not pred_joints_file.exists() or not target_joints_file.exists():
        print(f"❌ Missing joint files")
        return False

    pred_joints = np.load(pred_joints_file)  # (T, J, 3)
    target_joints = np.load(target_joints_file)  # (T, J, 3)

    with open(meta_file, 'r') as f:
        meta = json.load(f)

    mpjpe = meta.get('metrics', {}).get('MPJPE', 0)
    subject_name = meta.get('run_name', 'unknown')
    num_frames_orig = meta.get('num_frames_original', pred_joints.shape[0])
    num_frames_opt = pred_joints.shape[0]

    print(f"  Subject: {subject_name}")
    print(f"  Frames: {num_frames_opt} (original: {num_frames_orig})")
    print(f"  Joints: {pred_joints.shape[1]}")
    print(f"  MPJPE: {mpjpe:.2f} mm")

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

    # Calculate global axis limits
    all_points = np.vstack([target_joints.reshape(-1, 3), pred_joints.reshape(-1, 3)])
    margin = 0.2
    xlim = [all_points[:, 0].min() - margin, all_points[:, 0].max() + margin]
    ylim = [all_points[:, 1].min() - margin, all_points[:, 1].max() + margin]
    zlim = [all_points[:, 2].min() - margin, all_points[:, 2].max() + margin]

    # Create figure with 3 subplots
    print(f"🎬 Creating triple view animation...")
    fig = plt.figure(figsize=(24, 8))
    ax1 = fig.add_subplot(131, projection='3d')  # Target
    ax2 = fig.add_subplot(132, projection='3d')  # Predicted
    ax3 = fig.add_subplot(133, projection='3d')  # Overlay

    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()

        target = target_joints[frame]
        pred = pred_joints[frame]

        # ========================================
        # Left: Target only (Blue)
        # ========================================
        ax1.scatter(target[:, 0], target[:, 1], target[:, 2],
                   c='blue', s=50, alpha=0.7)
        for connection in smpl_skeleton:
            if connection[0] < len(target) and connection[1] < len(target):
                ax1.plot([target[connection[0], 0], target[connection[1], 0]],
                        [target[connection[0], 1], target[connection[1], 1]],
                        [target[connection[0], 2], target[connection[1], 2]],
                        'b-', alpha=0.5, linewidth=2)

        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_zlim(zlim)
        ax1.set_title(f'Target (AddBiomechanics)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.view_init(elev=20, azim=frame * 2)

        # ========================================
        # Middle: Predicted only (Red)
        # ========================================
        ax2.scatter(pred[:, 0], pred[:, 1], pred[:, 2],
                   c='red', s=50, alpha=0.7)
        for connection in smpl_skeleton:
            if connection[0] < len(pred) and connection[1] < len(pred):
                ax2.plot([pred[connection[0], 0], pred[connection[1], 0]],
                        [pred[connection[0], 1], pred[connection[1], 1]],
                        [pred[connection[0], 2], pred[connection[1], 2]],
                        'r-', alpha=0.5, linewidth=2)

        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_zlim(zlim)
        ax2.set_title(f'Predicted (SMPL)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.view_init(elev=20, azim=frame * 2)

        # ========================================
        # Right: Overlay (Both)
        # ========================================
        # Target (blue, lighter)
        ax3.scatter(target[:, 0], target[:, 1], target[:, 2],
                   c='blue', s=50, alpha=0.4, label='Target')
        for connection in smpl_skeleton:
            if connection[0] < len(target) and connection[1] < len(target):
                ax3.plot([target[connection[0], 0], target[connection[1], 0]],
                        [target[connection[0], 1], target[connection[1], 1]],
                        [target[connection[0], 2], target[connection[1], 2]],
                        'b-', alpha=0.3, linewidth=2)

        # Predicted (red, lighter)
        ax3.scatter(pred[:, 0], pred[:, 1], pred[:, 2],
                   c='red', s=50, alpha=0.4, label='Predicted')
        for connection in smpl_skeleton:
            if connection[0] < len(pred) and connection[1] < len(pred):
                ax3.plot([pred[connection[0], 0], pred[connection[1], 0]],
                        [pred[connection[0], 1], pred[connection[1], 1]],
                        [pred[connection[0], 2], pred[connection[1], 2]],
                        'r-', alpha=0.3, linewidth=2)

        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)
        ax3.set_zlim(zlim)
        ax3.set_title(f'Overlay (Blue=Target, Red=Predicted)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend(loc='upper right')
        ax3.view_init(elev=20, azim=frame * 2)

        # Overall title
        fig.suptitle(f'{subject_name} | MPJPE: {mpjpe:.2f} mm | Frame {frame+1}/{len(pred_joints)} | Frames: {num_frames_orig}→{num_frames_opt}',
                     fontsize=16, fontweight='bold')

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(pred_joints), interval=50)

    # Save as MP4
    print(f"💾 Saving to: {output_mp4}")
    writer = FFMpegWriter(fps=20, bitrate=5000)  # Higher bitrate for better quality
    anim.save(output_mp4, writer=writer)

    plt.close(fig)
    print(f"✅ Saved: {output_mp4}")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_smpl_triple.py <result_dir> [output.mp4]")
        print("Example: python visualize_smpl_triple.py /path/to/Carter2023_P020_split3 /tmp/p020_triple.mp4")
        sys.exit(1)

    result_dir = sys.argv[1]
    output_mp4 = sys.argv[2] if len(sys.argv) > 2 else "/tmp/smpl_viz_triple.mp4"

    success = visualize_subject_triple(result_dir, output_mp4)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
