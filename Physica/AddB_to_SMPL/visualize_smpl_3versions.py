#!/usr/bin/env python3
"""
Create 3 versions of SMPL visualization:
1. Target only
2. Predicted only
3. Overlay (both together)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import json
from pathlib import Path
import sys

def visualize_subject_3versions(result_dir, output_prefix):
    """
    Visualize a single subject's SMPL results in 3 versions
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

    print(f"  Frames: {pred_joints.shape[0]}")
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

    # ========================================
    # Version 1: Target only
    # ========================================
    print(f"🎬 Creating Version 1: Target only...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update_target(frame):
        ax.clear()
        target = target_joints[frame]

        ax.scatter(target[:, 0], target[:, 1], target[:, 2],
                   c='blue', s=50, alpha=0.7, label='Target (AddB)')

        for connection in smpl_skeleton:
            if connection[0] < len(target) and connection[1] < len(target):
                ax.plot([target[connection[0], 0], target[connection[1], 0]],
                        [target[connection[0], 1], target[connection[1], 1]],
                        [target[connection[0], 2], target[connection[1], 2]],
                        'b-', alpha=0.5, linewidth=2)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_title(f'Target (AddBiomechanics)\nFrame {frame+1}/{len(pred_joints)}', fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=frame * 2)

    anim1 = FuncAnimation(fig, update_target, frames=len(pred_joints), interval=50)
    output1 = f"{output_prefix}_target.mp4"
    writer = FFMpegWriter(fps=20, bitrate=2000)
    anim1.save(output1, writer=writer)
    plt.close(fig)
    print(f"✅ Saved: {output1}")

    # ========================================
    # Version 2: Predicted only
    # ========================================
    print(f"🎬 Creating Version 2: Predicted only...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update_pred(frame):
        ax.clear()
        pred = pred_joints[frame]

        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2],
                   c='red', s=50, alpha=0.7, label='Predicted (SMPL)')

        for connection in smpl_skeleton:
            if connection[0] < len(pred) and connection[1] < len(pred):
                ax.plot([pred[connection[0], 0], pred[connection[1], 0]],
                        [pred[connection[0], 1], pred[connection[1], 1]],
                        [pred[connection[0], 2], pred[connection[1], 2]],
                        'r-', alpha=0.5, linewidth=2)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_title(f'Predicted (SMPL) - MPJPE: {mpjpe:.2f} mm\nFrame {frame+1}/{len(pred_joints)}', fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=frame * 2)

    anim2 = FuncAnimation(fig, update_pred, frames=len(pred_joints), interval=50)
    output2 = f"{output_prefix}_predicted.mp4"
    writer = FFMpegWriter(fps=20, bitrate=2000)
    anim2.save(output2, writer=writer)
    plt.close(fig)
    print(f"✅ Saved: {output2}")

    # ========================================
    # Version 3: Overlay (both together)
    # ========================================
    print(f"🎬 Creating Version 3: Overlay...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update_overlay(frame):
        ax.clear()
        target = target_joints[frame]
        pred = pred_joints[frame]

        # Target (blue)
        ax.scatter(target[:, 0], target[:, 1], target[:, 2],
                   c='blue', s=50, alpha=0.5, label='Target')
        for connection in smpl_skeleton:
            if connection[0] < len(target) and connection[1] < len(target):
                ax.plot([target[connection[0], 0], target[connection[1], 0]],
                        [target[connection[0], 1], target[connection[1], 1]],
                        [target[connection[0], 2], target[connection[1], 2]],
                        'b-', alpha=0.3, linewidth=2)

        # Predicted (red)
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2],
                   c='red', s=50, alpha=0.5, label='Predicted')
        for connection in smpl_skeleton:
            if connection[0] < len(pred) and connection[1] < len(pred):
                ax.plot([pred[connection[0], 0], pred[connection[1], 0]],
                        [pred[connection[0], 1], pred[connection[1], 1]],
                        [pred[connection[0], 2], pred[connection[1], 2]],
                        'r-', alpha=0.3, linewidth=2)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_title(f'Overlay: Target (Blue) vs Predicted (Red)\nMPJPE: {mpjpe:.2f} mm | Frame {frame+1}/{len(pred_joints)}', fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.view_init(elev=20, azim=frame * 2)

    anim3 = FuncAnimation(fig, update_overlay, frames=len(pred_joints), interval=50)
    output3 = f"{output_prefix}_overlay.mp4"
    writer = FFMpegWriter(fps=20, bitrate=2000)
    anim3.save(output3, writer=writer)
    plt.close(fig)
    print(f"✅ Saved: {output3}")

    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_smpl_3versions.py <result_dir> [output_prefix]")
        print("Example: python visualize_smpl_3versions.py /path/to/Carter2023_P020_split3 /tmp/p020")
        sys.exit(1)

    result_dir = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "/tmp/smpl_viz"

    success = visualize_subject_3versions(result_dir, output_prefix)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
