#!/usr/bin/env python3
"""
Simple SMPL visualization for completed results
Creates MP4 animation showing predicted vs target joints
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import json
from pathlib import Path
import sys

def visualize_subject(result_dir, output_mp4):
    """
    Visualize a single subject's SMPL results
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

    # Create figure
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # SMPL skeleton connections (simplified)
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

    def update(frame):
        ax1.clear()
        ax2.clear()

        # Target joints (AddBiomechanics)
        target = target_joints[frame]
        ax1.scatter(target[:, 0], target[:, 1], target[:, 2],
                   c='blue', s=50, alpha=0.6, label='Target (AddB)')
        for connection in smpl_skeleton:
            if connection[0] < len(target) and connection[1] < len(target):
                ax1.plot([target[connection[0], 0], target[connection[1], 0]],
                        [target[connection[0], 1], target[connection[1], 1]],
                        [target[connection[0], 2], target[connection[1], 2]],
                        'b-', alpha=0.3)

        # Predicted joints (SMPL)
        pred = pred_joints[frame]
        ax2.scatter(pred[:, 0], pred[:, 1], pred[:, 2],
                   c='red', s=50, alpha=0.6, label='Predicted (SMPL)')
        for connection in smpl_skeleton:
            if connection[0] < len(pred) and connection[1] < len(pred):
                ax2.plot([pred[connection[0], 0], pred[connection[1], 0]],
                        [pred[connection[0], 1], pred[connection[1], 1]],
                        [pred[connection[0], 2], pred[connection[1], 2]],
                        'r-', alpha=0.3)

        # Set same view limits for both
        all_points = np.vstack([target, pred])
        margin = 0.2
        ax1.set_xlim([all_points[:, 0].min() - margin, all_points[:, 0].max() + margin])
        ax1.set_ylim([all_points[:, 1].min() - margin, all_points[:, 1].max() + margin])
        ax1.set_zlim([all_points[:, 2].min() - margin, all_points[:, 2].max() + margin])

        ax2.set_xlim([all_points[:, 0].min() - margin, all_points[:, 0].max() + margin])
        ax2.set_ylim([all_points[:, 1].min() - margin, all_points[:, 1].max() + margin])
        ax2.set_zlim([all_points[:, 2].min() - margin, all_points[:, 2].max() + margin])

        ax1.set_title(f'Target (AddBiomechanics)\nFrame {frame+1}/{len(pred_joints)}', fontsize=12)
        ax2.set_title(f'Predicted (SMPL)\nMPJPE: {mpjpe:.2f} mm', fontsize=12)

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        # Set view angle
        ax1.view_init(elev=20, azim=frame * 2)  # Rotate
        ax2.view_init(elev=20, azim=frame * 2)

    # Create animation
    print(f"🎬 Creating animation...")
    anim = FuncAnimation(fig, update, frames=len(pred_joints), interval=50)

    # Save as MP4
    print(f"💾 Saving to: {output_mp4}")
    writer = FFMpegWriter(fps=20, bitrate=2000)
    anim.save(output_mp4, writer=writer)

    plt.close(fig)
    print(f"✅ Saved: {output_mp4}")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_smpl_simple.py <result_dir> [output.mp4]")
        print("Example: python visualize_smpl_simple.py /path/to/Carter2023_Formatted_With_Arm_P002_split0")
        sys.exit(1)

    result_dir = sys.argv[1]
    output_mp4 = sys.argv[2] if len(sys.argv) > 2 else "/tmp/smpl_viz.mp4"

    success = visualize_subject(result_dir, output_mp4)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
