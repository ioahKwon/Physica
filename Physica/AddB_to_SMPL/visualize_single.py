"""
Simple visualization script for single SMPL fitting result
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse
from pathlib import Path

# Lower body joint indices (SMPL)
LOWER_BODY_JOINTS = [0, 1, 2, 4, 5, 7, 8, 10, 11]  # pelvis, hips, knees, ankles, feet

# Lower body skeleton connections
# SMPL indices: [0, 1, 2, 4, 5, 7, 8, 10, 11]
# Mapped to:    [0, 1, 2, 3, 4, 5, 6, 7,  8 ]
# Names:  pelvis, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, left_foot, right_foot
LOWER_BODY_SKELETON = [
    (0, 1), (0, 2),  # pelvis → hips
    (1, 3), (2, 4),  # hips → knees
    (3, 5), (4, 6),  # knees → ankles
    (5, 7), (6, 8)   # ankles → feet
]

def visualize_skeleton_video(pred_joints_path, target_joints_path, output_path,
                             lower_body_only=False, fps=30, subsample=1):
    """
    Create side-by-side video: Predicted (left) vs Target (right)
    """
    # Load data
    pred_joints = np.load(pred_joints_path)  # [T, 24, 3]
    target_joints = np.load(target_joints_path)  # [T, num_joints, 3]

    print(f"Pred joints shape: {pred_joints.shape}")
    print(f"Target joints shape: {target_joints.shape}")

    # Subsample
    pred_joints = pred_joints[::subsample]
    target_joints = target_joints[::subsample]
    num_frames = len(pred_joints)

    # Filter to lower body if requested
    if lower_body_only:
        joint_indices = LOWER_BODY_JOINTS
        skeleton = LOWER_BODY_SKELETON
    else:
        joint_indices = list(range(24))
        skeleton = []  # Full body skeleton would go here

    # Setup figure
    fig = plt.figure(figsize=(16, 8))
    ax_pred = fig.add_subplot(121, projection='3d')
    ax_target = fig.add_subplot(122, projection='3d')

    def draw_skeleton(ax, joints, skeleton, color, title):
        ax.clear()
        ax.set_title(title, fontsize=16, fontweight='bold')

        # Draw joints
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                  c=color, s=50, alpha=0.8, edgecolors='black', linewidth=0.5)

        # Draw skeleton connections
        for i, j in skeleton:
            if i < len(joints) and j < len(joints):
                xs = [joints[i, 0], joints[j, 0]]
                ys = [joints[i, 1], joints[j, 1]]
                zs = [joints[i, 2], joints[j, 2]]
                ax.plot(xs, ys, zs, color, linewidth=2, alpha=0.6)

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=10, azim=45)

    def compute_bounds(pred_joints, target_joints):
        """Compute scene bounds"""
        # Flip Y for predicted (SMPL Y is inverted)
        pred_flipped = pred_joints.copy()
        pred_flipped[:, :, 1] = -pred_flipped[:, :, 1]

        all_points = np.concatenate([
            pred_flipped.reshape(-1, 3),
            target_joints.reshape(-1, 3)
        ], axis=0)

        # Remove NaN
        valid_mask = ~np.isnan(all_points).any(axis=1)
        all_points = all_points[valid_mask]

        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)

        # Add padding
        padding = (maxs - mins) * 0.1
        mins -= padding
        maxs += padding

        return mins, maxs

    # Compute bounds
    mins, maxs = compute_bounds(pred_joints, target_joints)

    def update(frame_idx):
        # Get joints for this frame
        pred_full = pred_joints[frame_idx]
        target_full = target_joints[frame_idx]

        # Flip Y axis for predicted (SMPL has inverted Y)
        pred_full_copy = pred_full.copy()
        pred_full_copy[:, 1] = -pred_full_copy[:, 1]

        # Filter to specified joints
        pred_filtered = pred_full_copy[joint_indices]

        # For target, only show points (no skeleton)
        target_mapped = []
        for i in range(target_full.shape[0]):
            if not np.isnan(target_full[i]).any():
                target_mapped.append(target_full[i])
        target_mapped = np.array(target_mapped) if target_mapped else np.zeros((0, 3))

        # Draw predicted skeleton (left)
        draw_skeleton(ax_pred, pred_filtered, skeleton, 'b',
                     f'Predicted SMPL (Frame {frame_idx}/{num_frames})')

        # Draw target points (right)
        ax_target.clear()
        ax_target.set_title(f'Target AddBiomechanics (Frame {frame_idx}/{num_frames})',
                           fontsize=16, fontweight='bold')
        if len(target_mapped) > 0:
            ax_target.scatter(target_mapped[:, 0], target_mapped[:, 1], target_mapped[:, 2],
                            c='red', s=100, alpha=0.8, edgecolors='black', linewidth=1, marker='o')

        # Set consistent bounds and view
        for ax in [ax_pred, ax_target]:
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=10, azim=45)

        return ax_pred, ax_target

    # Create animation
    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps, blit=False)

    # Save
    writer = FFMpegWriter(fps=fps, bitrate=5000, codec='libx264')
    print(f"Saving video to {output_path}...")
    anim.save(output_path, writer=writer)
    print("Done!")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_joints', required=True, help='Path to pred_joints.npy')
    parser.add_argument('--target_joints', required=True, help='Path to target_joints.npy')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--lower_body_only', action='store_true')

    args = parser.parse_args()

    visualize_skeleton_video(
        args.pred_joints,
        args.target_joints,
        args.output,
        lower_body_only=args.lower_body_only,
        fps=args.fps,
        subsample=args.subsample
    )
