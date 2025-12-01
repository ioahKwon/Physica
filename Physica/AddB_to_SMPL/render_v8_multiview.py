#!/usr/bin/env python3
"""
Render SMPL skeleton from FIX 8 results with multiple views
"""

import numpy as np
import cv2
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel


# SMPL kinematic tree (24 joints)
SMPL_PARENT = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
    18, 19, 20, 21
]

# Joint names for visualization
JOINT_NAMES = [
    'pelvis', 'L_hip', 'R_hip', 'spine1', 'L_knee', 'R_knee',
    'spine2', 'L_ankle', 'R_ankle', 'spine3', 'L_foot', 'R_foot',
    'neck', 'L_collar', 'R_collar', 'head', 'L_shoulder', 'R_shoulder',
    'L_elbow', 'R_elbow', 'L_wrist', 'R_wrist', 'L_hand', 'R_hand'
]


def read_smpl_joints(smpl_params_path, smpl_model_path, max_frames=None):
    """Read SMPL joints only"""
    device = torch.device('cpu')
    smpl_model = SMPLModel(smpl_model_path, device=device)

    data = np.load(smpl_params_path)
    poses = data['poses']
    trans = data['trans']
    betas = data['betas']

    num_frames = len(poses)
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    all_joints = []

    print(f"Reading SMPL joints ({num_frames} frames)...")

    for i in range(num_frames):
        if i % 50 == 0:
            print(f"  Frame {i}/{num_frames}")

        betas_t = torch.from_numpy(betas).float().to(device).unsqueeze(0)

        if len(poses.shape) == 3:
            poses_frame = poses[i].reshape(-1)
        else:
            poses_frame = poses[i]

        poses_t = torch.from_numpy(poses_frame).float().to(device)
        trans_t = torch.from_numpy(trans[i]).float().to(device)

        with torch.no_grad():
            _, joints = smpl_model.forward(betas_t[0], poses_t, trans_t)

        if joints.dim() == 3 and joints.shape[0] == 1:
            joints = joints.squeeze(0)

        all_joints.append(joints.cpu().numpy())

    return np.stack(all_joints, axis=0)


def project_3d_to_2d(points_3d, img_width, img_height, scale=500, center_point=None, view='sagittal'):
    """
    Project 3D points to 2D with different views

    SMPL coordinate system:
    - X: right (+) / left (-)
    - Y: up (+) / down (-)
    - Z: back (+) / front (-)

    Views:
    - 'sagittal': side view (X-Y plane, looking from right side)
    - 'frontal': front view (Z-Y plane, looking from front)
    - 'transverse': top view (X-Z plane, looking from top)
    """
    if view == 'sagittal':
        # Side view: X (horizontal) vs Y (vertical)
        # Positive X = right side of screen
        points_2d = points_3d[:, [0, 1]].copy()
    elif view == 'frontal':
        # Front view: Z (horizontal) vs Y (vertical)
        # Negative Z = right side of screen (person facing us)
        points_2d = points_3d[:, [2, 1]].copy()
        points_2d[:, 0] = -points_2d[:, 0]  # Flip Z for intuitive viewing
    elif view == 'transverse':
        # Top view: X (horizontal) vs Z (vertical)
        points_2d = points_3d[:, [0, 2]].copy()
    else:
        raise ValueError(f"Unknown view: {view}")

    if center_point is not None:
        if view == 'sagittal':
            center_2d = center_point[[0, 1]]
        elif view == 'frontal':
            center_2d = np.array([-center_point[2], center_point[1]])
        elif view == 'transverse':
            center_2d = center_point[[0, 2]]
        points_2d -= center_2d

    points_2d *= scale
    points_2d[:, 0] += img_width // 2
    points_2d[:, 1] += img_height // 2
    points_2d[:, 1] = img_height - points_2d[:, 1]  # Flip Y axis

    return points_2d.astype(int)


def render_skeleton(img, joints_2d, color=(0, 255, 0), thickness=3):
    """Draw skeleton connections"""
    for child_idx, parent_idx in enumerate(SMPL_PARENT):
        if parent_idx < 0:
            continue

        pt_child = tuple(joints_2d[child_idx])
        pt_parent = tuple(joints_2d[parent_idx])

        if (0 <= pt_child[0] < img.shape[1] and 0 <= pt_child[1] < img.shape[0] and
            0 <= pt_parent[0] < img.shape[1] and 0 <= pt_parent[1] < img.shape[0]):
            cv2.line(img, pt_child, pt_parent, color, thickness, cv2.LINE_AA)

    # Draw joints with labels for key spine joints
    key_joints = [0, 3, 6, 9, 12]  # pelvis, spine1, spine2, spine3, neck
    for i, pt in enumerate(joints_2d):
        pt_tuple = tuple(pt)
        if 0 <= pt_tuple[0] < img.shape[1] and 0 <= pt_tuple[1] < img.shape[0]:
            if i in key_joints:
                # Larger circle for key joints
                cv2.circle(img, pt_tuple, 8, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, pt_tuple, 8, (255, 255, 255), 2, cv2.LINE_AA)
                # Add label
                cv2.putText(img, JOINT_NAMES[i], (pt_tuple[0]+12, pt_tuple[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.circle(img, pt_tuple, 6, color, -1, cv2.LINE_AA)
                cv2.circle(img, pt_tuple, 6, (255, 255, 255), 1, cv2.LINE_AA)


def render_frame(joints_3d, img_width, img_height, view='sagittal'):
    """Render a single frame"""
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    pelvis = joints_3d[0]
    joints_2d = project_3d_to_2d(joints_3d, img_width, img_height,
                                  scale=400, center_point=pelvis, view=view)

    render_skeleton(img, joints_2d, color=(0, 200, 0), thickness=4)

    return img


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True, choices=['spine2', 'spine3', 'v9', 'v10'],
                        help='Which version to render (spine2, spine3, v9, or v10)')
    parser.add_argument('--view', required=True, choices=['sagittal', 'frontal', 'transverse'],
                        help='View angle: sagittal (side), frontal (front), transverse (top)')
    args = parser.parse_args()

    if args.version == 'spine2':
        smpl_params_path = '/tmp/v8_spine2_p020_split5/foot_orient_loss_with_arm_carter2023_p020/smpl_params.npz'
        output_path = f'/egr/research-zijunlab/kwonjoon/smpl_skeleton_v8_spine2_{args.view}_p020_split5.mp4'
        title = f'FIX 8 spine2: back → spine2 ({args.view} view)'
        mpjpe = 64.75
    elif args.version == 'spine3':
        smpl_params_path = '/tmp/v8_spine3_p020_split5/foot_orient_loss_with_arm_carter2023_p020/smpl_params.npz'
        output_path = f'/egr/research-zijunlab/kwonjoon/smpl_skeleton_v8_spine3_{args.view}_p020_split5.mp4'
        title = f'FIX 8 spine3: back → spine3 ({args.view} view)'
        mpjpe = 66.18
    elif args.version == 'v9':
        smpl_params_path = '/tmp/v9_foot_orient_p020_split5/v9_foot_orient_with_arm_carter2023_p020/smpl_params.npz'
        output_path = f'/egr/research-zijunlab/kwonjoon/smpl_skeleton_v9_{args.view}_p020_split5.mp4'
        title = f'FIX 9: foot orient loss ({args.view} view)'
        mpjpe = 90.53
    else:  # v10
        smpl_params_path = '/tmp/v10_foot_orient_late_p020_split5/v10_foot_orient_late_with_arm_carter2023_p020/smpl_params.npz'
        output_path = f'/egr/research-zijunlab/kwonjoon/smpl_skeleton_v10_{args.view}_p020_split5.mp4'
        title = f'FIX 10: foot orient loss late-stage ({args.view} view)'
        mpjpe = 83.80

    smpl_model_path = '/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl'

    print("=" * 80)
    print(f"FIX 8 ({args.version}) - P020_split5 Skeleton Video ({args.view} view)")
    print("=" * 80)
    print(f"Input:  {smpl_params_path}")
    print(f"Output: {output_path}")
    print()

    # Read SMPL joints
    print("[1] Reading SMPL joints...")
    joints = read_smpl_joints(smpl_params_path, smpl_model_path, max_frames=200)
    print(f"    Total frames: {len(joints)}")
    print(f"    Joints: {joints.shape[1]}")

    # Setup video writer
    fps = 30
    img_width = 800
    img_height = 800

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (img_width, img_height))

    # Render all frames
    print(f"\n[2] Rendering video ({len(joints)} frames, {fps} FPS)...")
    for i in range(len(joints)):
        if i % 50 == 0:
            print(f"    Frame {i}/{len(joints)}")

        img = render_frame(joints[i], img_width, img_height, view=args.view)

        # Add text
        cv2.putText(img, f'Frame {i}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, title, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
        cv2.putText(img, f'MPJPE: {mpjpe:.2f}mm', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)

        out.write(img)

    out.release()

    print(f"\nDone! Video saved: {output_path}")
    print(f"Length: {len(joints)/fps:.1f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
