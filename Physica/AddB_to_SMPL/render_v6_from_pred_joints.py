#!/usr/bin/env python3
"""
Render SMPL skeleton from pred_joints.npy (saved SMPL forward pass results)
"""

import numpy as np
import cv2

# SMPL kinematic tree (24 joints)
SMPL_PARENT = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
    18, 19, 20, 21
]


def project_3d_to_2d(points_3d, img_width, img_height, scale=500, center_point=None):
    """
    Simple orthographic projection from 3D to 2D with camera following
    """
    points_2d = points_3d[:, [0, 1]].copy()

    if center_point is not None:
        center_2d = center_point[[0, 1]]
        points_2d -= center_2d

    points_2d *= scale
    points_2d[:, 0] += img_width // 2
    points_2d[:, 1] += img_height // 2
    points_2d[:, 1] = img_height - points_2d[:, 1]

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

    for pt in joints_2d:
        pt_tuple = tuple(pt)
        if 0 <= pt_tuple[0] < img.shape[1] and 0 <= pt_tuple[1] < img.shape[0]:
            cv2.circle(img, pt_tuple, 6, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt_tuple, 6, (255, 255, 255), 1, cv2.LINE_AA)


def render_frame(joints_3d, img_width, img_height):
    """Render a single frame"""
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    pelvis = joints_3d[0]
    joints_2d = project_3d_to_2d(joints_3d, img_width, img_height, scale=400, center_point=pelvis)

    render_skeleton(img, joints_2d, color=(0, 200, 0), thickness=4)

    return img


def main():
    # Load pred_joints.npy (saved SMPL forward pass results)
    pred_joints_path = '/tmp/v6_p020_test/foot_orient_loss_with_arm_carter2023_p020/pred_joints.npy'
    output_path = '/egr/research-zijunlab/kwonjoon/smpl_skeleton_v6_from_pred.mp4'

    print("=" * 80)
    print("FIX 6 - Skeleton from pred_joints.npy")
    print("=" * 80)
    print(f"Input:  {pred_joints_path}")
    print(f"Output: {output_path}")
    print()

    joints = np.load(pred_joints_path)  # [T, 24, 3]
    print(f"[1] Loaded joints: shape={joints.shape}")
    print(f"    Frames: {len(joints)}, Joints: {joints.shape[1]}")

    fps = 30
    img_width = 800
    img_height = 800

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (img_width, img_height))

    print(f"\n[2] Rendering video ({len(joints)} frames, {fps} FPS)...")
    for i in range(len(joints)):
        if i % 50 == 0:
            print(f"    Frame {i}/{len(joints)}")

        img = render_frame(joints[i], img_width, img_height)

        cv2.putText(img, f'Frame {i}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, 'FIX 6: From pred_joints.npy', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
        cv2.putText(img, f'MPJPE: 29.92mm', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)

        out.write(img)

    out.release()

    print(f"\nDone! Video saved: {output_path}")
    print(f"Length: {len(joints)/fps:.1f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
