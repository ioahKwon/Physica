#!/usr/bin/env python3
"""
Render AddBiomechanics GT skeleton from target_joints.npy
"""

import numpy as np
import cv2

# AddBiomechanics joint hierarchy (based on the data we've seen)
# This is approximate - we'll visualize all connections
ADDB_CONNECTIONS = [
    # Pelvis to spine
    (0, 7),  # pelvis to back
    # Pelvis to legs
    (0, 1),  # pelvis to right hip
    (1, 2),  # right hip to right knee
    (2, 3),  # right knee to right ankle
    (3, 4),  # right ankle to right subtalar
    (4, 5),  # right subtalar to right mtp
    (0, 6),  # pelvis to left hip
    (6, 7),  # left hip to left knee (assuming symmetry)
    (7, 8),  # left knee to left ankle (approximate)
    # Arms (approximate - based on typical biomechanics structure)
    (0, 11),  # spine to right shoulder
    (11, 12), # right shoulder to right elbow
    (12, 13), # right elbow to right wrist
    (0, 14),  # spine to left shoulder
    (14, 15), # left shoulder to left elbow
    (15, 16), # left elbow to left wrist
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


def render_skeleton(img, joints_2d, color=(0, 0, 255), thickness=3):
    """Draw skeleton connections"""
    # Draw all joints as dots first
    for i, pt in enumerate(joints_2d):
        pt_tuple = tuple(pt)
        if 0 <= pt_tuple[0] < img.shape[1] and 0 <= pt_tuple[1] < img.shape[0]:
            cv2.circle(img, pt_tuple, 8, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt_tuple, 8, (255, 255, 255), 2, cv2.LINE_AA)
            # Add joint number for debugging
            cv2.putText(img, str(i), (pt_tuple[0]+10, pt_tuple[1]+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Draw connections
    for joint1, joint2 in ADDB_CONNECTIONS:
        if joint1 < len(joints_2d) and joint2 < len(joints_2d):
            pt1 = tuple(joints_2d[joint1])
            pt2 = tuple(joints_2d[joint2])

            if (0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0] and
                0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]):
                cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)


def render_frame(joints_3d, img_width, img_height):
    """Render a single frame"""
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # Camera following pelvis (joint 0)
    pelvis = joints_3d[0]
    joints_2d = project_3d_to_2d(joints_3d, img_width, img_height, scale=400, center_point=pelvis)

    # Render skeleton (red for GT)
    render_skeleton(img, joints_2d, color=(0, 0, 200), thickness=4)

    return img


def main():
    # Load AddBiomechanics GT data - split5
    target_joints_path = '/tmp/v7_p020_split5/foot_orient_loss_with_arm_carter2023_p020/target_joints.npy'
    output_path = '/egr/research-zijunlab/kwonjoon/addb_gt_skeleton_p020_split5.mp4'

    print("=" * 80)
    print("AddBiomechanics GT Skeleton Video - P020_split5")
    print("=" * 80)
    print(f"Input:  {target_joints_path}")
    print(f"Output: {output_path}")
    print()

    # Load target joints
    joints = np.load(target_joints_path)  # [T, N, 3]
    print(f"[1] Loaded GT joints: shape={joints.shape}")
    print(f"    Frames: {len(joints)}, Joints: {joints.shape[1]}")
    print()

    # Check for NaN values
    nan_count = np.isnan(joints).sum()
    if nan_count > 0:
        print(f"    Warning: {nan_count} NaN values found in data")

    # Setup video writer
    fps = 30
    img_width = 800
    img_height = 800

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (img_width, img_height))

    # Render all frames
    print(f"[2] Rendering video ({len(joints)} frames, {fps} FPS)...")
    for i in range(len(joints)):
        if i % 50 == 0:
            print(f"    Frame {i}/{len(joints)}")

        img = render_frame(joints[i], img_width, img_height)

        # Add text
        cv2.putText(img, f'Frame {i}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, 'AddBiomechanics GT', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 150), 2)
        cv2.putText(img, 'P020_split0', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 2)

        out.write(img)

    out.release()

    print(f"\nDone! Video saved: {output_path}")
    print(f"Length: {len(joints)/fps:.1f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
