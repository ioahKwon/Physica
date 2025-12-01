#!/usr/bin/env python3
"""
Render AddBiomechanics joints as video
"""

import numpy as np
import cv2
import nimblephysics as nimble
from pathlib import Path
import sys


def read_addb_joints(b3d_path, trial_idx=0, max_frames=None):
    """Read AddBiomechanics joint positions"""
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    trial_length = subject.getTrialLength(trial_idx)

    if max_frames is not None:
        trial_length = min(trial_length, max_frames)

    frames = subject.readFrames(
        trial=trial_idx,
        startFrame=0,
        numFramesToRead=trial_length,
        stride=1,
        includeSensorData=False,
        includeProcessingPasses=True
    )

    joint_positions = []
    for frame in frames:
        if len(frame.processingPasses) > 0:
            jc = np.asarray(frame.processingPasses[0].jointCenters, dtype=np.float32)
            joint_positions.append(jc.reshape(-1, 3))

    return np.array(joint_positions)


def render_skeleton_frame(joints, img_width=800, img_height=800):
    """Render single frame of skeleton"""
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # 2D projection (front view: X-Y plane, Y-up)
    joints_2d = joints[:, [0, 1]].copy()  # X, Y
    joints_2d[:, 1] = -joints_2d[:, 1]  # Y축 뒤집기 (이미지 좌표계)

    # Normalize and center
    joints_2d_center = joints_2d.mean(axis=0)
    range_val = max(
        joints_2d[:, 0].max() - joints_2d[:, 0].min(),
        joints_2d[:, 1].max() - joints_2d[:, 1].min()
    )

    if range_val < 1e-6:
        range_val = 1.0

    scale = min(img_width, img_height) * 0.8 / range_val
    joints_2d = (joints_2d - joints_2d_center) * scale
    joints_2d[:, 0] += img_width // 2
    joints_2d[:, 1] += img_height // 2

    # AddBiomechanics 연결 (20 joints)
    # 실제 joint 순서에 맞는 올바른 연결
    # 0:pelvis, 1-5:right_leg, 6-10:left_leg, 11:back, 12-15:right_arm, 16-19:left_arm
    connections = [
        # Right leg chain (pelvis -> hip_r -> knee_r -> ankle_r -> subtalar_r -> mtp_r)
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),

        # Left leg chain (pelvis -> hip_l -> knee_l -> ankle_l -> subtalar_l -> mtp_l)
        (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),

        # Spine (pelvis -> back)
        (0, 11),

        # Right arm chain (back -> acromial_r -> elbow_r -> radioulnar_r -> radius_hand_r)
        (11, 12), (12, 13), (13, 14), (14, 15),

        # Left arm chain (back -> acromial_l -> elbow_l -> radioulnar_l -> radius_hand_l)
        (11, 16), (16, 17), (17, 18), (18, 19),
    ]

    # Draw bones
    for i, j in connections:
        if i < len(joints_2d) and j < len(joints_2d):
            pt1 = tuple(joints_2d[i].astype(int))
            pt2 = tuple(joints_2d[j].astype(int))
            cv2.line(img, pt1, pt2, (100, 100, 255), 3)

    # Draw joints
    for i, pt in enumerate(joints_2d):
        pt_int = tuple(pt.astype(int))
        cv2.circle(img, pt_int, 6, (255, 100, 100), -1)
        cv2.circle(img, pt_int, 6, (0, 0, 0), 1)

    return img


def main():
    b3d_path = '/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Carter2023_Formatted_With_Arm/P020_split2/P020_split2.b3d'
    output_path = '/egr/research-zijunlab/kwonjoon/addb_joints_p020_200frames.mp4'

    print("AddBiomechanics 원본 joint 비디오 생성 중...")
    print(f"입력: {b3d_path}")
    print(f"출력: {output_path}")

    # Read joints
    print("\n[1] AddBiomechanics 데이터 읽는 중...")
    joints = read_addb_joints(b3d_path, trial_idx=0, max_frames=200)
    print(f"    총 프레임: {len(joints)}")
    print(f"    Joint 개수: {joints.shape[1]}")

    # Setup video writer
    fps = 30
    img_width = 800
    img_height = 800

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (img_width, img_height))

    # Render all frames
    print(f"\n[2] 비디오 렌더링 중 ({len(joints)} 프레임, {fps} FPS)...")
    for i, frame_joints in enumerate(joints):
        if i % 50 == 0:
            print(f"    프레임 {i}/{len(joints)}")

        img = render_skeleton_frame(frame_joints, img_width, img_height)

        # Add frame number
        cv2.putText(img, f'Frame {i}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, 'AddBiomechanics Original', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(img)

    out.release()

    print(f"\n완료! 비디오 저장됨: {output_path}")
    print(f"길이: {len(joints)/fps:.1f}초")


if __name__ == '__main__':
    main()
