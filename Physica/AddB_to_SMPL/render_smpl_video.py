#!/usr/bin/env python3
"""
Render SMPL joints as video
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


def read_smpl_joints(smpl_params_path, smpl_model_path, max_frames=None):
    """Read SMPL joint positions"""
    device = torch.device('cpu')
    smpl_model = SMPLModel(smpl_model_path, device=device)

    data = np.load(smpl_params_path)
    poses = data['poses']
    trans = data['trans']

    num_frames = len(poses)
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    all_joints = []

    print(f"SMPL 파라미터에서 joint 추출 중 ({num_frames} 프레임)...")
    for i in range(num_frames):
        if i % 100 == 0:
            print(f"  프레임 {i}/{num_frames}")

        # Zero betas for pose-only
        betas_frame = np.zeros((1, 10), dtype=np.float32)

        if len(poses.shape) == 3:
            poses_frame = poses[i:i+1].reshape(1, -1)
        else:
            poses_frame = poses[i:i+1]

        trans_frame = trans[i:i+1]

        # Convert to tensors
        betas_t = torch.from_numpy(betas_frame).float().to(device)
        poses_t = torch.from_numpy(poses_frame).float().to(device)
        trans_t = torch.from_numpy(trans_frame).float().to(device)

        # Get joints
        with torch.no_grad():
            vertices, joints = smpl_model.forward(betas=betas_t, poses=poses_t, trans=trans_t)

        all_joints.append(joints.cpu().numpy()[0])

    return np.array(all_joints)


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

    # Draw bones using kinematic tree
    for i, parent in enumerate(SMPL_PARENT):
        if parent >= 0:
            pt1 = tuple(joints_2d[parent].astype(int))
            pt2 = tuple(joints_2d[i].astype(int))
            cv2.line(img, pt1, pt2, (100, 100, 255), 3)

    # Draw joints
    for i, pt in enumerate(joints_2d):
        pt_int = tuple(pt.astype(int))
        cv2.circle(img, pt_int, 6, (255, 100, 100), -1)
        cv2.circle(img, pt_int, 6, (0, 0, 0), 1)

    return img


def main():
    smpl_params_path = '/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames/Carter2023_Formatted_With_Arm_P020_split2/with_arm_carter2023_p020/smpl_params.npz'
    smpl_model_path = '/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl'
    output_path = '/egr/research-zijunlab/kwonjoon/smpl_joints_p020_full.mp4'

    print("SMPL 최적화 결과 joint 비디오 생성 중...")
    print(f"입력: {smpl_params_path}")
    print(f"출력: {output_path}")

    # Read joints
    print("\n[1] SMPL 파라미터에서 joint 읽는 중...")
    joints = read_smpl_joints(smpl_params_path, smpl_model_path, max_frames=500)
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
        cv2.putText(img, 'SMPL Optimized (OLD - BUGGY)', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(img)

    out.release()

    print(f"\n완료! 비디오 저장됨: {output_path}")
    print(f"길이: {len(joints)/fps:.1f}초")


if __name__ == '__main__':
    main()
