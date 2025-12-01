#!/usr/bin/env python3
"""
SMPL skeleton을 2D projection으로 시각화 (정면 뷰)
"""

import numpy as np
import torch
import cv2
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel


# SMPL kinematic tree (24 joints) - from SMPL model
SMPL_PARENT = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
    18, 19, 20, 21
]

JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]


def render_skeleton_2d(smpl_params_path, smpl_model_path, output_jpg, frame_idx=0):
    """
    SMPL skeleton을 2D projection으로 렌더링 (정면 뷰)

    Args:
        smpl_params_path: smpl_params.npz 파일 경로
        smpl_model_path: SMPL 모델 경로
        output_jpg: 출력 JPG 파일 경로
        frame_idx: 렌더링할 프레임 번호
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SMPL 모델 로드
    smpl_model = SMPLModel(smpl_model_path, device=device)

    # SMPL parameters 로드
    data = np.load(smpl_params_path)
    betas = data['betas']
    poses = data['poses']
    trans = data['trans']

    T = poses.shape[0]
    print(f"총 {T} 프레임, {frame_idx}번 프레임 렌더링 중...")

    # 지정된 프레임만 추출
    betas_frame = np.zeros((1, 10), dtype=np.float32)

    if len(poses.shape) == 3:
        poses_frame = poses[frame_idx:frame_idx+1].reshape(1, -1)
    else:
        poses_frame = poses[frame_idx:frame_idx+1]

    trans_frame = trans[frame_idx:frame_idx+1]

    # Tensor로 변환
    betas_t = torch.from_numpy(betas_frame).float().to(device)
    poses_t = torch.from_numpy(poses_frame).float().to(device)
    trans_t = torch.from_numpy(trans_frame).float().to(device)

    # SMPL forward pass to get joints
    with torch.no_grad():
        vertices, joints = smpl_model.forward(betas=betas_t, poses=poses_t, trans=trans_t)

    joints = joints.cpu().numpy()[0]  # (24, 3)

    print(f"  Joints shape: {joints.shape}")
    print(f"  Joints range: X=[{joints[:, 0].min():.3f}, {joints[:, 0].max():.3f}], "
          f"Y=[{joints[:, 1].min():.3f}, {joints[:, 1].max():.3f}], "
          f"Z=[{joints[:, 2].min():.3f}, {joints[:, 2].max():.3f}]")

    # 2D projection (정면 뷰: X-Y plane, Y-up)
    # X: 좌우, Y: 상하
    img_width, img_height = 1920, 1080
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # 좌표 정규화 및 이미지 좌표로 변환
    center = joints.mean(axis=0)

    # X-Y plane projection
    joints_2d = joints[:, [0, 1]].copy()  # X, Y
    joints_2d[:, 1] = -joints_2d[:, 1]  # Y축 반전 (이미지 좌표계)

    # 정규화
    range_val = max(
        joints_2d[:, 0].max() - joints_2d[:, 0].min(),
        joints_2d[:, 1].max() - joints_2d[:, 1].min()
    )

    scale = min(img_width, img_height) * 0.8 / range_val
    joints_2d_center = joints_2d.mean(axis=0)
    joints_2d = (joints_2d - joints_2d_center) * scale
    joints_2d[:, 0] += img_width // 2
    joints_2d[:, 1] += img_height // 2

    joints_2d = joints_2d.astype(int)

    # Bone 그리기
    for i, parent in enumerate(SMPL_PARENT):
        if parent >= 0:
            pt1 = tuple(joints_2d[parent])
            pt2 = tuple(joints_2d[i])
            cv2.line(img, pt1, pt2, (100, 100, 255), 3)

    # Joint 그리기
    for i, joint_2d in enumerate(joints_2d):
        cv2.circle(img, tuple(joint_2d), 8, (255, 0, 0), -1)
        # Joint 번호 표시
        cv2.putText(img, str(i), (joint_2d[0] + 10, joint_2d[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 타이틀 추가
    cv2.putText(img, f"SMPL Skeleton - Frame {frame_idx} (Front View)",
               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

    # 저장
    cv2.imwrite(output_jpg, img)
    print(f"✓ 저장 완료: {output_jpg}")


def main():
    parser = argparse.ArgumentParser(description='SMPL skeleton 2D 렌더링')
    parser.add_argument('--smpl_params', type=str, required=True)
    parser.add_argument('--smpl_model', type=str, default='models/smpl_model.pkl')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--frame', type=int, default=0, help='렌더링할 프레임 번호')
    args = parser.parse_args()

    render_skeleton_2d(args.smpl_params, args.smpl_model, args.output, args.frame)


if __name__ == '__main__':
    main()
