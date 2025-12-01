#!/usr/bin/env python3
"""
SMPL skeleton (joints only) 시각화
"""

import numpy as np
import torch
import cv2
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel


# SMPL kinematic tree (24 joints)
SMPL_PARENT = [
    -1,  # 0: Pelvis (root)
    0,   # 1: L_Hip
    0,   # 2: R_Hip
    0,   # 3: Spine1
    1,   # 4: L_Knee
    2,   # 5: R_Knee
    3,   # 6: Spine2
    4,   # 7: L_Ankle
    5,   # 8: R_Ankle
    6,   # 9: Spine3
    7,   # 10: L_Foot
    8,   # 11: R_Foot
    9,   # 12: Neck
    9,   # 13: L_Collar
    9,   # 14: R_Collar
    12,  # 15: Head
    13,  # 16: L_Shoulder
    14,  # 17: R_Shoulder
    16,  # 18: L_Elbow
    17,  # 19: R_Elbow
    18,  # 20: L_Wrist
    19,  # 21: R_Wrist
    20,  # 22: L_Hand
    21,  # 23: R_Hand
]


def render_skeleton_frame(smpl_params_path, smpl_model_path, output_jpg, frame_idx=0):
    """
    SMPL skeleton (joints only) 렌더링

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

    # 3D plot으로 skeleton 그리기
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Joint 그리기
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               c='blue', marker='o', s=100, alpha=0.8)

    # Bone 그리기
    for i, parent in enumerate(SMPL_PARENT):
        if parent >= 0:
            # Draw line from parent to child
            xs = [joints[parent, 0], joints[i, 0]]
            ys = [joints[parent, 1], joints[i, 1]]
            zs = [joints[parent, 2], joints[i, 2]]
            ax.plot(xs, ys, zs, 'b-', linewidth=2, alpha=0.7)

    # Joint 번호 표시 (옵션)
    for i, joint in enumerate(joints):
        ax.text(joint[0], joint[1], joint[2], str(i),
                fontsize=8, color='red')

    # Axis 설정
    center = joints.mean(axis=0)
    max_range = np.abs(joints - center).max()

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'SMPL Skeleton - Frame {frame_idx}')

    # Y-up 좌표계 설정
    ax.view_init(elev=10, azim=60)

    # 저장
    plt.tight_layout()
    plt.savefig(output_jpg, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"✓ 저장 완료: {output_jpg}")


def main():
    parser = argparse.ArgumentParser(description='SMPL skeleton 렌더링')
    parser.add_argument('--smpl_params', type=str, required=True)
    parser.add_argument('--smpl_model', type=str, default='models/smpl_model.pkl')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--frame', type=int, default=0, help='렌더링할 프레임 번호')
    args = parser.parse_args()

    render_skeleton_frame(args.smpl_params, args.smpl_model, args.output, args.frame)


if __name__ == '__main__':
    main()
