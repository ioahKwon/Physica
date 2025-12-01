#!/usr/bin/env python3
"""
SMPL pose 프레임별 시각화 - PyRender 사용
"""

import numpy as np
import torch
import trimesh
import pyrender
import cv2
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel


def visualize_smpl_frames(smpl_params_path, smpl_model_path, output_mp4, fps=30):
    """
    SMPL parameters를 프레임별로 시각화

    Args:
        smpl_params_path: smpl_params.npz 파일 경로
        smpl_model_path: SMPL 모델 경로
        output_mp4: 출력 MP4 파일 경로
        fps: 프레임 레이트
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # SMPL 모델 로드
    print(f"\nSMPL 모델 로드 중: {smpl_model_path}")
    smpl_model = SMPLModel(smpl_model_path, device=device)

    # SMPL parameters 로드
    print(f"\nSMPL parameters 로드 중: {smpl_params_path}")
    data = np.load(smpl_params_path)

    betas = data['betas']
    poses = data['poses']
    trans = data['trans']

    T = poses.shape[0]
    print(f"  총 프레임 수: {T}")
    print(f"  Betas shape: {betas.shape}")
    print(f"  Poses shape: {poses.shape}")
    print(f"  Trans shape: {trans.shape}")

    # Betas 확장
    if len(betas.shape) == 1:
        betas_expanded = np.tile(betas, (T, 1))
    else:
        betas_expanded = betas[:T]

    # Poses 형태 확인
    if len(poses.shape) == 3:
        poses_flat = poses.reshape(T, -1)
    else:
        poses_flat = poses[:T]

    # Tensor로 변환
    betas_t = torch.from_numpy(betas_expanded).float().to(device)
    poses_t = torch.from_numpy(poses_flat).float().to(device)
    trans_t = torch.from_numpy(trans[:T]).float().to(device)

    # SMPL 메시 생성
    print("\nSMPL 메시 생성 중...")
    with torch.no_grad():
        vertices, _ = smpl_model.forward(betas=betas_t, poses=poses_t, trans=trans_t)

    vertices = vertices.cpu().numpy()  # (T, 6890, 3)
    faces = smpl_model.faces

    print(f"  Vertices shape: {vertices.shape}")
    print(f"  Faces shape: {faces.shape}")

    # 비디오 렌더링
    print(f"\n프레임별 렌더링 중...")
    render_frames_to_video(vertices, faces, output_mp4, fps, T)

    print(f"\n✓ 비디오 저장 완료: {output_mp4}")


def render_single_frame(vertices, faces, resolution=(1920, 1080)):
    """
    단일 SMPL 메시 프레임 렌더링

    Args:
        vertices: (N, 3) 버텍스 좌표
        faces: (F, 3) face indices
        resolution: (width, height)

    Returns:
        color_image: (height, width, 3) RGB 이미지
    """
    # Scene 생성
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])

    # Trimesh 생성 및 색상 설정
    mesh_trimesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False
    )
    # 파란색 계열
    mesh_trimesh.visual.vertex_colors = [0.7, 0.8, 1.0, 1.0]

    # Pyrender mesh로 변환
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True)
    scene.add(mesh_pyrender)

    # 카메라 설정
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    # 중심과 크기 계산
    center = vertices.mean(axis=0)
    bbox_size = (vertices.max(axis=0) - vertices.min(axis=0)).max()

    # 카메라 위치 (Y-up 좌표계 기준)
    cam_distance = bbox_size * 2.5

    camera_pose = np.array([
        [0.866, 0, 0.5, center[0] + cam_distance * 0.5],
        [0, 1, 0, center[1] + bbox_size * 0.2],
        [-0.5, 0, 0.866, center[2] + cam_distance * 0.866],
        [0, 0, 0, 1]
    ])

    scene.add(camera, pose=camera_pose)

    # 조명 추가
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    # 렌더링
    renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    color_img, _ = renderer.render(scene)
    renderer.delete()

    return color_img


def render_frames_to_video(vertices_all, faces, output_mp4, fps, T):
    """
    모든 프레임을 렌더링하여 비디오로 저장

    Args:
        vertices_all: (T, 6890, 3) 모든 프레임의 버텍스
        faces: (F, 3) face indices
        output_mp4: 출력 비디오 경로
        fps: 프레임 레이트
        T: 총 프레임 수
    """
    width, height = 1920, 1080

    # 비디오 writer 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))

    # 각 프레임 렌더링
    for t in tqdm(range(T), desc="렌더링"):
        # 프레임 렌더링
        img = render_single_frame(vertices_all[t], faces, resolution=(width, height))

        # 텍스트 오버레이 추가
        img_with_text = add_frame_text(img, t, T)

        # BGR로 변환하여 저장
        out.write(cv2.cvtColor(img_with_text, cv2.COLOR_RGB2BGR))

    out.release()


def add_frame_text(img, frame, total_frames):
    """
    이미지에 프레임 정보 텍스트 추가

    Args:
        img: (H, W, 3) 이미지
        frame: 현재 프레임 번호
        total_frames: 총 프레임 수

    Returns:
        텍스트가 추가된 이미지
    """
    img = img.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    color_white = (255, 255, 255)
    color_blue = (255, 200, 150)

    # 제목
    title = f"SMPL Pose Visualization"
    cv2.putText(img, title, (50, 60),
                font, font_scale, color_white, thickness, cv2.LINE_AA)

    # 프레임 정보
    frame_text = f"Frame: {frame + 1} / {total_frames}"
    cv2.putText(img, frame_text, (50, 120),
                font, 0.9, color_blue, 2, cv2.LINE_AA)

    return img


def main():
    parser = argparse.ArgumentParser(description='SMPL pose 프레임별 시각화')
    parser.add_argument('--smpl_params', type=str, required=True,
                       help='smpl_params.npz 파일 경로')
    parser.add_argument('--smpl_model', type=str,
                       default='models/smpl_model.pkl',
                       help='SMPL 모델 경로')
    parser.add_argument('--output', type=str, required=True,
                       help='출력 MP4 파일 경로')
    parser.add_argument('--fps', type=int, default=30,
                       help='프레임 레이트 (기본값: 30)')
    args = parser.parse_args()

    visualize_smpl_frames(args.smpl_params, args.smpl_model, args.output, args.fps)


if __name__ == '__main__':
    main()
