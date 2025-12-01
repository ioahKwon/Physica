#!/usr/bin/env python3
"""
단일 SMPL 프레임을 JPG 이미지로 렌더링
"""

import numpy as np
import torch
import trimesh
import pyrender
import cv2
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel


def render_frame_to_image(smpl_params_path, smpl_model_path, output_jpg, frame_idx=0):
    """
    단일 SMPL 프레임을 이미지로 렌더링

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
    # Shape을 zero로 설정 (표준 체형)
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

    # SMPL 메시 생성
    with torch.no_grad():
        vertices, _ = smpl_model.forward(betas=betas_t, poses=poses_t, trans=trans_t)

    vertices = vertices.cpu().numpy()[0]  # (6890, 3)
    faces = smpl_model.faces

    # 렌더링
    img = render_mesh(vertices, faces)

    # JPG로 저장
    cv2.imwrite(output_jpg, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"✓ 저장 완료: {output_jpg}")


def render_mesh(vertices, faces, resolution=(1920, 1080)):
    """단일 메시 렌더링"""
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])

    mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh_trimesh.visual.vertex_colors = [0.7, 0.8, 1.0, 1.0]

    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True)
    scene.add(mesh_pyrender)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    center = vertices.mean(axis=0)
    bbox_size = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    cam_distance = bbox_size * 2.5

    camera_pose = np.array([
        [0.866, 0, 0.5, center[0] + cam_distance * 0.5],
        [0, 1, 0, center[1] + bbox_size * 0.2],
        [-0.5, 0, 0.866, center[2] + cam_distance * 0.866],
        [0, 0, 0, 1]
    ])

    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    color_img, _ = renderer.render(scene)
    renderer.delete()

    return color_img


def main():
    parser = argparse.ArgumentParser(description='단일 SMPL 프레임 렌더링')
    parser.add_argument('--smpl_params', type=str, required=True)
    parser.add_argument('--smpl_model', type=str, default='models/smpl_model.pkl')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--frame', type=int, default=0, help='렌더링할 프레임 번호')
    args = parser.parse_args()

    render_frame_to_image(args.smpl_params, args.smpl_model, args.output, args.frame)


if __name__ == '__main__':
    main()
