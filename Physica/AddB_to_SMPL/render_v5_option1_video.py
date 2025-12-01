#!/usr/bin/env python3
"""
Render SMPL mesh + skeleton video from FIX 5 Option 1 results
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


def read_smpl_data(smpl_params_path, smpl_model_path, max_frames=None):
    """Read SMPL vertices and joints"""
    device = torch.device('cpu')
    smpl_model = SMPLModel(smpl_model_path, device=device)

    data = np.load(smpl_params_path)
    poses = data['poses']
    trans = data['trans']
    betas = data['betas']

    num_frames = len(poses)
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    all_verts = []
    all_joints = []

    print(f"SMPL 파라미터에서 vertices와 joints 추출 중 ({num_frames} 프레임)...")

    for i in range(num_frames):
        if i % 50 == 0:
            print(f"  프레임 {i}/{num_frames}")

        # Use optimized betas
        betas_t = torch.from_numpy(betas).float().to(device).unsqueeze(0)

        if len(poses.shape) == 3:
            poses_frame = poses[i].reshape(-1)
        else:
            poses_frame = poses[i]

        poses_t = torch.from_numpy(poses_frame).float().to(device)
        trans_t = torch.from_numpy(trans[i]).float().to(device)

        # Get vertices and joints
        with torch.no_grad():
            verts, joints = smpl_model.forward(betas_t[0], poses_t, trans_t)

        # Squeeze batch dimension if present
        if verts.dim() == 3 and verts.shape[0] == 1:
            verts = verts.squeeze(0)
        if joints.dim() == 3 and joints.shape[0] == 1:
            joints = joints.squeeze(0)

        all_verts.append(verts.cpu().numpy())
        all_joints.append(joints.cpu().numpy())

    return np.stack(all_verts, axis=0), np.stack(all_joints, axis=0)


def project_3d_to_2d(points_3d, img_width, img_height, scale=500):
    """
    Simple orthographic projection from 3D to 2D
    """
    # Orthographic projection (ignore Z for now, just use X and Y)
    points_2d = points_3d[:, [0, 1]].copy()

    # Scale and center
    points_2d *= scale
    points_2d[:, 0] += img_width // 2
    points_2d[:, 1] += img_height // 2

    # Flip Y axis (image coordinates go down)
    points_2d[:, 1] = img_height - points_2d[:, 1]

    return points_2d.astype(int)


def render_mesh_wireframe(img, verts_2d, faces, depth_order, color=(100, 150, 255), thickness=1):
    """
    Render mesh as wireframe with depth ordering
    """
    # Sort faces by depth (back to front)
    sorted_faces = faces[np.argsort(-depth_order)]

    for face in sorted_faces:
        pts = verts_2d[face]

        # Draw triangle edges
        for i in range(3):
            pt1 = tuple(pts[i])
            pt2 = tuple(pts[(i + 1) % 3])

            # Check if points are within image bounds
            if (0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0] and
                0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]):
                cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)


def render_skeleton(img, joints_2d, color=(0, 255, 0), thickness=2):
    """Draw skeleton connections"""
    for child_idx, parent_idx in enumerate(SMPL_PARENT):
        if parent_idx < 0:
            continue

        pt_child = tuple(joints_2d[child_idx])
        pt_parent = tuple(joints_2d[parent_idx])

        # Check bounds
        if (0 <= pt_child[0] < img.shape[1] and 0 <= pt_child[1] < img.shape[0] and
            0 <= pt_parent[0] < img.shape[1] and 0 <= pt_parent[1] < img.shape[0]):
            cv2.line(img, pt_child, pt_parent, color, thickness, cv2.LINE_AA)

    # Draw joints
    for pt in joints_2d:
        pt_tuple = tuple(pt)
        if 0 <= pt_tuple[0] < img.shape[1] and 0 <= pt_tuple[1] < img.shape[0]:
            cv2.circle(img, pt_tuple, 4, color, -1, cv2.LINE_AA)


def render_frame(verts_3d, joints_3d, faces, img_width, img_height):
    """
    Render a single frame with mesh wireframe + skeleton
    """
    # Create white background
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # Project vertices and joints to 2D
    verts_2d = project_3d_to_2d(verts_3d, img_width, img_height, scale=400)
    joints_2d = project_3d_to_2d(joints_3d, img_width, img_height, scale=400)

    # Compute face depths (average Z of vertices)
    face_depths = verts_3d[faces].mean(axis=1)[:, 2]

    # Render mesh wireframe (light blue)
    render_mesh_wireframe(img, verts_2d, faces, face_depths,
                          color=(200, 180, 150), thickness=1)

    # Render skeleton on top (green)
    render_skeleton(img, joints_2d, color=(0, 200, 0), thickness=3)

    return img


def main():
    smpl_params_path = '/tmp/v5_comparison/v5_option1_pelvis_constraint/foot_orient_loss_with_arm_carter2023_p020/smpl_params.npz'
    smpl_model_path = '/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl'
    output_path = '/egr/research-zijunlab/kwonjoon/smpl_mesh_v5_option1_p020.mp4'

    print("=" * 80)
    print("FIX 5 Option 1 - SMPL Mesh + Skeleton Video")
    print("=" * 80)
    print(f"입력: {smpl_params_path}")
    print(f"출력: {output_path}")
    print()

    # Read SMPL data
    print("[1] SMPL 데이터 읽는 중...")
    verts, joints = read_smpl_data(smpl_params_path, smpl_model_path, max_frames=200)
    print(f"    총 프레임: {len(verts)}")
    print(f"    Vertex 개수: {verts.shape[1]}")
    print(f"    Joint 개수: {joints.shape[1]}")

    # Get SMPL faces
    device = torch.device('cpu')
    smpl_model = SMPLModel(smpl_model_path, device=device)
    faces = smpl_model.faces.cpu().numpy()
    print(f"    Face 개수: {len(faces)}")

    # Setup video writer
    fps = 30
    img_width = 800
    img_height = 800

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (img_width, img_height))

    # Render all frames
    print(f"\n[2] 비디오 렌더링 중 ({len(verts)} 프레임, {fps} FPS)...")
    for i in range(len(verts)):
        if i % 50 == 0:
            print(f"    프레임 {i}/{len(verts)}")

        img = render_frame(verts[i], joints[i], faces, img_width, img_height)

        # Add text
        cv2.putText(img, f'Frame {i}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, 'FIX 5 Option 1: Pelvis Upright', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
        cv2.putText(img, f'MPJPE: 29.63mm', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
        cv2.putText(img, 'Mesh + Skeleton', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        out.write(img)

    out.release()

    print(f"\n완료! 비디오 저장됨: {output_path}")
    print(f"길이: {len(verts)/fps:.1f}초")
    print("=" * 80)


if __name__ == '__main__':
    main()
