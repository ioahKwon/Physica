#!/usr/bin/env python3
"""
Visualize worst case fitting result (Tiziana2019 Subject12, 180mm MPJPE)
Render SMPL mesh with B3D markers to show fitting failure
"""

import torch
import numpy as np
import nimblephysics as nimble
from pathlib import Path
import argparse
import json
import sys
import trimesh
import pyrender
import cv2
from tqdm import tqdm

# Import SMPL model
sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPL

def render_frame(smpl_mesh, markers, camera_pose, resolution=(1920, 1080)):
    """
    Render a single frame with SMPL mesh and markers
    """
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

    # Add SMPL mesh (blue-ish)
    mesh_trimesh = trimesh.Trimesh(
        vertices=smpl_mesh.vertices,
        faces=smpl_mesh.faces,
        vertex_colors=[0.7, 0.7, 0.9, 1.0]
    )
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True)
    scene.add(mesh_pyrender)

    # Add markers as red spheres
    marker_radius = 0.02
    for marker_pos in markers:
        marker_mesh = trimesh.creation.icosphere(subdivisions=2, radius=marker_radius)
        marker_mesh.vertices += marker_pos
        marker_mesh.visual.vertex_colors = [1.0, 0.0, 0.0, 1.0]  # Red
        scene.add(pyrender.Mesh.from_trimesh(marker_mesh, smooth=False))

    # Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=camera_pose)

    # Lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render
    renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    color, depth = renderer.render(scene)
    renderer.delete()

    return color

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b3d', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject12/Subject12.b3d')
    parser.add_argument('--result_dir', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames/Tiziana2019_Formatted_With_Arm_Subject12')
    parser.add_argument('--smpl_model', type=str, default='models/smpl_model.pkl')
    parser.add_argument('--output_video', type=str, default='/tmp/worst_case_subject12.mp4')
    parser.add_argument('--max_frames', type=int, default=200, help='Max frames to render')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--resolution', type=str, default='1920x1080')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("WORST CASE VISUALIZATION - Tiziana2019 Subject12 (180mm MPJPE)")
    print("="*80)
    print(f"B3D: {args.b3d}")
    print(f"Result: {args.result_dir}")
    print(f"Output: {args.output_video}")
    print("="*80 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load SMPL model
    print("[1/5] Loading SMPL model...")
    smpl = SMPL(args.smpl_model, device=device)

    # 2. Load SMPL parameters
    print("[2/5] Loading SMPL fit results...")
    result_path = Path(args.result_dir)
    inner_dirs = list(result_path.glob("with_arm_*"))
    if not inner_dirs:
        print(f"ERROR: No with_arm_* directory found in {result_path}")
        sys.exit(1)

    npz_file = inner_dirs[0] / "smpl_params.npz"
    if not npz_file.exists():
        print(f"ERROR: {npz_file} does not exist")
        sys.exit(1)

    data = np.load(npz_file)
    betas = torch.from_numpy(data['betas']).float().to(device)
    poses = torch.from_numpy(data['poses']).float().to(device)  # (T, 24, 3)
    trans = torch.from_numpy(data['trans']).float().to(device)  # (T, 3)

    T = poses.shape[0]
    print(f"  Loaded {T} frames")

    # Load meta to confirm MPJPE
    meta_file = inner_dirs[0] / "meta.json"
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    mpjpe = meta.get('metrics', {}).get('MPJPE', 0)
    print(f"  MPJPE: {mpjpe:.2f}mm")

    # 3. Load B3D markers
    print("[3/5] Loading B3D markers...")
    subject = nimble.biomechanics.SubjectOnDisk(args.b3d)
    num_frames_b3d = subject.getTrials()[0].getNumFrames()
    print(f"  B3D has {num_frames_b3d} frames")

    # Sample frames to match SMPL
    if num_frames_b3d > args.max_frames:
        indices = np.linspace(0, num_frames_b3d-1, args.max_frames, dtype=int)
    else:
        indices = np.arange(num_frames_b3d)

    # Load all marker data
    markers_list = []
    trial = subject.getTrials()[0]
    for idx in indices:
        frame = trial.readFrames(int(idx), 1)[0]
        marker_obs = frame.markerObservations
        markers = np.array([pos for pos in marker_obs.values()])
        markers_list.append(markers)

    print(f"  Loaded {len(markers_list)} frames of markers")

    # Interpolate SMPL params if needed
    if T != len(indices):
        print(f"  Interpolating SMPL params from {T} to {len(indices)} frames...")
        from scipy.interpolate import interp1d

        t_old = np.linspace(0, 1, T)
        t_new = np.linspace(0, 1, len(indices))

        poses_np = poses.cpu().numpy().reshape(T, -1)
        poses_interp = interp1d(t_old, poses_np, axis=0, kind='linear')(t_new)
        poses = torch.from_numpy(poses_interp.reshape(len(indices), 24, 3)).float().to(device)

        trans_np = trans.cpu().numpy()
        trans_interp = interp1d(t_old, trans_np, axis=0, kind='linear')(t_new)
        trans = torch.from_numpy(trans_interp).float().to(device)

        T = len(indices)

    # 4. Generate SMPL meshes
    print("[4/5] Generating SMPL meshes...")
    smpl_meshes = []

    with torch.no_grad():
        for i in tqdm(range(T), desc="Generating meshes"):
            output = smpl(
                betas=betas.unsqueeze(0),
                body_pose=poses[i:i+1, 1:],
                global_orient=poses[i:i+1, 0:1],
                transl=trans[i:i+1]
            )
            vertices = output.vertices[0].cpu().numpy()
            faces = smpl.faces

            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            smpl_meshes.append(mesh)

    # 5. Render video
    print(f"[5/5] Rendering video ({args.output_video})...")

    # Set up camera pose (view from front)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.5],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Parse resolution
    w, h = map(int, args.resolution.split('x'))

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, args.fps, (w, h))

    for i in tqdm(range(T), desc="Rendering frames"):
        frame_img = render_frame(smpl_meshes[i], markers_list[i], camera_pose, resolution=(w, h))

        # Add text overlay
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
        cv2.putText(frame_img, f"Subject12 - MPJPE: {mpjpe:.2f}mm", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame_img, f"Frame: {i+1}/{T}", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame_img, "Blue: SMPL Mesh | Red: B3D Markers", (50, h-50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        out.write(frame_img)

    out.release()

    print(f"\n✓ Video saved to: {args.output_video}")
    print(f"  Duration: {T/args.fps:.1f}s")
    print(f"  Resolution: {w}×{h}")
    print(f"  FPS: {args.fps}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
