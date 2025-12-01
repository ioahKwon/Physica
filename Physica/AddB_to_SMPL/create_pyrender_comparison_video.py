#!/usr/bin/env python3
"""
Create high-quality side-by-side SMPL mesh comparison video using Pyrender
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


def create_pyrender_comparison_video(baseline_npz, fixed_npz, smpl_model_path, output_mp4):
    """
    Create side-by-side comparison video with Pyrender rendering

    Args:
        baseline_npz: Path to baseline smpl_params.npz
        fixed_npz: Path to fixed smpl_params.npz
        smpl_model_path: Path to SMPL model
        output_mp4: Output MP4 file path
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load SMPL model
    print(f"\nLoading SMPL model from {smpl_model_path}...")
    smpl_model = SMPLModel(smpl_model_path, device=device)

    # Load baseline results
    print(f"\nLoading baseline from {baseline_npz}...")
    baseline_data = np.load(baseline_npz)
    betas_baseline = baseline_data['betas']
    poses_baseline = baseline_data['poses']
    trans_baseline = baseline_data['trans']
    T_baseline = poses_baseline.shape[0]
    print(f"  Loaded {T_baseline} frames")

    # Load fixed results
    print(f"\nLoading fixed from {fixed_npz}...")
    fixed_data = np.load(fixed_npz)
    betas_fixed = fixed_data['betas']
    poses_fixed = fixed_data['poses']
    trans_fixed = fixed_data['trans']
    T_fixed = poses_fixed.shape[0]
    print(f"  Loaded {T_fixed} frames")

    T = min(T_baseline, T_fixed)
    print(f"\nUsing {T} frames for comparison")

    # Generate SMPL meshes for both
    print("\nGenerating SMPL meshes for baseline...")
    vertices_baseline = generate_smpl_vertices(
        smpl_model, betas_baseline, poses_baseline, trans_baseline, device, T
    )

    print("Generating SMPL meshes for fixed...")
    vertices_fixed = generate_smpl_vertices(
        smpl_model, betas_fixed, poses_fixed, trans_fixed, device, T
    )

    # Get SMPL faces
    faces = smpl_model.faces

    # Compute foot rotation magnitudes for display
    baseline_l_rot = np.linalg.norm(poses_baseline[:T, 10, :], axis=1)
    baseline_r_rot = np.linalg.norm(poses_baseline[:T, 11, :], axis=1)
    fixed_l_rot = np.linalg.norm(poses_fixed[:T, 10, :], axis=1)
    fixed_r_rot = np.linalg.norm(poses_fixed[:T, 11, :], axis=1)

    print(f"\nFoot rotation statistics:")
    print(f"  Baseline - Left:  mean={np.degrees(baseline_l_rot.mean()):.2f}°, max={np.degrees(baseline_l_rot.max()):.2f}°")
    print(f"  Baseline - Right: mean={np.degrees(baseline_r_rot.mean()):.2f}°, max={np.degrees(baseline_r_rot.max()):.2f}°")
    print(f"  Fixed - Left:  mean={np.degrees(fixed_l_rot.mean()):.2f}°, max={np.degrees(fixed_l_rot.max()):.2f}°")
    print(f"  Fixed - Right: mean={np.degrees(fixed_r_rot.mean()):.2f}°, max={np.degrees(fixed_r_rot.max()):.2f}°")

    # Create visualization
    print(f"\nRendering high-quality comparison video with Pyrender...")
    render_comparison_video(
        vertices_baseline, vertices_fixed, faces,
        baseline_l_rot, baseline_r_rot,
        fixed_l_rot, fixed_r_rot,
        output_mp4, T
    )

    print(f"\n✓ Saved video to {output_mp4}")


def generate_smpl_vertices(smpl_model, betas, poses, trans, device, T):
    """Generate SMPL mesh vertices"""
    # Expand betas if needed
    if len(betas.shape) == 1:
        betas_expanded = np.tile(betas, (T, 1))
    else:
        betas_expanded = betas[:T]

    # Reshape poses if needed
    if len(poses.shape) == 3:
        poses_flat = poses[:T].reshape(T, -1)
    else:
        poses_flat = poses[:T]

    betas_t = torch.from_numpy(betas_expanded).float().to(device)
    poses_t = torch.from_numpy(poses_flat).float().to(device)
    trans_t = torch.from_numpy(trans[:T]).float().to(device)

    with torch.no_grad():
        vertices, _ = smpl_model.forward(betas=betas_t, poses=poses_t, trans=trans_t)

    return vertices.cpu().numpy()  # (T, 6890, 3)


def render_single_mesh(vertices, faces, color, resolution=(960, 1080)):
    """
    Render a single SMPL mesh with Pyrender

    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) face indices
        color: RGBA color as list [R, G, B, A]
        resolution: (width, height) tuple

    Returns:
        color_image: (height, width, 3) RGB image
    """
    # Create scene
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])

    # Create trimesh and add color
    mesh_trimesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False
    )
    mesh_trimesh.visual.vertex_colors = color

    # Convert to pyrender mesh
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True)
    scene.add(mesh_pyrender)

    # Set up camera (looking from front-right angle for Y-up coordinate system)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    # Compute center and bounds
    center = vertices.mean(axis=0)
    bbox_size = (vertices.max(axis=0) - vertices.min(axis=0)).max()

    # Camera position: slightly to the right and elevated for Y-up coordinates
    # In Y-up: X=right, Y=up, Z=forward
    cam_distance = bbox_size * 2.5

    camera_pose = np.array([
        [0.866, 0, 0.5, center[0] + cam_distance * 0.5],
        [0, 1, 0, center[1] + bbox_size * 0.2],
        [-0.5, 0, 0.866, center[2] + cam_distance * 0.866],
        [0, 0, 0, 1]
    ])

    scene.add(camera, pose=camera_pose)

    # Add directional lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render
    renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    color_img, _ = renderer.render(scene)
    renderer.delete()

    return color_img


def render_comparison_video(
    vertices_baseline, vertices_fixed, faces,
    baseline_l, baseline_r, fixed_l, fixed_r,
    output_mp4, T
):
    """
    Render side-by-side comparison video

    Args:
        vertices_baseline, vertices_fixed: (T, 6890, 3) vertex arrays
        faces: (F, 3) face indices
        baseline_l, baseline_r, fixed_l, fixed_r: (T,) rotation magnitudes
        output_mp4: output file path
        T: number of frames
    """
    # Video settings
    fps = 30
    width_per_panel = 960
    height = 1080
    total_width = width_per_panel * 2

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (total_width, height))

    # Colors: blue-ish for baseline, green-ish for fixed
    color_baseline = [0.6, 0.6, 1.0, 1.0]  # Light blue
    color_fixed = [0.6, 1.0, 0.6, 1.0]     # Light green

    print(f"Rendering {T} frames...")
    for t in tqdm(range(T)):
        # Render baseline (left panel)
        img_baseline = render_single_mesh(
            vertices_baseline[t], faces, color_baseline,
            resolution=(width_per_panel, height)
        )

        # Render fixed (right panel)
        img_fixed = render_single_mesh(
            vertices_fixed[t], faces, color_fixed,
            resolution=(width_per_panel, height)
        )

        # Combine side-by-side
        combined = np.hstack([img_baseline, img_fixed])

        # Add text overlays
        combined = add_text_overlay(
            combined, t, T,
            baseline_l[t], baseline_r[t],
            fixed_l[t], fixed_r[t],
            width_per_panel
        )

        # Write frame (convert RGB to BGR for OpenCV)
        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    out.release()


def add_text_overlay(img, frame, total_frames, bl, br, fl, fr, width_per_panel):
    """
    Add text overlays to the combined image

    Args:
        img: (H, W, 3) image array
        frame: current frame number
        total_frames: total number of frames
        bl, br: baseline left/right foot rotation (radians)
        fl, fr: fixed left/right foot rotation (radians)
        width_per_panel: width of each panel

    Returns:
        img with text overlays
    """
    img = img.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color_white = (255, 255, 255)
    color_blue = (255, 200, 150)
    color_green = (150, 255, 150)

    # Title
    title = f"SMPL Mesh Comparison - Frame {frame}/{total_frames}"
    cv2.putText(img, title, (int(width_per_panel * 0.7), 40),
                font, 0.9, color_white, 2, cv2.LINE_AA)

    # Left panel (Baseline)
    cv2.putText(img, "Baseline", (50, 80),
                font, font_scale, color_blue, thickness, cv2.LINE_AA)
    cv2.putText(img, f"L foot: {np.degrees(bl):.1f}°", (50, 120),
                font, 0.6, color_white, 1, cv2.LINE_AA)
    cv2.putText(img, f"R foot: {np.degrees(br):.1f}°", (50, 150),
                font, 0.6, color_white, 1, cv2.LINE_AA)

    # Right panel (Fixed)
    cv2.putText(img, "Fixed (Option 4)", (width_per_panel + 50, 80),
                font, font_scale, color_green, thickness, cv2.LINE_AA)
    cv2.putText(img, f"L foot: {np.degrees(fl):.1f}°", (width_per_panel + 50, 120),
                font, 0.6, color_white, 1, cv2.LINE_AA)
    cv2.putText(img, f"R foot: {np.degrees(fr):.1f}°", (width_per_panel + 50, 150),
                font, 0.6, color_white, 1, cv2.LINE_AA)

    return img


def main():
    parser = argparse.ArgumentParser(description='Create Pyrender-based comparison video')
    parser.add_argument('--baseline', type=str, required=True,
                       help='Path to baseline smpl_params.npz')
    parser.add_argument('--fixed', type=str, required=True,
                       help='Path to fixed smpl_params.npz')
    parser.add_argument('--smpl_model', type=str,
                       default='models/smpl_model.pkl',
                       help='Path to SMPL model')
    parser.add_argument('--output', type=str, required=True,
                       help='Output MP4 file path')
    args = parser.parse_args()

    create_pyrender_comparison_video(args.baseline, args.fixed, args.smpl_model, args.output)


if __name__ == '__main__':
    main()
