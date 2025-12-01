#!/usr/bin/env python3
"""
3D mesh visualization for SMPL fitting results (Version 2 - Optimized)
Renders SMPL body mesh as wireframe with AddB target joints
Uses simpler rendering for better performance and compatibility
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import json
import argparse
from pathlib import Path
import pickle


def load_smpl_model(model_path):
    """Load SMPL model from pickle file"""
    with open(model_path, 'rb') as f:
        smpl_data = pickle.load(f, encoding='latin1')
    return smpl_data


def smpl_forward(smpl_data, beta, pose, trans):
    """
    Simple SMPL forward pass (numpy implementation)
    Only applies shape and translation (simplified - no pose)

    Args:
        smpl_data: SMPL model data
        beta: Shape parameters (10,)
        pose: Pose parameters (72,) - NOT USED in this simplified version
        trans: Translation (3,)

    Returns:
        vertices: (6890, 3)
    """
    v_template = smpl_data['v_template']
    shapedirs = smpl_data['shapedirs']
    faces = smpl_data['f']

    # Apply shape blendshape
    num_betas = min(beta.shape[0], shapedirs.shape[2])
    shapedirs_reshaped = shapedirs[:, :, :num_betas].reshape(-1, num_betas)
    shape_offsets = np.dot(shapedirs_reshaped, beta[:num_betas])
    v_shaped = v_template + shape_offsets.reshape(-1, 3)

    # Apply translation
    vertices = v_shaped + trans

    return vertices, faces


def get_body_edges_from_joints(joint_positions):
    """
    Create body edges from joint positions for wireframe-like body visualization

    Args:
        joint_positions: (24, 3) SMPL joint positions

    Returns:
        edges: list of (start, end) joint index pairs
    """
    # SMPL skeleton connections (same as before)
    edges = [
        # Lower body
        (0, 1), (0, 2),  # pelvis to hips
        (1, 4), (2, 5),  # hips to knees
        (4, 7), (5, 8),  # knees to ankles
        (7, 10), (8, 11),  # ankles to feet
        # Upper body
        (0, 3),  # pelvis to spine
        (3, 6), (6, 9),  # spine chain
        (9, 12), (9, 13), (9, 14),  # neck/shoulders
        (13, 16), (14, 17),  # shoulders to elbows
        (16, 18), (17, 19),  # elbows to wrists
        (18, 20), (19, 21),  # wrists to hands
        (20, 22), (21, 23),  # hands
    ]
    return edges


def get_mesh_edges(vertices, faces, sample_rate=10):
    """
    Extract edges from mesh faces for wireframe rendering
    Sample edges to reduce complexity

    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) face indices
        sample_rate: Use every Nth face

    Returns:
        edges: list of edge coordinates for plotting
    """
    faces_sampled = faces[::sample_rate]
    edges = []

    for face in faces_sampled:
        # Each triangle has 3 edges
        for i in range(3):
            v1 = vertices[face[i]]
            v2 = vertices[face[(i+1) % 3]]
            edges.append((v1, v2))

    return edges


def load_results(result_dir):
    """Load prediction, target, and SMPL parameters"""
    result_dir = Path(result_dir)

    pred_joints = np.load(result_dir / 'pred_joints.npy')
    target_joints = np.load(result_dir / 'target_joints.npy')

    smpl_params = np.load(result_dir / 'smpl_params.npz')
    betas = smpl_params['betas']
    poses = smpl_params['poses']
    trans = smpl_params['trans']

    with open(result_dir / 'meta.json', 'r') as f:
        meta = json.load(f)

    return pred_joints, target_joints, betas, poses, trans, meta


def detect_body_type(meta):
    """Detect if upper body was optimized"""
    joint_mapping = meta.get('joint_mapping', {})
    if not joint_mapping:
        return "no_arm"

    mapped_smpl_indices = set(int(v) for v in joint_mapping.values())
    upper_body_indices = {3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
    has_upper_body_optimized = bool(mapped_smpl_indices & upper_body_indices)

    return "with_arm" if has_upper_body_optimized else "no_arm"


def create_mesh_animation(result_dir, smpl_model_path, output_path=None,
                          fps=30, rotation_degrees=180, wireframe_density=20):
    """
    Create 3D mesh animation with SMPL body shape (wireframe version)

    Args:
        result_dir: Path to result directory
        smpl_model_path: Path to SMPL model pickle file
        output_path: Output MP4 path
        fps: Frames per second
        rotation_degrees: Total rotation in degrees
        wireframe_density: Sample every Nth face for wireframe (higher = less dense)
    """
    # Load data
    print("Loading results...")
    pred_joints, target_joints, betas, poses, trans, meta = load_results(result_dir)

    # Load SMPL model
    print(f"Loading SMPL model from {smpl_model_path}...")
    smpl_data = load_smpl_model(smpl_model_path)

    # Auto-detect body type
    body_type = detect_body_type(meta)
    print(f'Detected body type: {body_type.upper()}')

    # Handle joint mapping
    joint_mapping = meta.get('joint_mapping', {})
    if pred_joints.shape[1] != target_joints.shape[1] and len(joint_mapping) > 0:
        addb_to_smpl = {int(k): int(v) for k, v in joint_mapping.items()}
        addb_indices = sorted(addb_to_smpl.keys())
        smpl_indices = [addb_to_smpl[i] for i in addb_indices]

        pred_joints_vis = pred_joints[:, smpl_indices, :]
        target_joints_vis = target_joints[:, addb_indices, :]
    else:
        pred_joints_vis = pred_joints
        target_joints_vis = target_joints

    num_frames = pred_joints.shape[0]

    # Compute SMPL mesh for all frames
    print(f"Computing SMPL mesh for {num_frames} frames...")
    vertices_all = []
    for t in range(num_frames):
        vertices, faces = smpl_forward(smpl_data, betas, poses[t], trans[t])
        vertices_all.append(vertices)
    vertices_all = np.array(vertices_all)

    # Get skeleton edges
    skeleton_edges = get_body_edges_from_joints(pred_joints[0])

    # Compute global bounds
    all_points = np.concatenate([
        vertices_all.reshape(-1, 3),
        pred_joints.reshape(-1, 3),
        target_joints.reshape(-1, 3)
    ], axis=0)
    all_points = all_points[~np.isnan(all_points).any(axis=1)]

    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)

    # Add padding
    padding = 0.2
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range
    z_min -= padding * z_range
    z_max += padding * z_range

    # Setup figure
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_data):
        frame_idx, angle = frame_data
        ax.clear()

        # Get current frame
        vertices = vertices_all[frame_idx]
        pred = pred_joints[frame_idx]
        target_vis = target_joints_vis[frame_idx]

        # Draw mesh wireframe (sample edges)
        mesh_edges = get_mesh_edges(vertices, faces, sample_rate=wireframe_density)
        for v1, v2 in mesh_edges:
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]],
                   'cyan', linewidth=0.3, alpha=0.3)

        # Draw SMPL skeleton (thicker lines)
        for i, j in skeleton_edges:
            if not (np.isnan(pred[i]).any() or np.isnan(pred[j]).any()):
                ax.plot([pred[i, 0], pred[j, 0]],
                       [pred[i, 1], pred[j, 1]],
                       [pred[i, 2], pred[j, 2]],
                       'blue', linewidth=2.5, alpha=0.8)

        # Plot SMPL joints
        pred_valid = pred[~np.isnan(pred).any(axis=1)]
        if len(pred_valid) > 0:
            ax.scatter(pred_valid[:, 0], pred_valid[:, 1], pred_valid[:, 2],
                      c='blue', s=60, marker='o', label='SMPL (Predicted)',
                      alpha=0.9, edgecolors='darkblue', linewidths=1.5, depthshade=False)

        # Plot target joints (AddB)
        valid_target = target_vis[~np.isnan(target_vis).any(axis=1)]
        if len(valid_target) > 0:
            ax.scatter(valid_target[:, 0], valid_target[:, 1], valid_target[:, 2],
                      c='red', s=80, marker='o', label='AddB (Target)',
                      alpha=1.0, edgecolors='darkred', linewidths=2,
                      depthshade=False, zorder=10)

        # Compute frame MPJPE
        valid_mask = ~(np.isnan(pred_joints_vis[frame_idx]).any(axis=1) |
                       np.isnan(target_vis).any(axis=1))
        if valid_mask.sum() > 0:
            mpjpe_frame = np.linalg.norm(
                pred_joints_vis[frame_idx][valid_mask] - target_vis[valid_mask],
                axis=1
            ).mean() * 1000
        else:
            mpjpe_frame = 0.0

        # Set fixed bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # Set viewing angle
        ax.view_init(elev=20, azim=angle)

        # Labels and title
        ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=11, fontweight='bold')

        subject_name = meta.get('run_name', 'Unknown')
        overall_mpjpe = meta.get('MPJPE', 0.0)

        title_suffix = "FULL BODY MESH" if body_type == "with_arm" else "LOWER BODY MESH"

        ax.set_title(
            f'{subject_name} - {title_suffix}\n'
            f'Frame {frame_idx + 1}/{num_frames} | '
            f'Frame MPJPE: {mpjpe_frame:.2f}mm | '
            f'Overall MPJPE: {overall_mpjpe:.2f}mm',
            fontsize=13, fontweight='bold', pad=20
        )

        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Add ground plane
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
                              np.linspace(y_min, y_max, 10))
        zz = np.ones_like(xx) * z_min
        ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

        return ax,

    # Generate frame data with rotation
    angles = np.linspace(0, rotation_degrees, num_frames)
    frame_data = [(i, angle) for i, angle in enumerate(angles)]

    print(f'Creating 3D MESH animation with {num_frames} frames...')
    print(f'Rotation: {rotation_degrees}° total ({rotation_degrees/num_frames:.2f}° per frame)')
    print(f'Wireframe density: 1/{wireframe_density} faces')

    anim = FuncAnimation(fig, update, frames=frame_data,
                        interval=1000/fps, blit=False)

    # Save
    if output_path is None:
        subject_name = meta.get('run_name', 'result')
        output_path = Path(result_dir) / f'{subject_name}_3d_mesh.mp4'
    else:
        output_path = Path(output_path)

    print(f'Saving 3D MESH MP4 to: {output_path}')
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    anim.save(str(output_path), writer=writer)

    plt.close()
    print(f'✓ 3D Mesh Video saved: {output_path}')
    print(f'  Body type: {body_type.upper()}')
    print(f'  File size: {output_path.stat().st_size / 1024:.0f}KB')

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Create 3D mesh visualization with SMPL body shape (wireframe)'
    )
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Path to result directory')
    parser.add_argument('--smpl_model', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl',
                       help='Path to SMPL model pickle file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output MP4 path (default: auto-generated)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--rotation_degrees', type=int, default=180,
                       help='Total rotation in degrees (default: 180)')
    parser.add_argument('--wireframe_density', type=int, default=20,
                       help='Sample every Nth face for wireframe (default: 20)')

    args = parser.parse_args()
    create_mesh_animation(args.result_dir, args.smpl_model, args.output,
                         args.fps, args.rotation_degrees, args.wireframe_density)


if __name__ == '__main__':
    main()
