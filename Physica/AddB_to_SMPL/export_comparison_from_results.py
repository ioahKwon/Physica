#!/usr/bin/env python3
"""
Export SMPL and AddBiomechanics models directly from optimization results
This uses the exact same data that was used during optimization (target_joints.npy)
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import sys
import trimesh
from tqdm import tqdm

# Import SMPL model
sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel


def create_skeleton_mesh(joints: np.ndarray, joint_radius: float = 0.01, bone_radius: float = 0.005) -> trimesh.Trimesh:
    """Create a mesh representation of skeleton from joint positions

    Joint indices (from AddB data):
     0: ground_pelvis
     1: hip_r,  2: walker_knee_r,  3: ankle_r,  4: subtalar_r,  5: mtp_r
     6: hip_l,  7: walker_knee_l,  8: ankle_l,  9: subtalar_l, 10: mtp_l
    11: back
    12: acromial_r, 13: elbow_r, 14: radioulnar_r, 15: radius_hand_r
    16: acromial_l, 17: elbow_l, 18: radioulnar_l, 19: radius_hand_l
    """
    meshes = []

    # Define skeleton connectivity based on actual AddB joint structure
    bone_pairs = [
        # Pelvis to spine
        (0, 11),  # pelvis to back

        # Right leg
        (0, 1),   # pelvis to hip_r
        (1, 2),   # hip_r to knee_r
        (2, 3),   # knee_r to ankle_r
        (3, 4),   # ankle_r to subtalar_r
        (4, 5),   # subtalar_r to mtp_r (toe)

        # Left leg
        (0, 6),   # pelvis to hip_l
        (6, 7),   # hip_l to knee_l
        (7, 8),   # knee_l to ankle_l
        (8, 9),   # ankle_l to subtalar_l
        (9, 10),  # subtalar_l to mtp_l (toe)

        # Right arm
        (11, 12), # back to shoulder_r
        (12, 13), # shoulder_r to elbow_r
        (13, 14), # elbow_r to radioulnar_r
        (14, 15), # radioulnar_r to wrist_r

        # Left arm
        (11, 16), # back to shoulder_l
        (16, 17), # shoulder_l to elbow_l
        (17, 18), # elbow_l to radioulnar_l
        (18, 19), # radioulnar_l to wrist_l
    ]

    # Create joint spheres (red)
    for joint_pos in joints:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=joint_radius)
        sphere.vertices += joint_pos
        sphere.visual.vertex_colors = [1.0, 0.0, 0.0, 1.0]
        meshes.append(sphere)

    # Create bone cylinders (blue)
    for j1, j2 in bone_pairs:
        if j1 < len(joints) and j2 < len(joints):
            p1, p2 = joints[j1], joints[j2]
            bone_length = np.linalg.norm(p2 - p1)

            if bone_length > 1e-6:
                cylinder = trimesh.creation.cylinder(
                    radius=bone_radius,
                    height=bone_length,
                    sections=8
                )

                bone_dir = (p2 - p1) / bone_length
                z_axis = np.array([0, 0, 1])
                rot_axis = np.cross(z_axis, bone_dir)
                rot_axis_norm = np.linalg.norm(rot_axis)

                if rot_axis_norm > 1e-6:
                    rot_axis /= rot_axis_norm
                    cos_angle = np.dot(z_axis, bone_dir)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    K = np.array([
                        [0, -rot_axis[2], rot_axis[1]],
                        [rot_axis[2], 0, -rot_axis[0]],
                        [-rot_axis[1], rot_axis[0], 0]
                    ])
                    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
                else:
                    R = np.eye(3) if bone_dir[2] > 0 else np.diag([1, 1, -1])

                cylinder.vertices = cylinder.vertices @ R.T
                cylinder.vertices += (p1 + p2) / 2
                cylinder.visual.vertex_colors = [0.0, 0.0, 1.0, 1.0]
                meshes.append(cylinder)

    return trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()


def export_comparison_from_results(
    result_dir: Path,
    output_dir: Path,
    smpl_model_path: Path,
    frame_range: tuple = None,
    smpl_color: tuple = (0.7, 0.8, 1.0, 0.5),
    joint_radius: float = 0.015,
    bone_radius: float = 0.007
):
    """
    Export both SMPL mesh and AddB skeleton from optimization results directory

    Args:
        result_dir: Directory containing smpl_params.npz and target_joints.npy
        output_dir: Directory to save OBJ files
        smpl_model_path: Path to SMPL model (.pkl)
        frame_range: Optional (start, end) frame indices
        smpl_color: RGBA color for SMPL mesh
        joint_radius: Radius of AddB joint spheres
        bone_radius: Radius of AddB bone cylinders
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SMPL + AddBiomechanics COMPARISON EXPORT (FROM RESULTS)")
    print("="*80)

    # -------------------------------------------------------------------------
    # STEP 1: Load SMPL Model and Parameters
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading SMPL model and parameters...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl_model = SMPLModel(model_path=str(smpl_model_path), device=device)

    if smpl_model.faces is None:
        raise ValueError("SMPL model does not contain face information")

    # Load SMPL parameters
    smpl_params_path = result_dir / "smpl_params.npz"
    if not smpl_params_path.exists():
        raise FileNotFoundError(f"smpl_params.npz not found in {result_dir}")

    data = np.load(smpl_params_path)
    betas = torch.tensor(data['betas'], dtype=torch.float32, device=device)
    poses = torch.tensor(data['poses'], dtype=torch.float32, device=device)
    trans = torch.tensor(data['trans'], dtype=torch.float32, device=device)

    smpl_num_frames = poses.shape[0]
    print(f"  SMPL frames: {smpl_num_frames}")
    print(f"  Betas: {betas.shape}, Poses: {poses.shape}, Trans: {trans.shape}")

    faces = smpl_model.faces.cpu().numpy()

    # -------------------------------------------------------------------------
    # STEP 2: Load Target Joints (AddB GT)
    # -------------------------------------------------------------------------
    print("\n[2/4] Loading AddBiomechanics target joints...")

    target_joints_path = result_dir / "target_joints.npy"
    if not target_joints_path.exists():
        raise FileNotFoundError(f"target_joints.npy not found in {result_dir}")

    target_joints = np.load(target_joints_path)  # [T, N_joints, 3]
    addb_num_frames = target_joints.shape[0]
    addb_num_joints = target_joints.shape[1]

    print(f"  AddB frames: {addb_num_frames}")
    print(f"  AddB joints: {addb_num_joints}")
    print(f"  Joint range (frame 0): {target_joints[0].min(axis=0)} to {target_joints[0].max(axis=0)}")

    # -------------------------------------------------------------------------
    # STEP 3: Determine Export Frame Range
    # -------------------------------------------------------------------------
    print("\n[3/4] Determining export frame range...")

    max_frames = min(smpl_num_frames, addb_num_frames)

    if frame_range:
        export_start = frame_range[0]
        export_end = min(frame_range[1], max_frames)
    else:
        export_start = 0
        export_end = max_frames

    print(f"  Exporting {export_end - export_start} frames ({export_start} to {export_end-1})")

    # -------------------------------------------------------------------------
    # STEP 4: Export Both Models
    # -------------------------------------------------------------------------
    print(f"\n[4/4] Exporting SMPL meshes and AddB skeletons...")

    for frame_idx in tqdm(range(export_start, export_end), desc="Exporting frames"):
        # Export SMPL mesh
        pose = poses[frame_idx].reshape(24, 3)
        t = trans[frame_idx]

        with torch.no_grad():
            vertices, _ = smpl_model.forward(betas, pose, t)

        vertices_np = vertices.cpu().numpy()
        smpl_mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces, process=False)
        smpl_mesh.visual.vertex_colors = [int(c*255) for c in smpl_color]

        smpl_output_path = output_dir / f"smpl_frame_{frame_idx:04d}.obj"
        smpl_mesh.export(str(smpl_output_path))

        # Export AddB skeleton
        addb_skeleton = create_skeleton_mesh(
            target_joints[frame_idx],
            joint_radius=joint_radius,
            bone_radius=bone_radius
        )

        addb_output_path = output_dir / f"addb_frame_{frame_idx:04d}.obj"
        addb_skeleton.export(str(addb_output_path))

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("EXPORT COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total frames exported: {export_end - export_start}")
    print(f"\nFiles created:")
    print(f"  SMPL meshes: smpl_frame_XXXX.obj ({export_end - export_start} files)")
    print(f"  AddB skeletons: addb_frame_XXXX.obj ({export_end - export_start} files)")
    print(f"\n" + "="*80)
    print("HOW TO VISUALIZE IN MESHLAB:")
    print("="*80)
    print(f"\n1. Open MeshLab")
    print(f"\n2. Load both models for a single frame:")
    print(f"   File -> Import Mesh -> Select both:")
    print(f"     - {output_dir}/smpl_frame_0000.obj")
    print(f"     - {output_dir}/addb_frame_0000.obj")
    print(f"\n3. Both models should be aligned now (same coordinate system)")
    print()

    return export_end - export_start


def main():
    parser = argparse.ArgumentParser(
        description="Export SMPL and AddBiomechanics from optimization results directory"
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        required=True,
        help='Directory containing smpl_params.npz and target_joints.npy'
    )
    parser.add_argument(
        '--smpl_model',
        type=str,
        default='models/smpl_model.pkl',
        help='Path to SMPL model pickle file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save OBJ files'
    )
    parser.add_argument(
        '--start_frame',
        type=int,
        default=None,
        help='Start frame index (default: 0)'
    )
    parser.add_argument(
        '--end_frame',
        type=int,
        default=None,
        help='End frame index (default: all frames)'
    )
    parser.add_argument(
        '--joint_radius',
        type=float,
        default=0.015,
        help='Radius of AddB joint spheres (default: 0.015m)'
    )
    parser.add_argument(
        '--bone_radius',
        type=float,
        default=0.007,
        help='Radius of AddB bone cylinders (default: 0.007m)'
    )

    args = parser.parse_args()

    # Convert paths
    result_dir = Path(args.result_dir)
    smpl_model_path = Path(args.smpl_model)
    output_dir = Path(args.output_dir)

    # Validate input paths
    if not result_dir.exists():
        print(f"Error: Result directory not found: {result_dir}")
        sys.exit(1)

    if not smpl_model_path.exists():
        print(f"Error: SMPL model file not found: {smpl_model_path}")
        sys.exit(1)

    # Determine frame range
    frame_range = None
    if args.start_frame is not None or args.end_frame is not None:
        start = args.start_frame if args.start_frame is not None else 0
        end = args.end_frame if args.end_frame is not None else float('inf')
        frame_range = (start, int(end))

    # Export
    export_comparison_from_results(
        result_dir=result_dir,
        smpl_model_path=smpl_model_path,
        output_dir=output_dir,
        frame_range=frame_range,
        joint_radius=args.joint_radius,
        bone_radius=args.bone_radius
    )


if __name__ == '__main__':
    main()
