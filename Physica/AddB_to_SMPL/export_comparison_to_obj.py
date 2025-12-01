#!/usr/bin/env python3
"""
Export both SMPL and AddBiomechanics models to OBJ files for simultaneous MeshLab visualization
This script combines both export_smpl_to_obj.py and export_addb_to_obj.py
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import sys
import trimesh
from tqdm import tqdm

# Import modules
sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel

try:
    import nimblephysics as nimble
except ImportError as exc:
    raise RuntimeError(
        "nimblephysics is required to read AddBiomechanics .b3d files. "
        "Install nimblephysics inside the current environment."
    ) from exc


def load_addb_joints(
    b3d_path: str,
    trial: int = 0,
    processing_pass: int = 0,
    start_frame: int = 0,
    num_frames: int = -1
) -> tuple:
    """Load joint positions from AddBiomechanics .b3d file"""
    subj = nimble.biomechanics.SubjectOnDisk(b3d_path)
    total = subj.getTrialLength(trial)
    count = total - start_frame if num_frames < 0 else min(num_frames, total - start_frame)

    frames = subj.readFrames(
        trial=trial,
        startFrame=start_frame,
        numFramesToRead=count,
        includeProcessingPasses=True,
        includeSensorData=False,
        stride=1,
        contactThreshold=1.0
    )

    if len(frames) == 0:
        raise RuntimeError("Failed to read frames from .b3d file")

    def frame_joint_centers(frame):
        if hasattr(frame, 'processingPasses') and len(frame.processingPasses) > 0:
            idx = min(processing_pass, len(frame.processingPasses) - 1)
            return np.asarray(frame.processingPasses[idx].jointCenters, dtype=np.float32)
        return np.asarray(frame.jointCenters, dtype=np.float32)

    first = frame_joint_centers(frames[0])
    num_joints = first.size // 3
    joints = np.zeros((len(frames), num_joints, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        data = frame_joint_centers(frame)
        joints[i] = data.reshape(-1, 3)[:num_joints]

    dt = float(subj.getTrialTimestep(trial))
    skel = subj.readSkel(0)
    joint_names = [skel.getJoint(i).getName() for i in range(num_joints)]

    return joints, joint_names, dt


def create_skeleton_mesh(joints: np.ndarray, joint_radius: float = 0.01, bone_radius: float = 0.005) -> trimesh.Trimesh:
    """Create a mesh representation of skeleton from joint positions"""
    meshes = []

    # Define skeleton connectivity
    bone_pairs = [
        (0, 1),   # pelvis to spine
        (0, 2),   # pelvis to left hip
        (2, 3),   # left hip to left knee
        (3, 4),   # left knee to left ankle
        (0, 5),   # pelvis to right hip
        (5, 6),   # right hip to right knee
        (6, 7),   # right knee to right ankle
    ]

    # Add more bones if joints exist
    if len(joints) > 8:
        bone_pairs.extend([(1, 8), (8, 9)])
    if len(joints) > 10:
        bone_pairs.append((9, 10))
    if len(joints) > 11:
        bone_pairs.extend([(1, 11), (11, 12)])
    if len(joints) > 13:
        bone_pairs.append((12, 13))
    if len(joints) > 14:
        bone_pairs.append((4, 14))
    if len(joints) > 15:
        bone_pairs.append((7, 15))

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


def export_comparison(
    b3d_path: Path,
    smpl_params_path: Path,
    smpl_model_path: Path,
    output_dir: Path,
    frame_range: tuple = None,
    smpl_color: tuple = (0.7, 0.8, 1.0, 0.5),
    joint_radius: float = 0.015,
    bone_radius: float = 0.007
):
    """
    Export both SMPL mesh and AddB skeleton to OBJ files for comparison

    Args:
        b3d_path: Path to .b3d file
        smpl_params_path: Path to smpl_params.npz file
        smpl_model_path: Path to SMPL model (.pkl)
        output_dir: Directory to save OBJ files
        frame_range: Optional (start, end) frame indices
        smpl_color: RGBA color for SMPL mesh
        joint_radius: Radius of AddB joint spheres
        bone_radius: Radius of AddB bone cylinders
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SMPL + AddBiomechanics COMPARISON EXPORT")
    print("="*80)

    # -------------------------------------------------------------------------
    # STEP 1: Load SMPL Model and Parameters
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading SMPL model and parameters...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl_model = SMPLModel(model_path=str(smpl_model_path), device=device)

    if smpl_model.faces is None:
        raise ValueError("SMPL model does not contain face information")

    data = np.load(smpl_params_path)
    betas = torch.tensor(data['betas'], dtype=torch.float32, device=device)
    poses = torch.tensor(data['poses'], dtype=torch.float32, device=device)
    trans = torch.tensor(data['trans'], dtype=torch.float32, device=device)

    smpl_num_frames = poses.shape[0]
    print(f"  SMPL frames: {smpl_num_frames}")
    print(f"  Betas: {betas.shape}, Poses: {poses.shape}, Trans: {trans.shape}")

    faces = smpl_model.faces.cpu().numpy()

    # -------------------------------------------------------------------------
    # STEP 2: Load AddBiomechanics Data
    # -------------------------------------------------------------------------
    print("\n[2/5] Loading AddBiomechanics skeleton...")

    start_frame = frame_range[0] if frame_range else 0
    num_frames = (frame_range[1] - start_frame) if frame_range else -1

    addb_joints, joint_names, dt = load_addb_joints(
        str(b3d_path),
        trial=0,
        processing_pass=0,
        start_frame=start_frame,
        num_frames=num_frames
    )

    addb_num_frames = addb_joints.shape[0]
    print(f"  AddB frames: {addb_num_frames}")
    print(f"  Joints: {addb_joints.shape[1]}")
    print(f"  Joint names: {joint_names}")

    # -------------------------------------------------------------------------
    # STEP 3: Determine Export Frame Range
    # -------------------------------------------------------------------------
    print("\n[3/5] Determining export frame range...")

    # Use minimum of both
    max_frames = min(smpl_num_frames, addb_num_frames)
    if frame_range:
        export_start = 0  # Already offset when loading
        export_end = max_frames
    else:
        export_start = 0
        export_end = max_frames

    print(f"  Exporting {export_end - export_start} frames (0 to {export_end-1})")

    # -------------------------------------------------------------------------
    # STEP 4: Export SMPL Meshes
    # -------------------------------------------------------------------------
    print(f"\n[4/5] Exporting SMPL meshes...")

    for frame_idx in tqdm(range(export_start, export_end), desc="Exporting SMPL"):
        pose = poses[frame_idx].reshape(24, 3)
        t = trans[frame_idx]

        with torch.no_grad():
            vertices, _ = smpl_model.forward(betas, pose, t)

        vertices_np = vertices.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces, process=False)
        mesh.visual.vertex_colors = [int(c*255) for c in smpl_color]

        actual_frame = start_frame + frame_idx
        output_path = output_dir / f"smpl_frame_{actual_frame:04d}.obj"
        mesh.export(str(output_path))

    # -------------------------------------------------------------------------
    # STEP 5: Export AddB Skeletons
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Exporting AddBiomechanics skeletons...")

    for frame_idx in tqdm(range(export_start, export_end), desc="Exporting AddB"):
        skeleton_mesh = create_skeleton_mesh(
            addb_joints[frame_idx],
            joint_radius=joint_radius,
            bone_radius=bone_radius
        )

        actual_frame = start_frame + frame_idx
        output_path = output_dir / f"addb_frame_{actual_frame:04d}.obj"
        skeleton_mesh.export(str(output_path))

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
    print(f"\n3. View both models simultaneously to compare alignment")
    print(f"\n4. Tips:")
    print(f"   - Use 'Show Layer Dialog' to toggle visibility")
    print(f"   - Adjust transparency in 'Render' menu")
    print(f"   - Use 'Align' tools to overlay if needed")
    print(f"\n5. For animation, load sequential frames manually")
    print()

    return export_end - export_start


def main():
    parser = argparse.ArgumentParser(
        description="Export both SMPL and AddBiomechanics to OBJ for simultaneous MeshLab visualization"
    )
    parser.add_argument(
        '--b3d',
        type=str,
        required=True,
        help='Path to AddBiomechanics .b3d file'
    )
    parser.add_argument(
        '--smpl_params',
        type=str,
        required=True,
        help='Path to smpl_params.npz file'
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
    b3d_path = Path(args.b3d)
    smpl_params_path = Path(args.smpl_params)
    smpl_model_path = Path(args.smpl_model)
    output_dir = Path(args.output_dir)

    # Validate input paths
    if not b3d_path.exists():
        print(f"Error: .b3d file not found: {b3d_path}")
        sys.exit(1)

    if not smpl_params_path.exists():
        print(f"Error: SMPL parameters file not found: {smpl_params_path}")
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
    export_comparison(
        b3d_path=b3d_path,
        smpl_params_path=smpl_params_path,
        smpl_model_path=smpl_model_path,
        output_dir=output_dir,
        frame_range=frame_range,
        joint_radius=args.joint_radius,
        bone_radius=args.bone_radius
    )


if __name__ == '__main__':
    main()
