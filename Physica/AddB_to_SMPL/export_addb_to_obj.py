#!/usr/bin/env python3
"""
Export AddBiomechanics (OpenSIM) skeleton to OBJ files for MeshLab visualization
Loads .b3d file and exports skeleton/joints as mesh for each frame
"""

import numpy as np
from pathlib import Path
import argparse
import sys
import trimesh
from tqdm import tqdm

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
    """
    Load joint positions from AddBiomechanics .b3d file

    Args:
        b3d_path: Path to .b3d file
        trial: Trial index
        processing_pass: Processing pass index
        start_frame: Start frame index
        num_frames: Number of frames to read (-1 for all)

    Returns:
        joints: [T, N_joints, 3] array of joint positions
        joint_names: List of joint names
        dt: Time step
    """
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
    if first.ndim != 1 or first.size % 3 != 0:
        raise ValueError("Unexpected joint center layout in .b3d file")

    num_joints = first.size // 3
    joints = np.zeros((len(frames), num_joints, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        data = frame_joint_centers(frame)
        joints[i] = data.reshape(-1, 3)[:num_joints]

    dt = float(subj.getTrialTimestep(trial))

    # Get joint names from skeleton
    skel = subj.readSkel(0)
    joint_names = [skel.getJoint(i).getName() for i in range(num_joints)]

    return joints, joint_names, dt


def create_skeleton_mesh(joints: np.ndarray, joint_radius: float = 0.01, bone_radius: float = 0.005) -> trimesh.Trimesh:
    """
    Create a mesh representation of skeleton from joint positions

    Args:
        joints: [N_joints, 3] array of joint positions
        joint_radius: Radius of joint spheres
        bone_radius: Radius of bone cylinders

    Returns:
        trimesh.Trimesh: Combined mesh of joints and bones
    """
    meshes = []

    # Define skeleton connectivity (common biomechanics skeleton structure)
    # This is a simplified version - adjust based on your specific skeleton
    bone_pairs = [
        # Spine/torso
        (0, 1),  # pelvis to spine
        # Left leg
        (0, 2),  # pelvis to left hip
        (2, 3),  # left hip to left knee
        (3, 4),  # left knee to left ankle
        # Right leg
        (0, 5),  # pelvis to right hip
        (5, 6),  # right hip to right knee
        (6, 7),  # right knee to right ankle
    ]

    # Try to add more bones if joints exist
    if len(joints) > 8:
        # Left arm (if exists)
        bone_pairs.extend([
            (1, 8),   # spine to left shoulder
            (8, 9),   # left shoulder to left elbow
        ])
    if len(joints) > 10:
        bone_pairs.extend([
            (9, 10),  # left elbow to left wrist
        ])
    if len(joints) > 11:
        # Right arm
        bone_pairs.extend([
            (1, 11),  # spine to right shoulder
            (11, 12), # right shoulder to right elbow
        ])
    if len(joints) > 13:
        bone_pairs.extend([
            (12, 13), # right elbow to right wrist
        ])

    # Add toe joints if they exist
    if len(joints) > 4:
        bone_pairs.append((4, 14 if len(joints) > 14 else 4))  # left ankle to left toe
    if len(joints) > 7:
        bone_pairs.append((7, 15 if len(joints) > 15 else 7))  # right ankle to right toe

    # Create joint spheres
    for i, joint_pos in enumerate(joints):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=joint_radius)
        sphere.vertices += joint_pos
        sphere.visual.vertex_colors = [1.0, 0.0, 0.0, 1.0]  # Red for joints
        meshes.append(sphere)

    # Create bone cylinders
    for j1, j2 in bone_pairs:
        if j1 < len(joints) and j2 < len(joints):
            p1 = joints[j1]
            p2 = joints[j2]
            bone_length = np.linalg.norm(p2 - p1)

            if bone_length > 1e-6:  # Avoid zero-length bones
                # Create cylinder along Z axis
                cylinder = trimesh.creation.cylinder(
                    radius=bone_radius,
                    height=bone_length,
                    sections=8
                )

                # Compute rotation to align cylinder with bone direction
                bone_dir = (p2 - p1) / bone_length
                z_axis = np.array([0, 0, 1])

                # Rotation axis (cross product)
                rot_axis = np.cross(z_axis, bone_dir)
                rot_axis_norm = np.linalg.norm(rot_axis)

                if rot_axis_norm > 1e-6:
                    rot_axis = rot_axis / rot_axis_norm
                    # Rotation angle
                    cos_angle = np.dot(z_axis, bone_dir)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

                    # Create rotation matrix using Rodrigues' formula
                    K = np.array([
                        [0, -rot_axis[2], rot_axis[1]],
                        [rot_axis[2], 0, -rot_axis[0]],
                        [-rot_axis[1], rot_axis[0], 0]
                    ])
                    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
                else:
                    # Bones are aligned or opposite to z-axis
                    R = np.eye(3) if bone_dir[2] > 0 else np.diag([1, 1, -1])

                # Apply transformation
                cylinder.vertices = cylinder.vertices @ R.T
                cylinder.vertices += (p1 + p2) / 2  # Translate to midpoint

                cylinder.visual.vertex_colors = [0.0, 0.0, 1.0, 1.0]  # Blue for bones
                meshes.append(cylinder)

    # Combine all meshes
    if meshes:
        combined = trimesh.util.concatenate(meshes)
        return combined
    else:
        # Return empty mesh if no joints
        return trimesh.Trimesh()


def export_addb_sequence(
    b3d_path: Path,
    output_dir: Path,
    frame_range: tuple = None,
    prefix: str = "addb",
    joint_radius: float = 0.015,
    bone_radius: float = 0.007
):
    """
    Export AddBiomechanics skeleton sequence to OBJ files

    Args:
        b3d_path: Path to .b3d file
        output_dir: Directory to save OBJ files
        frame_range: Optional (start, end) frame indices
        prefix: Prefix for output files (default: "addb")
        joint_radius: Radius of joint spheres
        bone_radius: Radius of bone cylinders
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load AddBiomechanics data
    print(f"[1/2] Loading AddBiomechanics data from {b3d_path}")

    start_frame = frame_range[0] if frame_range else 0
    num_frames = (frame_range[1] - start_frame) if frame_range else -1

    joints, joint_names, dt = load_addb_joints(
        str(b3d_path),
        trial=0,
        processing_pass=0,
        start_frame=start_frame,
        num_frames=num_frames
    )

    num_total_frames = joints.shape[0]
    print(f"  Total frames: {num_total_frames}")
    print(f"  Number of joints: {joints.shape[1]}")
    print(f"  Time step: {dt:.4f}s")
    print(f"  Joint names: {joint_names}")

    # Export each frame
    print(f"[2/2] Exporting {num_total_frames} frames to OBJ files...")

    for frame_idx in tqdm(range(num_total_frames), desc="Exporting AddB"):
        # Create skeleton mesh for this frame
        skeleton_mesh = create_skeleton_mesh(
            joints[frame_idx],
            joint_radius=joint_radius,
            bone_radius=bone_radius
        )

        # Export to OBJ
        actual_frame_idx = start_frame + frame_idx
        output_path = output_dir / f"{prefix}_frame_{actual_frame_idx:04d}.obj"
        skeleton_mesh.export(str(output_path))

    print(f"\nExport complete! Files saved to: {output_dir}")
    print(f"Total OBJ files created: {num_total_frames}")

    return num_total_frames


def main():
    parser = argparse.ArgumentParser(
        description="Export AddBiomechanics skeleton to OBJ files for MeshLab"
    )
    parser.add_argument(
        '--b3d',
        type=str,
        required=True,
        help='Path to .b3d file'
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
        '--prefix',
        type=str,
        default='addb',
        help='Prefix for output OBJ files (default: "addb")'
    )
    parser.add_argument(
        '--joint_radius',
        type=float,
        default=0.015,
        help='Radius of joint spheres (default: 0.015m)'
    )
    parser.add_argument(
        '--bone_radius',
        type=float,
        default=0.007,
        help='Radius of bone cylinders (default: 0.007m)'
    )

    args = parser.parse_args()

    # Convert paths
    b3d_path = Path(args.b3d)
    output_dir = Path(args.output_dir)

    # Validate input paths
    if not b3d_path.exists():
        print(f"Error: .b3d file not found: {b3d_path}")
        sys.exit(1)

    # Determine frame range
    frame_range = None
    if args.start_frame is not None or args.end_frame is not None:
        start = args.start_frame if args.start_frame is not None else 0
        end = args.end_frame if args.end_frame is not None else float('inf')
        frame_range = (start, int(end))

    # Export
    print("\n" + "="*80)
    print("ADDBIOMECHANICS SKELETON TO OBJ EXPORTER")
    print("="*80)

    export_addb_sequence(
        b3d_path=b3d_path,
        output_dir=output_dir,
        frame_range=frame_range,
        prefix=args.prefix,
        joint_radius=args.joint_radius,
        bone_radius=args.bone_radius
    )

    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print(f"\nYou can now open the OBJ files in MeshLab:")
    print(f"  meshlab {output_dir}/{args.prefix}_frame_*.obj")
    print()


if __name__ == '__main__':
    main()
