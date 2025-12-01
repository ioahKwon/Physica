#!/usr/bin/env python3
"""
Export SMPL mesh to OBJ files for MeshLab visualization
Loads fitted SMPL parameters and exports each frame as an OBJ file
Optionally exports joints as point cloud for visualization
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

# SMPL joint names for reference
SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand'
]

# SMPL skeleton bones (parent -> child)
SMPL_BONES = [
    (0, 1),   # pelvis -> left_hip
    (0, 2),   # pelvis -> right_hip
    (0, 3),   # pelvis -> spine1
    (1, 4),   # left_hip -> left_knee
    (2, 5),   # right_hip -> right_knee
    (3, 6),   # spine1 -> spine2
    (4, 7),   # left_knee -> left_ankle
    (5, 8),   # right_knee -> right_ankle
    (6, 9),   # spine2 -> spine3
    (7, 10),  # left_ankle -> left_foot
    (8, 11),  # right_ankle -> right_foot
    (9, 12),  # spine3 -> neck
    (9, 13),  # spine3 -> left_collar
    (9, 14),  # spine3 -> right_collar
    (12, 15), # neck -> head
    (13, 16), # left_collar -> left_shoulder
    (14, 17), # right_collar -> right_shoulder
    (16, 18), # left_shoulder -> left_elbow
    (17, 19), # right_shoulder -> right_elbow
    (18, 20), # left_elbow -> left_wrist
    (19, 21), # right_elbow -> right_wrist
    (20, 22), # left_wrist -> left_hand
    (21, 23), # right_wrist -> right_hand
]

# AddB joint names (from the dataset)
ADDB_JOINT_NAMES = [
    'ground_pelvis', 'hip_r', 'walker_knee_r', 'ankle_r', 'subtalar_r', 'mtp_r',
    'hip_l', 'walker_knee_l', 'ankle_l', 'subtalar_l', 'mtp_l', 'back',
    'acromial_r', 'elbow_r', 'radioulnar_r', 'radius_hand_r',
    'acromial_l', 'elbow_l', 'radioulnar_l', 'radius_hand_l'
]

# AddB skeleton bones (parent -> child)
ADDB_BONES = [
    (0, 1),   # pelvis -> hip_r
    (0, 6),   # pelvis -> hip_l
    (0, 11),  # pelvis -> back
    (1, 2),   # hip_r -> knee_r
    (2, 3),   # knee_r -> ankle_r
    (3, 4),   # ankle_r -> subtalar_r
    (4, 5),   # subtalar_r -> mtp_r
    (6, 7),   # hip_l -> knee_l
    (7, 8),   # knee_l -> ankle_l
    (8, 9),   # ankle_l -> subtalar_l
    (9, 10),  # subtalar_l -> mtp_l
    (11, 12), # back -> acromial_r (shoulder_r)
    (11, 16), # back -> acromial_l (shoulder_l)
    (12, 13), # acromial_r -> elbow_r
    (13, 14), # elbow_r -> radioulnar_r
    (14, 15), # radioulnar_r -> radius_hand_r
    (16, 17), # acromial_l -> elbow_l
    (17, 18), # elbow_l -> radioulnar_l
    (18, 19), # radioulnar_l -> radius_hand_l
]


def write_skeleton_obj(filepath: Path, joints: np.ndarray, bones: list,
                       joint_names: list = None, sphere_radius: float = 0.015,
                       bone_radius: float = 0.008):
    """
    Write skeleton as OBJ file with joints (spheres) and bones (cylinders).

    Args:
        filepath: Output path
        joints: Joint positions [N, 3]
        bones: List of (parent_idx, child_idx) tuples
        joint_names: Optional list of joint names
        sphere_radius: Radius for joint spheres (default 1.5cm)
        bone_radius: Radius for bone cylinders (default 0.8cm)
    """
    with open(filepath, 'w') as f:
        f.write("# Skeleton with joints and bones\n")
        f.write(f"# Total joints: {len(joints)}\n")
        f.write(f"# Total bones: {len(bones)}\n\n")

        vertex_idx = 1

        # Write joints as octahedrons
        f.write("# === JOINTS ===\n")
        for i, joint in enumerate(joints):
            name = joint_names[i] if joint_names and i < len(joint_names) else f"joint_{i}"
            f.write(f"# Joint {i}: {name}\n")

            x, y, z = joint
            offsets = [
                (sphere_radius, 0, 0), (-sphere_radius, 0, 0),
                (0, sphere_radius, 0), (0, -sphere_radius, 0),
                (0, 0, sphere_radius), (0, 0, -sphere_radius)
            ]

            for ox, oy, oz in offsets:
                f.write(f"v {x+ox:.6f} {y+oy:.6f} {z+oz:.6f}\n")

            base = vertex_idx
            f.write(f"f {base+2} {base} {base+4}\n")
            f.write(f"f {base+2} {base+4} {base+1}\n")
            f.write(f"f {base+2} {base+1} {base+5}\n")
            f.write(f"f {base+2} {base+5} {base}\n")
            f.write(f"f {base+3} {base+4} {base}\n")
            f.write(f"f {base+3} {base+1} {base+4}\n")
            f.write(f"f {base+3} {base+5} {base+1}\n")
            f.write(f"f {base+3} {base} {base+5}\n")

            vertex_idx += 6
            f.write("\n")

        # Write bones as cylinders (approximated with 6-sided prisms)
        f.write("# === BONES ===\n")
        num_sides = 6
        for parent_idx, child_idx in bones:
            if parent_idx >= len(joints) or child_idx >= len(joints):
                continue

            p1 = joints[parent_idx]
            p2 = joints[child_idx]

            # Compute bone direction and perpendicular vectors
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue

            direction = direction / length

            # Find perpendicular vectors
            if abs(direction[1]) < 0.9:
                perp1 = np.cross(direction, np.array([0, 1, 0]))
            else:
                perp1 = np.cross(direction, np.array([1, 0, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(direction, perp1)

            parent_name = joint_names[parent_idx] if joint_names and parent_idx < len(joint_names) else f"j{parent_idx}"
            child_name = joint_names[child_idx] if joint_names and child_idx < len(joint_names) else f"j{child_idx}"
            f.write(f"# Bone: {parent_name} -> {child_name}\n")

            # Create vertices for cylinder
            base_start = vertex_idx
            for end_point in [p1, p2]:
                for j in range(num_sides):
                    angle = 2 * np.pi * j / num_sides
                    offset = bone_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                    v = end_point + offset
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Create faces for cylinder sides
            for j in range(num_sides):
                j_next = (j + 1) % num_sides
                v1 = base_start + j
                v2 = base_start + j_next
                v3 = base_start + num_sides + j_next
                v4 = base_start + num_sides + j
                f.write(f"f {v1} {v2} {v3} {v4}\n")

            vertex_idx += num_sides * 2
            f.write("\n")


def write_joints_obj(filepath: Path, joints: np.ndarray, joint_names: list = None):
    """
    Write joints as OBJ file with vertices and optional names as comments.
    Creates small spheres at each joint location for better visibility.
    (Legacy function - use write_skeleton_obj for full skeleton)
    """
    with open(filepath, 'w') as f:
        f.write("# Joint positions as vertices\n")
        f.write(f"# Total joints: {len(joints)}\n\n")

        vertex_idx = 1
        sphere_radius = 0.015  # 1.5cm radius spheres

        for i, joint in enumerate(joints):
            name = joint_names[i] if joint_names and i < len(joint_names) else f"joint_{i}"
            f.write(f"# Joint {i}: {name}\n")

            # Create a small icosphere at each joint
            # Simple approximation: 6 vertices forming an octahedron
            x, y, z = joint
            offsets = [
                (sphere_radius, 0, 0), (-sphere_radius, 0, 0),
                (0, sphere_radius, 0), (0, -sphere_radius, 0),
                (0, 0, sphere_radius), (0, 0, -sphere_radius)
            ]

            for ox, oy, oz in offsets:
                f.write(f"v {x+ox:.6f} {y+oy:.6f} {z+oz:.6f}\n")

            # Create faces for octahedron
            base = vertex_idx
            # Top pyramid (vertex at +Y)
            f.write(f"f {base+2} {base} {base+4}\n")
            f.write(f"f {base+2} {base+4} {base+1}\n")
            f.write(f"f {base+2} {base+1} {base+5}\n")
            f.write(f"f {base+2} {base+5} {base}\n")
            # Bottom pyramid (vertex at -Y)
            f.write(f"f {base+3} {base+4} {base}\n")
            f.write(f"f {base+3} {base+1} {base+4}\n")
            f.write(f"f {base+3} {base+5} {base+1}\n")
            f.write(f"f {base+3} {base} {base+5}\n")

            vertex_idx += 6
            f.write("\n")


def export_smpl_sequence(
    smpl_params_path: Path,
    smpl_model_path: Path,
    output_dir: Path,
    frame_range: tuple = None,
    prefix: str = "smpl",
    export_joints: bool = False,
    target_joints_path: Path = None
):
    """
    Export SMPL mesh sequence to OBJ files

    Args:
        smpl_params_path: Path to smpl_params.npz file
        smpl_model_path: Path to SMPL model (.pkl)
        output_dir: Directory to save OBJ files
        frame_range: Optional (start, end) frame indices
        prefix: Prefix for output files (default: "smpl")
        export_joints: If True, export SMPL joints as separate OBJ files
        target_joints_path: Optional path to target_joints.npy for comparison
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SMPL model
    print(f"[1/3] Loading SMPL model from {smpl_model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl_model = SMPLModel(model_path=str(smpl_model_path), device=device)

    if smpl_model.faces is None:
        raise ValueError("SMPL model does not contain face information")

    # Load SMPL parameters
    print(f"[2/3] Loading SMPL parameters from {smpl_params_path}")
    data = np.load(smpl_params_path)

    betas = torch.tensor(data['betas'], dtype=torch.float32, device=device)  # [10]
    poses = torch.tensor(data['poses'], dtype=torch.float32, device=device)  # [T, 24, 3]
    trans = torch.tensor(data['trans'], dtype=torch.float32, device=device)  # [T, 3]

    num_frames = poses.shape[0]
    print(f"  Total frames: {num_frames}")
    print(f"  Betas shape: {betas.shape}")
    print(f"  Poses shape: {poses.shape}")
    print(f"  Trans shape: {trans.shape}")

    # Determine frame range
    if frame_range is None:
        start_frame, end_frame = 0, num_frames
    else:
        start_frame, end_frame = frame_range
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)

    print(f"  Exporting frames {start_frame} to {end_frame-1}")

    # Get faces as numpy array
    faces = smpl_model.faces.cpu().numpy()

    # Load target joints if provided
    target_joints = None
    if target_joints_path and target_joints_path.exists():
        target_joints = np.load(target_joints_path)
        print(f"  Target joints loaded: {target_joints.shape}")

    # Export each frame
    joint_info = " + joints" if export_joints else ""
    print(f"[3/3] Exporting {end_frame - start_frame} frames to OBJ files{joint_info}...")

    for frame_idx in tqdm(range(start_frame, end_frame), desc="Exporting SMPL"):
        # Get SMPL vertices for this frame
        pose = poses[frame_idx].reshape(24, 3)  # [24, 3]
        t = trans[frame_idx]  # [3]

        with torch.no_grad():
            vertices, joints = smpl_model.forward(betas, pose, t)

        # Convert to numpy
        vertices_np = vertices.cpu().numpy()
        joints_np = joints.cpu().numpy()

        # Create trimesh
        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces, process=False)

        # Export mesh to OBJ
        output_path = output_dir / f"{prefix}_frame_{frame_idx:04d}.obj"
        mesh.export(str(output_path))

        # Export joints/skeleton if requested
        if export_joints:
            # Export SMPL skeleton (joints + bones)
            skeleton_path = output_dir / f"{prefix}_skeleton_frame_{frame_idx:04d}.obj"
            write_skeleton_obj(skeleton_path, joints_np, SMPL_BONES, SMPL_JOINT_NAMES)

            # Export AddB target skeleton if available
            if target_joints is not None and frame_idx < len(target_joints):
                target_path = output_dir / f"{prefix}_addb_frame_{frame_idx:04d}.obj"
                write_skeleton_obj(target_path, target_joints[frame_idx], ADDB_BONES, ADDB_JOINT_NAMES)

    print(f"\nExport complete! Files saved to: {output_dir}")
    print(f"Total OBJ files created: {end_frame - start_frame}")

    return end_frame - start_frame


def main():
    parser = argparse.ArgumentParser(
        description="Export SMPL mesh sequence to OBJ files for MeshLab"
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
        '--prefix',
        type=str,
        default='smpl',
        help='Prefix for output OBJ files (default: "smpl")'
    )
    parser.add_argument(
        '--export_joints',
        action='store_true',
        help='Export SMPL joints as separate OBJ files (small spheres)'
    )
    parser.add_argument(
        '--target_joints',
        type=str,
        default=None,
        help='Path to target_joints.npy for comparison (optional)'
    )

    args = parser.parse_args()

    # Convert paths
    smpl_params_path = Path(args.smpl_params)
    smpl_model_path = Path(args.smpl_model)
    output_dir = Path(args.output_dir)

    # Validate input paths
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
    print("\n" + "="*80)
    print("SMPL MESH TO OBJ EXPORTER")
    print("="*80)

    target_joints_path = Path(args.target_joints) if args.target_joints else None

    export_smpl_sequence(
        smpl_params_path=smpl_params_path,
        smpl_model_path=smpl_model_path,
        output_dir=output_dir,
        frame_range=frame_range,
        prefix=args.prefix,
        export_joints=args.export_joints,
        target_joints_path=target_joints_path
    )

    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print(f"\nYou can now open the OBJ files in MeshLab:")
    print(f"  meshlab {output_dir}/{args.prefix}_frame_*.obj")
    print()


if __name__ == '__main__':
    main()
